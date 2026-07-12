"""T2SMark SD3.5 formal 复现边界的轻量测试。"""

from __future__ import annotations

from dataclasses import asdict, replace
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace
from typing import Any

import pytest

from experiments.runtime import repository_environment
from experiments.protocol.attacks import attack_config_digest
from external_baseline.primary.sd35_method_faithful_common import formal_image_attack_config
import paper_experiments.runners.t2smark_formal_reproduction as t2smark_runtime
from paper_experiments.runners.t2smark_formal_reproduction import (
    DEFAULT_FORMAL_IMAGE_ATTACK_FAMILIES,
    T2SMarkFormalReproductionConfig,
    build_t2smark_formal_protocol_binding,
    build_t2smark_formal_checkpoint_contract,
    build_t2smark_formal_run_readiness,
    package_t2smark_formal_reproduction_outputs,
    should_run_official,
    validate_t2smark_formal_protocol_config,
    write_t2smark_formal_protocol_binding,
)
from external_baseline.primary.t2smark.adapter.formal_unit_checkpoint import (
    aggregate_t2smark_formal_unit_records,
    build_t2smark_formal_unit_record,
    inspect_t2smark_formal_unit_records,
    write_or_validate_t2smark_formal_unit_contract,
    write_t2smark_formal_unit_record,
)
from paper_experiments.runners.t2smark_source_runtime import (
    _verify_formal_source,
    configured_attack_names,
    verify_exact_t2smark_protocol_worktree,
)
from paper_experiments.runners.closure_package_selection import (
    CLOSURE_PACKAGE_FAMILY_SPECS,
    inspect_closure_package,
)
from tests.helpers.formal_execution_lock import build_test_formal_execution_lock
from tests.helpers.scientific_execution_binding import (
    write_test_scientific_execution_binding,
)


pytestmark = pytest.mark.quick
FORMAL_EXECUTION_LOCK = build_test_formal_execution_lock()


def test_t2smark_packaging_module_imports_without_scientific_dependencies() -> None:
    """CPU 父打包路径不得因导入模块而加载 PIL, torch 或指标模型依赖."""

    program = """
import builtins

blocked_roots = {
    "PIL",
    "diffusers",
    "lpips",
    "open_clip",
    "torch",
    "torchvision",
    "transformers",
}
original_import = builtins.__import__

def guarded_import(name, *args, **kwargs):
    if name.split(".", 1)[0] in blocked_roots:
        raise ModuleNotFoundError("blocked scientific dependency: " + name)
    return original_import(name, *args, **kwargs)

builtins.__import__ = guarded_import
import paper_experiments.runners.t2smark_formal_reproduction
"""
    completed = subprocess.run(
        [sys.executable, "-c", program],
        cwd=Path(__file__).resolve().parents[2],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


@pytest.fixture(autouse=True)
def _publish_formal_execution_lock(monkeypatch: pytest.MonkeyPatch) -> None:
    """把 T2SMark 归档夹具绑定到确定性正式执行锁."""

    monkeypatch.setattr(
        repository_environment,
        "require_published_formal_execution_lock",
        lambda _root: dict(FORMAL_EXECUTION_LOCK),
    )


def _write_results(path: Path, *, sample_count: int, missing_attack_name: str = "") -> None:
    """写出只用于复用门禁的 T2SMark results.json。"""

    attack_names = configured_attack_names(DEFAULT_FORMAL_IMAGE_ATTACK_FAMILIES)
    attack_identities = {
        attack_name: {
            "attack_id": config.attack_id,
            "resource_profile": config.resource_profile,
            "attack_config_digest": attack_config_digest(config),
        }
        for attack_name in attack_names
        for config in (formal_image_attack_config(attack_name),)
    }
    payload = {
        str(index): {
            "image_only_detection": {
                "clean_score": 0.1,
                "watermarked_score": 0.9,
            },
            "formal_attacks": {
                attack_name: {
                    **attack_identities[attack_name],
                    "attack_name": attack_name,
                    "attacked_negative": {
                        **attack_identities[attack_name],
                        "detection_score": 0.1,
                    },
                    "attacked_positive": {
                        **attack_identities[attack_name],
                        "detection_score": 0.9,
                    },
                }
                for attack_name in attack_names
                if attack_name != missing_attack_name
            }
        }
        for index in range(sample_count)
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _source_report() -> dict[str, object]:
    """构造只用于轻量复用测试的精确源码报告。"""

    return {
        "official_repository_commit": "1" * 40,
        "protocol_patch_sha256": "2" * 64,
        "source_worktree_exact": True,
        "source_worktree_digest": "3" * 64,
        "patched_source_sha256": {
            "option.py": "5" * 64,
            "run_sd35.py": "6" * 64,
        },
    }


def _prompt_report(prompt_count: int, digest: str = "4" * 64) -> dict[str, object]:
    """构造 canonical Prompt 绑定输入。"""

    return {
        "prompt_protocol_name": "paper_probe_paper_prompt_protocol",
        "prompt_protocol_digest": digest,
        "selected_prompt_count": prompt_count,
        "paper_run_prompt_protocol_ready": True,
    }


def _paper_run(
    prompt_count: int,
    prompt_file: str = "configs/paper_main_probe_paper_prompts.txt",
) -> SimpleNamespace:
    """构造协议绑定使用的最小论文运行对象。"""

    return SimpleNamespace(
        run_name="probe_paper",
        protocol_profile="probe_paper_fixed_fpr_0_1",
        prompt_count=prompt_count,
        target_fpr=0.1,
        prompt_set="probe_paper",
        prompt_file=prompt_file,
        sample_count=prompt_count,
    )


def test_t2smark_formal_rejects_unlocked_openclip_source_branch(tmp_path: Path) -> None:
    """正式入口不得激活官方源码中按可变标签下载 OpenCLIP 的分支。"""

    config = T2SMarkFormalReproductionConfig(clip_test_num=1)

    with pytest.raises(ValueError, match="未受治理的 OpenCLIP"):
        validate_t2smark_formal_protocol_config(config, root_path=tmp_path)


def test_t2smark_protocol_binding_records_disabled_source_clip_branch() -> None:
    """复用协议必须显式绑定官方源码 OpenCLIP 分支处于禁用状态。"""

    binding = build_t2smark_formal_protocol_binding(
        T2SMarkFormalReproductionConfig(prompt_limit=1),
        paper_run=_paper_run(1),
        prompt_report=_prompt_report(1),
        source_report=_source_report(),
    )

    assert binding["clip_test_num"] == 0


def _write_reuse_binding(
    tmp_path: Path,
    config: T2SMarkFormalReproductionConfig,
    *,
    prompt_digest: str = "4" * 64,
) -> tuple[Path, Path, Path, dict[str, object]]:
    """写出与当前 results/settings 字节绑定的正式复用记录。"""

    results_path = tmp_path / "results.json"
    settings_path = tmp_path / "settings.json"
    binding_path = tmp_path / "slm_formal_protocol_binding.json"
    _write_results(results_path, sample_count=config.prompt_limit)
    settings_path.write_text('{"settings":"formal"}\n', encoding="utf-8")
    expected_binding = build_t2smark_formal_protocol_binding(
        config,
        paper_run=_paper_run(config.prompt_limit),
        prompt_report=_prompt_report(config.prompt_limit, prompt_digest),
        source_report=_source_report(),
    )
    write_t2smark_formal_protocol_binding(
        binding_path,
        expected_binding,
        results_path=results_path,
        settings_path=settings_path,
    )
    return results_path, settings_path, binding_path, expected_binding


def test_t2smark_reuse_requires_every_formal_attack_for_every_prompt(tmp_path: Path) -> None:
    """只有每个 Prompt 都包含全部正式攻击时才允许复用 T2SMark 结果。"""

    config = T2SMarkFormalReproductionConfig(prompt_limit=2, save_clean_pair=False)
    results_path, settings_path, binding_path, expected_binding = _write_reuse_binding(tmp_path, config)

    should_run, reason = should_run_official(
        config,
        results_path,
        protocol_binding_path=binding_path,
        settings_path=settings_path,
        expected_protocol_binding=expected_binding,
    )

    assert should_run is False
    assert reason == "existing_results_found"


def test_t2smark_reuse_rejects_incomplete_formal_attack_matrix(tmp_path: Path) -> None:
    """缺少任一正式攻击时必须重新运行官方生成与检测链路。"""

    results_path = tmp_path / "results.json"
    missing_attack_name = configured_attack_names(DEFAULT_FORMAL_IMAGE_ATTACK_FAMILIES)[-1]
    config = T2SMarkFormalReproductionConfig(prompt_limit=2, save_clean_pair=False)
    _write_results(results_path, sample_count=2, missing_attack_name=missing_attack_name)
    settings_path = tmp_path / "settings.json"
    binding_path = tmp_path / "slm_formal_protocol_binding.json"
    settings_path.write_text('{"settings":"formal"}\n', encoding="utf-8")
    expected_binding = build_t2smark_formal_protocol_binding(
        config,
        paper_run=_paper_run(2),
        prompt_report=_prompt_report(2),
        source_report=_source_report(),
    )
    write_t2smark_formal_protocol_binding(
        binding_path,
        expected_binding,
        results_path=results_path,
        settings_path=settings_path,
    )

    should_run, reason = should_run_official(
        config,
        results_path,
        protocol_binding_path=binding_path,
        settings_path=settings_path,
        expected_protocol_binding=expected_binding,
    )

    assert should_run is True
    assert reason == "existing_results_formal_attack_count_mismatch"


@pytest.mark.parametrize(
    ("changed_field", "changed_value"),
    (
        ("prompt_digest", "9" * 64),
        ("seed", 20260711),
        ("model_revision", "8" * 40),
        ("num_inference_steps", 21),
        ("num_inversion_steps", 21),
        ("guidance_scale", 5.0),
    ),
)
def test_t2smark_reuse_rejects_protocol_binding_mismatch(
    tmp_path: Path,
    changed_field: str,
    changed_value: object,
) -> None:
    """Prompt 或公平预算发生变化时不得复用旧官方结果。"""

    config = T2SMarkFormalReproductionConfig(prompt_limit=2, save_clean_pair=False)
    results_path, settings_path, binding_path, _ = _write_reuse_binding(tmp_path, config)
    changed_config = config
    prompt_digest = "4" * 64
    if changed_field == "prompt_digest":
        prompt_digest = str(changed_value)
    else:
        changed_payload = asdict(config)
        changed_payload[changed_field] = changed_value
        changed_config = T2SMarkFormalReproductionConfig(**changed_payload)
    expected_binding = build_t2smark_formal_protocol_binding(
        changed_config,
        paper_run=_paper_run(2),
        prompt_report=_prompt_report(2, prompt_digest),
        source_report=_source_report(),
    )

    should_run, reason = should_run_official(
        changed_config,
        results_path,
        protocol_binding_path=binding_path,
        settings_path=settings_path,
        expected_protocol_binding=expected_binding,
    )

    assert should_run is True
    assert reason == "existing_results_protocol_binding_mismatch"


def test_t2smark_reuse_rejects_source_revision_or_patch_mismatch(tmp_path: Path) -> None:
    """源码 revision 或固定补丁摘要变化时必须重新生成。"""

    config = T2SMarkFormalReproductionConfig(prompt_limit=2, save_clean_pair=False)
    results_path, settings_path, binding_path, _ = _write_reuse_binding(tmp_path, config)
    changed_source = _source_report()
    changed_source["protocol_patch_sha256"] = "8" * 64
    expected_binding = build_t2smark_formal_protocol_binding(
        config,
        paper_run=_paper_run(2),
        prompt_report=_prompt_report(2),
        source_report=changed_source,
    )

    should_run, reason = should_run_official(
        config,
        results_path,
        protocol_binding_path=binding_path,
        settings_path=settings_path,
        expected_protocol_binding=expected_binding,
    )

    assert should_run is True
    assert reason == "existing_results_protocol_binding_mismatch"


def test_t2smark_run_readiness_requires_formal_import_validation() -> None:
    """其他运行条件全部通过时, formal import 失败仍必须使运行失败。"""

    assert build_t2smark_formal_run_readiness(
        dependency_environment_ready=True,
        official_ready=True,
        adapter_ready=True,
        prompt_ready=True,
        pair_quality_ready=True,
        formal_attack_ready=True,
        formal_import_validation_ready=False,
        formal_unit_set_ready=True,
    ) is False
    assert build_t2smark_formal_run_readiness(
        dependency_environment_ready=True,
        official_ready=True,
        adapter_ready=True,
        prompt_ready=True,
        pair_quality_ready=True,
        formal_attack_ready=True,
        formal_import_validation_ready=True,
        formal_unit_set_ready=True,
    ) is True
    assert build_t2smark_formal_run_readiness(
        dependency_environment_ready=False,
        official_ready=True,
        adapter_ready=True,
        prompt_ready=True,
        pair_quality_ready=True,
        formal_attack_ready=True,
        formal_import_validation_ready=True,
        formal_unit_set_ready=True,
    ) is False


def test_t2smark_source_worktree_must_equal_fixed_patch(tmp_path: Path) -> None:
    """固定补丁之外的 tracked 或 untracked 改动必须被源码门禁拒绝。"""

    source_dir = tmp_path / "source"
    source_dir.mkdir()
    subprocess.run(["git", "init"], cwd=source_dir, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=source_dir, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=source_dir, check=True)
    (source_dir / "option.py").write_text("VALUE = 1\n", encoding="utf-8")
    (source_dir / "run_sd35.py").write_text("def run():\n    return 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "option.py", "run_sd35.py"], cwd=source_dir, check=True)
    subprocess.run(["git", "commit", "-m", "base"], cwd=source_dir, check=True, capture_output=True)
    (source_dir / "option.py").write_text("VALUE = 2\n", encoding="utf-8")
    (source_dir / "run_sd35.py").write_text("def run():\n    return 2\n", encoding="utf-8")
    patch_path = tmp_path / "formal_protocol.diff"
    diff_result = subprocess.run(
        ["git", "diff", "--binary", "HEAD", "--", "option.py", "run_sd35.py"],
        cwd=source_dir,
        check=True,
        capture_output=True,
    )
    patch_path.write_bytes(diff_result.stdout)

    report = verify_exact_t2smark_protocol_worktree(source_dir, patch_path)
    assert report["source_worktree_exact"] is True
    assert len(str(report["source_worktree_digest"])) == 64

    extra_path = source_dir / "extra.txt"
    extra_path.write_text("unexpected\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="未跟踪文件"):
        verify_exact_t2smark_protocol_worktree(source_dir, patch_path)
    extra_path.unlink()
    (source_dir / "run_sd35.py").write_text("def run():\n    return 3\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="不等于固定 revision"):
        verify_exact_t2smark_protocol_worktree(source_dir, patch_path)


def test_t2smark_fixed_patch_passes_exact_model_revision() -> None:
    """T2SMark 固定源码补丁必须把40位模型 revision 传入 loader。"""

    root = Path(__file__).resolve().parents[2]
    patch_text = (
        root / "external_baseline/primary/t2smark/adapter/formal_protocol_git_diff.txt"
    ).read_text(encoding="utf-8")

    assert '+    parser.add_argument("--model_revision", type=str, required=True)' in patch_text
    assert "+    revision=args.model_revision" in patch_text
    assert '+    parser.add_argument("--slm_unit_contract", type=str, required=True)' in patch_text
    assert "build_t2smark_formal_unit_record" in patch_text
    assert 'prompt_identity["split"] == "test"' in patch_text
    assert "encode_with_exact_clean_base" in patch_text
    assert "torch.random.get_rng_state()" in patch_text
    assert "torch.random.set_rng_state" in patch_text
    assert "utils.set_random_seed(args.seed + prompt_id)" in patch_text
    assert "clean_base_latent_digest_random" in patch_text
    assert "t2smark_secret_material_digest_random" in patch_text
    assert "return_base=True" not in patch_text


def test_t2smark_fixed_patch_applies_to_registered_source_snapshot(
    tmp_path: Path,
) -> None:
    """固定补丁必须可应用到登记源码快照并形成精确正式工作树."""

    root = Path(__file__).resolve().parents[2]
    registered_source = root / "external_baseline/primary/t2smark/source"
    patch_path = (
        root / "external_baseline/primary/t2smark/adapter/formal_protocol_git_diff.txt"
    )
    clean_source = tmp_path / "source"
    clean_source.mkdir()
    subprocess.run(["git", "init"], cwd=clean_source, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=clean_source,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=clean_source,
        check=True,
    )
    for relative_path in ("option.py", "run_sd35.py"):
        content = subprocess.check_output(
            ["git", "show", f"HEAD:{relative_path}"],
            cwd=registered_source,
        )
        (clean_source / relative_path).write_bytes(content)
    subprocess.run(
        ["git", "add", "option.py", "run_sd35.py"],
        cwd=clean_source,
        check=True,
    )
    subprocess.run(
        ["git", "commit", "-m", "base"],
        cwd=clean_source,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "apply", "--unidiff-zero", "--check", str(patch_path)],
        cwd=clean_source,
        check=True,
    )
    subprocess.run(
        ["git", "apply", "--unidiff-zero", str(patch_path)],
        cwd=clean_source,
        check=True,
    )

    report = verify_exact_t2smark_protocol_worktree(clean_source, patch_path)
    _verify_formal_source(clean_source / "run_sd35.py")
    assert report["source_worktree_exact"] is True


class _FixtureCuda:
    """提供科学单元来源构造所需的最小 CUDA 身份接口."""

    @staticmethod
    def is_available() -> bool:
        return True

    @staticmethod
    def current_device() -> int:
        return 0

    @staticmethod
    def device_count() -> int:
        return 1

    @staticmethod
    def get_device_capability(_index: int) -> tuple[int, int]:
        return (7, 5)

    @staticmethod
    def get_device_name(_index: int) -> str:
        return "NVIDIA T4"


FIXTURE_TORCH = SimpleNamespace(
    __version__="2.11.0+cu128",
    version=SimpleNamespace(cuda="12.8"),
    cuda=_FixtureCuda(),
)


def _unit_runtime_environment() -> dict[str, object]:
    """构造逐 Prompt 来源校验使用的完整隔离环境报告."""

    return {
        "dependency_environment_ready": True,
        "formal_execution_lock_ready": True,
        "isolated_scientific_context_ready": True,
        "dependency_profile_id": "t2smark_sd35_gpu",
        "dependency_profile_digest": "1" * 64,
        "direct_requirements_digest": "2" * 64,
        "complete_hash_lock_digest": "3" * 64,
        "formal_execution_commit": FORMAL_EXECUTION_LOCK["formal_execution_commit"],
        "formal_execution_lock_digest": FORMAL_EXECUTION_LOCK[
            "formal_execution_lock_digest"
        ],
        "python_version": "3.12.13",
        "package_versions": {"torch": "2.11.0+cu128"},
        "cuda_version": "12.8",
        "gpu_name": "NVIDIA T4",
        "device_count": 1,
        "isolated_scientific_context": {
            "dependency_environment_report_actual_digest": "4" * 64,
            "current_python_executable_sha256": "5" * 64,
        },
    }


def test_t2smark_prompt_units_resume_only_missing_and_reject_damage(
    tmp_path: Path,
) -> None:
    """dev/calibration/test 均形成单元, 续跑只返回缺失索引且损坏单元闭锁."""

    config = T2SMarkFormalReproductionConfig(
        prompt_limit=3,
        minimum_prompt_protocol_count=3,
    )
    prompt_rows = [
        {
            "prompt_id": f"prompt_{index:06d}",
            "prompt_index": index,
            "prompt_set": "probe_paper",
            "split": split,
            "prompt_text": f"prompt {index}",
            "prompt_digest": str(index + 1) * 64,
        }
        for index, split in enumerate(("dev", "calibration", "test"))
    ]
    protocol_binding = build_t2smark_formal_protocol_binding(
        config,
        paper_run=_paper_run(3),
        prompt_report=_prompt_report(3),
        source_report=_source_report(),
    )
    contract = build_t2smark_formal_checkpoint_contract(
        config,
        paper_run=_paper_run(3),
        prompt_rows=prompt_rows,
        protocol_binding=protocol_binding,
        source_report=_source_report(),
        formal_execution_lock=FORMAL_EXECUTION_LOCK,
    )
    artifact_root = tmp_path / "workspace_a" / "outputs" / "t2smark" / "run"
    watermarked_path = artifact_root / "images" / "00000.png"
    clean_path = artifact_root / "quality_pairs" / "clean" / "00000.png"
    watermarked_path.parent.mkdir(parents=True)
    clean_path.parent.mkdir(parents=True)
    watermarked_path.write_bytes(b"watermarked")
    clean_path.write_bytes(b"clean")
    result = {
        "robustness": {
            "norm1_no_w": 0.1,
            "norm1_w": 0.9,
            "acc_key": 1.0,
            "acc_msg": 1.0,
        },
        "image_only_detection": {
            "clean_score": 0.1,
            "watermarked_score": 0.9,
        },
        "pair_quality": {
            "pair_quality_protocol": "strict_clean_watermarked_pair",
            "clean_base_latent_digest_random": "6" * 64,
            "clean_image_path": str(clean_path),
            "clean_image_digest": t2smark_runtime.file_digest(clean_path),
            "watermarked_image_path": str(watermarked_path),
            "watermarked_image_digest": t2smark_runtime.file_digest(
                watermarked_path
            ),
        },
        "formal_attacks": {},
    }
    runtime_environment = _unit_runtime_environment()
    record = build_t2smark_formal_unit_record(
        contract=contract,
        prompt_index=0,
        result=result,
        artifact_root=artifact_root,
        runtime_environment=runtime_environment,
        torch_module=FIXTURE_TORCH,
        execution_device_name="cuda:0",
        random_identity_random={
            "clean_base_latent_digest_random": "6" * 64,
            "t2smark_secret_material_digest_random": "7" * 64,
        },
    )
    assert not Path(record["result"]["pair_quality"]["clean_image_path"]).is_absolute()
    assert not Path(
        record["result"]["pair_quality"]["watermarked_image_path"]
    ).is_absolute()
    unit_path = artifact_root / "slm_formal_units" / "00000.json"
    write_t2smark_formal_unit_record(
        unit_path,
        record,
        contract=contract,
        artifact_root=artifact_root,
        runtime_environment=runtime_environment,
    )

    records, missing = inspect_t2smark_formal_unit_records(
        unit_path.parent,
        contract=contract,
        artifact_root=artifact_root,
        runtime_environment=runtime_environment,
    )
    assert sorted(records) == [0]
    assert missing == (1, 2)

    moved_artifact_root = (
        tmp_path / "workspace_b" / "outputs" / "t2smark" / "run"
    )
    shutil.copytree(artifact_root, moved_artifact_root)
    moved_records, moved_missing = inspect_t2smark_formal_unit_records(
        moved_artifact_root / "slm_formal_units",
        contract=contract,
        artifact_root=moved_artifact_root,
        runtime_environment=runtime_environment,
    )
    assert sorted(moved_records) == [0]
    assert moved_missing == (1, 2)

    forbidden_attack_path = (
        artifact_root
        / "formal_attacks"
        / "00000_jpeg_compression_attacked_positive.png"
    )
    forbidden_attack_path.parent.mkdir(parents=True, exist_ok=True)
    forbidden_attack_path.write_bytes(b"forbidden dev attack")
    with pytest.raises(ValueError, match="split 协议之外"):
        inspect_t2smark_formal_unit_records(
            unit_path.parent,
            contract=contract,
            artifact_root=artifact_root,
            runtime_environment=runtime_environment,
        )
    forbidden_attack_path.unlink()

    watermarked_path.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="摘要漂移"):
        inspect_t2smark_formal_unit_records(
            unit_path.parent,
            contract=contract,
            artifact_root=artifact_root,
            runtime_environment=runtime_environment,
        )


def test_t2smark_unit_contract_excludes_workspace_and_control_paths() -> None:
    """Drive、checkout 路径和续跑控制开关不得改变科学单元身份."""

    first = T2SMarkFormalReproductionConfig(
        output_dir="outputs/t2smark_formal_reproduction",
        drive_output_dir="/content/drive/session_a",
        prompt_file="/content/checkout_a/configs/paper_main_probe_paper_prompts.txt",
    )
    second = replace(
        first,
        output_dir="outputs/t2smark_formal_reproduction",
        drive_output_dir="/mnt/drive/session_b",
        prompt_file="/workspace_b/configs/paper_main_probe_paper_prompts.txt",
        reuse_existing=False,
        force_generate=True,
        timeout_seconds=123,
        enable_workflow_progress_bar=False,
    )
    prompt_rows = [
        {
            "prompt_id": "prompt_000000",
            "prompt_index": 0,
            "prompt_set": "probe_paper",
            "split": "test",
            "prompt_text": "a ceramic fox",
            "prompt_digest": "1" * 64,
        }
    ]
    protocol_binding = build_t2smark_formal_protocol_binding(
        first,
        paper_run=_paper_run(1),
        prompt_report=_prompt_report(1),
        source_report=_source_report(),
    )
    first_contract = build_t2smark_formal_checkpoint_contract(
        first,
        paper_run=_paper_run(1, "/content/checkout_a/configs/prompts.txt"),
        prompt_rows=prompt_rows,
        protocol_binding=protocol_binding,
        source_report=_source_report(),
        formal_execution_lock=FORMAL_EXECUTION_LOCK,
    )
    second_contract = build_t2smark_formal_checkpoint_contract(
        second,
        paper_run=_paper_run(1, "/workspace_b/configs/prompts.txt"),
        prompt_rows=prompt_rows,
        protocol_binding=protocol_binding,
        source_report=_source_report(),
        formal_execution_lock=FORMAL_EXECUTION_LOCK,
    )

    assert first_contract == second_contract


def _write_package_fixture(
    root_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path]:
    """写出 calibration 与 test 均闭合的 T2SMark 精确白名单 fixture."""

    from PIL import Image

    code_version = "b" * 40
    monkeypatch.setattr(t2smark_runtime, "resolve_code_version", lambda _root: code_version)
    config = T2SMarkFormalReproductionConfig(
        output_dir="outputs/t2smark_formal_reproduction",
        drive_output_dir=str(root_path / "drive"),
        prompt_set="probe_paper",
        prompt_file="configs/paper_main_probe_paper_prompts.txt",
        t2smark_run_name="t",
        prompt_limit=2,
        minimum_prompt_protocol_count=2,
        target_fpr=0.1,
    )
    paper_run = _paper_run(2)
    monkeypatch.setattr(
        t2smark_runtime,
        "validate_t2smark_formal_protocol_config",
        lambda _config, *, root_path: paper_run,
    )
    monkeypatch.setattr(t2smark_runtime, "build_paper_run_config", lambda _root: paper_run)
    paths = t2smark_runtime.output_paths(root_path, config)
    def validate_fixture_candidates(
        rows: Any,
        **_kwargs: object,
    ) -> dict[str, object]:
        """隔离封装测试中的论文规模门禁, 保留候选内容精确重建校验."""

        materialized = tuple(rows)
        return {
            "formal_import_validation_ready": True,
            "accepted_formal_import_count": len(materialized),
            "rejected_formal_import_count": 0,
            "formal_import_issue_count": 0,
        }

    monkeypatch.setattr(
        t2smark_runtime,
        "validate_primary_baseline_formal_import_rows",
        validate_fixture_candidates,
    )
    attack_names = configured_attack_names(config.formal_attack_families)
    source_report = _source_report()
    prompt_report = _prompt_report(2)
    prompt_rows = [
        {
            "prompt_id": f"prompt_{index:06d}",
            "prompt_index": index,
            "prompt_set": "probe_paper",
            "split": split,
            "prompt_text": f"formal test prompt {index}",
            "prompt_digest": str(7 + index) * 64,
        }
        for index, split in enumerate(("calibration", "test"))
    ]
    runtime_environment = _unit_runtime_environment()

    for path_value in (
        paths["official_settings"],
        paths["prompt_dataset"],
        paths["prompt_plan"],
        paths["image_pairs"],
        paths["pair_quality_summary"],
        paths["adapter_observations"],
        paths["adapter_manifest"],
        paths["adapter_artifact_manifest"],
        paths["validation_report"],
        paths["environment_report"],
        paths["source_prepare_result"],
        paths["official_command_result"],
        paths["adapter_command_result"],
    ):
        path_value.parent.mkdir(parents=True, exist_ok=True)

    result_rows: dict[str, dict[str, object]] = {}
    image_pairs: list[dict[str, object]] = []
    for index, prompt_row in enumerate(prompt_rows):
        sample_name = f"{index:05d}.png"
        watermarked_path = paths["official_images"] / sample_name
        clean_path = paths["official_run_dir"] / "quality_pairs" / "clean" / sample_name
        watermarked_path.parent.mkdir(parents=True, exist_ok=True)
        clean_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (16, 16), color=(30 + index, 40, 50)).save(clean_path)
        Image.new("RGB", (16, 16), color=(31 + index, 40, 50)).save(watermarked_path)
        formal_attacks: dict[str, object] = {}
        if prompt_row["split"] == "test":
            for attack_name in attack_names:
                attack_config = formal_image_attack_config(attack_name)
                attack_identity = {
                    "attack_id": attack_config.attack_id,
                    "resource_profile": attack_config.resource_profile,
                    "attack_config_digest": attack_config_digest(attack_config),
                }
                role_rows: dict[str, object] = {}
                for role_offset, sample_role in enumerate(
                    ("attacked_negative", "attacked_positive")
                ):
                    attack_path = paths["official_run_dir"] / "formal_attacks" / (
                        f"{index:05d}_{attack_name}_{sample_role}.png"
                    )
                    attack_path.parent.mkdir(parents=True, exist_ok=True)
                    Image.new(
                        "RGB",
                        (16, 16),
                        color=(60 + role_offset, 70, 80),
                    ).save(attack_path)
                    role_rows[sample_role] = {
                        **attack_identity,
                        "detection_score": 0.2 if sample_role == "attacked_negative" else 0.8,
                        "attacked_image_path": str(attack_path),
                        "attacked_image_digest": t2smark_runtime.file_digest(attack_path),
                    }
                formal_attacks[attack_name] = {
                    **attack_identity,
                    "attack_family": attack_config.attack_family,
                    "attack_name": attack_name,
                    **role_rows,
                }
        clean_digest = t2smark_runtime.file_digest(clean_path)
        watermarked_digest = t2smark_runtime.file_digest(watermarked_path)
        clean_base_digest = ("6" if index == 0 else "8") * 64
        result_rows[str(index)] = {
            "robustness": {
                "norm1_no_w": 0.1 + index * 0.01,
                "norm1_w": 0.9 - index * 0.01,
                "acc_key": 1.0,
                "acc_msg": 1.0,
            },
            "image_only_detection": {
                "clean_score": 0.1 + index * 0.01,
                "watermarked_score": 0.9 - index * 0.01,
            },
            "formal_attacks": formal_attacks,
            "pair_quality": {
                "pair_quality_protocol": "strict_clean_watermarked_pair",
                "clean_base_latent_digest_random": clean_base_digest,
                "clean_image_path": str(clean_path),
                "clean_image_digest": clean_digest,
                "watermarked_image_path": str(watermarked_path),
                "watermarked_image_digest": watermarked_digest,
            },
        }
        image_pairs.append(
            {
                "image_id": f"image_{index:06d}",
                "event_id": f"image_{index:06d}",
                "prompt_id": prompt_row["prompt_id"],
                "prompt_index": index,
                "prompt_set": "probe_paper",
                "split": prompt_row["split"],
                "clean_image_path": clean_path.relative_to(root_path).as_posix(),
                "clean_image_digest": clean_digest,
                "watermarked_image_path": watermarked_path.relative_to(root_path).as_posix(),
                "watermarked_image_digest": watermarked_digest,
                "generated_image_path": watermarked_path.relative_to(root_path).as_posix(),
                "generated_image_digest": watermarked_digest,
                "pair_quality_protocol": "strict_clean_watermarked_pair",
                "strict_pair_quality_ready": True,
            }
        )

    t2smark_runtime.write_json(paths["official_settings"], {"settings": "formal"})
    expected_binding = build_t2smark_formal_protocol_binding(
        config,
        paper_run=paper_run,
        prompt_report=prompt_report,
        source_report=source_report,
    )
    unit_contract = build_t2smark_formal_checkpoint_contract(
        config,
        paper_run=paper_run,
        prompt_rows=prompt_rows,
        protocol_binding=expected_binding,
        source_report=source_report,
        formal_execution_lock=FORMAL_EXECUTION_LOCK,
    )
    write_or_validate_t2smark_formal_unit_contract(
        paths["official_unit_contract"],
        unit_contract,
    )
    for index in range(len(prompt_rows)):
        clean_base_digest = ("6" if index == 0 else "8") * 64
        unit_record = build_t2smark_formal_unit_record(
            contract=unit_contract,
            prompt_index=index,
            result=result_rows[str(index)],
            artifact_root=paths["official_run_dir"],
            runtime_environment=runtime_environment,
            torch_module=FIXTURE_TORCH,
            execution_device_name="cuda:0",
            random_identity_random={
                "clean_base_latent_digest_random": clean_base_digest,
                "t2smark_secret_material_digest_random": ("7" if index == 0 else "9") * 64,
            },
        )
        write_t2smark_formal_unit_record(
            paths["official_unit_dir"] / f"{index:05d}.json",
            unit_record,
            contract=unit_contract,
            artifact_root=paths["official_run_dir"],
            runtime_environment=runtime_environment,
        )
    unit_records, missing_indices = inspect_t2smark_formal_unit_records(
        paths["official_unit_dir"],
        contract=unit_contract,
        artifact_root=paths["official_run_dir"],
        runtime_environment=runtime_environment,
    )
    assert not missing_indices
    rebuilt_results, unit_aggregate = aggregate_t2smark_formal_unit_records(
        unit_records,
        contract=unit_contract,
        artifact_root=paths["official_run_dir"],
        runtime_environment=runtime_environment,
    )
    official_results = t2smark_runtime._rebuild_t2smark_official_payload(
        rebuilt_results,
        unit_aggregate,
    )
    t2smark_runtime.write_json(paths["official_results"], official_results)
    persisted_binding = write_t2smark_formal_protocol_binding(
        paths["official_protocol_binding"],
        expected_binding,
        results_path=paths["official_results"],
        settings_path=paths["official_settings"],
    )
    t2smark_runtime.write_json(
        paths["prompt_dataset"],
        {
            "annotations": [
                {"caption": row["prompt_text"], **row} for row in prompt_rows
            ]
        },
    )
    t2smark_runtime.write_json(paths["prompt_plan"], prompt_rows)
    t2smark_runtime.write_json(paths["image_pairs"], image_pairs)
    observations, adapter_manifest = t2smark_runtime.build_t2smark_observations(
        image_pairs=image_pairs,
        t2smark_results=rebuilt_results,
        target_fpr=config.target_fpr,
        evidence_root=root_path,
    )
    t2smark_runtime.write_json(paths["adapter_observations"], observations)
    adapter_manifest.pop("adapter_digest", None)
    adapter_manifest.update(
        {
            "baseline_observations_path": paths["adapter_observations"]
            .relative_to(root_path)
            .as_posix(),
            "image_pairs_path": paths["image_pairs"].relative_to(root_path).as_posix(),
            "t2smark_results_path": paths["official_results"]
            .relative_to(root_path)
            .as_posix(),
            "generation_protocol": {
                "model_id": config.model_id,
                "model_revision": config.model_revision,
                "num_inference_steps": config.num_inference_steps,
                "guidance_scale": config.guidance_scale,
            },
            "detection_protocol": {
                "input_access_mode": "image_only",
                "num_inversion_steps": config.num_inversion_steps,
                "target_fpr": config.target_fpr,
            },
        }
    )
    adapter_manifest["adapter_digest"] = t2smark_runtime.build_stable_digest(
        adapter_manifest
    )
    t2smark_runtime.write_json(paths["adapter_manifest"], adapter_manifest)
    t2smark_runtime.write_json(paths["adapter_artifact_manifest"], adapter_manifest)
    t2smark_runtime.write_json(
        paths["pair_quality_summary"],
        {"strict_pair_quality_ready": True, "measured_strict_pair_quality_count": 2},
    )
    paths["pair_quality_metrics"].write_text(
        "image_id,ssim\nimage_000000,1.0\nimage_000001,1.0\n",
        encoding="utf-8",
    )
    for path_value in (
        paths["source_prepare_result"],
        paths["official_command_result"],
        paths["adapter_command_result"],
    ):
        t2smark_runtime.write_json(path_value, {})
    t2smark_runtime.write_json(paths["environment_report"], runtime_environment)
    import_report = t2smark_runtime.build_candidate_records_and_validation(
        root_path,
        config,
        paths,
        prompt_report,
    )
    validation = import_report["validation_report"]
    candidate_count = int(import_report["candidate_record_count"])
    assert validation["formal_import_validation_ready"] is True
    summary = {
        "generated_at": "2026-07-11T00:00:00+00:00",
        "baseline_id": "t2smark",
        "paper_claim_scale": "probe_paper",
        "target_fpr": 0.1,
        "run_decision": "pass",
        "t2smark_formal_reproduction_ready": True,
        "t2smark_formal_attack_ready": True,
        "t2smark_formal_unit_set_ready": True,
        "t2smark_formal_unit_record_count": 2,
        "t2smark_formal_unit_records_digest": unit_aggregate[
            "formal_unit_records_digest"
        ],
        "t2smark_formal_unit_aggregate_digest": unit_aggregate[
            "formal_unit_aggregate_digest"
        ],
        "scientific_unit_provenance_ready": True,
        "t2smark_strict_pair_quality_ready": True,
        "formal_import_validation_ready": True,
        "selected_prompt_count": 2,
        "formal_import_candidate_record_count": candidate_count,
        "official_protocol_binding_digest": expected_binding["protocol_binding_digest"],
        "metadata": {
            "prompt_report": prompt_report,
            "official_report": {
                "source_report": source_report,
                "official_protocol_binding": persisted_binding,
            },
        },
    }
    t2smark_runtime.write_json(paths["summary"], summary)
    run_manifest = {
        "code_version": code_version,
        "formal_execution_run_lock": FORMAL_EXECUTION_LOCK,
        "config": asdict(config),
        "metadata": {"run_decision": "pass", "formal_import_validation_ready": True},
    }
    t2smark_runtime.write_json(paths["manifest"], run_manifest)
    write_test_scientific_execution_binding(
        repository_root=root_path,
        artifact_dir=paths["output_dir"],
        artifact_role="t2smark_formal_reproduction",
        paper_run_name="probe_paper",
        profile_id="t2smark_sd35_gpu",
        summary_file_name=paths["summary"].name,
        manifest_file_name=paths["manifest"].name,
        formal_execution_lock=FORMAL_EXECUTION_LOCK,
        execution_route="isolated_t2smark_workflow",
    )
    return paths["output_dir"], root_path / "drive"


def test_t2smark_package_requires_pass_validation_and_exact_whitelist(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """pass 结果可打包, 但任何旧文件或 formal validation 失败都必须阻断。"""

    output_dir, drive_dir = _write_package_fixture(tmp_path, monkeypatch)
    record = package_t2smark_formal_reproduction_outputs(
        root=tmp_path,
        output_dir="outputs/t2smark_formal_reproduction",
        drive_output_dir=str(drive_dir),
        archive_name="external_baseline_official_reference_package_t2smark_test.zip",
    )
    assert (tmp_path / record.archive_path).is_file()
    spec = next(
        item
        for item in CLOSURE_PACKAGE_FAMILY_SPECS
        if item.package_family == "official_reference_t2smark"
    )
    candidate = inspect_closure_package(
        tmp_path / record.archive_path,
        spec=spec,
        paper_run_name="probe_paper",
        target_fpr=0.1,
    )
    assert candidate.package_family == "official_reference_t2smark"
    old_archive_path = output_dir / "external_baseline_official_reference_package_t2smark_old.zip"
    old_archive_path.write_bytes(b"old archive is not a package member")
    with pytest.raises(RuntimeError, match="旧运行或非白名单"):
        package_t2smark_formal_reproduction_outputs(
            root=tmp_path,
            output_dir="outputs/t2smark_formal_reproduction",
            drive_output_dir=str(drive_dir),
            archive_name="external_baseline_official_reference_package_t2smark_test.zip",
        )
    old_archive_path.unlink()
    adapter_manifest_path = output_dir / "t2smark_adapter/t2smark_slm_adapter_manifest.json"
    adapter_artifact_manifest_path = (
        output_dir / "t2smark_adapter/artifacts/t2smark_slm_adapter_manifest.json"
    )
    adapter_manifest = json.loads(adapter_manifest_path.read_text(encoding="utf-8"))
    canonical_adapter_manifest = dict(adapter_manifest)
    adapter_manifest["baseline_observations_path"] = str(
        output_dir / "t2smark_adapter/baseline_observations.json"
    )
    adapter_manifest.pop("adapter_digest")
    adapter_manifest["adapter_digest"] = t2smark_runtime.build_stable_digest(
        adapter_manifest
    )
    t2smark_runtime.write_json(adapter_manifest_path, adapter_manifest)
    t2smark_runtime.write_json(adapter_artifact_manifest_path, adapter_manifest)
    with pytest.raises(RuntimeError, match="不是可迁移路径"):
        package_t2smark_formal_reproduction_outputs(
            root=tmp_path,
            output_dir="outputs/t2smark_formal_reproduction",
            drive_output_dir=str(drive_dir),
            archive_name="external_baseline_official_reference_package_t2smark_test.zip",
        )
    t2smark_runtime.write_json(adapter_manifest_path, canonical_adapter_manifest)
    t2smark_runtime.write_json(
        adapter_artifact_manifest_path,
        canonical_adapter_manifest,
    )
    stale_path = output_dir / "stale_previous_run.json"
    stale_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="旧运行或非白名单"):
        package_t2smark_formal_reproduction_outputs(
            root=tmp_path,
            output_dir="outputs/t2smark_formal_reproduction",
            drive_output_dir=str(drive_dir),
            archive_name="external_baseline_official_reference_package_t2smark_test.zip",
        )
    stale_path.unlink()
    stale_directory = output_dir / "stale_empty_directory"
    stale_directory.mkdir()
    with pytest.raises(RuntimeError, match="层级不是精确白名单"):
        package_t2smark_formal_reproduction_outputs(
            root=tmp_path,
            output_dir="outputs/t2smark_formal_reproduction",
            drive_output_dir=str(drive_dir),
            archive_name="external_baseline_official_reference_package_t2smark_test.zip",
        )
    stale_directory.rmdir()
    symbolic_link = output_dir / "unexpected_symbolic_link.json"
    try:
        symbolic_link.symlink_to(output_dir / "t2smark_official_results.json")
    except OSError:
        pass
    else:
        with pytest.raises(RuntimeError, match="链接"):
            package_t2smark_formal_reproduction_outputs(
                root=tmp_path,
                output_dir="outputs/t2smark_formal_reproduction",
                drive_output_dir=str(drive_dir),
                archive_name="external_baseline_official_reference_package_t2smark_test.zip",
            )
        symbolic_link.unlink()
    candidate_path = output_dir / "t2smark_formal_import_candidate_records.jsonl"
    original_candidates = candidate_path.read_text(encoding="utf-8")
    candidate_rows = [
        json.loads(line) for line in original_candidates.splitlines() if line.strip()
    ]
    candidate_rows[0]["true_positive_rate"] = 0.0
    candidate_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in candidate_rows),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="candidate records 无法"):
        package_t2smark_formal_reproduction_outputs(
            root=tmp_path,
            output_dir="outputs/t2smark_formal_reproduction",
            drive_output_dir=str(drive_dir),
            archive_name="external_baseline_official_reference_package_t2smark_test.zip",
        )
    candidate_path.write_text(original_candidates, encoding="utf-8")
    validation_path = output_dir / "t2smark_formal_import_validation_report.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    validation["formal_import_validation_ready"] = False
    validation_path.write_text(json.dumps(validation), encoding="utf-8")
    with pytest.raises(RuntimeError, match="formal import validation"):
        package_t2smark_formal_reproduction_outputs(
            root=tmp_path,
            output_dir="outputs/t2smark_formal_reproduction",
            drive_output_dir=str(drive_dir),
            archive_name="external_baseline_official_reference_package_t2smark_test.zip",
        )


@pytest.mark.quick
@pytest.mark.parametrize("nested_reparse", (False, True))
def test_t2smark_inventory_rejects_root_or_nested_reparse_point(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    nested_reparse: bool,
) -> None:
    """结果根和任一嵌套目录都不得通过 Windows reparse point 逃逸。"""

    output_dir = tmp_path / "t2smark_results"
    output_dir.mkdir()
    reparse_path = output_dir
    if nested_reparse:
        reparse_path = output_dir / "nested"
        reparse_path.mkdir()
    original_is_junction = getattr(Path, "is_junction", lambda _path: False)

    def simulated_is_junction(path: Path) -> bool:
        """仅把目标测试路径模拟为 junction, 其他路径保留平台行为。"""

        return path == reparse_path or bool(original_is_junction(path))

    monkeypatch.setattr(Path, "is_junction", simulated_is_junction, raising=False)
    with pytest.raises(RuntimeError, match="reparse point"):
        t2smark_runtime._inventory_t2smark_result_tree(output_dir)


def test_t2smark_package_survives_cross_workspace_restore(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """完整 outputs 快照搬迁后可直接重建并打包, 不依赖旧 checkout 路径."""

    source_root = tmp_path / "workspace_a"
    restored_root = tmp_path / "workspace_b"
    _write_package_fixture(source_root, monkeypatch)
    shutil.copytree(source_root / "outputs", restored_root / "outputs")
    restored_drive = restored_root / "drive"
    record = package_t2smark_formal_reproduction_outputs(
        root=restored_root,
        output_dir="outputs/t2smark_formal_reproduction",
        drive_output_dir=str(restored_drive),
        archive_name="external_baseline_official_reference_package_t2smark_restored.zip",
    )

    assert (restored_root / record.archive_path).is_file()
