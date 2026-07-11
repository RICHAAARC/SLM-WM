"""验证 Shallow Diffuse 官方参考环境补充表 受治理导入 协议。"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import pytest

from experiments.runtime import repository_environment
from paper_experiments.baselines import (
    SHALLOW_DIFFUSE_OFFICIAL_REFERENCE_PROTOCOL_NAME,
    build_shallow_diffuse_official_reference_record,
    build_shallow_diffuse_official_reference_schema,
    validate_shallow_diffuse_official_reference_records,
)
from paper_experiments.runners.shallow_diffuse_official_reference import (
    ShallowDiffuseOfficialReferenceConfig,
    build_reference_record_report,
    build_default_config,
    build_official_command,
    ensure_shallow_diffuse_source_available,
    output_paths,
    package_shallow_diffuse_official_reference_outputs,
    normalize_metric_summary,
    parse_metric_text,
    patch_shallow_diffuse_model_repository_layout,
    prepare_shallow_diffuse_dependency_environment,
    prepare_shallow_diffuse_model_repository,
    write_shallow_diffuse_official_reference_outputs,
)
from paper_experiments.runners.openclip_checkpoint_runtime import (
    DEFAULT_OPENCLIP_CHECKPOINT_PATH,
    OPENCLIP_CHECKPOINT_FILENAME,
    OPENCLIP_CHECKPOINT_SHA256,
    OPENCLIP_CHECKPOINT_SIZE_BYTES,
    OPENCLIP_MODEL_NAME,
    OPENCLIP_REPOSITORY_ID,
    OPENCLIP_REVISION,
)
from paper_experiments.runners.closure_package_selection import (
    CLOSURE_PACKAGE_FAMILY_SPECS,
    inspect_closure_package,
)
from tests.helpers.formal_execution_lock import build_test_formal_execution_lock


FORMAL_EXECUTION_LOCK = build_test_formal_execution_lock()
DEPENDENCY_PROFILE_ID = "shallow_diffuse_official_py39_cu117"
SOURCE_PROVENANCE = {
    "source_worktree_digest": "a" * 64,
    "source_patch_sha256": "b" * 64,
    "prompt_dataset_repository_id": "Gustavosta/Stable-Diffusion-Prompts",
    "prompt_dataset_revision": "d816d4a05cb89bde39dd99284c459801e1e7e69a",
    "official_model_repository_id": "Manojb/stable-diffusion-2-1-base",
    "official_model_revision": "0094d483a120f3f33dafbd187ea4aa60d10de75c",
    "model_snapshot_content_digest": "d" * 64,
    "openclip_model_name": OPENCLIP_MODEL_NAME,
    "openclip_repository_id": OPENCLIP_REPOSITORY_ID,
    "openclip_revision": OPENCLIP_REVISION,
    "openclip_checkpoint_filename": OPENCLIP_CHECKPOINT_FILENAME,
    "openclip_checkpoint_sha256": OPENCLIP_CHECKPOINT_SHA256,
    "openclip_checkpoint_size_bytes": OPENCLIP_CHECKPOINT_SIZE_BYTES,
    "openclip_snapshot_content_digest": "c" * 64,
}
READY_FLAGS = {
    "official_source_ready": True,
    "source_identity_ready": True,
    "source_worktree_exact": True,
    "official_environment_report_ready": True,
    "official_execution_ready": True,
    "required_metrics_ready": True,
    "model_source_ready": True,
    "openclip_source_ready": True,
    "official_result_summary_ready": True,
    "governed_import_ready": True,
}
MODEL_REPOSITORY_REPORT = {
    "local_model_repository_ready": True,
    "official_model_id": "Manojb/stable-diffusion-2-1-base",
    "official_model_revision": "0094d483a120f3f33dafbd187ea4aa60d10de75c",
    "model_snapshot_content": {
        "repository_id": "Manojb/stable-diffusion-2-1-base",
        "revision": "0094d483a120f3f33dafbd187ea4aa60d10de75c",
        "allow_patterns": [
            "feature_extractor/*",
            "model_index.json",
            "safety_checker/*",
            "scheduler/*",
            "text_encoder/*",
            "tokenizer/*",
            "unet/*",
            "vae/*",
        ],
        "snapshot_content_digest": "d" * 64,
    },
}
OPENCLIP_REPORT = {
    "openclip_checkpoint_ready": True,
    **{key: value for key, value in SOURCE_PROVENANCE.items() if key.startswith("openclip_")},
    "model_snapshot_content": {
        "repository_id": OPENCLIP_REPOSITORY_ID,
        "revision": OPENCLIP_REVISION,
        "allow_patterns": [OPENCLIP_CHECKPOINT_FILENAME],
        "file_count": 1,
        "files": [
            {
                "path": OPENCLIP_CHECKPOINT_FILENAME,
                "size_bytes": OPENCLIP_CHECKPOINT_SIZE_BYTES,
                "sha256": OPENCLIP_CHECKPOINT_SHA256,
            }
        ],
        "snapshot_content_digest": "c" * 64,
    },
}


class _DependencyProfileFixture(SimpleNamespace):
    """提供 runner 所需的最小不可变依赖身份接口。"""

    def to_dict(self) -> dict[str, object]:
        """返回可写入环境报告的 profile 记录。"""

        return dict(vars(self))


def _dependency_profile(*, ready: bool) -> _DependencyProfileFixture:
    """构造不依赖仓库 registry 当前状态的固定 profile 夹具。"""

    return _DependencyProfileFixture(
        profile_name=DEPENDENCY_PROFILE_ID,
        profile_digest="f" * 64,
        direct_requirements_digest="d" * 64,
        complete_hash_lock_path=(
            f"configs/dependency_profiles/{DEPENDENCY_PROFILE_ID}_lock.txt"
        ),
        complete_hash_lock_present=ready,
        complete_hash_lock_digest="e" * 64 if ready else None,
        complete_hash_lock_dependency_count=24 if ready else 0,
        formal_ready=ready,
        readiness_blockers=() if ready else ("complete_hash_lock_missing",),
    )


def _ready_dependency_profile() -> _DependencyProfileFixture:
    """返回具有完整哈希锁的 Shallow Diffuse 固定依赖 profile 夹具。"""

    return _dependency_profile(ready=True)


def _write_isolated_dependency_environment_report(
    root: Path,
    python_executable: Path,
    profile: _DependencyProfileFixture,
    *,
    lock_digest: str | None = None,
) -> tuple[dict[str, object], Path]:
    """写出隔离环境 API 的最小成功报告。"""

    resolved_lock_digest = lock_digest or profile.complete_hash_lock_digest
    dependency_preparation_report = {
        "profile_id": DEPENDENCY_PROFILE_ID,
        "profile_digest": profile.profile_digest,
        "complete_hash_lock_digest": resolved_lock_digest,
        "decision": "pass",
        "failure_reasons": [],
        "repository_commit_state": {"all_committed": True},
        "installation": {"attempted": True, "return_code": 0},
        "runtime_comparison": {
            "decision": "pass",
            "environment_match": True,
            "mismatches": [],
        },
    }
    report: dict[str, object] = {
        "report_schema": "isolated_dependency_environment_preparation_report",
        "schema_version": 1,
        "profile_id": DEPENDENCY_PROFILE_ID,
        "profile_digest": profile.profile_digest,
        "direct_requirements_digest": profile.direct_requirements_digest,
        "complete_hash_lock_digest": resolved_lock_digest,
        "complete_hash_lock_dependency_count": profile.complete_hash_lock_dependency_count,
        "provisioned": True,
        "formal_preparation_completed": True,
        "formal_ready": True,
        "decision": "pass",
        "failure_reasons": [],
        "supports_paper_claim": False,
        "python_executable_path": str(python_executable),
        "python_executable_sha256": "a" * 64,
        "python_executable_sha256_after_preparation": "a" * 64,
        "dependency_preparation_report_digest": "b" * 64,
        "dependency_preparation_report": dependency_preparation_report,
        "provision_report": {
            "decision": "provisioned",
            "provisioned": True,
            "profile_digest": profile.profile_digest,
        },
        "command_results": [
            {
                "operation": "dependency_profile_preparation",
                "return_code": 0,
            }
        ],
    }
    report_path = (
        root
        / "outputs"
        / "dependency_profiles"
        / DEPENDENCY_PROFILE_ID
        / "isolated_dependency_environment_report.json"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False),
        encoding="utf-8",
    )
    return report, report_path
@pytest.fixture(autouse=True)
def _select_pilot_paper(monkeypatch: pytest.MonkeyPatch) -> None:
    """本模块的官方参考归档夹具固定使用 pilot_paper."""

    monkeypatch.setenv("SLM_WM_PAPER_RUN_NAME", "pilot_paper")
    monkeypatch.setattr(
        repository_environment,
        "require_published_formal_execution_lock",
        lambda _root: dict(FORMAL_EXECUTION_LOCK),
    )


@pytest.mark.quick
def test_shallow_diffuse_official_reference_record_validates_when_all_boundaries_ready() -> None:
    """官方固定 profile 复现记录满足证据边界时应通过补充表导入校验。"""

    record = build_shallow_diffuse_official_reference_record(
        official_entrypoint="external_baseline/primary/shallow_diffuse/source/run_shallow_diffuse_t2i.py",
        official_repository_commit="c80c553fdf66fda8db735d77a9d56538b7a0ade8",
        official_environment_profile=DEPENDENCY_PROFILE_ID,
        baseline_result_source="outputs/shallow_diffuse_official_reference/summary.json",
        baseline_result_source_digest="digest",
        evidence_paths=["outputs/shallow_diffuse_official_reference/summary.json"],
        source_provenance=SOURCE_PROVENANCE,
        metric_values={
            "sample_count": 5,
            "positive_count": 5,
            "negative_count": 5,
            "auc": 0.9,
            "accuracy": 0.8,
            "true_positive_rate_at_one_percent_fpr": 0.7,
            "clip_score_mean": -0.1,
            "watermarked_clip_score_mean": -0.2,
        },
        ready_flags=READY_FLAGS,
    )

    report = validate_shallow_diffuse_official_reference_records([record])
    schema = build_shallow_diffuse_official_reference_schema()

    assert schema["reference_protocol_name"] == SHALLOW_DIFFUSE_OFFICIAL_REFERENCE_PROTOCOL_NAME
    assert record["supplemental_table_role"] == "supplemental_method_fidelity_reference"
    assert record["main_table_eligible"] is False
    assert report["reference_import_ready"] is True
    assert report["accepted_reference_record_count"] == 1

    incomplete_record = dict(record)
    incomplete_record.pop("watermarked_clip_score_mean")
    incomplete_report = validate_shallow_diffuse_official_reference_records([incomplete_record])
    assert incomplete_report["reference_import_ready"] is False
    assert any(
        issue["reason"] == "watermarked_clip_score_mean_required"
        for issue in incomplete_report["issues"]
    )


@pytest.mark.quick
def test_shallow_diffuse_official_reference_rejects_main_table_eligibility() -> None:
    """官方固定 profile 参考记录不得伪装为主表同协议结果。"""

    record = build_shallow_diffuse_official_reference_record(
        official_entrypoint="external_baseline/primary/shallow_diffuse/source/run_shallow_diffuse_t2i.py",
        official_repository_commit="c80c553fdf66fda8db735d77a9d56538b7a0ade8",
        official_environment_profile=DEPENDENCY_PROFILE_ID,
        baseline_result_source="outputs/shallow_diffuse_official_reference/summary.json",
        baseline_result_source_digest="digest",
        evidence_paths=["outputs/shallow_diffuse_official_reference/summary.json"],
        source_provenance=SOURCE_PROVENANCE,
        metric_values={
            "sample_count": 5,
            "positive_count": 5,
            "negative_count": 5,
            "auc": 0.9,
            "accuracy": 0.8,
            "true_positive_rate_at_one_percent_fpr": 0.7,
            "clip_score_mean": 0.0,
            "watermarked_clip_score_mean": 0.0,
        },
        ready_flags=READY_FLAGS,
    )
    record["main_table_eligible"] = True

    report = validate_shallow_diffuse_official_reference_records([record])
    reasons = {issue["reason"] for issue in report["issues"]}

    assert report["reference_import_ready"] is False
    assert "official_reference_must_not_enter_main_table" in reasons


@pytest.mark.quick
def test_shallow_diffuse_official_reference_patches_source_runtime_boundaries(tmp_path: Path) -> None:
    """helper 应把 Shallow Diffuse 官方源码运行补丁记录为可审计产物。"""

    source_dir = tmp_path / "external_baseline" / "primary" / "shallow_diffuse" / "source"
    source_dir.mkdir(parents=True)
    entrypoint = source_dir / "run_shallow_diffuse_t2i.py"
    entrypoint.write_text(
        "pipe = InversableStableDiffusionPipeline.from_pretrained(\n"
        "        args.model_id,\n"
        "        scheduler=scheduler,\n"
        "        torch_dtype=torch.float16,\n"
        "        revision='fp16',\n"
        "        )\n"
        "    suffixes = ['none', 'jpeg', 'gaussianblur', 'gaussianstd', 'colorjitter','randomdrop', 'saltandpepper', 'resizerestore','vaebmshj', 'vaecheng', 'diff']\n"
        "    attackers = {\n"
        "        'none': image_distortion_none,\n"
        "        'diffpure': image_distortion_diffpure,\n"
        "    }\n"
        "    sims = measure_similarity([x0_no_w_img, averaged_image], current_prompt, ref_model, ref_clip_preprocess, ref_tokenizer, device)\n"
        "    w_no_sim = sim[0].item()\n"
        "    avg_sim = sims[1].item()\n",
        encoding="utf-8",
    )
    attackers = source_dir / "attackers.py"
    attackers.write_text(
        "import os\n"
        "from compressai.zoo import bmshj2018_factorized, bmshj2018_hyperprior, mbt2018_mean, mbt2018, cheng2020_anchor\n"
        "def initialize_attackers(args, device):\n"
        "    global vae_attacker1, vae_attacker2, diff_attacker\n"
        "    if args.vae_attack_model_name1 is not None and args.vae_attack_model_name2 is not None:\n"
        "        vae_attacker1 = VAEWMAttacker(args.vae_attack_model_name1, quality=3, metric='mse', device=device)\n"
        "    att_pipe = ReSDPipeline.from_pretrained(\"stabilityai/stable-diffusion-2-1\", torch_dtype=torch.float16, revision=\"fp16\")\n"
        "    diff_attacker = DiffWMAttacker(att_pipe, batch_size=1, noise_step=60, captions={})\n"
        "\n"
        "def image_distortion_none(imgs, seed, args):\n"
        "    return\n",
        encoding="utf-8",
    )
    optim_utils = source_dir / "optim_utils.py"
    optim_utils.write_text(
        "import torch\n"
        "import torch.nn.functional as F\n"
        "from datasets import load_dataset\n"
        "def get_dataset(args):\n"
        "    return load_dataset(args.dataset)['test']\n"
        "def get_watermarking_pattern(pipe, args, device):\n"
        "    gt_init = pipe.get_random_latents()\n"
        "    gt_patch = torch.fft.fft2(gt_init)\n"
        "    return gt_patch\n"
        "def inject_watermark(init_latents_w, watermarking_mask, gt_patch, w_injection):\n"
        "    if 'complex2' == w_injection:\n"
        "        init_latents_w_fft = torch.fft.fft2(init_latents_w)\n"
        "        init_latents_w = torch.fft.ifft2(init_latents_w_fft).real\n"
        "        return init_latents_w\n",
        encoding="utf-8",
    )
    config = ShallowDiffuseOfficialReferenceConfig(
        output_dir="outputs/shallow_diffuse_official_reference",
        source_dir="external_baseline/primary/shallow_diffuse/source",
        patch_model_repository_layout=True,
        require_cuda=False,
    )
    paths = output_paths(tmp_path, config)
    paths["output_dir"].mkdir(parents=True, exist_ok=True)

    report = patch_shallow_diffuse_model_repository_layout(tmp_path, config, paths)
    patched_entrypoint = entrypoint.read_text(encoding="utf-8")
    patched_attackers = attackers.read_text(encoding="utf-8")
    patched_optim_utils = optim_utils.read_text(encoding="utf-8")

    assert report["patch_applied"] is True
    assert "remove_fp16_revision_branch" in report["patch_items"]
    assert "environment_controlled_attacker_suffixes" in report["patch_items"]
    assert "fix_reference_similarity_variable" in report["patch_items"]
    assert "lazy_heavy_attacker_initialization" in report["patch_items"]
    assert "float32_fft_for_profile_cuda" in report["patch_items"]
    assert "preserve_latent_dtype_after_fft_injection" in report["patch_items"]
    assert "revision='fp16'" not in patched_entrypoint
    assert "SLM_WM_SHALLOW_DIFFUSE_OFFICIAL_ATTACKER_NAMES" in patched_entrypoint
    assert "w_no_sim = sims[0].item()" in patched_entrypoint
    assert "w_no_sim = sim[0].item()" not in patched_entrypoint
    assert "compressai_required_for_vae_attackers" in patched_attackers
    assert "def slm_wm_fft2_float32" in patched_optim_utils
    assert "gt_patch = slm_wm_fft2_float32(gt_init)" in patched_optim_utils
    assert "init_latents_w_fft = slm_wm_fft2_float32(init_latents_w)" in patched_optim_utils
    assert "slm_wm_init_latents_dtype = init_latents_w.dtype" in patched_optim_utils
    assert "torch.fft.ifft2(init_latents_w_fft).real.to(slm_wm_init_latents_dtype)" in patched_optim_utils
    assert "revision='d816d4a05cb89bde39dd99284c459801e1e7e69a'" in patched_optim_utils
    assert "pin_prompt_dataset_revision" in report["patch_items"]
    assert report["prompt_dataset_revision"] == "d816d4a05cb89bde39dd99284c459801e1e7e69a"
    assert report["similarity_variable_ready"] is True
    assert report["prompt_dataset_revision_ready"] is True
    assert report["source_patch_postcondition_ready"] is True


@pytest.mark.quick
def test_shallow_diffuse_source_patch_rejects_missing_similarity_postcondition(
    tmp_path: Path,
) -> None:
    """动态补丁未形成正确 sims 读取语句时必须立即阻断。"""

    source_dir = tmp_path / "external_baseline" / "primary" / "shallow_diffuse" / "source"
    source_dir.mkdir(parents=True)
    (source_dir / "run_shallow_diffuse_t2i.py").write_text(
        "print('missing similarity assignment')\n",
        encoding="utf-8",
    )
    (source_dir / "attackers.py").write_text("def initialize_attackers(args, device):\n    pass\n", encoding="utf-8")
    (source_dir / "optim_utils.py").write_text(
        "from datasets import load_dataset\n"
        "def get_dataset(args):\n"
        "    return load_dataset(args.dataset)['test']\n",
        encoding="utf-8",
    )
    config = ShallowDiffuseOfficialReferenceConfig(
        source_dir="external_baseline/primary/shallow_diffuse/source",
        require_cuda=False,
    )
    paths = output_paths(tmp_path, config)
    paths["output_dir"].mkdir(parents=True, exist_ok=True)

    with pytest.raises(RuntimeError, match="必需源码补丁后置条件未满足"):
        patch_shallow_diffuse_model_repository_layout(tmp_path, config, paths)

    report = json.loads(paths["source_patch_result"].read_text(encoding="utf-8"))
    assert report["similarity_variable_ready"] is False
    assert report["prompt_dataset_revision_ready"] is True
    assert report["source_patch_postcondition_ready"] is False


@pytest.mark.quick
def test_shallow_diffuse_official_reference_prepares_local_model_repository(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """本地模型目录应补齐固定 transformers 版本需要的 model_index 项。"""

    local_model_dir = tmp_path / "runtime_model" / "stable_diffusion_2_1_base"
    config = ShallowDiffuseOfficialReferenceConfig(
        output_dir="outputs/shallow_diffuse_official_reference",
        source_dir="external_baseline/primary/shallow_diffuse/source",
        official_model_id="Manojb/stable-diffusion-2-1-base",
        local_model_repository_dir=str(local_model_dir),
        prepare_local_model_repository=True,
        patch_model_index_for_pinned_transformers=True,
        require_cuda=False,
    )
    paths = output_paths(tmp_path, config)
    paths["output_dir"].mkdir(parents=True, exist_ok=True)

    def fake_ensure_hugging_face_snapshot_files(
        repository_dir: str | Path,
        *,
        report_path: str | Path,
        repository_id: str,
        revision: str,
        allow_patterns: tuple[str, ...],
        token: str | None,
    ) -> dict[str, object]:
        assert repository_id == "Manojb/stable-diffusion-2-1-base"
        assert revision == "0094d483a120f3f33dafbd187ea4aa60d10de75c"
        assert "model_index.json" in allow_patterns
        local_dir = Path(repository_dir)
        local_dir.mkdir(parents=True, exist_ok=True)
        (local_dir / "model_index.json").write_text(
            json.dumps(
                {
                    "_class_name": "StableDiffusionPipeline",
                    "feature_extractor": ["transformers", "CLIPImageProcessor"],
                    "scheduler": ["diffusers", "PNDMScheduler"],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return {"download_requested": True, "snapshot_path": str(local_dir)}

    monkeypatch.setattr(
        "paper_experiments.runners.shallow_diffuse_official_reference.ensure_hugging_face_snapshot_files",
        fake_ensure_hugging_face_snapshot_files,
    )

    report = prepare_shallow_diffuse_model_repository(tmp_path, config, paths)
    patched_index = json.loads((local_model_dir / "model_index.json").read_text(encoding="utf-8"))

    assert report["local_model_repository_ready"] is True
    assert report["official_model_revision"] == "0094d483a120f3f33dafbd187ea4aa60d10de75c"
    assert len(report["model_snapshot_content"]["snapshot_content_digest"]) == 64
    assert report["model_index_patch_applied"] is True
    assert report["effective_official_model_id"] == str(local_model_dir)
    assert patched_index["feature_extractor"] == ["transformers", "CLIPFeatureExtractor"]


@pytest.mark.quick
def test_shallow_diffuse_official_reference_prepares_fixed_dependency_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """只有隔离环境 API 报告与固定 profile 身份完全一致时才允许运行。"""

    dependency_python = tmp_path / "shallow_env" / "bin" / "python"
    dependency_python.parent.mkdir(parents=True)
    dependency_python.write_text("#!/bin/sh\n", encoding="utf-8")
    profile = _ready_dependency_profile()
    config = ShallowDiffuseOfficialReferenceConfig(
        output_dir="outputs/shallow_diffuse_official_reference",
        source_dir="external_baseline/primary/shallow_diffuse/source",
        require_cuda=False,
    )
    paths = output_paths(tmp_path, config)
    paths["output_dir"].mkdir(parents=True, exist_ok=True)

    def fake_prepare_environment(
        profile_id: str,
        *,
        repository_root: Path,
    ) -> tuple[dict[str, object], Path]:
        assert profile_id == DEPENDENCY_PROFILE_ID
        assert repository_root == tmp_path
        return _write_isolated_dependency_environment_report(
            tmp_path,
            dependency_python,
            profile,
        )

    monkeypatch.setattr(
        "paper_experiments.runners.official_reference_dependency_environment.get_dependency_profile",
        lambda _profile_id, _registry_path: profile,
    )
    monkeypatch.setattr(
        "paper_experiments.runners.official_reference_dependency_environment.require_dependency_profile_ready",
        lambda _profile_id, _registry_path: profile,
    )
    monkeypatch.setattr(
        "paper_experiments.runners.official_reference_dependency_environment.prepare_isolated_dependency_environment",
        fake_prepare_environment,
    )

    monkeypatch.setattr(
        "paper_experiments.runners.official_reference_dependency_environment.validate_official_reference_dependency_environment_report",
        lambda *_args, **_kwargs: SimpleNamespace(
            validation_errors=(),
            dependency_python_executable=str(dependency_python),
            dependency_installation_performed=True,
            isolated_dependency_environment_report_digest="f" * 64,
            passed=True,
        ),
    )

    report = prepare_shallow_diffuse_dependency_environment(tmp_path, config, paths)

    assert report["dependency_environment_requested"] is True
    assert report["dependency_environment_ready"] is True
    assert report["dependency_profile_id"] == DEPENDENCY_PROFILE_ID
    assert report["dependency_environment_report_valid"] is True
    assert report["dependency_environment_materialized"] is True
    assert len(report["command_results"]) == 1


@pytest.mark.quick
def test_shallow_diffuse_dependency_profile_missing_lock_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """固定 profile 缺少完整哈希锁时不得调用隔离环境 API。"""

    profile = _dependency_profile(ready=False)
    config = ShallowDiffuseOfficialReferenceConfig(require_cuda=False)
    paths = output_paths(tmp_path, config)
    paths["output_dir"].mkdir(parents=True, exist_ok=True)

    def reject_profile(_profile_id: str, _registry_path: Path) -> object:
        raise RuntimeError("完整哈希锁缺失")

    monkeypatch.setattr(
        "paper_experiments.runners.official_reference_dependency_environment.get_dependency_profile",
        lambda _profile_id, _registry_path: profile,
    )
    monkeypatch.setattr(
        "paper_experiments.runners.official_reference_dependency_environment.require_dependency_profile_ready",
        reject_profile,
    )

    report = prepare_shallow_diffuse_dependency_environment(tmp_path, config, paths)

    assert report["dependency_environment_ready"] is False
    assert report["dependency_profile_ready"] is False
    assert report["dependency_lock_ready"] is False
    assert report["dependency_environment_failure_reason"] == "dependency_hash_lock_not_ready"
    assert report["command_results"] == []


@pytest.mark.quick
def test_shallow_diffuse_official_reference_parses_metric_text_and_custom_python(tmp_path: Path) -> None:
    """官方日志解析与固定 profile Python 可执行文件配置应保持可审计。"""

    config = ShallowDiffuseOfficialReferenceConfig(
        source_dir="external_baseline/primary/shallow_diffuse/source",
        sample_count=5,
        edit_time_list="0.3",
        num_inference_steps=50,
        attacker_names="none",
    )

    metrics = parse_metric_text(
        "CLIP scores\n"
        "clip_score_mean: 0.39522719383239746\n"
        "avg_clip_score_mean: 0.3972677767276764\n"
        "auc: 0.95, acc: 0.84, TPR@1%FPR: 0.72\n",
        sample_count=5,
    )
    command = build_official_command(
        tmp_path,
        config,
        "/opt/shallow-diffuse-profile/bin/python",
    )

    assert metrics["sample_count"] == 5
    assert metrics["clip_score_mean"] == 0.39522719383239746
    assert metrics["watermarked_clip_score_mean"] == 0.3972677767276764
    assert metrics["auc"] == 0.95
    assert command[0] == "/opt/shallow-diffuse-profile/bin/python"
    assert command[command.index("--dataset") + 1] == "Gustavosta/Stable-Diffusion-Prompts"
    assert "--edit_time_list" in command
    assert command[command.index("--reference_model") + 1] == "ViT-g-14"
    assert Path(command[command.index("--reference_model_pretrain") + 1]).name == OPENCLIP_CHECKPOINT_FILENAME
    assert command[command.index("--end") + 1] == "5"


@pytest.mark.quick
def test_shallow_diffuse_metric_summary_requires_all_scientific_metrics() -> None:
    """缺失任一官方科学指标时不得用 0 补齐正式摘要。"""

    incomplete = normalize_metric_summary(
        {
            "auc": 0.95,
            "accuracy": 0.84,
            "true_positive_rate_at_one_percent_fpr": 0.72,
            "clip_score_mean": 0.39,
        },
        sample_count=5,
    )
    complete = normalize_metric_summary(
        {
            "auc": 0.95,
            "accuracy": 0.84,
            "true_positive_rate_at_one_percent_fpr": 0.72,
            "clip_score_mean": 0.39,
            "watermarked_clip_score_mean": 0.38,
        },
        sample_count=5,
    )

    assert incomplete == {}
    assert complete["sample_count"] == 5
    assert complete["watermarked_clip_score_mean"] == 0.38


@pytest.mark.quick
def test_shallow_diffuse_record_requires_successful_current_official_command(
    tmp_path: Path,
) -> None:
    """历史指标字典不能绕过本次官方命令成功门禁生成记录。"""

    config = ShallowDiffuseOfficialReferenceConfig(
        output_dir="outputs/shallow_diffuse_official_reference",
        source_dir="external_baseline/primary/shallow_diffuse/source",
        sample_count=5,
        require_cuda=False,
    )
    paths = output_paths(tmp_path, config)
    paths["output_dir"].mkdir(parents=True, exist_ok=True)
    metric_summary = normalize_metric_summary(
        {
            "auc": 0.95,
            "accuracy": 0.84,
            "true_positive_rate_at_one_percent_fpr": 0.72,
            "clip_score_mean": 0.39,
            "watermarked_clip_score_mean": 0.38,
        },
        sample_count=5,
    )
    source_status = {
        "official_entrypoint": "external_baseline/primary/shallow_diffuse/source/run_shallow_diffuse_t2i.py",
        "official_entrypoint_ready": True,
        "official_repository_commit": "c80c553fdf66fda8db735d77a9d56538b7a0ade8",
        "source_identity_ready": True,
        "source_worktree_exact": True,
        **{key: value for key, value in SOURCE_PROVENANCE.items() if not key.startswith("openclip_")},
    }
    openclip_report = OPENCLIP_REPORT

    failed = build_reference_record_report(
        tmp_path,
        config,
        paths,
        metric_summary,
        {"official_command_requested": True, "return_code": 1},
        source_status,
        {"dependency_environment_ready": True, "dependency_environment_profile_id": DEPENDENCY_PROFILE_ID},
        MODEL_REPOSITORY_REPORT,
        openclip_report,
    )
    succeeded = build_reference_record_report(
        tmp_path,
        config,
        paths,
        metric_summary,
        {"official_command_requested": True, "return_code": 0},
        source_status,
        {"dependency_environment_ready": True, "dependency_environment_profile_id": DEPENDENCY_PROFILE_ID},
        MODEL_REPOSITORY_REPORT,
        openclip_report,
    )

    assert failed["record_count"] == 0
    assert succeeded["record_count"] == 1
    assert succeeded["validation"]["reference_import_ready"] is True


@pytest.mark.quick
def test_shallow_diffuse_official_reference_rejects_non_git_source_cache(tmp_path: Path) -> None:
    """正式导入不得接受无法核验提交身份的普通源码目录。"""

    source_dir = tmp_path / "external_baseline" / "primary" / "shallow_diffuse" / "source"
    source_dir.mkdir(parents=True)
    (source_dir / "run_shallow_diffuse_t2i.py").write_text("print('shallow diffuse official entry')\n", encoding="utf-8")
    (source_dir / "attackers.py").write_text("print('attackers')\n", encoding="utf-8")
    config = ShallowDiffuseOfficialReferenceConfig(
        output_dir="outputs/shallow_diffuse_official_reference",
        drive_output_dir=str(tmp_path / "drive"),
        source_dir="external_baseline/primary/shallow_diffuse/source",
        sample_count=5,
        run_official_command=False,
        require_cuda=False,
    )

    with pytest.raises(RuntimeError, match="不是可验证的 Git checkout"):
        write_shallow_diffuse_official_reference_outputs(config, root=tmp_path)


@pytest.mark.quick
def test_shallow_diffuse_official_reference_package_embeds_archive_self_description(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """打包结果应包含归档摘要、归档 manifest 和输入清单。"""

    code_version = "b" * 40
    output_dir = tmp_path / "outputs" / "shallow_diffuse_official_reference" / "pilot_paper"
    output_dir.mkdir(parents=True)
    (output_dir / "shallow_diffuse_official_reference_summary.json").write_text(
        json.dumps(
            {
                "baseline_id": "shallow_diffuse",
                "paper_claim_scale": "pilot_paper",
                "target_fpr": 0.01,
                "run_decision": "pass",
                "shallow_diffuse_official_reference_ready": True,
                "reference_import_ready": True,
                "dependency_profile_id": DEPENDENCY_PROFILE_ID,
                "dependency_environment_profile_id": DEPENDENCY_PROFILE_ID,
                "dependency_profile_ready": True,
                "dependency_lock_ready": True,
                "dependency_environment_materialized": True,
                "dependency_environment_report_valid": True,
                "dependency_profile_digest": "f" * 64,
                "dependency_lock_digest": "e" * 64,
                "model_source_ready": True,
                "model_snapshot_scope_ready": True,
                "model_source_repository_id": "Manojb/stable-diffusion-2-1-base",
                "model_source_revision": "0094d483a120f3f33dafbd187ea4aa60d10de75c",
                "model_snapshot_content_digest": "4" * 64,
                "openclip_source_ready": True,
                "openclip_repository_id": OPENCLIP_REPOSITORY_ID,
                "openclip_revision": OPENCLIP_REVISION,
                "openclip_checkpoint_filename": OPENCLIP_CHECKPOINT_FILENAME,
                "openclip_checkpoint_sha256": OPENCLIP_CHECKPOINT_SHA256,
                "openclip_checkpoint_size_bytes": OPENCLIP_CHECKPOINT_SIZE_BYTES,
                "openclip_snapshot_content_digest": "5" * 64,
                "official_command_requested": True,
                "official_command_return_code": 0,
                "official_execution_ready": True,
                "required_metrics_ready": True,
                "official_command_result_ready": True,
                "governed_reference_record_count": 1,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (output_dir / "manifest.local.json").write_text(
        json.dumps(
                {
                    "code_version": code_version,
                    "formal_execution_run_lock": FORMAL_EXECUTION_LOCK,
                    "metadata": {"run_decision": "pass"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "shallow_diffuse_official_reference_records.jsonl").write_text(
        json.dumps({"baseline_id": "shallow_diffuse"}) + "\n",
        encoding="utf-8",
    )
    (output_dir / "shallow_diffuse_official_reference_validation_report.json").write_text(
        json.dumps({"reference_import_ready": True}) + "\n",
        encoding="utf-8",
    )

    record = package_shallow_diffuse_official_reference_outputs(
        root=tmp_path,
        output_dir="outputs/shallow_diffuse_official_reference",
        drive_output_dir=str(tmp_path / "drive" / "SLM" / "external_baseline_official_reference"),
        archive_name="external_baseline_official_reference_package_shallow_diffuse_test.zip",
    )

    archive_path = tmp_path / record.archive_path
    expected_entries = {
        "outputs/shallow_diffuse_official_reference/pilot_paper/shallow_diffuse_official_reference_summary.json",
        "outputs/shallow_diffuse_official_reference/pilot_paper/shallow_diffuse_official_reference_package_input_manifest.json",
        "outputs/shallow_diffuse_official_reference/pilot_paper/shallow_diffuse_official_reference_archive_summary.json",
        "outputs/shallow_diffuse_official_reference/pilot_paper/shallow_diffuse_official_reference_archive_manifest.local.json",
    }
    with ZipFile(archive_path) as archive:
        names = set(archive.namelist())
        package_manifest = json.loads(
            archive.read(
                "outputs/shallow_diffuse_official_reference/pilot_paper/shallow_diffuse_official_reference_package_input_manifest.json"
            ).decode("utf-8")
        )

    assert expected_entries <= names
    assert package_manifest["entry_count"] == len(names)
    assert record.archive_digest
    assert record.drive_archive_digest
    spec = next(
        item
        for item in CLOSURE_PACKAGE_FAMILY_SPECS
        if item.package_family == "official_reference_shallow_diffuse"
    )
    candidate = inspect_closure_package(
        archive_path,
        spec=spec,
        paper_run_name="pilot_paper",
        target_fpr=0.01,
    )
    assert candidate.package_family == "official_reference_shallow_diffuse"


@pytest.mark.quick
def test_shallow_diffuse_official_reference_cold_start_clones_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Colab 冷启动缺少官方源码时, helper 应按登记表补齐 source 缓存。"""

    registry_path = tmp_path / "external_baseline" / "source_registry.json"
    registry_path.parent.mkdir(parents=True)
    registry_path.write_text(
        json.dumps(
            {
                "baseline_sources": [
                    {
                        "baseline_id": "shallow_diffuse",
                        "source_dir": "external_baseline/primary/shallow_diffuse/source",
                        "official_repository_url": "git@github.com:liwd190019/Shallow-Diffuse.git",
                        "official_repository_commit": "c80c553fdf66fda8db735d77a9d56538b7a0ade8",
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    config = ShallowDiffuseOfficialReferenceConfig(
        output_dir="outputs/shallow_diffuse_official_reference",
        source_dir="external_baseline/primary/shallow_diffuse/source",
        require_cuda=False,
    )
    paths = output_paths(tmp_path, config)

    def fake_run_command(command: list[str], *, cwd: Path, timeout_seconds: int) -> dict[str, object]:
        if command[:2] == ["git", "clone"]:
            source_dir = Path(command[-1])
            source_dir.mkdir(parents=True, exist_ok=True)
            (source_dir / "run_shallow_diffuse_t2i.py").write_text("print('official source')\n", encoding="utf-8")
            (source_dir / "README.md").write_text("# Shallow Diffuse\n", encoding="utf-8")
        return {"command": command, "return_code": 0, "stdout": "", "stderr": ""}

    monkeypatch.setattr("paper_experiments.runners.shallow_diffuse_official_reference.run_command", fake_run_command)

    report = ensure_shallow_diffuse_source_available(tmp_path, config, paths)

    assert report["source_available"] is True
    assert report["source_downloaded"] is True
    assert report["official_entrypoint_ready"] is True
    assert report["official_repository_url"] == "https://github.com/liwd190019/Shallow-Diffuse.git"
    assert paths["source_prepare_result"].is_file()


@pytest.mark.quick
def test_shallow_diffuse_official_reference_default_config_reads_runtime_parameters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Notebook 参数层可传运行参数, 但依赖身份必须保持固定。"""

    monkeypatch.setenv("SLM_WM_SHALLOW_DIFFUSE_EDIT_TIME_LIST", "0.3")
    monkeypatch.setenv("SLM_WM_SHALLOW_DIFFUSE_OFFICIAL_ATTACKER_NAMES", "none")
    monkeypatch.setenv("SLM_WM_SHALLOW_DIFFUSE_W_PATTERN", "complex2_ring")
    monkeypatch.setenv("SLM_WM_SHALLOW_DIFFUSE_W_MEASUREMENT", "l1_complex2")
    monkeypatch.setenv("SLM_WM_SHALLOW_DIFFUSE_OFFICIAL_MODEL_ID", "Manojb/stable-diffusion-2-1-base")

    config = build_default_config()

    assert "official_python_executable" not in config.__dataclass_fields__
    assert config.dependency_profile_id == DEPENDENCY_PROFILE_ID
    assert config.edit_time_list == "0.3"
    assert config.attacker_names == "none"
    assert config.w_pattern == "complex2_ring"
    assert config.w_measurement == "l1_complex2"
    assert config.official_model_id == "Manojb/stable-diffusion-2-1-base"
    assert config.dataset == "Gustavosta/Stable-Diffusion-Prompts"
    assert config.dataset_revision == "d816d4a05cb89bde39dd99284c459801e1e7e69a"
    assert config.reference_model == "ViT-g-14"
    assert config.reference_model_checkpoint_path == DEFAULT_OPENCLIP_CHECKPOINT_PATH
    assert Path(config.reference_model_checkpoint_path).name == OPENCLIP_CHECKPOINT_FILENAME


@pytest.mark.quick
def test_shallow_diffuse_official_reference_rejects_alternate_dependency_profile() -> None:
    """正式参考不得由调用方替换固定依赖 profile。"""

    with pytest.raises(ValueError, match="固定依赖 profile"):
        ShallowDiffuseOfficialReferenceConfig(dependency_profile_id="workflow_orchestrator")


@pytest.mark.quick
def test_shallow_diffuse_official_reference_rejects_unregistered_prompt_dataset() -> None:
    """正式配置不得把精确 Gustavosta 数据来源替换为其他仓库。"""

    with pytest.raises(ValueError, match="精确 Prompt 数据集 revision"):
        ShallowDiffuseOfficialReferenceConfig(dataset="example/mutable-prompts")


