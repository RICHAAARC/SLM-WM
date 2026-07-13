"""验证 Gaussian Shading 官方参考环境补充表 受治理导入 协议。"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import pytest

from experiments.runtime import repository_environment
from paper_experiments.baselines import (
    GAUSSIAN_SHADING_OFFICIAL_REFERENCE_PROTOCOL_NAME,
    build_gaussian_shading_official_reference_record,
    build_gaussian_shading_official_reference_schema,
    validate_gaussian_shading_official_reference_records,
)
from paper_experiments.runners.gaussian_shading_official_reference import (
    GaussianShadingOfficialReferenceConfig,
    DEFAULT_OFFICIAL_MODEL_ID,
    DEFAULT_OFFICIAL_MODEL_REVISION,
    GAUSSIAN_SHADING_PACKAGE_GENERATED_FILE_NAMES,
    GAUSSIAN_SHADING_PACKAGE_ROOT_FILE_WHITELIST,
    build_default_config,
    build_official_command,
    build_reference_record_report,
    ensure_gaussian_shading_source_available,
    openclip_checkpoint_source_is_exact,
    output_paths,
    package_gaussian_shading_official_reference_outputs,
    parse_metric_text,
    patch_gaussian_shading_model_repository_layout,
    prepare_gaussian_shading_dependency_environment,
    prepare_gaussian_shading_model_repository,
    write_gaussian_shading_official_reference_outputs,
)
from paper_experiments.runners.model_snapshot_runtime import DIFFUSERS_PIPELINE_ALLOW_PATTERNS
from paper_experiments.runners.official_reference_unit_runtime import (
    build_official_reference_config_digest,
)
from paper_experiments.runners.openclip_checkpoint_runtime import (
    OPENCLIP_CHECKPOINT_FILENAME,
    OPENCLIP_CHECKPOINT_SHA256,
    OPENCLIP_CHECKPOINT_SIZE_BYTES,
    OPENCLIP_ALLOW_PATTERNS,
    OPENCLIP_MODEL_NAME,
    OPENCLIP_REPOSITORY_ID,
    OPENCLIP_REVISION,
    OPENCLIP_SOURCE_NAME,
    OPENCLIP_USAGE_ROLE,
)
from paper_experiments.runners.closure_package_selection import (
    CLOSURE_PACKAGE_FAMILY_SPECS,
    ClosurePackageSelectionError,
    inspect_closure_package,
)
from tests.helpers.formal_execution_lock import build_test_formal_execution_lock


FORMAL_EXECUTION_LOCK = build_test_formal_execution_lock()
DEPENDENCY_PROFILE_ID = "gaussian_shading_official_py38_cu117"
OFFICIAL_SCIENTIFIC_CONFIG = {"test": True}
OFFICIAL_SCIENTIFIC_CONFIG_DIGEST = build_official_reference_config_digest(
    "gaussian_shading",
    OFFICIAL_SCIENTIFIC_CONFIG,
)
SOURCE_PROVENANCE = {
    "source_worktree_digest": "a" * 64,
    "source_patch_sha256": "b" * 64,
    "prompt_dataset_repository_id": "Gustavosta/Stable-Diffusion-Prompts",
    "prompt_dataset_revision": "d816d4a05cb89bde39dd99284c459801e1e7e69a",
    "official_model_repository_id": DEFAULT_OFFICIAL_MODEL_ID,
    "official_model_revision": DEFAULT_OFFICIAL_MODEL_REVISION,
    "model_snapshot_content_digest": "c" * 64,
    "openclip_source_name": OPENCLIP_SOURCE_NAME,
    "openclip_usage_role": OPENCLIP_USAGE_ROLE,
    "openclip_model_name": OPENCLIP_MODEL_NAME,
    "openclip_repository_id": OPENCLIP_REPOSITORY_ID,
    "openclip_revision": OPENCLIP_REVISION,
    "openclip_checkpoint_filename": OPENCLIP_CHECKPOINT_FILENAME,
    "openclip_checkpoint_sha256": OPENCLIP_CHECKPOINT_SHA256,
    "openclip_checkpoint_size_bytes": OPENCLIP_CHECKPOINT_SIZE_BYTES,
    "openclip_snapshot_content_digest": "d" * 64,
    "official_scientific_config": OFFICIAL_SCIENTIFIC_CONFIG,
    "official_scientific_config_digest": OFFICIAL_SCIENTIFIC_CONFIG_DIGEST,
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
    """返回具有完整哈希锁的 Gaussian Shading 固定依赖 profile 夹具。"""

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
def test_gaussian_shading_official_reference_record_validates_when_all_boundaries_ready() -> None:
    """官方固定 profile 复现记录满足证据边界时应通过补充表导入校验。"""

    record = build_gaussian_shading_official_reference_record(
        official_command_requested=True,
        official_command_return_code=0,
        official_entrypoint="external_baseline/primary/gaussian_shading/source/run_gaussian_shading.py",
        official_repository_commit="09c678fadc7545acf7be12647ddf2a5e66f6a9dc",
        official_environment_profile=DEPENDENCY_PROFILE_ID,
        baseline_result_source="outputs/gaussian_shading_official_reference/summary.json",
        baseline_result_source_digest="e" * 64,
        evidence_paths=["outputs/gaussian_shading_official_reference/summary.json"],
        source_provenance=SOURCE_PROVENANCE,
        metric_values={
            "sample_count": 5,
            "positive_count": 5,
            "detection_true_positive_rate": 0.8,
            "traceability_true_positive_rate": 0.6,
            "mean_bit_accuracy": 0.9,
            "std_bit_accuracy": 0.1,
            "mean_clip_score": -0.25,
            "std_clip_score": 0.03,
        },
        ready_flags={
            "official_command_succeeded": True,
            "official_source_ready": True,
            "source_identity_ready": True,
            "source_worktree_exact": True,
            "official_environment_report_ready": True,
            "official_result_summary_ready": True,
            "model_source_ready": True,
            "openclip_source_ready": True,
            "governed_import_ready": True,
        },
    )

    report = validate_gaussian_shading_official_reference_records([record])
    schema = build_gaussian_shading_official_reference_schema()

    assert schema["reference_protocol_name"] == GAUSSIAN_SHADING_OFFICIAL_REFERENCE_PROTOCOL_NAME
    assert schema["expected_openclip_provenance"]["openclip_revision"] == OPENCLIP_REVISION
    assert record["official_command_requested"] is True
    assert record["official_command_return_code"] == 0
    assert record["openclip_checkpoint_sha256"] == OPENCLIP_CHECKPOINT_SHA256
    assert record["supplemental_table_role"] == "supplemental_method_fidelity_reference"
    assert record["main_table_eligible"] is False
    assert report["reference_import_ready"] is True
    assert report["accepted_reference_record_count"] == 1

    tampered_record = json.loads(json.dumps(record))
    tampered_record["official_scientific_config"]["test"] = False
    tampered_report = validate_gaussian_shading_official_reference_records(
        [tampered_record]
    )
    assert "official_scientific_config_digest_mismatch" in {
        issue["reason"] for issue in tampered_report["issues"]
    }


@pytest.mark.quick
def test_gaussian_shading_official_reference_rejects_main_table_eligibility() -> None:
    """官方固定 profile 参考记录不得伪装为主表同协议结果。"""

    record = build_gaussian_shading_official_reference_record(
        official_command_requested=True,
        official_command_return_code=0,
        official_entrypoint="external_baseline/primary/gaussian_shading/source/run_gaussian_shading.py",
        official_repository_commit="09c678fadc7545acf7be12647ddf2a5e66f6a9dc",
        official_environment_profile=DEPENDENCY_PROFILE_ID,
        baseline_result_source="outputs/gaussian_shading_official_reference/summary.json",
        baseline_result_source_digest="e" * 64,
        evidence_paths=["outputs/gaussian_shading_official_reference/summary.json"],
        source_provenance=SOURCE_PROVENANCE,
        metric_values={
            "sample_count": 5,
            "positive_count": 5,
            "detection_true_positive_rate": 0.8,
            "traceability_true_positive_rate": 0.6,
            "mean_bit_accuracy": 0.9,
            "std_bit_accuracy": 0.1,
            "mean_clip_score": 0.25,
            "std_clip_score": 0.03,
        },
        ready_flags={
            "official_command_succeeded": True,
            "official_source_ready": True,
            "source_identity_ready": True,
            "source_worktree_exact": True,
            "official_environment_report_ready": True,
            "official_result_summary_ready": True,
            "model_source_ready": True,
            "openclip_source_ready": True,
            "governed_import_ready": True,
        },
    )
    record["main_table_eligible"] = True

    report = validate_gaussian_shading_official_reference_records([record])
    reasons = {issue["reason"] for issue in report["issues"]}

    assert report["reference_import_ready"] is False
    assert "official_reference_must_not_enter_main_table" in reasons


@pytest.mark.quick
def test_gaussian_shading_official_reference_rejects_openclip_provenance_drift() -> None:
    """OpenCLIP checkpoint 任一来源字段漂移都必须阻断 受治理导入."""

    provenance = dict(SOURCE_PROVENANCE)
    provenance["openclip_checkpoint_sha256"] = "0" * 64
    record = build_gaussian_shading_official_reference_record(
        official_command_requested=True,
        official_command_return_code=0,
        official_entrypoint="external_baseline/primary/gaussian_shading/source/run_gaussian_shading.py",
        official_repository_commit="09c678fadc7545acf7be12647ddf2a5e66f6a9dc",
        official_environment_profile=DEPENDENCY_PROFILE_ID,
        baseline_result_source="outputs/gaussian_shading_official_reference/metric_summary.json",
        baseline_result_source_digest="e" * 64,
        evidence_paths=["outputs/gaussian_shading_official_reference/metric_summary.json"],
        source_provenance=provenance,
        metric_values={
            "sample_count": 5,
            "positive_count": 5,
            "detection_true_positive_rate": 0.8,
            "traceability_true_positive_rate": 0.6,
            "mean_bit_accuracy": 0.9,
            "std_bit_accuracy": 0.1,
            "mean_clip_score": 0.25,
            "std_clip_score": 0.03,
        },
        ready_flags={field_name: True for field_name in (
            "official_command_succeeded",
            "official_source_ready",
            "source_identity_ready",
            "source_worktree_exact",
            "official_environment_report_ready",
            "official_result_summary_ready",
            "model_source_ready",
            "openclip_source_ready",
            "governed_import_ready",
        )},
    )

    report = validate_gaussian_shading_official_reference_records([record])

    assert report["reference_import_ready"] is False
    assert any(
        issue["field_name"] == "openclip_checkpoint_sha256"
        and issue["reason"] == "openclip_checkpoint_sha256_registered_value_required"
        for issue in report["issues"]
    )


@pytest.mark.quick
def test_gaussian_shading_official_reference_patches_model_repository_layout(tmp_path: Path) -> None:
    """公开镜像缺少 fp16 分支时, helper 应把官方入口补丁记录为可审计产物。"""

    source_dir = tmp_path / "external_baseline" / "primary" / "gaussian_shading" / "source"
    source_dir.mkdir(parents=True)
    entrypoint = source_dir / "run_gaussian_shading.py"
    entrypoint.write_text(
        "from watermark import *\n"
        "def main(args):\n"
        "    #CLIP Scores\n"
        "    clip_scores = []\n"
        "    for i in tqdm(range(args.num)):\n"
        "        seed = i + args.gen_seed\n"
        "        current_prompt = 'prompt'\n"
        "        acc_metric = 1.0\n"
        "        clip_socre = 1.0\n"
        "        clip_scores.append(clip_socre)\n"
        "    tpr_detection, tpr_traceability, acc = 1, 1, [1.0]\n"
        "    save_metrics(args, tpr_detection, tpr_traceability, acc, clip_scores)\n"
        "pipe = InversableStableDiffusionPipeline.from_pretrained(\n"
        "            args.model_path,\n"
        "            scheduler=scheduler,\n"
        "            torch_dtype=torch.float16,\n"
        "            revision='fp16',\n"
        ")\n",
        encoding="utf-8",
    )
    optim_utils = source_dir / "optim_utils.py"
    optim_utils.write_text(
        "from datasets import load_dataset\n"
        "def get_dataset(args):\n"
        "    return load_dataset(args.dataset_path)['train']\n",
        encoding="utf-8",
    )
    config = GaussianShadingOfficialReferenceConfig(
        output_dir="outputs/gaussian_shading_official_reference",
        source_dir="external_baseline/primary/gaussian_shading/source",
        official_model_id="Manojb/stable-diffusion-2-1-base",
        patch_model_repository_layout=True,
        require_cuda=False,
    )
    paths = output_paths(tmp_path, config)
    paths["output_dir"].mkdir(parents=True, exist_ok=True)

    report = patch_gaussian_shading_model_repository_layout(tmp_path, config, paths)
    patched_text = entrypoint.read_text(encoding="utf-8")
    patched_optim_text = optim_utils.read_text(encoding="utf-8")
    saved_report = json.loads(paths["source_patch_result"].read_text(encoding="utf-8"))

    assert report["patch_applied"] is True
    assert "revision='fp16'" not in patched_text
    assert "公开镜像没有 fp16 分支" in patched_text
    assert "revision='d816d4a05cb89bde39dd99284c459801e1e7e69a'" in patched_optim_text
    assert "pin_prompt_dataset_revision" in report["patch_items"]
    assert saved_report["prompt_dataset_revision"] == "d816d4a05cb89bde39dd99284c459801e1e7e69a"
    assert saved_report["official_model_id"] == "Manojb/stable-diffusion-2-1-base"
    assert saved_report["upstream_official_model_id"] == "stabilityai/stable-diffusion-2-1-base"


@pytest.mark.quick
def test_gaussian_shading_official_reference_prepares_local_model_repository(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """本地模型目录应补齐固定 diffusers 版本需要的 model_index 项。"""

    local_model_dir = tmp_path / "runtime_model" / "stable_diffusion_2_1_base"
    config = GaussianShadingOfficialReferenceConfig(
        output_dir="outputs/gaussian_shading_official_reference",
        source_dir="external_baseline/primary/gaussian_shading/source",
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
        "paper_experiments.runners.gaussian_shading_official_reference.ensure_hugging_face_snapshot_files",
        fake_ensure_hugging_face_snapshot_files,
    )

    report = prepare_gaussian_shading_model_repository(tmp_path, config, paths)
    patched_index = json.loads((local_model_dir / "model_index.json").read_text(encoding="utf-8"))
    saved_report = json.loads(paths["model_repository_prepare_result"].read_text(encoding="utf-8"))

    assert report["local_model_repository_ready"] is True
    assert report["official_model_revision"] == "0094d483a120f3f33dafbd187ea4aa60d10de75c"
    assert len(report["model_snapshot_content"]["snapshot_content_digest"]) == 64
    assert report["model_index_patch_applied"] is True
    assert report["effective_official_model_id"] == str(local_model_dir)
    assert patched_index["feature_extractor"] == ["transformers", "CLIPFeatureExtractor"]
    assert saved_report["model_index_feature_extractor"] == ["transformers", "CLIPFeatureExtractor"]


@pytest.mark.quick
def test_gaussian_shading_dependency_profile_missing_lock_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """固定 profile 缺少完整哈希锁时不得调用隔离环境 API。"""

    profile = _dependency_profile(ready=False)
    config = GaussianShadingOfficialReferenceConfig(require_cuda=False)
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

    report = prepare_gaussian_shading_dependency_environment(tmp_path, config, paths)

    assert report["dependency_environment_ready"] is False
    assert report["dependency_profile_ready"] is False
    assert report["dependency_lock_ready"] is False
    assert report["dependency_environment_failure_reason"] == "dependency_hash_lock_not_ready"
    assert report["command_results"] == []


@pytest.mark.quick
def test_gaussian_shading_dependency_profile_requires_valid_isolated_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """隔离环境 API 报告身份闭合时才能标记环境已就绪。"""

    dependency_python = tmp_path / "gaussian_env" / "bin" / "python"
    dependency_python.parent.mkdir(parents=True)
    dependency_python.write_text("#!/bin/sh\n", encoding="utf-8")
    profile = _ready_dependency_profile()
    config = GaussianShadingOfficialReferenceConfig(
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

    report = prepare_gaussian_shading_dependency_environment(tmp_path, config, paths)

    assert report["dependency_environment_ready"] is True
    assert report["dependency_environment_materialized"] is True
    assert report["dependency_environment_report_valid"] is True
    assert report["dependency_environment_profile_id"] == DEPENDENCY_PROFILE_ID
    assert len(report["command_results"]) == 1


@pytest.mark.quick
def test_gaussian_shading_dependency_profile_rejects_report_lock_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """隔离环境报告中的哈希锁摘要漂移时必须保留诊断并阻断运行。"""

    dependency_python = tmp_path / "gaussian_env" / "bin" / "python"
    dependency_python.parent.mkdir(parents=True)
    dependency_python.write_text("#!/bin/sh\n", encoding="utf-8")
    profile = _ready_dependency_profile()
    config = GaussianShadingOfficialReferenceConfig(
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
            lock_digest="0" * 64,
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

    report = prepare_gaussian_shading_dependency_environment(tmp_path, config, paths)

    assert report["dependency_environment_ready"] is False
    assert report["dependency_environment_report_valid"] is False
    assert report["dependency_environment_failure_reason"] == "isolated_dependency_environment_report_invalid"
    assert "complete_hash_lock_digest_mismatch" in report["dependency_environment_validation_errors"]


@pytest.mark.quick
def test_gaussian_shading_official_reference_rejects_non_git_source_cache(tmp_path: Path) -> None:
    """正式导入不得接受无法核验提交身份的普通源码目录."""

    source_dir = tmp_path / "external_baseline" / "primary" / "gaussian_shading" / "source"
    source_dir.mkdir(parents=True)
    (source_dir / "run_gaussian_shading.py").write_text("print('gaussian shading official entry')\n", encoding="utf-8")
    (source_dir / "requirements.txt").write_text("diffusers==0.11.1\ntransformers==4.34.0\n", encoding="utf-8")
    config = GaussianShadingOfficialReferenceConfig(
        output_dir="outputs/gaussian_shading_official_reference",
        drive_output_dir=str(tmp_path / "drive"),
        source_dir="external_baseline/primary/gaussian_shading/source",
        sample_count=5,
        require_cuda=False,
    )

    with pytest.raises(RuntimeError, match="不是可验证的 Git checkout"):
        write_gaussian_shading_official_reference_outputs(config, root=tmp_path)


@pytest.mark.quick
def test_gaussian_shading_official_reference_package_embeds_archive_self_description(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """打包结果应包含归档摘要、归档 manifest 和输入清单。"""

    code_version = "b" * 40
    output_dir = tmp_path / "outputs" / "gaussian_shading_official_reference" / "pilot_paper"
    output_dir.mkdir(parents=True)
    (output_dir / "gaussian_shading_official_reference_summary.json").write_text(
        json.dumps(
            {
                "baseline_id": "gaussian_shading",
                "paper_claim_scale": "pilot_paper",
                "target_fpr": 0.1,
                "run_decision": "pass",
                "gaussian_shading_official_reference_ready": True,
                "reference_import_ready": True,
                "official_command_requested": True,
                "official_command_return_code": 0,
                "official_command_succeeded": True,
                "official_unit_coverage_ready": True,
                "official_unit_batch_size": 10,
                "official_unit_expected_count": 7,
                "official_unit_completed_count": 7,
                "official_unit_records_digest": "6" * 64,
                    "official_unit_observations_digest": "7" * 64,
                    "official_unit_command_identities_digest": "9" * 64,
                "scientific_unit_provenance": {
                    "scientific_unit_provenance_ready": True
                },
                "official_scientific_config": {"test": True},
                "official_scientific_config_digest": "8" * 64,
                "scientific_metrics_complete": True,
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
                "openclip_source_name": OPENCLIP_SOURCE_NAME,
                "openclip_usage_role": OPENCLIP_USAGE_ROLE,
                "openclip_model_name": OPENCLIP_MODEL_NAME,
                "openclip_repository_id": OPENCLIP_REPOSITORY_ID,
                "openclip_revision": OPENCLIP_REVISION,
                "openclip_checkpoint_filename": OPENCLIP_CHECKPOINT_FILENAME,
                "openclip_checkpoint_sha256": OPENCLIP_CHECKPOINT_SHA256,
                "openclip_checkpoint_size_bytes": OPENCLIP_CHECKPOINT_SIZE_BYTES,
                "openclip_snapshot_content_digest": "5" * 64,
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
    (output_dir / "gaussian_shading_official_reference_records.jsonl").write_text(
        json.dumps({"baseline_id": "gaussian_shading"}) + "\n",
        encoding="utf-8",
    )
    (output_dir / "gaussian_shading_official_reference_validation_report.json").write_text(
        json.dumps({"reference_import_ready": True}) + "\n",
        encoding="utf-8",
    )
    (output_dir / "gaussian_shading_official_metric_summary.json").write_text(
        "{}\n",
        encoding="utf-8",
    )
    (output_dir / "gaussian_shading_model_repository_prepare_result.json").write_text(
        "{}\n",
        encoding="utf-8",
    )
    (output_dir / "gaussian_shading_openclip_checkpoint_prepare_result.json").write_text(
        "{}\n",
        encoding="utf-8",
    )
    for relative_path in sorted(
        GAUSSIAN_SHADING_PACKAGE_ROOT_FILE_WHITELIST
        - GAUSSIAN_SHADING_PACKAGE_GENERATED_FILE_NAMES
    ):
        required_path = output_dir / relative_path
        if not required_path.exists():
            required_path.parent.mkdir(parents=True, exist_ok=True)
            required_path.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        "paper_experiments.runners.gaussian_shading_official_reference.validate_persisted_official_reference_units",
        lambda **_kwargs: {
            "official_unit_coverage_ready": True,
            "official_unit_expected_count": 7,
            "official_unit_completed_count": 7,
            "official_unit_records_digest": "6" * 64,
                "official_unit_observations_digest": "7" * 64,
                "official_unit_command_identities_digest": "9" * 64,
            "scientific_unit_provenance": {
                "scientific_unit_provenance_ready": True
            },
            "official_scientific_config": {"test": True},
            "official_scientific_config_digest": "8" * 64,
            "official_unit_commands": [],
            "metric_summary": {},
            "stable_unit_identity": {
                "formal_execution_commit": code_version,
                "formal_execution_lock_digest": FORMAL_EXECUTION_LOCK[
                    "formal_execution_lock_digest"
                ],
                "official_repository_commit": None,
                "source_patch_sha256": None,
                "source_worktree_digest": None,
            },
        },
    )
    monkeypatch.setattr(
        "paper_experiments.runners.gaussian_shading_official_reference._validate_packaged_gaussian_shading_reference_evidence",
        lambda *_args: None,
    )
    monkeypatch.setattr(
        "paper_experiments.runners.gaussian_shading_official_reference.validate_official_reference_scientific_config_and_commands",
        lambda **_kwargs: None,
    )

    record = package_gaussian_shading_official_reference_outputs(
        root=tmp_path,
        output_dir="outputs/gaussian_shading_official_reference",
        drive_output_dir=str(tmp_path / "drive" / "SLM" / "external_baseline_official_reference"),
        archive_name="external_baseline_official_reference_package_gaussian_shading_test.zip",
    )

    archive_path = tmp_path / record.archive_path
    expected_entries = {
        "outputs/gaussian_shading_official_reference/pilot_paper/gaussian_shading_official_reference_summary.json",
        "outputs/gaussian_shading_official_reference/pilot_paper/gaussian_shading_official_reference_package_input_manifest.json",
        "outputs/gaussian_shading_official_reference/pilot_paper/gaussian_shading_official_reference_archive_summary.json",
        "outputs/gaussian_shading_official_reference/pilot_paper/gaussian_shading_official_reference_archive_manifest.local.json",
    }
    with ZipFile(archive_path) as archive:
        names = set(archive.namelist())
        package_manifest = json.loads(
            archive.read(
                "outputs/gaussian_shading_official_reference/pilot_paper/gaussian_shading_official_reference_package_input_manifest.json"
            ).decode("utf-8")
        )
        embedded_summary = json.loads(
            archive.read(
                "outputs/gaussian_shading_official_reference/pilot_paper/gaussian_shading_official_reference_archive_summary.json"
            ).decode("utf-8")
        )
        embedded_manifest = json.loads(
            archive.read(
                "outputs/gaussian_shading_official_reference/pilot_paper/gaussian_shading_official_reference_archive_manifest.local.json"
            ).decode("utf-8")
        )

    local_summary = json.loads(
        (output_dir / "gaussian_shading_official_reference_archive_summary.json").read_text(encoding="utf-8")
    )
    local_manifest = json.loads(
        (output_dir / "gaussian_shading_official_reference_archive_manifest.local.json").read_text(encoding="utf-8")
    )

    assert expected_entries <= names
    assert package_manifest["entry_count"] == len(names)
    assert package_manifest["embedded_digest_scope"] == "external_summary_records_final_archive_digest"
    assert embedded_summary["metadata"]["embedded_digest_scope"] == "external_summary_records_final_archive_digest"
    assert embedded_manifest["metadata"]["embedded_digest_scope"] == "external_summary_records_final_archive_digest"
    assert local_summary["archive_digest"] == record.archive_digest
    assert local_summary["drive_archive_digest"] == record.drive_archive_digest
    assert local_manifest["metadata"]["archive_digest"] == record.archive_digest
    assert local_manifest["metadata"]["drive_archive_digest"] == record.drive_archive_digest
    spec = next(
        item
        for item in CLOSURE_PACKAGE_FAMILY_SPECS
        if item.package_family == "official_reference_gaussian_shading"
    )
    with pytest.raises(
        ClosurePackageSelectionError,
        match="缺少必要成员|科学执行证据摘要非法",
    ):
        inspect_closure_package(
            archive_path,
            spec=spec,
            paper_run_name="pilot_paper",
            target_fpr=0.1,
        )


@pytest.mark.quick
def test_gaussian_shading_official_reference_cold_start_clones_source(
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
                        "baseline_id": "gaussian_shading",
                        "source_dir": "external_baseline/primary/gaussian_shading/source",
                        "official_repository_url": "git@github.com:bsmhmmlf/Gaussian-Shading.git",
                        "official_repository_commit": "09c678fadc7545acf7be12647ddf2a5e66f6a9dc",
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    config = GaussianShadingOfficialReferenceConfig(
        output_dir="outputs/gaussian_shading_official_reference",
        source_dir="external_baseline/primary/gaussian_shading/source",
        require_cuda=False,
    )
    paths = output_paths(tmp_path, config)

    def fake_run_command(command: list[str], *, cwd: Path, timeout_seconds: int) -> dict[str, object]:
        if command[:2] == ["git", "clone"]:
            source_dir = Path(command[-1])
            source_dir.mkdir(parents=True, exist_ok=True)
            (source_dir / "run_gaussian_shading.py").write_text("print('official source')\n", encoding="utf-8")
            (source_dir / "requirements.txt").write_text("diffusers==0.11.1\n", encoding="utf-8")
        return {"command": command, "return_code": 0, "stdout": "", "stderr": ""}

    monkeypatch.setattr("paper_experiments.runners.gaussian_shading_official_reference.run_command", fake_run_command)

    report = ensure_gaussian_shading_source_available(tmp_path, config, paths)

    assert report["source_available"] is True
    assert report["source_downloaded"] is True
    assert report["official_entrypoint_ready"] is True
    assert report["official_repository_url"] == "https://github.com/bsmhmmlf/Gaussian-Shading.git"
    assert paths["source_prepare_result"].is_file()


@pytest.mark.quick
def test_gaussian_shading_official_reference_parses_metric_text_and_custom_python(tmp_path: Path) -> None:
    """官方日志解析与固定 profile Python 可执行文件配置应保持可审计。"""

    config = GaussianShadingOfficialReferenceConfig(
        source_dir="external_baseline/primary/gaussian_shading/source",
        sample_count=5,
        official_model_id="/content/model",
    )
    paths = output_paths(tmp_path, config)

    metrics = parse_metric_text(
        "tpr_detection:0.8      tpr_traceability:0.6      mean_acc:0.9      std_acc:0.1      "
        "mean_clip_score:0.25      std_clip_score:0.03\n",
        sample_count=5,
    )
    command = build_official_command(
        tmp_path,
        config,
        paths,
        "/opt/gaussian-shading-profile/bin/python",
    )

    assert metrics["sample_count"] == 5
    assert metrics["detection_true_positive_rate"] == 0.8
    assert metrics["mean_bit_accuracy"] == 0.9
    assert command[0] == "/opt/gaussian-shading-profile/bin/python"
    assert command[command.index("--dataset_path") + 1] == "Gustavosta/Stable-Diffusion-Prompts"
    assert command[command.index("--reference_model") + 1] == OPENCLIP_MODEL_NAME
    assert Path(command[command.index("--reference_model_pretrain") + 1]).name == OPENCLIP_CHECKPOINT_FILENAME
    assert "--num" in command
    assert command[command.index("--num") + 1] == "5"


@pytest.mark.quick
def test_gaussian_shading_metric_parser_does_not_fabricate_missing_scientific_metrics() -> None:
    """官方文本缺失指标时不得以 0 补齐并伪造完整结果."""

    metrics = parse_metric_text("tpr_detection:0.8\n", sample_count=5)

    assert metrics == {
        "sample_count": 5,
        "positive_count": 5,
        "detection_true_positive_rate": 0.8,
    }


@pytest.mark.quick
def test_gaussian_shading_record_requires_current_successful_official_command(tmp_path: Path) -> None:
    """历史或未执行命令的指标不得生成受治理记录."""

    config = GaussianShadingOfficialReferenceConfig(
        sample_count=5,
        require_cuda=False,
    )
    paths = output_paths(tmp_path, config)
    paths["output_dir"].mkdir(parents=True, exist_ok=True)
    source_status = {
        "official_entrypoint": "external_baseline/primary/gaussian_shading/source/run_gaussian_shading.py",
        "official_entrypoint_ready": True,
        "official_repository_commit": "09c678fadc7545acf7be12647ddf2a5e66f6a9dc",
        "source_identity_ready": True,
        "source_worktree_exact": True,
        **SOURCE_PROVENANCE,
    }
    model_report = {
        "local_model_repository_ready": True,
        "official_model_id": DEFAULT_OFFICIAL_MODEL_ID,
        "official_model_revision": DEFAULT_OFFICIAL_MODEL_REVISION,
        "model_snapshot_content": {
            "repository_id": DEFAULT_OFFICIAL_MODEL_ID,
            "revision": DEFAULT_OFFICIAL_MODEL_REVISION,
            "allow_patterns": sorted(DIFFUSERS_PIPELINE_ALLOW_PATTERNS),
            "snapshot_content_digest": "c" * 64,
        },
    }
    openclip_report = {
        "openclip_checkpoint_requested": True,
        "openclip_checkpoint_ready": True,
        "openclip_source_name": OPENCLIP_SOURCE_NAME,
        "openclip_usage_role": OPENCLIP_USAGE_ROLE,
        "openclip_model_name": OPENCLIP_MODEL_NAME,
        "openclip_repository_id": OPENCLIP_REPOSITORY_ID,
        "openclip_revision": OPENCLIP_REVISION,
        "openclip_checkpoint_filename": OPENCLIP_CHECKPOINT_FILENAME,
        "openclip_checkpoint_sha256": OPENCLIP_CHECKPOINT_SHA256,
        "openclip_checkpoint_size_bytes": OPENCLIP_CHECKPOINT_SIZE_BYTES,
        "openclip_snapshot_content_digest": "d" * 64,
        "model_snapshot_content": {
            "repository_id": OPENCLIP_REPOSITORY_ID,
            "revision": OPENCLIP_REVISION,
            "allow_patterns": list(OPENCLIP_ALLOW_PATTERNS),
            "file_count": 1,
            "files": [
                {
                    "path": OPENCLIP_CHECKPOINT_FILENAME,
                    "size_bytes": OPENCLIP_CHECKPOINT_SIZE_BYTES,
                    "sha256": OPENCLIP_CHECKPOINT_SHA256,
                }
            ],
            "snapshot_content_digest": "d" * 64,
        },
    }
    metrics = {
        "sample_count": 5,
        "positive_count": 5,
        "detection_true_positive_rate": 0.8,
        "traceability_true_positive_rate": 0.6,
        "mean_bit_accuracy": 0.9,
        "std_bit_accuracy": 0.1,
        "mean_clip_score": 0.25,
        "std_clip_score": 0.03,
    }

    report = build_reference_record_report(
        tmp_path,
        config,
        paths,
        metrics,
        source_status,
        {"dependency_environment_ready": True, "dependency_environment_profile_id": DEPENDENCY_PROFILE_ID},
        model_report,
        openclip_report,
        {"official_command_requested": False, "return_code": 0},
    )

    assert report["record_count"] == 0
    assert report["official_command_succeeded"] is False
    assert report["openclip_source_ready"] is True
    assert openclip_checkpoint_source_is_exact(openclip_report) is True
    openclip_report["model_snapshot_content"]["files"][0]["sha256"] = "0" * 64
    assert openclip_checkpoint_source_is_exact(openclip_report) is False
    assert paths["reference_records"].read_text(encoding="utf-8") == ""
    assert not paths["official_metric_summary"].exists()


@pytest.mark.quick
def test_gaussian_shading_official_reference_default_config_uses_fixed_dependency_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    """Notebook 参数层不能覆盖隔离解释器或固定依赖身份。"""

    monkeypatch.setenv("SLM_WM_GAUSSIAN_SHADING_OFFICIAL_MODEL_ID", "Manojb/stable-diffusion-2-1-base")
    monkeypatch.setenv("SLM_WM_GAUSSIAN_SHADING_PATCH_MODEL_REPOSITORY_LAYOUT", "1")
    monkeypatch.setenv("SLM_WM_GAUSSIAN_SHADING_PREPARE_LOCAL_MODEL_REPOSITORY", "1")
    monkeypatch.setenv(
        "SLM_WM_GAUSSIAN_SHADING_LOCAL_MODEL_REPOSITORY_DIR",
        "/content/gaussian_shading_model_repository/stable_diffusion_2_1_base",
    )
    monkeypatch.setenv("SLM_WM_GAUSSIAN_SHADING_PATCH_MODEL_INDEX_FOR_PINNED_TRANSFORMERS", "1")

    config = build_default_config()

    assert "official_python_executable" not in config.__dataclass_fields__
    assert config.dependency_profile_id == DEPENDENCY_PROFILE_ID
    assert config.official_model_id == "Manojb/stable-diffusion-2-1-base"
    assert config.dataset_path == "Gustavosta/Stable-Diffusion-Prompts"
    assert config.dataset_revision == "d816d4a05cb89bde39dd99284c459801e1e7e69a"
    assert config.upstream_official_model_id == "stabilityai/stable-diffusion-2-1-base"
    assert config.patch_model_repository_layout is True
    assert config.prepare_local_model_repository is True
    assert config.local_model_repository_dir == "/content/gaussian_shading_model_repository/stable_diffusion_2_1_base"
    assert config.patch_model_index_for_pinned_transformers is True
    assert "summary_import_path" not in config.__dataclass_fields__
    assert "log_import_path" not in config.__dataclass_fields__
    assert "run_official_command" not in config.__dataclass_fields__


@pytest.mark.quick
def test_gaussian_shading_official_reference_rejects_alternate_dependency_profile() -> None:
    """正式参考不得由调用方替换固定依赖 profile。"""

    with pytest.raises(ValueError, match="固定依赖 profile"):
        GaussianShadingOfficialReferenceConfig(dependency_profile_id="workflow_orchestrator")


@pytest.mark.quick
def test_gaussian_shading_official_reference_rejects_unregistered_prompt_dataset() -> None:
    """正式配置不得把精确 Gustavosta 数据来源替换为其他仓库."""

    with pytest.raises(ValueError, match="精确 Prompt 数据集 revision"):
        GaussianShadingOfficialReferenceConfig(
            dataset_path="example/mutable-prompts",
        )


