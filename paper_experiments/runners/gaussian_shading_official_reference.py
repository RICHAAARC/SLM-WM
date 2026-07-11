"""Gaussian Shading 官方参考环境复现与受治理导入的 Colab 辅助函数。

该辅助模块服务补充表方法忠实度审计。它不把固定 Stable Diffusion 2.1 profile 结果混入 SD3.5 主表,
而是把官方命令、运行日志、环境报告、指标摘要和受治理导入记录统一写入 outputs/。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import shutil
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

from experiments.protocol.paper_run_config import build_paper_run_config, resolve_count_from_environment
from experiments.runtime import repository_environment
from experiments.runtime.model_sources import get_model_source
from paper_experiments.baselines import (
    build_gaussian_shading_official_reference_record,
    build_gaussian_shading_official_reference_schema,
    validate_gaussian_shading_official_reference_records,
)
from paper_experiments.baselines.gaussian_shading_official_reference import REQUIRED_METRIC_FIELDS
from experiments.artifacts.artifact_manifest import build_artifact_manifest
from paper_experiments.runners.external_source_runtime import (
    bind_successful_official_command_execution_evidence,
    build_registered_source_patch_evidence,
    inspect_cuda_with_python_executable,
    load_baseline_registry_item,
    normalize_repository_url,
    prepare_registered_source_checkout,
    run_command,
)
from experiments.runtime.progress import (
    call_runner_with_progress_status,
    emit_progress_status,
    progress_bar,
    run_quiet_subprocess_with_progress,
    update_progress,
)
from experiments.runtime.archive_naming import utc_archive_token
from experiments.runtime.repository_environment import (
    build_runtime_environment_report,
    file_digest,
)
from paper_experiments.runners.model_snapshot_runtime import (
    DEFAULT_SHARED_HUGGING_FACE_SNAPSHOT_ROOT,
    DIFFUSERS_PIPELINE_ALLOW_PATTERNS,
    build_model_snapshot_content,
    build_shared_hugging_face_snapshot_dir,
    ensure_hugging_face_snapshot_files,
    validate_frozen_model_source,
)
from paper_experiments.runners.official_reference_dependency_environment import (
    prepare_official_reference_dependency_environment,
)
from paper_experiments.runners.openclip_checkpoint_runtime import (
    DEFAULT_OPENCLIP_CHECKPOINT_PATH,
    OPENCLIP_CHECKPOINT_FILENAME,
    OPENCLIP_CHECKPOINT_SHA256,
    OPENCLIP_CHECKPOINT_SIZE_BYTES,
    OPENCLIP_ALLOW_PATTERNS,
    OPENCLIP_MODEL_NAME,
    OPENCLIP_REPOSITORY_ID,
    OPENCLIP_REVISION,
    OPENCLIP_SOURCE_NAME,
    OPENCLIP_USAGE_ROLE,
    write_openclip_checkpoint_report,
)

DEFAULT_OUTPUT_DIR = "outputs/gaussian_shading_official_reference"
DEFAULT_DRIVE_OUTPUT_DIR = ""
DEFAULT_SOURCE_DIR = "external_baseline/primary/gaussian_shading/source"
EXPECTED_PATCHED_SOURCE_PATHS = (
    "optim_utils.py",
    "run_gaussian_shading.py",
)
DEFAULT_RUN_NAME = "gaussian_shading_official_reference"
DEFAULT_SAMPLE_COUNT = 700
DEFAULT_OUTPUT_SUBDIR = "official_output"
_OFFICIAL_MODEL_SOURCE = get_model_source("manojb_stable_diffusion_2_1_base")
_PROMPT_DATASET_SOURCE = get_model_source("gustavosta_stable_diffusion_prompts")
DEFAULT_UPSTREAM_OFFICIAL_MODEL_ID = "stabilityai/stable-diffusion-2-1-base"
DEFAULT_OFFICIAL_MODEL_ID = _OFFICIAL_MODEL_SOURCE.repository_id
DEFAULT_OFFICIAL_MODEL_REVISION = _OFFICIAL_MODEL_SOURCE.revision
DEFAULT_PROMPT_DATASET_ID = _PROMPT_DATASET_SOURCE.repository_id
DEFAULT_PROMPT_DATASET_REVISION = _PROMPT_DATASET_SOURCE.revision
DEFAULT_MODEL_SOURCE_NOTE = (
    "Gaussian Shading 官方 README 默认使用 stabilityai/stable-diffusion-2-1-base; "
    "当该模型不可直接访问时, 默认改用公开镜像并保留该模型来源说明。"
)
DEFAULT_LOCAL_MODEL_REPOSITORY_DIR = str(
    build_shared_hugging_face_snapshot_dir(
        DEFAULT_OFFICIAL_MODEL_ID,
        DEFAULT_OFFICIAL_MODEL_REVISION,
    )
)
DEFAULT_DEPENDENCY_PROFILE_ID = "gaussian_shading_official_py38_cu117"
@dataclass(frozen=True)
class GaussianShadingOfficialReferenceConfig:
    """描述 Gaussian Shading 官方参考环境复现与导入所需配置。"""

    output_dir: str = DEFAULT_OUTPUT_DIR
    drive_output_dir: str = field(default_factory=lambda: build_paper_run_config(".").drive_dir("external_baseline_official_reference"))
    source_dir: str = DEFAULT_SOURCE_DIR
    run_name: str = DEFAULT_RUN_NAME
    sample_count: int = DEFAULT_SAMPLE_COUNT
    official_output_subdir: str = DEFAULT_OUTPUT_SUBDIR
    official_model_id: str = DEFAULT_OFFICIAL_MODEL_ID
    official_model_revision: str = DEFAULT_OFFICIAL_MODEL_REVISION
    upstream_official_model_id: str = DEFAULT_UPSTREAM_OFFICIAL_MODEL_ID
    model_source_note: str = DEFAULT_MODEL_SOURCE_NOTE
    dataset_path: str = DEFAULT_PROMPT_DATASET_ID
    dataset_revision: str = DEFAULT_PROMPT_DATASET_REVISION
    fpr: float = 0.000001
    channel_copy: int = 1
    hw_copy: int = 8
    user_number: int = 1000000
    gen_seed: int = 0
    image_length: int = 512
    guidance_scale: float = 7.5
    num_inference_steps: int = 50
    num_inversion_steps: int = 50
    use_chacha: bool = True
    reference_model: str = OPENCLIP_MODEL_NAME
    reference_model_checkpoint_path: str = DEFAULT_OPENCLIP_CHECKPOINT_PATH
    openclip_cache_root: str = DEFAULT_SHARED_HUGGING_FACE_SNAPSHOT_ROOT
    patch_model_repository_layout: bool = True
    prepare_local_model_repository: bool = True
    local_model_repository_dir: str = DEFAULT_LOCAL_MODEL_REPOSITORY_DIR
    patch_model_index_for_pinned_transformers: bool = True
    dependency_profile_id: str = DEFAULT_DEPENDENCY_PROFILE_ID
    require_cuda: bool = True
    timeout_seconds: int = 86400
    enable_workflow_progress_bar: bool = True

    def __post_init__(self) -> None:
        """集中校验官方命令的严格环境边界。"""

        if (
            self.dataset_path != DEFAULT_PROMPT_DATASET_ID
            or self.dataset_revision != DEFAULT_PROMPT_DATASET_REVISION
        ):
            raise ValueError("Gaussian Shading 正式参考必须使用登记的精确 Prompt 数据集 revision")
        if self.reference_model != OPENCLIP_MODEL_NAME:
            raise ValueError("Gaussian Shading 正式参考必须使用登记的 ViT-g-14 OpenCLIP 编码器")
        if Path(self.reference_model_checkpoint_path).name != OPENCLIP_CHECKPOINT_FILENAME:
            raise ValueError("Gaussian Shading OpenCLIP 预训练参数必须指向登记的本地 checkpoint")
        if self.dependency_profile_id != DEFAULT_DEPENDENCY_PROFILE_ID:
            raise ValueError("Gaussian Shading 正式参考必须使用固定依赖 profile")


@dataclass(frozen=True)
class GaussianShadingOfficialReferenceArchiveRecord:
    """记录 Gaussian Shading 官方参考压缩包与 Drive 镜像信息。"""

    archive_path: str
    archive_digest: str
    archive_entry_count: int
    drive_archive_path: str
    drive_archive_digest: str
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """转换为 JSON 兼容字典。"""

        return asdict(self)


def stable_json_text(value: Any) -> str:
    """以稳定顺序序列化 JSON。"""

    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def write_json(path: Path, payload: Any) -> None:
    """写出 JSON 文件并创建父目录。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(stable_json_text(payload), encoding="utf-8")


def read_json(path: Path) -> Any:
    """读取 JSON 文件。"""

    return json.loads(path.read_text(encoding="utf-8-sig"))


def relative_or_absolute(path: Path, root_path: Path) -> str:
    """优先记录相对仓库根目录的路径。"""

    try:
        return path.resolve().relative_to(root_path.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def run_command_with_progress_status(
    command: list[str],
    *,
    cwd: Path,
    timeout_seconds: int,
    progress: object | None = None,
    progress_profile: str = "",
) -> dict[str, Any]:
    """调用可替换 argv runner, 并兼容测试中的轻量 fake。"""

    return call_runner_with_progress_status(
        run_command,
        command,
        cwd=cwd,
        timeout_seconds=timeout_seconds,
        progress=progress,
        progress_profile=progress_profile,
    )


def output_paths(root_path: Path, config: GaussianShadingOfficialReferenceConfig) -> dict[str, Path]:
    """集中构造 Gaussian Shading 官方参考 workflow 的输出路径。"""

    paper_run = build_paper_run_config(root_path)
    configured_output_root = (root_path / config.output_dir).resolve()
    expected_output_root = (root_path / DEFAULT_OUTPUT_DIR).resolve()
    if configured_output_root != expected_output_root:
        raise ValueError("Gaussian Shading 官方参考输出根目录必须使用正式 outputs family")
    output_dir = expected_output_root / paper_run.run_name
    official_output_dir = output_dir / config.official_output_subdir
    return {
        "output_dir": output_dir,
        "official_output_dir": official_output_dir,
        "official_metric_text": official_output_dir / "Identity.txt",
        "official_command_result": output_dir / "gaussian_shading_official_command_result.json",
        "source_prepare_result": output_dir / "gaussian_shading_official_source_prepare_result.json",
        "source_patch_result": output_dir / "gaussian_shading_official_source_patch_result.json",
        "model_repository_prepare_result": output_dir / "gaussian_shading_model_repository_prepare_result.json",
        "openclip_checkpoint_prepare_result": output_dir / "gaussian_shading_openclip_checkpoint_prepare_result.json",
        "dependency_environment_prepare_result": output_dir / "gaussian_shading_dependency_environment_prepare_result.json",
        "official_stdout": output_dir / "gaussian_shading_official_stdout.txt",
        "official_stderr": output_dir / "gaussian_shading_official_stderr.txt",
        "official_metric_summary": output_dir / "gaussian_shading_official_metric_summary.json",
        "reference_schema": output_dir / "gaussian_shading_official_reference_schema.json",
        "reference_records": output_dir / "gaussian_shading_official_reference_records.jsonl",
        "reference_validation": output_dir / "gaussian_shading_official_reference_validation_report.json",
        "environment_report": output_dir / "gaussian_shading_official_reference_environment_report.json",
        "summary": output_dir / "gaussian_shading_official_reference_summary.json",
        "manifest": output_dir / "manifest.local.json",
    }


def prepare_gaussian_shading_dependency_environment(
    root_path: Path,
    config: GaussianShadingOfficialReferenceConfig,
    paths: dict[str, Path],
    progress: object | None = None,
) -> dict[str, Any]:
    """使用共享协议准备并核验固定依赖 profile。"""

    return prepare_official_reference_dependency_environment(
        root_path,
        config.dependency_profile_id,
        paths["dependency_environment_prepare_result"],
        progress=progress,
    )
def source_report(root_path: Path, config: GaussianShadingOfficialReferenceConfig) -> dict[str, Any]:
    """检查官方 Gaussian Shading 源码快照和 requirements 是否存在。"""

    source_dir = (root_path / config.source_dir).resolve()
    entrypoint = source_dir / "run_gaussian_shading.py"
    requirements = source_dir / "requirements.txt"
    return {
        "source_dir": relative_or_absolute(source_dir, root_path),
        "official_entrypoint": relative_or_absolute(entrypoint, root_path),
        "requirements_path": relative_or_absolute(requirements, root_path),
        "source_dir_ready": source_dir.is_dir(),
        "official_entrypoint_ready": entrypoint.is_file(),
        "requirements_ready": requirements.is_file(),
        "requirements_text": requirements.read_text(encoding="utf-8") if requirements.is_file() else "",
    }


def ensure_gaussian_shading_source_available(
    root_path: Path,
    config: GaussianShadingOfficialReferenceConfig,
    paths: dict[str, Path],
    progress: object | None = None,
) -> dict[str, Any]:
    """在 Colab 冷启动环境中按登记表补齐 Gaussian Shading 官方源码。"""

    initial_report = source_report(root_path, config)
    if initial_report["official_entrypoint_ready"]:
        source_prepare_report = {
            **initial_report,
            "source_available": True,
            "source_downloaded": False,
            "source_prepare_reason": "existing_source_entrypoint_found",
        }
        write_json(paths["source_prepare_result"], source_prepare_report)
        return source_prepare_report

    registry_item = load_baseline_registry_item(root_path, "gaussian_shading")
    source_dir = (root_path / config.source_dir).resolve()
    source_entry = source_dir / "run_gaussian_shading.py"
    if source_dir.exists() and any(source_dir.iterdir()):
        source_prepare_report = {
            **initial_report,
            "source_available": False,
            "source_downloaded": False,
            "source_prepare_reason": "source_dir_exists_without_official_entrypoint",
            "official_repository_url": normalize_repository_url(str(registry_item.get("official_repository_url", ""))),
            "official_repository_commit": registry_item.get("official_repository_commit", ""),
        }
        write_json(paths["source_prepare_result"], source_prepare_report)
        return source_prepare_report

    source_dir.parent.mkdir(parents=True, exist_ok=True)
    repository_url = normalize_repository_url(str(registry_item["official_repository_url"]))
    clone_result = run_command_with_progress_status(
        ["git", "clone", repository_url, str(source_dir)],
        cwd=root_path,
        timeout_seconds=300,
        progress=progress,
        progress_profile="operation=gaussian_shading_source_clone",
    )
    checkout_result: dict[str, Any] = {"command": [], "return_code": 0, "stdout": "", "stderr": ""}
    if clone_result["return_code"] == 0 and registry_item.get("official_repository_commit"):
        checkout_result = run_command_with_progress_status(
            ["git", "checkout", str(registry_item["official_repository_commit"])],
            cwd=source_dir,
            timeout_seconds=300,
            progress=progress,
            progress_profile="operation=gaussian_shading_source_checkout",
        )
    refreshed_report = source_report(root_path, config)
    source_prepare_report = {
        **refreshed_report,
        "source_available": source_entry.is_file() and clone_result["return_code"] == 0 and checkout_result["return_code"] == 0,
        "source_downloaded": clone_result["return_code"] == 0,
        "source_prepare_reason": "source_cloned_from_registry",
        "official_repository_url": repository_url,
        "official_repository_commit": registry_item.get("official_repository_commit", ""),
        "clone_return_code": clone_result["return_code"],
        "checkout_return_code": checkout_result["return_code"],
        "clone_result": clone_result,
        "checkout_result": checkout_result,
    }
    write_json(paths["source_prepare_result"], source_prepare_report)
    return source_prepare_report


def patch_gaussian_shading_model_repository_layout(
    root_path: Path,
    config: GaussianShadingOfficialReferenceConfig,
    paths: dict[str, Path],
) -> dict[str, Any]:
    """应用模型镜像兼容补丁并锁定官方 Prompt 数据集 revision."""

    source_dir = (root_path / config.source_dir).resolve()
    entrypoint = source_dir / "run_gaussian_shading.py"
    optim_utils_path = source_dir / "optim_utils.py"
    report = {
        "patch_requested": bool(config.patch_model_repository_layout),
        "patch_applied": False,
        "patch_skipped": False,
        "official_entrypoint": relative_or_absolute(entrypoint, root_path),
        "optim_utils_path": relative_or_absolute(optim_utils_path, root_path),
        "official_model_id": config.official_model_id,
        "official_model_revision": config.official_model_revision,
        "upstream_official_model_id": config.upstream_official_model_id,
        "model_source_note": config.model_source_note,
        "prompt_dataset_source": _PROMPT_DATASET_SOURCE.to_dict(),
        "prompt_dataset_repository_id": config.dataset_path,
        "prompt_dataset_revision": config.dataset_revision,
        "patch_reason": "mirror_model_layout_and_exact_prompt_dataset_revision",
        "patch_items": [],
    }
    if not config.patch_model_repository_layout:
        report.update({"patch_skipped": True, "patch_skip_reason": "patch_model_repository_layout_disabled"})
        write_json(paths["source_patch_result"], report)
        return report
    if not entrypoint.is_file() or not optim_utils_path.is_file():
        report.update({"patch_skipped": True, "patch_skip_reason": "official_source_file_missing"})
        write_json(paths["source_patch_result"], report)
        return report

    source_text = entrypoint.read_text(encoding="utf-8")
    optim_text = optim_utils_path.read_text(encoding="utf-8")
    before_digest = file_digest(entrypoint)
    optim_before_digest = file_digest(optim_utils_path)
    patched_source_text = source_text
    patched_optim_text = optim_text
    marker = "# SLM-WM: 公开镜像没有 fp16 分支, 因此从 main 分支加载模型权重。"
    target = "            revision='fp16',\n"
    if marker not in patched_source_text and target in patched_source_text:
        patched_source_text = patched_source_text.replace(target, f"            {marker}\n")
        report["patch_items"].append("remove_fp16_revision_branch")

    dataset_targets = (
        "load_dataset(args.dataset)",
        "load_dataset(args.dataset_path)",
    )
    for dataset_target in dataset_targets:
        if dataset_target in patched_optim_text:
            dataset_argument = dataset_target.removeprefix("load_dataset(").removesuffix(")")
            patched_optim_text = patched_optim_text.replace(
                dataset_target,
                f"load_dataset({dataset_argument}, revision='{config.dataset_revision}')",
            )
    if patched_optim_text != optim_text:
        report["patch_items"].append("pin_prompt_dataset_revision")

    if patched_source_text == source_text and patched_optim_text == optim_text:
        report.update(
            {
                "patch_skipped": True,
                "patch_skip_reason": "source_runtime_patch_already_present_or_targets_missing",
                "entrypoint_digest_before": before_digest,
                "entrypoint_digest_after": before_digest,
                "optim_utils_digest_before": optim_before_digest,
                "optim_utils_digest_after": optim_before_digest,
            }
        )
        write_json(paths["source_patch_result"], report)
        return report
    if patched_source_text != source_text:
        entrypoint.write_text(patched_source_text, encoding="utf-8", newline="\n")
    if patched_optim_text != optim_text:
        optim_utils_path.write_text(patched_optim_text, encoding="utf-8", newline="\n")
    report.update(
        {
            "patch_applied": True,
            "entrypoint_digest_before": before_digest,
            "entrypoint_digest_after": file_digest(entrypoint),
            "optim_utils_digest_before": optim_before_digest,
            "optim_utils_digest_after": file_digest(optim_utils_path),
        }
    )
    write_json(paths["source_patch_result"], report)
    return report


def prepare_gaussian_shading_model_repository(
    root_path: Path,
    config: GaussianShadingOfficialReferenceConfig,
    paths: dict[str, Path],
    progress: object | None = None,
) -> dict[str, Any]:
    """准备本地模型目录并补齐固定 diffusers 版本所需的 model_index 项。"""

    validate_frozen_model_source(
        config.official_model_id,
        config.official_model_revision,
        expected_repository_id=DEFAULT_OFFICIAL_MODEL_ID,
        expected_revision=DEFAULT_OFFICIAL_MODEL_REVISION,
    )

    local_model_path = Path(config.local_model_repository_dir).expanduser()
    model_index_path = local_model_path / "model_index.json"
    report: dict[str, Any] = {
        "local_model_repository_requested": bool(config.prepare_local_model_repository),
        "local_model_repository_ready": False,
        "local_model_repository_path": str(local_model_path),
        "official_model_id": config.official_model_id,
        "official_model_revision": config.official_model_revision,
        "upstream_official_model_id": config.upstream_official_model_id,
        "effective_official_model_id": config.official_model_id,
        "model_index_patch_requested": bool(config.patch_model_index_for_pinned_transformers),
        "model_index_patch_applied": False,
        "model_source_note": config.model_source_note,
    }
    if not config.prepare_local_model_repository:
        report.update({"local_model_repository_skipped": True, "skip_reason": "prepare_local_model_repository_disabled"})
        write_json(paths["model_repository_prepare_result"], report)
        return report

    try:
        emit_progress_status(
            progress,
            profile=f"operation=gaussian_shading_model_snapshot_materialize model={config.official_model_id}",
        )
        snapshot_materialization = ensure_hugging_face_snapshot_files(
            local_model_path,
            report_path=paths["model_repository_prepare_result"],
            repository_id=config.official_model_id,
            revision=config.official_model_revision,
            allow_patterns=DIFFUSERS_PIPELINE_ALLOW_PATTERNS,
            token=os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or None,
        )
    except Exception as error:
        report.update(
            {
                "failure_reason": "model_snapshot_materialization_failed",
                "snapshot_validation_error": f"{type(error).__name__}:{error}",
            }
        )
        write_json(paths["model_repository_prepare_result"], report)
        return report

    report["download_result"] = {**snapshot_materialization, "return_code": 0}
    report["model_snapshot_allow_patterns"] = list(DIFFUSERS_PIPELINE_ALLOW_PATTERNS)
    if not model_index_path.is_file():
        report.update({"local_model_repository_ready": False, "failure_reason": "model_index_missing_after_download"})
        write_json(paths["model_repository_prepare_result"], report)
        return report

    before_digest = file_digest(model_index_path)
    model_index = read_json(model_index_path)
    report["model_index_digest_before"] = before_digest
    feature_extractor = model_index.get("feature_extractor")
    if (
        config.patch_model_index_for_pinned_transformers
        and isinstance(feature_extractor, list)
        and len(feature_extractor) >= 2
        and feature_extractor[0] == "transformers"
        and feature_extractor[1] == "CLIPImageProcessor"
    ):
        feature_extractor[1] = "CLIPFeatureExtractor"
        write_json(model_index_path, model_index)
        report["model_index_patch_applied"] = True
        report["model_index_patch_reason"] = "pinned_diffusers_uses_clip_feature_extractor"
    elif not config.patch_model_index_for_pinned_transformers:
        report["model_index_patch_skip_reason"] = "patch_model_index_for_pinned_transformers_disabled"
    else:
        report["model_index_patch_skip_reason"] = "model_index_feature_extractor_already_compatible"

    report.update(
        {
            "model_index_digest_after": file_digest(model_index_path),
            "model_index_feature_extractor": read_json(model_index_path).get("feature_extractor"),
            "local_model_repository_ready": True,
            "effective_official_model_id": str(local_model_path),
            "model_snapshot_content": build_model_snapshot_content(
                local_model_path,
                repository_id=config.official_model_id,
                revision=config.official_model_revision,
                allow_patterns=DIFFUSERS_PIPELINE_ALLOW_PATTERNS,
            ),
        }
    )
    write_json(paths["model_repository_prepare_result"], report)
    return report


def build_official_command(
    root_path: Path,
    config: GaussianShadingOfficialReferenceConfig,
    paths: dict[str, Path],
    dependency_python_executable: str | Path,
) -> list[str]:
    """构造 Gaussian Shading 官方固定依赖 profile 入口命令。"""

    source_dir = (root_path / config.source_dir).resolve()
    entrypoint = source_dir / "run_gaussian_shading.py"
    python_executable = str(dependency_python_executable).strip()
    if not python_executable:
        raise ValueError("Gaussian Shading 官方命令必须使用已核验的隔离 Python 解释器")
    output_path = str(paths["official_output_dir"]) + os.sep
    command = [
        python_executable,
        str(entrypoint),
        "--num",
        str(max(1, int(config.sample_count))),
        "--fpr",
        str(config.fpr),
        "--channel_copy",
        str(config.channel_copy),
        "--hw_copy",
        str(config.hw_copy),
        "--user_number",
        str(config.user_number),
        "--gen_seed",
        str(config.gen_seed),
        "--image_length",
        str(config.image_length),
        "--guidance_scale",
        str(config.guidance_scale),
        "--num_inference_steps",
        str(config.num_inference_steps),
        "--num_inversion_steps",
        str(config.num_inversion_steps),
        "--dataset_path",
        str(config.dataset_path),
        "--model_path",
        str(config.official_model_id),
        "--output_path",
        output_path,
    ]
    if config.use_chacha:
        command.append("--chacha")
    command.extend(
        [
            "--reference_model",
            config.reference_model,
            "--reference_model_pretrain",
            config.reference_model_checkpoint_path,
        ]
    )
    return command


def parse_metric_text(text: str, sample_count: int) -> dict[str, Any]:
    """从本次官方命令生成的 Identity.txt 解析科学指标."""

    patterns = {
        "detection_true_positive_rate": r"tpr_detection:\s*([0-9eE+\-.]+)",
        "traceability_true_positive_rate": r"tpr_traceability:\s*([0-9eE+\-.]+)",
        "mean_bit_accuracy": r"mean_acc:\s*([0-9eE+\-.]+)",
        "std_bit_accuracy": r"std_acc:\s*([0-9eE+\-.]+)",
        "mean_clip_score": r"mean_clip_score:\s*([0-9eE+\-.]+)",
        "std_clip_score": r"std_clip_score:\s*([0-9eE+\-.]+)",
    }
    metrics: dict[str, Any] = {
        "sample_count": int(sample_count),
        "positive_count": int(sample_count),
    }
    for field_name, pattern in patterns.items():
        matches = re.findall(pattern, text)
        if matches:
            metrics[field_name] = float(matches[-1])
    return metrics


def normalize_metric_summary(payload: dict[str, Any], sample_count: int) -> dict[str, Any]:
    """规范化本次官方输出字段, 不为缺失科学指标填充 0."""

    normalized: dict[str, Any] = {
        "sample_count": int(sample_count),
        "positive_count": int(sample_count),
    }
    aliases = {
        "tpr_detection": "detection_true_positive_rate",
        "tpr_traceability": "traceability_true_positive_rate",
        "mean_acc": "mean_bit_accuracy",
        "std_acc": "std_bit_accuracy",
    }
    for key, value in payload.items():
        target_key = aliases.get(str(key), str(key))
        if target_key in REQUIRED_METRIC_FIELDS:
            normalized[target_key] = value
    return normalized


def metric_summary_has_complete_scientific_values(
    metric_summary: dict[str, Any],
    sample_count: int,
) -> bool:
    """要求全部正式科学指标均来自本次输出且样本计数精确一致."""

    if any(field_name not in metric_summary for field_name in REQUIRED_METRIC_FIELDS):
        return False
    try:
        normalized_sample_count = int(metric_summary["sample_count"])
        positive_count = int(metric_summary["positive_count"])
        scientific_values = {
            field_name: float(metric_summary[field_name])
            for field_name in REQUIRED_METRIC_FIELDS
            if not field_name.endswith("count")
        }
    except (TypeError, ValueError):
        return False
    return all(
        (
            normalized_sample_count == int(sample_count),
            positive_count == int(sample_count),
            all(math.isfinite(value) for value in scientific_values.values()),
            all(
                0.0 <= scientific_values[field_name] <= 1.0
                for field_name in (
                    "detection_true_positive_rate",
                    "traceability_true_positive_rate",
                    "mean_bit_accuracy",
                    "std_bit_accuracy",
                    "std_clip_score",
                )
            ),
            -1.0 <= scientific_values["mean_clip_score"] <= 1.0,
        )
    )


def run_official_command(
    root_path: Path,
    config: GaussianShadingOfficialReferenceConfig,
    paths: dict[str, Path],
    dependency_python_executable: str | Path,
    progress: object | None = None,
    device_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """执行本次 Gaussian Shading 官方命令并保存独立运行证据."""

    paths["official_metric_text"].unlink(missing_ok=True)
    resolved_device_report = device_report or inspect_cuda_with_python_executable(
        dependency_python_executable,
        require_cuda=config.require_cuda,
        cwd=root_path,
    )
    if config.require_cuda and resolved_device_report.get("decision") != "pass":
        error_text = ",".join(
            str(reason)
            for reason in resolved_device_report.get("failure_reasons", ())
        ) or "isolated_cuda_inspection_failed"
        paths["official_stdout"].write_text("", encoding="utf-8")
        paths["official_stderr"].write_text(error_text, encoding="utf-8")
        result = {
            "official_command_requested": True,
            "official_command": [],
            "return_code": 97,
            "stdout_path": relative_or_absolute(paths["official_stdout"], root_path),
            "stderr_path": relative_or_absolute(paths["official_stderr"], root_path),
            "error": error_text,
        }
        write_json(paths["official_command_result"], result)
        return result
    source_status = source_report(root_path, config)
    if not source_status["official_entrypoint_ready"]:
        paths["official_stdout"].write_text("", encoding="utf-8")
        paths["official_stderr"].write_text("gaussian_shading_official_source_entrypoint_missing", encoding="utf-8")
        result = {
            "official_command_requested": True,
            "official_command": build_official_command(
                root_path,
                config,
                paths,
                dependency_python_executable,
            ),
            "return_code": 96,
            "stdout_path": relative_or_absolute(paths["official_stdout"], root_path),
            "stderr_path": relative_or_absolute(paths["official_stderr"], root_path),
            "error": "gaussian_shading_official_source_entrypoint_missing",
        }
        write_json(paths["official_command_result"], result)
        return result
    paths["official_output_dir"].mkdir(parents=True, exist_ok=True)
    command = build_official_command(
        root_path,
        config,
        paths,
        dependency_python_executable,
    )
    source_dir = (root_path / config.source_dir).resolve()
    env = os.environ.copy()
    env.setdefault("WANDB_MODE", "disabled")
    try:
        completed = run_quiet_subprocess_with_progress(
            command,
            cwd=source_dir,
            env=env,
            timeout_seconds=int(config.timeout_seconds),
            progress=progress,
            progress_profile=f"operation=gaussian_shading_official_command samples={config.sample_count}",
        )
    except Exception as error:
        paths["official_stdout"].write_text("", encoding="utf-8")
        paths["official_stderr"].write_text(f"{type(error).__name__}:{error}", encoding="utf-8")
        result = {
            "official_command_requested": True,
            "official_command": command,
            "return_code": 98,
            "stdout_path": relative_or_absolute(paths["official_stdout"], root_path),
            "stderr_path": relative_or_absolute(paths["official_stderr"], root_path),
            "error": f"{type(error).__name__}:{error}",
        }
        write_json(paths["official_command_result"], result)
        return result
    paths["official_stdout"].write_text(completed.stdout, encoding="utf-8")
    paths["official_stderr"].write_text(completed.stderr, encoding="utf-8")
    result = {
        "official_command_requested": True,
        "official_command": command,
        "return_code": int(completed.returncode),
        "stdout_path": relative_or_absolute(paths["official_stdout"], root_path),
        "stderr_path": relative_or_absolute(paths["official_stderr"], root_path),
        "official_metric_text_path": relative_or_absolute(paths["official_metric_text"], root_path),
    }
    result = bind_successful_official_command_execution_evidence(
        result,
        baseline_id="gaussian_shading",
        command=command,
        working_directory=source_dir,
        dependency_python_executable=dependency_python_executable,
        cuda_inspection_report=resolved_device_report,
    )
    write_json(paths["official_command_result"], result)
    return result


def write_official_command_skip_result(
    root_path: Path,
    paths: dict[str, Path],
    *,
    return_code: int,
    reason: str,
) -> dict[str, Any]:
    """在前置环境不可用时写出官方命令跳过诊断。"""

    paths["official_stdout"].write_text("", encoding="utf-8")
    paths["official_stderr"].write_text(reason, encoding="utf-8")
    result = {
        "official_command_requested": True,
        "official_command": [],
        "return_code": int(return_code),
        "stdout_path": relative_or_absolute(paths["official_stdout"], root_path),
        "stderr_path": relative_or_absolute(paths["official_stderr"], root_path),
        "error": reason,
    }
    write_json(paths["official_command_result"], result)
    return result


def model_repository_source_is_exact(model_repository_report: dict[str, Any]) -> bool:
    """核验本次官方模型快照的仓库、revision 和内容摘要."""

    snapshot_content = model_repository_report.get("model_snapshot_content")
    return isinstance(snapshot_content, dict) and all(
        (
            model_repository_report.get("local_model_repository_ready") is True,
            model_repository_report.get("official_model_id") == DEFAULT_OFFICIAL_MODEL_ID,
            model_repository_report.get("official_model_revision") == DEFAULT_OFFICIAL_MODEL_REVISION,
            snapshot_content.get("repository_id") == DEFAULT_OFFICIAL_MODEL_ID,
            snapshot_content.get("revision") == DEFAULT_OFFICIAL_MODEL_REVISION,
            snapshot_content.get("allow_patterns") == sorted(DIFFUSERS_PIPELINE_ALLOW_PATTERNS),
            len(str(snapshot_content.get("snapshot_content_digest", ""))) == 64,
        )
    )


def openclip_checkpoint_source_is_exact(openclip_report: dict[str, Any]) -> bool:
    """核验本次 OpenCLIP checkpoint 的全部登记来源字段."""

    snapshot_content = openclip_report.get("model_snapshot_content")
    expected_files = [
        {
            "path": OPENCLIP_CHECKPOINT_FILENAME,
            "size_bytes": OPENCLIP_CHECKPOINT_SIZE_BYTES,
            "sha256": OPENCLIP_CHECKPOINT_SHA256,
        }
    ]
    return isinstance(snapshot_content, dict) and all(
        (
            openclip_report.get("openclip_checkpoint_requested") is True,
            openclip_report.get("openclip_checkpoint_ready") is True,
            openclip_report.get("openclip_source_name") == OPENCLIP_SOURCE_NAME,
            openclip_report.get("openclip_usage_role") == OPENCLIP_USAGE_ROLE,
            openclip_report.get("openclip_model_name") == OPENCLIP_MODEL_NAME,
            openclip_report.get("openclip_repository_id") == OPENCLIP_REPOSITORY_ID,
            openclip_report.get("openclip_revision") == OPENCLIP_REVISION,
            openclip_report.get("openclip_checkpoint_filename") == OPENCLIP_CHECKPOINT_FILENAME,
            openclip_report.get("openclip_checkpoint_sha256") == OPENCLIP_CHECKPOINT_SHA256,
            openclip_report.get("openclip_checkpoint_size_bytes") == OPENCLIP_CHECKPOINT_SIZE_BYTES,
            len(str(openclip_report.get("openclip_snapshot_content_digest", ""))) == 64,
            snapshot_content.get("repository_id") == OPENCLIP_REPOSITORY_ID,
            snapshot_content.get("revision") == OPENCLIP_REVISION,
            snapshot_content.get("allow_patterns") == list(OPENCLIP_ALLOW_PATTERNS),
            snapshot_content.get("file_count") == 1,
            snapshot_content.get("files") == expected_files,
            snapshot_content.get("snapshot_content_digest")
            == openclip_report.get("openclip_snapshot_content_digest"),
        )
    )


def build_reference_record_report(
    root_path: Path,
    config: GaussianShadingOfficialReferenceConfig,
    paths: dict[str, Path],
    metric_summary: dict[str, Any],
    source_status: dict[str, Any],
    dependency_environment_report: dict[str, Any],
    model_repository_report: dict[str, Any],
    openclip_report: dict[str, Any],
    official_report: dict[str, Any],
) -> dict[str, Any]:
    """仅为本次成功命令和完整科学证据构造 governed record."""

    schema = build_gaussian_shading_official_reference_schema()
    write_json(paths["reference_schema"], schema)
    paths["reference_records"].write_text("", encoding="utf-8")
    paths["official_metric_summary"].unlink(missing_ok=True)

    official_command_succeeded = all(
        (
            official_report.get("official_command_requested") is True,
            int(official_report.get("return_code", -1)) == 0,
        )
    )
    scientific_metrics_complete = metric_summary_has_complete_scientific_values(
        metric_summary,
        config.sample_count,
    )
    source_ready = all(
        (
            source_status.get("official_entrypoint_ready") is True,
            source_status.get("source_identity_ready") is True,
            source_status.get("source_worktree_exact") is True,
            source_status.get("prompt_dataset_repository_id") == DEFAULT_PROMPT_DATASET_ID,
            source_status.get("prompt_dataset_revision") == DEFAULT_PROMPT_DATASET_REVISION,
        )
    )
    environment_ready = dependency_environment_report.get("dependency_environment_ready") is True
    model_source_ready = model_repository_source_is_exact(model_repository_report)
    openclip_source_ready = openclip_checkpoint_source_is_exact(openclip_report)
    governed_import_ready = all(
        (
            official_command_succeeded,
            scientific_metrics_complete,
            source_ready,
            environment_ready,
            model_source_ready,
            openclip_source_ready,
        )
    )
    if not governed_import_ready:
        validation = validate_gaussian_shading_official_reference_records([])
        write_json(paths["reference_validation"], validation)
        return {
            "record_count": 0,
            "validation": validation,
            "official_command_succeeded": official_command_succeeded,
            "scientific_metrics_complete": scientific_metrics_complete,
            "source_ready": source_ready,
            "environment_ready": environment_ready,
            "model_source_ready": model_source_ready,
            "openclip_source_ready": openclip_source_ready,
        }

    write_json(paths["official_metric_summary"], metric_summary)
    local_evidence_paths: list[str] = []
    for candidate in (
        paths["official_metric_summary"],
        paths["official_command_result"],
        paths["official_stdout"],
        paths["official_stderr"],
        paths["official_metric_text"],
        paths["source_prepare_result"],
        paths["source_patch_result"],
        paths["model_repository_prepare_result"],
        paths["openclip_checkpoint_prepare_result"],
        paths["dependency_environment_prepare_result"],
        paths["environment_report"],
    ):
        if candidate.is_file():
            local_evidence_paths.append(relative_or_absolute(candidate, root_path))
    result_source = relative_or_absolute(paths["official_metric_summary"], root_path)
    result_digest = file_digest(paths["official_metric_summary"])
    snapshot_content = model_repository_report["model_snapshot_content"]
    record = build_gaussian_shading_official_reference_record(
        official_command_requested=True,
        official_command_return_code=0,
        official_entrypoint=str(source_status.get("official_entrypoint", "")),
        official_repository_commit=str(source_status.get("official_repository_commit", "")),
        official_environment_profile=str(
            dependency_environment_report.get("dependency_environment_profile_id", "")
        ),
        baseline_result_source=result_source,
        baseline_result_source_digest=result_digest,
        evidence_paths=sorted(set(local_evidence_paths)),
        source_provenance={
            **source_status,
            "official_model_repository_id": model_repository_report["official_model_id"],
            "official_model_revision": model_repository_report["official_model_revision"],
            "model_snapshot_content_digest": snapshot_content["snapshot_content_digest"],
            **openclip_report,
        },
        metric_values=metric_summary,
        ready_flags={
            "official_command_succeeded": official_command_succeeded,
            "official_source_ready": source_ready,
            "source_identity_ready": True,
            "source_worktree_exact": True,
            "official_environment_report_ready": environment_ready,
            "official_result_summary_ready": scientific_metrics_complete,
            "model_source_ready": model_source_ready,
            "openclip_source_ready": openclip_source_ready,
            "governed_import_ready": governed_import_ready,
        },
    )
    paths["reference_records"].write_text(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    validation = validate_gaussian_shading_official_reference_records([record])
    write_json(paths["reference_validation"], validation)
    return {
        "record_count": 1,
        "validation": validation,
        "record_digest": record["reference_record_digest"],
        "official_command_succeeded": official_command_succeeded,
        "scientific_metrics_complete": scientific_metrics_complete,
        "source_ready": source_ready,
        "environment_ready": environment_ready,
        "model_source_ready": model_source_ready,
        "openclip_source_ready": openclip_source_ready,
    }


def write_gaussian_shading_official_reference_outputs(
    config: GaussianShadingOfficialReferenceConfig,
    root: str | Path = ".",
) -> dict[str, Any]:
    """执行 Gaussian Shading 官方参考 workflow 并写出摘要、清单和受治理导入记录。"""

    root_path = Path(root).resolve()
    formal_execution_run_lock = (
        repository_environment.require_published_formal_execution_lock(root_path)
    )
    paths = output_paths(root_path, config)
    paths["output_dir"].mkdir(parents=True, exist_ok=True)
    paths["official_output_dir"].mkdir(parents=True, exist_ok=True)
    with progress_bar(10, desc="gaussian shading official reference", enabled=config.enable_workflow_progress_bar) as run_progress:
        effective_config = config
        source_status = ensure_gaussian_shading_source_available(root_path, effective_config, paths, progress=run_progress)
        source_identity_report = prepare_registered_source_checkout(
            root_path,
            "gaussian_shading",
            (root_path / effective_config.source_dir).resolve(),
        )
        source_status.update(source_identity_report)
        write_json(paths["source_prepare_result"], source_status)
        update_progress(run_progress, profile="operation=ensure_gaussian_shading_source")
        emit_progress_status(run_progress, profile="operation=patch_gaussian_shading_source status=running")
        source_patch_report = patch_gaussian_shading_model_repository_layout(root_path, effective_config, paths)
        source_patch_evidence = build_registered_source_patch_evidence(
            root_path,
            "gaussian_shading",
            (root_path / effective_config.source_dir).resolve(),
            EXPECTED_PATCHED_SOURCE_PATHS,
        )
        source_patch_report.update(source_patch_evidence)
        source_status.update(source_patch_evidence)
        source_status.update(
            {
                "prompt_dataset_source": _PROMPT_DATASET_SOURCE.to_dict(),
                "prompt_dataset_repository_id": effective_config.dataset_path,
                "prompt_dataset_revision": effective_config.dataset_revision,
            }
        )
        write_json(paths["source_patch_result"], source_patch_report)
        write_json(paths["source_prepare_result"], source_status)
        update_progress(run_progress, profile="operation=patch_gaussian_shading_source")
        dependency_environment_report = prepare_gaussian_shading_dependency_environment(
            root_path,
            config,
            paths,
            progress=run_progress,
        )
        update_progress(run_progress, profile="operation=prepare_gaussian_shading_dependency_environment")
        emit_progress_status(run_progress, profile="operation=ensure_cuda status=running")
        device_report = inspect_cuda_with_python_executable(
            str(dependency_environment_report.get("dependency_python_executable", "")),
            require_cuda=effective_config.require_cuda,
            cwd=root_path,
        )
        update_progress(run_progress, profile="operation=ensure_cuda")
        should_prepare_model_repository = (
            dependency_environment_report.get("dependency_environment_ready") is True
        )
        model_repository_config = effective_config if should_prepare_model_repository else replace(effective_config, prepare_local_model_repository=False)
        model_repository_report = prepare_gaussian_shading_model_repository(root_path, model_repository_config, paths, progress=run_progress)
        update_progress(run_progress, profile="operation=prepare_gaussian_shading_model_repository")
        if model_repository_report.get("local_model_repository_ready"):
            effective_config = replace(effective_config, official_model_id=str(model_repository_report["effective_official_model_id"]))
        openclip_report = write_openclip_checkpoint_report(
            paths["openclip_checkpoint_prepare_result"],
            requested=(
                should_prepare_model_repository
                and model_repository_report.get("local_model_repository_ready") is True
            ),
            cache_root=effective_config.openclip_cache_root,
            token=os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or None,
        )
        if openclip_report.get("openclip_checkpoint_ready"):
            effective_config = replace(
                effective_config,
                reference_model_checkpoint_path=str(openclip_report["openclip_checkpoint_path"]),
            )
        if not dependency_environment_report.get("dependency_environment_ready"):
            official_report = write_official_command_skip_result(
                root_path,
                paths,
                return_code=95,
                reason="gaussian_shading_dependency_environment_prepare_failed",
            )
        elif not model_repository_report.get("local_model_repository_ready"):
            official_report = write_official_command_skip_result(
                root_path,
                paths,
                return_code=94,
                reason="gaussian_shading_model_snapshot_prepare_failed",
            )
        elif not openclip_report.get("openclip_checkpoint_ready"):
            official_report = write_official_command_skip_result(
                root_path,
                paths,
                return_code=93,
                reason="gaussian_shading_openclip_checkpoint_prepare_failed",
            )
        else:
            official_report = run_official_command(
                root_path,
                effective_config,
                paths,
                dependency_environment_report["dependency_python_executable"],
                progress=run_progress,
                device_report=device_report,
            )
        update_progress(run_progress, profile="operation=gaussian_shading_official_command")
        emit_progress_status(run_progress, profile="operation=parse_gaussian_shading_metrics status=running")
        command_metrics: dict[str, Any] = {}
        if official_report.get("return_code") == 0 and paths["official_metric_text"].is_file():
            command_metrics = parse_metric_text(
                paths["official_metric_text"].read_text(encoding="utf-8", errors="ignore"),
                effective_config.sample_count,
            )
        metric_summary = (
            normalize_metric_summary(command_metrics, effective_config.sample_count)
            if command_metrics
            else {}
        )
        update_progress(run_progress, profile="operation=parse_gaussian_shading_metrics")
        emit_progress_status(run_progress, profile="operation=write_environment_report status=running")
        environment_report = build_runtime_environment_report(
            "workflow_orchestrator",
            verified_formal_execution_lock=formal_execution_run_lock,
        )
        environment_report["gaussian_shading_official_reference_device_report"] = device_report
        environment_report["gaussian_shading_official_reference_source_report"] = source_status
        environment_report["gaussian_shading_official_reference_source_patch_report"] = source_patch_report
        environment_report["gaussian_shading_official_reference_model_repository_report"] = model_repository_report
        environment_report["gaussian_shading_official_reference_openclip_checkpoint_report"] = openclip_report
        environment_report["gaussian_shading_official_reference_dependency_environment_report"] = dependency_environment_report
        write_json(paths["environment_report"], environment_report)
        update_progress(run_progress, profile="operation=write_environment_report")
        emit_progress_status(run_progress, profile="operation=build_reference_record status=running")
        record_report = build_reference_record_report(
            root_path,
            effective_config,
            paths,
            metric_summary,
            source_status,
            dependency_environment_report,
            model_repository_report,
            openclip_report,
            official_report,
        )
        update_progress(run_progress, profile=f"operation=build_reference_record records={record_report.get('record_count', 0)}")
    validation = record_report["validation"]
    model_snapshot_content = model_repository_report.get("model_snapshot_content")
    model_snapshot_digest = (
        str(model_snapshot_content.get("snapshot_content_digest", ""))
        if isinstance(model_snapshot_content, dict)
        else ""
    )
    model_snapshot_scope_ready = isinstance(model_snapshot_content, dict) and all(
        (
            model_snapshot_content.get("repository_id") == DEFAULT_OFFICIAL_MODEL_ID,
            model_snapshot_content.get("revision") == DEFAULT_OFFICIAL_MODEL_REVISION,
            model_snapshot_content.get("allow_patterns")
            == sorted(DIFFUSERS_PIPELINE_ALLOW_PATTERNS),
        )
    )
    model_source_ready = model_repository_source_is_exact(model_repository_report)
    source_revision_ready = all(
        (
            source_status.get("source_identity_ready") is True,
            source_status.get("source_worktree_exact") is True,
            len(str(source_status.get("official_repository_commit", ""))) == 40,
            len(str(source_status.get("source_patch_sha256", ""))) == 64,
            len(str(source_status.get("source_worktree_digest", ""))) == 64,
            source_status.get("prompt_dataset_repository_id") == DEFAULT_PROMPT_DATASET_ID,
            source_status.get("prompt_dataset_revision") == DEFAULT_PROMPT_DATASET_REVISION,
        )
    )
    openclip_source_ready = openclip_checkpoint_source_is_exact(openclip_report)
    official_command_succeeded = all(
        (
            official_report.get("official_command_requested") is True,
            int(official_report.get("return_code", -1)) == 0,
        )
    )
    scientific_metrics_complete = metric_summary_has_complete_scientific_values(
        metric_summary,
        effective_config.sample_count,
    )
    run_ready = all(
        (
            validation.get("reference_import_ready") is True,
            official_command_succeeded,
            scientific_metrics_complete,
        dependency_environment_report.get("dependency_environment_ready") is True,
            model_source_ready,
            source_revision_ready,
            openclip_source_ready,
        )
    )
    unsupported_reason = "" if run_ready else "gaussian_shading_official_reference_result_missing_or_invalid"
    if dependency_environment_report.get("dependency_environment_ready") is not True:
        unsupported_reason = "dependency_profile_environment_not_ready"
    elif not official_command_succeeded:
        unsupported_reason = "official_command_failed_formal_archive_blocked"
    elif not scientific_metrics_complete:
        unsupported_reason = "official_command_scientific_metrics_incomplete"
    paper_run = build_paper_run_config(root_path)
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "baseline_id": "gaussian_shading",
        "run_decision": "pass" if run_ready else "fail",
        "gaussian_shading_official_reference_ready": run_ready,
        "official_command_requested": bool(official_report.get("official_command_requested")),
        "official_command_return_code": int(official_report.get("return_code", -1)),
        "official_command_succeeded": official_command_succeeded,
        "scientific_metrics_complete": scientific_metrics_complete,
        "scientific_metric_fields": list(REQUIRED_METRIC_FIELDS),
        "sample_count": int(effective_config.sample_count),
        "paper_claim_scale": paper_run.run_name,
        "target_fpr": paper_run.target_fpr,
        "dependency_environment_requested": bool(dependency_environment_report.get("dependency_environment_requested")),
        "dependency_environment_ready": bool(dependency_environment_report.get("dependency_environment_ready")),
        "dependency_environment_profile_id": str(dependency_environment_report.get("dependency_environment_profile_id", "")),
        "dependency_profile_ready": bool(dependency_environment_report.get("dependency_profile_ready")),
        "dependency_profile_id": str(dependency_environment_report.get("dependency_profile_id", "")),
        "dependency_lock_ready": bool(dependency_environment_report.get("dependency_lock_ready")),
        "dependency_environment_materialized": bool(dependency_environment_report.get("dependency_environment_materialized")),
        "dependency_environment_report_valid": bool(dependency_environment_report.get("dependency_environment_report_valid")),
        "dependency_profile_digest": str(dependency_environment_report.get("dependency_profile_digest", "")),
        "dependency_lock_digest": str(dependency_environment_report.get("dependency_lock_digest", "")),
        "source_patch_applied": bool(source_patch_report.get("patch_applied")),
        "source_revision_ready": source_revision_ready,
        "official_repository_commit": str(source_status.get("official_repository_commit", "")),
        "source_patch_sha256": str(source_status.get("source_patch_sha256", "")),
        "source_worktree_digest": str(source_status.get("source_worktree_digest", "")),
        "prompt_dataset_repository_id": str(source_status.get("prompt_dataset_repository_id", "")),
        "prompt_dataset_revision": str(source_status.get("prompt_dataset_revision", "")),
        "local_model_repository_ready": bool(model_repository_report.get("local_model_repository_ready")),
        "model_source_ready": model_source_ready,
        "model_source_repository_id": str(model_repository_report.get("official_model_id", "")),
        "model_source_revision": str(model_repository_report.get("official_model_revision", "")),
        "model_snapshot_content_digest": model_snapshot_digest,
        "model_snapshot_scope_ready": model_snapshot_scope_ready,
        "openclip_source_ready": openclip_source_ready,
        "openclip_source_name": str(openclip_report.get("openclip_source_name", "")),
        "openclip_usage_role": str(openclip_report.get("openclip_usage_role", "")),
        "openclip_model_name": str(openclip_report.get("openclip_model_name", "")),
        "openclip_repository_id": str(openclip_report.get("openclip_repository_id", "")),
        "openclip_revision": str(openclip_report.get("openclip_revision", "")),
        "openclip_checkpoint_filename": str(openclip_report.get("openclip_checkpoint_filename", "")),
        "openclip_checkpoint_sha256": str(openclip_report.get("openclip_checkpoint_sha256", "")),
        "openclip_checkpoint_size_bytes": int(openclip_report.get("openclip_checkpoint_size_bytes", 0)),
        "openclip_snapshot_content_digest": str(openclip_report.get("openclip_snapshot_content_digest", "")),
        "model_index_patch_applied": bool(model_repository_report.get("model_index_patch_applied")),
        "official_model_id": effective_config.official_model_id,
        "upstream_official_model_id": effective_config.upstream_official_model_id,
        "governed_reference_record_count": int(record_report.get("record_count", 0)),
        "reference_import_ready": bool(validation.get("reference_import_ready")),
        "main_table_eligible": False,
        "supports_paper_claim": False,
        "unsupported_reason": unsupported_reason,
        "summary_path": relative_or_absolute(paths["summary"], root_path),
        "environment_report_path": relative_or_absolute(paths["environment_report"], root_path),
        "reference_records_path": relative_or_absolute(paths["reference_records"], root_path),
        "reference_validation_path": relative_or_absolute(paths["reference_validation"], root_path),
        "official_metric_text_path": relative_or_absolute(paths["official_metric_text"], root_path),
        "metadata": {
            "source_report": source_status,
            "source_patch_report": source_patch_report,
            "model_repository_report": model_repository_report,
            "openclip_checkpoint_report": openclip_report,
            "dependency_environment_report": dependency_environment_report,
            "official_report": official_report,
            "validation": validation,
        },
    }
    write_json(paths["summary"], summary)
    output_paths_for_manifest = [
        relative_or_absolute(paths["summary"], root_path),
        relative_or_absolute(paths["environment_report"], root_path),
        relative_or_absolute(paths["reference_schema"], root_path),
        relative_or_absolute(paths["reference_records"], root_path),
        relative_or_absolute(paths["reference_validation"], root_path),
    ]
    for optional_path in (
        paths["official_command_result"],
        paths["official_stdout"],
        paths["official_stderr"],
        paths["official_metric_summary"],
        paths["official_metric_text"],
        paths["source_prepare_result"],
        paths["source_patch_result"],
        paths["model_repository_prepare_result"],
        paths["openclip_checkpoint_prepare_result"],
        paths["dependency_environment_prepare_result"],
    ):
        if optional_path.exists():
            output_paths_for_manifest.append(relative_or_absolute(optional_path, root_path))
    formal_execution_run_lock = repository_environment.validate_formal_execution_lock_pair(
        formal_execution_run_lock,
        repository_environment.require_published_formal_execution_lock(root_path),
        formal_execution_run_lock["formal_execution_commit"],
    )
    manifest = build_artifact_manifest(
        artifact_id="gaussian_shading_official_reference_manifest",
        artifact_type="local_manifest",
        input_paths=(relative_or_absolute(root_path / config.source_dir, root_path),),
        output_paths=tuple(output_paths_for_manifest + [relative_or_absolute(paths["manifest"], root_path)]),
        config=asdict(effective_config),
        code_version=formal_execution_run_lock["formal_execution_commit"],
        rebuild_command="调用 paper_experiments.runners.gaussian_shading_official_reference",
        metadata={
            "run_decision": summary["run_decision"],
            "gaussian_shading_official_reference_ready": run_ready,
            "main_table_eligible": False,
            "supports_paper_claim": False,
        },
    ).to_dict()
    manifest["formal_execution_run_lock"] = formal_execution_run_lock
    write_json(paths["manifest"], manifest)
    return summary


def build_default_config() -> GaussianShadingOfficialReferenceConfig:
    """从环境变量构造默认 Colab 运行配置。"""

    paper_run = build_paper_run_config(".")
    return GaussianShadingOfficialReferenceConfig(
        output_dir=os.environ.get("SLM_WM_GAUSSIAN_SHADING_OFFICIAL_OUTPUT_DIR", DEFAULT_OUTPUT_DIR),
        drive_output_dir=os.environ.get("SLM_WM_GAUSSIAN_SHADING_OFFICIAL_DRIVE_OUTPUT_DIR", paper_run.drive_dir("external_baseline_official_reference")),
        source_dir=os.environ.get("SLM_WM_GAUSSIAN_SHADING_OFFICIAL_SOURCE_DIR", DEFAULT_SOURCE_DIR),
        run_name=os.environ.get("SLM_WM_GAUSSIAN_SHADING_OFFICIAL_RUN_NAME", DEFAULT_RUN_NAME),
        sample_count=resolve_count_from_environment("SLM_WM_GAUSSIAN_SHADING_OFFICIAL_SAMPLE_COUNT", default_value=paper_run.sample_count),
        official_output_subdir=os.environ.get("SLM_WM_GAUSSIAN_SHADING_OFFICIAL_OUTPUT_SUBDIR", DEFAULT_OUTPUT_SUBDIR),
        official_model_id=os.environ.get("SLM_WM_GAUSSIAN_SHADING_OFFICIAL_MODEL_ID", DEFAULT_OFFICIAL_MODEL_ID),
        official_model_revision=os.environ.get(
            "SLM_WM_GAUSSIAN_SHADING_OFFICIAL_MODEL_REVISION",
            DEFAULT_OFFICIAL_MODEL_REVISION,
        ),
        upstream_official_model_id=os.environ.get(
            "SLM_WM_GAUSSIAN_SHADING_UPSTREAM_OFFICIAL_MODEL_ID", DEFAULT_UPSTREAM_OFFICIAL_MODEL_ID
        ),
        model_source_note=os.environ.get("SLM_WM_GAUSSIAN_SHADING_MODEL_SOURCE_NOTE", DEFAULT_MODEL_SOURCE_NOTE),
        dataset_path=DEFAULT_PROMPT_DATASET_ID,
        dataset_revision=DEFAULT_PROMPT_DATASET_REVISION,
        fpr=float(os.environ.get("SLM_WM_GAUSSIAN_SHADING_FPR", "0.000001")),
        channel_copy=int(os.environ.get("SLM_WM_GAUSSIAN_SHADING_CHANNEL_COPY", "1")),
        hw_copy=int(os.environ.get("SLM_WM_GAUSSIAN_SHADING_HW_COPY", "8")),
        user_number=int(os.environ.get("SLM_WM_GAUSSIAN_SHADING_USER_NUMBER", "1000000")),
        gen_seed=int(os.environ.get("SLM_WM_GAUSSIAN_SHADING_GEN_SEED", "0")),
        image_length=int(os.environ.get("SLM_WM_GAUSSIAN_SHADING_IMAGE_LENGTH", "512")),
        guidance_scale=float(os.environ.get("SLM_WM_GAUSSIAN_SHADING_GUIDANCE_SCALE", "7.5")),
        num_inference_steps=int(os.environ.get("SLM_WM_GAUSSIAN_SHADING_NUM_INFERENCE_STEPS", "50")),
        num_inversion_steps=int(os.environ.get("SLM_WM_GAUSSIAN_SHADING_NUM_INVERSION_STEPS", "50")),
        use_chacha=os.environ.get("SLM_WM_GAUSSIAN_SHADING_USE_CHACHA", "1") != "0",
        reference_model=OPENCLIP_MODEL_NAME,
        reference_model_checkpoint_path=DEFAULT_OPENCLIP_CHECKPOINT_PATH,
        openclip_cache_root=os.environ.get(
            "SLM_WM_OPENCLIP_CACHE_ROOT",
            DEFAULT_SHARED_HUGGING_FACE_SNAPSHOT_ROOT,
        ),
        patch_model_repository_layout=os.environ.get("SLM_WM_GAUSSIAN_SHADING_PATCH_MODEL_REPOSITORY_LAYOUT", "1") != "0",
        prepare_local_model_repository=os.environ.get("SLM_WM_GAUSSIAN_SHADING_PREPARE_LOCAL_MODEL_REPOSITORY", "1") != "0",
        local_model_repository_dir=os.environ.get(
            "SLM_WM_GAUSSIAN_SHADING_LOCAL_MODEL_REPOSITORY_DIR", DEFAULT_LOCAL_MODEL_REPOSITORY_DIR
        ),
        patch_model_index_for_pinned_transformers=os.environ.get(
            "SLM_WM_GAUSSIAN_SHADING_PATCH_MODEL_INDEX_FOR_PINNED_TRANSFORMERS", "1"
        )
        != "0",
        dependency_profile_id=DEFAULT_DEPENDENCY_PROFILE_ID,
        require_cuda=os.environ.get("SLM_WM_GAUSSIAN_SHADING_OFFICIAL_REQUIRE_CUDA", "1") != "0",
        timeout_seconds=int(os.environ.get("SLM_WM_GAUSSIAN_SHADING_OFFICIAL_TIMEOUT_SECONDS", "86400")),
        enable_workflow_progress_bar=os.environ.get("SLM_WM_ENABLE_WORKFLOW_PROGRESS_BAR", "1") != "0",
    )


def run_default_gaussian_shading_official_reference_plan(root: str | Path = ".") -> dict[str, Any]:
    """运行默认 Gaussian Shading 官方参考 workflow。"""

    return write_gaussian_shading_official_reference_outputs(config=build_default_config(), root=root)


def collect_package_entries(root_path: Path, output_dir: Path, archive_path: Path) -> tuple[Path, ...]:
    """仅收集当前论文层级的 Gaussian Shading outputs family 文件。"""

    entries: list[Path] = []
    if output_dir.exists():
        for path in sorted(output_dir.rglob("*")):
            if path.is_file() and path.resolve() != archive_path.resolve() and path.suffix.lower() != ".zip":
                entries.append(path)
    unique_entries: list[Path] = []
    for entry in entries:
        if entry not in unique_entries:
            unique_entries.append(entry)
    return tuple(unique_entries)


def package_gaussian_shading_official_reference_outputs(
    root: str | Path = ".",
    output_dir: str = DEFAULT_OUTPUT_DIR,
    drive_output_dir: str | None = None,
    archive_name: str | None = None,
) -> GaussianShadingOfficialReferenceArchiveRecord:
    """打包 Gaussian Shading 官方参考产物并镜像到 Google Drive。"""

    root_path = Path(root).resolve()
    formal_execution_package_lock = (
        repository_environment.require_published_formal_execution_lock(root_path)
    )
    paper_run = build_paper_run_config(root_path)
    resolved_drive_output_dir = drive_output_dir or paper_run.drive_dir("external_baseline_official_reference")
    configured_output_root = (root_path / output_dir).resolve()
    expected_output_root = (root_path / DEFAULT_OUTPUT_DIR).resolve()
    if configured_output_root != expected_output_root:
        raise ValueError("Gaussian Shading 官方参考打包根目录必须使用正式 outputs family")
    source_dir = expected_output_root / paper_run.run_name
    required_runtime_paths = (
        source_dir / "gaussian_shading_official_reference_summary.json",
        source_dir / "manifest.local.json",
        source_dir / "gaussian_shading_official_reference_records.jsonl",
        source_dir / "gaussian_shading_official_reference_validation_report.json",
    )
    missing_runtime_paths = [path for path in required_runtime_paths if not path.is_file()]
    if missing_runtime_paths:
        raise FileNotFoundError("Gaussian Shading 正式参考输出不完整, 不得打包")
    run_summary = read_json(required_runtime_paths[0])
    run_manifest = read_json(required_runtime_paths[1])
    formal_execution_run_lock = repository_environment.validate_formal_execution_lock_pair(
        run_manifest.get("formal_execution_run_lock"),
        formal_execution_package_lock,
        run_manifest.get("code_version"),
    )
    if not all(
        (
            run_summary.get("run_decision") == "pass",
            run_summary.get("gaussian_shading_official_reference_ready") is True,
            run_summary.get("reference_import_ready") is True,
            run_summary.get("official_command_requested") is True,
            run_summary.get("official_command_return_code") == 0,
            run_summary.get("official_command_succeeded") is True,
            run_summary.get("scientific_metrics_complete") is True,
            run_summary.get("baseline_id") == "gaussian_shading",
            run_summary.get("dependency_environment_profile_id") == DEFAULT_DEPENDENCY_PROFILE_ID,
            run_summary.get("dependency_profile_id") == DEFAULT_DEPENDENCY_PROFILE_ID,
            run_summary.get("dependency_profile_ready") is True,
            run_summary.get("dependency_lock_ready") is True,
            run_summary.get("dependency_environment_materialized") is True,
            run_summary.get("dependency_environment_report_valid") is True,
            re.fullmatch(r"[0-9a-f]{64}", str(run_summary.get("dependency_profile_digest", ""))) is not None,
            re.fullmatch(r"[0-9a-f]{64}", str(run_summary.get("dependency_lock_digest", ""))) is not None,
            run_summary.get("model_source_ready") is True,
            run_summary.get("model_snapshot_scope_ready") is True,
            run_summary.get("openclip_source_ready") is True,
            run_summary.get("openclip_source_name") == OPENCLIP_SOURCE_NAME,
            run_summary.get("openclip_usage_role") == OPENCLIP_USAGE_ROLE,
            run_summary.get("openclip_model_name") == OPENCLIP_MODEL_NAME,
            run_summary.get("openclip_repository_id") == OPENCLIP_REPOSITORY_ID,
            run_summary.get("openclip_revision") == OPENCLIP_REVISION,
            run_summary.get("openclip_checkpoint_filename") == OPENCLIP_CHECKPOINT_FILENAME,
            run_summary.get("openclip_checkpoint_sha256") == OPENCLIP_CHECKPOINT_SHA256,
            run_summary.get("openclip_checkpoint_size_bytes") == OPENCLIP_CHECKPOINT_SIZE_BYTES,
            len(str(run_summary.get("openclip_snapshot_content_digest", ""))) == 64,
            run_summary.get("model_source_repository_id") == DEFAULT_OFFICIAL_MODEL_ID,
            run_summary.get("model_source_revision") == DEFAULT_OFFICIAL_MODEL_REVISION,
            len(str(run_summary.get("model_snapshot_content_digest", ""))) == 64,
            run_summary.get("paper_claim_scale") == paper_run.run_name,
            float(run_summary.get("target_fpr", -1.0)) == paper_run.target_fpr,
        )
    ):
        raise RuntimeError("Gaussian Shading 正式参考身份或 ready 门禁未通过")
    resolved_archive_name = archive_name or (
        "external_baseline_official_reference_package_gaussian_shading_"
        f"{utc_archive_token()}_{formal_execution_package_lock['formal_execution_commit'][:7]}.zip"
    )
    if (
        Path(resolved_archive_name).name != resolved_archive_name
        or not resolved_archive_name.startswith(
            "external_baseline_official_reference_package_gaussian_shading_"
        )
        or Path(resolved_archive_name).suffix.lower() != ".zip"
    ):
        raise ValueError("Gaussian Shading archive_name 未匹配正式命名")
    archive_path = source_dir / resolved_archive_name
    package_manifest_path = source_dir / "gaussian_shading_official_reference_package_input_manifest.json"
    summary_path = source_dir / "gaussian_shading_official_reference_archive_summary.json"
    manifest_path = source_dir / "gaussian_shading_official_reference_archive_manifest.local.json"
    for stale_path in (package_manifest_path, summary_path, manifest_path):
        if stale_path.exists():
            stale_path.unlink()
    content_entries = collect_package_entries(root_path, source_dir, archive_path)
    entries = tuple((*content_entries, package_manifest_path, summary_path, manifest_path))
    package_manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "paper_run_name": paper_run.run_name,
        "target_fpr": paper_run.target_fpr,
        "baseline_id": "gaussian_shading",
        "formal_execution_run_lock": formal_execution_run_lock,
        "formal_execution_package_lock": formal_execution_package_lock,
        "entry_paths": [entry.relative_to(root_path).as_posix() for entry in content_entries],
        "entry_sha256": {
            entry.relative_to(root_path).as_posix(): file_digest(entry)
            for entry in content_entries
        },
        "entry_count": len(entries),
        "embedded_digest_scope": "external_summary_records_final_archive_digest",
    }
    write_json(package_manifest_path, package_manifest)
    preliminary_record = GaussianShadingOfficialReferenceArchiveRecord(
        archive_path=relative_or_absolute(archive_path, root_path),
        archive_digest="",
        archive_entry_count=len(entries),
        drive_archive_path=str(Path(resolved_drive_output_dir).expanduser() / resolved_archive_name),
        drive_archive_digest="",
        metadata={
            "drive_output_dir": str(Path(resolved_drive_output_dir).expanduser()),
            "embedded_digest_scope": "external_summary_records_final_archive_digest",
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    write_json(summary_path, preliminary_record.to_dict())
    archive_manifest = build_artifact_manifest(
        artifact_id="gaussian_shading_official_reference_archive_manifest",
        artifact_type="local_manifest",
        input_paths=tuple(
            [entry.relative_to(root_path).as_posix() for entry in content_entries]
            + [package_manifest_path.relative_to(root_path).as_posix()]
        ),
        output_paths=(
            archive_path.relative_to(root_path).as_posix(),
            summary_path.relative_to(root_path).as_posix(),
            manifest_path.relative_to(root_path).as_posix(),
        ),
        config={
            "archive_name": resolved_archive_name,
            "paper_run_name": paper_run.run_name,
            "target_fpr": paper_run.target_fpr,
            "drive_output_dir": str(Path(resolved_drive_output_dir).expanduser()),
        },
        code_version=formal_execution_package_lock["formal_execution_commit"],
        rebuild_command="调用 paper_experiments.runners.gaussian_shading_official_reference",
        metadata={
            "embedded_digest_scope": "external_summary_records_final_archive_digest",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "paper_run_name": paper_run.run_name,
            "target_fpr": paper_run.target_fpr,
            "baseline_id": "gaussian_shading",
            "main_table_eligible": False,
        },
    ).to_dict()
    archive_manifest["formal_execution_run_lock"] = formal_execution_run_lock
    archive_manifest["formal_execution_package_lock"] = formal_execution_package_lock
    write_json(manifest_path, archive_manifest)
    with ZipFile(archive_path, mode="w", compression=ZIP_DEFLATED) as archive:
        for entry in entries:
            archive.write(entry, entry.relative_to(root_path).as_posix())
    try:
        final_package_lock = (
            repository_environment.require_published_formal_execution_lock(root_path)
        )
        repository_environment.validate_formal_execution_lock_pair(
            formal_execution_package_lock,
            final_package_lock,
            formal_execution_package_lock["formal_execution_commit"],
        )
    except Exception:
        archive_path.unlink(missing_ok=True)
        raise
    drive_dir = Path(resolved_drive_output_dir).expanduser()
    drive_dir.mkdir(parents=True, exist_ok=True)
    mirrored_path = drive_dir / resolved_archive_name
    shutil.copy2(archive_path, mirrored_path)
    record = GaussianShadingOfficialReferenceArchiveRecord(
        archive_path=relative_or_absolute(archive_path, root_path),
        archive_digest=file_digest(archive_path),
        archive_entry_count=len(entries),
        drive_archive_path=str(mirrored_path),
        drive_archive_digest=file_digest(mirrored_path),
        metadata={"drive_output_dir": str(drive_dir), "generated_at": datetime.now(timezone.utc).isoformat()},
    )
    write_json(summary_path, record.to_dict())
    archive_manifest.setdefault("metadata", {})["archive_digest"] = record.archive_digest
    archive_manifest.setdefault("metadata", {})["drive_archive_digest"] = record.drive_archive_digest
    write_json(manifest_path, archive_manifest)
    return record
