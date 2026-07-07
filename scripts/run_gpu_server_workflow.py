"""运行独立 GPU 服务器论文 workflow。

该脚本提供不依赖 Colab Notebook 和 Google Drive 挂载的命令行入口。它只在
当前进程内设置服务器本地结果根目录, 然后调用已有 repository workflow
helper 和打包函数。现有 Colab Notebook 入口仍继续使用原有
`/content/drive/MyDrive/SLM/...` 落盘约定, 不受该脚本影响。
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.protocol.paper_run_config import RUN_DEFAULTS, build_paper_run_config, normalize_paper_run_name
from experiments.runtime.archive_naming import build_workflow_archive_name
from paper_experiments.runners.external_baseline_method_faithful import DEFAULT_FORMAL_IMAGE_ATTACK_FAMILIES

WorkflowRunner = Callable[[str | Path], dict[str, Any]]
WorkflowPackager = Callable[..., Any]

PRIMARY_BASELINE_IDS = ("tree_ring", "gaussian_shading", "shallow_diffuse", "t2smark")
OFFICIAL_REFERENCE_IDS = ("tree_ring", "gaussian_shading", "shallow_diffuse", "t2smark")
WORKFLOW_ALIASES = {
    "external_baseline_tree_ring": ("external_baseline_method_faithful", "tree_ring"),
    "external_baseline_gaussian_shading": ("external_baseline_method_faithful", "gaussian_shading"),
    "external_baseline_shallow_diffuse": ("external_baseline_method_faithful", "shallow_diffuse"),
    "external_baseline_t2smark": ("external_baseline_method_faithful", "t2smark"),
}
WORKFLOW_CHOICES = (
    "attention_geometry",
    "attention_latent_injection",
    "aligned_rescoring",
    "threshold_calibration",
    "real_attack_evaluation",
    "conventional_geometric_attack_evaluation",
    "dataset_level_quality",
    "external_baseline_method_faithful",
    *WORKFLOW_ALIASES.keys(),
    "official_reference_tree_ring",
    "official_reference_gaussian_shading",
    "official_reference_shallow_diffuse",
    "official_reference_t2smark",
)
PUBLISH_DIR_NAMES = {
    "attention_geometry": "attention_geometry",
    "attention_latent_injection": "attention_latent_injection",
    "aligned_rescoring": "aligned_rescoring",
    "threshold_calibration": "threshold_calibration",
    "real_attack_evaluation": "real_attack_evaluation",
    "conventional_geometric_attack_evaluation": "conventional_geometric_attack_evaluation",
    "dataset_level_quality": "dataset_level_quality",
    "external_baseline_method_faithful": "external_baseline_method_faithful",
    "official_reference_tree_ring": "external_baseline_official_reference",
    "official_reference_gaussian_shading": "external_baseline_official_reference",
    "official_reference_shallow_diffuse": "external_baseline_official_reference",
    "official_reference_t2smark": "external_baseline_official_reference",
}
LEGACY_RUNTIME_DIR_NAME = "gpu_server_runtime_assets"


@dataclass(frozen=True)
class ServerWorkflowSelection:
    """记录服务器入口实际运行的 workflow 与可选 baseline。"""

    requested_workflow_name: str
    workflow_name: str
    baseline_id: str

    def to_dict(self) -> dict[str, Any]:
        """转换为 JSON 兼容字典。"""

        return asdict(self)


def stable_json_text(value: Any) -> str:
    """把 JSON 兼容对象转为稳定文本。"""

    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def write_json(path: Path, payload: Any) -> None:
    """写出 JSON 文件并自动创建父目录。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(stable_json_text(payload), encoding="utf-8")


def file_digest(path: Path) -> str:
    """计算文件 SHA-256 摘要。"""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def set_env(name: str, value: str | int | float) -> None:
    """写入当前进程环境变量, 让已有 workflow helper 读取同一配置。"""

    os.environ[name] = str(value)


def set_default_env(name: str, value: str | int | float) -> None:
    """仅在调用方没有显式覆盖时写入默认环境变量。"""

    os.environ.setdefault(name, str(value))


def target_fpr_text(value: float) -> str:
    """生成稳定的 FPR 文本表示。"""

    return f"{float(value):g}"


def default_result_root(paper_run_name: str) -> Path:
    """返回服务器脚本默认的本地结果根目录。"""

    return Path("outputs") / "gpu_server_results" / paper_run_name


def resolve_workflow_selection(requested_workflow_name: str, baseline_id: str = "") -> ServerWorkflowSelection:
    """解析 workflow alias 与 baseline 选择。"""

    if requested_workflow_name in WORKFLOW_ALIASES:
        workflow_name, alias_baseline_id = WORKFLOW_ALIASES[requested_workflow_name]
        return ServerWorkflowSelection(requested_workflow_name, workflow_name, alias_baseline_id)
    resolved_baseline_id = baseline_id.strip()
    if requested_workflow_name == "external_baseline_method_faithful" and resolved_baseline_id not in PRIMARY_BASELINE_IDS:
        raise ValueError("external_baseline_method_faithful 必须通过 --baseline-id 指定主表 baseline")
    return ServerWorkflowSelection(requested_workflow_name, requested_workflow_name, resolved_baseline_id)


def configure_common_server_environment(
    *,
    root: str | Path,
    paper_run_name: str,
    result_root: str | Path,
    sample_count_token: str,
    target_fpr_override: str,
) -> dict[str, Any]:
    """配置服务器本地论文运行环境。

    该函数不调用 Colab 专用配置 helper, 因此不会把结果根目录重置到
    `/content/drive/MyDrive/SLM`。它属于服务器入口配置解析层, 用于复用
    现有 repository workflow 的环境变量接口。
    """

    root_path = Path(root).resolve()
    normalized_run_name = normalize_paper_run_name(paper_run_name)
    defaults = RUN_DEFAULTS[normalized_run_name]
    resolved_result_root = Path(result_root).expanduser()
    if not resolved_result_root.is_absolute():
        resolved_result_root = (root_path / resolved_result_root).resolve()
    resolved_result_root.mkdir(parents=True, exist_ok=True)

    set_env("SLM_WM_PAPER_RUN_NAME", normalized_run_name)
    set_env("SLM_WM_PROMPT_SET", defaults["prompt_set"])
    set_env("SLM_WM_PROMPT_FILE", defaults["prompt_file"])
    set_env("SLM_WM_DRIVE_RESULT_ROOT", resolved_result_root.as_posix())
    set_env("SLM_WM_PAPER_RUN_SAMPLE_COUNT", sample_count_token)
    set_env("SLM_WM_PAPER_RUN_TARGET_FPR", target_fpr_override or defaults["target_fpr"])
    os.environ.pop("SLM_WM_PAPER_RUN_MINIMUM_CLEAN_NEGATIVE_COUNT", None)
    os.environ.pop("SLM_WM_PAPER_RUN_DATASET_QUALITY_MINIMUM_COUNT", None)

    paper_run = build_paper_run_config(root_path)
    resolved_target_fpr = target_fpr_text(paper_run.target_fpr)
    set_env("SLM_WM_PAPER_RUN_TARGET_FPR", resolved_target_fpr)
    set_env("SLM_WM_PROTOCOL_PROFILE", f"{paper_run.run_name}_fixed_fpr_{resolved_target_fpr.replace('.', '_')}")
    set_env("SLM_WM_PAPER_RUN_EXPECTED_SAMPLE_COUNT", paper_run.sample_count)
    set_env("SLM_WM_PAPER_RUN_MINIMUM_CLEAN_NEGATIVE_COUNT", paper_run.minimum_clean_negative_count)
    set_env("SLM_WM_PAPER_RUN_DATASET_QUALITY_MINIMUM_COUNT", paper_run.dataset_level_quality_minimum_count)
    return {
        "root": root_path.as_posix(),
        "paper_run": paper_run.to_dict(),
        "result_root": resolved_result_root.as_posix(),
        "target_fpr": resolved_target_fpr,
        "sample_count_token": sample_count_token,
    }


def configure_workflow_environment(selection: ServerWorkflowSelection, result_root: Path) -> None:
    """配置单个服务器 workflow 所需的本地路径。"""

    workflow_name = selection.workflow_name
    sample_count_token = os.environ["SLM_WM_PAPER_RUN_SAMPLE_COUNT"]
    set_env("SLM_WM_ATTENTION_GEOMETRY_DRIVE_DIR", (result_root / "attention_geometry").as_posix())
    set_env("SLM_WM_ATTENTION_INJECTION_DRIVE_DIR", (result_root / "attention_latent_injection").as_posix())
    set_env("SLM_WM_ALIGNED_RESCORING_DRIVE_DIR", (result_root / "aligned_rescoring").as_posix())
    set_env("SLM_WM_THRESHOLD_CALIBRATION_DRIVE_DIR", (result_root / "threshold_calibration").as_posix())
    set_env("SLM_WM_REAL_ATTACK_EVALUATION_DRIVE_DIR", (result_root / "real_attack_evaluation").as_posix())
    set_env("SLM_WM_DRIVE_OUTPUT_DIR", (result_root / PUBLISH_DIR_NAMES.get(workflow_name, workflow_name)).as_posix())
    set_env("SLM_WM_ATTENTION_SUBSPACE_RECORDS", sample_count_token)
    set_env("SLM_WM_ALIGNED_RESCORING_SUBSPACE_RECORDS", sample_count_token)
    set_env("SLM_WM_ALIGNED_RESCORING_CARRIER_COUNT", sample_count_token)
    set_env("SLM_WM_REAL_ATTACK_SOURCE_COUNT", sample_count_token)
    set_env("SLM_WM_CONVENTIONAL_GEOMETRIC_ATTACK_SOURCE_COUNT", sample_count_token)
    set_default_env("SLM_WM_REAL_ATTACK_SOURCE_IMAGE_DIR", "outputs/aligned_rescoring/aligned_images")
    set_default_env("SLM_WM_ENABLE_PIPELINE_PROGRESS_BAR", "0")
    set_default_env("SLM_WM_ENABLE_ATTACK_PROGRESS_BAR", "1")
    set_default_env("SLM_WM_ENABLE_CARRIER_PROGRESS_BAR", "1")
    set_default_env("SLM_WM_ENABLE_WORKFLOW_PROGRESS_BAR", "1")
    set_default_env("WANDB_MODE", "disabled")

    if workflow_name == "dataset_level_quality":
        set_env("SLM_WM_DATASET_QUALITY_DRIVE_DIR", (result_root / "dataset_level_quality").as_posix())
        set_env("SLM_WM_FORMAL_MIN_SAMPLE_COUNT", build_paper_run_config(".").dataset_level_quality_minimum_count)
    if workflow_name == "external_baseline_method_faithful":
        configure_external_baseline_environment(selection.baseline_id, result_root, sample_count_token)
    if workflow_name.startswith("official_reference_"):
        configure_official_reference_environment(workflow_name, result_root, sample_count_token)


def configure_external_baseline_environment(baseline_id: str, result_root: Path, sample_count_token: str) -> None:
    """配置 method-faithful baseline 的服务器本地环境。"""

    set_env("SLM_WM_PRIMARY_BASELINE_METHODS", baseline_id)
    set_env("SLM_WM_EXTERNAL_BASELINE_DRIVE_OUTPUT_DIR", (result_root / "external_baseline_method_faithful").as_posix())
    set_env("SLM_WM_EXTERNAL_BASELINE_PRIOR_DRIVE_DIR", (result_root / "external_baseline_method_faithful").as_posix())
    set_env("SLM_WM_T2SMARK_ROBUST_TEST_NUM", sample_count_token)
    set_env("SLM_WM_PRIMARY_BASELINE_MAX_SAMPLES", sample_count_token)
    set_default_env("SLM_WM_T2SMARK_MODEL_ID", "stabilityai/stable-diffusion-3.5-medium")
    set_default_env("SLM_WM_T2SMARK_RUN_NAME", "t2smark_sd35_medium_method_faithful")
    set_default_env("SLM_WM_T2SMARK_CLIP_TEST_NUM", "0")
    set_default_env("SLM_WM_T2SMARK_NUM_INFERENCE_STEPS", "8")
    set_default_env("SLM_WM_T2SMARK_NUM_INVERSION_STEPS", "3")
    set_default_env("SLM_WM_T2SMARK_GUIDANCE_SCALE", "4.0")
    set_default_env("SLM_WM_EXTERNAL_BASELINE_REUSE_EXISTING", "1")
    set_default_env("SLM_WM_EXTERNAL_BASELINE_REUSE_DRIVE", "1")
    set_default_env("SLM_WM_EXTERNAL_BASELINE_REQUIRE_CUDA", "1")
    set_default_env("SLM_WM_TREE_RING_ADAPTER_MODE", "method_faithful_sd35")
    set_default_env("SLM_WM_GAUSSIAN_SHADING_ADAPTER_MODE", "method_faithful_sd35")
    set_default_env("SLM_WM_SHALLOW_DIFFUSE_ADAPTER_MODE", "method_faithful_sd35")
    set_default_env("SLM_WM_TREE_RING_ATTACK_FAMILIES", DEFAULT_FORMAL_IMAGE_ATTACK_FAMILIES)
    set_default_env("SLM_WM_GAUSSIAN_SHADING_ATTACK_FAMILIES", DEFAULT_FORMAL_IMAGE_ATTACK_FAMILIES)
    set_default_env("SLM_WM_SHALLOW_DIFFUSE_ATTACK_FAMILIES", DEFAULT_FORMAL_IMAGE_ATTACK_FAMILIES)


def configure_official_reference_environment(workflow_name: str, result_root: Path, sample_count_token: str) -> None:
    """配置 official reference 的服务器本地 legacy 环境路径。"""

    method_id = workflow_name.removeprefix("official_reference_")
    prefix = method_id.upper()
    if method_id == "tree_ring":
        prefix = "TREE_RING"
    if method_id == "gaussian_shading":
        prefix = "GAUSSIAN_SHADING"
    if method_id == "shallow_diffuse":
        prefix = "SHALLOW_DIFFUSE"
    if method_id == "t2smark":
        set_env("SLM_WM_T2SMARK_FULL_MAIN_DRIVE_OUTPUT_DIR", (result_root / "external_baseline_official_reference").as_posix())
        set_env("SLM_WM_T2SMARK_FULL_MAIN_PROMPT_LIMIT", sample_count_token)
        set_env("SLM_WM_T2SMARK_FULL_MAIN_TARGET_FPR", os.environ["SLM_WM_PAPER_RUN_TARGET_FPR"])
        set_env("SLM_WM_T2SMARK_FULL_MAIN_FIXED_FPR_READY", "1")
        set_env("SLM_WM_T2SMARK_FULL_MAIN_ATTACK_MATRIX_READY", "1")
        set_default_env("SLM_WM_T2SMARK_MODEL_ID", "stabilityai/stable-diffusion-3.5-medium")
        set_default_env("SLM_WM_T2SMARK_FULL_MAIN_REQUIRE_CUDA", "1")
        return

    runtime_root = result_root / LEGACY_RUNTIME_DIR_NAME / method_id
    model_dir = runtime_root / "model_repository" / "stable_diffusion_2_1_base"
    legacy_env_dir = runtime_root / "legacy_env"
    set_env(f"SLM_WM_{prefix}_OFFICIAL_DRIVE_OUTPUT_DIR", (result_root / "external_baseline_official_reference").as_posix())
    set_env(f"SLM_WM_{prefix}_OFFICIAL_SAMPLE_COUNT", sample_count_token)
    set_default_env(f"SLM_WM_{prefix}_OFFICIAL_RUN_COMMAND", "1")
    set_default_env(f"SLM_WM_{prefix}_OFFICIAL_REQUIRE_CUDA", "1")
    set_default_env(f"SLM_WM_{prefix}_OFFICIAL_TIMEOUT_SECONDS", "86400")
    set_default_env(f"SLM_WM_{prefix}_OFFICIAL_MODEL_ID", "Manojb/stable-diffusion-2-1-base")
    set_default_env(f"SLM_WM_{prefix}_UPSTREAM_OFFICIAL_MODEL_ID", "stabilityai/stable-diffusion-2-1-base")
    set_default_env(f"SLM_WM_{prefix}_PATCH_MODEL_REPOSITORY_LAYOUT", "1")
    set_default_env(f"SLM_WM_{prefix}_PREPARE_LOCAL_MODEL_REPOSITORY", "1")
    set_default_env(f"SLM_WM_{prefix}_PATCH_MODEL_INDEX_FOR_LEGACY_TRANSFORMERS", "1")
    set_default_env(f"SLM_WM_{prefix}_LOCAL_MODEL_REPOSITORY_DIR", model_dir.as_posix())
    set_default_env(f"SLM_WM_{prefix}_OFFICIAL_PYTHON_EXECUTABLE", "")
    set_default_env(f"SLM_WM_{prefix}_PREPARE_LEGACY_ENV", "1")
    set_default_env(f"SLM_WM_{prefix}_LEGACY_ENV_PREFIX", legacy_env_dir.as_posix())
    set_default_env(f"SLM_WM_{prefix}_MICROMAMBA_PATH", (runtime_root / "bin" / "micromamba").as_posix())

    if method_id == "gaussian_shading":
        set_default_env("SLM_WM_GAUSSIAN_SHADING_LEGACY_PYTHON_VERSION", "3.8")
        set_default_env("SLM_WM_GAUSSIAN_SHADING_STRICT_OFFICIAL_ENV", "1")
        set_default_env("SLM_WM_GAUSSIAN_SHADING_ALLOW_COMPATIBLE_ENV_FALLBACK", "1")
        set_default_env("SLM_WM_GAUSSIAN_SHADING_USE_CHACHA", "1")
    else:
        set_default_env(f"SLM_WM_{prefix}_LEGACY_PYTHON_VERSION", "3.9")


def workflow_publish_dir(result_root: Path, workflow_name: str) -> Path:
    """返回某个 workflow 的服务器发布目录。"""

    return result_root / PUBLISH_DIR_NAMES[workflow_name]


def load_workflow_runner(workflow_name: str) -> WorkflowRunner:
    """延迟导入运行函数, 避免服务器入口加载无关重依赖。"""

    if workflow_name == "attention_geometry":
        from experiments.runners.attention_geometry_capture import run_default_attention_geometry_plan

        return run_default_attention_geometry_plan
    if workflow_name == "attention_latent_injection":
        from experiments.runners.attention_latent_injection import run_default_attention_latent_injection_plan

        return run_default_attention_latent_injection_plan
    if workflow_name == "aligned_rescoring":
        from experiments.runners.aligned_rescoring import run_default_aligned_rescoring_plan

        return run_default_aligned_rescoring_plan
    if workflow_name == "external_baseline_method_faithful":
        from paper_experiments.runners.external_baseline_method_faithful import run_default_external_baseline_method_faithful_plan

        return run_default_external_baseline_method_faithful_plan
    if workflow_name == "official_reference_tree_ring":
        from paper_experiments.runners.tree_ring_official_reference import run_default_tree_ring_official_reference_plan

        return run_default_tree_ring_official_reference_plan
    if workflow_name == "official_reference_gaussian_shading":
        from paper_experiments.runners.gaussian_shading_official_reference import (
            run_default_gaussian_shading_official_reference_plan,
        )

        return run_default_gaussian_shading_official_reference_plan
    if workflow_name == "official_reference_shallow_diffuse":
        from paper_experiments.runners.shallow_diffuse_official_reference import (
            run_default_shallow_diffuse_official_reference_plan,
        )

        return run_default_shallow_diffuse_official_reference_plan
    if workflow_name == "official_reference_t2smark":
        from paper_experiments.runners.t2smark_full_main_reproduction import (
            run_default_t2smark_full_main_reproduction_plan,
        )

        return run_default_t2smark_full_main_reproduction_plan
    raise ValueError(f"该 workflow 需要专用输入包调度: {workflow_name}")


def load_workflow_packager(workflow_name: str) -> WorkflowPackager:
    """延迟导入打包函数, 使服务器入口不依赖 Notebook。"""

    if workflow_name == "attention_geometry":
        from experiments.runners.attention_geometry_capture import package_attention_geometry_outputs

        return package_attention_geometry_outputs
    if workflow_name == "attention_latent_injection":
        from experiments.runners.attention_latent_injection import package_attention_latent_injection_outputs

        return package_attention_latent_injection_outputs
    if workflow_name == "aligned_rescoring":
        from experiments.runners.aligned_rescoring import package_aligned_rescoring_outputs

        return package_aligned_rescoring_outputs
    if workflow_name == "threshold_calibration":
        from experiments.runners.threshold_calibration import package_threshold_calibration_outputs

        return package_threshold_calibration_outputs
    if workflow_name == "real_attack_evaluation":
        from experiments.runners.real_attack_evaluation import package_real_attack_evaluation_outputs

        return package_real_attack_evaluation_outputs
    if workflow_name == "conventional_geometric_attack_evaluation":
        from experiments.runners.conventional_geometric_attack_evaluation import (
            package_conventional_geometric_attack_evaluation_outputs,
        )

        return package_conventional_geometric_attack_evaluation_outputs
    if workflow_name == "dataset_level_quality":
        from experiments.runners.dataset_level_quality import package_dataset_level_quality_outputs

        return package_dataset_level_quality_outputs
    if workflow_name == "external_baseline_method_faithful":
        from paper_experiments.runners.external_baseline_method_faithful import package_external_baseline_method_faithful_outputs

        return package_external_baseline_method_faithful_outputs
    if workflow_name == "official_reference_tree_ring":
        from paper_experiments.runners.tree_ring_official_reference import package_tree_ring_official_reference_outputs

        return package_tree_ring_official_reference_outputs
    if workflow_name == "official_reference_gaussian_shading":
        from paper_experiments.runners.gaussian_shading_official_reference import package_gaussian_shading_official_reference_outputs

        return package_gaussian_shading_official_reference_outputs
    if workflow_name == "official_reference_shallow_diffuse":
        from paper_experiments.runners.shallow_diffuse_official_reference import package_shallow_diffuse_official_reference_outputs

        return package_shallow_diffuse_official_reference_outputs
    if workflow_name == "official_reference_t2smark":
        from paper_experiments.runners.t2smark_full_main_reproduction import package_t2smark_full_main_reproduction_outputs

        return package_t2smark_full_main_reproduction_outputs
    raise ValueError(f"未知服务器 workflow: {workflow_name}")


def run_workflow_with_local_inputs(workflow_name: str, root: Path, result_root: Path) -> dict[str, Any]:
    """运行需要从本地结果根目录读取前序包的 workflow。"""

    if workflow_name == "threshold_calibration":
        from experiments.runners.threshold_calibration import run_default_threshold_calibration_from_drive_plan

        return run_default_threshold_calibration_from_drive_plan(
            root=root,
            attention_injection_drive_dir=(result_root / "attention_latent_injection").as_posix(),
            aligned_rescoring_drive_dir=(result_root / "aligned_rescoring").as_posix(),
            target_fpr=float(os.environ["SLM_WM_PAPER_RUN_TARGET_FPR"]),
            max_content_records="all",
            minimum_clean_negative_count=os.environ.get("SLM_WM_PAPER_RUN_MINIMUM_CLEAN_NEGATIVE_COUNT"),
        )
    if workflow_name == "real_attack_evaluation":
        from experiments.runners.real_attack_evaluation import run_default_real_attack_evaluation_from_drive_plan

        return run_default_real_attack_evaluation_from_drive_plan(
            root=root,
            aligned_rescoring_drive_dir=(result_root / "aligned_rescoring").as_posix(),
            threshold_calibration_drive_dir=(result_root / "threshold_calibration").as_posix(),
            require_threshold_package=True,
        )
    if workflow_name == "conventional_geometric_attack_evaluation":
        from experiments.runners.conventional_geometric_attack_evaluation import (
            run_default_conventional_geometric_attack_evaluation_from_drive_plan,
        )

        return run_default_conventional_geometric_attack_evaluation_from_drive_plan(
            root=root,
            aligned_rescoring_drive_dir=(result_root / "aligned_rescoring").as_posix(),
            threshold_calibration_drive_dir=(result_root / "threshold_calibration").as_posix(),
            require_threshold_package=True,
        )
    if workflow_name == "dataset_level_quality":
        from experiments.runners.dataset_level_quality import run_default_dataset_level_quality_from_drive_plan

        return run_default_dataset_level_quality_from_drive_plan(
            root=root,
            real_attack_evaluation_drive_dir=(result_root / "real_attack_evaluation").as_posix(),
            aligned_rescoring_drive_dir=(result_root / "aligned_rescoring").as_posix(),
            formal_min_sample_count=int(os.environ.get("SLM_WM_FORMAL_MIN_SAMPLE_COUNT", "100")),
        )
    return load_workflow_runner(workflow_name)(root)


def normalize_archive_record(record: Any) -> dict[str, Any]:
    """把不同打包函数返回值归一化为字典。"""

    if hasattr(record, "to_dict"):
        return record.to_dict()
    if isinstance(record, dict):
        return record
    raise TypeError(f"无法识别的打包返回类型: {type(record).__name__}")


def write_ready_manifest(
    *,
    root: Path,
    result_root: Path,
    selection: ServerWorkflowSelection,
    archive_record: dict[str, Any],
    run_summary: dict[str, Any],
    environment_report: dict[str, Any],
) -> dict[str, Any]:
    """为汇总服务器写出归档就绪清单与 SHA-256 sidecar。"""

    archive_path_text = str(archive_record.get("drive_archive_path") or archive_record.get("archive_path") or "")
    archive_path = Path(archive_path_text).expanduser()
    if not archive_path.is_absolute():
        archive_path = (root / archive_path).resolve()
    digest = str(archive_record.get("drive_archive_digest") or archive_record.get("archive_digest") or "")
    if not digest:
        digest = file_digest(archive_path)
    sidecar_path = archive_path.with_suffix(archive_path.suffix + ".sha256")
    sidecar_path.write_text(f"{digest}  {archive_path.name}\n", encoding="utf-8")
    ready_payload = {
        "server_workflow_ready": archive_path.is_file(),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "selection": selection.to_dict(),
        "archive_path": archive_path.as_posix(),
        "archive_digest": digest,
        "archive_size_bytes": archive_path.stat().st_size if archive_path.is_file() else 0,
        "sha256_sidecar_path": sidecar_path.as_posix(),
        "result_root": result_root.as_posix(),
        "environment_report": environment_report,
        "run_summary": run_summary,
    }
    ready_manifest_path = archive_path.with_suffix(archive_path.suffix + ".ready.json")
    write_json(ready_manifest_path, ready_payload)
    return ready_payload


def execute_server_workflow(
    *,
    root: str | Path,
    workflow_name: str,
    paper_run_name: str,
    result_root: str | Path,
    sample_count_token: str,
    baseline_id: str,
    target_fpr_override: str,
    dry_run: bool,
    skip_package: bool,
) -> dict[str, Any]:
    """执行单个服务器 workflow, 并把结果包发布到本地结果根目录。"""

    root_path = Path(root).resolve()
    os.chdir(root_path)
    environment_report = configure_common_server_environment(
        root=root_path,
        paper_run_name=paper_run_name,
        result_root=result_root,
        sample_count_token=sample_count_token,
        target_fpr_override=target_fpr_override,
    )
    resolved_result_root = Path(environment_report["result_root"])
    selection = resolve_workflow_selection(workflow_name, baseline_id)
    configure_workflow_environment(selection, resolved_result_root)
    publish_dir = workflow_publish_dir(resolved_result_root, selection.workflow_name)
    publish_dir.mkdir(parents=True, exist_ok=True)
    archive_name = build_workflow_archive_name(
        selection.workflow_name,
        root=root_path,
        baseline_id=selection.baseline_id,
    )
    plan = {
        "server_workflow_plan_ready": True,
        "selection": selection.to_dict(),
        "publish_dir": publish_dir.as_posix(),
        "archive_name": archive_name,
        "environment_report": environment_report,
        "skip_package": skip_package,
        "dry_run": dry_run,
    }
    if dry_run:
        return plan

    run_summary = run_workflow_with_local_inputs(selection.workflow_name, root_path, resolved_result_root)
    if skip_package:
        return {
            **plan,
            "run_summary": run_summary,
            "package_skipped": True,
        }
    packager = load_workflow_packager(selection.workflow_name)
    archive_record = normalize_archive_record(
        packager(
            root=root_path,
            drive_output_dir=publish_dir.as_posix(),
            archive_name=archive_name,
        )
    )
    ready_manifest = write_ready_manifest(
        root=root_path,
        result_root=resolved_result_root,
        selection=selection,
        archive_record=archive_record,
        run_summary=run_summary,
        environment_report=environment_report,
    )
    return {
        **plan,
        "run_summary": run_summary,
        "archive_record": archive_record,
        "ready_manifest": ready_manifest,
    }


def build_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器。"""

    parser = argparse.ArgumentParser(description="运行独立 GPU 服务器 workflow。")
    parser.add_argument("--root", default=".", help="仓库根目录。")
    parser.add_argument("--workflow", required=True, choices=WORKFLOW_CHOICES, help="要运行的 workflow 名称。")
    parser.add_argument("--paper-run-name", default="full_paper", choices=sorted(RUN_DEFAULTS), help="论文运行层级。")
    parser.add_argument("--result-root", default="", help="服务器本地结果根目录; 为空时写入 outputs/gpu_server_results。")
    parser.add_argument("--sample-count", default="all", help="样本数量; all 表示使用当前论文运行层级全部 prompt。")
    parser.add_argument("--baseline-id", default="", choices=("", *PRIMARY_BASELINE_IDS), help="method-faithful baseline 标识。")
    parser.add_argument("--target-fpr", default="", help="可选 fixed-FPR 覆盖值; 为空时使用论文运行层级默认值。")
    parser.add_argument("--dry-run", action="store_true", help="只输出服务器运行计划, 不运行真实 GPU 任务。")
    parser.add_argument("--skip-package", action="store_true", help="运行 workflow 后不打包发布。")
    return parser


def main() -> None:
    """命令行入口。"""

    args = build_parser().parse_args()
    result_root = args.result_root or default_result_root(args.paper_run_name)
    result = execute_server_workflow(
        root=args.root,
        workflow_name=args.workflow,
        paper_run_name=args.paper_run_name,
        result_root=result_root,
        sample_count_token=args.sample_count,
        baseline_id=args.baseline_id,
        target_fpr_override=args.target_fpr,
        dry_run=args.dry_run,
        skip_package=args.skip_package,
    )
    print(stable_json_text(result), end="")


if __name__ == "__main__":
    main()





