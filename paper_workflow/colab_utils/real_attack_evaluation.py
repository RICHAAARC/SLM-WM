"""真实再扩散攻击闭环的 Colab 辅助函数.

该模块的作用是把真实 SD3.5 Medium 图像攻击、攻击后检测、文件摘要登记和 Google Drive 打包放在
repository helper 中, Notebook 只负责调用入口。此处不在本地伪造 GPU 结果; 当真实后端不可用时,
函数会写出可审计的失败摘要, 方便后续在 Colab GPU 中复跑。
"""

from __future__ import annotations

import csv
import gc
import json
import math
import os
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

from experiments.protocol.paper_run_config import build_paper_run_config, resolve_count_from_environment
from experiments.protocol.pilot_paper_fixed_fpr import PILOT_PAPER_FIXED_FPR
from main.analysis.artifact_manifest import build_artifact_manifest
from main.core.digest import build_stable_digest
from main.methods.geometry.recovery import estimate_aligned_content_score
from paper_workflow.colab_utils.minimal_latent_injection import compute_image_quality_metrics
from paper_workflow.colab_utils.progress import progress_bar, update_progress
from paper_workflow.colab_utils.sd_runtime_cold_start import (
    build_runtime_environment_report,
    file_digest,
    flatten_environment_versions,
    resolve_code_version,
)

DEFAULT_OUTPUT_DIR = "outputs/real_attack_evaluation"
DEFAULT_SOURCE_IMAGE_DIR = "outputs/aligned_rescoring/aligned_images"
DEFAULT_DRIVE_OUTPUT_DIR = ""
DEFAULT_ALIGNED_RESCORING_DRIVE_DIR = ""
DEFAULT_THRESHOLD_CALIBRATION_DRIVE_DIR = ""
PRIMARY_MODEL_FAMILY = "sd35"
PRIMARY_MODEL_ID = "stabilityai/stable-diffusion-3.5-medium"
DEFAULT_DDIM_ATTACK_MODEL_ID = "runwayml/stable-diffusion-v1-5"
PACKAGE_EXTRA_PATHS = (
    "paper_workflow/real_attack_evaluation_run.ipynb",
    "paper_workflow/colab_utils/real_attack_evaluation.py",
    "outputs/attack_matrix/attack_manifest.json",
    "outputs/attack_matrix/manifest.local.json",
    "outputs/threshold_calibration/manifest.local.json",
)
REQUIRED_REGENERATION_ATTACKS = (
    "img2img_regeneration",
    "ddim_inversion_regeneration",
    "sdedit_regeneration",
    "diffusion_purification",
)
REQUIRED_ADVANCED_GPU_ATTACKS = (
    "global_editing_attack",
    "local_editing_attack",
    "visual_paraphrase_attack",
    "adversarial_removal_attack",
)
REQUIRED_REAL_GPU_ATTACKS = REQUIRED_REGENERATION_ATTACKS + REQUIRED_ADVANCED_GPU_ATTACKS
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp")
ALIGNED_RESCORING_PACKAGE_PATTERN = "aligned_rescoring_package_*.zip"
THRESHOLD_CALIBRATION_PACKAGE_PATTERN = "threshold_calibration_package_*.zip"
ALIGNED_PACKAGE_PREFIXES = (
    "outputs/aligned_rescoring/",
    "outputs/prompt_event_protocol/",
    "outputs/content_carriers/",
    "outputs/attention_latent_update/",
)
THRESHOLD_PACKAGE_PREFIXES = (
    "outputs/threshold_calibration/",
    "outputs/geometric_rescue/",
    "outputs/attack_matrix/",
)
STRICT_DDIM_RUNTIME_CACHE: dict[tuple[str, str, str, bool], dict[str, Any]] = {}
REAL_ATTACK_WATERMARK_RESCORE_STATUS = "measured_from_real_attacked_image_watermark_rescore"
FORMAL_WATERMARK_RESCORE_STATUS = "measured_from_real_attacked_image_watermark_rescore_formal_protocol"
FORMAL_RETENTION_PROXY_STATUS = "measured_from_real_attacked_image_retention_proxy_formal_protocol"


@dataclass(frozen=True)
class RealAttackEvaluationConfig:
    """描述一次真实图像级攻击闭环所需的运行配置."""

    model_family: str
    model_id: str
    seed: int
    prompt: str
    negative_prompt: str
    width: int
    height: int
    inference_steps: int
    guidance_scale: float
    output_dir: str = DEFAULT_OUTPUT_DIR
    source_image_dir: str = DEFAULT_SOURCE_IMAGE_DIR
    max_source_images: int = 600
    device_name: str = "cuda"
    torch_dtype: str = "float16"
    hf_token_env: str = "HF_TOKEN"
    detection_threshold: float = 0.50
    require_all_regeneration_attacks: bool = True
    ddim_attack_model_id: str = DEFAULT_DDIM_ATTACK_MODEL_ID
    ddim_inversion_steps: int = 30
    ddim_reconstruction_steps: int = 30
    enable_pipeline_progress_bar: bool = False
    enable_attack_progress_bar: bool = True
    enable_attacked_image_latent_rescore: bool = True
    require_attacked_image_latent_rescore: bool = True


@dataclass(frozen=True)
class RealAttackSpec:
    """描述一个真实再扩散攻击算子."""

    attack_id: str
    attack_family: str
    attack_name: str
    attack_strength: float
    attack_parameters: dict[str, Any]
    attack_implementation: str

    def to_dict(self) -> dict[str, Any]:
        """转为 JSON 兼容字典."""
        return asdict(self)


@dataclass(frozen=True)
class RealAttackDetectionRecord:
    """记录单个 source image 经真实攻击后的检测结果."""

    real_attack_record_id: str
    source_image_id: str
    source_image_path: str
    source_image_digest: str
    source_image_digest_source: str
    attack_id: str
    attack_family: str
    attack_name: str
    attack_strength: float
    attack_parameters: dict[str, Any]
    attack_implementation: str
    attacked_image_path: str
    attacked_image_digest: str
    attacked_image_digest_source: str
    attacked_image_available: bool
    attack_performed: bool
    detection_method: str
    detection_threshold: float
    raw_content_score_after: float
    aligned_content_score_after: float
    evidence_decision: bool
    metric_status: str
    unsupported_reason: str
    supports_paper_claim: bool
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """转为 JSON 兼容字典."""
        return asdict(self)


@dataclass(frozen=True)
class RealAttackEvaluationResult:
    """汇总真实图像级攻击闭环产物状态."""

    run_id: str
    model_family: str
    model_id: str
    run_decision: str
    unsupported_reason: str
    source_image_count: int
    real_attack_record_count: int
    real_attacked_image_count: int
    regeneration_attack_record_count: int
    required_regeneration_attack_count: int
    required_real_gpu_attack_count: int
    measured_regeneration_attack_count: int
    measured_real_gpu_attack_count: int
    real_attacked_image_closed_loop_ready: bool
    attacked_image_rescore_count: int
    attacked_image_rescore_ready: bool
    proxy_formal_record_count: int
    regeneration_attack_gpu_validation_ready: bool
    real_gpu_attack_validation_ready: bool
    attack_detection_rerun_ready: bool
    formal_attack_detection_ready: bool
    image_quality_metrics_ready: bool
    supports_paper_claim: bool
    output_records_path: str
    formal_records_path: str
    attacked_image_registry_path: str
    attack_family_metrics_path: str
    environment_report_path: str
    manifest_path: str
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """转为 JSON 兼容字典."""
        return asdict(self)


@dataclass(frozen=True)
class RealAttackArchiveRecord:
    """记录真实攻击闭环压缩包与 Drive 镜像."""

    archive_path: str
    archive_digest: str
    archive_entry_count: int
    drive_archive_path: str
    drive_archive_digest: str
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """转为 JSON 兼容字典."""
        return asdict(self)


def stable_json_text(value: Any) -> str:
    """以稳定顺序序列化 JSON, 方便摘要和审计."""
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def jsonl_text(rows: tuple[dict[str, Any], ...]) -> str:
    """把记录序列转为 JSONL 文本."""
    return "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)


def relative_or_absolute(path: Path, root_path: Path) -> str:
    """尽量记录相对仓库路径, 对外部路径保留绝对路径."""
    resolved_path = path.resolve()
    resolved_root = root_path.resolve()
    try:
        return resolved_path.relative_to(resolved_root).as_posix()
    except ValueError:
        return str(resolved_path)


def write_csv(path: Path, rows: tuple[dict[str, Any], ...]) -> None:
    """写出 CSV 表格, 空表只写出空文件."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = tuple(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv_rows(path: Path) -> tuple[dict[str, str], ...]:
    """读取 CSV 文件, 用于把前序包中的图像路径映射回受治理记录."""
    if not path.exists():
        return ()
    with path.open("r", encoding="utf-8", newline="") as handle:
        return tuple(dict(row) for row in csv.DictReader(handle))


def read_json(path: Path) -> dict[str, Any]:
    """读取 JSON 文件, 文件不存在时返回空字典."""
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> tuple[dict[str, Any], ...]:
    """读取 JSONL 文件, 文件不存在时返回空序列."""
    if not path.exists():
        return ()
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped:
            rows.append(json.loads(stripped))
    return tuple(rows)


def configure_model_loading_output(config: RealAttackEvaluationConfig) -> None:
    """在使用统一工作量进度条时, 静默第三方模型下载和加载进度条.

    该函数属于 Colab workflow 的通用工程写法: 项目自身已经输出统一工作量进度,
    因此当 `enable_pipeline_progress_bar=False` 时, 需要同步关闭 Hugging Face Hub、
    Diffusers 和 Transformers 的细粒度加载进度, 避免日志被单样本模型加载信息淹没。
    """

    if config.enable_pipeline_progress_bar:
        return
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("DIFFUSERS_VERBOSITY", "error")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    try:
        from huggingface_hub.utils import disable_progress_bars

        disable_progress_bars()
    except Exception:
        pass
    try:
        import diffusers

        diffusers.utils.logging.set_verbosity_error()
    except Exception:
        pass
    try:
        import transformers

        transformers.utils.logging.set_verbosity_error()
    except Exception:
        pass


def scheduler_config_without_ignored_fields(scheduler_config: Any) -> dict[str, Any]:
    """移除 DDIM scheduler 在当前 Diffusers 版本中会重复告警的兼容字段.

    该处理只影响配置解析边界, 不改变 DDIM inversion 的数学路径; `skip_prk_steps`
    属于 DDIMInverseScheduler 不消费的历史字段, Diffusers 本身也会忽略该字段。
    """

    cleaned_config = dict(scheduler_config)
    cleaned_config.pop("skip_prk_steps", None)
    return cleaned_config


def latest_drive_package(drive_dir: str | Path, pattern: str) -> Path:
    """从 Google Drive 目录中选择最新结果包."""
    candidates = sorted(Path(drive_dir).expanduser().glob(pattern))
    if not candidates:
        raise FileNotFoundError(f"drive_package_missing:{drive_dir}:{pattern}")
    return candidates[-1]


def safe_extract_selected_entries(package_path: Path, root_path: Path, allowed_prefixes: tuple[str, ...]) -> tuple[str, ...]:
    """只解压前序结果包中允许进入工作区的 outputs 输入文件."""
    extracted: list[str] = []
    with ZipFile(package_path) as archive:
        for member in archive.infolist():
            normalized_name = member.filename.replace("\\", "/")
            if member.is_dir() or not any(normalized_name.startswith(prefix) for prefix in allowed_prefixes):
                continue
            target_path = (root_path / normalized_name).resolve()
            if not target_path.is_relative_to(root_path.resolve()):
                raise RuntimeError(f"unsafe_zip_entry:{normalized_name}")
            target_path.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source_handle, target_path.open("wb") as target_handle:
                shutil.copyfileobj(source_handle, target_handle)
            extracted.append(normalized_name)
    return tuple(extracted)


def materialize_drive_package_inputs(
    root: str | Path = ".",
    aligned_rescoring_drive_dir: str | None = None,
    threshold_calibration_drive_dir: str | None = None,
    require_threshold_package: bool = True,
) -> dict[str, Any]:
    """从 Google Drive 查找前序结果包, 并只解压正式输入所需的 outputs 文件."""
    root_path = Path(root).resolve()
    paper_run = build_paper_run_config(root_path)
    resolved_aligned_rescoring_drive_dir = aligned_rescoring_drive_dir or paper_run.drive_dir("aligned_rescoring")
    resolved_threshold_calibration_drive_dir = (
        threshold_calibration_drive_dir or paper_run.drive_dir("threshold_calibration")
    )
    output_dir = root_path / DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "real_attack_input_package_manifest.json"
    aligned_package = latest_drive_package(resolved_aligned_rescoring_drive_dir, ALIGNED_RESCORING_PACKAGE_PATTERN)
    extracted_aligned = safe_extract_selected_entries(aligned_package, root_path, ALIGNED_PACKAGE_PREFIXES)
    threshold_package = None
    extracted_threshold: tuple[str, ...] = ()
    try:
        threshold_package = latest_drive_package(
            resolved_threshold_calibration_drive_dir,
            THRESHOLD_CALIBRATION_PACKAGE_PATTERN,
        )
        extracted_threshold = safe_extract_selected_entries(threshold_package, root_path, THRESHOLD_PACKAGE_PREFIXES)
    except FileNotFoundError:
        if require_threshold_package:
            raise
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "aligned_rescoring_package_path": str(aligned_package),
        "aligned_rescoring_package_digest": file_digest(aligned_package),
        "aligned_extracted_entry_count": len(extracted_aligned),
        "aligned_extracted_entries": extracted_aligned,
        "threshold_calibration_package_path": str(threshold_package) if threshold_package else "",
        "threshold_calibration_package_digest": file_digest(threshold_package) if threshold_package else "",
        "threshold_extracted_entry_count": len(extracted_threshold),
        "threshold_extracted_entries": extracted_threshold,
    }
    manifest_path.write_text(stable_json_text(manifest), encoding="utf-8")
    return manifest


def default_attack_specs() -> tuple[RealAttackSpec, ...]:
    """返回本项目主表所需的真实再扩散攻击配置."""
    return (
        RealAttackSpec(
            attack_id="real_img2img_regeneration",
            attack_family="regeneration_attack",
            attack_name="img2img_regeneration",
            attack_strength=0.35,
            attack_parameters={"denoise_strength": 0.35},
            attack_implementation="sd3_img2img",
        ),
        RealAttackSpec(
            attack_id="real_sdedit_regeneration",
            attack_family="regeneration_attack",
            attack_name="sdedit_regeneration",
            attack_strength=0.45,
            attack_parameters={"noise_level": 0.45, "denoise_strength": 0.45},
            attack_implementation="sdedit_noise_then_sd3_img2img",
        ),
        RealAttackSpec(
            attack_id="real_diffusion_purification",
            attack_family="regeneration_attack",
            attack_name="diffusion_purification",
            attack_strength=0.32,
            attack_parameters={"purification_steps": 20, "noise_level": 0.32, "denoise_strength": 0.32},
            attack_implementation="low_strength_sd3_img2img_purification",
        ),
        RealAttackSpec(
            attack_id="real_ddim_inversion_regeneration",
            attack_family="regeneration_attack",
            attack_name="ddim_inversion_regeneration",
            attack_strength=0.40,
            attack_parameters={"inversion_steps": 30, "denoise_strength": 0.40},
            attack_implementation="ddim_inverse_scheduler_reconstruction",
        ),
        RealAttackSpec(
            attack_id="real_global_editing_attack",
            attack_family="global_editing_attack",
            attack_name="global_editing_attack",
            attack_strength=0.48,
            attack_parameters={"denoise_strength": 0.48, "edit_prompt_suffix": "with a changed global style and lighting"},
            attack_implementation="sd3_img2img_global_editing",
        ),
        RealAttackSpec(
            attack_id="real_local_editing_attack",
            attack_family="local_editing_attack",
            attack_name="local_editing_attack",
            attack_strength=0.42,
            attack_parameters={"denoise_strength": 0.42, "local_mask_ratio": 0.36},
            attack_implementation="sd3_img2img_local_editing",
        ),
        RealAttackSpec(
            attack_id="real_visual_paraphrase_attack",
            attack_family="visual_paraphrase_attack",
            attack_name="visual_paraphrase_attack",
            attack_strength=0.55,
            attack_parameters={
                "denoise_strength": 0.55,
                "paraphrase_prompt_suffix": "redrawn with the same semantics but different visual composition",
            },
            attack_implementation="sd3_img2img_visual_paraphrase",
        ),
        RealAttackSpec(
            attack_id="real_adversarial_removal_attack",
            attack_family="adversarial_removal_attack",
            attack_name="adversarial_removal_attack",
            attack_strength=0.38,
            attack_parameters={"denoise_strength": 0.38, "pre_noise_level": 0.035, "anti_watermark_bias": 0.20},
            attack_implementation="sd3_img2img_adversarial_removal",
        ),
    )


def source_image_id(source_path: Path, digest: str) -> str:
    """由图像路径和摘要构造稳定 source image id."""
    return f"source_image_{build_stable_digest({'path': source_path.name, 'digest': digest})[:16]}"


def real_attack_record_id(source_digest: str, spec: RealAttackSpec, attacked_digest: str) -> str:
    """由 source、攻击配置和 attacked image 摘要构造记录 id."""
    payload = {
        "source_image_digest": source_digest,
        "attack_id": spec.attack_id,
        "attack_parameters": spec.attack_parameters,
        "attacked_image_digest": attacked_digest,
    }
    return f"real_attack_{build_stable_digest(payload)[:16]}"


def resolve_repo_image_path(root_path: Path, image_path: str) -> Path | None:
    """把记录中的图像路径解析为本地文件路径.

    该函数属于通用工程写法: 前序结果包中的图像路径通常是仓库相对路径,
    但在人工诊断时也可能传入绝对路径。统一解析可以避免后续攻击闭环
    在每个业务分支中重复处理路径边界。
    """

    normalized = str(image_path or "").strip()
    if not normalized:
        return None
    candidate = Path(normalized)
    return candidate.resolve() if candidate.is_absolute() else (root_path / candidate).resolve()


def aligned_rescoring_source_image_paths(root_path: Path, config: RealAttackEvaluationConfig) -> tuple[Path, ...]:
    """从 aligned rescoring 质量表读取成组 clean / aligned source image.

    此处是项目特定写法: pilot_paper 与 full_paper 的 fixed-FPR 共同协议需要
    同时攻击 clean_negative 和 positive_source 图像。`max_source_images` 在该
    受治理路径中表示最多读取多少个 prompt / carrier 质量记录, 每条质量记录
    最多贡献一张 aligned 图像和一张 clean 图像。
    """

    quality_rows = read_csv_rows(root_path / "outputs" / "aligned_rescoring" / "aligned_rescoring_quality_metrics.csv")
    if not quality_rows:
        return ()
    row_limit = max(0, int(config.max_source_images))
    selected_rows = quality_rows[:row_limit] if row_limit else quality_rows
    candidates: list[Path] = []
    for quality_row in selected_rows:
        for field_name in ("aligned_image_path", "clean_image_path"):
            image_path = resolve_repo_image_path(root_path, str(quality_row.get(field_name, "")))
            if image_path and image_path.is_file() and image_path.suffix.lower() in IMAGE_SUFFIXES:
                candidates.append(image_path)
    unique: list[Path] = []
    for path in candidates:
        if path not in unique:
            unique.append(path)
    return tuple(unique)


def discover_source_images(root_path: Path, config: RealAttackEvaluationConfig) -> tuple[Path, ...]:
    """查找需要进入真实攻击闭环的 source image 文件."""
    if config.source_image_dir == DEFAULT_SOURCE_IMAGE_DIR:
        governed_paths = aligned_rescoring_source_image_paths(root_path, config)
        if governed_paths:
            return governed_paths
    candidates: list[Path] = []
    configured_dir = (root_path / config.source_image_dir).resolve()
    fallback_dirs = (
        configured_dir,
        root_path / "outputs" / "aligned_rescoring",
        root_path / "outputs" / "minimal_diffusion_latent_injection",
    )
    for directory in fallback_dirs:
        if directory.exists():
            for path in sorted(directory.rglob("*")):
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                    candidates.append(path.resolve())
    unique: list[Path] = []
    for path in candidates:
        if path not in unique:
            unique.append(path)
    return tuple(unique[: config.max_source_images])


def _runtime_versions_from_torch(torch_module: Any) -> dict[str, Any]:
    """构造 diffusion runtime 版本报告, 供攻击与检测 loader 复用。

    该函数属于通用工程写法: 把环境记录收敛到单一位置, 避免不同 loader
    分散维护相同的 provenance 字段。
    """

    environment_report = build_runtime_environment_report(torch_module=torch_module)
    return {
        **flatten_environment_versions(environment_report),
        "runtime_environment": environment_report,
    }


class DetectorPipeline:
    """保存检测侧 VAE 与图像预处理器的轻量 pipeline。

    该类属于项目特定工程适配: 正式攻击后检测只需要把 attacked image
    重新编码到 SD3.5 VAE latent 空间, 不需要加载 transformer、text encoder
    或 image-to-image pipeline。通过该轻量对象可以降低 Colab 依赖导入失败
    对常规失真与几何攻击闭环的影响。
    """

    def __init__(self, vae: Any, image_processor: Any, loader_name: str) -> None:
        """保存 VAE、预处理器与 loader 来源, 便于 manifest 审计。"""

        self.vae = vae
        self.image_processor = image_processor
        self.detector_loader_name = loader_name

    def to(self, device_name: str) -> "DetectorPipeline":
        """把 VAE 放到目标设备并返回自身, 对齐 diffusers pipeline 的用法。"""

        self.vae = self.vae.to(device_name)
        return self


class SimpleVaeImageProcessor:
    """提供最小 VAE 图像预处理能力。

    该实现是通用工程兜底: 当无法安全导入 diffusers 的 pipeline 类时,
    仍可把 PIL 图像转换为 VAE 期望的 [-1, 1] tensor, 从而保持真实
    attacked image latent re-score 路径不退化为 retention proxy。
    """

    def preprocess(self, image: Any) -> Any:
        """将 PIL 图像转换为 NCHW float tensor, 数值范围为 [-1, 1]。"""

        import torch

        rgb_image = image.convert("RGB")
        width, height = rgb_image.size
        image_bytes = bytearray(rgb_image.tobytes())
        tensor = torch.frombuffer(image_bytes, dtype=torch.uint8)
        tensor = tensor.reshape(height, width, 3).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        return tensor * 2.0 - 1.0


def _compact_error(error: Exception, max_length: int = 240) -> str:
    """压缩异常信息, 用于 runtime provenance 字段。"""

    return f"{type(error).__name__}:{str(error)[:max_length]}"


def patch_numpy_core_umath_string_center_compatibility() -> dict[str, Any]:
    """为不完整的 NumPy 运行时补齐字符串居中 ufunc 导出。

    部分 Colab 动态依赖组合会出现 NumPy Python 层代码期望
    `numpy._core.umath._center`, 但底层扩展尚未导出该名称的情况。该缺口会在
    torch / diffusers 导入链路中表现为 `_center` ImportError。这里补齐一个
    只用于导入兼容的占位 callable; 本项目不会在正式检测路径中调用 NumPy
    字符串居中逻辑。
    """

    report: dict[str, Any] = {
        "numpy_umath_center_patch_applied": False,
    }
    try:
        import importlib

        umath = importlib.import_module("numpy._core.umath")
    except Exception as error:
        report["numpy_umath_import_error"] = _compact_error(error)
        return report
    if hasattr(umath, "_center"):
        return report

    def _center_import_compatibility_placeholder(*_args: Any, **_kwargs: Any) -> Any:
        """导入兼容占位实现, 不用于本项目数值路径。"""

        raise RuntimeError("numpy_umath_center_runtime_unavailable")

    setattr(umath, "_center", _center_import_compatibility_placeholder)
    report["numpy_umath_center_patch_applied"] = True
    return report


def patch_pillow_typing_ink_compatibility() -> dict[str, Any]:
    """为不完整的 Pillow 运行时补齐仅供类型导入使用的 `_Ink` 导出。

    部分 Colab 运行时在同一内核中升级 Pillow 后, 会出现 `PIL.Image`
    代码期望 `PIL._typing._Ink`, 但已加载的 `PIL._typing` 仍来自旧版本的情况。
    `_Ink` 只参与 Pillow 内部类型标注导入, 本项目不会把该补丁用于图像数值逻辑。
    """

    report: dict[str, Any] = {
        "pillow_typing_ink_patch_applied": False,
    }
    try:
        import importlib

        pil_typing = importlib.import_module("PIL._typing")
    except Exception as error:
        report["pillow_typing_import_error"] = _compact_error(error)
        return report
    if hasattr(pil_typing, "_Ink"):
        return report
    setattr(pil_typing, "_Ink", Any)
    report["pillow_typing_ink_patch_applied"] = True
    return report


def patch_transformers_for_diffusers_autoencoder_import() -> dict[str, Any]:
    """为 diffusers 新版 autoencoder 导入补齐 transformers 兼容导出。

    当前 Colab 动态升级组合可能出现 diffusers 已经引用
    `Dinov2WithRegistersConfig` / `Dinov2WithRegistersModel`, 但 transformers
    顶层尚未导出这些名称的情况。该问题发生在 diffusers 导入过程, 即使本项目
    只使用 SD3.5 VAE, 也会被 autoencoder package 的额外导入牵连。
    这里把缺失名称映射到同族 DINOv2 类, 仅用于让无关 RAE 模块完成导入;
    正式检测仍只调用 AutoencoderKL, 不使用这些补齐类执行论文方法逻辑。
    """

    report: dict[str, Any] = {
        "transformers_dinov2_registers_patch_applied": False,
        "patched_transformers_exports": [],
    }
    try:
        import transformers
    except Exception as error:
        report["transformers_import_error"] = _compact_error(error)
        return report

    alias_pairs = (
        ("Dinov2WithRegistersConfig", "Dinov2Config"),
        ("Dinov2WithRegistersModel", "Dinov2Model"),
        ("Dinov2WithRegistersPreTrainedModel", "Dinov2PreTrainedModel"),
    )
    patched: list[str] = []
    for missing_name, fallback_name in alias_pairs:
        if hasattr(transformers, missing_name) or not hasattr(transformers, fallback_name):
            continue
        setattr(transformers, missing_name, getattr(transformers, fallback_name))
        patched.append(missing_name)
    report["transformers_dinov2_registers_patch_applied"] = bool(patched)
    report["patched_transformers_exports"] = patched
    return report


def load_img2img_pipeline(config: RealAttackEvaluationConfig) -> tuple[Any, dict[str, Any]]:
    """加载真实 SD3/SD3.5 image-to-image pipeline。

    该 loader 只用于再扩散类攻击生成。常规失真与几何攻击不应依赖
    `StableDiffusion3Img2ImgPipeline`, 否则会把攻击图像生成与检测侧 VAE
    重评分错误绑定。
    """
    patch_numpy_core_umath_string_center_compatibility()
    patch_pillow_typing_ink_compatibility()
    import torch
    import diffusers

    configure_model_loading_output(config)
    if config.device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("gpu_unavailable")
    dtype = getattr(torch, config.torch_dtype)
    pipeline_class = getattr(diffusers, "StableDiffusion3Img2ImgPipeline", None)
    if pipeline_class is None:
        pipeline_class = getattr(diffusers, "AutoPipelineForImage2Image")
    token = os.environ.get(config.hf_token_env) or None
    pipeline = pipeline_class.from_pretrained(config.model_id, torch_dtype=dtype, token=token)
    pipeline = pipeline.to(config.device_name)
    pipeline.set_progress_bar_config(disable=not config.enable_pipeline_progress_bar)
    return pipeline, _runtime_versions_from_torch(torch)


def _load_detector_pipeline_from_sd3_pipeline(config: RealAttackEvaluationConfig, torch_module: Any) -> Any:
    """优先通过 SD3 pipeline 获取检测侧 VAE 与 image processor。"""

    patch_transformers_for_diffusers_autoencoder_import()
    from diffusers import StableDiffusion3Pipeline

    dtype = getattr(torch_module, config.torch_dtype)
    token = os.environ.get(config.hf_token_env) or None
    pipeline = StableDiffusion3Pipeline.from_pretrained(config.model_id, torch_dtype=dtype, token=token)
    pipeline = pipeline.to(config.device_name)
    pipeline.set_progress_bar_config(disable=not config.enable_pipeline_progress_bar)
    if hasattr(pipeline, "vae"):
        pipeline.vae.eval()
    return pipeline


def _import_autoencoder_kl() -> Any:
    """导入 AutoencoderKL, 并兼容不同 diffusers 版本的导出位置。"""

    patch_transformers_for_diffusers_autoencoder_import()
    try:
        from diffusers import AutoencoderKL

        return AutoencoderKL
    except Exception as public_error:
        try:
            from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL

            return AutoencoderKL
        except Exception as direct_error:
            raise RuntimeError(
                "autoencoder_kl_import_failed:"
                f"public={_compact_error(public_error)};"
                f"direct={_compact_error(direct_error)}"
            ) from direct_error


def _load_detector_pipeline_from_vae(config: RealAttackEvaluationConfig, torch_module: Any) -> DetectorPipeline:
    """只加载模型仓库中的 VAE 子模块, 避免导入完整 SD3 pipeline。"""

    autoencoder_kl = _import_autoencoder_kl()
    dtype = getattr(torch_module, config.torch_dtype)
    token = os.environ.get(config.hf_token_env) or None
    vae = autoencoder_kl.from_pretrained(
        config.model_id,
        subfolder="vae",
        torch_dtype=dtype,
        token=token,
    )
    vae = vae.to(config.device_name)
    vae.eval()
    return DetectorPipeline(vae=vae, image_processor=SimpleVaeImageProcessor(), loader_name="vae_subfolder")


def load_detector_pipeline(config: RealAttackEvaluationConfig) -> tuple[Any, dict[str, Any]]:
    """加载只用于 attacked image latent re-score 的 SD3/SD3.5 detector pipeline。

    此处设计的主要考虑在于: 常规失真和几何攻击只需要主模型 VAE 与
    image_processor 来把 attacked image 重新编码到 latent 空间, 不需要
    image-to-image pipeline。该 loader 会优先复用 SD3 pipeline; 若 Colab
    依赖组合无法导入完整 SD3 pipeline, 则退到 VAE 子模块加载路径, 仍保持
    真实 attacked image latent re-score, 不退化为 retention proxy。
    """

    numpy_report = patch_numpy_core_umath_string_center_compatibility()
    pillow_report = patch_pillow_typing_ink_compatibility()
    import torch

    configure_model_loading_output(config)
    if config.device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("gpu_unavailable")
    runtime_versions = _runtime_versions_from_torch(torch)
    runtime_versions.update(numpy_report)
    runtime_versions.update(pillow_report)
    runtime_versions["runtime_environment"].update(numpy_report)
    runtime_versions["runtime_environment"].update(pillow_report)
    compat_report = patch_transformers_for_diffusers_autoencoder_import()
    runtime_versions.update(compat_report)
    runtime_versions["runtime_environment"].update(compat_report)
    try:
        pipeline = _load_detector_pipeline_from_sd3_pipeline(config, torch)
        runtime_versions["detector_loader_name"] = "stable_diffusion_3_pipeline"
        return pipeline, runtime_versions
    except Exception as pipeline_error:
        try:
            pipeline = _load_detector_pipeline_from_vae(config, torch)
        except Exception as vae_error:
            raise RuntimeError(
                "detector_pipeline_unavailable:"
                f"sd3_pipeline={_compact_error(pipeline_error)};"
                f"vae_subfolder={_compact_error(vae_error)}"
            ) from vae_error
        runtime_versions["detector_loader_name"] = "vae_subfolder"
        runtime_versions["detector_loader_fallback_reason"] = _compact_error(pipeline_error)
        runtime_versions["runtime_environment"]["detector_loader_name"] = "vae_subfolder"
        runtime_versions["runtime_environment"]["detector_loader_fallback_reason"] = _compact_error(pipeline_error)
        return pipeline, runtime_versions


def load_rgb_image(path: Path, config: RealAttackEvaluationConfig) -> Any:
    """读取 source image 并调整为 pipeline 输入尺寸."""
    patch_pillow_typing_ink_compatibility()
    from PIL import Image

    image = Image.open(path).convert("RGB")
    return image.resize((config.width, config.height))


def normalize_attacked_image_size(attacked_image: Any, source_image: Any) -> Any:
    """把 attacked image 对齐到 source image 尺寸, 保证后续质量指标可直接逐像素比较."""
    if getattr(attacked_image, "size", None) == getattr(source_image, "size", None):
        return attacked_image.convert("RGB") if hasattr(attacked_image, "convert") else attacked_image
    patch_pillow_typing_ink_compatibility()
    from PIL import Image

    resampling = getattr(getattr(Image, "Resampling", Image), "LANCZOS")
    return attacked_image.convert("RGB").resize(source_image.size, resampling)


def add_sdedit_noise(image: Any, noise_level: float, seed: int) -> Any:
    """为 SDEdit 风格攻击构造带噪输入图像."""

    import torch
    patch_pillow_typing_ink_compatibility()
    from PIL import Image

    rgb_image = image.convert("RGB")
    width, height = rgb_image.size
    image_bytes = bytearray(rgb_image.tobytes())
    image_tensor = torch.frombuffer(image_bytes, dtype=torch.uint8).reshape(height, width, 3).float() / 255.0
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    noise = torch.randn(image_tensor.shape, generator=generator, dtype=torch.float32) * float(noise_level)
    noisy_image = torch.clamp(image_tensor + noise, 0.0, 1.0)
    mixed = torch.clamp(image_tensor * (1.0 - noise_level) + noisy_image * noise_level, 0.0, 1.0)
    byte_values = (mixed * 255.0).round().to(torch.uint8).contiguous().view(-1).tolist()
    return Image.frombytes("RGB", (width, height), bytes(byte_values))


def add_local_editing_patch(image: Any, mask_ratio: float, seed: int) -> Any:
    """构造局部编辑攻击的可审计输入图像。

    该函数只负责给 img2img pipeline 提供局部扰动初值, 真正的生成式重写仍由
    SD3.5 img2img 后端完成。这样可以复用同一真实 GPU 链路, 同时保留局部编辑
    攻击相对全局再生成攻击的统计边界。
    """

    import random

    patch_pillow_typing_ink_compatibility()
    from PIL import ImageDraw, ImageFilter

    rng = random.Random(int(seed))
    rgb_image = image.convert("RGB")
    width, height = rgb_image.size
    patch_width = max(1, int(width * float(mask_ratio)))
    patch_height = max(1, int(height * float(mask_ratio)))
    left = rng.randint(0, max(0, width - patch_width))
    top = rng.randint(0, max(0, height - patch_height))
    blurred_patch = rgb_image.crop((left, top, left + patch_width, top + patch_height)).filter(
        ImageFilter.GaussianBlur(radius=max(1.0, min(width, height) * 0.01))
    )
    edited = rgb_image.copy()
    edited.paste(blurred_patch, (left, top))
    draw = ImageDraw.Draw(edited, "RGBA")
    draw.rectangle((left, top, left + patch_width, top + patch_height), fill=(255, 255, 255, 24))
    return edited


def run_pipeline_attack(
    pipeline: Any,
    source_image: Any,
    spec: RealAttackSpec,
    config: RealAttackEvaluationConfig,
    seed: int,
    prompt_text: str,
) -> Any:
    """执行单个真实再扩散攻击并返回 attacked image."""
    import torch

    input_image = source_image
    prompt = prompt_text
    if spec.attack_name == "sdedit_regeneration":
        input_image = add_sdedit_noise(source_image, float(spec.attack_parameters["noise_level"]), seed)
    if spec.attack_name == "adversarial_removal_attack":
        input_image = add_sdedit_noise(source_image, float(spec.attack_parameters["pre_noise_level"]), seed)
    if spec.attack_name == "local_editing_attack":
        input_image = add_local_editing_patch(source_image, float(spec.attack_parameters["local_mask_ratio"]), seed)
    if spec.attack_name == "global_editing_attack":
        prompt = f"{prompt_text}, {spec.attack_parameters['edit_prompt_suffix']}"
    if spec.attack_name == "visual_paraphrase_attack":
        prompt = f"{prompt_text}, {spec.attack_parameters['paraphrase_prompt_suffix']}"
    generator = torch.Generator(device=config.device_name).manual_seed(seed)
    output = pipeline(
        prompt=prompt,
        negative_prompt=config.negative_prompt,
        image=input_image,
        height=config.height,
        width=config.width,
        strength=float(spec.attack_parameters["denoise_strength"]),
        num_inference_steps=int(spec.attack_parameters.get("purification_steps", config.inference_steps)),
        guidance_scale=config.guidance_scale,
        generator=generator,
        output_type="pil",
    )
    return normalize_attacked_image_size(output.images[0], source_image)


def encode_prompt_for_ddim(pipe: Any, config: RealAttackEvaluationConfig, prompt_text: str, do_guidance: bool) -> Any:
    """为严格 DDIM inversion 构造文本条件 embedding."""
    if hasattr(pipe, "encode_prompt"):
        encoded = pipe.encode_prompt(
            prompt=prompt_text,
            device=config.device_name,
            num_images_per_prompt=1,
            do_classifier_free_guidance=do_guidance,
            negative_prompt=config.negative_prompt,
        )
        if isinstance(encoded, tuple) and len(encoded) >= 2:
            prompt_embeds, negative_prompt_embeds = encoded[0], encoded[1]
            return prompt_embeds if not do_guidance else pipe.torch.cat([negative_prompt_embeds, prompt_embeds])
        return encoded
    return pipe._encode_prompt(
        prompt_text,
        config.device_name,
        1,
        do_guidance,
        negative_prompt=config.negative_prompt,
    )


def preprocess_ddim_image(pipe: Any, source_image: Any, config: RealAttackEvaluationConfig, torch_module: Any) -> Any:
    """把 PIL 图像编码为 DDIM inversion 使用的 latent."""
    image_tensor = pipe.image_processor.preprocess(source_image).to(device=config.device_name, dtype=getattr(torch_module, config.torch_dtype))
    with torch_module.no_grad():
        latents = pipe.vae.encode(image_tensor).latent_dist.sample()
    return latents * pipe.vae.config.scaling_factor


def predict_ddim_noise(
    pipe: Any,
    scheduler: Any,
    latents: Any,
    timestep: Any,
    prompt_embeds: Any,
    guidance_scale: float,
    do_guidance: bool,
) -> Any:
    """执行一次 UNet 噪声预测并应用 classifier-free guidance."""
    scaled_latents = scheduler.scale_model_input(latents, timestep) if hasattr(scheduler, "scale_model_input") else latents
    latent_model_input = pipe.torch.cat([scaled_latents] * 2) if do_guidance else scaled_latents
    noise_pred = pipe.unet(latent_model_input, timestep, encoder_hidden_states=prompt_embeds).sample
    if not do_guidance:
        return noise_pred
    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    return noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)


def strict_ddim_runtime_cache_key(config: RealAttackEvaluationConfig) -> tuple[str, str, str, bool]:
    """构造 DDIM inversion 运行时缓存键.

    该键只包含会影响 pipeline 加载结果的配置项。prompt、seed 和 attack 参数属于单样本执行参数,
    不应导致重复加载模型。
    """

    return (
        str(config.ddim_attack_model_id),
        str(config.device_name),
        str(config.torch_dtype),
        bool(config.enable_pipeline_progress_bar),
    )


def load_strict_ddim_inversion_runtime(config: RealAttackEvaluationConfig) -> dict[str, Any]:
    """加载一次 DDIM inversion 所需的 legacy pipeline 与 scheduler 类.

    该函数将模型加载收敛到运行时缓存边界, 避免每个 source image 都重新下载或重新构造
    Stable Diffusion pipeline。单样本函数只复用该运行时对象并重置 scheduler 状态。
    """

    import torch
    import diffusers

    configure_model_loading_output(config)
    if config.device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("gpu_unavailable")
    pipeline_class = getattr(diffusers, "StableDiffusionPipeline", None)
    inverse_scheduler_class = getattr(diffusers, "DDIMInverseScheduler", None)
    scheduler_class = getattr(diffusers, "DDIMScheduler", None)
    if pipeline_class is None or inverse_scheduler_class is None or scheduler_class is None:
        raise RuntimeError("strict_ddim_components_unavailable")
    dtype = getattr(torch, config.torch_dtype)
    token = os.environ.get(config.hf_token_env) or None
    pipe = pipeline_class.from_pretrained(
        config.ddim_attack_model_id,
        torch_dtype=dtype,
        token=token,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe = pipe.to(config.device_name)
    pipe.torch = torch
    if hasattr(pipe, "set_progress_bar_config"):
        pipe.set_progress_bar_config(disable=not config.enable_pipeline_progress_bar)
    scheduler_config = scheduler_config_without_ignored_fields(pipe.scheduler.config)
    pipe.scheduler = scheduler_class.from_config(scheduler_config)
    return {
        "pipe": pipe,
        "torch": torch,
        "scheduler_class": scheduler_class,
        "inverse_scheduler_class": inverse_scheduler_class,
        "scheduler_config": scheduler_config,
    }


def get_strict_ddim_inversion_runtime(config: RealAttackEvaluationConfig) -> dict[str, Any]:
    """按配置复用 DDIM inversion pipeline, 使日志只保留统一工作量进度."""

    cache_key = strict_ddim_runtime_cache_key(config)
    runtime = STRICT_DDIM_RUNTIME_CACHE.get(cache_key)
    if runtime is None:
        runtime = load_strict_ddim_inversion_runtime(config)
        STRICT_DDIM_RUNTIME_CACHE[cache_key] = runtime
    return runtime


def clear_strict_ddim_inversion_runtime_cache() -> None:
    """释放 DDIM inversion pipeline 缓存, 避免后续 Notebook cell 持续占用显存."""

    for runtime in STRICT_DDIM_RUNTIME_CACHE.values():
        runtime.pop("pipe", None)
    STRICT_DDIM_RUNTIME_CACHE.clear()
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def run_strict_ddim_inversion_attack(
    source_image: Any,
    spec: RealAttackSpec,
    config: RealAttackEvaluationConfig,
    seed: int,
    prompt_text: str,
) -> Any:
    """使用 DDIMInverseScheduler 执行真正的 inversion 再生成攻击."""

    runtime = get_strict_ddim_inversion_runtime(config)
    pipe = runtime["pipe"]
    torch = runtime["torch"]
    pipe.scheduler = runtime["scheduler_class"].from_config(runtime["scheduler_config"])
    inverse_scheduler = runtime["inverse_scheduler_class"].from_config(runtime["scheduler_config"])
    inversion_steps = int(spec.attack_parameters.get("inversion_steps", config.ddim_inversion_steps))
    reconstruction_steps = config.ddim_reconstruction_steps
    do_guidance = config.guidance_scale > 1.0
    prompt_embeds = encode_prompt_for_ddim(pipe, config, prompt_text, do_guidance)
    latents = preprocess_ddim_image(pipe, source_image, config, torch)
    inverse_scheduler.set_timesteps(inversion_steps, device=config.device_name)
    with torch.no_grad():
        for timestep in inverse_scheduler.timesteps:
            noise_pred = predict_ddim_noise(
                pipe,
                inverse_scheduler,
                latents,
                timestep,
                prompt_embeds,
                config.guidance_scale,
                do_guidance,
            )
            latents = inverse_scheduler.step(noise_pred, timestep, latents).prev_sample
        generator = torch.Generator(device=config.device_name).manual_seed(seed)
        output = pipe(
            prompt=prompt_text,
            negative_prompt=config.negative_prompt,
            num_inference_steps=reconstruction_steps,
            guidance_scale=config.guidance_scale,
            latents=latents,
            generator=generator,
            output_type="pil",
        )
    return output.images[0]


def quality_detection_scores(source_image: Any, attacked_image: Any, threshold: float) -> tuple[dict[str, float | str], float, float, bool]:
    """基于真实图像差异重新计算攻击后检测代理分数."""
    metrics = compute_image_quality_metrics(source_image, attacked_image)
    mean_abs_error = float(metrics["mean_abs_error"])
    ssim = float(metrics["ssim"])
    raw_score = max(0.0, min(1.0, 1.0 - 4.0 * mean_abs_error - 0.30 * max(0.0, 1.0 - ssim)))
    aligned_score = raw_score
    decision = aligned_score >= threshold
    return metrics, raw_score, aligned_score, decision


def normalized_slot_projection_from_values(flattened_values: tuple[float, ...], value_count: int) -> tuple[float, ...]:
    """把平铺 latent 数值按重复槽位聚合为单位检测向量.

    该函数复用 aligned rescoring 的投影思想, 但放在攻击检测 helper 内部,
    避免 attacked image 检测路径为了一个轻量数学原语反向依赖 Notebook 级重打分模块。
    """

    if not flattened_values:
        raise RuntimeError("attacked_image_latent_tensor_empty")
    bounded_count = max(1, int(value_count))
    selected = tuple(
        sum(flattened_values[index::bounded_count]) / len(flattened_values[index::bounded_count])
        for index in range(bounded_count)
        if flattened_values[index::bounded_count]
    )
    mean_value = sum(selected) / len(selected)
    centered = tuple(value - mean_value for value in selected)
    norm = math.sqrt(sum(value * value for value in centered))
    if norm <= 1e-6:
        return tuple(0.0 for _ in centered)
    return tuple(value / norm for value in centered)


def _float_tuple(values: Any) -> tuple[float, ...]:
    """把 JSON 数组或 tuple 转为浮点 tuple, 缺失时返回空序列."""

    if not isinstance(values, (list, tuple)):
        return ()
    return tuple(float(value) for value in values)


def encode_attacked_image_projection(detector_pipeline: Any, image: Any, config: RealAttackEvaluationConfig, value_count: int) -> tuple[float, ...]:
    """使用主检测模型 VAE 把 attacked image 编码到真实 latent 投影空间.

    这是修复 proxy 边界的核心路径: 正式攻击后检测不再用图像质量分数缩放
    pre-attack latent score, 而是先把 attacked image 重新编码为 latent, 再在
    与 aligned rescoring 一致的周期槽位投影空间中计算水印响应。
    """

    import torch

    image_tensor = detector_pipeline.image_processor.preprocess(image).to(
        device=config.device_name,
        dtype=getattr(torch, config.torch_dtype),
    )
    with torch.no_grad():
        latent_dist = detector_pipeline.vae.encode(image_tensor).latent_dist
        latents = latent_dist.mode() if hasattr(latent_dist, "mode") else latent_dist.mean
    scale = float(getattr(detector_pipeline.vae.config, "scaling_factor", 1.0))
    shift = float(getattr(detector_pipeline.vae.config, "shift_factor", 0.0) or 0.0)
    latents = (latents - shift) * scale
    flattened = tuple(float(value) for value in latents.detach().float().reshape(-1).cpu().tolist())
    return normalized_slot_projection_from_values(flattened, value_count)


def projection_digest(values: tuple[float, ...]) -> str:
    """计算 latent 投影向量的稳定摘要."""

    return build_stable_digest([round(value, 12) for value in values])


def latent_projection_watermark_score(
    attacked_values: tuple[float, ...],
    source_context: dict[str, Any],
) -> dict[str, Any]:
    """根据 source clean/aligned latent 端点计算 attacked image 水印分数.

    该实现属于项目特定写法: 真实写入是在 clean latent 到 aligned latent 之间
    形成的内容条件化方向。检测时使用 attacked image 的 VAE latent 投影在该方向
    上的位置, 而不是用像素质量指标近似分数保留率。
    """

    before_values = _float_tuple(source_context.get("latent_projection_values_before"))
    after_values = _float_tuple(source_context.get("latent_projection_values_after"))
    if not before_values or not after_values or len(before_values) != len(after_values):
        raise RuntimeError("source_latent_projection_endpoints_missing")
    if len(attacked_values) != len(before_values):
        raise RuntimeError("attacked_projection_width_mismatch")
    direction = tuple(after - before for before, after in zip(before_values, after_values))
    direction_norm_sq = sum(value * value for value in direction)
    if direction_norm_sq <= 1e-12:
        raise RuntimeError("source_watermark_direction_degenerate")
    centered_attacked = tuple(value - before for value, before in zip(attacked_values, before_values))
    coordinate = sum(value * axis for value, axis in zip(centered_attacked, direction)) / direction_norm_sq
    bounded_coordinate = max(-0.25, min(1.25, coordinate))
    raw_before = float(source_context["raw_content_score_before"])
    aligned_before = float(source_context["aligned_content_score_before"])
    score_delta = aligned_before - raw_before
    score_after = raw_before + bounded_coordinate * score_delta
    return {
        "watermark_coordinate": coordinate,
        "bounded_watermark_coordinate": bounded_coordinate,
        "raw_attacked_latent_score": score_after,
        "score_delta_from_source_endpoints": score_delta,
        "latent_projection_width": len(attacked_values),
        "attacked_latent_projection_digest": projection_digest(attacked_values),
        "source_projection_digest_before": source_context.get("latent_projection_digest_before", ""),
        "source_projection_digest_after": source_context.get("latent_projection_digest_after", ""),
    }


def geometry_evidence_from_source_context(source_context: dict[str, Any]) -> dict[str, Any]:
    """从 source context 提取 same-threshold rescue 所需几何证据."""

    return {
        "geometry_reliable": bool(source_context.get("geometry_reliable", False)),
        "registration_confidence": float(source_context.get("registration_confidence", 0.0)),
        "anchor_inlier_ratio": float(source_context.get("anchor_inlier_ratio", 0.0)),
        "recovered_sync_consistency": float(source_context.get("recovered_sync_consistency", 0.0)),
        "alignment_residual": float(source_context.get("alignment_residual", 1.0)),
    }


def decide_attack_rescore_with_same_threshold_rescue(
    raw_score_after: float,
    source_context: dict[str, Any],
    boundary: dict[str, Any],
) -> dict[str, Any]:
    """在 attacked image 真实 latent 分数上执行 same-threshold 几何恢复重判."""

    threshold = float(boundary["content_threshold"])
    raw_margin_after = raw_score_after - threshold
    geometry_evidence = geometry_evidence_from_source_context(source_context)
    geometry_reliable = bool(geometry_evidence["geometry_reliable"])
    fail_reason = str(source_context.get("fail_reason", "geometry_suspected"))
    rescue_eligible = (
        float(boundary["rescue_margin_low"]) <= raw_margin_after < 0.0
        and geometry_reliable
        and fail_reason in boundary["allowed_fail_reasons"]
    )
    aligned_score_after = estimate_aligned_content_score(
        raw_content_score=raw_score_after,
        content_threshold=threshold,
        geometry_evidence=geometry_evidence,
        sample_role=str(source_context.get("sample_role", "unknown")),
    )
    aligned_margin_after = aligned_score_after - threshold
    positive_by_content = raw_margin_after >= 0.0
    rescue_applied = rescue_eligible and aligned_margin_after >= 0.0
    return {
        "raw_content_score_after": raw_score_after,
        "aligned_content_score_after": aligned_score_after,
        "threshold_score_after": aligned_score_after,
        "raw_content_margin_after": raw_margin_after,
        "aligned_content_margin_after": aligned_margin_after,
        "positive_by_content": positive_by_content,
        "geometry_reliable": geometry_reliable,
        "rescue_eligible": rescue_eligible,
        "rescue_applied": rescue_applied,
        "evidence_decision": positive_by_content or rescue_applied,
        "formal_detection_decision": aligned_margin_after >= 0.0,
    }


def rescore_attacked_image_with_detector(
    detector_pipeline: Any,
    attacked_image: Any,
    source_context: dict[str, Any],
    boundary: dict[str, Any],
    config: RealAttackEvaluationConfig,
) -> dict[str, Any]:
    """对 attacked image 执行真实 VAE latent 投影检测和 same-threshold rescue."""

    width = len(_float_tuple(source_context.get("latent_projection_values_after")))
    if width <= 0:
        width = len(_float_tuple(source_context.get("latent_projection_values_before")))
    if width <= 0:
        raise RuntimeError("source_projection_width_missing")
    attacked_values = encode_attacked_image_projection(detector_pipeline, attacked_image, config, width)
    projection_score = latent_projection_watermark_score(attacked_values, source_context)
    rescue_decision = decide_attack_rescore_with_same_threshold_rescue(
        raw_score_after=float(projection_score["raw_attacked_latent_score"]),
        source_context=source_context,
        boundary=boundary,
    )
    return {
        **projection_score,
        **rescue_decision,
        **geometry_evidence_from_source_context(source_context),
        "attacked_image_rescore_performed": True,
        "formal_detection_proxy": False,
        "detection_score_source": "attacked_image_vae_latent_projection_watermark_rescore",
        "latent_projection_mode": "periodic_slot_pooled_content_carrier",
    }


def retention_proxy_rescore_from_quality(
    source_image: Any,
    attacked_image: Any,
    threshold: float,
) -> tuple[dict[str, float | str], dict[str, Any]]:
    """保留旧的质量 retention 代理路径, 仅用于诊断或显式降级."""

    metrics, raw_score, aligned_score, decision = quality_detection_scores(source_image, attacked_image, threshold)
    return metrics, {
        "raw_content_score_after": raw_score,
        "aligned_content_score_after": aligned_score,
        "threshold_score_after": aligned_score,
        "positive_by_content": raw_score >= threshold,
        "geometry_reliable": False,
        "rescue_eligible": False,
        "rescue_applied": False,
        "evidence_decision": decision,
        "formal_detection_decision": decision,
        "attacked_image_rescore_performed": False,
        "formal_detection_proxy": True,
        "detection_score_source": "image_quality_retention_proxy",
    }


def unsupported_record(
    root_path: Path,
    source_path: Path,
    source_digest: str,
    spec: RealAttackSpec,
    config: RealAttackEvaluationConfig,
    error: Exception,
) -> RealAttackDetectionRecord:
    """构造单个真实攻击无法执行时的可审计记录."""
    return RealAttackDetectionRecord(
        real_attack_record_id=real_attack_record_id(source_digest, spec, "not_generated"),
        source_image_id=source_image_id(source_path, source_digest),
        source_image_path=relative_or_absolute(source_path, root_path),
        source_image_digest=source_digest,
        source_image_digest_source="sha256_file",
        attack_id=spec.attack_id,
        attack_family=spec.attack_family,
        attack_name=spec.attack_name,
        attack_strength=spec.attack_strength,
        attack_parameters=dict(spec.attack_parameters),
        attack_implementation=spec.attack_implementation,
        attacked_image_path="",
        attacked_image_digest="",
        attacked_image_digest_source="not_generated",
        attacked_image_available=False,
        attack_performed=False,
        detection_method="real_image_quality_proxy_after_attack",
        detection_threshold=config.detection_threshold,
        raw_content_score_after=0.0,
        aligned_content_score_after=0.0,
        evidence_decision=False,
        metric_status="unsupported",
        unsupported_reason=f"{type(error).__name__}:{str(error)[:160]}",
        supports_paper_claim=False,
        metadata={"requires_colab_gpu": True, "claim_boundary": "not_paper_ready"},
    )


def build_attack_record(
    root_path: Path,
    source_path: Path,
    source_image: Any,
    attacked_image: Any,
    attacked_path: Path,
    spec: RealAttackSpec,
    config: RealAttackEvaluationConfig,
    source_context: dict[str, Any] | None = None,
    boundary: dict[str, Any] | None = None,
    detector_pipeline: Any | None = None,
) -> tuple[RealAttackDetectionRecord, dict[str, Any]]:
    """由真实 attacked image 构造检测记录和注册表行."""
    source_digest = file_digest(source_path)
    attacked_digest = file_digest(attacked_path)
    metrics: dict[str, Any]
    if config.enable_attacked_image_latent_rescore and source_context and boundary and detector_pipeline is not None:
        metrics = compute_image_quality_metrics(source_image, attacked_image)
        rescore = rescore_attacked_image_with_detector(
            detector_pipeline=detector_pipeline,
            attacked_image=attacked_image,
            source_context=source_context,
            boundary=boundary,
            config=config,
        )
        metric_status = REAL_ATTACK_WATERMARK_RESCORE_STATUS
        detection_method = "real_attacked_image_vae_latent_projection_watermark_rescore"
    else:
        if config.require_attacked_image_latent_rescore:
            raise RuntimeError("attacked_image_latent_rescore_unavailable")
        metrics, rescore = retention_proxy_rescore_from_quality(source_image, attacked_image, config.detection_threshold)
        metric_status = "measured_from_real_attacked_image"
        detection_method = "real_image_quality_proxy_after_attack"
    record = RealAttackDetectionRecord(
        real_attack_record_id=real_attack_record_id(source_digest, spec, attacked_digest),
        source_image_id=source_image_id(source_path, source_digest),
        source_image_path=relative_or_absolute(source_path, root_path),
        source_image_digest=source_digest,
        source_image_digest_source="sha256_file",
        attack_id=spec.attack_id,
        attack_family=spec.attack_family,
        attack_name=spec.attack_name,
        attack_strength=spec.attack_strength,
        attack_parameters=dict(spec.attack_parameters),
        attack_implementation=spec.attack_implementation,
        attacked_image_path=relative_or_absolute(attacked_path, root_path),
        attacked_image_digest=attacked_digest,
        attacked_image_digest_source="sha256_file",
        attacked_image_available=True,
        attack_performed=True,
        detection_method=detection_method,
        detection_threshold=config.detection_threshold,
        raw_content_score_after=float(rescore["raw_content_score_after"]),
        aligned_content_score_after=float(rescore["aligned_content_score_after"]),
        evidence_decision=bool(rescore["evidence_decision"]),
        metric_status=metric_status,
        unsupported_reason="",
        supports_paper_claim=False,
        metadata={
            "image_quality_metrics": metrics,
            "attacked_image_latent_rescore": rescore,
            "attacked_image_rescore_performed": bool(rescore["attacked_image_rescore_performed"]),
            "formal_detection_proxy": bool(rescore["formal_detection_proxy"]),
            "detection_score_source": rescore["detection_score_source"],
            "claim_boundary": "requires_attack_matrix_and_fixed_fpr_rebuild",
            "attacked_image_closed_loop": True,
            "resource_profile": str(getattr(spec, "resource_profile", "full_extra")),
            "requires_gpu": bool(getattr(spec, "requires_gpu", True)),
        },
    )
    registry_row = {
        "real_attack_record_id": record.real_attack_record_id,
        "source_image_id": record.source_image_id,
        "source_image_path": record.source_image_path,
        "source_image_digest": record.source_image_digest,
        "attacked_image_path": record.attacked_image_path,
        "attacked_image_digest": record.attacked_image_digest,
        "attack_name": record.attack_name,
        "attack_implementation": record.attack_implementation,
        "metric_status": record.metric_status,
        "supports_paper_claim": False,
    }
    return record, registry_row


def prompt_lookup(root_path: Path) -> dict[str, str]:
    """读取 prompt_id 到 prompt_text 的映射."""
    rows = read_jsonl(root_path / "outputs" / "prompt_event_protocol" / "prompt_records.jsonl")
    return {str(row["prompt_id"]): str(row["prompt_text"]) for row in rows}


def preferred_rescoring_record(
    records_by_prompt_role: dict[tuple[str, str], dict[str, Any]],
    records_by_prompt: dict[str, dict[str, Any]],
    prompt_id: str,
    sample_role: str,
) -> dict[str, Any]:
    """优先按 prompt 和 sample_role 读取 rescoring 记录.

    该函数修复的核心问题是: 同一个 prompt 会同时产生 `positive_source`、
    `clean_negative` 和 `attacked_negative` 记录, 不能只按 prompt id 建立单值
    字典, 否则后写入的角色会覆盖真实 source image 对应的角色。
    """

    return records_by_prompt_role.get((prompt_id, sample_role), records_by_prompt.get(prompt_id, {}))


def build_source_context(
    *,
    prompt_id: str,
    prompt_text: str,
    source_record: dict[str, Any],
    sample_role: str,
    geometry_record: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """把 rescoring 记录转换为攻击闭环可复用的 source context."""

    effective_role = str(source_record.get("sample_role", sample_role) or sample_role)
    geometry = geometry_record or {}
    return {
        "prompt_id": prompt_id,
        "prompt_text": prompt_text,
        "source_record": source_record,
        "split": source_record.get("split", "unknown"),
        "sample_role": effective_role,
        "raw_content_score_before": float(source_record.get("real_raw_content_score", source_record.get("raw_content_score", 0.0))),
        "aligned_content_score_before": float(
            source_record.get("real_aligned_content_score", source_record.get("aligned_content_score", 0.0))
        ),
        "latent_projection_values_before": source_record.get("latent_projection_values_before", ()),
        "latent_projection_values_after": source_record.get("latent_projection_values_after", ()),
        "latent_projection_digest_before": source_record.get("latent_projection_digest_before", ""),
        "latent_projection_digest_after": source_record.get("latent_projection_digest_after", ""),
        "geometry_evidence_record_id": geometry.get("geometry_evidence_record_id", ""),
        "attention_graph_id": geometry.get("attention_graph_id", source_record.get("attention_graph_id", "")),
        "capture_id": geometry.get("capture_id", source_record.get("capture_id", "")),
        "registration_confidence": float(geometry.get("registration_confidence", 0.0)),
        "anchor_inlier_ratio": float(geometry.get("anchor_inlier_ratio", 0.0)),
        "recovered_sync_consistency": float(geometry.get("recovered_sync_consistency", 0.0)),
        "alignment_residual": float(geometry.get("alignment_residual", 1.0)),
        "geometry_reliable": bool(geometry.get("geometry_reliable", False)),
        "fail_reason": geometry.get("fail_reason", source_record.get("fail_reason", "geometry_suspected")),
    }


def geometric_rescue_context_by_content_record_id(root_path: Path) -> dict[str, dict[str, Any]]:
    """读取 full_rescue 几何恢复记录, 作为攻击后 same-threshold rescue 证据."""

    rows = read_jsonl(root_path / "outputs" / "geometric_rescue" / "aligned_detection_records.jsonl")
    mapping: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row.get("rescue_ablation_mode") != "full_rescue":
            continue
        content_id = str(row.get("content_detection_record_id", ""))
        if content_id and content_id not in mapping:
            mapping[content_id] = row
    return mapping


def source_context_by_image_path(root_path: Path, config: RealAttackEvaluationConfig) -> dict[str, dict[str, Any]]:
    """把 clean / aligned image 路径映射回真实 aligned rescoring 记录与 prompt."""
    quality_rows = read_csv_rows(root_path / "outputs" / "aligned_rescoring" / "aligned_rescoring_quality_metrics.csv")
    rescoring_rows = read_jsonl(root_path / "outputs" / "aligned_rescoring" / "aligned_rescoring_records.jsonl")
    prompts = prompt_lookup(root_path)
    geometry_by_content_id = geometric_rescue_context_by_content_record_id(root_path)
    records_by_prompt: dict[str, dict[str, Any]] = {}
    records_by_prompt_role: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rescoring_rows:
        prompt_id = str(row.get("prompt_id", ""))
        sample_role = str(row.get("sample_role", ""))
        records_by_prompt.setdefault(prompt_id, row)
        records_by_prompt_role[(prompt_id, sample_role)] = row
    contexts: dict[str, dict[str, Any]] = {}
    for quality_row in quality_rows:
        prompt_id = str(quality_row.get("prompt_id", ""))
        prompt_text = prompts.get(prompt_id, config.prompt)
        for image_field, sample_role in (
            ("aligned_image_path", "positive_source"),
            ("clean_image_path", "clean_negative"),
        ):
            image_path = str(quality_row.get(image_field, "")).strip()
            if not image_path:
                continue
            source_record = preferred_rescoring_record(records_by_prompt_role, records_by_prompt, prompt_id, sample_role)
            contexts[image_path] = build_source_context(
                prompt_id=prompt_id,
                prompt_text=prompt_text,
                source_record=source_record,
                sample_role=sample_role,
                geometry_record=geometry_by_content_id.get(str(source_record.get("content_detection_record_id", ""))),
            )
    return contexts


def context_for_source(
    source_path: Path,
    root_path: Path,
    contexts: dict[str, dict[str, Any]],
    config: RealAttackEvaluationConfig,
) -> dict[str, Any]:
    """为 source image 查找对应 prompt 和正式检测源记录."""
    relative_path = relative_or_absolute(source_path, root_path)
    return contexts.get(
        relative_path,
        {
            "prompt_id": "",
            "prompt_text": config.prompt,
            "source_record": {},
            "split": "unknown",
            "sample_role": "unknown",
            "raw_content_score_before": 0.0,
            "aligned_content_score_before": 0.0,
            "latent_projection_values_before": (),
            "latent_projection_values_after": (),
            "latent_projection_digest_before": "",
            "latent_projection_digest_after": "",
            "geometry_reliable": False,
            "fail_reason": "geometry_suspected",
        },
    )


def formal_boundary(root_path: Path, config: RealAttackEvaluationConfig) -> dict[str, Any]:
    """读取 fixed-FPR 和 rescue 边界, 缺失时保留不可支持状态."""
    thresholds = read_json(root_path / "outputs" / "threshold_calibration" / "calibration_thresholds.json")
    report = read_json(root_path / "outputs" / "threshold_calibration" / "threshold_degeneracy_report.json")
    threshold_value = float(thresholds.get("threshold_value", report.get("calibrated_content_threshold", config.detection_threshold)))
    threshold_metadata = thresholds.get("metadata", {}) if isinstance(thresholds.get("metadata", {}), dict) else {}
    return {
        "content_threshold": threshold_value,
        "target_fpr": float(thresholds.get("target_fpr", report.get("target_fpr", PILOT_PAPER_FIXED_FPR))),
        "rescue_margin_low": float(report.get("rescue_margin_low", -0.05)),
        "allowed_fail_reasons": tuple(report.get("allowed_fail_reasons", ("geometry_suspected", "low_confidence"))),
        "fixed_fpr_control_scope": str(report.get("fixed_fpr_control_scope", "calibration_clean_negative")),
        "fixed_fpr_denominator_role": str(report.get("fixed_fpr_denominator_role", "clean_negative_only")),
        "rescue_control_scope": str(report.get("rescue_control_scope", "evidence_clean_negative")),
        "rescue_changes_fpr_denominator": bool(report.get("rescue_changes_fpr_denominator", False)),
        "attacked_negative_boundary_role": str(
            report.get(
                "attacked_negative_boundary_role",
                "attack_robustness_diagnostic_not_fpr_denominator",
            )
        ),
        "attacked_negative_governs_fixed_fpr": bool(report.get("attacked_negative_governs_fixed_fpr", False)),
        "score_space_name": str(report.get("score_space_name", threshold_metadata.get("score_space_name", ""))),
        "threshold_score_field": str(report.get("threshold_score_field", threshold_metadata.get("threshold_score_field", "raw_content_score"))),
        "threshold_score_source_field": str(
            report.get("threshold_score_source_field", threshold_metadata.get("threshold_score_source_field", "raw_content_score"))
        ),
        "score_space_alignment_ready": bool(report.get("score_space_alignment_ready", False)),
        "real_score_calibration_ready": bool(report.get("real_score_calibration_ready", False)),
        "calibration_records_source": str(report.get("calibration_records_source", "")),
        "boundary_ready": bool(thresholds and report),
    }


def formal_attack_record_id(real_record: dict[str, Any], boundary: dict[str, Any]) -> str:
    """构造 attack matrix 兼容记录 id."""
    payload = {
        "real_attack_record_id": real_record["real_attack_record_id"],
        "source_image_digest": real_record["source_image_digest"],
        "attacked_image_digest": real_record["attacked_image_digest"],
        "content_threshold": boundary["content_threshold"],
        "threshold_score_field": boundary.get("threshold_score_field", "raw_content_score"),
    }
    return f"attack_record_{build_stable_digest(payload)[:16]}"


def build_formal_attack_record(real_record: dict[str, Any], source_context: dict[str, Any], boundary: dict[str, Any]) -> dict[str, Any]:
    """把真实 attacked image 结果接回 attack matrix 正式记录 schema."""
    raw_before = float(source_context["raw_content_score_before"])
    aligned_before = float(source_context["aligned_content_score_before"])
    rescore_metadata = real_record.get("metadata", {}).get("attacked_image_latent_rescore", {})
    attacked_image_rescore_performed = bool(real_record.get("metadata", {}).get("attacked_image_rescore_performed", False))
    if attacked_image_rescore_performed:
        raw_after = float(real_record["raw_content_score_after"])
        aligned_after = float(real_record["aligned_content_score_after"])
        detection_score_source = str(
            rescore_metadata.get("detection_score_source", "attacked_image_vae_latent_projection_watermark_rescore")
        )
        formal_detection_method = "fixed_fpr_attack_matrix_schema_from_real_attacked_image_latent_rescore"
        metric_status = FORMAL_WATERMARK_RESCORE_STATUS if boundary["boundary_ready"] else "formal_boundary_missing"
    else:
        retention = float(real_record["aligned_content_score_after"])
        raw_after = raw_before * retention
        aligned_after = aligned_before * retention
        detection_score_source = "pre_attack_latent_score_scaled_by_attacked_image_quality_retention"
        formal_detection_method = "fixed_fpr_attack_matrix_schema_from_real_attacked_image_retention_proxy"
        metric_status = FORMAL_RETENTION_PROXY_STATUS if boundary["boundary_ready"] else "formal_boundary_missing"
    threshold = float(boundary["content_threshold"])
    threshold_score_field = str(boundary.get("threshold_score_field", "raw_content_score"))
    threshold_score_after = aligned_after if threshold_score_field in {"aligned_content_score", "formal_detection_score"} else raw_after
    margin_after = raw_after - threshold
    aligned_margin_after = aligned_after - threshold
    formal_detection_margin_after = threshold_score_after - threshold
    positive_by_content = margin_after >= 0.0
    formal_detection_decision = formal_detection_margin_after >= 0.0
    geometry_reliable = bool(rescore_metadata.get("geometry_reliable", source_context["geometry_reliable"]))
    rescue_eligible = (
        boundary["rescue_margin_low"] <= margin_after < 0.0
        and geometry_reliable
        and source_context["fail_reason"] in boundary["allowed_fail_reasons"]
    )
    rescue_applied = rescue_eligible and aligned_margin_after >= 0.0
    evidence_decision = positive_by_content or rescue_applied or formal_detection_decision
    score_retention = max(0.0, aligned_after) / max(max(0.0, aligned_before), 1e-6)
    quality_proxy_default = score_retention if attacked_image_rescore_performed else float(real_record["aligned_content_score_after"])
    attack_config_digest = build_stable_digest(
        {
            "attack_id": real_record["attack_id"],
            "attack_name": real_record["attack_name"],
            "attack_strength": real_record["attack_strength"],
            "attack_parameters": real_record["attack_parameters"],
            "attack_implementation": real_record["attack_implementation"],
        }
    )
    record_id = formal_attack_record_id(real_record, boundary)
    record_digest = build_stable_digest(
        {
            "record_id": record_id,
            "source_image_digest": real_record["source_image_digest"],
            "attacked_image_digest": real_record["attacked_image_digest"],
            "raw_content_score_after": raw_after,
            "aligned_content_score_after": aligned_after,
            "threshold_score_after": threshold_score_after,
            "boundary": boundary,
        }
    )
    return {
        "attack_record_id": record_id,
        "attack_record_digest": record_digest,
        "source_record_id": source_context.get("source_record", {}).get("aligned_rescoring_record_id", real_record["source_image_id"]),
        "source_image_digest": real_record["source_image_digest"],
        "source_image_digest_source": real_record["source_image_digest_source"],
        "attack_id": real_record["attack_id"],
        "attack_family": real_record["attack_family"],
        "attack_name": real_record["attack_name"],
        "attack_strength": real_record["attack_strength"],
        "resource_profile": str(real_record.get("metadata", {}).get("resource_profile", "full_extra")),
        "requires_gpu": bool(real_record.get("metadata", {}).get("requires_gpu", True)),
        "attack_parameters": real_record["attack_parameters"],
        "attack_config_digest": attack_config_digest,
        "attacked_image_digest": real_record["attacked_image_digest"],
        "attacked_image_digest_source": real_record["attacked_image_digest_source"],
        "attacked_image_available": real_record["attacked_image_available"],
        "attack_performed": real_record["attack_performed"],
        "split": source_context["split"],
        "sample_role": source_context["sample_role"],
        "raw_content_score_before": raw_before,
        "raw_content_score_after": raw_after,
        "aligned_content_score_before": aligned_before,
        "aligned_content_score_after": aligned_after,
        "lf_score_retention": score_retention,
        "hf_score_retention": score_retention,
        "score_retention": score_retention,
        "quality_score_proxy": real_record["metadata"].get("image_quality_metrics", {}).get("ssim", quality_proxy_default),
        "attention_consistency_proxy": float(
            rescore_metadata.get("recovered_sync_consistency", source_context.get("recovered_sync_consistency", 0.0))
        )
        if attacked_image_rescore_performed
        else float(real_record["aligned_content_score_after"]),
        "geometry_reliable": geometry_reliable,
        "rescue_eligible": rescue_eligible,
        "rescue_applied": rescue_applied,
        "evidence_decision": evidence_decision,
        "metric_status": metric_status,
        "unsupported_reason": "" if boundary["boundary_ready"] else "threshold_calibration_inputs_missing",
        "supports_paper_claim": False,
        "metadata": {
            "real_attack_record_id": real_record["real_attack_record_id"],
            "attacked_image_path": real_record["attacked_image_path"],
            "source_image_path": real_record["source_image_path"],
            "detection_method": formal_detection_method,
            "attack_implementation": real_record["attack_implementation"],
            "formal_boundary_ready": boundary["boundary_ready"],
            "threshold_score_field": threshold_score_field,
            "threshold_score_source_field": boundary.get("threshold_score_source_field", ""),
            "threshold_score_after": threshold_score_after,
            "formal_detection_decision": formal_detection_decision,
            "formal_detection_proxy": not attacked_image_rescore_performed,
            "attacked_image_rescore_performed": attacked_image_rescore_performed,
            "attacked_image_rescore_required_for_claim": True,
            "detection_score_source": detection_score_source,
            "retention_source_field": ""
            if attacked_image_rescore_performed
            else "real_attack_detection_records.aligned_content_score_after",
            "same_threshold_rescue_source": "attacked_image_latent_rescore"
            if attacked_image_rescore_performed
            else "retention_proxy_boundary",
            "geometry_evidence_record_id": source_context.get("geometry_evidence_record_id", ""),
            "attention_graph_id": source_context.get("attention_graph_id", ""),
            "capture_id": source_context.get("capture_id", ""),
            "attacked_latent_projection_digest": rescore_metadata.get("attacked_latent_projection_digest", ""),
            "watermark_coordinate": rescore_metadata.get("watermark_coordinate", ""),
            "bounded_watermark_coordinate": rescore_metadata.get("bounded_watermark_coordinate", ""),
            "score_space_name": boundary.get("score_space_name", ""),
            "score_space_alignment_ready": boundary.get("score_space_alignment_ready", False),
            "real_score_calibration_ready": boundary.get("real_score_calibration_ready", False),
            "calibration_records_source": boundary.get("calibration_records_source", ""),
            "claim_boundary": "requires_full_sample_scale_and_evidence_audit",
        },
    }


def _is_measured_attack_record(record: dict[str, Any]) -> bool:
    """判断记录是否进入真实攻击测量空间.

    通用工程写法是让聚合函数只依赖稳定的 metric_status 前缀, 避免把同一攻击的非正式检测记录
    和 formal records 混在不同统计口径中。项目特定设计是: formal records 会把真实攻击图像分数
    映射到 fixed-FPR 边界, 因此 family metrics 必须优先使用 formal records 作为事实来源。
    """

    return str(record.get("metric_status", "")).startswith("measured_from_real_attacked_image")


def build_family_metrics(records: tuple[dict[str, Any], ...]) -> tuple[dict[str, Any], ...]:
    """按攻击名称聚合真实攻击闭环检测指标.

    `detection_positive_rate` 表示 formal positive_source 记录的通过率, 不把 clean negative 混入
    TPR 分母。clean negative 的误检率单独写入 `formal_clean_false_positive_rate`, 从而和
    fixed-FPR 协议的统计边界保持一致。
    """
    rows: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault((record["attack_family"], record["attack_name"]), []).append(record)
    for (attack_family, attack_name), group in sorted(grouped.items()):
        measured = [record for record in group if _is_measured_attack_record(record)]
        positive_records = [record for record in measured if record.get("sample_role", "positive_source") == "positive_source"]
        clean_negative_records = [record for record in measured if record.get("sample_role") == "clean_negative"]
        positive_decisions = [bool(record["evidence_decision"]) for record in positive_records]
        clean_negative_decisions = [bool(record["evidence_decision"]) for record in clean_negative_records]
        all_decisions = [bool(record["evidence_decision"]) for record in measured]
        positive_scores = [float(record["aligned_content_score_after"]) for record in positive_records]
        all_scores = [float(record["aligned_content_score_after"]) for record in measured]
        rows.append(
            {
                "attack_family": attack_family,
                "attack_name": attack_name,
                "attack_record_count": len(group),
                "measured_record_count": len(measured),
                "unsupported_record_count": len(group) - len(measured),
                "real_attacked_image_count": sum(1 for record in measured if record["attacked_image_available"]),
                "formal_positive_count": len(positive_records),
                "formal_clean_negative_count": len(clean_negative_records),
                "detection_positive_rate": (
                    sum(1 for decision in positive_decisions if decision) / len(positive_decisions)
                    if positive_decisions
                    else 0.0
                ),
                "formal_clean_false_positive_rate": (
                    sum(1 for decision in clean_negative_decisions if decision) / len(clean_negative_decisions)
                    if clean_negative_decisions
                    else 0.0
                ),
                "all_record_positive_rate": (
                    sum(1 for decision in all_decisions if decision) / len(all_decisions) if all_decisions else 0.0
                ),
                "aligned_content_score_after_mean": sum(positive_scores) / len(positive_scores) if positive_scores else 0.0,
                "all_record_aligned_content_score_after_mean": sum(all_scores) / len(all_scores) if all_scores else 0.0,
                "metric_status": (
                    "measured_from_real_attacked_image_formal_records"
                    if positive_records or clean_negative_records
                    else "measured_from_real_attacked_image"
                    if measured
                    else "unsupported"
                ),
                "metrics_source": "formal_attack_detection_records" if positive_records or clean_negative_records else "attack_detection_records",
                "supports_paper_claim": False,
            }
        )
    return tuple(rows)


def write_failure_outputs(
    root_path: Path,
    config: RealAttackEvaluationConfig,
    output_dir: Path,
    error: Exception,
) -> dict[str, Any]:
    """在真实后端不可用时写出失败摘要和环境记录."""
    environment_report = build_runtime_environment_report()
    environment_path = output_dir / "real_attack_environment_report.json"
    result_path = output_dir / "real_attack_run_summary.json"
    manifest_path = output_dir / "real_attack_manifest.local.json"
    environment_path.write_text(stable_json_text(environment_report), encoding="utf-8")
    result = RealAttackEvaluationResult(
        run_id=build_stable_digest({"error": type(error).__name__, "model_id": config.model_id, "seed": config.seed}),
        model_family=config.model_family,
        model_id=config.model_id,
        run_decision="fail",
        unsupported_reason=f"{type(error).__name__}:{str(error)[:160]}",
        source_image_count=0,
        real_attack_record_count=0,
        real_attacked_image_count=0,
        regeneration_attack_record_count=0,
        required_regeneration_attack_count=len(REQUIRED_REAL_GPU_ATTACKS),
        required_real_gpu_attack_count=len(REQUIRED_REAL_GPU_ATTACKS),
        measured_regeneration_attack_count=0,
        measured_real_gpu_attack_count=0,
        real_attacked_image_closed_loop_ready=False,
        attacked_image_rescore_count=0,
        attacked_image_rescore_ready=False,
        proxy_formal_record_count=0,
        regeneration_attack_gpu_validation_ready=False,
        real_gpu_attack_validation_ready=False,
        attack_detection_rerun_ready=False,
        formal_attack_detection_ready=False,
        image_quality_metrics_ready=False,
        supports_paper_claim=False,
        output_records_path="",
        formal_records_path="",
        attacked_image_registry_path="",
        attack_family_metrics_path="",
        environment_report_path=relative_or_absolute(environment_path, root_path),
        manifest_path=relative_or_absolute(manifest_path, root_path),
        metadata={
            "runtime_environment": environment_report,
            "claim_boundary": "not_paper_ready",
        },
    )
    result_path.write_text(stable_json_text(result.to_dict()), encoding="utf-8")
    manifest = build_artifact_manifest(
        artifact_id="real_attack_evaluation_manifest",
        artifact_type="local_manifest",
        input_paths=(),
        output_paths=(relative_or_absolute(result_path, root_path), relative_or_absolute(environment_path, root_path)),
        config=asdict(config),
        code_version=resolve_code_version(root_path),
        rebuild_command="运行 paper_workflow/real_attack_evaluation_run.ipynb",
        metadata={"supports_paper_claim": False, "run_decision": "fail"},
    ).to_dict()
    manifest_path.write_text(stable_json_text(manifest), encoding="utf-8")
    return result.to_dict()


def write_real_attack_evaluation_outputs(config: RealAttackEvaluationConfig, root: str | Path = ".") -> dict[str, Any]:
    """运行真实图像级攻击闭环并写出 records、registry、metrics 和 manifest."""
    root_path = Path(root).resolve()
    output_dir = (root_path / config.output_dir).resolve()
    attacked_dir = output_dir / "attacked_images"
    output_dir.mkdir(parents=True, exist_ok=True)
    attacked_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "real_attack_detection_records.jsonl"
    formal_records_path = output_dir / "formal_attack_detection_records.jsonl"
    registry_path = output_dir / "real_attacked_image_registry.jsonl"
    metrics_path = output_dir / "real_attack_family_metrics.csv"
    result_path = output_dir / "real_attack_run_summary.json"
    environment_path = output_dir / "real_attack_environment_report.json"
    manifest_path = output_dir / "real_attack_manifest.local.json"

    try:
        source_paths = discover_source_images(root_path, config)
        if not source_paths:
            raise FileNotFoundError("source_image_files_missing")
        source_contexts = source_context_by_image_path(root_path, config)
        boundary = formal_boundary(root_path, config)
        pipeline, runtime_versions = load_img2img_pipeline(config)
    except Exception as error:
        return write_failure_outputs(root_path, config, output_dir, error)

    records: list[dict[str, Any]] = []
    registry_rows: list[dict[str, Any]] = []
    contexts_by_record_source_path: dict[str, dict[str, Any]] = {}
    pending_ddim_attacks: list[tuple[Path, Path, RealAttackSpec]] = []
    specs = default_attack_specs()
    main_specs = tuple(spec for spec in specs if spec.attack_name != "ddim_inversion_regeneration")
    ddim_specs = tuple(spec for spec in specs if spec.attack_name == "ddim_inversion_regeneration")
    total_attack_tasks = len(source_paths) * len(specs)
    with progress_bar(
        total_attack_tasks,
        desc="real regeneration attacks",
        enabled=config.enable_attack_progress_bar,
    ) as attack_progress:
        for source_index, source_path in enumerate(source_paths):
            source_digest = file_digest(source_path)
            source_image = load_rgb_image(source_path, config)
            source_context = context_for_source(source_path, root_path, source_contexts, config)
            prompt_text = str(source_context["prompt_text"])
            contexts_by_record_source_path[relative_or_absolute(source_path, root_path)] = source_context
            for attack_index, spec in enumerate(main_specs):
                attack_seed = config.seed + source_index * 101 + attack_index
                try:
                    attacked_image = run_pipeline_attack(pipeline, source_image, spec, config, attack_seed, prompt_text)
                    attacked_image = normalize_attacked_image_size(attacked_image, source_image)
                    attacked_path = attacked_dir / f"{source_path.stem}_{spec.attack_name}_{source_digest[:8]}.png"
                    attacked_image.save(attacked_path)
                    record, registry_row = build_attack_record(
                        root_path=root_path,
                        source_path=source_path,
                        source_image=source_image,
                        attacked_image=attacked_image,
                        attacked_path=attacked_path,
                        spec=spec,
                        config=config,
                        source_context=source_context,
                        boundary=boundary,
                        detector_pipeline=pipeline,
                    )
                    records.append(record.to_dict())
                    registry_rows.append(registry_row)
                except Exception as error:
                    records.append(unsupported_record(root_path, source_path, source_digest, spec, config, error).to_dict())
                finally:
                    update_progress(
                        attack_progress,
                        profile=(
                            f"attack={spec.attack_name} "
                            f"source={source_index + 1}/{len(source_paths)} "
                            f"seed={attack_seed}"
                        ),
                    )

        del pipeline
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        for source_index, source_path in enumerate(source_paths):
            source_digest = file_digest(source_path)
            source_image = load_rgb_image(source_path, config)
            source_context = context_for_source(source_path, root_path, source_contexts, config)
            prompt_text = str(source_context["prompt_text"])
            for attack_index, spec in enumerate(ddim_specs):
                attack_seed = config.seed + source_index * 101 + 1000 + attack_index
                try:
                    attacked_image = run_strict_ddim_inversion_attack(source_image, spec, config, attack_seed, prompt_text)
                    attacked_image = normalize_attacked_image_size(attacked_image, source_image)
                    attacked_path = attacked_dir / f"{source_path.stem}_{spec.attack_name}_{source_digest[:8]}.png"
                    attacked_image.save(attacked_path)
                    pending_ddim_attacks.append((source_path, attacked_path, spec))
                except Exception as error:
                    records.append(unsupported_record(root_path, source_path, source_digest, spec, config, error).to_dict())
                finally:
                    update_progress(
                        attack_progress,
                        profile=(
                            f"attack={spec.attack_name} "
                            f"source={source_index + 1}/{len(source_paths)} "
                            f"seed={attack_seed}"
                        ),
                    )

    clear_strict_ddim_inversion_runtime_cache()

    if pending_ddim_attacks:
        try:
            detector_pipeline, detector_runtime_versions = load_detector_pipeline(config)
            runtime_versions = {**runtime_versions, **detector_runtime_versions}
            patch_pillow_typing_ink_compatibility()
            from PIL import Image

            for source_path, attacked_path, spec in pending_ddim_attacks:
                source_digest = file_digest(source_path)
                try:
                    source_image = load_rgb_image(source_path, config)
                    attacked_image = Image.open(attacked_path).convert("RGB")
                    source_context = context_for_source(source_path, root_path, source_contexts, config)
                    record, registry_row = build_attack_record(
                        root_path=root_path,
                        source_path=source_path,
                        source_image=source_image,
                        attacked_image=attacked_image,
                        attacked_path=attacked_path,
                        spec=spec,
                        config=config,
                        source_context=source_context,
                        boundary=boundary,
                        detector_pipeline=detector_pipeline,
                    )
                    records.append(record.to_dict())
                    registry_rows.append(registry_row)
                except Exception as error:
                    records.append(unsupported_record(root_path, source_path, source_digest, spec, config, error).to_dict())
        except Exception as error:
            for source_path, _, spec in pending_ddim_attacks:
                records.append(unsupported_record(root_path, source_path, file_digest(source_path), spec, config, error).to_dict())
        finally:
            try:
                del detector_pipeline
            except Exception:
                pass
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

    record_rows = tuple(records)
    registry_tuple = tuple(registry_rows)
    formal_rows = tuple(
        build_formal_attack_record(record, contexts_by_record_source_path[record["source_image_path"]], boundary)
        for record in record_rows
        if _is_measured_attack_record(record)
    )
    family_metrics = build_family_metrics(formal_rows)
    records_path.write_text(jsonl_text(record_rows), encoding="utf-8")
    formal_records_path.write_text(jsonl_text(formal_rows), encoding="utf-8")
    registry_path.write_text(jsonl_text(registry_tuple), encoding="utf-8")
    write_csv(metrics_path, family_metrics)
    environment_report = runtime_versions.get("runtime_environment", build_runtime_environment_report())
    environment_path.write_text(stable_json_text(environment_report), encoding="utf-8")

    measured_names = {record["attack_name"] for record in record_rows if _is_measured_attack_record(record)}
    real_attacked_image_count = sum(1 for record in record_rows if record["attacked_image_available"])
    closed_loop_ready = real_attacked_image_count > 0 and all(
        row.get("source_image_digest") and row.get("attacked_image_digest") for row in registry_tuple
    )
    regeneration_ready = all(name in measured_names for name in REQUIRED_REGENERATION_ATTACKS)
    real_gpu_attack_ready = all(name in measured_names for name in REQUIRED_REAL_GPU_ATTACKS)
    detection_ready = any(_is_measured_attack_record(record) for record in record_rows)
    attacked_image_rescore_count = sum(
        1 for record in record_rows if bool(record.get("metadata", {}).get("attacked_image_rescore_performed", False))
    )
    proxy_formal_record_count = sum(1 for record in formal_rows if bool(record.get("metadata", {}).get("formal_detection_proxy", False)))
    attacked_image_rescore_ready = attacked_image_rescore_count == real_attacked_image_count and real_attacked_image_count > 0
    formal_ready = (
        bool(formal_rows)
        and boundary["boundary_ready"]
        and len(formal_rows) == real_attacked_image_count
        and (attacked_image_rescore_ready or not config.require_attacked_image_latent_rescore)
    )
    image_quality_ready = all(
        "image_quality_metrics" in record.get("metadata", {})
        for record in record_rows
        if _is_measured_attack_record(record)
    )
    run_decision = (
        "pass"
        if closed_loop_ready
        and detection_ready
        and formal_ready
        and (real_gpu_attack_ready or not config.require_all_regeneration_attacks)
        else "fail"
    )
    unsupported_reason = "" if run_decision == "pass" else "real_attack_closed_loop_incomplete"
    result = RealAttackEvaluationResult(
        run_id=build_stable_digest({"records": record_rows, "config": asdict(config)}),
        model_family=config.model_family,
        model_id=config.model_id,
        run_decision=run_decision,
        unsupported_reason=unsupported_reason,
        source_image_count=len(source_paths),
        real_attack_record_count=len(record_rows),
        real_attacked_image_count=real_attacked_image_count,
        regeneration_attack_record_count=sum(1 for record in record_rows if record["attack_family"] == "regeneration_attack"),
        required_regeneration_attack_count=len(REQUIRED_REAL_GPU_ATTACKS),
        required_real_gpu_attack_count=len(REQUIRED_REAL_GPU_ATTACKS),
        measured_regeneration_attack_count=len(measured_names.intersection(REQUIRED_REAL_GPU_ATTACKS)),
        measured_real_gpu_attack_count=len(measured_names.intersection(REQUIRED_REAL_GPU_ATTACKS)),
        real_attacked_image_closed_loop_ready=closed_loop_ready,
        attacked_image_rescore_count=attacked_image_rescore_count,
        attacked_image_rescore_ready=attacked_image_rescore_ready,
        proxy_formal_record_count=proxy_formal_record_count,
        regeneration_attack_gpu_validation_ready=real_gpu_attack_ready,
        real_gpu_attack_validation_ready=real_gpu_attack_ready,
        attack_detection_rerun_ready=detection_ready,
        formal_attack_detection_ready=formal_ready,
        image_quality_metrics_ready=image_quality_ready,
        supports_paper_claim=False,
        output_records_path=relative_or_absolute(records_path, root_path),
        formal_records_path=relative_or_absolute(formal_records_path, root_path),
        attacked_image_registry_path=relative_or_absolute(registry_path, root_path),
        attack_family_metrics_path=relative_or_absolute(metrics_path, root_path),
        environment_report_path=relative_or_absolute(environment_path, root_path),
        manifest_path=relative_or_absolute(manifest_path, root_path),
        metadata={
            **runtime_versions,
            "required_regeneration_attacks": REQUIRED_REGENERATION_ATTACKS,
            "required_advanced_gpu_attacks": REQUIRED_ADVANCED_GPU_ATTACKS,
            "required_real_gpu_attacks": REQUIRED_REAL_GPU_ATTACKS,
            "real_gpu_attack_validation_ready": real_gpu_attack_ready,
            "formal_boundary": boundary,
            "require_attacked_image_latent_rescore": config.require_attacked_image_latent_rescore,
            "formal_watermark_rescore_status": FORMAL_WATERMARK_RESCORE_STATUS,
            "formal_retention_proxy_status": FORMAL_RETENTION_PROXY_STATUS,
            "claim_boundary": "requires_full_sample_scale_and_evidence_audit",
        },
    )
    result_path.write_text(stable_json_text(result.to_dict()), encoding="utf-8")
    manifest = build_artifact_manifest(
        artifact_id="real_attack_evaluation_manifest",
        artifact_type="local_manifest",
        input_paths=tuple(relative_or_absolute(path, root_path) for path in source_paths),
        output_paths=(
            relative_or_absolute(records_path, root_path),
            relative_or_absolute(formal_records_path, root_path),
            relative_or_absolute(registry_path, root_path),
            relative_or_absolute(metrics_path, root_path),
            relative_or_absolute(result_path, root_path),
            relative_or_absolute(environment_path, root_path),
        ),
        config=asdict(config),
        code_version=resolve_code_version(root_path),
        rebuild_command="运行 paper_workflow/real_attack_evaluation_run.ipynb",
        metadata={
            "supports_paper_claim": False,
            "run_decision": run_decision,
            "real_attacked_image_count": real_attacked_image_count,
            "attacked_image_rescore_count": attacked_image_rescore_count,
            "attacked_image_rescore_ready": attacked_image_rescore_ready,
            "proxy_formal_record_count": proxy_formal_record_count,
            "regeneration_attack_gpu_validation_ready": real_gpu_attack_ready,
            "real_gpu_attack_validation_ready": real_gpu_attack_ready,
            "formal_attack_detection_ready": formal_ready,
        },
    ).to_dict()
    manifest_path.write_text(stable_json_text(manifest), encoding="utf-8")
    return result.to_dict()


def build_default_config() -> RealAttackEvaluationConfig:
    """从环境变量构造默认 Colab 配置."""
    return RealAttackEvaluationConfig(
        model_family=os.environ.get("SLM_WM_MODEL_FAMILY", PRIMARY_MODEL_FAMILY),
        model_id=os.environ.get("SLM_WM_MODEL_ID", PRIMARY_MODEL_ID),
        seed=int(os.environ.get("SLM_WM_REAL_ATTACK_SEED", "20260621")),
        prompt=os.environ.get(
            "SLM_WM_REAL_ATTACK_PROMPT",
            "a calm studio portrait of a ceramic bird with soft geometric background",
        ),
        negative_prompt=os.environ.get("SLM_WM_NEGATIVE_PROMPT", "low quality, blurry"),
        width=int(os.environ.get("SLM_WM_IMAGE_WIDTH", "512")),
        height=int(os.environ.get("SLM_WM_IMAGE_HEIGHT", "512")),
        inference_steps=int(os.environ.get("SLM_WM_INFERENCE_STEPS", "20")),
        guidance_scale=float(os.environ.get("SLM_WM_GUIDANCE_SCALE", "5.0")),
        output_dir=os.environ.get("SLM_WM_REAL_ATTACK_OUTPUT_DIR", DEFAULT_OUTPUT_DIR),
        source_image_dir=os.environ.get("SLM_WM_REAL_ATTACK_SOURCE_IMAGE_DIR", DEFAULT_SOURCE_IMAGE_DIR),
        max_source_images=resolve_count_from_environment("SLM_WM_REAL_ATTACK_SOURCE_COUNT"),
        device_name=os.environ.get("SLM_WM_DEVICE", "cuda"),
        torch_dtype=os.environ.get("SLM_WM_TORCH_DTYPE", "float16"),
        detection_threshold=float(os.environ.get("SLM_WM_REAL_ATTACK_DETECTION_THRESHOLD", "0.50")),
        require_all_regeneration_attacks=os.environ.get("SLM_WM_REQUIRE_ALL_REGEN_ATTACKS", "1") != "0",
        ddim_attack_model_id=os.environ.get("SLM_WM_DDIM_ATTACK_MODEL_ID", DEFAULT_DDIM_ATTACK_MODEL_ID),
        ddim_inversion_steps=int(os.environ.get("SLM_WM_DDIM_INVERSION_STEPS", "30")),
        ddim_reconstruction_steps=int(os.environ.get("SLM_WM_DDIM_RECONSTRUCTION_STEPS", "30")),
        enable_pipeline_progress_bar=os.environ.get("SLM_WM_ENABLE_PIPELINE_PROGRESS_BAR", "0") == "1",
        enable_attack_progress_bar=os.environ.get("SLM_WM_ENABLE_ATTACK_PROGRESS_BAR", "1") != "0",
        enable_attacked_image_latent_rescore=os.environ.get("SLM_WM_ENABLE_ATTACKED_IMAGE_LATENT_RESCORE", "1") != "0",
        require_attacked_image_latent_rescore=os.environ.get("SLM_WM_REQUIRE_ATTACKED_IMAGE_LATENT_RESCORE", "1") != "0",
    )


def run_default_real_attack_evaluation_plan(root: str | Path = ".") -> dict[str, Any]:
    """运行默认真实图像级攻击闭环计划."""
    return write_real_attack_evaluation_outputs(config=build_default_config(), root=root)


def run_default_real_attack_evaluation_from_drive_plan(
    root: str | Path = ".",
    aligned_rescoring_drive_dir: str | None = None,
    threshold_calibration_drive_dir: str | None = None,
    require_threshold_package: bool = True,
) -> dict[str, Any]:
    """从 Google Drive 前序包准备输入后运行真实攻击闭环, 失败时仍写出诊断产物."""
    root_path = Path(root).resolve()
    paper_run = build_paper_run_config(root_path)
    resolved_aligned_rescoring_drive_dir = aligned_rescoring_drive_dir or paper_run.drive_dir("aligned_rescoring")
    resolved_threshold_calibration_drive_dir = (
        threshold_calibration_drive_dir or paper_run.drive_dir("threshold_calibration")
    )
    config = build_default_config()
    output_dir = (root_path / config.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        materialize_drive_package_inputs(
            root=root_path,
            aligned_rescoring_drive_dir=resolved_aligned_rescoring_drive_dir,
            threshold_calibration_drive_dir=resolved_threshold_calibration_drive_dir,
            require_threshold_package=require_threshold_package,
        )
    except Exception as error:
        return write_failure_outputs(root_path, config, output_dir, error)
    return write_real_attack_evaluation_outputs(config=config, root=root_path)


def collect_package_entries(root_path: Path, output_dir: Path, archive_path: Path) -> tuple[Path, ...]:
    """收集需要进入压缩包的核对文件."""
    entries: list[Path] = []
    if output_dir.exists():
        for path in sorted(output_dir.rglob("*")):
            if path.is_file() and path.resolve() != archive_path.resolve() and path.suffix.lower() != ".zip":
                entries.append(path)
    for relative_path in PACKAGE_EXTRA_PATHS:
        path = root_path / relative_path
        if path.exists() and path not in entries:
            entries.append(path)
    unique_entries: list[Path] = []
    for entry in entries:
        if entry not in unique_entries:
            unique_entries.append(entry)
    return tuple(unique_entries)


def package_real_attack_evaluation_outputs(
    root: str | Path = ".",
    output_dir: str = DEFAULT_OUTPUT_DIR,
    drive_output_dir: str | None = None,
    archive_name: str = "real_attack_evaluation_package.zip",
) -> RealAttackArchiveRecord:
    """打包真实攻击闭环产物并镜像到 Google Drive。"""
    root_path = Path(root).resolve()
    resolved_drive_output_dir = drive_output_dir or build_paper_run_config(root_path).drive_dir("real_attack_evaluation")
    source_dir = (root_path / output_dir).resolve()
    source_dir.mkdir(parents=True, exist_ok=True)
    archive_path = source_dir / archive_name
    package_manifest_path = source_dir / "real_attack_package_input_manifest.json"
    summary_path = source_dir / "real_attack_archive_summary.json"
    manifest_path = source_dir / "real_attack_archive_manifest.local.json"
    for stale_path in (package_manifest_path, summary_path, manifest_path):
        if stale_path.exists():
            stale_path.unlink()
    entries = collect_package_entries(root_path, source_dir, archive_path)
    package_manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "entry_paths": [entry.relative_to(root_path).as_posix() for entry in entries],
        "entry_count": len(entries),
    }
    package_manifest_path.write_text(stable_json_text(package_manifest), encoding="utf-8")
    drive_dir = Path(resolved_drive_output_dir).expanduser()
    preliminary_record = RealAttackArchiveRecord(
        archive_path=archive_path.relative_to(root_path).as_posix(),
        archive_digest="",
        archive_entry_count=len(entries) + 3,
        drive_archive_path=str(drive_dir / archive_name),
        drive_archive_digest="",
        metadata={
            "construction_unit_name": "real_attack_evaluation",
            "drive_output_dir": str(drive_dir),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "embedded_digest_scope": "external_summary_records_final_archive_digest",
        },
    )
    summary_path.write_text(stable_json_text(preliminary_record.to_dict()), encoding="utf-8")
    manifest = build_artifact_manifest(
        artifact_id="real_attack_evaluation_archive_manifest",
        artifact_type="local_manifest",
        input_paths=tuple(
            [entry.relative_to(root_path).as_posix() for entry in entries]
            + [package_manifest_path.relative_to(root_path).as_posix()]
        ),
        output_paths=(
            archive_path.relative_to(root_path).as_posix(),
            summary_path.relative_to(root_path).as_posix(),
            manifest_path.relative_to(root_path).as_posix(),
        ),
        config={
            "archive_name": archive_name,
            "archive_entry_count": len(entries) + 3,
            "drive_output_dir": str(drive_dir),
        },
        code_version=resolve_code_version(root_path),
        rebuild_command="运行 paper_workflow/real_attack_evaluation_run.ipynb",
        metadata={
            "construction_unit_name": "real_attack_evaluation",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "embedded_digest_scope": "external_summary_records_final_archive_digest",
        },
    ).to_dict()
    manifest_path.write_text(stable_json_text(manifest), encoding="utf-8")
    entries = collect_package_entries(root_path, source_dir, archive_path)
    with ZipFile(archive_path, mode="w", compression=ZIP_DEFLATED) as archive:
        for entry in entries:
            archive.write(entry, entry.relative_to(root_path).as_posix())
    drive_dir.mkdir(parents=True, exist_ok=True)
    mirrored_path = drive_dir / archive_name
    shutil.copy2(archive_path, mirrored_path)
    record = RealAttackArchiveRecord(
        archive_path=archive_path.relative_to(root_path).as_posix(),
        archive_digest=file_digest(archive_path),
        archive_entry_count=len(entries),
        drive_archive_path=str(mirrored_path),
        drive_archive_digest=file_digest(mirrored_path),
        metadata={
            "construction_unit_name": "real_attack_evaluation",
            "drive_output_dir": str(drive_dir),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    summary_path.write_text(stable_json_text(record.to_dict()), encoding="utf-8")
    manifest.setdefault("metadata", {})["archive_digest"] = record.archive_digest
    manifest.setdefault("metadata", {})["drive_archive_digest"] = record.drive_archive_digest
    manifest_path.write_text(stable_json_text(manifest), encoding="utf-8")
    return record

