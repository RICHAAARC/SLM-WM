"""常规失真与几何变换真实图像级攻击闭环 helper.

该模块只执行不依赖 diffusion 模型的图像攻击, 例如 JPEG、噪声、模糊、缩放、裁剪和旋转。
Notebook 仍然只作为入口, 正式记录、注册表、metrics 和 manifest 均由该 helper 写出。
"""

from __future__ import annotations

import csv
import io
import json
import os
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

from experiments.protocol.attacks import AttackConfig, default_attack_configs
from experiments.protocol.paper_run_config import build_paper_run_config, resolve_count_from_environment
from main.analysis.artifact_manifest import build_artifact_manifest
from main.core.digest import build_stable_digest
from paper_workflow.colab_utils.progress import progress_bar, update_progress
from paper_workflow.colab_utils.real_attack_evaluation import (
    ALIGNED_PACKAGE_PREFIXES,
    ALIGNED_RESCORING_PACKAGE_PATTERN,
    DEFAULT_SOURCE_IMAGE_DIR,
    IMAGE_SUFFIXES,
    PRIMARY_MODEL_FAMILY,
    PRIMARY_MODEL_ID,
    THRESHOLD_CALIBRATION_PACKAGE_PATTERN,
    THRESHOLD_PACKAGE_PREFIXES,
    build_attack_record,
    build_family_metrics,
    build_formal_attack_record,
    context_for_source,
    file_digest,
    formal_boundary,
    jsonl_text,
    latest_drive_package,
    load_img2img_pipeline,
    load_rgb_image,
    normalize_attacked_image_size,
    read_csv_rows,
    read_jsonl,
    relative_or_absolute,
    safe_extract_selected_entries,
    discover_source_images as discover_governed_source_images,
    source_context_by_image_path,
    stable_json_text,
    unsupported_record,
    write_csv,
)
from paper_workflow.colab_utils.sd_runtime_cold_start import build_runtime_environment_report, resolve_code_version

DEFAULT_OUTPUT_DIR = "outputs/conventional_geometric_attack_evaluation"
DEFAULT_DRIVE_OUTPUT_DIR = ""
PACKAGE_EXTRA_PATHS = (
    "paper_workflow/conventional_geometric_attack_evaluation_run.ipynb",
    "paper_workflow/colab_utils/conventional_geometric_attack_evaluation.py",
    "paper_workflow/colab_utils/progress.py",
    "outputs/attack_matrix/attack_manifest.json",
    "outputs/attack_matrix/manifest.local.json",
    "outputs/threshold_calibration/manifest.local.json",
)
CONVENTIONAL_GEOMETRIC_FAMILIES = ("standard_distortion", "geometric_transform")


@dataclass(frozen=True)
class ConventionalGeometricAttackEvaluationConfig:
    """描述常规失真与几何变换攻击闭环配置."""

    seed: int
    prompt: str
    negative_prompt: str
    width: int
    height: int
    model_family: str = PRIMARY_MODEL_FAMILY
    model_id: str = PRIMARY_MODEL_ID
    output_dir: str = DEFAULT_OUTPUT_DIR
    source_image_dir: str = DEFAULT_SOURCE_IMAGE_DIR
    max_source_images: int = 600
    detection_threshold: float = 0.50
    device_name: str = "cuda"
    torch_dtype: str = "float16"
    hf_token_env: str = "HF_TOKEN"
    inference_steps: int = 20
    guidance_scale: float = 5.0
    enable_pipeline_progress_bar: bool = False
    enable_attack_progress_bar: bool = True
    enable_attacked_image_latent_rescore: bool = True
    require_attacked_image_latent_rescore: bool = True


@dataclass(frozen=True)
class ConventionalGeometricAttackSpec:
    """适配真实攻击记录构造函数所需的攻击配置视图.

    该结构属于通用工程写法: 将 AttackConfig 中的协议字段转换为
    build_attack_record 可复用的属性集合, 避免在业务循环里动态构造临时类型。
    """

    attack_id: str
    attack_family: str
    attack_name: str
    attack_strength: float
    attack_parameters: dict[str, Any]
    attack_implementation: str
    resource_profile: str
    requires_gpu: bool = False


@dataclass(frozen=True)
class ConventionalGeometricAttackEvaluationResult:
    """汇总常规失真与几何变换攻击闭环状态."""

    run_id: str
    run_decision: str
    unsupported_reason: str
    source_image_count: int
    image_attack_record_count: int
    real_attacked_image_count: int
    measured_attack_name_count: int
    required_attack_name_count: int
    real_attacked_image_closed_loop_ready: bool
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
        """转换为 JSON 兼容字典."""

        return asdict(self)


@dataclass(frozen=True)
class ConventionalGeometricAttackArchiveRecord:
    """记录常规失真与几何变换攻击压缩包与 Drive 镜像."""

    archive_path: str
    archive_digest: str
    archive_entry_count: int
    drive_archive_path: str
    drive_archive_digest: str
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """转换为 JSON 兼容字典."""

        return asdict(self)


def conventional_attack_configs() -> tuple[AttackConfig, ...]:
    """返回需要真实图像级执行的非 diffusion 攻击配置."""

    return tuple(config for config in default_attack_configs() if config.attack_family in CONVENTIONAL_GEOMETRIC_FAMILIES)


def conventional_attack_spec(config: AttackConfig) -> ConventionalGeometricAttackSpec:
    """把攻击矩阵配置转换为真实图像攻击记录构造所需的 spec."""

    return ConventionalGeometricAttackSpec(
        attack_id=config.attack_id,
        attack_family=config.attack_family,
        attack_name=config.attack_name,
        attack_strength=config.attack_strength,
        attack_parameters=dict(config.attack_parameters),
        attack_implementation=f"pil_{config.attack_name}",
        resource_profile=config.resource_profile,
        requires_gpu=config.requires_gpu,
    )


def materialize_drive_package_inputs(
    root: str | Path = ".",
    aligned_rescoring_drive_dir: str | None = None,
    threshold_calibration_drive_dir: str | None = None,
    require_threshold_package: bool = True,
) -> dict[str, Any]:
    """从 Google Drive 物化 aligned rescoring 与 threshold calibration 输入."""

    root_path = Path(root).resolve()
    paper_run = build_paper_run_config(root_path)
    resolved_aligned_dir = aligned_rescoring_drive_dir or paper_run.drive_dir("aligned_rescoring")
    resolved_threshold_dir = threshold_calibration_drive_dir or paper_run.drive_dir("threshold_calibration")
    output_dir = root_path / DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "conventional_geometric_attack_input_package_manifest.json"
    aligned_package = latest_drive_package(resolved_aligned_dir, ALIGNED_RESCORING_PACKAGE_PATTERN)
    extracted_aligned = safe_extract_selected_entries(aligned_package, root_path, ALIGNED_PACKAGE_PREFIXES)
    threshold_package = None
    extracted_threshold: tuple[str, ...] = ()
    try:
        threshold_package = latest_drive_package(resolved_threshold_dir, THRESHOLD_CALIBRATION_PACKAGE_PATTERN)
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


def discover_source_images(root_path: Path, config: ConventionalGeometricAttackEvaluationConfig) -> tuple[Path, ...]:
    """查找进入常规失真与几何变换攻击闭环的 source image."""

    if config.source_image_dir == DEFAULT_SOURCE_IMAGE_DIR:
        governed_paths = discover_governed_source_images(root_path, config)  # type: ignore[arg-type]
        if governed_paths:
            return governed_paths
    configured_dir = (root_path / config.source_image_dir).resolve()
    candidates: list[Path] = []
    for directory in (configured_dir, root_path / "outputs" / "aligned_rescoring"):
        if directory.exists():
            for path in sorted(directory.rglob("*")):
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                    candidates.append(path.resolve())
    unique: list[Path] = []
    for path in candidates:
        if path not in unique:
            unique.append(path)
    return tuple(unique[: config.max_source_images])


def _center_crop(image: Any, ratio: float) -> Any:
    """按中心裁剪比例裁剪图像."""

    width, height = image.size
    crop_width = max(1, int(width * ratio))
    crop_height = max(1, int(height * ratio))
    left = max(0, (width - crop_width) // 2)
    top = max(0, (height - crop_height) // 2)
    return image.crop((left, top, left + crop_width, top + crop_height))


def apply_conventional_geometric_attack(source_image: Any, config: AttackConfig, seed: int) -> Any:
    """执行单个 CPU 图像攻击并返回 attacked image."""

    import numpy as np
    from PIL import Image, ImageFilter

    resampling = getattr(getattr(Image, "Resampling", Image), "BICUBIC")
    image = source_image.convert("RGB")
    if config.attack_name == "jpeg_compression":
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=int(config.attack_parameters["quality"]))
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")
    if config.attack_name == "gaussian_noise":
        rng = np.random.default_rng(seed)
        array = np.asarray(image, dtype=np.float32) / 255.0
        noise = rng.normal(0.0, float(config.attack_parameters["sigma"]), size=array.shape).astype(np.float32)
        noised = np.clip(array + noise, 0.0, 1.0)
        return Image.fromarray((noised * 255.0).round().astype(np.uint8), mode="RGB")
    if config.attack_name == "gaussian_blur":
        return image.filter(ImageFilter.GaussianBlur(radius=float(config.attack_parameters["radius"])))
    if config.attack_name == "resize":
        width, height = image.size
        scale = float(config.attack_parameters["scale"])
        resized = image.resize((max(1, int(width * scale)), max(1, int(height * scale))), resampling)
        return resized.resize((width, height), resampling)
    if config.attack_name == "crop":
        return _center_crop(image, float(config.attack_parameters["crop_ratio"])).resize(image.size, resampling)
    if config.attack_name == "rotation":
        return image.rotate(float(config.attack_parameters["degrees"]), resample=resampling, expand=False)
    if config.attack_name == "crop_resize":
        cropped = _center_crop(image, float(config.attack_parameters["crop_ratio"]))
        return cropped.resize(image.size, resampling)
    if config.attack_name == "composite_geometric_attacks":
        cropped = _center_crop(image, float(config.attack_parameters["crop_ratio"]))
        rotated = cropped.rotate(float(config.attack_parameters["degrees"]), resample=resampling, expand=False)
        width, height = image.size
        scale = float(config.attack_parameters["resize_scale"])
        resized = rotated.resize((max(1, int(width * scale)), max(1, int(height * scale))), resampling)
        return resized.resize(image.size, resampling)
    raise ValueError(f"unsupported_conventional_geometric_attack:{config.attack_name}")


def write_failure_outputs(
    root_path: Path,
    config: ConventionalGeometricAttackEvaluationConfig,
    output_dir: Path,
    error: Exception,
) -> dict[str, Any]:
    """在输入缺失或执行失败时写出可打包诊断产物."""

    environment_report = build_runtime_environment_report()
    environment_path = output_dir / "conventional_geometric_attack_environment_report.json"
    result_path = output_dir / "conventional_geometric_attack_run_summary.json"
    manifest_path = output_dir / "conventional_geometric_attack_manifest.local.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    environment_path.write_text(stable_json_text(environment_report), encoding="utf-8")
    result = ConventionalGeometricAttackEvaluationResult(
        run_id=build_stable_digest({"error": type(error).__name__, "seed": config.seed}),
        run_decision="fail",
        unsupported_reason=f"{type(error).__name__}:{str(error)[:160]}",
        source_image_count=0,
        image_attack_record_count=0,
        real_attacked_image_count=0,
        measured_attack_name_count=0,
        required_attack_name_count=len(conventional_attack_configs()),
        real_attacked_image_closed_loop_ready=False,
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
        metadata={"runtime_environment": environment_report, "claim_boundary": "not_paper_ready"},
    )
    result_path.write_text(stable_json_text(result.to_dict()), encoding="utf-8")
    manifest = build_artifact_manifest(
        artifact_id="conventional_geometric_attack_evaluation_manifest",
        artifact_type="local_manifest",
        input_paths=(),
        output_paths=(relative_or_absolute(result_path, root_path), relative_or_absolute(environment_path, root_path)),
        config=asdict(config),
        code_version=resolve_code_version(root_path),
        rebuild_command="运行 paper_workflow/conventional_geometric_attack_evaluation_run.ipynb",
        metadata={"run_decision": "fail", "supports_paper_claim": False},
    ).to_dict()
    manifest_path.write_text(stable_json_text(manifest), encoding="utf-8")
    return result.to_dict()


def write_conventional_geometric_attack_evaluation_outputs(
    config: ConventionalGeometricAttackEvaluationConfig,
    root: str | Path = ".",
) -> dict[str, Any]:
    """执行常规失真与几何变换真实图像级攻击闭环."""

    root_path = Path(root).resolve()
    output_dir = (root_path / config.output_dir).resolve()
    attacked_dir = output_dir / "attacked_images"
    output_dir.mkdir(parents=True, exist_ok=True)
    attacked_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "conventional_geometric_attack_detection_records.jsonl"
    formal_records_path = output_dir / "formal_attack_detection_records.jsonl"
    registry_path = output_dir / "conventional_geometric_attacked_image_registry.jsonl"
    metrics_path = output_dir / "conventional_geometric_attack_family_metrics.csv"
    result_path = output_dir / "conventional_geometric_attack_run_summary.json"
    environment_path = output_dir / "conventional_geometric_attack_environment_report.json"
    manifest_path = output_dir / "conventional_geometric_attack_manifest.local.json"

    try:
        source_paths = discover_source_images(root_path, config)
        if not source_paths:
            raise FileNotFoundError("source_image_files_missing")
        source_contexts = source_context_by_image_path(root_path, config)  # type: ignore[arg-type]
        boundary = formal_boundary(root_path, config)  # type: ignore[arg-type]
        detector_pipeline, detector_runtime_versions = load_img2img_pipeline(config)  # type: ignore[arg-type]
    except Exception as error:
        return write_failure_outputs(root_path, config, output_dir, error)

    attack_configs = conventional_attack_configs()
    records: list[dict[str, Any]] = []
    registry_rows: list[dict[str, Any]] = []
    contexts_by_record_source_path: dict[str, dict[str, Any]] = {}
    total_tasks = len(source_paths) * len(attack_configs)
    with progress_bar(total_tasks, desc="conventional/geometric image attacks", enabled=config.enable_attack_progress_bar) as task_progress:
        for source_index, source_path in enumerate(source_paths):
            source_digest = file_digest(source_path)
            source_image = load_rgb_image(source_path, config)  # type: ignore[arg-type]
            source_context = context_for_source(source_path, root_path, source_contexts, config)  # type: ignore[arg-type]
            contexts_by_record_source_path[relative_or_absolute(source_path, root_path)] = source_context
            for attack_index, attack_config in enumerate(attack_configs):
                seed = config.seed + source_index * 101 + attack_index
                spec = conventional_attack_spec(attack_config)
                try:
                    attacked_image = apply_conventional_geometric_attack(source_image, attack_config, seed)
                    attacked_image = normalize_attacked_image_size(attacked_image, source_image)
                    attacked_path = attacked_dir / f"{source_path.stem}_{attack_config.attack_name}_{source_digest[:8]}.png"
                    attacked_image.save(attacked_path)
                    record, registry_row = build_attack_record(
                        root_path=root_path,
                        source_path=source_path,
                        source_image=source_image,
                        attacked_image=attacked_image,
                        attacked_path=attacked_path,
                        spec=spec,  # type: ignore[arg-type]
                        config=config,  # type: ignore[arg-type]
                        source_context=source_context,
                        boundary=boundary,
                        detector_pipeline=detector_pipeline,
                    )
                    records.append(record.to_dict())
                    registry_rows.append(registry_row)
                except Exception as error:
                    records.append(unsupported_record(root_path, source_path, source_digest, spec, config, error).to_dict())  # type: ignore[arg-type]
                finally:
                    update_progress(
                        task_progress,
                        profile=(
                            f"attack={attack_config.attack_name} "
                            f"source={source_index + 1}/{len(source_paths)} "
                            f"seed={seed} resource_profile={attack_config.resource_profile}"
                        ),
                    )

    try:
        del detector_pipeline
    except Exception:
        pass

    record_rows = tuple(records)
    registry_tuple = tuple(registry_rows)
    formal_rows = tuple(
        build_formal_attack_record(record, contexts_by_record_source_path[record["source_image_path"]], boundary)
        for record in record_rows
        if str(record.get("metric_status", "")).startswith("measured_from_real_attacked_image")
    )
    family_metrics = build_family_metrics(formal_rows)
    records_path.write_text(jsonl_text(record_rows), encoding="utf-8")
    formal_records_path.write_text(jsonl_text(formal_rows), encoding="utf-8")
    registry_path.write_text(jsonl_text(registry_tuple), encoding="utf-8")
    write_csv(metrics_path, family_metrics)
    environment_report = detector_runtime_versions.get("runtime_environment", build_runtime_environment_report())
    environment_path.write_text(stable_json_text(environment_report), encoding="utf-8")

    measured_names = {
        record["attack_name"]
        for record in record_rows
        if str(record.get("metric_status", "")).startswith("measured_from_real_attacked_image")
    }
    real_attacked_image_count = sum(1 for record in record_rows if record["attacked_image_available"])
    closed_loop_ready = real_attacked_image_count > 0 and all(
        row.get("source_image_digest") and row.get("attacked_image_digest") for row in registry_tuple
    )
    detection_ready = any(str(record.get("metric_status", "")).startswith("measured_from_real_attacked_image") for record in record_rows)
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
        if str(record.get("metric_status", "")).startswith("measured_from_real_attacked_image")
    )
    required_names = {config.attack_name for config in attack_configs}
    run_decision = "pass" if closed_loop_ready and detection_ready and formal_ready and required_names.issubset(measured_names) else "fail"
    result = ConventionalGeometricAttackEvaluationResult(
        run_id=build_stable_digest({"records": record_rows, "config": asdict(config)}),
        run_decision=run_decision,
        unsupported_reason="" if run_decision == "pass" else "conventional_geometric_attack_closed_loop_incomplete",
        source_image_count=len(source_paths),
        image_attack_record_count=len(record_rows),
        real_attacked_image_count=real_attacked_image_count,
        measured_attack_name_count=len(measured_names.intersection(required_names)),
        required_attack_name_count=len(required_names),
        real_attacked_image_closed_loop_ready=closed_loop_ready,
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
            "runtime_environment": environment_report,
            "required_attack_names": sorted(required_names),
            "measured_attack_names": sorted(measured_names),
            "formal_boundary": boundary,
            "attacked_image_rescore_count": attacked_image_rescore_count,
            "attacked_image_rescore_ready": attacked_image_rescore_ready,
            "proxy_formal_record_count": proxy_formal_record_count,
            "claim_boundary": "requires_attack_matrix_rebuild_and_evidence_audit",
        },
    )
    result_path.write_text(stable_json_text(result.to_dict()), encoding="utf-8")
    manifest = build_artifact_manifest(
        artifact_id="conventional_geometric_attack_evaluation_manifest",
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
        rebuild_command="运行 paper_workflow/conventional_geometric_attack_evaluation_run.ipynb",
        metadata={
            "run_decision": run_decision,
            "real_attacked_image_count": real_attacked_image_count,
            "formal_attack_detection_ready": formal_ready,
            "attacked_image_rescore_count": attacked_image_rescore_count,
            "attacked_image_rescore_ready": attacked_image_rescore_ready,
            "proxy_formal_record_count": proxy_formal_record_count,
            "supports_paper_claim": False,
        },
    ).to_dict()
    manifest_path.write_text(stable_json_text(manifest), encoding="utf-8")
    return result.to_dict()


def build_default_config() -> ConventionalGeometricAttackEvaluationConfig:
    """从环境变量构造默认常规失真与几何变换攻击配置."""

    return ConventionalGeometricAttackEvaluationConfig(
        seed=int(os.environ.get("SLM_WM_CONVENTIONAL_GEOMETRIC_ATTACK_SEED", "20260624")),
        prompt=os.environ.get("SLM_WM_REAL_ATTACK_PROMPT", "a calm studio portrait of a ceramic bird with soft geometric background"),
        negative_prompt=os.environ.get("SLM_WM_NEGATIVE_PROMPT", "low quality, blurry"),
        width=int(os.environ.get("SLM_WM_IMAGE_WIDTH", "512")),
        height=int(os.environ.get("SLM_WM_IMAGE_HEIGHT", "512")),
        model_family=os.environ.get("SLM_WM_MODEL_FAMILY", PRIMARY_MODEL_FAMILY),
        model_id=os.environ.get("SLM_WM_MODEL_ID", PRIMARY_MODEL_ID),
        output_dir=os.environ.get("SLM_WM_CONVENTIONAL_GEOMETRIC_ATTACK_OUTPUT_DIR", DEFAULT_OUTPUT_DIR),
        source_image_dir=os.environ.get("SLM_WM_REAL_ATTACK_SOURCE_IMAGE_DIR", DEFAULT_SOURCE_IMAGE_DIR),
        max_source_images=resolve_count_from_environment("SLM_WM_CONVENTIONAL_GEOMETRIC_ATTACK_SOURCE_COUNT"),
        detection_threshold=float(os.environ.get("SLM_WM_REAL_ATTACK_DETECTION_THRESHOLD", "0.50")),
        device_name=os.environ.get("SLM_WM_DEVICE", "cuda"),
        torch_dtype=os.environ.get("SLM_WM_TORCH_DTYPE", "float16"),
        inference_steps=int(os.environ.get("SLM_WM_INFERENCE_STEPS", "20")),
        guidance_scale=float(os.environ.get("SLM_WM_GUIDANCE_SCALE", "5.0")),
        enable_pipeline_progress_bar=os.environ.get("SLM_WM_ENABLE_PIPELINE_PROGRESS_BAR", "0") == "1",
        enable_attack_progress_bar=os.environ.get("SLM_WM_ENABLE_ATTACK_PROGRESS_BAR", "1") != "0",
        enable_attacked_image_latent_rescore=os.environ.get("SLM_WM_ENABLE_ATTACKED_IMAGE_LATENT_RESCORE", "1") != "0",
        require_attacked_image_latent_rescore=os.environ.get("SLM_WM_REQUIRE_ATTACKED_IMAGE_LATENT_RESCORE", "1") != "0",
    )


def run_default_conventional_geometric_attack_evaluation_plan(root: str | Path = ".") -> dict[str, Any]:
    """运行默认常规失真与几何变换攻击计划."""

    return write_conventional_geometric_attack_evaluation_outputs(build_default_config(), root=root)


def run_default_conventional_geometric_attack_evaluation_from_drive_plan(
    root: str | Path = ".",
    aligned_rescoring_drive_dir: str | None = None,
    threshold_calibration_drive_dir: str | None = None,
    require_threshold_package: bool = True,
) -> dict[str, Any]:
    """先从 Drive 物化输入包, 再运行常规失真与几何变换攻击闭环."""

    root_path = Path(root).resolve()
    config = build_default_config()
    output_dir = (root_path / config.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        materialize_drive_package_inputs(
            root=root_path,
            aligned_rescoring_drive_dir=aligned_rescoring_drive_dir,
            threshold_calibration_drive_dir=threshold_calibration_drive_dir,
            require_threshold_package=require_threshold_package,
        )
    except Exception as error:
        return write_failure_outputs(root_path, config, output_dir, error)
    return write_conventional_geometric_attack_evaluation_outputs(config, root=root_path)


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


def package_conventional_geometric_attack_evaluation_outputs(
    root: str | Path = ".",
    output_dir: str = DEFAULT_OUTPUT_DIR,
    drive_output_dir: str | None = None,
    archive_name: str = "conventional_geometric_attack_evaluation_package.zip",
) -> ConventionalGeometricAttackArchiveRecord:
    """打包常规失真与几何变换攻击产物并镜像到 Google Drive."""

    root_path = Path(root).resolve()
    resolved_drive_output_dir = drive_output_dir or build_paper_run_config(root_path).drive_dir(
        "conventional_geometric_attack_evaluation"
    )
    source_dir = (root_path / output_dir).resolve()
    source_dir.mkdir(parents=True, exist_ok=True)
    archive_path = source_dir / archive_name
    package_manifest_path = source_dir / "conventional_geometric_attack_package_input_manifest.json"
    summary_path = source_dir / "conventional_geometric_attack_archive_summary.json"
    manifest_path = source_dir / "conventional_geometric_attack_archive_manifest.local.json"
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
    preliminary_record = ConventionalGeometricAttackArchiveRecord(
        archive_path=relative_or_absolute(archive_path, root_path),
        archive_digest="",
        archive_entry_count=len(entries) + 3,
        drive_archive_path=str(Path(resolved_drive_output_dir).expanduser() / archive_name),
        drive_archive_digest="",
        metadata={
            "construction_unit_name": "conventional_geometric_attack_evaluation",
            "drive_output_dir": str(Path(resolved_drive_output_dir).expanduser()),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "embedded_digest_scope": "conventional_geometric_attack_final_archive_digest",
        },
    )
    summary_path.write_text(stable_json_text(preliminary_record.to_dict()), encoding="utf-8")
    archive_manifest = build_artifact_manifest(
        artifact_id="conventional_geometric_attack_archive_manifest",
        artifact_type="local_manifest",
        input_paths=tuple([entry.relative_to(root_path).as_posix() for entry in entries] + [package_manifest_path.relative_to(root_path).as_posix()]),
        output_paths=(
            archive_path.relative_to(root_path).as_posix(),
            summary_path.relative_to(root_path).as_posix(),
            manifest_path.relative_to(root_path).as_posix(),
        ),
        config={"archive_name": archive_name, "drive_output_dir": str(Path(resolved_drive_output_dir).expanduser())},
        code_version=resolve_code_version(root_path),
        rebuild_command="运行 paper_workflow/conventional_geometric_attack_evaluation_run.ipynb",
        metadata={"archive_digest_embedded": False, "supports_paper_claim": False},
    ).to_dict()
    manifest_path.write_text(stable_json_text(archive_manifest), encoding="utf-8")
    with ZipFile(archive_path, "w", ZIP_DEFLATED) as archive:
        for entry in (*entries, package_manifest_path, summary_path, manifest_path):
            archive.write(entry, arcname=entry.relative_to(root_path).as_posix())
    archive_digest = file_digest(archive_path)
    final_record = ConventionalGeometricAttackArchiveRecord(
        archive_path=relative_or_absolute(archive_path, root_path),
        archive_digest=archive_digest,
        archive_entry_count=len(entries) + 3,
        drive_archive_path=str(Path(resolved_drive_output_dir).expanduser() / archive_name),
        drive_archive_digest="",
        metadata={**preliminary_record.metadata, "archive_digest_embedded": True},
    )
    summary_path.write_text(stable_json_text(final_record.to_dict()), encoding="utf-8")
    archive_manifest["metadata"]["archive_digest_embedded"] = True
    archive_manifest["metadata"]["archive_digest"] = archive_digest
    manifest_path.write_text(stable_json_text(archive_manifest), encoding="utf-8")
    with ZipFile(archive_path, "w", ZIP_DEFLATED) as archive:
        for entry in (*entries, package_manifest_path, summary_path, manifest_path):
            archive.write(entry, arcname=entry.relative_to(root_path).as_posix())
    archive_digest = file_digest(archive_path)
    drive_dir = Path(resolved_drive_output_dir).expanduser()
    drive_dir.mkdir(parents=True, exist_ok=True)
    drive_archive_path = drive_dir / archive_name
    shutil.copy2(archive_path, drive_archive_path)
    drive_digest = file_digest(drive_archive_path)
    completed_record = ConventionalGeometricAttackArchiveRecord(
        archive_path=relative_or_absolute(archive_path, root_path),
        archive_digest=archive_digest,
        archive_entry_count=len(entries) + 3,
        drive_archive_path=str(drive_archive_path),
        drive_archive_digest=drive_digest,
        metadata={**final_record.metadata, "drive_digest_verified": drive_digest == archive_digest},
    )
    summary_path.write_text(stable_json_text(completed_record.to_dict()), encoding="utf-8")
    return completed_record
