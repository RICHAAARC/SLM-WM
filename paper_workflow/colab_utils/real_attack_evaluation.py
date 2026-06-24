"""真实再扩散攻击闭环的 Colab 辅助函数.

该模块的作用是把真实 SD3.5 Medium 图像攻击、攻击后检测、文件摘要登记和 Google Drive 打包放在
repository helper 中, Notebook 只负责调用入口。此处不在本地伪造 GPU 结果; 当真实后端不可用时,
函数会写出可审计的失败摘要, 方便后续在 Colab GPU 中复跑。
"""

from __future__ import annotations

import csv
import gc
import json
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
from paper_workflow.colab_utils.minimal_latent_injection import compute_image_quality_metrics
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
    measured_regeneration_attack_count: int
    real_attacked_image_closed_loop_ready: bool
    regeneration_attack_gpu_validation_ready: bool
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


def discover_source_images(root_path: Path, config: RealAttackEvaluationConfig) -> tuple[Path, ...]:
    """查找需要进入真实攻击闭环的 source image 文件."""
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


def load_img2img_pipeline(config: RealAttackEvaluationConfig) -> tuple[Any, dict[str, Any]]:
    """加载真实 SD3/SD3.5 image-to-image pipeline."""
    import torch
    import diffusers

    if config.device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("gpu_unavailable")
    dtype = getattr(torch, config.torch_dtype)
    pipeline_class = getattr(diffusers, "StableDiffusion3Img2ImgPipeline", None)
    if pipeline_class is None:
        pipeline_class = getattr(diffusers, "AutoPipelineForImage2Image")
    token = os.environ.get(config.hf_token_env) or None
    pipeline = pipeline_class.from_pretrained(config.model_id, torch_dtype=dtype, token=token)
    pipeline = pipeline.to(config.device_name)
    pipeline.set_progress_bar_config(disable=False)
    environment_report = build_runtime_environment_report(torch_module=torch)
    runtime_versions = {
        **flatten_environment_versions(environment_report),
        "runtime_environment": environment_report,
    }
    return pipeline, runtime_versions


def load_rgb_image(path: Path, config: RealAttackEvaluationConfig) -> Any:
    """读取 source image 并调整为 pipeline 输入尺寸."""
    from PIL import Image

    image = Image.open(path).convert("RGB")
    return image.resize((config.width, config.height))


def normalize_attacked_image_size(attacked_image: Any, source_image: Any) -> Any:
    """把 attacked image 对齐到 source image 尺寸, 保证后续质量指标可直接逐像素比较."""
    if getattr(attacked_image, "size", None) == getattr(source_image, "size", None):
        return attacked_image.convert("RGB") if hasattr(attacked_image, "convert") else attacked_image
    from PIL import Image

    resampling = getattr(getattr(Image, "Resampling", Image), "LANCZOS")
    return attacked_image.convert("RGB").resize(source_image.size, resampling)


def add_sdedit_noise(image: Any, noise_level: float, seed: int) -> Any:
    """为 SDEdit 风格攻击构造带噪输入图像."""
    import numpy as np
    from PIL import Image

    rng = np.random.default_rng(seed)
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    noise = rng.normal(loc=0.0, scale=noise_level, size=array.shape).astype(np.float32)
    mixed = np.clip(array * (1.0 - noise_level) + noise_level * np.clip(array + noise, 0.0, 1.0), 0.0, 1.0)
    return Image.fromarray((mixed * 255.0).round().astype(np.uint8), mode="RGB")


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
    if spec.attack_name == "sdedit_regeneration":
        input_image = add_sdedit_noise(source_image, float(spec.attack_parameters["noise_level"]), seed)
    generator = torch.Generator(device=config.device_name).manual_seed(seed)
    output = pipeline(
        prompt=prompt_text,
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


def run_strict_ddim_inversion_attack(
    source_image: Any,
    spec: RealAttackSpec,
    config: RealAttackEvaluationConfig,
    seed: int,
    prompt_text: str,
) -> Any:
    """使用 DDIMInverseScheduler 执行真正的 inversion 再生成攻击."""
    import torch
    import diffusers

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
    pipe.scheduler = scheduler_class.from_config(pipe.scheduler.config)
    inverse_scheduler = inverse_scheduler_class.from_config(pipe.scheduler.config)
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
) -> tuple[RealAttackDetectionRecord, dict[str, Any]]:
    """由真实 attacked image 构造检测记录和注册表行."""
    source_digest = file_digest(source_path)
    attacked_digest = file_digest(attacked_path)
    metrics, raw_score, aligned_score, decision = quality_detection_scores(
        source_image, attacked_image, config.detection_threshold
    )
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
        detection_method="real_image_quality_proxy_after_attack",
        detection_threshold=config.detection_threshold,
        raw_content_score_after=raw_score,
        aligned_content_score_after=aligned_score,
        evidence_decision=decision,
        metric_status="measured_from_real_attacked_image",
        unsupported_reason="",
        supports_paper_claim=False,
        metadata={
            "image_quality_metrics": metrics,
            "claim_boundary": "requires_attack_matrix_and_fixed_fpr_rebuild",
            "attacked_image_closed_loop": True,
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


def source_context_by_image_path(root_path: Path, config: RealAttackEvaluationConfig) -> dict[str, dict[str, Any]]:
    """把 aligned image 路径映射回真实 aligned rescoring 记录与 prompt."""
    quality_rows = read_csv_rows(root_path / "outputs" / "aligned_rescoring" / "aligned_rescoring_quality_metrics.csv")
    rescoring_rows = read_jsonl(root_path / "outputs" / "aligned_rescoring" / "aligned_rescoring_records.jsonl")
    prompts = prompt_lookup(root_path)
    records_by_prompt = {str(row.get("prompt_id", "")): row for row in rescoring_rows}
    contexts: dict[str, dict[str, Any]] = {}
    for quality_row in quality_rows:
        image_path = str(quality_row.get("aligned_image_path", ""))
        prompt_id = str(quality_row.get("prompt_id", ""))
        source_record = records_by_prompt.get(prompt_id, {})
        contexts[image_path] = {
            "prompt_id": prompt_id,
            "prompt_text": prompts.get(prompt_id, config.prompt),
            "source_record": source_record,
            "split": source_record.get("split", "unknown"),
            "sample_role": source_record.get("sample_role", "unknown"),
            "raw_content_score_before": float(source_record.get("real_raw_content_score", source_record.get("raw_content_score", 0.0))),
            "aligned_content_score_before": float(
                source_record.get("real_aligned_content_score", source_record.get("aligned_content_score", 0.0))
            ),
            "geometry_reliable": bool(source_record.get("aligned_rescoring_ready", False)),
            "fail_reason": source_record.get("fail_reason", "geometry_suspected"),
        }
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
            "geometry_reliable": False,
            "fail_reason": "geometry_suspected",
        },
    )


def formal_boundary(root_path: Path, config: RealAttackEvaluationConfig) -> dict[str, Any]:
    """读取 fixed-FPR 和 rescue 边界, 缺失时保留不可支持状态."""
    thresholds = read_json(root_path / "outputs" / "threshold_calibration" / "calibration_thresholds.json")
    report = read_json(root_path / "outputs" / "threshold_calibration" / "threshold_degeneracy_report.json")
    threshold_value = float(thresholds.get("threshold_value", report.get("calibrated_content_threshold", config.detection_threshold)))
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
        "boundary_ready": bool(thresholds and report),
    }


def formal_attack_record_id(real_record: dict[str, Any], boundary: dict[str, Any]) -> str:
    """构造 attack matrix 兼容记录 id."""
    payload = {
        "real_attack_record_id": real_record["real_attack_record_id"],
        "source_image_digest": real_record["source_image_digest"],
        "attacked_image_digest": real_record["attacked_image_digest"],
        "content_threshold": boundary["content_threshold"],
    }
    return f"attack_record_{build_stable_digest(payload)[:16]}"


def build_formal_attack_record(real_record: dict[str, Any], source_context: dict[str, Any], boundary: dict[str, Any]) -> dict[str, Any]:
    """把真实 attacked image 结果接回 attack matrix 正式记录 schema."""
    raw_before = float(source_context["raw_content_score_before"])
    aligned_before = float(source_context["aligned_content_score_before"])
    retention = float(real_record["aligned_content_score_after"])
    raw_after = max(0.0, min(1.0, raw_before * retention))
    aligned_after = max(0.0, min(1.0, aligned_before * retention))
    threshold = float(boundary["content_threshold"])
    margin_after = raw_after - threshold
    aligned_margin_after = aligned_after - threshold
    positive_by_content = margin_after >= 0.0
    geometry_reliable = bool(source_context["geometry_reliable"])
    rescue_eligible = (
        boundary["rescue_margin_low"] <= margin_after < 0.0
        and geometry_reliable
        and source_context["fail_reason"] in boundary["allowed_fail_reasons"]
    )
    rescue_applied = rescue_eligible and aligned_margin_after >= 0.0
    evidence_decision = positive_by_content or rescue_applied
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
        "resource_profile": "full_extra",
        "requires_gpu": True,
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
        "lf_score_retention": retention,
        "hf_score_retention": retention,
        "score_retention": retention,
        "quality_score_proxy": real_record["metadata"].get("image_quality_metrics", {}).get("ssim", retention),
        "attention_consistency_proxy": retention,
        "geometry_reliable": geometry_reliable,
        "rescue_eligible": rescue_eligible,
        "rescue_applied": rescue_applied,
        "evidence_decision": evidence_decision,
        "metric_status": "measured_from_real_attacked_image_formal_protocol" if boundary["boundary_ready"] else "formal_boundary_missing",
        "unsupported_reason": "" if boundary["boundary_ready"] else "threshold_calibration_inputs_missing",
        "supports_paper_claim": False,
        "metadata": {
            "real_attack_record_id": real_record["real_attack_record_id"],
            "attacked_image_path": real_record["attacked_image_path"],
            "source_image_path": real_record["source_image_path"],
            "detection_method": "fixed_fpr_attack_matrix_schema_from_real_attacked_image",
            "attack_implementation": real_record["attack_implementation"],
            "formal_boundary_ready": boundary["boundary_ready"],
            "claim_boundary": "requires_full_sample_scale_and_evidence_audit",
        },
    }


def build_family_metrics(records: tuple[dict[str, Any], ...]) -> tuple[dict[str, Any], ...]:
    """按攻击名称聚合真实攻击闭环检测指标."""
    rows: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault((record["attack_family"], record["attack_name"]), []).append(record)
    for (attack_family, attack_name), group in sorted(grouped.items()):
        measured = [record for record in group if record["metric_status"] == "measured_from_real_attacked_image"]
        decisions = [bool(record["evidence_decision"]) for record in measured]
        scores = [float(record["aligned_content_score_after"]) for record in measured]
        rows.append(
            {
                "attack_family": attack_family,
                "attack_name": attack_name,
                "attack_record_count": len(group),
                "measured_record_count": len(measured),
                "unsupported_record_count": len(group) - len(measured),
                "real_attacked_image_count": sum(1 for record in measured if record["attacked_image_available"]),
                "detection_positive_rate": sum(1 for decision in decisions if decision) / len(decisions) if decisions else 0.0,
                "aligned_content_score_after_mean": sum(scores) / len(scores) if scores else 0.0,
                "metric_status": "measured_from_real_attacked_image" if measured else "unsupported",
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
        required_regeneration_attack_count=len(REQUIRED_REGENERATION_ATTACKS),
        measured_regeneration_attack_count=0,
        real_attacked_image_closed_loop_ready=False,
        regeneration_attack_gpu_validation_ready=False,
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
    specs = default_attack_specs()
    main_specs = tuple(spec for spec in specs if spec.attack_name != "ddim_inversion_regeneration")
    ddim_specs = tuple(spec for spec in specs if spec.attack_name == "ddim_inversion_regeneration")
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
                )
                records.append(record.to_dict())
                registry_rows.append(registry_row)
            except Exception as error:
                records.append(unsupported_record(root_path, source_path, source_digest, spec, config, error).to_dict())

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
                record, registry_row = build_attack_record(
                    root_path=root_path,
                    source_path=source_path,
                    source_image=source_image,
                    attacked_image=attacked_image,
                    attacked_path=attacked_path,
                    spec=spec,
                    config=config,
                )
                records.append(record.to_dict())
                registry_rows.append(registry_row)
            except Exception as error:
                records.append(unsupported_record(root_path, source_path, source_digest, spec, config, error).to_dict())

    record_rows = tuple(records)
    registry_tuple = tuple(registry_rows)
    formal_rows = tuple(
        build_formal_attack_record(record, contexts_by_record_source_path[record["source_image_path"]], boundary)
        for record in record_rows
        if record["metric_status"] == "measured_from_real_attacked_image"
    )
    family_metrics = build_family_metrics(record_rows)
    records_path.write_text(jsonl_text(record_rows), encoding="utf-8")
    formal_records_path.write_text(jsonl_text(formal_rows), encoding="utf-8")
    registry_path.write_text(jsonl_text(registry_tuple), encoding="utf-8")
    write_csv(metrics_path, family_metrics)
    environment_report = runtime_versions.get("runtime_environment", build_runtime_environment_report())
    environment_path.write_text(stable_json_text(environment_report), encoding="utf-8")

    measured_names = {record["attack_name"] for record in record_rows if record["metric_status"] == "measured_from_real_attacked_image"}
    real_attacked_image_count = sum(1 for record in record_rows if record["attacked_image_available"])
    closed_loop_ready = real_attacked_image_count > 0 and all(
        row.get("source_image_digest") and row.get("attacked_image_digest") for row in registry_tuple
    )
    regeneration_ready = all(name in measured_names for name in REQUIRED_REGENERATION_ATTACKS)
    detection_ready = any(record["metric_status"] == "measured_from_real_attacked_image" for record in record_rows)
    formal_ready = bool(formal_rows) and boundary["boundary_ready"] and len(formal_rows) == real_attacked_image_count
    image_quality_ready = all(
        "image_quality_metrics" in record.get("metadata", {})
        for record in record_rows
        if record["metric_status"] == "measured_from_real_attacked_image"
    )
    run_decision = (
        "pass"
        if closed_loop_ready
        and detection_ready
        and formal_ready
        and (regeneration_ready or not config.require_all_regeneration_attacks)
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
        required_regeneration_attack_count=len(REQUIRED_REGENERATION_ATTACKS),
        measured_regeneration_attack_count=len(measured_names.intersection(REQUIRED_REGENERATION_ATTACKS)),
        real_attacked_image_closed_loop_ready=closed_loop_ready,
        regeneration_attack_gpu_validation_ready=regeneration_ready,
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
            "formal_boundary": boundary,
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
            "regeneration_attack_gpu_validation_ready": regeneration_ready,
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
