"""Colab 最小 diffusion latent injection helper."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

from main.analysis.artifact_manifest import build_artifact_manifest
from main.core.digest import build_stable_digest
from main.methods.algorithm_primitives import (
    build_semantic_risk_field,
    compose_latent_update,
    derive_attention_carrier_stub,
    derive_tail_carrier,
    derive_lf_carrier,
    estimate_safe_basis,
    project_latent_mask,
)
from experiments.runtime.repository_environment import (
    build_runtime_environment_report,
    file_digest,
    flatten_environment_versions,
    resolve_code_version,
    tensor_digest,
)


DEFAULT_OUTPUT_DIR = "outputs/minimal_diffusion_latent_injection"
DEFAULT_DRIVE_OUTPUT_DIR = "/content/drive/MyDrive/SLM/minimal_diffusion_latent_injection"
PRIMARY_MODEL_FAMILY = "sd35"
FALLBACK_MODEL_FAMILY = "sd3"


@dataclass(frozen=True)
class InjectionRunConfig:
    """描述一次最小 latent injection 运行所需的配置."""

    model_family: str
    model_id: str
    model_priority: str
    prompt: str
    negative_prompt: str
    seed: int
    width: int
    height: int
    inference_steps: int
    guidance_scale: float
    injection_strength: float
    injection_step_indices: tuple[int, ...]
    watermark_key_digest: str
    output_dir: str = DEFAULT_OUTPUT_DIR
    device_name: str = "cuda"
    torch_dtype: str = "float16"
    hf_token_env: str = "HF_TOKEN"

    def __post_init__(self) -> None:
        """集中校验配置边界, 避免运行路径重复构造错误信息."""
        invalid_positive_fields = {
            name: value
            for name, value in {
                "width": self.width,
                "height": self.height,
                "inference_steps": self.inference_steps,
            }.items()
            if value <= 0
        }
        if invalid_positive_fields:
            raise ValueError(f"配置正整数边界无效: {invalid_positive_fields}")
        if self.guidance_scale <= 0.0 or self.injection_strength < 0.0:
            raise ValueError("guidance_scale 必须为正数, injection_strength 不得为负数")
        if any(step_index < 0 or step_index >= self.inference_steps for step_index in self.injection_step_indices):
            raise ValueError("injection_step_indices 必须位于采样步数边界内")


@dataclass(frozen=True)
class LatentUpdateRecord:
    """记录一次 latent injection 对采样 latent 的最小扰动摘要."""

    injection_id: str
    model_family: str
    model_id: str
    model_priority: str
    update_index: int
    trajectory_index: int
    timestep: float
    injection_strength: float
    carrier_digest: str
    latent_digest_before: str
    latent_digest_after: str
    update_norm: float
    latent_norm_before: float
    latent_norm_after: float
    relative_update_norm: float
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """转为 JSON 兼容字典."""
        return asdict(self)


@dataclass(frozen=True)
class InjectionRunResult:
    """保存一次 clean / watermarked paired image 运行摘要."""

    injection_id: str
    model_family: str
    model_id: str
    model_priority: str
    run_decision: str
    unsupported_reason: str
    clean_image_path: str
    watermarked_image_path: str
    clean_image_digest: str
    watermarked_image_digest: str
    latent_update_count: int
    psnr: float | str
    ssim: float
    mse: float
    mean_abs_error: float
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """转为 JSON 兼容字典."""
        return asdict(self)


@dataclass(frozen=True)
class InjectionArchiveRecord:
    """记录最小 latent injection 产物压缩包与 Drive 镜像."""

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
    """把 JSON 兼容对象转为稳定、可读文本."""
    return json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n"


def json_line(value: Any) -> str:
    """把 JSON 兼容对象转为 JSONL 单行文本."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n"


def build_injection_id(config: InjectionRunConfig) -> str:
    """根据运行配置生成稳定 injection id."""
    return build_stable_digest(
        {
            "model_family": config.model_family,
            "model_id": config.model_id,
            "model_priority": config.model_priority,
            "prompt": config.prompt,
            "seed": config.seed,
            "width": config.width,
            "height": config.height,
            "inference_steps": config.inference_steps,
            "guidance_scale": config.guidance_scale,
            "injection_strength": config.injection_strength,
            "injection_step_indices": config.injection_step_indices,
            "watermark_key_digest": config.watermark_key_digest,
        }
    )


def build_prompt_digest(config: InjectionRunConfig) -> str:
    """生成 prompt 与模型配置摘要."""
    return build_stable_digest(
        {
            "prompt": config.prompt,
            "negative_prompt": config.negative_prompt,
            "model_id": config.model_id,
            "seed": config.seed,
        }
    )


def import_runtime_dependencies() -> tuple[Any, Any, Any, Any]:
    """延迟导入真实模型和图像依赖."""
    import torch
    import diffusers
    from diffusers import StableDiffusion3Pipeline

    return None, torch, diffusers, StableDiffusion3Pipeline


def load_pipeline(config: InjectionRunConfig) -> tuple[Any, dict[str, Any]]:
    """加载真实 SD pipeline 并移动到目标设备."""
    _, torch, diffusers, pipeline_class = import_runtime_dependencies()
    if config.device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("gpu_unavailable")
    dtype = getattr(torch, config.torch_dtype)
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


def tensor_norm(tensor: Any) -> float:
    """计算 tensor 的二范数."""
    return float(tensor.detach().float().norm().item())


def stable_unit_sequence(parts: tuple[str, ...], count: int) -> tuple[float, ...]:
    """根据稳定字符串材料生成 [-1, 1] 区间内的确定性数值序列."""
    if count <= 0:
        raise ValueError("count 必须为正数")
    values: list[float] = []
    for index in range(count):
        payload = "|".join((*parts, str(index))).encode("utf-8")
        digest = hashlib.sha256(payload).hexdigest()
        unit = int(digest[:16], 16) / float(0xFFFFFFFFFFFFFFFF)
        values.append(unit * 2.0 - 1.0)
    return tuple(values)


def unit_interval(values: tuple[float, ...]) -> tuple[float, ...]:
    """把 [-1, 1] 确定性序列映射到 [0, 1], 作为 semantic proxy 输入."""
    return tuple((value + 1.0) / 2.0 for value in values)


def derive_core_carrier_values(
    config: InjectionRunConfig,
    trajectory_index: int,
    carrier_width: int,
) -> tuple[tuple[float, ...], dict[str, Any]]:
    """复用核心算法原语生成可平铺到真实 latent 的 carrier 向量."""
    prompt_digest = build_prompt_digest(config)
    seed_parts = (
        config.model_id,
        prompt_digest,
        str(config.seed),
        str(trajectory_index),
        config.watermark_key_digest,
    )
    latent_proxy = stable_unit_sequence((*seed_parts, "latent_proxy"), carrier_width)
    semantic_values = unit_interval(stable_unit_sequence((*seed_parts, "semantic"), carrier_width))
    texture_values = unit_interval(stable_unit_sequence((*seed_parts, "texture"), carrier_width))
    stability_values = unit_interval(stable_unit_sequence((*seed_parts, "stability"), carrier_width))
    saliency_values = unit_interval(stable_unit_sequence((*seed_parts, "saliency"), carrier_width))
    attention_values = unit_interval(stable_unit_sequence((*seed_parts, "attention"), carrier_width))
    risk_field = build_semantic_risk_field(
        semantic_values=semantic_values,
        texture_values=texture_values,
        stability_values=stability_values,
        saliency_values=saliency_values,
        attention_stability_values=attention_values,
    )
    projection = project_latent_mask(latent_proxy, mask_values=saliency_values)
    safe_basis = estimate_safe_basis(latent_proxy, projection, risk_field, basis_rank=4)
    event_digest = build_stable_digest(
        {
            "injection_id": build_injection_id(config),
            "trajectory_index": trajectory_index,
            "prompt_digest": prompt_digest,
        }
    )
    lf_carrier = derive_lf_carrier(safe_basis, key=config.watermark_key_digest, event_digest=event_digest)
    tail_carrier = derive_tail_carrier(safe_basis, risk_field, key=config.watermark_key_digest, event_digest=event_digest)
    attention_carrier = derive_attention_carrier_stub(
        safe_basis,
        key=config.watermark_key_digest,
        event_digest=event_digest,
    )
    composition = compose_latent_update(lf_carrier, tail_carrier, attention_carrier)
    metadata = {
        "carrier_source": "core_algorithm_primitives",
        "carrier_width": carrier_width,
        "lf_carrier_digest": lf_carrier.carrier_digest,
        "tail_carrier_digest": tail_carrier.carrier_digest,
        "attention_carrier_digest": attention_carrier.carrier_digest,
        "core_update_digest": composition.update_digest,
    }
    return composition.combined_update_values, metadata


def make_carrier_tensor(latents: Any, config: InjectionRunConfig, trajectory_index: int) -> tuple[Any, dict[str, Any]]:
    """构造与当前 latent 同形状的确定性 core carrier tensor."""
    _, torch, _, _ = import_runtime_dependencies()
    carrier_width = int(os.environ.get("SLM_WM_CARRIER_WIDTH", "512"))
    if carrier_width < 2:
        raise ValueError("SLM_WM_CARRIER_WIDTH 必须大于等于2")
    carrier_values, metadata = derive_core_carrier_values(config, trajectory_index, carrier_width)
    base = torch.tensor(carrier_values, device=latents.device, dtype=latents.dtype)
    repeat_count = math.ceil(latents.numel() / base.numel())
    tiled = base.repeat(repeat_count)[: latents.numel()].reshape(latents.shape)
    carrier = tiled / tiled.detach().float().std().clamp_min(1e-6).to(latents.dtype)
    return carrier, metadata


def compute_image_quality_metrics(clean_image: Any, watermarked_image: Any) -> dict[str, float | str]:
    """计算 paired image 的轻量质量指标."""
    import torch

    def _image_tensor(image: Any) -> Any:
        """将 PIL 图像转成 HWC float tensor, 避免质量指标路径依赖 NumPy。"""

        rgb_image = image.convert("RGB")
        width, height = rgb_image.size
        image_bytes = bytearray(rgb_image.tobytes())
        return torch.frombuffer(image_bytes, dtype=torch.uint8).reshape(height, width, 3).float() / 255.0

    clean = _image_tensor(clean_image)
    watermarked = _image_tensor(watermarked_image)
    diff = clean - watermarked
    mse = float((diff * diff).mean().item())
    mean_abs_error = float(diff.abs().mean().item())
    psnr: float | str = "inf" if mse == 0.0 else float(20.0 * math.log10(1.0 / math.sqrt(mse)))
    clean_mean = float(clean.mean().item())
    watermarked_mean = float(watermarked.mean().item())
    clean_var = float(clean.var(unbiased=False).item())
    watermarked_var = float(watermarked.var(unbiased=False).item())
    covariance = float(((clean - clean_mean) * (watermarked - watermarked_mean)).mean().item())
    c1 = 0.01**2
    c2 = 0.03**2
    ssim = float(((2 * clean_mean * watermarked_mean + c1) * (2 * covariance + c2)) / ((clean_mean**2 + watermarked_mean**2 + c1) * (clean_var + watermarked_var + c2)))
    return {"psnr": psnr, "ssim": ssim, "mse": mse, "mean_abs_error": mean_abs_error}


def build_failure_result(config: InjectionRunConfig, error: Exception) -> tuple[InjectionRunResult, tuple[LatentUpdateRecord, ...]]:
    """把真实后端不可用状态转为可审计失败摘要."""
    injection_id = build_injection_id(config)
    environment_report = build_runtime_environment_report()
    result = InjectionRunResult(
        injection_id=injection_id,
        model_family=config.model_family,
        model_id=config.model_id,
        model_priority=config.model_priority,
        run_decision="fail",
        unsupported_reason=type(error).__name__,
        clean_image_path="",
        watermarked_image_path="",
        clean_image_digest="",
        watermarked_image_digest="",
        latent_update_count=0,
        psnr=0.0,
        ssim=0.0,
        mse=0.0,
        mean_abs_error=0.0,
        metadata={
            **flatten_environment_versions(environment_report),
            "error_message": str(error),
            "runtime_environment": environment_report,
            "supports_paper_claim": False,
        },
    )
    return result, ()


def run_single_injection(config: InjectionRunConfig) -> tuple[InjectionRunResult, tuple[LatentUpdateRecord, ...], Any, Any]:
    """运行 clean 与 watermarked 生成, 并在回调中执行最小 latent injection."""
    _, torch, _, _ = import_runtime_dependencies()
    injection_id = build_injection_id(config)
    pipeline, runtime_versions = load_pipeline(config)
    clean_generator = torch.Generator(device=config.device_name).manual_seed(config.seed)
    watermarked_generator = torch.Generator(device=config.device_name).manual_seed(config.seed)
    common_kwargs = {
        "prompt": config.prompt,
        "negative_prompt": config.negative_prompt,
        "width": config.width,
        "height": config.height,
        "num_inference_steps": config.inference_steps,
        "guidance_scale": config.guidance_scale,
        "output_type": "pil",
    }
    clean_output = pipeline(generator=clean_generator, **common_kwargs)
    update_records: list[LatentUpdateRecord] = []
    update_index_by_step = {step_index: order for order, step_index in enumerate(config.injection_step_indices)}

    def inject_latents(pipe: Any, trajectory_index: int, timestep: Any, callback_kwargs: dict[str, Any]) -> dict[str, Any]:
        latents = callback_kwargs.get("latents")
        if latents is None or trajectory_index not in update_index_by_step:
            return callback_kwargs
        carrier, carrier_metadata = make_carrier_tensor(latents, config, trajectory_index)
        update = carrier * config.injection_strength
        injected = latents + update
        latent_norm_before = tensor_norm(latents)
        latent_norm_after = tensor_norm(injected)
        update_norm = tensor_norm(update)
        update_records.append(
            LatentUpdateRecord(
                injection_id=injection_id,
                model_family=config.model_family,
                model_id=config.model_id,
                model_priority=config.model_priority,
                update_index=update_index_by_step[trajectory_index],
                trajectory_index=int(trajectory_index),
                timestep=float(timestep),
                injection_strength=config.injection_strength,
                carrier_digest=tensor_digest(carrier.detach().float().cpu()),
                latent_digest_before=tensor_digest(latents.detach().float().cpu()),
                latent_digest_after=tensor_digest(injected.detach().float().cpu()),
                update_norm=update_norm,
                latent_norm_before=latent_norm_before,
                latent_norm_after=latent_norm_after,
                relative_update_norm=update_norm / max(latent_norm_before, 1e-12),
                metadata={**carrier_metadata, "supports_paper_claim": False},
            )
        )
        callback_kwargs["latents"] = injected
        return callback_kwargs

    watermarked_output = pipeline(
        generator=watermarked_generator,
        callback_on_step_end=inject_latents,
        callback_on_step_end_tensor_inputs=["latents"],
        **common_kwargs,
    )
    clean_image = clean_output.images[0]
    watermarked_image = watermarked_output.images[0]
    metrics = compute_image_quality_metrics(clean_image, watermarked_image)
    result = InjectionRunResult(
        injection_id=injection_id,
        model_family=config.model_family,
        model_id=config.model_id,
        model_priority=config.model_priority,
        run_decision="pass" if update_records else "fail",
        unsupported_reason="" if update_records else "latent_callback_unavailable",
        clean_image_path="",
        watermarked_image_path="",
        clean_image_digest="",
        watermarked_image_digest="",
        latent_update_count=len(update_records),
        metadata={**runtime_versions, "supports_paper_claim": False},
        **metrics,
    )
    return result, tuple(update_records), clean_image, watermarked_image


def write_single_injection_outputs(config: InjectionRunConfig, root: str | Path = ".") -> dict[str, Any]:
    """运行单模型最小 latent injection 并写出受治理产物."""
    root_path = Path(root).resolve()
    output_dir = (root_path / config.output_dir).resolve()
    clean_dir = output_dir / "clean_images"
    watermarked_dir = output_dir / "watermarked_images"
    clean_dir.mkdir(parents=True, exist_ok=True)
    watermarked_dir.mkdir(parents=True, exist_ok=True)
    try:
        result, update_records, clean_image, watermarked_image = run_single_injection(config)
        clean_path = clean_dir / f"{config.model_family}_{config.seed}.png"
        watermarked_path = watermarked_dir / f"{config.model_family}_{config.seed}.png"
        clean_image.save(clean_path)
        watermarked_image.save(watermarked_path)
        result = InjectionRunResult(
            **{
                **result.to_dict(),
                "clean_image_path": clean_path.relative_to(root_path).as_posix(),
                "watermarked_image_path": watermarked_path.relative_to(root_path).as_posix(),
                "clean_image_digest": file_digest(clean_path),
                "watermarked_image_digest": file_digest(watermarked_path),
            }
        )
    except Exception as error:  # pragma: no cover - 该路径依赖 Colab、GPU 与远程模型状态.
        result, update_records = build_failure_result(config, error)

    result_path = output_dir / f"{config.model_family}_injection_result.json"
    updates_path = output_dir / f"{config.model_family}_latent_update_records.jsonl"
    metrics_path = output_dir / f"{config.model_family}_paired_quality_metrics.csv"
    environment_path = output_dir / f"{config.model_family}_environment_report.json"
    manifest_path = output_dir / f"{config.model_family}_manifest.local.json"
    environment_report = result.metadata.get("runtime_environment")
    if environment_report is None:
        environment_report = build_runtime_environment_report()
    environment_report_relative_path = environment_path.relative_to(root_path).as_posix()
    result = InjectionRunResult(
        **{
            **result.to_dict(),
            "metadata": {
                **result.metadata,
                "environment_report_path": environment_report_relative_path,
            },
        }
    )
    result_path.write_text(stable_json_text(result.to_dict()), encoding="utf-8")
    updates_path.write_text("".join(json_line(record.to_dict()) for record in update_records), encoding="utf-8")
    environment_path.write_text(stable_json_text(environment_report), encoding="utf-8")
    with metrics_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "injection_id",
                "model_family",
                "model_id",
                "model_priority",
                "clean_image_path",
                "watermarked_image_path",
                "psnr",
                "ssim",
                "mse",
                "mean_abs_error",
                "run_decision",
                "unsupported_reason",
            ),
        )
        writer.writeheader()
        writer.writerow(
            {
                "injection_id": result.injection_id,
                "model_family": result.model_family,
                "model_id": result.model_id,
                "model_priority": result.model_priority,
                "clean_image_path": result.clean_image_path,
                "watermarked_image_path": result.watermarked_image_path,
                "psnr": result.psnr,
                "ssim": result.ssim,
                "mse": result.mse,
                "mean_abs_error": result.mean_abs_error,
                "run_decision": result.run_decision,
                "unsupported_reason": result.unsupported_reason,
            }
        )
    output_paths = tuple(
        path.relative_to(root_path).as_posix()
        for path in (result_path, updates_path, metrics_path, environment_path, manifest_path)
    )
    manifest = build_artifact_manifest(
        artifact_id=f"{config.model_family}_minimal_latent_injection_manifest",
        artifact_type="local_manifest",
        input_paths=(
            "experiments/runners/minimal_latent_injection.py",
        ),
        output_paths=output_paths,
        config={
            "model_family": config.model_family,
            "model_id": config.model_id,
            "model_priority": config.model_priority,
            "prompt_digest": build_prompt_digest(config),
            "seed": config.seed,
            "injection_strength": config.injection_strength,
            "injection_step_indices": config.injection_step_indices,
            "watermark_key_digest": config.watermark_key_digest,
            "latent_update_count": result.latent_update_count,
            "environment_report_path": environment_report_relative_path,
        },
        code_version=resolve_code_version(root_path),
        rebuild_command="调用 experiments.runners.minimal_latent_injection",
        metadata={
            "construction_unit_name": "minimal_diffusion_latent_injection",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "run_decision": result.run_decision,
            "unsupported_reason": result.unsupported_reason,
        },
    ).to_dict()
    manifest_path.write_text(stable_json_text(manifest), encoding="utf-8")
    return result.to_dict()


def build_default_configs(model_selection: str = "both") -> tuple[InjectionRunConfig, ...]:
    """根据环境变量构造默认模型运行计划."""
    watermark_key_digest = build_stable_digest(
        {"watermark_key": os.environ.get("SLM_WM_WATERMARK_KEY", "slm_wm_colab_probe_key")}
    )
    common = {
        "prompt": os.environ.get("SLM_WM_PROMPT", "a high quality photograph of a glass sphere on a wooden table"),
        "negative_prompt": os.environ.get("SLM_WM_NEGATIVE_PROMPT", "low quality, blurry"),
        "seed": int(os.environ.get("SLM_WM_SEED", "1703")),
        "width": int(os.environ.get("SLM_WM_WIDTH", "512")),
        "height": int(os.environ.get("SLM_WM_HEIGHT", "512")),
        "inference_steps": int(os.environ.get("SLM_WM_INFERENCE_STEPS", "28")),
        "guidance_scale": float(os.environ.get("SLM_WM_GUIDANCE_SCALE", "4.5")),
        "injection_strength": float(os.environ.get("SLM_WM_INJECTION_STRENGTH", "0.035")),
        "injection_step_indices": tuple(
            int(value.strip())
            for value in os.environ.get("SLM_WM_INJECTION_STEPS", "8,14,20").split(",")
            if value.strip()
        ),
        "watermark_key_digest": watermark_key_digest,
    }
    configs = {
        PRIMARY_MODEL_FAMILY: InjectionRunConfig(
            model_family=PRIMARY_MODEL_FAMILY,
            model_id=os.environ.get("SLM_WM_SD35_MODEL_ID", "stabilityai/stable-diffusion-3.5-medium"),
            model_priority="primary",
            **common,
        ),
        FALLBACK_MODEL_FAMILY: InjectionRunConfig(
            model_family=FALLBACK_MODEL_FAMILY,
            model_id=os.environ.get("SLM_WM_SD3_MODEL_ID", "stabilityai/stable-diffusion-3-medium-diffusers"),
            model_priority="compatibility_fallback",
            **common,
        ),
    }
    normalized = model_selection.strip().lower()
    if normalized == FALLBACK_MODEL_FAMILY:
        return (configs[FALLBACK_MODEL_FAMILY],)
    if normalized == "auto":
        return (configs[PRIMARY_MODEL_FAMILY],)
    return (configs[PRIMARY_MODEL_FAMILY], configs[FALLBACK_MODEL_FAMILY])


def run_default_injection_plan(root: str | Path = ".", model_selection: str = "both") -> tuple[dict[str, Any], ...]:
    """运行默认最小 latent injection 计划."""
    results: list[dict[str, Any]] = []
    for config in build_default_configs(model_selection=model_selection):
        results.append(write_single_injection_outputs(config=config, root=root))
    return tuple(results)


def collect_archive_entries(source_dir: Path, archive_path: Path) -> tuple[Path, ...]:
    """收集需要进入压缩包的产物, 避免把历史压缩包递归打包."""
    entries: list[Path] = []
    for path in sorted(source_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.resolve() == archive_path.resolve() or path.suffix.lower() == ".zip":
            continue
        entries.append(path)
    return tuple(entries)


def package_injection_outputs(
    root: str | Path = ".",
    output_dir: str = DEFAULT_OUTPUT_DIR,
    drive_output_dir: str = DEFAULT_DRIVE_OUTPUT_DIR,
    archive_name: str = "minimal_latent_injection_package.zip",
) -> InjectionArchiveRecord:
    """把最小 latent injection 产物打包为 zip, 并镜像到 Google Drive 目录."""
    root_path = Path(root).resolve()
    source_dir = (root_path / output_dir).resolve()
    archive_path = source_dir / archive_name
    summary_path = source_dir / "minimal_latent_injection_archive_summary.json"
    manifest_path = source_dir / "minimal_latent_injection_archive_manifest.local.json"
    if not source_dir.exists():
        raise FileNotFoundError(f"injection_output_dir_missing: {source_dir}")
    entries = collect_archive_entries(source_dir=source_dir, archive_path=archive_path)
    with ZipFile(archive_path, mode="w", compression=ZIP_DEFLATED) as archive:
        for entry in entries:
            archive.write(entry, entry.relative_to(source_dir).as_posix())
    drive_dir = Path(drive_output_dir).expanduser()
    drive_dir.mkdir(parents=True, exist_ok=True)
    mirrored_path = drive_dir / archive_name
    shutil.copy2(archive_path, mirrored_path)
    record = InjectionArchiveRecord(
        archive_path=archive_path.relative_to(root_path).as_posix(),
        archive_digest=file_digest(archive_path),
        archive_entry_count=len(entries),
        drive_archive_path=str(mirrored_path),
        drive_archive_digest=file_digest(mirrored_path),
        metadata={
            "construction_unit_name": "minimal_diffusion_latent_injection",
            "drive_output_dir": str(drive_dir),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    summary_path.write_text(stable_json_text(record.to_dict()), encoding="utf-8")
    manifest = build_artifact_manifest(
        artifact_id="minimal_latent_injection_archive_manifest",
        artifact_type="local_manifest",
        input_paths=(
            "experiments/runners/minimal_latent_injection.py",
            output_dir,
        ),
        output_paths=(
            archive_path.relative_to(root_path).as_posix(),
            summary_path.relative_to(root_path).as_posix(),
            manifest_path.relative_to(root_path).as_posix(),
        ),
        config={
            "archive_name": archive_name,
            "archive_entry_count": len(entries),
            "drive_output_dir": str(drive_dir),
        },
        code_version=resolve_code_version(root_path),
        rebuild_command="调用 experiments.runners.minimal_latent_injection",
        metadata={
            "construction_unit_name": "minimal_diffusion_latent_injection",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "archive_digest": record.archive_digest,
            "drive_archive_digest": record.drive_archive_digest,
        },
    ).to_dict()
    manifest_path.write_text(stable_json_text(manifest), encoding="utf-8")
    return record
