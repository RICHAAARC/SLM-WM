"""Colab 真实 attention-relative latent injection helper。"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

from main.analysis.artifact_manifest import build_artifact_manifest
from main.core.digest import build_stable_digest
from paper_workflow.colab_utils.minimal_latent_injection import (
    compute_image_quality_metrics,
    import_runtime_dependencies,
    load_pipeline,
    tensor_norm,
)
from paper_workflow.colab_utils.sd_runtime_cold_start import (
    build_runtime_environment_report,
    file_digest,
    resolve_code_version,
    tensor_digest,
)
from scripts.write_attention_latent_update_outputs import write_attention_latent_update_outputs
from scripts.write_content_carrier_outputs import write_content_carrier_outputs
from scripts.write_prompt_event_protocol import write_prompt_event_protocol_outputs
from scripts.write_semantic_subspace_outputs import write_semantic_subspace_outputs

DEFAULT_OUTPUT_DIR = "outputs/attention_latent_injection"
DEFAULT_METHOD_OUTPUT_DIR = "outputs/attention_latent_update"
DEFAULT_DRIVE_OUTPUT_DIR = "/content/drive/MyDrive/SLM/attention_latent_injection"
DEFAULT_GEOMETRY_DRIVE_DIR = "/content/drive/MyDrive/SLM/attention_geometry"
PRIMARY_MODEL_FAMILY = "sd35"
PRIMARY_MODEL_ID = "stabilityai/stable-diffusion-3.5-medium"
GEOMETRY_PACKAGE_PATTERN = "attention_geometry_package_*.zip"


@dataclass(frozen=True)
class AttentionLatentInjectionConfig:
    """描述真实 attention-relative latent injection 运行配置。"""

    model_family: str
    model_id: str
    prompt: str
    negative_prompt: str
    seed: int
    width: int
    height: int
    inference_steps: int
    guidance_scale: float
    attention_runtime_strength: float
    injection_step_indices: tuple[int, ...]
    output_dir: str = DEFAULT_OUTPUT_DIR
    method_output_dir: str = DEFAULT_METHOD_OUTPUT_DIR
    geometry_drive_dir: str = DEFAULT_GEOMETRY_DRIVE_DIR
    attention_geometry_package_path: str = ""
    max_subspace_records: int = 16
    attention_carrier_index: int = 0
    device_name: str = "cuda"
    torch_dtype: str = "float16"
    hf_token_env: str = "HF_TOKEN"

    def __post_init__(self) -> None:
        """集中校验配置边界, 避免业务路径重复构造错误信息。"""
        positive_fields = {
            "width": self.width,
            "height": self.height,
            "inference_steps": self.inference_steps,
            "max_subspace_records": self.max_subspace_records,
        }
        invalid_fields = {name: value for name, value in positive_fields.items() if value <= 0}
        if invalid_fields:
            raise ValueError(f"配置正整数边界无效: {invalid_fields}")
        if self.guidance_scale <= 0.0 or self.attention_runtime_strength < 0.0:
            raise ValueError("guidance_scale 必须为正数, attention_runtime_strength 不得为负数")
        if any(index < 0 or index >= self.inference_steps for index in self.injection_step_indices):
            raise ValueError("injection_step_indices 必须位于采样步数边界内")


@dataclass(frozen=True)
class AttentionLatentUpdateRecord:
    """记录真实 latent callback 中一次 attention-relative update。"""

    injection_id: str
    model_family: str
    model_id: str
    update_index: int
    trajectory_index: int
    timestep: float
    attention_runtime_strength: float
    carrier_id: str
    attention_graph_id: str
    capture_id: str
    carrier_digest: str
    latent_digest_before: str
    latent_digest_after: str
    update_norm: float
    latent_norm_before: float
    latent_norm_after: float
    relative_update_norm: float
    relation_loss_before: float
    relation_loss_after: float
    relation_loss_delta: float
    relation_consistency_before: float
    relation_consistency_after: float
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """转为 JSON 兼容字典。"""
        return asdict(self)


@dataclass(frozen=True)
class AttentionLatentInjectionResult:
    """保存真实 attention-relative latent injection 摘要。"""

    injection_id: str
    model_family: str
    model_id: str
    run_decision: str
    unsupported_reason: str
    clean_image_path: str
    watermarked_image_path: str
    clean_image_digest: str
    watermarked_image_digest: str
    latent_update_count: int
    selected_attention_carrier_id: str
    attention_geometry_package_path: str
    method_manifest_path: str
    image_quality_metrics_ready: bool
    full_method_claim_ready: bool
    psnr: float | str
    ssim: float
    mse: float
    mean_abs_error: float
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """转为 JSON 兼容字典。"""
        return asdict(self)


@dataclass(frozen=True)
class AttentionLatentInjectionArchiveRecord:
    """记录真实 attention injection 产物压缩包与 Drive 镜像。"""

    archive_path: str
    archive_digest: str
    archive_entry_count: int
    drive_archive_path: str
    drive_archive_digest: str
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """转为 JSON 兼容字典。"""
        return asdict(self)


def stable_json_text(value: Any) -> str:
    """把 JSON 兼容对象转为稳定文本。"""
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def json_line(value: Any) -> str:
    """把 JSON 兼容对象转为 JSONL 单行文本。"""
    return json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n"


def build_injection_id(config: AttentionLatentInjectionConfig, carrier_record: dict[str, Any]) -> str:
    """根据模型配置和 attention carrier 生成稳定 injection id。"""
    return build_stable_digest(
        {
            "model_family": config.model_family,
            "model_id": config.model_id,
            "prompt": config.prompt,
            "seed": config.seed,
            "width": config.width,
            "height": config.height,
            "inference_steps": config.inference_steps,
            "guidance_scale": config.guidance_scale,
            "attention_runtime_strength": config.attention_runtime_strength,
            "injection_step_indices": config.injection_step_indices,
            "carrier_id": carrier_record["carrier_id"],
            "attention_relative_carrier_digest": carrier_record["attention_relative_carrier_digest"],
        }
    )


def read_jsonl(path: Path) -> tuple[dict[str, Any], ...]:
    """读取 JSONL 文件。"""
    return tuple(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def latest_geometry_package(directory: Path) -> Path | None:
    """查找目录中最新的 attention geometry 压缩包。"""
    if not directory.exists():
        return None
    candidates = sorted(directory.glob(GEOMETRY_PACKAGE_PATTERN), key=lambda path: path.name, reverse=True)
    return candidates[0] if candidates else None


def resolve_geometry_package(config: AttentionLatentInjectionConfig, root_path: Path) -> Path:
    """解析真实 attention geometry 输入包路径。"""
    explicit = config.attention_geometry_package_path.strip()
    if explicit:
        candidate = Path(explicit).expanduser()
        return candidate.resolve() if candidate.is_absolute() else (root_path / candidate).resolve()
    drive_candidate = latest_geometry_package(Path(config.geometry_drive_dir).expanduser())
    if drive_candidate is not None:
        return drive_candidate.resolve()
    local_candidate = latest_geometry_package(root_path / "outputs")
    if local_candidate is not None:
        return local_candidate.resolve()
    raise FileNotFoundError("attention_geometry_package_missing")


def materialize_geometry_package(config: AttentionLatentInjectionConfig, root_path: Path) -> Path:
    """把外部 geometry 包复制到 outputs 下, 便于完整打包核对。"""
    source_path = resolve_geometry_package(config, root_path)
    package_dir = root_path / config.output_dir / "input_packages"
    package_dir.mkdir(parents=True, exist_ok=True)
    target_path = package_dir / source_path.name
    if source_path.resolve() != target_path.resolve():
        shutil.copy2(source_path, target_path)
    return target_path


def prepare_attention_method_outputs(config: AttentionLatentInjectionConfig, root_path: Path) -> dict[str, Any]:
    """重建 prompt、semantic、content 与 attention update 输入链。"""
    geometry_package_path = materialize_geometry_package(config, root_path)
    write_prompt_event_protocol_outputs(root=root_path)
    write_semantic_subspace_outputs(root=root_path, max_records=config.max_subspace_records)
    write_content_carrier_outputs(root=root_path, max_records=config.max_subspace_records)
    method_manifest = write_attention_latent_update_outputs(
        root=root_path,
        output_dir=config.method_output_dir,
        attention_geometry_package_path=geometry_package_path,
        max_subspace_records=config.max_subspace_records,
    )
    return {
        "geometry_package_path": geometry_package_path,
        "method_manifest": method_manifest,
        "method_manifest_path": root_path / config.method_output_dir / "manifest.local.json",
        "method_summary_path": root_path / config.method_output_dir / "attention_update_summary.json",
        "carrier_records_path": root_path / config.method_output_dir / "attention_carrier_records.jsonl",
    }


def select_active_carrier(carrier_records_path: Path, carrier_index: int) -> dict[str, Any]:
    """选择可执行 active update 的 attention carrier。"""
    records = read_jsonl(carrier_records_path)
    active_records = [
        record
        for record in records
        if record.get("fallback_mode") == "active_update" and bool(record.get("attention_update_stable", False))
    ]
    if not active_records:
        raise RuntimeError("active_attention_carrier_missing")
    ordered = sorted(active_records, key=lambda record: (-float(record.get("relation_loss_delta", 0.0)), record["carrier_id"]))
    return ordered[carrier_index % len(ordered)]


def attention_carrier_tensor(latents: Any, carrier_record: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    """把 attention carrier update_values 平铺为与 latent 同形状的 tensor。"""
    _, torch, _, _ = import_runtime_dependencies()
    update_values = tuple(float(value) for value in carrier_record["update_values"])
    base = torch.tensor(update_values, device=latents.device, dtype=latents.dtype)
    repeat_count = math.ceil(latents.numel() / base.numel())
    tiled = base.repeat(repeat_count)[: latents.numel()].reshape(latents.shape)
    centered = tiled - tiled.detach().float().mean().to(latents.dtype)
    carrier = centered / centered.detach().float().std().clamp_min(1e-6).to(latents.dtype)
    metadata = {
        "carrier_source": "attention_relative_latent_update",
        "carrier_id": carrier_record["carrier_id"],
        "attention_relative_carrier_digest": carrier_record["attention_relative_carrier_digest"],
        "attention_graph_id": carrier_record["attention_graph_id"],
        "capture_id": carrier_record["capture_id"],
        "fallback_mode": carrier_record["fallback_mode"],
        "relation_loss_before": carrier_record["relation_loss_before"],
        "relation_loss_after": carrier_record["relation_loss_after"],
        "relation_loss_delta": carrier_record["relation_loss_delta"],
        "relation_consistency_before": carrier_record["relation_consistency_before"],
        "relation_consistency_after": carrier_record["relation_consistency_after"],
    }
    return carrier, metadata


def run_attention_latent_injection(
    config: AttentionLatentInjectionConfig,
    carrier_record: dict[str, Any],
    geometry_package_path: Path,
    method_manifest_path: Path,
) -> tuple[AttentionLatentInjectionResult, tuple[AttentionLatentUpdateRecord, ...], Any, Any]:
    """运行真实 SD3.5 clean / attention-watermarked paired generation。"""
    _, torch, _, _ = import_runtime_dependencies()
    injection_id = build_injection_id(config, carrier_record)
    pipeline, runtime_versions = load_pipeline(config)  # type: ignore[arg-type]
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
    update_index_by_step = {trajectory_index: index for index, trajectory_index in enumerate(config.injection_step_indices)}
    update_records: list[AttentionLatentUpdateRecord] = []

    def inject_latents(pipe: Any, trajectory_index: int, timestep: Any, callback_kwargs: dict[str, Any]) -> dict[str, Any]:
        latents = callback_kwargs.get("latents")
        if latents is None or trajectory_index not in update_index_by_step:
            return callback_kwargs
        carrier, carrier_metadata = attention_carrier_tensor(latents, carrier_record)
        update = carrier * config.attention_runtime_strength
        injected = latents + update
        latent_norm_before = tensor_norm(latents)
        latent_norm_after = tensor_norm(injected)
        update_norm = tensor_norm(update)
        update_records.append(
            AttentionLatentUpdateRecord(
                injection_id=injection_id,
                model_family=config.model_family,
                model_id=config.model_id,
                update_index=update_index_by_step[trajectory_index],
                trajectory_index=int(trajectory_index),
                timestep=float(timestep),
                attention_runtime_strength=config.attention_runtime_strength,
                carrier_id=carrier_record["carrier_id"],
                attention_graph_id=carrier_record["attention_graph_id"],
                capture_id=carrier_record["capture_id"],
                carrier_digest=tensor_digest(carrier.detach().float().cpu()),
                latent_digest_before=tensor_digest(latents.detach().float().cpu()),
                latent_digest_after=tensor_digest(injected.detach().float().cpu()),
                update_norm=update_norm,
                latent_norm_before=latent_norm_before,
                latent_norm_after=latent_norm_after,
                relative_update_norm=update_norm / max(latent_norm_before, 1e-12),
                relation_loss_before=float(carrier_record["relation_loss_before"]),
                relation_loss_after=float(carrier_record["relation_loss_after"]),
                relation_loss_delta=float(carrier_record["relation_loss_delta"]),
                relation_consistency_before=float(carrier_record["relation_consistency_before"]),
                relation_consistency_after=float(carrier_record["relation_consistency_after"]),
                metadata={
                    **carrier_metadata,
                    "geometry_package_path": str(geometry_package_path),
                    "method_manifest_path": str(method_manifest_path),
                    "supports_paper_claim": False,
                },
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
    result = AttentionLatentInjectionResult(
        injection_id=injection_id,
        model_family=config.model_family,
        model_id=config.model_id,
        run_decision="pass" if update_records else "fail",
        unsupported_reason="" if update_records else "attention_latent_callback_unavailable",
        clean_image_path="",
        watermarked_image_path="",
        clean_image_digest="",
        watermarked_image_digest="",
        latent_update_count=len(update_records),
        selected_attention_carrier_id=carrier_record["carrier_id"],
        attention_geometry_package_path=str(geometry_package_path),
        method_manifest_path=str(method_manifest_path),
        image_quality_metrics_ready=bool(update_records),
        full_method_claim_ready=False,
        metadata={**runtime_versions, "supports_paper_claim": False},
        **metrics,
    )
    return result, tuple(update_records), clean_image, watermarked_image


def build_failure_result(
    config: AttentionLatentInjectionConfig,
    carrier_record: dict[str, Any] | None,
    geometry_package_path: Path | None,
    method_manifest_path: Path | None,
    error: Exception,
) -> tuple[AttentionLatentInjectionResult, tuple[AttentionLatentUpdateRecord, ...]]:
    """把真实后端不可用状态转为可审计失败摘要。"""
    environment_report = build_runtime_environment_report()
    fallback_carrier = carrier_record or {}
    injection_id = build_stable_digest({"error": type(error).__name__, "model_id": config.model_id, "seed": config.seed})
    result = AttentionLatentInjectionResult(
        injection_id=injection_id,
        model_family=config.model_family,
        model_id=config.model_id,
        run_decision="fail",
        unsupported_reason=type(error).__name__,
        clean_image_path="",
        watermarked_image_path="",
        clean_image_digest="",
        watermarked_image_digest="",
        latent_update_count=0,
        selected_attention_carrier_id=str(fallback_carrier.get("carrier_id", "")),
        attention_geometry_package_path="" if geometry_package_path is None else str(geometry_package_path),
        method_manifest_path="" if method_manifest_path is None else str(method_manifest_path),
        image_quality_metrics_ready=False,
        full_method_claim_ready=False,
        psnr=0.0,
        ssim=0.0,
        mse=0.0,
        mean_abs_error=0.0,
        metadata={
            "error_message": str(error),
            "runtime_environment": environment_report,
            "supports_paper_claim": False,
        },
    )
    return result, ()


def write_attention_latent_injection_outputs(
    config: AttentionLatentInjectionConfig,
    root: str | Path = ".",
) -> dict[str, Any]:
    """运行真实 attention-relative latent injection 并写出受治理产物。"""
    root_path = Path(root).resolve()
    output_dir = (root_path / config.output_dir).resolve()
    clean_dir = output_dir / "clean_images"
    watermarked_dir = output_dir / "watermarked_images"
    output_dir.mkdir(parents=True, exist_ok=True)
    clean_dir.mkdir(parents=True, exist_ok=True)
    watermarked_dir.mkdir(parents=True, exist_ok=True)
    carrier_record: dict[str, Any] | None = None
    geometry_package_path: Path | None = None
    method_manifest_path: Path | None = None

    try:
        prepared = prepare_attention_method_outputs(config, root_path)
        geometry_package_path = prepared["geometry_package_path"]
        method_manifest_path = prepared["method_manifest_path"]
        carrier_record = select_active_carrier(prepared["carrier_records_path"], config.attention_carrier_index)
        result, update_records, clean_image, watermarked_image = run_attention_latent_injection(
            config=config,
            carrier_record=carrier_record,
            geometry_package_path=geometry_package_path,
            method_manifest_path=method_manifest_path,
        )
        clean_path = clean_dir / f"{config.model_family}_{config.seed}_attention_clean.png"
        watermarked_path = watermarked_dir / f"{config.model_family}_{config.seed}_attention_watermarked.png"
        clean_image.save(clean_path)
        watermarked_image.save(watermarked_path)
        result = AttentionLatentInjectionResult(
            **{
                **result.to_dict(),
                "clean_image_path": clean_path.relative_to(root_path).as_posix(),
                "watermarked_image_path": watermarked_path.relative_to(root_path).as_posix(),
                "clean_image_digest": file_digest(clean_path),
                "watermarked_image_digest": file_digest(watermarked_path),
                "attention_geometry_package_path": geometry_package_path.relative_to(root_path).as_posix(),
                "method_manifest_path": method_manifest_path.relative_to(root_path).as_posix(),
            }
        )
    except Exception as error:  # pragma: no cover - 该路径依赖 Colab、GPU 与远程模型状态。
        result, update_records = build_failure_result(config, carrier_record, geometry_package_path, method_manifest_path, error)

    result_path = output_dir / "attention_latent_injection_result.json"
    updates_path = output_dir / "attention_latent_update_records.jsonl"
    metrics_path = output_dir / "attention_paired_quality_metrics.csv"
    environment_path = output_dir / "attention_injection_environment_report.json"
    manifest_path = output_dir / "attention_latent_injection_manifest.local.json"
    environment_report = result.metadata.get("runtime_environment") or build_runtime_environment_report()
    environment_path.write_text(stable_json_text(environment_report), encoding="utf-8")
    result = AttentionLatentInjectionResult(
        **{
            **result.to_dict(),
            "metadata": {
                **result.metadata,
                "environment_report_path": environment_path.relative_to(root_path).as_posix(),
            },
        }
    )
    result_path.write_text(stable_json_text(result.to_dict()), encoding="utf-8")
    updates_path.write_text("".join(json_line(record.to_dict()) for record in update_records), encoding="utf-8")
    with metrics_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "injection_id",
                "model_family",
                "model_id",
                "selected_attention_carrier_id",
                "clean_image_path",
                "watermarked_image_path",
                "psnr",
                "ssim",
                "mse",
                "mean_abs_error",
                "image_quality_metrics_ready",
                "full_method_claim_ready",
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
                "selected_attention_carrier_id": result.selected_attention_carrier_id,
                "clean_image_path": result.clean_image_path,
                "watermarked_image_path": result.watermarked_image_path,
                "psnr": result.psnr,
                "ssim": result.ssim,
                "mse": result.mse,
                "mean_abs_error": result.mean_abs_error,
                "image_quality_metrics_ready": result.image_quality_metrics_ready,
                "full_method_claim_ready": result.full_method_claim_ready,
                "run_decision": result.run_decision,
                "unsupported_reason": result.unsupported_reason,
            }
        )
    output_paths = tuple(
        path.relative_to(root_path).as_posix()
        for path in (result_path, updates_path, metrics_path, environment_path, manifest_path)
    )
    input_paths = [
        "paper_workflow/attention_latent_injection_run.ipynb",
        "paper_workflow/colab_utils/attention_latent_injection.py",
        "scripts/write_attention_latent_update_outputs.py",
    ]
    if result.attention_geometry_package_path:
        input_paths.append(result.attention_geometry_package_path)
    if result.method_manifest_path:
        input_paths.append(result.method_manifest_path)
    manifest = build_artifact_manifest(
        artifact_id="attention_latent_injection_manifest",
        artifact_type="local_manifest",
        input_paths=tuple(input_paths),
        output_paths=output_paths,
        config={
            "model_family": config.model_family,
            "model_id": config.model_id,
            "prompt_digest": build_stable_digest({"prompt": config.prompt, "negative_prompt": config.negative_prompt}),
            "seed": config.seed,
            "attention_runtime_strength": config.attention_runtime_strength,
            "injection_step_indices": config.injection_step_indices,
            "latent_update_count": result.latent_update_count,
            "selected_attention_carrier_id": result.selected_attention_carrier_id,
            "image_quality_metrics_ready": result.image_quality_metrics_ready,
            "full_method_claim_ready": result.full_method_claim_ready,
        },
        code_version=resolve_code_version(root_path),
        rebuild_command="运行 paper_workflow/attention_latent_injection_run.ipynb",
        metadata={
            "construction_unit_name": "attention_latent_injection",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "run_decision": result.run_decision,
            "unsupported_reason": result.unsupported_reason,
            "supports_paper_claim": False,
        },
    ).to_dict()
    manifest_path.write_text(stable_json_text(manifest), encoding="utf-8")
    return result.to_dict()


def build_default_config() -> AttentionLatentInjectionConfig:
    """根据环境变量构造默认真实 attention latent injection 配置。"""
    return AttentionLatentInjectionConfig(
        model_family=PRIMARY_MODEL_FAMILY,
        model_id=os.environ.get("SLM_WM_SD35_MODEL_ID", PRIMARY_MODEL_ID),
        prompt=os.environ.get("SLM_WM_PROMPT", "a high quality photograph of a glass sphere on a wooden table"),
        negative_prompt=os.environ.get("SLM_WM_NEGATIVE_PROMPT", "low quality, blurry"),
        seed=int(os.environ.get("SLM_WM_SEED", "1703")),
        width=int(os.environ.get("SLM_WM_WIDTH", "512")),
        height=int(os.environ.get("SLM_WM_HEIGHT", "512")),
        inference_steps=int(os.environ.get("SLM_WM_INFERENCE_STEPS", "20")),
        guidance_scale=float(os.environ.get("SLM_WM_GUIDANCE_SCALE", "4.5")),
        attention_runtime_strength=float(os.environ.get("SLM_WM_ATTENTION_RUNTIME_STRENGTH", "0.025")),
        injection_step_indices=tuple(
            int(value.strip())
            for value in os.environ.get("SLM_WM_ATTENTION_INJECTION_STEPS", "6,10,14").split(",")
            if value.strip()
        ),
        output_dir=os.environ.get("SLM_WM_ATTENTION_INJECTION_OUTPUT_DIR", DEFAULT_OUTPUT_DIR),
        method_output_dir=os.environ.get("SLM_WM_ATTENTION_METHOD_OUTPUT_DIR", DEFAULT_METHOD_OUTPUT_DIR),
        geometry_drive_dir=os.environ.get("SLM_WM_ATTENTION_GEOMETRY_DRIVE_DIR", DEFAULT_GEOMETRY_DRIVE_DIR),
        attention_geometry_package_path=os.environ.get("SLM_WM_ATTENTION_GEOMETRY_PACKAGE_PATH", ""),
        max_subspace_records=int(os.environ.get("SLM_WM_ATTENTION_SUBSPACE_RECORDS", "16")),
        attention_carrier_index=int(os.environ.get("SLM_WM_ATTENTION_CARRIER_INDEX", "0")),
    )


def run_default_attention_latent_injection_plan(root: str | Path = ".") -> dict[str, Any]:
    """运行默认真实 attention latent injection 计划。"""
    return write_attention_latent_injection_outputs(config=build_default_config(), root=root)


def collect_package_entries(root_path: Path, output_dirs: tuple[Path, ...], archive_path: Path) -> tuple[Path, ...]:
    """收集需要进入压缩包的核对文件。"""
    entries: list[Path] = []
    for output_dir in output_dirs:
        if not output_dir.exists():
            continue
        for path in sorted(output_dir.rglob("*")):
            if path.is_file() and path.resolve() != archive_path.resolve():
                entries.append(path)
    unique_entries = []
    for entry in entries:
        if entry not in unique_entries:
            unique_entries.append(entry)
    return tuple(unique_entries)


def package_attention_latent_injection_outputs(
    root: str | Path = ".",
    output_dir: str = DEFAULT_OUTPUT_DIR,
    method_output_dir: str = DEFAULT_METHOD_OUTPUT_DIR,
    drive_output_dir: str = DEFAULT_DRIVE_OUTPUT_DIR,
    archive_name: str = "attention_latent_injection_package.zip",
) -> AttentionLatentInjectionArchiveRecord:
    """打包真实 attention latent injection 产物并镜像到 Google Drive。"""
    root_path = Path(root).resolve()
    source_dir = (root_path / output_dir).resolve()
    method_dir = (root_path / method_output_dir).resolve()
    source_dir.mkdir(parents=True, exist_ok=True)
    archive_path = source_dir / archive_name
    package_manifest_path = source_dir / "attention_latent_injection_package_input_manifest.json"
    summary_path = source_dir / "attention_latent_injection_archive_summary.json"
    manifest_path = source_dir / "attention_latent_injection_archive_manifest.local.json"
    entries = collect_package_entries(root_path, (source_dir, method_dir), archive_path)
    package_manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "entry_paths": [entry.relative_to(root_path).as_posix() for entry in entries],
        "entry_count": len(entries),
    }
    package_manifest_path.write_text(stable_json_text(package_manifest), encoding="utf-8")
    entries = tuple((*entries, package_manifest_path))
    with ZipFile(archive_path, mode="w", compression=ZIP_DEFLATED) as archive:
        for entry in entries:
            archive.write(entry, entry.relative_to(root_path).as_posix())
    drive_dir = Path(drive_output_dir).expanduser()
    drive_dir.mkdir(parents=True, exist_ok=True)
    mirrored_path = drive_dir / archive_name
    shutil.copy2(archive_path, mirrored_path)
    record = AttentionLatentInjectionArchiveRecord(
        archive_path=archive_path.relative_to(root_path).as_posix(),
        archive_digest=file_digest(archive_path),
        archive_entry_count=len(entries),
        drive_archive_path=str(mirrored_path),
        drive_archive_digest=file_digest(mirrored_path),
        metadata={
            "construction_unit_name": "attention_latent_injection",
            "drive_output_dir": str(drive_dir),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    summary_path.write_text(stable_json_text(record.to_dict()), encoding="utf-8")
    manifest = build_artifact_manifest(
        artifact_id="attention_latent_injection_archive_manifest",
        artifact_type="local_manifest",
        input_paths=tuple(entry.relative_to(root_path).as_posix() for entry in entries),
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
        rebuild_command="运行 paper_workflow/attention_latent_injection_run.ipynb",
        metadata={
            "construction_unit_name": "attention_latent_injection",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "archive_digest": record.archive_digest,
            "drive_archive_digest": record.drive_archive_digest,
        },
    ).to_dict()
    manifest_path.write_text(stable_json_text(manifest), encoding="utf-8")
    return record
