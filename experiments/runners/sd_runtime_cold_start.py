"""真实 SD runtime 诊断 probe 的正式 runner。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import os
from pathlib import Path
import shutil
import time
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

from experiments.runtime.repository_environment import (
    COLAB_DYNAMIC_DEPENDENCY_INSTALL_COMMAND,
    RUNTIME_ENVIRONMENT_PACKAGES,
    build_runtime_environment_report,
    file_digest,
    flatten_environment_versions,
    json_line,
    read_package_version,
    resolve_code_version,
    stable_json_text,
    tensor_digest,
)
from main.analysis.artifact_manifest import build_artifact_manifest
from main.core.digest import build_stable_digest


DEFAULT_OUTPUT_DIR = "outputs/real_sd_runtime_probe"
DEFAULT_DRIVE_OUTPUT_DIR = "/content/drive/MyDrive/SLM/real_sd_runtime_probe"
PRIMARY_MODEL_FAMILY = "sd35"
FALLBACK_MODEL_FAMILY = "sd3"


@dataclass(frozen=True)
class RealRuntimeConfig:
    """描述一次真实 SD runtime 调用所需的最小配置。"""

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
    output_dir: str = DEFAULT_OUTPUT_DIR
    device_name: str = "cuda"
    torch_dtype: str = "float16"
    hf_token_env: str = "HF_TOKEN"


@dataclass(frozen=True)
class LatentTrajectoryRecord:
    """保存真实采样回调中的 latent 摘要。"""

    probe_id: str
    model_family: str
    model_id: str
    model_priority: str
    trajectory_index: int
    timestep: float
    latent_digest: str
    latent_shape: tuple[int, ...]
    latent_mean: float
    latent_std: float
    latent_min: float
    latent_max: float

    def to_dict(self) -> dict[str, Any]:
        """转为 JSON 兼容字典。"""
        return asdict(self)


@dataclass(frozen=True)
class RealRuntimeResult:
    """保存一次真实 SD runtime 调用的结果摘要。"""

    probe_id: str
    model_family: str
    model_id: str
    model_priority: str
    probe_decision: str
    unsupported_reason: str
    prompt_digest: str
    seed: int
    image_path: str
    image_digest: str
    trajectory_entry_count: int
    pipeline_class: str
    device_name: str
    torch_dtype: str
    elapsed_seconds: float
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """转为 JSON 兼容字典。"""
        return asdict(self)


@dataclass(frozen=True)
class ProbeArchiveRecord:
    """记录真实 SD runtime 产物压缩包与 Drive 镜像。"""

    archive_path: str
    archive_digest: str
    archive_entry_count: int
    drive_archive_path: str
    drive_archive_digest: str
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """转为 JSON 兼容字典。"""
        return asdict(self)


def import_runtime_dependencies() -> tuple[Any, Any, Any]:
    """延迟导入真实 SD runtime 依赖。"""
    import torch
    import diffusers
    from diffusers import StableDiffusion3Pipeline

    return torch, diffusers, StableDiffusion3Pipeline


def build_probe_id(config: RealRuntimeConfig) -> str:
    """根据配置生成稳定 probe id。"""
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
        }
    )


def build_prompt_digest(config: RealRuntimeConfig) -> str:
    """根据 prompt 与模型配置生成稳定摘要。"""
    return build_stable_digest(
        {
            "prompt": config.prompt,
            "negative_prompt": config.negative_prompt,
            "model_id": config.model_id,
            "seed": config.seed,
        }
    )


def load_pipeline(config: RealRuntimeConfig) -> tuple[Any, dict[str, Any]]:
    """加载真实 SD pipeline 并放入目标设备。"""
    torch, diffusers, pipeline_class = import_runtime_dependencies()
    if config.device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("gpu_unavailable")
    dtype = getattr(torch, config.torch_dtype)
    token = os.environ.get(config.hf_token_env) or None
    pipeline = pipeline_class.from_pretrained(
        config.model_id,
        torch_dtype=dtype,
        token=token,
    )
    pipeline = pipeline.to(config.device_name)
    pipeline.set_progress_bar_config(disable=False)
    environment_report = build_runtime_environment_report(torch_module=torch)
    runtime_versions = {
        **flatten_environment_versions(environment_report),
        "runtime_environment": environment_report,
    }
    return pipeline, runtime_versions


def run_real_runtime(config: RealRuntimeConfig) -> tuple[RealRuntimeResult, tuple[LatentTrajectoryRecord, ...], Any]:
    """执行真实 SD 推理并捕获 latent trajectory。"""
    torch, _, _ = import_runtime_dependencies()
    probe_id = build_probe_id(config)
    trajectory_records: list[LatentTrajectoryRecord] = []
    started_at = time.time()
    pipeline, runtime_versions = load_pipeline(config)
    generator = torch.Generator(device=config.device_name).manual_seed(config.seed)

    def capture_latents(pipe: Any, trajectory_index: int, timestep: Any, callback_kwargs: dict[str, Any]) -> dict[str, Any]:
        latents = callback_kwargs.get("latents")
        if latents is not None:
            detached = latents.detach().float().cpu()
            trajectory_records.append(
                LatentTrajectoryRecord(
                    probe_id=probe_id,
                    model_family=config.model_family,
                    model_id=config.model_id,
                    model_priority=config.model_priority,
                    trajectory_index=int(trajectory_index),
                    timestep=float(timestep),
                    latent_digest=tensor_digest(detached),
                    latent_shape=tuple(int(value) for value in detached.shape),
                    latent_mean=float(detached.mean().item()),
                    latent_std=float(detached.std().item()),
                    latent_min=float(detached.min().item()),
                    latent_max=float(detached.max().item()),
                )
            )
        return callback_kwargs

    output = pipeline(
        prompt=config.prompt,
        negative_prompt=config.negative_prompt,
        width=config.width,
        height=config.height,
        num_inference_steps=config.inference_steps,
        guidance_scale=config.guidance_scale,
        generator=generator,
        output_type="pil",
        callback_on_step_end=capture_latents,
        callback_on_step_end_tensor_inputs=["latents"],
    )
    elapsed_seconds = time.time() - started_at
    result = RealRuntimeResult(
        probe_id=probe_id,
        model_family=config.model_family,
        model_id=config.model_id,
        model_priority=config.model_priority,
        probe_decision="pass" if trajectory_records else "fail",
        unsupported_reason="" if trajectory_records else "latent_callback_unavailable",
        prompt_digest=build_prompt_digest(config),
        seed=config.seed,
        image_path="",
        image_digest="",
        trajectory_entry_count=len(trajectory_records),
        pipeline_class=type(pipeline).__name__,
        device_name=config.device_name,
        torch_dtype=config.torch_dtype,
        elapsed_seconds=elapsed_seconds,
        metadata=runtime_versions,
    )
    return result, tuple(trajectory_records), output.images[0]


def build_failure_result(config: RealRuntimeConfig, error: Exception) -> RealRuntimeResult:
    """把不可用真实后端转为可审计失败摘要。"""
    environment_report = build_runtime_environment_report()
    return RealRuntimeResult(
        probe_id=build_probe_id(config),
        model_family=config.model_family,
        model_id=config.model_id,
        model_priority=config.model_priority,
        probe_decision="fail",
        unsupported_reason=type(error).__name__,
        prompt_digest=build_prompt_digest(config),
        seed=config.seed,
        image_path="",
        image_digest="",
        trajectory_entry_count=0,
        pipeline_class="unavailable",
        device_name=config.device_name,
        torch_dtype=config.torch_dtype,
        elapsed_seconds=0.0,
        metadata={
            **flatten_environment_versions(environment_report),
            "error_message": str(error),
            "runtime_environment": environment_report,
        },
    )


def write_single_model_outputs(config: RealRuntimeConfig, root: str | Path = ".") -> dict[str, Any]:
    """运行单个真实 SD runtime 调用并写出受治理输出。"""
    root_path = Path(root).resolve()
    output_dir = (root_path / config.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    try:
        result, trajectory_records, image = run_real_runtime(config)
        image_path = image_dir / f"{config.model_family}_{config.seed}.png"
        image.save(image_path)
        result = RealRuntimeResult(
            **{
                **result.to_dict(),
                "image_path": image_path.relative_to(root_path).as_posix(),
                "image_digest": file_digest(image_path),
            }
        )
    except Exception as error:  # pragma: no cover - 该路径依赖 Colab、GPU 与远程模型状态。
        result = build_failure_result(config, error)
        trajectory_records = ()

    generation_path = output_dir / f"{config.model_family}_generation_record.json"
    trajectory_path = output_dir / f"{config.model_family}_latent_trajectory_records.jsonl"
    summary_path = output_dir / f"{config.model_family}_runtime_summary.json"
    environment_path = output_dir / f"{config.model_family}_environment_report.json"
    manifest_path = output_dir / f"{config.model_family}_manifest.local.json"
    environment_report = result.metadata.get("runtime_environment")
    if environment_report is None:
        environment_report = build_runtime_environment_report()
    environment_report_relative_path = environment_path.relative_to(root_path).as_posix()
    result = RealRuntimeResult(
        **{
            **result.to_dict(),
            "metadata": {
                **result.metadata,
                "environment_report_path": environment_report_relative_path,
            },
        }
    )

    summary = {
        "construction_unit_name": "minimal_diffusion_latent_injection",
        "probe_decision": result.probe_decision,
        "unsupported_reason": result.unsupported_reason,
        "probe_id": result.probe_id,
        "model_family": result.model_family,
        "model_id": result.model_id,
        "model_priority": result.model_priority,
        "trajectory_entry_count": result.trajectory_entry_count,
        "image_digest": result.image_digest,
        "environment_report_path": environment_report_relative_path,
        "metadata": result.metadata,
    }
    generation_path.write_text(stable_json_text(result.to_dict()), encoding="utf-8")
    trajectory_path.write_text("".join(json_line(record.to_dict()) for record in trajectory_records), encoding="utf-8")
    environment_path.write_text(stable_json_text(environment_report), encoding="utf-8")
    summary_path.write_text(stable_json_text(summary), encoding="utf-8")

    output_paths = tuple(
        path.relative_to(root_path).as_posix()
        for path in (generation_path, trajectory_path, summary_path, environment_path, manifest_path)
    )
    manifest = build_artifact_manifest(
        artifact_id=f"{config.model_family}_real_sd_runtime_manifest",
        artifact_type="local_manifest",
        input_paths=(
            "experiments/runners/sd_runtime_cold_start.py",
        ),
        output_paths=output_paths,
        config={
            "model_family": config.model_family,
            "model_id": config.model_id,
            "model_priority": config.model_priority,
            "prompt_digest": result.prompt_digest,
            "seed": config.seed,
            "trajectory_entry_count": result.trajectory_entry_count,
            "environment_report_path": environment_report_relative_path,
        },
        code_version=resolve_code_version(root_path),
        rebuild_command="调用 experiments.runners.sd_runtime_cold_start",
        metadata={
            "construction_unit_name": "minimal_diffusion_latent_injection",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "probe_decision": result.probe_decision,
            "unsupported_reason": result.unsupported_reason,
        },
    ).to_dict()
    manifest_path.write_text(stable_json_text(manifest), encoding="utf-8")
    return summary


def build_default_configs(model_selection: str = "both") -> tuple[RealRuntimeConfig, ...]:
    """根据环境变量构造 Colab 默认模型运行计划。"""
    common = {
        "prompt": os.environ.get("SLM_WM_PROMPT", "a high quality photograph of a glass sphere on a wooden table"),
        "negative_prompt": os.environ.get("SLM_WM_NEGATIVE_PROMPT", "low quality, blurry"),
        "seed": int(os.environ.get("SLM_WM_SEED", "1703")),
        "width": int(os.environ.get("SLM_WM_WIDTH", "512")),
        "height": int(os.environ.get("SLM_WM_HEIGHT", "512")),
        "inference_steps": int(os.environ.get("SLM_WM_INFERENCE_STEPS", "28")),
        "guidance_scale": float(os.environ.get("SLM_WM_GUIDANCE_SCALE", "4.5")),
        "output_dir": DEFAULT_OUTPUT_DIR,
    }
    configs = {
        PRIMARY_MODEL_FAMILY: RealRuntimeConfig(
            model_family=PRIMARY_MODEL_FAMILY,
            model_id=os.environ.get("SLM_WM_SD35_MODEL_ID", "stabilityai/stable-diffusion-3.5-medium"),
            model_priority="primary",
            **common,
        ),
        FALLBACK_MODEL_FAMILY: RealRuntimeConfig(
            model_family=FALLBACK_MODEL_FAMILY,
            model_id=os.environ.get("SLM_WM_SD3_MODEL_ID", "stabilityai/stable-diffusion-3-medium-diffusers"),
            model_priority="compatibility_fallback",
            **common,
        ),
    }
    normalized = model_selection.strip().lower()
    if normalized == "both":
        return (configs[PRIMARY_MODEL_FAMILY], configs[FALLBACK_MODEL_FAMILY])
    if normalized == FALLBACK_MODEL_FAMILY:
        return (configs[FALLBACK_MODEL_FAMILY],)
    return (configs[PRIMARY_MODEL_FAMILY],)


def run_default_model_plan(root: str | Path = ".", model_selection: str = "both") -> tuple[dict[str, Any], ...]:
    """运行默认模型计划, both 模式下同时运行主模型和兼容模型。"""
    summaries: list[dict[str, Any]] = []
    configs = build_default_configs(model_selection=model_selection)
    for config in configs:
        summaries.append(write_single_model_outputs(config=config, root=root))
    if model_selection.strip().lower() == "auto" and summaries and summaries[0]["probe_decision"] != "pass":
        fallback_config = build_default_configs(model_selection=FALLBACK_MODEL_FAMILY)[0]
        summaries.append(write_single_model_outputs(config=fallback_config, root=root))
    return tuple(summaries)


def collect_archive_entries(source_dir: Path, archive_path: Path) -> tuple[Path, ...]:
    """收集需要进入压缩包的受治理产物, 避免把历史压缩包递归打包。"""
    entries: list[Path] = []
    for path in sorted(source_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.resolve() == archive_path.resolve() or path.suffix.lower() == ".zip":
            continue
        entries.append(path)
    return tuple(entries)


def package_probe_outputs(
    root: str | Path = ".",
    output_dir: str = DEFAULT_OUTPUT_DIR,
    drive_output_dir: str = DEFAULT_DRIVE_OUTPUT_DIR,
    archive_name: str = "real_sd_runtime_probe_package.zip",
) -> ProbeArchiveRecord:
    """把真实 SD runtime 产物打包为 zip, 并镜像到 Google Drive 目录。"""
    root_path = Path(root).resolve()
    source_dir = (root_path / output_dir).resolve()
    archive_path = source_dir / archive_name
    summary_path = source_dir / "real_sd_runtime_probe_archive_summary.json"
    manifest_path = source_dir / "real_sd_runtime_probe_archive_manifest.local.json"

    if not source_dir.exists():
        raise FileNotFoundError(f"probe_output_dir_missing: {source_dir}")

    entries = collect_archive_entries(source_dir=source_dir, archive_path=archive_path)
    with ZipFile(archive_path, mode="w", compression=ZIP_DEFLATED) as archive:
        for entry in entries:
            archive.write(entry, entry.relative_to(source_dir).as_posix())

    drive_dir = Path(drive_output_dir).expanduser()
    drive_dir.mkdir(parents=True, exist_ok=True)
    mirrored_path = drive_dir / archive_name
    shutil.copy2(archive_path, mirrored_path)

    record = ProbeArchiveRecord(
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
        artifact_id="real_sd_runtime_probe_archive_manifest",
        artifact_type="local_manifest",
        input_paths=(
            "experiments/runners/sd_runtime_cold_start.py",
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
        rebuild_command="调用 experiments.runners.sd_runtime_cold_start",
        metadata={
            "construction_unit_name": "minimal_diffusion_latent_injection",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "archive_digest": record.archive_digest,
            "drive_archive_digest": record.drive_archive_digest,
        },
    ).to_dict()
    manifest_path.write_text(stable_json_text(manifest), encoding="utf-8")
    return record
