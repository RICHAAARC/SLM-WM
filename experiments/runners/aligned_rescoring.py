"""真实 aligned rescoring 服务器与 Notebook 共用 runner。"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from functools import lru_cache
import json
import os
from pathlib import Path
import shutil
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

from main.analysis.artifact_manifest import build_artifact_manifest
from main.core.digest import build_stable_digest
from experiments.protocol.paper_run_config import (
    DEFAULT_CONTENT_BASIS_RANK,
    DEFAULT_CONTENT_VECTOR_WIDTH,
    build_paper_run_config,
    resolve_count_from_environment,
)
from main.methods.detection.scores import compute_unified_content_score
from experiments.runners.attention_latent_injection import (
    build_content_update_lookup,
    compose_runtime_watermark_tensor,
    load_content_records_by_prompt,
    materialize_geometry_package,
    prepare_attention_method_outputs,
    select_runtime_content_update,
)
from experiments.runtime.diffusion.sd3_pipeline_runtime import (
    compute_image_quality_metrics,
    import_runtime_dependencies,
    load_pipeline,
    tensor_norm,
)
from experiments.runtime.repository_environment import (
    build_runtime_environment_report,
    file_digest,
    resolve_code_version,
    tensor_digest,
)

DEFAULT_OUTPUT_DIR = "outputs/aligned_rescoring"
DEFAULT_METHOD_OUTPUT_DIR = "outputs/attention_latent_update"
DEFAULT_DRIVE_OUTPUT_DIR = ""
DEFAULT_GEOMETRY_DRIVE_DIR = ""
PRIMARY_MODEL_FAMILY = "sd35"
PRIMARY_MODEL_ID = "stabilityai/stable-diffusion-3.5-medium"
DEFAULT_CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
DEFAULT_LPIPS_NETWORK = "alex"
PAIR_PERCEPTUAL_DEPENDENCY_INSTALL_COMMAND = "%pip install -q --upgrade lpips"
PACKAGE_EXTRA_PATHS = (
    "experiments/runners/aligned_rescoring.py",
    "outputs/prompt_event_protocol/prompt_records.jsonl",
    "outputs/content_carriers/content_detection_records.jsonl",
    "outputs/content_carriers/manifest.local.json",
    "outputs/attention_latent_update/attention_carrier_records.jsonl",
    "outputs/attention_latent_update/manifest.local.json",
)


@dataclass(frozen=True)
class AlignedRescoringConfig:
    """描述真实 aligned rescoring 运行配置。"""

    model_family: str
    model_id: str
    seed: int
    width: int
    height: int
    inference_steps: int
    guidance_scale: float
    attention_runtime_strength: float
    injection_step_indices: tuple[int, ...]
    output_dir: str = DEFAULT_OUTPUT_DIR
    method_output_dir: str = DEFAULT_METHOD_OUTPUT_DIR
    geometry_drive_dir: str = field(default_factory=lambda: build_paper_run_config(".").drive_dir("attention_geometry"))
    attention_geometry_package_path: str = ""
    max_subspace_records: int = 700
    max_rescore_carriers: int = 700
    content_vector_width: int = DEFAULT_CONTENT_VECTOR_WIDTH
    content_basis_rank: int = DEFAULT_CONTENT_BASIS_RANK
    negative_prompt: str = "low quality, blurry"
    device_name: str = "cuda"
    torch_dtype: str = "float16"
    hf_token_env: str = "HF_TOKEN"
    enable_pair_perceptual_metrics: bool = True
    require_pair_perceptual_metrics: bool = True
    clip_model_id: str = DEFAULT_CLIP_MODEL_ID
    lpips_network: str = DEFAULT_LPIPS_NETWORK
    perceptual_metric_device_name: str = "cpu"
    enable_pipeline_progress_bar: bool = False
    enable_carrier_progress_bar: bool = True

    def __post_init__(self) -> None:
        """集中校验配置边界, 避免运行路径重复构造错误信息。"""
        positive_fields = {
            "width": self.width,
            "height": self.height,
            "inference_steps": self.inference_steps,
            "max_subspace_records": self.max_subspace_records,
            "max_rescore_carriers": self.max_rescore_carriers,
            "content_vector_width": self.content_vector_width,
            "content_basis_rank": self.content_basis_rank,
        }
        invalid_fields = {name: value for name, value in positive_fields.items() if value <= 0}
        if invalid_fields:
            raise ValueError(f"配置正整数边界无效: {invalid_fields}")
        if self.guidance_scale <= 0.0 or self.attention_runtime_strength < 0.0:
            raise ValueError("guidance_scale 必须为正数, attention_runtime_strength 不得为负数")
        if any(index < 0 or index >= self.inference_steps for index in self.injection_step_indices):
            raise ValueError("injection_step_indices 必须位于采样步数边界内")
        if self.require_pair_perceptual_metrics and not self.enable_pair_perceptual_metrics:
            raise ValueError("要求 pair perceptual metrics 时必须启用指标计算")
        if not self.clip_model_id.strip():
            raise ValueError("clip_model_id 不能为空")
        if self.lpips_network not in {"alex", "vgg", "squeeze"}:
            raise ValueError("lpips_network 必须为 alex、vgg 或 squeeze")


@dataclass(frozen=True)
class AlignedRescoringRecord:
    """记录真实 latent 对齐前后的内容重打分。"""

    aligned_rescoring_record_id: str
    content_detection_record_id: str
    prompt_id: str
    prompt_digest: str
    split: str
    sample_role: str
    carrier_id: str
    attention_graph_id: str
    capture_id: str
    content_update_digest: str
    content_chain_digest: str
    raw_content_score: float
    aligned_content_score: float
    real_raw_content_score: float
    real_aligned_content_score: float
    real_rescoring_score_gain: float
    real_lf_score_before: float
    real_lf_score_after: float
    real_hf_score_before: float
    real_hf_score_after: float
    real_combined_score_before: float
    real_combined_score_after: float
    real_lf_hf_fusion_score_before: float
    real_lf_hf_fusion_score_after: float
    latent_digest_before: str
    latent_digest_after: str
    latent_projection_digest_before: str
    latent_projection_digest_after: str
    latent_projection_values_before: tuple[float, ...]
    latent_projection_values_after: tuple[float, ...]
    aligned_rescoring_ready: bool
    metric_status: str
    full_method_claim_ready: bool
    supports_paper_claim: bool
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """转为 JSON 兼容字典。"""
        return asdict(self)


@dataclass(frozen=True)
class AlignedRescoringResult:
    """保存真实 aligned rescoring 运行摘要。"""

    run_id: str
    model_family: str
    model_id: str
    run_decision: str
    unsupported_reason: str
    aligned_rescoring_record_count: int
    real_aligned_rescore_count: int
    selected_attention_carrier_count: int
    image_quality_metrics_ready: bool
    perceptual_metrics_ready: bool
    full_method_claim_ready: bool
    output_records_path: str
    quality_metrics_path: str
    method_manifest_path: str
    attention_geometry_package_path: str
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """转为 JSON 兼容字典。"""
        return asdict(self)


@dataclass(frozen=True)
class AlignedRescoringArchiveRecord:
    """记录真实 aligned rescoring 压缩包与 Drive 镜像。"""

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


def parse_bool_environment(name: str, default: bool) -> bool:
    """解析布尔环境变量, 把配置合法性收敛到配置加载边界。"""
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    normalized = raw_value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"{name} 必须使用布尔值文本")


def configure_pipeline_progress_bar(pipeline: Any, config: AlignedRescoringConfig) -> str:
    """配置 Diffusers 单次推理进度条, 使长流程默认只展示 carrier 总进度。"""
    if not hasattr(pipeline, "set_progress_bar_config"):
        return "pipeline_progress_bar_config_unavailable"
    pipeline.set_progress_bar_config(disable=not config.enable_pipeline_progress_bar)
    return "pipeline_progress_bar_enabled" if config.enable_pipeline_progress_bar else "pipeline_progress_bar_disabled"


def iterate_carriers_with_progress(
    carrier_records: tuple[dict[str, Any], ...],
    config: AlignedRescoringConfig,
) -> Any:
    """按 carrier 维度提供总进度, 避免每次 diffusion 调用刷屏。"""
    enumerated_records = enumerate(carrier_records)
    if not config.enable_carrier_progress_bar:
        return enumerated_records
    try:
        from tqdm.auto import tqdm
    except Exception:
        return enumerated_records
    return tqdm(enumerated_records, total=len(carrier_records), desc="aligned rescoring carriers")


def read_jsonl(path: Path) -> tuple[dict[str, Any], ...]:
    """读取 JSONL 文件。"""
    return tuple(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def build_run_id(config: AlignedRescoringConfig, carrier_ids: tuple[str, ...]) -> str:
    """根据真实运行配置生成稳定 run id。"""
    return build_stable_digest(
        {
            "model_family": config.model_family,
            "model_id": config.model_id,
            "seed": config.seed,
            "width": config.width,
            "height": config.height,
            "inference_steps": config.inference_steps,
            "guidance_scale": config.guidance_scale,
            "attention_runtime_strength": config.attention_runtime_strength,
            "injection_step_indices": config.injection_step_indices,
            "content_vector_width": config.content_vector_width,
            "content_basis_rank": config.content_basis_rank,
            "carrier_ids": carrier_ids,
            "enable_pair_perceptual_metrics": config.enable_pair_perceptual_metrics,
            "require_pair_perceptual_metrics": config.require_pair_perceptual_metrics,
            "clip_model_id": config.clip_model_id,
            "lpips_network": config.lpips_network,
        }
    )


def prompt_digest(prompt_text: str, config: AlignedRescoringConfig) -> str:
    """生成 prompt 与运行配置摘要。"""
    return build_stable_digest(
        {
            "prompt": prompt_text,
            "negative_prompt": config.negative_prompt,
            "model_id": config.model_id,
            "seed": config.seed,
        }
    )


def select_active_carriers(carrier_records_path: Path, max_count: int) -> tuple[dict[str, Any], ...]:
    """选择可执行 active update 的 attention carrier。"""
    records = read_jsonl(carrier_records_path)
    active_records = [
        record
        for record in records
        if record.get("fallback_mode") == "active_update" and bool(record.get("attention_update_stable", False))
    ]
    ordered = sorted(active_records, key=lambda record: (-float(record.get("relation_loss_delta", 0.0)), record["carrier_id"]))
    if not ordered:
        raise RuntimeError("active_attention_carrier_missing")
    return tuple(ordered[:max_count])


def prompt_text_by_id(root_path: Path) -> dict[str, str]:
    """读取 prompt_id 到 prompt_text 的映射。"""
    records_path = root_path / "outputs" / "prompt_event_protocol" / "prompt_records.jsonl"
    return {record["prompt_id"]: record["prompt_text"] for record in read_jsonl(records_path)}


def normalized_slot_projection_from_values(flattened_values: tuple[float, ...], value_count: int) -> tuple[float, ...]:
    """把平铺 latent 数值按重复槽位聚合为单位检测向量。"""
    if not flattened_values:
        raise RuntimeError("latent_tensor_empty")
    if value_count <= 1:
        selected = flattened_values[:1]
    else:
        selected = tuple(
            sum(flattened_values[index::value_count]) / len(flattened_values[index::value_count])
            for index in range(value_count)
            if flattened_values[index::value_count]
        )
    mean_value = sum(selected) / len(selected)
    centered = tuple(value - mean_value for value in selected)
    norm = sum(value * value for value in centered) ** 0.5
    if norm <= 1e-6:
        return tuple(0.0 for _ in centered)
    return tuple(value / norm for value in centered)


def latent_projection_values(latents: Any, value_count: int) -> tuple[float, ...]:
    """把真实 latent tensor 按 carrier 重复槽位池化为内容检测向量。"""
    flattened = tuple(float(value) for value in latents.detach().float().reshape(-1).cpu().tolist())
    return normalized_slot_projection_from_values(flattened, value_count)


def single_content_update_width(
    content_records: tuple[dict[str, Any], ...],
    content_updates: dict[str, Any],
) -> int:
    """确认同一重打分批次中的 content update 宽度一致。"""

    widths = {
        record["content_detection_record_id"]: len(content_updates[record["content_detection_record_id"]].combined_update_values)
        for record in content_records
    }
    unique_widths = sorted(set(widths.values()))
    if len(unique_widths) != 1:
        raise RuntimeError(f"content_update_width_mismatch:{widths}")
    return unique_widths[0]


def build_rescoring_records(
    config: AlignedRescoringConfig,
    carrier_record: dict[str, Any],
    prompt_text: str,
    content_records: tuple[dict[str, Any], ...],
    content_updates: dict[str, Any],
    latent_before: Any,
    latent_after: Any,
    update_count: int,
    runtime_content_record: dict[str, Any] | None = None,
    runtime_content_update: Any | None = None,
    latent_boundary_metadata: dict[str, Any] | None = None,
) -> tuple[AlignedRescoringRecord, ...]:
    """根据真实 latent 对齐前后状态构造内容重打分记录。"""
    if not content_records:
        return ()
    content_update_width = single_content_update_width(content_records, content_updates)
    values_before = latent_projection_values(latent_before, content_update_width)
    values_after = latent_projection_values(latent_after, content_update_width)
    digest_before = build_stable_digest([round(value, 12) for value in values_before])
    digest_after = build_stable_digest([round(value, 12) for value in values_after])
    records: list[AlignedRescoringRecord] = []
    current_prompt_digest = prompt_digest(prompt_text, config)
    for content_record in content_records:
        update = content_updates[content_record["content_detection_record_id"]]
        score_before = compute_unified_content_score(values_before, update)
        score_after = compute_unified_content_score(values_after, update)
        payload = {
            "content_detection_record_id": content_record["content_detection_record_id"],
            "carrier_id": carrier_record["carrier_id"],
            "latent_projection_digest_before": digest_before,
            "latent_projection_digest_after": digest_after,
            "score_digest_after": score_after.score_digest,
        }
        record_digest = build_stable_digest(payload)
        records.append(
            AlignedRescoringRecord(
                aligned_rescoring_record_id=f"aligned_rescoring_{record_digest[:16]}",
                content_detection_record_id=content_record["content_detection_record_id"],
                prompt_id=content_record["prompt_id"],
                prompt_digest=current_prompt_digest,
                split=content_record["split"],
                sample_role=str(content_record.get("metadata", {}).get("sample_role", "unknown")),
                carrier_id=carrier_record["carrier_id"],
                attention_graph_id=carrier_record["attention_graph_id"],
                capture_id=carrier_record["capture_id"],
                content_update_digest=update.content_update_digest,
                content_chain_digest=update.content_chain_digest,
                raw_content_score=float(content_record["content_score"]),
                aligned_content_score=score_after.content_score,
                real_raw_content_score=score_before.content_score,
                real_aligned_content_score=score_after.content_score,
                real_rescoring_score_gain=score_after.content_score - score_before.content_score,
                real_lf_score_before=score_before.lf_score,
                real_lf_score_after=score_after.lf_score,
                real_hf_score_before=score_before.hf_score,
                real_hf_score_after=score_after.hf_score,
                real_combined_score_before=score_before.combined_score,
                real_combined_score_after=score_after.combined_score,
                real_lf_hf_fusion_score_before=score_before.lf_hf_fusion_score,
                real_lf_hf_fusion_score_after=score_after.lf_hf_fusion_score,
                latent_digest_before=tensor_digest(latent_before.detach().float().cpu()),
                latent_digest_after=tensor_digest(latent_after.detach().float().cpu()),
                latent_projection_digest_before=digest_before,
                latent_projection_digest_after=digest_after,
                latent_projection_values_before=values_before,
                latent_projection_values_after=values_after,
                aligned_rescoring_ready=True,
                metric_status="measured_from_real_latent_projection",
                full_method_claim_ready=False,
                supports_paper_claim=False,
                metadata={
                    "model_family": config.model_family,
                    "model_id": config.model_id,
                    "attention_runtime_strength": config.attention_runtime_strength,
                    "latent_update_count": update_count,
                    "score_source": "real_sd_latent_projection",
                    "latent_projection_mode": "periodic_slot_pooled_content_carrier",
                    "content_update_width": content_update_width,
                    "content_update_width_consistent": True,
                    "detector_input_access_mode": "generation_latent_trace_required",
                    "blind_image_detector": False,
                    "latent_trace_detection_claim_boundary": "not_blind_image_watermark_detection",
                    "baseline_fairness_boundary": "external_baseline_comparison_requires_matching_detector_access",
                    "content_vector_width": config.content_vector_width,
                    "content_basis_rank": config.content_basis_rank,
                    "formal_score_source": score_after.metadata.get(
                        "formal_score_source",
                        "lf_hf_consistency_guarded_combined_correlation",
                    ),
                    "runtime_content_update_digest": ""
                    if runtime_content_update is None
                    else runtime_content_update.content_update_digest,
                    "runtime_content_detection_record_id": ""
                    if runtime_content_record is None
                    else runtime_content_record.get("content_detection_record_id", ""),
                    "runtime_content_sample_role": ""
                    if runtime_content_record is None
                    else runtime_content_record.get("metadata", {}).get("sample_role", ""),
                    "supports_paper_claim": False,
                    **(latent_boundary_metadata or {}),
                },
            )
        )
    return tuple(records)


def unsupported_perceptual_metric_rows(status: str) -> dict[str, Any]:
    """返回统一的未测量感知指标状态。"""
    return {
        "lpips": "unsupported",
        "lpips_status": status,
        "lpips_error_type": "",
        "lpips_error_message": "",
        "clip_score": "unsupported",
        "clip_score_clean": "unsupported",
        "clip_score_aligned": "unsupported",
        "clip_score_delta": "unsupported",
        "clip_score_status": status,
        "clip_score_error_type": "",
        "clip_score_error_message": "",
        "fid": "unsupported",
        "fid_status": "dataset_level_metric_not_computed_in_pair_run",
        "kid": "unsupported",
        "kid_status": "dataset_level_metric_not_computed_in_pair_run",
        "perceptual_metrics_ready": False,
    }


def metric_error_message(error: Exception, limit: int = 240) -> str:
    """压缩指标异常信息, 避免 CSV 中写入过长 traceback。"""
    return str(error).replace("\n", " ")[:limit]


def image_to_metric_tensor(image: Any, device: Any) -> Any:
    """把 PIL 图像转为 LPIPS 需要的 [-1, 1] BCHW tensor。"""
    import torch

    rgb_image = image.convert("RGB")
    width, height = rgb_image.size
    image_bytes = bytearray(rgb_image.tobytes())
    image_tensor = torch.frombuffer(image_bytes, dtype=torch.uint8).reshape(height, width, 3).float()
    image_tensor = image_tensor / 127.5 - 1.0
    return image_tensor.permute(2, 0, 1).unsqueeze(0).to(device)


@lru_cache(maxsize=8)
def load_lpips_metric_model(lpips_network: str, perceptual_metric_device_name: str) -> Any:
    """加载并缓存 LPIPS 模型, 避免每个 carrier 重复初始化感知损失网络。"""
    import lpips

    _, torch, _, _ = import_runtime_dependencies()
    device = torch.device(perceptual_metric_device_name)
    metric_model = lpips.LPIPS(net=lpips_network).to(device)
    metric_model.eval()
    return metric_model


def compute_lpips_metric(clean_image: Any, aligned_image: Any, config: AlignedRescoringConfig) -> dict[str, Any]:
    """计算成对图像 LPIPS, 不可用时返回可审计 status。"""
    if not config.enable_pair_perceptual_metrics:
        return {"lpips": "unsupported", "lpips_status": "disabled", "lpips_error_type": "", "lpips_error_message": ""}
    try:
        import lpips
        _, torch, _, _ = import_runtime_dependencies()
    except ModuleNotFoundError as error:
        return {
            "lpips": "unsupported",
            "lpips_status": "optional_dependency_not_installed",
            "lpips_error_type": type(error).__name__,
            "lpips_error_message": metric_error_message(error),
        }
    except Exception as error:
        return {
            "lpips": "unsupported",
            "lpips_status": "optional_dependency_import_error",
            "lpips_error_type": type(error).__name__,
            "lpips_error_message": metric_error_message(error),
        }
    try:
        device = torch.device(config.perceptual_metric_device_name)
        if device.type == "cuda" and not torch.cuda.is_available():
            return {
                "lpips": "unsupported",
                "lpips_status": "gpu_unavailable",
                "lpips_error_type": "",
                "lpips_error_message": "",
            }
        metric_model = load_lpips_metric_model(config.lpips_network, config.perceptual_metric_device_name)
        clean_tensor = image_to_metric_tensor(clean_image, device)
        aligned_tensor = image_to_metric_tensor(aligned_image, device)
        with torch.no_grad():
            metric_value = metric_model(clean_tensor, aligned_tensor)
        return {
            "lpips": float(metric_value.detach().float().cpu().reshape(-1)[0].item()),
            "lpips_status": "measured",
            "lpips_error_type": "",
            "lpips_error_message": "",
        }
    except Exception as error:
        return {
            "lpips": "unsupported",
            "lpips_status": "metric_runtime_error",
            "lpips_error_type": type(error).__name__,
            "lpips_error_message": metric_error_message(error),
        }


def move_batch_to_device(batch: dict[str, Any], device: Any) -> dict[str, Any]:
    """把 transformers processor 生成的 batch 移动到目标设备。"""
    return {name: value.to(device) if hasattr(value, "to") else value for name, value in batch.items()}


def unsupported_clip_metric(status: str, error: Exception | None = None) -> dict[str, Any]:
    """构造统一的 CLIP 指标不可用状态。"""
    return {
        "clip_score": "unsupported",
        "clip_score_clean": "unsupported",
        "clip_score_aligned": "unsupported",
        "clip_score_delta": "unsupported",
        "clip_score_status": status,
        "clip_score_error_type": "" if error is None else type(error).__name__,
        "clip_score_error_message": "" if error is None else metric_error_message(error),
    }


def measured_clip_metric(clean_score: float, aligned_score: float) -> dict[str, Any]:
    """构造统一的 CLIP 实测指标行。"""
    return {
        "clip_score": aligned_score,
        "clip_score_clean": clean_score,
        "clip_score_aligned": aligned_score,
        "clip_score_delta": aligned_score - clean_score,
        "clip_score_status": "measured",
        "clip_score_error_type": "",
        "clip_score_error_message": "",
    }


def read_output_value(outputs: Any, field_name: str) -> Any:
    """兼容 transformers 输出对象和 dict 输出。"""
    if hasattr(outputs, field_name):
        return getattr(outputs, field_name)
    if isinstance(outputs, dict):
        return outputs.get(field_name)
    return None


def clip_scores_from_features(image_features: Any, text_features: Any) -> tuple[float, float]:
    """由 CLIP image/text embeddings 计算 clean 和 aligned 的余弦相似度。"""
    image_features = image_features.float()
    text_features = text_features.float()
    image_features = image_features / image_features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    scores = (image_features @ text_features.T).reshape(-1)
    return float(scores[0].detach().cpu().item()), float(scores[1].detach().cpu().item())


def compute_clip_scores_with_feature_api(model: Any, image_inputs: dict[str, Any], text_inputs: dict[str, Any]) -> tuple[float, float]:
    """使用 CLIPModel embedding API 计算图文相似度。"""
    image_kwargs = {name: image_inputs[name] for name in ("pixel_values",) if name in image_inputs}
    text_kwargs = {name: text_inputs[name] for name in ("input_ids", "attention_mask", "position_ids") if name in text_inputs}
    image_features = model.get_image_features(**image_kwargs)
    text_features = model.get_text_features(**text_kwargs)
    return clip_scores_from_features(image_features, text_features)


def compute_clip_scores_with_forward_api(
    model: Any,
    processor: Any,
    clean_image: Any,
    aligned_image: Any,
    prompt_text: str,
    device: Any,
) -> tuple[float, float]:
    """使用 CLIPModel forward 输出计算图文相似度, 兼容新版或裁剪版 API。"""
    joint_inputs = processor(
        text=[prompt_text],
        images=[clean_image.convert("RGB"), aligned_image.convert("RGB")],
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    joint_inputs = move_batch_to_device(joint_inputs, device)
    outputs = model(**joint_inputs)
    image_features = read_output_value(outputs, "image_embeds")
    text_features = read_output_value(outputs, "text_embeds")
    if image_features is not None and text_features is not None:
        return clip_scores_from_features(image_features, text_features)
    logits_per_image = read_output_value(outputs, "logits_per_image")
    if logits_per_image is None:
        raise AttributeError("clip_forward_output_missing_image_text_scores")
    scores = logits_per_image.float().reshape(-1)
    return float(scores[0].detach().cpu().item()), float(scores[1].detach().cpu().item())


@lru_cache(maxsize=8)
def load_clip_metric_objects(
    clip_model_id: str,
    perceptual_metric_device_name: str,
    hf_token: str,
) -> tuple[Any, Any]:
    """加载并缓存 CLIP processor / model, 让成对图文分数计算复用同一模型对象。"""
    _, torch, _, _ = import_runtime_dependencies()
    from transformers import CLIPModel, CLIPProcessor

    device = torch.device(perceptual_metric_device_name)
    model_kwargs = {"token": hf_token} if hf_token else {}
    processor = CLIPProcessor.from_pretrained(clip_model_id, **model_kwargs)
    model = CLIPModel.from_pretrained(clip_model_id, **model_kwargs).to(device)
    model.eval()
    return processor, model


def compute_clip_pair_metrics(
    clean_image: Any,
    aligned_image: Any,
    prompt_text: str,
    config: AlignedRescoringConfig,
) -> dict[str, Any]:
    """计算 clean / aligned 图像相对 prompt 的 CLIP score。"""
    if not config.enable_pair_perceptual_metrics:
        return unsupported_clip_metric("disabled")
    try:
        _, torch, _, _ = import_runtime_dependencies()
        from transformers import CLIPModel, CLIPProcessor
    except ModuleNotFoundError as error:
        return unsupported_clip_metric("optional_dependency_not_installed", error)
    except Exception as error:
        return unsupported_clip_metric("optional_dependency_import_error", error)
    try:
        device = torch.device(config.perceptual_metric_device_name)
        if device.type == "cuda" and not torch.cuda.is_available():
            return unsupported_clip_metric("gpu_unavailable")
        token = os.environ.get(config.hf_token_env) or ""
        processor, model = load_clip_metric_objects(
            config.clip_model_id,
            config.perceptual_metric_device_name,
            token,
        )
        image_inputs = processor(
            images=[clean_image.convert("RGB"), aligned_image.convert("RGB")],
            return_tensors="pt",
        )
        text_inputs = processor(text=[prompt_text], return_tensors="pt", padding=True, truncation=True)
        image_inputs = move_batch_to_device(image_inputs, device)
        text_inputs = move_batch_to_device(text_inputs, device)
        with torch.no_grad():
            try:
                clean_score, aligned_score = compute_clip_scores_with_feature_api(model, image_inputs, text_inputs)
            except AttributeError:
                clean_score, aligned_score = compute_clip_scores_with_forward_api(
                    model,
                    processor,
                    clean_image,
                    aligned_image,
                    prompt_text,
                    device,
                )
        return measured_clip_metric(clean_score, aligned_score)
    except Exception as error:
        return unsupported_clip_metric("metric_runtime_error", error)


def compute_pair_perceptual_metrics(
    clean_image: Any,
    aligned_image: Any,
    prompt_text: str,
    config: AlignedRescoringConfig,
) -> dict[str, Any]:
    """计算 paired LPIPS 与 CLIP 指标, FID / KID 保持数据集级边界。"""
    if not config.enable_pair_perceptual_metrics:
        return unsupported_perceptual_metric_rows("disabled")
    lpips_row = compute_lpips_metric(clean_image, aligned_image, config)
    clip_row = compute_clip_pair_metrics(clean_image, aligned_image, prompt_text, config)
    pair_ready = lpips_row.get("lpips_status") == "measured" and clip_row.get("clip_score_status") == "measured"
    return {
        **unsupported_perceptual_metric_rows("not_measured"),
        **lpips_row,
        **clip_row,
        "perceptual_metrics_ready": pair_ready,
    }


def pair_metric_status_summary(rows: list[dict[str, Any]]) -> str:
    """汇总 LPIPS / CLIP 状态, 使失败摘要能直接定位缺失指标。"""
    if not rows:
        return "quality_rows_missing"
    statuses = []
    for row in rows:
        statuses.append(
            "lpips="
            + str(row.get("lpips_status", "missing"))
            + ";clip_score="
            + str(row.get("clip_score_status", "missing"))
        )
    return "|".join(sorted(set(statuses)))


def update_aligned_latent_snapshot(
    snapshot: dict[str, Any],
    *,
    latents: Any,
    aligned: Any,
    update: Any,
    trajectory_index: int,
    timestep: Any,
    runtime_content_record: dict[str, Any],
    runtime_content_update: Any,
    carrier_metadata: dict[str, Any],
) -> None:
    """维护 aligned rescoring 的 raw/aligned latent 边界。

    该函数属于真实 Colab workflow 的关键边界修复: `latent_before` 只在第一次
    注入前记录一次, 表示尚未被任何 watermark update 污染的 clean latent;
    `latent_after` 则随每次注入更新, 最终表示所有注入完成后的 aligned latent。
    这样可以避免旧实现把最后一次注入前的 latent 误当作 raw latent。
    """

    current_trajectory_index = int(trajectory_index)
    current_timestep = float(timestep)
    if "latent_before" not in snapshot:
        snapshot.update(
            {
                "latent_before": latents.detach().clone(),
                "latent_projection_boundary_before": "first_clean_latent_before_any_injection",
                "first_injection_trajectory_index": current_trajectory_index,
                "first_injection_timestep": current_timestep,
                "latent_norm_before": tensor_norm(latents),
            }
        )
    snapshot.update(
        {
            "latent_after": aligned.detach().clone(),
            "latent_projection_boundary_after": "final_aligned_latent_after_all_injections",
            "trajectory_index": current_trajectory_index,
            "timestep": current_timestep,
            "final_injection_trajectory_index": current_trajectory_index,
            "final_injection_timestep": current_timestep,
            "update_norm": tensor_norm(update),
            "latent_norm_after": tensor_norm(aligned),
            "runtime_content_update_digest": runtime_content_update.content_update_digest,
            "runtime_content_detection_record_id": runtime_content_record.get("content_detection_record_id", ""),
            "carrier_metadata": carrier_metadata,
        }
    )


def latent_boundary_metadata_from_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    """从 runtime snapshot 中抽取会随 records 落盘的 latent 边界元数据。"""

    return {
        "latent_projection_boundary_before": snapshot["latent_projection_boundary_before"],
        "latent_projection_boundary_after": snapshot["latent_projection_boundary_after"],
        "first_injection_trajectory_index": snapshot["first_injection_trajectory_index"],
        "first_injection_timestep": snapshot["first_injection_timestep"],
        "final_injection_trajectory_index": snapshot["final_injection_trajectory_index"],
        "final_injection_timestep": snapshot["final_injection_timestep"],
    }


def run_carrier_rescoring(
    pipeline: Any,
    config: AlignedRescoringConfig,
    carrier_record: dict[str, Any],
    prompt_text: str,
    content_records: tuple[dict[str, Any], ...],
    content_updates: dict[str, Any],
    run_index: int,
) -> tuple[tuple[AlignedRescoringRecord, ...], dict[str, Any], Any, Any]:
    """对单个 attention carrier 执行真实 latent 对齐和内容重打分。"""
    _, torch, _, _ = import_runtime_dependencies()
    seed = config.seed + run_index
    common_kwargs = {
        "prompt": prompt_text,
        "negative_prompt": config.negative_prompt,
        "width": config.width,
        "height": config.height,
        "num_inference_steps": config.inference_steps,
        "guidance_scale": config.guidance_scale,
        "output_type": "pil",
    }
    clean_generator = torch.Generator(device=config.device_name).manual_seed(seed)
    aligned_generator = torch.Generator(device=config.device_name).manual_seed(seed)
    clean_output = pipeline(generator=clean_generator, **common_kwargs)
    update_steps = set(config.injection_step_indices)
    latent_snapshot: dict[str, Any] = {}
    update_count = 0
    runtime_content_record, runtime_content_update = select_runtime_content_update(content_records, content_updates)

    def align_latents(pipe: Any, trajectory_index: int, timestep: Any, callback_kwargs: dict[str, Any]) -> dict[str, Any]:
        nonlocal update_count
        latents = callback_kwargs.get("latents")
        if latents is None or trajectory_index not in update_steps:
            return callback_kwargs
        carrier, carrier_metadata = compose_runtime_watermark_tensor(
            latents=latents,
            carrier_record=carrier_record,
            content_record=runtime_content_record,
            content_update=runtime_content_update,
        )
        update = carrier * config.attention_runtime_strength
        aligned = latents + update
        update_aligned_latent_snapshot(
            latent_snapshot,
            latents=latents,
            aligned=aligned,
            update=update,
            trajectory_index=trajectory_index,
            timestep=timestep,
            runtime_content_record=runtime_content_record,
            runtime_content_update=runtime_content_update,
            carrier_metadata=carrier_metadata,
        )
        update_count += 1
        callback_kwargs["latents"] = aligned
        return callback_kwargs

    aligned_output = pipeline(
        generator=aligned_generator,
        callback_on_step_end=align_latents,
        callback_on_step_end_tensor_inputs=["latents"],
        **common_kwargs,
    )
    if not latent_snapshot:
        raise RuntimeError("aligned_latent_snapshot_missing")
    records = build_rescoring_records(
        config=config,
        carrier_record=carrier_record,
        prompt_text=prompt_text,
        content_records=content_records,
        content_updates=content_updates,
        latent_before=latent_snapshot["latent_before"],
        latent_after=latent_snapshot["latent_after"],
        update_count=update_count,
        runtime_content_record=runtime_content_record,
        runtime_content_update=runtime_content_update,
        latent_boundary_metadata=latent_boundary_metadata_from_snapshot(latent_snapshot),
    )
    quality_metrics = compute_image_quality_metrics(clean_output.images[0], aligned_output.images[0])
    perceptual_metrics = compute_pair_perceptual_metrics(
        clean_image=clean_output.images[0],
        aligned_image=aligned_output.images[0],
        prompt_text=prompt_text,
        config=config,
    )
    quality_row = {
        "carrier_id": carrier_record["carrier_id"],
        "prompt_id": carrier_record.get("metadata", {}).get("prompt_id", ""),
        "attention_graph_id": carrier_record["attention_graph_id"],
        "capture_id": carrier_record["capture_id"],
        "trajectory_index": latent_snapshot["trajectory_index"],
        "timestep": latent_snapshot["timestep"],
        "update_norm": latent_snapshot["update_norm"],
        "latent_norm_before": latent_snapshot["latent_norm_before"],
        "latent_norm_after": latent_snapshot["latent_norm_after"],
        "image_quality_metrics_ready": True,
        **quality_metrics,
        **perceptual_metrics,
    }
    return records, quality_row, clean_output.images[0], aligned_output.images[0]


def build_failure_result(config: AlignedRescoringConfig, error: Exception) -> AlignedRescoringResult:
    """把真实后端不可用状态转为可审计失败摘要。"""
    environment_report = build_runtime_environment_report()
    return AlignedRescoringResult(
        run_id=build_stable_digest({"error": type(error).__name__, "model_id": config.model_id, "seed": config.seed}),
        model_family=config.model_family,
        model_id=config.model_id,
        run_decision="fail",
        unsupported_reason=type(error).__name__,
        aligned_rescoring_record_count=0,
        real_aligned_rescore_count=0,
        selected_attention_carrier_count=0,
        image_quality_metrics_ready=False,
        perceptual_metrics_ready=False,
        full_method_claim_ready=False,
        output_records_path="",
        quality_metrics_path="",
        method_manifest_path="",
        attention_geometry_package_path="",
        metadata={
            "error_message": str(error),
            "runtime_environment": environment_report,
            "enable_pair_perceptual_metrics": config.enable_pair_perceptual_metrics,
            "require_pair_perceptual_metrics": config.require_pair_perceptual_metrics,
            "clip_model_id": config.clip_model_id,
            "lpips_network": config.lpips_network,
            "enable_pipeline_progress_bar": config.enable_pipeline_progress_bar,
            "enable_carrier_progress_bar": config.enable_carrier_progress_bar,
            "supports_paper_claim": False,
        },
    )


def write_quality_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    """写出 aligned rescoring 质量指标表。"""
    fieldnames = [
        "carrier_id",
        "prompt_id",
        "attention_graph_id",
        "capture_id",
        "trajectory_index",
        "timestep",
        "update_norm",
        "latent_norm_before",
        "latent_norm_after",
        "clean_image_path",
        "aligned_image_path",
        "clean_image_digest",
        "aligned_image_digest",
        "psnr",
        "ssim",
        "mse",
        "mean_abs_error",
        "lpips",
        "lpips_status",
        "lpips_error_type",
        "lpips_error_message",
        "clip_score",
        "clip_score_clean",
        "clip_score_aligned",
        "clip_score_delta",
        "clip_score_status",
        "clip_score_error_type",
        "clip_score_error_message",
        "fid",
        "fid_status",
        "kid",
        "kid_status",
        "image_quality_metrics_ready",
        "perceptual_metrics_ready",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_csv_rows(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    """按固定字段写出 CSV, 缺失字段留空。"""

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field_name: row.get(field_name, "") for field_name in fieldnames})


def aligned_component_audit_rows(records: list[AlignedRescoringRecord]) -> list[dict[str, Any]]:
    """构造逐记录 LF/HF/combined 分量审计表。

    该表属于诊断与论文解释层: 正式 fixed-FPR 判定仍使用
    `real_aligned_content_score`, 但该表显式展示 LF、HF 和融合分数如何变化,
    便于定位 clean negative 高尾或 positive 低尾样本。
    """

    rows: list[dict[str, Any]] = []
    for record in records:
        metadata = record.metadata
        rows.append(
            {
                "aligned_rescoring_record_id": record.aligned_rescoring_record_id,
                "content_detection_record_id": record.content_detection_record_id,
                "prompt_id": record.prompt_id,
                "split": record.split,
                "sample_role": record.sample_role,
                "carrier_id": record.carrier_id,
                "real_raw_content_score": record.real_raw_content_score,
                "real_aligned_content_score": record.real_aligned_content_score,
                "real_rescoring_score_gain": record.real_rescoring_score_gain,
                "real_lf_score_before": record.real_lf_score_before,
                "real_lf_score_after": record.real_lf_score_after,
                "real_lf_score_gain": record.real_lf_score_after - record.real_lf_score_before,
                "real_hf_score_before": record.real_hf_score_before,
                "real_hf_score_after": record.real_hf_score_after,
                "real_hf_score_gain": record.real_hf_score_after - record.real_hf_score_before,
                "real_combined_score_before": record.real_combined_score_before,
                "real_combined_score_after": record.real_combined_score_after,
                "real_combined_score_gain": record.real_combined_score_after - record.real_combined_score_before,
                "real_lf_hf_fusion_score_before": record.real_lf_hf_fusion_score_before,
                "real_lf_hf_fusion_score_after": record.real_lf_hf_fusion_score_after,
                "real_lf_hf_fusion_score_gain": (
                    record.real_lf_hf_fusion_score_after - record.real_lf_hf_fusion_score_before
                ),
                "content_vector_width": metadata.get("content_vector_width", ""),
                "content_basis_rank": metadata.get("content_basis_rank", ""),
                "latent_projection_boundary_before": metadata.get("latent_projection_boundary_before", ""),
                "latent_projection_boundary_after": metadata.get("latent_projection_boundary_after", ""),
                "formal_score_source": metadata.get("formal_score_source", ""),
                "metric_status": record.metric_status,
                "supports_paper_claim": record.supports_paper_claim,
            }
        )
    return rows


def _mean(values: list[float]) -> float:
    """计算均值, 空列表返回 0。"""

    return sum(values) / len(values) if values else 0.0


def aligned_component_summary_rows(component_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """按 sample role 汇总内容分量, 并补充 positive-clean delta。"""

    sample_roles = sorted({str(row["sample_role"]) for row in component_rows})
    score_fields = (
        "real_raw_content_score",
        "real_aligned_content_score",
        "real_rescoring_score_gain",
        "real_lf_score_before",
        "real_lf_score_after",
        "real_hf_score_before",
        "real_hf_score_after",
        "real_combined_score_before",
        "real_combined_score_after",
        "real_lf_hf_fusion_score_before",
        "real_lf_hf_fusion_score_after",
    )
    rows: list[dict[str, Any]] = []
    means_by_role: dict[str, dict[str, float]] = {}
    for sample_role in sample_roles:
        role_rows = [row for row in component_rows if row["sample_role"] == sample_role]
        means = {
            field_name: _mean([float(row[field_name]) for row in role_rows])
            for field_name in score_fields
        }
        means_by_role[sample_role] = means
        rows.append(
            {
                "summary_scope": sample_role,
                "record_count": len(role_rows),
                **{f"{field_name}_mean": value for field_name, value in means.items()},
                "supports_paper_claim": False,
            }
        )
    if "positive_source" in means_by_role and "clean_negative" in means_by_role:
        delta = {
            field_name: means_by_role["positive_source"][field_name] - means_by_role["clean_negative"][field_name]
            for field_name in score_fields
        }
        rows.append(
            {
                "summary_scope": "positive_source_minus_clean_negative",
                "record_count": min(
                    len([row for row in component_rows if row["sample_role"] == "positive_source"]),
                    len([row for row in component_rows if row["sample_role"] == "clean_negative"]),
                ),
                **{f"{field_name}_mean": value for field_name, value in delta.items()},
                "supports_paper_claim": False,
            }
        )
    return rows


def aligned_tail_audit_rows(records: list[AlignedRescoringRecord], tail_count: int = 20) -> list[dict[str, Any]]:
    """构造高尾 negative 与低尾 positive 样本审计表。"""

    specs = (
        ("low_tail_positive", "positive_source", False),
        ("high_tail_clean_negative", "clean_negative", True),
        ("high_tail_attacked_negative", "attacked_negative", True),
    )
    rows: list[dict[str, Any]] = []
    for tail_bucket, sample_role, reverse in specs:
        role_records = [record for record in records if record.sample_role == sample_role]
        ranked = sorted(role_records, key=lambda record: record.real_aligned_content_score, reverse=reverse)
        for rank, record in enumerate(ranked[:tail_count], start=1):
            rows.append(
                {
                    "tail_bucket": tail_bucket,
                    "tail_rank": rank,
                    "aligned_rescoring_record_id": record.aligned_rescoring_record_id,
                    "content_detection_record_id": record.content_detection_record_id,
                    "prompt_id": record.prompt_id,
                    "split": record.split,
                    "sample_role": record.sample_role,
                    "carrier_id": record.carrier_id,
                    "real_raw_content_score": record.real_raw_content_score,
                    "real_aligned_content_score": record.real_aligned_content_score,
                    "real_rescoring_score_gain": record.real_rescoring_score_gain,
                    "real_lf_hf_fusion_score_after": record.real_lf_hf_fusion_score_after,
                    "latent_projection_digest_before": record.latent_projection_digest_before,
                    "latent_projection_digest_after": record.latent_projection_digest_after,
                    "metric_status": record.metric_status,
                    "supports_paper_claim": record.supports_paper_claim,
                }
            )
    return rows


def write_aligned_rescoring_outputs(config: AlignedRescoringConfig, root: str | Path = ".") -> dict[str, Any]:
    """运行真实 aligned rescoring 并写出受治理产物。"""
    root_path = Path(root).resolve()
    output_dir = (root_path / config.output_dir).resolve()
    clean_dir = output_dir / "clean_images"
    aligned_dir = output_dir / "aligned_images"
    output_dir.mkdir(parents=True, exist_ok=True)
    clean_dir.mkdir(parents=True, exist_ok=True)
    aligned_dir.mkdir(parents=True, exist_ok=True)
    records: list[AlignedRescoringRecord] = []
    quality_rows: list[dict[str, Any]] = []
    geometry_package_path: Path | None = None
    method_manifest_path: Path | None = None
    carrier_records: tuple[dict[str, Any], ...] = ()
    runtime_versions: dict[str, Any] = {}

    try:
        geometry_package_path = materialize_geometry_package(config, root_path)  # type: ignore[arg-type]
        prepared = prepare_attention_method_outputs(config, root_path)  # type: ignore[arg-type]
        method_manifest_path = prepared["method_manifest_path"]
        carrier_records = select_active_carriers(prepared["carrier_records_path"], config.max_rescore_carriers)
        prompt_lookup = prompt_text_by_id(root_path)
        content_by_prompt = load_content_records_by_prompt(root_path)
        selected_content_records = tuple(
            record
            for carrier_record in carrier_records
            for record in content_by_prompt.get(str(carrier_record.get("metadata", {}).get("prompt_id", "")), ())
        )
        content_updates = build_content_update_lookup(root_path, selected_content_records)
        pipeline, runtime_versions = load_pipeline(config)  # type: ignore[arg-type]
        pipeline_progress_bar_status = configure_pipeline_progress_bar(pipeline, config)
        runtime_versions = {
            **runtime_versions,
            "enable_pair_perceptual_metrics": config.enable_pair_perceptual_metrics,
            "require_pair_perceptual_metrics": config.require_pair_perceptual_metrics,
            "clip_model_id": config.clip_model_id,
            "lpips_network": config.lpips_network,
            "perceptual_metric_device_name": config.perceptual_metric_device_name,
            "pair_perceptual_dependency_install_command": PAIR_PERCEPTUAL_DEPENDENCY_INSTALL_COMMAND,
            "enable_pipeline_progress_bar": config.enable_pipeline_progress_bar,
            "enable_carrier_progress_bar": config.enable_carrier_progress_bar,
            "pipeline_progress_bar_status": pipeline_progress_bar_status,
        }
        for run_index, carrier_record in iterate_carriers_with_progress(carrier_records, config):
            current_prompt_id = str(carrier_record.get("metadata", {}).get("prompt_id", ""))
            current_prompt_text = prompt_lookup[current_prompt_id]
            current_content_records = content_by_prompt.get(current_prompt_id, ())
            carrier_records_out, quality_row, clean_image, aligned_image = run_carrier_rescoring(
                pipeline=pipeline,
                config=config,
                carrier_record=carrier_record,
                prompt_text=current_prompt_text,
                content_records=current_content_records,
                content_updates=content_updates,
                run_index=run_index,
            )
            clean_path = clean_dir / f"{config.model_family}_{config.seed + run_index}_{carrier_record['carrier_id']}_clean.png"
            aligned_path = aligned_dir / f"{config.model_family}_{config.seed + run_index}_{carrier_record['carrier_id']}_aligned.png"
            clean_image.save(clean_path)
            aligned_image.save(aligned_path)
            quality_rows.append(
                {
                    **quality_row,
                    "clean_image_path": clean_path.relative_to(root_path).as_posix(),
                    "aligned_image_path": aligned_path.relative_to(root_path).as_posix(),
                    "clean_image_digest": file_digest(clean_path),
                    "aligned_image_digest": file_digest(aligned_path),
                }
            )
            records.extend(carrier_records_out)
        perceptual_metrics_ready = (
            all(bool(row.get("perceptual_metrics_ready", False)) for row in quality_rows) if quality_rows else False
        )
        required_pair_metrics_missing = config.require_pair_perceptual_metrics and not perceptual_metrics_ready
        run_decision = "pass" if records and not required_pair_metrics_missing else "fail"
        unsupported_reason = ""
        if not records:
            unsupported_reason = "aligned_rescoring_records_missing"
        elif required_pair_metrics_missing:
            unsupported_reason = "pair_perceptual_metrics_missing:" + pair_metric_status_summary(quality_rows)
        result = AlignedRescoringResult(
            run_id=build_run_id(config, tuple(record["carrier_id"] for record in carrier_records)),
            model_family=config.model_family,
            model_id=config.model_id,
            run_decision=run_decision,
            unsupported_reason=unsupported_reason,
            aligned_rescoring_record_count=len(records),
            real_aligned_rescore_count=sum(1 for record in records if record.aligned_rescoring_ready),
            selected_attention_carrier_count=len(carrier_records),
            image_quality_metrics_ready=bool(quality_rows),
            perceptual_metrics_ready=perceptual_metrics_ready,
            full_method_claim_ready=False,
            output_records_path="",
            quality_metrics_path="",
            method_manifest_path="" if method_manifest_path is None else method_manifest_path.relative_to(root_path).as_posix(),
            attention_geometry_package_path="" if geometry_package_path is None else geometry_package_path.relative_to(root_path).as_posix(),
            metadata={
                **runtime_versions,
                "latent_projection_mode": "periodic_slot_pooled_content_carrier",
                "content_basis_rank": config.content_basis_rank,
                "supports_paper_claim": False,
            },
        )
    except Exception as error:  # pragma: no cover - 该路径依赖 Colab、GPU 与远程模型状态。
        result = build_failure_result(config, error)

    records_path = output_dir / "aligned_rescoring_records.jsonl"
    result_path = output_dir / "aligned_rescoring_result.json"
    quality_path = output_dir / "aligned_rescoring_quality_metrics.csv"
    component_audit_path = output_dir / "aligned_rescoring_component_audit.csv"
    component_summary_path = output_dir / "aligned_rescoring_component_summary.csv"
    tail_audit_path = output_dir / "aligned_rescoring_tail_audit.csv"
    environment_path = output_dir / "aligned_rescoring_environment_report.json"
    manifest_path = output_dir / "aligned_rescoring_manifest.local.json"
    environment_report = result.metadata.get("runtime_environment") or build_runtime_environment_report()
    if isinstance(environment_report, dict):
        environment_report = {
            **environment_report,
            "pair_perceptual_dependency_install_command": PAIR_PERCEPTUAL_DEPENDENCY_INSTALL_COMMAND,
            "enable_pair_perceptual_metrics": config.enable_pair_perceptual_metrics,
            "require_pair_perceptual_metrics": config.require_pair_perceptual_metrics,
            "clip_model_id": config.clip_model_id,
            "lpips_network": config.lpips_network,
        }
    records_path.write_text("".join(json_line(record.to_dict()) for record in records), encoding="utf-8")
    write_quality_rows(quality_path, quality_rows)
    component_rows = aligned_component_audit_rows(records)
    component_summary_rows = aligned_component_summary_rows(component_rows)
    tail_rows = aligned_tail_audit_rows(records)
    component_fieldnames = [
        "aligned_rescoring_record_id",
        "content_detection_record_id",
        "prompt_id",
        "split",
        "sample_role",
        "carrier_id",
        "real_raw_content_score",
        "real_aligned_content_score",
        "real_rescoring_score_gain",
        "real_lf_score_before",
        "real_lf_score_after",
        "real_lf_score_gain",
        "real_hf_score_before",
        "real_hf_score_after",
        "real_hf_score_gain",
        "real_combined_score_before",
        "real_combined_score_after",
        "real_combined_score_gain",
        "real_lf_hf_fusion_score_before",
        "real_lf_hf_fusion_score_after",
        "real_lf_hf_fusion_score_gain",
        "content_vector_width",
        "content_basis_rank",
        "latent_projection_boundary_before",
        "latent_projection_boundary_after",
        "formal_score_source",
        "metric_status",
        "supports_paper_claim",
    ]
    summary_fieldnames = [
        "summary_scope",
        "record_count",
        *[f"{field_name}_mean" for field_name in (
            "real_raw_content_score",
            "real_aligned_content_score",
            "real_rescoring_score_gain",
            "real_lf_score_before",
            "real_lf_score_after",
            "real_hf_score_before",
            "real_hf_score_after",
            "real_combined_score_before",
            "real_combined_score_after",
            "real_lf_hf_fusion_score_before",
            "real_lf_hf_fusion_score_after",
        )],
        "supports_paper_claim",
    ]
    tail_fieldnames = [
        "tail_bucket",
        "tail_rank",
        "aligned_rescoring_record_id",
        "content_detection_record_id",
        "prompt_id",
        "split",
        "sample_role",
        "carrier_id",
        "real_raw_content_score",
        "real_aligned_content_score",
        "real_rescoring_score_gain",
        "real_lf_hf_fusion_score_after",
        "latent_projection_digest_before",
        "latent_projection_digest_after",
        "metric_status",
        "supports_paper_claim",
    ]
    write_csv_rows(component_audit_path, component_rows, component_fieldnames)
    write_csv_rows(component_summary_path, component_summary_rows, summary_fieldnames)
    write_csv_rows(tail_audit_path, tail_rows, tail_fieldnames)
    environment_path.write_text(stable_json_text(environment_report), encoding="utf-8")
    result = AlignedRescoringResult(
        **{
            **result.to_dict(),
            "output_records_path": records_path.relative_to(root_path).as_posix(),
            "quality_metrics_path": quality_path.relative_to(root_path).as_posix(),
            "metadata": {
                **result.metadata,
                "environment_report_path": environment_path.relative_to(root_path).as_posix(),
                "component_audit_path": component_audit_path.relative_to(root_path).as_posix(),
                "component_summary_path": component_summary_path.relative_to(root_path).as_posix(),
                "tail_audit_path": tail_audit_path.relative_to(root_path).as_posix(),
            },
        }
    )
    result_path.write_text(stable_json_text(result.to_dict()), encoding="utf-8")
    output_paths = tuple(
        path.relative_to(root_path).as_posix()
        for path in (
            records_path,
            result_path,
            quality_path,
            component_audit_path,
            component_summary_path,
            tail_audit_path,
            environment_path,
            manifest_path,
        )
    )
    input_paths = [
            "experiments/runners/aligned_rescoring.py",
        "experiments/runners/attention_latent_injection.py",
    ]
    if result.method_manifest_path:
        input_paths.append(result.method_manifest_path)
    if result.attention_geometry_package_path:
        input_paths.append(result.attention_geometry_package_path)
    manifest = build_artifact_manifest(
        artifact_id="aligned_rescoring_manifest",
        artifact_type="local_manifest",
        input_paths=tuple(input_paths),
        output_paths=output_paths,
        config={
            "model_family": config.model_family,
            "model_id": config.model_id,
            "seed": config.seed,
            "attention_runtime_strength": config.attention_runtime_strength,
            "injection_step_indices": config.injection_step_indices,
            "content_vector_width": config.content_vector_width,
            "content_basis_rank": config.content_basis_rank,
            "max_subspace_records": config.max_subspace_records,
            "max_rescore_carriers": config.max_rescore_carriers,
            "aligned_rescoring_record_count": result.aligned_rescoring_record_count,
            "real_aligned_rescore_count": result.real_aligned_rescore_count,
            "enable_pair_perceptual_metrics": config.enable_pair_perceptual_metrics,
            "require_pair_perceptual_metrics": config.require_pair_perceptual_metrics,
            "clip_model_id": config.clip_model_id,
            "lpips_network": config.lpips_network,
            "perceptual_metrics_ready": result.perceptual_metrics_ready,
            "enable_pipeline_progress_bar": config.enable_pipeline_progress_bar,
            "enable_carrier_progress_bar": config.enable_carrier_progress_bar,
            "full_method_claim_ready": result.full_method_claim_ready,
        },
        code_version=resolve_code_version(root_path),
        rebuild_command="调用 experiments.runners.aligned_rescoring",
        metadata={
            "construction_unit_name": "aligned_rescoring",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "run_decision": result.run_decision,
            "unsupported_reason": result.unsupported_reason,
            "supports_paper_claim": False,
        },
    ).to_dict()
    manifest_path.write_text(stable_json_text(manifest), encoding="utf-8")
    return result.to_dict()


def build_default_config() -> AlignedRescoringConfig:
    """根据环境变量构造默认真实 aligned rescoring 配置。"""
    paper_run = build_paper_run_config(".")
    return AlignedRescoringConfig(
        model_family=PRIMARY_MODEL_FAMILY,
        model_id=os.environ.get("SLM_WM_SD35_MODEL_ID", PRIMARY_MODEL_ID),
        seed=int(os.environ.get("SLM_WM_SEED", "1703")),
        width=int(os.environ.get("SLM_WM_WIDTH", "512")),
        height=int(os.environ.get("SLM_WM_HEIGHT", "512")),
        inference_steps=int(os.environ.get("SLM_WM_INFERENCE_STEPS", str(paper_run.inference_steps))),
        guidance_scale=float(os.environ.get("SLM_WM_GUIDANCE_SCALE", str(paper_run.guidance_scale))),
        attention_runtime_strength=float(
            os.environ.get("SLM_WM_ATTENTION_RUNTIME_STRENGTH", str(paper_run.attention_runtime_strength))
        ),
        injection_step_indices=tuple(
            int(value.strip())
            for value in os.environ.get(
                "SLM_WM_ATTENTION_INJECTION_STEPS",
                ",".join(str(value) for value in paper_run.attention_injection_steps),
            ).split(",")
            if value.strip()
        ),
        output_dir=os.environ.get("SLM_WM_ALIGNED_RESCORING_OUTPUT_DIR", DEFAULT_OUTPUT_DIR),
        method_output_dir=os.environ.get("SLM_WM_ATTENTION_METHOD_OUTPUT_DIR", DEFAULT_METHOD_OUTPUT_DIR),
        geometry_drive_dir=os.environ.get(
            "SLM_WM_ATTENTION_GEOMETRY_DRIVE_DIR",
            paper_run.drive_dir("attention_geometry"),
        ),
        attention_geometry_package_path=os.environ.get("SLM_WM_ATTENTION_GEOMETRY_PACKAGE_PATH", ""),
        max_subspace_records=resolve_count_from_environment(
            "SLM_WM_ALIGNED_RESCORING_SUBSPACE_RECORDS",
            default_value=paper_run.sample_count,
        ),
        max_rescore_carriers=resolve_count_from_environment(
            "SLM_WM_ALIGNED_RESCORING_CARRIER_COUNT",
            default_value=paper_run.sample_count,
        ),
        content_vector_width=int(os.environ.get("SLM_WM_CONTENT_VECTOR_WIDTH", str(paper_run.content_vector_width))),
        content_basis_rank=int(os.environ.get("SLM_WM_CONTENT_BASIS_RANK", str(paper_run.content_basis_rank))),
        negative_prompt=os.environ.get("SLM_WM_NEGATIVE_PROMPT", "low quality, blurry"),
        enable_pair_perceptual_metrics=parse_bool_environment("SLM_WM_ENABLE_PAIR_PERCEPTUAL_METRICS", True),
        require_pair_perceptual_metrics=parse_bool_environment("SLM_WM_REQUIRE_PAIR_PERCEPTUAL_METRICS", True),
        clip_model_id=os.environ.get("SLM_WM_CLIP_MODEL_ID", DEFAULT_CLIP_MODEL_ID),
        lpips_network=os.environ.get("SLM_WM_LPIPS_NETWORK", DEFAULT_LPIPS_NETWORK),
        perceptual_metric_device_name=os.environ.get("SLM_WM_PERCEPTUAL_METRIC_DEVICE", "cpu"),
        enable_pipeline_progress_bar=parse_bool_environment("SLM_WM_ENABLE_PIPELINE_PROGRESS_BAR", False),
        enable_carrier_progress_bar=parse_bool_environment("SLM_WM_ENABLE_CARRIER_PROGRESS_BAR", True),
    )


def run_default_aligned_rescoring_plan(root: str | Path = ".") -> dict[str, Any]:
    """运行默认真实 aligned rescoring 计划。"""
    return write_aligned_rescoring_outputs(config=build_default_config(), root=root)


def collect_package_entries(root_path: Path, output_dir: Path, method_output_dir: Path, archive_path: Path) -> tuple[Path, ...]:
    """收集需要进入压缩包的核对文件。"""
    entries: list[Path] = []
    for source_dir in (output_dir, method_output_dir):
        if not source_dir.exists():
            continue
        for path in sorted(source_dir.rglob("*")):
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


def package_aligned_rescoring_outputs(
    root: str | Path = ".",
    output_dir: str = DEFAULT_OUTPUT_DIR,
    method_output_dir: str = DEFAULT_METHOD_OUTPUT_DIR,
    drive_output_dir: str | None = None,
    archive_name: str = "aligned_rescoring_package.zip",
) -> AlignedRescoringArchiveRecord:
    """打包真实 aligned rescoring 产物并镜像到 Google Drive。"""
    root_path = Path(root).resolve()
    resolved_drive_output_dir = drive_output_dir or build_paper_run_config(root_path).drive_dir("aligned_rescoring")
    source_dir = (root_path / output_dir).resolve()
    method_dir = (root_path / method_output_dir).resolve()
    source_dir.mkdir(parents=True, exist_ok=True)
    archive_path = source_dir / archive_name
    package_manifest_path = source_dir / "aligned_rescoring_package_input_manifest.json"
    summary_path = source_dir / "aligned_rescoring_archive_summary.json"
    manifest_path = source_dir / "aligned_rescoring_archive_manifest.local.json"
    entries = collect_package_entries(root_path, source_dir, method_dir, archive_path)
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
    drive_dir = Path(resolved_drive_output_dir).expanduser()
    drive_dir.mkdir(parents=True, exist_ok=True)
    mirrored_path = drive_dir / archive_name
    shutil.copy2(archive_path, mirrored_path)
    record = AlignedRescoringArchiveRecord(
        archive_path=archive_path.relative_to(root_path).as_posix(),
        archive_digest=file_digest(archive_path),
        archive_entry_count=len(entries),
        drive_archive_path=str(mirrored_path),
        drive_archive_digest=file_digest(mirrored_path),
        metadata={
            "construction_unit_name": "aligned_rescoring",
            "drive_output_dir": str(drive_dir),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    summary_path.write_text(stable_json_text(record.to_dict()), encoding="utf-8")
    manifest = build_artifact_manifest(
        artifact_id="aligned_rescoring_archive_manifest",
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
        rebuild_command="调用 experiments.runners.aligned_rescoring",
        metadata={
            "construction_unit_name": "aligned_rescoring",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "archive_digest": record.archive_digest,
            "drive_archive_digest": record.drive_archive_digest,
        },
    ).to_dict()
    manifest_path.write_text(stable_json_text(manifest), encoding="utf-8")
    return record



