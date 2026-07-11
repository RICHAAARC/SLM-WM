"""T2SMark 严格成对图像质量证据构造工具。"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from functools import lru_cache
import json
import os
from pathlib import Path
from typing import Any, Iterable

from PIL import Image

from experiments.runtime.image_metrics import compute_image_quality_metrics
from experiments.runtime.model_sources import get_model_source, require_registered_model_reference
from main.core.digest import build_stable_digest


STRICT_PAIR_PROTOCOL = "strict_clean_watermarked_pair"
_CLIP_MODEL_SOURCE = get_model_source("openai_clip_vit_base_patch32")
DEFAULT_CLIP_MODEL_ID = _CLIP_MODEL_SOURCE.repository_id
DEFAULT_CLIP_MODEL_REVISION = _CLIP_MODEL_SOURCE.revision
DEFAULT_LPIPS_NETWORK = "alex"


@dataclass(frozen=True)
class T2SMarkPairPerceptualConfig:
    """描述 T2SMark pair-level 感知质量指标配置。"""

    enable_pair_perceptual_metrics: bool = True
    clip_model_id: str = DEFAULT_CLIP_MODEL_ID
    clip_model_revision: str = DEFAULT_CLIP_MODEL_REVISION
    lpips_network: str = DEFAULT_LPIPS_NETWORK
    perceptual_metric_device_name: str = "cpu"
    hf_token_env: str = "HF_TOKEN"

    def __post_init__(self) -> None:
        """集中校验配置边界, 避免指标循环中重复构造错误信息。"""

        if self.lpips_network not in {"alex", "vgg", "squeeze"}:
            raise ValueError("lpips_network 必须为 alex、vgg 或 squeeze")
        if not self.clip_model_id.strip():
            raise ValueError("clip_model_id 不能为空")
        if len(self.clip_model_revision) != 40 or any(
            character not in "0123456789abcdef" for character in self.clip_model_revision
        ):
            raise ValueError("clip_model_revision 必须是40位小写十六进制提交")
        require_registered_model_reference(
            self.clip_model_id,
            self.clip_model_revision,
            required_usage_role="t2smark_pair_quality_encoder",
        )


def stable_json_text(value: Any) -> str:
    """把 JSON 兼容对象转换为稳定文本。"""

    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def read_json(path: Path) -> Any:
    """读取 JSON 文件。"""

    return json.loads(path.read_text(encoding="utf-8-sig"))


def relative_or_absolute(path: Path, root_path: Path) -> str:
    """优先记录相对仓库根目录路径。"""

    try:
        return path.resolve().relative_to(root_path.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def resolve_image_path(root_path: Path, image_path_text: str) -> Path:
    """将 image pair 中的图像路径解析为绝对路径。"""

    raw_path = Path(str(image_path_text or ""))
    return raw_path.resolve() if raw_path.is_absolute() else (root_path / raw_path).resolve()


def strict_pair_ready(row: dict[str, Any], root_path: Path) -> bool:
    """判断一条 T2SMark image pair 是否具备严格 clean/watermarked 图像对。"""

    clean_path = resolve_image_path(root_path, str(row.get("clean_image_path", "")))
    watermarked_path = resolve_image_path(root_path, str(row.get("watermarked_image_path", "")))
    return bool(row.get("clean_image_digest")) and bool(row.get("watermarked_image_digest")) and clean_path.is_file() and watermarked_path.is_file()


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


@lru_cache(maxsize=4)
def load_lpips_metric_model(lpips_network: str, device_name: str) -> Any:
    """加载并缓存 LPIPS 模型, 避免每个 T2SMark 图像对重复初始化网络。"""

    import lpips
    import torch

    device = torch.device(device_name)
    metric_model = lpips.LPIPS(net=lpips_network).to(device)
    metric_model.eval()
    return metric_model


def compute_lpips_metric(clean_image: Any, watermarked_image: Any, config: T2SMarkPairPerceptualConfig) -> dict[str, Any]:
    """计算 T2SMark clean/watermarked 成对图像 LPIPS。"""

    if not config.enable_pair_perceptual_metrics:
        return {"lpips": "unsupported", "lpips_status": "disabled", "lpips_error_type": "", "lpips_error_message": ""}
    try:
        import torch
        import lpips  # noqa: F401
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
            return {"lpips": "unsupported", "lpips_status": "gpu_unavailable", "lpips_error_type": "", "lpips_error_message": ""}
        metric_model = load_lpips_metric_model(config.lpips_network, config.perceptual_metric_device_name)
        clean_tensor = image_to_metric_tensor(clean_image, device)
        watermarked_tensor = image_to_metric_tensor(watermarked_image, device)
        with torch.no_grad():
            metric_value = metric_model(clean_tensor, watermarked_tensor)
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
        "clip_score_clean": "unsupported",
        "clip_score_watermarked": "unsupported",
        "clip_score_delta": "unsupported",
        "clip_score_status": status,
        "clip_score_error_type": "" if error is None else type(error).__name__,
        "clip_score_error_message": "" if error is None else metric_error_message(error),
    }


def clip_scores_from_features(image_features: Any, text_features: Any) -> tuple[float, float]:
    """由 CLIP image/text embeddings 计算 clean 和 watermarked 的余弦相似度。"""

    image_features = image_features.float()
    text_features = text_features.float()
    image_features = image_features / image_features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    scores = (image_features @ text_features.T).reshape(-1)
    return float(scores[0].detach().cpu().item()), float(scores[1].detach().cpu().item())


@lru_cache(maxsize=4)
def load_clip_metric_objects(
    clip_model_id: str,
    clip_model_revision: str,
    device_name: str,
    hf_token: str,
) -> tuple[Any, Any]:
    """加载并缓存 CLIP processor / model。"""

    import torch
    from transformers import CLIPModel, CLIPProcessor

    device = torch.device(device_name)
    require_registered_model_reference(
        clip_model_id,
        clip_model_revision,
        required_usage_role="t2smark_pair_quality_encoder",
    )
    model_kwargs = {"revision": clip_model_revision}
    if hf_token:
        model_kwargs["token"] = hf_token
    processor = CLIPProcessor.from_pretrained(clip_model_id, **model_kwargs)
    model = CLIPModel.from_pretrained(clip_model_id, **model_kwargs).to(device)
    model.eval()
    return processor, model


def compute_clip_pair_metrics(
    clean_image: Any,
    watermarked_image: Any,
    prompt_text: str,
    config: T2SMarkPairPerceptualConfig,
) -> dict[str, Any]:
    """计算 T2SMark clean/watermarked 图像相对 prompt 的 CLIP score。"""

    if not config.enable_pair_perceptual_metrics:
        return unsupported_clip_metric("disabled")
    try:
        import torch
        from transformers import CLIPModel, CLIPProcessor  # noqa: F401
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
            config.clip_model_revision,
            config.perceptual_metric_device_name,
            token,
        )
        image_inputs = processor(
            images=[clean_image.convert("RGB"), watermarked_image.convert("RGB")],
            return_tensors="pt",
        )
        text_inputs = processor(text=[prompt_text], return_tensors="pt", padding=True, truncation=True)
        image_inputs = move_batch_to_device(image_inputs, device)
        text_inputs = move_batch_to_device(text_inputs, device)
        with torch.no_grad():
            image_kwargs = {name: image_inputs[name] for name in ("pixel_values",) if name in image_inputs}
            text_kwargs = {name: text_inputs[name] for name in ("input_ids", "attention_mask", "position_ids") if name in text_inputs}
            image_features = model.get_image_features(**image_kwargs)
            text_features = model.get_text_features(**text_kwargs)
            clean_score, watermarked_score = clip_scores_from_features(image_features, text_features)
        return {
            "clip_score_clean": clean_score,
            "clip_score_watermarked": watermarked_score,
            "clip_score_delta": watermarked_score - clean_score,
            "clip_score_status": "measured",
            "clip_score_error_type": "",
            "clip_score_error_message": "",
        }
    except Exception as error:
        return unsupported_clip_metric("metric_runtime_error", error)


def unsupported_perceptual_metrics(status: str) -> dict[str, Any]:
    """构造未计算感知指标的统一字段。"""

    return {
        "lpips": "unsupported",
        "lpips_status": status,
        "lpips_error_type": "",
        "lpips_error_message": "",
        "clip_score_clean": "unsupported",
        "clip_score_watermarked": "unsupported",
        "clip_score_delta": "unsupported",
        "clip_score_status": status,
        "clip_score_error_type": "",
        "clip_score_error_message": "",
        "perceptual_metrics_ready": False,
    }


def compute_pair_perceptual_metrics(
    clean_image: Any,
    watermarked_image: Any,
    prompt_text: str,
    config: T2SMarkPairPerceptualConfig,
) -> dict[str, Any]:
    """计算 T2SMark pair-level LPIPS 与 CLIP 指标。"""

    if not config.enable_pair_perceptual_metrics:
        return unsupported_perceptual_metrics("disabled")
    lpips_row = compute_lpips_metric(clean_image, watermarked_image, config)
    clip_row = compute_clip_pair_metrics(clean_image, watermarked_image, prompt_text, config)
    return {
        **unsupported_perceptual_metrics("not_measured"),
        **lpips_row,
        **clip_row,
        "perceptual_metrics_ready": lpips_row.get("lpips_status") == "measured"
        and clip_row.get("clip_score_status") == "measured",
    }


def build_t2smark_strict_pair_quality_rows(
    image_pairs: Iterable[dict[str, Any]],
    *,
    root_path: Path,
    perceptual_config: T2SMarkPairPerceptualConfig | None = None,
) -> list[dict[str, Any]]:
    """基于严格 clean/watermarked 图像对构造 T2SMark pair-level 质量行。

    该函数属于完整论文实验层的证据重建逻辑: 它只消费已经落盘的 clean 图像与
    watermarked 图像, 不重新生成外部 baseline, 因而可在结果闭合入口复用历史包。
    """

    rows: list[dict[str, Any]] = []
    resolved_perceptual_config = perceptual_config or T2SMarkPairPerceptualConfig(enable_pair_perceptual_metrics=False)
    for index, pair in enumerate(image_pairs):
        clean_path = resolve_image_path(root_path, str(pair.get("clean_image_path", "")))
        watermarked_path = resolve_image_path(root_path, str(pair.get("watermarked_image_path", "")))
        ready = strict_pair_ready(pair, root_path)
        metrics: dict[str, Any]
        status: str
        if ready:
            with Image.open(clean_path) as clean_image, Image.open(watermarked_path) as watermarked_image:
                metrics = compute_image_quality_metrics(clean_image, watermarked_image)
                perceptual_metrics = compute_pair_perceptual_metrics(
                    clean_image,
                    watermarked_image,
                    str(pair.get("prompt_text") or ""),
                    resolved_perceptual_config,
                )
            status = "measured"
        else:
            metrics = {"psnr": "unsupported", "ssim": "unsupported", "mse": "unsupported", "mean_abs_error": "unsupported"}
            perceptual_metrics = unsupported_perceptual_metrics("strict_pair_image_missing")
            status = "strict_pair_image_missing"
        payload = {
            "baseline_id": "t2smark",
            "image_id": str(pair.get("image_id") or f"t2smark_pair_{index:05d}"),
            "prompt_id": str(pair.get("prompt_id") or ""),
            "prompt_index": int(pair.get("prompt_index", index) or index),
            "prompt_set": str(pair.get("prompt_set") or ""),
            "split": str(pair.get("split") or "test"),
            "pair_quality_protocol": STRICT_PAIR_PROTOCOL,
            "clean_image_path": relative_or_absolute(clean_path, root_path) if clean_path.is_file() else str(pair.get("clean_image_path", "")),
            "clean_image_digest": str(pair.get("clean_image_digest") or ""),
            "watermarked_image_path": relative_or_absolute(watermarked_path, root_path)
            if watermarked_path.is_file()
            else str(pair.get("watermarked_image_path", "")),
            "watermarked_image_digest": str(pair.get("watermarked_image_digest") or ""),
            "psnr": metrics["psnr"],
            "ssim": metrics["ssim"],
            "mse": metrics["mse"],
            "mean_abs_error": metrics["mean_abs_error"],
            "lpips": perceptual_metrics["lpips"],
            "lpips_status": perceptual_metrics["lpips_status"],
            "lpips_error_type": perceptual_metrics["lpips_error_type"],
            "lpips_error_message": perceptual_metrics["lpips_error_message"],
            "clip_score_clean": perceptual_metrics["clip_score_clean"],
            "clip_score_watermarked": perceptual_metrics["clip_score_watermarked"],
            "clip_score_delta": perceptual_metrics["clip_score_delta"],
            "clip_score_status": perceptual_metrics["clip_score_status"],
            "clip_score_error_type": perceptual_metrics["clip_score_error_type"],
            "clip_score_error_message": perceptual_metrics["clip_score_error_message"],
            "perceptual_metrics_ready": perceptual_metrics["perceptual_metrics_ready"],
            "pair_quality_status": status,
            "strict_pair_quality_ready": ready,
            "supports_paper_claim": False,
        }
        payload["pair_quality_record_digest"] = build_stable_digest(payload)
        rows.append(payload)
    return rows


def summarize_t2smark_strict_pair_quality(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """汇总 T2SMark 严格 pair-level 质量状态。"""

    row_values = list(rows)
    measured_rows = [row for row in row_values if row.get("pair_quality_status") == "measured"]
    ready = bool(row_values) and len(measured_rows) == len(row_values)
    return {
        "baseline_id": "t2smark",
        "pair_quality_protocol": STRICT_PAIR_PROTOCOL,
        "strict_pair_quality_record_count": len(row_values),
        "measured_strict_pair_quality_count": len(measured_rows),
        "strict_pair_quality_ready": ready,
        "psnr_mean": _mean_numeric(measured_rows, "psnr"),
        "ssim_mean": _mean_numeric(measured_rows, "ssim"),
        "mse_mean": _mean_numeric(measured_rows, "mse"),
        "mean_abs_error_mean": _mean_numeric(measured_rows, "mean_abs_error"),
        "lpips_mean": _mean_numeric([row for row in measured_rows if row.get("lpips_status") == "measured"], "lpips"),
        "clip_score_clean_mean": _mean_numeric(
            [row for row in measured_rows if row.get("clip_score_status") == "measured"],
            "clip_score_clean",
        ),
        "clip_score_watermarked_mean": _mean_numeric(
            [row for row in measured_rows if row.get("clip_score_status") == "measured"],
            "clip_score_watermarked",
        ),
        "clip_score_delta_mean": _mean_numeric(
            [row for row in measured_rows if row.get("clip_score_status") == "measured"],
            "clip_score_delta",
        ),
        "perceptual_metrics_ready_count": sum(1 for row in measured_rows if row.get("perceptual_metrics_ready") is True),
        "lpips_status_values": sorted({str(row.get("lpips_status") or "") for row in row_values if row.get("lpips_status")}),
        "clip_score_status_values": sorted(
            {str(row.get("clip_score_status") or "") for row in row_values if row.get("clip_score_status")}
        ),
        "supports_paper_claim": False,
    }


def _mean_numeric(rows: list[dict[str, Any]], field_name: str) -> float | str:
    """计算数值字段均值, 空集合返回 unsupported。"""

    values: list[float] = []
    for row in rows:
        try:
            values.append(float(row[field_name]))
        except (KeyError, TypeError, ValueError):
            continue
    if not values:
        return "unsupported"
    return sum(values) / len(values)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    """写出 CSV 表格。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


PAIR_QUALITY_FIELDNAMES = [
    "baseline_id",
    "image_id",
    "prompt_id",
    "prompt_index",
    "prompt_set",
    "split",
    "pair_quality_protocol",
    "clean_image_path",
    "clean_image_digest",
    "watermarked_image_path",
    "watermarked_image_digest",
    "psnr",
    "ssim",
    "mse",
    "mean_abs_error",
    "lpips",
    "lpips_status",
    "lpips_error_type",
    "lpips_error_message",
    "clip_score_clean",
    "clip_score_watermarked",
    "clip_score_delta",
    "clip_score_status",
    "clip_score_error_type",
    "clip_score_error_message",
    "perceptual_metrics_ready",
    "pair_quality_status",
    "strict_pair_quality_ready",
    "supports_paper_claim",
    "pair_quality_record_digest",
]


def write_t2smark_strict_pair_quality_outputs(
    *,
    root_path: Path,
    image_pairs_path: Path,
    metrics_path: Path,
    summary_path: Path,
    enable_pair_perceptual_metrics: bool = False,
    clip_model_id: str = DEFAULT_CLIP_MODEL_ID,
    clip_model_revision: str = DEFAULT_CLIP_MODEL_REVISION,
    lpips_network: str = DEFAULT_LPIPS_NETWORK,
    perceptual_metric_device_name: str = "cpu",
    hf_token_env: str = "HF_TOKEN",
) -> dict[str, Any]:
    """从 T2SMark image_pairs 写出严格 pair-level 质量表和摘要。"""

    image_pairs = [dict(row) for row in read_json(image_pairs_path)] if image_pairs_path.is_file() else []
    perceptual_config = T2SMarkPairPerceptualConfig(
        enable_pair_perceptual_metrics=enable_pair_perceptual_metrics,
        clip_model_id=clip_model_id,
        clip_model_revision=clip_model_revision,
        lpips_network=lpips_network,
        perceptual_metric_device_name=perceptual_metric_device_name,
        hf_token_env=hf_token_env,
    )
    rows = build_t2smark_strict_pair_quality_rows(
        image_pairs,
        root_path=root_path,
        perceptual_config=perceptual_config,
    )
    summary = summarize_t2smark_strict_pair_quality(rows)
    summary["image_pairs_path"] = relative_or_absolute(image_pairs_path, root_path)
    summary["pair_quality_metrics_path"] = relative_or_absolute(metrics_path, root_path)
    summary["enable_pair_perceptual_metrics"] = enable_pair_perceptual_metrics
    summary["clip_model_id"] = clip_model_id
    summary["clip_model_revision"] = clip_model_revision
    summary["lpips_network"] = lpips_network
    summary["perceptual_metric_device_name"] = perceptual_metric_device_name
    summary["pair_quality_rows_digest"] = build_stable_digest(rows)
    write_csv(metrics_path, rows, PAIR_QUALITY_FIELDNAMES)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(stable_json_text(summary), encoding="utf-8")
    return summary
