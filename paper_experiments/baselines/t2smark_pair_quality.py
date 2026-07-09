"""T2SMark 严格成对图像质量证据构造工具。"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

from PIL import Image

from experiments.runtime.diffusion.sd3_pipeline_runtime import compute_image_quality_metrics
from main.core.digest import build_stable_digest


STRICT_PAIR_PROTOCOL = "strict_clean_watermarked_pair"


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


def build_t2smark_strict_pair_quality_rows(
    image_pairs: Iterable[dict[str, Any]],
    *,
    root_path: Path,
) -> list[dict[str, Any]]:
    """基于严格 clean/watermarked 图像对构造 T2SMark pair-level 质量行。

    该函数属于完整论文实验层的证据重建逻辑: 它只消费已经落盘的 clean 图像与
    watermarked 图像, 不重新生成外部 baseline, 因而可在结果闭合入口复用历史包。
    """

    rows: list[dict[str, Any]] = []
    for index, pair in enumerate(image_pairs):
        clean_path = resolve_image_path(root_path, str(pair.get("clean_image_path", "")))
        watermarked_path = resolve_image_path(root_path, str(pair.get("watermarked_image_path", "")))
        ready = strict_pair_ready(pair, root_path)
        metrics: dict[str, Any]
        status: str
        if ready:
            with Image.open(clean_path) as clean_image, Image.open(watermarked_path) as watermarked_image:
                metrics = compute_image_quality_metrics(clean_image, watermarked_image)
            status = "measured"
        else:
            metrics = {"psnr": "unsupported", "ssim": "unsupported", "mse": "unsupported", "mean_abs_error": "unsupported"}
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
            "lpips": "unsupported",
            "lpips_status": "not_computed_in_t2smark_strict_pair_quality",
            "clip_score_clean": "unsupported",
            "clip_score_watermarked": "unsupported",
            "clip_score_delta": "unsupported",
            "clip_score_status": "not_computed_in_t2smark_strict_pair_quality",
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
        "lpips_status": "not_computed_in_t2smark_strict_pair_quality",
        "clip_score_status": "not_computed_in_t2smark_strict_pair_quality",
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
    "clip_score_clean",
    "clip_score_watermarked",
    "clip_score_delta",
    "clip_score_status",
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
) -> dict[str, Any]:
    """从 T2SMark image_pairs 写出严格 pair-level 质量表和摘要。"""

    image_pairs = [dict(row) for row in read_json(image_pairs_path)] if image_pairs_path.is_file() else []
    rows = build_t2smark_strict_pair_quality_rows(image_pairs, root_path=root_path)
    summary = summarize_t2smark_strict_pair_quality(rows)
    summary["image_pairs_path"] = relative_or_absolute(image_pairs_path, root_path)
    summary["pair_quality_metrics_path"] = relative_or_absolute(metrics_path, root_path)
    summary["pair_quality_rows_digest"] = build_stable_digest(rows)
    write_csv(metrics_path, rows, PAIR_QUALITY_FIELDNAMES)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(stable_json_text(summary), encoding="utf-8")
    return summary
