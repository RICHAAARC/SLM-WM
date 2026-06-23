"""数据集级图像质量证据的轻量治理协议。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from PIL import Image

from main.core.digest import build_stable_digest

PIXEL_FEATURE_BACKEND = "pixel_histogram_feature_proxy"
FORMAL_FID_KID_BLOCKER = "requires_inception_feature_backend"


@dataclass(frozen=True)
class DatasetQualityImageRecord:
    """记录一组 source / comparison 图像对进入数据集级质量协议的事实。

    该对象属于通用工程写法: 它只记录图像路径、摘要、配对角色和特征后端, 不在记录层直接声明正式
    FID / KID 结论。正式论文级 FID / KID 需要后续替换为 Inception 特征后端和足够样本规模。
    """

    dataset_quality_record_id: str
    dataset_quality_record_digest: str
    image_pair_index: int
    image_pair_role: str
    source_image_path: str
    source_image_digest: str
    comparison_image_path: str
    comparison_image_digest: str
    feature_backend: str
    supports_paper_claim: bool

    def to_dict(self) -> dict[str, Any]:
        """转换为 JSON 兼容字典。"""

        return asdict(self)


def _as_relative_or_absolute(path: Path, root_path: Path) -> str:
    """优先返回相对仓库根目录路径, 便于 manifest 复用。"""

    try:
        return path.resolve().relative_to(root_path).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _row_path(root_path: Path, row: Mapping[str, Any], field_name: str) -> Path:
    """把记录中的图像路径解析为绝对路径。"""

    value = Path(str(row.get(field_name, "") or ""))
    return value.resolve() if value.is_absolute() else (root_path / value).resolve()


def _row_digest(row: Mapping[str, Any], field_name: str, path: Path) -> str:
    """读取已有 digest, 缺失时根据文件内容生成 digest。"""

    existing = str(row.get(field_name, "") or "")
    if existing:
        return existing
    if not path.is_file():
        return "missing_file_digest_unavailable"
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def extract_pixel_histogram_feature(image_path: Path, image_size: int = 32, hist_bins: int = 8) -> np.ndarray:
    """从图像中提取轻量 RGB 统计特征。

    该函数不是 Inception 特征提取器, 只用于本地小样本治理入口和链路测试。后续如果要支撑正式 FID / KID,
    应在相同记录协议下替换为 Inception 或论文约定的视觉特征后端。
    """

    with Image.open(image_path) as image:
        image_rgb = image.convert("RGB").resize((image_size, image_size))
    array = np.asarray(image_rgb, dtype=np.float64) / 255.0
    channel_mean = array.mean(axis=(0, 1))
    channel_std = array.std(axis=(0, 1))
    histogram_parts = []
    for channel_index in range(3):
        hist, _ = np.histogram(array[:, :, channel_index], bins=hist_bins, range=(0.0, 1.0), density=True)
        histogram_parts.append(hist / max(float(hist.sum()), 1.0))
    return np.concatenate([channel_mean, channel_std, *histogram_parts]).astype(np.float64)


def _diagonal_gaussian_fid(source_features: np.ndarray, comparison_features: np.ndarray) -> float:
    """计算对角协方差近似 FID proxy, 避免引入重型线性代数依赖。"""

    source_mean = source_features.mean(axis=0)
    comparison_mean = comparison_features.mean(axis=0)
    source_var = source_features.var(axis=0)
    comparison_var = comparison_features.var(axis=0)
    mean_term = np.sum((source_mean - comparison_mean) ** 2)
    covariance_term = np.sum(source_var + comparison_var - 2.0 * np.sqrt(np.maximum(source_var * comparison_var, 0.0)))
    return float(max(mean_term + covariance_term, 0.0))


def _polynomial_kernel(values_a: np.ndarray, values_b: np.ndarray) -> np.ndarray:
    """计算 KID proxy 使用的三阶多项式核矩阵。"""

    feature_dim = max(int(values_a.shape[1]), 1)
    return ((values_a @ values_b.T) / feature_dim + 1.0) ** 3


def _biased_polynomial_mmd(source_features: np.ndarray, comparison_features: np.ndarray) -> float:
    """计算有偏 MMD 形式的 KID proxy, 使极小样本也有稳定可审计输出。"""

    source_kernel = _polynomial_kernel(source_features, source_features).mean()
    comparison_kernel = _polynomial_kernel(comparison_features, comparison_features).mean()
    cross_kernel = _polynomial_kernel(source_features, comparison_features).mean()
    return float(max(source_kernel + comparison_kernel - 2.0 * cross_kernel, 0.0))


def build_dataset_quality_image_records(
    registry_rows: Iterable[Mapping[str, Any]],
    root_path: Path,
) -> tuple[DatasetQualityImageRecord, ...]:
    """从真实攻击图像 registry 构造数据集级质量图像记录。"""

    records: list[DatasetQualityImageRecord] = []
    for index, row in enumerate(registry_rows):
        source_path = _row_path(root_path, row, "source_image_path")
        comparison_path = _row_path(root_path, row, "attacked_image_path")
        source_digest = _row_digest(row, "source_image_digest", source_path)
        comparison_digest = _row_digest(row, "attacked_image_digest", comparison_path)
        image_pair_role = str(row.get("attack_name", "") or "comparison_image")
        payload = {
            "image_pair_index": index,
            "image_pair_role": image_pair_role,
            "source_image_path": _as_relative_or_absolute(source_path, root_path),
            "source_image_digest": source_digest,
            "comparison_image_path": _as_relative_or_absolute(comparison_path, root_path),
            "comparison_image_digest": comparison_digest,
            "feature_backend": PIXEL_FEATURE_BACKEND,
            "supports_paper_claim": False,
        }
        digest = build_stable_digest(payload)
        records.append(
            DatasetQualityImageRecord(
                dataset_quality_record_id=f"dataset_quality_record_{digest[:16]}",
                dataset_quality_record_digest=digest,
                image_pair_index=index,
                image_pair_role=image_pair_role,
                source_image_path=payload["source_image_path"],
                source_image_digest=source_digest,
                comparison_image_path=payload["comparison_image_path"],
                comparison_image_digest=comparison_digest,
                feature_backend=PIXEL_FEATURE_BACKEND,
                supports_paper_claim=False,
            )
        )
    return tuple(records)


def build_dataset_quality_metric_rows(
    records: Iterable[DatasetQualityImageRecord],
    root_path: Path,
) -> list[dict[str, Any]]:
    """构造数据集级质量指标表。

    表中同时保留正式 FID / KID 的 unsupported 行和小样本 proxy 行, 用于明确统计边界。
    """

    record_values = tuple(records)
    source_paths = tuple((root_path / record.source_image_path).resolve() for record in record_values)
    comparison_paths = tuple((root_path / record.comparison_image_path).resolve() for record in record_values)
    missing_image_file_count = sum(1 for path in source_paths + comparison_paths if not path.is_file())
    source_count = len(source_paths)
    comparison_count = len(comparison_paths)
    metric_context = {
        "feature_backend": PIXEL_FEATURE_BACKEND,
        "source_image_count": source_count,
        "comparison_image_count": comparison_count,
        "sample_pair_count": len(record_values),
        "supports_paper_claim": False,
    }
    rows = [
        {
            "quality_metric_name": "fid",
            "quality_metric_value": "unsupported",
            "metric_status": FORMAL_FID_KID_BLOCKER,
            "paper_metric_name": "fid",
            **metric_context,
        },
        {
            "quality_metric_name": "kid",
            "quality_metric_value": "unsupported",
            "metric_status": FORMAL_FID_KID_BLOCKER,
            "paper_metric_name": "kid",
            **metric_context,
        },
    ]
    if not record_values:
        return rows
    if missing_image_file_count:
        rows.extend(
            [
                {
                    "quality_metric_name": "fid_pixel_feature_proxy",
                    "quality_metric_value": "unsupported",
                    "metric_status": "image_file_missing",
                    "paper_metric_name": "fid",
                    **metric_context,
                },
                {
                    "quality_metric_name": "kid_pixel_feature_proxy",
                    "quality_metric_value": "unsupported",
                    "metric_status": "image_file_missing",
                    "paper_metric_name": "kid",
                    **metric_context,
                },
            ]
        )
        return rows
    source_features = [extract_pixel_histogram_feature(path) for path in source_paths]
    comparison_features = [extract_pixel_histogram_feature(path) for path in comparison_paths]
    source_array = np.vstack(source_features)
    comparison_array = np.vstack(comparison_features)
    rows.extend(
        [
            {
                "quality_metric_name": "fid_pixel_feature_proxy",
                "quality_metric_value": _diagonal_gaussian_fid(source_array, comparison_array),
                "metric_status": "measured_small_sample_proxy",
                "paper_metric_name": "fid",
                **metric_context,
            },
            {
                "quality_metric_name": "kid_pixel_feature_proxy",
                "quality_metric_value": _biased_polynomial_mmd(source_array, comparison_array),
                "metric_status": "measured_small_sample_proxy",
                "paper_metric_name": "kid",
                **metric_context,
            },
        ]
    )
    return rows


def build_dataset_quality_summary(
    records: Iterable[DatasetQualityImageRecord],
    metric_rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """聚合数据集级质量证据摘要。"""

    record_values = tuple(records)
    rows = tuple(metric_rows)
    proxy_ready = any(str(row.get("metric_status")) == "measured_small_sample_proxy" for row in rows)
    formal_ready = any(
        str(row.get("quality_metric_name")) in {"fid", "kid"} and str(row.get("metric_status")) == "measured"
        for row in rows
    )
    return {
        "construction_unit_name": "dataset_level_quality_evidence",
        "dataset_quality_record_count": len(record_values),
        "source_image_count": len(record_values),
        "comparison_image_count": len(record_values),
        "sample_pair_count": len(record_values),
        "feature_backend": PIXEL_FEATURE_BACKEND,
        "dataset_level_quality_proxy_ready": proxy_ready,
        "formal_fid_kid_ready": formal_ready,
        "paper_claim_ready": False,
        "unsupported_reason": FORMAL_FID_KID_BLOCKER if not formal_ready else "",
        "supports_paper_claim": False,
    }
