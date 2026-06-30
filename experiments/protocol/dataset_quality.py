"""数据集级图像质量证据的轻量治理协议。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from pathlib import Path
import time
from typing import Any, Iterable, Mapping

import numpy as np
from PIL import Image

from main.core.digest import build_stable_digest

PIXEL_FEATURE_BACKEND = "pixel_histogram_feature_proxy"
FORMAL_FEATURE_BACKEND = "inception_feature_backend"
FORMAL_FID_KID_BLOCKER = "requires_inception_feature_backend"
FORMAL_FID_KID_SAMPLE_BLOCKER = "requires_full_main_sample_scale"
FORMAL_FID_KID_NUMERIC_BLOCKER = "requires_covariance_square_root_backend"
FORMAL_FID_CPU_MAX_FEATURE_DIM = 512
FORMAL_KID_EXACT_MAX_SAMPLE_COUNT = 3000
FORMAL_KID_SUBSET_COUNT = 8
FORMAL_KID_SUBSET_SIZE = 512


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


def _resolve_existing_image_path(
    image_path_text: str,
    root_path: Path,
    image_search_roots: Iterable[Path] = (),
) -> Path:
    """在仓库根目录和补充图像根目录中解析图像路径。

    该函数属于通用工程写法: 记录层保留原始相对路径, 度量层只在需要读取图像时解析实际文件位置。
    这样可以复用同一批 records, 同时允许前序 Colab 产物从 Google Drive ZIP 中解包到受治理的 outputs 子目录后参与计算。
    """

    raw_path = Path(image_path_text)
    if raw_path.is_absolute():
        return raw_path.resolve()
    candidates = [(root_path / raw_path).resolve()]
    candidates.extend((Path(search_root) / raw_path).resolve() for search_root in image_search_roots)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


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


def _metric_progress_enabled() -> bool:
    """判断是否输出数据集级质量指标重建进度。"""

    value = os.environ.get("SLM_WM_DATASET_QUALITY_METRIC_PROGRESS", "1").strip().lower()
    return value not in {"0", "false", "no", "off"}


def _metric_progress_interval_items() -> int:
    """读取图像级 proxy 进度刷新间隔."""

    return max(1, int(os.environ.get("SLM_WM_DATASET_QUALITY_PROGRESS_INTERVAL_ITEMS", "50")))


def _should_emit_metric_item_progress(completed: int, total: int) -> bool:
    """判断图像级 proxy 进度是否需要刷新."""

    interval = _metric_progress_interval_items()
    return completed in {0, total} or completed % interval == 0


def _emit_metric_progress(
    *,
    started_at: float,
    completed: int,
    total: int,
    profile: str,
) -> None:
    """输出长耗时数据集级指标的总体进度。

    该函数属于通用工程写法: 它只报告当前指标重建所处的资源环节, 不参与
    FID / KID 数值逻辑。这样可以避免用户在 Colab 中看到特征提取完成后
    长时间没有任何反馈, 同时保持核心指标计算路径可复用。
    """

    if not _metric_progress_enabled():
        return
    elapsed_seconds = max(0.0, time.monotonic() - started_at)
    percent = completed / max(total, 1) * 100.0
    eta_seconds = elapsed_seconds * max(total - completed, 0) / completed if completed > 0 else 0.0
    print(
        (
            f"工作量进度 | dataset-level metric reconstruction | {completed}/{total} ({percent:.1f}%) | "
            f"elapsed={elapsed_seconds / 60.0:.1f} min | eta={eta_seconds / 60.0:.1f} min | "
            f"profile={profile}"
        ),
        flush=True,
    )


def _feature_array_or_none(feature_values: Any) -> np.ndarray | None:
    """把可选特征矩阵规整为二维浮点数组。

    该函数属于配置解析边界: 业务逻辑只消费已经规整的矩阵, 避免在指标构造路径中重复处理列表、空值和维度差异。
    """

    if feature_values is None:
        return None
    array = np.asarray(feature_values, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        return None
    return array


def _fid_cpu_feature_dim_limit() -> int:
    """读取 CPU FID 最大特征维度, 防止无 GPU 会话长时间阻塞。"""

    return int(os.environ.get("SLM_WM_FID_CPU_MAX_FEATURE_DIM", str(FORMAL_FID_CPU_MAX_FEATURE_DIM)))


def _sqrt_trace_fid_from_covariances(
    source_mean: np.ndarray,
    comparison_mean: np.ndarray,
    source_covariance: np.ndarray,
    comparison_covariance: np.ndarray,
) -> float:
    """用对称特征分解计算完整协方差 FID。

    这一实现属于通用工程写法: FID 只需要 `sqrt(C1 C2)` 的迹, 不需要构造
    SciPy `sqrtm` 返回的完整复数矩阵。先构造 `sqrt(C1) C2 sqrt(C1)` 这个
    对称半正定矩阵, 再对其特征值开方求和, 可以显著降低数值后处理的
    常数开销, 并避免 Colab CPU 在 2048 维 Inception 特征上长时间无反馈。
    """

    source_covariance = (source_covariance + source_covariance.T) * 0.5
    comparison_covariance = (comparison_covariance + comparison_covariance.T) * 0.5
    source_eigenvalues, source_eigenvectors = np.linalg.eigh(source_covariance)
    source_eigenvalues = np.clip(source_eigenvalues, 0.0, None)
    source_covariance_sqrt = (source_eigenvectors * np.sqrt(source_eigenvalues)[None, :]) @ source_eigenvectors.T
    middle = source_covariance_sqrt @ comparison_covariance @ source_covariance_sqrt
    middle = (middle + middle.T) * 0.5
    middle_eigenvalues = np.linalg.eigvalsh(middle)
    covariance_sqrt_trace = np.sqrt(np.clip(middle_eigenvalues, 0.0, None)).sum()
    mean_term = np.sum((source_mean - comparison_mean) ** 2)
    covariance_term = np.trace(source_covariance) + np.trace(comparison_covariance) - 2.0 * covariance_sqrt_trace
    return float(max(mean_term + covariance_term, 0.0))


def _torch_gaussian_fid(
    source_features: np.ndarray,
    comparison_features: np.ndarray,
    *,
    progress: Any = None,
) -> float | None:
    """优先使用 PyTorch 后端计算完整协方差 FID。

    项目特定考虑在于: Colab `pilot_paper` 会产生 2048 维 Inception 特征,
    SciPy `sqrtm` 在 CPU 上可能阻塞数小时。这里在 CUDA 可用时使用
    `torch.linalg.eigh` 计算同一 FID 公式; CUDA 不可用且维度过高时返回
    `None`, 由上层写出受治理的 numeric blocker, 而不是让 Notebook 静默卡住。
    """

    try:
        import torch
    except Exception:
        return None
    feature_dim = int(source_features.shape[1])
    use_cuda = bool(torch.cuda.is_available())
    if not use_cuda and feature_dim > _fid_cpu_feature_dim_limit():
        return None
    device = torch.device("cuda" if use_cuda else "cpu")
    dtype = torch.float32 if use_cuda else torch.float64
    try:
        with torch.no_grad():
            if progress is not None:
                progress("fid_torch_feature_transfer")
            source_tensor = torch.as_tensor(source_features, dtype=dtype, device=device)
            comparison_tensor = torch.as_tensor(comparison_features, dtype=dtype, device=device)
            if progress is not None:
                progress("fid_torch_covariance")
            source_mean = source_tensor.mean(dim=0)
            comparison_mean = comparison_tensor.mean(dim=0)
            source_centered = source_tensor - source_mean
            comparison_centered = comparison_tensor - comparison_mean
            source_denominator = max(int(source_tensor.shape[0]) - 1, 1)
            comparison_denominator = max(int(comparison_tensor.shape[0]) - 1, 1)
            source_covariance = source_centered.T.matmul(source_centered) / source_denominator
            comparison_covariance = comparison_centered.T.matmul(comparison_centered) / comparison_denominator
            source_covariance = (source_covariance + source_covariance.T) * 0.5
            comparison_covariance = (comparison_covariance + comparison_covariance.T) * 0.5
            if progress is not None:
                progress("fid_torch_covariance_square_root")
            source_eigenvalues, source_eigenvectors = torch.linalg.eigh(source_covariance)
            source_eigenvalues = torch.clamp(source_eigenvalues, min=0.0)
            source_covariance_sqrt = (
                source_eigenvectors * torch.sqrt(source_eigenvalues).unsqueeze(0)
            ).matmul(source_eigenvectors.T)
            middle = source_covariance_sqrt.matmul(comparison_covariance).matmul(source_covariance_sqrt)
            middle = (middle + middle.T) * 0.5
            middle_eigenvalues = torch.linalg.eigvalsh(middle)
            covariance_sqrt_trace = torch.sqrt(torch.clamp(middle_eigenvalues, min=0.0)).sum()
            mean_term = torch.sum((source_mean - comparison_mean) ** 2)
            covariance_term = torch.trace(source_covariance) + torch.trace(comparison_covariance) - 2.0 * covariance_sqrt_trace
            value = torch.clamp(mean_term + covariance_term, min=0.0).detach().cpu().item()
    except Exception:
        return None
    return float(value)


def _exact_gaussian_fid(
    source_features: np.ndarray,
    comparison_features: np.ndarray,
    *,
    progress: Any = None,
) -> float | None:
    """使用完整协方差矩阵计算 FID。

    此处设计的主要考虑在于: 正式 FID 必须基于视觉特征分布的均值与协方差,
    不能复用轻量 pixel proxy 的对角近似。实现优先使用 CUDA 上的对称
    特征分解; 在无 GPU 且特征维度过高时返回 numeric blocker, 防止
    repository runner 或 Notebook 长时间卡死。
    """

    torch_value = _torch_gaussian_fid(source_features, comparison_features, progress=progress)
    if torch_value is not None:
        return torch_value
    feature_dim = int(source_features.shape[1])
    if feature_dim > _fid_cpu_feature_dim_limit():
        return None
    if progress is not None:
        progress("fid_numpy_covariance")
    source_mean = source_features.mean(axis=0)
    comparison_mean = comparison_features.mean(axis=0)
    source_covariance = np.atleast_2d(np.cov(source_features, rowvar=False))
    comparison_covariance = np.atleast_2d(np.cov(comparison_features, rowvar=False))
    if progress is not None:
        progress("fid_numpy_covariance_square_root")
    return _sqrt_trace_fid_from_covariances(
        source_mean,
        comparison_mean,
        source_covariance,
        comparison_covariance,
    )


def _kid_exact_max_sample_count() -> int:
    """读取 KID 完整核矩阵计算的样本上限。"""

    return int(os.environ.get("SLM_WM_KID_EXACT_MAX_SAMPLE_COUNT", str(FORMAL_KID_EXACT_MAX_SAMPLE_COUNT)))


def _kid_subset_count() -> int:
    """读取大样本 KID 确定性子集数量。"""

    return max(1, int(os.environ.get("SLM_WM_KID_SUBSET_COUNT", str(FORMAL_KID_SUBSET_COUNT))))


def _kid_subset_size() -> int:
    """读取大样本 KID 确定性子集大小。"""

    return max(2, int(os.environ.get("SLM_WM_KID_SUBSET_SIZE", str(FORMAL_KID_SUBSET_SIZE))))


def _unbiased_polynomial_mmd_exact(source_features: np.ndarray, comparison_features: np.ndarray) -> float:
    """使用三阶多项式核计算完整无偏 MMD。"""

    torch_value = _torch_unbiased_polynomial_mmd_exact(source_features, comparison_features)
    if torch_value is not None:
        return torch_value
    source_count = source_features.shape[0]
    comparison_count = comparison_features.shape[0]
    source_kernel = _polynomial_kernel(source_features, source_features)
    comparison_kernel = _polynomial_kernel(comparison_features, comparison_features)
    cross_kernel = _polynomial_kernel(source_features, comparison_features)
    source_term = (source_kernel.sum() - np.trace(source_kernel)) / max(source_count * (source_count - 1), 1)
    comparison_term = (comparison_kernel.sum() - np.trace(comparison_kernel)) / max(
        comparison_count * (comparison_count - 1),
        1,
    )
    cross_term = cross_kernel.mean()
    return float(max(source_term + comparison_term - 2.0 * cross_term, 0.0))


def _torch_unbiased_polynomial_mmd_exact(
    source_features: np.ndarray,
    comparison_features: np.ndarray,
) -> float | None:
    """在 CUDA 可用时计算完整 KID 核矩阵, 避免 Colab CPU 后处理过慢。"""

    try:
        import torch
    except Exception:
        return None
    if not torch.cuda.is_available():
        return None
    try:
        with torch.no_grad():
            device = torch.device("cuda")
            source_tensor = torch.as_tensor(source_features, dtype=torch.float32, device=device)
            comparison_tensor = torch.as_tensor(comparison_features, dtype=torch.float32, device=device)
            feature_dim = max(int(source_tensor.shape[1]), 1)
            source_kernel = (source_tensor.matmul(source_tensor.T) / feature_dim + 1.0) ** 3
            comparison_kernel = (comparison_tensor.matmul(comparison_tensor.T) / feature_dim + 1.0) ** 3
            cross_kernel = (source_tensor.matmul(comparison_tensor.T) / feature_dim + 1.0) ** 3
            source_count = int(source_tensor.shape[0])
            comparison_count = int(comparison_tensor.shape[0])
            source_term = (source_kernel.sum() - torch.trace(source_kernel)) / max(source_count * (source_count - 1), 1)
            comparison_term = (comparison_kernel.sum() - torch.trace(comparison_kernel)) / max(
                comparison_count * (comparison_count - 1),
                1,
            )
            cross_term = cross_kernel.mean()
            value = torch.clamp(source_term + comparison_term - 2.0 * cross_term, min=0.0).detach().cpu().item()
    except Exception:
        return None
    return float(value)


def _deterministic_subset_indices(total: int, subset_size: int, subset_index: int, subset_count: int) -> np.ndarray:
    """生成可复现的大样本 KID 子集索引。"""

    if total <= subset_size:
        return np.arange(total)
    if subset_count <= 1:
        start = max(0, (total - subset_size) // 2)
    else:
        start = round(subset_index * (total - subset_size) / (subset_count - 1))
    return np.arange(int(start), int(start) + subset_size)


def _deterministic_subset_polynomial_mmd(source_features: np.ndarray, comparison_features: np.ndarray) -> float:
    """使用确定性子集估计大样本 KID, 避免构造超大核矩阵。"""

    subset_size = min(_kid_subset_size(), source_features.shape[0], comparison_features.shape[0])
    subset_count = _kid_subset_count()
    values: list[float] = []
    for subset_index in range(subset_count):
        source_indices = _deterministic_subset_indices(source_features.shape[0], subset_size, subset_index, subset_count)
        comparison_indices = _deterministic_subset_indices(
            comparison_features.shape[0],
            subset_size,
            subset_index,
            subset_count,
        )
        values.append(
            _unbiased_polynomial_mmd_exact(
                source_features[source_indices],
                comparison_features[comparison_indices],
            )
        )
    return float(max(np.mean(values), 0.0))


def _unbiased_polynomial_mmd(source_features: np.ndarray, comparison_features: np.ndarray) -> float:
    """使用三阶多项式核计算 KID 的无偏 MMD 形式。"""

    max_count = max(int(source_features.shape[0]), int(comparison_features.shape[0]))
    if max_count <= _kid_exact_max_sample_count():
        return _unbiased_polynomial_mmd_exact(source_features, comparison_features)
    return _deterministic_subset_polynomial_mmd(source_features, comparison_features)


def _formal_metric_rows(
    source_features: Any,
    comparison_features: Any,
    *,
    sample_pair_count: int,
    formal_min_sample_count: int,
) -> list[dict[str, Any]]:
    """构造正式 FID / KID 指标行。

    通用工程写法是把正式特征后端、样本规模和数值后端的边界集中在该函数中。
    项目特定约束是: 小样本链路即使提供了特征记录, 也只能输出受治理的阻断状态, 不能被提升为投稿级结论。
    """

    source_array = _feature_array_or_none(source_features)
    comparison_array = _feature_array_or_none(comparison_features)
    source_count = int(source_array.shape[0]) if source_array is not None else sample_pair_count
    comparison_count = int(comparison_array.shape[0]) if comparison_array is not None else sample_pair_count
    metric_context = {
        "feature_backend": FORMAL_FEATURE_BACKEND,
        "source_image_count": source_count,
        "comparison_image_count": comparison_count,
        "sample_pair_count": sample_pair_count,
        "supports_paper_claim": False,
    }
    progress_started_at = time.monotonic()

    def metric_progress(profile: str) -> None:
        """输出当前正式质量指标重建位置。"""

        _emit_metric_progress(
            started_at=progress_started_at,
            completed=1,
            total=4,
            profile=profile,
        )

    if source_array is None or comparison_array is None:
        metric_status = FORMAL_FID_KID_BLOCKER
        fid_value: str | float = "unsupported"
        kid_value: str | float = "unsupported"
    elif (
        source_array.shape != comparison_array.shape
        or source_array.shape[0] < formal_min_sample_count
        or comparison_array.shape[0] < formal_min_sample_count
    ):
        metric_status = FORMAL_FID_KID_SAMPLE_BLOCKER
        fid_value = "unsupported"
        kid_value = "unsupported"
    else:
        metric_progress("formal_fid_start")
        fid_result = _exact_gaussian_fid(source_array, comparison_array, progress=metric_progress)
        if fid_result is None:
            metric_status = FORMAL_FID_KID_NUMERIC_BLOCKER
            fid_value = "unsupported"
            kid_value = "unsupported"
            _emit_metric_progress(
                started_at=progress_started_at,
                completed=4,
                total=4,
                profile="formal_metrics_numeric_blocker",
            )
        else:
            _emit_metric_progress(
                started_at=progress_started_at,
                completed=3,
                total=4,
                profile="formal_kid_start",
            )
            metric_status = "measured"
            fid_value = fid_result
            kid_value = _unbiased_polynomial_mmd(source_array, comparison_array)
            _emit_metric_progress(
                started_at=progress_started_at,
                completed=4,
                total=4,
                profile="formal_metrics_done",
            )
    return [
        {
            "quality_metric_name": "fid",
            "quality_metric_value": fid_value,
            "metric_status": metric_status,
            "paper_metric_name": "fid",
            **metric_context,
        },
        {
            "quality_metric_name": "kid",
            "quality_metric_value": kid_value,
            "metric_status": metric_status,
            "paper_metric_name": "kid",
            **metric_context,
        },
    ]


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
    image_search_roots: Iterable[Path] = (),
    formal_source_features: Any = None,
    formal_comparison_features: Any = None,
    formal_min_sample_count: int = 50,
) -> list[dict[str, Any]]:
    """构造数据集级质量指标表。

    表中同时保留正式 FID / KID 的 unsupported 行和小样本 proxy 行, 用于明确统计边界。
    """

    record_values = tuple(records)
    image_root_values = tuple(Path(path).resolve() for path in image_search_roots)
    source_paths = tuple(
        _resolve_existing_image_path(record.source_image_path, root_path, image_root_values)
        for record in record_values
    )
    comparison_paths = tuple(
        _resolve_existing_image_path(record.comparison_image_path, root_path, image_root_values)
        for record in record_values
    )
    missing_image_file_count = sum(1 for path in source_paths + comparison_paths if not path.is_file())
    source_count = len(source_paths)
    comparison_count = len(comparison_paths)
    pixel_metric_context = {
        "feature_backend": PIXEL_FEATURE_BACKEND,
        "source_image_count": source_count,
        "comparison_image_count": comparison_count,
        "sample_pair_count": len(record_values),
        "supports_paper_claim": False,
    }
    rows = _formal_metric_rows(
        formal_source_features,
        formal_comparison_features,
        sample_pair_count=len(record_values),
        formal_min_sample_count=formal_min_sample_count,
    )
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
                    **pixel_metric_context,
                },
                {
                    "quality_metric_name": "kid_pixel_feature_proxy",
                    "quality_metric_value": "unsupported",
                    "metric_status": "image_file_missing",
                    "paper_metric_name": "kid",
                    **pixel_metric_context,
                },
            ]
        )
        return rows
    pixel_started_at = time.monotonic()
    pixel_total = len(source_paths) + len(comparison_paths)
    _emit_metric_progress(
        started_at=pixel_started_at,
        completed=0,
        total=pixel_total,
        profile="pixel_proxy_start",
    )
    source_features = []
    for source_index, path in enumerate(source_paths, start=1):
        source_features.append(extract_pixel_histogram_feature(path))
        completed = source_index
        if _should_emit_metric_item_progress(completed, pixel_total):
            _emit_metric_progress(
                started_at=pixel_started_at,
                completed=completed,
                total=pixel_total,
                profile=f"pixel_proxy_source={source_index}/{len(source_paths)}",
            )
    comparison_features = []
    for comparison_index, path in enumerate(comparison_paths, start=1):
        comparison_features.append(extract_pixel_histogram_feature(path))
        completed = len(source_paths) + comparison_index
        if _should_emit_metric_item_progress(completed, pixel_total):
            _emit_metric_progress(
                started_at=pixel_started_at,
                completed=completed,
                total=pixel_total,
                profile=f"pixel_proxy_comparison={comparison_index}/{len(comparison_paths)}",
            )
    source_array = np.vstack(source_features)
    comparison_array = np.vstack(comparison_features)
    rows.extend(
        [
            {
                "quality_metric_name": "fid_pixel_feature_proxy",
                "quality_metric_value": _diagonal_gaussian_fid(source_array, comparison_array),
                "metric_status": "measured_small_sample_proxy",
                "paper_metric_name": "fid",
                **pixel_metric_context,
            },
            {
                "quality_metric_name": "kid_pixel_feature_proxy",
                "quality_metric_value": _biased_polynomial_mmd(source_array, comparison_array),
                "metric_status": "measured_small_sample_proxy",
                "paper_metric_name": "kid",
                **pixel_metric_context,
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
    measured_formal_metric_names = {
        str(row.get("quality_metric_name"))
        for row in rows
        if str(row.get("quality_metric_name")) in {"fid", "kid"} and str(row.get("metric_status")) == "measured"
    }
    formal_ready = measured_formal_metric_names == {"fid", "kid"}
    formal_status_values = tuple(
        str(row.get("metric_status"))
        for row in rows
        if str(row.get("quality_metric_name")) in {"fid", "kid"} and str(row.get("metric_status"))
    )
    if formal_ready:
        formal_fid_kid_claim_blocker = ""
        dataset_quality_claim_boundary = "formal_fid_kid_measured_but_paper_claim_requires_evidence_closure"
    elif not formal_status_values:
        formal_fid_kid_claim_blocker = "formal_fid_kid_metric_rows_missing"
        dataset_quality_claim_boundary = "formal_feature_backend_missing_for_dataset_quality_claim"
    else:
        formal_fid_kid_claim_blocker = next(
            (
                status
                for status in formal_status_values
                if status
                in {
                    FORMAL_FID_KID_BLOCKER,
                    FORMAL_FID_KID_SAMPLE_BLOCKER,
                    FORMAL_FID_KID_NUMERIC_BLOCKER,
                }
            ),
            next((status for status in formal_status_values if status != "measured"), "formal_fid_kid_not_measured"),
        )
        dataset_quality_claim_boundary = (
            "formal_feature_backend_missing_for_dataset_quality_claim"
            if formal_fid_kid_claim_blocker == FORMAL_FID_KID_BLOCKER
            else "formal_feature_backend_ready_but_formal_fid_kid_blocked"
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
        "formal_fid_kid_metric_names_ready": formal_ready,
        "formal_fid_kid_claim_gate_ready": formal_ready,
        "formal_fid_kid_claim_blocker": formal_fid_kid_claim_blocker,
        "dataset_quality_proxy_only": proxy_ready and not formal_ready,
        "dataset_quality_claim_boundary": dataset_quality_claim_boundary,
        "paper_claim_ready": False,
        "unsupported_reason": formal_fid_kid_claim_blocker if not formal_ready else "",
        "supports_paper_claim": False,
    }
