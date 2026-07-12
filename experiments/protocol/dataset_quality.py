"""数据集级图像质量证据的轻量治理协议。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import os
from pathlib import Path
import time
from typing import Any, Iterable, Mapping

import numpy as np
from main.core.digest import build_stable_digest

FORMAL_FEATURE_BACKEND = "inception_feature_backend"
FORMAL_FEATURE_EXTRACTOR_ID = "torch_fidelity_0_4_0_inception_v3_compat_2048"
FORMAL_FID_KID_BLOCKER = "requires_inception_feature_backend"
FORMAL_FID_KID_SAMPLE_BLOCKER = "requires_full_main_sample_scale"
FORMAL_FID_KID_NUMERIC_BLOCKER = "requires_covariance_square_root_backend"
FORMAL_FID_CPU_MAX_FEATURE_DIM = 512
FORMAL_KID_EXACT_MAX_SAMPLE_COUNT = 1000
FORMAL_KID_SUBSET_COUNT = 100
FORMAL_KID_SUBSET_SIZE = 1000
FORMAL_KID_RNG_SEED = 2020
SAFE_ELEMENTWISE_KERNEL_MAX_OPERATIONS = 2_000_000


@dataclass(frozen=True)
class DatasetQualityImageRecord:
    """记录一组 source / comparison 图像对进入数据集级质量协议的事实。

    该对象属于通用工程写法: 它只记录图像路径、摘要、配对角色和特征后端, 不在记录层直接声明正式
    FID / KID 结论。正式论文级 FID / KID 由后续 Inception 特征导入和样本规模门禁决定。
    """

    dataset_quality_record_id: str
    dataset_quality_record_digest: str
    prompt_id: str
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


def _polynomial_kernel(values_a: np.ndarray, values_b: np.ndarray) -> np.ndarray:
    """计算正式 KID 使用的三阶多项式核矩阵。"""

    feature_dim = max(int(values_a.shape[1]), 1)
    operation_count = int(values_a.shape[0]) * int(values_b.shape[0]) * feature_dim
    if operation_count <= SAFE_ELEMENTWISE_KERNEL_MAX_OPERATIONS:
        dot_products = np.sum(values_a[:, None, :] * values_b[None, :, :], axis=2)
    else:
        dot_products = values_a @ values_b.T
    return (dot_products / feature_dim + 1.0) ** 3


def _metric_progress_enabled() -> bool:
    """判断是否输出数据集级质量指标重建进度。"""

    value = os.environ.get("SLM_WM_DATASET_QUALITY_METRIC_PROGRESS", "1").strip().lower()
    return value not in {"0", "false", "no", "off"}


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
    allow_high_dimensional_cpu: bool = False,
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
    if (
        not use_cuda
        and feature_dim > _fid_cpu_feature_dim_limit()
        and not allow_high_dimensional_cpu
    ):
        return None
    device = torch.device("cuda" if use_cuda else "cpu")
    # 正式指标统一使用 float64,使 Colab GPU 生成值与 CPU 独立重算值共享
    # 可审计的数值精度边界,避免设备差异被误判为证据漂移.
    dtype = torch.float64
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
    allow_high_dimensional_cpu: bool = False,
) -> float | None:
    """使用完整协方差矩阵计算 FID。

    正式 FID 基于 Inception 特征分布的均值与完整协方差。实现优先使用
    CUDA 上的对称特征分解; 在无 GPU 且特征维度过高时返回明确的数值后端
    阻断状态, 防止运行入口长时间无界阻塞。
    """

    torch_value = _torch_gaussian_fid(
        source_features,
        comparison_features,
        progress=progress,
        allow_high_dimensional_cpu=allow_high_dimensional_cpu,
    )
    if torch_value is not None:
        return torch_value
    feature_dim = int(source_features.shape[1])
    if feature_dim > _fid_cpu_feature_dim_limit() and not allow_high_dimensional_cpu:
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
    """返回冻结的 KID 完整核矩阵样本上限。"""

    return FORMAL_KID_EXACT_MAX_SAMPLE_COUNT


def _kid_subset_count() -> int:
    """返回冻结的 KID 随机子集数量。"""

    return FORMAL_KID_SUBSET_COUNT


def _kid_subset_size() -> int:
    """返回冻结的 KID 随机子集大小。"""

    return FORMAL_KID_SUBSET_SIZE


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
    # 无偏 KID 是有限样本 U-statistic, 合法估计值可能略小于0.
    # 此处不得截断, 否则会把正式估计量改成有偏的非标准指标.
    return float(source_term + comparison_term - 2.0 * cross_term)


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
            source_tensor = torch.as_tensor(source_features, dtype=torch.float64, device=device)
            comparison_tensor = torch.as_tensor(
                comparison_features,
                dtype=torch.float64,
                device=device,
            )
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
            value = (
                source_term + comparison_term - 2.0 * cross_term
            ).detach().cpu().item()
    except Exception:
        return None
    return float(value)


def _formal_random_subset_polynomial_mmd(
    source_features: np.ndarray,
    comparison_features: np.ndarray,
    *,
    subset_count: int = FORMAL_KID_SUBSET_COUNT,
    subset_size: int = FORMAL_KID_SUBSET_SIZE,
    rng_seed: int = FORMAL_KID_RNG_SEED,
) -> float:
    """使用冻结随机种子和均匀无放回子集估计大样本 KID。

    该实现遵循标准 KID 的随机子集 U-statistic 定义。完整特征行摘要仅用于
    消除 JSONL 物化顺序差异, 不参与样本入选概率;每个样本在每轮子集中仍
    具有相同的无放回抽样概率。
    """

    if subset_count <= 0 or subset_size < 2:
        raise ValueError("KID 子集数量必须为正数且子集大小至少为2")

    # KID 是分布指标,结果不应依赖 feature record 的物化顺序.使用完整
    # float64 行字节的 SHA-256 排序,可以在2048维和7000样本规模下以一次
    # 线性扫描确定顺序,避免按每个特征维度执行一次排序比较.
    def digest_order(features: np.ndarray) -> np.ndarray:
        """返回由完整特征行摘要确定的稳定顺序。"""

        rows = np.ascontiguousarray(features, dtype=np.float64)
        digests = np.asarray(
            [hashlib.sha256(row.tobytes()).digest() for row in rows],
            dtype="S32",
        )
        return np.argsort(digests, kind="stable")

    source_order = digest_order(source_features)
    comparison_order = digest_order(comparison_features)
    source_features = source_features[source_order]
    comparison_features = comparison_features[comparison_order]
    resolved_subset_size = min(
        subset_size,
        source_features.shape[0],
        comparison_features.shape[0],
    )
    random_state = np.random.RandomState(rng_seed)
    values: list[float] = []
    for _ in range(subset_count):
        source_indices = random_state.choice(
            source_features.shape[0],
            resolved_subset_size,
            replace=False,
        )
        comparison_indices = random_state.choice(
            comparison_features.shape[0],
            resolved_subset_size,
            replace=False,
        )
        values.append(
            _unbiased_polynomial_mmd_exact(
                source_features[source_indices],
                comparison_features[comparison_indices],
            )
        )
    return float(np.mean(values))


def _unbiased_polynomial_mmd(source_features: np.ndarray, comparison_features: np.ndarray) -> float:
    """使用三阶多项式核计算 KID 的无偏 MMD 形式。"""

    max_count = max(int(source_features.shape[0]), int(comparison_features.shape[0]))
    if max_count <= _kid_exact_max_sample_count():
        return _unbiased_polynomial_mmd_exact(source_features, comparison_features)
    return _formal_random_subset_polynomial_mmd(source_features, comparison_features)


def _formal_metric_rows(
    source_features: Any,
    comparison_features: Any,
    *,
    sample_pair_count: int,
    formal_min_sample_count: int,
    allow_high_dimensional_cpu: bool = False,
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
        fid_result = _exact_gaussian_fid(
            source_array,
            comparison_array,
            progress=metric_progress,
            allow_high_dimensional_cpu=allow_high_dimensional_cpu,
        )
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
        prompt_id = str(row.get("prompt_id", "") or "")
        payload = {
            "prompt_id": prompt_id,
            "image_pair_index": index,
            "image_pair_role": image_pair_role,
            "source_image_path": _as_relative_or_absolute(source_path, root_path),
            "source_image_digest": source_digest,
            "comparison_image_path": _as_relative_or_absolute(comparison_path, root_path),
            "comparison_image_digest": comparison_digest,
            "feature_backend": FORMAL_FEATURE_BACKEND,
            "supports_paper_claim": False,
        }
        digest = build_stable_digest(payload)
        records.append(
            DatasetQualityImageRecord(
                dataset_quality_record_id=f"dataset_quality_record_{digest[:16]}",
                dataset_quality_record_digest=digest,
                prompt_id=prompt_id,
                image_pair_index=index,
                image_pair_role=image_pair_role,
                source_image_path=payload["source_image_path"],
                source_image_digest=source_digest,
                comparison_image_path=payload["comparison_image_path"],
                comparison_image_digest=comparison_digest,
                feature_backend=FORMAL_FEATURE_BACKEND,
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
    """构造数据集级正式质量指标表。

    该函数只返回由正式 Inception 特征计算的 FID / KID 行。
    特征缺失、样本规模不足或数值后端不可用时返回明确阻断状态。
    """

    record_values = tuple(records)
    return _formal_metric_rows(
        formal_source_features,
        formal_comparison_features,
        sample_pair_count=len(record_values),
        formal_min_sample_count=formal_min_sample_count,
    )


def rebuild_formal_fid_kid_metric_rows(
    source_features: Any,
    comparison_features: Any,
    *,
    sample_pair_count: int,
) -> list[dict[str, Any]]:
    """从已物化的正式特征独立重算 FID 与 KID.

    该函数用于 CPU 结果闭合.与指标生成侧不同,结果闭合不能因特征维度较高而
    跳过数值计算,否则只能验证文件摘要,无法验证论文表中的指标值.调用方
    必须已经完成 feature record 的身份,角色,维度和有限值检查;此处只执行
    与正式生成路径相同的 FID / KID 数学算子,并要求完整样本规模.
    """

    if sample_pair_count <= 0:
        raise ValueError("正式 FID/KID 重算需要正样本对数量")
    return _formal_metric_rows(
        source_features,
        comparison_features,
        sample_pair_count=sample_pair_count,
        formal_min_sample_count=sample_pair_count,
        allow_high_dimensional_cpu=True,
    )


def build_dataset_quality_summary(
    records: Iterable[DatasetQualityImageRecord],
    metric_rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """聚合正式 Inception FID / KID 证据摘要。"""

    record_values = tuple(records)
    rows = tuple(metric_rows)
    measured_formal_metric_names = {
        str(row.get("quality_metric_name"))
        for row in rows
        if str(row.get("quality_metric_name")) in {"fid", "kid"}
        and str(row.get("metric_status")) == "measured"
    }
    formal_ready = measured_formal_metric_names == {"fid", "kid"}
    formal_status_values = tuple(
        str(row.get("metric_status"))
        for row in rows
        if str(row.get("quality_metric_name")) in {"fid", "kid"}
        and str(row.get("metric_status"))
    )
    if formal_ready:
        blocker = ""
        claim_boundary = "formal_fid_kid_measured_but_paper_claim_requires_evidence_closure"
    elif not formal_status_values:
        blocker = "formal_fid_kid_metric_rows_missing"
        claim_boundary = "formal_feature_backend_missing_for_dataset_quality_claim"
    else:
        blocker = next(
            (
                status
                for status in formal_status_values
                if status in {
                    FORMAL_FID_KID_BLOCKER,
                    FORMAL_FID_KID_SAMPLE_BLOCKER,
                    FORMAL_FID_KID_NUMERIC_BLOCKER,
                }
            ),
            next((status for status in formal_status_values if status != "measured"), "formal_fid_kid_not_measured"),
        )
        claim_boundary = (
            "formal_feature_backend_missing_for_dataset_quality_claim"
            if blocker == FORMAL_FID_KID_BLOCKER
            else "formal_feature_backend_ready_but_formal_fid_kid_blocked"
        )
    return {
        "construction_unit_name": "dataset_level_quality_evidence",
        "dataset_quality_record_count": len(record_values),
        "formal_quality_metric_count": len(rows),
        "source_image_count": len(record_values),
        "comparison_image_count": len(record_values),
        "sample_pair_count": len(record_values),
        "feature_backend": FORMAL_FEATURE_BACKEND,
        "formal_feature_backend": FORMAL_FEATURE_BACKEND,
        "primary_metric_backend": FORMAL_FEATURE_BACKEND,
        "formal_fid_kid_ready": formal_ready,
        "formal_fid_kid_metric_names_ready": formal_ready,
        "formal_fid_kid_claim_gate_ready": formal_ready,
        "formal_fid_kid_claim_blocker": blocker,
        "dataset_quality_claim_boundary": claim_boundary,
        "paper_claim_ready": False,
        "unsupported_reason": blocker if not formal_ready else "",
        "supports_paper_claim": False,
    }
