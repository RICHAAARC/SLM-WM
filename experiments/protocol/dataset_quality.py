"""数据集级图像质量证据的轻量治理协议。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
import os
from pathlib import Path
import time
from typing import Any, Iterable, Mapping

import numpy as np
from main.core.digest import build_stable_digest

FORMAL_FEATURE_BACKEND = "inception_feature_backend"
FORMAL_FEATURE_EXTRACTOR_ID = "torch_fidelity_0_4_0_inception_v3_compat_2048"
FORMAL_DATASET_QUALITY_ATTACK_NAME = "watermark_embedding"
FORMAL_DATASET_QUALITY_IMAGE_PAIR_ROLE = "clean_to_watermarked"
FORMAL_DATASET_QUALITY_METRIC_NAMES = ("fid", "kid_mean", "kid_std")
FORMAL_FID_KID_BLOCKER = "requires_inception_feature_backend"
FORMAL_FID_KID_SAMPLE_BLOCKER = "requires_full_main_sample_scale"
FORMAL_FID_KID_NUMERIC_BLOCKER = "requires_covariance_square_root_backend"
FORMAL_FID_CPU_MAX_FEATURE_DIM = 512
FORMAL_KID_EXACT_MAX_SAMPLE_COUNT = 1000
FORMAL_KID_SUBSET_COUNT = 100
FORMAL_KID_SUBSET_SIZE = 1000
FORMAL_KID_RNG_SEED = 2020
SAFE_EINSUM_KERNEL_MAX_OPERATIONS = 20_000_000
SAFE_ELEMENTWISE_FID_CROSS_MAX_OPERATIONS = 20_000_000
LOW_RANK_JACOBI_MAX_SAMPLE_COUNT = 128
LOW_RANK_JACOBI_RELATIVE_TOLERANCE = 1e-13


def formal_dataset_quality_metric_protocol() -> dict[str, Any]:
    """返回正式 FID/KID 数学算子的冻结身份。

    该记录将特征提取器与统计估计器分开声明。这样生成侧和 CPU 结果闭合侧
    可以在不信任环境变量的前提下复算同一公式。KID 先按冻结的小端 float64
    行摘要构造项目 canonical population, 再执行 torch-fidelity v0.4.0 同参数
    的均匀无放回抽样。冻结子集大小1000是上限; 实际子集大小同时受两侧
    样本数量约束, 从而让70/700/7000三档规模使用同一估计规则。任意外部
    文件行顺序只有先执行同一 canonicalization, 才能复现项目冻结数值。
    """

    payload = {
        "feature_backend": FORMAL_FEATURE_BACKEND,
        "feature_extractor_id": FORMAL_FEATURE_EXTRACTOR_ID,
        "feature_dimension": 2048,
        "fid_numeric_dtype": "float64",
        "fid_covariance_denominator": "sample_count_minus_one",
        "fid_covariance_square_root": (
            "adaptive_exact_low_rank_svd_or_symmetric_psd_eigendecomposition"
        ),
        "fid_low_rank_trace_identity": "nuclear_norm_centered_cross_gram",
        "fid_small_sample_svd_backend": "one_sided_jacobi_float64",
        "fid_small_sample_jacobi_max_count": LOW_RANK_JACOBI_MAX_SAMPLE_COUNT,
        "fid_small_sample_jacobi_relative_tolerance": (
            LOW_RANK_JACOBI_RELATIVE_TOLERANCE
        ),
        "kid_estimator": "unbiased_polynomial_mmd",
        "kid_polynomial_degree": 3,
        "kid_polynomial_gamma": "inverse_feature_dimension",
        "kid_polynomial_coefficient": 1.0,
        "kid_subset_sampling": "uniform_without_replacement",
        "kid_population_order": (
            "sha256_little_endian_float64_c_order_feature_row_bytes"
        ),
        "kid_subset_count": FORMAL_KID_SUBSET_COUNT,
        "kid_subset_size": FORMAL_KID_SUBSET_SIZE,
        "kid_effective_subset_size_rule": (
            "minimum_configured_size_source_count_comparison_count"
        ),
        "kid_full_sample_u_statistic_equivalence": True,
        "kid_rng_seed": FORMAL_KID_RNG_SEED,
        "kid_exact_max_sample_count": FORMAL_KID_EXACT_MAX_SAMPLE_COUNT,
        "kid_reported_statistics": ["mean", "std"],
        "kid_subset_std_ddof": 0,
        "kid_subset_std_semantics": (
            "population_standard_deviation_across_subset_mmd_estimates"
        ),
        "kid_subset_std_is_standard_error": False,
        "kid_output_scale": 1.0,
        "kid_full_sample_subset_std": 0.0,
        "kid_effective_subset_size_by_paper_run": {
            "probe_paper": 70,
            "pilot_paper": 700,
            "full_paper": 1000,
        },
    }
    return {
        **payload,
        "formal_metric_protocol_digest": build_stable_digest(payload),
    }


@dataclass(frozen=True)
class DatasetQualityImageRecord:
    """记录一组 source / comparison 图像对进入数据集级质量协议的事实。

    该对象属于通用工程写法: 它只记录图像路径、摘要、配对角色和特征后端, 不在记录层直接声明正式
    FID / KID 结论。正式论文级 FID / KID 由后续 Inception 特征导入和样本规模门禁决定。
    """

    dataset_quality_record_id: str
    dataset_quality_record_digest: str
    run_id: str
    prompt_id: str
    attack_name: str
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
    if operation_count <= SAFE_EINSUM_KERNEL_MAX_OPERATIONS:
        # 小样本使用 NumPy 自身的确定性张量收缩, 避免宿主进程加载
        # PyTorch 后再次初始化系统 BLAS 线程池。该分支不改变核公式。
        dot_products = np.einsum(
            "id,jd->ij",
            values_a,
            values_b,
            optimize=False,
        )
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


def _one_sided_jacobi_singular_value_sum(matrix: np.ndarray) -> float:
    """用单边 Jacobi 正交化计算小矩阵全部奇异值之和。

    该实现仅用于样本数不超过128的 FID 样本空间矩阵。它只执行 float64
    向量内积和二维旋转，不调用宿主 BLAS/LAPACK 线程池；因此可在同一进程
    已加载 PyTorch 的 Windows 审计环境中稳定复算。Jacobi 旋转计算的是完整
    奇异值集合，不是截断、随机化或近似低秩估计。
    """

    rotated = np.asarray(matrix, dtype=np.float64).copy()
    if rotated.ndim != 2 or min(rotated.shape) == 0:
        raise ValueError("Jacobi SVD 需要非空二维矩阵")
    column_count = int(rotated.shape[1])
    relative_tolerance = LOW_RANK_JACOBI_RELATIVE_TOLERANCE
    maximum_sweeps = max(12, column_count)
    converged = False
    for _ in range(maximum_sweeps):
        rotation_count = 0
        for left_index in range(column_count - 1):
            for right_index in range(left_index + 1, column_count):
                left = rotated[:, left_index].copy()
                right = rotated[:, right_index].copy()
                left_energy = float(np.sum(left * left))
                right_energy = float(np.sum(right * right))
                cross_energy = float(np.sum(left * right))
                scale = math.sqrt(max(left_energy * right_energy, 0.0))
                if scale == 0.0 or abs(cross_energy) <= relative_tolerance * scale:
                    continue
                tau = (right_energy - left_energy) / (2.0 * cross_energy)
                tangent = math.copysign(
                    1.0 / (abs(tau) + math.sqrt(1.0 + tau * tau)),
                    tau,
                )
                cosine = 1.0 / math.sqrt(1.0 + tangent * tangent)
                sine = tangent * cosine
                rotated[:, left_index] = cosine * left - sine * right
                rotated[:, right_index] = sine * left + cosine * right
                rotation_count += 1
        if rotation_count == 0:
            converged = True
            break
    if not converged:
        maximum_relative_cross_energy = 0.0
        for left_index in range(column_count - 1):
            for right_index in range(left_index + 1, column_count):
                left = rotated[:, left_index]
                right = rotated[:, right_index]
                scale = math.sqrt(
                    max(
                        float(np.sum(left * left))
                        * float(np.sum(right * right)),
                        0.0,
                    )
                )
                if scale > 0.0:
                    maximum_relative_cross_energy = max(
                        maximum_relative_cross_energy,
                        abs(float(np.sum(left * right))) / scale,
                    )
        if maximum_relative_cross_energy > relative_tolerance * 10.0:
            raise ArithmeticError("小样本 Jacobi SVD 未达到冻结正交收敛阈值")
    return float(
        sum(
            math.sqrt(max(float(np.sum(rotated[:, index] ** 2)), 0.0))
            for index in range(column_count)
        )
    )


def _low_rank_gaussian_fid(
    source_features: np.ndarray,
    comparison_features: np.ndarray,
    *,
    progress: Any = None,
) -> float:
    """用样本空间低秩恒等式精确计算高维 FID。

    设中心化并按样本协方差分母缩放后的特征矩阵为 ``A`` 与 ``B``，则
    ``Cov(A)=A.T@A``、``Cov(B)=B.T@B``，FID 的协方差交叉迹严格等于
    ``A@B.T`` 的核范数。因而当样本数小于特征维数时，只需对样本空间矩阵
    做 SVD，不需要构造或近似2048×2048协方差矩阵。该变换保持标准 FID
    数学值不变，并使 probe/pilot 的 CPU 独立复算能够实际完成。
    """

    source_mean = source_features.mean(axis=0)
    comparison_mean = comparison_features.mean(axis=0)
    source_denominator = max(int(source_features.shape[0]) - 1, 1)
    comparison_denominator = max(int(comparison_features.shape[0]) - 1, 1)
    source_scaled = (
        source_features - source_mean
    ) / math.sqrt(source_denominator)
    comparison_scaled = (
        comparison_features - comparison_mean
    ) / math.sqrt(comparison_denominator)
    if progress is not None:
        progress("fid_low_rank_cross_gram")
    cross_operation_count = (
        int(source_scaled.shape[0])
        * int(comparison_scaled.shape[0])
        * int(source_scaled.shape[1])
    )
    if cross_operation_count <= SAFE_ELEMENTWISE_FID_CROSS_MAX_OPERATIONS:
        # 小样本使用 NumPy 自身的确定性张量收缩, 避免宿主进程已经加载
        # PyTorch 时再次初始化系统 BLAS 线程池。较大样本仍使用矩阵乘法。
        cross_gram = np.einsum(
            "id,jd->ij",
            source_scaled,
            comparison_scaled,
            optimize=False,
        )
    else:
        cross_gram = source_scaled @ comparison_scaled.T
    if progress is not None:
        progress("fid_low_rank_singular_values")
    if max(cross_gram.shape) <= LOW_RANK_JACOBI_MAX_SAMPLE_COUNT:
        covariance_sqrt_trace = _one_sided_jacobi_singular_value_sum(
            cross_gram
        )
    else:
        covariance_sqrt_trace = np.linalg.svd(
            cross_gram,
            compute_uv=False,
        ).sum()
    mean_term = np.sum((source_mean - comparison_mean) ** 2)
    covariance_term = (
        np.sum(source_scaled**2)
        + np.sum(comparison_scaled**2)
        - 2.0 * covariance_sqrt_trace
    )
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
    """按矩阵秩自适应选择数学等价的精确 FID 算法。

    当样本空间不大于特征空间时使用低秩核范数恒等式；否则使用完整协方差
    的对称半正定分解。两条分支计算同一个标准 FID，不属于近似或代理指标。
    """

    feature_dim = int(source_features.shape[1])
    if max(source_features.shape[0], comparison_features.shape[0]) <= feature_dim:
        try:
            return _low_rank_gaussian_fid(
                source_features,
                comparison_features,
                progress=progress,
            )
        except (ArithmeticError, ValueError):
            return None
    torch_value = _torch_gaussian_fid(
        source_features,
        comparison_features,
        progress=progress,
        allow_high_dimensional_cpu=allow_high_dimensional_cpu,
    )
    if torch_value is not None:
        return torch_value
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


def _effective_kid_subset_size(
    source_count: int,
    comparison_count: int,
    configured_subset_size: int = FORMAL_KID_SUBSET_SIZE,
) -> int:
    """按两侧样本数量解析正式 KID 的实际子集大小。

    `torch-fidelity` 要求调用时的 `kid_subset_size` 不超过任一输入集合。
    因此1000是冻结上限, 实际值为该上限与两侧样本数量的最小值。对
    probe 和 pilot 而言, 每轮无放回抽样都会选中完整集合, 其均值严格等于
    一次完整无偏 U-statistic; full 则继续执行100个大小为1000的随机子集。
    """

    if source_count < 2 or comparison_count < 2:
        raise ValueError("KID 两侧样本数量都必须至少为2")
    if configured_subset_size < 2:
        raise ValueError("KID 配置子集大小必须至少为2")
    return min(configured_subset_size, source_count, comparison_count)


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
) -> tuple[float, float]:
    """返回冻结随机子集 KID 的均值与总体标准差。

    该实现遵循标准 KID 的随机子集 U-statistic 定义。完整特征行摘要仅用于
    消除 JSONL 物化顺序差异, 不参与样本入选概率;每个样本在每轮子集中仍
    具有相同的无放回抽样概率。标准差使用 ``ddof=0``, 表示各子集估计值的
    总体离散程度, 不是均值的标准误或置信区间。
    """

    if subset_count <= 0 or subset_size < 2:
        raise ValueError("KID 子集数量必须为正数且子集大小至少为2")

    # KID 是分布指标,结果不应依赖 feature record 的物化顺序.每行先转换为
    # little-endian float64 的 C-order 字节,再按完整 SHA-256 摘要排序.该规则
    # 同时消除宿主字节序差异,并保持每轮均匀无放回抽样的入选概率不变.
    def digest_order(features: np.ndarray) -> np.ndarray:
        """返回由完整特征行摘要确定的稳定顺序。"""

        rows = np.ascontiguousarray(features, dtype=np.dtype("<f8"))
        digests = np.asarray(
            [hashlib.sha256(row.tobytes(order="C")).digest() for row in rows],
            dtype="S32",
        )
        return np.argsort(digests, kind="stable")

    source_order = digest_order(source_features)
    comparison_order = digest_order(comparison_features)
    source_features = source_features[source_order]
    comparison_features = comparison_features[comparison_order]
    resolved_subset_size = _effective_kid_subset_size(
        int(source_features.shape[0]),
        int(comparison_features.shape[0]),
        subset_size,
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
    value_array = np.asarray(values, dtype=np.float64)
    return (
        float(np.mean(value_array)),
        float(np.std(value_array, ddof=0)),
    )


def _unbiased_polynomial_mmd(
    source_features: np.ndarray,
    comparison_features: np.ndarray,
) -> tuple[float, float]:
    """返回正式 KID 的子集均值与总体标准差。"""

    max_count = max(int(source_features.shape[0]), int(comparison_features.shape[0]))
    if max_count <= _kid_exact_max_sample_count():
        # 当实际子集覆盖完整集合时, 每轮无放回抽样只改变行排列, 不改变
        # U-statistic 数值.因此总体标准差按冻结协议显式记录为0.
        return (
            _unbiased_polynomial_mmd_exact(
                source_features,
                comparison_features,
            ),
            0.0,
        )
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
        kid_mean_value: str | float = "unsupported"
        kid_std_value: str | float = "unsupported"
    elif (
        source_array.shape != comparison_array.shape
        or source_array.shape[0] < formal_min_sample_count
        or comparison_array.shape[0] < formal_min_sample_count
    ):
        metric_status = FORMAL_FID_KID_SAMPLE_BLOCKER
        fid_value = "unsupported"
        kid_mean_value = "unsupported"
        kid_std_value = "unsupported"
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
            kid_mean_value = "unsupported"
            kid_std_value = "unsupported"
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
            kid_mean_value, kid_std_value = _unbiased_polynomial_mmd(
                source_array,
                comparison_array,
            )
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
            "quality_metric_name": "kid_mean",
            "quality_metric_value": kid_mean_value,
            "metric_status": metric_status,
            "paper_metric_name": "kid_mean",
            **metric_context,
        },
        {
            "quality_metric_name": "kid_std",
            "quality_metric_value": kid_std_value,
            "metric_status": metric_status,
            "paper_metric_name": "kid_std",
            **metric_context,
        },
    ]


def build_dataset_quality_image_records(
    registry_rows: Iterable[Mapping[str, Any]],
    root_path: Path,
) -> tuple[DatasetQualityImageRecord, ...]:
    """从主方法 clean/watermarked registry 构造数据集级质量记录。"""

    records: list[DatasetQualityImageRecord] = []
    resolved_source_paths: set[str] = set()
    resolved_comparison_paths: set[str] = set()
    run_ids: set[str] = set()
    for index, row in enumerate(registry_rows):
        run_id = str(row.get("run_id", "") or "")
        prompt_id = str(row.get("prompt_id", "") or "")
        attack_name = str(row.get("attack_name", "") or "")
        image_pair_role = str(row.get("image_pair_role", "") or "")
        if (
            not run_id
            or run_id in run_ids
            or not prompt_id
            or attack_name != FORMAL_DATASET_QUALITY_ATTACK_NAME
            or image_pair_role != FORMAL_DATASET_QUALITY_IMAGE_PAIR_ROLE
            or row.get("supports_paper_claim") is not False
        ):
            raise ValueError(
                "数据集质量 registry 必须是唯一 run_id 的正式 clean-to-watermarked 图像对"
            )
        source_path = _row_path(root_path, row, "source_image_path")
        comparison_path = _row_path(root_path, row, "attacked_image_path")
        source_identity = os.path.normcase(str(source_path.resolve()))
        comparison_identity = os.path.normcase(str(comparison_path.resolve()))
        if (
            source_identity == comparison_identity
            or source_identity in resolved_source_paths
            or comparison_identity in resolved_comparison_paths
            or source_identity in resolved_comparison_paths
            or comparison_identity in resolved_source_paths
        ):
            raise ValueError(
                "正式质量样本要求 source/comparison 实际文件路径逐对不同、角色内唯一且跨角色不相交"
            )
        source_digest = _row_digest(row, "source_image_digest", source_path)
        comparison_digest = _row_digest(row, "attacked_image_digest", comparison_path)
        payload = {
            "run_id": run_id,
            "prompt_id": prompt_id,
            "attack_name": attack_name,
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
                run_id=run_id,
                prompt_id=prompt_id,
                attack_name=attack_name,
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
        run_ids.add(run_id)
        resolved_source_paths.add(source_identity)
        resolved_comparison_paths.add(comparison_identity)
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
        if str(row.get("quality_metric_name"))
        in FORMAL_DATASET_QUALITY_METRIC_NAMES
        and str(row.get("metric_status")) == "measured"
    }
    formal_ready = measured_formal_metric_names == set(
        FORMAL_DATASET_QUALITY_METRIC_NAMES
    )
    formal_status_values = tuple(
        str(row.get("metric_status"))
        for row in rows
        if str(row.get("quality_metric_name"))
        in FORMAL_DATASET_QUALITY_METRIC_NAMES
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
    metric_protocol = formal_dataset_quality_metric_protocol()
    kid_effective_subset_size = (
        _effective_kid_subset_size(
            len(record_values),
            len(record_values),
        )
        if len(record_values) >= 2
        else 0
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
        "formal_metric_protocol": metric_protocol,
        "formal_metric_protocol_digest": metric_protocol[
            "formal_metric_protocol_digest"
        ],
        "kid_effective_subset_size": kid_effective_subset_size,
        "formal_fid_kid_ready": formal_ready,
        "formal_fid_kid_metric_names_ready": formal_ready,
        "formal_fid_kid_component_ready": formal_ready,
        "formal_fid_kid_component_blocker": blocker,
        "dataset_quality_claim_boundary": claim_boundary,
        "paper_claim_ready": False,
        "unsupported_reason": blocker if not formal_ready else "",
        "supports_paper_claim": False,
    }
