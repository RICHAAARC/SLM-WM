"""验证正式 KID 保持无偏 U-statistic 的有符号取值."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from experiments.protocol.dataset_quality import (
    FORMAL_DATASET_QUALITY_METRIC_NAMES,
    _effective_kid_subset_size,
    _formal_random_subset_polynomial_mmd,
    _low_rank_gaussian_fid,
    _unbiased_polynomial_mmd,
    _unbiased_polynomial_mmd_exact,
    formal_dataset_quality_metric_protocol,
)


pytestmark = pytest.mark.quick


def _reference_unbiased_polynomial_mmd(
    source: np.ndarray,
    comparison: np.ndarray,
) -> float:
    """用独立直译公式计算单个 KID 无偏 U-statistic。"""

    feature_dimension = int(source.shape[1])

    def kernel(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        """计算 torch-fidelity v0.4.0 默认三阶多项式核。"""

        dot_products = np.sum(
            left[:, None, :] * right[None, :, :],
            axis=2,
        )
        return (dot_products / feature_dimension + 1.0) ** 3

    source_kernel = kernel(source, source)
    comparison_kernel = kernel(comparison, comparison)
    cross_kernel = kernel(source, comparison)
    sample_count = int(source.shape[0])
    denominator = sample_count * (sample_count - 1)
    return float(
        (source_kernel.sum() - np.trace(source_kernel)) / denominator
        + (comparison_kernel.sum() - np.trace(comparison_kernel))
        / denominator
        - 2.0 * cross_kernel.mean()
    )


def _reference_random_subset_kid(
    source: np.ndarray,
    comparison: np.ndarray,
    *,
    subset_count: int,
    subset_size: int,
    rng_seed: int,
) -> tuple[float, float]:
    """独立复现 canonical population 上的 KID 子集均值与标准差。"""

    def canonical_rows(features: np.ndarray) -> np.ndarray:
        """按冻结 little-endian float64 行摘要构造 canonical population。"""

        rows = np.ascontiguousarray(features, dtype=np.dtype("<f8"))
        order = sorted(
            range(len(rows)),
            key=lambda index: hashlib.sha256(
                rows[index].tobytes(order="C")
            ).digest(),
        )
        return rows[np.asarray(order, dtype=np.int64)]

    source_rows = canonical_rows(source)
    comparison_rows = canonical_rows(comparison)
    random_state = np.random.RandomState(rng_seed)
    values = []
    for _ in range(subset_count):
        source_indices = random_state.choice(
            len(source_rows), subset_size, replace=False
        )
        comparison_indices = random_state.choice(
            len(comparison_rows), subset_size, replace=False
        )
        values.append(
            _reference_unbiased_polynomial_mmd(
                source_rows[source_indices],
                comparison_rows[comparison_indices],
            )
        )
    value_array = np.asarray(values, dtype=np.float64)
    return float(value_array.mean()), float(value_array.std(ddof=0))


def test_formal_kid_protocol_uses_reference_random_subset_settings() -> None:
    """正式 KID 必须冻结参考子集参数及样本规模自适应规则。"""

    protocol = formal_dataset_quality_metric_protocol()

    assert protocol["kid_estimator"] == "unbiased_polynomial_mmd"
    assert protocol["kid_subset_sampling"] == "uniform_without_replacement"
    assert protocol["kid_population_order"] == (
        "sha256_little_endian_float64_c_order_feature_row_bytes"
    )
    assert protocol["kid_subset_count"] == 100
    assert protocol["kid_subset_size"] == 1000
    assert protocol["kid_effective_subset_size_rule"] == (
        "minimum_configured_size_source_count_comparison_count"
    )
    assert protocol["kid_full_sample_u_statistic_equivalence"] is True
    assert protocol["kid_rng_seed"] == 2020
    assert protocol["kid_reported_statistics"] == ["mean", "std"]
    assert protocol["kid_subset_std_ddof"] == 0
    assert protocol["kid_subset_std_semantics"] == (
        "population_standard_deviation_across_subset_mmd_estimates"
    )
    assert protocol["kid_subset_std_is_standard_error"] is False
    assert protocol["kid_output_scale"] == 1.0
    assert protocol["kid_full_sample_subset_std"] == 0.0
    assert protocol["kid_effective_subset_size_by_paper_run"] == {
        "probe_paper": 70,
        "pilot_paper": 700,
        "full_paper": 1000,
    }
    assert FORMAL_DATASET_QUALITY_METRIC_NAMES == (
        "fid",
        "kid_mean",
        "kid_std",
    )
    assert protocol["fid_covariance_square_root"] == (
        "adaptive_exact_low_rank_svd_or_symmetric_psd_eigendecomposition"
    )
    assert protocol["fid_low_rank_trace_identity"] == (
        "nuclear_norm_centered_cross_gram"
    )
    assert protocol["fid_small_sample_svd_backend"] == (
        "one_sided_jacobi_float64"
    )
    assert protocol["fid_small_sample_jacobi_max_count"] == 128
    assert protocol["fid_small_sample_jacobi_relative_tolerance"] == 1e-13
    assert len(protocol["formal_metric_protocol_digest"]) == 64


@pytest.mark.parametrize(
    ("sample_count", "expected_subset_size"),
    ((70, 70), (700, 700), (7000, 1000)),
)
def test_formal_kid_effective_subset_size_tracks_paper_scale(
    sample_count: int,
    expected_subset_size: int,
) -> None:
    """三档论文规模只改变实际样本子集大小, 不改变 KID 估计规则。"""

    assert _effective_kid_subset_size(sample_count, sample_count) == (
        expected_subset_size
    )


def test_low_rank_fid_is_exactly_equivalent_to_full_covariance_formula() -> None:
    """正交协方差的样本空间 SVD 必须得到可手算的完整 FID。"""

    source = np.asarray(((-1.0, 0.0), (1.0, 0.0)), dtype=np.float64)
    comparison = np.asarray(((1.0, -2.0), (1.0, 2.0)), dtype=np.float64)

    # 两组均值距离平方为1，样本协方差分别为 diag(2, 0) 与
    # diag(0, 8)。交叉平方根迹为0，因此完整 FID 为1+2+8=11。
    assert _low_rank_gaussian_fid(source, comparison) == pytest.approx(
        11.0,
        abs=1e-12,
    )


def test_unbiased_kid_preserves_legitimate_negative_estimate() -> None:
    """有限样本无偏 KID 可以为负, 正式实现不得把它截断为0."""

    source = np.asarray(
        (
            (0.1257302211, -0.1321048633, 0.6404226504),
            (0.1049001172, -0.5356693732, 0.3615950549),
        ),
        dtype=np.float64,
    )
    comparison = np.asarray(
        (
            (1.3040000451, 0.9470809631, -0.7037352358),
            (-1.2654214710, -0.6232744625, 0.0413259793),
        ),
        dtype=np.float64,
    )

    value = _unbiased_polynomial_mmd_exact(source, comparison)

    assert value == pytest.approx(-0.2960922093, abs=1e-6)


def test_unbiased_kid_matches_independent_reference_formula() -> None:
    """正式单子集 KID 必须与独立直译的官方无偏公式一致。"""

    source = np.asarray(
        ((0.0, 1.0), (1.0, 0.0), (0.5, 0.25), (-0.5, 0.75)),
        dtype=np.float64,
    )
    comparison = np.asarray(
        ((0.2, 0.8), (0.9, -0.1), (0.4, 0.5), (-0.3, 0.6)),
        dtype=np.float64,
    )

    assert _unbiased_polynomial_mmd_exact(
        source,
        comparison,
    ) == pytest.approx(
        _reference_unbiased_polynomial_mmd(source, comparison),
        abs=1e-12,
    )


@pytest.mark.parametrize("sample_count", (70, 700, 1000))
def test_full_population_kid_reports_explicit_zero_std(
    monkeypatch: pytest.MonkeyPatch,
    sample_count: int,
) -> None:
    """完整集合子集只改变行排列, KID 标准差必须显式为0。"""

    monkeypatch.setattr(
        "experiments.protocol.dataset_quality._unbiased_polynomial_mmd_exact",
        lambda source, comparison: 0.125,
    )
    source = np.zeros((sample_count, 1), dtype=np.float64)
    comparison = np.ones((sample_count, 1), dtype=np.float64)

    assert _unbiased_polynomial_mmd(source, comparison) == (0.125, 0.0)


def test_random_subset_kid_mean_and_std_match_independent_reference() -> None:
    """随机子集 KID 的 mean/std 必须逐轮匹配独立参考实现。"""

    source = np.arange(51, dtype=np.float64).reshape(17, 3) / 13.0
    comparison = source * 0.85 + np.asarray((0.2, -0.1, 0.05))
    expected = _reference_random_subset_kid(
        source,
        comparison,
        subset_count=7,
        subset_size=5,
        rng_seed=2020,
    )

    actual = _formal_random_subset_polynomial_mmd(
        source,
        comparison,
        subset_count=7,
        subset_size=5,
        rng_seed=2020,
    )

    assert actual == pytest.approx(expected, abs=1e-12)
    assert actual[1] >= 0.0


def test_deterministic_subset_kid_is_feature_record_order_invariant(
) -> None:
    """冻结随机子集 KID 不得随 feature record 排列顺序变化."""

    source = np.arange(30, dtype=np.float64).reshape(10, 3) / 10.0
    comparison = source * 0.9 + 0.2

    expected = _formal_random_subset_polynomial_mmd(
        source,
        comparison,
        subset_count=3,
        subset_size=4,
    )
    permuted = _formal_random_subset_polynomial_mmd(
        source[[4, 1, 9, 0, 7, 3, 8, 2, 6, 5]],
        comparison[[8, 0, 5, 2, 9, 1, 6, 4, 7, 3]],
        subset_count=3,
        subset_size=4,
    )

    assert permuted == pytest.approx(expected, abs=1e-12)


def test_canonical_population_is_independent_of_input_byte_order() -> None:
    """同一数值集合的 big-endian 输入不得改变 canonical KID 结果。"""

    source = np.arange(36, dtype=np.float64).reshape(12, 3) / 9.0
    comparison = source * 0.95 - 0.1
    native = _formal_random_subset_polynomial_mmd(
        source,
        comparison,
        subset_count=5,
        subset_size=4,
    )
    big_endian = _formal_random_subset_polynomial_mmd(
        source.astype(">f8"),
        comparison.astype(">f8"),
        subset_count=5,
        subset_size=4,
    )

    assert big_endian == pytest.approx(native, abs=1e-12)
