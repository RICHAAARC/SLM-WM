"""验证正式 KID 保持无偏 U-statistic 的有符号取值."""

from __future__ import annotations

import numpy as np
import pytest

from experiments.protocol.dataset_quality import (
    _formal_random_subset_polynomial_mmd,
    _unbiased_polynomial_mmd_exact,
)


pytestmark = pytest.mark.quick


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
