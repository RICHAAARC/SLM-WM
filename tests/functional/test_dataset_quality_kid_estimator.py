"""验证正式 KID 保持无偏 U-statistic 的有符号取值."""

from __future__ import annotations

import numpy as np
import pytest

from experiments.protocol.dataset_quality import _unbiased_polynomial_mmd_exact


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
