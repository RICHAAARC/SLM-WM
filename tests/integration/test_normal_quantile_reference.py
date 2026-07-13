"""验证 Q20 标准正态表的完整 MPFR 正确舍入证书."""

from __future__ import annotations

import pytest

from main.core.normal_quantile_table import (
    NORMAL_QUANTILE_COUNT,
    NORMAL_QUANTILE_REFERENCE_VERIFICATION_DIGEST,
)
from tools.harness.verify_normal_quantile_reference import (
    verify_normal_quantile_reference,
)


@pytest.mark.integration
@pytest.mark.slow
def test_all_positive_quantile_entries_have_mpfr_rounding_certificate() -> None:
    """逐项复验正半轴表值,作为离线发布门禁而不进入默认测试路径."""

    pytest.importorskip("gmpy2")

    report = verify_normal_quantile_reference()

    assert report["normal_quantile_reference_verification_ready"] is True
    assert report["normal_quantile_reference_mismatch_count"] == 0
    assert report[
        "normal_quantile_reference_verified_positive_entry_count"
    ] == (NORMAL_QUANTILE_COUNT // 2)
    assert report["normal_quantile_reference_verification_digest"] == (
        NORMAL_QUANTILE_REFERENCE_VERIFICATION_DIGEST
    )
    assert report["supports_paper_claim"] is False
