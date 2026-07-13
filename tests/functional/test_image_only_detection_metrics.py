"""验证仅图像检测原始指标重建."""

from __future__ import annotations

import math

import pytest

from experiments.artifacts.image_only_detection_metrics import (
    build_image_only_test_metric_rows,
)
from tests.helpers.formal_detection_record import bind_formal_detection_record


pytestmark = pytest.mark.quick


def detection_record(
    *,
    decision: bool,
    psnr: float | str,
    mse: float = 0.01,
) -> dict[str, object]:
    """构造单条 test split 检测记录."""

    return bind_formal_detection_record({
        "split": "test",
        "sample_role": "positive_source",
        "attack_family": "clean",
        "attack_name": "none",
        "resource_profile": "clean",
        "formal_evidence_positive": decision,
        "content_score": 0.8,
        "source_to_evaluated_ssim": 1.0,
        "source_to_evaluated_psnr": psnr,
        "source_to_evaluated_mse": mse,
    })


def test_metric_rows_preserve_exact_match_infinite_psnr_mean() -> None:
    """同组图像全部完全相同时, 正无穷 PSNR 均值具有明确数学语义."""

    rows = build_image_only_test_metric_rows(
        [detection_record(decision=True, psnr="inf", mse=0.0)],
        0.1,
    )

    assert rows[0]["positive_rate"] == 1.0
    assert math.isinf(rows[0]["source_to_evaluated_psnr_mean"])


def test_metric_rows_reject_non_boolean_formal_decision() -> None:
    """非空字符串不得被 Python truthiness 误解释为正式阳性."""

    record = detection_record(decision=True, psnr=30.0)
    record["formal_evidence_positive"] = "false"

    with pytest.raises(ValueError, match="必须是布尔值"):
        build_image_only_test_metric_rows([record], 0.1)


def test_metric_rows_reject_incomplete_ssim_coverage() -> None:
    """质量均值不得选择性跳过缺失 SSIM 的正式样本."""

    record = detection_record(decision=True, psnr=30.0)
    record.pop("source_to_evaluated_ssim")

    with pytest.raises(ValueError, match="每条.*SSIM"):
        build_image_only_test_metric_rows([record], 0.1)


@pytest.mark.parametrize(
    "field_name",
    (
        "content_score",
        "source_to_evaluated_ssim",
        "source_to_evaluated_psnr",
    ),
)
def test_metric_rows_reject_boolean_numeric_measurements(field_name: str) -> None:
    """布尔值虽然属于 Python 整数子类, 但不得充当任何正式测量值."""

    record = detection_record(decision=True, psnr=30.0)
    record[field_name] = True

    with pytest.raises(ValueError, match="不得使用布尔值"):
        build_image_only_test_metric_rows([record], 0.1)


def test_metric_rows_reject_boolean_target_fpr() -> None:
    """协议数值 target_fpr 也不得接受 Python 布尔值."""

    with pytest.raises(ValueError, match="target_fpr.*不得使用布尔值"):
        build_image_only_test_metric_rows(
            [detection_record(decision=True, psnr=30.0)],
            True,
        )


def test_metric_rows_reject_missing_psnr_coverage() -> None:
    """PSNR 均值必须覆盖聚合组中的每条正式 test record."""

    record = detection_record(decision=True, psnr=30.0)
    record.pop("source_to_evaluated_psnr")

    with pytest.raises(ValueError, match="每条.*PSNR"):
        build_image_only_test_metric_rows([record], 0.1)


@pytest.mark.parametrize("psnr", (math.nan, -math.inf))
def test_metric_rows_reject_invalid_non_finite_psnr(psnr: float) -> None:
    """NaN 与负无穷 PSNR 没有可用的正式聚合语义."""

    with pytest.raises(ValueError, match="PSNR.*有限或为有效正无穷"):
        build_image_only_test_metric_rows(
            [detection_record(decision=True, psnr=psnr)],
            0.1,
        )


def test_metric_rows_reject_infinite_psnr_without_zero_mse() -> None:
    """正无穷 PSNR 必须由同一记录中的零 MSE 证明图像完全一致."""

    with pytest.raises(ValueError, match="正无穷 PSNR.*MSE 为0"):
        build_image_only_test_metric_rows(
            [detection_record(decision=True, psnr=math.inf, mse=0.01)],
            0.1,
        )


def test_metric_rows_reject_mixed_finite_and_infinite_psnr_group() -> None:
    """同一均值不得混合有限失真样本与完全一致样本后声称正无穷."""

    records = [
        detection_record(decision=True, psnr=math.inf, mse=0.0),
        detection_record(decision=True, psnr=30.0, mse=0.001),
    ]

    with pytest.raises(ValueError, match="不得混合有限 PSNR 与正无穷 PSNR"):
        build_image_only_test_metric_rows(records, 0.1)
