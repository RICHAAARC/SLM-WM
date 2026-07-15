"""fixed-FPR 结果记录转换与聚合来源门禁的轻量功能测试."""

from __future__ import annotations

from pathlib import Path

import pytest

from experiments.protocol.paper_fixed_fpr import (
    bounded_hoeffding_confidence_interval,
)
from paper_experiments.runners.paper_claim_provenance import (
    PaperClaimAggregateRequiredError,
)
from scripts.write_paper_result_records import (
    attach_metric_fields,
    write_paper_result_record_outputs,
)


pytestmark = pytest.mark.quick


def test_result_metric_writer_preserves_negative_ssim_and_signed_ci() -> None:
    """负 SSIM 必须原样保留, 且使用范围宽度为2的 Hoeffding 区间."""

    payload = attach_metric_fields(
        {},
        positive_count=34,
        negative_count=34,
        attack_record_count=68,
        supported_record_count=34,
        true_positive_rate=0.8,
        false_positive_rate=0.05,
        clean_false_positive_rate=0.05,
        attacked_false_positive_rate=0.05,
        quality_score_mean=-0.25,
        score_retention_mean=0.7,
        confidence_level=0.95,
    )
    expected_quality_ci = bounded_hoeffding_confidence_interval(
        -0.25,
        34,
        0.95,
        lower_bound=-1.0,
        upper_bound=1.0,
    )
    expected_retention_ci = bounded_hoeffding_confidence_interval(
        0.7,
        34,
        0.95,
    )

    assert payload["quality_score_mean"] == -0.25
    assert (
        payload["quality_score_ci_low"],
        payload["quality_score_ci_high"],
    ) == pytest.approx(expected_quality_ci)
    assert (
        payload["score_retention_ci_low"],
        payload["score_retention_ci_high"],
    ) == pytest.approx(expected_retention_ci)


@pytest.mark.parametrize("paper_run_name", ("probe_paper", "pilot_paper", "full_paper"))
def test_formal_result_writer_requires_validated_exact9_aggregate_before_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    paper_run_name: str,
) -> None:
    """三个统计层级都必须在读取输入和创建输出前通过同一聚合门禁."""

    monkeypatch.setenv("SLM_WM_PAPER_RUN_NAME", paper_run_name)

    with pytest.raises(
        PaperClaimAggregateRequiredError,
        match="版本化精确9重复聚合证据验证",
    ):
        write_paper_result_record_outputs(
            root=tmp_path,
            output_dir=tmp_path / "outside_outputs",
            baseline_records_path=tmp_path / "missing.jsonl",
            require_existing_evidence=True,
        )

    assert not (tmp_path / "outputs").exists()
    assert not (tmp_path / "outside_outputs").exists()
