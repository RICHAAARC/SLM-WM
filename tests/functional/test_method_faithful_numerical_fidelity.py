"""验证 method-faithful baseline 的官方关键算子数值忠实度."""

from __future__ import annotations

from copy import deepcopy

import pytest

from main.core.digest import build_stable_digest
from paper_experiments.baselines.method_faithful_numerical_fidelity import (
    METHOD_FAITHFUL_NUMERICAL_FIDELITY_BASELINE_IDS,
    MethodFaithfulNumericalFidelityError,
    build_method_faithful_numerical_fidelity_report,
    validate_method_faithful_numerical_fidelity_report,
)


pytestmark = pytest.mark.quick


@pytest.mark.parametrize(
    "baseline_id",
    METHOD_FAITHFUL_NUMERICAL_FIDELITY_BASELINE_IDS,
)
def test_registered_official_operators_match_method_faithful_adapter(
    baseline_id: str,
) -> None:
    """登记 commit 的官方算子必须在同一确定性 Tensor 上匹配适配器."""

    pytest.importorskip("torch")
    report = build_method_faithful_numerical_fidelity_report(".", baseline_id)
    validated = validate_method_faithful_numerical_fidelity_report(
        report,
        expected_baseline_id=baseline_id,
    )

    assert validated["baseline_id"] == baseline_id
    assert validated["official_source_read_mode"] == "immutable_git_commit_blob"
    assert validated["method_faithful_numerical_fidelity_ready"] is True
    assert validated["operator_record_count"] == len(validated["operator_records"])
    assert all(
        record["numerical_fidelity_ready"] is True
        and record["max_absolute_error"] <= record["absolute_tolerance"]
        for record in validated["operator_records"]
    )


def test_numerical_fidelity_report_rejects_modified_operator_value() -> None:
    """任一比较值或门禁字段被修改后不得继续进入正式证据链."""

    pytest.importorskip("torch")
    report = build_method_faithful_numerical_fidelity_report(".", "tree_ring")
    modified = deepcopy(report)
    modified["operator_records"][0]["max_absolute_error"] = 1.0

    with pytest.raises(
        MethodFaithfulNumericalFidelityError,
        match="无法通过独立复验",
    ):
        validate_method_faithful_numerical_fidelity_report(
            modified,
            expected_baseline_id="tree_ring",
        )


def test_numerical_fidelity_rebuild_rejects_self_consistent_false_claim() -> None:
    """即使同步重算全部摘要, 超过容差的数值也不得伪装成 ready."""

    pytest.importorskip("torch")
    report = build_method_faithful_numerical_fidelity_report(".", "tree_ring")
    modified = deepcopy(report)
    record = modified["operator_records"][1]
    record["max_absolute_error"] = 0.5
    record["numerical_fidelity_ready"] = True
    record["comparison_record_digest"] = build_stable_digest(
        {
            key: value
            for key, value in record.items()
            if key != "comparison_record_digest"
        }
    )
    modified["operator_records_digest"] = build_stable_digest(
        modified["operator_records"]
    )
    modified["numerical_fidelity_report_digest"] = build_stable_digest(
        {
            key: value
            for key, value in modified.items()
            if key != "numerical_fidelity_report_digest"
        }
    )

    with pytest.raises(
        MethodFaithfulNumericalFidelityError,
        match="无法通过独立复验",
    ):
        validate_method_faithful_numerical_fidelity_report(
            modified,
            expected_baseline_id="tree_ring",
        )
