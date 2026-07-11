"""统一核验主方法与四个外部 baseline 的 fixed-FPR 阈值事实。"""

from __future__ import annotations

from dataclasses import fields
import math
from typing import Any, Iterable, Mapping

from experiments.protocol.fixed_fpr_observation_audit import (
    FORMAL_THRESHOLD_SOURCE,
    audit_fixed_fpr_observation_threshold,
)
from experiments.runners.image_only_dataset_runtime import (
    FrozenEvidenceProtocol,
    apply_frozen_evidence_protocol,
    calibrate_complete_evidence_protocol,
)


MAIN_THRESHOLD_SOURCE = "calibration_clean_negative_complete_evidence_conformal"


def _same_protocol_value(left: Any, right: Any) -> bool:
    """比较协议字段，浮点值使用严格绝对容差。"""

    if isinstance(left, int | float) and isinstance(right, int | float):
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-12)
    return left == right


def audit_main_method_fixed_fpr(
    observation_rows: Iterable[Mapping[str, Any]],
    frozen_protocol: Mapping[str, Any],
    *,
    target_fpr: float,
    expected_calibration_negative_count: int,
    expected_test_negative_count: int,
) -> dict[str, Any]:
    """从主方法 observation 重算包含 rescue 的完整冻结判定。"""

    rows = tuple(dict(row) for row in observation_rows)
    calibration_rows = tuple(
        row
        for row in rows
        if row.get("split") == "calibration"
        and row.get("sample_role") == "clean_negative"
        and not row.get("attack_id")
    )
    test_rows = tuple(
        row
        for row in rows
        if row.get("split") == "test"
        and row.get("sample_role") == "clean_negative"
        and not row.get("attack_id")
    )
    protocol_field_names = tuple(field.name for field in fields(FrozenEvidenceProtocol))
    protocol_fields_ready = all(name in frozen_protocol for name in protocol_field_names)
    target_ready = protocol_fields_ready and math.isclose(
        float(frozen_protocol.get("target_fpr", math.nan)),
        float(target_fpr),
        rel_tol=0.0,
        abs_tol=1e-12,
    )
    recomputed: FrozenEvidenceProtocol | None = None
    if calibration_rows and "rescue_margin_low" in frozen_protocol:
        recomputed = calibrate_complete_evidence_protocol(
            calibration_rows,
            float(target_fpr),
            float(frozen_protocol["rescue_margin_low"]),
        )
    recomputed_payload = recomputed.to_dict() if recomputed is not None else {}
    protocol_value_ready = bool(
        recomputed is not None
        and protocol_fields_ready
        and all(
            _same_protocol_value(frozen_protocol[name], recomputed_payload[name])
            for name in protocol_field_names
        )
    )
    decision_ready = False
    if recomputed is not None:
        reapplied = apply_frozen_evidence_protocol(rows, recomputed)
        derived_fields = (
            "frozen_content_threshold",
            "formal_raw_content_margin",
            "formal_aligned_content_margin",
            "formal_positive_by_content",
            "formal_content_failure_reason",
            "formal_rescue_applied",
            "formal_evidence_positive",
            "formal_metric_status",
        )
        decision_ready = len(reapplied) == len(rows) and all(
            all(
                _same_protocol_value(original.get(field_name), resolved.get(field_name))
                for field_name in derived_fields
            )
            and str(original.get("frozen_threshold_digest", ""))
            == recomputed.threshold_digest
            for original, resolved in zip(rows, reapplied)
        )
    count_ready = (
        len(calibration_rows) == int(expected_calibration_negative_count)
        and len(test_rows) == int(expected_test_negative_count)
    )
    ready = all((target_ready, protocol_value_ready, decision_ready, count_ready))
    return {
        "method_id": "slm_wm",
        "threshold_source": MAIN_THRESHOLD_SOURCE,
        "target_fpr": float(target_fpr),
        "calibration_clean_negative_count": len(calibration_rows),
        "test_clean_negative_count": len(test_rows),
        "calibrated_detection_threshold": (
            None if recomputed is None else recomputed.content_threshold
        ),
        "threshold_digest": "" if recomputed is None else recomputed.threshold_digest,
        "protocol_target_ready": target_ready,
        "protocol_value_ready": protocol_value_ready,
        "detection_decision_ready": decision_ready,
        "split_count_ready": count_ready,
        "fixed_fpr_threshold_ready": ready,
        "supports_paper_claim": False,
    }


def audit_baseline_fixed_fpr(
    method_id: str,
    observation_rows: Iterable[Mapping[str, Any]],
    *,
    target_fpr: float,
    expected_calibration_negative_count: int,
    expected_test_negative_count: int,
    declared_threshold: float | None = None,
    declared_threshold_digest: str = "",
) -> dict[str, Any]:
    """从外部 baseline observation 重算共享 conformal 阈值与逐条判定。"""

    rows = tuple(dict(row) for row in observation_rows)
    audit = audit_fixed_fpr_observation_threshold(
        rows,
        target_fpr=float(target_fpr),
        expected_calibration_negative_count=int(expected_calibration_negative_count),
    )
    test_rows = tuple(
        row
        for row in rows
        if row.get("split") == "test"
        and row.get("sample_role") == "clean_negative"
        and row.get("attack_family") == "clean"
    )
    test_count_ready = len(test_rows) == int(expected_test_negative_count)
    declared_threshold_ready = bool(
        declared_threshold is not None
        and audit.frozen_threshold is not None
        and math.isclose(
            float(declared_threshold),
            float(audit.frozen_threshold),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    )
    declared_digest_ready = (
        len(str(declared_threshold_digest)) == 64
        and str(declared_threshold_digest) == audit.threshold_digest
    )
    ready = all(
        (
            audit.fixed_fpr_ready,
            test_count_ready,
            declared_threshold_ready,
            declared_digest_ready,
        )
    )
    return {
        "method_id": str(method_id),
        "threshold_source": FORMAL_THRESHOLD_SOURCE,
        "target_fpr": float(target_fpr),
        "calibration_clean_negative_count": audit.calibration_negative_count,
        "test_clean_negative_count": len(test_rows),
        "calibrated_detection_threshold": audit.frozen_threshold,
        "threshold_digest": audit.threshold_digest,
        "protocol_target_ready": True,
        "protocol_value_ready": declared_threshold_ready and declared_digest_ready,
        "detection_decision_ready": audit.detection_decision_ready,
        "split_count_ready": audit.calibration_count_ready and test_count_ready,
        "fixed_fpr_threshold_ready": ready,
        "supports_paper_claim": False,
    }


def build_fixed_fpr_threshold_audit_report(
    rows: Iterable[Mapping[str, Any]],
    *,
    paper_run_name: str,
    target_fpr: float,
    expected_method_ids: Iterable[str] = (
        "slm_wm",
        "tree_ring",
        "gaussian_shading",
        "shallow_diffuse",
        "t2smark",
    ),
) -> dict[str, Any]:
    """汇总五个方法的统一 fixed-FPR 阈值审计结论。"""

    materialized = tuple(dict(row) for row in rows)
    expected_ids = tuple(str(value) for value in expected_method_ids)
    actual_ids = tuple(str(row.get("method_id", "")) for row in materialized)
    identity_ready = (
        len(actual_ids) == len(expected_ids)
        and len(set(actual_ids)) == len(actual_ids)
        and set(actual_ids) == set(expected_ids)
    )
    all_rows_ready = bool(materialized) and all(
        row.get("fixed_fpr_threshold_ready") is True
        and math.isclose(
            float(row.get("target_fpr", math.nan)),
            float(target_fpr),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        for row in materialized
    )
    ready = identity_ready and all_rows_ready
    return {
        "paper_claim_scale": str(paper_run_name),
        "target_fpr": float(target_fpr),
        "expected_method_ids": list(expected_ids),
        "audited_method_ids": list(actual_ids),
        "audited_method_count": len(materialized),
        "method_identity_ready": identity_ready,
        "all_method_thresholds_ready": all_rows_ready,
        "fixed_fpr_threshold_audit_ready": ready,
        "supports_paper_claim": ready,
    }
