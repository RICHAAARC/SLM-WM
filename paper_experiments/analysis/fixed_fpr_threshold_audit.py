"""统一核验主方法与四个外部 baseline 的 fixed-FPR 阈值事实。"""

from __future__ import annotations

from dataclasses import fields
import math
import re
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
from main.core.digest import build_stable_digest
from paper_experiments.analysis.formal_record_statistics import (
    validate_frozen_evidence_protocol_record,
)


MAIN_THRESHOLD_SOURCE = (
    "calibration_clean_negative_complete_evidence_empirical_fixed_fpr"
)
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
FIXED_FPR_THRESHOLD_METHOD_IDS = (
    "slm_wm",
    "tree_ring",
    "gaussian_shading",
    "shallow_diffuse",
    "t2smark",
)


def _same_protocol_value(left: Any, right: Any) -> bool:
    """比较协议字段，浮点值使用严格绝对容差。"""

    if type(left) is not type(right):
        return False
    if type(left) is float:
        return math.isclose(left, right, rel_tol=0.0, abs_tol=1e-12)
    return left == right


def audit_main_method_fixed_fpr(
    observation_rows: Iterable[Mapping[str, Any]],
    frozen_protocol: Mapping[str, Any],
    *,
    observation_source_sha256: str,
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
    protocol_fields_ready = set(frozen_protocol) == set(protocol_field_names)
    declared_protocol: FrozenEvidenceProtocol | None = None
    if protocol_fields_ready:
        try:
            declared_protocol = validate_frozen_evidence_protocol_record(
                frozen_protocol,
                expected_target_fpr=target_fpr,
            )
        except (TypeError, ValueError):
            declared_protocol = None
    target_ready = declared_protocol is not None
    recomputed: FrozenEvidenceProtocol | None = None
    if calibration_rows and declared_protocol is not None:
        recomputed = calibrate_complete_evidence_protocol(
            calibration_rows,
            float(target_fpr),
            declared_protocol.rescue_margin_low,
        )
    recomputed_payload = recomputed.to_dict() if recomputed is not None else {}
    protocol_value_ready = bool(
        recomputed is not None
        and declared_protocol is not None
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
        "observation_source_sha256": str(observation_source_sha256),
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
    observation_source_sha256: str,
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
        "observation_source_sha256": str(observation_source_sha256),
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
) -> dict[str, Any]:
    """汇总五个方法的统一 fixed-FPR 阈值审计结论。"""

    materialized = tuple(dict(row) for row in rows)
    expected_ids = FIXED_FPR_THRESHOLD_METHOD_IDS
    actual_ids = tuple(str(row.get("method_id", "")) for row in materialized)
    identity_ready = (
        len(actual_ids) == len(expected_ids)
        and len(set(actual_ids)) == len(actual_ids)
        and set(actual_ids) == set(expected_ids)
    )
    canonical_rows = sorted(
        materialized,
        key=lambda row: (
            str(row.get("method_id", "")),
            build_stable_digest(row),
        ),
    )
    threshold_audit_rows_digest = build_stable_digest(canonical_rows)
    method_observation_source_sha256_map = (
        {
            str(row["method_id"]): str(row.get("observation_source_sha256", ""))
            for row in canonical_rows
        }
        if identity_ready
        else {}
    )
    method_threshold_digest_map = (
        {
            str(row["method_id"]): str(row.get("threshold_digest", ""))
            for row in canonical_rows
        }
        if identity_ready
        else {}
    )
    threshold_observation_binding_ready = bool(
        identity_ready
        and set(method_observation_source_sha256_map) == set(expected_ids)
        and set(method_threshold_digest_map) == set(expected_ids)
        and all(
            SHA256_PATTERN.fullmatch(digest) is not None
            for digest in method_observation_source_sha256_map.values()
        )
        and all(
            SHA256_PATTERN.fullmatch(digest) is not None
            for digest in method_threshold_digest_map.values()
        )
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
    ready = identity_ready and all_rows_ready and threshold_observation_binding_ready
    return {
        "paper_claim_scale": str(paper_run_name),
        "target_fpr": float(target_fpr),
        "expected_method_ids": list(expected_ids),
        "audited_method_ids": list(actual_ids),
        "audited_method_count": len(materialized),
        "method_observation_source_sha256_map": method_observation_source_sha256_map,
        "method_threshold_digest_map": method_threshold_digest_map,
        "threshold_audit_rows_digest": threshold_audit_rows_digest,
        "method_identity_ready": identity_ready,
        "all_method_thresholds_ready": all_rows_ready,
        "threshold_observation_binding_ready": threshold_observation_binding_ready,
        "fixed_fpr_threshold_audit_ready": ready,
        "supports_paper_claim": ready,
    }


def build_fixed_fpr_threshold_manifest_config(
    report: Mapping[str, Any],
) -> dict[str, Any]:
    """从阈值报告构造无自由字段的正式 manifest 配置."""

    return {
        "paper_claim_scale": str(report.get("paper_claim_scale", "")),
        "target_fpr": float(report.get("target_fpr", math.nan)),
        "method_observation_source_sha256_map": dict(
            report.get("method_observation_source_sha256_map", {})
        ),
        "method_threshold_digest_map": dict(
            report.get("method_threshold_digest_map", {})
        ),
        "threshold_audit_rows_digest": str(
            report.get("threshold_audit_rows_digest", "")
        ),
        "threshold_observation_binding_ready": (
            report.get("threshold_observation_binding_ready") is True
        ),
    }
