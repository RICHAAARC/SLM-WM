from __future__ import annotations

import pytest

from experiments.protocol.fixed_fpr_observation_audit import (
    FORMAL_THRESHOLD_SOURCE,
    audit_fixed_fpr_observation_threshold,
    conformal_threshold_from_clean_negative_scores,
)
from experiments.runners.image_only_dataset_runtime import (
    apply_frozen_evidence_protocol,
    calibrate_complete_evidence_protocol,
)
from paper_experiments.analysis.fixed_fpr_threshold_audit import (
    audit_baseline_fixed_fpr,
    audit_main_method_fixed_fpr,
    build_fixed_fpr_threshold_audit_report,
)


pytestmark = pytest.mark.quick


def _main_method_rows() -> tuple[tuple[dict[str, object], ...], dict[str, object]]:
    """构造包含 calibration 和 test clean negative 的主方法记录。"""

    raw_rows = tuple(
        {
            "prompt_id": f"prompt-{index}",
            "split": split,
            "sample_role": "clean_negative",
            "attack_id": "",
            "content_score": score,
            "aligned_content_score": score,
            "attention_geometry_score": 0.0,
            "geometry_reliable": False,
        }
        for index, (split, score) in enumerate(
            (
                ("calibration", 0.1),
                ("calibration", 0.2),
                ("calibration", 0.3),
                ("test", 0.15),
                ("test", 0.35),
            )
        )
    )
    protocol = calibrate_complete_evidence_protocol(
        raw_rows[:3],
        target_fpr=0.25,
        rescue_margin_low=-0.05,
    )
    return apply_frozen_evidence_protocol(raw_rows, protocol), protocol.to_dict()


def _baseline_rows() -> tuple[dict[str, object], ...]:
    """构造共享 calibration 冻结阈值的 baseline observation。"""

    threshold = conformal_threshold_from_clean_negative_scores((0.1, 0.2, 0.3), 0.25)
    return tuple(
        {
            "prompt_id": f"prompt-{index}",
            "event_id": f"event-{index}",
            "split": split,
            "sample_role": "clean_negative",
            "attack_family": "clean",
            "score": score,
            "threshold": threshold,
            "threshold_source": FORMAL_THRESHOLD_SOURCE,
            "detection_decision": score >= threshold,
        }
        for index, (split, score) in enumerate(
            (
                ("calibration", 0.1),
                ("calibration", 0.2),
                ("calibration", 0.3),
                ("test", 0.15),
                ("test", 0.35),
            )
        )
    )


def test_main_method_threshold_audit_recomputes_complete_rescue_protocol() -> None:
    rows, protocol = _main_method_rows()

    result = audit_main_method_fixed_fpr(
        rows,
        protocol,
        target_fpr=0.25,
        expected_calibration_negative_count=3,
        expected_test_negative_count=2,
    )

    assert result["fixed_fpr_threshold_ready"] is True
    assert result["protocol_value_ready"] is True
    tampered = [dict(row) for row in rows]
    tampered[0]["formal_evidence_positive"] = not bool(
        tampered[0]["formal_evidence_positive"]
    )
    failed = audit_main_method_fixed_fpr(
        tampered,
        protocol,
        target_fpr=0.25,
        expected_calibration_negative_count=3,
        expected_test_negative_count=2,
    )
    assert failed["detection_decision_ready"] is False
    assert failed["fixed_fpr_threshold_ready"] is False

    tampered_margin = [dict(row) for row in rows]
    tampered_margin[0]["formal_raw_content_margin"] = 999.0
    failed_margin = audit_main_method_fixed_fpr(
        tampered_margin,
        protocol,
        target_fpr=0.25,
        expected_calibration_negative_count=3,
        expected_test_negative_count=2,
    )
    assert failed_margin["detection_decision_ready"] is False


def test_baseline_threshold_audit_binds_recomputed_threshold_and_digest() -> None:
    rows = _baseline_rows()
    primitive_audit = audit_fixed_fpr_observation_threshold(
        rows,
        target_fpr=0.25,
        expected_calibration_negative_count=3,
    )

    result = audit_baseline_fixed_fpr(
        "tree_ring",
        rows,
        target_fpr=0.25,
        expected_calibration_negative_count=3,
        expected_test_negative_count=2,
        declared_threshold=primitive_audit.frozen_threshold,
        declared_threshold_digest=primitive_audit.threshold_digest,
    )

    assert result["fixed_fpr_threshold_ready"] is True
    failed = audit_baseline_fixed_fpr(
        "tree_ring",
        rows,
        target_fpr=0.25,
        expected_calibration_negative_count=3,
        expected_test_negative_count=2,
        declared_threshold=primitive_audit.frozen_threshold,
        declared_threshold_digest="0" * 64,
    )
    assert failed["protocol_value_ready"] is False
    assert failed["fixed_fpr_threshold_ready"] is False


def test_threshold_audit_report_requires_exact_five_method_identity_set() -> None:
    method_ids = (
        "slm_wm",
        "tree_ring",
        "gaussian_shading",
        "shallow_diffuse",
        "t2smark",
    )
    rows = tuple(
        {
            "method_id": method_id,
            "target_fpr": 0.1,
            "fixed_fpr_threshold_ready": True,
        }
        for method_id in method_ids
    )

    report = build_fixed_fpr_threshold_audit_report(
        rows,
        paper_run_name="probe_paper",
        target_fpr=0.1,
    )
    assert report["fixed_fpr_threshold_audit_ready"] is True

    duplicate_rows = (*rows[:-1], dict(rows[0]))
    failed = build_fixed_fpr_threshold_audit_report(
        duplicate_rows,
        paper_run_name="probe_paper",
        target_fpr=0.1,
    )
    assert failed["method_identity_ready"] is False
    assert failed["fixed_fpr_threshold_audit_ready"] is False
