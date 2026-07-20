from __future__ import annotations

from dataclasses import replace
import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

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
from experiments.protocol.image_only_evidence import (
    partition_calibration_prompt_ids,
)
from main.core.digest import build_stable_digest
from main.methods.detection.image_only import (
    project_image_only_measurement_record,
    recompute_image_only_measurement_digest_payload,
)
from paper_experiments.analysis.fixed_fpr_threshold_audit import (
    MAIN_THRESHOLD_SOURCE,
    audit_baseline_fixed_fpr,
    audit_main_method_fixed_fpr,
    build_fixed_fpr_threshold_audit_report,
    build_fixed_fpr_threshold_manifest_config,
)
from tests.helpers.formal_detection_record import bind_formal_detection_record


pytestmark = pytest.mark.quick


OBSERVATION_SOURCE_SHA256 = "a" * 64


def _raw_measurements(
    rows: tuple[dict[str, object], ...] | list[dict[str, object]],
) -> tuple[dict[str, object], ...]:
    """显式投影最终记录中的阈值无关测量原子。"""

    return tuple(project_image_only_measurement_record(row) for row in rows)


def _bind_attention_alignment_gate(
    record: dict[str, object],
) -> dict[str, object]:
    """为检测夹具绑定预注册注意力配准门禁."""

    return bind_formal_detection_record(record)


def _main_method_rows() -> tuple[tuple[dict[str, object], ...], dict[str, object]]:
    """构造包含 calibration 和 test clean negative 的主方法记录。"""

    raw_rows = tuple(
        _bind_attention_alignment_gate({
            "prompt_id": f"prompt-{index}",
            "split": split,
            "sample_role": "clean_negative",
            "detection_key_role": "registered_watermark_key",
            "attack_id": None,
            "content_score": score,
            "aligned_content_score": score,
            "attention_geometry_score": 0.0,
            "registration_confidence": 0.0,
            "attention_sync_score": 0.0,
            "geometry_reliable": False,
            "alignment": {
                "registration_geometry_reliable": False,
            },
        })
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
    )
    return apply_frozen_evidence_protocol(raw_rows, protocol), protocol.to_dict()


def _baseline_rows() -> tuple[dict[str, object], ...]:
    """构造共享 calibration 冻结阈值的 baseline observation。"""

    calibration_scores = {
        f"prompt-{index}": score
        for index, score in enumerate((0.1, 0.2, 0.3))
    }
    _, threshold_freeze_ids, _ = partition_calibration_prompt_ids(
        calibration_scores
    )
    threshold = conformal_threshold_from_clean_negative_scores(
        (calibration_scores[prompt_id] for prompt_id in threshold_freeze_ids),
        0.25,
    )
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
        observation_source_sha256=OBSERVATION_SOURCE_SHA256,
        target_fpr=0.25,
        expected_calibration_source_negative_count=3,
        expected_test_negative_count=2,
    )

    assert result["fixed_fpr_threshold_ready"] is True
    assert result["protocol_value_ready"] is True
    assert result["threshold_source"] == MAIN_THRESHOLD_SOURCE
    assert "conformal" not in result["threshold_source"]
    detector_config_digest = protocol[
        "image_only_measurement_config_digest"
    ]
    assert all(
        row["image_only_measurement_config_digest"]
        == row["frozen_image_only_measurement_config_digest"]
        == detector_config_digest
        for row in rows
    )
    tampered = [dict(row) for row in rows]
    tampered[0]["formal_evidence_positive"] = not bool(
        tampered[0]["formal_evidence_positive"]
    )
    failed = audit_main_method_fixed_fpr(
        tampered,
        protocol,
        observation_source_sha256=OBSERVATION_SOURCE_SHA256,
        target_fpr=0.25,
        expected_calibration_source_negative_count=3,
        expected_test_negative_count=2,
    )
    assert failed["detection_decision_ready"] is False
    assert failed["fixed_fpr_threshold_ready"] is False

    tampered_margin = [dict(row) for row in rows]
    tampered_margin[0]["formal_raw_content_margin"] = 999.0
    failed_margin = audit_main_method_fixed_fpr(
        tampered_margin,
        protocol,
        observation_source_sha256=OBSERVATION_SOURCE_SHA256,
        target_fpr=0.25,
        expected_calibration_source_negative_count=3,
        expected_test_negative_count=2,
    )
    assert failed_margin["detection_decision_ready"] is False


@pytest.mark.parametrize(
    ("record_index", "field_name", "invalid_value"),
    (
        (0, "attack_condition", "jpeg_compression"),
        (3, "detection_key_role", "registered_wrong_key_negative"),
    ),
)
def test_main_method_threshold_audit_rejects_non_registered_clean_rows(
    record_index: int,
    field_name: str,
    invalid_value: str,
) -> None:
    """攻击条件或 wrong-key 身份不得冒充 registered-key clean negative。"""

    rows, protocol = _main_method_rows()
    changed = [dict(row) for row in rows]
    changed[record_index][field_name] = invalid_value

    result = audit_main_method_fixed_fpr(
        changed,
        protocol,
        observation_source_sha256=OBSERVATION_SOURCE_SHA256,
        target_fpr=0.25,
        expected_calibration_source_negative_count=3,
        expected_test_negative_count=2,
    )

    assert result["split_count_ready"] is False
    assert result["fixed_fpr_threshold_ready"] is False


def test_calibrator_rejects_hidden_attack_condition_on_clean_negative() -> None:
    """calibrator 不得只因 `attack_id` 为空就接受攻击记录。"""

    rows, _protocol = _main_method_rows()
    changed = list(_raw_measurements(rows[:3]))
    changed[0] = {
        **changed[0],
        "attack_condition": "jpeg_compression",
    }

    with pytest.raises(ValueError, match="registered-key clean negatives"):
        calibrate_complete_evidence_protocol(changed, target_fpr=0.25)


def test_calibrator_rejects_nonempty_attack_id_on_clean_negative() -> None:
    """JSON null 表示未攻击，但非空攻击 ID 仍必须失败关闭。"""

    rows, _protocol = _main_method_rows()
    changed = list(_raw_measurements(rows[:3]))
    changed[0] = {
        **changed[0],
        "attack_id": "jpeg_attack_0001",
    }

    with pytest.raises(ValueError, match="registered-key clean negatives"):
        calibrate_complete_evidence_protocol(changed, target_fpr=0.25)


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    (
        ("attention_anchor_count", 12.0),
        ("lf_carrier_protocol_digest", "f" * 64),
        ("tail_carrier_protocol_digest", "f" * 64),
        ("geometry_protocol_calibration_ready", 1),
        ("unexpected_protocol_field", "forbidden"),
    ),
)
def test_main_method_threshold_audit_rejects_protocol_type_and_schema_drift(
    field_name: str,
    invalid_value: object,
) -> None:
    """正式阈值审计必须拒绝宽松数值相等和未知协议字段."""

    rows, protocol = _main_method_rows()
    changed_protocol = {
        **protocol,
        field_name: invalid_value,
    }

    result = audit_main_method_fixed_fpr(
        rows,
        changed_protocol,
        observation_source_sha256=OBSERVATION_SOURCE_SHA256,
        target_fpr=0.25,
        expected_calibration_source_negative_count=3,
        expected_test_negative_count=2,
    )

    assert result["protocol_value_ready"] is False
    assert result["detection_decision_ready"] is False
    assert result["fixed_fpr_threshold_ready"] is False


def test_calibration_rejects_mixed_low_frequency_carrier_identity() -> None:
    """Calibration 记录不得混用不同的 LF 检测权重身份."""

    rows, _protocol = _main_method_rows()
    mixed_rows = list(_raw_measurements(rows[:3]))
    mixed_rows[1] = bind_formal_detection_record(
        mixed_rows[1],
        method_role="lf_only_content",
        lf_weight=1.0,
        tail_robust_weight=0.0,
        tail_fraction=0.20,
    )

    with pytest.raises(
        ValueError,
        match="混用了测量配置或载体身份",
    ):
        calibrate_complete_evidence_protocol(
            mixed_rows,
            target_fpr=0.25,
        )


def test_calibration_rejects_low_frequency_carrier_protocol_drift() -> None:
    """Calibration 记录不得混入摘要漂移的 LF 载体协议."""

    rows, _protocol = _main_method_rows()
    mixed_rows = list(_raw_measurements(rows[:3]))
    drifted = dict(mixed_rows[1])
    drifted["lf_carrier_protocol_digest"] = "f" * 64
    drifted["measurement_digest"] = build_stable_digest(
        recompute_image_only_measurement_digest_payload(drifted)
    )
    mixed_rows[1] = drifted

    with pytest.raises(ValueError):
        calibrate_complete_evidence_protocol(
            mixed_rows,
            target_fpr=0.25,
        )


def test_calibration_rejects_tail_carrier_protocol_drift() -> None:
    """Calibration 记录不得混入摘要漂移的尾部载体协议."""

    rows, _protocol = _main_method_rows()
    mixed_rows = list(_raw_measurements(rows[:3]))
    drifted = dict(mixed_rows[1])
    drifted["tail_carrier_protocol_digest"] = "f" * 64
    drifted["measurement_digest"] = build_stable_digest(
        recompute_image_only_measurement_digest_payload(drifted)
    )
    mixed_rows[1] = drifted

    with pytest.raises(ValueError):
        calibrate_complete_evidence_protocol(
            mixed_rows,
            target_fpr=0.25,
        )


def test_apply_frozen_protocol_rejects_low_frequency_record_drift() -> None:
    """冻结协议应用时必须拒绝记录级 LF 权重漂移."""

    rows, protocol_record = _main_method_rows()
    protocol = calibrate_complete_evidence_protocol(
        _raw_measurements(rows[:3]),
        target_fpr=0.25,
    )
    assert protocol.to_dict() == protocol_record
    drifted = bind_formal_detection_record(
        project_image_only_measurement_record(rows[3]),
        method_role="lf_only_content",
        lf_weight=1.0,
        tail_robust_weight=0.0,
        tail_fraction=0.20,
    )

    with pytest.raises(
        ValueError,
        match="配置或载体身份不一致",
    ):
        apply_frozen_evidence_protocol((drifted,), protocol)


def test_apply_rejects_test_record_with_different_detector_config() -> None:
    """测试 split 不得混用 calibration 未冻结的检测器配置身份."""

    rows, _protocol_record = _main_method_rows()
    protocol = calibrate_complete_evidence_protocol(
        _raw_measurements(rows[:3]),
        target_fpr=0.25,
    )
    drifted_metadata = {
        **dict(rows[3]["metadata"]),
        "model_id": "fixture/different-detector-model",
    }
    drifted = bind_formal_detection_record(
        {
            **project_image_only_measurement_record(rows[3]),
            "metadata": drifted_metadata,
        }
    )

    with pytest.raises(ValueError, match="配置或载体身份"):
        apply_frozen_evidence_protocol((drifted,), protocol)


@pytest.mark.parametrize(
    ("field_name", "value"),
    (
        ("content_threshold", 999.0),
        ("threshold_digest", "f" * 64),
        ("calibration_false_positive_rate", 0.999),
        ("content_threshold", 1),
        ("rescue_margin_low", -1),
        ("geometry_score_threshold", 0),
        ("registration_confidence_threshold", 0),
        ("attention_sync_score_threshold", 0),
    ),
)
def test_apply_frozen_protocol_rejects_protocol_integrity_drift(
    field_name: str,
    value: object,
) -> None:
    """协议应用时必须先拒绝冻结协议正文、派生率或摘要漂移."""

    rows, _protocol_record = _main_method_rows()
    protocol = calibrate_complete_evidence_protocol(
        _raw_measurements(rows[:3]),
        target_fpr=0.25,
    )
    drifted = replace(protocol, **{field_name: value})

    with pytest.raises(
        ValueError,
        match="阈值摘要|假阳性率|连续参数|配置或载体身份|几何 rescue 冻结参数",
    ):
        apply_frozen_evidence_protocol(
            (project_image_only_measurement_record(rows[3]),),
            drifted,
        )


def test_baseline_threshold_audit_binds_recomputed_threshold_and_digest() -> None:
    rows = _baseline_rows()
    primitive_audit = audit_fixed_fpr_observation_threshold(
        rows,
        target_fpr=0.25,
        expected_calibration_source_negative_count=3,
    )

    result = audit_baseline_fixed_fpr(
        "tree_ring",
        rows,
        observation_source_sha256=OBSERVATION_SOURCE_SHA256,
        target_fpr=0.25,
        expected_calibration_source_negative_count=3,
        expected_test_negative_count=2,
        declared_threshold=primitive_audit.frozen_threshold,
        declared_threshold_digest=primitive_audit.threshold_digest,
    )

    assert result["fixed_fpr_threshold_ready"] is True
    failed = audit_baseline_fixed_fpr(
        "tree_ring",
        rows,
        observation_source_sha256=OBSERVATION_SOURCE_SHA256,
        target_fpr=0.25,
        expected_calibration_source_negative_count=3,
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
            "threshold_digest": f"{index + 1:x}" * 64,
            "observation_source_sha256": f"{index + 6:x}" * 64,
            "calibration_partition_digest": "b" * 64,
            "threshold_freeze_prompt_id_digest": "c" * 64,
            "fixed_fpr_threshold_ready": True,
        }
        for index, method_id in enumerate(method_ids)
    )

    report = build_fixed_fpr_threshold_audit_report(
        rows,
        paper_run_name="probe_paper",
        target_fpr=0.1,
    )
    assert report["fixed_fpr_threshold_audit_ready"] is True
    assert report["threshold_observation_binding_ready"] is True
    assert report["shared_calibration_partition_ready"] is True
    assert report["method_observation_source_sha256_map"] == {
        method_id: f"{index + 6:x}" * 64
        for index, method_id in enumerate(method_ids)
    }
    assert report["method_threshold_digest_map"] == {
        method_id: f"{index + 1:x}" * 64
        for index, method_id in enumerate(method_ids)
    }
    reordered = build_fixed_fpr_threshold_audit_report(
        reversed(rows),
        paper_run_name="probe_paper",
        target_fpr=0.1,
    )
    assert reordered["threshold_audit_rows_digest"] == report[
        "threshold_audit_rows_digest"
    ]

    duplicate_rows = (*rows[:-1], dict(rows[0]))
    failed = build_fixed_fpr_threshold_audit_report(
        duplicate_rows,
        paper_run_name="probe_paper",
        target_fpr=0.1,
    )
    assert failed["method_identity_ready"] is False
    assert failed["threshold_observation_binding_ready"] is False
    assert failed["fixed_fpr_threshold_audit_ready"] is False

    incomplete = build_fixed_fpr_threshold_audit_report(
        rows[:-1],
        paper_run_name="probe_paper",
        target_fpr=0.1,
    )
    assert incomplete["expected_method_ids"] == list(method_ids)
    assert incomplete["method_identity_ready"] is False
    assert incomplete["fixed_fpr_threshold_audit_ready"] is False

    malformed_digest_rows = tuple(dict(row) for row in rows)
    malformed_digest_rows[0]["observation_source_sha256"] = "A" * 64
    malformed = build_fixed_fpr_threshold_audit_report(
        malformed_digest_rows,
        paper_run_name="probe_paper",
        target_fpr=0.1,
    )
    assert malformed["all_method_thresholds_ready"] is True
    assert malformed["threshold_observation_binding_ready"] is False
    assert malformed["fixed_fpr_threshold_audit_ready"] is False

    different_partition = tuple(dict(row) for row in rows)
    different_partition[0]["threshold_freeze_prompt_id_digest"] = "d" * 64
    partition_failed = build_fixed_fpr_threshold_audit_report(
        different_partition,
        paper_run_name="probe_paper",
        target_fpr=0.1,
    )
    assert partition_failed["shared_calibration_partition_ready"] is False
    assert partition_failed["supports_paper_claim"] is False
