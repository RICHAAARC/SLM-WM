"""验证 GPU 方法真实性与资源预算使用两个独立门禁."""

from __future__ import annotations

from pathlib import Path

import pytest

from experiments.protocol import gpu_method_qualification as qualification
from experiments.protocol.image_only_evidence import (
    apply_frozen_evidence_protocol,
)
from experiments.runners.image_only_dataset_runtime import (
    calibrate_complete_evidence_protocol,
)
from experiments.runners.semantic_watermark_runtime import (
    SemanticWatermarkRuntimeConfig,
)
from tests.helpers.formal_detection_record import bind_formal_detection_record


pytestmark = pytest.mark.quick


def _protocol_and_formal_record():
    raw_records = tuple(
        bind_formal_detection_record(
            {
                "prompt_id": f"calibration-{index}",
                "split": "calibration",
                "sample_role": "clean_negative",
                "detection_key_role": "registered_watermark_key",
                "attack_id": "",
                "content_score": score,
                "aligned_content_score": score,
                "attention_geometry_score": 0.0,
                "registration_confidence": 0.0,
                "attention_sync_score": 0.0,
                "geometry_reliable": False,
                "alignment": {"registration_geometry_reliable": False},
            }
        )
        for index, score in enumerate((0.1, 0.2, 0.3))
    )
    protocol = calibrate_complete_evidence_protocol(
        raw_records,
        target_fpr=0.25,
    )
    return protocol, apply_frozen_evidence_protocol(raw_records, protocol)[0]


def test_keyed_prg_cross_platform_known_answer_rebuilds_exact_bytes() -> None:
    """当前平台必须重建与 Windows 冻结值逐字节相同的 PRG 向量."""

    report = qualification.rebuild_keyed_prg_known_answer_report(
        Path("configs/keyed_prg_cross_platform_known_answer.json")
    )

    assert report["known_answer_protocol_identity_ready"] is True
    assert report["keyed_prg_cross_platform_known_answer_ready"] is True
    assert all(
        row["known_answer_ready"] is True
        for row in report["known_answer_vector_reports"]
    )


def test_resource_budget_failure_does_not_change_operator_readiness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """运行超预算只能阻断资源门禁, 不能否定方法算子事实."""

    monkeypatch.setattr(
        qualification,
        "build_gpu_operator_preflight_report",
        lambda *_args, **_kwargs: {
            "gpu_operator_preflight_ready": True,
            "supports_paper_claim": False,
            "gpu_operator_preflight_report_digest": "a" * 64,
        },
    )
    report = qualification.build_gpu_method_qualification_report(
        runtime_result={"run_id": "single_prompt_gpu_observation"},
        update_records=(),
        detection_records=(),
        config=SemanticWatermarkRuntimeConfig(),
        known_answer_path=(
            "configs/keyed_prg_cross_platform_known_answer.json"
        ),
        resource_observation={
            "peak_gpu_memory_bytes": 40,
            "single_prompt_wall_time_seconds": 20.0,
            "estimated_probe_total_gpu_hours": 12.0,
        },
        registered_budget={
            "maximum_peak_gpu_memory_bytes": 20,
            "maximum_single_prompt_wall_time_seconds": 10.0,
            "maximum_estimated_probe_total_gpu_hours": 6.0,
        },
    )

    assert report["gpu_operator_preflight_ready"] is True
    assert report["gpu_resource_budget_ready"] is False
    assert report["gpu_resource_budget"][
        "affects_gpu_operator_preflight_ready"
    ] is False
    assert report["supports_paper_claim"] is False


def test_resource_budget_requires_observations_and_registered_limits() -> None:
    """缺少预算依据时应报告未评估, 不能伪装为资源可执行."""

    report = qualification.build_gpu_resource_budget_report(None, None)

    assert report["gpu_resource_budget_ready"] is False
    assert report["resource_budget_evaluation_status"] == (
        "not_evaluated_missing_observation_or_registered_limit"
    )
    assert report["affects_gpu_operator_preflight_ready"] is False


def test_threshold_free_measurement_cannot_claim_formal_blind_detection() -> None:
    """metadata布尔值不得替代冻结阈值协议实际物化的判定字段。"""

    record = {
        "image_only_measurement_config_digest": "a" * 64,
        "metadata": {
            "measurement_status": "threshold_independent_image_only_evidence",
            "same_threshold_geometry_rescue_protocol": True,
        },
        "lf_score": 0.1,
        "tail_robust_score": 0.2,
        "content_score": 0.3,
    }

    protocol, _ = _protocol_and_formal_record()
    assert qualification._formal_same_threshold_decision_ready(
        record,
        protocol,
    ) is False


def test_formal_decision_requires_and_accepts_the_validated_protocol() -> None:
    """完整协议物化的记录可由同一协议逐字重建。"""

    protocol, record = _protocol_and_formal_record()

    assert qualification._formal_same_threshold_decision_ready(
        record,
        protocol,
    ) is True


@pytest.mark.parametrize("digest", ("0" * 64, "f" * 64))
def test_formal_decision_rejects_self_consistent_record_with_unbound_digest(
    digest: str,
) -> None:
    """任意64hex摘要不能替代完整FrozenEvidenceProtocol身份。"""

    protocol, record = _protocol_and_formal_record()
    record["frozen_threshold_digest"] = digest

    assert qualification._formal_same_threshold_decision_ready(
        record,
        protocol,
    ) is False
