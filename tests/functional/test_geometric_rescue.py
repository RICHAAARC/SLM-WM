"""同阈值几何恢复重判的轻量功能测试。"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pytest

from main.methods.detection import SameThresholdRescueConfig, decide_same_threshold_geometric_rescue
from scripts.write_geometric_rescue_outputs import write_geometric_rescue_outputs


def json_line(value: dict[str, object]) -> str:
    """将字典转为 JSONL 单行。"""
    return json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n"


def reliable_geometry_record() -> dict[str, object]:
    """构造可靠几何证据记录。"""
    return {
        "geometry_evidence_record_id": "geometry_evidence_test",
        "attention_graph_id": "attention_graph_test",
        "capture_id": "real_attention_test",
        "registration_confidence": 0.82,
        "anchor_inlier_ratio": 0.75,
        "recovered_sync_consistency": 0.88,
        "alignment_residual": 0.18,
        "geometry_reliable": True,
        "direct_positive_decision": False,
        "supports_paper_claim": False,
    }


@pytest.mark.quick
def test_same_threshold_rescue_uses_aligned_content_score_without_direct_geometry_positive() -> None:
    """几何链只能帮助同阈值内容重判, 不能直接给出正式 positive。"""
    config = SameThresholdRescueConfig(content_threshold=0.75, rescue_margin_low=-0.05)
    content_record = {
        "content_detection_record_id": "content_test",
        "prompt_id": "prompt_test",
        "split": "calibration",
        "content_score": 0.72,
        "metadata": {"sample_role": "positive_source"},
    }
    rescued = decide_same_threshold_geometric_rescue(
        content_record=content_record,
        geometry_record=reliable_geometry_record(),
        aligned_content_score=0.76,
        config=config,
        fail_reason="geometry_suspected",
    )
    blocked = decide_same_threshold_geometric_rescue(
        content_record=content_record,
        geometry_record=reliable_geometry_record(),
        aligned_content_score=0.76,
        config=config,
        fail_reason="geometry_suspected",
        rescue_ablation_mode="no_rescue",
    )
    geo_audit = decide_same_threshold_geometric_rescue(
        content_record=content_record,
        geometry_record=reliable_geometry_record(),
        aligned_content_score=0.72,
        config=config,
        fail_reason="geometry_suspected",
        rescue_ablation_mode="geo_direct_positive_audit",
    )

    assert rescued.positive_by_content is False
    assert rescued.rescue_eligible is True
    assert rescued.rescue_applied is True
    assert rescued.evidence_decision is True
    assert rescued.direct_positive_decision is False
    assert blocked.rescue_applied is False
    assert blocked.evidence_decision is False
    assert geo_audit.evidence_decision is False
    assert geo_audit.geo_direct_positive_audit_decision is True


def write_fixture_attention_package(path: Path) -> None:
    """写出最小真实 attention injection 包 fixture。"""
    geometry = reliable_geometry_record()
    nested_buffer = io.BytesIO()
    with ZipFile(nested_buffer, mode="w", compression=ZIP_DEFLATED) as nested:
        nested.writestr("outputs/attention_geometry/geometry_evidence_records.jsonl", json_line(geometry))
        nested.writestr(
            "outputs/attention_geometry/geometry_evidence_summary.json",
            json.dumps(
                {
                    "attention_geometry_ready": True,
                    "real_attention_capture_count": 1,
                    "protocol_decision": "pass",
                    "supports_paper_claim": False,
                },
                ensure_ascii=False,
            ),
        )
    with ZipFile(path, mode="w", compression=ZIP_DEFLATED) as archive:
        archive.writestr(
            "outputs/attention_latent_injection/attention_latent_injection_result.json",
            json.dumps(
                {
                    "run_decision": "pass",
                    "latent_update_count": 3,
                    "image_quality_metrics_ready": True,
                    "full_method_claim_ready": False,
                },
                ensure_ascii=False,
            ),
        )
        archive.writestr(
            "outputs/attention_latent_update/attention_update_summary.json",
            json.dumps(
                {
                    "attention_geometry_ready": True,
                    "active_update_count": 1,
                    "protocol_decision": "pass",
                    "supports_paper_claim": False,
                },
                ensure_ascii=False,
            ),
        )
        archive.writestr(
            "outputs/attention_latent_update/attention_carrier_records.jsonl",
            json_line(
                {
                    "fallback_mode": "active_update",
                    "attention_graph_id": "attention_graph_test",
                    "carrier_id": "attention_carrier_test",
                    "metadata": {"prompt_id": "prompt_a"},
                }
            ),
        )
        archive.writestr(
            "outputs/attention_latent_injection/input_packages/attention_geometry_package_fixture.zip",
            nested_buffer.getvalue(),
        )


@pytest.mark.quick
def test_geometric_rescue_outputs_are_rebuildable_from_governed_records(tmp_path: Path) -> None:
    """几何恢复产物应由内容记录和真实 attention injection 包重建。"""
    content_dir = tmp_path / "outputs" / "content_carriers"
    content_dir.mkdir(parents=True)
    content_records_path = content_dir / "content_detection_records.jsonl"
    content_records = [
        {
            "content_detection_record_id": "content_positive",
            "prompt_id": "prompt_a",
            "split": "calibration",
            "content_score": 0.72,
            "metadata": {"sample_role": "positive_source"},
        },
        {
            "content_detection_record_id": "content_clean",
            "prompt_id": "prompt_b",
            "split": "calibration",
            "content_score": 0.74,
            "metadata": {"sample_role": "clean_negative"},
        },
        {
            "content_detection_record_id": "content_attacked",
            "prompt_id": "prompt_c",
            "split": "calibration",
            "content_score": 0.76,
            "metadata": {"sample_role": "attacked_negative"},
        },
    ]
    content_records_path.write_text("".join(json_line(record) for record in content_records), encoding="utf-8")
    package_path = tmp_path / "outputs" / "attention_latent_injection_package_fixture.zip"
    write_fixture_attention_package(package_path)

    manifest = write_geometric_rescue_outputs(
        root=tmp_path,
        content_records_path=content_records_path,
        attention_injection_package_path=package_path,
        max_content_records=None,
    )
    output_dir = tmp_path / "outputs" / "geometric_rescue"
    records_path = output_dir / "aligned_detection_records.jsonl"
    metrics_path = output_dir / "rescue_metrics_summary.csv"
    audit_path = output_dir / "geometry_rescue_audit.json"

    assert records_path.exists()
    records = [json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 18
    assert all(record["direct_positive_decision"] is False for record in records)

    metrics_rows = list(csv.DictReader(metrics_path.open(encoding="utf-8")))
    full_row = next(row for row in metrics_rows if row["rescue_ablation_mode"] == "full_rescue")
    assert int(full_row["aligned_detection_record_count"]) == 3
    assert float(full_row["evidence_clean_fpr"]) >= float(full_row["raw_content_clean_fpr"])

    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert audit["protocol_decision"] == "pass"
    assert audit["direct_positive_decision_used"] is False
    assert audit["full_method_claim_ready"] is False
    assert manifest["artifact_id"] == "geometric_rescue_manifest"
