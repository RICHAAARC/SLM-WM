"""主表 external baseline 小样本证据边界测试。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.baselines import (
    EXCLUDED_OPERATING_POINTS,
    build_primary_baseline_small_sample_evidence_records,
    build_primary_baseline_small_sample_evidence_summary,
)
from scripts.write_primary_baseline_small_sample_evidence_outputs import (
    write_primary_baseline_small_sample_evidence_outputs,
)


def candidate_row(baseline_id: str, resource_profile: str) -> dict[str, object]:
    """构造一个具备证据路径但尚未通过正式导入的候选记录。"""

    return {
        "baseline_id": baseline_id,
        "baseline_result_record_id": f"record_{baseline_id}",
        "baseline_result_digest": f"digest_{baseline_id}",
        "resource_profile": resource_profile,
        "positive_count": 5,
        "negative_count": 5,
        "supported_record_count": 10,
        "attack_record_count": 10,
        "evidence_paths": [f"outputs/external_baseline_results/{baseline_id}.json"],
        "formal_evidence_paths_ready": True,
        "supports_paper_claim": False,
    }


def validation_report() -> dict[str, object]:
    """构造 validator 拒绝报告, 用于验证小样本边界不会变成论文级 claim。"""

    return {
        "issues": [
            {"row_index": 0, "baseline_id": "tree_ring", "field_name": "resource_profile", "reason": "full_main_resource_profile_required"},
            {"row_index": 1, "baseline_id": "gaussian_shading", "field_name": "resource_profile", "reason": "full_main_resource_profile_required"},
            {"row_index": 2, "baseline_id": "shallow_diffuse", "field_name": "resource_profile", "reason": "full_main_resource_profile_required"},
            {"row_index": 3, "baseline_id": "t2smark", "field_name": "fixed_fpr_baseline_calibration_ready", "reason": "fixed_fpr_baseline_calibration_ready_required"},
        ]
    }


@pytest.mark.quick
def test_small_sample_evidence_records_are_ready_but_not_paper_claims() -> None:
    """四个 baseline 均有小样本证据时, 摘要应 ready, 但论文级 claim 仍保持关闭。"""

    rows = [
        candidate_row("tree_ring", "gpu_smoke"),
        candidate_row("gaussian_shading", "gpu_smoke"),
        candidate_row("shallow_diffuse", "gpu_smoke"),
        candidate_row("t2smark", "full_main"),
    ]

    records = build_primary_baseline_small_sample_evidence_records(rows, validation_report())
    summary = build_primary_baseline_small_sample_evidence_summary(records)

    assert len(records) == 4
    assert summary["small_sample_evidence_ready"] is True
    assert summary["covered_primary_baseline_count"] == 4
    assert summary["formal_import_ready_count"] == 0
    assert summary["formal_full_paper_run_requested"] is False
    assert summary["formal_full_paper_run_permitted"] is False
    assert tuple(summary["excluded_operating_points"]) == EXCLUDED_OPERATING_POINTS
    assert summary["paper_claim_ready"] is False
    assert all(record.supports_paper_claim is False for record in records)
    assert all(record.excluded_operating_points == EXCLUDED_OPERATING_POINTS for record in records)


@pytest.mark.quick
def test_small_sample_evidence_writer_outputs_rebuildable_artifacts(tmp_path: Path) -> None:
    """写出脚本应从候选 records 和 validator 报告重建小样本证据产物。"""

    candidate_path = tmp_path / "outputs" / "external_baseline_results" / "baseline_result_records.jsonl"
    validation_path = tmp_path / "outputs" / "external_baseline_results" / "baseline_result_candidate_validation_report.json"
    candidate_path.parent.mkdir(parents=True)
    rows = [
        candidate_row("tree_ring", "gpu_smoke"),
        candidate_row("gaussian_shading", "gpu_smoke"),
        candidate_row("shallow_diffuse", "gpu_smoke"),
        candidate_row("t2smark", "full_main"),
    ]
    candidate_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    validation_path.write_text(json.dumps(validation_report(), ensure_ascii=False), encoding="utf-8")

    manifest = write_primary_baseline_small_sample_evidence_outputs(root=tmp_path)
    output_dir = tmp_path / "outputs" / "primary_baseline_small_sample_evidence"
    summary = json.loads((output_dir / "primary_baseline_small_sample_evidence_summary.json").read_text(encoding="utf-8"))
    records = [
        json.loads(line)
        for line in (output_dir / "primary_baseline_small_sample_evidence_records.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert manifest["artifact_id"] == "primary_baseline_small_sample_evidence_manifest"
    assert len(records) == 4
    assert summary["small_sample_evidence_ready"] is True
    assert summary["supports_paper_claim"] is False
    assert all(str(path).startswith("outputs/") for path in manifest["output_paths"])
