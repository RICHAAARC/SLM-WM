"""pilot_paper 论文结果分析表与失败案例图的轻量功能测试。"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from scripts.write_pilot_paper_result_analysis_outputs import write_pilot_paper_result_analysis_outputs


pytestmark = pytest.mark.quick


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    """写出测试用 JSONL。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _result_record(method_id: str, attack_name: str, tpr: float, ci_low: float, ci_high: float) -> dict[str, object]:
    """构造一条最小结果记录。"""

    return {
        "paper_claim_scale": "pilot_paper",
        "method_id": method_id,
        "attack_family": "standard_distortion",
        "attack_name": attack_name,
        "resource_profile": "full_main",
        "metric_status": "measured",
        "true_positive_rate": tpr,
        "true_positive_rate_ci_low": ci_low,
        "true_positive_rate_ci_high": ci_high,
        "false_positive_rate": 0.0,
        "false_positive_rate_ci_low": 0.0,
        "false_positive_rate_ci_high": 0.0,
        "clean_false_positive_rate": 0.0,
        "clean_false_positive_rate_ci_low": 0.0,
        "clean_false_positive_rate_ci_high": 0.0,
        "attacked_false_positive_rate": 0.0,
        "attacked_false_positive_rate_ci_low": 0.0,
        "attacked_false_positive_rate_ci_high": 0.0,
        "positive_count": 10,
        "negative_count": 10,
        "bootstrap_iteration_count": 1000,
        "confidence_level": 0.95,
        "supports_paper_claim": True,
    }


def test_pilot_paper_result_analysis_rebuilds_tables_and_failure_figure(tmp_path: Path) -> None:
    """结果分析脚本应从 governed records 重建 CI 表、优势表和失败案例 SVG。"""

    result_records_path = tmp_path / "outputs" / "pilot_paper_fixed_fpr_results" / "pilot_paper_result_records.jsonl"
    _write_jsonl(
        result_records_path,
        [
            _result_record("slm_wm_current", "jpeg_compression", 0.9, 0.82, 0.95),
            _result_record("tree_ring", "jpeg_compression", 0.6, 0.50, 0.70),
            _result_record("gaussian_shading", "jpeg_compression", 0.5, 0.40, 0.60),
            _result_record("shallow_diffuse", "jpeg_compression", 0.7, 0.60, 0.78),
            _result_record("t2smark", "jpeg_compression", 0.65, 0.55, 0.74),
        ],
    )
    attacked_image = (
        tmp_path
        / "outputs"
        / "real_attack_evaluation"
        / "attacked_images"
        / "sample_aligned_jpeg_compression.png"
    )
    attacked_image.parent.mkdir(parents=True)
    attacked_image.write_bytes(b"not_a_real_png_for_svg_href_only")
    real_records_path = tmp_path / "outputs" / "real_attack_evaluation" / "formal_attack_detection_records.jsonl"
    _write_jsonl(
        real_records_path,
        [
            {
                "sample_role": "positive_source",
                "attack_family": "standard_distortion",
                "attack_name": "jpeg_compression",
                "source_record_id": "source_1",
                "attack_record_id": "attack_1",
                "aligned_content_score_after": 0.1,
                "aligned_content_score_before": 0.9,
                "score_retention": 0.11,
                "evidence_decision": False,
                "attacked_image_digest": "attacked_digest",
                "source_image_digest": "source_digest",
                "metadata": {
                    "attacked_image_path": "outputs/real_attack_evaluation/attacked_images/sample_aligned_jpeg_compression.png"
                },
                "supports_paper_claim": True,
            }
        ],
    )
    conventional_records_path = (
        tmp_path / "outputs" / "conventional_geometric_attack_evaluation" / "formal_attack_detection_records.jsonl"
    )
    _write_jsonl(conventional_records_path, [])

    manifest = write_pilot_paper_result_analysis_outputs(
        root=tmp_path,
        result_records_path=result_records_path,
        real_attack_formal_records_path=real_records_path,
        conventional_attack_formal_records_path=conventional_records_path,
        failure_case_limit=4,
    )

    output_dir = tmp_path / "outputs" / "pilot_paper_result_analysis"
    bootstrap_rows = list(csv.DictReader((output_dir / "bootstrap_ci_table.csv").open(encoding="utf-8")))
    superiority_rows = list(csv.DictReader((output_dir / "per_attack_superiority_table.csv").open(encoding="utf-8")))
    failure_rows = [
        json.loads(line)
        for line in (output_dir / "failure_case_records.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    svg_text = (output_dir / "failure_case_figure.svg").read_text(encoding="utf-8")
    summary = json.loads((output_dir / "result_analysis_summary.json").read_text(encoding="utf-8"))

    assert manifest["artifact_id"] == "pilot_paper_result_analysis_manifest"
    assert len(bootstrap_rows) == 5
    assert len(superiority_rows) == 1
    assert superiority_rows[0]["best_baseline_id"] == "shallow_diffuse"
    assert float(superiority_rows[0]["slm_minus_best_baseline_tpr"]) == pytest.approx(0.2)
    assert superiority_rows[0]["superiority_claim_ready"] == "True"
    assert len(failure_rows) == 1
    assert failure_rows[0]["attacked_image_digest"] == "attacked_digest"
    assert "jpeg_compression" in svg_text
    assert summary["failure_case_figure_ready"] is True
