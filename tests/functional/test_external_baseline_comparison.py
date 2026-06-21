"""外部 baseline 共同协议对比的轻量功能测试。"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from experiments.baselines import default_baseline_specs
from scripts.write_external_baseline_comparison_outputs import write_external_baseline_comparison_outputs


@pytest.mark.quick
def test_default_baseline_specs_keep_missing_results_unsupported() -> None:
    """默认外部 baseline 只登记协议 adapter, 不伪造外部复现结果。"""
    specs = default_baseline_specs()

    assert len(specs) == 8
    assert {spec.comparison_group for spec in specs} == {"primary", "supplemental"}
    assert all(spec.baseline_adapter_ready for spec in specs)
    assert not any(spec.baseline_reproduced_result_ready for spec in specs)
    assert not any(spec.baseline_imported_result_ready for spec in specs)
    assert {spec.unsupported_reason for spec in specs} == {"external_baseline_result_missing"}


def write_input_artifacts(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    """写出外部 baseline 对比脚本需要的最小上游输入。"""
    attack_dir = tmp_path / "outputs" / "attack_matrix"
    threshold_dir = tmp_path / "outputs" / "threshold_calibration"
    attack_dir.mkdir(parents=True)
    threshold_dir.mkdir(parents=True)
    attack_manifest_path = attack_dir / "attack_manifest.json"
    attack_family_metrics_path = attack_dir / "attack_family_metrics.csv"
    attack_matrix_manifest_path = attack_dir / "manifest.local.json"
    threshold_report_path = threshold_dir / "threshold_degeneracy_report.json"

    attack_manifest_path.write_text(
        json.dumps(
            {
                "attack_metrics_ready": True,
                "evaluation_boundary": {
                    "target_fpr": 0.05,
                    "calibrated_content_threshold": 0.50,
                    "rescue_margin_low": -0.05,
                    "allowed_fail_reasons": ["geometry_suspected", "low_confidence"],
                },
                "supports_paper_claim": False,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    with attack_family_metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "attack_family",
                "attack_name",
                "resource_profile",
                "metric_status",
                "attack_record_count",
                "supported_record_count",
                "unsupported_record_count",
                "positive_count",
                "negative_count",
                "true_positive_rate",
                "false_positive_rate",
                "clean_false_positive_rate",
                "attacked_false_positive_rate",
                "quality_score_proxy_mean",
                "score_retention_mean",
                "supports_paper_claim",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "attack_family": "standard_distortion",
                "attack_name": "jpeg_compression",
                "resource_profile": "full_main",
                "metric_status": "measured_from_local_proxy",
                "attack_record_count": 6,
                "supported_record_count": 6,
                "unsupported_record_count": 0,
                "positive_count": 2,
                "negative_count": 4,
                "true_positive_rate": 0.5,
                "false_positive_rate": 0.25,
                "clean_false_positive_rate": 0.0,
                "attacked_false_positive_rate": 0.5,
                "quality_score_proxy_mean": 0.9,
                "score_retention_mean": 0.8,
                "supports_paper_claim": False,
            }
        )
        writer.writerow(
            {
                "attack_family": "regeneration_attack",
                "attack_name": "img2img_regeneration",
                "resource_profile": "full_extra",
                "metric_status": "unsupported",
                "attack_record_count": 6,
                "supported_record_count": 0,
                "unsupported_record_count": 6,
                "positive_count": 0,
                "negative_count": 0,
                "true_positive_rate": 0.0,
                "false_positive_rate": 0.0,
                "clean_false_positive_rate": 0.0,
                "attacked_false_positive_rate": 0.0,
                "quality_score_proxy_mean": 0.0,
                "score_retention_mean": 0.0,
                "supports_paper_claim": False,
            }
        )
    attack_matrix_manifest_path.write_text(
        json.dumps({"artifact_id": "attack_matrix_manifest", "config_digest": "digest"}, ensure_ascii=False),
        encoding="utf-8",
    )
    threshold_report_path.write_text(
        json.dumps({"target_fpr": 0.05, "threshold_degenerate": False, "supports_paper_claim": False}, ensure_ascii=False),
        encoding="utf-8",
    )
    return attack_manifest_path, attack_family_metrics_path, attack_matrix_manifest_path, threshold_report_path


@pytest.mark.quick
def test_external_baseline_outputs_are_rebuildable_and_claim_safe(tmp_path: Path) -> None:
    """外部 baseline 对比产物应由上游攻击矩阵和 baseline spec 重建, 且不支持论文主张。"""
    attack_manifest_path, attack_family_metrics_path, attack_matrix_manifest_path, threshold_report_path = (
        write_input_artifacts(tmp_path)
    )

    manifest = write_external_baseline_comparison_outputs(
        root=tmp_path,
        attack_manifest_path=attack_manifest_path,
        attack_family_metrics_path=attack_family_metrics_path,
        attack_matrix_manifest_path=attack_matrix_manifest_path,
        threshold_report_path=threshold_report_path,
    )
    output_dir = tmp_path / "outputs" / "external_baseline_comparison"
    observations = [
        json.loads(line)
        for line in (output_dir / "baseline_observations.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    imported_records = [
        json.loads(line)
        for line in (output_dir / "baseline_result_records.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    baseline_rows = list(csv.DictReader((output_dir / "baseline_metrics.csv").open(encoding="utf-8")))
    comparison_rows = list(csv.DictReader((output_dir / "baseline_comparison_table.csv").open(encoding="utf-8")))
    runtime_report = json.loads((output_dir / "baseline_runtime_report.json").read_text(encoding="utf-8"))

    assert manifest["artifact_id"] == "external_baseline_comparison_manifest"
    assert len(observations) == len(default_baseline_specs()) * 2
    assert imported_records == []
    assert len(baseline_rows) == len(default_baseline_specs())
    assert runtime_report["comparison_protocol_ready"] is True
    assert runtime_report["baseline_results_ready"] is False
    assert runtime_report["imported_baseline_result_count"] == 0
    assert {row["metric_status"] for row in baseline_rows} == {"unsupported"}
    assert any(row["method_id"] == "slm_wm_current" for row in comparison_rows)
    assert all(row["supports_paper_claim"] == "False" for row in comparison_rows)


@pytest.mark.quick
def test_external_baseline_imported_records_flow_into_comparison_table(tmp_path: Path) -> None:
    """受治理导入结果应进入观测记录, 聚合指标和共同协议对比表。"""
    attack_manifest_path, attack_family_metrics_path, attack_matrix_manifest_path, threshold_report_path = (
        write_input_artifacts(tmp_path)
    )
    source_root = tmp_path / "external_baseline" / "primary" / "tree_ring" / "source"
    source_root.mkdir(parents=True)
    source_registry_path = tmp_path / "external_baseline" / "source_registry.json"
    source_registry_path.write_text(
        json.dumps(
            {
                "baseline_sources": [
                    {
                        "baseline_id": "tree_ring",
                        "source_status": "downloaded",
                        "source_dir": "external_baseline/primary/tree_ring/source",
                        "official_repository_url": "git@example.invalid/tree-ring.git",
                        "official_repository_commit": "abc123",
                        "official_repository_branch": "main",
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    result_dir = tmp_path / "outputs" / "external_baseline_results"
    result_dir.mkdir(parents=True)
    evidence_path = result_dir / "tree_ring_metrics.csv"
    evidence_path.write_text("baseline_id,true_positive_rate\ntree_ring,0.7\n", encoding="utf-8")
    result_records_path = result_dir / "baseline_result_records.jsonl"
    result_records_path.write_text(
        json.dumps(
            {
                "baseline_id": "tree_ring",
                "attack_family": "standard_distortion",
                "attack_name": "jpeg_compression",
                "resource_profile": "full_main",
                "comparable_operating_point": "fixed_fpr_0.05",
                "result_protocol_name": "primary_baseline_formal_import_protocol",
                "result_source_type": "governed_import",
                "baseline_result_source": "outputs/external_baseline_results/tree_ring_metrics.csv",
                "baseline_result_source_digest": "tree_ring_digest",
                "metric_status": "measured",
                "positive_count": 10,
                "negative_count": 20,
                "attack_record_count": 30,
                "supported_record_count": 30,
                "true_positive_rate": 0.7,
                "false_positive_rate": 0.05,
                "clean_false_positive_rate": 0.0,
                "attacked_false_positive_rate": 0.1,
                "quality_score_proxy_mean": 0.88,
                "score_retention_mean": 0.77,
                "prompt_protocol_name": "paper_main_full_prompt_protocol",
                "prompt_protocol_digest": "prompt_digest",
                "adapter_boundary": "method_faithful_sd35_adapter_reproduction",
                "evidence_paths": ["outputs/external_baseline_results/tree_ring_metrics.csv"],
                "method_faithful_adapter_ready": True,
                "full_main_prompt_protocol_ready": True,
                "fixed_fpr_baseline_calibration_ready": True,
                "attack_matrix_baseline_detection_ready": True,
                "formal_evidence_paths_ready": True,
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    write_external_baseline_comparison_outputs(
        root=tmp_path,
        attack_manifest_path=attack_manifest_path,
        attack_family_metrics_path=attack_family_metrics_path,
        attack_matrix_manifest_path=attack_matrix_manifest_path,
        threshold_report_path=threshold_report_path,
        baseline_result_records_path=result_records_path,
        baseline_source_registry_path=source_registry_path,
    )

    output_dir = tmp_path / "outputs" / "external_baseline_comparison"
    baseline_rows = list(csv.DictReader((output_dir / "baseline_metrics.csv").open(encoding="utf-8")))
    comparison_rows = list(csv.DictReader((output_dir / "baseline_comparison_table.csv").open(encoding="utf-8")))
    runtime_report = json.loads((output_dir / "baseline_runtime_report.json").read_text(encoding="utf-8"))
    validation_report = json.loads((output_dir / "baseline_formal_import_validation_report.json").read_text(encoding="utf-8"))
    tree_row = next(row for row in baseline_rows if row["baseline_id"] == "tree_ring")
    tree_comparison_row = next(row for row in comparison_rows if row["method_id"] == "tree_ring")

    assert runtime_report["baseline_source_registry_ready"] is True
    assert runtime_report["official_source_ready_count"] == 1
    assert runtime_report["imported_baseline_result_count"] == 1
    assert runtime_report["accepted_formal_import_count"] == 1
    assert validation_report["formal_import_validation_ready"] is True
    assert runtime_report["baseline_result_ready_count"] == 1
    assert runtime_report["baseline_results_ready"] is False
    assert tree_row["metric_status"] == "measured"
    assert tree_row["baseline_official_code_ready"] == "True"
    assert tree_row["baseline_imported_result_ready"] == "True"
    assert tree_row["true_positive_rate"] == "0.7"
    assert tree_comparison_row["comparison_scope"] == "common_protocol_governed_result"
    assert tree_comparison_row["attacked_false_positive_rate"] == "0.1"
    assert tree_comparison_row["supports_paper_claim"] == "False"
