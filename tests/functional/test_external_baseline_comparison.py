"""外部 baseline 共同协议对比的轻量功能测试。"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from paper_experiments.baselines import default_baseline_specs
from scripts.write_external_baseline_comparison_outputs import (
    align_comparison_table_claim_scope,
    build_runtime_report,
    write_external_baseline_comparison_outputs,
)


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
                    "target_fpr": 0.01,
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
                "quality_score_mean",
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
                "metric_status": "measured_real_attacked_image_image_only_detection",
                "attack_record_count": 6,
                "supported_record_count": 6,
                "unsupported_record_count": 0,
                "positive_count": 2,
                "negative_count": 4,
                "true_positive_rate": 0.5,
                "false_positive_rate": 0.25,
                "clean_false_positive_rate": 0.0,
                "attacked_false_positive_rate": 0.5,
                "quality_score_mean": 0.9,
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
                "quality_score_mean": 0.0,
                "score_retention_mean": 0.0,
                "supports_paper_claim": False,
            }
        )
    attack_matrix_manifest_path.write_text(
        json.dumps({"artifact_id": "attack_matrix_manifest", "config_digest": "digest"}, ensure_ascii=False),
        encoding="utf-8",
    )
    threshold_report_path.write_text(
        json.dumps({"target_fpr": 0.01, "threshold_degenerate": False, "supports_paper_claim": False}, ensure_ascii=False),
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
    evidence_path_report = json.loads(
        (output_dir / "baseline_formal_evidence_path_resolution_report.json").read_text(encoding="utf-8")
    )

    assert manifest["artifact_id"] == "external_baseline_comparison_manifest"
    assert len(observations) == len(default_baseline_specs()) * 2
    assert imported_records == []
    assert len(baseline_rows) == len(default_baseline_specs())
    assert runtime_report["comparison_protocol_ready"] is False
    assert runtime_report["baseline_results_ready"] is False
    assert runtime_report["imported_baseline_result_count"] == 0
    assert runtime_report["primary_baseline_formal_ready"] is False
    assert runtime_report["formal_result_ready_count"] == 0
    assert runtime_report["blocked_primary_baseline_ids"] == [
        "tree_ring",
        "gaussian_shading",
        "shallow_diffuse",
        "t2smark",
    ]
    assert runtime_report["formal_template_record_count"] == 8
    assert runtime_report["missing_formal_template_count"] == 8
    assert runtime_report["formal_evidence_collection_task_count"] == 8
    assert runtime_report["missing_formal_evidence_collection_task_count"] == 8
    assert runtime_report["primary_baseline_formal_evidence_collection_ready"] is False
    assert runtime_report["formal_evidence_path_resolution_report_path"].endswith(
        "baseline_formal_evidence_path_resolution_report.json"
    )
    assert runtime_report["formal_evidence_path_reference_count"] == 0
    assert runtime_report["existing_formal_evidence_path_count"] == 0
    assert runtime_report["direct_formal_evidence_path_count"] == 0
    assert runtime_report["search_resolved_formal_evidence_path_count"] == 0
    assert runtime_report["missing_formal_evidence_path_count"] == 0
    assert runtime_report["formal_evidence_path_resolution_ready"] is False
    assert runtime_report["evidence_search_roots"] == []
    assert evidence_path_report["candidate_record_count"] == 0
    assert evidence_path_report["supports_paper_claim"] is False
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
                "comparable_operating_point": "fixed_fpr_0.01",
                "result_protocol_name": "primary_baseline_formal_import_protocol",
                "result_source_type": "governed_import",
                "baseline_result_source": "outputs/external_baseline_results/tree_ring_metrics.csv",
                "baseline_result_source_digest": "tree_ring_digest",
                "metric_status": "measured",
                "positive_count": 340,
                "negative_count": 340,
                "attacked_negative_count": 340,
                "attack_record_count": 680,
                "supported_record_count": 680,
                "true_positive_rate": 0.7,
                "false_positive_rate": 0.0,
                "clean_false_positive_rate": 0.0,
                "attacked_false_positive_rate": 0.1,
                "quality_score_mean": 0.88,
                "score_retention_mean": 0.77,
                "prompt_protocol_name": "paper_main_pilot_paper_prompt_protocol",
                "prompt_protocol_digest": "prompt_digest",
                "adapter_boundary": "method_faithful_sd35_adapter_reproduction",
                "evidence_paths": ["outputs/external_baseline_results/tree_ring_metrics.csv"],
                "method_faithful_adapter_ready": True,
                "paper_run_prompt_protocol_ready": True,
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
    evidence_path_report = json.loads(
        (output_dir / "baseline_formal_evidence_path_resolution_report.json").read_text(encoding="utf-8")
    )
    tree_row = next(row for row in baseline_rows if row["baseline_id"] == "tree_ring")
    tree_comparison_row = next(row for row in comparison_rows if row["method_id"] == "tree_ring")

    assert runtime_report["baseline_source_registry_ready"] is True
    assert runtime_report["official_source_ready_count"] == 1
    assert runtime_report["imported_baseline_result_count"] == 1
    assert runtime_report["accepted_formal_import_count"] == 1
    assert validation_report["formal_import_validation_ready"] is True
    assert runtime_report["primary_baseline_formal_ready"] is False
    assert runtime_report["formal_result_ready_count"] == 1
    assert runtime_report["blocked_primary_baseline_ids"] == ["gaussian_shading", "shallow_diffuse", "t2smark"]
    assert runtime_report["dominant_formal_import_blocking_reasons"] == ["candidate_record_missing"]
    assert runtime_report["formal_template_record_count"] == 8
    assert runtime_report["formal_template_coverage_ready_count"] == 0
    assert runtime_report["missing_formal_template_count"] == 7
    assert runtime_report["primary_baseline_formal_template_coverage_ready"] is False
    assert runtime_report["formal_evidence_collection_task_count"] == 8
    assert runtime_report["ready_formal_evidence_collection_task_count"] == 1
    assert runtime_report["missing_formal_evidence_collection_task_count"] == 7
    assert runtime_report["primary_baseline_formal_evidence_collection_ready"] is False
    assert runtime_report["formal_evidence_path_reference_count"] == 1
    assert runtime_report["existing_formal_evidence_path_count"] == 1
    assert runtime_report["direct_formal_evidence_path_count"] == 1
    assert runtime_report["search_resolved_formal_evidence_path_count"] == 0
    assert runtime_report["missing_formal_evidence_path_count"] == 0
    assert runtime_report["formal_evidence_path_resolution_ready"] is True
    assert runtime_report["evidence_search_roots"] == []
    assert runtime_report["formal_evidence_path_missing_baseline_ids"] == []
    assert evidence_path_report["formal_evidence_path_reference_count"] == 1
    assert evidence_path_report["formal_evidence_path_resolution_ready"] is True
    assert runtime_report["baseline_result_ready_count"] == 1
    assert runtime_report["baseline_results_ready"] is False
    assert tree_row["metric_status"] == "measured"
    assert tree_row["baseline_official_code_ready"] == "True"
    assert tree_row["baseline_imported_result_ready"] == "True"
    assert tree_row["true_positive_rate"] == "0.7"
    assert tree_comparison_row["comparison_scope"] == "common_protocol_governed_result"
    assert tree_comparison_row["attacked_false_positive_rate"] == "0.1"
    assert tree_comparison_row["supports_paper_claim"] == "False"


@pytest.mark.quick
def test_external_baseline_primary_comparison_rows_share_common_claim_scope() -> None:
    """主表 baseline 全部完成时, 对比表应与共同协议使用同一 claim 标记口径。"""

    baseline_rows = [
        {
            "baseline_id": baseline_id,
            "baseline_family": "diffusion_watermark",
            "baseline_name": baseline_id,
            "comparison_group": "primary",
            "baseline_adapter_ready": True,
            "baseline_official_code_ready": False,
            "baseline_reproduced_result_ready": False,
            "baseline_imported_result_ready": True,
            "baseline_result_source": f"outputs/external_baseline_results/{baseline_id}.csv",
            "baseline_protocol_compatible": True,
            "baseline_requires_gpu": True,
            "baseline_requires_training": False,
            "baseline_observation_count": 18,
            "baseline_result_ready_count": 18,
            "unsupported_record_count": 0,
            "metric_status": "measured",
            "true_positive_rate": 0.7,
            "false_positive_rate": 0.01,
            "clean_false_positive_rate": 0.0,
            "attacked_false_positive_rate": 0.02,
            "quality_score_mean": 0.9,
            "score_retention_mean": 0.8,
            "unsupported_reason": "",
            "supports_paper_claim": False,
        }
        for baseline_id in ("tree_ring", "gaussian_shading", "shallow_diffuse", "t2smark")
    ]
    baseline_rows.append(
        {
            "baseline_id": "stable_signature",
            "baseline_family": "decoder_signature_watermark",
            "baseline_name": "Stable Signature",
            "comparison_group": "supplemental",
            "baseline_adapter_ready": True,
            "baseline_official_code_ready": False,
            "baseline_reproduced_result_ready": False,
            "baseline_imported_result_ready": False,
            "baseline_result_source": "not_available",
            "baseline_protocol_compatible": True,
            "baseline_requires_gpu": True,
            "baseline_requires_training": False,
            "baseline_observation_count": 18,
            "baseline_result_ready_count": 0,
            "unsupported_record_count": 18,
            "metric_status": "unsupported",
            "true_positive_rate": "unsupported",
            "false_positive_rate": "unsupported",
            "clean_false_positive_rate": "unsupported",
            "attacked_false_positive_rate": "unsupported",
            "quality_score_mean": "unsupported",
            "score_retention_mean": "unsupported",
            "unsupported_reason": "external_baseline_result_missing",
            "supports_paper_claim": False,
        }
    )
    comparison_rows = [
        {
            "method_id": "slm_wm_current",
            "method_role": "proposed_method",
            "comparison_scope": "common_protocol_real_image_detection",
            "common_prompt_protocol_ready": True,
            "common_attack_protocol_ready": True,
            "common_threshold_protocol_ready": True,
            "metric_status": "measured_real_attacked_image_image_only_detection",
            "true_positive_rate": 0.84,
            "false_positive_rate": 0.01,
            "clean_false_positive_rate": 0.0,
            "attacked_false_positive_rate": 0.02,
            "quality_score_mean": 0.9,
            "score_retention_mean": 0.8,
            "supports_paper_claim": False,
        },
        *[
            {
                "method_id": row["baseline_id"],
                "method_role": f"external_baseline_{row['comparison_group']}",
                "comparison_scope": "common_protocol_governed_result"
                if row["metric_status"] != "unsupported"
                else "common_protocol_result_missing",
                "common_prompt_protocol_ready": True,
                "common_attack_protocol_ready": True,
                "common_threshold_protocol_ready": True,
                "metric_status": row["metric_status"],
                "true_positive_rate": row["true_positive_rate"],
                "false_positive_rate": row["false_positive_rate"],
                "clean_false_positive_rate": row["clean_false_positive_rate"],
                "attacked_false_positive_rate": row["attacked_false_positive_rate"],
                "quality_score_mean": row["quality_score_mean"],
                "score_retention_mean": row["score_retention_mean"],
                "supports_paper_claim": False,
            }
            for row in baseline_rows
        ],
    ]
    runtime_report = build_runtime_report(
        {
            "attack_metrics_ready": True,
            "supports_paper_claim": True,
            "evaluation_boundary": {"target_fpr": 0.01},
        },
        {"threshold_degenerate": False},
        baseline_rows,
        tuple(),
        {"baseline_sources": [{"baseline_id": "tree_ring"}]},
        imported_result_count=72,
        formal_import_validation={
            "formal_import_validation_ready": True,
            "input_record_count": 72,
            "accepted_formal_import_count": 72,
            "rejected_formal_import_count": 0,
            "formal_import_issue_count": 0,
        },
        formal_import_readiness_summary={
            "primary_baseline_formal_ready": True,
            "formal_result_ready_count": 4,
            "blocked_primary_baseline_ids": [],
        },
        formal_template_coverage_summary={
            "primary_baseline_formal_template_coverage_ready": True,
            "formal_template_record_count": 36,
            "candidate_template_match_count": 36,
            "accepted_template_match_count": 36,
            "formal_template_coverage_ready_count": 4,
            "missing_candidate_template_count": 0,
            "missing_formal_template_count": 0,
        },
        formal_evidence_collection_summary={
            "formal_evidence_collection_task_count": 36,
            "ready_formal_evidence_collection_task_count": 36,
            "missing_formal_evidence_collection_task_count": 0,
            "primary_baseline_formal_evidence_collection_ready": True,
        },
        formal_evidence_path_summary={
            "formal_evidence_path_reference_count": 72,
            "existing_formal_evidence_path_count": 72,
            "direct_formal_evidence_path_count": 72,
            "search_resolved_formal_evidence_path_count": 0,
            "missing_formal_evidence_path_count": 0,
            "formal_evidence_path_resolution_ready": True,
            "evidence_search_roots": [],
            "formal_evidence_path_missing_baseline_ids": [],
        },
        formal_evidence_path_summary_path="outputs/external_baseline_comparison/baseline_formal_evidence_path_resolution_report.json",
    )

    align_comparison_table_claim_scope(baseline_rows, comparison_rows, runtime_report)

    slm_row = next(row for row in comparison_rows if row["method_id"] == "slm_wm_current")
    primary_rows = [row for row in comparison_rows if row["method_role"] == "external_baseline_primary"]
    supplemental_row = next(row for row in comparison_rows if row["method_id"] == "stable_signature")

    assert runtime_report["primary_baseline_results_ready"] is True
    assert runtime_report["baseline_results_ready"] is False
    assert runtime_report["comparison_table_supports_paper_claim"] is True
    assert runtime_report["supports_paper_claim"] is True
    assert slm_row["method_role"] == "proposed_method_governed_result"
    assert slm_row["comparison_scope"] == "common_protocol_governed_result"
    assert slm_row["metric_status"] == "measured_from_attack_matrix_formal_records"
    assert slm_row["supports_paper_claim"] is True
    assert all(row["supports_paper_claim"] is True for row in primary_rows)
    assert all(row["supports_paper_claim"] is True for row in baseline_rows if row["comparison_group"] == "primary")
    assert supplemental_row["supports_paper_claim"] is False


@pytest.mark.quick
def test_external_baseline_evidence_paths_can_resolve_from_explicit_mirror_root(tmp_path: Path) -> None:
    """显式镜像根目录中的 evidence 文件应能进入 schema 层路径解析, 但不改变正式导入规则。"""

    attack_manifest_path, attack_family_metrics_path, attack_matrix_manifest_path, threshold_report_path = (
        write_input_artifacts(tmp_path)
    )
    result_dir = tmp_path / "outputs" / "external_baseline_results"
    result_dir.mkdir(parents=True)
    mirror_root = tmp_path / "drive_mirror" / "SLM"
    mirror_evidence_dir = mirror_root / "external_baseline_method_faithful"
    mirror_evidence_dir.mkdir(parents=True)
    mirror_evidence_path = mirror_evidence_dir / "tree_ring_metrics.csv"
    mirror_evidence_path.write_text("baseline_id,true_positive_rate\ntree_ring,0.7\n", encoding="utf-8")
    result_records_path = result_dir / "baseline_result_records.jsonl"
    result_records_path.write_text(
        json.dumps(
            {
                "baseline_id": "tree_ring",
                "attack_family": "standard_distortion",
                "attack_name": "jpeg_compression",
                "resource_profile": "full_main",
                "comparable_operating_point": "fixed_fpr_0.01",
                "result_protocol_name": "primary_baseline_formal_import_protocol",
                "result_source_type": "governed_import",
                "baseline_result_source": "outputs/external_baseline_results/tree_ring_metrics.csv",
                "baseline_result_source_digest": "tree_ring_digest",
                "metric_status": "measured",
                "positive_count": 340,
                "negative_count": 340,
                "attacked_negative_count": 340,
                "attack_record_count": 680,
                "supported_record_count": 680,
                "true_positive_rate": 0.7,
                "false_positive_rate": 0.0,
                "clean_false_positive_rate": 0.0,
                "attacked_false_positive_rate": 0.1,
                "quality_score_mean": 0.88,
                "score_retention_mean": 0.77,
                "prompt_protocol_name": "paper_main_pilot_paper_prompt_protocol",
                "prompt_protocol_digest": "prompt_digest",
                "adapter_boundary": "method_faithful_sd35_adapter_reproduction",
                "evidence_paths": ["outputs/external_baseline_results/tree_ring_metrics.csv"],
                "method_faithful_adapter_ready": True,
                "paper_run_prompt_protocol_ready": True,
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
        evidence_search_roots=(mirror_root,),
    )

    output_dir = tmp_path / "outputs" / "external_baseline_comparison"
    runtime_report = json.loads((output_dir / "baseline_runtime_report.json").read_text(encoding="utf-8"))
    validation_report = json.loads((output_dir / "baseline_formal_import_validation_report.json").read_text(encoding="utf-8"))
    evidence_path_report = json.loads(
        (output_dir / "baseline_formal_evidence_path_resolution_report.json").read_text(encoding="utf-8")
    )

    assert validation_report["formal_import_validation_ready"] is True
    assert runtime_report["formal_evidence_path_reference_count"] == 1
    assert runtime_report["existing_formal_evidence_path_count"] == 1
    assert runtime_report["direct_formal_evidence_path_count"] == 0
    assert runtime_report["search_resolved_formal_evidence_path_count"] == 1
    assert runtime_report["missing_formal_evidence_path_count"] == 0
    assert runtime_report["formal_evidence_path_resolution_ready"] is True
    assert runtime_report["evidence_search_roots"] == [mirror_root.resolve().as_posix()]
    assert evidence_path_report["resolved_formal_evidence_paths"] == [mirror_evidence_path.resolve().as_posix()]

