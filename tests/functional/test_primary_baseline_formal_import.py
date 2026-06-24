"""主表 external baseline 正式结果导入协议的轻量功能测试。"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from experiments.baselines import (
    build_primary_baseline_formal_evidence_collection_rows,
    build_primary_baseline_formal_evidence_collection_summary,
    build_primary_baseline_formal_import_schema,
    build_primary_baseline_formal_template_coverage_rows,
    build_primary_baseline_formal_template_coverage_summary,
    build_t2smark_full_main_candidate_records,
    build_tree_ring_method_faithful_candidate_records,
    validate_primary_baseline_formal_import_rows,
)
from scripts.write_primary_baseline_formal_import_protocol import write_primary_baseline_formal_import_protocol_outputs


def formal_tree_ring_row(evidence_path: str) -> dict[str, object]:
    """构造一条满足正式导入 schema 的最小主表 baseline 记录。"""

    return {
        "baseline_id": "tree_ring",
        "attack_family": "standard_distortion",
        "attack_name": "jpeg_compression",
        "resource_profile": "full_main",
        "comparable_operating_point": "fixed_fpr_0.05",
        "result_protocol_name": "primary_baseline_formal_import_protocol",
        "result_source_type": "governed_import",
        "baseline_result_source": evidence_path,
        "baseline_result_source_digest": "digest",
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
        "prompt_protocol_name": "paper_main_full_paper_prompt_protocol",
        "prompt_protocol_digest": "prompt_digest",
        "adapter_boundary": "method_faithful_sd35_adapter_reproduction",
        "evidence_paths": [evidence_path],
        "method_faithful_adapter_ready": True,
        "full_main_prompt_protocol_ready": True,
        "fixed_fpr_baseline_calibration_ready": True,
        "attack_matrix_baseline_detection_ready": True,
        "formal_evidence_paths_ready": True,
    }


@pytest.mark.quick
def test_formal_import_validator_accepts_governed_full_main_record(tmp_path: Path) -> None:
    """完整边界均满足时, validator 应接受主表正式导入记录。"""

    evidence_path = tmp_path / "outputs" / "external_baseline_results" / "tree_ring_metrics.csv"
    evidence_path.parent.mkdir(parents=True)
    evidence_path.write_text("baseline_id,true_positive_rate\ntree_ring,0.7\n", encoding="utf-8")
    row = formal_tree_ring_row("outputs/external_baseline_results/tree_ring_metrics.csv")

    report = validate_primary_baseline_formal_import_rows([row], evidence_root=tmp_path, target_fpr=0.05)

    assert report["overall_decision"] == "pass"
    assert report["accepted_formal_import_count"] == 1
    assert report["formal_import_validation_ready"] is True
    assert report["accepted_records"][0]["baseline_id"] == "tree_ring"


@pytest.mark.quick
def test_formal_import_validator_rejects_smoke_boundary_and_missing_readiness(tmp_path: Path) -> None:
    """GPU smoke adapter observation 不得被升级为主表正式结果。"""

    evidence_path = tmp_path / "outputs" / "external_baseline_results" / "tree_ring_metrics.csv"
    evidence_path.parent.mkdir(parents=True)
    evidence_path.write_text("baseline_id,true_positive_rate\ntree_ring,0.7\n", encoding="utf-8")
    row = formal_tree_ring_row("outputs/external_baseline_results/tree_ring_metrics.csv")
    row["adapter_boundary"] = "sd35_latent_smoke_adapter_not_formal_external_baseline_evidence"
    row["fixed_fpr_baseline_calibration_ready"] = False

    report = validate_primary_baseline_formal_import_rows([row], evidence_root=tmp_path, target_fpr=0.05)
    reasons = {issue["reason"] for issue in report["issues"]}

    assert report["overall_decision"] == "fail"
    assert report["accepted_formal_import_count"] == 0
    assert "adapter_boundary_not_formal" in reasons
    assert "fixed_fpr_baseline_calibration_ready_required" in reasons


@pytest.mark.quick
def test_t2smark_candidate_records_remain_rejected_until_attack_and_threshold_ready(tmp_path: Path) -> None:
    """T2SMark full-main 候选记录在攻击矩阵和 fixed-FPR 未闭合前应保持未通过正式导入。"""

    evidence_path = tmp_path / "outputs" / "t2smark_full_main_reproduction" / "results.json"
    evidence_path.parent.mkdir(parents=True)
    evidence_path.write_text('{"0":{"robustness":{"norm1_no_w":0.1,"norm1_w":0.9}}}\n', encoding="utf-8")
    observations = [
        {"baseline_id": "t2smark", "attack_family": "clean", "attack_condition": "clean_none", "sample_role": "clean_negative", "detection_decision": False},
        {"baseline_id": "t2smark", "attack_family": "clean", "attack_condition": "clean_none", "sample_role": "positive_source", "detection_decision": True},
    ]

    records = build_t2smark_full_main_candidate_records(
        observation_rows=observations,
        target_fpr=0.05,
        baseline_result_source="outputs/t2smark_full_main_reproduction/results.json",
        baseline_result_source_digest="digest",
        evidence_paths=["outputs/t2smark_full_main_reproduction/results.json"],
        prompt_protocol_digest="prompt_digest",
        full_main_prompt_protocol_ready=True,
        fixed_fpr_baseline_calibration_ready=False,
        attack_matrix_baseline_detection_ready=False,
    )
    report = validate_primary_baseline_formal_import_rows(records, evidence_root=tmp_path, target_fpr=0.05)
    reasons = {issue["reason"] for issue in report["issues"]}

    assert len(records) == 1
    assert records[0]["adapter_boundary"] == "sd35_medium_native_official_reproduction"
    assert report["accepted_formal_import_count"] == 0
    assert "fixed_fpr_baseline_calibration_ready_required" in reasons
    assert "attack_matrix_baseline_detection_ready_required" in reasons


@pytest.mark.quick
def test_tree_ring_method_faithful_candidate_records_are_schema_compatible(tmp_path: Path) -> None:
    """Tree-Ring 方法忠实 observations 应能聚合为 formal import 候选记录。"""

    evidence_path = tmp_path / "outputs" / "tree_ring_method_faithful" / "baseline_observations.json"
    evidence_path.parent.mkdir(parents=True)
    evidence_path.write_text("[]\n", encoding="utf-8")
    observations = [
        {
            "baseline_id": "tree_ring",
            "attack_family": "clean",
            "attack_condition": "clean_none",
            "sample_role": "clean_negative",
            "detection_decision": False,
            "quality_score_proxy": 1.0,
            "score_retention_proxy": 1.0,
        },
        {
            "baseline_id": "tree_ring",
            "attack_family": "clean",
            "attack_condition": "clean_none",
            "sample_role": "positive_source",
            "detection_decision": True,
            "quality_score_proxy": 1.0,
            "score_retention_proxy": 1.0,
        },
    ]

    records = build_tree_ring_method_faithful_candidate_records(
        observation_rows=observations,
        target_fpr=0.05,
        baseline_result_source="outputs/tree_ring_method_faithful/baseline_observations.json",
        baseline_result_source_digest="digest",
        evidence_paths=["outputs/tree_ring_method_faithful/baseline_observations.json"],
        prompt_protocol_digest="prompt_digest",
        full_main_prompt_protocol_ready=True,
        fixed_fpr_baseline_calibration_ready=True,
        attack_matrix_baseline_detection_ready=True,
    )
    report = validate_primary_baseline_formal_import_rows(records, evidence_root=tmp_path, target_fpr=0.05)

    assert len(records) == 1
    assert records[0]["baseline_id"] == "tree_ring"
    assert records[0]["adapter_boundary"] == "method_faithful_sd35_adapter_reproduction"
    assert records[0]["result_source_type"] == "governed_import"
    assert report["accepted_formal_import_count"] == 1


@pytest.mark.quick
def test_formal_template_coverage_requires_matching_full_main_records(tmp_path: Path) -> None:
    """正式模板覆盖应检查候选记录是否覆盖共同协议要求的 full-main 攻击模板。"""

    evidence_path = tmp_path / "outputs" / "external_baseline_results" / "tree_ring_metrics.csv"
    evidence_path.parent.mkdir(parents=True)
    evidence_path.write_text("baseline_id,true_positive_rate\ntree_ring,0.7\n", encoding="utf-8")
    accepted_row = formal_tree_ring_row("outputs/external_baseline_results/tree_ring_metrics.csv")
    missing_template = {
        "baseline_id": "tree_ring",
        "attack_family": "standard_distortion",
        "attack_name": "gaussian_noise",
        "resource_profile": "full_main",
        "comparable_operating_point": "fixed_fpr_0.05",
    }
    template_rows = [
        {
            "baseline_id": "tree_ring",
            "attack_family": "standard_distortion",
            "attack_name": "jpeg_compression",
            "resource_profile": "full_main",
            "comparable_operating_point": "fixed_fpr_0.05",
        },
        missing_template,
    ]
    report = validate_primary_baseline_formal_import_rows([accepted_row], evidence_root=tmp_path, target_fpr=0.05)

    coverage_rows = build_primary_baseline_formal_template_coverage_rows(template_rows, [accepted_row], report)
    coverage_summary = build_primary_baseline_formal_template_coverage_summary(coverage_rows)
    tree_row = next(row for row in coverage_rows if row["baseline_id"] == "tree_ring")

    assert tree_row["expected_formal_template_count"] == 2
    assert tree_row["accepted_template_match_count"] == 1
    assert tree_row["missing_formal_template_count"] == 1
    assert tree_row["formal_template_coverage_ready"] is False
    assert coverage_summary["primary_baseline_formal_template_coverage_ready"] is False


@pytest.mark.quick
def test_formal_template_coverage_separates_candidate_and_accepted_matches(tmp_path: Path) -> None:
    """已有候选但未通过 validator 时, 摘要应保留候选覆盖进度并继续阻断正式结论。"""

    evidence_path = tmp_path / "outputs" / "external_baseline_results" / "tree_ring_metrics.csv"
    evidence_path.parent.mkdir(parents=True)
    evidence_path.write_text("baseline_id,true_positive_rate\ntree_ring,0.7\n", encoding="utf-8")
    candidate_row = formal_tree_ring_row("outputs/external_baseline_results/tree_ring_metrics.csv")
    candidate_row["fixed_fpr_baseline_calibration_ready"] = False
    template_rows = [
        {
            "baseline_id": "tree_ring",
            "attack_family": "standard_distortion",
            "attack_name": "jpeg_compression",
            "resource_profile": "full_main",
            "comparable_operating_point": "fixed_fpr_0.05",
        }
    ]
    report = validate_primary_baseline_formal_import_rows([candidate_row], evidence_root=tmp_path, target_fpr=0.05)

    coverage_rows = build_primary_baseline_formal_template_coverage_rows(template_rows, [candidate_row], report)
    coverage_summary = build_primary_baseline_formal_template_coverage_summary(coverage_rows)

    assert report["accepted_formal_import_count"] == 0
    assert coverage_summary["candidate_template_match_count"] == 1
    assert coverage_summary["accepted_template_match_count"] == 0
    assert coverage_summary["missing_candidate_template_count"] == 0
    assert coverage_summary["missing_formal_template_count"] == 1
    assert coverage_summary["primary_baseline_formal_template_coverage_ready"] is False


@pytest.mark.quick
def test_formal_evidence_collection_plan_marks_missing_templates(tmp_path: Path) -> None:
    """正式证据收集计划应把未通过正式导入的模板转换为可执行补证任务。"""

    evidence_path = tmp_path / "outputs" / "external_baseline_results" / "tree_ring_metrics.csv"
    evidence_path.parent.mkdir(parents=True)
    evidence_path.write_text("baseline_id,true_positive_rate\ntree_ring,0.7\n", encoding="utf-8")
    accepted_row = formal_tree_ring_row("outputs/external_baseline_results/tree_ring_metrics.csv")
    missing_template = {
        "baseline_id": "tree_ring",
        "attack_family": "standard_distortion",
        "attack_name": "gaussian_noise",
        "resource_profile": "full_main",
        "comparable_operating_point": "fixed_fpr_0.05",
        "required_metric_fields": ["true_positive_rate"],
        "required_source_fields": ["baseline_result_source"],
    }
    template_rows = [
        {
            "baseline_id": "tree_ring",
            "attack_family": "standard_distortion",
            "attack_name": "jpeg_compression",
            "resource_profile": "full_main",
            "comparable_operating_point": "fixed_fpr_0.05",
            "required_metric_fields": ["true_positive_rate"],
            "required_source_fields": ["baseline_result_source"],
        },
        missing_template,
    ]
    report = validate_primary_baseline_formal_import_rows([accepted_row], evidence_root=tmp_path, target_fpr=0.05)

    collection_rows = build_primary_baseline_formal_evidence_collection_rows(template_rows, [accepted_row], report)
    collection_summary = build_primary_baseline_formal_evidence_collection_summary(collection_rows)
    missing_row = next(row for row in collection_rows if row["attack_name"] == "gaussian_noise")

    assert len(collection_rows) == 2
    assert missing_row["formal_evidence_collection_ready"] is False
    assert "generate_full_main_baseline_result_record" in missing_row["required_collection_actions"]
    assert collection_summary["formal_evidence_collection_task_count"] == 2
    assert collection_summary["missing_formal_evidence_collection_task_count"] == 1
    assert collection_summary["primary_baseline_formal_evidence_collection_ready"] is False


@pytest.mark.quick
def test_formal_import_protocol_writer_outputs_schema_template_and_validation(tmp_path: Path) -> None:
    """协议写出脚本应生成 schema、模板、候选校验报告和 manifest。"""

    source_root = tmp_path / "external_baseline" / "primary" / "tree_ring" / "source"
    source_root.mkdir(parents=True)
    registry_path = tmp_path / "external_baseline" / "source_registry.json"
    registry_path.write_text(
        json.dumps(
            {
                "baseline_sources": [
                    {
                        "baseline_id": "tree_ring",
                        "baseline_name": "Tree-Ring",
                        "baseline_family": "diffusion_latent_watermark",
                        "comparison_group": "primary",
                        "source_status": "downloaded",
                        "source_dir": "external_baseline/primary/tree_ring/source",
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    attack_dir = tmp_path / "outputs" / "attack_matrix"
    attack_dir.mkdir(parents=True)
    attack_manifest_path = attack_dir / "attack_manifest.json"
    attack_metrics_path = attack_dir / "attack_family_metrics.csv"
    attack_manifest_path.write_text(
        json.dumps({"evaluation_boundary": {"target_fpr": 0.05}}, ensure_ascii=False),
        encoding="utf-8",
    )
    with attack_metrics_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["attack_family", "attack_name", "resource_profile"])
        writer.writeheader()
        writer.writerow({"attack_family": "standard_distortion", "attack_name": "jpeg_compression", "resource_profile": "full_main"})

    manifest = write_primary_baseline_formal_import_protocol_outputs(
        root=tmp_path,
        source_registry_path=registry_path,
        attack_manifest_path=attack_manifest_path,
        attack_family_metrics_path=attack_metrics_path,
    )
    output_dir = tmp_path / "outputs" / "primary_baseline_formal_import"
    schema = json.loads((output_dir / "primary_baseline_formal_import_schema.json").read_text(encoding="utf-8"))
    validation = json.loads((output_dir / "primary_baseline_formal_import_validation_report.json").read_text(encoding="utf-8"))
    template_rows = [
        json.loads(line)
        for line in (output_dir / "primary_baseline_formal_result_template.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    readiness_rows = list(
        csv.DictReader((output_dir / "primary_baseline_formal_import_readiness.csv").open(encoding="utf-8"))
    )
    coverage_rows = list(
        csv.DictReader((output_dir / "primary_baseline_formal_template_coverage.csv").open(encoding="utf-8"))
    )
    coverage_summary = json.loads(
        (output_dir / "primary_baseline_formal_template_coverage_summary.json").read_text(encoding="utf-8")
    )
    collection_rows = [
        json.loads(line)
        for line in (output_dir / "primary_baseline_formal_evidence_collection_plan.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    collection_summary = json.loads(
        (output_dir / "primary_baseline_formal_evidence_collection_summary.json").read_text(encoding="utf-8")
    )
    summary = json.loads((output_dir / "primary_baseline_formal_import_summary.json").read_text(encoding="utf-8"))

    assert manifest["artifact_id"] == "primary_baseline_formal_import_protocol_manifest"
    assert schema == build_primary_baseline_formal_import_schema(target_fpr=0.05)
    assert len(template_rows) == 4
    assert validation["input_record_count"] == 0
    assert len(readiness_rows) == 4
    assert len(coverage_rows) == 4
    assert coverage_summary["formal_template_record_count"] == 4
    assert coverage_summary["missing_formal_template_count"] == 4
    assert len(collection_rows) == 4
    assert collection_summary["formal_evidence_collection_task_count"] == 4
    assert collection_summary["missing_formal_evidence_collection_task_count"] == 4
    assert summary["primary_baseline_formal_ready"] is False
    assert all(str(path).startswith("outputs/") for path in manifest["output_paths"])
