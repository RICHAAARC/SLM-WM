"""主表 external baseline 正式结果导入协议的轻量功能测试。"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from paper_experiments.baselines import (
    build_primary_baseline_formal_evidence_collection_rows,
    build_primary_baseline_formal_evidence_collection_summary,
    build_primary_baseline_formal_import_schema,
    build_primary_baseline_formal_template_coverage_rows,
    build_primary_baseline_formal_template_coverage_summary,
    build_t2smark_formal_candidate_records,
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
        "comparable_operating_point": "fixed_fpr_0.01",
        "result_protocol_name": "primary_baseline_formal_import_protocol",
        "result_source_type": "governed_import",
        "baseline_result_source": evidence_path,
        "baseline_result_source_digest": "digest",
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
        "evidence_paths": [evidence_path],
        "method_faithful_adapter_ready": True,
        "paper_run_prompt_protocol_ready": True,
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

    report = validate_primary_baseline_formal_import_rows([row], evidence_root=tmp_path, target_fpr=0.01)

    assert report["overall_decision"] == "pass"
    assert report["accepted_formal_import_count"] == 1
    assert report["formal_import_validation_ready"] is True
    assert report["accepted_records"][0]["baseline_id"] == "tree_ring"


@pytest.mark.quick
def test_formal_import_validator_rejects_duplicate_formal_template_key(tmp_path: Path) -> None:
    """baseline 正式导入不得包含重复的 baseline × attack 模板键。"""

    evidence_path = tmp_path / "outputs" / "external_baseline_results" / "tree_ring_metrics.csv"
    evidence_path.parent.mkdir(parents=True)
    evidence_path.write_text("baseline_id,true_positive_rate\ntree_ring,0.7\n", encoding="utf-8")
    row = formal_tree_ring_row("outputs/external_baseline_results/tree_ring_metrics.csv")

    report = validate_primary_baseline_formal_import_rows(
        [row, dict(row)],
        evidence_root=tmp_path,
        target_fpr=0.01,
    )

    assert report["formal_import_validation_ready"] is False
    assert report["accepted_formal_import_count"] == 1
    assert {issue["reason"] for issue in report["issues"]} == {"duplicate_formal_template_key"}


@pytest.mark.quick
def test_formal_import_validator_requires_complete_scale_and_fixed_fpr_confidence(tmp_path: Path) -> None:
    """baseline 必须覆盖完整 test split 且 clean FPR 置信上界不超过目标。"""

    evidence_path = tmp_path / "outputs" / "external_baseline_results" / "tree_ring_metrics.csv"
    evidence_path.parent.mkdir(parents=True)
    evidence_path.write_text("baseline_id,true_positive_rate\ntree_ring,0.7\n", encoding="utf-8")
    incomplete_row = formal_tree_ring_row("outputs/external_baseline_results/tree_ring_metrics.csv")
    incomplete_row["positive_count"] = 100
    high_upper_bound_row = formal_tree_ring_row("outputs/external_baseline_results/tree_ring_metrics.csv")
    high_upper_bound_row["false_positive_rate"] = 1 / 340
    high_upper_bound_row["clean_false_positive_rate"] = 1 / 340

    incomplete_report = validate_primary_baseline_formal_import_rows(
        [incomplete_row], evidence_root=tmp_path, target_fpr=0.01
    )
    upper_bound_report = validate_primary_baseline_formal_import_rows(
        [high_upper_bound_row], evidence_root=tmp_path, target_fpr=0.01
    )

    assert "complete_test_positive_count_required" in {
        issue["reason"] for issue in incomplete_report["issues"]
    }
    assert "clean_fpr_confidence_upper_bound_exceeds_target" in {
        issue["reason"] for issue in upper_bound_report["issues"]
    }


@pytest.mark.quick
def test_formal_import_protocol_switches_prompt_schema_to_full_paper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """主表 baseline 正式导入 schema 和 validator 应跟随 full_paper 运行层级。"""

    config_dir = tmp_path / "configs"
    config_dir.mkdir(parents=True)
    (config_dir / "paper_main_full_paper_prompts.txt").write_text("a full paper prompt\n", encoding="utf-8")
    evidence_path = tmp_path / "outputs" / "external_baseline_results" / "tree_ring_metrics.csv"
    evidence_path.parent.mkdir(parents=True)
    evidence_path.write_text("baseline_id,true_positive_rate\ntree_ring,0.7\n", encoding="utf-8")
    monkeypatch.setenv("SLM_WM_PAPER_RUN_NAME", "full_paper")
    row = formal_tree_ring_row("outputs/external_baseline_results/tree_ring_metrics.csv")
    row["prompt_protocol_name"] = "paper_main_full_paper_prompt_protocol"
    row["comparable_operating_point"] = "fixed_fpr_0.001"
    row["positive_count"] = 3400
    row["negative_count"] = 3400
    row["attacked_negative_count"] = 3400
    row["attack_record_count"] = 6800
    row["supported_record_count"] = 6800

    schema = build_primary_baseline_formal_import_schema(target_fpr=0.001, root=tmp_path)
    report = validate_primary_baseline_formal_import_rows([row], evidence_root=tmp_path, target_fpr=0.001)

    assert schema["paper_claim_scale"] == "full_paper"
    assert schema["prompt_protocol_name"] == "paper_main_full_paper_prompt_protocol"
    assert schema["allowed_resource_profiles"] == ["full_main", "full_extra"]
    assert report["overall_decision"] == "pass"
    assert report["accepted_formal_import_count"] == 1


@pytest.mark.quick
def test_formal_import_validator_rejects_incomplete_adapter_boundary_and_missing_readiness(tmp_path: Path) -> None:
    """method-faithful adapter observation 不得被升级为主表正式结果。"""

    evidence_path = tmp_path / "outputs" / "external_baseline_results" / "tree_ring_metrics.csv"
    evidence_path.parent.mkdir(parents=True)
    evidence_path.write_text("baseline_id,true_positive_rate\ntree_ring,0.7\n", encoding="utf-8")
    row = formal_tree_ring_row("outputs/external_baseline_results/tree_ring_metrics.csv")
    row["adapter_boundary"] = "sd35_method_faithful_adapter_not_formal_external_baseline_evidence"
    row["fixed_fpr_baseline_calibration_ready"] = False

    report = validate_primary_baseline_formal_import_rows([row], evidence_root=tmp_path, target_fpr=0.01)
    reasons = {issue["reason"] for issue in report["issues"]}

    assert report["overall_decision"] == "fail"
    assert report["accepted_formal_import_count"] == 0
    assert "adapter_boundary_not_formal" in reasons
    assert "fixed_fpr_baseline_calibration_ready_required" in reasons


@pytest.mark.quick
def test_t2smark_candidate_records_remain_rejected_until_attack_and_threshold_ready(tmp_path: Path) -> None:
    """T2SMark formal 候选记录在攻击矩阵和 fixed-FPR 未闭合前应保持未通过正式导入。"""

    evidence_path = tmp_path / "outputs" / "t2smark_formal_reproduction" / "results.json"
    evidence_path.parent.mkdir(parents=True)
    evidence_path.write_text('{"0":{"robustness":{"norm1_no_w":0.1,"norm1_w":0.9}}}\n', encoding="utf-8")
    observations = [
        {"baseline_id": "t2smark", "attack_family": "clean", "attack_condition": "clean_none", "sample_role": "clean_negative", "detection_decision": False},
        {"baseline_id": "t2smark", "attack_family": "standard_distortion", "attack_condition": "jpeg_compression", "sample_role": "attacked_positive", "detection_decision": True, "quality_score": 1.0, "score_retention": 1.0},
        {"baseline_id": "t2smark", "attack_family": "standard_distortion", "attack_condition": "jpeg_compression", "sample_role": "attacked_negative", "detection_decision": False, "quality_score": 1.0, "score_retention": 1.0},
        {"baseline_id": "t2smark", "attack_family": "regeneration_attack", "attack_condition": "img2img_regeneration", "sample_role": "attacked_positive", "detection_decision": True, "quality_score": 0.9, "score_retention": 0.8},
        {"baseline_id": "t2smark", "attack_family": "regeneration_attack", "attack_condition": "img2img_regeneration", "sample_role": "attacked_negative", "detection_decision": False, "quality_score": 0.9, "score_retention": 0.8},
    ]

    records = build_t2smark_formal_candidate_records(
        observation_rows=observations,
        target_fpr=0.01,
        baseline_result_source="outputs/t2smark_formal_reproduction/results.json",
        baseline_result_source_digest="digest",
        evidence_paths=["outputs/t2smark_formal_reproduction/results.json"],
        prompt_protocol_digest="prompt_digest",
        paper_run_prompt_protocol_ready=True,
        fixed_fpr_baseline_calibration_ready=False,
        attack_matrix_baseline_detection_ready=False,
    )
    report = validate_primary_baseline_formal_import_rows(records, evidence_root=tmp_path, target_fpr=0.01)
    reasons = {issue["reason"] for issue in report["issues"]}

    assert len(records) == 2
    assert {record["resource_profile"] for record in records} == {"full_main", "full_extra"}
    assert all(record["adapter_boundary"] == "sd35_medium_native_official_reproduction" for record in records)
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
        *[
            {
                "baseline_id": "tree_ring",
                "attack_family": "clean",
                "attack_condition": "clean_none",
                "sample_role": "clean_negative",
                "detection_decision": False,
                "quality_score": 1.0,
                "score_retention": 1.0,
            }
            for _ in range(340)
        ],
        *[
            {
                "baseline_id": "tree_ring",
                "attack_family": "standard_distortion",
                "attack_condition": "jpeg_compression",
                "sample_role": "attacked_positive",
                "detection_decision": True,
                "quality_score": 1.0,
                "score_retention": 1.0,
            }
            for _ in range(340)
        ],
        *[
            {
                "baseline_id": "tree_ring",
                "attack_family": "standard_distortion",
                "attack_condition": "jpeg_compression",
                "sample_role": "attacked_negative",
                "detection_decision": False,
                "quality_score": 1.0,
                "score_retention": 1.0,
            }
            for _ in range(340)
        ],
    ]

    records = build_tree_ring_method_faithful_candidate_records(
        observation_rows=observations,
        target_fpr=0.01,
        baseline_result_source="outputs/tree_ring_method_faithful/baseline_observations.json",
        baseline_result_source_digest="digest",
        evidence_paths=["outputs/tree_ring_method_faithful/baseline_observations.json"],
        prompt_protocol_digest="prompt_digest",
        paper_run_prompt_protocol_ready=True,
        fixed_fpr_baseline_calibration_ready=True,
        attack_matrix_baseline_detection_ready=True,
    )
    report = validate_primary_baseline_formal_import_rows(records, evidence_root=tmp_path, target_fpr=0.01)

    assert len(records) == 1
    assert records[0]["baseline_id"] == "tree_ring"
    assert records[0]["adapter_boundary"] == "method_faithful_sd35_adapter_reproduction"
    assert records[0]["result_source_type"] == "governed_import"
    assert report["accepted_formal_import_count"] == 1


@pytest.mark.quick
def test_formal_template_coverage_requires_matching_formal_attack_records(tmp_path: Path) -> None:
    """正式模板覆盖应检查候选记录是否覆盖共同协议要求的攻击模板。"""

    evidence_path = tmp_path / "outputs" / "external_baseline_results" / "tree_ring_metrics.csv"
    evidence_path.parent.mkdir(parents=True)
    evidence_path.write_text("baseline_id,true_positive_rate\ntree_ring,0.7\n", encoding="utf-8")
    accepted_row = formal_tree_ring_row("outputs/external_baseline_results/tree_ring_metrics.csv")
    missing_template = {
        "baseline_id": "tree_ring",
        "attack_family": "standard_distortion",
        "attack_name": "gaussian_noise",
        "resource_profile": "full_main",
        "comparable_operating_point": "fixed_fpr_0.01",
    }
    template_rows = [
        {
            "baseline_id": "tree_ring",
            "attack_family": "standard_distortion",
            "attack_name": "jpeg_compression",
            "resource_profile": "full_main",
            "comparable_operating_point": "fixed_fpr_0.01",
        },
        missing_template,
    ]
    report = validate_primary_baseline_formal_import_rows([accepted_row], evidence_root=tmp_path, target_fpr=0.01)

    coverage_rows = build_primary_baseline_formal_template_coverage_rows(template_rows, [accepted_row], report)
    coverage_summary = build_primary_baseline_formal_template_coverage_summary(coverage_rows)
    tree_row = next(row for row in coverage_rows if row["baseline_id"] == "tree_ring")

    assert tree_row["expected_formal_template_count"] == 2
    assert tree_row["accepted_template_match_count"] == 1
    assert tree_row["missing_formal_template_count"] == 1
    assert tree_row["formal_template_coverage_ready"] is False
    assert coverage_summary["primary_baseline_formal_template_coverage_ready"] is False


@pytest.mark.quick
def test_formal_template_coverage_rejects_unexpected_and_duplicate_accepted_records() -> None:
    """正式模板覆盖必须与当前攻击模板严格相等, 不得接受额外或重复记录。"""

    template = {
        "baseline_id": "tree_ring",
        "attack_family": "standard_distortion",
        "attack_name": "jpeg_compression",
        "resource_profile": "full_main",
        "comparable_operating_point": "fixed_fpr_0.01",
    }
    unexpected = {
        **template,
        "attack_family": "unregistered_attack",
        "attack_name": "unregistered_attack",
    }
    accepted_records = [template, dict(template), unexpected]
    report = {"accepted_records": accepted_records}

    coverage_rows = build_primary_baseline_formal_template_coverage_rows(
        [template],
        accepted_records,
        report,
    )
    coverage_summary = build_primary_baseline_formal_template_coverage_summary(coverage_rows)
    tree_row = next(row for row in coverage_rows if row["baseline_id"] == "tree_ring")

    assert tree_row["missing_formal_template_count"] == 0
    assert tree_row["unexpected_accepted_record_count"] == 1
    assert tree_row["duplicate_accepted_template_count"] == 1
    assert tree_row["formal_template_coverage_ready"] is False
    assert coverage_summary["unexpected_accepted_record_count"] == 1
    assert coverage_summary["duplicate_accepted_template_count"] == 1


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
            "comparable_operating_point": "fixed_fpr_0.01",
        }
    ]
    report = validate_primary_baseline_formal_import_rows([candidate_row], evidence_root=tmp_path, target_fpr=0.01)

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
        "comparable_operating_point": "fixed_fpr_0.01",
        "required_metric_fields": ["true_positive_rate"],
        "required_source_fields": ["baseline_result_source"],
    }
    template_rows = [
        {
            "baseline_id": "tree_ring",
            "attack_family": "standard_distortion",
            "attack_name": "jpeg_compression",
            "resource_profile": "full_main",
            "comparable_operating_point": "fixed_fpr_0.01",
            "required_metric_fields": ["true_positive_rate"],
            "required_source_fields": ["baseline_result_source"],
        },
        missing_template,
    ]
    report = validate_primary_baseline_formal_import_rows([accepted_row], evidence_root=tmp_path, target_fpr=0.01)

    collection_rows = build_primary_baseline_formal_evidence_collection_rows(template_rows, [accepted_row], report)
    collection_summary = build_primary_baseline_formal_evidence_collection_summary(collection_rows)
    missing_row = next(row for row in collection_rows if row["attack_name"] == "gaussian_noise")

    assert len(collection_rows) == 2
    assert missing_row["formal_evidence_collection_ready"] is False
    assert "generate_pilot_paper_baseline_result_record" in missing_row["required_collection_actions"]
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
        json.dumps({"evaluation_boundary": {"target_fpr": 0.01}}, ensure_ascii=False),
        encoding="utf-8",
    )
    with attack_metrics_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["attack_family", "attack_name", "resource_profile"])
        writer.writeheader()
        writer.writerow({"attack_family": "standard_distortion", "attack_name": "jpeg_compression", "resource_profile": "full_main"})
        writer.writerow(
            {
                "attack_family": "regeneration_attack",
                "attack_name": "img2img_regeneration",
                "resource_profile": "full_extra",
            }
        )
    candidate_records_path = tmp_path / "outputs" / "external_baseline_results" / "baseline_result_records.jsonl"
    candidate_records_path.parent.mkdir(parents=True)
    candidate_records_path.write_text("", encoding="utf-8")

    manifest = write_primary_baseline_formal_import_protocol_outputs(
        root=tmp_path,
        source_registry_path=registry_path,
        attack_manifest_path=attack_manifest_path,
        attack_family_metrics_path=attack_metrics_path,
        candidate_records_path=candidate_records_path,
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
    assert schema == build_primary_baseline_formal_import_schema(target_fpr=0.01)
    assert len(template_rows) == 8
    assert {row["resource_profile"] for row in template_rows} == {"full_main", "full_extra"}
    assert validation["input_record_count"] == 0
    assert len(readiness_rows) == 4
    assert len(coverage_rows) == 4
    assert coverage_summary["formal_template_record_count"] == 8
    assert coverage_summary["missing_formal_template_count"] == 8
    assert len(collection_rows) == 8
    assert collection_summary["formal_evidence_collection_task_count"] == 8
    assert collection_summary["missing_formal_evidence_collection_task_count"] == 8
    assert summary["primary_baseline_formal_ready"] is False
    assert all(str(path).startswith("outputs/") for path in manifest["output_paths"])

