"""论文结果闭合语义门禁的轻量功能测试。"""

from __future__ import annotations

import csv
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

import pytest

from experiments.ablations.runtime_rerun import (
    FORMAL_RUNTIME_RERUN_ABLATION_IDS,
    FORMAL_RUNTIME_RERUN_ABLATION_SPEC_DIGEST,
)
from experiments.protocol.prompts import build_prompt_records
from main.core.digest import build_stable_digest
from paper_experiments.analysis.result_closure_gate import (
    ResultClosureGateInput,
    build_result_closure_gate_checks,
    build_result_closure_gate_report,
)
from scripts.write_result_closure_gate_outputs import main, write_result_closure_gate_outputs


SCALE = "probe_paper"
TARGET_FPR = 0.1
PROMPT_COUNT = 70
TEST_COUNT = 34
MAIN_THRESHOLD_DIGEST = "1" * 64
PROMPT_SPLIT_DIGEST = "6" * 64
ATTACK_MATRIX_DIGEST = "7" * 64
FIXED_FPR_PROTOCOL_DIGEST = "8" * 64
PROMPT_ID_DIGEST = build_stable_digest(
    sorted(
        record.prompt_id
        for record in build_prompt_records(
            SCALE,
            tuple(f"a governed prompt {index}" for index in range(PROMPT_COUNT)),
        )
    )
)
FEATURE_RECORDS_TEXT = "{}\n"
FEATURE_RECORDS_SHA256 = hashlib.sha256(FEATURE_RECORDS_TEXT.encode("utf-8")).hexdigest()


def manifest(
    artifact_id: str,
    output_paths: tuple[str, ...],
    metadata: dict[str, object],
    *,
    input_paths: tuple[str, ...] = (),
    config: dict[str, object] | None = None,
) -> dict[str, object]:
    """构造满足通用 provenance schema 的测试 manifest。"""

    return {
        "artifact_id": artifact_id,
        "artifact_type": "local_manifest",
        "input_paths": list(input_paths),
        "output_paths": list(output_paths),
        "config_digest": "a" * 64,
        "code_version": "test-code-version",
        "rebuild_command": "python scripts/test_builder.py",
        "config": config or {},
        "metadata": metadata,
    }


def formal_result_record() -> dict[str, object]:
    """构造正文摘要可独立重算的最小正式结果记录。"""

    payload: dict[str, object] = {
        "paper_claim_scale": SCALE,
        "target_fpr": TARGET_FPR,
        "method_id": "slm_wm_current",
        "attack_family": "compression",
        "attack_name": "jpeg_70",
        "resource_profile": "full_main",
        "positive_count": TEST_COUNT,
        "negative_count": TEST_COUNT,
        "attacked_negative_count": TEST_COUNT,
        "prompt_split_digest": PROMPT_SPLIT_DIGEST,
        "attack_matrix_digest": ATTACK_MATRIX_DIGEST,
        "fixed_fpr_protocol_digest": FIXED_FPR_PROTOCOL_DIGEST,
        "strict_formal_result_ready": True,
        "supports_paper_claim": True,
    }
    digest = build_stable_digest(payload)
    payload["pilot_paper_result_record_digest"] = digest
    payload["pilot_paper_result_record_id"] = f"pilot_paper_result_record_{digest[:16]}"
    return payload


def ready_bundle() -> ResultClosureGateInput:
    """构造证据完整且允许至少一个逐攻击比较未显著胜出的输入包。"""

    attack_record_count = 2 * 2 * TEST_COUNT
    attack_report = {
        "paper_run_name": SCALE,
        "evaluation_boundary": {"target_fpr": TARGET_FPR, "threshold_digest": MAIN_THRESHOLD_DIGEST},
        "attack_record_count": attack_record_count,
        "performed_attack_record_count": attack_record_count,
        "formal_real_attack_record_count": attack_record_count,
        "formal_image_attack_record_count": attack_record_count,
        "real_attacked_image_count": attack_record_count,
        "expected_attack_ids": ["jpeg_70", "regeneration_025"],
        "actual_attack_ids": ["jpeg_70", "regeneration_025"],
        "missing_attack_ids": [],
        "unexpected_attack_ids": [],
        "expected_attack_split_role_count": TEST_COUNT,
        "attack_split_role_counts": {
            "jpeg_70|positive_source": TEST_COUNT,
            "jpeg_70|clean_negative": TEST_COUNT,
            "regeneration_025|positive_source": TEST_COUNT,
            "regeneration_025|clean_negative": TEST_COUNT,
        },
        "required_real_gpu_attack_count": 1,
        "measured_real_gpu_attack_count": 1,
        "gpu_attack_real_measurement_missing_count": 0,
        "real_attacked_image_closed_loop_ready": True,
        "formal_attack_detection_ready": True,
        "attack_metrics_ready": True,
        "attack_record_coverage_ready": True,
        "real_gpu_attack_validation_ready": True,
        "full_method_claim_ready": True,
        "supports_paper_claim": True,
    }
    attack_metadata = {
        "protocol_decision": "pass",
        "full_method_claim_ready": True,
        "supports_paper_claim": True,
    }
    threshold_rows = tuple(
        {
            "method_id": method_id,
            "target_fpr": TARGET_FPR,
            "test_clean_negative_count": TEST_COUNT,
            "threshold_digest": MAIN_THRESHOLD_DIGEST if method_id == "slm_wm" else str(index + 2) * 64,
            "protocol_target_ready": True,
            "protocol_value_ready": True,
            "detection_decision_ready": True,
            "split_count_ready": True,
            "fixed_fpr_threshold_ready": True,
            "supports_paper_claim": False,
        }
        for index, method_id in enumerate(
            ("slm_wm", "tree_ring", "gaussian_shading", "shallow_diffuse", "t2smark")
        )
    )
    threshold_report = {
        "paper_claim_scale": SCALE,
        "target_fpr": TARGET_FPR,
        "method_identity_ready": True,
        "all_method_thresholds_ready": True,
        "fixed_fpr_threshold_audit_ready": True,
        "supports_paper_claim": True,
    }
    primary_evidence_summary = {
        "paper_claim_scale": SCALE,
        "target_fpr": TARGET_FPR,
        "primary_baseline_count": 4,
        "adapter_run_ready_count": 4,
        "adapter_run_ready_ids": ["tree_ring", "gaussian_shading", "shallow_diffuse", "t2smark"],
        "formal_result_ready_count": 4,
        "formal_result_ready_ids": ["tree_ring", "gaussian_shading", "shallow_diffuse", "t2smark"],
        "input_baseline_ids": ["tree_ring", "gaussian_shading", "shallow_diffuse", "t2smark"],
        "primary_baseline_formal_ready": True,
        "blocking_reasons": [],
        "input_observation_count": PROMPT_COUNT * 4,
        "input_command_result_count": 4,
        "t2smark_formal_evidence_digest": "9" * 64,
        "supports_paper_claim": False,
    }
    baseline_report = {
        "paper_claim_scale": SCALE,
        "target_fpr": TARGET_FPR,
        "accepted_formal_import_count": 5,
        "rejected_formal_import_count": 0,
        "formal_import_issue_count": 0,
        "missing_candidate_template_count": 0,
        "missing_formal_template_count": 0,
        "unexpected_candidate_record_count": 0,
        "unexpected_accepted_record_count": 0,
        "duplicate_candidate_template_count": 0,
        "duplicate_accepted_template_count": 0,
        "missing_formal_evidence_collection_task_count": 0,
        "missing_formal_evidence_path_count": 0,
        "baseline_results_ready": False,
        "comparison_protocol_ready": True,
        "comparison_table_supports_paper_claim": True,
        "primary_baseline_formal_ready": True,
        "primary_baseline_results_ready": True,
        "primary_baseline_formal_template_coverage_ready": True,
        "primary_baseline_formal_evidence_collection_ready": True,
        "formal_import_validation_ready": True,
        "formal_evidence_path_resolution_ready": True,
        "baseline_source_registry_ready": True,
        "supports_paper_claim": True,
    }
    record_summary = {
        "paper_claim_scale": SCALE,
        "pilot_paper_result_record_count": 1,
        "pilot_paper_template_record_count": 1,
        "pilot_paper_template_covered_count": 1,
        "pilot_paper_template_missing_count": 0,
        "accepted_pilot_paper_import_count": 1,
        "accepted_pilot_paper_claim_record_count": 1,
        "pilot_paper_template_coverage_ready": True,
        "pilot_paper_result_import_ready": True,
        "pilot_paper_claim_record_ready": True,
        "supports_paper_claim": True,
    }
    common_summary = {
        "paper_claim_scale": SCALE,
        "paper_target_fpr": TARGET_FPR,
        "expected_target_fpr": TARGET_FPR,
        "paper_prompt_count": PROMPT_COUNT,
        "pilot_paper_import_template_count": 1,
        "accepted_pilot_paper_import_count": 1,
        "accepted_pilot_paper_claim_record_count": 1,
        "pilot_paper_negative_count_minimum_required": TEST_COUNT,
        "minimum_result_positive_count": TEST_COUNT,
        "minimum_result_negative_count": TEST_COUNT,
        "minimum_result_attacked_negative_count": TEST_COUNT,
        "paper_run_result_missing_template_count": 0,
        "paper_run_result_unexpected_template_count": 0,
        "paper_run_result_duplicate_template_count": 0,
        "paper_run_allows_paper_claim": True,
        "strict_formal_evidence_required": True,
        "pilot_paper_common_protocol_ready": True,
        "paper_run_workflow_validation_ready": True,
        "pilot_paper_prompt_split_ready": True,
        "paper_prompt_split_ready": True,
        "pilot_paper_result_import_ready": True,
        "pilot_paper_claim_record_ready": True,
        "paper_run_result_import_coverage_ready": True,
        "paper_run_template_registry_unique": True,
        "pilot_paper_evidence_coverage_ready": True,
        "pilot_paper_effectiveness_gate_ready": True,
        "slm_wm_fixed_fpr_boundary_ready": True,
        "paper_run_claim_ready": True,
        "paper_run_supports_superiority_claim": True,
        "paper_claim_ready": True,
    }
    common_schema = {
        "paper_claim_scale": SCALE,
        "target_fpr": TARGET_FPR,
        "prompt_split_digest": PROMPT_SPLIT_DIGEST,
        "attack_matrix_digest": ATTACK_MATRIX_DIGEST,
        "fixed_fpr_protocol_digest": FIXED_FPR_PROTOCOL_DIGEST,
    }
    analysis_summary = {
        "paper_claim_scale": SCALE,
        "target_fpr": TARGET_FPR,
        "result_record_count": 1,
        "expected_result_record_count": 1,
        "actual_result_record_count": 1,
        "unique_result_record_key_count": 1,
        "confidence_interval_row_count": 1,
        "expected_superiority_row_count": 1,
        "per_attack_superiority_row_count": 1,
        "superiority_claim_ready_count": 0,
        "duplicate_result_record_count": 0,
        "missing_result_record_count": 0,
        "unexpected_result_record_count": 0,
        "failure_case_figure_ready": True,
        "result_template_coverage_ready": True,
        "per_attack_ci_coverage_ready": True,
        "per_attack_superiority_evaluation_ready": True,
        "universal_per_attack_superiority_claim_ready": False,
        "supports_paper_claim": True,
    }
    ablation_summary = {
        "paper_claim_scale": SCALE,
        "target_fpr": TARGET_FPR,
        "split_counts": {"test": TEST_COUNT},
        "record_count": PROMPT_COUNT * len(FORMAL_RUNTIME_RERUN_ABLATION_IDS),
        "prompt_count": PROMPT_COUNT,
        "ablation_count": len(FORMAL_RUNTIME_RERUN_ABLATION_IDS),
        "per_ablation_calibration_count": len(FORMAL_RUNTIME_RERUN_ABLATION_IDS),
        "generation_rerun_count": PROMPT_COUNT * len(FORMAL_RUNTIME_RERUN_ABLATION_IDS),
        "attack_and_detection_rerun_count": PROMPT_COUNT
        * len(FORMAL_RUNTIME_RERUN_ABLATION_IDS),
        "expected_ablation_ids": list(FORMAL_RUNTIME_RERUN_ABLATION_IDS),
        "actual_ablation_ids": list(FORMAL_RUNTIME_RERUN_ABLATION_IDS),
        "ablation_spec_digest": FORMAL_RUNTIME_RERUN_ABLATION_SPEC_DIGEST,
        "ablation_exact_set_ready": True,
        "ablation_claim_gate_ready": True,
        "protocol_decision": "pass",
        "supports_paper_claim": True,
    }
    quality_summary = {
        "paper_claim_scale": SCALE,
        "target_fpr": TARGET_FPR,
        "expected_prompt_count": PROMPT_COUNT,
        "registry_prompt_count": PROMPT_COUNT,
        "duplicate_registry_prompt_id_count": 0,
        "missing_registry_prompt_id_count": 0,
        "unexpected_registry_prompt_id_count": 0,
        "canonical_prompt_id_digest": PROMPT_ID_DIGEST,
        "registry_prompt_id_digest": PROMPT_ID_DIGEST,
        "prompt_registry_exact_set_ready": True,
        "sample_pair_count": PROMPT_COUNT,
        "source_image_count": PROMPT_COUNT,
        "comparison_image_count": PROMPT_COUNT,
        "formal_fid_kid_ready": True,
        "formal_fid_kid_metric_names_ready": True,
        "formal_feature_backend_ready": True,
        "formal_sample_scale_ready": True,
        "canonical_formal_feature_extractor_ready": True,
        "formal_feature_record_count": PROMPT_COUNT * 2,
        "expected_feature_pair_count": PROMPT_COUNT,
        "accepted_feature_pair_count": PROMPT_COUNT,
        "missing_feature_pair_count": 0,
        "feature_issue_count": 0,
        "formal_feature_records_sha256": FEATURE_RECORDS_SHA256,
        "formal_fid_kid_claim_gate_ready": True,
    }
    quality_feature_report = {
        "paper_run_name": SCALE,
        "target_fpr": TARGET_FPR,
        "canonical_prompt_id_digest": PROMPT_ID_DIGEST,
        "registry_prompt_id_digest": PROMPT_ID_DIGEST,
        "prompt_registry_exact_set_ready": True,
        "formal_feature_record_count": PROMPT_COUNT * 2,
        "expected_feature_pair_count": PROMPT_COUNT,
        "accepted_feature_pair_count": PROMPT_COUNT,
        "missing_feature_pair_count": 0,
        "feature_issue_count": 0,
        "formal_feature_records_sha256": FEATURE_RECORDS_SHA256,
    }
    quality_metrics = tuple(
        {
            "quality_metric_name": metric_name,
            "quality_metric_value": "0.125",
            "metric_status": "measured",
            "source_image_count": str(PROMPT_COUNT),
            "comparison_image_count": str(PROMPT_COUNT),
            "sample_pair_count": str(PROMPT_COUNT),
        }
        for metric_name in ("fid", "kid")
    )
    builder_report = {
        "artifact_builder_ready": True,
        "paper_artifact_claim_ready": True,
        "paper_artifact_audit_ready": True,
        "blocked_artifact_count": 0,
    }
    blocker_report = {
        "submission_ready": True,
        "artifact_builder_ready": True,
        "paper_artifact_claim_ready": True,
        "paper_artifact_audit_ready": True,
        "full_method_claim_ready": True,
        "supports_paper_claim": True,
        "blocking_claim_count": 0,
        "critical_gap_count": 0,
        "gap_count": 0,
    }
    submission_report = {
        "readiness_decision": "ready",
        "submission_ready": True,
        "package_freeze_allowed": True,
        "artifact_builder_ready": True,
        "paper_artifact_claim_ready": True,
        "release_dry_run_ready": True,
        "required_input_count": 0,
        "critical_required_input_count": 0,
        "blocking_claim_count": 0,
    }
    entry_report = {
        "entry_review_decision": "ready_for_user_audit",
        "entry_review_ready": True,
        "user_audit_required": True,
        "evidence_closure_allowed": True,
        "blocked_review_item_count": 0,
        "required_input_count": 0,
        "critical_required_input_count": 0,
        "blocking_claim_count": 0,
        "primary_baseline_results_ready": True,
        "formal_import_validation_ready": True,
        "accepted_formal_import_count": 5,
        "formal_evidence_path_resolution_ready": True,
        "formal_fid_kid_ready": True,
        "formal_sample_scale_ready": True,
        "formal_feature_backend_ready": True,
    }
    return ResultClosureGateInput(
        expected_paper_claim_scale=SCALE,
        expected_target_fpr=TARGET_FPR,
        expected_prompt_count=PROMPT_COUNT,
        expected_test_count=TEST_COUNT,
        expected_prompt_id_digest=PROMPT_ID_DIGEST,
        attack_report=attack_report,
        attack_manifest=manifest(
            f"{SCALE}_attack_matrix_manifest",
            (
                f"outputs/attack_matrix/{SCALE}/attack_manifest.json",
                f"outputs/attack_matrix/{SCALE}/manifest.local.json",
            ),
            attack_metadata,
        ),
        threshold_audit_report=threshold_report,
        threshold_audit_rows=threshold_rows,
        threshold_audit_manifest=manifest(
            "fixed_fpr_threshold_audit_manifest",
            (
                f"outputs/fixed_fpr_threshold_audit/{SCALE}/threshold_audit_rows.csv",
                f"outputs/fixed_fpr_threshold_audit/{SCALE}/threshold_audit_report.json",
                f"outputs/fixed_fpr_threshold_audit/{SCALE}/manifest.local.json",
            ),
            threshold_report,
        ),
        primary_baseline_evidence_summary=primary_evidence_summary,
        primary_baseline_evidence_manifest=manifest(
            "primary_baseline_evidence_manifest",
            (
                f"outputs/primary_baseline_evidence/{SCALE}/primary_baseline_evidence_records.jsonl",
                f"outputs/primary_baseline_evidence/{SCALE}/primary_baseline_evidence_summary.json",
                f"outputs/primary_baseline_evidence/{SCALE}/manifest.local.json",
            ),
            primary_evidence_summary,
        ),
        baseline_report=baseline_report,
        baseline_manifest=manifest(
            "external_baseline_comparison_manifest",
            (
                f"outputs/external_baseline_comparison/{SCALE}/baseline_runtime_report.json",
                f"outputs/external_baseline_comparison/{SCALE}/manifest.local.json",
            ),
            baseline_report,
        ),
        result_records=(formal_result_record(),),
        result_record_summary=record_summary,
        result_record_manifest=manifest(
            "pilot_paper_fixed_fpr_result_records_manifest",
            (
                f"outputs/pilot_paper_fixed_fpr_results/{SCALE}/pilot_paper_result_records.jsonl",
                f"outputs/pilot_paper_fixed_fpr_results/{SCALE}/pilot_paper_result_record_summary.json",
                f"outputs/pilot_paper_fixed_fpr_results/{SCALE}/manifest.local.json",
            ),
            record_summary,
        ),
        common_protocol_summary=common_summary,
        common_protocol_schema=common_schema,
        common_protocol_manifest=manifest(
            "pilot_paper_fixed_fpr_common_protocol_manifest",
            (
                f"outputs/pilot_paper_fixed_fpr_common_protocol/{SCALE}/pilot_paper_result_import_schema.json",
                f"outputs/pilot_paper_fixed_fpr_common_protocol/{SCALE}/pilot_paper_common_protocol_summary.json",
                f"outputs/pilot_paper_fixed_fpr_common_protocol/{SCALE}/manifest.local.json",
            ),
            common_summary,
        ),
        result_analysis_summary=analysis_summary,
        result_analysis_manifest=manifest(
            "pilot_paper_result_analysis_manifest",
            (
                f"outputs/pilot_paper_result_analysis/{SCALE}/confidence_interval_table.csv",
                f"outputs/pilot_paper_result_analysis/{SCALE}/per_attack_superiority_table.csv",
                f"outputs/pilot_paper_result_analysis/{SCALE}/failure_case_records.jsonl",
                f"outputs/pilot_paper_result_analysis/{SCALE}/failure_case_figure.svg",
                f"outputs/pilot_paper_result_analysis/{SCALE}/result_analysis_summary.json",
                f"outputs/pilot_paper_result_analysis/{SCALE}/manifest.local.json",
            ),
            analysis_summary,
        ),
        ablation_summary=ablation_summary,
        ablation_manifest=manifest(
            "formal_mechanism_ablation_manifest",
            (
                f"outputs/formal_mechanism_ablation/{SCALE}/per_ablation_frozen_protocols.json",
                f"outputs/formal_mechanism_ablation/{SCALE}/mechanism_ablation_metrics.csv",
                f"outputs/formal_mechanism_ablation/{SCALE}/ablation_claim_summary.json",
                f"outputs/formal_mechanism_ablation/{SCALE}/manifest.local.json",
            ),
            {
                "protocol_decision": "pass",
                "expected_ablation_ids": list(FORMAL_RUNTIME_RERUN_ABLATION_IDS),
                "actual_ablation_ids": list(FORMAL_RUNTIME_RERUN_ABLATION_IDS),
                "ablation_spec_digest": FORMAL_RUNTIME_RERUN_ABLATION_SPEC_DIGEST,
                "ablation_exact_set_ready": True,
                "generation_rerun_required": True,
                "per_ablation_calibration_required": True,
                "supports_paper_claim": True,
            },
            config={
                "expected_ablation_ids": list(FORMAL_RUNTIME_RERUN_ABLATION_IDS),
                "actual_ablation_ids": list(FORMAL_RUNTIME_RERUN_ABLATION_IDS),
                "ablation_spec_digest": FORMAL_RUNTIME_RERUN_ABLATION_SPEC_DIGEST,
                "ablation_exact_set_ready": True,
            },
        ),
        dataset_quality_summary=quality_summary,
        dataset_quality_feature_report=quality_feature_report,
        dataset_quality_metrics=quality_metrics,
        dataset_quality_feature_records_sha256=FEATURE_RECORDS_SHA256,
        dataset_quality_manifest=manifest(
            "dataset_level_quality_manifest",
            (
                f"outputs/dataset_level_quality/{SCALE}/dataset_quality_formal_feature_records.jsonl",
                f"outputs/dataset_level_quality/{SCALE}/dataset_quality_formal_feature_import_report.json",
                f"outputs/dataset_level_quality/{SCALE}/dataset_quality_metrics.csv",
                f"outputs/dataset_level_quality/{SCALE}/dataset_quality_summary.json",
                f"outputs/dataset_level_quality/{SCALE}/manifest.local.json",
            ),
            quality_summary,
            config={
                "canonical_prompt_id_digest": PROMPT_ID_DIGEST,
                "registry_prompt_id_digest": PROMPT_ID_DIGEST,
                "prompt_registry_exact_set_ready": True,
                "accepted_feature_pair_count": PROMPT_COUNT,
                "missing_feature_pair_count": 0,
                "feature_issue_count": 0,
                "formal_feature_record_count": PROMPT_COUNT * 2,
                "formal_feature_records_sha256": FEATURE_RECORDS_SHA256,
            },
        ),
        evidence_builder_report=builder_report,
        evidence_blocker_report=blocker_report,
        evidence_audit_manifest=manifest(
            "paper_artifact_evidence_audit_manifest",
            (
                f"outputs/paper_artifact_evidence_audit/{SCALE}/artifact_builder_readiness_report.json",
                f"outputs/paper_artifact_evidence_audit/{SCALE}/submission_blocker_report.json",
                f"outputs/paper_artifact_evidence_audit/{SCALE}/manifest.local.json",
            ),
            blocker_report,
            input_paths=(
                f"outputs/image_only_dataset_runtime/{SCALE}/dataset_runtime_summary.json",
                f"outputs/formal_mechanism_ablation/{SCALE}/ablation_claim_summary.json",
                f"outputs/dataset_level_quality/{SCALE}/dataset_quality_summary.json",
            ),
        ),
        submission_readiness_report=submission_report,
        submission_readiness_manifest=manifest(
            "submission_readiness_manifest",
            (
                f"outputs/submission_readiness/{SCALE}/readiness_blocker_report.json",
                f"outputs/submission_readiness/{SCALE}/submission_readiness_manifest.local.json",
            ),
            submission_report,
        ),
        entry_review_report=entry_report,
        entry_review_manifest=manifest(
            "evidence_closure_entry_review_manifest",
            (
                f"outputs/evidence_closure_entry_review/{SCALE}/entry_review_report.json",
                f"outputs/evidence_closure_entry_review/{SCALE}/manifest.local.json",
            ),
            entry_report,
            input_paths=(f"outputs/dataset_level_quality/{SCALE}/dataset_quality_summary.json",),
        ),
    )


def write_json(path: Path, payload: object) -> None:
    """写出测试 JSON 文件。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def write_jsonl(path: Path, rows: tuple[dict[str, object], ...]) -> None:
    """写出测试 JSONL 文件。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def write_csv(path: Path, rows: tuple[dict[str, object], ...]) -> None:
    """写出测试 CSV 文件。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_bundle_inputs(root: Path, bundle: ResultClosureGateInput) -> None:
    """按正式默认路径写出脚本测试需要的输入证据。"""

    json_paths = {
        f"outputs/attack_matrix/{SCALE}/attack_manifest.json": bundle.attack_report,
        f"outputs/attack_matrix/{SCALE}/manifest.local.json": bundle.attack_manifest,
        f"outputs/fixed_fpr_threshold_audit/{SCALE}/threshold_audit_report.json": bundle.threshold_audit_report,
        f"outputs/fixed_fpr_threshold_audit/{SCALE}/manifest.local.json": bundle.threshold_audit_manifest,
        f"outputs/primary_baseline_evidence/{SCALE}/primary_baseline_evidence_summary.json": bundle.primary_baseline_evidence_summary,
        f"outputs/primary_baseline_evidence/{SCALE}/manifest.local.json": bundle.primary_baseline_evidence_manifest,
        f"outputs/external_baseline_comparison/{SCALE}/baseline_runtime_report.json": bundle.baseline_report,
        f"outputs/external_baseline_comparison/{SCALE}/manifest.local.json": bundle.baseline_manifest,
        f"outputs/pilot_paper_fixed_fpr_results/{SCALE}/pilot_paper_result_record_summary.json": bundle.result_record_summary,
        f"outputs/pilot_paper_fixed_fpr_results/{SCALE}/manifest.local.json": bundle.result_record_manifest,
        f"outputs/pilot_paper_fixed_fpr_common_protocol/{SCALE}/pilot_paper_common_protocol_summary.json": bundle.common_protocol_summary,
        f"outputs/pilot_paper_fixed_fpr_common_protocol/{SCALE}/pilot_paper_result_import_schema.json": bundle.common_protocol_schema,
        f"outputs/pilot_paper_fixed_fpr_common_protocol/{SCALE}/manifest.local.json": bundle.common_protocol_manifest,
        f"outputs/pilot_paper_result_analysis/{SCALE}/result_analysis_summary.json": bundle.result_analysis_summary,
        f"outputs/pilot_paper_result_analysis/{SCALE}/manifest.local.json": bundle.result_analysis_manifest,
        f"outputs/formal_mechanism_ablation/{SCALE}/ablation_claim_summary.json": bundle.ablation_summary,
        f"outputs/formal_mechanism_ablation/{SCALE}/manifest.local.json": bundle.ablation_manifest,
        f"outputs/dataset_level_quality/{SCALE}/dataset_quality_summary.json": bundle.dataset_quality_summary,
        f"outputs/dataset_level_quality/{SCALE}/dataset_quality_formal_feature_import_report.json": bundle.dataset_quality_feature_report,
        f"outputs/dataset_level_quality/{SCALE}/manifest.local.json": bundle.dataset_quality_manifest,
        f"outputs/paper_artifact_evidence_audit/{SCALE}/artifact_builder_readiness_report.json": bundle.evidence_builder_report,
        f"outputs/paper_artifact_evidence_audit/{SCALE}/submission_blocker_report.json": bundle.evidence_blocker_report,
        f"outputs/paper_artifact_evidence_audit/{SCALE}/manifest.local.json": bundle.evidence_audit_manifest,
        f"outputs/submission_readiness/{SCALE}/readiness_blocker_report.json": bundle.submission_readiness_report,
        f"outputs/submission_readiness/{SCALE}/submission_readiness_manifest.local.json": bundle.submission_readiness_manifest,
        f"outputs/evidence_closure_entry_review/{SCALE}/entry_review_report.json": bundle.entry_review_report,
        f"outputs/evidence_closure_entry_review/{SCALE}/manifest.local.json": bundle.entry_review_manifest,
    }
    for relative_path, payload in json_paths.items():
        write_json(root / relative_path, payload)
    write_csv(
        root / f"outputs/fixed_fpr_threshold_audit/{SCALE}/threshold_audit_rows.csv",
        bundle.threshold_audit_rows,
    )
    write_jsonl(
        root / f"outputs/pilot_paper_fixed_fpr_results/{SCALE}/pilot_paper_result_records.jsonl",
        bundle.result_records,
    )
    write_csv(
        root / f"outputs/dataset_level_quality/{SCALE}/dataset_quality_metrics.csv",
        bundle.dataset_quality_metrics,
    )
    feature_records_path = (
        root
        / f"outputs/dataset_level_quality/{SCALE}/dataset_quality_formal_feature_records.jsonl"
    )
    feature_records_path.write_bytes(FEATURE_RECORDS_TEXT.encode("utf-8"))
    prompt_path = root / "configs/paper_main_probe_paper_prompts.txt"
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_path.write_text(
        "\n".join(f"a governed prompt {index}" for index in range(PROMPT_COUNT)) + "\n",
        encoding="utf-8",
    )


@pytest.mark.quick
def test_result_closure_gate_passes_only_when_all_semantic_evidence_is_ready() -> None:
    """完整逐攻击披露可闭合, 不要求每个攻击均形成显著优势。"""

    bundle = ready_bundle()
    checks = build_result_closure_gate_checks(bundle)
    report = build_result_closure_gate_report(bundle, checks)

    assert report["result_closure_ready"] is True
    assert report["evidence_closure_allowed"] is True
    assert report["closure_decision"] == "pass"
    assert report["blocked_check_count"] == 0
    assert all(row["check_status"] == "pass" for row in checks)
    assert bundle.result_analysis_summary["superiority_claim_ready_count"] == 0
    assert bundle.result_analysis_summary["universal_per_attack_superiority_claim_ready"] is False


@pytest.mark.quick
def test_result_closure_gate_blocks_unscoped_fpr_and_inexact_attack_roles() -> None:
    """baseline、质量或攻击角色边界不精确时必须阻断证据闭合。"""

    bundle = ready_bundle()
    polluted_attack_report = {
        **bundle.attack_report,
        "performed_attack_record_count": int(bundle.attack_report["attack_record_count"]) - 1,
        "attack_split_role_counts": {
            **dict(bundle.attack_report["attack_split_role_counts"]),
            "jpeg_70|unexpected_role": TEST_COUNT,
        },
    }
    blocked_bundle = replace(
        bundle,
        attack_report=polluted_attack_report,
        baseline_report={**bundle.baseline_report, "target_fpr": 0.01},
        dataset_quality_summary={**bundle.dataset_quality_summary, "target_fpr": 0.01},
    )

    report = build_result_closure_gate_report(
        blocked_bundle,
        build_result_closure_gate_checks(blocked_bundle),
    )

    assert report["result_closure_ready"] is False
    assert "target_fpr_consistent" in report["blocked_check_ids"]
    assert "test_split_count_consistent" in report["blocked_check_ids"]
    assert "attack_matrix_ready" in report["blocked_check_ids"]


@pytest.mark.quick
def test_result_closure_gate_blocks_mixed_scope_and_entry_review_denial() -> None:
    """混入其他层级结果或入口拒绝时, 门禁必须 fail-closed。"""

    bundle = ready_bundle()
    mixed_record = dict(bundle.result_records[0])
    mixed_record["paper_claim_scale"] = "pilot_paper"
    mixed_digest_payload = {
        key: value
        for key, value in mixed_record.items()
        if key not in {"pilot_paper_result_record_digest", "pilot_paper_result_record_id"}
    }
    mixed_digest = build_stable_digest(mixed_digest_payload)
    mixed_record["pilot_paper_result_record_digest"] = mixed_digest
    mixed_record["pilot_paper_result_record_id"] = f"pilot_paper_result_record_{mixed_digest[:16]}"
    denied_entry = {**bundle.entry_review_report, "evidence_closure_allowed": False}
    incomplete_primary_evidence = {
        **bundle.primary_baseline_evidence_summary,
        "formal_result_ready_count": 3,
        "formal_result_ready_ids": ["tree_ring", "gaussian_shading", "shallow_diffuse"],
        "primary_baseline_formal_ready": False,
    }
    blocked_bundle = replace(
        bundle,
        result_records=(mixed_record,),
        primary_baseline_evidence_summary=incomplete_primary_evidence,
        entry_review_report=denied_entry,
    )

    checks = build_result_closure_gate_checks(blocked_bundle)
    report = build_result_closure_gate_report(blocked_bundle, checks)

    assert report["result_closure_ready"] is False
    assert report["closure_decision"] == "blocked"
    assert "current_run_scope_consistent" in report["blocked_check_ids"]
    assert "primary_baseline_evidence_ready" in report["blocked_check_ids"]
    assert "evidence_closure_entry_ready" in report["blocked_check_ids"]


@pytest.mark.quick
def test_result_closure_gate_rejects_six_item_ablation_summary() -> None:
    """旧的6项消融即使自行声明 ready 也不得通过正式8项门禁。"""

    bundle = ready_bundle()
    six_ids = list(FORMAL_RUNTIME_RERUN_ABLATION_IDS[:6])
    six_item_summary = {
        **bundle.ablation_summary,
        "record_count": PROMPT_COUNT * 6,
        "ablation_count": 6,
        "per_ablation_calibration_count": 6,
        "generation_rerun_count": PROMPT_COUNT * 6,
        "attack_and_detection_rerun_count": PROMPT_COUNT * 6,
        "actual_ablation_ids": six_ids,
        "ablation_spec_digest": build_stable_digest(six_ids),
        "ablation_exact_set_ready": True,
    }
    blocked_bundle = replace(bundle, ablation_summary=six_item_summary)

    report = build_result_closure_gate_report(
        blocked_bundle,
        build_result_closure_gate_checks(blocked_bundle),
    )

    assert report["result_closure_ready"] is False
    assert "formal_ablation_ready" in report["blocked_check_ids"]


@pytest.mark.quick
def test_result_closure_gate_rejects_incomplete_feature_coverage_and_metric_rows() -> None:
    """特征缺配对或 FID/KID 行数不精确时必须阻断质量证据。"""

    bundle = ready_bundle()
    incomplete_report = {
        **bundle.dataset_quality_feature_report,
        "accepted_feature_pair_count": PROMPT_COUNT - 1,
        "missing_feature_pair_count": 1,
    }
    blocked_bundle = replace(
        bundle,
        dataset_quality_feature_report=incomplete_report,
        dataset_quality_metrics=bundle.dataset_quality_metrics[:1],
    )

    report = build_result_closure_gate_report(
        blocked_bundle,
        build_result_closure_gate_checks(blocked_bundle),
    )

    assert report["result_closure_ready"] is False
    assert "formal_fid_kid_ready" in report["blocked_check_ids"]


@pytest.mark.quick
def test_result_closure_writer_is_run_scoped_and_require_pass_returns_nonzero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """脚本应按 current run 写出 manifest, 阻断时 `--require-pass` 返回非零。"""

    monkeypatch.setenv("SLM_WM_PAPER_RUN_NAME", SCALE)
    bundle = ready_bundle()
    write_bundle_inputs(tmp_path, bundle)

    report = write_result_closure_gate_outputs(root=tmp_path)
    output_dir = tmp_path / "outputs/result_closure_gate" / SCALE
    written_report = json.loads((output_dir / "result_closure_gate_report.json").read_text(encoding="utf-8"))
    written_manifest = json.loads((output_dir / "manifest.local.json").read_text(encoding="utf-8"))

    assert report["result_closure_ready"] is True
    assert written_report["result_closure_ready"] is True
    assert written_manifest["artifact_id"] == f"{SCALE}_result_closure_gate_manifest"
    assert written_manifest["metadata"]["result_closure_ready"] is True
    source_map = written_report["closure_source_file_sha256"]
    assert set(source_map) == set(written_manifest["input_paths"])
    assert all(
        hashlib.sha256((tmp_path / path).read_bytes()).hexdigest() == digest
        for path, digest in source_map.items()
    )
    assert written_report["closure_source_file_digest"] == build_stable_digest(source_map)
    assert written_manifest["metadata"]["closure_source_file_sha256"] == source_map
    assert written_manifest["metadata"]["closure_source_file_digest"] == written_report[
        "closure_source_file_digest"
    ]
    assert written_manifest["metadata"]["report_digest"] == hashlib.sha256(
        (output_dir / "result_closure_gate_report.json").read_bytes()
    ).hexdigest()
    assert written_manifest["metadata"]["expected_prompt_id_digest"] == PROMPT_ID_DIGEST

    denied_entry_path = (
        tmp_path / f"outputs/evidence_closure_entry_review/{SCALE}/entry_review_report.json"
    )
    denied_entry = {**bundle.entry_review_report, "evidence_closure_allowed": False}
    write_json(denied_entry_path, denied_entry)
    monkeypatch.setattr(
        sys,
        "argv",
        ["write_result_closure_gate_outputs.py", "--root", str(tmp_path), "--require-pass"],
    )
    with pytest.raises(SystemExit) as exit_info:
        main()
    assert exit_info.value.code == 1
