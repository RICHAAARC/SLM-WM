"""外部 baseline 共同协议对比的轻量功能测试。"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from experiments.protocol.attacks import attack_config_digest, resolve_formal_attack_config
from experiments.protocol.fixed_fpr_observation_audit import (
    FORMAL_THRESHOLD_SOURCE,
    audit_fixed_fpr_observation_threshold,
    conformal_threshold_from_clean_negative_scores,
)
from main.core.digest import build_stable_digest
from paper_experiments.baselines import (
    build_baseline_observations,
    default_baseline_specs,
)
from scripts.write_external_baseline_comparison_outputs import (
    align_comparison_table_claim_scope,
    build_runtime_report,
)


_JPEG_CONFIG = resolve_formal_attack_config(
    attack_family="standard_distortion",
    attack_name="jpeg_compression",
)
FORMAL_JPEG_IDENTITY = {
    "attack_id": _JPEG_CONFIG.attack_id,
    "resource_profile": _JPEG_CONFIG.resource_profile,
    "attack_config_digest": attack_config_digest(_JPEG_CONFIG),
}
_REGENERATION_CONFIG = resolve_formal_attack_config(
    attack_family="regeneration_attack",
    attack_name="img2img_regeneration",
    resource_profile="full_extra",
)
FORMAL_REGENERATION_IDENTITY = {
    "attack_id": _REGENERATION_CONFIG.attack_id,
    "attack_config_digest": attack_config_digest(_REGENERATION_CONFIG),
}


@pytest.fixture(autouse=True)
def _select_pilot_paper(monkeypatch: pytest.MonkeyPatch) -> None:
    """本模块使用 pilot_paper 的统一 FPR=0.01 夹具."""

    monkeypatch.setenv("SLM_WM_PAPER_RUN_NAME", "pilot_paper")


def write_pilot_threshold_observation_evidence(
    path: Path,
    *,
    evidence_path: str,
) -> dict[str, object]:
    """写出可重算阈值且逐 Prompt 覆盖的 pilot_paper observation evidence。"""

    calibration_scores = [index / 330.0 for index in range(330)]
    threshold = conformal_threshold_from_clean_negative_scores(calibration_scores, target_fpr=0.01)

    def observation(
        *,
        split: str,
        prompt_id: str,
        event_id: str,
        attack_family: str,
        attack_condition: str,
        sample_role: str,
        score: float,
        quality_score: float | None = None,
    ) -> dict[str, object]:
        row: dict[str, object] = {
            "baseline_id": "tree_ring",
            "split": split,
            "prompt_id": prompt_id,
            "event_id": event_id,
            "attack_family": attack_family,
            "attack_condition": attack_condition,
            "sample_role": sample_role,
            "score": score,
            "threshold": threshold,
            "threshold_source": FORMAL_THRESHOLD_SOURCE,
            "detection_decision": score >= threshold,
        }
        if quality_score is not None:
            row["quality_score"] = quality_score
        if sample_role in {"attacked_negative", "attacked_positive"}:
            row.update(FORMAL_JPEG_IDENTITY)
        return row

    observations = [
        *[
            observation(
                split="calibration",
                prompt_id=f"calibration_{index:03d}",
                event_id=f"calibration_clean_negative_{index:03d}",
                attack_family="clean",
                attack_condition="clean_none",
                sample_role="clean_negative",
                score=score,
            )
            for index, score in enumerate(calibration_scores)
        ],
        *[
            observation(
                split="test",
                prompt_id=f"test_{index:03d}",
                event_id=f"test_clean_negative_{index:03d}",
                attack_family="clean",
                attack_condition="clean_none",
                sample_role="clean_negative",
                score=threshold - 1.0,
            )
            for index in range(340)
        ],
        *[
            observation(
                split="test",
                prompt_id=f"test_{index:03d}",
                event_id=f"test_attacked_positive_{index:03d}",
                attack_family="standard_distortion",
                attack_condition="jpeg_compression",
                sample_role="attacked_positive",
                score=threshold + 1.0 if index < 238 else threshold - 1.0,
                quality_score=0.88,
            )
            for index in range(340)
        ],
        *[
            observation(
                split="test",
                prompt_id=f"test_{index:03d}",
                event_id=f"test_attacked_negative_{index:03d}",
                attack_family="standard_distortion",
                attack_condition="jpeg_compression",
                sample_role="attacked_negative",
                score=threshold + 1.0 if index < 34 else threshold - 1.0,
                quality_score=0.88,
            )
            for index in range(340)
        ],
    ]
    path.write_text(json.dumps(observations, ensure_ascii=False), encoding="utf-8")
    audit = audit_fixed_fpr_observation_threshold(
        observations,
        target_fpr=0.01,
        expected_calibration_source_negative_count=330,
    )
    assert audit.fixed_fpr_ready is True
    return {
        "evaluation_split": "test",
        "calibrated_detection_threshold": threshold,
        "threshold_source": FORMAL_THRESHOLD_SOURCE,
        "calibration_clean_negative_count": 330,
        "test_clean_negative_count": 340,
        "threshold_digest": audit.threshold_digest,
        "fixed_fpr_observation_evidence_path": evidence_path,
        "fixed_fpr_observation_evidence_digest": build_stable_digest(observations),
    }


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


@pytest.mark.quick
def test_baseline_observations_require_explicit_target_fpr() -> None:
    """共同 baseline 观测不得从缺失边界回退到任一数值工作点。"""

    with pytest.raises(ValueError, match="显式提供 target_fpr"):
        build_baseline_observations((), (), {})


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
                "attack_id",
                "attack_family",
                "attack_name",
                "resource_profile",
                "attack_config_digest",
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
                **FORMAL_JPEG_IDENTITY,
                "attack_family": "standard_distortion",
                "attack_name": "jpeg_compression",
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
                **FORMAL_REGENERATION_IDENTITY,
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
            {
                "target_fpr": 0.01,
                "fixed_fpr_threshold_audit_ready": True,
                "all_method_thresholds_ready": True,
                "supports_paper_claim": True,
            },
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
