"""pilot_paper 论文结果分析表与失败案例图的轻量功能测试。"""

from __future__ import annotations

import base64
import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.write_paper_result_analysis_outputs import (
    build_result_template_coverage,
)
from paper_experiments.analysis.result_analysis_payload import (
    PAIRED_SUPERIORITY_FIELDNAMES,
    result_analysis_payload_binding_ready,
)


pytestmark = pytest.mark.quick


@pytest.fixture(autouse=True)
def _select_pilot_paper(monkeypatch: pytest.MonkeyPatch) -> None:
    """本模块的结果分析夹具固定使用 pilot_paper."""

    monkeypatch.setenv("SLM_WM_PAPER_RUN_NAME", "pilot_paper")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    """写出测试用 JSONL。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_paired_superiority_inputs(root: Path) -> None:
    """写出结果分析必须绑定的四方法总体配对优势证据."""

    output_dir = root / "outputs" / "paired_superiority_analysis" / "pilot_paper"
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_ids = ("tree_ring", "gaussian_shading", "shallow_diffuse", "t2smark")
    summary = {
        "paper_claim_scale": "pilot_paper",
        "target_fpr": 0.1,
        "paired_superiority_exact_set_ready": True,
        "paired_superiority_scale_ready": True,
        "overall_paired_superiority_ready": True,
        "paired_outcome_set_digest": "a" * 64,
        "paired_superiority_rows_digest": "b" * 64,
        "paired_superiority_protocol_digest": "c" * 64,
        "quality_matching_protocol_schema": (
            "paired_prompt_embedding_ssim_caliper"
        ),
        "quality_matching_protocol_digest": "1" * 64,
        "quality_metric_name": "embedding_pair_ssim",
        "quality_match_caliper": 0.02,
        "minimum_matched_prompt_fraction": 0.8,
        "quality_matched_row_count": len(baseline_ids),
        "quality_matched_ready_ids": list(baseline_ids),
        "quality_matched_exact_set_ready": True,
        "overall_quality_matched_superiority_ready": True,
        "quality_matched_rows_digest": "2" * 64,
        "quality_matching_uses_detection_labels": False,
        "supports_quality_matched_paper_claim": True,
        "paired_test_prompt_count": 340,
        "paired_test_prompt_id_digest": "d" * 64,
        "expected_attack_count": 1,
        "paired_attack_registry_digest": "e" * 64,
        "threshold_audit_rows_digest": "f" * 64,
        "claim_p_value_method": "bounded_hoeffding_prompt_cluster_mean",
        "sharp_null_diagnostic_method": "exact_prompt_cluster_sign_flip_dp",
        "bootstrap_analysis_schema": "paired_prompt_cluster_bootstrap",
        "bootstrap_bit_generator": "PCG64",
        "bootstrap_quantile_method": "linear",
        "bootstrap_resample_count": 100_000,
        "confidence_level": 0.95,
        "method_observation_source_sha256_map": {
            "slm_wm": "1" * 64,
            "tree_ring": "2" * 64,
            "gaussian_shading": "3" * 64,
            "shallow_diffuse": "4" * 64,
            "t2smark": "5" * 64,
        },
        "method_observation_source_path_map": {
            method_id: f"outputs/observations/{method_id}.json"
            for method_id in ("slm_wm", *baseline_ids)
        },
        "method_threshold_digest_map": {
            "slm_wm": "6" * 64,
            "tree_ring": "7" * 64,
            "gaussian_shading": "8" * 64,
            "shallow_diffuse": "9" * 64,
            "t2smark": "a" * 64,
        },
        "supports_paper_claim": True,
    }
    (output_dir / "paired_superiority_summary.json").write_text(
        json.dumps(summary),
        encoding="utf-8",
    )
    with (output_dir / "paired_superiority_table.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=PAIRED_SUPERIORITY_FIELDNAMES,
        )
        writer.writeheader()
        writer.writerows(
            {
                "baseline_id": baseline_id,
                "paired_prompt_count": 340,
                "paired_attack_count": 1,
                "paired_observation_count": 340,
                "mean_paired_true_positive_rate_difference": 0.2,
                "mean_paired_difference_ci_low": 0.1,
                "mean_paired_difference_ci_high": 0.3,
                "positive_prompt_cluster_count": 340,
                "negative_prompt_cluster_count": 0,
                "tied_prompt_cluster_count": 0,
                "one_sided_bounded_hoeffding_mean_p_value": 0.001,
                "one_sided_exact_prompt_cluster_sign_flip_p_value": 0.001,
                "exact_prompt_cluster_sign_flip_p_value_is_diagnostic": True,
                "sharp_null_diagnostic_method": (
                    "exact_prompt_cluster_sign_flip_dp"
                ),
                "claim_p_value_method": (
                    "bounded_hoeffding_prompt_cluster_mean"
                ),
                "holm_adjusted_p_value": 0.004,
                "confidence_level": 0.95,
                "bootstrap_resample_count": 100_000,
                "bootstrap_seed_digest_random": "3" * 64,
                "bootstrap_analysis_schema": "paired_prompt_cluster_bootstrap",
                "bootstrap_bit_generator": "PCG64",
                "bootstrap_quantile_method": "linear",
                "proposed_method_threshold_digest": "6" * 64,
                "baseline_method_threshold_digest": "7" * 64,
                "paired_test_prompt_id_digest": "d" * 64,
                "paired_attack_registry_digest": "e" * 64,
                "paired_outcome_set_digest": "a" * 64,
                "protocol_digest": "c" * 64,
                "paired_superiority_ready": True,
                "quality_matching_protocol_schema": (
                    "paired_prompt_embedding_ssim_caliper"
                ),
                "quality_matching_protocol_digest": "1" * 64,
                "quality_metric_name": "embedding_pair_ssim",
                "quality_match_caliper": 0.02,
                "minimum_matched_prompt_fraction": 0.8,
                "total_quality_prompt_count": 340,
                "minimum_matched_prompt_count": 272,
                "matched_prompt_count": 340,
                "unmatched_prompt_count": 0,
                "matched_prompt_fraction": 1.0,
                "proposed_embedding_pair_ssim_mean": 0.95,
                "baseline_embedding_pair_ssim_mean": 0.95,
                "mean_embedding_pair_ssim_gap": 0.0,
                "max_absolute_embedding_pair_ssim_gap": 0.0,
                "quality_match_coverage_ready": True,
                "quality_matched_observation_count": 340,
                "quality_matched_mean_paired_true_positive_rate_difference": 0.2,
                "quality_matched_mean_paired_difference_ci_low": 0.1,
                "quality_matched_mean_paired_difference_ci_high": 0.3,
                "quality_matched_holm_adjusted_p_value": 0.004,
                "quality_matched_superiority_ready": True,
                "quality_matched_row_digest": "4" * 64,
                "supports_paper_claim": True,
            }
            for baseline_id in baseline_ids
        )
    (output_dir / "manifest.local.json").write_text(
        json.dumps(
            {
                "artifact_id": "paired_superiority_analysis_manifest",
                "metadata": summary,
            }
        ),
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
        "confidence_interval_method": "bounded_hoeffding",
        "confidence_level": 0.95,
        "supports_paper_claim": True,
    }


def test_result_template_coverage_counts_duplicate_keys() -> None:
    """结果模板覆盖必须显式阻断重复的 method × attack 记录。"""

    record = _result_record("slm_wm_current", "jpeg_compression", 0.9, 0.82, 0.95)
    coverage = build_result_template_coverage([record, dict(record)])

    assert coverage["actual_result_record_count"] == 2
    assert coverage["unique_result_record_key_count"] == 1
    assert coverage["duplicate_result_record_count"] == 1
    assert coverage["result_template_coverage_ready"] is False
