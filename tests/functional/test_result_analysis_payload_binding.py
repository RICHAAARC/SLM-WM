"""验证结果分析表图的精确路径、摘要与 manifest 绑定."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from main.core.digest import build_stable_digest
from experiments.artifacts.attack_family_metrics import build_attack_family_metrics
from experiments.protocol.dataset_quality import (
    FORMAL_FEATURE_BACKEND,
    FORMAL_FEATURE_EXTRACTOR_ID,
    rebuild_formal_fid_kid_metric_rows,
)
from paper_experiments.analysis.result_analysis_payload import (
    RESULT_ANALYSIS_PAYLOAD_FILE_NAMES,
    ResultAnalysisSemanticError,
    build_confidence_interval_rows,
    build_failure_case_records,
    build_failure_case_svg_text,
    build_governed_paper_payload_path_map,
    build_main_comparison_rows_from_result_records,
    build_per_attack_superiority_rows,
    build_result_analysis_manifest_config,
    build_result_analysis_payload_binding,
    rebuild_and_validate_result_analysis_semantics,
    result_analysis_payload_binding_ready,
)
from scripts.write_pilot_paper_complete_result_package import (
    collect_result_closure_source_entries,
)


pytestmark = pytest.mark.quick


def _ready_payload(tmp_path: Path) -> tuple[dict[str, object], dict[str, object], dict[str, str]]:
    """写出四类最小 payload 并构造相互一致的 summary 与 manifest."""

    output_dir = (
        tmp_path / "outputs/pilot_paper_result_analysis/probe_paper"
    )
    output_dir.mkdir(parents=True)
    contents = {
        "main_confidence_interval_table": b"method_id,tpr\nslm_wm_current,0.9\n",
        "per_attack_superiority_table": b"attack_name,margin\njpeg,0.2\n",
        "failure_case_records": b'{"attack_name":"jpeg"}\n',
        "failure_case_figure": b"<svg><title>failure</title></svg>\n",
    }
    for role, file_name in RESULT_ANALYSIS_PAYLOAD_FILE_NAMES.items():
        (output_dir / file_name).write_bytes(contents[role])
    binding = build_result_analysis_payload_binding(
        repository_root=tmp_path,
        output_dir=output_dir,
    )
    summary: dict[str, object] = {
        "paper_claim_scale": "probe_paper",
        "failure_case_limit": 12,
        "result_analysis_semantic_rebuild_digest": "e" * 64,
        "result_analysis_semantic_rebuild_ready": True,
        **binding,
    }
    config = build_result_analysis_manifest_config(summary)
    manifest: dict[str, object] = {
        "output_paths": [
            *binding["result_analysis_payload_path_map"].values(),
            "outputs/pilot_paper_result_analysis/probe_paper/result_analysis_summary.json",
            "outputs/pilot_paper_result_analysis/probe_paper/manifest.local.json",
        ],
        "config_digest": build_stable_digest(config),
        "metadata": dict(summary),
    }
    actual_source_sha256 = {
        path: binding["result_analysis_payload_sha256_map"][role]
        for role, path in binding["result_analysis_payload_path_map"].items()
    }
    return summary, manifest, actual_source_sha256


def test_result_analysis_payload_binding_requires_exact_bytes_and_roles(
    tmp_path: Path,
) -> None:
    """完整角色集合、实际字节、summary 和 manifest 一致时才通过."""

    summary, manifest, actual_source_sha256 = _ready_payload(tmp_path)

    assert result_analysis_payload_binding_ready(
        summary=summary,
        manifest=manifest,
        actual_source_sha256=actual_source_sha256,
    )

    drifted_source_sha256 = dict(actual_source_sha256)
    failure_path = summary["result_analysis_payload_path_map"][
        "failure_case_figure"
    ]
    drifted_source_sha256[failure_path] = "f" * 64
    assert not result_analysis_payload_binding_ready(
        summary=summary,
        manifest=manifest,
        actual_source_sha256=drifted_source_sha256,
    )


def test_result_analysis_payload_binding_rejects_missing_role_or_manifest_drift(
    tmp_path: Path,
) -> None:
    """删除角色或只改 manifest metadata 都不得保留 ready 状态."""

    summary, manifest, actual_source_sha256 = _ready_payload(tmp_path)
    incomplete_summary = dict(summary)
    incomplete_path_map = dict(summary["result_analysis_payload_path_map"])
    incomplete_path_map.pop("failure_case_records")
    incomplete_summary["result_analysis_payload_path_map"] = incomplete_path_map
    assert not result_analysis_payload_binding_ready(
        summary=incomplete_summary,
        manifest=manifest,
        actual_source_sha256=actual_source_sha256,
    )

    drifted_manifest = dict(manifest)
    drifted_metadata = dict(manifest["metadata"])
    drifted_metadata["result_analysis_payload_digest"] = "0" * 64
    drifted_manifest["metadata"] = drifted_metadata
    assert not result_analysis_payload_binding_ready(
        summary=summary,
        manifest=drifted_manifest,
        actual_source_sha256=actual_source_sha256,
    )


def test_result_analysis_payload_binding_requires_canonical_exact_paths(
    tmp_path: Path,
) -> None:
    """同名后缀、绝对前缀或反斜杠都不能冒充规范仓库相对路径."""

    summary, manifest, actual_source_sha256 = _ready_payload(tmp_path)
    for forged_path in (
        "archive/outputs/pilot_paper_result_analysis/probe_paper/failure_case_figure.svg",
        "outputs\\pilot_paper_result_analysis\\probe_paper\\failure_case_figure.svg",
    ):
        forged_summary = copy.deepcopy(summary)
        original_path = forged_summary["result_analysis_payload_path_map"][
            "failure_case_figure"
        ]
        forged_summary["result_analysis_payload_path_map"][
            "failure_case_figure"
        ] = forged_path
        forged_sha = dict(actual_source_sha256)
        forged_sha[forged_path] = forged_sha.pop(original_path)
        forged_digest_payload = {
            "result_analysis_payload_path_map": forged_summary[
                "result_analysis_payload_path_map"
            ],
            "result_analysis_payload_sha256_map": forged_summary[
                "result_analysis_payload_sha256_map"
            ],
        }
        forged_summary["result_analysis_payload_digest"] = build_stable_digest(
            forged_digest_payload
        )
        forged_manifest = copy.deepcopy(manifest)
        forged_manifest["metadata"] = copy.deepcopy(forged_summary)
        forged_manifest["output_paths"] = [
            forged_path if path == original_path else path
            for path in forged_manifest["output_paths"]
        ]
        forged_manifest["config_digest"] = build_stable_digest(
            build_result_analysis_manifest_config(forged_summary)
        )

        assert not result_analysis_payload_binding_ready(
            summary=forged_summary,
            manifest=forged_manifest,
            actual_source_sha256=forged_sha,
        )


def test_complete_package_revalidates_result_analysis_payload_bytes(
    tmp_path: Path,
) -> None:
    """完整包收集边界必须再次读取 closure source map 中的表图字节."""

    summary, _, actual_source_sha256 = _ready_payload(tmp_path)
    required_payload_paths = build_governed_paper_payload_path_map("probe_paper")
    for role, relative_path in required_payload_paths.items():
        if relative_path in actual_source_sha256:
            continue
        path = tmp_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"role,value\n{role},1\n", encoding="utf-8")
        actual_source_sha256[relative_path] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    gate_dir = tmp_path / "outputs/result_closure_gate/probe_paper"
    gate_dir.mkdir(parents=True)
    (gate_dir / "result_closure_gate_report.json").write_text(
        json.dumps({"closure_source_file_sha256": actual_source_sha256}),
        encoding="utf-8",
    )
    (gate_dir / "manifest.local.json").write_text("{}\n", encoding="utf-8")

    entries, declared_map, ready = collect_result_closure_source_entries(
        tmp_path,
        paper_run_name="probe_paper",
        excluded_paths=(),
    )

    assert ready
    assert declared_map == actual_source_sha256
    assert len(entries) == len(actual_source_sha256) + 2

    incomplete_map = dict(actual_source_sha256)
    incomplete_map.pop(required_payload_paths["quality_table"])
    (gate_dir / "result_closure_gate_report.json").write_text(
        json.dumps({"closure_source_file_sha256": incomplete_map}),
        encoding="utf-8",
    )
    _, _, ready_without_quality = collect_result_closure_source_entries(
        tmp_path,
        paper_run_name="probe_paper",
        excluded_paths=(),
    )
    assert not ready_without_quality
    (gate_dir / "result_closure_gate_report.json").write_text(
        json.dumps({"closure_source_file_sha256": actual_source_sha256}),
        encoding="utf-8",
    )

    failure_figure_path = tmp_path / summary["result_analysis_payload_path_map"][
        "failure_case_figure"
    ]
    failure_figure_path.write_text("<svg>tampered</svg>\n", encoding="utf-8")
    _, _, ready_after_drift = collect_result_closure_source_entries(
        tmp_path,
        paper_run_name="probe_paper",
        excluded_paths=(),
    )
    assert not ready_after_drift


def _semantic_result_record(
    method_id: str,
    *,
    true_positive_rate: float,
    quality_score_mean: float,
) -> dict[str, object]:
    """构造可同时重建主表、CI 表和逐攻击表的正式记录."""

    ci_low = max(0.0, true_positive_rate - 0.1)
    ci_high = min(1.0, true_positive_rate + 0.1)
    return {
        "paper_claim_scale": "probe_paper",
        "method_id": method_id,
        "attack_id": "standard_distortion_jpeg_compression_full_main",
        "attack_family": "standard_distortion",
        "attack_name": "jpeg_compression",
        "resource_profile": "full_main",
        "attack_config_digest": "a" * 64,
        "metric_status": "measured",
        "true_positive_rate": true_positive_rate,
        "true_positive_rate_ci_low": ci_low,
        "true_positive_rate_ci_high": ci_high,
        "false_positive_rate": 0.0,
        "false_positive_rate_ci_low": 0.0,
        "false_positive_rate_ci_high": 0.1,
        "clean_false_positive_rate": 0.0,
        "clean_false_positive_rate_ci_low": 0.0,
        "clean_false_positive_rate_ci_high": 0.1,
        "attacked_false_positive_rate": 0.0,
        "attacked_false_positive_rate_ci_low": 0.0,
        "attacked_false_positive_rate_ci_high": 0.1,
        "positive_count": 1,
        "negative_count": 1,
        "supported_record_count": 1,
        "quality_score_mean": quality_score_mean,
        "confidence_interval_method": "bounded_hoeffding",
        "confidence_level": 0.95,
        "supports_paper_claim": True,
    }


def _semantic_attack_records() -> list[dict[str, object]]:
    """构造一正一负的逐样本攻击记录, 正样本同时是一条真实失败案例."""

    common = {
        "attack_id": "standard_distortion_jpeg_compression_full_main",
        "attack_family": "standard_distortion",
        "attack_name": "jpeg_compression",
        "resource_profile": "full_main",
        "attack_config_digest": "a" * 64,
        "geometry_reliable": True,
        "formal_rescue_applied": False,
        "supports_paper_claim": True,
    }
    return [
        {
            **common,
            "sample_role": "positive_source",
            "formal_evidence_positive": False,
            "quality_score": 0.8,
            "quality_ssim": 0.8,
            "quality_psnr": 30.0,
            "score_retention": 0.5,
            "lf_score_retention": 0.6,
            "tail_score_retention": 0.4,
            "evidence_decision": False,
            "aligned_content_score_after": 0.2,
            "aligned_content_score_before": 0.8,
            "source_record_id": "source_1",
            "attack_record_id": "attack_1",
            "attacked_image_digest": "b" * 64,
            "source_image_digest": "c" * 64,
            "metadata": {
                "attacked_image_path": (
                    "outputs/attack_matrix/probe_paper/attacked_images/sample_1.png"
                )
            },
        },
        {
            **common,
            "sample_role": "clean_negative",
            "formal_evidence_positive": False,
            "evidence_decision": False,
        },
    ]


def _semantic_feature_records() -> list[dict[str, object]]:
    """构造两个 source/comparison 配对的正式特征记录."""

    rows = []
    vectors = {
        "quality_1": ([0.0, 1.0], [0.1, 1.1]),
        "quality_2": ([2.0, 3.0], [2.1, 3.1]),
    }
    for record_id, (source, comparison) in vectors.items():
        for role, vector in (("source", source), ("comparison", comparison)):
            rows.append(
                {
                    "dataset_quality_record_id": record_id,
                    "dataset_quality_image_role": role,
                    "feature_backend": FORMAL_FEATURE_BACKEND,
                    "feature_extractor_id": FORMAL_FEATURE_EXTRACTOR_ID,
                    "feature_dimension": 2,
                    "feature_vector": vector,
                    "supports_paper_claim": False,
                }
            )
    return rows


def _semantic_paired_rows() -> list[dict[str, object]]:
    """构造3个正结果和1个真实负结果的配对统计表."""

    rows = []
    for index, baseline_id in enumerate(
        ("tree_ring", "gaussian_shading", "shallow_diffuse", "t2smark")
    ):
        negative = index == 0
        mean = -0.1 if negative else 0.1
        ci_low = -0.2 if negative else 0.05
        ci_high = 0.0 if negative else 0.15
        p_value = 0.5 if negative else 0.01
        ready = not negative
        rows.append(
            {
                "baseline_id": baseline_id,
                "paired_prompt_count": 2,
                "paired_attack_count": 1,
                "paired_observation_count": 2,
                "mean_paired_true_positive_rate_difference": mean,
                "mean_paired_difference_ci_low": ci_low,
                "mean_paired_difference_ci_high": ci_high,
                "positive_prompt_cluster_count": 1 if ready else 0,
                "negative_prompt_cluster_count": 0 if ready else 1,
                "tied_prompt_cluster_count": 1,
                "one_sided_bounded_hoeffding_mean_p_value": p_value,
                "one_sided_exact_prompt_cluster_sign_flip_p_value": p_value,
                "exact_prompt_cluster_sign_flip_p_value_is_diagnostic": True,
                "sharp_null_diagnostic_method": "exact_prompt_cluster_sign_flip_dp",
                "claim_p_value_method": "bounded_hoeffding_prompt_cluster_mean",
                "holm_adjusted_p_value": p_value,
                "confidence_level": 0.95,
                "bootstrap_resample_count": 100_000,
                "bootstrap_seed_digest_random": f"{index + 1}" * 64,
                "bootstrap_analysis_schema": "paired_prompt_cluster_bootstrap",
                "bootstrap_bit_generator": "PCG64",
                "bootstrap_quantile_method": "linear",
                "proposed_method_threshold_digest": "d" * 64,
                "baseline_method_threshold_digest": "e" * 64,
                "paired_test_prompt_id_digest": "f" * 64,
                "paired_attack_registry_digest": "1" * 64,
                "paired_outcome_set_digest": "2" * 64,
                "protocol_digest": "3" * 64,
                "paired_superiority_ready": ready,
                "quality_matching_protocol_schema": (
                    "paired_prompt_embedding_ssim_caliper"
                ),
                "quality_matching_protocol_digest": "4" * 64,
                "quality_metric_name": "embedding_pair_ssim",
                "quality_match_caliper": 0.02,
                "minimum_matched_prompt_fraction": 0.8,
                "total_quality_prompt_count": 2,
                "minimum_matched_prompt_count": 2,
                "matched_prompt_count": 2,
                "unmatched_prompt_count": 0,
                "matched_prompt_fraction": 1.0,
                "proposed_embedding_pair_ssim_mean": 0.95,
                "baseline_embedding_pair_ssim_mean": 0.95,
                "mean_embedding_pair_ssim_gap": 0.0,
                "max_absolute_embedding_pair_ssim_gap": 0.0,
                "quality_match_coverage_ready": True,
                "quality_matched_observation_count": 2,
                "quality_matched_mean_paired_true_positive_rate_difference": mean,
                "quality_matched_mean_paired_difference_ci_low": ci_low,
                "quality_matched_mean_paired_difference_ci_high": ci_high,
                "quality_matched_holm_adjusted_p_value": p_value,
                "quality_matched_superiority_ready": ready,
                "quality_matched_row_digest": "5" * 64,
                "supports_paper_claim": ready,
            }
        )
    return rows


def _semantic_payload_kwargs() -> dict[str, object]:
    """构造七类 payload 及其独立正式来源."""

    result_records = [
        _semantic_result_record(
            "slm_wm_current", true_positive_rate=0.0, quality_score_mean=0.8
        ),
        _semantic_result_record(
            "tree_ring", true_positive_rate=0.7, quality_score_mean=0.7
        ),
        _semantic_result_record(
            "gaussian_shading", true_positive_rate=0.6, quality_score_mean=0.7
        ),
        _semantic_result_record(
            "shallow_diffuse", true_positive_rate=0.5, quality_score_mean=0.7
        ),
        _semantic_result_record(
            "t2smark", true_positive_rate=0.4, quality_score_mean=0.7
        ),
    ]
    attack_records = _semantic_attack_records()
    attack_metrics = list(
        build_attack_family_metrics(attack_records, 0.1, True)
    )
    feature_records = _semantic_feature_records()
    quality_rows = rebuild_formal_fid_kid_metric_rows(
        [[0.0, 1.0], [2.0, 3.0]],
        [[0.1, 1.1], [2.1, 3.1]],
        sample_pair_count=2,
    )
    failure_rows = build_failure_case_records(attack_records, limit=12)
    figure_path = (
        "outputs/pilot_paper_result_analysis/probe_paper/failure_case_figure.svg"
    )
    return {
        "paper_claim_scale": "probe_paper",
        "governed_payload_path_map": build_governed_paper_payload_path_map(
            "probe_paper"
        ),
        "target_fpr": 0.1,
        "result_records": result_records,
        "attack_detection_records": attack_records,
        "attack_family_metrics": attack_metrics,
        "baseline_comparison_rows": (
            build_main_comparison_rows_from_result_records(result_records)
        ),
        "dataset_quality_feature_records": feature_records,
        "dataset_quality_metric_rows": quality_rows,
        "expected_quality_pair_count": 2,
        "paired_superiority_rows": _semantic_paired_rows(),
        "confidence_interval_rows": build_confidence_interval_rows(
            result_records
        ),
        "per_attack_superiority_rows": build_per_attack_superiority_rows(
            result_records
        ),
        "failure_case_rows": failure_rows,
        "failure_case_svg_text": build_failure_case_svg_text(
            failure_rows,
            failure_figure_path=figure_path,
        ),
        "failure_figure_path": figure_path,
        "failure_case_limit": 12,
    }


def test_result_analysis_semantics_rebuilds_all_payloads_and_keeps_negative_results() -> None:
    """七类 payload 可重建时通过, 配对与逐攻击负结果保持为 False."""

    kwargs = _semantic_payload_kwargs()
    evidence = rebuild_and_validate_result_analysis_semantics(**kwargs)

    assert evidence["governed_paper_payload_semantic_rebuild_ready"] is True
    assert evidence["paired_superiority_negative_result_count"] == 1


def test_result_analysis_semantics_rejects_zero_failure_disclosure_limit() -> None:
    """存在真实 false-negative 时不得用 limit=0 生成看似无失败的表图。"""

    kwargs = _semantic_payload_kwargs()
    kwargs["failure_case_limit"] = 0
    kwargs["failure_case_rows"] = []
    kwargs["failure_case_svg_text"] = build_failure_case_svg_text(
        [],
        failure_figure_path=str(kwargs["failure_figure_path"]),
    )
    with pytest.raises(ResultAnalysisSemanticError):
        rebuild_and_validate_result_analysis_semantics(**kwargs)


def test_per_attack_superiority_uses_largest_baseline_ci_upper_bound() -> None:
    """点估计较低但 CI 上界更高的 baseline 必须成为保守比较对象。"""

    rows = [
        _semantic_result_record(
            "slm_wm_current",
            true_positive_rate=0.9,
            quality_score_mean=0.9,
        )
    ]
    for method_id, true_positive_rate in zip(
        ("tree_ring", "gaussian_shading", "shallow_diffuse", "t2smark"),
        (0.7, 0.69, 0.5, 0.4),
        strict=True,
    ):
        rows.append(
            _semantic_result_record(
                method_id,
                true_positive_rate=true_positive_rate,
                quality_score_mean=0.8,
            )
        )
    rows[1]["true_positive_rate_ci_high"] = 0.79
    rows[2]["true_positive_rate_ci_high"] = 0.95

    result = build_per_attack_superiority_rows(rows)[0]

    assert result["best_baseline_id"] == "gaussian_shading"
    assert result["conservative_ci_margin"] == pytest.approx(-0.15)
    assert result["superiority_claim_ready"] is False


@pytest.mark.parametrize(
    "payload_role",
    (
        "main_comparison_table",
        "attack_table",
        "quality_table",
        "main_confidence_interval_table",
        "per_attack_superiority_table",
        "failure_case_records",
        "failure_case_figure",
        "payload_paths",
    ),
)
def test_result_analysis_semantics_fail_closed_on_each_payload_cell(
    payload_role: str,
) -> None:
    """任一表格单元、失败记录或 SVG 漂移都必须阻断语义证据链."""

    kwargs = _semantic_payload_kwargs()
    if payload_role == "main_comparison_table":
        kwargs["baseline_comparison_rows"][0]["true_positive_rate"] = 0.1
    elif payload_role == "attack_table":
        kwargs["attack_family_metrics"][0]["quality_ssim_mean"] = 0.7
    elif payload_role == "quality_table":
        kwargs["dataset_quality_metric_rows"][0]["quality_metric_value"] += 0.1
    elif payload_role == "main_confidence_interval_table":
        kwargs["confidence_interval_rows"][0]["true_positive_rate_ci_high"] = 0.9
    elif payload_role == "per_attack_superiority_table":
        kwargs["per_attack_superiority_rows"][0]["conservative_ci_margin"] = 0.1
    elif payload_role == "failure_case_records":
        kwargs["failure_case_rows"][0]["attacked_image_digest"] = "0" * 64
    elif payload_role == "payload_paths":
        kwargs["governed_payload_path_map"]["quality_table"] = (
            "archive/outputs/dataset_level_quality/probe_paper/"
            "dataset_quality_metrics.csv"
        )
    else:
        kwargs["failure_case_svg_text"] += "<!-- drift -->\n"

    with pytest.raises(ResultAnalysisSemanticError):
        rebuild_and_validate_result_analysis_semantics(**kwargs)
