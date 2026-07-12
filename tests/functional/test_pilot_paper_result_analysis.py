"""pilot_paper 论文结果分析表与失败案例图的轻量功能测试。"""

from __future__ import annotations

import base64
import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.write_pilot_paper_result_analysis_outputs import (
    build_result_template_coverage,
    write_pilot_paper_result_analysis_outputs,
)
from paper_experiments.analysis.result_analysis_payload import (
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
        "target_fpr": 0.01,
        "paired_superiority_exact_set_ready": True,
        "paired_superiority_scale_ready": True,
        "overall_paired_superiority_ready": True,
        "paired_outcome_set_digest": "a" * 64,
        "paired_superiority_rows_digest": "b" * 64,
        "paired_superiority_protocol_digest": "c" * 64,
        "paired_test_prompt_count": 340,
        "paired_test_prompt_id_digest": "d" * 64,
        "expected_attack_count": 1,
        "paired_attack_registry_digest": "e" * 64,
        "threshold_audit_rows_digest": "f" * 64,
        "claim_p_value_method": "bounded_hoeffding_prompt_cluster_mean",
        "sharp_null_diagnostic_method": "exact_prompt_cluster_sign_flip_dp",
        "bootstrap_analysis_schema": "paired_prompt_cluster_bootstrap_v1",
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
            fieldnames=("baseline_id", "paired_superiority_ready", "supports_paper_claim"),
        )
        writer.writeheader()
        writer.writerows(
            {
                "baseline_id": baseline_id,
                "paired_superiority_ready": True,
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


def test_pilot_paper_result_analysis_rebuilds_tables_and_failure_figure(tmp_path: Path) -> None:
    """结果分析脚本应从 governed records 重建 CI 表、优势表和失败案例 SVG。"""

    _write_paired_superiority_inputs(tmp_path)
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
    attacked_image.write_bytes(
        base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
        )
    )
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
    manifest = write_pilot_paper_result_analysis_outputs(
        root=tmp_path,
        result_records_path=result_records_path,
        attack_detection_records_path=real_records_path,
        failure_case_limit=12,
    )

    output_dir = tmp_path / "outputs" / "pilot_paper_result_analysis" / "pilot_paper"
    confidence_interval_rows = list(
        csv.DictReader((output_dir / "confidence_interval_table.csv").open(encoding="utf-8"))
    )
    superiority_rows = list(csv.DictReader((output_dir / "per_attack_superiority_table.csv").open(encoding="utf-8")))
    failure_rows = [
        json.loads(line)
        for line in (output_dir / "failure_case_records.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    svg_text = (output_dir / "failure_case_figure.svg").read_text(encoding="utf-8")
    summary = json.loads((output_dir / "result_analysis_summary.json").read_text(encoding="utf-8"))

    assert manifest["artifact_id"] == "pilot_paper_result_analysis_manifest"
    assert len(confidence_interval_rows) == 5
    assert len(superiority_rows) == 1
    assert superiority_rows[0]["best_baseline_id"] == "shallow_diffuse"
    assert float(superiority_rows[0]["slm_minus_best_baseline_tpr"]) == pytest.approx(0.2)
    assert superiority_rows[0]["superiority_claim_ready"] == "True"
    assert len(failure_rows) == 1
    assert failure_rows[0]["attacked_image_digest"] == "attacked_digest"
    assert "jpeg_compression" in svg_text
    assert "placeholder" not in svg_text
    assert summary["failure_case_figure_ready"] is True
    payload_path_map = summary["result_analysis_payload_path_map"]
    payload_sha256_map = summary["result_analysis_payload_sha256_map"]
    actual_source_sha256 = {
        path: hashlib.sha256((tmp_path / path).read_bytes()).hexdigest()
        for path in payload_path_map.values()
    }
    assert {
        role: actual_source_sha256[path]
        for role, path in payload_path_map.items()
    } == payload_sha256_map
    assert result_analysis_payload_binding_ready(
        summary=summary,
        manifest=manifest,
        actual_source_sha256=actual_source_sha256,
    )
    assert summary["result_template_coverage_ready"] is False
    assert summary["supports_paper_claim"] is False


def test_result_analysis_rejects_failure_record_without_attacked_image(tmp_path: Path) -> None:
    """失败记录缺少实际攻击图像时必须停止生成论文图。"""

    _write_paired_superiority_inputs(tmp_path)
    result_records_path = tmp_path / "outputs" / "fixed_fpr" / "result_records.jsonl"
    _write_jsonl(
        result_records_path,
        [
            _result_record("slm_wm_current", "jpeg_compression", 0.9, 0.82, 0.95),
            _result_record("tree_ring", "jpeg_compression", 0.6, 0.50, 0.70),
        ],
    )
    real_records_path = tmp_path / "outputs" / "attacks" / "formal_detection_records.jsonl"
    _write_jsonl(
        real_records_path,
        [
            {
                "sample_role": "positive_source",
                "attack_family": "standard_distortion",
                "attack_name": "jpeg_compression",
                "aligned_content_score_after": 0.1,
                "evidence_decision": False,
                "metadata": {"attacked_image_path": "outputs/attacks/missing.png"},
                "supports_paper_claim": True,
            }
        ],
    )
    with pytest.raises(FileNotFoundError, match="失败案例攻击图像不存在"):
        write_pilot_paper_result_analysis_outputs(
            root=tmp_path,
            result_records_path=result_records_path,
            attack_detection_records_path=real_records_path,
        )


def test_result_analysis_discloses_nonwinning_attack_without_blocking_complete_analysis(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """逐攻击未显著胜出时应如实披露, 但完整分析证据仍可支持论文使用。"""

    monkeypatch.setenv("SLM_WM_PAPER_RUN_NAME", "pilot_paper")
    _write_paired_superiority_inputs(tmp_path)
    monkeypatch.setattr(
        "scripts.write_pilot_paper_result_analysis_outputs.default_attack_configs",
        lambda: (
            SimpleNamespace(
                enabled=True,
                resource_profile="full_main",
                attack_family="standard_distortion",
                attack_name="jpeg_compression",
            ),
        ),
    )

    result_records_path = tmp_path / "outputs" / "fixed_fpr" / "result_records.jsonl"
    _write_jsonl(
        result_records_path,
        [
            _result_record("slm_wm_current", "jpeg_compression", 0.6, 0.50, 0.70),
            _result_record("tree_ring", "jpeg_compression", 0.7, 0.60, 0.78),
            _result_record("gaussian_shading", "jpeg_compression", 0.55, 0.45, 0.65),
            _result_record("shallow_diffuse", "jpeg_compression", 0.65, 0.55, 0.75),
            _result_record("t2smark", "jpeg_compression", 0.5, 0.40, 0.60),
        ],
    )
    real_records_path = tmp_path / "outputs" / "attacks" / "formal_detection_records.jsonl"
    _write_jsonl(real_records_path, [])

    write_pilot_paper_result_analysis_outputs(
        root=tmp_path,
        result_records_path=result_records_path,
        attack_detection_records_path=real_records_path,
    )
    summary = json.loads(
        (
            tmp_path
            / "outputs"
            / "pilot_paper_result_analysis"
            / "pilot_paper"
            / "result_analysis_summary.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    assert summary["superiority_claim_ready_count"] == 0
    assert summary["per_attack_ci_coverage_ready"] is True
    assert summary["per_attack_superiority_evaluation_ready"] is True
    assert summary["universal_per_attack_superiority_claim_ready"] is False
    assert summary["supports_paper_claim"] is True


def test_result_template_coverage_counts_duplicate_keys() -> None:
    """结果模板覆盖必须显式阻断重复的 method × attack 记录。"""

    record = _result_record("slm_wm_current", "jpeg_compression", 0.9, 0.82, 0.95)
    coverage = build_result_template_coverage([record, dict(record)])

    assert coverage["actual_result_record_count"] == 2
    assert coverage["unique_result_record_key_count"] == 1
    assert coverage["duplicate_result_record_count"] == 1
    assert coverage["result_template_coverage_ready"] is False
