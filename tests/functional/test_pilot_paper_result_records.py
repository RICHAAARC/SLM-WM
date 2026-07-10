"""pilot_paper fixed-FPR 结果记录物化层的轻量功能测试。"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from zipfile import ZipFile

import pytest

from scripts.write_pilot_paper_result_records import (
    ensure_output_dir_under_outputs,
    materialize_output_entries,
    write_pilot_paper_result_record_outputs,
)


def json_line(value: dict[str, object]) -> str:
    """将测试记录转换为 JSONL 单行。"""

    return json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n"


def write_pilot_prompt_file(repo_root: Path, prompt_count: int = 240) -> None:
    """写入满足 pilot_paper fixed-FPR 最小 clean negative 数量的 prompt 文件。"""

    config_dir = repo_root / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    prompt_path = config_dir / "paper_main_pilot_paper_prompts.txt"
    prompt_path.write_text(
        "\n".join(f"a controlled pilot_paper prompt for result import {index}" for index in range(prompt_count)) + "\n",
        encoding="utf-8",
    )


def write_full_prompt_file(repo_root: Path, prompt_count: int = 240) -> None:
    """写入满足 full_paper fixed-FPR 最小 clean negative 数量的 prompt 文件。"""

    config_dir = repo_root / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    prompt_path = config_dir / "paper_main_full_paper_prompts.txt"
    prompt_path.write_text(
        "\n".join(f"a controlled full_paper prompt for result import {index}" for index in range(prompt_count)) + "\n",
        encoding="utf-8",
    )


def write_csv_rows(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    """写出测试 CSV 表格。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_attack_matrix_inputs(repo_root: Path) -> None:
    """构造 SLM-WM 当前方法的最小攻击矩阵聚合证据。"""

    attack_dir = repo_root / "outputs" / "attack_matrix"
    real_attack_dir = repo_root / "outputs" / "real_attack_evaluation"
    quality_dir = repo_root / "outputs" / "dataset_level_quality"
    attack_dir.mkdir(parents=True, exist_ok=True)
    real_attack_dir.mkdir(parents=True, exist_ok=True)
    quality_dir.mkdir(parents=True, exist_ok=True)
    write_csv_rows(
        attack_dir / "attack_family_metrics.csv",
        [
            {
                "attack_family": "standard_distortion",
                "attack_name": "jpeg_compression",
                "resource_profile": "full_main",
                "metric_status": "measured_from_real_attacked_image_watermark_rescore_formal_protocol",
                "attack_record_count": 240,
                "supported_record_count": 240,
                "unsupported_record_count": 0,
                "positive_count": 120,
                "negative_count": 120,
                "true_positive_rate": 0.8,
                "false_positive_rate": 0.0,
                "clean_false_positive_rate": 0.0,
                "attacked_false_positive_rate": 0.0,
                "quality_score_mean": 0.92,
                "quality_score_proxy_mean": 0.92,
                "score_retention_mean": 0.81,
                "lf_score_retention_mean": 0.80,
                "hf_score_retention_mean": 0.82,
                "attention_consistency_proxy_mean": 0.77,
                "geometry_reliable_rate": 1.0,
                "rescue_rate": 0.0,
                "supports_paper_claim": False,
            }
        ],
        [
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
            "quality_score_proxy_mean",
            "score_retention_mean",
            "lf_score_retention_mean",
            "hf_score_retention_mean",
            "attention_consistency_proxy_mean",
            "geometry_reliable_rate",
            "rescue_rate",
            "supports_paper_claim",
        ],
    )
    (attack_dir / "attack_manifest.json").write_text('{"artifact_id":"attack_matrix_manifest"}\n', encoding="utf-8")
    (attack_dir / "attack_detection_records.jsonl").write_text(json_line({"attack_record_id": "attack_a"}), encoding="utf-8")
    (real_attack_dir / "formal_attack_detection_records.jsonl").write_text(
        json_line({"attack_record_id": "real_attack_a"}),
        encoding="utf-8",
    )
    write_csv_rows(
        quality_dir / "dataset_quality_metrics.csv",
        [
            {"quality_metric_name": "fid", "metric_status": "measured"},
            {"quality_metric_name": "kid", "metric_status": "measured"},
        ],
        ["quality_metric_name", "metric_status"],
    )
    (quality_dir / "dataset_quality_summary.json").write_text(
        json.dumps(
            {
                "formal_fid_kid_metric_names_ready": True,
                "formal_fid_kid_claim_gate_ready": True,
                "canonical_formal_feature_extractor_ready": True,
                "formal_fid_kid_claim_blocker": "",
                "dataset_quality_claim_boundary": "formal_fid_kid_measured",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


def baseline_candidate_row() -> dict[str, object]:
    """构造已经通过外部 baseline 受治理导入报告接受的候选记录。"""

    return {
        "baseline_id": "tree_ring",
        "attack_family": "standard_distortion",
        "attack_name": "jpeg_compression",
        "resource_profile": "full_main",
        "comparable_operating_point": "fixed_fpr_0.01",
        "result_protocol_name": "primary_baseline_formal_import_protocol",
        "result_source_type": "governed_import",
        "baseline_result_source": "outputs/external_baseline_results/baseline_result_records.jsonl",
        "baseline_result_source_digest": "baseline_source_digest",
        "metric_status": "measured",
        "prompt_protocol_name": "paper_main_pilot_paper_prompt_protocol",
        "prompt_protocol_digest": "pilot_prompt_digest",
        "adapter_boundary": "method_faithful_sd35_adapter_reproduction",
        "evidence_paths": ["outputs/external_baseline_results/baseline_result_records.jsonl"],
        "method_faithful_adapter_ready": True,
        "full_main_prompt_protocol_ready": True,
        "fixed_fpr_baseline_calibration_ready": True,
        "attack_matrix_baseline_detection_ready": True,
        "formal_evidence_paths_ready": True,
        "positive_count": 120,
        "negative_count": 120,
        "attack_record_count": 240,
        "supported_record_count": 240,
        "true_positive_rate": 0.6,
        "false_positive_rate": 0.0,
        "clean_false_positive_rate": 0.0,
        "attacked_false_positive_rate": 0.0,
        "quality_score_mean": 0.88,
        "score_retention_mean": 0.70,
        "baseline_result_record_id": "primary_baseline_formal_result_test",
        "baseline_result_digest": "baseline_result_digest",
        "supports_paper_claim": False,
    }


def write_baseline_inputs(repo_root: Path, accepted: bool = True) -> None:
    """写出外部 baseline 候选记录和受治理导入校验报告。"""

    baseline_dir = repo_root / "outputs" / "external_baseline_results"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    row = baseline_candidate_row()
    (baseline_dir / "baseline_result_records.jsonl").write_text(json_line(row), encoding="utf-8")
    validation_report = {
        "protocol_name": "primary_baseline_formal_import_protocol",
        "target_fpr": 0.01,
        "accepted_records": [row] if accepted else [],
        "formal_import_validation_ready": accepted,
        "supports_paper_claim": False,
    }
    (baseline_dir / "baseline_result_candidate_validation_report.json").write_text(
        json.dumps(validation_report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def append_diagnostic_records_outside_template(repo_root: Path) -> None:
    """追加不属于 pilot_paper 17 类攻击模板的诊断记录。"""

    metrics_path = repo_root / "outputs" / "attack_matrix" / "attack_family_metrics.csv"
    rows = list(csv.DictReader(metrics_path.open(encoding="utf-8")))
    fieldnames = list(rows[0].keys())
    probe_row = dict(rows[0])
    probe_row.update(
        {
            "resource_profile": "probe",
            "metric_status": "measured_from_local_proxy",
            "supports_paper_claim": "False",
        }
    )
    write_csv_rows(metrics_path, rows + [probe_row], fieldnames)

    baseline_dir = repo_root / "outputs" / "external_baseline_results"
    baseline_rows = [
        json.loads(line)
        for line in (baseline_dir / "baseline_result_records.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    clean_row = dict(baseline_rows[0])
    clean_row.update(
        {
            "attack_family": "clean",
            "attack_name": "clean_none",
            "resource_profile": "full_main",
            "baseline_result_record_id": "primary_baseline_formal_result_clean_none",
            "baseline_result_digest": "baseline_result_digest_clean_none",
        }
    )
    all_rows = baseline_rows + [clean_row]
    (baseline_dir / "baseline_result_records.jsonl").write_text(
        "".join(json_line(row) for row in all_rows),
        encoding="utf-8",
    )
    validation_report = json.loads(
        (baseline_dir / "baseline_result_candidate_validation_report.json").read_text(encoding="utf-8")
    )
    validation_report["accepted_records"] = all_rows
    (baseline_dir / "baseline_result_candidate_validation_report.json").write_text(
        json.dumps(validation_report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


@pytest.mark.quick
def test_pilot_paper_result_writer_materializes_slm_and_governed_baseline_records(tmp_path: Path) -> None:
    """结果物化脚本应把方法主流程与已接受 baseline 候选统一转换为 pilot_paper 记录。"""

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    write_pilot_prompt_file(repo_root)
    write_attack_matrix_inputs(repo_root)
    write_baseline_inputs(repo_root, accepted=True)

    manifest = write_pilot_paper_result_record_outputs(root=repo_root, require_existing_evidence=True)
    output_dir = repo_root / "outputs" / "pilot_paper_fixed_fpr_results"
    records = [
        json.loads(line)
        for line in (output_dir / "pilot_paper_result_records.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    validation = json.loads(
        (output_dir / "pilot_paper_result_import_validation_report.json").read_text(encoding="utf-8")
    )
    coverage_rows = list(csv.DictReader((output_dir / "pilot_paper_result_template_coverage.csv").open(encoding="utf-8")))
    summary = json.loads((output_dir / "pilot_paper_result_record_summary.json").read_text(encoding="utf-8"))

    assert manifest["artifact_id"] == "pilot_paper_fixed_fpr_result_records_manifest"
    assert {row["method_id"] for row in records} == {"tree_ring"}
    assert all(row["result_protocol_name"] == "pilot_paper_fixed_fpr_common_protocol" for row in records)
    assert all(row["target_fpr"] == 0.01 for row in records)
    assert all(row["paper_claim_scale"] == "pilot_paper" for row in records)
    assert all(row["evidence_paths"] for row in records)
    baseline_record = next(row for row in records if row["method_id"] == "tree_ring")
    assert baseline_record["detector_input_access_mode"] == "method_native_or_final_image"
    assert validation["pilot_paper_result_import_ready"] is True
    assert validation["accepted_pilot_paper_import_count"] == 1
    assert validation["accepted_pilot_paper_claim_record_count"] == 1
    assert summary["pilot_paper_result_record_count"] == 1
    assert summary["pilot_paper_template_coverage_ready"] is False
    assert summary["pilot_paper_template_missing_count"] > 0
    assert any(row["template_covered"] == "True" for row in coverage_rows)
    assert all(path.startswith("outputs/") for path in manifest["output_paths"])


@pytest.mark.quick
def test_pilot_paper_result_writer_excludes_diagnostic_records_outside_template(tmp_path: Path) -> None:
    """clean_none 与 probe 诊断记录不应进入 pilot_paper claim 比较集合。"""

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    write_pilot_prompt_file(repo_root)
    write_attack_matrix_inputs(repo_root)
    write_baseline_inputs(repo_root, accepted=True)
    append_diagnostic_records_outside_template(repo_root)

    write_pilot_paper_result_record_outputs(root=repo_root, require_existing_evidence=True)
    output_dir = repo_root / "outputs" / "pilot_paper_fixed_fpr_results"
    records = [
        json.loads(line)
        for line in (output_dir / "pilot_paper_result_records.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    validation = json.loads(
        (output_dir / "pilot_paper_result_import_validation_report.json").read_text(encoding="utf-8")
    )

    assert len(records) == 1
    assert {row["method_id"] for row in records} == {"tree_ring"}
    assert {row["attack_name"] for row in records} == {"jpeg_compression"}
    assert {row["resource_profile"] for row in records} == {"full_main"}
    assert all(row["supports_paper_claim"] is True for row in records)
    assert validation["accepted_pilot_paper_import_count"] == 1
    assert validation["accepted_pilot_paper_claim_record_count"] == 1


@pytest.mark.quick
def test_result_writer_switches_records_to_full_paper_claim_scale(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """结果记录写出脚本应随论文运行配置切换协议名和主张层级。"""

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    write_full_prompt_file(repo_root)
    write_attack_matrix_inputs(repo_root)
    write_baseline_inputs(repo_root, accepted=True)
    monkeypatch.setenv("SLM_WM_PAPER_RUN_NAME", "full_paper")

    write_pilot_paper_result_record_outputs(root=repo_root, require_existing_evidence=True)
    output_dir = repo_root / "outputs" / "pilot_paper_fixed_fpr_results"
    records = [
        json.loads(line)
        for line in (output_dir / "pilot_paper_result_records.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    summary = json.loads((output_dir / "pilot_paper_result_record_summary.json").read_text(encoding="utf-8"))

    assert {row["method_id"] for row in records} == {"tree_ring"}
    assert all(row["result_protocol_name"] == "full_paper_fixed_fpr_common_protocol" for row in records)
    assert all(row["result_claim_scope"] == "full_claim" for row in records)
    assert all(row["prompt_protocol_name"] == "paper_main_full_paper_prompt_protocol" for row in records)
    assert all(row["paper_claim_scale"] == "full_paper" for row in records)
    assert summary["paper_claim_scale"] == "full_paper"


@pytest.mark.quick
def test_result_writer_blocks_retention_proxy_from_claim_boundary(tmp_path: Path) -> None:
    """攻击后检测仅为 retention proxy 时, 结果记录不得支撑论文主张。"""

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    write_pilot_prompt_file(repo_root)
    write_attack_matrix_inputs(repo_root)
    write_baseline_inputs(repo_root, accepted=True)
    metrics_path = repo_root / "outputs" / "attack_matrix" / "attack_family_metrics.csv"
    text = metrics_path.read_text(encoding="utf-8")
    metrics_path.write_text(
        text.replace(
            "measured_from_real_attacked_image_watermark_rescore_formal_protocol",
            "measured_from_real_attacked_image_retention_proxy_formal_protocol",
        ),
        encoding="utf-8",
    )

    write_pilot_paper_result_record_outputs(root=repo_root, require_existing_evidence=True)
    output_dir = repo_root / "outputs" / "pilot_paper_fixed_fpr_results"
    records = [
        json.loads(line)
        for line in (output_dir / "pilot_paper_result_records.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    baseline_record = next(row for row in records if row["method_id"] == "tree_ring")

    assert {row["method_id"] for row in records} == {"tree_ring"}
    assert baseline_record["supports_paper_claim"] is True


@pytest.mark.quick
def test_result_writer_accepts_image_only_method_records(tmp_path: Path) -> None:
    """仅图像检测结果应替代生成轨迹检测记录进入方法主张边界。"""

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    write_pilot_prompt_file(repo_root)
    write_attack_matrix_inputs(repo_root)
    write_baseline_inputs(repo_root, accepted=False)
    runtime_dir = repo_root / "outputs" / "image_only_dataset_runtime" / "pilot_paper"
    runtime_dir.mkdir(parents=True)
    metric_rows = [
        {
            "attack_family": "clean",
            "attack_name": "none",
            "resource_profile": "clean",
            "sample_role": "clean_negative",
            "record_count": 340,
            "positive_count": 0,
            "positive_rate": 0.0,
            "positive_rate_upper_95": 0.009,
            "content_score_mean": 0.01,
            "fixed_fpr_upper_bound_ready": True,
        },
        {
            "attack_family": "clean",
            "attack_name": "none",
            "resource_profile": "clean",
            "sample_role": "positive_source",
            "record_count": 340,
            "positive_count": 330,
            "positive_rate": 330 / 340,
            "positive_rate_upper_95": 0.99,
            "content_score_mean": 0.80,
            "fixed_fpr_upper_bound_ready": False,
        },
        {
            "attack_family": "standard_distortion",
            "attack_name": "jpeg_compression",
            "resource_profile": "full_main",
            "sample_role": "clean_negative",
            "record_count": 340,
            "positive_count": 1,
            "positive_rate": 1 / 340,
            "positive_rate_upper_95": 0.02,
            "content_score_mean": 0.02,
            "fixed_fpr_upper_bound_ready": False,
        },
        {
            "attack_family": "standard_distortion",
            "attack_name": "jpeg_compression",
            "resource_profile": "full_main",
            "sample_role": "positive_source",
            "record_count": 340,
            "positive_count": 300,
            "positive_rate": 300 / 340,
            "positive_rate_upper_95": 0.92,
            "content_score_mean": 0.60,
            "fixed_fpr_upper_bound_ready": False,
        },
    ]
    metrics_path = runtime_dir / "test_detection_metrics.csv"
    with metrics_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metric_rows[0]))
        writer.writeheader()
        writer.writerows(metric_rows)
    (runtime_dir / "dataset_runtime_summary.json").write_text(
        json.dumps(
            {
                "protocol_decision": "pass",
                "supports_paper_claim": True,
                "clean_test_fixed_fpr_upper_bound_ready": True,
                "wrong_key_test_fixed_fpr_upper_bound_ready": True,
                "scientific_operator_gate_ready": True,
                "attack_record_coverage_ready": True,
                "attacked_image_evidence_chain_ready": True,
                "real_gpu_attack_validation_ready": True,
                "paired_ssim_mean": 0.99,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (runtime_dir / "image_only_detection_records.jsonl").write_text("{}\n", encoding="utf-8")
    (runtime_dir / "manifest.local.json").write_text("{}\n", encoding="utf-8")

    write_pilot_paper_result_record_outputs(root=repo_root, require_existing_evidence=True)
    records = [
        json.loads(line)
        for line in (
            repo_root / "outputs" / "pilot_paper_fixed_fpr_results" / "pilot_paper_result_records.jsonl"
        ).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert {record["method_id"] for record in records} == {"slm_wm_current"}
    assert records[0]["detector_input_access_mode"] == "image_key_public_model_only"
    assert records[0]["blind_image_detector"] is True
    assert records[0]["generation_latent_trace_required"] is False
    assert records[0]["supports_paper_claim"] is True


@pytest.mark.quick
def test_pilot_paper_result_writer_keeps_unaccepted_baseline_out_of_claim_boundary(tmp_path: Path) -> None:
    """baseline 候选未被受治理导入报告接受时, 转换后的记录不得支撑 pilot_paper 主张。"""

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    write_pilot_prompt_file(repo_root)
    write_attack_matrix_inputs(repo_root)
    write_baseline_inputs(repo_root, accepted=False)

    write_pilot_paper_result_record_outputs(root=repo_root, require_existing_evidence=True)
    output_dir = repo_root / "outputs" / "pilot_paper_fixed_fpr_results"
    records = [
        json.loads(line)
        for line in (output_dir / "pilot_paper_result_records.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    validation = json.loads(
        (output_dir / "pilot_paper_result_import_validation_report.json").read_text(encoding="utf-8")
    )
    assert records == []
    assert validation["pilot_paper_result_import_ready"] is False
    assert validation["accepted_pilot_paper_claim_record_count"] == 0
    assert validation["pilot_paper_claim_record_ready"] is False


@pytest.mark.quick
def test_pilot_paper_package_materialization_only_extracts_outputs_entries(tmp_path: Path) -> None:
    """Drive 结果包物化只允许释放 outputs/ 下的受治理条目。"""

    package_path = tmp_path / "drive" / "pilot_package.zip"
    package_path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(package_path, "w") as archive:
        archive.writestr("outputs/upstream/result.json", '{"ready": true}\n')
        archive.writestr("metadata/result.json", '{"ignored": true}\n')
        archive.writestr("outputs/../escape.json", '{"blocked": true}\n')

    report = materialize_output_entries(tmp_path, (package_path,))

    assert (tmp_path / "outputs" / "upstream" / "result.json").is_file()
    assert not (tmp_path / "escape.json").exists()
    assert report["input_package_count"] == 1
    assert report["materialized_output_entry_count"] == 1
    assert report["skipped_output_entry_count"] == 2
    assert "metadata/result.json" in report["skipped_output_entries"]
    assert "outputs/../escape.json" in report["skipped_output_entries"]


@pytest.mark.quick
def test_pilot_paper_writer_materialize_only_does_not_require_prompt_config(tmp_path: Path) -> None:
    """仅物化模式应可在重建前先释放 Google Drive 结果包中的 outputs 条目。"""

    package_path = tmp_path / "drive" / "pilot_package.zip"
    package_path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(package_path, "w") as archive:
        archive.writestr("outputs/upstream/result.json", '{"ready": true}\n')

    manifest = write_pilot_paper_result_record_outputs(
        root=tmp_path,
        package_paths=(package_path,),
        materialize_only=True,
    )
    output_dir = tmp_path / "outputs" / "pilot_paper_fixed_fpr_results"
    report = json.loads((output_dir / "pilot_paper_materialization_report.json").read_text(encoding="utf-8"))

    assert manifest["artifact_id"] == "pilot_paper_result_record_materialization_manifest"
    assert report["materialized_output_entry_count"] == 1
    assert (tmp_path / "outputs" / "upstream" / "result.json").is_file()
    assert not (output_dir / "pilot_paper_result_records.jsonl").exists()


@pytest.mark.quick
def test_pilot_paper_result_writer_rejects_output_dir_outside_outputs(tmp_path: Path) -> None:
    """结果物化脚本不得把持久化文件写到 outputs/ 之外。"""

    with pytest.raises(ValueError, match="outputs/"):
        ensure_output_dir_under_outputs(tmp_path, "reports/pilot_paper")
