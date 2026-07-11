"""当前论文运行层级 fixed-FPR 结果记录物化层的轻量功能测试。"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from zipfile import ZipFile

import pytest

from experiments.protocol.attacks import attack_config_digest, resolve_formal_attack_config
from experiments.protocol.pilot_paper_fixed_fpr import (
    bounded_hoeffding_confidence_interval,
)
from scripts.write_pilot_paper_result_records import (
    attach_metric_fields,
    materialize_output_entries,
    write_pilot_paper_result_record_outputs,
)
from tests.helpers.closure_input_lock import write_test_closure_input_lock


pytestmark = pytest.mark.quick


PAPER_SCALE = {
    "probe_paper": {"prompt_count": 70, "test_count": 34, "target_fpr": 0.1},
    "pilot_paper": {"prompt_count": 700, "test_count": 340, "target_fpr": 0.01},
    "full_paper": {"prompt_count": 7000, "test_count": 3400, "target_fpr": 0.001},
}
PRIMARY_BASELINE_IDS = ("tree_ring", "gaussian_shading", "shallow_diffuse", "t2smark")
BASELINE_THRESHOLD_DIGESTS = {
    baseline_id: f"{index + 2:x}" * 64
    for index, baseline_id in enumerate(PRIMARY_BASELINE_IDS)
}


def test_result_metric_writer_preserves_negative_ssim_and_signed_ci() -> None:
    """负 SSIM 必须原样保留, 且使用范围宽度为2的 Hoeffding 区间."""

    payload = attach_metric_fields(
        {},
        positive_count=34,
        negative_count=34,
        attack_record_count=68,
        supported_record_count=34,
        true_positive_rate=0.8,
        false_positive_rate=0.05,
        clean_false_positive_rate=0.05,
        attacked_false_positive_rate=0.05,
        quality_score_mean=-0.25,
        score_retention_mean=0.7,
        confidence_level=0.95,
    )
    expected_quality_ci = bounded_hoeffding_confidence_interval(
        -0.25,
        34,
        0.95,
        lower_bound=-1.0,
        upper_bound=1.0,
    )
    expected_retention_ci = bounded_hoeffding_confidence_interval(
        0.7,
        34,
        0.95,
    )

    assert payload["quality_score_mean"] == -0.25
    assert (
        payload["quality_score_ci_low"],
        payload["quality_score_ci_high"],
    ) == pytest.approx(expected_quality_ci)
    assert (
        payload["score_retention_ci_low"],
        payload["score_retention_ci_high"],
    ) == pytest.approx(expected_retention_ci)


def json_line(value: dict[str, object]) -> str:
    """将测试记录转换为 JSONL 单行。"""

    return json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n"


def write_csv_rows(path: Path, rows: list[dict[str, object]]) -> None:
    """写出字段一致的测试 CSV 表格。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_prompt_file(repo_root: Path, paper_claim_scale: str) -> None:
    """写出当前运行层级要求的完整 Prompt 集。"""

    prompt_count = int(PAPER_SCALE[paper_claim_scale]["prompt_count"])
    prompt_path = repo_root / "configs" / f"paper_main_{paper_claim_scale}_prompts.txt"
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_path.write_text(
        "\n".join(f"a governed {paper_claim_scale} prompt {index}" for index in range(prompt_count)) + "\n",
        encoding="utf-8",
    )


def write_dataset_quality_inputs(repo_root: Path, paper_claim_scale: str) -> None:
    """写出正式 Inception FID/KID 质量门禁输入。"""

    quality_dir = repo_root / "outputs" / "dataset_level_quality" / paper_claim_scale
    write_csv_rows(
        quality_dir / "dataset_quality_metrics.csv",
        [
            {"quality_metric_name": "fid", "metric_status": "measured"},
            {"quality_metric_name": "kid", "metric_status": "measured"},
        ],
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


def write_image_only_runtime_inputs(repo_root: Path, paper_claim_scale: str) -> None:
    """写出仅图像盲检数据集运行的最小正式证据链。"""

    test_count = int(PAPER_SCALE[paper_claim_scale]["test_count"])
    runtime_dir = repo_root / "outputs" / "image_only_dataset_runtime" / paper_claim_scale
    runtime_dir.mkdir(parents=True, exist_ok=True)
    attack_config = resolve_formal_attack_config(
        attack_family="standard_distortion",
        attack_name="jpeg_compression",
    )
    metric_rows = [
        {
            "attack_family": "clean",
            "attack_name": "none",
            "resource_profile": "clean",
            "sample_role": "clean_negative",
            "record_count": test_count,
            "positive_count": 0,
            "positive_rate": 0.0,
            "positive_rate_upper_95": float(PAPER_SCALE[paper_claim_scale]["target_fpr"]) * 0.9,
            "content_score_mean": 0.01,
            "source_to_evaluated_ssim_mean": 1.0,
            "fixed_fpr_upper_bound_ready": True,
        },
        {
            "attack_family": "clean",
            "attack_name": "none",
            "resource_profile": "clean",
            "sample_role": "positive_source",
            "record_count": test_count,
            "positive_count": test_count - 1,
            "positive_rate": (test_count - 1) / test_count,
            "positive_rate_upper_95": 1.0,
            "content_score_mean": 0.80,
            "source_to_evaluated_ssim_mean": 1.0,
            "fixed_fpr_upper_bound_ready": False,
        },
        {
            "attack_family": "standard_distortion",
            "attack_name": "jpeg_compression",
            "resource_profile": "full_main",
            "sample_role": "clean_negative",
            "record_count": test_count,
            "positive_count": 0,
            "positive_rate": 0.0,
            "positive_rate_upper_95": float(PAPER_SCALE[paper_claim_scale]["target_fpr"]) * 0.9,
            "content_score_mean": 0.02,
            "source_to_evaluated_ssim_mean": 0.95,
            "fixed_fpr_upper_bound_ready": True,
        },
        {
            "attack_family": "standard_distortion",
            "attack_name": "jpeg_compression",
            "resource_profile": "full_main",
            "sample_role": "positive_source",
            "record_count": test_count,
            "positive_count": test_count - 2,
            "positive_rate": (test_count - 2) / test_count,
            "positive_rate_upper_95": 1.0,
            "content_score_mean": 0.60,
            "source_to_evaluated_ssim_mean": 0.88,
            "fixed_fpr_upper_bound_ready": False,
        },
    ]
    write_csv_rows(runtime_dir / "test_detection_metrics.csv", metric_rows)
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
                "frozen_threshold_digest": "1" * 64,
                "paired_ssim_mean": 0.99,
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    (runtime_dir / "image_only_detection_records.jsonl").write_text(
        json_line(
            {
                "detector_input_access_mode": "image_key_public_model_only",
                "sample_role": "positive_source",
                "attack_id": attack_config.attack_id,
                "attack_family": attack_config.attack_family,
                "attack_name": "jpeg_compression",
                "resource_profile": attack_config.resource_profile,
                "attack_config_digest": attack_config_digest(attack_config),
                "evidence_decision": True,
                "supports_paper_claim": True,
            }
        ),
        encoding="utf-8",
    )
    (runtime_dir / "manifest.local.json").write_text(
        json.dumps(
            {
                "artifact_id": f"{paper_claim_scale}_image_only_dataset_runtime_manifest",
                "artifact_type": "local_manifest",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


def baseline_candidate_row(
    paper_claim_scale: str,
    baseline_id: str = "tree_ring",
) -> dict[str, object]:
    """构造已经通过外部 baseline 受治理导入报告接受的候选记录。"""

    test_count = int(PAPER_SCALE[paper_claim_scale]["test_count"])
    target_fpr = float(PAPER_SCALE[paper_claim_scale]["target_fpr"])
    attack_config = resolve_formal_attack_config(
        attack_family="standard_distortion",
        attack_name="jpeg_compression",
    )
    return {
        "baseline_id": baseline_id,
        "attack_id": attack_config.attack_id,
        "attack_family": "standard_distortion",
        "attack_name": "jpeg_compression",
        "resource_profile": "full_main",
        "attack_config_digest": attack_config_digest(attack_config),
        "comparable_operating_point": f"fixed_fpr_{target_fpr}",
        "result_protocol_name": "primary_baseline_formal_import_protocol",
        "result_source_type": "governed_import",
        "baseline_result_source": (
            f"outputs/external_baseline_results/{paper_claim_scale}/baseline_result_records.jsonl"
        ),
        "baseline_result_source_digest": "baseline_source_digest",
        "metric_status": "measured",
        "prompt_protocol_name": f"paper_main_{paper_claim_scale}_prompt_protocol",
        "prompt_protocol_digest": f"{paper_claim_scale}_prompt_digest",
        "adapter_boundary": "method_faithful_sd35_adapter_reproduction",
        "evidence_paths": [
            f"outputs/external_baseline_results/{paper_claim_scale}/baseline_result_records.jsonl"
        ],
        "method_faithful_adapter_ready": True,
        "paper_run_prompt_protocol_ready": True,
        "fixed_fpr_baseline_calibration_ready": True,
        "attack_matrix_baseline_detection_ready": True,
        "formal_evidence_paths_ready": True,
        "threshold_digest": BASELINE_THRESHOLD_DIGESTS[baseline_id],
        "positive_count": test_count,
        "negative_count": test_count,
        "attacked_negative_count": test_count,
        "attack_record_count": 2 * test_count,
        "supported_record_count": test_count,
        "true_positive_rate": 0.6,
        "false_positive_rate": 0.0,
        "clean_false_positive_rate": 0.0,
        "attacked_false_positive_rate": 0.0,
        "quality_score_mean": 0.88,
        "score_retention_mean": 0.70,
        "baseline_result_record_id": "primary_baseline_formal_result_test",
        "baseline_result_digest": "baseline_result_digest",
        "supports_paper_claim": True,
    }


def write_baseline_inputs(repo_root: Path, paper_claim_scale: str, *, accepted: bool) -> None:
    """写出外部 baseline 候选记录和受治理导入校验报告。"""

    baseline_dir = repo_root / "outputs" / "external_baseline_results" / paper_claim_scale
    baseline_dir.mkdir(parents=True, exist_ok=True)
    rows = [baseline_candidate_row(paper_claim_scale, baseline_id) for baseline_id in PRIMARY_BASELINE_IDS]
    (baseline_dir / "baseline_result_records.jsonl").write_text(
        "".join(json_line(row) for row in rows),
        encoding="utf-8",
    )
    validation_report = {
        "protocol_name": "primary_baseline_formal_import_protocol",
        "target_fpr": PAPER_SCALE[paper_claim_scale]["target_fpr"],
        "accepted_records": rows if accepted else [],
        "formal_import_validation_ready": accepted,
        "supports_paper_claim": accepted,
    }
    (baseline_dir / "baseline_result_candidate_validation_report.json").write_text(
        json.dumps(validation_report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_formal_inputs(repo_root: Path, paper_claim_scale: str, *, baseline_accepted: bool) -> None:
    """写出当前运行层级结果闭合所需的全部正式输入。"""

    write_prompt_file(repo_root, paper_claim_scale)
    write_dataset_quality_inputs(repo_root, paper_claim_scale)
    write_image_only_runtime_inputs(repo_root, paper_claim_scale)
    write_baseline_inputs(repo_root, paper_claim_scale, accepted=baseline_accepted)
    write_test_closure_input_lock(
        repo_root,
        paper_run_name=paper_claim_scale,
        target_fpr=float(PAPER_SCALE[paper_claim_scale]["target_fpr"]),
    )


def read_result_records(repo_root: Path, paper_claim_scale: str = "pilot_paper") -> list[dict[str, object]]:
    """读取结果物化脚本写出的共同协议记录。"""

    path = (
        repo_root
        / "outputs"
        / "pilot_paper_fixed_fpr_results"
        / paper_claim_scale
        / "pilot_paper_result_records.jsonl"
    )
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def append_clean_baseline_record(repo_root: Path) -> None:
    """追加不属于正式攻击比较模板的 clean baseline 记录。"""

    baseline_dir = repo_root / "outputs" / "external_baseline_results" / "pilot_paper"
    records_path = baseline_dir / "baseline_result_records.jsonl"
    rows = [json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    clean_row = dict(rows[0])
    clean_row.update(
        {
            "attack_family": "clean",
            "attack_name": "clean_none",
            "baseline_result_record_id": "primary_baseline_formal_result_clean_none",
            "baseline_result_digest": "baseline_result_digest_clean_none",
        }
    )
    all_rows = rows + [clean_row]
    records_path.write_text("".join(json_line(row) for row in all_rows), encoding="utf-8")
    report_path = baseline_dir / "baseline_result_candidate_validation_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["accepted_records"] = all_rows
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def test_materialization_rejects_cross_run_path_overwrite(tmp_path: Path) -> None:
    """两个结果包对同一 outputs 路径给出不同内容时必须停止闭合。"""

    packages = []
    for index, content in enumerate(("run-a", "run-b")):
        package = tmp_path / f"package-{index}.zip"
        with ZipFile(package, "w") as archive:
            archive.writestr("outputs/shared/result.json", content)
        packages.append(package)

    with pytest.raises(RuntimeError, match="跨运行覆盖"):
        materialize_output_entries(tmp_path, packages)


def test_result_writer_rejects_missing_explicit_package(tmp_path: Path) -> None:
    """显式结果包不存在时必须停止物化。"""

    with pytest.raises(FileNotFoundError, match="显式论文结果包不存在"):
        write_pilot_paper_result_record_outputs(
            root=tmp_path,
            package_paths=(tmp_path / "missing.zip",),
            materialize_only=True,
        )


def test_result_writer_materializes_method_and_governed_baseline_records(tmp_path: Path) -> None:
    """结果物化脚本应统一转换仅图像盲检方法记录与正式 baseline 记录。"""

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    write_formal_inputs(repo_root, "pilot_paper", baseline_accepted=True)

    manifest = write_pilot_paper_result_record_outputs(root=repo_root, require_existing_evidence=True)
    output_dir = repo_root / "outputs" / "pilot_paper_fixed_fpr_results" / "pilot_paper"
    records = read_result_records(repo_root)
    validation = json.loads((output_dir / "pilot_paper_result_import_validation_report.json").read_text(encoding="utf-8"))
    coverage_rows = list(csv.DictReader((output_dir / "pilot_paper_result_template_coverage.csv").open(encoding="utf-8")))
    summary = json.loads((output_dir / "pilot_paper_result_record_summary.json").read_text(encoding="utf-8"))

    assert manifest["artifact_id"] == "pilot_paper_fixed_fpr_result_records_manifest"
    assert {row["method_id"] for row in records} == {"slm_wm_current", *PRIMARY_BASELINE_IDS}
    assert all(row["result_protocol_name"] == "pilot_paper_fixed_fpr_common_protocol" for row in records)
    assert all(row["target_fpr"] == 0.01 for row in records)
    assert all(row["positive_count"] == 340 for row in records)
    assert all(row["negative_count"] == 340 for row in records)
    assert all(row["supports_paper_claim"] is True for row in records)
    assert all(row["attack_id"] == "jpeg_compression_main" for row in records)
    assert all(len(row["attack_config_digest"]) == 64 for row in records)
    method_record = next(row for row in records if row["method_id"] == "slm_wm_current")
    assert method_record["detector_input_access_mode"] == "image_key_public_model_only"
    assert method_record["blind_image_detector"] is True
    assert method_record["generation_latent_trace_required"] is False
    assert validation["pilot_paper_result_import_ready"] is True
    assert validation["accepted_pilot_paper_import_count"] == 5
    assert validation["accepted_pilot_paper_claim_record_count"] == 5
    assert summary["pilot_paper_result_record_count"] == 5
    assert len(summary["method_threshold_digest_map"]) == 5
    assert len(summary["result_record_set_digest"]) == 64
    assert summary["common_code_version"] == "abc1234"
    assert summary["pilot_paper_template_coverage_ready"] is False
    assert any(row["template_covered"] == "True" for row in coverage_rows)
    assert all(path.startswith("outputs/") for path in manifest["output_paths"])


def test_result_writer_rejects_records_outside_attack_template(tmp_path: Path) -> None:
    """clean baseline 记录不属于正式攻击模板时必须停止物化。"""

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    write_formal_inputs(repo_root, "pilot_paper", baseline_accepted=True)
    append_clean_baseline_record(repo_root)

    with pytest.raises(ValueError, match="不属于当前 method × attack 模板"):
        write_pilot_paper_result_record_outputs(root=repo_root, require_existing_evidence=True)


def test_result_writer_switches_records_to_full_paper_claim_scale(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """结果记录应只切换统计规模与 fixed-FPR 层级, 不切换方法机制。"""

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    write_formal_inputs(repo_root, "full_paper", baseline_accepted=True)
    monkeypatch.setenv("SLM_WM_PAPER_RUN_NAME", "full_paper")

    write_pilot_paper_result_record_outputs(root=repo_root, require_existing_evidence=True)
    records = read_result_records(repo_root, "full_paper")

    assert {row["method_id"] for row in records} == {"slm_wm_current", *PRIMARY_BASELINE_IDS}
    assert all(row["result_protocol_name"] == "full_paper_fixed_fpr_common_protocol" for row in records)
    assert all(row["result_claim_scope"] == "full_claim" for row in records)
    assert all(row["prompt_protocol_name"] == "paper_main_full_paper_prompt_protocol" for row in records)
    assert all(row["paper_claim_scale"] == "full_paper" for row in records)
    assert all(row["positive_count"] == 3400 for row in records)
    assert all(row["negative_count"] == 3400 for row in records)


def test_result_writer_rejects_missing_formal_runtime_inputs(tmp_path: Path) -> None:
    """缺少仅图像盲检正式输入时必须失败, 不得回退到其他结果路径。"""

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    write_prompt_file(repo_root, "pilot_paper")
    write_dataset_quality_inputs(repo_root, "pilot_paper")
    write_baseline_inputs(repo_root, "pilot_paper", accepted=True)
    write_test_closure_input_lock(
        repo_root,
        paper_run_name="pilot_paper",
        target_fpr=0.01,
    )

    with pytest.raises(FileNotFoundError, match="论文结果记录缺少正式输入"):
        write_pilot_paper_result_record_outputs(root=repo_root, require_existing_evidence=True)


def test_result_writer_rejects_unaccepted_baseline_record(tmp_path: Path) -> None:
    """任一 baseline 未被受治理报告接受时必须停止正式结果物化。"""

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    write_formal_inputs(repo_root, "pilot_paper", baseline_accepted=False)

    with pytest.raises(ValueError, match="未通过严格正式门禁"):
        write_pilot_paper_result_record_outputs(root=repo_root, require_existing_evidence=True)


def test_package_materialization_only_extracts_outputs_entries(tmp_path: Path) -> None:
    """结果包物化只允许释放 outputs/ 下的受治理条目。"""

    package_path = tmp_path / "drive" / "paper_package.zip"
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


def test_writer_materialize_only_does_not_require_prompt_config(tmp_path: Path) -> None:
    """仅物化模式应可在重建前先释放结果包中的 outputs 条目。"""

    package_path = tmp_path / "drive" / "paper_package.zip"
    package_path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(package_path, "w") as archive:
        archive.writestr("outputs/upstream/result.json", '{"ready": true}\n')

    manifest = write_pilot_paper_result_record_outputs(
        root=tmp_path,
        package_paths=(package_path,),
        materialize_only=True,
    )
    output_dir = tmp_path / "outputs" / "pilot_paper_fixed_fpr_results" / "pilot_paper"
    report = json.loads((output_dir / "pilot_paper_materialization_report.json").read_text(encoding="utf-8"))

    assert manifest["artifact_id"] == "pilot_paper_result_record_materialization_manifest"
    assert report["materialized_output_entry_count"] == 1
    assert (tmp_path / "outputs" / "upstream" / "result.json").is_file()
    assert not (output_dir / "pilot_paper_result_records.jsonl").exists()
