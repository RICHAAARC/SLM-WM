"""主表 external baseline 候选结果导入脚本的轻量功能测试。"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from zipfile import ZipFile

import pytest

from scripts.write_primary_baseline_result_candidates import (
    GPU_SMOKE_OBSERVATIONS_ENTRY,
    T2SMARK_CANDIDATE_RECORDS_ENTRY,
    write_primary_baseline_result_candidate_outputs,
)


def method_observations() -> list[dict[str, object]]:
    """构造覆盖三个主表 baseline 的最小 observation 集合。"""

    rows: list[dict[str, object]] = []
    for baseline_id in ("tree_ring", "gaussian_shading", "shallow_diffuse"):
        rows.extend(
            [
                {
                    "baseline_id": baseline_id,
                    "attack_family": "clean",
                    "attack_condition": "clean_none",
                    "sample_role": "clean_negative",
                    "detection_decision": False,
                    "prompt_id": "prompt_000",
                    "prompt_text": "a ceramic fox on a wooden desk",
                    "quality_score_proxy": 1.0,
                    "score_retention_proxy": 1.0,
                },
                {
                    "baseline_id": baseline_id,
                    "attack_family": "clean",
                    "attack_condition": "clean_none",
                    "sample_role": "positive_source",
                    "detection_decision": True,
                    "prompt_id": "prompt_000",
                    "prompt_text": "a ceramic fox on a wooden desk",
                    "quality_score_proxy": 1.0,
                    "score_retention_proxy": 1.0,
                },
            ]
        )
    return rows


def t2smark_smoke_observations() -> list[dict[str, object]]:
    """构造 T2SMark GPU smoke adapter 的最小 observation 集合。"""

    return [
        {
            "baseline_id": "t2smark",
            "attack_family": "clean",
            "attack_condition": "clean_none",
            "sample_role": "clean_negative",
            "detection_decision": False,
            "prompt_id": "prompt_000",
            "prompt_text": "a ceramic fox on a wooden desk",
        },
        {
            "baseline_id": "t2smark",
            "attack_family": "clean",
            "attack_condition": "clean_none",
            "sample_role": "positive_source",
            "detection_decision": True,
            "prompt_id": "prompt_000",
            "prompt_text": "a ceramic fox on a wooden desk",
        },
    ]


def t2smark_candidate_record() -> dict[str, object]:
    """构造一个尚未完成共同协议闭合的 T2SMark 候选记录。"""

    return {
        "adapter_boundary": "sd35_medium_native_official_reproduction",
        "attack_family": "clean",
        "attack_matrix_baseline_detection_ready": False,
        "attack_name": "clean_none",
        "attack_record_count": 10,
        "attacked_false_positive_rate": 0.0,
        "baseline_id": "t2smark",
        "baseline_result_source": "outputs/t2smark_full_main_reproduction/results.json",
        "baseline_result_source_digest": "source_digest",
        "clean_false_positive_rate": 0.0,
        "comparable_operating_point": "fixed_fpr_0.01",
        "evidence_paths": ["outputs/t2smark_full_main_reproduction/results.json"],
        "false_positive_rate": 0.0,
        "fixed_fpr_baseline_calibration_ready": False,
        "formal_evidence_paths_ready": True,
        "full_main_prompt_protocol_ready": False,
        "method_faithful_adapter_ready": True,
        "metric_status": "measured",
        "negative_count": 5,
        "positive_count": 5,
        "prompt_protocol_digest": "prompt_digest",
        "prompt_protocol_name": "paper_main_pilot_paper_prompt_protocol",
        "quality_score_proxy_mean": 1.0,
        "resource_profile": "full_main",
        "result_protocol_name": "primary_baseline_formal_import_protocol",
        "result_source_type": "official_reproduction",
        "score_retention_mean": 1.0,
        "supported_record_count": 10,
        "supports_paper_claim": False,
        "true_positive_rate": 1.0,
    }


def write_text_package(path: Path, entries: dict[str, str]) -> None:
    """写出测试用 zip 结果包。"""

    with ZipFile(path, "w") as archive:
        for entry_name, text in entries.items():
            archive.writestr(entry_name, text)


@pytest.mark.quick
def test_primary_baseline_candidate_writer_imports_packages_without_promoting_smoke_results(tmp_path: Path) -> None:
    """候选导入脚本应保留候选证据, 但不能把小样本 GPU 链路升级为主表结论。"""

    attack_dir = tmp_path / "outputs" / "attack_matrix"
    attack_dir.mkdir(parents=True)
    (attack_dir / "attack_manifest.json").write_text(
        json.dumps({"evaluation_boundary": {"target_fpr": 0.01}}, ensure_ascii=False),
        encoding="utf-8",
    )
    smoke_package_path = tmp_path / "external_baseline_gpu_smoke_package.zip"
    t2smark_package_path = tmp_path / "t2smark_full_main_reproduction_package.zip"
    write_text_package(
        smoke_package_path,
        {GPU_SMOKE_OBSERVATIONS_ENTRY: json.dumps(method_observations(), ensure_ascii=False)},
    )
    write_text_package(
        t2smark_package_path,
        {T2SMARK_CANDIDATE_RECORDS_ENTRY: json.dumps(t2smark_candidate_record(), ensure_ascii=False) + "\n"},
    )

    manifest = write_primary_baseline_result_candidate_outputs(
        root=tmp_path,
        external_gpu_smoke_package_path=smoke_package_path,
        t2smark_full_main_package_path=t2smark_package_path,
    )
    output_dir = tmp_path / "outputs" / "external_baseline_results"
    records = [
        json.loads(line)
        for line in (output_dir / "baseline_result_records.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    validation = json.loads(
        (output_dir / "baseline_result_candidate_validation_report.json").read_text(encoding="utf-8")
    )
    readiness_rows = list(csv.DictReader((output_dir / "baseline_formal_import_readiness.csv").open(encoding="utf-8")))
    readiness_summary = json.loads(
        (output_dir / "baseline_formal_import_readiness_summary.json").read_text(encoding="utf-8")
    )
    reasons = {issue["reason"] for issue in validation["issues"]}

    assert manifest["artifact_id"] == "primary_baseline_result_candidate_import_manifest"
    assert len(records) == 4
    assert {row["baseline_id"] for row in records} == {
        "tree_ring",
        "gaussian_shading",
        "shallow_diffuse",
        "t2smark",
    }
    assert all(row["evidence_paths"] for row in records)
    assert validation["accepted_formal_import_count"] == 3
    assert validation["rejected_formal_import_count"] == 1
    assert len(readiness_rows) == 4
    assert readiness_summary["primary_baseline_formal_ready"] is False
    assert readiness_summary["formal_result_ready_count"] == 3
    assert readiness_summary["blocked_primary_baseline_ids"] == ["t2smark"]
    tree_readiness = next(row for row in readiness_rows if row["baseline_id"] == "tree_ring")
    t2smark_readiness = next(row for row in readiness_rows if row["baseline_id"] == "t2smark")
    assert tree_readiness["missing_resource_profile_full_main"] == "False"
    assert t2smark_readiness["missing_resource_profile_full_main"] == "False"
    assert tree_readiness["blocking_reasons"] == ""
    assert "full_main_resource_profile_required" not in reasons
    assert "full_main_prompt_protocol_ready_required" in reasons
    assert "fixed_fpr_baseline_calibration_ready_required" in reasons
    assert "attack_matrix_baseline_detection_ready_required" in reasons
    assert "evidence_path_missing" not in reasons


@pytest.mark.quick
def test_primary_baseline_candidate_writer_imports_t2smark_smoke_when_full_main_package_missing(
    tmp_path: Path,
) -> None:
    """T2SMark full-main 包缺失时, writer 应保留 GPU smoke observation 作为小样本候选。"""

    attack_dir = tmp_path / "outputs" / "attack_matrix"
    attack_dir.mkdir(parents=True)
    (attack_dir / "attack_manifest.json").write_text(
        json.dumps({"evaluation_boundary": {"target_fpr": 0.01}}, ensure_ascii=False),
        encoding="utf-8",
    )
    smoke_package_path = tmp_path / "external_baseline_gpu_smoke_package.zip"
    write_text_package(
        smoke_package_path,
        {GPU_SMOKE_OBSERVATIONS_ENTRY: json.dumps(method_observations() + t2smark_smoke_observations(), ensure_ascii=False)},
    )

    write_primary_baseline_result_candidate_outputs(
        root=tmp_path,
        external_gpu_smoke_package_path=smoke_package_path,
    )
    output_dir = tmp_path / "outputs" / "external_baseline_results"
    records = [
        json.loads(line)
        for line in (output_dir / "baseline_result_records.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    validation = json.loads(
        (output_dir / "baseline_result_candidate_validation_report.json").read_text(encoding="utf-8")
    )
    t2smark_row = next(row for row in records if row["baseline_id"] == "t2smark")
    reasons = {issue["reason"] for issue in validation["issues"] if issue["baseline_id"] == "t2smark"}

    assert len(records) == 4
    assert t2smark_row["resource_profile"] == "full_main"
    assert t2smark_row["adapter_boundary"] == "sd35_medium_native_official_reproduction"
    assert t2smark_row["result_source_type"] == "official_reproduction"
    assert t2smark_row["formal_evidence_paths_ready"] is True
    assert validation["accepted_formal_import_count"] == 4
    assert "full_main_resource_profile_required" not in reasons
