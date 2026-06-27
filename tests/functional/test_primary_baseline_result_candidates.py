"""主表 external baseline 候选结果导入脚本的轻量功能测试。"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from zipfile import ZipFile

import pytest

from scripts.write_primary_baseline_result_candidates import (
    METHOD_FAITHFUL_OBSERVATIONS_ENTRY,
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


def t2smark_method_faithful_observations() -> list[dict[str, object]]:
    """构造 T2SMark method-faithful adapter 的最小 observation 集合。"""

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


def t2smark_attack_method_faithful_observations() -> list[dict[str, object]]:
    """构造 T2SMark 非 clean 攻击 observation, 用于验证缺失攻击项合并。"""

    return [
        {
            "baseline_id": "t2smark",
            "attack_family": "standard_distortion",
            "attack_condition": "gaussian_blur",
            "sample_role": "attacked_negative",
            "detection_decision": False,
            "prompt_id": "prompt_000",
            "prompt_text": "a ceramic fox on a wooden desk",
            "quality_score_proxy": 0.9,
            "score_retention_proxy": 0.8,
        },
        {
            "baseline_id": "t2smark",
            "attack_family": "standard_distortion",
            "attack_condition": "gaussian_blur",
            "sample_role": "attacked_positive",
            "detection_decision": True,
            "prompt_id": "prompt_000",
            "prompt_text": "a ceramic fox on a wooden desk",
            "quality_score_proxy": 0.9,
            "score_retention_proxy": 0.8,
        },
    ]


def regeneration_observations() -> list[dict[str, object]]:
    """构造覆盖再生成攻击的最小 method-faithful observation 集合。"""

    return [
        {
            "baseline_id": "tree_ring",
            "attack_family": "regeneration_attack",
            "attack_name": "img2img_regeneration",
            "attack_condition": "img2img_regeneration",
            "sample_role": "attacked_negative",
            "detection_decision": False,
            "prompt_id": "prompt_000",
            "prompt_text": "a ceramic fox on a wooden desk",
            "quality_score_proxy": 1.0,
            "score_retention_proxy": 0.8,
        },
        {
            "baseline_id": "tree_ring",
            "attack_family": "regeneration_attack",
            "attack_name": "img2img_regeneration",
            "attack_condition": "img2img_regeneration",
            "sample_role": "attacked_positive",
            "detection_decision": True,
            "prompt_id": "prompt_000",
            "prompt_text": "a ceramic fox on a wooden desk",
            "quality_score_proxy": 1.0,
            "score_retention_proxy": 0.8,
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
def test_primary_baseline_candidate_writer_imports_packages_without_promoting_method_faithful_results(tmp_path: Path) -> None:
    """候选导入脚本应保留候选证据, 但不能把小样本 GPU 链路升级为主表结论。"""

    attack_dir = tmp_path / "outputs" / "attack_matrix"
    attack_dir.mkdir(parents=True)
    (attack_dir / "attack_manifest.json").write_text(
        json.dumps({"evaluation_boundary": {"target_fpr": 0.01}}, ensure_ascii=False),
        encoding="utf-8",
    )
    method_faithful_package_path = tmp_path / "external_baseline_method_faithful_package.zip"
    t2smark_package_path = tmp_path / "t2smark_full_main_reproduction_package.zip"
    write_text_package(
        method_faithful_package_path,
        {METHOD_FAITHFUL_OBSERVATIONS_ENTRY: json.dumps(method_observations(), ensure_ascii=False)},
    )
    write_text_package(
        t2smark_package_path,
        {T2SMARK_CANDIDATE_RECORDS_ENTRY: json.dumps(t2smark_candidate_record(), ensure_ascii=False) + "\n"},
    )

    manifest = write_primary_baseline_result_candidate_outputs(
        root=tmp_path,
        external_method_faithful_package_path=method_faithful_package_path,
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
def test_primary_baseline_candidate_writer_preserves_regeneration_resource_profile(tmp_path: Path) -> None:
    """再生成攻击候选记录应使用攻击矩阵中的 full_extra 资源档位。"""

    attack_dir = tmp_path / "outputs" / "attack_matrix"
    attack_dir.mkdir(parents=True)
    (attack_dir / "attack_manifest.json").write_text(
        json.dumps({"evaluation_boundary": {"target_fpr": 0.01}}, ensure_ascii=False),
        encoding="utf-8",
    )
    method_faithful_package_path = tmp_path / "external_baseline_method_faithful_package.zip"
    write_text_package(
        method_faithful_package_path,
        {METHOD_FAITHFUL_OBSERVATIONS_ENTRY: json.dumps(method_observations() + regeneration_observations(), ensure_ascii=False)},
    )

    write_primary_baseline_result_candidate_outputs(
        root=tmp_path,
        external_method_faithful_package_path=method_faithful_package_path,
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
    regeneration_row = next(row for row in records if row["attack_name"] == "img2img_regeneration")

    assert regeneration_row["resource_profile"] == "full_extra"
    assert validation["accepted_formal_import_count"] == 4
    assert "allowed_resource_profile_required" not in {issue["reason"] for issue in validation["issues"]}


@pytest.mark.quick
def test_primary_baseline_candidate_writer_imports_t2smark_method_faithful_when_full_main_package_missing(
    tmp_path: Path,
) -> None:
    """T2SMark full-main 包缺失时, writer 应保留 method-faithful observation 作为小样本候选。"""

    attack_dir = tmp_path / "outputs" / "attack_matrix"
    attack_dir.mkdir(parents=True)
    (attack_dir / "attack_manifest.json").write_text(
        json.dumps({"evaluation_boundary": {"target_fpr": 0.01}}, ensure_ascii=False),
        encoding="utf-8",
    )
    method_faithful_package_path = tmp_path / "external_baseline_method_faithful_package.zip"
    write_text_package(
        method_faithful_package_path,
        {METHOD_FAITHFUL_OBSERVATIONS_ENTRY: json.dumps(method_observations() + t2smark_method_faithful_observations(), ensure_ascii=False)},
    )

    write_primary_baseline_result_candidate_outputs(
        root=tmp_path,
        external_method_faithful_package_path=method_faithful_package_path,
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


@pytest.mark.quick
def test_primary_baseline_candidate_writer_merges_t2smark_formal_and_method_faithful_attack_rows(
    tmp_path: Path,
) -> None:
    """T2SMark 专用复现包只含 clean 时, 不得丢弃 GPU 观测中的攻击矩阵记录。"""

    attack_dir = tmp_path / "outputs" / "attack_matrix"
    attack_dir.mkdir(parents=True)
    (attack_dir / "attack_manifest.json").write_text(
        json.dumps({"evaluation_boundary": {"target_fpr": 0.01}}, ensure_ascii=False),
        encoding="utf-8",
    )
    method_faithful_package_path = tmp_path / "external_baseline_method_faithful_package.zip"
    t2smark_package_path = tmp_path / "t2smark_full_main_reproduction_package.zip"
    write_text_package(
        method_faithful_package_path,
        {
            METHOD_FAITHFUL_OBSERVATIONS_ENTRY: json.dumps(
                method_observations() + t2smark_method_faithful_observations() + t2smark_attack_method_faithful_observations(),
                ensure_ascii=False,
            )
        },
    )
    write_text_package(
        t2smark_package_path,
        {T2SMARK_CANDIDATE_RECORDS_ENTRY: json.dumps(t2smark_candidate_record(), ensure_ascii=False) + "\n"},
    )

    write_primary_baseline_result_candidate_outputs(
        root=tmp_path,
        external_method_faithful_package_path=method_faithful_package_path,
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
    t2smark_keys = {
        (row["attack_family"], row["attack_name"], row["resource_profile"])
        for row in records
        if row["baseline_id"] == "t2smark"
    }

    assert ("clean", "clean_none", "full_main") in t2smark_keys
    assert ("standard_distortion", "gaussian_blur", "full_main") in t2smark_keys
    assert len([row for row in records if row["baseline_id"] == "t2smark"]) == 2
    assert validation["accepted_formal_import_count"] == 4


@pytest.mark.quick
def test_primary_baseline_candidate_writer_merges_split_baseline_observations(tmp_path: Path) -> None:
    """四个单 baseline 包物化后, writer 应合并 split_observations 而不是只读最后一次 execution。"""

    attack_dir = tmp_path / "outputs" / "attack_matrix"
    split_dir = tmp_path / "outputs" / "external_baseline_method_faithful" / "split_observations"
    attack_dir.mkdir(parents=True)
    split_dir.mkdir(parents=True)
    (attack_dir / "attack_manifest.json").write_text(
        json.dumps({"evaluation_boundary": {"target_fpr": 0.01}}, ensure_ascii=False),
        encoding="utf-8",
    )
    for baseline_id in ("tree_ring", "gaussian_shading", "shallow_diffuse"):
        rows = [row for row in method_observations() if row["baseline_id"] == baseline_id]
        (split_dir / f"{baseline_id}_baseline_observations.json").write_text(
            json.dumps(rows, ensure_ascii=False),
            encoding="utf-8",
        )
    (split_dir / "t2smark_baseline_observations.json").write_text(
        json.dumps(t2smark_method_faithful_observations(), ensure_ascii=False),
        encoding="utf-8",
    )

    write_primary_baseline_result_candidate_outputs(root=tmp_path)
    output_dir = tmp_path / "outputs" / "external_baseline_results"
    records = [
        json.loads(line)
        for line in (output_dir / "baseline_result_records.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert {row["baseline_id"] for row in records} == {
        "tree_ring",
        "gaussian_shading",
        "shallow_diffuse",
        "t2smark",
    }
