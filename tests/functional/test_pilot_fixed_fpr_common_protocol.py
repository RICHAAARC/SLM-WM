"""pilot fixed-FPR=0.01 共同协议的轻量功能测试。"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from experiments.protocol.attacks import default_attack_configs
from experiments.protocol.pilot_fixed_fpr import (
    PilotFixedFprConfig,
    build_attack_matrix_digest,
    build_fixed_fpr_protocol_digest,
    build_pilot_attack_matrix_rows,
    build_pilot_prompt_split_summary,
    build_pilot_result_import_schema,
    validate_pilot_result_import_rows,
)
from experiments.protocol.prompts import build_prompt_records
from scripts.write_pilot_fixed_fpr_common_protocol_outputs import write_pilot_fixed_fpr_common_protocol_outputs


def write_pilot_prompts(repo_root: Path, prompt_count: int = 240) -> Path:
    """写入测试用 pilot prompt 配置。"""

    config_dir = repo_root / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    prompt_path = config_dir / "paper_main_pilot_prompts.txt"
    prompt_lines = [f"a controlled city pilot prompt variant {index}" for index in range(prompt_count)]
    prompt_path.write_text("\n".join(prompt_lines) + "\n", encoding="utf-8")
    return prompt_path


@pytest.mark.quick
def test_writer_outputs_pilot_common_protocol_with_shared_boundaries(tmp_path: Path) -> None:
    """写出脚本应冻结同一 prompt split、同一攻击矩阵和同一 fixed-FPR 协议。"""

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    write_pilot_prompts(repo_root)

    manifest = write_pilot_fixed_fpr_common_protocol_outputs(root=repo_root)
    output_dir = repo_root / "outputs" / "pilot_fixed_fpr_common_protocol"
    summary = json.loads((output_dir / "pilot_common_protocol_summary.json").read_text(encoding="utf-8"))
    prompt_summary = json.loads((output_dir / "pilot_prompt_split_summary.json").read_text(encoding="utf-8"))
    schema = json.loads((output_dir / "pilot_result_import_schema.json").read_text(encoding="utf-8"))
    validation = json.loads((output_dir / "pilot_result_import_validation_report.json").read_text(encoding="utf-8"))
    method_rows = list(csv.DictReader((output_dir / "pilot_method_registry.csv").open(encoding="utf-8")))
    attack_rows = list(csv.DictReader((output_dir / "pilot_attack_matrix.csv").open(encoding="utf-8")))
    template_rows = [
        json.loads(line)
        for line in (output_dir / "pilot_result_import_template.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert manifest["artifact_id"] == "pilot_fixed_fpr_common_protocol_manifest"
    assert summary["pilot_common_protocol_ready"] is True
    assert summary["pilot_result_import_ready"] is False
    assert summary["pilot_supports_superiority_claim"] is False
    assert summary["supports_paper_claim"] is False
    assert prompt_summary["prompt_set"] == "pilot"
    assert prompt_summary["target_fpr"] == 0.01
    assert prompt_summary["prompt_split_ready"] is True
    assert validation["input_record_count"] == 0
    assert validation["pilot_result_import_ready"] is False
    assert {row["method_id"] for row in method_rows} == {
        "slm_wm_current",
        "tree_ring",
        "gaussian_shading",
        "shallow_diffuse",
        "t2smark",
    }
    assert len(template_rows) == len(method_rows) * len(attack_rows)
    assert {row["prompt_split_digest"] for row in method_rows} == {prompt_summary["prompt_split_digest"]}
    assert {row["attack_matrix_digest"] for row in method_rows} == {schema["attack_matrix_digest"]}
    assert {row["fixed_fpr_protocol_digest"] for row in method_rows} == {schema["fixed_fpr_protocol_digest"]}
    assert all(row["target_fpr"] == 0.01 for row in template_rows)
    assert all(row["result_claim_scope"] == "pilot_not_full_main_paper_claim" for row in template_rows)
    assert all("true_positive_rate_ci_low" in row["required_metric_fields"] for row in template_rows)
    assert all(path.startswith("outputs/") for path in manifest["output_paths"])


def pilot_result_row(schema: dict[str, object], evidence_path: str) -> dict[str, object]:
    """构造一条满足 pilot 导入 schema 的最小结果记录。"""

    return {
        "method_id": "tree_ring",
        "attack_family": "standard_distortion",
        "attack_name": "jpeg_compression",
        "resource_profile": "full_main",
        "target_fpr": 0.01,
        "result_protocol_name": schema["result_protocol_name"],
        "result_scope": schema["result_scope"],
        "result_claim_scope": schema["result_claim_scope"],
        "prompt_protocol_name": schema["prompt_protocol_name"],
        "prompt_split_digest": schema["prompt_split_digest"],
        "attack_matrix_digest": schema["attack_matrix_digest"],
        "fixed_fpr_protocol_digest": schema["fixed_fpr_protocol_digest"],
        "bootstrap_iteration_count": schema["bootstrap_iteration_count"],
        "confidence_level": schema["confidence_level"],
        "baseline_result_source": evidence_path,
        "baseline_result_source_digest": "digest",
        "evidence_paths": [evidence_path],
        "positive_count": 20,
        "negative_count": 30,
        "attack_record_count": 50,
        "supported_record_count": 50,
        "true_positive_rate": 0.80,
        "true_positive_rate_ci_low": 0.70,
        "true_positive_rate_ci_high": 0.90,
        "false_positive_rate": 0.01,
        "false_positive_rate_ci_low": 0.00,
        "false_positive_rate_ci_high": 0.05,
        "clean_false_positive_rate": 0.00,
        "clean_false_positive_rate_ci_low": 0.00,
        "clean_false_positive_rate_ci_high": 0.03,
        "attacked_false_positive_rate": 0.02,
        "attacked_false_positive_rate_ci_low": 0.00,
        "attacked_false_positive_rate_ci_high": 0.08,
        "quality_score_mean": 0.88,
        "quality_score_ci_low": 0.84,
        "quality_score_ci_high": 0.91,
        "score_retention_mean": 0.76,
        "score_retention_ci_low": 0.70,
        "score_retention_ci_high": 0.82,
        "supports_paper_claim": False,
    }


@pytest.mark.quick
def test_pilot_import_validator_accepts_governed_bootstrap_record(tmp_path: Path) -> None:
    """带 bootstrap 置信区间的 pilot 结果应能进入受治理导入协议。"""

    config = PilotFixedFprConfig()
    prompt_records = build_prompt_records(
        "pilot",
        tuple(f"a controlled city pilot prompt variant {index}" for index in range(240)),
    )
    prompt_summary = build_pilot_prompt_split_summary(prompt_records, config)
    attack_rows = build_pilot_attack_matrix_rows(default_attack_configs(), config)
    schema = build_pilot_result_import_schema(
        prompt_split_digest=prompt_summary["prompt_split_digest"],
        attack_matrix_digest=build_attack_matrix_digest(attack_rows),
        fixed_fpr_protocol_digest=build_fixed_fpr_protocol_digest(config),
        config=config,
    )
    evidence_path = tmp_path / "outputs" / "pilot_fixed_fpr_results" / "tree_ring_metrics.json"
    evidence_path.parent.mkdir(parents=True)
    evidence_path.write_text('{"true_positive_rate": 0.8}\n', encoding="utf-8")
    row = pilot_result_row(schema, "outputs/pilot_fixed_fpr_results/tree_ring_metrics.json")

    report = validate_pilot_result_import_rows(
        [row],
        schema,
        evidence_root=tmp_path,
        require_existing_evidence=True,
    )

    assert report["pilot_result_import_ready"] is True
    assert report["accepted_pilot_import_count"] == 1
    assert report["accepted_records"][0]["method_id"] == "tree_ring"
    assert report["supports_paper_claim"] is False


@pytest.mark.quick
def test_pilot_import_validator_rejects_paper_claim_boundary(tmp_path: Path) -> None:
    """pilot 导入记录不得声明为 full-main 论文主张。"""

    config = PilotFixedFprConfig()
    prompt_records = build_prompt_records(
        "pilot",
        tuple(f"a controlled city pilot prompt variant {index}" for index in range(240)),
    )
    prompt_summary = build_pilot_prompt_split_summary(prompt_records, config)
    attack_rows = build_pilot_attack_matrix_rows(default_attack_configs(), config)
    schema = build_pilot_result_import_schema(
        prompt_split_digest=prompt_summary["prompt_split_digest"],
        attack_matrix_digest=build_attack_matrix_digest(attack_rows),
        fixed_fpr_protocol_digest=build_fixed_fpr_protocol_digest(config),
        config=config,
    )
    row = pilot_result_row(schema, "outputs/pilot_fixed_fpr_results/tree_ring_metrics.json")
    row["supports_paper_claim"] = True

    report = validate_pilot_result_import_rows([row], schema, evidence_root=tmp_path)
    reasons = {issue["reason"] for issue in report["issues"]}

    assert report["accepted_pilot_import_count"] == 0
    assert "pilot_result_must_not_support_paper_claim" in reasons
