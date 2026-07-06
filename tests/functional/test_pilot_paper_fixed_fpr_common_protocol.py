"""pilot_paper fixed-FPR=0.01 共同协议的轻量功能测试。"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from experiments.protocol.attacks import default_attack_configs
from experiments.protocol.pilot_paper_fixed_fpr import (
    PilotPaperFixedFprConfig,
    build_attack_matrix_digest,
    build_fixed_fpr_protocol_digest,
    build_pilot_paper_attack_matrix_rows,
    build_pilot_paper_common_protocol_summary,
    build_pilot_paper_method_registry_rows,
    build_pilot_paper_prompt_split_summary,
    build_pilot_paper_result_import_template_rows,
    build_pilot_paper_result_import_schema,
    validate_pilot_paper_result_import_rows,
)
from experiments.protocol.prompts import build_prompt_records
from scripts.write_pilot_paper_fixed_fpr_common_protocol_outputs import write_pilot_paper_fixed_fpr_common_protocol_outputs


def write_pilot_paper_prompts(repo_root: Path, prompt_count: int = 250) -> Path:
    """写入测试用 pilot_paper prompt 配置。"""

    config_dir = repo_root / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    prompt_path = config_dir / "paper_main_pilot_paper_prompts.txt"
    prompt_lines = [f"a controlled city pilot_paper prompt variant {index}" for index in range(prompt_count)]
    prompt_path.write_text("\n".join(prompt_lines) + "\n", encoding="utf-8")
    return prompt_path


def write_full_paper_prompts(repo_root: Path, prompt_count: int = 250) -> Path:
    """写入测试用 full_paper prompt 配置。"""

    config_dir = repo_root / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    prompt_path = config_dir / "paper_main_full_paper_prompts.txt"
    prompt_lines = [f"a controlled city full_paper prompt variant {index}" for index in range(prompt_count)]
    prompt_path.write_text("\n".join(prompt_lines) + "\n", encoding="utf-8")
    return prompt_path


def write_probe_paper_prompts(repo_root: Path, prompt_count: int = 60) -> Path:
    """写入测试用 probe_paper prompt 配置。"""

    config_dir = repo_root / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    prompt_path = config_dir / "paper_main_probe_paper_prompts.txt"
    prompt_lines = [f"a controlled city probe_paper prompt variant {index}" for index in range(prompt_count)]
    prompt_path.write_text("\n".join(prompt_lines) + "\n", encoding="utf-8")
    return prompt_path


@pytest.mark.quick
def test_writer_outputs_pilot_paper_common_protocol_with_shared_boundaries(tmp_path: Path) -> None:
    """写出脚本应冻结同一 prompt split、同一攻击矩阵和同一 fixed-FPR 协议。"""

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    write_pilot_paper_prompts(repo_root)

    manifest = write_pilot_paper_fixed_fpr_common_protocol_outputs(root=repo_root)
    output_dir = repo_root / "outputs" / "pilot_paper_fixed_fpr_common_protocol"
    summary = json.loads((output_dir / "pilot_paper_common_protocol_summary.json").read_text(encoding="utf-8"))
    prompt_summary = json.loads((output_dir / "pilot_paper_prompt_split_summary.json").read_text(encoding="utf-8"))
    schema = json.loads((output_dir / "pilot_paper_result_import_schema.json").read_text(encoding="utf-8"))
    validation = json.loads((output_dir / "pilot_paper_result_import_validation_report.json").read_text(encoding="utf-8"))
    method_rows = list(csv.DictReader((output_dir / "pilot_paper_method_registry.csv").open(encoding="utf-8")))
    attack_rows = list(csv.DictReader((output_dir / "pilot_paper_attack_matrix.csv").open(encoding="utf-8")))
    template_rows = [
        json.loads(line)
        for line in (output_dir / "pilot_paper_result_import_template.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert manifest["artifact_id"] == "pilot_paper_fixed_fpr_common_protocol_manifest"
    assert summary["pilot_paper_common_protocol_ready"] is True
    assert summary["pilot_paper_result_import_ready"] is False
    assert summary["pilot_paper_supports_superiority_claim"] is False
    assert summary["pilot_paper_claim_ready"] is False
    assert summary["full_paper_claim_ready"] is False
    assert summary["supports_paper_claim"] is False
    assert prompt_summary["prompt_set"] == "pilot_paper"
    assert prompt_summary["target_fpr"] == 0.01
    assert prompt_summary["prompt_split_ready"] is True
    assert schema["minimum_result_positive_count"] == 100
    assert schema["minimum_result_negative_count"] == 100
    assert validation["input_record_count"] == 0
    assert validation["pilot_paper_result_import_ready"] is False
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
    assert all(row["result_claim_scope"] == "pilot_paper_paper_claim" for row in template_rows)
    assert all("true_positive_rate_ci_low" in row["required_metric_fields"] for row in template_rows)
    assert all(path.startswith("outputs/") for path in manifest["output_paths"])


@pytest.mark.quick
def test_common_protocol_blocks_superiority_claim_when_slm_wm_tpr_is_below_baselines() -> None:
    """证据覆盖完整但 SLM-WM TPR 低于 baseline 时, 不得支持优势性主张。"""

    config = PilotPaperFixedFprConfig()
    prompt_summary = {
        "prompt_split_ready": True,
        "pilot_paper_prompt_count": 240,
        "prompt_split_digest": "prompt_digest",
    }
    attack_rows = build_pilot_paper_attack_matrix_rows(default_attack_configs(), config)
    method_rows = build_pilot_paper_method_registry_rows(
        prompt_split_digest="prompt_digest",
        attack_matrix_digest="attack_digest",
        fixed_fpr_protocol_digest="fixed_fpr_digest",
        config=config,
    )
    template_rows = build_pilot_paper_result_import_template_rows(method_rows, attack_rows, config)
    accepted_records = []
    for row in template_rows:
        method_id = str(row["method_id"])
        accepted_records.append(
            {
                **row,
                "true_positive_rate": 0.01 if method_id == "slm_wm_current" else 0.50,
                "false_positive_rate": 0.001,
                "supports_paper_claim": True,
            }
        )
    summary = build_pilot_paper_common_protocol_summary(
        prompt_summary=prompt_summary,
        attack_rows=attack_rows,
        method_rows=method_rows,
        template_rows=template_rows,
        import_validation_report={
            "pilot_paper_result_import_ready": True,
            "accepted_pilot_paper_import_count": len(accepted_records),
            "accepted_records": accepted_records,
        },
        config=config,
    )

    assert summary["pilot_paper_evidence_coverage_ready"] is True
    assert summary["pilot_paper_effectiveness_gate_ready"] is False
    assert summary["pilot_paper_effectiveness_gate_reason"] == "slm_wm_tpr_not_above_best_baseline"
    assert summary["pilot_paper_supports_superiority_claim"] is False
    assert summary["paper_claim_ready"] is False


@pytest.mark.quick
def test_writer_switches_common_protocol_to_full_paper_without_logic_fork(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """同一共同协议写出脚本应仅凭论文运行配置切换到 full_paper。"""

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    write_full_paper_prompts(repo_root)
    monkeypatch.setenv("SLM_WM_PAPER_RUN_NAME", "full_paper")

    write_pilot_paper_fixed_fpr_common_protocol_outputs(root=repo_root)
    output_dir = repo_root / "outputs" / "pilot_paper_fixed_fpr_common_protocol"
    summary = json.loads((output_dir / "pilot_paper_common_protocol_summary.json").read_text(encoding="utf-8"))
    prompt_summary = json.loads((output_dir / "pilot_paper_prompt_split_summary.json").read_text(encoding="utf-8"))
    schema = json.loads((output_dir / "pilot_paper_result_import_schema.json").read_text(encoding="utf-8"))
    template_rows = [
        json.loads(line)
        for line in (output_dir / "pilot_paper_result_import_template.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert summary["paper_claim_scale"] == "full_paper"
    assert summary["result_protocol_name"] == "full_paper_fixed_fpr_common_protocol"
    assert summary["result_claim_scope"] == "full_paper_paper_claim"
    assert summary["full_paper_claim_ready"] is False
    assert prompt_summary["prompt_set"] == "full_paper"
    assert prompt_summary["prompt_protocol_name"] == "paper_main_full_paper_prompt_protocol"
    assert schema["paper_claim_scale"] == "full_paper"
    assert schema["prompt_set"] == "full_paper"
    assert schema["prompt_protocol_name"] == "paper_main_full_paper_prompt_protocol"
    assert schema["result_protocol_name"] == "full_paper_fixed_fpr_common_protocol"
    assert all(row["paper_claim_scale"] == "full_paper" for row in template_rows)
    assert all(row["result_claim_scope"] == "full_paper_paper_claim" for row in template_rows)


@pytest.mark.quick
def test_writer_switches_common_protocol_to_probe_paper_without_logic_fork(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """同一共同协议写出脚本应支持 probe_paper 小规模流程对齐验证。"""

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    write_probe_paper_prompts(repo_root)
    monkeypatch.setenv("SLM_WM_PAPER_RUN_NAME", "probe_paper")

    write_pilot_paper_fixed_fpr_common_protocol_outputs(root=repo_root)
    output_dir = repo_root / "outputs" / "pilot_paper_fixed_fpr_common_protocol"
    summary = json.loads((output_dir / "pilot_paper_common_protocol_summary.json").read_text(encoding="utf-8"))
    prompt_summary = json.loads((output_dir / "pilot_paper_prompt_split_summary.json").read_text(encoding="utf-8"))
    schema = json.loads((output_dir / "pilot_paper_result_import_schema.json").read_text(encoding="utf-8"))
    template_rows = [
        json.loads(line)
        for line in (output_dir / "pilot_paper_result_import_template.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert summary["paper_claim_scale"] == "probe_paper"
    assert summary["result_protocol_name"] == "probe_paper_fixed_fpr_common_protocol"
    assert summary["result_claim_scope"] == "probe_paper_paper_claim"
    assert summary["probe_paper_claim_ready"] is False
    assert prompt_summary["prompt_set"] == "probe_paper"
    assert prompt_summary["pilot_paper_prompt_count"] == 60
    assert prompt_summary["target_fpr"] == 0.1
    assert prompt_summary["prompt_protocol_name"] == "paper_main_probe_paper_prompt_protocol"
    assert schema["paper_claim_scale"] == "probe_paper"
    assert schema["prompt_set"] == "probe_paper"
    assert schema["target_fpr"] == 0.1
    assert schema["prompt_protocol_name"] == "paper_main_probe_paper_prompt_protocol"
    assert schema["result_protocol_name"] == "probe_paper_fixed_fpr_common_protocol"
    assert all(row["paper_claim_scale"] == "probe_paper" for row in template_rows)
    assert all(row["target_fpr"] == 0.1 for row in template_rows)
    assert all(row["result_claim_scope"] == "probe_paper_paper_claim" for row in template_rows)


def pilot_paper_result_row(schema: dict[str, object], evidence_path: str) -> dict[str, object]:
    """构造一条满足 pilot_paper 导入 schema 的最小结果记录。"""

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
        "positive_count": 120,
        "negative_count": 120,
        "attack_record_count": 240,
        "supported_record_count": 240,
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
        "supports_paper_claim": True,
        "paper_claim_scale": "pilot_paper",
    }


@pytest.mark.quick
def test_pilot_paper_import_validator_accepts_governed_bootstrap_record(tmp_path: Path) -> None:
    """带 bootstrap 置信区间的 pilot_paper 结果应能进入受治理导入协议。"""

    config = PilotPaperFixedFprConfig()
    prompt_records = build_prompt_records(
        "pilot_paper",
        tuple(f"a controlled city pilot_paper prompt variant {index}" for index in range(250)),
    )
    prompt_summary = build_pilot_paper_prompt_split_summary(prompt_records, config)
    attack_rows = build_pilot_paper_attack_matrix_rows(default_attack_configs(), config)
    schema = build_pilot_paper_result_import_schema(
        prompt_split_digest=prompt_summary["prompt_split_digest"],
        attack_matrix_digest=build_attack_matrix_digest(attack_rows),
        fixed_fpr_protocol_digest=build_fixed_fpr_protocol_digest(config),
        config=config,
    )
    evidence_path = tmp_path / "outputs" / "pilot_paper_fixed_fpr_results" / "tree_ring_metrics.json"
    evidence_path.parent.mkdir(parents=True)
    evidence_path.write_text('{"true_positive_rate": 0.8}\n', encoding="utf-8")
    row = pilot_paper_result_row(schema, "outputs/pilot_paper_fixed_fpr_results/tree_ring_metrics.json")

    report = validate_pilot_paper_result_import_rows(
        [row],
        schema,
        evidence_root=tmp_path,
        require_existing_evidence=True,
    )

    assert report["pilot_paper_result_import_ready"] is True
    assert report["accepted_pilot_paper_import_count"] == 1
    assert report["accepted_records"][0]["method_id"] == "tree_ring"
    assert report["accepted_pilot_paper_claim_record_count"] == 1
    assert report["supports_paper_claim"] is True


@pytest.mark.quick
def test_pilot_paper_import_validator_rejects_full_paper_claim_boundary(tmp_path: Path) -> None:
    """pilot_paper 导入记录不得声明为 full_paper 论文主张。"""

    config = PilotPaperFixedFprConfig()
    prompt_records = build_prompt_records(
        "pilot_paper",
        tuple(f"a controlled city pilot_paper prompt variant {index}" for index in range(250)),
    )
    prompt_summary = build_pilot_paper_prompt_split_summary(prompt_records, config)
    attack_rows = build_pilot_paper_attack_matrix_rows(default_attack_configs(), config)
    schema = build_pilot_paper_result_import_schema(
        prompt_split_digest=prompt_summary["prompt_split_digest"],
        attack_matrix_digest=build_attack_matrix_digest(attack_rows),
        fixed_fpr_protocol_digest=build_fixed_fpr_protocol_digest(config),
        config=config,
    )
    row = pilot_paper_result_row(schema, "outputs/pilot_paper_fixed_fpr_results/tree_ring_metrics.json")
    row["result_claim_scope"] = "full_paper_paper_claim"
    row["paper_claim_scale"] = "full_paper"

    report = validate_pilot_paper_result_import_rows([row], schema, evidence_root=tmp_path)
    reasons = {issue["reason"] for issue in report["issues"]}

    assert report["accepted_pilot_paper_import_count"] == 0
    assert "protocol_value_mismatch" in reasons
    assert "pilot_paper_claim_scale_required" in reasons


@pytest.mark.quick
def test_pilot_paper_import_validator_rejects_small_sample_result_records(tmp_path: Path) -> None:
    """低于 pilot_paper fixed-FPR 统计边界的记录不得进入受治理导入协议。"""

    config = PilotPaperFixedFprConfig()
    prompt_records = build_prompt_records(
        "pilot_paper",
        tuple(f"a controlled city pilot_paper prompt variant {index}" for index in range(250)),
    )
    prompt_summary = build_pilot_paper_prompt_split_summary(prompt_records, config)
    attack_rows = build_pilot_paper_attack_matrix_rows(default_attack_configs(), config)
    schema = build_pilot_paper_result_import_schema(
        prompt_split_digest=prompt_summary["prompt_split_digest"],
        attack_matrix_digest=build_attack_matrix_digest(attack_rows),
        fixed_fpr_protocol_digest=build_fixed_fpr_protocol_digest(config),
        config=config,
    )
    row = pilot_paper_result_row(schema, "outputs/pilot_paper_fixed_fpr_results/tree_ring_metrics.json")
    row["positive_count"] = 5
    row["negative_count"] = 5

    report = validate_pilot_paper_result_import_rows([row], schema, evidence_root=tmp_path)
    reasons = {issue["reason"] for issue in report["issues"]}

    assert report["accepted_pilot_paper_import_count"] == 0
    assert "pilot_paper_minimum_sample_count_required" in reasons
