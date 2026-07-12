"""pilot_paper fixed-FPR=0.01 共同协议的轻量功能测试。"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from experiments.protocol.attacks import (
    attack_config_digest,
    default_attack_configs,
    resolve_formal_attack_config,
)
from experiments.protocol.pilot_paper_fixed_fpr import (
    PilotPaperFixedFprConfig,
    bounded_hoeffding_confidence_interval,
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
from tests.helpers.formal_prompt_source import copy_governed_prompt_file
from scripts.write_pilot_paper_fixed_fpr_common_protocol_outputs import write_pilot_paper_fixed_fpr_common_protocol_outputs
from tests.helpers.closure_input_lock import write_test_closure_input_lock


PRIMARY_BASELINE_IDS = ("tree_ring", "gaussian_shading", "shallow_diffuse", "t2smark")


@pytest.fixture(autouse=True)
def _select_pilot_paper(monkeypatch: pytest.MonkeyPatch) -> None:
    """本模块未显式切换层级的协议夹具固定使用 pilot_paper."""

    monkeypatch.setenv("SLM_WM_PAPER_RUN_NAME", "pilot_paper")


def paired_superiority_summary(
    config: PilotPaperFixedFprConfig,
    *,
    attack_count: int,
    ready: bool,
) -> dict[str, object]:
    """构造与当前共同协议规模一致的配对统计摘要."""

    row: dict[str, object] = {
        "paper_claim_scale": config.paper_run_name,
        "target_fpr": config.target_fpr,
        "expected_test_count": config.minimum_clean_negative_count,
        "expected_attack_count": attack_count,
        "paired_prompt_counts": [config.minimum_clean_negative_count],
        "paired_attack_counts": [attack_count],
        "primary_baseline_ids": list(PRIMARY_BASELINE_IDS),
        "paired_superiority_ready_ids": list(PRIMARY_BASELINE_IDS) if ready else [],
        "paired_superiority_row_count": len(PRIMARY_BASELINE_IDS),
        "paired_superiority_exact_set_ready": True,
        "paired_superiority_scale_ready": True,
        "overall_paired_superiority_ready": ready,
        "paired_outcome_set_digest": "a" * 64,
        "paired_superiority_rows_digest": "b" * 64,
        "paired_superiority_protocol_digest": "c" * 64,
        "paired_test_prompt_count": config.minimum_clean_negative_count,
        "paired_test_prompt_id_digest": "d" * 64,
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
            for method_id in ("slm_wm", *PRIMARY_BASELINE_IDS)
        },
        "method_threshold_digest_map": {
            "slm_wm": "6" * 64,
            "tree_ring": "7" * 64,
            "gaussian_shading": "8" * 64,
            "shallow_diffuse": "9" * 64,
            "t2smark": "a" * 64,
        },
        "supports_paper_claim": ready,
    }
    return row


def write_paired_superiority_inputs(
    root: Path,
    *,
    paper_run_name: str,
    target_fpr: float,
    expected_test_count: int,
    ready: bool = False,
) -> None:
    """写出共同协议 writer 所需的 run-scoped 配对统计证据."""

    config = PilotPaperFixedFprConfig(
        paper_run_name=paper_run_name,
        prompt_set=paper_run_name,
        prompt_file=f"configs/paper_main_{paper_run_name}_prompts.txt",
        prompt_protocol_name=f"paper_main_{paper_run_name}_prompt_protocol",
        result_protocol_name=f"{paper_run_name}_fixed_fpr_common_protocol",
        result_scope=f"{paper_run_name}_common_protocol",
        result_claim_scope={
            "probe_paper": "probe_claim",
            "pilot_paper": "pilot_claim",
            "full_paper": "full_claim",
        }[paper_run_name],
        target_fpr=target_fpr,
        minimum_clean_negative_count=expected_test_count,
    )
    summary = paired_superiority_summary(
        config,
        attack_count=len(default_attack_configs()),
        ready=ready,
    )
    output_dir = root / "outputs" / "paired_superiority_analysis" / paper_run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "paired_superiority_summary.json").write_text(
        json.dumps(summary),
        encoding="utf-8",
    )
    (output_dir / "manifest.local.json").write_text(
        json.dumps(
            {
                "artifact_id": "paired_superiority_analysis_manifest",
                "code_version": "a" * 40,
                "metadata": summary,
            }
        ),
        encoding="utf-8",
    )


def write_pilot_paper_prompts(repo_root: Path) -> Path:
    """写入受治理的 pilot_paper Prompt 配置。"""

    return copy_governed_prompt_file(repo_root, "pilot_paper")


def write_full_paper_prompts(repo_root: Path) -> Path:
    """写入受治理的 full_paper Prompt 配置。"""

    return copy_governed_prompt_file(repo_root, "full_paper")


def write_probe_paper_prompts(repo_root: Path) -> Path:
    """写入受治理的 probe_paper Prompt 配置。"""

    return copy_governed_prompt_file(repo_root, "probe_paper")


@pytest.mark.quick
def test_writer_outputs_pilot_paper_common_protocol_with_shared_boundaries(tmp_path: Path) -> None:
    """写出脚本应冻结同一 prompt split、同一攻击矩阵和同一 fixed-FPR 协议。"""

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    write_pilot_paper_prompts(repo_root)
    write_test_closure_input_lock(
        repo_root,
        paper_run_name="pilot_paper",
        target_fpr=0.01,
    )
    write_paired_superiority_inputs(
        repo_root,
        paper_run_name="pilot_paper",
        target_fpr=0.01,
        expected_test_count=340,
    )

    manifest = write_pilot_paper_fixed_fpr_common_protocol_outputs(root=repo_root)
    output_dir = repo_root / "outputs" / "pilot_paper_fixed_fpr_common_protocol" / "pilot_paper"
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
    assert schema["minimum_result_positive_count"] == 340
    assert schema["minimum_result_negative_count"] == 340
    assert summary["paired_test_prompt_count"] == 340
    assert schema["paired_test_prompt_id_digest"] == "d" * 64
    assert summary["paired_attack_registry_digest"] == "e" * 64
    assert summary["threshold_audit_rows_digest"] == "f" * 64
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
    assert all(row["result_claim_scope"] == "pilot_claim" for row in template_rows)
    assert all(
        row["required_result_record_path"]
        == "outputs/pilot_paper_fixed_fpr_results/pilot_paper/pilot_paper_result_records.jsonl"
        for row in template_rows
    )
    assert all("true_positive_rate_ci_low" in row["required_metric_fields"] for row in template_rows)
    assert all(path.startswith("outputs/") for path in manifest["output_paths"])


@pytest.mark.quick
def test_common_protocol_accepts_complete_paired_superiority_evidence() -> None:
    """点估计与真实配对统计同时通过时才允许形成当前层级论文结论."""

    config = PilotPaperFixedFprConfig()
    prompt_summary = {
        "prompt_split_ready": True,
        "pilot_paper_prompt_count": 700,
        "test_prompt_id_digest": "d" * 64,
    }
    attack_rows = build_pilot_paper_attack_matrix_rows(default_attack_configs(), config)
    method_rows = build_pilot_paper_method_registry_rows(
        prompt_split_digest="prompt_digest",
        attack_matrix_digest="attack_digest",
        fixed_fpr_protocol_digest="fixed_fpr_digest",
        config=config,
    )
    template_rows = build_pilot_paper_result_import_template_rows(
        method_rows,
        attack_rows,
        config,
    )
    accepted_records = [
        {
            **row,
            "true_positive_rate": (
                0.9 if row["method_id"] == "slm_wm_current" else 0.5
            ),
            "false_positive_rate": 0.001,
            "supports_paper_claim": True,
        }
        for row in template_rows
    ]

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
        paired_superiority_summary=paired_superiority_summary(
            config,
            attack_count=len(attack_rows),
            ready=True,
        ),
        config=config,
    )

    assert summary["paired_superiority_ready"] is True
    assert summary["pilot_paper_effectiveness_gate_ready"] is True
    assert summary["paper_claim_ready"] is True
    assert summary["supports_paper_claim"] is True

    paired_summary = paired_superiority_summary(
        config,
        attack_count=len(attack_rows),
        ready=True,
    )
    mismatched = build_pilot_paper_common_protocol_summary(
        prompt_summary=prompt_summary,
        attack_rows=attack_rows,
        method_rows=method_rows,
        template_rows=template_rows,
        import_validation_report={
            "pilot_paper_result_import_ready": True,
            "accepted_pilot_paper_import_count": len(accepted_records),
            "accepted_records": accepted_records,
        },
        paired_superiority_summary={
            **paired_summary,
            "paired_test_prompt_id_digest": "e" * 64,
        },
        config=config,
    )
    assert mismatched["paired_superiority_ready"] is False
    assert mismatched["paper_claim_ready"] is False


@pytest.mark.quick
def test_common_protocol_blocks_superiority_claim_when_slm_wm_tpr_is_below_baselines() -> None:
    """证据覆盖完整但 SLM-WM TPR 低于 baseline 时, 不得支持优势性主张。"""

    config = PilotPaperFixedFprConfig()
    prompt_summary = {
        "prompt_split_ready": True,
        "pilot_paper_prompt_count": 700,
        "test_prompt_id_digest": "d" * 64,
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
        paired_superiority_summary=paired_superiority_summary(
            config,
            attack_count=len(attack_rows),
            ready=True,
        ),
        config=config,
    )

    assert summary["pilot_paper_evidence_coverage_ready"] is True
    assert summary["pilot_paper_effectiveness_gate_ready"] is False
    assert summary["pilot_paper_effectiveness_gate_reason"] == "slm_wm_tpr_not_above_best_baseline"
    assert summary["pilot_paper_supports_superiority_claim"] is False
    assert summary["paper_claim_ready"] is False


@pytest.mark.quick
def test_common_protocol_blocks_duplicate_method_attack_records() -> None:
    """共同协议必须阻断重复的 method × attack 记录, 避免重复行改变聚合统计。"""

    config = PilotPaperFixedFprConfig()
    prompt_summary = {
        "prompt_split_ready": True,
        "pilot_paper_prompt_count": 700,
        "test_prompt_id_digest": "d" * 64,
    }
    attack_rows = build_pilot_paper_attack_matrix_rows(default_attack_configs(), config)
    method_rows = build_pilot_paper_method_registry_rows(
        prompt_split_digest="prompt_digest",
        attack_matrix_digest="attack_digest",
        fixed_fpr_protocol_digest="fixed_fpr_digest",
        config=config,
    )
    template_rows = build_pilot_paper_result_import_template_rows(method_rows, attack_rows, config)
    accepted_records = [
        {
            **row,
            "true_positive_rate": 0.9 if row["method_id"] == "slm_wm_current" else 0.5,
            "false_positive_rate": 0.0,
            "supports_paper_claim": True,
        }
        for row in template_rows
    ]
    accepted_records.append(dict(accepted_records[0]))

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
        paired_superiority_summary=paired_superiority_summary(
            config,
            attack_count=len(attack_rows),
            ready=True,
        ),
        config=config,
    )

    assert summary["paper_run_result_duplicate_template_count"] == 1
    assert summary["paper_run_result_import_coverage_ready"] is False
    assert summary["pilot_paper_effectiveness_gate_ready"] is False
    assert summary["pilot_paper_effectiveness_gate_reason"] == "duplicate_method_attack_template_records"
    assert summary["supports_paper_claim"] is False


@pytest.mark.quick
def test_writer_switches_common_protocol_to_full_paper_without_logic_fork(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """同一共同协议写出脚本应仅凭论文运行配置切换到 full_paper。"""

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    write_full_paper_prompts(repo_root)
    write_test_closure_input_lock(
        repo_root,
        paper_run_name="full_paper",
        target_fpr=0.001,
    )
    write_paired_superiority_inputs(
        repo_root,
        paper_run_name="full_paper",
        target_fpr=0.001,
        expected_test_count=3400,
    )
    monkeypatch.setenv("SLM_WM_PAPER_RUN_NAME", "full_paper")

    write_pilot_paper_fixed_fpr_common_protocol_outputs(root=repo_root)
    output_dir = repo_root / "outputs" / "pilot_paper_fixed_fpr_common_protocol" / "full_paper"
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
    assert summary["result_claim_scope"] == "full_claim"
    assert summary["full_paper_claim_ready"] is False
    assert prompt_summary["prompt_set"] == "full_paper"
    assert prompt_summary["prompt_protocol_name"] == "paper_main_full_paper_prompt_protocol"
    assert schema["paper_claim_scale"] == "full_paper"
    assert schema["prompt_set"] == "full_paper"
    assert schema["prompt_protocol_name"] == "paper_main_full_paper_prompt_protocol"
    assert schema["result_protocol_name"] == "full_paper_fixed_fpr_common_protocol"
    assert all(row["paper_claim_scale"] == "full_paper" for row in template_rows)
    assert all(row["result_claim_scope"] == "full_claim" for row in template_rows)


@pytest.mark.quick
def test_writer_switches_common_protocol_to_probe_paper_without_logic_fork(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """同一共同协议写出脚本应支持 probe_paper 小规模流程对齐验证。"""

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    write_probe_paper_prompts(repo_root)
    write_test_closure_input_lock(
        repo_root,
        paper_run_name="probe_paper",
        target_fpr=0.1,
    )
    write_paired_superiority_inputs(
        repo_root,
        paper_run_name="probe_paper",
        target_fpr=0.1,
        expected_test_count=34,
    )
    monkeypatch.setenv("SLM_WM_PAPER_RUN_NAME", "probe_paper")

    write_pilot_paper_fixed_fpr_common_protocol_outputs(root=repo_root)
    output_dir = repo_root / "outputs" / "pilot_paper_fixed_fpr_common_protocol" / "probe_paper"
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
    assert summary["result_claim_scope"] == "probe_claim"
    assert summary["pilot_paper_common_protocol_ready"] is True
    assert summary["paper_run_allows_paper_claim"] is True
    assert summary["probe_paper_workflow_validation_ready"] is False
    assert summary["probe_paper_claim_ready"] is False
    assert summary["probe_claim_ready"] is False
    assert prompt_summary["prompt_set"] == "probe_paper"
    assert prompt_summary["pilot_paper_prompt_count"] == 70
    assert prompt_summary["target_fpr"] == 0.1
    assert prompt_summary["pilot_paper_negative_count_minimum_required"] == 34
    assert prompt_summary["prompt_split_ready"] is True
    assert prompt_summary["prompt_protocol_name"] == "paper_main_probe_paper_prompt_protocol"
    assert schema["paper_claim_scale"] == "probe_paper"
    assert schema["prompt_set"] == "probe_paper"
    assert schema["target_fpr"] == 0.1
    assert schema["minimum_result_positive_count"] == 34
    assert schema["paper_run_allows_paper_claim"] is True
    assert schema["paper_run_claim_type"] == "probe_claim"
    assert schema["prompt_protocol_name"] == "paper_main_probe_paper_prompt_protocol"
    assert schema["result_protocol_name"] == "probe_paper_fixed_fpr_common_protocol"
    assert all(row["paper_claim_scale"] == "probe_paper" for row in template_rows)
    assert all(row["target_fpr"] == 0.1 for row in template_rows)
    assert all(row["result_claim_scope"] == "probe_claim" for row in template_rows)


def pilot_paper_result_row(schema: dict[str, object], evidence_path: str) -> dict[str, object]:
    """构造一条满足 pilot_paper 导入 schema 的最小结果记录。"""

    attack_config = resolve_formal_attack_config(
        attack_family="standard_distortion",
        attack_name="jpeg_compression",
    )
    row: dict[str, object] = {
        "method_id": "tree_ring",
        "attack_id": attack_config.attack_id,
        "attack_family": "standard_distortion",
        "attack_name": "jpeg_compression",
        "resource_profile": "full_main",
        "attack_config_digest": attack_config_digest(attack_config),
        "target_fpr": 0.01,
        "result_protocol_name": schema["result_protocol_name"],
        "result_scope": schema["result_scope"],
        "result_claim_scope": schema["result_claim_scope"],
        "prompt_protocol_name": schema["prompt_protocol_name"],
        "prompt_split_digest": schema["prompt_split_digest"],
        "attack_matrix_digest": schema["attack_matrix_digest"],
        "fixed_fpr_protocol_digest": schema["fixed_fpr_protocol_digest"],
        "method_threshold_digest": "1" * 64,
        "confidence_interval_method": schema["confidence_interval_method"],
        "confidence_level": schema["confidence_level"],
        "baseline_result_source": evidence_path,
        "baseline_result_source_digest": "digest",
        "evidence_paths": [evidence_path],
        "positive_count": 340,
        "negative_count": 340,
        "attacked_negative_count": 340,
        "attack_record_count": 680,
        "supported_record_count": 340,
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
        "strict_formal_result_ready": True,
        "supports_paper_claim": True,
        "paper_claim_scale": "pilot_paper",
    }
    metric_bounds = schema["metric_bounds"]
    ci_count_fields = schema["ci_count_fields"]
    assert isinstance(metric_bounds, dict)
    assert isinstance(ci_count_fields, dict)
    for low_name, value_name, high_name in schema["ci_field_groups"]:
        lower_bound, upper_bound = metric_bounds[value_name]
        sample_count = int(row[str(ci_count_fields[value_name])])
        low, high = bounded_hoeffding_confidence_interval(
            float(row[value_name]),
            sample_count,
            float(schema["confidence_level"]),
            lower_bound=float(lower_bound),
            upper_bound=float(upper_bound),
        )
        row[low_name] = low
        row[high_name] = high
    return row


@pytest.mark.quick
def test_pilot_paper_import_validator_accepts_governed_confidence_interval_record(tmp_path: Path) -> None:
    """带 Hoeffding 置信区间的 pilot_paper 结果应能进入受治理导入协议。"""

    config = PilotPaperFixedFprConfig()
    prompt_records = build_prompt_records(
        "pilot_paper",
        tuple(f"a controlled city pilot_paper prompt variant {index}" for index in range(700)),
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
    quality_ci = bounded_hoeffding_confidence_interval(
        -0.25,
        340,
        0.95,
        lower_bound=-1.0,
        upper_bound=1.0,
    )
    row["quality_score_mean"] = -0.25
    row["quality_score_ci_low"] = quality_ci[0]
    row["quality_score_ci_high"] = quality_ci[1]

    report = validate_pilot_paper_result_import_rows(
        [row],
        schema,
        evidence_root=tmp_path,
        require_existing_evidence=True,
    )

    assert report["pilot_paper_result_import_ready"] is True
    assert report["accepted_pilot_paper_import_count"] == 1
    assert report["accepted_records"][0]["method_id"] == "tree_ring"
    assert report["accepted_records"][0]["quality_score_mean"] == -0.25
    assert report["accepted_pilot_paper_claim_record_count"] == 1
    assert report["supports_paper_claim"] is True


@pytest.mark.quick
def test_paper_result_schema_rejects_unit_range_ssim_interval() -> None:
    """旧 [0,1] Hoeffding 区间不得冒充 signed-range SSIM 正式区间."""

    config = PilotPaperFixedFprConfig()
    schema = build_pilot_paper_result_import_schema(
        prompt_split_digest="a" * 64,
        attack_matrix_digest="b" * 64,
        fixed_fpr_protocol_digest="c" * 64,
        config=config,
    )
    row = pilot_paper_result_row(schema, "outputs/result.json")
    old_low, old_high = bounded_hoeffding_confidence_interval(
        float(row["quality_score_mean"]),
        int(row["positive_count"]),
        float(schema["confidence_level"]),
    )
    row["quality_score_ci_low"] = old_low
    row["quality_score_ci_high"] = old_high

    report = validate_pilot_paper_result_import_rows([row], schema)

    assert report["pilot_paper_result_import_ready"] is False
    assert "confidence_interval_must_match_bounded_hoeffding" in {
        issue["reason"] for issue in report["issues"]
    }


@pytest.mark.quick
def test_paper_result_schema_rejects_post_labeled_attack_identity(
    tmp_path: Path,
) -> None:
    """结果记录的攻击身份必须与正式 AttackConfig 精确一致."""

    config = PilotPaperFixedFprConfig()
    prompt_records = build_prompt_records(
        "pilot_paper",
        tuple(f"formal prompt {index}" for index in range(700)),
    )
    prompt_summary = build_pilot_paper_prompt_split_summary(prompt_records, config)
    attack_rows = build_pilot_paper_attack_matrix_rows(default_attack_configs(), config)
    schema = build_pilot_paper_result_import_schema(
        prompt_split_digest=prompt_summary["prompt_split_digest"],
        attack_matrix_digest=build_attack_matrix_digest(attack_rows),
        fixed_fpr_protocol_digest=build_fixed_fpr_protocol_digest(config),
        config=config,
    )
    row = pilot_paper_result_row(schema, "outputs/result.json")
    row["attack_id"] = "jpeg_compression_probe"

    report = validate_pilot_paper_result_import_rows(
        [row],
        schema,
        evidence_root=tmp_path,
    )

    assert report["accepted_pilot_paper_import_count"] == 0
    assert "formal_attack_identity_must_match_attack_config" in {
        issue["reason"] for issue in report["issues"]
    }


@pytest.mark.quick
def test_pilot_paper_import_validator_rejects_duplicate_template_key(tmp_path: Path) -> None:
    """行级导入校验器必须阻断重复的 method × attack 模板键。"""

    config = PilotPaperFixedFprConfig()
    prompt_records = build_prompt_records(
        "pilot_paper",
        tuple(f"a controlled city pilot_paper prompt variant {index}" for index in range(700)),
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
        [row, dict(row)],
        schema,
        evidence_root=tmp_path,
        require_existing_evidence=True,
    )

    assert report["pilot_paper_result_import_ready"] is False
    assert report["accepted_pilot_paper_import_count"] == 1
    assert report["pilot_paper_claim_record_ready"] is False
    assert report["supports_paper_claim"] is False
    assert {issue["reason"] for issue in report["issues"]} == {"duplicate_result_template_key"}


@pytest.mark.quick
def test_pilot_paper_import_validator_rejects_full_paper_claim_boundary(tmp_path: Path) -> None:
    """pilot_paper 导入记录不得声明为 full_paper 论文主张。"""

    config = PilotPaperFixedFprConfig()
    prompt_records = build_prompt_records(
        "pilot_paper",
        tuple(f"a controlled city pilot_paper prompt variant {index}" for index in range(700)),
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
    row["result_claim_scope"] = "full_claim"
    row["paper_claim_scale"] = "full_paper"

    report = validate_pilot_paper_result_import_rows([row], schema, evidence_root=tmp_path)
    reasons = {issue["reason"] for issue in report["issues"]}

    assert report["accepted_pilot_paper_import_count"] == 0
    assert "protocol_value_mismatch" in reasons
    assert "pilot_paper_claim_scale_required" in reasons


@pytest.mark.quick
def test_pilot_paper_import_validator_rejects_incomplete_statistical_scale(tmp_path: Path) -> None:
    """低于 pilot_paper fixed-FPR 统计边界的记录不得进入受治理导入协议。"""

    config = PilotPaperFixedFprConfig()
    prompt_records = build_prompt_records(
        "pilot_paper",
        tuple(f"a controlled city pilot_paper prompt variant {index}" for index in range(700)),
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


@pytest.mark.quick
def test_paper_import_validator_rejects_nonformal_marked_result_records(tmp_path: Path) -> None:
    """三层正式论文结果导入都必须拒绝诊断性证据标记。"""

    config = PilotPaperFixedFprConfig()
    prompt_records = build_prompt_records(
        "pilot_paper",
        tuple(f"a controlled city pilot_paper prompt variant {index}" for index in range(700)),
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
    row["metric_status"] = "measured_from_local_proxy"

    report = validate_pilot_paper_result_import_rows([row], schema, evidence_root=tmp_path)
    reasons = {issue["reason"] for issue in report["issues"]}

    assert report["accepted_pilot_paper_import_count"] == 0
    assert "nonformal_result_marker_rejected" in reasons
