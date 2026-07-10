"""验证 LF/尾部截断内容载体与统一内容分数。"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from main.methods.carrier import (
    CONTENT_MODES,
    compose_content_update,
    derive_tail_content_carrier,
    derive_lf_content_carrier,
)
from main.methods.detection import compute_unified_content_score
from experiments.artifacts.content_carrier_outputs import write_content_carrier_outputs


SELECTED_INDICES = (0, 2, 5, 7)
BASIS_DIGEST = "a" * 64
ROUTE_DIGEST = "b" * 64
EVENT_DIGEST = "c" * 64
KEY_MATERIAL = "unit_content_key"
VECTOR_WIDTH = 8


def build_lf_tail_carriers():
    """构造稳定的 LF 与尾部截断载体, 供多个测试复用。"""
    lf_carrier = derive_lf_content_carrier(
        selected_indices=SELECTED_INDICES,
        basis_digest=BASIS_DIGEST,
        route_digest=ROUTE_DIGEST,
        event_digest=EVENT_DIGEST,
        key_material=KEY_MATERIAL,
        vector_width=VECTOR_WIDTH,
    )
    tail_carrier = derive_tail_content_carrier(
        selected_indices=SELECTED_INDICES,
        basis_digest=BASIS_DIGEST,
        route_digest=ROUTE_DIGEST,
        event_digest=EVENT_DIGEST,
        key_material=KEY_MATERIAL,
        vector_width=VECTOR_WIDTH,
        tail_fraction=0.25,
    )
    tail_without_truncation = derive_tail_content_carrier(
        selected_indices=SELECTED_INDICES,
        basis_digest=BASIS_DIGEST,
        route_digest=ROUTE_DIGEST,
        event_digest=EVENT_DIGEST,
        key_material=KEY_MATERIAL,
        vector_width=VECTOR_WIDTH,
        tail_fraction=0.25,
        tail_truncation_enabled=False,
    )
    return lf_carrier, tail_carrier, tail_without_truncation


@pytest.mark.quick
def test_lf_tail_carrier_digests_are_stable() -> None:
    """相同输入应导出相同载体摘要, 并保持非 claim 边界。"""
    first_lf, first_tail, first_tail_without_truncation = build_lf_tail_carriers()
    second_lf, second_tail, second_tail_without_truncation = build_lf_tail_carriers()

    assert first_lf.lf_content_carrier_digest == second_lf.lf_content_carrier_digest
    assert first_tail.tail_content_carrier_digest == second_tail.tail_content_carrier_digest
    assert first_tail_without_truncation.tail_content_carrier_digest == second_tail_without_truncation.tail_content_carrier_digest
    assert first_lf.update_values == second_lf.update_values
    assert first_tail.update_values == second_tail.update_values
    assert first_lf.supports_paper_claim is False
    assert first_tail.supports_paper_claim is False
    assert first_tail.retained_fraction < 1.0
    assert first_tail_without_truncation.retained_fraction == 1.0


@pytest.mark.quick
def test_content_modes_change_updates_without_branch_vote() -> None:
    """机制开关应真实改变组合 update, 但不引入 LF/尾部截断独立阈值投票。"""
    lf_carrier, tail_carrier, tail_without_truncation = build_lf_tail_carriers()
    updates = {
        "full_content_chain": compose_content_update(lf_carrier, tail_carrier, "full_content_chain"),
        "lf_only": compose_content_update(lf_carrier, tail_carrier, "lf_only"),
        "tail_only": compose_content_update(lf_carrier, tail_carrier, "tail_only"),
        "no_tail": compose_content_update(lf_carrier, tail_carrier, "no_tail"),
        "no_tail_truncation": compose_content_update(lf_carrier, tail_without_truncation, "no_tail_truncation"),
        "no_lf": compose_content_update(lf_carrier, tail_carrier, "no_lf"),
    }
    full_update = updates["full_content_chain"]

    assert set(updates) == set(CONTENT_MODES)
    assert updates["lf_only"].tail_enabled is False
    assert updates["tail_only"].lf_enabled is False
    assert updates["no_tail"].tail_enabled is False
    assert updates["no_lf"].lf_enabled is False
    assert updates["no_tail_truncation"].tail_truncation_enabled is False
    for name, update in updates.items():
        assert len(update.combined_update_values) == VECTOR_WIDTH
        assert update.supports_paper_claim is False
        if name != "full_content_chain":
            assert update.content_update_digest != full_update.content_update_digest
    assert updates["lf_only"].combined_update_values == updates["no_tail"].combined_update_values
    assert updates["tail_only"].combined_update_values == updates["no_lf"].combined_update_values
    assert updates["no_tail_truncation"].combined_update_values != full_update.combined_update_values


@pytest.mark.quick
def test_unified_content_score_keeps_fixed_fpr_boundary() -> None:
    """正式内容分数应同时约束 combined 方向和 LF/尾部截断分支一致性。"""
    lf_carrier, tail_carrier, _ = build_lf_tail_carriers()
    content_update = compose_content_update(lf_carrier, tail_carrier, "full_content_chain")
    score = compute_unified_content_score(content_update.combined_update_values, content_update)

    assert score.lambda_lf > score.lambda_tail
    assert score.used_independent_branch_vote is False
    assert score.fixed_fpr_ready is True
    assert score.supports_paper_claim is False
    assert score.content_score == pytest.approx(min(score.combined_score, score.lf_tail_fusion_score))
    assert score.lf_tail_fusion_score == pytest.approx(0.70 * score.lf_score + 0.30 * score.tail_score)
    assert score.content_score > 0.0
    with pytest.raises(ValueError):
        compute_unified_content_score(content_update.combined_update_values, content_update, lambda_lf=0.40, lambda_tail=0.60)


@pytest.mark.quick
def test_formal_score_uses_branch_consistency_guard() -> None:
    """正式分数应被 LF/尾部截断一致性门控压低, 避免 wrong-key 高尾。"""
    lf_carrier, tail_carrier, _ = build_lf_tail_carriers()
    content_update = compose_content_update(lf_carrier, tail_carrier, "full_content_chain")
    observed_values = content_update.combined_update_values

    score = compute_unified_content_score(observed_values, content_update)

    assert score.combined_score == pytest.approx(1.0)
    assert score.content_score == pytest.approx(score.lf_tail_fusion_score)
    assert score.content_score <= score.combined_score


def write_semantic_inputs(repo_root: Path) -> None:
    """写入内容载体脚本所需的最小语义输入。"""
    semantic_dir = repo_root / "outputs" / "semantic_subspace"
    semantic_dir.mkdir(parents=True)
    subspace_records = [
        {
            "basis_digest": "1" * 64,
            "basis_strategy": "semantic_safe_basis",
            "prompt_id": "prompt_alpha",
            "prompt_set": "probe",
            "route_projection_digest": "2" * 64,
            "selected_indices": [0, 2, 5, 7],
            "semantic_mask_enabled": True,
            "split": "calibration",
            "subspace_plan_id": "plan_alpha",
            "supports_paper_claim": False,
        },
        {
            "basis_digest": "3" * 64,
            "basis_strategy": "semantic_safe_basis",
            "prompt_id": "prompt_beta",
            "prompt_set": "probe",
            "route_projection_digest": "4" * 64,
            "selected_indices": [1, 3, 4, 6],
            "semantic_mask_enabled": True,
            "split": "test",
            "subspace_plan_id": "plan_beta",
            "supports_paper_claim": False,
        },
    ]
    route_records = [
        {
            "prompt_id": "prompt_alpha",
            "prompt_set": "probe",
            "route_digest": "5" * 64,
            "route_id": "route_alpha",
            "split": "calibration",
            "supports_paper_claim": False,
        },
        {
            "prompt_id": "prompt_beta",
            "prompt_set": "probe",
            "route_digest": "6" * 64,
            "route_id": "route_beta",
            "split": "test",
            "supports_paper_claim": False,
        },
    ]
    (semantic_dir / "subspace_plan_records.jsonl").write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in subspace_records),
        encoding="utf-8",
    )
    (semantic_dir / "semantic_route_records.jsonl").write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in route_records),
        encoding="utf-8",
    )
    (semantic_dir / "manifest.local.json").write_text("{}\n", encoding="utf-8")


@pytest.mark.quick
def test_content_carrier_writer_creates_governed_outputs(tmp_path: Path) -> None:
    """内容载体写出脚本应生成 records, tables, summary 与 manifest。"""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    write_semantic_inputs(repo_root)

    manifest = write_content_carrier_outputs(root=repo_root, max_records=2)
    output_dir = repo_root / "outputs" / "content_carriers"
    summary = json.loads((output_dir / "content_carrier_summary.json").read_text(encoding="utf-8"))
    records = (output_dir / "content_detection_records.jsonl").read_text(encoding="utf-8").splitlines()
    score_rows = list(csv.DictReader((output_dir / "lf_tail_score_table.csv").open(encoding="utf-8")))
    distribution_rows = list(csv.DictReader((output_dir / "content_score_distribution.csv").open(encoding="utf-8")))

    assert manifest["metadata"]["protocol_decision"] == "pass"
    assert manifest["metadata"]["supports_paper_claim"] is False
    assert summary["content_detection_record_count"] == 6
    assert summary["score_count"] == 6
    assert summary["fixed_fpr_ready"] is True
    assert summary["used_independent_branch_vote"] is False
    assert summary["supports_paper_claim"] is False
    assert len(records) == 6
    assert len(score_rows) == 6
    assert distribution_rows
    assert (output_dir / "paired_quality_metrics.csv").exists()
    assert set(manifest["output_paths"]) == {
        "outputs/content_carriers/content_detection_records.jsonl",
        "outputs/content_carriers/lf_tail_score_table.csv",
        "outputs/content_carriers/paired_quality_metrics.csv",
        "outputs/content_carriers/content_score_distribution.csv",
        "outputs/content_carriers/content_carrier_summary.json",
        "outputs/content_carriers/manifest.local.json",
    }


@pytest.mark.quick
def test_content_carrier_writer_rejects_output_outside_outputs(tmp_path: Path) -> None:
    """内容载体写出脚本应拒绝 outputs 之外的持久化目录。"""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()

    with pytest.raises(ValueError):
        write_content_carrier_outputs(root=repo_root, output_dir=repo_root / "outside")
