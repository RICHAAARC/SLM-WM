"""验证注意力图和几何证据的轻量行为。"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from experiments.runtime.diffusion.attention_capture import build_qk_attention_capture_record
from main.methods.geometry import attention_from_query_key, build_attention_graph_record, build_geometry_evidence_record
from scripts.write_attention_geometry_outputs import write_attention_geometry_outputs


@pytest.mark.quick
def test_query_key_attention_rows_are_probabilities() -> None:
    """Q/K 注意力公式应产生逐行归一化的概率矩阵。"""
    query_vectors = ((1.0, 0.0), (0.0, 1.0))
    key_vectors = ((1.0, 0.0), (0.0, 1.0))

    matrix = attention_from_query_key(query_vectors, key_vectors)
    capture = build_qk_attention_capture_record(
        run_id="run_alpha",
        model_family="sd35",
        model_id="model_alpha",
        attention_layer="joint_attention_block",
        query_vectors=query_vectors,
        key_vectors=key_vectors,
        capture_backend="qk_hook",
    )

    assert len(matrix) == 2
    assert all(abs(sum(row) - 1.0) < 1e-12 for row in matrix)
    assert matrix[0][0] > matrix[0][1]
    assert capture.unsupported_reason == ""
    assert capture.metadata["capture_is_synthetic"] is False
    assert capture.metadata["supports_paper_claim"] is False


@pytest.mark.quick
def test_attention_graph_and_geometry_evidence_are_stable() -> None:
    """相同注意力矩阵应产生稳定锚点图和几何证据摘要。"""
    matrix = (
        (0.50, 0.20, 0.20, 0.10),
        (0.25, 0.45, 0.20, 0.10),
        (0.20, 0.25, 0.45, 0.10),
        (0.10, 0.20, 0.20, 0.50),
    )
    first_graph = build_attention_graph_record("capture_alpha", "attention_block", "a" * 64, matrix)
    second_graph = build_attention_graph_record("capture_alpha", "attention_block", "a" * 64, matrix)
    evidence = build_geometry_evidence_record(first_graph)

    assert first_graph.anchor_graph_digest == second_graph.anchor_graph_digest
    assert first_graph.attention_graph_id == second_graph.attention_graph_id
    assert first_graph.stable_token_indices
    assert first_graph.relative_relation_values
    assert 0.0 <= evidence.registration_confidence <= 1.0
    assert 0.0 <= evidence.anchor_inlier_ratio <= 1.0
    assert 0.0 <= evidence.recovered_sync_consistency <= 1.0
    assert 0.0 <= evidence.alignment_residual <= 1.0
    assert evidence.direct_positive_decision is False
    assert evidence.supports_paper_claim is False


def write_attention_inputs(repo_root: Path) -> None:
    """写入注意力几何脚本所需的最小输入。"""
    content_dir = repo_root / "outputs" / "content_carriers"
    runtime_dir = repo_root / "outputs" / "sd_runtime_adapter"
    content_dir.mkdir(parents=True)
    runtime_dir.mkdir(parents=True)
    (content_dir / "manifest.local.json").write_text(json.dumps({"artifact_id": "content"}) + "\n", encoding="utf-8")
    (content_dir / "content_carrier_summary.json").write_text(
        json.dumps({"protocol_decision": "pass", "supports_paper_claim": False}) + "\n",
        encoding="utf-8",
    )
    (runtime_dir / "manifest.local.json").write_text(json.dumps({"artifact_id": "runtime"}) + "\n", encoding="utf-8")
    records = [
        {
            "run_id": "run_alpha",
            "model_family": "sd35",
            "model_id": "model_alpha",
            "capture_id": "capture_alpha",
            "attention_layer": "attention_early",
            "attention_map_digest": "1" * 64,
            "attention_shape": [4, 4],
            "attention_mean": 0.25,
            "attention_entropy": 1.0,
            "capture_backend": "qk_hook",
            "unsupported_reason": "",
            "metadata": {"capture_is_synthetic": False, "supports_paper_claim": False},
        },
        {
            "run_id": "run_beta",
            "model_family": "sd35",
            "model_id": "model_beta",
            "capture_id": "capture_beta",
            "attention_layer": "attention_late",
            "attention_map_digest": "2" * 64,
            "attention_shape": [4, 4],
            "attention_mean": 0.25,
            "attention_entropy": 1.0,
            "capture_backend": "digest_replay",
            "unsupported_reason": "real_attention_hook_unavailable",
            "metadata": {"capture_is_synthetic": True, "supports_paper_claim": False},
        },
    ]
    (runtime_dir / "attention_capture_records.jsonl").write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


@pytest.mark.quick
def test_attention_geometry_writer_creates_governed_outputs(tmp_path: Path) -> None:
    """注意力几何脚本应生成 records, table, summary 与 manifest。"""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    write_attention_inputs(repo_root)

    manifest = write_attention_geometry_outputs(root=repo_root)
    output_dir = repo_root / "outputs" / "attention_geometry"
    summary = json.loads((output_dir / "geometry_evidence_summary.json").read_text(encoding="utf-8"))
    graph_records = (output_dir / "attention_graph_records.jsonl").read_text(encoding="utf-8").splitlines()
    evidence_records = (output_dir / "geometry_evidence_records.jsonl").read_text(encoding="utf-8").splitlines()
    relation_rows = list(csv.DictReader((output_dir / "attention_relation_consistency.csv").open(encoding="utf-8")))

    assert manifest["metadata"]["protocol_decision"] == "pass"
    assert manifest["metadata"]["supports_paper_claim"] is False
    assert summary["attention_capture_record_count"] == 2
    assert summary["attention_graph_record_count"] == 2
    assert summary["geometry_evidence_record_count"] == 2
    assert summary["real_attention_capture_count"] == 1
    assert summary["unsupported_capture_count"] == 1
    assert summary["direct_positive_decision_used"] is False
    assert summary["attention_geometry_ready"] is False
    assert len(graph_records) == 2
    assert len(evidence_records) == 2
    assert len(relation_rows) == 2
    assert set(manifest["output_paths"]) == {
        "outputs/attention_geometry/attention_graph_records.jsonl",
        "outputs/attention_geometry/geometry_evidence_records.jsonl",
        "outputs/attention_geometry/attention_relation_consistency.csv",
        "outputs/attention_geometry/geometry_evidence_summary.json",
        "outputs/attention_geometry/manifest.local.json",
    }


@pytest.mark.quick
def test_attention_geometry_writer_rejects_output_outside_outputs(tmp_path: Path) -> None:
    """注意力几何脚本应拒绝 outputs 之外的持久化目录。"""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()

    with pytest.raises(ValueError):
        write_attention_geometry_outputs(root=repo_root, output_dir=repo_root / "outside")
