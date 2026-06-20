"""验证 attention-relative latent update 的轻量行为。"""

from __future__ import annotations

import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pytest

from main.methods.carrier.attention import derive_attention_relative_carrier, simulate_attention_update_strengths
from scripts.write_attention_latent_update_outputs import write_attention_latent_update_outputs


def sample_attention_graph() -> dict[str, object]:
    """构造最小 attention graph 输入。"""
    return {
        "attention_graph_id": "attention_graph_unit",
        "capture_id": "capture_unit",
        "attention_layer": "attention_unit",
        "attention_map_digest": "a" * 64,
        "attention_shape": [4, 4],
        "stable_token_indices": [0, 1, 2],
        "relative_relation_values": [
            [0, 1, 0.55],
            [1, 0, 0.45],
            [0, 2, 0.35],
            [2, 0, 0.30],
        ],
        "attention_relation_consistency": 0.80,
        "anchor_graph_digest": "b" * 64,
        "unsupported_reason": "",
        "supports_paper_claim": False,
        "metadata": {"attention_matrix_source": "preview", "capture_is_synthetic": False},
    }


def sample_geometry_evidence(reliable: bool = True) -> dict[str, object]:
    """构造最小几何证据输入。"""
    return {
        "geometry_evidence_record_id": "geometry_evidence_unit",
        "attention_graph_id": "attention_graph_unit",
        "capture_id": "capture_unit",
        "attention_relation_consistency": 0.80,
        "anchor_inlier_ratio": 0.75,
        "registration_confidence": 0.80,
        "recovered_sync_consistency": 0.85,
        "alignment_residual": 0.20,
        "geometry_reliable": reliable,
        "direct_positive_decision": False,
        "unsupported_reason": "",
        "supports_paper_claim": False,
        "metadata": {"anchor_graph_digest": "b" * 64},
    }


def sample_subspace_record() -> dict[str, object]:
    """构造最小语义安全子空间输入。"""
    return {
        "prompt_id": "prompt_unit",
        "prompt_set": "probe",
        "split": "test",
        "basis_strategy": "semantic_safe_basis",
        "basis_digest": "c" * 64,
        "route_projection_digest": "d" * 64,
        "selected_indices": [0, 2, 4, 6],
        "subspace_plan_id": "subspace_unit",
        "supports_paper_claim": False,
    }


def sample_route_record() -> dict[str, object]:
    """构造最小语义路由输入。"""
    return {
        "prompt_id": "prompt_unit",
        "route_id": "route_unit",
        "route_digest": "e" * 64,
        "route_label": "semantic_conditioned_route",
        "split": "test",
        "supports_paper_claim": False,
    }


@pytest.mark.quick
def test_attention_relative_carrier_reduces_relation_loss() -> None:
    """可靠几何证据应产生 active update, 且关系损失下降。"""
    carrier = derive_attention_relative_carrier(
        attention_graph=sample_attention_graph(),
        geometry_evidence=sample_geometry_evidence(reliable=True),
        subspace_record=sample_subspace_record(),
        route_record=sample_route_record(),
        vector_width=8,
        embedding_strength=0.08,
    )
    rows = simulate_attention_update_strengths(carrier, (0.0, 1.0))

    assert carrier.fallback_mode == "active_update"
    assert carrier.attention_update_stable is True
    assert carrier.relation_loss_after < carrier.relation_loss_before
    assert carrier.projected_update_norm > 0.0
    assert carrier.supports_paper_claim is False
    assert rows[-1]["relation_loss_delta"] > 0.0


@pytest.mark.quick
def test_attention_relative_carrier_falls_back_when_geometry_is_unreliable() -> None:
    """不可靠几何证据只能保留 evidence-only 边界。"""
    carrier = derive_attention_relative_carrier(
        attention_graph=sample_attention_graph(),
        geometry_evidence=sample_geometry_evidence(reliable=False),
        subspace_record=sample_subspace_record(),
        route_record=sample_route_record(),
        vector_width=8,
        embedding_strength=0.08,
    )

    assert carrier.fallback_mode == "evidence_only"
    assert carrier.unsupported_reason == "geometry_evidence_unreliable"
    assert carrier.attention_update_stable is False
    assert all(value == 0.0 for value in carrier.update_values)


def write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    """写入 JSONL 测试输入。"""
    path.write_text("".join(json.dumps(record, sort_keys=True) + "\n" for record in records), encoding="utf-8")


def write_ready_geometry_package(path: Path) -> None:
    """写入 ready attention geometry 压缩包。"""
    graph = sample_attention_graph()
    evidence = sample_geometry_evidence(reliable=True)
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr(
            "outputs/attention_geometry/geometry_evidence_summary.json",
            json.dumps(
                {
                    "attention_geometry_ready": True,
                    "attention_graph_record_count": 1,
                    "geometry_evidence_record_count": 1,
                    "real_attention_capture_count": 1,
                    "unsupported_capture_count": 0,
                    "protocol_decision": "pass",
                    "supports_paper_claim": False,
                }
            ),
        )
        archive.writestr("outputs/attention_geometry/manifest.local.json", json.dumps({"artifact_id": "attention_geometry_manifest"}))
        archive.writestr("outputs/attention_geometry/attention_graph_records.jsonl", json.dumps(graph, sort_keys=True) + "\n")
        archive.writestr("outputs/attention_geometry/geometry_evidence_records.jsonl", json.dumps(evidence, sort_keys=True) + "\n")


@pytest.mark.quick
def test_attention_latent_update_writer_uses_ready_geometry_package(tmp_path: Path) -> None:
    """写出脚本应从 ready 几何包生成 carrier、stability、quality 和 manifest。"""
    repo_root = tmp_path / "repo"
    semantic_dir = repo_root / "outputs" / "semantic_subspace"
    content_dir = repo_root / "outputs" / "content_carriers"
    semantic_dir.mkdir(parents=True)
    content_dir.mkdir(parents=True)
    (semantic_dir / "manifest.local.json").write_text('{"artifact_id":"semantic"}\n', encoding="utf-8")
    (content_dir / "manifest.local.json").write_text('{"artifact_id":"content"}\n', encoding="utf-8")
    (content_dir / "content_carrier_summary.json").write_text('{"fixed_fpr_ready":true}\n', encoding="utf-8")
    write_jsonl(semantic_dir / "subspace_plan_records.jsonl", [sample_subspace_record()])
    write_jsonl(semantic_dir / "semantic_route_records.jsonl", [sample_route_record()])
    package_path = repo_root / "outputs" / "attention_geometry_package_unit.zip"
    write_ready_geometry_package(package_path)

    manifest = write_attention_latent_update_outputs(
        root=repo_root,
        attention_geometry_package_path=package_path,
        max_subspace_records=1,
    )
    output_dir = repo_root / "outputs" / "attention_latent_update"
    summary = json.loads((output_dir / "attention_update_summary.json").read_text(encoding="utf-8"))

    assert manifest["metadata"]["protocol_decision"] == "pass"
    assert summary["attention_geometry_ready"] is True
    assert summary["attention_carrier_record_count"] == 1
    assert summary["active_update_count"] == 1
    assert summary["image_quality_metrics_ready"] is False
    assert summary["full_method_claim_ready"] is False
    assert (output_dir / "attention_carrier_records.jsonl").exists()
    assert (output_dir / "attention_update_stability.csv").exists()
    assert (output_dir / "attention_update_quality_metrics.csv").exists()
