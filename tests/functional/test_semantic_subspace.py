"""验证语义掩码与安全子空间方法。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from main.methods.semantic import build_risk_field, build_semantic_route, project_mask_to_latent
from main.methods.subspace import build_safe_basis_plan, build_trajectory_features, estimate_approximate_jvp
from scripts.write_semantic_subspace_outputs import write_semantic_subspace_outputs


@pytest.mark.quick
def test_different_masks_produce_different_routes() -> None:
    """不同语义掩码应产生不同 route 摘要。"""
    latent_values = (0.2, -0.1, 0.4, -0.3, 0.6, -0.2, 0.1, -0.5)
    risk_field = build_risk_field(
        semantic_values=(0.2, 0.3, 0.4, 0.5, 0.7, 0.6, 0.2, 0.3),
        texture_values=(0.2, 0.3, 0.7, 0.8, 0.9, 0.6, 0.2, 0.7),
        stability_values=(0.8, 0.7, 0.9, 0.6, 0.8, 0.5, 0.7, 0.9),
        saliency_values=(0.2, 0.2, 0.3, 0.3, 0.6, 0.5, 0.2, 0.4),
    )
    first_mask = project_mask_to_latent(latent_values, (0.9, 0.7, 0.5, 0.3), "prompt_semantic_feature_mask")
    second_mask = project_mask_to_latent(latent_values, (0.2, 0.4, 0.8, 1.0), "prompt_semantic_feature_mask")
    first_route = build_semantic_route("prompt_a", "balanced_scene", risk_field, first_mask)
    second_route = build_semantic_route("prompt_b", "balanced_scene", risk_field, second_mask)

    assert first_route.route_digest != second_route.route_digest
    assert first_route.supports_paper_claim is False
    assert second_route.supports_paper_claim is False


@pytest.mark.quick
def test_semantic_mask_changes_safe_basis() -> None:
    """关闭语义掩码后安全基底应发生变化。"""
    latent_values = (0.2, -0.1, 0.4, -0.3, 0.6, -0.2, 0.1, -0.5)
    risk_field = build_risk_field(
        semantic_values=(0.8, 0.3, 0.4, 0.2, 0.7, 0.6, 0.1, 0.3),
        texture_values=(0.2, 0.6, 0.7, 0.8, 0.2, 0.6, 0.4, 0.7),
        stability_values=(0.8, 0.4, 0.9, 0.6, 0.3, 0.5, 0.7, 0.9),
        saliency_values=(0.6, 0.2, 0.3, 0.8, 0.6, 0.1, 0.2, 0.4),
    )
    latent_mask = project_mask_to_latent(latent_values, (0.2, 0.9, 0.4, 0.8), "prompt_semantic_feature_mask")
    route = build_semantic_route("prompt_a", "human_centric", risk_field, latent_mask)
    features = build_trajectory_features(latent_mask)
    jvp = estimate_approximate_jvp(features)
    semantic_basis = build_safe_basis_plan(features, jvp, risk_field, route)
    no_mask_basis = build_safe_basis_plan(
        features,
        jvp,
        risk_field,
        route,
        semantic_mask_enabled=False,
        basis_strategy="no_semantic_mask",
    )
    global_basis = build_safe_basis_plan(features, jvp, risk_field, route, basis_strategy="global_nullspace")
    diagnostic_basis = build_safe_basis_plan(features, jvp, risk_field, route, basis_strategy="diagnostic_basis")

    assert semantic_basis.basis_digest != no_mask_basis.basis_digest
    assert semantic_basis.basis_digest
    assert no_mask_basis.basis_digest
    assert global_basis.basis_digest
    assert diagnostic_basis.basis_digest
    assert jvp.approximate_jvp_digest


def write_prompt_records(repo_root: Path) -> None:
    """写入最小 prompt records, 用于测试脚本输出。"""
    output_dir = repo_root / "outputs" / "prompt_event_protocol"
    output_dir.mkdir(parents=True)
    records = [
        {
            "prompt_id": "prompt_alpha",
            "prompt_set": "probe",
            "prompt_index": 0,
            "prompt_text": "a cat near a window",
            "prompt_digest": "a" * 64,
            "semantic_tags": ["animal", "object"],
            "risk_profile": "animal_centric",
            "split": "calibration",
            "supports_paper_claim": False,
        },
        {
            "prompt_id": "prompt_beta",
            "prompt_set": "probe",
            "prompt_index": 1,
            "prompt_text": "a boat on a lake",
            "prompt_digest": "b" * 64,
            "semantic_tags": ["water"],
            "risk_profile": "natural_scene",
            "split": "test",
            "supports_paper_claim": False,
        },
    ]
    (output_dir / "prompt_records.jsonl").write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    (output_dir / "manifest.local.json").write_text("{}\n", encoding="utf-8")


@pytest.mark.quick
def test_semantic_subspace_writer_outputs_manifest(tmp_path: Path) -> None:
    """语义子空间写出脚本应生成可重建 manifest。"""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    write_prompt_records(repo_root)

    manifest = write_semantic_subspace_outputs(root=repo_root)
    output_dir = repo_root / "outputs" / "semantic_subspace"
    summary = json.loads((output_dir / "semantic_subspace_summary.json").read_text(encoding="utf-8"))

    assert manifest["metadata"]["protocol_decision"] == "pass"
    assert manifest["metadata"]["supports_paper_claim"] is False
    assert summary["semantic_route_record_count"] == 2
    assert summary["semantic_mask_changed_basis_count"] == 2
    assert (output_dir / "mask_projection_reports" / "mask_projection_reports.jsonl").exists()


@pytest.mark.quick
def test_semantic_subspace_writer_rejects_non_outputs_dir(tmp_path: Path) -> None:
    """语义子空间写出脚本应拒绝 outputs 之外的目录。"""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    write_prompt_records(repo_root)

    with pytest.raises(ValueError):
        write_semantic_subspace_outputs(root=repo_root, output_dir=repo_root / "outside")
