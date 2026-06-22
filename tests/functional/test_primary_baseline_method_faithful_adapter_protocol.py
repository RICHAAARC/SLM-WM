"""主表 external baseline 方法忠实 SD3.5 适配协议的轻量功能测试。"""

from __future__ import annotations

import json
from pathlib import Path
from zipfile import ZipFile

import pytest

from experiments.baselines import (
    METHOD_FAITHFUL_ADAPTER_BOUNDARY,
    build_method_faithful_adapter_status_records,
    build_method_faithful_adapter_summary,
    build_primary_baseline_method_faithful_adapter_schema,
)
from scripts.write_primary_baseline_method_faithful_adapter_protocol import (
    write_primary_baseline_method_faithful_adapter_protocol_outputs,
)


def observation_row(
    *,
    baseline_id: str,
    sample_role: str,
    adapter_boundary: str,
    detection_decision: bool,
    image_index: int,
) -> dict[str, object]:
    """构造满足协议字段要求的最小 observation。"""

    return {
        "event_id": f"{baseline_id}_{sample_role}_{image_index}",
        "baseline_id": baseline_id,
        "score": 0.91 if detection_decision else 0.02,
        "threshold": 0.5,
        "score_name": "baseline_detection_score",
        "higher_is_positive": True,
        "detection_decision": detection_decision,
        "sample_role": sample_role,
        "attack_family": "clean",
        "attack_condition": "clean_none",
        "prompt_id": f"prompt_{image_index}",
        "image_id": f"{baseline_id}_image_{image_index}",
        "adapter_boundary": adapter_boundary,
        "formal_result_claim": False,
        "supports_paper_claim": False,
        "image_path": f"outputs/{baseline_id}/{sample_role}_{image_index}.png",
        "image_digest": f"digest_{baseline_id}_{sample_role}_{image_index}",
    }


def mixed_observations() -> list[dict[str, object]]:
    """构造 Tree-Ring ready、Gaussian smoke 被拒绝、T2SMark native 的混合记录。"""

    rows: list[dict[str, object]] = []
    for index, sample_role in enumerate(("clean_negative", "positive_source")):
        rows.append(
            observation_row(
                baseline_id="tree_ring",
                sample_role=sample_role,
                adapter_boundary=METHOD_FAITHFUL_ADAPTER_BOUNDARY,
                detection_decision=sample_role == "positive_source",
                image_index=index,
            )
        )
        rows.append(
            observation_row(
                baseline_id="gaussian_shading",
                sample_role=sample_role,
                adapter_boundary="sd35_latent_smoke_adapter_not_formal_external_baseline_evidence",
                detection_decision=sample_role == "positive_source",
                image_index=index,
            )
        )
        rows.append(
            observation_row(
                baseline_id="t2smark",
                sample_role=sample_role,
                adapter_boundary="sd35_medium_native_official_reproduction",
                detection_decision=sample_role == "positive_source",
                image_index=index,
            )
        )
    return rows


@pytest.mark.quick
def test_method_faithful_adapter_schema_freezes_primary_baseline_boundary() -> None:
    """schema 应明确主表 adapter 边界、native baseline 和不合格 smoke 边界。"""

    schema = build_primary_baseline_method_faithful_adapter_schema()

    assert schema["protocol_name"] == "primary_baseline_method_faithful_adapter_protocol"
    assert schema["method_faithful_adapter_required_ids"] == ["tree_ring", "gaussian_shading", "shallow_diffuse"]
    assert schema["native_official_reproduction_ids"] == ["t2smark"]
    assert schema["accepted_adapter_boundary"] == METHOD_FAITHFUL_ADAPTER_BOUNDARY
    assert "sd35_latent_smoke_adapter_not_formal_external_baseline_evidence" in schema["rejected_smoke_adapter_boundaries"]
    assert schema["supports_paper_claim"] is False


@pytest.mark.quick
def test_method_faithful_adapter_records_reject_smoke_boundary() -> None:
    """协议记录应允许 Tree-Ring method-faithful 候选, 同时拒绝 Gaussian smoke observation。"""

    records = build_method_faithful_adapter_status_records(mixed_observations())
    summary = build_method_faithful_adapter_summary(records)

    tree_ring = next(row for row in records if row["baseline_id"] == "tree_ring")
    gaussian = next(row for row in records if row["baseline_id"] == "gaussian_shading")
    shallow = next(row for row in records if row["baseline_id"] == "shallow_diffuse")
    t2smark = next(row for row in records if row["baseline_id"] == "t2smark")

    assert tree_ring["method_faithful_adapter_ready"] is True
    assert tree_ring["formal_import_candidate_allowed"] is True
    assert tree_ring["score_protocol_ready"] is True
    assert tree_ring["image_provenance_ready"] is True
    assert gaussian["method_faithful_adapter_ready"] is False
    assert "smoke_adapter_boundary_rejected" in gaussian["blocking_reasons"]
    assert shallow["method_faithful_adapter_ready"] is False
    assert "method_faithful_observations_missing" in shallow["blocking_reasons"]
    assert t2smark["protocol_role"] == "native_official_reproduction"
    assert t2smark["formal_import_candidate_allowed"] is False
    assert summary["method_faithful_adapter_ready_ids"] == ["tree_ring"]
    assert summary["missing_method_faithful_adapter_ids"] == ["gaussian_shading", "shallow_diffuse"]
    assert summary["method_faithful_adapter_protocol_ready"] is False
    assert summary["supports_paper_claim"] is False


@pytest.mark.quick
def test_method_faithful_adapter_writer_can_read_smoke_package(tmp_path: Path) -> None:
    """协议写出脚本应可从 GPU smoke zip 包读取 observation 并写出治理产物。"""

    observations_path = tmp_path / "outputs" / "external_baseline_gpu_smoke" / "execution" / "baseline_observations.json"
    observations_path.parent.mkdir(parents=True)
    observations_path.write_text(json.dumps(mixed_observations(), ensure_ascii=False), encoding="utf-8")
    package_path = tmp_path / "outputs" / "external_baseline_gpu_smoke_package.zip"
    with ZipFile(package_path, "w") as archive:
        archive.write(observations_path, "outputs/external_baseline_gpu_smoke/execution/baseline_observations.json")
    observations_path.unlink()

    manifest = write_primary_baseline_method_faithful_adapter_protocol_outputs(
        root=tmp_path,
        smoke_package_path=package_path,
    )

    output_dir = tmp_path / "outputs" / "primary_baseline_method_faithful_adapter_protocol"
    records = [
        json.loads(line)
        for line in (output_dir / "primary_baseline_method_faithful_adapter_status_records.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    summary = json.loads(
        (output_dir / "primary_baseline_method_faithful_adapter_summary.json").read_text(encoding="utf-8")
    )

    assert manifest["artifact_id"] == "primary_baseline_method_faithful_adapter_protocol_manifest"
    assert len(records) == 4
    assert summary["input_observation_count"] == 6
    assert summary["formal_import_candidate_allowed_ids"] == ["tree_ring"]
    assert summary["smoke_package_path"] == "outputs/external_baseline_gpu_smoke_package.zip"
    assert all(str(path).startswith("outputs/") for path in manifest["output_paths"])
