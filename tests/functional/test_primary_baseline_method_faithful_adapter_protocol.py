"""主表 external baseline 方法忠实 SD3.5 适配协议的轻量功能测试。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from paper_experiments.baselines import (
    METHOD_FAITHFUL_ADAPTER_BOUNDARY,
    build_method_faithful_adapter_status_records,
    build_method_faithful_adapter_summary,
    build_primary_baseline_method_faithful_adapter_schema,
)
from scripts.write_primary_baseline_method_faithful_adapter_protocol import (
    write_primary_baseline_method_faithful_adapter_protocol_outputs,
)
from paper_experiments.baselines.method_faithful_observation_collection import (
    METHOD_FAITHFUL_BASELINE_IDS,
)
from tests.helpers.method_faithful_collection import (
    formal_observation_rows,
    write_complete_collection,
    write_current_paper_protocol,
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
    """构造 Tree-Ring ready、Gaussian incomplete adapter 被拒绝、T2SMark native 的混合记录。"""

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
                adapter_boundary="sd35_method_faithful_adapter_not_formal_external_baseline_evidence",
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
    """schema 应明确主表 adapter 边界、native baseline 和不合格 adapter 边界。"""

    schema = build_primary_baseline_method_faithful_adapter_schema()

    assert schema["protocol_name"] == "primary_baseline_method_faithful_adapter_protocol"
    assert schema["method_faithful_adapter_required_ids"] == ["tree_ring", "gaussian_shading", "shallow_diffuse"]
    assert schema["native_official_reproduction_ids"] == ["t2smark"]
    assert schema["accepted_adapter_boundary"] == METHOD_FAITHFUL_ADAPTER_BOUNDARY
    assert "sd35_method_faithful_adapter_not_formal_external_baseline_evidence" in schema["rejected_incomplete_adapter_boundaries"]
    assert schema["supports_paper_claim"] is False


@pytest.mark.quick
def test_method_faithful_adapter_records_reject_incomplete_adapter_boundary() -> None:
    """协议记录应允许 Tree-Ring method-faithful 候选, 同时拒绝 Gaussian method-faithful observation。"""

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
    assert "incomplete_adapter_boundary_rejected" in gaussian["blocking_reasons"]
    assert shallow["method_faithful_adapter_ready"] is False
    assert "method_faithful_observations_missing" in shallow["blocking_reasons"]
    assert t2smark["protocol_role"] == "native_official_reproduction"
    assert t2smark["formal_import_candidate_allowed"] is False
    assert summary["method_faithful_adapter_ready_ids"] == ["tree_ring"]
    assert summary["missing_method_faithful_adapter_ids"] == ["gaussian_shading", "shallow_diffuse"]
    assert summary["method_faithful_adapter_protocol_ready"] is False
    assert summary["supports_paper_claim"] is False


def write_exact_collection(
    collection_root: Path,
    prompts: list[dict[str, object]],
    protocol: object,
) -> None:
    """写出三个 baseline 的 exact-set observation collection。"""

    write_complete_collection(
        collection_root,
        {
            baseline_id: formal_observation_rows(baseline_id, prompts, protocol)
            for baseline_id in METHOD_FAITHFUL_BASELINE_IDS
        },
        prompts,
        protocol,
    )


@pytest.mark.quick
def test_method_faithful_adapter_writer_reads_exact_collection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """协议写出脚本必须从三个 baseline 的 exact-set collection 读取。"""

    collection_root = tmp_path / "outputs" / "external_baseline_method_faithful"
    monkeypatch.setenv("SLM_WM_PAPER_RUN_NAME", "probe_paper")
    prompts, protocol = write_current_paper_protocol(tmp_path)
    write_exact_collection(collection_root, prompts, protocol)

    manifest = write_primary_baseline_method_faithful_adapter_protocol_outputs(
        root=tmp_path,
        collection_root=collection_root,
    )

    output_dir = (
        tmp_path
        / "outputs"
        / "primary_baseline_method_faithful_adapter_protocol"
        / "probe_paper"
    )
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
    assert summary["input_observation_count"] > 6
    assert summary["formal_import_candidate_allowed_ids"] == list(METHOD_FAITHFUL_BASELINE_IDS)
    assert summary["input_baseline_ids"] == list(METHOD_FAITHFUL_BASELINE_IDS)
    assert all(str(path).startswith("outputs/") for path in manifest["output_paths"])

