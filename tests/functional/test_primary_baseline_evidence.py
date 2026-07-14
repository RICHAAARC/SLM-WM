"""主表 external baseline 证据边界审计的轻量功能测试。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from paper_experiments.baselines import build_primary_baseline_evidence_records, build_primary_baseline_evidence_summary
from scripts.write_primary_baseline_evidence_outputs import write_primary_baseline_evidence_outputs
from paper_experiments.baselines.method_faithful_observation_collection import (
    METHOD_FAITHFUL_BASELINE_IDS,
    canonical_prompt_protocol_digest,
)
from tests.helpers.method_faithful_collection import (
    formal_observation_rows,
    write_complete_collection,
    write_current_paper_protocol,
)


def write_source_registry(tmp_path: Path) -> Path:
    """写出包含四个主表 baseline 的最小来源登记。"""

    rows = []
    for baseline_id in ("tree_ring", "gaussian_shading", "shallow_diffuse", "t2smark"):
        rows.append(
            {
                "baseline_id": baseline_id,
                "comparison_group": "primary",
                "source_status": "downloaded",
                "source_dir": f"external_baseline/primary/{baseline_id}/source",
                "official_repository_commit": f"{baseline_id}_commit",
                "adapter_status": "sd35_native_result_adapter_ready"
                if baseline_id == "t2smark"
                else "sd35_method_faithful_adapter_ready",
                "model_alignment_status": "sd35_medium_native_entrypoint"
                if baseline_id == "t2smark"
                else "sd35_medium_adapter_required",
            }
        )
    registry_path = tmp_path / "external_baseline" / "source_registry.json"
    registry_path.parent.mkdir(parents=True)
    registry_path.write_text(json.dumps({"baseline_sources": rows}, ensure_ascii=False), encoding="utf-8")
    return registry_path


def write_method_faithful_outputs(tmp_path: Path) -> tuple[Path, Path]:
    """写出四个主表 baseline 的最小 method-faithful 命令结果和 observation。"""

    output_dir = tmp_path / "outputs" / "external_baseline_method_faithful" / "execution"
    output_dir.mkdir(parents=True)
    command_results = []
    observations = []
    for baseline_id in ("tree_ring", "gaussian_shading", "shallow_diffuse", "t2smark"):
        command_results.append(
            {
                "baseline_id": baseline_id,
                "return_code": 0,
                "observation_count": 2,
                "output_path": f"outputs/{baseline_id}.json",
            }
        )
        for sample_role in ("clean_negative", "positive_source"):
            observations.append(
                {
                    "event_id": f"{baseline_id}_{sample_role}",
                    "baseline_id": baseline_id,
                    "sample_role": sample_role,
                    "execution_device": "cuda" if baseline_id != "t2smark" else "",
                    "latent_shape": [16, 64, 64] if baseline_id != "t2smark" else [],
                }
            )
    command_results_path = output_dir / "baseline_command_results.json"
    observations_path = output_dir / "baseline_observations.json"
    command_results_path.write_text(json.dumps(command_results, ensure_ascii=False), encoding="utf-8")
    observations_path.write_text(json.dumps(observations, ensure_ascii=False), encoding="utf-8")
    return command_results_path, observations_path


def write_evidence_collection(
    tmp_path: Path,
    prompts: list[dict[str, object]],
    protocol: object,
) -> Path:
    """写出三个 method-faithful baseline 的 exact-set transfer 集合。"""

    collection_root = tmp_path / "outputs" / "external_baseline_method_faithful"
    write_complete_collection(
        collection_root,
        {
            baseline_id: formal_observation_rows(baseline_id, prompts, protocol)
            for baseline_id in METHOD_FAITHFUL_BASELINE_IDS
        },
        prompts,
        protocol,
    )
    return collection_root


def write_t2smark_formal_evidence(
    tmp_path: Path,
    prompts: list[dict[str, object]],
    protocol: object,
) -> Path:
    """写出独立 T2SMark formal runner 的完整受治理测试证据。"""

    output_dir = tmp_path / "outputs" / "t2smark_formal_reproduction"
    official_results = (
        output_dir
        / "t2smark_official"
        / "t2smark_sd35_medium_probe_paper"
        / "results.json"
    )
    adapter_dir = output_dir / "t2smark_adapter"
    observations_path = adapter_dir / "baseline_observations.json"
    adapter_manifest_path = adapter_dir / "t2smark_slm_adapter_manifest.json"
    prompt_plan_path = output_dir / "t2smark_formal_prompt_plan.json"
    rows = formal_observation_rows("t2smark", prompts, protocol)
    required_attacks = sorted(
        {
            str(row["attack_name"])
            for row in rows
            if str(row["sample_role"]).startswith("attacked_")
        }
    )
    for path, payload in (
        (official_results, {"formal_results": True}),
        (prompt_plan_path, prompts),
        (output_dir / "t2smark_formal_image_pairs.json", [{"strict_pair_quality_ready": True}]),
        (observations_path, rows),
        (
            adapter_manifest_path,
            {
                "artifact_name": "t2smark_slm_adapter_manifest.json",
                "baseline_id": "t2smark",
                "adapter_status": "sd35_native_result_adapter_ready",
                "observation_count": len(rows),
                "strict_pair_quality_ready": True,
                "missing_result_indices": [],
                "formal_attack_names": required_attacks,
                "threshold_source": "nested_calibration_threshold_freeze_conformal_v1",
            },
        ),
        (
            output_dir / "t2smark_formal_adapter_command_result.json",
            {
                "command": [
                    "python",
                    "external_baseline/primary/t2smark/adapter/run_slm_eval.py",
                    "--num-inference-steps",
                    "20",
                    "--num-inversion-steps",
                    "20",
                    "--guidance-scale",
                    "4.5",
                    "--target-fpr",
                    "0.1",
                ],
                "return_code": 0,
            },
        ),
        (
            output_dir / "t2smark_formal_import_validation_report.json",
            {"formal_import_validation_ready": True},
        ),
        (
            output_dir / "t2smark_formal_strict_pair_quality_summary.json",
            {"strict_pair_quality_ready": True},
        ),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    prompt_digest = canonical_prompt_protocol_digest(prompts)
    summary = {
        "run_decision": "pass",
        "t2smark_formal_reproduction_ready": True,
        "paper_run_prompt_protocol_ready": True,
        "t2smark_formal_attack_ready": True,
        "t2smark_strict_pair_quality_ready": True,
        "formal_import_validation_ready": True,
        "paper_claim_scale": "probe_paper",
        "selected_prompt_count": len(prompts),
        "target_fpr": 0.1,
        "metadata": {"prompt_report": {"prompt_protocol_digest": prompt_digest}},
    }
    manifest = {
        "artifact_id": "t2smark_formal_reproduction_manifest",
        "metadata": {
            "run_decision": "pass",
            "t2smark_formal_reproduction_ready": True,
        },
        "config": {
            "prompt_set": "probe_paper",
            "model_id": "stabilityai/stable-diffusion-3.5-medium",
            "num_inference_steps": 20,
            "num_inversion_steps": 20,
            "guidance_scale": 4.5,
            "target_fpr": 0.1,
        },
    }
    (output_dir / "t2smark_formal_reproduction_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "t2smark_formal_reproduction_manifest.local.json").write_text(
        json.dumps(manifest, ensure_ascii=False),
        encoding="utf-8",
    )
    return output_dir


@pytest.mark.quick
def test_primary_baseline_evidence_distinguishes_method_faithful_from_formal_result(tmp_path: Path) -> None:
    """四个主表 baseline method-faithful 全通过时, 正式结果仍应因协议缺口保持未就绪。"""

    registry_path = write_source_registry(tmp_path)
    command_results_path, observations_path = write_method_faithful_outputs(tmp_path)
    source_registry = json.loads(registry_path.read_text(encoding="utf-8"))
    command_results = json.loads(command_results_path.read_text(encoding="utf-8"))
    observations = json.loads(observations_path.read_text(encoding="utf-8"))

    records = build_primary_baseline_evidence_records(
        source_registry=source_registry,
        command_results=command_results,
        observation_rows=observations,
    )
    summary = build_primary_baseline_evidence_summary(records)

    assert summary["adapter_run_ready_count"] == 4
    assert summary["formal_result_ready_count"] == 0
    assert "fixed_fpr_baseline_calibration_required" in summary["blocking_reasons"]
    tree_ring = next(row for row in records if row["baseline_id"] == "tree_ring")
    t2smark = next(row for row in records if row["baseline_id"] == "t2smark")
    assert tree_ring["adapter_run_ready"] is True
    assert tree_ring["method_faithful_adapter_ready"] is False
    assert "method_faithful_sd35_adapter_required" in tree_ring["blocking_reasons"]
    assert t2smark["method_faithful_adapter_ready"] is True
    assert "method_faithful_sd35_adapter_required" not in t2smark["blocking_reasons"]
    assert not any(row["supports_paper_claim"] for row in records)


@pytest.mark.quick
def test_primary_baseline_evidence_accepts_tree_ring_method_faithful_boundary(tmp_path: Path) -> None:
    """Tree-Ring observation 明确来自方法忠实 adapter 时, 方法忠实边界应视为就绪。"""

    registry_path = write_source_registry(tmp_path)
    source_registry = json.loads(registry_path.read_text(encoding="utf-8"))
    command_results = [
        {
            "baseline_id": "tree_ring",
            "return_code": 0,
            "observation_count": 2,
            "output_path": "outputs/tree_ring.json",
        }
    ]
    observations = [
        {
            "event_id": "tree_ring_clean",
            "baseline_id": "tree_ring",
            "sample_role": "clean_negative",
            "execution_device": "cuda",
            "latent_shape": [1, 16, 64, 64],
            "adapter_boundary": "method_faithful_sd35_adapter_reproduction",
        },
        {
            "event_id": "tree_ring_positive",
            "baseline_id": "tree_ring",
            "sample_role": "positive_source",
            "execution_device": "cuda",
            "latent_shape": [1, 16, 64, 64],
            "adapter_boundary": "method_faithful_sd35_adapter_reproduction",
        },
    ]

    records = build_primary_baseline_evidence_records(
        source_registry=source_registry,
        command_results=command_results,
        observation_rows=observations,
    )

    tree_ring = next(row for row in records if row["baseline_id"] == "tree_ring")
    assert tree_ring["adapter_run_ready"] is True
    assert tree_ring["method_faithful_adapter_ready"] is True
    assert "method_faithful_sd35_adapter_required" not in tree_ring["blocking_reasons"]
    assert "fixed_fpr_baseline_calibration_required" in tree_ring["blocking_reasons"]


@pytest.mark.quick
def test_primary_baseline_evidence_writer_outputs_records_summary_and_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """证据边界脚本应写出 records、summary 和 manifest, 且所有输出位于 outputs/。"""

    monkeypatch.setenv("SLM_WM_PAPER_RUN_NAME", "probe_paper")
    registry_path = write_source_registry(tmp_path)
    prompts, protocol = write_current_paper_protocol(tmp_path)
    collection_root = write_evidence_collection(tmp_path, prompts, protocol)
    t2smark_output_dir = write_t2smark_formal_evidence(tmp_path, prompts, protocol)

    manifest = write_primary_baseline_evidence_outputs(
        root=tmp_path,
        source_registry_path=registry_path,
        collection_root=collection_root,
        t2smark_formal_output_dir=t2smark_output_dir,
    )

    output_dir = tmp_path / "outputs" / "primary_baseline_evidence" / "probe_paper"
    records = [
        json.loads(line)
        for line in (output_dir / "primary_baseline_evidence_records.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    summary = json.loads((output_dir / "primary_baseline_evidence_summary.json").read_text(encoding="utf-8"))

    assert manifest["artifact_id"] == "primary_baseline_evidence_manifest"
    assert len(records) == 4
    assert summary["adapter_run_ready_count"] == 4
    assert summary["formal_result_ready_count"] == 4
    assert summary["primary_baseline_formal_ready"] is True
    assert summary["input_baseline_ids"] == [*METHOD_FAITHFUL_BASELINE_IDS, "t2smark"]
    assert summary["t2smark_formal_evidence_digest"]
    assert len(summary["primary_baseline_evidence_records_digest"]) == 64
    assert manifest["metadata"]["primary_baseline_evidence_records_digest"] == summary[
        "primary_baseline_evidence_records_digest"
    ]
    assert all(row["formal_evidence_paths"] for row in records)
    assert summary["supports_paper_claim"] is False
    assert all(str(path).startswith("outputs/") for path in manifest["output_paths"])


@pytest.mark.quick
def test_primary_baseline_evidence_writer_rejects_command_digest_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """transfer manifest 绑定的 command result 被修改后必须停止证据审计。"""

    monkeypatch.setenv("SLM_WM_PAPER_RUN_NAME", "probe_paper")
    registry_path = write_source_registry(tmp_path)
    prompts, protocol = write_current_paper_protocol(tmp_path)
    collection_root = write_evidence_collection(tmp_path, prompts, protocol)
    t2smark_output_dir = write_t2smark_formal_evidence(tmp_path, prompts, protocol)
    command_path = collection_root / "split_observations" / "tree_ring_baseline_command_results.json"
    command_path.write_text("[]\n", encoding="utf-8")

    with pytest.raises(ValueError, match="摘要与 transfer manifest 不一致"):
        write_primary_baseline_evidence_outputs(
            root=tmp_path,
            source_registry_path=registry_path,
            collection_root=collection_root,
            t2smark_formal_output_dir=t2smark_output_dir,
        )


@pytest.mark.quick
def test_primary_baseline_evidence_writer_requires_independent_t2smark_formal_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """三个 method-faithful source 不得替代独立 T2SMark formal evidence。"""

    monkeypatch.setenv("SLM_WM_PAPER_RUN_NAME", "probe_paper")
    registry_path = write_source_registry(tmp_path)
    prompts, protocol = write_current_paper_protocol(tmp_path)
    collection_root = write_evidence_collection(tmp_path, prompts, protocol)

    with pytest.raises(FileNotFoundError, match="T2SMark formal evidence 缺少文件"):
        write_primary_baseline_evidence_outputs(
            root=tmp_path,
            source_registry_path=registry_path,
            collection_root=collection_root,
        )

