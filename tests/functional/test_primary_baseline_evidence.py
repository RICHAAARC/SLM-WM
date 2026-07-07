"""主表 external baseline 证据边界审计的轻量功能测试。"""

from __future__ import annotations

import json
from pathlib import Path
from zipfile import ZipFile

import pytest

from paper_experiments.baselines import build_primary_baseline_evidence_records, build_primary_baseline_evidence_summary
from scripts.write_primary_baseline_evidence_outputs import write_primary_baseline_evidence_outputs


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
def test_primary_baseline_evidence_writer_outputs_records_summary_and_manifest(tmp_path: Path) -> None:
    """证据边界脚本应写出 records、summary 和 manifest, 且所有输出位于 outputs/。"""

    registry_path = write_source_registry(tmp_path)
    command_results_path, observations_path = write_method_faithful_outputs(tmp_path)

    manifest = write_primary_baseline_evidence_outputs(
        root=tmp_path,
        source_registry_path=registry_path,
        command_results_path=command_results_path,
        observations_path=observations_path,
    )

    output_dir = tmp_path / "outputs" / "primary_baseline_evidence"
    records = [
        json.loads(line)
        for line in (output_dir / "primary_baseline_evidence_records.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    summary = json.loads((output_dir / "primary_baseline_evidence_summary.json").read_text(encoding="utf-8"))

    assert manifest["artifact_id"] == "primary_baseline_evidence_manifest"
    assert len(records) == 4
    assert summary["adapter_run_ready_count"] == 4
    assert summary["primary_baseline_formal_ready"] is False
    assert summary["supports_paper_claim"] is False
    assert all(str(path).startswith("outputs/") for path in manifest["output_paths"])


@pytest.mark.quick
def test_primary_baseline_evidence_writer_can_read_method_faithful_package(tmp_path: Path) -> None:
    """证据边界脚本应可直接读取 method-faithful zip 包中的命令结果和 observation。"""

    registry_path = write_source_registry(tmp_path)
    command_results_path, observations_path = write_method_faithful_outputs(tmp_path)
    package_path = tmp_path / "outputs" / "external_baseline_method_faithful_package.zip"
    package_path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(package_path, "w") as archive:
        archive.write(registry_path, "external_baseline/source_registry.json")
        archive.write(
            command_results_path,
            "outputs/external_baseline_method_faithful/execution/baseline_command_results.json",
        )
        archive.write(
            observations_path,
            "outputs/external_baseline_method_faithful/execution/baseline_observations.json",
        )
    command_results_path.unlink()
    observations_path.unlink()

    write_primary_baseline_evidence_outputs(
        root=tmp_path,
        source_registry_path=tmp_path / "missing_registry.json",
        method_faithful_package_path=package_path,
    )

    summary = json.loads(
        (tmp_path / "outputs" / "primary_baseline_evidence" / "primary_baseline_evidence_summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert summary["adapter_run_ready_count"] == 4
    assert summary["method_faithful_package_path"] == "outputs/external_baseline_method_faithful_package.zip"

