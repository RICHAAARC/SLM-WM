"""主表外部 baseline 复现计划的轻量功能测试。"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from experiments.baselines import PRIMARY_BASELINE_IDS, build_primary_baseline_execution_plans
from scripts.write_primary_baseline_reproduction_plan import write_primary_baseline_reproduction_plan


def write_primary_source_registry(tmp_path: Path) -> Path:
    """写出包含 4 个主表 baseline 的最小来源登记。"""
    sources = []
    for baseline_id in PRIMARY_BASELINE_IDS:
        source_dir = f"external_baseline/primary/{baseline_id}/source"
        (tmp_path / source_dir).mkdir(parents=True)
        (tmp_path / source_dir / "README.md").write_text("source cache\n", encoding="utf-8")
        sources.append(
            {
                "baseline_id": baseline_id,
                "baseline_name": baseline_id.replace("_", " ").title(),
                "baseline_family": "test_family",
                "comparison_group": "primary",
                "source_dir": source_dir,
                "source_status": "downloaded",
                "official_repository_url": f"git@example.invalid/{baseline_id}.git",
                "official_repository_commit": f"{baseline_id}_commit",
                "official_repository_branch": "main",
                "paper_claim_support": False,
            }
        )
    registry_path = tmp_path / "external_baseline" / "source_registry.json"
    registry_path.write_text(
        json.dumps({"registry_name": "external_baseline_source_registry", "baseline_sources": sources}, ensure_ascii=False),
        encoding="utf-8",
    )
    return registry_path


def write_attack_inputs(tmp_path: Path) -> tuple[Path, Path]:
    """写出生成结果导入模板所需的攻击矩阵输入。"""
    attack_dir = tmp_path / "outputs" / "attack_matrix"
    attack_dir.mkdir(parents=True)
    attack_manifest_path = attack_dir / "attack_manifest.json"
    attack_metrics_path = attack_dir / "attack_family_metrics.csv"
    attack_manifest_path.write_text(
        json.dumps({"attack_metrics_ready": True, "evaluation_boundary": {"target_fpr": 0.05}}, ensure_ascii=False),
        encoding="utf-8",
    )
    with attack_metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["attack_family", "attack_name", "resource_profile"])
        writer.writeheader()
        writer.writerow(
            {
                "attack_family": "standard_distortion",
                "attack_name": "jpeg_compression",
                "resource_profile": "probe",
            }
        )
        writer.writerow(
            {
                "attack_family": "regeneration_attack",
                "attack_name": "img2img_regeneration",
                "resource_profile": "full_main",
            }
        )
    return attack_manifest_path, attack_metrics_path


@pytest.mark.quick
def test_primary_execution_plan_uses_only_primary_baselines(tmp_path: Path) -> None:
    """主表复现计划应只覆盖 4 个主表 baseline, 且不支持论文主张。"""
    registry_path = write_primary_source_registry(tmp_path)
    registry = json.loads(registry_path.read_text(encoding="utf-8"))

    plans = build_primary_baseline_execution_plans(registry, root=tmp_path)

    assert {row["baseline_id"] for row in plans} == set(PRIMARY_BASELINE_IDS)
    assert all(row["comparison_group"] == "primary" for row in plans)
    assert all(row["source_entry_ready"] for row in plans)
    assert all(row["result_import_required"] for row in plans)
    assert not any(row["supports_paper_claim"] for row in plans)
    t2smark_plan = next(row for row in plans if row["baseline_id"] == "t2smark")
    assert t2smark_plan["model_alignment_status"] == "sd35_medium_native_entrypoint"


@pytest.mark.quick
def test_primary_reproduction_writer_outputs_plan_and_result_templates(tmp_path: Path) -> None:
    """复现计划脚本应输出计划, 结果模板, 运行报告和 manifest。"""
    registry_path = write_primary_source_registry(tmp_path)
    attack_manifest_path, attack_metrics_path = write_attack_inputs(tmp_path)

    manifest = write_primary_baseline_reproduction_plan(
        root=tmp_path,
        source_registry_path=registry_path,
        attack_manifest_path=attack_manifest_path,
        attack_family_metrics_path=attack_metrics_path,
    )

    output_dir = tmp_path / "outputs" / "primary_baseline_reproduction"
    plan_rows = [
        json.loads(line)
        for line in (output_dir / "primary_baseline_execution_plan.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    template_rows = [
        json.loads(line)
        for line in (output_dir / "primary_baseline_result_record_template.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    report = json.loads((output_dir / "primary_baseline_reproduction_report.json").read_text(encoding="utf-8"))

    assert manifest["artifact_id"] == "primary_baseline_reproduction_manifest"
    assert len(plan_rows) == 4
    assert len(template_rows) == 8
    assert report["primary_baseline_plan_ready"] is True
    assert report["result_import_template_ready"] is True
    assert report["baseline_results_ready"] is False
    assert report["supports_paper_claim"] is False
    assert {row["comparable_operating_point"] for row in template_rows} == {"fixed_fpr_0.05"}
