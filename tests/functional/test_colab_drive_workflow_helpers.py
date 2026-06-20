"""验证 Colab Drive workflow helper 的轻量行为。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from paper_workflow.colab_utils.dependency_check import build_dependency_report
from paper_workflow.colab_utils.drive_paths import build_drive_workflow_paths
from paper_workflow.colab_utils.drive_workflow import run_colab_drive_workflow
from paper_workflow.colab_utils.mount_drive import build_drive_mount_report


def write_fake_manifest(repo_root: Path) -> None:
    """写入一个最小本地 manifest, 用于模拟已存在的受治理产物。"""
    artifact_dir = repo_root / "outputs" / "core_package_boundary_freeze"
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "core_boundary_report.json").write_text('{"decision":"pass"}\n', encoding="utf-8")
    manifest = {
        "artifact_id": "core_package_boundary_manifest",
        "artifact_type": "local_manifest",
        "input_paths": ["outputs/audit_reports/harness_audit_summary.json"],
        "output_paths": [
            "outputs/core_package_boundary_freeze/core_boundary_report.json",
            "outputs/core_package_boundary_freeze/manifest.local.json",
        ],
        "config_digest": "digest",
        "code_version": "test",
        "rebuild_command": "python scripts/write_core_package_boundary_outputs.py",
        "metadata": {"decision": "pass"},
    }
    (artifact_dir / "manifest.local.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


@pytest.mark.quick
def test_drive_workflow_mirrors_outputs_and_reloads(tmp_path: Path) -> None:
    """Drive workflow 应镜像 outputs 文件, 写入 manifest, 并能按 manifest 重载校验。"""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    write_fake_manifest(repo_root)
    drive_root = tmp_path / "drive_root"

    summary = run_colab_drive_workflow(root=repo_root, drive_root=drive_root)

    local_output_dir = repo_root / "outputs" / "colab_drive_workflow"
    drive_workflow_dir = drive_root / "colab_drive_workflow"
    manifest = json.loads((drive_workflow_dir / "manifest.json").read_text(encoding="utf-8"))
    reload_line = (local_output_dir / "reload_smoke_record.jsonl").read_text(encoding="utf-8").strip()
    reload_record = json.loads(reload_line)

    assert summary["workflow_decision"] == "pass"
    assert summary["mirrored_file_count"] >= 2
    assert manifest["metadata"]["workflow_decision"] == "pass"
    assert reload_record["reload_decision"] == "pass"
    assert (local_output_dir / "local_output_sync_report.json").exists()
    assert (drive_workflow_dir / "local_outputs" / "outputs" / "core_package_boundary_freeze" / "manifest.local.json").exists()


@pytest.mark.quick
def test_local_output_dir_must_stay_under_outputs(tmp_path: Path) -> None:
    """本地持久化输出目录必须收敛在 outputs 下。"""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()

    with pytest.raises(ValueError):
        build_drive_workflow_paths(root=repo_root, local_output_dir=repo_root / "outside")


@pytest.mark.quick
def test_mount_report_can_skip_drive_action() -> None:
    """非 Colab 本地测试应能跳过真实挂载并保留可审计原因。"""
    report = build_drive_mount_report(perform_mount=False).to_dict()

    assert report["mount_decision"] == "skipped"
    assert report["unsupported_reason"] == "mount_not_requested"


@pytest.mark.quick
def test_dependency_report_is_not_claim_evidence() -> None:
    """依赖快照只能作为工程 provenance, 不能支持论文 claim。"""
    report = build_dependency_report(("json",))

    assert report["dependency_count"] == 1
    assert report["missing_dependency_count"] == 0
    assert report["supports_paper_claim"] is False
    assert "json" in report["package_versions"]
