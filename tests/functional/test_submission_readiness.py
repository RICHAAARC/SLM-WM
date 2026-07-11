"""投稿就绪门禁链路的轻量功能测试。"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from paper_experiments.analysis.submission_readiness import (
    SubmissionReadinessInput,
    build_release_profile_rows,
    build_required_evidence_rows,
    build_submission_readiness_report,
)
from scripts.write_submission_readiness_outputs import write_submission_readiness_outputs


def sample_gap_rows() -> tuple[dict, ...]:
    """构造最小证据缺口行, 用于验证阻断判定。"""
    return (
        {
            "gap_id": "gap_real_attacked_image_closed_loop",
            "gap_area": "attack_matrix",
            "blocker_severity": "critical",
            "required_action": "补齐真实 attacked image 闭环。",
            "related_artifacts": "outputs/attack_matrix/attacked_images",
            "closes_claim_ids": "claim_attack_robustness_under_common_matrix",
            "recommended_order": "1",
            "supports_paper_claim": "False",
        },
        {
            "gap_id": "gap_baseline_results",
            "gap_area": "baseline_comparison",
            "blocker_severity": "critical",
            "required_action": "补齐外部 baseline 结果。",
            "related_artifacts": "outputs/external_baseline_comparison/baseline_comparison_table.csv",
            "closes_claim_ids": "claim_baseline_superiority",
            "recommended_order": "2",
            "supports_paper_claim": "False",
        },
    )


def make_input_bundle() -> SubmissionReadinessInput:
    """构造投稿就绪门禁的最小输入。"""
    return SubmissionReadinessInput(
        evidence_manifest={"artifact_id": "paper_artifact_evidence_audit_manifest"},
        builder_report={"artifact_builder_ready": True, "paper_ready_artifact_count": 0},
        blocker_report={
            "submission_ready": False,
            "blocking_claim_count": 2,
            "recommended_next_action": "先补齐关键证据缺口。",
        },
        evidence_gaps=sample_gap_rows(),
        release_profiles=(
            {"profile_name": "minimal_method_package", "copied_files": ["main/__init__.py"], "missing_paths": [], "dry_run": True},
            {"profile_name": "paper_artifact_rebuild_package", "copied_files": ["main/__init__.py"], "missing_paths": [], "dry_run": True},
        ),
    )


@pytest.mark.quick
def test_submission_readiness_remains_blocked_when_evidence_gaps_exist() -> None:
    """即使 release dry-run 可运行, 未关闭证据缺口时也不得允许投稿冻结。"""
    bundle = make_input_bundle()
    required_rows = build_required_evidence_rows(bundle)
    release_rows = build_release_profile_rows(bundle)
    report = build_submission_readiness_report(bundle, required_rows, release_rows)

    assert report["readiness_decision"] == "blocked"
    assert report["submission_ready"] is False
    assert report["package_freeze_allowed"] is False
    assert report["release_dry_run_ready"] is True
    assert report["required_input_count"] == 2
    assert report["critical_required_input_count"] == 2
    assert report["primary_blockers"] == ["gap_real_attacked_image_closed_loop", "gap_baseline_results"]
    assert all(row["input_ready"] is False for row in required_rows)
    assert all(row["release_package_allowed"] is False for row in release_rows)


@pytest.mark.quick
def test_submission_readiness_requires_all_paper_artifacts_and_allows_complete_freeze() -> None:
    """只有证据缺口为空且所有论文产物达到 claim 门禁时才允许冻结。"""

    blocked_bundle = SubmissionReadinessInput(
        evidence_manifest={"artifact_id": "paper_artifact_evidence_audit_manifest"},
        builder_report={"artifact_builder_ready": True, "paper_artifact_claim_ready": False},
        blocker_report={"submission_ready": True, "paper_artifact_claim_ready": False},
        evidence_gaps=(),
        release_profiles=make_input_bundle().release_profiles,
    )
    blocked_release_rows = build_release_profile_rows(blocked_bundle)
    blocked_report = build_submission_readiness_report(blocked_bundle, (), blocked_release_rows)

    assert blocked_report["submission_ready"] is False
    assert all(row["release_package_allowed"] is False for row in blocked_release_rows)

    ready_bundle = SubmissionReadinessInput(
        evidence_manifest={"artifact_id": "paper_artifact_evidence_audit_manifest"},
        builder_report={"artifact_builder_ready": True, "paper_artifact_claim_ready": True},
        blocker_report={"submission_ready": True, "paper_artifact_claim_ready": True},
        evidence_gaps=(),
        release_profiles=make_input_bundle().release_profiles,
    )
    ready_release_rows = build_release_profile_rows(ready_bundle)
    ready_report = build_submission_readiness_report(ready_bundle, (), ready_release_rows)

    assert ready_report["readiness_decision"] == "ready"
    assert ready_report["submission_ready"] is True
    assert ready_report["paper_artifact_claim_ready"] is True
    assert all(row["release_package_allowed"] is True for row in ready_release_rows)


def write_json(path: Path, value: dict) -> None:
    """以 UTF-8 写入 JSON fixture。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def write_csv(path: Path, rows: tuple[dict, ...]) -> None:
    """写入证据缺口 CSV fixture。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_minimal_repository_files(tmp_path: Path) -> None:
    """写入 release dry-run 所需的最小仓库文件。"""
    for directory in [
        "main/core",
        "main/methods",
        "configs",
        "experiments/protocol",
        "paper_experiments/baselines",
        "external_baseline/primary/tree_ring/adapter",
        "scripts",
        "tests/functional",
        "docs",
    ]:
        (tmp_path / directory).mkdir(parents=True, exist_ok=True)
    for file_path in [
        "main/__init__.py",
        "main/core/__init__.py",
        "main/methods/__init__.py",
        "configs/model_sd35.yaml",
        "experiments/protocol/__init__.py",
        "paper_experiments/__init__.py",
        "paper_experiments/baselines/__init__.py",
        "external_baseline/README.md",
        "external_baseline/source_registry.json",
        "external_baseline/primary/tree_ring/adapter/run_slm_eval.py",
        "scripts/write_artifact.py",
        "tests/functional/test_artifact.py",
        "README.md",
        "pyproject.toml",
        "docs/artifact_rebuild.md",
        "docs/field_registry.md",
        "docs/file_organization.md",
        "docs/release_boundary.md",
        "docs/release_layer_boundary.md",
        "docs/extraction_profiles.md",
        "docs/intermediate_state_governance.md",
    ]:
        (tmp_path / file_path).write_text("# fixture\n", encoding="utf-8")


def write_upstream_audit_outputs(tmp_path: Path) -> None:
    """写入投稿就绪脚本所需的上游审计产物。"""
    audit_dir = tmp_path / "outputs" / "paper_artifact_evidence_audit"
    write_json(audit_dir / "manifest.local.json", {"artifact_id": "paper_artifact_evidence_audit_manifest"})
    write_json(audit_dir / "artifact_builder_readiness_report.json", {"artifact_builder_ready": True, "paper_ready_artifact_count": 0})
    write_json(
        audit_dir / "submission_blocker_report.json",
        {"submission_ready": False, "blocking_claim_count": 2, "recommended_next_action": "先补齐关键证据缺口。"},
    )
    write_csv(audit_dir / "evidence_gap_list.csv", sample_gap_rows())


@pytest.mark.quick
def test_submission_readiness_outputs_are_rebuildable_and_claim_safe(tmp_path: Path) -> None:
    """脚本应从证据审计产物重建投稿就绪门禁报告, 且保持 claim 安全边界。"""
    write_minimal_repository_files(tmp_path)
    write_upstream_audit_outputs(tmp_path)

    manifest = write_submission_readiness_outputs(root=tmp_path)
    output_dir = tmp_path / "outputs" / "submission_readiness"
    required_rows = list(csv.DictReader((output_dir / "required_evidence_inputs.csv").open(encoding="utf-8")))
    release_rows = list(csv.DictReader((output_dir / "release_profile_dry_run.csv").open(encoding="utf-8")))
    report = json.loads((output_dir / "readiness_blocker_report.json").read_text(encoding="utf-8"))

    assert manifest["artifact_id"] == "submission_readiness_manifest"
    assert manifest["metadata"]["readiness_decision"] == "blocked"
    assert report["submission_ready"] is False
    assert report["release_dry_run_ready"] is True
    assert {row["required_input_id"] for row in required_rows} == {"gap_real_attacked_image_closed_loop", "gap_baseline_results"}
    assert {row["release_profile_name"] for row in release_rows} == {
        "full_experiment_execution_package",
        "minimal_method_package",
        "paper_artifact_rebuild_package",
    }
    assert all(row["supports_paper_claim"] == "False" for row in required_rows + release_rows)
