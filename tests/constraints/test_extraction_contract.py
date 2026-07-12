"""验证论文附件抽离契约。"""

from __future__ import annotations

from pathlib import Path
import subprocess

import pytest

from scripts.extract_release_package import (
    PROFILES,
    _initialize_standalone_repository,
    extract_profile,
)
from tools.harness.audits.audit_dependency_boundaries import run_audit as run_dependency_boundary_audit
from tools.harness.audits.audit_release_extraction_contract import run_audit as run_release_extraction_audit


@pytest.mark.constraint
def test_dependency_boundaries_pass_for_template() -> None:
    """模板自身必须保持核心方法层可抽离。"""
    report = run_dependency_boundary_audit(Path.cwd())
    assert report["decision"] == "pass"


@pytest.mark.constraint
def test_release_extraction_contract_pass_for_template() -> None:
    """模板必须提供最小论文附件抽离规则。"""
    report = run_release_extraction_audit(Path.cwd())
    assert report["decision"] == "pass"


@pytest.mark.constraint
def test_minimal_method_package_dry_run_excludes_governance_layer(tmp_path: Path) -> None:
    """最小方法包抽离清单不得包含外层治理目录。"""
    manifest = extract_profile(Path.cwd(), tmp_path / "minimal_method_package", "minimal_method_package", dry_run=True)
    copied_files = manifest["copied_files"]
    assert copied_files
    assert all(not path.startswith(".codex/") for path in copied_files)
    assert all(not path.startswith("tools/") for path in copied_files)
    assert all(not path.startswith("tests/") for path in copied_files)
    assert all(not path.startswith("experiments/") for path in copied_files)
    assert all(not path.startswith("paper_experiments/") for path in copied_files)
    assert all(not path.startswith("paper_workflow/") for path in copied_files)
    assert "configs/model_sd35.yaml" in copied_files
    assert "configs/model_source_registry.json" in copied_files
    assert all("_prompts.txt" not in path for path in copied_files)


@pytest.mark.constraint
def test_paper_artifact_rebuild_package_includes_full_experiment_layer(tmp_path: Path) -> None:
    """论文产物重建包必须包含完整论文实验层, 但不包含 Colab 运行层。"""
    manifest = extract_profile(
        Path.cwd(),
        tmp_path / "paper_artifact_rebuild_package",
        "paper_artifact_rebuild_package",
        dry_run=True,
    )
    copied_files = manifest["copied_files"]
    assert any(path.startswith("paper_experiments/") for path in copied_files)
    assert all(not path.startswith("paper_workflow/") for path in copied_files)


@pytest.mark.constraint
def test_paper_experiment_execution_package_excludes_colab_and_tests(
    tmp_path: Path,
) -> None:
    """服务器论文实验包必须包含正式执行层, 但排除 Colab 与开发测试层。"""
    manifest = extract_profile(
        Path.cwd(),
        tmp_path / "paper_experiment_execution_package",
        "paper_experiment_execution_package",
        dry_run=True,
    )
    copied_files = manifest["copied_files"]
    assert any(path.startswith("paper_experiments/") for path in copied_files)
    assert any(path.startswith("experiments/") for path in copied_files)
    assert all(not path.startswith("paper_workflow/") for path in copied_files)
    assert any(path.startswith("external_baseline/primary/") for path in copied_files)
    assert all("/source/" not in path for path in copied_files)
    assert all(not path.startswith("tests/") for path in copied_files)
    assert "scripts/validate_extracted_package.py" in copied_files
    assert manifest["standalone_repository"] is True
    assert manifest["complete_dependency_locks_required"] is True


@pytest.mark.constraint
def test_standalone_profiles_require_complete_locks_and_real_entrypoints() -> None:
    """两个可运行抽离 profile 必须共享完整锁与独立 Git 身份契约。"""

    for profile_name in (
        "paper_artifact_rebuild_package",
        "paper_experiment_execution_package",
    ):
        profile = PROFILES[profile_name]
        assert profile.standalone_repository is True
        assert profile.complete_dependency_locks_required is True
        assert "scripts/validate_extracted_package.py" in profile.required_entrypoints


@pytest.mark.constraint
def test_standalone_repository_is_clean_detached_and_self_contained(
    tmp_path: Path,
) -> None:
    """抽离包必须创建自己的根提交, 不能引用开发仓库 Git 元数据。"""

    package_root = tmp_path / "standalone_package"
    package_root.mkdir()
    (package_root / ".gitignore").write_text("__pycache__/\n", encoding="utf-8")
    (package_root / "payload.txt").write_text("受治理代码包\n", encoding="utf-8")
    commit = _initialize_standalone_repository(
        package_root,
        PROFILES["paper_experiment_execution_package"],
    )

    assert len(commit) == 40
    assert (
        subprocess.run(
            ["git", "symbolic-ref", "-q", "HEAD"],
            cwd=package_root,
            check=False,
            capture_output=True,
        ).returncode
        == 1
    )
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=package_root,
        check=True,
        capture_output=True,
        text=True,
    )
    assert status.stdout == ""
    assert (package_root / ".git").is_dir()
