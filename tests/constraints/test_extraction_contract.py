"""验证论文附件抽离契约。"""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

import pytest

from scripts.extract_release_package import (
    PROFILES,
    _initialize_standalone_repository,
    extract_profile,
)
from tools.harness.audits.audit_dependency_boundaries import run_audit as run_dependency_boundary_audit
from tools.harness.audits.audit_release_extraction_contract import run_audit as run_release_extraction_audit


def _run_git(root: Path, *arguments: str) -> str:
    """在测试临时目录中构造正式抽离所需的 clean 源提交。"""

    completed = subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
        shell=False,
    )
    return completed.stdout.strip()


def _build_clean_minimal_source_repository(tmp_path: Path) -> Path:
    """只复制最小 profile 的源输入, 避免测试依赖开发工作树状态。"""

    repository_root = Path.cwd()
    source_root = tmp_path / "source_repository"
    source_root.mkdir()
    shutil.copytree(
        repository_root / "main",
        source_root / "main",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    for relative in (
        "configs/core_method_dependency_identity.json",
        "configs/model_sd35.yaml",
        "configs/model_source_registry.json",
        "docs/core_method_package_readme.md",
        "scripts/validate_core_method_package.py",
        "pyproject.toml",
    ):
        source = repository_root / relative
        target = source_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    _run_git(source_root, "init", "--quiet", "--initial-branch=main")
    _run_git(source_root, "config", "user.name", "SLM-WM Test")
    _run_git(source_root, "config", "user.email", "test@slm-wm.invalid")
    _run_git(source_root, "add", "--all")
    _run_git(source_root, "commit", "--quiet", "--no-gpg-sign", "-m", "构造最小包测试源")
    return source_root


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
    assert "configs/core_method_dependency_identity.json" in copied_files
    assert "README.md" in copied_files
    assert "validate_core_method_package.py" in copied_files
    assert manifest["standalone_repository"] is True
    assert manifest["complete_dependency_locks_required"] is False
    assert manifest["required_entrypoints"] == ["validate_core_method_package.py"]
    readme_record = next(
        record
        for record in manifest["copied_file_records"]
        if record["path"] == "README.md"
    )
    assert readme_record["source_path"] == "docs/core_method_package_readme.md"
    assert all("_prompts.txt" not in path for path in copied_files)


@pytest.mark.constraint
def test_core_dependency_identity_covers_main_third_party_imports() -> None:
    """最小包依赖身份必须覆盖 main 中延迟导入的第三方科学依赖。"""

    imported_roots: set[str] = set()
    for path in Path("main").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_roots.update(
                    alias.name.split(".", maxsplit=1)[0]
                    for alias in node.names
                )
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_roots.add(node.module.split(".", maxsplit=1)[0])
    third_party_imports = imported_roots - set(sys.stdlib_module_names) - {
        "__future__",
        "main",
    }
    dependency_identity = json.loads(
        Path("configs/core_method_dependency_identity.json").read_text(
            encoding="utf-8"
        )
    )

    assert third_party_imports == {"torch"}
    assert dependency_identity["runtime_dependencies"] == ["torch>=2.11,<2.12"]


@pytest.mark.quick
def test_minimal_method_package_is_detached_importable_and_builds_only_main(
    tmp_path: Path,
) -> None:
    """正式最小包必须脱离开发仓库完成验证, 并只构建 main 包。"""

    source_root = _build_clean_minimal_source_repository(tmp_path)
    package_root = tmp_path / "minimal_method_package"
    manifest = extract_profile(
        source_root,
        package_root,
        "minimal_method_package",
        dry_run=False,
    )

    assert manifest["source_repository_commit"] == _run_git(source_root, "rev-parse", "HEAD")
    assert _run_git(package_root, "status", "--porcelain=v1", "--untracked-files=all") == ""
    assert (
        subprocess.run(
            ["git", "symbolic-ref", "-q", "HEAD"],
            cwd=package_root,
            check=False,
            capture_output=True,
            shell=False,
        ).returncode
        == 1
    )
    assert (package_root / "README.md").read_bytes() == (
        source_root / "docs/core_method_package_readme.md"
    ).read_bytes()
    assert (package_root / "README.md").read_bytes() != (
        Path.cwd() / "README.md"
    ).read_bytes()
    assert not (package_root / "configs/dependency_profile_registry.json").exists()
    assert not any((package_root / "configs").glob("dependency_profiles/*_lock.txt"))
    dependency_identity = json.loads(
        (package_root / "configs/core_method_dependency_identity.json").read_text(
            encoding="utf-8"
        )
    )
    assert dependency_identity["runtime_dependencies"] == ["torch>=2.11,<2.12"]

    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    validation = subprocess.run(
        [
            sys.executable,
            "-I",
            str(package_root / "validate_core_method_package.py"),
            "--root",
            str(package_root),
        ],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        shell=False,
    )
    assert validation.returncode == 0, validation.stderr or validation.stdout
    validation_report = json.loads(validation.stdout)
    assert validation_report["decision"] == "pass"
    assert "main.methods.subspace.jacobian_nullspace" in validation_report["imported_modules"]
    assert _run_git(package_root, "status", "--porcelain=v1", "--untracked-files=all") == ""

    build_source = tmp_path / "build_source"
    shutil.copytree(
        package_root,
        build_source,
        ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc"),
    )
    wheel_directory = tmp_path / "wheel_output"
    wheel_directory.mkdir()
    build = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--disable-pip-version-check",
            "--no-build-isolation",
            "--no-deps",
            "--wheel-dir",
            str(wheel_directory),
            str(build_source),
        ],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        shell=False,
    )
    assert build.returncode == 0, build.stderr or build.stdout
    wheels = list(wheel_directory.glob("*.whl"))
    assert len(wheels) == 1
    with zipfile.ZipFile(wheels[0]) as archive:
        members = archive.namelist()
    python_sources = [name for name in members if name.endswith(".py")]
    assert python_sources
    assert all(name.startswith("main/") for name in python_sources)
    assert all(not name.startswith("configs/") for name in members)

    install_directory = tmp_path / "installed_core_package"
    install = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--no-index",
            "--no-deps",
            "--target",
            str(install_directory),
            str(wheels[0]),
        ],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        shell=False,
    )
    assert install.returncode == 0, install.stderr or install.stdout
    isolated_import_code = (
        "import importlib,json,pkgutil,sys;"
        "sys.dont_write_bytecode=True;"
        "sys.path.insert(0,sys.argv[1]);"
        "package=importlib.import_module('main');"
        "names=['main'];"
        "names.extend(item.name for item in pkgutil.walk_packages("
        "package.__path__,prefix='main.'));"
        "[importlib.import_module(name) for name in names];"
        "print(json.dumps(sorted(names)))"
    )
    installed_import = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            isolated_import_code,
            str(install_directory),
        ],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        shell=False,
    )
    assert installed_import.returncode == 0, (
        installed_import.stderr or installed_import.stdout
    )
    installed_modules = json.loads(installed_import.stdout)
    assert "main.methods.geometry.differentiable_attention" in installed_modules
    assert "main.methods.subspace.jacobian_nullspace" in installed_modules


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
