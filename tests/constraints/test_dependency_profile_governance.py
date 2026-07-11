"""验证正式依赖 profile 的 harness 治理规则."""

from __future__ import annotations

from pathlib import Path
import shutil

import pytest

from experiments.runtime.dependency_profiles import (
    load_dependency_profile_registry,
)
from tools.harness.audits.audit_dependency_profile_governance import run_audit


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _copy_governance_fixture(tmp_path: Path) -> Path:
    """复制依赖 registry、文档、内层实现与薄 CLI 的最小审计输入."""

    files = (
        "configs/dependency_profile_registry.json",
        "experiments/runtime/dependency_profiles.py",
        "experiments/runtime/dependency_preparation.py",
        "experiments/runtime/isolated_dependency_environment.py",
        "experiments/runtime/isolated_scientific_execution.py",
        "experiments/runtime/repository_environment.py",
        "scripts/prepare_dependency_profile.py",
        "scripts/prepare_isolated_dependency_environment.py",
        "scripts/materialize_dependency_lock_candidate.py",
        "scripts/write_dependency_lock_review_bundle.py",
        "scripts/write_reviewed_dependency_hash_lock.py",
        "docs/field_registry.md",
        "docs/builds/formal_dependency_environment.md",
    )
    for relative_path in files:
        source_path = REPOSITORY_ROOT / relative_path
        target_path = tmp_path / relative_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target_path)
    shutil.copytree(
        REPOSITORY_ROOT / "configs/dependency_profiles",
        tmp_path / "configs/dependency_profiles",
    )
    for relative_directory in (
        "main",
        "paper_experiments",
        "paper_workflow/notebooks",
        "paper_workflow/colab_utils",
    ):
        (tmp_path / relative_directory).mkdir(parents=True, exist_ok=True)
    return tmp_path


@pytest.mark.constraint
def test_repository_dependency_profile_governance_passes() -> None:
    """当前仓库必须通过正式依赖结构、字段和业务路径审计."""

    report = run_audit(REPOSITORY_ROOT)
    profiles = load_dependency_profile_registry()
    expected_missing_count = sum(
        not profile.complete_hash_lock_present for profile in profiles.values()
    )
    expected_ready_count = len(profiles) - expected_missing_count
    expected_direct_count = sum(
        len(profile.direct_requirements) for profile in profiles.values()
    )

    assert report["decision"] == "pass"
    assert report["summary"]["profile_count"] == len(profiles)
    assert report["summary"]["direct_dependency_count"] == expected_direct_count
    assert report["summary"]["missing_lock_count"] == expected_missing_count
    assert report["summary"]["fail_closed_missing_lock_count"] == (
        expected_missing_count
    )
    assert report["summary"]["ready_lock_count"] == expected_ready_count


@pytest.mark.constraint
def test_fixture_hash_lock_counts_follow_actual_registry_state(tmp_path: Path) -> None:
    """缺失锁必须 fail-closed, 已提交锁必须进入动态 ready 计数."""

    fixture_root = _copy_governance_fixture(tmp_path)
    report = run_audit(fixture_root)
    profiles = load_dependency_profile_registry(
        fixture_root / "configs/dependency_profile_registry.json"
    )
    expected_missing_count = sum(
        not profile.complete_hash_lock_present for profile in profiles.values()
    )
    expected_ready_count = len(profiles) - expected_missing_count
    expected_direct_count = sum(
        len(profile.direct_requirements) for profile in profiles.values()
    )

    assert report["decision"] == "pass"
    assert report["summary"]["direct_dependency_count"] == expected_direct_count
    assert report["summary"]["missing_lock_count"] == expected_missing_count
    assert report["summary"]["fail_closed_missing_lock_count"] == (
        expected_missing_count
    )
    assert report["summary"]["ready_lock_count"] == expected_ready_count
    assert report["violations"] == []


@pytest.mark.constraint
@pytest.mark.parametrize(
    ("relative_path", "source", "reason"),
    (
        (
            "paper_workflow/notebooks/example.ipynb",
            '{"cells": [{"source": ["%pip install package==1.0"]}]}',
            "notebook_pip_magic_forbidden",
        ),
        (
            "paper_experiments/runtime_install.py",
            'command = ["pip", "install", "--upgrade", "package"]\n',
            "dependency_upgrade_option_forbidden",
        ),
        (
            "paper_workflow/colab_utils/free_spec.py",
            'PACKAGE_SPECS = os.environ.get("PACKAGE_SPECS")\n',
            "free_dependency_spec_override_forbidden",
        ),
        (
            "scripts/mutable_download.py",
            'url = "https://example.invalid/runtime/latest"\n',
            "mutable_latest_dependency_forbidden",
        ),
    ),
)
def test_dynamic_dependency_rules_in_business_paths_are_rejected(
    tmp_path: Path,
    relative_path: str,
    source: str,
    reason: str,
) -> None:
    """Notebook 和业务路径不得形成第二套动态依赖规则."""

    fixture_root = _copy_governance_fixture(tmp_path)
    target_path = fixture_root / relative_path
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(source, encoding="utf-8")

    report = run_audit(fixture_root)

    assert report["decision"] == "fail"
    assert reason in {violation["reason"] for violation in report["violations"]}


@pytest.mark.constraint
def test_obsolete_colab_constraint_file_is_rejected(tmp_path: Path) -> None:
    """仓库不得保留绕过 dependency registry 的旁路约束文件入口."""

    fixture_root = _copy_governance_fixture(tmp_path)
    obsolete_path = fixture_root / "configs/colab_sd35_runtime_constraints.txt"
    obsolete_path.write_text("diffusers==0.38.0\n", encoding="utf-8")

    report = run_audit(fixture_root)

    assert report["decision"] == "fail"
    assert "obsolete_dependency_path_present" in {
        violation["reason"] for violation in report["violations"]
    }


@pytest.mark.constraint
def test_inner_layer_cannot_reference_outer_dependency_script(tmp_path: Path) -> None:
    """方法与实验内层必须调用 runtime API, 不能反向调用 scripts 层."""

    fixture_root = _copy_governance_fixture(tmp_path)
    invalid_path = fixture_root / "experiments/runtime/invalid_dependency_entry.py"
    invalid_path.write_text(
        'COMMAND = "python scripts/prepare_dependency_profile.py --profile x"\n',
        encoding="utf-8",
    )

    report = run_audit(fixture_root)

    assert report["decision"] == "fail"
    assert "inner_layer_references_outer_dependency_script" in {
        violation["reason"] for violation in report["violations"]
    }


@pytest.mark.constraint
def test_dependency_scripts_must_remain_thin_forwarders(tmp_path: Path) -> None:
    """外层 CLI 只能转发到 experiments runtime, 不得重新承载实现."""

    fixture_root = _copy_governance_fixture(tmp_path)
    script_path = fixture_root / "scripts/prepare_dependency_profile.py"
    script_path.write_text(
        script_path.read_text(encoding="utf-8") + "\nimport subprocess\n",
        encoding="utf-8",
    )

    report = run_audit(fixture_root)

    assert report["decision"] == "fail"
    assert "dependency_script_contains_inner_implementation" in {
        violation["reason"] for violation in report["violations"]
    }


@pytest.mark.constraint
def test_non_exact_direct_dependency_is_rejected(tmp_path: Path) -> None:
    """直接依赖中出现版本范围时 registry 与 harness 必须同时 fail-closed."""

    fixture_root = _copy_governance_fixture(tmp_path)
    direct_path = fixture_root / "configs/dependency_profiles/workflow_orchestrator_direct.txt"
    direct_path.write_text(
        direct_path.read_text(encoding="utf-8").replace(
            "packaging==25.0",
            "packaging>=25.0",
        ),
        encoding="utf-8",
    )

    report = run_audit(fixture_root)

    assert report["decision"] == "fail"
    assert "dependency_profile_registry_invalid" in {
        violation["reason"] for violation in report["violations"]
    }


@pytest.mark.constraint
def test_scientific_context_gate_cannot_be_removed(tmp_path: Path) -> None:
    """运行环境层不得移除科学子解释器上下文 blocker 门禁."""

    fixture_root = _copy_governance_fixture(tmp_path)
    environment_path = fixture_root / "experiments/runtime/repository_environment.py"
    environment_path.write_text(
        environment_path.read_text(encoding="utf-8").replace(
            '"isolated_context_environment_report_path_missing"',
            '"scientific_context_check_removed"',
        ),
        encoding="utf-8",
    )

    report = run_audit(fixture_root)

    assert report["decision"] == "fail"
    assert "isolated_scientific_context_contract_missing" in {
        violation["reason"] for violation in report["violations"]
    }


@pytest.mark.constraint
def test_field_registry_rejects_dynamic_install_fields(tmp_path: Path) -> None:
    """正式字段表不得重新登记 Notebook 动态安装命令."""

    fixture_root = _copy_governance_fixture(tmp_path)
    field_registry_path = fixture_root / "docs/field_registry.md"
    with field_registry_path.open("a", encoding="utf-8") as handle:
        handle.write(
            "| pip_install_command | runtime | none | false | false | false | "
            "不允许的动态命令字段。 |\n"
        )

    report = run_audit(fixture_root)

    assert report["decision"] == "fail"
    assert "dynamic_dependency_fields_forbidden" in {
        violation["reason"] for violation in report["violations"]
    }
