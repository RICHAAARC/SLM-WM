"""验证三套官方参考复现共享的依赖环境报告协议."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from paper_experiments.runners.official_reference_dependency_environment import (
    validate_official_reference_dependency_environment_report,
)
from tests.helpers.formal_execution_lock import build_test_formal_execution_lock


PROFILE_ID = "tree_ring_official_py39_cu117"


def _ready_profile() -> SimpleNamespace:
    """返回共享 validator 所需的固定 profile 身份."""

    return SimpleNamespace(
        profile_name=PROFILE_ID,
        profile_digest="f" * 64,
        direct_requirements_digest="d" * 64,
        complete_hash_lock_digest="e" * 64,
        complete_hash_lock_dependency_count=24,
    )


def _valid_isolated_report(
    python_executable: Path,
    output_root: Path,
) -> tuple[dict[str, object], dict[str, object]]:
    """构造包含 provision、安装和运行时闭合证据的最小正式报告."""

    formal_execution_lock = build_test_formal_execution_lock("a" * 40)
    python_digest = hashlib.sha256(python_executable.read_bytes()).hexdigest()
    dependency_preparation_report = {
        "profile_id": PROFILE_ID,
        "profile_digest": "f" * 64,
        "complete_hash_lock_digest": "e" * 64,
        "decision": "pass",
        "repository_commit_state": {"all_committed": True},
        "installation": {"attempted": True, "return_code": 0},
        "runtime_comparison": {
            "decision": "pass",
            "environment_match": True,
            "mismatches": [],
        },
        "formal_execution_lock": formal_execution_lock,
    }
    provision_report = {
        "decision": "provisioned",
        "provisioned": True,
        "profile_digest": "f" * 64,
        "formal_execution_lock": formal_execution_lock,
    }
    dependency_report_path = output_root / "dependency_profile_report.json"
    provision_report_path = output_root / "isolated_python_provision_report.json"
    dependency_report_path.parent.mkdir(parents=True, exist_ok=True)
    dependency_report_path.write_text(
        json.dumps(dependency_preparation_report),
        encoding="utf-8",
    )
    provision_report_path.write_text(
        json.dumps(provision_report),
        encoding="utf-8",
    )
    report = {
        "report_schema": "isolated_dependency_environment_preparation_report",
        "schema_version": 1,
        "profile_id": PROFILE_ID,
        "profile_digest": "f" * 64,
        "direct_requirements_digest": "d" * 64,
        "complete_hash_lock_digest": "e" * 64,
        "complete_hash_lock_dependency_count": 24,
        "provisioned": True,
        "formal_preparation_completed": True,
        "formal_ready": True,
        "decision": "pass",
        "failure_reasons": [],
        "supports_paper_claim": False,
        "python_executable_path": str(python_executable),
        "python_executable_sha256": python_digest,
        "python_executable_sha256_after_preparation": python_digest,
        "dependency_preparation_report_path": str(dependency_report_path),
        "dependency_preparation_report_digest": hashlib.sha256(
            dependency_report_path.read_bytes()
        ).hexdigest(),
        "dependency_preparation_report": dependency_preparation_report,
        "provision_report_path": str(provision_report_path),
        "provision_report_digest": hashlib.sha256(
            provision_report_path.read_bytes()
        ).hexdigest(),
        "provision_report": provision_report,
        "formal_execution_lock": formal_execution_lock,
        "formal_execution_commit": formal_execution_lock[
            "formal_execution_commit"
        ],
        "formal_execution_lock_digest": formal_execution_lock[
            "formal_execution_lock_digest"
        ],
        "formal_execution_lock_ready": True,
    }
    return report, formal_execution_lock


@pytest.mark.quick
def test_shared_dependency_environment_validator_accepts_closed_report(
    tmp_path: Path,
) -> None:
    """身份、锁、安装、运行时和解释器证据完整时应返回可信解释器."""

    python_executable = tmp_path / "profile_env" / "bin" / "python"
    python_executable.parent.mkdir(parents=True)
    python_executable.write_text("#!/bin/sh\n", encoding="utf-8")
    report, formal_execution_lock = _valid_isolated_report(
        python_executable,
        tmp_path / "nested_reports",
    )
    report_path = tmp_path / "outputs" / "dependency_profiles" / PROFILE_ID / "report.json"
    report_path.parent.mkdir(parents=True)
    report_path.write_text(json.dumps(report), encoding="utf-8")

    validation = validate_official_reference_dependency_environment_report(
        report,
        report_path,
        _ready_profile(),
        expected_formal_execution_lock=formal_execution_lock,
    )

    assert validation.passed is True
    assert validation.validation_errors == ()
    assert validation.dependency_python_executable == str(python_executable)
    assert validation.dependency_installation_performed is True
    assert len(validation.isolated_dependency_environment_report_digest) == 64


@pytest.mark.quick
def test_shared_dependency_environment_validator_rejects_lock_and_file_drift(
    tmp_path: Path,
) -> None:
    """内存报告与持久化报告或锁摘要漂移时必须 fail-closed."""

    python_executable = tmp_path / "profile_env" / "bin" / "python"
    python_executable.parent.mkdir(parents=True)
    python_executable.write_text("#!/bin/sh\n", encoding="utf-8")
    persisted_report, formal_execution_lock = _valid_isolated_report(
        python_executable,
        tmp_path / "nested_reports",
    )
    report_path = tmp_path / "outputs" / "dependency_profiles" / PROFILE_ID / "report.json"
    report_path.parent.mkdir(parents=True)
    report_path.write_text(json.dumps(persisted_report), encoding="utf-8")
    observed_report = dict(persisted_report)
    observed_report["complete_hash_lock_digest"] = "0" * 64

    validation = validate_official_reference_dependency_environment_report(
        observed_report,
        report_path,
        _ready_profile(),
        expected_formal_execution_lock=formal_execution_lock,
    )

    assert validation.passed is False
    assert "isolated_environment_report_content_mismatch" in validation.validation_errors
    assert "complete_hash_lock_digest_mismatch" in validation.validation_errors
    assert validation.isolated_dependency_environment_report_digest == ""


@pytest.mark.quick
def test_shared_dependency_environment_validator_rejects_python_content_drift(
    tmp_path: Path,
) -> None:
    """隔离解释器在报告形成后发生字节漂移时必须阻断官方命令."""

    python_executable = tmp_path / "profile_env" / "bin" / "python"
    python_executable.parent.mkdir(parents=True)
    python_executable.write_text("#!/bin/sh\n", encoding="utf-8")
    report, formal_execution_lock = _valid_isolated_report(
        python_executable,
        tmp_path / "nested_reports",
    )
    report_path = (
        tmp_path
        / "outputs"
        / "dependency_profiles"
        / PROFILE_ID
        / "report.json"
    )
    report_path.parent.mkdir(parents=True)
    report_path.write_text(json.dumps(report), encoding="utf-8")
    python_executable.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")

    validation = validate_official_reference_dependency_environment_report(
        report,
        report_path,
        _ready_profile(),
        expected_formal_execution_lock=formal_execution_lock,
    )

    assert validation.passed is False
    assert (
        "dependency_python_executable_content_mismatch"
        in validation.validation_errors
    )
