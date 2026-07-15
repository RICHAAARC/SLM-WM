"""验证 fresh-host 到隔离 GPU 方法资格化入口的完整编排接线."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.protocol.gpu_method_qualification import (
    GPU_METHOD_QUALIFICATION_SCHEMA,
)
from main.core.digest import build_stable_digest
from scripts import formal_workflow_entry
from scripts import gpu_method_qualification_host_workflow as workflow
from scripts import run_formal_workflow_host as host


def _sha256(path: Path) -> str:
    """计算测试资格化报告的真实文件摘要."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _qualification_report(
    *,
    repository_commit: str,
    paper_run_name: str,
    prompt_id: str,
    operator_ready: bool,
) -> dict[str, object]:
    """构造满足宿主交叉复验规则的最小资格化报告."""

    report: dict[str, object] = {
        "qualification_report_schema": GPU_METHOD_QUALIFICATION_SCHEMA,
        "qualification_binding": {
            "code_version": repository_commit,
            "dependency_profile_id": workflow.SCIENTIFIC_PROFILE_ID,
            "input_summary": {
                "paper_run_name": paper_run_name,
                "prompt_id": prompt_id,
            },
        },
        "gpu_operator_preflight_ready": operator_ready,
        "gpu_resource_budget_ready": False,
        "supports_paper_claim": False,
    }
    report["qualification_report_digest"] = build_stable_digest(report)
    return report


@pytest.mark.quick
@pytest.mark.parametrize("operator_ready", (False, True))
def test_host_workflow_uses_exact_sd35_child_and_operator_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operator_ready: bool,
) -> None:
    """资源超限不得篡改方法结论, 方法门禁必须传回宿主状态."""

    root = tmp_path / "repository"
    report_dir = root / "outputs/gpu_method_qualification/runtime_run"
    report_dir.mkdir(parents=True)
    repository_commit = "a" * 40
    paper_run_name = "probe_paper"
    prompt_id = "probe_prompt_0001"
    qualification_report = _qualification_report(
        repository_commit=repository_commit,
        paper_run_name=paper_run_name,
        prompt_id=prompt_id,
        operator_ready=operator_ready,
    )
    qualification_report_path = (
        report_dir / "gpu_method_qualification_report.json"
    )
    qualification_report_path.write_text(
        json.dumps(qualification_report, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    invocation = {
        "report_schema": workflow.QUALIFICATION_INVOCATION_RESULT_SCHEMA,
        "schema_version": 1,
        "gpu_method_qualification_report_path": (
            qualification_report_path.relative_to(root).as_posix()
        ),
        "gpu_method_qualification_report_sha256": _sha256(
            qualification_report_path
        ),
        "gpu_method_qualification_report_digest": qualification_report[
            "qualification_report_digest"
        ],
        "gpu_operator_preflight_ready": operator_ready,
        "gpu_resource_budget_ready": False,
        "supports_paper_claim": False,
    }
    captured: dict[str, object] = {}

    def fake_execute_isolated(
        profile_id,
        child_argv,
        *,
        execution_report_path,
        repository_root,
    ):
        """物化真实格式的隔离执行报告, 不伪装为 CUDA 科学证据."""

        captured.update(
            {
                "profile_id": profile_id,
                "child_argv": list(child_argv),
                "repository_root": Path(repository_root),
            }
        )
        isolated_report = {
            "report_schema": "isolated_scientific_execution_report",
            "schema_version": 1,
            "profile_id": profile_id,
            "profile_digest": "b" * 64,
            "direct_requirements_digest": "c" * 64,
            "complete_hash_lock_digest": "d" * 64,
            "complete_hash_lock_dependency_count": 10,
            "dependency_environment_report_path": (
                "outputs/dependency_profiles/sd35/report.json"
            ),
            "dependency_environment_report_digest": "e" * 64,
            "dependency_environment_report_valid": True,
            "python_executable_path": "/managed/python",
            "python_executable_sha256": "f" * 64,
            "formal_execution_commit": repository_commit,
            "formal_execution_lock_ready": True,
            "formal_execution_lock_revalidated_before_child": True,
            "formal_execution_lock_revalidated_after_child": True,
            "python_executable_revalidated_before_child": True,
            "python_executable_revalidated_after_child": True,
            "dependency_environment_report_revalidated_before_child": True,
            "dependency_environment_report_revalidated_after_child": True,
            "execution": {
                "return_code": 0 if operator_ready else 1,
                "stdout": "runtime log\n" + json.dumps(invocation),
                "stderr": "",
            },
            "decision": "pass" if operator_ready else "fail",
            "failure_reasons": (
                [] if operator_ready else ["scientific_child_command_failed"]
            ),
            "supports_paper_claim": False,
        }
        persisted_path = Path(execution_report_path)
        persisted_path.parent.mkdir(parents=True, exist_ok=True)
        persisted_path.write_text(
            json.dumps(isolated_report, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )
        return isolated_report, persisted_path

    monkeypatch.setattr(
        workflow,
        "execute_isolated_scientific_command",
        fake_execute_isolated,
    )
    result = workflow.run_gpu_method_qualification_host_workflow(
        root=root,
        repository_commit=repository_commit,
        paper_run_name=paper_run_name,
        prompt_id=prompt_id,
        result_path="outputs/host/qualification_result.json",
        known_answer="configs/keyed_prg_cross_platform_known_answer.json",
        qualification_output_root="outputs/gpu_method_qualification",
    )

    assert captured["profile_id"] == "sd35_method_runtime_gpu"
    assert captured["repository_root"] == root.resolve()
    child_argv = captured["child_argv"]
    assert child_argv[0] == str(
        root.resolve() / "scripts/run_gpu_method_qualification.py"
    )
    assert child_argv[child_argv.index("--prompt-id") + 1] == prompt_id
    assert result["decision"] == ("pass" if operator_ready else "fail")
    assert result["return_code"] == (0 if operator_ready else 1)
    assert result["workflow_summary"]["gpu_resource_budget_ready"] is False
    assert result["supports_paper_claim"] is False


@pytest.mark.quick
def test_formal_host_builds_qualification_child_command() -> None:
    """公开宿主子命令必须进入统一 formal_workflow_entry, 而非旁路执行科学代码."""

    arguments = host.build_parser().parse_args(
        [
            "--repository-commit",
            "a" * 40,
            "qualification",
            "--paper-run-name",
            "probe_paper",
            "--prompt-id",
            "probe_prompt_0001",
            "--registered-budget",
            "configs/gpu_budget.json",
            "--result-path",
            "outputs/host/qualification_result.json",
        ]
    )
    bootstrap_identity = {
        "profile_id": "workflow_orchestrator",
        "python_version": "3.12.13",
        "complete_hash_lock_digest": "b" * 64,
        "python_executable": "/managed/python",
        "python_executable_sha256": "c" * 64,
    }
    command = host.build_child_command(
        arguments,
        Path("/managed/python"),
        Path("/repository"),
        bootstrap_identity,
    )

    assert command[:3] == [
        str(Path("/managed/python")),
        "-I",
        str(Path("/repository/scripts/formal_workflow_entry.py")),
    ]
    assert command[3] == "qualification"
    assert command[command.index("--prompt-id") + 1] == "probe_prompt_0001"
    assert command[command.index("--registered-budget") + 1] == (
        "configs/gpu_budget.json"
    )
    assert "--workflow" not in command
    assert "--randomization-repeat-id" not in command


@pytest.mark.quick
@pytest.mark.parametrize("operator_ready", (False, True))
def test_formal_entry_propagates_qualification_decision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operator_ready: bool,
) -> None:
    """父编排结果必须由方法门禁决定, 资源门禁不得改变成功状态."""

    bootstrap_identity = {
        "profile_id": "workflow_orchestrator",
        "python_version": "3.12.13",
        "complete_hash_lock_digest": "b" * 64,
        "python_executable": "/managed/python",
        "python_executable_sha256": "c" * 64,
    }
    monkeypatch.setattr(
        formal_workflow_entry,
        "_validate_bootstrap_identity",
        lambda _arguments: bootstrap_identity,
    )
    monkeypatch.setattr(
        formal_workflow_entry,
        "build_formal_execution_lock",
        lambda _root, commit: {"formal_execution_commit": commit},
    )
    monkeypatch.setattr(
        formal_workflow_entry,
        "publish_formal_execution_lock",
        lambda lock: lock,
    )
    monkeypatch.setattr(
        formal_workflow_entry,
        "_require_workflow_orchestrator_environment",
        lambda _root: {
            "profile_id": "workflow_orchestrator",
            "complete_hash_lock_digest": "b" * 64,
        },
    )
    monkeypatch.setattr(
        formal_workflow_entry,
        "run_gpu_method_qualification_host_workflow",
        lambda **_kwargs: {
            "workflow_summary": {
                "workflow_completion_state": (
                    "gpu_operator_preflight_ready"
                    if operator_ready
                    else "gpu_operator_preflight_failed"
                ),
                "gpu_operator_preflight_ready": operator_ready,
                "gpu_resource_budget_ready": False,
                "supports_paper_claim": False,
            },
            "workflow_environment": {
                "scientific_profile_id": "sd35_method_runtime_gpu"
            },
            "decision": "pass" if operator_ready else "fail",
            "failure_reasons": (
                [] if operator_ready else ["gpu_operator_preflight_not_ready"]
            ),
            "supports_paper_claim": False,
        },
    )
    arguments = argparse.Namespace(
        operation="qualification",
        root=str(tmp_path),
        repository_commit="a" * 40,
        paper_run_name="probe_paper",
        result_path="outputs/host/qualification_result.json",
        orchestrator_profile_id="workflow_orchestrator",
        orchestrator_python_version="3.12.13",
        orchestrator_lock_digest="b" * 64,
        orchestrator_python_executable="/managed/python",
        orchestrator_python_executable_sha256="c" * 64,
        workflow=None,
        persistent_output_dir="",
        package_search_root="",
        randomization_repeat_id="",
        prompt_id="probe_prompt_0001",
        known_answer="configs/keyed_prg_cross_platform_known_answer.json",
        registered_budget="",
        qualification_output_root="outputs/gpu_method_qualification",
    )

    payload = formal_workflow_entry.execute(arguments)

    assert payload["decision"] == ("pass" if operator_ready else "fail")
    assert payload["session_execution_decision"] == payload["decision"]
    assert payload["workflow_name"] == "gpu_method_qualification"
    assert payload["workflow_summary"]["gpu_resource_budget_ready"] is False
    assert payload["supports_paper_claim"] is False


@pytest.mark.quick
def test_host_workflow_rejects_report_digest_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """宿主不得只信任子进程返回码或 stdout 中的报告路径."""

    root = tmp_path / "repository"
    report_path = (
        root
        / "outputs/gpu_method_qualification/runtime_run"
        / "gpu_method_qualification_report.json"
    )
    report_path.parent.mkdir(parents=True)
    report = _qualification_report(
        repository_commit="a" * 40,
        paper_run_name="probe_paper",
        prompt_id="probe_prompt_0001",
        operator_ready=True,
    )
    report_path.write_text(json.dumps(report), encoding="utf-8")
    invocation = {
        "report_schema": workflow.QUALIFICATION_INVOCATION_RESULT_SCHEMA,
        "schema_version": 1,
        "gpu_method_qualification_report_path": report_path.relative_to(
            root
        ).as_posix(),
        "gpu_method_qualification_report_sha256": "0" * 64,
        "gpu_method_qualification_report_digest": report[
            "qualification_report_digest"
        ],
        "gpu_operator_preflight_ready": True,
        "gpu_resource_budget_ready": False,
        "supports_paper_claim": False,
    }

    def fake_execute(*_args, execution_report_path, **_kwargs):
        isolated = {
            "report_schema": "isolated_scientific_execution_report",
            "schema_version": 1,
            "profile_id": workflow.SCIENTIFIC_PROFILE_ID,
            "profile_digest": "b" * 64,
            "direct_requirements_digest": "c" * 64,
            "complete_hash_lock_digest": "d" * 64,
            "complete_hash_lock_dependency_count": 10,
            "dependency_environment_report_valid": True,
            "dependency_environment_report_digest": "e" * 64,
            "python_executable_sha256": "f" * 64,
            "formal_execution_commit": "a" * 40,
            "formal_execution_lock_ready": True,
            "formal_execution_lock_revalidated_before_child": True,
            "formal_execution_lock_revalidated_after_child": True,
            "python_executable_revalidated_before_child": True,
            "python_executable_revalidated_after_child": True,
            "dependency_environment_report_revalidated_before_child": True,
            "dependency_environment_report_revalidated_after_child": True,
            "execution": {
                "return_code": 0,
                "stdout": json.dumps(invocation),
                "stderr": "",
            },
            "decision": "pass",
            "supports_paper_claim": False,
        }
        path = Path(execution_report_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(isolated), encoding="utf-8")
        return isolated, path

    monkeypatch.setattr(
        workflow,
        "execute_isolated_scientific_command",
        fake_execute,
    )
    with pytest.raises(ValueError, match="报告文件摘要不一致"):
        workflow.run_gpu_method_qualification_host_workflow(
            root=root,
            repository_commit="a" * 40,
            paper_run_name="probe_paper",
            prompt_id="probe_prompt_0001",
            result_path="outputs/host/result.json",
            known_answer="configs/keyed_prg_cross_platform_known_answer.json",
            qualification_output_root="outputs/gpu_method_qualification",
        )


@pytest.mark.quick
def test_formal_entry_main_returns_nonzero_for_operator_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """资格化方法门禁失败必须穿透父编排并形成非零退出码."""

    monkeypatch.setattr(
        formal_workflow_entry,
        "execute",
        lambda _arguments: {
            "decision": "fail",
            "session_execution_decision": "fail",
            "workflow_completion_state": "gpu_operator_preflight_failed",
            "workflow_name": "gpu_method_qualification",
            "paper_run_name": "probe_paper",
            "profile_id": "workflow_orchestrator",
            "orchestrator_bootstrap_identity": {
                "complete_hash_lock_digest": "b" * 64,
                "python_executable_sha256": "c" * 64,
            },
            "supports_paper_claim": False,
        },
    )
    monkeypatch.setattr(
        formal_workflow_entry,
        "_write_result",
        lambda _root, path, _payload: Path(path),
    )
    exit_code = formal_workflow_entry.main(
        [
            "qualification",
            "--root",
            str(tmp_path),
            "--repository-commit",
            "a" * 40,
            "--paper-run-name",
            "probe_paper",
            "--result-path",
            "outputs/host/result.json",
            "--orchestrator-profile-id",
            "workflow_orchestrator",
            "--orchestrator-python-version",
            "3.12.13",
            "--orchestrator-lock-digest",
            "b" * 64,
            "--orchestrator-python-executable",
            "/managed/python",
            "--orchestrator-python-executable-sha256",
            "c" * 64,
            "--prompt-id",
            "probe_prompt_0001",
        ]
    )

    assert exit_code == 1
