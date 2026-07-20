"""验证精确父解释器内部入口只调用现有正式 workflow."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import formal_workflow_entry as entry


BOOTSTRAP_IDENTITY = {
    "profile_id": "workflow_orchestrator",
    "python_version": "3.12.13",
    "complete_hash_lock_digest": "b" * 64,
    "python_executable": "/runtime/workflow_orchestrator/bin/python",
    "python_executable_sha256": "c" * 64,
}


@pytest.mark.quick
def test_gpu_entry_delegates_to_independent_server_workflow(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """内部入口应把 GPU 配置与持久化解析交给独立服务器 workflow."""

    captured: dict[str, object] = {}

    def fake_run_workflow(
        *arguments: object,
        **keywords: object,
    ) -> dict[str, object]:
        captured["run_arguments"] = arguments
        captured["run_keywords"] = keywords
        return {
            "workflow_summary": {"run_decision": "pass"},
            "archive_record": {},
        }

    monkeypatch.setattr(entry, "run_workflow", fake_run_workflow)
    arguments = argparse.Namespace(
        workflow="external_baseline_tree_ring",
        paper_run_name="probe_paper",
        persistent_output_dir="",
        repository_commit="a" * 40,
        randomization_repeat_id="seed_00_key_00",
        calibration_only=False,
        expected_reference_registry_digest="1" * 64,
        expected_reference_registry_file_sha256="2" * 64,
        calibration_content_strength_sensitivity=False,
    )
    result = entry._gpu_result(arguments, tmp_path)

    assert captured["run_arguments"] == (
        "external_baseline_tree_ring",
        "probe_paper",
        "a" * 40,
        tmp_path,
        None,
    )
    assert captured["run_keywords"] == {
        "randomization_repeat_id": "seed_00_key_00",
        "calibration_only": False,
        "expected_reference_registry_digest": "1" * 64,
        "expected_reference_registry_file_sha256": "2" * 64,
        "content_strength_common_multiplier": 1.0,
        "calibration_content_strength_sensitivity": False,
    }
    assert result["workflow_summary"] == {"run_decision": "pass"}


@pytest.mark.quick
@pytest.mark.parametrize("nominal_ready", (False, True))
def test_content_strength_sensitivity_runs_all_candidates_and_only_selects_nominal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    nominal_ready: bool,
) -> None:
    """三候选固定全跑；非名义候选永不晋升，名义失败即science_blocked。"""

    calls: list[tuple[float, str]] = []

    def fake_run_workflow(
        *_arguments: object,
        **keywords: object,
    ) -> dict[str, object]:
        multiplier = float(keywords["content_strength_common_multiplier"])
        persistent = str(_arguments[4])
        calls.append((multiplier, persistent))
        compatible = nominal_ready if multiplier == 1.0 else True
        candidate_role = {
            0.75: "content_strength_075",
            1.0: "content_strength_100",
            1.25: "content_strength_125",
        }[multiplier]
        return {
            "workflow_summary": {
                "workflow_decision": "calibration_complete",
                "workflow_completion_state": "calibration_complete",
                "content_strength_candidate_role": candidate_role,
                "calibration_protocol_summary": {
                    "protocol_decision": "calibration_complete",
                    "calibration_detection_record_count": 33,
                    "test_prompt_execution_count": 0,
                    "attack_execution_count": 0,
                    "content_strength_common_multiplier": multiplier,
                    "candidate_qualification_compatible": compatible,
                    "formal_parameter_selection_eligible": (
                        multiplier == 1.0 and compatible
                    ),
                },
            },
            "workflow_environment": {},
            "archive_record": {},
        }

    monkeypatch.setattr(entry, "run_workflow", fake_run_workflow)
    arguments = argparse.Namespace(
        workflow="image_only_dataset",
        paper_run_name="probe_paper",
        persistent_output_dir=str(tmp_path / "persistent"),
        repository_commit="a" * 40,
        randomization_repeat_id="seed_00_key_00",
        calibration_only=True,
        expected_reference_registry_digest="1" * 64,
        expected_reference_registry_file_sha256="2" * 64,
        calibration_content_strength_sensitivity=True,
    )

    result = entry._gpu_result(arguments, tmp_path)

    assert [multiplier for multiplier, _ in calls] == [0.75, 1.0, 1.25]
    assert [Path(path).name for _, path in calls] == [
        "content_strength_075",
        "content_strength_100",
        "content_strength_125",
    ]
    summary = result["workflow_summary"]
    assert summary["all_candidate_execution_count"] == 3
    assert summary["all_calibration_prompt_execution_count"] == 99
    assert summary["non_nominal_candidates_descriptive_only"] is True
    assert summary["selected_content_strength_common_multiplier"] == (
        1.0 if nominal_ready else None
    )
    assert summary["protocol_decision"] == (
        "nominal_reference_qualification_compatible"
        if nominal_ready
        else "science_blocked"
    )


@pytest.mark.quick
@pytest.mark.parametrize(
    "invalid_session_shape",
    ("flat_stub", "missing_nested_summary", "outer_role_drift"),
)
def test_content_strength_sensitivity_rejects_invalid_outer_session_shape(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    invalid_session_shape: str,
) -> None:
    """候选聚合必须消费正式session外层身份与嵌套calibration摘要。"""

    call_index = 0

    def fake_run_workflow(
        *_arguments: object,
        **keywords: object,
    ) -> dict[str, object]:
        nonlocal call_index
        multiplier = float(keywords["content_strength_common_multiplier"])
        candidate_role = (
            "content_strength_075",
            "content_strength_100",
            "content_strength_125",
        )[call_index]
        call_index += 1
        calibration_summary = {
            "protocol_decision": "calibration_complete",
            "calibration_detection_record_count": 33,
            "test_prompt_execution_count": 0,
            "attack_execution_count": 0,
            "content_strength_common_multiplier": multiplier,
            "candidate_qualification_compatible": True,
            "formal_parameter_selection_eligible": multiplier == 1.0,
        }
        if invalid_session_shape == "flat_stub":
            session = calibration_summary
        else:
            session = {
                "workflow_decision": "calibration_complete",
                "workflow_completion_state": "calibration_complete",
                "content_strength_candidate_role": (
                    "content_strength_125"
                    if invalid_session_shape == "outer_role_drift"
                    and multiplier == 0.75
                    else candidate_role
                ),
            }
            if invalid_session_shape != "missing_nested_summary":
                session["calibration_protocol_summary"] = calibration_summary
        return {
            "workflow_summary": session,
            "workflow_environment": {},
            "archive_record": {},
        }

    monkeypatch.setattr(entry, "run_workflow", fake_run_workflow)
    arguments = argparse.Namespace(
        workflow="image_only_dataset",
        paper_run_name="probe_paper",
        persistent_output_dir=str(tmp_path / "persistent"),
        repository_commit="a" * 40,
        randomization_repeat_id="seed_00_key_00",
        calibration_only=True,
        expected_reference_registry_digest="1" * 64,
        expected_reference_registry_file_sha256="2" * 64,
        calibration_content_strength_sensitivity=True,
    )

    with pytest.raises(RuntimeError, match="content strength sensitivity"):
        entry._gpu_result(arguments, tmp_path)


@pytest.mark.quick
def test_result_writer_rejects_path_outside_outputs(tmp_path: Path) -> None:
    """Notebook 可读结果不得越过统一 outputs 根目录."""

    with pytest.raises(ValueError, match="outputs"):
        entry._write_result(tmp_path, tmp_path / "outside.json", {"decision": "pass"})

    path = entry._write_result(
        tmp_path,
        "outputs/formal_workflow_execution/probe_paper/result.json",
        {"decision": "pass"},
    )
    assert path.is_file()
    assert not path.with_name(path.name + ".partial").exists()


@pytest.mark.quick
def test_bootstrap_identity_revalidates_current_python_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """内部入口必须复算当前精确解释器摘要, 不能只信任宿主参数."""

    executable = tmp_path / "bin/python"
    executable.parent.mkdir(parents=True)
    executable.write_bytes(b"exact-cpython")
    executable_digest = hashlib.sha256(executable.read_bytes()).hexdigest()
    monkeypatch.setattr(entry.sys, "executable", str(executable))
    monkeypatch.setattr(entry.sys, "flags", SimpleNamespace(isolated=1))
    monkeypatch.setattr(entry.platform, "python_implementation", lambda: "CPython")
    monkeypatch.setattr(entry.platform, "python_version", lambda: "3.12.13")
    monkeypatch.setattr(entry.platform, "system", lambda: "Linux")
    monkeypatch.setattr(entry.platform, "machine", lambda: "x86_64")
    arguments = argparse.Namespace(
        orchestrator_profile_id="workflow_orchestrator",
        orchestrator_python_version="3.12.13",
        orchestrator_lock_digest="b" * 64,
        orchestrator_python_executable=str(executable),
        orchestrator_python_executable_sha256=executable_digest,
    )

    assert entry._validate_bootstrap_identity(arguments) == {
        **BOOTSTRAP_IDENTITY,
        "python_executable": executable.as_posix(),
        "python_executable_sha256": executable_digest,
    }
    arguments.orchestrator_python_executable_sha256 = "d" * 64
    with pytest.raises(RuntimeError, match="摘要不一致"):
        entry._validate_bootstrap_identity(arguments)


@pytest.mark.quick
def test_execute_records_bootstrap_and_verified_orchestrator_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """受治理结果必须同时记录宿主引导身份与父环境复验证据."""

    execution_lock = {"formal_execution_commit": "a" * 40}
    orchestrator_environment = {
        "profile_id": "workflow_orchestrator",
        "complete_hash_lock_digest": "b" * 64,
    }
    monkeypatch.setattr(
        entry,
        "_validate_bootstrap_identity",
        lambda arguments: dict(BOOTSTRAP_IDENTITY),
    )
    monkeypatch.setattr(
        entry,
        "build_formal_execution_lock",
        lambda root, commit: execution_lock,
    )
    monkeypatch.setattr(entry, "publish_formal_execution_lock", lambda lock: lock)
    monkeypatch.setattr(
        entry,
        "_gpu_result",
        lambda arguments, root: {
            "workflow_summary": {"workflow_decision": "complete"},
            "workflow_environment": {"persistent_output_dir": "/persistent"},
            "archive_record": {"archive_ready": True},
            "orchestrator_dependency_environment": orchestrator_environment,
            "randomization_scope": "active_repeat_component",
            "randomization_repeat_id": "seed_00_key_00",
        },
    )
    arguments = argparse.Namespace(
        root=str(tmp_path),
        operation="gpu",
        workflow="image_only_dataset",
        paper_run_name="probe_paper",
        repository_commit="a" * 40,
        randomization_repeat_id="seed_00_key_00",
    )

    result = entry.execute(arguments)

    assert result["orchestrator_bootstrap_identity"] == BOOTSTRAP_IDENTITY
    assert result["orchestrator_dependency_environment"] == orchestrator_environment
    assert result["formal_execution_lock"] == execution_lock
    assert result["workflow_environment"] == {
        "persistent_output_dir": "/persistent"
    }


@pytest.mark.quick
def test_content_runtime_smoke_is_a_named_orchestrator_operation() -> None:
    """The smoke operation carries all three explicit references through parsing."""

    arguments = entry.build_parser().parse_args(
        [
            "content_runtime_smoke",
            "--root", "/repository",
            "--repository-commit", "a" * 40,
            "--paper-run-name", "probe_paper",
            "--result-path", "outputs/smoke.json",
            "--orchestrator-profile-id", "workflow_orchestrator",
            "--orchestrator-python-version", "3.12.13",
            "--orchestrator-lock-digest", "b" * 64,
            "--orchestrator-python-executable", "/managed/python",
            "--orchestrator-python-executable-sha256", "c" * 64,
            "--prompt-id", "probe_prompt_0001",
            "--reference-gradient", "1.0",
            "--reference-response", "0.5",
            "--reference-sensitivity", "0.25",
        ]
    )
    assert arguments.operation == "content_runtime_smoke"
    assert arguments.reference_gradient == 1.0
    assert arguments.smoke_output_root == "outputs/content_runtime_smoke"
