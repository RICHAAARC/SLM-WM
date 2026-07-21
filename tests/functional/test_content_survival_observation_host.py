from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from scripts import run_content_survival_observation_host as host


pytestmark = pytest.mark.quick


def _synthetic_key() -> str:
    return "-".join(("synthetic", "content", "survival", "key", "cpu", "only"))


def _host_arguments(root: Path) -> argparse.Namespace:
    return argparse.Namespace(
        mode="host",
        repository_root=root,
        repository_commit="a" * 40,
        output_dir=Path("outputs/observation/run"),
        scientific_environment_root=root / "scientific-environments",
        scientific_managed_python_root=root / "managed-pythons",
        scientific_execution_report=Path("outputs/observation/scientific.json"),
        host_report=Path("outputs/observation/host.json"),
        orchestrator_runtime_root=root / "orchestrator",
    )


def _orchestrator_arguments(root: Path) -> argparse.Namespace:
    return argparse.Namespace(
        mode="orchestrator",
        repository_root=root,
        repository_commit="a" * 40,
        output_dir=Path("outputs/observation/run"),
        scientific_environment_root=root / "scientific-environments",
        scientific_managed_python_root=root / "managed-pythons",
        scientific_execution_report=Path("outputs/observation/scientific.json"),
        host_report=Path("outputs/observation/host.json"),
        orchestrator_profile_id="workflow_orchestrator",
        orchestrator_python_version="3.12.13",
        orchestrator_lock_digest="b" * 64,
        orchestrator_python_executable="/exact/orchestrator/python",
        orchestrator_python_sha256="c" * 64,
    )


def test_key_identity_digest_is_domain_separated_and_host_value_is_removed() -> None:
    key = _synthetic_key()
    environment = {host.KEY_ENVIRONMENT_NAME: key, "SAFE": "1"}

    resolved, digest = host._take_host_key_material(environment)

    assert resolved == key
    assert host.KEY_ENVIRONMENT_NAME not in environment
    assert digest == hashlib.sha256(
        host.KEY_DIGEST_DOMAIN + b"\0" + key.encode("utf-8")
    ).hexdigest()
    assert key not in digest


def test_missing_key_rejects_before_checkout_or_environment_preparation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv(host.KEY_ENVIRONMENT_NAME, raising=False)
    monkeypatch.setattr(host.sys, "flags", SimpleNamespace(isolated=1))
    called = {"checkout": 0, "prepare": 0, "process": 0}
    monkeypatch.setattr(
        host,
        "validate_clean_detached_checkout",
        lambda *_: called.__setitem__("checkout", called["checkout"] + 1),
    )
    monkeypatch.setattr(
        host,
        "prepare_exact_orchestrator",
        lambda **_: called.__setitem__("prepare", called["prepare"] + 1),
    )
    monkeypatch.setattr(
        host.subprocess,
        "run",
        lambda *_, **__: called.__setitem__("process", called["process"] + 1),
    )

    with pytest.raises(host.ContentSurvivalObservationHostError, match="explicitly"):
        host._run_host(_host_arguments(tmp_path))

    assert called == {"checkout": 0, "prepare": 0, "process": 0}


def test_host_removes_key_before_preparation_and_uses_stdin_only(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    key = _synthetic_key()
    monkeypatch.setenv(host.KEY_ENVIRONMENT_NAME, key)
    monkeypatch.setattr(host.sys, "flags", SimpleNamespace(isolated=1))
    seen: dict[str, object] = {}

    def validate(root: Path, commit: str) -> Path:
        assert host.KEY_ENVIRONMENT_NAME not in os.environ
        assert commit == "a" * 40
        return Path(root).resolve()

    def build_lock(root: Path, commit: str) -> dict[str, str]:
        assert host.KEY_ENVIRONMENT_NAME not in os.environ
        return {
            "formal_execution_commit": commit,
            "formal_execution_lock_digest": "d" * 64,
        }

    def prepare(**kwargs: object) -> tuple[Path, dict[str, str]]:
        assert host.KEY_ENVIRONMENT_NAME not in os.environ
        python = tmp_path / "orchestrator" / "bin" / "python"
        return python, {
            "profile_id": "workflow_orchestrator",
            "python_version": "3.12.13",
            "complete_hash_lock_digest": "b" * 64,
            "python_executable": str(python),
            "python_executable_sha256": "c" * 64,
        }

    def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        seen.update(command=command, **kwargs)
        assert key not in command
        assert kwargs["input"] == key
        assert host.KEY_ENVIRONMENT_NAME not in kwargs["env"]
        return subprocess.CompletedProcess(command, 0, "safe\n", "")

    monkeypatch.setattr(host, "validate_clean_detached_checkout", validate)
    monkeypatch.setattr(host, "build_formal_execution_lock", build_lock)
    monkeypatch.setattr(host, "prepare_exact_orchestrator", prepare)
    monkeypatch.setattr(host.subprocess, "run", run)

    assert host._run_host(_host_arguments(tmp_path)) == 0
    assert host.KEY_ENVIRONMENT_NAME not in os.environ
    persistent_snapshot = {name: value for name, value in seen.items() if name != "input"}
    assert key not in json.dumps(persistent_snapshot, default=str, sort_keys=True)


def test_only_exact_scientific_child_receives_key_in_memory_environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    key = _synthetic_key()
    root = tmp_path.resolve()
    output = Path("outputs/observation/run")
    seen: dict[str, object] = {}

    def process_runner(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        seen.update(command=command, **kwargs)
        environment = kwargs["env"]
        assert environment[host.KEY_ENVIRONMENT_NAME] == key
        assert key not in command
        return subprocess.CompletedProcess(command, 0, '{"decision":"pass"}\n', "")

    runner, state = host._scientific_command_runner(
        root=root,
        output_dir=output,
        key_material=key,
        process_runner=process_runner,
    )
    result = runner(
        [
            "/isolated/python",
            str(root / "scripts/run_content_survival_observation.py"),
            "--repository-root",
            str(root),
            "--output-dir",
            str(output),
        ],
        root,
        {"SLM_WM_FORMAL_EXECUTION_COMMIT": "a" * 40},
    )

    assert result == {
        "return_code": 0,
        "stdout": '{"decision":"pass"}\n',
        "stderr": "",
    }
    recorded = dict(seen)
    recorded["env"] = {
        name: value
        for name, value in dict(seen["env"]).items()
        if name != host.KEY_ENVIRONMENT_NAME
    }
    assert key not in json.dumps(recorded, default=str, sort_keys=True)
    assert state == {
        "consumed": True,
        "runner_invocation_count": 1,
        "target_launch_attempt_count": 1,
        "target_launch_completed_count": 1,
        "target_key_environment_prepared_count": 1,
        "rejected_non_target_count": 0,
        "rejected_duplicate_target_count": 0,
        "non_target_key_environment_prepared_count": 0,
    }
    assert key not in json.dumps(state, sort_keys=True)


def test_non_target_child_is_rejected_without_process_or_key_environment(
    tmp_path: Path,
) -> None:
    key = _synthetic_key()
    calls: list[object] = []
    runner, state = host._scientific_command_runner(
        root=tmp_path.resolve(),
        output_dir=Path("outputs/observation/run"),
        key_material=key,
        process_runner=lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    with pytest.raises(host.ContentSurvivalObservationHostError, match="restricted"):
        runner(["/isolated/python", "unrelated.py"], tmp_path, {})

    assert calls == []
    assert state["runner_invocation_count"] == 1
    assert state["rejected_non_target_count"] == 1
    assert state["target_launch_attempt_count"] == 0
    assert state["target_key_environment_prepared_count"] == 0
    assert state["non_target_key_environment_prepared_count"] == 0


def test_raw_key_in_child_output_is_redacted_and_fails_closed(tmp_path: Path) -> None:
    key = _synthetic_key()

    def leaking_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 0, f"stdout {key}", f"stderr {key}")

    root = tmp_path.resolve()
    output = Path("outputs/observation/run")
    runner, state = host._scientific_command_runner(
        root=root,
        output_dir=output,
        key_material=key,
        process_runner=leaking_runner,
    )
    result = runner(
        [
            "/isolated/python",
            str(root / "scripts/run_content_survival_observation.py"),
            "--repository-root",
            str(root),
            "--output-dir",
            str(output),
        ],
        root,
        {},
    )

    assert result["return_code"] == 86
    assert key not in result["stdout"]
    assert key not in result["stderr"]
    assert host._REDACTION in result["stdout"]
    assert host._REDACTION in result["stderr"]
    assert state["target_launch_attempt_count"] == 1
    assert state["target_launch_completed_count"] == 1


def test_scientific_command_runner_is_single_use_before_second_key_environment(
    tmp_path: Path,
) -> None:
    key = _synthetic_key()
    root = tmp_path.resolve()
    output = Path("outputs/observation/run")
    key_receipts: list[str] = []

    def process_runner(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        key_receipts.append(dict(kwargs["env"])[host.KEY_ENVIRONMENT_NAME])
        return subprocess.CompletedProcess(command, 0, "safe", "")

    runner, state = host._scientific_command_runner(
        root=root,
        output_dir=output,
        key_material=key,
        process_runner=process_runner,
    )
    command = [
        "/isolated/python",
        str(root / "scripts/run_content_survival_observation.py"),
        "--repository-root",
        str(root),
        "--output-dir",
        str(output),
    ]

    assert runner(command, root, {})["return_code"] == 0
    with pytest.raises(host.ContentSurvivalObservationHostError, match="single-use"):
        runner(command, root, {})

    assert key_receipts == [key]
    assert state == {
        "consumed": True,
        "runner_invocation_count": 2,
        "target_launch_attempt_count": 1,
        "target_launch_completed_count": 1,
        "target_key_environment_prepared_count": 1,
        "rejected_non_target_count": 0,
        "rejected_duplicate_target_count": 1,
        "non_target_key_environment_prepared_count": 0,
    }
    assert key not in json.dumps(state, sort_keys=True)


def test_orchestrator_persists_only_digest_role_and_fixed_identity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    key = _synthetic_key()
    root = tmp_path.resolve()
    (root / "outputs/observation").mkdir(parents=True)
    scientific_path = root / "outputs/observation/scientific.json"
    scientific_path.write_text('{"decision":"pass"}\n', encoding="utf-8")
    protocol = SimpleNamespace(
        payload={
            "roster": {
                "roster_semantic_digest": "1" * 64,
                "roster_artifact_file_sha256": "2" * 64,
            }
        },
        identity_record=lambda: {
            "protocol_version": "content_survival_observation_v1",
            "protocol_semantic_digest": "3" * 64,
        },
    )
    monkeypatch.setattr(host.sys, "flags", SimpleNamespace(isolated=1))
    monkeypatch.setattr(host.sys, "stdin", io.StringIO(key))
    monkeypatch.setattr(host, "_validate_orchestrator_process", lambda _: root)
    monkeypatch.setattr(
        host, "load_content_survival_observation_protocol", lambda _: protocol
    )
    monkeypatch.setattr(
        host,
        "build_formal_execution_lock",
        lambda _root, commit: {
            "formal_execution_commit": commit,
            "formal_execution_lock_digest": "4" * 64,
        },
    )
    received: dict[str, object] = {}
    target_keys: list[str] = []
    original_factory = host._scientific_command_runner

    def runner_factory(**kwargs: object):
        def process_runner(
            command: list[str], **process_kwargs: object
        ) -> subprocess.CompletedProcess[str]:
            target_keys.append(
                dict(process_kwargs["env"])[host.KEY_ENVIRONMENT_NAME]
            )
            return subprocess.CompletedProcess(command, 0, "safe", "")

        return original_factory(**kwargs, process_runner=process_runner)

    monkeypatch.setattr(host, "_scientific_command_runner", runner_factory)

    def execute(profile: str, argv: list[str], **kwargs: object):
        received.update(profile=profile, argv=argv, kwargs=kwargs)
        assert key not in argv
        execution = kwargs["command_runner"](
            ["/isolated/python", *argv],
            root,
            {"SLM_WM_FORMAL_EXECUTION_COMMIT": "a" * 40},
        )
        report = {
            "decision": "pass",
            "failure_reasons": [],
            "execution": {
                "attempted": True,
                "return_code": execution["return_code"],
            },
        }
        scientific_path.write_text(json.dumps(report) + "\n", encoding="utf-8")
        return report, scientific_path

    monkeypatch.setattr(host, "execute_isolated_scientific_command", execute)
    arguments = _orchestrator_arguments(root)

    assert host._run_orchestrator(arguments) == 0

    host_path = root / arguments.host_report
    persisted = host_path.read_text(encoding="utf-8")
    report = json.loads(persisted)
    assert key not in persisted
    assert report["key_material_identity"] == {
        "role": "registered_watermark_key_material",
        "digest_domain": host.KEY_DIGEST_DOMAIN.decode("ascii"),
        "domain_separated_sha256": host.content_survival_key_identity_digest(key),
        "raw_material_persisted": False,
    }
    assert report["formal_execution_lock"]["formal_execution_commit"] == "a" * 40
    assert report["protocol_identity"]["protocol_version"] == (
        "content_survival_observation_v1"
    )
    assert report["prompt_roster_semantic_digest"] == "1" * 64
    assert report["target_scientific_child_count"] == 1
    assert report["non_target_key_environment_count"] == 0
    assert report["scientific_runner_state"] == {
        "consumed": True,
        "runner_invocation_count": 1,
        "target_launch_attempt_count": 1,
        "target_launch_completed_count": 1,
        "target_key_environment_prepared_count": 1,
        "rejected_non_target_count": 0,
        "rejected_duplicate_target_count": 0,
        "non_target_key_environment_prepared_count": 0,
    }
    assert target_keys == [key]
    assert received["profile"] == host.SCIENTIFIC_PROFILE_ID
    assert received["argv"] == [
        str(root / "scripts/run_content_survival_observation.py"),
        "--repository-root",
        str(root),
        "--output-dir",
        str(arguments.output_dir),
    ]


def test_process_launch_oserror_reports_attempt_without_completed_child(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    key = _synthetic_key()
    root = tmp_path.resolve()
    (root / "outputs/observation").mkdir(parents=True)
    scientific_path = root / "outputs/observation/scientific.json"
    protocol = SimpleNamespace(
        payload={
            "roster": {
                "roster_semantic_digest": "1" * 64,
                "roster_artifact_file_sha256": "2" * 64,
            }
        },
        identity_record=lambda: {
            "protocol_version": "content_survival_observation_v1",
            "protocol_semantic_digest": "3" * 64,
        },
    )
    monkeypatch.setattr(host.sys, "flags", SimpleNamespace(isolated=1))
    monkeypatch.setattr(host.sys, "stdin", io.StringIO(key))
    monkeypatch.setattr(host, "_validate_orchestrator_process", lambda _: root)
    monkeypatch.setattr(
        host, "load_content_survival_observation_protocol", lambda _: protocol
    )
    monkeypatch.setattr(
        host,
        "build_formal_execution_lock",
        lambda _root, commit: {
            "formal_execution_commit": commit,
            "formal_execution_lock_digest": "4" * 64,
        },
    )
    process_calls = 0
    original_factory = host._scientific_command_runner

    def runner_factory(**kwargs: object):
        def process_runner(*_: object, **__: object) -> subprocess.CompletedProcess[str]:
            nonlocal process_calls
            process_calls += 1
            raise OSError("synthetic launch failure")

        return original_factory(**kwargs, process_runner=process_runner)

    monkeypatch.setattr(host, "_scientific_command_runner", runner_factory)

    def execute(_profile: str, argv: list[str], **kwargs: object):
        try:
            kwargs["command_runner"](
                ["/isolated/python", *argv],
                root,
                {"SLM_WM_FORMAL_EXECUTION_COMMIT": "a" * 40},
            )
        except OSError:
            report = {
                "decision": "fail",
                "failure_reasons": ["scientific_child_command_launch_failed"],
                "execution": {"attempted": True, "return_code": None},
            }
            scientific_path.write_text(json.dumps(report) + "\n", encoding="utf-8")
            return report, scientific_path
        raise AssertionError("process runner must raise OSError")

    monkeypatch.setattr(host, "execute_isolated_scientific_command", execute)
    arguments = _orchestrator_arguments(root)

    assert host._run_orchestrator(arguments) == 1

    persisted = (root / arguments.host_report).read_text(encoding="utf-8")
    report = json.loads(persisted)
    assert process_calls == 1
    assert report["decision"] == "fail"
    assert report["target_scientific_child_count"] == 0
    assert report["scientific_runner_state"] == {
        "consumed": True,
        "runner_invocation_count": 1,
        "target_launch_attempt_count": 1,
        "target_launch_completed_count": 0,
        "target_key_environment_prepared_count": 1,
        "rejected_non_target_count": 0,
        "rejected_duplicate_target_count": 0,
        "non_target_key_environment_prepared_count": 0,
    }
    assert key not in persisted


def test_empty_orchestrator_input_rejects_before_scientific_environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(host.sys, "flags", SimpleNamespace(isolated=1))
    monkeypatch.setattr(host.sys, "stdin", io.StringIO(""))
    monkeypatch.setattr(
        host, "_validate_orchestrator_process", lambda _: tmp_path.resolve()
    )
    called = {"protocol": 0, "scientific": 0}
    monkeypatch.setattr(
        host,
        "load_content_survival_observation_protocol",
        lambda _: called.__setitem__("protocol", called["protocol"] + 1),
    )
    monkeypatch.setattr(
        host,
        "execute_isolated_scientific_command",
        lambda *_, **__: called.__setitem__("scientific", called["scientific"] + 1),
    )

    with pytest.raises(host.ContentSurvivalObservationHostError, match="at least 16"):
        host._run_orchestrator(_orchestrator_arguments(tmp_path))

    assert called == {"protocol": 0, "scientific": 0}


def test_orchestrator_identity_drift_rejects_before_formal_lock(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv(host.KEY_ENVIRONMENT_NAME, raising=False)
    monkeypatch.setattr(host.sys, "executable", "/unexpected/python")
    called = {"formal_lock": 0}
    monkeypatch.setattr(
        host,
        "build_formal_execution_lock",
        lambda *_: called.__setitem__("formal_lock", called["formal_lock"] + 1),
    )

    with pytest.raises(host.ContentSurvivalObservationHostError, match="identity drifted"):
        host._validate_orchestrator_process(_orchestrator_arguments(tmp_path))

    assert called["formal_lock"] == 0


def test_host_report_and_command_records_contain_no_raw_key(tmp_path: Path) -> None:
    key = _synthetic_key()
    report_path = tmp_path / "outputs/host.json"
    report = {
        "key_material_identity": {
            "role": "registered_watermark_key_material",
            "domain_separated_sha256": host.content_survival_key_identity_digest(key),
        },
        "execution": {
            "argv": ["python", "observation.py"],
            "stdout": "safe",
            "stderr": "",
            "environment_overrides": {"SAFE_IDENTITY": "a" * 64},
        },
    }
    host._write_host_report(report_path, report)

    serialized = report_path.read_text(encoding="utf-8")
    assert key not in serialized
    assert not report_path.with_name(report_path.name + ".partial").exists()
    assert host.KEY_ENVIRONMENT_NAME not in serialized


def test_short_or_nul_key_is_rejected_without_displaying_value() -> None:
    for invalid in ("short", "valid-length-key\x00suffix"):
        with pytest.raises(host.ContentSurvivalObservationHostError) as raised:
            host.content_survival_key_identity_digest(invalid)
        assert invalid not in str(raised.value)
