"""Launch the content-survival observation with an explicit host-held key.

The raw key is removed from the bootstrap host environment before any
orchestrator preparation.  It crosses the exact orchestrator boundary only on
stdin, remains an in-memory value there, and is added to the environment of the
single validated observation scientific child.  Reports retain only a
domain-separated digest and role identity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Callable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.protocol.content_survival_observation import (  # noqa: E402
    load_content_survival_observation_protocol,
)
from experiments.runtime.isolated_scientific_execution import (  # noqa: E402
    execute_isolated_scientific_command,
)
from experiments.runtime.repository_environment import (  # noqa: E402
    FORMAL_EXECUTION_COMMIT_ENVIRONMENT_KEY,
    FORMAL_EXECUTION_LOCK_DIGEST_ENVIRONMENT_KEY,
    build_formal_execution_lock,
)
from scripts.run_formal_workflow_host import (  # noqa: E402
    FormalWorkflowHostError,
    _sanitized_environment,
    prepare_exact_orchestrator,
    validate_clean_detached_checkout,
)


KEY_ENVIRONMENT_NAME = "SLM_WM_KEY_MATERIAL"
KEY_DIGEST_DOMAIN = b"slm_wm_content_survival_host_key_identity_v1"
SCIENTIFIC_PROFILE_ID = "sd35_method_runtime_gpu"
HOST_REPORT_SCHEMA = "content_survival_observation_host_report"
HOST_REPORT_SCHEMA_VERSION = 1
_REDACTION = "[REDACTED_SLM_WM_KEY_MATERIAL]"

ProcessRunner = Callable[..., subprocess.CompletedProcess[str]]


class ContentSurvivalObservationHostError(RuntimeError):
    """Reject an unsafe or identity-drifted observation host launch."""


def content_survival_key_identity_digest(key_material: str) -> str:
    """Return the non-reversible, domain-separated identity of a supplied key."""

    if not isinstance(key_material, str) or len(key_material) < 16:
        raise ContentSurvivalObservationHostError(
            "explicit key material must contain at least 16 characters"
        )
    if "\x00" in key_material:
        raise ContentSurvivalObservationHostError(
            "explicit key material cannot contain a NUL character"
        )
    return hashlib.sha256(
        KEY_DIGEST_DOMAIN + b"\0" + key_material.encode("utf-8")
    ).hexdigest()


def _take_host_key_material(environment: dict[str, str]) -> tuple[str, str]:
    """Remove the explicit key from host process state before child preparation."""

    key_material = environment.pop(KEY_ENVIRONMENT_NAME, None)
    if key_material is None:
        raise ContentSurvivalObservationHostError(
            "SLM_WM_KEY_MATERIAL must be explicitly provided by the host"
        )
    digest = content_survival_key_identity_digest(key_material)
    return key_material, digest


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_output_file(root: Path, value: str | Path, label: str) -> Path:
    raw = Path(value).expanduser()
    resolved = raw.resolve() if raw.is_absolute() else (root / raw).resolve()
    try:
        resolved.relative_to((root / "outputs").resolve())
    except ValueError as exc:
        raise ContentSurvivalObservationHostError(
            f"{label} must stay under the repository outputs directory"
        ) from exc
    if resolved.suffix.lower() != ".json":
        raise ContentSurvivalObservationHostError(f"{label} must be a JSON file")
    return resolved


def _redacted_process_text(value: str, key_material: str) -> tuple[str, bool]:
    text = str(value or "")
    leaked = key_material in text
    return (text.replace(key_material, _REDACTION) if leaked else text, leaked)


def _scientific_command_runner(
    *,
    root: Path,
    output_dir: Path,
    key_material: str,
    prompt_count: int = 4,
    process_runner: ProcessRunner = subprocess.run,
) -> tuple[
    Callable[[Sequence[str], Path, Mapping[str, str]], Mapping[str, Any]],
    dict[str, int | bool],
]:
    """Create the only runner allowed to place the key in a child environment."""

    target_script = (root / "scripts/run_content_survival_observation.py").resolve()
    if prompt_count not in {1, 4}:
        raise ContentSurvivalObservationHostError("prompt count must be one or four")
    expected_tail = (
        str(target_script),
        "--repository-root",
        str(root),
        "--output-dir",
        str(output_dir),
    )
    if prompt_count != 4:
        expected_tail = (*expected_tail, "--prompt-count", str(prompt_count))
    state: dict[str, int | bool] = {
        "consumed": False,
        "runner_invocation_count": 0,
        "target_launch_attempt_count": 0,
        "target_launch_completed_count": 0,
        "target_key_environment_prepared_count": 0,
        "rejected_non_target_count": 0,
        "rejected_duplicate_target_count": 0,
        "non_target_key_environment_prepared_count": 0,
    }

    def run_target(
        command: Sequence[str],
        working_directory: Path,
        environment_overrides: Mapping[str, str],
    ) -> Mapping[str, Any]:
        state["runner_invocation_count"] = int(
            state["runner_invocation_count"]
        ) + 1
        normalized = tuple(str(token) for token in command)
        if (
            len(normalized) != len(expected_tail) + 1
            or normalized[1:] != expected_tail
            or Path(working_directory).resolve() != root
        ):
            state["rejected_non_target_count"] = int(
                state["rejected_non_target_count"]
            ) + 1
            raise ContentSurvivalObservationHostError(
                "raw key injection is restricted to the observation scientific child"
            )
        if any(key_material in token for token in normalized):
            raise ContentSurvivalObservationHostError(
                "raw key material cannot appear in scientific child argv"
            )
        if KEY_ENVIRONMENT_NAME in environment_overrides:
            raise ContentSurvivalObservationHostError(
                "shared scientific environment overrides cannot contain raw key material"
            )
        if KEY_ENVIRONMENT_NAME in os.environ:
            raise ContentSurvivalObservationHostError(
                "orchestrator environment unexpectedly contains raw key material"
            )
        if state["consumed"] is True:
            state["rejected_duplicate_target_count"] = int(
                state["rejected_duplicate_target_count"]
            ) + 1
            raise ContentSurvivalObservationHostError(
                "observation scientific child launch is single-use"
            )
        state["consumed"] = True
        state["target_launch_attempt_count"] = int(
            state["target_launch_attempt_count"]
        ) + 1
        environment = dict(os.environ)
        environment.update({str(key): str(value) for key, value in environment_overrides.items()})
        environment[KEY_ENVIRONMENT_NAME] = key_material
        state["target_key_environment_prepared_count"] = int(
            state["target_key_environment_prepared_count"]
        ) + 1
        completed = process_runner(
            list(normalized),
            cwd=root,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
            shell=False,
        )
        state["target_launch_completed_count"] = int(
            state["target_launch_completed_count"]
        ) + 1
        stdout, stdout_leaked = _redacted_process_text(completed.stdout, key_material)
        stderr, stderr_leaked = _redacted_process_text(completed.stderr, key_material)
        if stdout_leaked or stderr_leaked:
            stderr = (stderr + "\n" if stderr else "") + (
                "scientific child emitted raw key material; output was redacted"
            )
            return {"return_code": 86, "stdout": stdout, "stderr": stderr}
        return {
            "return_code": int(completed.returncode),
            "stdout": stdout,
            "stderr": stderr,
        }

    return run_target, state


def _validated_runner_state(
    report: Mapping[str, Any], state: Mapping[str, int | bool]
) -> dict[str, int | bool]:
    """Cross-check runner counters against the shared execution report."""

    expected_fields = {
        "consumed",
        "runner_invocation_count",
        "target_launch_attempt_count",
        "target_launch_completed_count",
        "target_key_environment_prepared_count",
        "rejected_non_target_count",
        "rejected_duplicate_target_count",
        "non_target_key_environment_prepared_count",
    }
    if set(state) != expected_fields:
        raise ContentSurvivalObservationHostError(
            "observation scientific runner state schema drifted"
        )
    resolved = dict(state)
    integer_fields = expected_fields - {"consumed"}
    if type(resolved["consumed"]) is not bool or any(
        type(resolved[field]) is not int or int(resolved[field]) < 0
        for field in integer_fields
    ):
        raise ContentSurvivalObservationHostError(
            "observation scientific runner state is invalid"
        )
    attempts = int(resolved["target_launch_attempt_count"])
    completed = int(resolved["target_launch_completed_count"])
    prepared = int(resolved["target_key_environment_prepared_count"])
    invocations = int(resolved["runner_invocation_count"])
    rejected = int(resolved["rejected_non_target_count"])
    duplicates = int(resolved["rejected_duplicate_target_count"])
    non_target_environments = int(
        resolved["non_target_key_environment_prepared_count"]
    )
    if any(
        (
            attempts not in {0, 1},
            completed not in {0, 1},
            prepared != attempts,
            completed > attempts,
            invocations != attempts + rejected + duplicates,
            resolved["consumed"] is not (attempts == 1),
            non_target_environments != 0,
        )
    ):
        raise ContentSurvivalObservationHostError(
            "observation scientific runner counters are inconsistent"
        )
    execution = report.get("execution")
    if type(execution) is not dict:
        raise ContentSurvivalObservationHostError(
            "shared scientific execution report is missing execution state"
        )
    decision = report.get("decision")
    if decision == "pass":
        if (
            (
                invocations,
                attempts,
                completed,
                prepared,
                rejected,
                duplicates,
                non_target_environments,
            )
            != (1, 1, 1, 1, 0, 0, 0)
            or execution.get("attempted") is not True
            or execution.get("return_code") != 0
        ):
            raise ContentSurvivalObservationHostError(
                "passing scientific report disagrees with runner state"
            )
    elif decision == "fail":
        if execution.get("attempted") is not (attempts == 1):
            raise ContentSurvivalObservationHostError(
                "failed scientific report disagrees with launch attempt"
            )
        if attempts == 0 and any((completed, prepared)):
            raise ContentSurvivalObservationHostError(
                "failed scientific report contains an impossible launch state"
            )
    else:
        raise ContentSurvivalObservationHostError(
            "shared scientific execution decision is invalid"
        )
    return resolved


def _write_host_report(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    payload = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    with partial.open("w", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    partial.replace(path)


def _orchestrator_arguments(arguments: argparse.Namespace) -> list[str]:
    command = [
        "-I",
        str(Path(__file__).resolve()),
        "orchestrator",
        "--repository-root",
        str(arguments.repository_root),
        "--repository-commit",
        arguments.repository_commit,
        "--output-dir",
        str(arguments.output_dir),
        "--scientific-environment-root",
        str(arguments.scientific_environment_root),
        "--scientific-managed-python-root",
        str(arguments.scientific_managed_python_root),
        "--scientific-execution-report",
        str(arguments.scientific_execution_report),
        "--host-report",
        str(arguments.host_report),
        "--orchestrator-profile-id",
        arguments.orchestrator_profile_id,
        "--orchestrator-python-version",
        arguments.orchestrator_python_version,
        "--orchestrator-lock-digest",
        arguments.orchestrator_lock_digest,
        "--orchestrator-python-executable",
        arguments.orchestrator_python_executable,
        "--orchestrator-python-sha256",
        arguments.orchestrator_python_sha256,
    ]
    if arguments.prompt_count != 4:
        command.extend(("--prompt-count", str(arguments.prompt_count)))
    return command


def _run_host(arguments: argparse.Namespace) -> int:
    if sys.flags.isolated != 1:
        raise ContentSurvivalObservationHostError(
            "observation host must be invoked with python -I"
        )
    key_material, _ = _take_host_key_material(os.environ)
    root = validate_clean_detached_checkout(
        arguments.repository_root, arguments.repository_commit
    )
    formal_lock = build_formal_execution_lock(root, arguments.repository_commit)
    python_executable, bootstrap_identity = prepare_exact_orchestrator(
        root=root,
        runtime_root=arguments.orchestrator_runtime_root,
    )
    internal = argparse.Namespace(
        **vars(arguments),
        orchestrator_profile_id=str(bootstrap_identity["profile_id"]),
        orchestrator_python_version=str(bootstrap_identity["python_version"]),
        orchestrator_lock_digest=str(bootstrap_identity["complete_hash_lock_digest"]),
        orchestrator_python_executable=str(bootstrap_identity["python_executable"]),
        orchestrator_python_sha256=str(bootstrap_identity["python_executable_sha256"]),
    )
    command = [str(python_executable), *_orchestrator_arguments(internal)]
    if any(key_material in token for token in command):
        raise ContentSurvivalObservationHostError(
            "raw key material cannot appear in orchestrator argv"
        )
    environment = _sanitized_environment(
        {
            "PATH": str(python_executable.parent)
            + os.pathsep
            + os.environ.get("PATH", ""),
            FORMAL_EXECUTION_COMMIT_ENVIRONMENT_KEY: str(
                formal_lock["formal_execution_commit"]
            ),
            FORMAL_EXECUTION_LOCK_DIGEST_ENVIRONMENT_KEY: str(
                formal_lock["formal_execution_lock_digest"]
            ),
        }
    )
    if KEY_ENVIRONMENT_NAME in environment:
        raise ContentSurvivalObservationHostError(
            "raw key material cannot enter the orchestrator environment"
        )
    completed = subprocess.run(
        command,
        cwd=root,
        env=environment,
        input=key_material,
        check=False,
        capture_output=True,
        text=True,
        shell=False,
    )
    stdout, stdout_leaked = _redacted_process_text(completed.stdout, key_material)
    stderr, stderr_leaked = _redacted_process_text(completed.stderr, key_material)
    if stdout:
        sys.stdout.write(stdout)
    if stderr:
        sys.stderr.write(stderr)
    if stdout_leaked or stderr_leaked:
        raise ContentSurvivalObservationHostError(
            "orchestrator output contained raw key material and was redacted"
        )
    return int(completed.returncode)


def _validate_orchestrator_process(arguments: argparse.Namespace) -> Path:
    root = Path(arguments.repository_root).resolve()
    if os.environ.get(KEY_ENVIRONMENT_NAME) is not None:
        raise ContentSurvivalObservationHostError(
            "orchestrator environment cannot contain raw key material"
        )
    executable = Path(sys.executable).resolve()
    if (
        arguments.orchestrator_profile_id != "workflow_orchestrator"
        or arguments.orchestrator_python_version != "3.12.13"
        or executable != Path(arguments.orchestrator_python_executable).resolve()
        or _sha256(executable) != arguments.orchestrator_python_sha256
    ):
        raise ContentSurvivalObservationHostError(
            "exact orchestrator interpreter identity drifted"
        )
    formal_lock = build_formal_execution_lock(root, arguments.repository_commit)
    if (
        os.environ.get(FORMAL_EXECUTION_COMMIT_ENVIRONMENT_KEY)
        != formal_lock["formal_execution_commit"]
        or os.environ.get(FORMAL_EXECUTION_LOCK_DIGEST_ENVIRONMENT_KEY)
        != formal_lock["formal_execution_lock_digest"]
    ):
        raise ContentSurvivalObservationHostError(
            "published formal execution lock drifted"
        )
    return root


def _run_orchestrator(arguments: argparse.Namespace) -> int:
    if sys.flags.isolated != 1:
        raise ContentSurvivalObservationHostError(
            "exact orchestrator must be invoked with python -I"
        )
    root = _validate_orchestrator_process(arguments)
    key_material = sys.stdin.read()
    key_digest = content_survival_key_identity_digest(key_material)
    protocol = load_content_survival_observation_protocol(root)
    output_dir = Path(arguments.output_dir)
    command_runner, runner_state = _scientific_command_runner(
        root=root,
        output_dir=output_dir,
        key_material=key_material,
        prompt_count=arguments.prompt_count,
    )
    scientific_command = [
        str(root / "scripts/run_content_survival_observation.py"),
        "--repository-root",
        str(root),
        "--output-dir",
        str(output_dir),
    ]
    if arguments.prompt_count != 4:
        scientific_command.extend(("--prompt-count", str(arguments.prompt_count)))
    report, report_path = execute_isolated_scientific_command(
        SCIENTIFIC_PROFILE_ID,
        scientific_command,
        execution_report_path=arguments.scientific_execution_report,
        repository_root=root,
        environment_root=arguments.scientific_environment_root,
        managed_python_root=arguments.scientific_managed_python_root,
        command_runner=command_runner,
    )
    validated_runner_state = _validated_runner_state(report, runner_state)
    report_path = Path(report_path).resolve()
    host_report_path = _resolve_output_file(root, arguments.host_report, "host_report")
    formal_lock = build_formal_execution_lock(root, arguments.repository_commit)
    host_report = {
        "report_schema": HOST_REPORT_SCHEMA,
        "schema_version": HOST_REPORT_SCHEMA_VERSION,
        "operation_kind": "content_survival_observation_host_execution",
        "repository_commit": arguments.repository_commit,
        "formal_execution_lock": formal_lock,
        "orchestrator_identity": {
            "profile_id": arguments.orchestrator_profile_id,
            "python_version": arguments.orchestrator_python_version,
            "complete_hash_lock_digest": arguments.orchestrator_lock_digest,
            "python_executable_sha256": arguments.orchestrator_python_sha256,
        },
        "scientific_profile_id": SCIENTIFIC_PROFILE_ID,
        "scientific_execution_report_path": report_path.relative_to(root).as_posix(),
        "scientific_execution_report_sha256": _sha256(report_path),
        "protocol_identity": protocol.identity_record(),
        "prompt_roster_semantic_digest": protocol.payload["roster"][
            "roster_semantic_digest"
        ],
        "prompt_roster_artifact_file_sha256": protocol.payload["roster"][
            "roster_artifact_file_sha256"
        ],
        "prompt_count": arguments.prompt_count,
        "key_material_identity": {
            "role": "registered_watermark_key_material",
            "digest_domain": KEY_DIGEST_DOMAIN.decode("ascii"),
            "domain_separated_sha256": key_digest,
            "raw_material_persisted": False,
        },
        "scientific_runner_state": validated_runner_state,
        "target_scientific_child_count": validated_runner_state[
            "target_launch_completed_count"
        ],
        "non_target_key_environment_count": validated_runner_state[
            "non_target_key_environment_prepared_count"
        ],
        "decision": "pass" if report.get("decision") == "pass" else "fail",
        "failure_reasons": list(report.get("failure_reasons", [])),
        "diagnostic_only": True,
        "supports_paper_claim": False,
        "candidate_promotion_allowed": False,
        "qualification_evidence": False,
    }
    _write_host_report(host_report_path, host_report)
    invocation = {
        "host_report_path": host_report_path.relative_to(root).as_posix(),
        "host_report_sha256": _sha256(host_report_path),
        "scientific_execution_report_path": report_path.relative_to(root).as_posix(),
        "decision": host_report["decision"],
        "supports_paper_claim": False,
    }
    print(json.dumps(invocation, ensure_ascii=False, sort_keys=True))
    return 0 if report.get("decision") == "pass" else 1


def _common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--repository-commit", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scientific-environment-root", type=Path, required=True)
    parser.add_argument("--scientific-managed-python-root", type=Path, required=True)
    parser.add_argument("--scientific-execution-report", type=Path, required=True)
    parser.add_argument("--host-report", type=Path, required=True)
    parser.add_argument("--prompt-count", type=int, choices=(1, 4), default=4)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the governed content-survival observation host chain."
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)
    host = subparsers.add_parser("host")
    _common_arguments(host)
    host.add_argument("--orchestrator-runtime-root", type=Path, required=True)
    orchestrator = subparsers.add_parser("orchestrator")
    _common_arguments(orchestrator)
    orchestrator.add_argument("--orchestrator-profile-id", required=True)
    orchestrator.add_argument("--orchestrator-python-version", required=True)
    orchestrator.add_argument("--orchestrator-lock-digest", required=True)
    orchestrator.add_argument("--orchestrator-python-executable", required=True)
    orchestrator.add_argument("--orchestrator-python-sha256", required=True)
    return parser


def main() -> int:
    arguments = build_parser().parse_args()
    if arguments.mode == "host":
        return _run_host(arguments)
    return _run_orchestrator(arguments)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ContentSurvivalObservationHostError, FormalWorkflowHostError) as exc:
        print(f"content survival observation host rejected: {exc}", file=sys.stderr)
        raise SystemExit(2) from None
