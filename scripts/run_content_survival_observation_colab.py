"""Run the content-survival observation from a thin Google Colab entrypoint.

The controller deliberately owns only platform concerns: an exact public
GitHub checkout, the A100 memory gate, secret-safe long-process supervision,
minimal on-disk status, and an explicit package-then-Drive handoff.  Scientific
method, model loading, validation, and the 24-cell workload remain in the
existing repository host chain.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
from typing import Any, Callable, Mapping, Sequence


PUBLIC_REPOSITORY_URL = "https://github.com/RICHAAARC/SLM-WM.git"
CONTENT_ROOT = Path("/content")
RUN_ROOT = CONTENT_ROOT / "slm_wm_content_survival_runs"
DELIVERY_ROOT = CONTENT_ROOT / "slm_wm_content_survival_deliveries"
RUN_REQUEST_PATH = CONTENT_ROOT / "slm_wm_content_survival_request.json"
HF_SECRET_NAME = "HF_TOKEN"
DRIVE_INPUT_DIRECTORY = Path("drive/MyDrive/SLM/content-survival/inputs")
DRIVE_RESULTS_DIRECTORY = Path("drive/MyDrive/SLM/content-survival/results")
DRIVE_RUN_REQUEST_NAME = "run_request.json"
DRIVE_WATERMARK_KEY_NAME = "watermark_raw_key.txt"
MINIMUM_GPU_MEMORY_MIB = 40000
HEARTBEAT_INTERVAL_SECONDS = 1800
IDLE_TIMEOUT_SECONDS = 1800
PROCESS_TERMINATION_GRACE_SECONDS = 10
STREAM_TAIL_LIMIT_BYTES = 32768
SAFE_COMMIT = re.compile(r"^[0-9a-f]{40}$")
SAFE_RUN_ID = re.compile(
    r"^content_survival_observation_colab_[0-9]{8}T[0-9]{6}Z_[0-9a-f]{7}$"
)
SAFE_GPU_UUID = re.compile(r"^GPU-[0-9a-fA-F-]{20,}$")
CLAIM_BOUNDARY = {
    "diagnostic_only": True,
    "supports_paper_claim": False,
    "candidate_promotion_allowed": False,
    "qualification_evidence": False,
}
WORKLOAD_IDENTITY = {
    "cell_count": 24,
    "chain_count": 148,
    "evaluation_count": 29304,
}
SECRET_USAGE_REQUIRED_STATUSES = frozenset({"running", "success", "scientific_failure"})

CommandRunner = Callable[..., subprocess.CompletedProcess[Any]]
SecretGetter = Callable[[str], str]
DriveMount = Callable[[str], None]
DriveUnmount = Callable[[], None]
DriveReader = Callable[[Path], bytes]


class ColabObservationError(RuntimeError):
    """Reject an unsafe or incorrect Colab controller operation."""


@dataclass(frozen=True)
class RunRequest:
    run_id: str
    repository_commit: str
    request_sha256: str


@dataclass(frozen=True)
class RunPaths:
    run_root: Path
    repository_root: Path
    runtime_root: Path
    cache_root: Path
    evidence_root: Path
    state_path: Path
    heartbeat_path: Path
    scientific_execution_path: Path
    host_execution_path: Path
    delivery_root: Path


@dataclass(frozen=True)
class GpuIdentity:
    name: str
    memory_total_mib: int
    memory_free_mib: int
    driver_version: str
    cuda_version: str
    uuid: str
    utilization_percent: int

    def record(self) -> dict[str, Any]:
        return {
            "gpu_name": self.name,
            "gpu_memory_total_mib": self.memory_total_mib,
            "gpu_memory_free_mib": self.memory_free_mib,
            "driver_version": self.driver_version,
            "cuda_version": self.cuda_version,
            "gpu_uuid": self.uuid,
            "gpu_utilization_percent": self.utilization_percent,
        }


@dataclass(frozen=True)
class ProcessCapture:
    return_code: int
    stdout_byte_count: int
    stderr_byte_count: int
    stdout_redacted_sha256: str
    stderr_redacted_sha256: str
    stdout_safe_tail: str
    stderr_safe_tail: str
    secret_detected: bool
    timed_out: bool
    process_group_terminated: bool

    def record(self) -> dict[str, Any]:
        return {
            "return_code": self.return_code,
            "stdout_byte_count": self.stdout_byte_count,
            "stderr_byte_count": self.stderr_byte_count,
            "stdout_redacted_sha256": self.stdout_redacted_sha256,
            "stderr_redacted_sha256": self.stderr_redacted_sha256,
            "stdout_safe_tail": self.stdout_safe_tail,
            "stderr_safe_tail": self.stderr_safe_tail,
            "secret_detected": self.secret_detected,
            "timed_out": self.timed_out,
            "process_group_terminated": self.process_group_terminated,
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    with partial.open("w", encoding="utf-8") as handle:
        handle.write(_stable_json(value))
        handle.flush()
        os.fsync(handle.fileno())
    partial.replace(path)


def _write_bytes_atomic(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    with partial.open("wb") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    partial.replace(path)


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ColabObservationError(f"{label} is unavailable or invalid") from exc
    if type(value) is not dict:
        raise ColabObservationError(f"{label} must be a JSON object")
    return value


def _under_content(path: Path, label: str) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(CONTENT_ROOT.resolve())
    except ValueError as exc:
        raise ColabObservationError(f"{label} must stay under /content") from exc
    return resolved


def _run_checked(
    command: Sequence[str],
    *,
    cwd: Path | None = None,
    runner: CommandRunner = subprocess.run,
    operation: str,
) -> subprocess.CompletedProcess[Any]:
    completed = runner(
        list(command),
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
        shell=False,
    )
    if completed.returncode != 0:
        raise ColabObservationError(f"{operation} failed")
    return completed


def _parse_run_request(raw: bytes) -> RunRequest:
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ColabObservationError("run request is invalid") from exc
    required = {
        "request_schema",
        "schema_version",
        "run_id",
        "repository_commit",
    }
    if type(payload) is not dict or set(payload) != required:
        raise ColabObservationError("run request fields drifted")
    if (
        payload["request_schema"]
        != "content_survival_observation_colab_run_request"
        or payload["schema_version"] != 1
    ):
        raise ColabObservationError("run request schema drifted")
    run_id = payload["run_id"]
    commit = payload["repository_commit"]
    if type(run_id) is not str or SAFE_RUN_ID.fullmatch(run_id) is None:
        raise ColabObservationError("run identity is invalid")
    if type(commit) is not str or SAFE_COMMIT.fullmatch(commit) is None:
        raise ColabObservationError("repository commit is invalid")
    if not run_id.endswith("_" + commit[:7]):
        raise ColabObservationError("run identity does not bind repository commit")
    return RunRequest(
        run_id=run_id,
        repository_commit=commit,
        request_sha256=hashlib.sha256(raw).hexdigest(),
    )


def load_run_request(path: Path = RUN_REQUEST_PATH) -> RunRequest:
    """Read one non-secret request for the currently published main commit."""

    request_path = _under_content(path, "run request")
    return _parse_run_request(request_path.read_bytes())


def _drive_mount_root() -> Path:
    return (CONTENT_ROOT / "drive").resolve()


def _drive_input_root() -> Path:
    return (CONTENT_ROOT / DRIVE_INPUT_DIRECTORY).resolve()


def _drive_results_root() -> Path:
    return (CONTENT_ROOT / DRIVE_RESULTS_DIRECTORY).resolve()


def _drive_request_path() -> Path:
    return _drive_input_root() / DRIVE_RUN_REQUEST_NAME


def _drive_watermark_key_path() -> Path:
    return _drive_input_root() / DRIVE_WATERMARK_KEY_NAME


def _default_drive_mount(mount_point: str) -> None:
    from google.colab import drive  # type: ignore[import-not-found]

    drive.mount(mount_point)


def _default_drive_unmount() -> None:
    from google.colab import drive  # type: ignore[import-not-found]

    drive.flush_and_unmount()


def _default_drive_read(path: Path) -> bytes:
    return path.read_bytes()


def prepare_drive_input(
    request_path: Path = RUN_REQUEST_PATH,
    *,
    drive_mount: DriveMount = _default_drive_mount,
    drive_unmount: DriveUnmount = _default_drive_unmount,
    drive_read: DriveReader = _default_drive_read,
) -> dict[str, Any]:
    """Import the governed non-secret request from the fixed Drive boundary."""

    destination = _under_content(request_path, "run request")
    raw: bytes
    drive_mount(str(_drive_mount_root()))
    try:
        raw = drive_read(_drive_request_path())
    except Exception as exc:
        raise ColabObservationError("Drive run request is unavailable") from exc
    finally:
        drive_unmount()
    request = _parse_run_request(raw)
    _write_bytes_atomic(destination, raw)
    return {
        "decision": "pass",
        "run_id": request.run_id,
        "repository_commit": request.repository_commit,
        "run_request_sha256": request.request_sha256,
        "local_request_path": str(destination),
    }


def _read_watermark_key_from_drive(
    *,
    drive_mount: DriveMount,
    drive_unmount: DriveUnmount,
    drive_read: DriveReader,
) -> str:
    """Read the fixed private key file and unmount Drive before returning it."""

    drive_mount(str(_drive_mount_root()))
    try:
        raw = drive_read(_drive_watermark_key_path())
    except Exception as exc:
        raise ColabObservationError("required Drive watermark key is unavailable") from exc
    finally:
        drive_unmount()
    try:
        value = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ColabObservationError("Drive watermark key is invalid") from exc
    if value.endswith("\r\n"):
        value = value[:-2]
    elif value.endswith("\n"):
        value = value[:-1]
    if len(value) < 16 or "\x00" in value or "\n" in value or "\r" in value:
        raise ColabObservationError("Drive watermark key is invalid")
    return value


def run_paths(request: RunRequest) -> RunPaths:
    root = (RUN_ROOT / request.run_id).resolve()
    repository = root / "repository"
    evidence = repository / "outputs/content_survival_observation_colab" / request.run_id
    return RunPaths(
        run_root=root,
        repository_root=repository,
        runtime_root=root / "runtime",
        cache_root=root / "cache",
        evidence_root=evidence,
        state_path=evidence / "controller_state.json",
        heartbeat_path=evidence / "resource_heartbeat.json",
        scientific_execution_path=evidence / "scientific_execution.json",
        host_execution_path=evidence / "host_execution.json",
        delivery_root=(DELIVERY_ROOT / request.run_id).resolve(),
    )


def _state_event(
    request: RunRequest,
    paths: RunPaths,
    *,
    run_status: str,
    operation: str,
    error_category: str | None = None,
    details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    prior = _read_json(paths.state_path, "controller state") if paths.state_path.is_file() else {}
    events = list(prior.get("events", []))
    events.append(
        {
            "operation": operation,
            "run_status": run_status,
            "observed_at": datetime.now(timezone.utc).isoformat(),
            "error_category": error_category,
        }
    )
    state = {
        "state_schema": "content_survival_observation_colab_controller_state",
        "schema_version": 1,
        "run_id": request.run_id,
        "repository_commit": request.repository_commit,
        "run_request_sha256": request.request_sha256,
        "run_status": run_status,
        "error_category": error_category,
        "events": events,
        "details": dict(details or {}),
        **CLAIM_BOUNDARY,
    }
    _write_json_atomic(paths.state_path, state)
    return state


def _remote_main_sha(
    *,
    cwd: Path | None = None,
    runner: CommandRunner = subprocess.run,
) -> str:
    result = _run_checked(
        ["git", "ls-remote", PUBLIC_REPOSITORY_URL, "refs/heads/main"],
        cwd=cwd,
        runner=runner,
        operation="published main lookup",
    )
    fields = result.stdout.strip().split()
    if len(fields) != 2 or fields[1] != "refs/heads/main" or SAFE_COMMIT.fullmatch(fields[0]) is None:
        raise ColabObservationError("published main response is invalid")
    return fields[0]


def _verify_checkout(paths: RunPaths, request: RunRequest, runner: CommandRunner) -> None:
    head = _run_checked(
        ["git", "rev-parse", "HEAD"],
        cwd=paths.repository_root,
        runner=runner,
        operation="repository HEAD verification",
    ).stdout.strip()
    status = _run_checked(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=paths.repository_root,
        runner=runner,
        operation="repository cleanliness verification",
    ).stdout
    symbolic = runner(
        ["git", "symbolic-ref", "-q", "HEAD"],
        cwd=paths.repository_root,
        check=False,
        capture_output=True,
        text=True,
        shell=False,
    )
    origin = _run_checked(
        ["git", "remote", "get-url", "origin"],
        cwd=paths.repository_root,
        runner=runner,
        operation="repository origin verification",
    ).stdout.strip()
    if (
        head != request.repository_commit
        or status
        or symbolic.returncode != 1
        or origin != PUBLIC_REPOSITORY_URL
    ):
        raise ColabObservationError("repository is not clean detached exact public commit")


def bootstrap_public_run(
    request_path: Path = RUN_REQUEST_PATH,
    *,
    bootstrap_root: Path,
    runner: CommandRunner = subprocess.run,
) -> dict[str, Any]:
    """Create an isolated run only when public main equals the requested commit."""

    request = load_run_request(request_path)
    paths = run_paths(request)
    if paths.run_root.exists():
        raise ColabObservationError("run root already exists")
    paths.run_root.mkdir(parents=True, exist_ok=False)
    try:
        bootstrap = _under_content(bootstrap_root, "bootstrap repository")
        bootstrap_origin = _run_checked(
            ["git", "remote", "get-url", "origin"],
            cwd=bootstrap,
            runner=runner,
            operation="bootstrap origin verification",
        ).stdout.strip()
        bootstrap_head = _run_checked(
            ["git", "rev-parse", "HEAD"],
            cwd=bootstrap,
            runner=runner,
            operation="bootstrap HEAD verification",
        ).stdout.strip()
        published = _remote_main_sha(cwd=bootstrap, runner=runner)
        if (
            bootstrap_origin != PUBLIC_REPOSITORY_URL
            or bootstrap_head != request.repository_commit
            or published != request.repository_commit
        ):
            raise ColabObservationError("bootstrap or published main commit drifted")
        _run_checked(
            ["git", "clone", "--no-checkout", PUBLIC_REPOSITORY_URL, str(paths.repository_root)],
            runner=runner,
            operation="isolated public repository clone",
        )
        clone_published = _remote_main_sha(cwd=paths.repository_root, runner=runner)
        if clone_published != request.repository_commit:
            raise ColabObservationError("published main changed during bootstrap")
        _run_checked(
            ["git", "checkout", "--detach", request.repository_commit],
            cwd=paths.repository_root,
            runner=runner,
            operation="detached checkout",
        )
        _verify_checkout(paths, request, runner)
        paths.runtime_root.mkdir()
        paths.cache_root.mkdir()
        paths.evidence_root.mkdir(parents=True)
        return _state_event(
            request,
            paths,
            run_status="bootstrap_ready",
            operation="public_main_checkout",
            details={
                "repository_url": PUBLIC_REPOSITORY_URL,
                "resolved_published_commit": published,
                "repository_root": str(paths.repository_root),
            },
        )
    except Exception as exc:
        paths.evidence_root.mkdir(parents=True, exist_ok=True)
        _state_event(
            request,
            paths,
            run_status="preflight_failure",
            operation="public_main_checkout",
            error_category="published_code_identity_rejected",
            details={"safe_error": type(exc).__name__},
        )
        raise


def validate_gpu_identity(value: Mapping[str, Any]) -> GpuIdentity:
    required = {
        "name",
        "memory_total_mib",
        "memory_free_mib",
        "driver_version",
        "cuda_version",
        "uuid",
        "utilization_percent",
    }
    if set(value) != required:
        raise ColabObservationError("GPU identity fields drifted")
    name = str(value["name"]).strip()
    total = value["memory_total_mib"]
    free = value["memory_free_mib"]
    utilization = value["utilization_percent"]
    if "A100" not in name.upper():
        raise ColabObservationError("content-survival observation requires an A100 GPU")
    if any(type(item) is not int for item in (total, free, utilization)):
        raise ColabObservationError("GPU numeric identity is invalid")
    if total < MINIMUM_GPU_MEMORY_MIB or free < MINIMUM_GPU_MEMORY_MIB:
        raise ColabObservationError("A100 total or available memory is below the 40 GB gate")
    if not 0 <= utilization <= 100:
        raise ColabObservationError("GPU utilization is invalid")
    uuid = str(value["uuid"]).strip()
    if SAFE_GPU_UUID.fullmatch(uuid) is None:
        raise ColabObservationError("GPU UUID is invalid")
    driver = str(value["driver_version"]).strip()
    cuda = str(value["cuda_version"]).strip()
    if not driver or not cuda:
        raise ColabObservationError("GPU software identity is absent")
    return GpuIdentity(name, total, free, driver, cuda, uuid, utilization)


def probe_gpu_identity(runner: CommandRunner = subprocess.run) -> GpuIdentity:
    row = _run_checked(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.free,driver_version,uuid,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        runner=runner,
        operation="GPU query",
    ).stdout.strip().splitlines()
    if len(row) != 1:
        raise ColabObservationError("exactly one visible GPU is required")
    fields = [item.strip() for item in row[0].split(",")]
    header = _run_checked(["nvidia-smi"], runner=runner, operation="CUDA query").stdout
    cuda = re.search(r"CUDA Version:\s*([0-9.]+)", header)
    if len(fields) != 6 or cuda is None:
        raise ColabObservationError("GPU query response is invalid")
    try:
        total, free, utilization = int(fields[1]), int(fields[2]), int(fields[5])
    except ValueError as exc:
        raise ColabObservationError("GPU numeric identity is invalid") from exc
    return validate_gpu_identity(
        {
            "name": fields[0],
            "memory_total_mib": total,
            "memory_free_mib": free,
            "driver_version": fields[3],
            "cuda_version": cuda.group(1),
            "uuid": fields[4],
            "utilization_percent": utilization,
        }
    )


def read_colab_secret(name: str, *, required: bool, getter: SecretGetter | None = None) -> str | None:
    if getter is None:
        try:
            from google.colab import userdata  # type: ignore[import-not-found]
        except (ImportError, ModuleNotFoundError) as exc:
            if required:
                raise ColabObservationError("Colab secret provider is unavailable") from exc
            return None
        getter = userdata.get
    try:
        value = getter(name)
    except Exception as exc:
        if required:
            raise ColabObservationError("required Colab secret is unavailable") from exc
        return None
    if value is None and not required:
        return None
    if not isinstance(value, str) or len(value) < 16 or "\x00" in value:
        raise ColabObservationError("Colab secret is invalid")
    return value


def preflight_run(
    request_path: Path = RUN_REQUEST_PATH,
    *,
    runner: CommandRunner = subprocess.run,
    gpu_probe: Callable[[], GpuIdentity] | None = None,
) -> dict[str, Any]:
    request = load_run_request(request_path)
    paths = run_paths(request)
    try:
        _verify_checkout(paths, request, runner)
        state = _read_json(paths.state_path, "controller state")
        if state.get("run_status") != "bootstrap_ready":
            raise ColabObservationError("preflight requires a fresh bootstrap")
        gpu = gpu_probe() if gpu_probe is not None else probe_gpu_identity(runner)
        if not (paths.repository_root / "scripts/run_content_survival_observation_host.py").is_file():
            raise ColabObservationError("existing scientific host is absent")
        if not (paths.repository_root / "configs/model_sd35.yaml").is_file():
            raise ColabObservationError("existing model runtime config is absent")
        report = {
            "preflight_schema": "content_survival_observation_colab_preflight",
            "schema_version": 1,
            "run_id": request.run_id,
            "repository_commit": request.repository_commit,
            "run_request_sha256": request.request_sha256,
            "colab_runtime_identity": os.environ.get("COLAB_RELEASE_TAG", "unknown"),
            "python": sys.version.split()[0],
            "gpu": gpu.record(),
            "workload": WORKLOAD_IDENTITY,
            **CLAIM_BOUNDARY,
        }
        _write_json_atomic(paths.evidence_root / "preflight_report.json", report)
        _state_event(request, paths, run_status="preflight_ready", operation="runtime_preflight")
        return report
    except Exception as exc:
        paths.evidence_root.mkdir(parents=True, exist_ok=True)
        _state_event(
            request,
            paths,
            run_status="preflight_failure",
            operation="runtime_preflight",
            error_category="runtime_preflight_rejected",
            details={"safe_error": type(exc).__name__},
        )
        raise


class _StreamCollector(threading.Thread):
    def __init__(self, stream: Any, secrets: Sequence[bytes]) -> None:
        super().__init__(daemon=True)
        self.stream = stream
        self.secrets = tuple(item for item in secrets if item)
        self.byte_count = 0
        self.secret_detected = False
        self.digest = hashlib.sha256()
        self.tail = bytearray()

    def _consume(self, payload: bytes) -> None:
        self.byte_count += len(payload)
        redacted = payload
        for secret in self.secrets:
            if secret in redacted:
                self.secret_detected = True
                redacted = redacted.replace(secret, b"[REDACTED_SECRET]")
        self.digest.update(redacted)
        self.tail.extend(redacted)
        if len(self.tail) > STREAM_TAIL_LIMIT_BYTES:
            del self.tail[:-STREAM_TAIL_LIMIT_BYTES]

    def run(self) -> None:
        overlap = max((len(item) for item in self.secrets), default=1) - 1
        pending = b""
        while True:
            chunk = self.stream.read(65536)
            if not chunk:
                break
            combined = pending + chunk
            cut = max(0, len(combined) - overlap) if overlap else len(combined)
            for secret in self.secrets:
                start = max(0, cut - len(secret) + 1)
                position = combined.find(secret, start)
                if position >= 0 and position < cut < position + len(secret):
                    cut = position
            if cut:
                self._consume(combined[:cut])
            pending = combined[cut:]
        if pending:
            self._consume(pending)

    def safe_tail(self) -> str:
        return bytes(self.tail).decode("utf-8", errors="replace")


def terminate_process_group(
    process: subprocess.Popen[bytes],
    *,
    process_group: int | None = None,
    grace_seconds: float = PROCESS_TERMINATION_GRACE_SECONDS,
) -> bool:
    """Terminate the dedicated host session and confirm its process group is gone."""

    group = process_group
    if group is None:
        try:
            group = os.getpgid(process.pid)
        except ProcessLookupError:
            group = process.pid
    try:
        os.killpg(group, 0)
    except ProcessLookupError:
        if process.poll() is None:
            process.wait(timeout=max(0.1, grace_seconds))
        return True
    try:
        os.killpg(group, signal.SIGTERM)
    except ProcessLookupError:
        pass
    deadline = time.monotonic() + max(0.1, grace_seconds)
    while time.monotonic() < deadline:
        try:
            os.killpg(group, 0)
        except ProcessLookupError:
            if process.poll() is None:
                process.wait(timeout=max(0.1, grace_seconds))
            return True
        time.sleep(0.02)
    try:
        os.killpg(group, signal.SIGKILL)
    except ProcessLookupError:
        pass
    if process.poll() is None:
        process.wait(timeout=max(0.1, grace_seconds))
    try:
        os.killpg(group, 0)
    except ProcessLookupError:
        return True
    return False


def _manifest_count(evidence_root: Path) -> int:
    return sum(1 for _ in (evidence_root / "scientific").rglob("cell_manifest.json"))


def execute_drained_process(
    command: Sequence[str],
    *,
    cwd: Path,
    environment: Mapping[str, str],
    secrets: Sequence[str],
    evidence_root: Path,
    heartbeat_identity: Mapping[str, Any],
    idle_timeout_seconds: float = IDLE_TIMEOUT_SECONDS,
    heartbeat_interval_seconds: float = HEARTBEAT_INTERVAL_SECONDS,
    poll_interval_seconds: float = 1.0,
) -> ProcessCapture:
    """Continuously drain output and own the host's complete descendant group."""

    if any(secret and secret in token for token in command for secret in secrets):
        raise ColabObservationError("raw secret entered process argv")
    process = subprocess.Popen(
        list(command),
        cwd=cwd,
        env=dict(environment),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=False,
        shell=False,
        start_new_session=True,
    )
    process_group = process.pid
    assert process.stdout is not None and process.stderr is not None
    encoded = [secret.encode("utf-8") for secret in secrets if secret]
    stdout = _StreamCollector(process.stdout, encoded)
    stderr = _StreamCollector(process.stderr, encoded)
    stdout.start()
    stderr.start()
    started = time.monotonic()
    last_progress = started
    last_heartbeat = started - heartbeat_interval_seconds
    prior_bytes = 0
    prior_manifests = _manifest_count(evidence_root)
    timed_out = False
    terminated = False
    try:
        while process.poll() is None:
            observed_bytes = stdout.byte_count + stderr.byte_count
            observed_manifests = _manifest_count(evidence_root)
            if observed_bytes > prior_bytes or observed_manifests > prior_manifests:
                last_progress = time.monotonic()
            prior_bytes = observed_bytes
            prior_manifests = observed_manifests
            now = time.monotonic()
            if now - last_heartbeat >= heartbeat_interval_seconds:
                _write_json_atomic(
                    evidence_root / "resource_heartbeat.json",
                    {
                        "observed_at": datetime.now(timezone.utc).isoformat(),
                        "host_pid": process.pid,
                        "host_alive": True,
                        "complete_cell_count": observed_manifests,
                        "idle_seconds": max(0.0, now - last_progress),
                        **dict(heartbeat_identity),
                        **CLAIM_BOUNDARY,
                    },
                )
                last_heartbeat = now
            if now - last_progress >= idle_timeout_seconds:
                timed_out = True
                terminated = terminate_process_group(process, process_group=process_group)
                break
            time.sleep(poll_interval_seconds)
    except BaseException:
        terminated = terminate_process_group(process, process_group=process_group)
        raise
    finally:
        try:
            os.killpg(process_group, 0)
        except ProcessLookupError:
            pass
        else:
            terminated = terminate_process_group(process, process_group=process_group)
    stdout.join(timeout=10)
    stderr.join(timeout=10)
    if stdout.is_alive() or stderr.is_alive():
        raise ColabObservationError("process output drain did not terminate")
    return ProcessCapture(
        return_code=int(process.returncode if process.returncode is not None else -1),
        stdout_byte_count=stdout.byte_count,
        stderr_byte_count=stderr.byte_count,
        stdout_redacted_sha256=stdout.digest.hexdigest(),
        stderr_redacted_sha256=stderr.digest.hexdigest(),
        stdout_safe_tail=stdout.safe_tail(),
        stderr_safe_tail=stderr.safe_tail(),
        secret_detected=stdout.secret_detected or stderr.secret_detected,
        timed_out=timed_out,
        process_group_terminated=terminated,
    )


def _host_command(paths: RunPaths, request: RunRequest) -> list[str]:
    return [
        sys.executable,
        "-I",
        str(paths.repository_root / "scripts/run_content_survival_observation_host.py"),
        "host",
        "--repository-root",
        str(paths.repository_root),
        "--repository-commit",
        request.repository_commit,
        "--output-dir",
        str((paths.evidence_root / "scientific").relative_to(paths.repository_root)),
        "--orchestrator-runtime-root",
        str(paths.runtime_root / "orchestrator"),
        "--scientific-environment-root",
        str(paths.runtime_root / "scientific_environments"),
        "--scientific-managed-python-root",
        str(paths.runtime_root / "managed_pythons"),
        "--scientific-execution-report",
        str(paths.scientific_execution_path.relative_to(paths.repository_root)),
        "--host-report",
        str(paths.host_execution_path.relative_to(paths.repository_root)),
    ]


def run_observation(
    request_path: Path = RUN_REQUEST_PATH,
    *,
    secret_getter: SecretGetter | None = None,
    drive_mount: DriveMount = _default_drive_mount,
    drive_unmount: DriveUnmount = _default_drive_unmount,
    drive_read: DriveReader = _default_drive_read,
    process_executor: Callable[..., ProcessCapture] = execute_drained_process,
) -> dict[str, Any]:
    request = load_run_request(request_path)
    paths = run_paths(request)
    state = _read_json(paths.state_path, "controller state")
    if state.get("run_status") != "preflight_ready":
        raise ColabObservationError("observation requires a passing preflight")
    try:
        watermark_secret = _read_watermark_key_from_drive(
            drive_mount=drive_mount,
            drive_unmount=drive_unmount,
            drive_read=drive_read,
        )
    except Exception as exc:
        _state_event(
            request,
            paths,
            run_status="preflight_failure",
            operation="drive_watermark_input",
            error_category="required_watermark_key_unavailable",
            details={"safe_error": type(exc).__name__},
        )
        raise
    hf_secret = read_colab_secret(HF_SECRET_NAME, required=False, getter=secret_getter)
    assert watermark_secret is not None
    _write_json_atomic(
        paths.evidence_root / "secret_usage.json",
        {
            "watermark_used": True,
            "hf_token_used": hf_secret is not None,
        },
    )
    environment = {key: value for key, value in os.environ.items() if key not in {"SLM_WM_KEY_MATERIAL", "HF_TOKEN"}}
    environment.update(
        {
            "SLM_WM_KEY_MATERIAL": watermark_secret,
            "HF_HOME": str(paths.cache_root / "huggingface"),
        }
    )
    if hf_secret is not None:
        environment["HF_TOKEN"] = hf_secret
    _state_event(request, paths, run_status="running", operation="scientific_host_execution")
    try:
        capture = process_executor(
            _host_command(paths, request),
            cwd=paths.repository_root,
            environment=environment,
            secrets=[value for value in (watermark_secret, hf_secret) if value],
            evidence_root=paths.evidence_root,
            heartbeat_identity={"run_id": request.run_id},
        )
        _write_json_atomic(paths.evidence_root / "process_output_record.json", capture.record())
        safe_tail = (capture.stdout_safe_tail + "\n" + capture.stderr_safe_tail).lower()
        if capture.secret_detected:
            category = "secret_output_detected"
        elif capture.timed_out:
            category = "scientific_host_idle_timeout"
        elif "out of memory" in safe_tail:
            category = "gpu_out_of_memory"
        elif capture.return_code != 0:
            category = "scientific_host_execution_failed"
        else:
            return _state_event(
                request,
                paths,
                run_status="success",
                operation="scientific_host_execution",
                details={"complete_cell_count": _manifest_count(paths.evidence_root)},
            )
        return _state_event(
            request,
            paths,
            run_status="scientific_failure",
            operation="scientific_host_execution",
            error_category=category,
            details={"process_output_record_sha256": _sha256(paths.evidence_root / "process_output_record.json")},
        )
    except Exception as exc:
        return _state_event(
            request,
            paths,
            run_status="scientific_failure",
            operation="scientific_host_execution",
            error_category="scientific_host_controller_error",
            details={"safe_error": type(exc).__name__},
        )


def _file_inventory(root: Path, *, excluded: frozenset[str] = frozenset()) -> list[dict[str, Any]]:
    records = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative in excluded:
            continue
        records.append({"path": relative, "size_bytes": path.stat().st_size, "sha256": _sha256(path)})
    return records


def _secret_scan(root: Path, secrets: Sequence[str]) -> dict[str, Any]:
    encoded = [value.encode("utf-8") for value in secrets if value]
    hits: list[str] = []
    for record in _file_inventory(root):
        path = root / record["path"]
        overlap = max((len(item) for item in encoded), default=1) - 1
        pending = b""
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                payload = pending + chunk
                if any(secret in payload for secret in encoded):
                    hits.append(record["path"])
                    break
                pending = payload[-overlap:] if overlap else b""
    if hits:
        raise ColabObservationError("local evidence contains raw secret material")
    return {"decision": "pass", "scanned_file_count": len(_file_inventory(root)), "secret_hit_count": 0}


def _package_secret_usage(
    paths: RunPaths,
    state: Mapping[str, Any],
) -> dict[str, bool]:
    """Validate which secrets were exposed to the scientific host."""

    if state.get("run_status") not in SECRET_USAGE_REQUIRED_STATUSES:
        return {"watermark_used": False, "hf_token_used": False}
    usage_path = paths.evidence_root / "secret_usage.json"
    if not usage_path.is_file():
        raise ColabObservationError("scientific secret usage state is absent")
    usage = _read_json(usage_path, "scientific secret usage state")
    expected_fields = {"watermark_used", "hf_token_used"}
    if (
        set(usage) != expected_fields
        or usage.get("watermark_used") is not True
        or type(usage.get("hf_token_used")) is not bool
    ):
        raise ColabObservationError("scientific secret usage state is invalid")
    return {"watermark_used": True, "hf_token_used": usage["hf_token_used"]}


def _verify_inventory(root: Path, inventory: Sequence[Mapping[str, Any]]) -> None:
    actual = _file_inventory(root, excluded=frozenset({"delivery_inventory.json"}))
    expected = [dict(item) for item in inventory]
    if actual != expected:
        raise ColabObservationError("delivery inventory verification failed")


def package_and_deliver_to_drive(
    request_path: Path = RUN_REQUEST_PATH,
    *,
    secret_getter: SecretGetter | None = None,
    drive_mount: DriveMount = _default_drive_mount,
    drive_unmount: DriveUnmount = _default_drive_unmount,
    drive_read: DriveReader = _default_drive_read,
    copy_file: Callable[[str, str], str] = shutil.copy2,
) -> dict[str, Any]:
    """Package from disk, verify locally, then and only then mount Drive."""

    request = load_run_request(request_path)
    paths = run_paths(request)
    paths.evidence_root.mkdir(parents=True, exist_ok=True)
    if not paths.state_path.is_file():
        _state_event(
            request,
            paths,
            run_status="preflight_failure",
            operation="package_recovery",
            error_category="controller_state_absent",
        )
    state = _read_json(paths.state_path, "controller state")
    usage = _package_secret_usage(paths, state)
    secrets: list[str] = []
    if usage["watermark_used"]:
        secrets.append(
            _read_watermark_key_from_drive(
                drive_mount=drive_mount,
                drive_unmount=drive_unmount,
                drive_read=drive_read,
            )
        )
    if usage["hf_token_used"]:
        hf_token = read_colab_secret(HF_SECRET_NAME, required=True, getter=secret_getter)
        assert hf_token is not None
        secrets.append(hf_token)
    scan = _secret_scan(paths.evidence_root, secrets)
    if paths.delivery_root.exists():
        raise ColabObservationError("local delivery root already exists")
    paths.delivery_root.mkdir(parents=True, exist_ok=False)
    staging = paths.delivery_root / "evidence"
    shutil.copytree(paths.evidence_root, staging)
    summary = {
        "delivery_schema": "content_survival_observation_colab_delivery",
        "schema_version": 1,
        "run_id": request.run_id,
        "repository_commit": request.repository_commit,
        "run_status": state.get("run_status"),
        "error_category": state.get("error_category"),
        "secret_scan": scan,
        "workload": WORKLOAD_IDENTITY,
        **CLAIM_BOUNDARY,
    }
    _write_json_atomic(staging / "delivery_summary.json", summary)
    inventory = _file_inventory(staging, excluded=frozenset({"delivery_inventory.json"}))
    _write_json_atomic(staging / "delivery_inventory.json", {"files": inventory})
    _secret_scan(staging, secrets)
    archive = paths.delivery_root / f"{request.run_id}.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        handle.add(staging, arcname=request.run_id)
    checksum = paths.delivery_root / f"{archive.name}.sha256"
    checksum.write_text(f"{_sha256(archive)}  {archive.name}\n", encoding="utf-8")
    with tempfile.TemporaryDirectory(dir=paths.delivery_root) as temporary:
        with tarfile.open(archive, "r:gz") as handle:
            handle.extractall(temporary, filter="data")
        unpacked = Path(temporary) / request.run_id
        unpacked_inventory = _read_json(unpacked / "delivery_inventory.json", "delivery inventory")
        files = unpacked_inventory.get("files")
        if type(files) is not list:
            raise ColabObservationError("delivery inventory is invalid")
        _verify_inventory(unpacked, files)
    _secret_scan(paths.delivery_root, secrets)
    drive_mount(str(_drive_mount_root()))
    try:
        results_root = _drive_results_root()
        results_root.mkdir(parents=True, exist_ok=True)
        archive_target = results_root / archive.name
        checksum_target = results_root / checksum.name
        if archive_target.exists() or checksum_target.exists():
            raise ColabObservationError("Drive delivery target already exists")
        copy_file(str(archive), str(archive_target))
        copy_file(str(checksum), str(checksum_target))
        if (
            _sha256(archive_target) != _sha256(archive)
            or checksum_target.read_text(encoding="utf-8")
            != checksum.read_text(encoding="utf-8")
        ):
            raise ColabObservationError("Drive delivery verification failed")
    finally:
        drive_unmount()
    return {
        "decision": "pass",
        "archive_path": str(archive_target),
        "checksum_path": str(checksum_target),
        "archive_sha256": _sha256(archive),
        "run_status": state.get("run_status"),
        **CLAIM_BOUNDARY,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the content-survival Colab adapter.")
    subparsers = parser.add_subparsers(dest="operation", required=True)
    prepare = subparsers.add_parser("prepare-drive-input")
    prepare.add_argument("--run-request-json", type=Path, default=RUN_REQUEST_PATH)
    for name in ("preflight", "run", "package-drive"):
        command = subparsers.add_parser(name)
        command.add_argument("--run-request-json", type=Path, default=RUN_REQUEST_PATH)
    bootstrap = subparsers.add_parser("bootstrap-public")
    bootstrap.add_argument("--run-request-json", type=Path, default=RUN_REQUEST_PATH)
    bootstrap.add_argument("--bootstrap-root", type=Path, required=True)
    return parser


def main() -> int:
    arguments = build_parser().parse_args()
    if arguments.operation == "prepare-drive-input":
        result = prepare_drive_input(arguments.run_request_json)
    elif arguments.operation == "bootstrap-public":
        result = bootstrap_public_run(arguments.run_request_json, bootstrap_root=arguments.bootstrap_root)
    elif arguments.operation == "preflight":
        result = preflight_run(arguments.run_request_json)
    elif arguments.operation == "run":
        result = run_observation(arguments.run_request_json)
    else:
        result = package_and_deliver_to_drive(arguments.run_request_json)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    if arguments.operation == "run":
        return 0 if result.get("run_status") == "success" else 1
    return 0 if result.get("decision", "pass") == "pass" else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ColabObservationError as exc:
        print(f"content survival Colab adapter rejected: {exc}", file=sys.stderr)
        raise SystemExit(2) from None
