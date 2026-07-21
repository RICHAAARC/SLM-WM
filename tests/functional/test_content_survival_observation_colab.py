from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import pytest

from scripts import run_content_survival_observation_colab as colab


pytestmark = pytest.mark.quick
COMMIT = "a3311cca62b6a33131b95fb6247644fa69f476e7"
WATERMARK = "synthetic-watermark-secret-for-cpu-tests"
HF_TOKEN = "synthetic-huggingface-secret-for-cpu-tests"
NOTEBOOK_SHA256 = {
    "colab_drive_cold_start_smoke.ipynb": "391466464b776cfc8342e3d5e59ffd04abc1e8e81e7a036076ceb626f7e0bb01",
    "content_survival_observation_colab.ipynb": "f84ec7d2d889b99516f55aee82f84d6e037cb1b329da76d5b15ce0acf9db164f",
    "dependency_lock_review_run.ipynb": "ee2fdcfb50b1739f9f79d85c2517336e0432c30d2fb8280d28070c9720a02b2d",
    "external_baseline_gaussian_shading_run.ipynb": "b6d3008d80d3269c97917e8bc89f746d0381826b2ff8dd99755a769a05c9a518",
    "external_baseline_shallow_diffuse_run.ipynb": "861071d48712494559f2b0c4fe4acb8eb55420a52eae3f113cb1ab84156da356",
    "external_baseline_tree_ring_run.ipynb": "2b766dda4f3d56c69f07238ab5195ab2ca6876073ec63ec8e5eaee3dee3c157b",
    "gpu_method_qualification_run.ipynb": "7408aff655d70e0b08db2de0a9e9028a25f4ab362532cc8508945731ebe72b7a",
    "official_reference_gaussian_shading_run.ipynb": "eb429f3ef97fe309a8ed96cd799e049a5d736fc8dc347c05de2621fe1b6ce755",
    "official_reference_shallow_diffuse_run.ipynb": "043b190508cb2cf0f5f87fb39e15433b9d4eed368039dbfc8e3c832b25ed22d9",
    "official_reference_t2smark_run.ipynb": "41d9371f353e29285f48b614e0146e8b7a48d0ee1fbb27d8819af98c25d6ce95",
    "official_reference_tree_ring_run.ipynb": "14c5405f4cb681852277c28a4765a9c86eb17a215fbee916c5fff575fb9d18cf",
    "randomization_repeat_evidence_run.ipynb": "e136207110053f3fdd79e0af2a256537af8e0dab064201762da499d2cd405412",
    "semantic_watermark_image_only_run.ipynb": "0679049bd3a5011b160a7a6f4ddaba1e1a15f427c8c0e0e9115ed3b63da6fd64",
}


def _completed(command: list[str], returncode: int = 0, stdout: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(command, returncode, stdout, "")


class FakeGit:
    def __init__(self, commit: str = COMMIT) -> None:
        self.commit = commit
        self.commands: list[list[str]] = []

    def __call__(self, command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        self.commands.append(command)
        if command[:2] == ["git", "ls-remote"]:
            return _completed(command, stdout=f"{self.commit}\trefs/heads/main\n")
        if command[:3] == ["git", "clone", "--no-checkout"]:
            destination = Path(command[-1])
            destination.mkdir(parents=True)
            (destination / ".git").mkdir()
            return _completed(command)
        if command[:4] == ["git", "remote", "get-url", "origin"]:
            return _completed(command, stdout=colab.PUBLIC_REPOSITORY_URL + "\n")
        if command[:3] == ["git", "rev-parse", "HEAD"]:
            return _completed(command, stdout=COMMIT + "\n")
        if command[:3] == ["git", "status", "--porcelain=v1"]:
            return _completed(command)
        if command[:4] == ["git", "symbolic-ref", "-q", "HEAD"]:
            return _completed(command, returncode=1)
        if command[:3] == ["git", "checkout", "--detach"]:
            return _completed(command)
        raise AssertionError(command)


@pytest.fixture
def local_content(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    content = tmp_path / "content"
    content.mkdir()
    monkeypatch.setattr(colab, "CONTENT_ROOT", content)
    monkeypatch.setattr(colab, "RUN_ROOT", content / "slm_wm_content_survival_runs")
    monkeypatch.setattr(colab, "DELIVERY_ROOT", content / "slm_wm_content_survival_deliveries")
    request_path = content / "slm_wm_content_survival_request.json"
    request_path.write_text(
        json.dumps(
            {
                "request_schema": "content_survival_observation_colab_run_request",
                "schema_version": 1,
                "run_id": "content_survival_observation_colab_20260721T180000Z_a3311cc",
                "repository_commit": COMMIT,
                "drive_delivery_directory": str(content / "drive/MyDrive/SLM/content-survival"),
            }
        ),
        encoding="utf-8",
    )
    bootstrap = content / "slm_wm_content_survival_bootstrap"
    (bootstrap / ".git").mkdir(parents=True)
    return request_path, bootstrap


def _secrets(name: str) -> str:
    values = {colab.WATERMARK_SECRET_NAME: WATERMARK, colab.HF_SECRET_NAME: HF_TOKEN}
    return values[name]


def _gpu(name: str = "NVIDIA A100-SXM4-40GB", total: int = 40536, free: int = 40100) -> colab.GpuIdentity:
    return colab.validate_gpu_identity(
        {
            "name": name,
            "memory_total_mib": total,
            "memory_free_mib": free,
            "driver_version": "550.54.15",
            "cuda_version": "12.8",
            "uuid": "GPU-12345678-1234-1234-1234-123456789abc",
            "utilization_percent": 0,
        }
    )


def _bootstrap_ready(request_path: Path, bootstrap: Path) -> tuple[colab.RunRequest, colab.RunPaths, FakeGit]:
    fake = FakeGit()
    colab.bootstrap_public_run(request_path, bootstrap_root=bootstrap, runner=fake)
    request = colab.load_run_request(request_path)
    paths = colab.run_paths(request)
    (paths.repository_root / "scripts").mkdir(exist_ok=True)
    (paths.repository_root / "scripts/run_content_survival_observation_host.py").write_text("# host\n", encoding="utf-8")
    (paths.repository_root / "configs").mkdir(exist_ok=True)
    (paths.repository_root / "configs/model_sd35.yaml").write_text("model_id: existing\n", encoding="utf-8")
    return request, paths, fake


def test_public_bootstrap_binds_remote_main_and_detached_commit(local_content: tuple[Path, Path]) -> None:
    request_path, bootstrap = local_content
    request, paths, fake = _bootstrap_ready(request_path, bootstrap)
    state = json.loads(paths.state_path.read_text(encoding="utf-8"))
    assert state["run_status"] == "bootstrap_ready"
    assert state["repository_commit"] == COMMIT
    assert state["details"]["resolved_published_commit"] == COMMIT
    assert ["git", "pull"] not in fake.commands
    assert all("token" not in " ".join(command).lower() for command in fake.commands)
    assert request.repository_commit == COMMIT


def test_public_bootstrap_rejects_unpublished_commit_and_leaves_packageable_state(
    local_content: tuple[Path, Path],
) -> None:
    request_path, bootstrap = local_content
    with pytest.raises(colab.ColabObservationError, match="drifted"):
        colab.bootstrap_public_run(request_path, bootstrap_root=bootstrap, runner=FakeGit("f" * 40))
    paths = colab.run_paths(colab.load_run_request(request_path))
    state = json.loads(paths.state_path.read_text(encoding="utf-8"))
    assert state["run_status"] == "preflight_failure"
    assert state["error_category"] == "published_code_identity_rejected"


@pytest.mark.parametrize(
    ("name", "total", "free", "accepted"),
    [
        ("NVIDIA A100-SXM4-40GB", 40536, 40000, True),
        ("NVIDIA A100 80GB PCIe", 81920, 80000, True),
        ("NVIDIA L4", 23034, 22000, False),
        ("NVIDIA A100-SXM4-40GB", 40536, 39000, False),
    ],
)
def test_gpu_gate_is_minimal_a100_40gb_gate(name: str, total: int, free: int, accepted: bool) -> None:
    value = {
        "name": name,
        "memory_total_mib": total,
        "memory_free_mib": free,
        "driver_version": "550.54.15",
        "cuda_version": "12.8",
        "uuid": "GPU-12345678-1234-1234-1234-123456789abc",
        "utilization_percent": 0,
    }
    if accepted:
        assert colab.validate_gpu_identity(value).memory_free_mib == free
    else:
        with pytest.raises(colab.ColabObservationError):
            colab.validate_gpu_identity(value)


def test_preflight_uses_existing_host_and_secret_without_model_governance(
    local_content: tuple[Path, Path],
) -> None:
    request_path, bootstrap = local_content
    _, paths, fake = _bootstrap_ready(request_path, bootstrap)
    report = colab.preflight_run(request_path, runner=fake, gpu_probe=_gpu, secret_getter=_secrets)
    assert report["gpu"]["gpu_name"].startswith("NVIDIA A100")
    assert report["workload"] == {"cell_count": 24, "chain_count": 148, "evaluation_count": 29304}
    text = (paths.evidence_root / "preflight_report.json").read_text(encoding="utf-8")
    assert WATERMARK not in text and HF_TOKEN not in text
    source = Path(colab.__file__).read_text(encoding="utf-8")
    assert "snapshot" not in source.lower()
    assert "local_files_only" not in source
    assert "TRANSFORMERS_OFFLINE" not in source


def test_long_process_output_is_continuously_drained(tmp_path: Path) -> None:
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    capture = colab.execute_drained_process(
        [sys.executable, "-c", "import os; os.write(1, b'x' * 4000000); os.write(2, b'y' * 4000000)"],
        cwd=tmp_path,
        environment=os.environ,
        secrets=[WATERMARK],
        evidence_root=evidence,
        heartbeat_identity={"run_id": "synthetic"},
        idle_timeout_seconds=10,
        heartbeat_interval_seconds=100,
        poll_interval_seconds=0.01,
    )
    assert capture.return_code == 0
    assert capture.stdout_byte_count == 4000000
    assert capture.stderr_byte_count == 4000000
    assert not capture.secret_detected


def test_process_output_secret_is_redacted_and_never_persisted(tmp_path: Path) -> None:
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    environment = dict(os.environ)
    environment["TEST_EMIT_SECRET"] = WATERMARK
    capture = colab.execute_drained_process(
        [sys.executable, "-c", "import os; os.write(1, os.environ['TEST_EMIT_SECRET'].encode())"],
        cwd=tmp_path,
        environment=environment,
        secrets=[WATERMARK],
        evidence_root=evidence,
        heartbeat_identity={"run_id": "synthetic"},
        idle_timeout_seconds=10,
        heartbeat_interval_seconds=100,
        poll_interval_seconds=0.01,
    )
    assert capture.secret_detected
    assert WATERMARK not in capture.stdout_safe_tail
    assert "REDACTED_SECRET" in capture.stdout_safe_tail


def test_idle_timeout_terminates_host_and_descendant_process_group(tmp_path: Path) -> None:
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    child_pid_path = tmp_path / "child.pid"
    code = (
        "import pathlib,subprocess,time; "
        f"p=subprocess.Popen(['{sys.executable}','-c','import time; time.sleep(30)']); "
        f"pathlib.Path({str(child_pid_path)!r}).write_text(str(p.pid)); time.sleep(30)"
    )
    capture = colab.execute_drained_process(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        environment=os.environ,
        secrets=[WATERMARK],
        evidence_root=evidence,
        heartbeat_identity={"run_id": "synthetic"},
        idle_timeout_seconds=0.2,
        heartbeat_interval_seconds=100,
        poll_interval_seconds=0.02,
    )
    assert capture.timed_out and capture.process_group_terminated
    child_pid = int(child_pid_path.read_text(encoding="utf-8"))
    deadline = time.monotonic() + 2
    while Path(f"/proc/{child_pid}").exists() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert not Path(f"/proc/{child_pid}").exists()


def test_run_passes_secrets_only_in_memory_and_persists_minimal_failure(
    local_content: tuple[Path, Path],
) -> None:
    request_path, bootstrap = local_content
    _, paths, fake = _bootstrap_ready(request_path, bootstrap)
    colab.preflight_run(request_path, runner=fake, gpu_probe=_gpu, secret_getter=_secrets)
    observed: dict[str, Any] = {}

    def execute(command: list[str], **kwargs: Any) -> colab.ProcessCapture:
        observed["command"] = command
        observed["environment"] = kwargs["environment"]
        return colab.ProcessCapture(1, 1, 3, "a" * 64, "b" * 64, "", "OOM", False, False, False)

    state = colab.run_observation(request_path, secret_getter=_secrets, process_executor=execute)
    assert state["run_status"] == "scientific_failure"
    assert state["error_category"] == "scientific_host_execution_failed"
    assert WATERMARK not in " ".join(observed["command"])
    assert HF_TOKEN not in " ".join(observed["command"])
    assert observed["environment"]["SLM_WM_KEY_MATERIAL"] == WATERMARK
    assert observed["environment"]["HF_TOKEN"] == HF_TOKEN
    persisted = "\n".join(path.read_text(encoding="utf-8") for path in paths.evidence_root.rglob("*.json"))
    assert WATERMARK not in persisted and HF_TOKEN not in persisted
    usage = json.loads((paths.evidence_root / "secret_usage.json").read_text(encoding="utf-8"))
    assert usage == {"watermark_used": True, "hf_token_used": True}


def test_failure_is_locally_verified_before_drive_mount(local_content: tuple[Path, Path]) -> None:
    request_path, bootstrap = local_content
    request, paths, _ = _bootstrap_ready(request_path, bootstrap)
    colab._state_event(
        request,
        paths,
        run_status="preflight_failure",
        operation="synthetic_preflight",
        error_category="synthetic_failure",
    )
    events: list[str] = []

    def mount(path: str) -> None:
        assert (paths.delivery_root / f"{request.run_id}.tar.gz").is_file()
        assert (paths.delivery_root / f"{request.run_id}.tar.gz.sha256").is_file()
        events.append("mount")
        Path(path).mkdir(parents=True, exist_ok=True)

    def copy(source: str, target: str) -> str:
        events.append(Path(source).suffix or Path(source).name)
        return str(Path(target).write_bytes(Path(source).read_bytes()))

    def unavailable(name: str) -> str:
        raise KeyError(name)

    result = colab.package_and_deliver_to_drive(
        request_path, secret_getter=unavailable, drive_mount=mount, copy_file=copy
    )
    assert result["decision"] == "pass"
    assert result["run_status"] == "preflight_failure"
    assert events[0] == "mount"
    assert events[1:] == [".gz", ".sha256"]


def test_package_cell_recovers_absent_state_from_disk(local_content: tuple[Path, Path]) -> None:
    request_path, _ = local_content
    request = colab.load_run_request(request_path)
    paths = colab.run_paths(request)
    mounted: list[str] = []

    def mount(path: str) -> None:
        mounted.append(path)
        Path(path).mkdir(parents=True, exist_ok=True)

    colab.package_and_deliver_to_drive(
        request_path,
        secret_getter=_secrets,
        drive_mount=mount,
    )
    state = json.loads(paths.state_path.read_text(encoding="utf-8"))
    assert state["run_status"] == "preflight_failure"
    assert state["error_category"] == "controller_state_absent"
    assert mounted == [str(colab.CONTENT_ROOT / "drive")]


def test_secret_leak_blocks_drive_mount(local_content: tuple[Path, Path]) -> None:
    request_path, bootstrap = local_content
    request, paths, _ = _bootstrap_ready(request_path, bootstrap)
    colab._state_event(request, paths, run_status="scientific_failure", operation="synthetic")
    colab._write_json_atomic(
        paths.evidence_root / "secret_usage.json",
        {
            "watermark_used": True,
            "hf_token_used": False,
        },
    )
    (paths.evidence_root / "leak.txt").write_text(WATERMARK, encoding="utf-8")
    mounted: list[str] = []
    with pytest.raises(colab.ColabObservationError, match="secret"):
        colab.package_and_deliver_to_drive(
            request_path,
            secret_getter=_secrets,
            drive_mount=mounted.append,
        )
    assert mounted == []


def test_used_watermark_missing_during_package_blocks_drive_and_copy(
    local_content: tuple[Path, Path],
) -> None:
    request_path, bootstrap = local_content
    request, paths, _ = _bootstrap_ready(request_path, bootstrap)
    colab._state_event(request, paths, run_status="scientific_failure", operation="synthetic")
    colab._write_json_atomic(
        paths.evidence_root / "secret_usage.json",
        {
            "watermark_used": True,
            "hf_token_used": False,
        },
    )
    (paths.evidence_root / "leak.txt").write_text(WATERMARK, encoding="utf-8")
    mounted: list[str] = []
    copied: list[tuple[str, str]] = []

    def unavailable(name: str) -> str:
        raise KeyError(name)

    def copy(source: str, target: str) -> str:
        copied.append((source, target))
        return target

    with pytest.raises(colab.ColabObservationError, match="unavailable"):
        colab.package_and_deliver_to_drive(
            request_path,
            secret_getter=unavailable,
            drive_mount=mounted.append,
            copy_file=copy,
        )
    assert mounted == []
    assert copied == []


def test_used_hf_token_missing_during_package_blocks_drive_and_copy(
    local_content: tuple[Path, Path],
) -> None:
    request_path, bootstrap = local_content
    request, paths, _ = _bootstrap_ready(request_path, bootstrap)
    colab._state_event(request, paths, run_status="success", operation="synthetic")
    colab._write_json_atomic(
        paths.evidence_root / "secret_usage.json",
        {
            "watermark_used": True,
            "hf_token_used": True,
        },
    )
    mounted: list[str] = []
    copied: list[tuple[str, str]] = []

    def watermark_only(name: str) -> str:
        if name == colab.WATERMARK_SECRET_NAME:
            return WATERMARK
        raise KeyError(name)

    def copy(source: str, target: str) -> str:
        copied.append((source, target))
        return target

    with pytest.raises(colab.ColabObservationError, match="unavailable"):
        colab.package_and_deliver_to_drive(
            request_path,
            secret_getter=watermark_only,
            drive_mount=mounted.append,
            copy_file=copy,
        )
    assert mounted == []
    assert copied == []


def test_scientific_state_without_secret_usage_blocks_package(
    local_content: tuple[Path, Path],
) -> None:
    request_path, bootstrap = local_content
    request, paths, _ = _bootstrap_ready(request_path, bootstrap)
    colab._state_event(request, paths, run_status="running", operation="synthetic")
    mounted: list[str] = []
    with pytest.raises(colab.ColabObservationError, match="usage state is absent"):
        colab.package_and_deliver_to_drive(
            request_path,
            secret_getter=_secrets,
            drive_mount=mounted.append,
        )
    assert mounted == []


def test_scientific_secret_usage_mismatch_blocks_package(
    local_content: tuple[Path, Path],
) -> None:
    request_path, bootstrap = local_content
    request, paths, _ = _bootstrap_ready(request_path, bootstrap)
    colab._state_event(request, paths, run_status="success", operation="synthetic")
    colab._write_json_atomic(
        paths.evidence_root / "secret_usage.json",
        {"watermark_used": True, "hf_token_used": "false"},
    )
    mounted: list[str] = []
    with pytest.raises(colab.ColabObservationError, match="usage state is invalid"):
        colab.package_and_deliver_to_drive(
            request_path,
            secret_getter=_secrets,
            drive_mount=mounted.append,
        )
    assert mounted == []


def test_scientific_failure_with_available_used_secrets_can_package(
    local_content: tuple[Path, Path],
) -> None:
    request_path, bootstrap = local_content
    request, paths, _ = _bootstrap_ready(request_path, bootstrap)
    colab._state_event(request, paths, run_status="scientific_failure", operation="synthetic")
    colab._write_json_atomic(
        paths.evidence_root / "secret_usage.json",
        {
            "watermark_used": True,
            "hf_token_used": True,
        },
    )
    mounted: list[str] = []

    def mount(path: str) -> None:
        mounted.append(path)
        Path(path).mkdir(parents=True, exist_ok=True)

    result = colab.package_and_deliver_to_drive(
        request_path,
        secret_getter=_secrets,
        drive_mount=mount,
    )
    assert result["decision"] == "pass"
    assert result["run_status"] == "scientific_failure"
    assert mounted == [str(colab.CONTENT_ROOT / "drive")]


def test_notebook_is_thin_output_free_and_package_cell_is_disk_independent() -> None:
    path = Path("paper_workflow/notebooks/content_survival_observation_colab.ipynb")
    payload = json.loads(path.read_text(encoding="utf-8"))
    code_cells = [cell for cell in payload["cells"] if cell["cell_type"] == "code"]
    assert len(code_cells) == 4
    assert all(cell["outputs"] == [] and cell["execution_count"] is None for cell in code_cells)
    combined = "\n".join("".join(cell["source"]) for cell in code_cells)
    assert colab.PUBLIC_REPOSITORY_URL in combined
    assert "git\", \"pull" not in combined
    assert colab.WATERMARK_SECRET_NAME not in combined
    assert colab.HF_SECRET_NAME not in combined
    assert "from_pretrained" not in combined
    assert "torch" not in combined.lower()
    last = "".join(code_cells[-1]["source"])
    assert "package-drive" in last
    assert "request_path =" in last and "controller =" in last and "bootstrap_root =" in last


def test_notebook_set_has_one_canonical_hash_per_retained_entrypoint() -> None:
    notebooks = sorted(Path("paper_workflow/notebooks").glob("*.ipynb"))
    names = {path.name for path in notebooks}
    assert names == set(NOTEBOOK_SHA256)
    observed = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in notebooks}
    assert observed == NOTEBOOK_SHA256
    assert not Path("notebooks/content_survival_observation_colab.ipynb").exists()
