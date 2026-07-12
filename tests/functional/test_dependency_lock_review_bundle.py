"""验证依赖锁资格化入口的解释器路由与审查包复制."""

from __future__ import annotations

from dataclasses import replace
import hashlib
from io import BytesIO
import json
from pathlib import Path
from typing import Any
from urllib.parse import quote
import zipfile

import pytest

from experiments.runtime.dependency_profiles import DependencyProfile, get_dependency_profile
import scripts.write_dependency_lock_review_bundle as review_bundle


FORMAL_EXECUTION_LOCK = {
    "formal_execution_lock_schema": "clean_detached_git_commit_v1",
    "formal_execution_commit": "a" * 40,
    "formal_execution_head_detached": True,
    "formal_execution_worktree_clean": True,
    "formal_execution_lock_ready": True,
    "formal_execution_lock_digest": "b" * 64,
}


@pytest.mark.quick
@pytest.mark.parametrize(
    "output",
    (
        "uv 0.11.28\n",
        "uv 0.11.28 (x86_64-unknown-linux-gnu)\n",
    ),
)
def test_qualification_uv_version_accepts_fixed_linux_wheel_outputs(
    output: str,
) -> None:
    """固定 Linux wheel 的两种真实版本输出都必须保持同一身份."""

    assert review_bundle._matches_qualification_uv_version_output(output)


@pytest.mark.quick
@pytest.mark.parametrize(
    "output",
    (
        "uv 0.11.27 (x86_64-unknown-linux-gnu)",
        "uv 0.11.28 (aarch64-unknown-linux-gnu)",
        "uv 0.11.28 extra",
    ),
)
def test_qualification_uv_version_rejects_version_or_platform_drift(
    output: str,
) -> None:
    """版本、平台或额外文本漂移不得绕过固定工具身份门禁."""

    assert not review_bundle._matches_qualification_uv_version_output(output)


def _profile(profile_id: str) -> DependencyProfile:
    """读取真实 profile 身份并构造尚未接收目标锁的候选 fixture.

    审查包测试验证的是从缺锁状态生成候选的控制流, 不应随仓库逐个提交
    正式锁而改变前置状态. 需要父编排锁 ready 的测试会在此 fixture 上显式
    构造 ready 记录, 从而分别覆盖两种状态.
    """

    return replace(
        get_dependency_profile(profile_id),
        complete_hash_lock_present=False,
        complete_hash_lock_digest=None,
        complete_hash_lock_dependency_count=0,
        locked_requirements=(),
        formal_ready=False,
        readiness_blockers=("complete_hash_lock_missing",),
    )


def _sha256(path: Path) -> str:
    """独立计算测试断言使用的文件 SHA-256."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _qualification_wheel_bytes() -> bytes:
    """构造可由真实 ZIP 提取门禁识别的确定性 uv wheel fixture."""

    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED) as archive:
        member = zipfile.ZipInfo("uv/uv", date_time=(2024, 1, 1, 0, 0, 0))
        member.external_attr = 0o100755 << 16
        archive.writestr(member, b"qualification uv executable")
    return buffer.getvalue()


def _write_qualification_tool_lock(repository_root: Path) -> Any:
    """写入匹配 fixture wheel 的固定 URL 工具锁并返回离线 downloader."""

    _, wheel_url, _ = review_bundle._read_qualification_tool_lock(
        review_bundle.ROOT / review_bundle.QUALIFICATION_TOOL_LOCK_RELATIVE_PATH
    )
    wheel_bytes = _qualification_wheel_bytes()
    wheel_digest = hashlib.sha256(wheel_bytes).hexdigest()
    tool_lock_path = repository_root / review_bundle.QUALIFICATION_TOOL_LOCK_RELATIVE_PATH
    tool_lock_path.parent.mkdir(parents=True, exist_ok=True)
    tool_lock_path.write_text(
        (
            f"uv=={review_bundle.UV_DISTRIBUTION_VERSION} "
            f"--wheel-url={wheel_url} --hash=sha256:{wheel_digest}\n"
        ),
        encoding="utf-8",
    )

    def download(requested_url: str, destination: Path) -> None:
        assert requested_url == wheel_url
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(wheel_bytes)

    return download


def _wheel_item(package_name: str, version: str) -> dict[str, Any]:
    """构造候选物化器可重新解析的 wheel report 条目."""

    normalized_name = package_name.lower().replace("_", "-").replace(".", "-")
    wheel_name = normalized_name.replace("-", "_")
    wheel_version = quote(version, safe=".!_")
    digest = hashlib.sha256(
        f"review-wheel:{normalized_name}=={version}".encode("utf-8")
    ).hexdigest()
    return {
        "download_info": {
            "url": (
                "https://packages.example.test/wheels/"
                f"{wheel_name}-{wheel_version}-py3-none-any.whl"
            ),
            "archive_info": {"hashes": {"sha256": digest}},
        },
        "is_direct": False,
        "is_yanked": False,
        "requested": True,
        "metadata": {"name": package_name, "version": version},
    }


def _bind_common_apis(
    monkeypatch: pytest.MonkeyPatch,
    profile: DependencyProfile,
) -> None:
    """为资格化函数注入稳定 profile 和正式执行锁."""

    monkeypatch.setattr(
        review_bundle,
        "get_dependency_profile",
        lambda profile_id, path: profile,
    )
    monkeypatch.setattr(
        review_bundle.repository_environment,
        "require_published_formal_execution_lock",
        lambda root: dict(FORMAL_EXECUTION_LOCK),
    )


def _write_candidate_artifacts(
    repository_root: Path,
    profile: DependencyProfile,
) -> tuple[dict[str, Any], Path]:
    """写入与候选物化器成功输出同形的轻量 fixture."""

    output_dir = (
        repository_root
        / review_bundle.candidate_materializer.OUTPUT_RELATIVE_ROOT
        / profile.profile_name
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = (
        output_dir / review_bundle.candidate_materializer.CANDIDATE_LOCK_FILE_NAME
    )
    pip_report_path = (
        output_dir / review_bundle.candidate_materializer.PIP_REPORT_FILE_NAME
    )
    provenance_path = (
        output_dir / review_bundle.candidate_materializer.PROVENANCE_FILE_NAME
    )
    pip_version = "25.1.1"
    install = []
    for specification in profile.direct_requirements:
        dependency = review_bundle.candidate_materializer.parse_exact_requirement_spec(
            specification
        )
        install.append(_wheel_item(dependency.package_name, dependency.version))
    pip_report = {
        "version": "1",
        "pip_version": pip_version,
        "install": install,
        "environment": {
            "implementation_name": "cpython",
            "implementation_version": profile.python_version,
            "os_name": "posix",
            "platform_machine": profile.machine,
            "platform_python_implementation": profile.python_implementation,
            "python_full_version": profile.python_version,
            "python_version": ".".join(profile.python_version.split(".")[:2]),
            "sys_platform": profile.operating_system,
        },
    }
    pip_report_path.write_text(
        json.dumps(pip_report, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    wheels, _ = review_bundle.candidate_materializer.load_resolved_wheels(
        pip_report_path,
        profile,
        expected_pip_version=pip_version,
    )
    candidate_path.write_text(
        review_bundle.candidate_materializer.candidate_lock_text(wheels),
        encoding="utf-8",
    )
    provenance = {
        "report_schema": review_bundle.candidate_materializer.PROVENANCE_SCHEMA,
        "schema_version": review_bundle.candidate_materializer.PROVENANCE_SCHEMA_VERSION,
        "profile_id": profile.profile_name,
        "profile_digest": profile.profile_digest,
        "cuda_version": profile.cuda_version,
        "pytorch_index_url": profile.pytorch_index_url,
        "torch_version": profile.torch_version,
        "torchvision_version": profile.torchvision_version,
        "direct_requirements_path": profile.direct_requirements_path,
        "direct_requirements_digest": profile.direct_requirements_digest,
        "formal_execution_lock": dict(FORMAL_EXECUTION_LOCK),
        "formal_execution_commit": FORMAL_EXECUTION_LOCK["formal_execution_commit"],
        "formal_execution_lock_digest": FORMAL_EXECUTION_LOCK[
            "formal_execution_lock_digest"
        ],
        "decision": "candidate_ready_for_review",
        "failure_reasons": [],
        "supports_paper_claim": False,
        "resolver_return_code": 0,
        "pip_version": pip_version,
        "pip_resolver_report_path": pip_report_path.relative_to(
            repository_root
        ).as_posix(),
        "candidate_lock_path": candidate_path.relative_to(
            repository_root
        ).as_posix(),
        "candidate_hash_source": (
            "pip_install_report.download_info.archive_info.hashes.sha256"
        ),
        "candidate_lock_dependency_count": len(wheels),
        "candidate_lock_logical_digest": (
            review_bundle.candidate_materializer.candidate_lock_logical_digest(wheels)
        ),
    }
    provenance_path.write_text(
        json.dumps(provenance, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return provenance, provenance_path


def _write_launcher_review_bundle_fixture(
    repository_root: Path,
    profile: DependencyProfile,
) -> Path:
    """写出 host launcher 完成后必须重新读取的三文件审查包."""

    bundle_dir = (
        repository_root
        / review_bundle.LOCAL_BUNDLE_RELATIVE_ROOT
        / profile.profile_name
    ).resolve()
    bundle_dir.mkdir(parents=True, exist_ok=True)
    file_names = {
        "candidate_lock": review_bundle.candidate_materializer.CANDIDATE_LOCK_FILE_NAME,
        "pip_resolver_report": (
            review_bundle.candidate_materializer.PIP_REPORT_FILE_NAME
        ),
        "candidate_provenance": (
            review_bundle.candidate_materializer.PROVENANCE_FILE_NAME
        ),
    }
    records = []
    for artifact_role, file_name in file_names.items():
        path = bundle_dir / file_name
        path.write_text(f"{artifact_role}:{profile.profile_name}\n", encoding="utf-8")
        records.append(
            {
                "artifact_role": artifact_role,
                "file_name": file_name,
                "source_path": f"outputs/source/{file_name}",
                "bundle_path": path.relative_to(repository_root).as_posix(),
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            }
        )
    manifest = review_bundle._build_manifest(
        profile,
        repository_root=repository_root,
        local_bundle_dir=bundle_dir,
        drive_bundle_dir=None,
        formal_execution_lock=dict(FORMAL_EXECUTION_LOCK),
    )
    manifest["files"] = records
    manifest["decision"] = review_bundle.SUCCESS_DECISION
    manifest["failure_reasons"] = []
    manifest["diagnostic_message"] = None
    manifest_path = bundle_dir / review_bundle.BUNDLE_MANIFEST_FILE_NAME
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def _qualification_command_runner(
    repository_root: Path,
    profile: DependencyProfile,
    *,
    write_review_bundle: bool,
) -> tuple[Any, list[dict[str, Any]]]:
    """构造不访问网络的精确 Python/uv 资格化命令 runner."""

    records: list[dict[str, Any]] = []

    def run(
        command: list[str],
        working_directory: Path,
        environment_overrides: dict[str, str],
    ) -> dict[str, Any]:
        record = {
            "command": list(command),
            "working_directory": working_directory,
            "environment_overrides": dict(environment_overrides),
        }
        records.append(record)
        if command[-1:] == ["--version"]:
            return {
                "return_code": 0,
                "stdout": f"uv {review_bundle.UV_DISTRIBUTION_VERSION}\n",
                "stderr": "",
            }
        if command[1:3] == ["python", "install"]:
            return {"return_code": 0, "stdout": "", "stderr": ""}
        if command[1:2] == ["venv"]:
            python_path = Path(command[-1]) / "bin/python"
            python_path.parent.mkdir(parents=True, exist_ok=True)
            python_path.write_bytes(b"exact orchestrator python")
            return {"return_code": 0, "stdout": "", "stderr": ""}
        if command[1:3] == ["-m", "ensurepip"]:
            return {"return_code": 0, "stdout": "", "stderr": ""}
        if command[1:2] == ["-c"]:
            return {
                "return_code": 0,
                "stdout": _profile("workflow_orchestrator").python_version + "\n",
                "stderr": "",
            }
        assert command[1] == "-I"
        assert Path(command[2]).name == "write_dependency_lock_review_bundle.py"
        assert environment_overrides[
            review_bundle.QUALIFICATION_CHILD_ENVIRONMENT_KEY
        ] == "1"
        assert command[-2:] == ["--profile", profile.profile_name]
        if write_review_bundle:
            _write_launcher_review_bundle_fixture(repository_root, profile)
        return {"return_code": 0, "stdout": "child complete\n", "stderr": ""}

    return run, records


@pytest.mark.quick
@pytest.mark.parametrize(
    "profile_id",
    review_bundle.REQUIRED_DEPENDENCY_PROFILE_NAMES,
)
def test_fresh_linux_host_launches_every_profile_from_exact_orchestrator_python(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    profile_id: str,
) -> None:
    """六个候选入口都必须先创建精确 orchestrator 子解释器."""

    profile = _profile(profile_id)
    ready_orchestrator = replace(
        _profile("workflow_orchestrator"),
        complete_hash_lock_present=True,
        complete_hash_lock_digest="c" * 64,
        complete_hash_lock_dependency_count=1,
        locked_requirements=(
            "uv==0.11.28 --hash=sha256:" + "d" * 64,
        ),
        formal_ready=True,
        readiness_blockers=(),
    )

    def get_profile(requested_profile_id: str, path: Path) -> DependencyProfile:
        if (
            profile.profile_name in review_bundle.ISOLATED_PYTHON_PROFILE_IDS
            and requested_profile_id == review_bundle.WORKFLOW_ORCHESTRATOR_PROFILE_ID
        ):
            return ready_orchestrator
        return _profile(requested_profile_id)

    monkeypatch.setattr(review_bundle, "get_dependency_profile", get_profile)
    monkeypatch.setattr(
        review_bundle.repository_environment,
        "require_published_formal_execution_lock",
        lambda root: dict(FORMAL_EXECUTION_LOCK),
    )
    monkeypatch.setattr(review_bundle.platform, "system", lambda: "Linux")
    monkeypatch.setattr(review_bundle.platform, "machine", lambda: "x86_64")
    tool_lock_path = tmp_path / review_bundle.QUALIFICATION_TOOL_LOCK_RELATIVE_PATH
    tool_downloader = _write_qualification_tool_lock(tmp_path)
    command_runner, command_records = _qualification_command_runner(
        tmp_path,
        profile,
        write_review_bundle=True,
    )

    report, report_path = review_bundle.launch_dependency_lock_qualification(
        profile.profile_name,
        repository_root=tmp_path,
        qualification_runtime_root=tmp_path / "qualification_runtime",
        command_runner=command_runner,
        tool_downloader=tool_downloader,
    )

    assert report["decision"] == review_bundle.QUALIFICATION_SUCCESS_DECISION
    assert report["failure_reasons"] == []
    assert report["supports_paper_claim"] is False
    assert report["manifest_path"].endswith(
        f"/{profile.profile_name}/{review_bundle.BUNDLE_MANIFEST_FILE_NAME}"
    )
    assert len(report["command_results"]) == 6
    assert report_path.is_file()
    assert json.loads(report_path.read_text(encoding="utf-8")) == report
    assert report["qualification_tool_wheel_sha256"] is not None
    assert report["qualification_tool_wheel_member"] == "uv/uv"
    assert report["qualification_tool_lock_path"].endswith(tool_lock_path.name)
    assert command_records[0]["command"][-1] == "--version"
    assert command_records[1]["command"][2] == "install"
    assert "3.12.13" in command_records[1]["command"]
    assert command_records[2]["command"][1] == "venv"
    assert all("pip" not in record["command"][1:3] for record in command_records)
    child_record = command_records[-1]
    assert child_record["command"][-2:] == ["--profile", profile.profile_name]
    child_python = Path(child_record["command"][0])
    assert child_python == Path(report["python_executable_path"])
    assert child_record["environment_overrides"][
        review_bundle.QUALIFICATION_PYTHON_DIGEST_ENVIRONMENT_KEY
    ] == report["python_executable_sha256"]
    assert child_record["environment_overrides"]["PATH"].startswith(
        str(child_python.parent)
    )


@pytest.mark.quick
def test_host_launcher_rejects_zero_exit_without_review_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """child 仅返回0但未写受治理审查包时不得形成成功报告."""

    profile = _profile("workflow_orchestrator")
    monkeypatch.setattr(
        review_bundle,
        "get_dependency_profile",
        lambda requested_profile_id, path: _profile(requested_profile_id),
    )
    monkeypatch.setattr(
        review_bundle.repository_environment,
        "require_published_formal_execution_lock",
        lambda root: dict(FORMAL_EXECUTION_LOCK),
    )
    monkeypatch.setattr(review_bundle.platform, "system", lambda: "Linux")
    monkeypatch.setattr(review_bundle.platform, "machine", lambda: "x86_64")
    tool_downloader = _write_qualification_tool_lock(tmp_path)
    command_runner, _ = _qualification_command_runner(
        tmp_path,
        profile,
        write_review_bundle=False,
    )

    report, _ = review_bundle.launch_dependency_lock_qualification(
        profile.profile_name,
        repository_root=tmp_path,
        qualification_runtime_root=tmp_path / "qualification_runtime",
        command_runner=command_runner,
        tool_downloader=tool_downloader,
    )

    assert report["decision"] == "fail"
    assert report["failure_reasons"] == [
        "dependency_lock_review_bundle_validation_failed"
    ]


@pytest.mark.quick
def test_host_launcher_rejects_unhashed_qualification_tool_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """host uv 输入缺少单 wheel SHA-256 时不得启动任何安装命令."""

    profile = _profile("workflow_orchestrator")
    monkeypatch.setattr(
        review_bundle,
        "get_dependency_profile",
        lambda requested_profile_id, path: _profile(requested_profile_id),
    )
    monkeypatch.setattr(
        review_bundle.repository_environment,
        "require_published_formal_execution_lock",
        lambda root: dict(FORMAL_EXECUTION_LOCK),
    )
    monkeypatch.setattr(review_bundle.platform, "system", lambda: "Linux")
    monkeypatch.setattr(review_bundle.platform, "machine", lambda: "x86_64")
    tool_lock_path = tmp_path / review_bundle.QUALIFICATION_TOOL_LOCK_RELATIVE_PATH
    tool_lock_path.parent.mkdir(parents=True, exist_ok=True)
    tool_lock_path.write_text("uv==0.11.28\n", encoding="utf-8")

    report, _ = review_bundle.launch_dependency_lock_qualification(
        profile.profile_name,
        repository_root=tmp_path,
        qualification_runtime_root=tmp_path / "qualification_runtime",
        command_runner=lambda *args: (_ for _ in ()).throw(
            AssertionError("工具锁失败后不得运行命令")
        ),
    )

    assert report["decision"] == "fail"
    assert report["failure_reasons"] == ["qualification_tool_lock_invalid"]


@pytest.mark.quick
def test_scientific_profile_rejects_before_download_when_orchestrator_lock_is_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """科学 profile 不得在父编排完整锁提交前下载或创建任何环境."""

    profile = _profile("sd35_method_runtime_gpu")
    monkeypatch.setattr(
        review_bundle,
        "get_dependency_profile",
        lambda requested_profile_id, path: _profile(requested_profile_id),
    )
    monkeypatch.setattr(
        review_bundle.repository_environment,
        "require_published_formal_execution_lock",
        lambda root: dict(FORMAL_EXECUTION_LOCK),
    )

    report, _ = review_bundle.launch_dependency_lock_qualification(
        profile.profile_name,
        repository_root=tmp_path,
        qualification_runtime_root=tmp_path / "qualification_runtime",
        command_runner=lambda *args: (_ for _ in ()).throw(
            AssertionError("父编排锁缺失后不得执行资格化命令")
        ),
        tool_downloader=lambda *args: (_ for _ in ()).throw(
            AssertionError("父编排锁缺失后不得下载资格化 wheel")
        ),
    )

    assert report["decision"] == "fail"
    assert report["failure_reasons"] == [
        "qualification_orchestrator_lock_unavailable"
    ]
    assert report["command_results"] == []


@pytest.mark.quick
def test_host_launcher_rejects_downloaded_uv_wheel_digest_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """URL 下载成功但 wheel 字节不匹配时不得执行 uv 或创建 Python."""

    profile = _profile("workflow_orchestrator")
    _bind_common_apis(monkeypatch, profile)
    monkeypatch.setattr(review_bundle.platform, "system", lambda: "Linux")
    monkeypatch.setattr(review_bundle.platform, "machine", lambda: "x86_64")
    tool_downloader = _write_qualification_tool_lock(tmp_path)

    def tampered_download(url: str, destination: Path) -> None:
        tool_downloader(url, destination)
        destination.write_bytes(destination.read_bytes() + b"tampered")

    report, _ = review_bundle.launch_dependency_lock_qualification(
        profile.profile_name,
        repository_root=tmp_path,
        qualification_runtime_root=tmp_path / "qualification_runtime",
        command_runner=lambda *args: (_ for _ in ()).throw(
            AssertionError("wheel 摘要不匹配后不得执行命令")
        ),
        tool_downloader=tampered_download,
    )

    assert report["decision"] == "fail"
    assert report["failure_reasons"] == [
        "qualification_tool_wheel_materialization_failed"
    ]
    assert report["command_results"] == []


@pytest.mark.quick
def test_default_qualification_runner_sanitizes_host_python_pip_and_uv_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """默认 runner 必须移除会改变解释器、索引和 uv 行为的宿主变量."""

    observed_environment: dict[str, str] = {}
    for key, value in {
        "CONDA_PREFIX": "/host/conda",
        "PIP_INDEX_URL": "https://mirror.example.test/simple",
        "PYTHONPATH": "/host/pythonpath",
        "UV_OFFLINE": "1",
        "VIRTUAL_ENV": "/host/venv",
    }.items():
        monkeypatch.setenv(key, value)

    def run(command: list[str], **kwargs: Any) -> Any:
        observed_environment.update(kwargs["env"])
        return review_bundle.subprocess.CompletedProcess(
            command,
            0,
            stdout="ok\n",
            stderr="",
        )

    monkeypatch.setattr(review_bundle.subprocess, "run", run)
    result = review_bundle._run_qualification_command(
        ["tool", "--version"],
        tmp_path,
        {"UV_PYTHON_INSTALL_DIR": "/qualified/python"},
    )

    assert result["return_code"] == 0
    assert observed_environment["UV_PYTHON_INSTALL_DIR"] == "/qualified/python"
    assert observed_environment["UV_NO_CONFIG"] == "1"
    assert observed_environment["PIP_CONFIG_FILE"] == review_bundle.os.devnull
    for key in (
        "CONDA_PREFIX",
        "PIP_INDEX_URL",
        "PYTHONPATH",
        "UV_OFFLINE",
        "VIRTUAL_ENV",
    ):
        assert key not in observed_environment


@pytest.mark.quick
def test_host_cli_requires_python_isolated_mode(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """普通 CLI 未使用 ``python -I`` 时必须在任何网络或环境操作前失败."""

    monkeypatch.delenv(
        review_bundle.QUALIFICATION_CHILD_ENVIRONMENT_KEY,
        raising=False,
    )
    assert review_bundle.sys.flags.isolated == 0

    exit_code = review_bundle.main(["--profile", "workflow_orchestrator"])

    assert exit_code == 2
    payload = json.loads(capsys.readouterr().err)
    assert payload["failure_reasons"] == ["qualification_host_python_not_isolated"]


@pytest.mark.quick
def test_qualification_child_requires_exact_python_path_digest_and_patch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """child 标记必须同时绑定 Python 路径、文件摘要和登记 patch."""

    python_executable = tmp_path / "qualification/bin/python"
    python_executable.parent.mkdir(parents=True, exist_ok=True)
    python_executable.write_bytes(b"exact child python")
    monkeypatch.setattr(review_bundle.sys, "executable", str(python_executable))
    monkeypatch.setattr(
        review_bundle,
        "get_dependency_profile",
        lambda profile_id, path: _profile("workflow_orchestrator"),
    )
    monkeypatch.setattr(
        review_bundle.platform,
        "python_implementation",
        lambda: "CPython",
    )
    monkeypatch.setattr(review_bundle.platform, "python_version", lambda: "3.12.13")
    monkeypatch.setattr(review_bundle.platform, "system", lambda: "Linux")
    monkeypatch.setattr(review_bundle.platform, "machine", lambda: "x86_64")
    monkeypatch.setenv(
        review_bundle.QUALIFICATION_PYTHON_ENVIRONMENT_KEY,
        str(python_executable),
    )
    monkeypatch.setenv(
        review_bundle.QUALIFICATION_PYTHON_DIGEST_ENVIRONMENT_KEY,
        _sha256(python_executable),
    )

    review_bundle._require_qualification_child_interpreter(tmp_path)

    monkeypatch.setenv(
        review_bundle.QUALIFICATION_PYTHON_DIGEST_ENVIRONMENT_KEY,
        "0" * 64,
    )
    with pytest.raises(RuntimeError, match="解释器门禁"):
        review_bundle._require_qualification_child_interpreter(tmp_path)


@pytest.mark.quick
def test_orchestrator_interpreter_writes_local_bundle_without_implicit_drive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """未给 Drive 参数时只应生成 outputs 审查包且不得创建 isolated Python."""

    profile = _profile("workflow_orchestrator")
    _bind_common_apis(monkeypatch, profile)

    def materialize(profile_id: str, *, repository_root: Path) -> tuple[dict[str, Any], Path]:
        assert profile_id == profile.profile_name
        return _write_candidate_artifacts(Path(repository_root), profile)

    monkeypatch.setattr(
        review_bundle.candidate_materializer,
        "materialize_dependency_lock_candidate",
        materialize,
    )
    monkeypatch.setattr(
        review_bundle,
        "prepare_dependency_profile",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("profile 解释器路径不得准备 orchestrator")
        ),
    )
    monkeypatch.setattr(
        review_bundle,
        "provision_isolated_dependency_python",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("profile 解释器路径不得创建 isolated Python")
        ),
    )

    manifest, manifest_path = review_bundle.write_dependency_lock_review_bundle(
        profile.profile_name,
        repository_root=tmp_path,
    )

    assert manifest["decision"] == review_bundle.SUCCESS_DECISION
    assert manifest["review_execution_mode"] == "orchestrator_interpreter"
    assert manifest["drive_bundle_dir"] is None
    assert manifest["drive_copy_performed"] is False
    assert manifest["supports_paper_claim"] is False
    assert len(manifest["files"]) == 3
    assert manifest_path == (
        tmp_path
        / "outputs/dependency_lock_review_bundles/workflow_orchestrator"
        / review_bundle.BUNDLE_MANIFEST_FILE_NAME
    ).resolve()
    for record in manifest["files"]:
        bundle_path = tmp_path / record["bundle_path"]
        assert bundle_path.is_file()
        assert record["sha256"] == _sha256(bundle_path)
        assert record["size_bytes"] == bundle_path.stat().st_size
    assert json.loads(manifest_path.read_text(encoding="utf-8")) == manifest


@pytest.mark.quick
@pytest.mark.parametrize(
    ("tampered_artifact", "expected_reason"),
    (
        ("candidate_lock", "candidate_lock_text_mismatch"),
        ("pip_resolver_report", "candidate_pip_report_revalidation_failed"),
        ("candidate_provenance", "candidate_lock_dependency_count_mismatch"),
    ),
)
def test_review_bundle_revalidates_candidate_artifact_closure_after_materialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tampered_artifact: str,
    expected_reason: str,
) -> None:
    """候选锁、pip 报告或 provenance 生成后被改写时必须失败闭合."""

    profile = _profile("workflow_orchestrator")
    _bind_common_apis(monkeypatch, profile)

    def materialize(
        profile_id: str,
        *,
        repository_root: Path,
    ) -> tuple[dict[str, Any], Path]:
        provenance, provenance_path = _write_candidate_artifacts(
            Path(repository_root),
            profile,
        )
        output_dir = provenance_path.parent
        if tampered_artifact == "candidate_lock":
            candidate_path = (
                output_dir
                / review_bundle.candidate_materializer.CANDIDATE_LOCK_FILE_NAME
            )
            candidate_path.write_text(
                candidate_path.read_text(encoding="utf-8") + "# tampered\n",
                encoding="utf-8",
            )
        elif tampered_artifact == "pip_resolver_report":
            report_path = (
                output_dir / review_bundle.candidate_materializer.PIP_REPORT_FILE_NAME
            )
            report = json.loads(report_path.read_text(encoding="utf-8"))
            report["install"][0]["metadata"]["version"] = "0.0.1"
            report_path.write_text(json.dumps(report), encoding="utf-8")
        else:
            provenance["candidate_lock_dependency_count"] += 1
            provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
        return provenance, provenance_path

    monkeypatch.setattr(
        review_bundle.candidate_materializer,
        "materialize_dependency_lock_candidate",
        materialize,
    )

    manifest, manifest_path = review_bundle.write_dependency_lock_review_bundle(
        profile.profile_name,
        repository_root=tmp_path,
    )

    assert manifest["decision"] == "fail"
    assert expected_reason in manifest["failure_reasons"]
    assert manifest["files"] == []
    assert json.loads(manifest_path.read_text(encoding="utf-8")) == manifest


@pytest.mark.quick
def test_explicit_drive_root_receives_profile_bundle_and_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """显式 Drive 根目录应收到同摘要的三个文件和 manifest."""

    profile = _profile("workflow_orchestrator")
    _bind_common_apis(monkeypatch, profile)
    monkeypatch.setattr(
        review_bundle.candidate_materializer,
        "materialize_dependency_lock_candidate",
        lambda profile_id, repository_root: _write_candidate_artifacts(
            Path(repository_root), profile
        ),
    )
    drive_root = tmp_path / "mounted_drive/dependency_lock_review_bundles"

    manifest, manifest_path = review_bundle.write_dependency_lock_review_bundle(
        profile.profile_name,
        repository_root=tmp_path,
        drive_output_dir=drive_root,
    )

    drive_bundle_dir = drive_root / profile.profile_name
    assert manifest["decision"] == review_bundle.SUCCESS_DECISION
    assert manifest["drive_copy_performed"] is True
    recorded_drive_dir = Path(str(manifest["drive_bundle_dir"]))
    if not recorded_drive_dir.is_absolute():
        recorded_drive_dir = tmp_path / recorded_drive_dir
    assert recorded_drive_dir.resolve() == drive_bundle_dir.resolve()
    for record in manifest["files"]:
        drive_path = Path(record["drive_path"])
        assert drive_path == (drive_bundle_dir / record["file_name"]).resolve()
        assert drive_path.is_file()
        assert _sha256(drive_path) == record["sha256"]
    drive_manifest = drive_bundle_dir / review_bundle.BUNDLE_MANIFEST_FILE_NAME
    assert drive_manifest.read_bytes() == manifest_path.read_bytes()
    validated_manifest, validated_manifest_path = (
        review_bundle._validate_written_review_bundle(
            profile,
            repository_root=tmp_path,
            formal_execution_lock=FORMAL_EXECUTION_LOCK,
            drive_output_dir=drive_root,
        )
    )
    assert validated_manifest == manifest
    assert validated_manifest_path == manifest_path

    first_drive_file = drive_bundle_dir / manifest["files"][0]["file_name"]
    first_drive_file.write_bytes(first_drive_file.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="摘要或大小"):
        review_bundle._validate_written_review_bundle(
            profile,
            repository_root=tmp_path,
            formal_execution_lock=FORMAL_EXECUTION_LOCK,
            drive_output_dir=drive_root,
        )


@pytest.mark.quick
def test_drive_profile_directory_rejects_unregistered_stale_entries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Drive 同 profile 目录存在协议外文件时不得生成可接收 manifest."""

    profile = _profile("workflow_orchestrator")
    _bind_common_apis(monkeypatch, profile)
    monkeypatch.setattr(
        review_bundle.candidate_materializer,
        "materialize_dependency_lock_candidate",
        lambda profile_id, repository_root: _write_candidate_artifacts(
            Path(repository_root), profile
        ),
    )
    drive_root = tmp_path / "mounted_drive/dependency_lock_review_bundles"
    drive_bundle_dir = drive_root / profile.profile_name
    drive_bundle_dir.mkdir(parents=True)
    stale_path = drive_bundle_dir / "unregistered.txt"
    stale_path.write_text("不得删除的协议外文件\n", encoding="utf-8")

    manifest, _ = review_bundle.write_dependency_lock_review_bundle(
        profile.profile_name,
        repository_root=tmp_path,
        drive_output_dir=drive_root,
    )

    assert manifest["decision"] == "fail"
    assert manifest["failure_reasons"] == ["drive_bundle_copy_failed"]
    assert stale_path.is_file()
    assert not (drive_bundle_dir / review_bundle.BUNDLE_MANIFEST_FILE_NAME).exists()


@pytest.mark.quick
@pytest.mark.parametrize(
    "profile_id",
    review_bundle.ISOLATED_PYTHON_PROFILE_IDS,
)
def test_isolated_python_profile_prepares_orchestrator_then_runs_child_materializer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    profile_id: str,
) -> None:
    """五个科学 profile 必须依次准备父环境、创建并运行子解释器."""

    profile = _profile(profile_id)
    _bind_common_apis(monkeypatch, profile)
    operations: list[str] = []

    def prepare_orchestrator(
        profile_id: str,
        *,
        repository_root: Path,
    ) -> tuple[dict[str, Any], Path]:
        operations.append("prepare_orchestrator")
        assert profile_id == "workflow_orchestrator"
        report_path = Path(repository_root) / "outputs/dependency_profiles/workflow_orchestrator/dependency_profile_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report = {"decision": "pass", "failure_reasons": []}
        report_path.write_text(json.dumps(report), encoding="utf-8")
        return report, report_path

    python_executable = tmp_path / "isolated_env/bin/python"

    def provision(
        profile_id: str,
        *,
        repository_root: Path,
    ) -> tuple[dict[str, Any], Path]:
        operations.append("provision_isolated_python")
        python_executable.parent.mkdir(parents=True, exist_ok=True)
        python_executable.write_bytes(b"python fixture")
        report_path = Path(repository_root) / f"outputs/dependency_profiles/{profile_id}/isolated_python_provision_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "decision": "provisioned",
            "provisioned": True,
            "failure_reasons": [],
            "python_executable_path": str(python_executable),
            "python_executable_sha256": _sha256(python_executable),
        }
        report_path.write_text(json.dumps(report), encoding="utf-8")
        return report, report_path

    def run_child(command: list[str], working_directory: Path) -> dict[str, Any]:
        operations.append("run_child_materializer")
        assert list(command) == [
            str(python_executable),
            "-I",
            str(tmp_path.resolve() / "scripts/materialize_dependency_lock_candidate.py"),
            "--profile",
            profile.profile_name,
        ]
        _write_candidate_artifacts(tmp_path, profile)
        return {"return_code": 0, "stdout": "候选已生成.\n", "stderr": ""}

    monkeypatch.setattr(review_bundle, "prepare_dependency_profile", prepare_orchestrator)
    monkeypatch.setattr(review_bundle, "provision_isolated_dependency_python", provision)
    monkeypatch.setattr(
        review_bundle.candidate_materializer,
        "materialize_dependency_lock_candidate",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("isolated Python profile 不得由父解释器直接物化")
        ),
    )

    manifest, _ = review_bundle.write_dependency_lock_review_bundle(
        profile.profile_name,
        repository_root=tmp_path,
        child_command_runner=run_child,
    )

    assert operations == [
        "prepare_orchestrator",
        "provision_isolated_python",
        "run_child_materializer",
    ]
    assert manifest["decision"] == review_bundle.SUCCESS_DECISION
    assert manifest["review_execution_mode"] == "isolated_python"
    assert manifest["orchestrator_preparation"]["decision"] == "pass"
    assert manifest["isolated_python_provision"]["decision"] == "provisioned"
    assert manifest["candidate_materialization"]["return_code"] == 0
    assert manifest["candidate_materialization"]["python_executable"] == str(
        python_executable
    )


@pytest.mark.quick
def test_isolated_python_profile_fails_before_provision_when_orchestrator_lock_is_absent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """orchestrator 完整锁未提交时应保留诊断并停止创建 isolated Python."""

    profile = _profile("gaussian_shading_official_py38_cu117")
    _bind_common_apis(monkeypatch, profile)
    report_path = (
        tmp_path
        / "outputs/dependency_profiles/workflow_orchestrator/dependency_profile_report.json"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(
            {"decision": "fail", "failure_reasons": ["complete_hash_lock_missing"]}
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        review_bundle,
        "prepare_dependency_profile",
        lambda profile_id, repository_root: (
            {"decision": "fail", "failure_reasons": ["complete_hash_lock_missing"]},
            report_path,
        ),
    )
    monkeypatch.setattr(
        review_bundle,
        "provision_isolated_dependency_python",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("orchestrator 未通过时不得创建 isolated Python")
        ),
    )

    manifest, manifest_path = review_bundle.write_dependency_lock_review_bundle(
        profile.profile_name,
        repository_root=tmp_path,
        child_command_runner=lambda *args: (_ for _ in ()).throw(
            AssertionError("orchestrator 未通过时不得执行子解释器")
        ),
    )

    assert manifest["decision"] == "fail"
    assert manifest["failure_reasons"] == [
        "isolated_python_candidate_materialization_failed"
    ]
    assert "workflow_orchestrator" in manifest["diagnostic_message"]
    assert manifest["orchestrator_preparation"]["failure_reasons"] == [
        "complete_hash_lock_missing"
    ]
    assert json.loads(manifest_path.read_text(encoding="utf-8")) == manifest


@pytest.mark.quick
def test_review_bundle_rejects_unpublished_code_identity_before_materialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """未发布精确代码身份锁时应持久化失败且不运行候选物化器."""

    profile = _profile("t2smark_sd35_gpu")
    monkeypatch.setattr(
        review_bundle,
        "get_dependency_profile",
        lambda profile_id, path: profile,
    )
    monkeypatch.setattr(
        review_bundle.repository_environment,
        "require_published_formal_execution_lock",
        lambda root: (_ for _ in ()).throw(
            review_bundle.repository_environment.FormalExecutionLockError(
                "没有已发布执行锁."
            )
        ),
    )
    monkeypatch.setattr(
        review_bundle.candidate_materializer,
        "materialize_dependency_lock_candidate",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("代码身份未通过时不得物化候选")
        ),
    )

    manifest, manifest_path = review_bundle.write_dependency_lock_review_bundle(
        profile.profile_name,
        repository_root=tmp_path,
    )

    assert manifest["decision"] == "fail"
    assert manifest["failure_reasons"] == ["formal_execution_lock_unavailable"]
    assert manifest["formal_execution_lock"] == {}
    assert manifest["diagnostic_message"] == "没有已发布执行锁."
    assert json.loads(manifest_path.read_text(encoding="utf-8")) == manifest
