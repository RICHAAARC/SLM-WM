"""验证依赖锁资格化入口的解释器路由与审查包复制."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
from urllib.parse import quote

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


def _profile(profile_id: str) -> DependencyProfile:
    """读取共享 registry 中的真实 profile 身份."""

    return get_dependency_profile(profile_id)


def _sha256(path: Path) -> str:
    """独立计算测试断言使用的文件 SHA-256."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


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
