"""验证审查包回传后只生成可提交锁, 不自动形成论文证据."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
from typing import Any
from urllib.parse import quote

import pytest

from experiments.runtime.dependency_profiles import get_dependency_profile
from experiments.runtime import repository_environment
from main.core.digest import build_stable_digest
from scripts import materialize_dependency_lock_candidate as materialization
from scripts import write_dependency_lock_review_bundle as review_bundle
from scripts import write_reviewed_dependency_hash_lock as lock_writer


COMMIT = "a" * 40


def _formal_execution_lock() -> dict[str, Any]:
    """构造通过共享验证器的正式候选生成代码锁."""

    payload = {
        "formal_execution_lock_schema": (
            repository_environment.FORMAL_EXECUTION_LOCK_SCHEMA
        ),
        "formal_execution_commit": COMMIT,
        "formal_execution_head_detached": True,
        "formal_execution_worktree_clean": True,
        "formal_execution_lock_ready": True,
    }
    return {
        **payload,
        "formal_execution_lock_digest": build_stable_digest(payload),
    }


def _sha256(path: Path) -> str:
    """计算 fixture 文件实际 SHA-256."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _copy_dependency_configs(repository_root: Path) -> None:
    """复制真实 registry 与直接输入, 保持目标锁缺失状态."""

    source = Path("configs").resolve()
    destination = repository_root / "configs"
    shutil.copytree(source, destination)
    for lock_path in (destination / "dependency_profiles").glob("*_lock.txt"):
        qualification_tool_name = (
            lock_writer.review_bundle.QUALIFICATION_TOOL_LOCK_RELATIVE_PATH.name
        )
        if lock_path.name == qualification_tool_name:
            continue
        lock_path.unlink()


def _wheel_item(package_name: str, version: str) -> dict[str, Any]:
    """构造可被真实候选解析器复验的 wheel report 条目."""

    normalized = package_name.lower().replace("_", "-").replace(".", "-")
    wheel_name = normalized.replace("-", "_")
    wheel_version = quote(version, safe=".!_")
    digest = hashlib.sha256(
        f"accepted-wheel:{normalized}=={version}".encode("utf-8")
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


def _write_valid_review_bundle(
    repository_root: Path,
    profile_id: str,
) -> Path:
    """根据真实 profile 直接输入生成语义闭合的轻量审查包."""

    profile = get_dependency_profile(
        profile_id,
        repository_root / "configs/dependency_profile_registry.json",
    )
    formal_lock = _formal_execution_lock()
    bundle_dir = repository_root / "review_bundle" / profile.profile_name
    bundle_dir.mkdir(parents=True, exist_ok=True)
    pip_report_path = bundle_dir / materialization.PIP_REPORT_FILE_NAME
    candidate_path = bundle_dir / materialization.CANDIDATE_LOCK_FILE_NAME
    provenance_path = bundle_dir / materialization.PROVENANCE_FILE_NAME
    pip_version = "24.3.1"
    install = []
    for specification in profile.direct_requirements:
        dependency = materialization.parse_exact_requirement_spec(specification)
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
    wheels, _ = materialization.load_resolved_wheels(
        pip_report_path,
        profile,
        expected_pip_version=pip_version,
    )
    candidate_path.write_bytes(
        materialization.candidate_lock_text(wheels).encode("utf-8"),
    )
    provenance = {
        "report_schema": materialization.PROVENANCE_SCHEMA,
        "schema_version": materialization.PROVENANCE_SCHEMA_VERSION,
        "profile_id": profile.profile_name,
        "profile_digest": profile.profile_digest,
        "cuda_version": profile.cuda_version,
        "pytorch_index_url": profile.pytorch_index_url,
        "torch_version": profile.torch_version,
        "torchvision_version": profile.torchvision_version,
        "direct_requirements_path": profile.direct_requirements_path,
        "direct_requirements_digest": profile.direct_requirements_digest,
        "formal_execution_lock": formal_lock,
        "formal_execution_commit": formal_lock["formal_execution_commit"],
        "formal_execution_lock_digest": formal_lock[
            "formal_execution_lock_digest"
        ],
        "decision": "candidate_ready_for_review",
        "failure_reasons": [],
        "supports_paper_claim": False,
        "resolver_return_code": 0,
        "pip_version": pip_version,
        "pip_resolver_report_path": "outputs/source/pip_resolver_report.json",
        "candidate_lock_path": "outputs/source/dependency_lock_candidate.txt",
        "candidate_hash_source": (
            "pip_install_report.download_info.archive_info.hashes.sha256"
        ),
        "candidate_lock_dependency_count": len(wheels),
        "candidate_lock_logical_digest": (
            materialization.candidate_lock_logical_digest(wheels)
        ),
    }
    provenance_path.write_text(
        json.dumps(provenance, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    roles = {
        "candidate_lock": candidate_path,
        "pip_resolver_report": pip_report_path,
        "candidate_provenance": provenance_path,
    }
    files = [
        {
            "artifact_role": role,
            "file_name": path.name,
            "source_path": f"outputs/source/{path.name}",
            "bundle_path": str(path),
            "sha256": _sha256(path),
            "size_bytes": path.stat().st_size,
        }
        for role, path in roles.items()
    ]
    manifest = review_bundle._build_manifest(
        profile,
        repository_root=repository_root,
        local_bundle_dir=bundle_dir,
        drive_bundle_dir=None,
        formal_execution_lock=formal_lock,
    )
    manifest["files"] = files
    manifest["decision"] = review_bundle.SUCCESS_DECISION
    manifest["failure_reasons"] = []
    manifest["diagnostic_message"] = None
    (bundle_dir / review_bundle.BUNDLE_MANIFEST_FILE_NAME).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return bundle_dir


def _clean_git_runner(
    command: list[str],
    repository_root: Path,
) -> dict[str, Any]:
    """模拟与候选提交一致的 attached 或 detached clean 工作树."""

    if command[1] == "rev-parse":
        return {"return_code": 0, "stdout": COMMIT + "\n", "stderr": ""}
    assert command[1] == "status"
    return {"return_code": 0, "stdout": "", "stderr": ""}


@pytest.mark.quick
@pytest.mark.parametrize(
    "profile_id",
    ("workflow_orchestrator", "sd35_method_runtime_gpu"),
)
def test_reviewed_bundle_writes_registry_lock_for_explicit_git_commit(
    tmp_path: Path,
    profile_id: str,
) -> None:
    """CPU 与 CUDA profile 审查包都只写入经过重建复验的规范锁."""

    _copy_dependency_configs(tmp_path)
    bundle_dir = _write_valid_review_bundle(tmp_path, profile_id)
    profile_before = get_dependency_profile(
        profile_id,
        tmp_path / "configs/dependency_profile_registry.json",
    )
    assert profile_before.formal_ready is False

    report, report_path = lock_writer.write_reviewed_dependency_hash_lock(
        profile_id,
        bundle_dir,
        profile_id,
        repository_root=tmp_path,
        git_command_runner=_clean_git_runner,
    )

    lock_path = tmp_path / profile_before.complete_hash_lock_path
    profile_after = get_dependency_profile(
        profile_id,
        tmp_path / "configs/dependency_profile_registry.json",
    )
    assert report["decision"] == lock_writer.SUCCESS_DECISION
    assert report["supports_paper_claim"] is False
    assert lock_path.is_file()
    assert profile_after.formal_ready is True
    assert profile_after.complete_hash_lock_digest == report[
        "complete_hash_lock_digest"
    ]
    assert profile_after.complete_hash_lock_dependency_count == report[
        "complete_hash_lock_dependency_count"
    ]
    assert json.loads(report_path.read_text(encoding="utf-8")) == report


@pytest.mark.quick
def test_reviewed_bundle_rejects_tampered_candidate_without_writing_lock(
    tmp_path: Path,
) -> None:
    """回传候选文件被改写时不得写入 registry 锁目标."""

    _copy_dependency_configs(tmp_path)
    profile_id = "workflow_orchestrator"
    bundle_dir = _write_valid_review_bundle(tmp_path, profile_id)
    candidate_path = bundle_dir / materialization.CANDIDATE_LOCK_FILE_NAME
    candidate_path.write_text(
        candidate_path.read_text(encoding="utf-8") + "# tampered\n",
        encoding="utf-8",
    )
    profile = get_dependency_profile(
        profile_id,
        tmp_path / "configs/dependency_profile_registry.json",
    )

    report, _ = lock_writer.write_reviewed_dependency_hash_lock(
        profile_id,
        bundle_dir,
        profile_id,
        repository_root=tmp_path,
        git_command_runner=_clean_git_runner,
    )

    assert report["decision"] == "fail"
    assert report["failure_reasons"] == ["review_bundle_validation_failed"]
    assert not (tmp_path / profile.complete_hash_lock_path).exists()


@pytest.mark.quick
def test_reviewed_bundle_rejects_head_different_from_candidate_commit(
    tmp_path: Path,
) -> None:
    """当前 HEAD 已漂移时不得把旧审查包写入新代码状态."""

    _copy_dependency_configs(tmp_path)
    profile_id = "workflow_orchestrator"
    bundle_dir = _write_valid_review_bundle(tmp_path, profile_id)

    def mismatched_git_runner(
        command: list[str],
        repository_root: Path,
    ) -> dict[str, Any]:
        if command[1] == "rev-parse":
            return {"return_code": 0, "stdout": "c" * 40 + "\n", "stderr": ""}
        return {"return_code": 0, "stdout": "", "stderr": ""}

    report, _ = lock_writer.write_reviewed_dependency_hash_lock(
        profile_id,
        bundle_dir,
        profile_id,
        repository_root=tmp_path,
        git_command_runner=mismatched_git_runner,
    )

    assert report["decision"] == "fail"
    assert report["failure_reasons"] == ["review_bundle_validation_failed"]
