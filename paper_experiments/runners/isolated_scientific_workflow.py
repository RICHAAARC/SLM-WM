"""在隔离科学解释器中运行完整外部 baseline workflow.

该模块位于完整论文实验层, 同时提供父进程调度和科学子进程入口.父进程只
负责选择依赖 profile,核验证据并写出不可变绑定; 子进程才导入并调用现有
科学 runner.这样 Notebook 当前解释器只需承担 CPU 编排, 科学实现仍可脱离
Notebook 在 GPU 服务器上复用.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any, Dict, Mapping, Sequence, Tuple, Union

from experiments.protocol.paper_run_config import build_paper_run_config
from experiments.runtime.isolated_scientific_execution import (
    DEPENDENCY_ENVIRONMENT_REPORT_DIGEST_ENVIRONMENT_KEY,
    DEPENDENCY_ENVIRONMENT_REPORT_PATH_ENVIRONMENT_KEY,
    REPORT_SCHEMA as EXECUTION_REPORT_SCHEMA,
    REPORT_SCHEMA_VERSION as EXECUTION_REPORT_SCHEMA_VERSION,
    execute_isolated_scientific_command,
)
from experiments.runtime.repository_environment import (
    FORMAL_EXECUTION_COMMIT_ENVIRONMENT_KEY,
    FORMAL_EXECUTION_LOCK_DIGEST_ENVIRONMENT_KEY,
    require_published_formal_execution_lock,
)
from experiments.runtime.scientific_execution_binding import (
    BINDING_FILE_NAME,
    BOUND_MANIFEST_DIGEST_SCOPE,
    DEPENDENCY_REPORT_FILE_NAME,
    DISPATCH_REPORT_FILE_NAME,
    EXECUTION_REPORT_FILE_NAME,
    file_sha256 as _file_sha256,
    scientific_manifest_payload_digest,
    validate_scientific_execution_binding,
    write_scientific_execution_binding,
)


PathLike = Union[str, Path]
RESULT_ENVELOPE_SCHEMA = "isolated_scientific_workflow_result"
RESULT_ENVELOPE_SCHEMA_VERSION = 1
SCIENTIFIC_DIRECTORY_NAME = "scientific_execution"
RESULT_ENVELOPE_FILE_NAME = "scientific_workflow_result_envelope.json"
SOURCE_EXECUTION_REPORT_FILE_NAME = "source_isolated_scientific_execution_report.json"

WORKFLOW_PROFILE_IDS = {
    "external_baseline_method_faithful": "sd35_method_runtime_gpu",
    "official_reference_t2smark": "t2smark_sd35_gpu",
}
WORKFLOW_ARTIFACT_ROLES = {
    "external_baseline_method_faithful": "external_baseline_method_faithful",
    "official_reference_t2smark": "t2smark_formal_reproduction",
}
METHOD_FAITHFUL_BASELINE_IDS = (
    "tree_ring",
    "gaussian_shading",
    "shallow_diffuse",
)


@dataclass(frozen=True)
class IsolatedScientificArchiveRecord:
    """保存由受验证科学解释器生成并复验的归档记录."""

    archive_path: str
    archive_digest: str
    archive_entry_count: int
    drive_archive_path: str
    drive_archive_digest: str
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """转换为服务器和 Notebook 共同使用的 JSON object."""

        return asdict(self)


def _is_sha256(value: Any) -> bool:
    """判断字段是否是规范的小写 SHA-256 文本."""

    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """使用稳定排版写出受治理 JSON object."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _read_json_object(path: Path, label: str) -> Dict[str, Any]:
    """读取并验证一个必需的 JSON object."""

    if not path.is_file():
        raise FileNotFoundError("{0} 不存在: {1}".format(label, path))
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise TypeError("{0} 必须是 JSON object".format(label))
    return payload


def _relative_path(path: Path, root: Path) -> str:
    """把 workflow 产物路径记录为仓库相对路径."""

    return path.resolve().relative_to(root.resolve()).as_posix()


def _require_within(path: Path, parent: Path, label: str) -> Path:
    """限制运行证据位于当前 workflow 输出范围内."""

    resolved = path.resolve()
    try:
        resolved.relative_to(parent.resolve())
    except ValueError as exc:
        raise ValueError("{0} 必须位于当前 workflow 输出范围内".format(label)) from exc
    return resolved


def _resolved_baseline_id() -> str:
    """读取 method-faithful 单 baseline 身份并拒绝集合外值."""

    baseline_id = os.environ.get("SLM_WM_PRIMARY_BASELINE_ID", "").strip()
    if baseline_id not in METHOD_FAITHFUL_BASELINE_IDS:
        raise ValueError("method-faithful workflow 必须指定唯一受支持 baseline_id")
    return baseline_id


def scientific_artifact_paths(
    root: PathLike,
    workflow_name: str,
) -> Dict[str, Path]:
    """构造科学子进程产物与父进程证据的唯一受治理路径."""

    root_path = Path(root).resolve()
    paper_run = build_paper_run_config(root_path)
    if workflow_name == "external_baseline_method_faithful":
        baseline_id = _resolved_baseline_id()
        output_scope = (
            root_path
            / "outputs"
            / "external_baseline_method_faithful"
            / paper_run.run_name
            / "run_records"
            / baseline_id
        )
        summary_path = output_scope / "{0}_summary.json".format(baseline_id)
        manifest_path = output_scope / "{0}_manifest.local.json".format(baseline_id)
    elif workflow_name == "official_reference_t2smark":
        output_scope = (
            root_path
            / "outputs"
            / "t2smark_formal_reproduction"
            / paper_run.run_name
        )
        summary_path = output_scope / "t2smark_formal_reproduction_summary.json"
        manifest_path = output_scope / "t2smark_formal_reproduction_manifest.local.json"
    else:
        raise ValueError("isolated_scientific_workflow_not_supported:{0}".format(workflow_name))

    output_scope = _require_within(
        output_scope,
        root_path / "outputs",
        "workflow 输出目录",
    )
    scientific_dir = output_scope / SCIENTIFIC_DIRECTORY_NAME
    return {
        "root": root_path,
        "output_scope": output_scope,
        "scientific_dir": scientific_dir,
        "summary": summary_path,
        "manifest": manifest_path,
        "result_envelope": scientific_dir / RESULT_ENVELOPE_FILE_NAME,
        "source_execution_report": scientific_dir / SOURCE_EXECUTION_REPORT_FILE_NAME,
        "execution_report": output_scope / EXECUTION_REPORT_FILE_NAME,
        "dependency_report": output_scope / DEPENDENCY_REPORT_FILE_NAME,
        "dispatch_report": output_scope / DISPATCH_REPORT_FILE_NAME,
        "execution_binding": output_scope / BINDING_FILE_NAME,
    }


def _run_child_scientific_workflow(root: Path, workflow_name: str) -> Mapping[str, Any]:
    """在隔离解释器内延迟导入并调用现有完整科学 runner."""

    if workflow_name == "external_baseline_method_faithful":
        from paper_experiments.runners.external_baseline_method_faithful import (
            run_default_external_baseline_method_faithful_plan,
        )

        return run_default_external_baseline_method_faithful_plan(root=root)
    if workflow_name == "official_reference_t2smark":
        from paper_experiments.runners.t2smark_formal_reproduction import (
            run_default_t2smark_formal_reproduction_plan,
        )

        return run_default_t2smark_formal_reproduction_plan(root=root)
    raise ValueError("isolated_scientific_workflow_not_supported:{0}".format(workflow_name))


def run_scientific_child(
    *,
    root: PathLike,
    workflow_name: str,
    result_envelope_path: PathLike,
) -> int:
    """运行科学 workflow 并把唯一跨进程结果 envelope 写入 outputs."""

    paths = scientific_artifact_paths(root, workflow_name)
    expected_envelope_path = paths["result_envelope"].resolve()
    supplied_envelope_path = Path(result_envelope_path).expanduser().resolve()
    if supplied_envelope_path != expected_envelope_path:
        raise ValueError("科学子进程 result envelope 路径与当前 workflow 不一致")

    summary = _run_child_scientific_workflow(paths["root"], workflow_name)
    if not isinstance(summary, Mapping):
        raise TypeError("科学 runner 必须返回 summary mapping")
    persisted_summary = _read_json_object(paths["summary"], "科学 runner summary")
    if dict(summary) != persisted_summary:
        raise RuntimeError("科学 runner 返回 summary 与持久化内容不一致")
    manifest = _read_json_object(paths["manifest"], "科学 runner manifest")
    formal_execution_lock = require_published_formal_execution_lock(paths["root"])

    source_dependency_path_value = os.environ.get(
        DEPENDENCY_ENVIRONMENT_REPORT_PATH_ENVIRONMENT_KEY,
        "",
    )
    source_dependency_digest = os.environ.get(
        DEPENDENCY_ENVIRONMENT_REPORT_DIGEST_ENVIRONMENT_KEY,
        "",
    )
    source_dependency_path = Path(source_dependency_path_value).resolve()
    if (
        not source_dependency_path_value
        or not source_dependency_path.is_file()
        or not _is_sha256(source_dependency_digest)
        or _file_sha256(source_dependency_path) != source_dependency_digest
    ):
        raise RuntimeError("科学子进程缺少经过验证的隔离依赖环境报告")
    if manifest.get("formal_execution_run_lock") != formal_execution_lock:
        raise RuntimeError("科学 runner manifest 未绑定当前正式执行锁")

    workflow_ready = (
        persisted_summary.get("external_baseline_method_faithful_ready") is True
        if workflow_name == "external_baseline_method_faithful"
        else persisted_summary.get("t2smark_formal_reproduction_ready") is True
    )
    child_passed = persisted_summary.get("run_decision") == "pass" and workflow_ready
    profile_id = WORKFLOW_PROFILE_IDS[workflow_name]
    paper_run_name = build_paper_run_config(paths["root"]).run_name
    envelope = {
        "report_schema": "scientific_command_dispatch_report",
        "result_schema": RESULT_ENVELOPE_SCHEMA,
        "schema_version": RESULT_ENVELOPE_SCHEMA_VERSION,
        "operation_kind": "isolated_scientific_workflow",
        "workflow_name": workflow_name,
        "paper_run_name": paper_run_name,
        "profile_id": profile_id,
        "decision": "pass" if child_passed else "fail",
        "child_decision": "pass" if child_passed else "fail",
        "summary": persisted_summary,
        "summary_path": _relative_path(paths["summary"], paths["root"]),
        "summary_sha256": _file_sha256(paths["summary"]),
        "manifest_path": _relative_path(paths["manifest"], paths["root"]),
        "manifest_sha256": _file_sha256(paths["manifest"]),
        "dependency_environment_report_digest": source_dependency_digest,
        "formal_execution_lock": formal_execution_lock,
        "formal_execution_commit": formal_execution_lock["formal_execution_commit"],
        "formal_execution_lock_digest": formal_execution_lock[
            "formal_execution_lock_digest"
        ],
        "supports_paper_claim": False,
    }
    _write_json(expected_envelope_path, envelope)
    return 0 if child_passed else 2


def _child_argv(paths: Mapping[str, Path], workflow_name: str) -> Tuple[str, ...]:
    """构造隔离 Python 要执行的模块参数, 不包含解释器路径."""

    return (
        "-m",
        "paper_experiments.runners.isolated_scientific_workflow",
        "--child-workflow",
        workflow_name,
        "--root",
        str(paths["root"]),
        "--result-envelope",
        str(paths["result_envelope"]),
    )


def _validate_source_execution_report(
    report: Mapping[str, Any],
    persisted_report: Mapping[str, Any],
    *,
    paths: Mapping[str, Path],
    workflow_name: str,
    child_argv: Sequence[str],
) -> Path:
    """严格验证 runtime API 刚写出的原始执行报告."""

    if dict(report) != dict(persisted_report):
        raise RuntimeError("隔离科学执行报告的返回值与持久化内容不一致")
    profile_id = WORKFLOW_PROFILE_IDS[workflow_name]
    expected_values = {
        "report_schema": EXECUTION_REPORT_SCHEMA,
        "schema_version": EXECUTION_REPORT_SCHEMA_VERSION,
        "operation_kind": "isolated_scientific_execution",
        "profile_id": profile_id,
        "dependency_environment_report_valid": True,
        "formal_execution_lock_ready": True,
        "python_executable_revalidated_before_child": True,
        "python_executable_revalidated_after_child": True,
        "dependency_environment_report_revalidated_before_child": True,
        "dependency_environment_report_revalidated_after_child": True,
        "formal_execution_lock_revalidated_before_child": True,
        "formal_execution_lock_revalidated_after_child": True,
        "child_argv_tail": list(child_argv),
        "execution_completed": True,
        "decision": "pass",
        "failure_reasons": [],
        "supports_paper_claim": False,
    }
    for field_name, expected_value in expected_values.items():
        if report.get(field_name) != expected_value:
            raise RuntimeError(
                "隔离科学执行报告字段不一致: {0}".format(field_name)
            )
    for field_name in (
        "profile_digest",
        "direct_requirements_digest",
        "complete_hash_lock_digest",
        "dependency_environment_report_digest",
        "python_executable_sha256",
        "formal_execution_lock_digest",
    ):
        if not _is_sha256(report.get(field_name)):
            raise RuntimeError("隔离科学执行报告摘要无效: {0}".format(field_name))
    if int(report.get("complete_hash_lock_dependency_count", 0)) <= 0:
        raise RuntimeError("隔离科学执行报告未绑定完整依赖锁包集合")
    if Path(str(report.get("execution_report_path", ""))).resolve() != paths[
        "source_execution_report"
    ].resolve():
        raise RuntimeError("隔离科学执行报告未绑定唯一 workflow 路径")
    execution = report.get("execution")
    if not isinstance(execution, Mapping):
        raise TypeError("隔离科学执行报告缺少 execution object")
    execution_argv = list(execution.get("argv", ()))
    if (
        execution.get("attempted") is not True
        or execution.get("return_code") != 0
        or not execution_argv
        or Path(str(execution_argv[0])).resolve()
        != Path(str(report.get("python_executable_path", ""))).resolve()
        or execution_argv[1:] != list(child_argv)
        or Path(str(execution.get("working_directory", ""))).resolve()
        != paths["root"].resolve()
    ):
        raise RuntimeError("隔离科学子命令执行证据不完整")
    formal_execution_lock = report.get("formal_execution_lock")
    if not isinstance(formal_execution_lock, Mapping):
        raise TypeError("隔离科学执行报告缺少正式执行锁")
    if (
        formal_execution_lock.get("formal_execution_commit")
        != report.get("formal_execution_commit")
        or formal_execution_lock.get("formal_execution_lock_digest")
        != report.get("formal_execution_lock_digest")
    ):
        raise RuntimeError("隔离科学执行报告的正式执行锁字段不一致")

    source_dependency_path = Path(
        str(report.get("dependency_environment_report_path", ""))
    ).resolve()
    if (
        not source_dependency_path.is_file()
        or _file_sha256(source_dependency_path)
        != report["dependency_environment_report_digest"]
    ):
        raise RuntimeError("隔离科学执行报告引用的依赖环境报告无效")
    expected_environment_overrides = {
        FORMAL_EXECUTION_COMMIT_ENVIRONMENT_KEY: report["formal_execution_commit"],
        FORMAL_EXECUTION_LOCK_DIGEST_ENVIRONMENT_KEY: report[
            "formal_execution_lock_digest"
        ],
        DEPENDENCY_ENVIRONMENT_REPORT_PATH_ENVIRONMENT_KEY: str(
            source_dependency_path
        ),
        DEPENDENCY_ENVIRONMENT_REPORT_DIGEST_ENVIRONMENT_KEY: report[
            "dependency_environment_report_digest"
        ],
    }
    if execution.get("environment_overrides") != expected_environment_overrides:
        raise RuntimeError("隔离科学子命令环境身份与执行报告不一致")
    return source_dependency_path


def _validate_result_envelope(
    envelope: Mapping[str, Any],
    *,
    report: Mapping[str, Any],
    paths: Mapping[str, Path],
    workflow_name: str,
) -> Dict[str, Any]:
    """核验子进程 envelope,原 summary,manifest 与执行身份."""

    expected_values = {
        "report_schema": "scientific_command_dispatch_report",
        "result_schema": RESULT_ENVELOPE_SCHEMA,
        "schema_version": RESULT_ENVELOPE_SCHEMA_VERSION,
        "operation_kind": "isolated_scientific_workflow",
        "workflow_name": workflow_name,
        "paper_run_name": build_paper_run_config(paths["root"]).run_name,
        "profile_id": WORKFLOW_PROFILE_IDS[workflow_name],
        "decision": "pass",
        "child_decision": "pass",
        "summary_path": _relative_path(paths["summary"], paths["root"]),
        "manifest_path": _relative_path(paths["manifest"], paths["root"]),
        "dependency_environment_report_digest": report[
            "dependency_environment_report_digest"
        ],
        "formal_execution_lock": report["formal_execution_lock"],
        "formal_execution_commit": report["formal_execution_commit"],
        "formal_execution_lock_digest": report["formal_execution_lock_digest"],
        "supports_paper_claim": False,
    }
    for field_name, expected_value in expected_values.items():
        if envelope.get(field_name) != expected_value:
            raise RuntimeError("科学子进程 result envelope 字段不一致: {0}".format(field_name))
    persisted_summary = _read_json_object(paths["summary"], "科学 runner summary")
    persisted_manifest = _read_json_object(paths["manifest"], "科学 runner manifest")
    if envelope.get("summary") != persisted_summary:
        raise RuntimeError("科学子进程 result envelope 未绑定原 summary")
    if (
        envelope.get("summary_sha256") != _file_sha256(paths["summary"])
        or envelope.get("manifest_sha256") != _file_sha256(paths["manifest"])
    ):
        raise RuntimeError("科学子进程 result envelope 的产物摘要不一致")
    if persisted_manifest.get("formal_execution_run_lock") != report[
        "formal_execution_lock"
    ]:
        raise RuntimeError("科学 runner manifest 与隔离执行锁不一致")
    return persisted_summary


def _verify_shared_binding_before_return(
    binding: Mapping[str, Any],
    *,
    paths: Mapping[str, Path],
    source_report: Mapping[str, Any],
    workflow_name: str,
) -> None:
    """在父进程返回前复核共享 binding 与本 workflow 的完整身份."""

    persisted_binding = _read_json_object(paths["execution_binding"], "科学执行绑定")
    if dict(binding) != persisted_binding:
        raise RuntimeError("科学执行绑定的内存值与持久化内容不一致")
    expected_values = {
        "profile_id": WORKFLOW_PROFILE_IDS[workflow_name],
        "profile_digest": source_report["profile_digest"],
        "direct_requirements_digest": source_report["direct_requirements_digest"],
        "complete_hash_lock_digest": source_report["complete_hash_lock_digest"],
        "formal_execution_lock": source_report["formal_execution_lock"],
        "formal_execution_commit": source_report["formal_execution_commit"],
        "formal_execution_lock_digest": source_report["formal_execution_lock_digest"],
        "scientific_execution_report_path": _relative_path(
            paths["execution_report"], paths["root"]
        ),
        "scientific_execution_report_digest": _file_sha256(paths["execution_report"]),
        "dependency_environment_report_path": _relative_path(
            paths["dependency_report"], paths["root"]
        ),
        "dependency_environment_report_digest": _file_sha256(paths["dependency_report"]),
        "scientific_command_dispatch_report_path": _relative_path(
            paths["dispatch_report"], paths["root"]
        ),
        "scientific_command_dispatch_report_digest": _file_sha256(paths["dispatch_report"]),
        "bound_summary_path": _relative_path(paths["summary"], paths["root"]),
        "bound_summary_digest": _file_sha256(paths["summary"]),
        "bound_manifest_path": _relative_path(paths["manifest"], paths["root"]),
        "bound_manifest_scientific_digest": scientific_manifest_payload_digest(
            _read_json_object(paths["manifest"], "科学 runner manifest")
        ),
        "bound_manifest_digest_scope": BOUND_MANIFEST_DIGEST_SCOPE,
        "decision": "pass",
        "supports_paper_claim": False,
    }
    for field_name, expected_value in expected_values.items():
        if persisted_binding.get(field_name) != expected_value:
            raise RuntimeError("共享科学执行绑定字段不一致: {0}".format(field_name))
    if _read_json_object(paths["dispatch_report"], "科学命令调度报告") != _read_json_object(
        paths["result_envelope"],
        "科学子进程 result envelope",
    ):
        raise RuntimeError("共享科学执行绑定未保存原 result envelope 快照")


def run_isolated_scientific_workflow(
    *,
    root: PathLike = ".",
    workflow_name: str,
) -> Dict[str, Any]:
    """在受治理科学子解释器中运行完整 workflow 并返回原 summary."""

    if workflow_name not in WORKFLOW_PROFILE_IDS:
        raise ValueError("isolated_scientific_workflow_not_supported:{0}".format(workflow_name))
    paths = scientific_artifact_paths(root, workflow_name)
    scientific_dir = _require_within(
        paths["scientific_dir"],
        paths["output_scope"],
        "科学执行证据目录",
    )
    if scientific_dir.is_symlink():
        scientific_dir.unlink()
    elif scientific_dir.exists():
        shutil.rmtree(scientific_dir)
    for evidence_key in (
        "execution_report",
        "dependency_report",
        "dispatch_report",
        "execution_binding",
    ):
        evidence_path = _require_within(
            paths[evidence_key],
            paths["output_scope"],
            "科学执行证据文件",
        )
        if evidence_path.is_file() or evidence_path.is_symlink():
            evidence_path.unlink()

    child_argv = _child_argv(paths, workflow_name)
    report, report_path = execute_isolated_scientific_command(
        WORKFLOW_PROFILE_IDS[workflow_name],
        child_argv,
        execution_report_path=paths["source_execution_report"],
        repository_root=paths["root"],
    )
    if Path(report_path).resolve() != paths["source_execution_report"].resolve():
        raise RuntimeError("隔离科学执行 API 返回了非预期报告路径")
    persisted_source_report = _read_json_object(
        paths["source_execution_report"],
        "隔离科学执行报告",
    )
    _validate_source_execution_report(
        report,
        persisted_source_report,
        paths=paths,
        workflow_name=workflow_name,
        child_argv=child_argv,
    )

    envelope_files = tuple(scientific_dir.glob("*result_envelope.json"))
    if envelope_files != (paths["result_envelope"],):
        raise RuntimeError("科学子进程必须写出唯一 result envelope")
    envelope = _read_json_object(paths["result_envelope"], "科学子进程 result envelope")
    summary = _validate_result_envelope(
        envelope,
        report=report,
        paths=paths,
        workflow_name=workflow_name,
    )
    paper_run_name = build_paper_run_config(paths["root"]).run_name
    binding, binding_path = write_scientific_execution_binding(
        paths["output_scope"],
        artifact_role=WORKFLOW_ARTIFACT_ROLES[workflow_name],
        paper_run_name=paper_run_name,
        summary_file_name=paths["summary"].name,
        manifest_file_name=paths["manifest"].name,
        execution_report_path=paths["source_execution_report"],
        dispatch_report_path=paths["result_envelope"],
        expected_profile_id=WORKFLOW_PROFILE_IDS[workflow_name],
        repository_root=paths["root"],
    )
    if Path(binding_path).resolve() != paths["execution_binding"].resolve():
        raise RuntimeError("共享科学执行绑定写入了非预期路径")
    validated_binding = validate_scientific_execution_binding(
        binding_path,
        expected_artifact_role=WORKFLOW_ARTIFACT_ROLES[workflow_name],
        expected_paper_run_name=paper_run_name,
        repository_root=paths["root"],
    )
    if validated_binding != binding:
        raise RuntimeError("共享科学执行绑定写入后发生变化")
    _verify_shared_binding_before_return(
        binding,
        paths=paths,
        source_report=report,
        workflow_name=workflow_name,
    )
    _require_within(paths["execution_report"], paths["output_scope"], "科学执行报告")
    _require_within(paths["dependency_report"], paths["output_scope"], "依赖环境报告")
    paths["source_execution_report"].unlink()
    return summary


def _archive_record_from_mapping(value: Any) -> IsolatedScientificArchiveRecord:
    """把科学子解释器返回值收敛为固定归档记录."""

    if not isinstance(value, Mapping):
        raise TypeError("隔离科学打包子进程未返回 archive record")
    metadata = value.get("metadata")
    if not isinstance(metadata, dict):
        raise TypeError("隔离科学打包 archive record 缺少 metadata")
    archive_entry_count = value.get("archive_entry_count")
    if isinstance(archive_entry_count, bool) or not isinstance(
        archive_entry_count,
        int,
    ):
        raise TypeError("隔离科学打包 archive_entry_count 必须是整数")
    if archive_entry_count <= 0:
        raise ValueError("隔离科学打包 archive_entry_count 必须大于0")
    return IsolatedScientificArchiveRecord(
        archive_path=str(value.get("archive_path", "")),
        archive_digest=str(value.get("archive_digest", "")),
        archive_entry_count=archive_entry_count,
        drive_archive_path=str(value.get("drive_archive_path", "")),
        drive_archive_digest=str(value.get("drive_archive_digest", "")),
        metadata=dict(metadata),
    )


def run_isolated_packaging_child(
    *,
    root: PathLike,
    workflow_name: str,
    drive_output_dir: str,
    archive_name: str | None,
) -> int:
    """在 T2SMark 科学解释器内执行只读验证和正式打包."""

    if workflow_name != "official_reference_t2smark":
        raise ValueError("隔离科学打包当前只支持 T2SMark")
    from paper_experiments.runners.t2smark_formal_reproduction import (
        package_t2smark_formal_reproduction_outputs,
    )

    record = package_t2smark_formal_reproduction_outputs(
        root=root,
        drive_output_dir=drive_output_dir,
        archive_name=archive_name,
    )
    print(
        json.dumps(
            {
                "decision": "pass",
                "archive_record": record.to_dict(),
                "supports_paper_claim": False,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


def package_isolated_scientific_workflow_outputs(
    *,
    root: PathLike = ".",
    workflow_name: str,
    drive_output_dir: str,
    archive_name: str | None = None,
) -> IsolatedScientificArchiveRecord:
    """复用已绑定科学解释器完成 T2SMark 打包并复验本地与镜像摘要."""

    if workflow_name != "official_reference_t2smark":
        raise ValueError("隔离科学打包当前只支持 T2SMark")
    if not str(drive_output_dir).strip():
        raise ValueError("隔离科学打包必须提供镜像目录")
    paths = scientific_artifact_paths(root, workflow_name)
    drive_output_path = Path(drive_output_dir).expanduser()
    if not drive_output_path.is_absolute():
        drive_output_path = paths["root"] / drive_output_path
    drive_output_path = drive_output_path.resolve()
    binding = validate_scientific_execution_binding(
        paths["execution_binding"],
        expected_artifact_role=WORKFLOW_ARTIFACT_ROLES[workflow_name],
        expected_paper_run_name=build_paper_run_config(paths["root"]).run_name,
        repository_root=paths["root"],
    )
    execution_report = _read_json_object(
        paths["execution_report"],
        "隔离科学执行报告",
    )
    python_path = Path(str(execution_report.get("python_executable_path", "")))
    python_digest = str(execution_report.get("python_executable_sha256", ""))
    dependency_path = paths["dependency_report"].resolve()
    dependency_digest = str(binding["dependency_environment_report_digest"])
    formal_execution_lock = require_published_formal_execution_lock(paths["root"])
    if (
        not python_path.is_file()
        or _file_sha256(python_path) != python_digest
        or not dependency_path.is_file()
        or _file_sha256(dependency_path) != dependency_digest
        or execution_report.get("formal_execution_lock") != formal_execution_lock
    ):
        raise RuntimeError("隔离科学打包前解释器,依赖报告或执行锁发生漂移")

    argv = [
        str(python_path),
        "-m",
        "paper_experiments.runners.isolated_scientific_workflow",
        "--package-workflow",
        workflow_name,
        "--root",
        str(paths["root"]),
        "--drive-output-dir",
        str(drive_output_path),
    ]
    if archive_name is not None:
        argv.extend(("--archive-name", archive_name))
    environment = os.environ.copy()
    environment.update(
        {
            FORMAL_EXECUTION_COMMIT_ENVIRONMENT_KEY: formal_execution_lock[
                "formal_execution_commit"
            ],
            FORMAL_EXECUTION_LOCK_DIGEST_ENVIRONMENT_KEY: formal_execution_lock[
                "formal_execution_lock_digest"
            ],
            DEPENDENCY_ENVIRONMENT_REPORT_PATH_ENVIRONMENT_KEY: str(dependency_path),
            DEPENDENCY_ENVIRONMENT_REPORT_DIGEST_ENVIRONMENT_KEY: dependency_digest,
        }
    )
    completed = subprocess.run(
        argv,
        cwd=paths["root"],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        shell=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "隔离科学打包失败: " + (completed.stderr.strip() or "未知错误")
        )
    if (
        not python_path.is_file()
        or _file_sha256(python_path) != python_digest
        or not dependency_path.is_file()
        or _file_sha256(dependency_path) != dependency_digest
        or require_published_formal_execution_lock(paths["root"])
        != formal_execution_lock
    ):
        raise RuntimeError("隔离科学打包后解释器,依赖报告或执行锁发生漂移")
    output_lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if not output_lines:
        raise RuntimeError("隔离科学打包未返回 JSON record")
    try:
        result = json.loads(output_lines[-1])
    except json.JSONDecodeError as error:
        raise RuntimeError("隔离科学打包返回值不是有效 JSON") from error
    if (
        not isinstance(result, dict)
        or result.get("decision") != "pass"
        or result.get("supports_paper_claim") is not False
    ):
        raise RuntimeError("隔离科学打包返回值未通过门禁")
    record = _archive_record_from_mapping(result.get("archive_record"))
    local_archive = (paths["root"] / record.archive_path).resolve()
    _require_within(local_archive, paths["output_scope"], "T2SMark 本地归档")
    drive_archive = Path(record.drive_archive_path).expanduser().resolve()
    _require_within(drive_archive, drive_output_path, "T2SMark 镜像归档")
    if (
        not local_archive.is_file()
        or _file_sha256(local_archive) != record.archive_digest
        or not drive_archive.is_file()
        or _file_sha256(drive_archive) != record.drive_archive_digest
        or record.archive_digest != record.drive_archive_digest
        or local_archive.name != drive_archive.name
    ):
        raise RuntimeError("隔离科学打包返回的本地或镜像归档摘要无效")
    return record


def build_argument_parser() -> argparse.ArgumentParser:
    """构造可脱离 Notebook 使用的科学子进程参数解析器."""

    parser = argparse.ArgumentParser(description="运行隔离科学 workflow 子进程")
    operation = parser.add_mutually_exclusive_group(required=True)
    operation.add_argument("--child-workflow", choices=tuple(WORKFLOW_PROFILE_IDS))
    operation.add_argument(
        "--package-workflow",
        choices=("official_reference_t2smark",),
    )
    parser.add_argument("--root", default=".")
    parser.add_argument("--result-envelope")
    parser.add_argument("--drive-output-dir")
    parser.add_argument("--archive-name")
    return parser


def main(argv: Union[Sequence[str], None] = None) -> int:
    """执行隔离科学 workflow 子进程 CLI."""

    args = build_argument_parser().parse_args(argv)
    if args.package_workflow is not None:
        if not args.drive_output_dir:
            raise ValueError("隔离科学打包必须提供 --drive-output-dir")
        return run_isolated_packaging_child(
            root=args.root,
            workflow_name=args.package_workflow,
            drive_output_dir=args.drive_output_dir,
            archive_name=args.archive_name,
        )
    if not args.result_envelope:
        raise ValueError("隔离科学 workflow 必须提供 --result-envelope")
    return run_scientific_child(
        root=args.root,
        workflow_name=args.child_workflow,
        result_envelope_path=args.result_envelope,
    )


if __name__ == "__main__":
    raise SystemExit(main())
