"""把隔离科学执行证据绑定到已经完成的正式实验产物.

该模块只处理文件摘要,执行报告校验和不可变证据快照, 不执行模型方法或统计
协议.外层 workflow 可以复用该能力, 而 ``scripts`` 也能在不依赖 Notebook 的
情况下验证待重新打包的正式产物.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import shutil
from typing import Any, Mapping

from experiments.runtime.isolated_dependency_environment import (
    validate_formal_dependency_environment_report,
)
from experiments.runtime.repository_environment import (
    FormalExecutionLockError,
    validate_formal_execution_lock_record,
)


BINDING_SCHEMA = "scientific_execution_binding"
BINDING_SCHEMA_VERSION = 2
BINDING_FILE_NAME = "scientific_execution_binding.json"
EXECUTION_REPORT_FILE_NAME = "isolated_scientific_execution_report.json"
DEPENDENCY_REPORT_FILE_NAME = "isolated_dependency_environment_report.json"
DISPATCH_REPORT_FILE_NAME = "scientific_command_dispatch_report.json"
BOUND_MANIFEST_DIGEST_SCOPE = "json_without_formal_execution_package_lock"


def file_sha256(path: Path) -> str:
    """计算文件的 SHA-256, 供执行证据和正式产物交叉绑定."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scientific_manifest_payload_digest(payload: Mapping[str, Any]) -> str:
    """计算不含打包边界锁的科学 manifest 稳定摘要.

    科学 runner 写出 ``formal_execution_run_lock`` 后即完成其 manifest. 打包器
    只允许补充 ``formal_execution_package_lock``.排除该唯一字段后计算摘要,
    可以同时证明科学内容未被改写并保留独立的真实打包边界锁.
    """

    scientific_payload = dict(payload)
    scientific_payload.pop("formal_execution_package_lock", None)
    encoded = json.dumps(
        scientific_payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def stable_json_file_digest(payload: Mapping[str, Any]) -> str:
    """按 runtime 报告固定排版计算 JSON 文件 SHA-256."""

    encoded = (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def semantic_command_sequence_digest(
    dispatch_report: Mapping[str, Any],
) -> str:
    """计算不含 stdout/stderr 的主方法科学命令序列摘要."""

    commands = dispatch_report.get("commands")
    if not isinstance(commands, list) or not all(
        isinstance(command, dict) for command in commands
    ):
        raise RuntimeError("主方法逐命令调度列表无效")
    normalized = [
        {
            "command_role": command.get("command_role"),
            "argv": command.get("argv"),
            "return_code": command.get("return_code"),
            "packaging_deferred": command.get("packaging_deferred"),
        }
        for command in commands
    ]
    encoded = json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_semantic_watermark_dispatch_report(
    dispatch_report: Mapping[str, Any],
    execution_report: Mapping[str, Any],
) -> None:
    """验证主方法隔离会话实际执行的内层工作负载命令."""

    child_argv_tail = execution_report.get("child_argv_tail")
    valid_session_tails = (
        ["-m", "experiments.runtime.semantic_watermark_scientific_session"],
        [
            "-m",
            "experiments.runtime.semantic_watermark_scientific_session",
            "--run-formal-ablation",
        ],
    )
    if child_argv_tail not in valid_session_tails:
        raise RuntimeError("主方法科学执行报告的会话命令无效")
    include_ablation = child_argv_tail == valid_session_tails[1]
    python_executable = str(execution_report.get("python_executable_path", ""))
    expected_commands = [
        (
            "image_only_dataset_runtime",
            [
                python_executable,
                "-m",
                "experiments.runners.image_only_dataset_workload",
            ],
        )
    ]
    if include_ablation:
        expected_commands.extend(
            (
                (
                    "runtime_rerun_ablation",
                    [
                        python_executable,
                        "-m",
                        "experiments.ablations.mechanism_ablation_workload",
                    ],
                ),
                (
                    "branch_risk_parameter_sensitivity",
                    [
                        python_executable,
                        "-m",
                        "experiments.ablations.branch_risk_sensitivity_workload",
                    ],
                ),
            )
        )
    commands = dispatch_report.get("commands")
    if not all(
        (
            bool(python_executable),
            dispatch_report.get("python_executable") == python_executable,
            dispatch_report.get("formal_ablation_requested") is include_ablation,
            dispatch_report.get("packaging_deferred") is True,
            dispatch_report.get("session_execution_decision") == "pass",
            dispatch_report.get("workflow_completion_state")
            == "repeat_component_complete",
            dispatch_report.get("paper_run_closed") is False,
            dispatch_report.get("result_closure_ready") is False,
            isinstance(commands, list),
            len(commands) == len(expected_commands)
            if isinstance(commands, list)
            else False,
        )
    ):
        raise RuntimeError("主方法逐命令调度身份无效")
    for command, (expected_role, expected_argv) in zip(
        commands,
        expected_commands,
    ):
        if not all(
            (
                isinstance(command, dict),
                command.get("command_role") == expected_role
                if isinstance(command, dict)
                else False,
                command.get("argv") == expected_argv
                if isinstance(command, dict)
                else False,
                command.get("return_code") == 0
                if isinstance(command, dict)
                else False,
                command.get("packaging_deferred") is True
                if isinstance(command, dict)
                else False,
            )
        ):
            raise RuntimeError("主方法逐命令调度内容无效")
    expected_artifact_roles = [
        "image_only_dataset_runtime",
        "dataset_level_quality",
    ]
    if include_ablation:
        expected_artifact_roles.extend(
            (
                "runtime_rerun_ablation",
                "branch_risk_parameter_sensitivity",
            )
        )
    artifact_records = dispatch_report.get("artifact_records")
    if not isinstance(artifact_records, list) or [
        record.get("artifact_role") if isinstance(record, dict) else None
        for record in artifact_records
    ] != expected_artifact_roles:
        raise RuntimeError("主方法调度报告的闭合产物角色集合无效")
    formal_lock = _validated_formal_execution_lock(
        execution_report.get("formal_execution_lock")
    )
    for record in artifact_records:
        assert isinstance(record, dict)
        digest_fields = (
            "summary_sha256",
            "manifest_sha256_at_session",
            "manifest_scientific_digest",
        )
        relative_paths = (
            str(record.get("summary_path", "")),
            str(record.get("manifest_path", "")),
        )
        if not all(
            (
                dispatch_report.get("artifact_validation_mode")
                == "completed_or_revalidated_in_current_session",
                record.get("summary_protocol_decision") == "pass",
                record.get("formal_execution_run_lock") == formal_lock,
                all(
                    isinstance(record.get(field_name), str)
                    and len(record[field_name]) == 64
                    and all(
                        character in "0123456789abcdef"
                        for character in record[field_name]
                    )
                    for field_name in digest_fields
                ),
                all(
                    bool(path)
                    and not PurePosixPath(path).is_absolute()
                    and ".." not in PurePosixPath(path).parts
                    for path in relative_paths
                ),
            )
        ):
            raise RuntimeError("主方法调度报告的产物摘要记录无效")


def validate_semantic_watermark_dispatch_artifact_snapshot(
    dispatch_report: Mapping[str, Any],
    *,
    artifact_role: str,
    summary_path: str,
    summary_sha256: str,
    manifest_path: str,
    manifest_payload: Mapping[str, Any],
    manifest_sha256: str | None,
    formal_execution_lock: Mapping[str, Any],
) -> None:
    """复验一个角色化产物快照与主方法调度报告的因果绑定."""

    records = dispatch_report.get("artifact_records")
    matching = (
        [
            record
            for record in records
            if isinstance(record, dict)
            and record.get("artifact_role") == artifact_role
        ]
        if isinstance(records, list)
        else []
    )
    if len(matching) != 1:
        raise RuntimeError("主方法调度报告缺少唯一角色化产物记录")
    record = matching[0]
    ready = all(
        (
            record.get("summary_path") == summary_path,
            record.get("summary_sha256") == summary_sha256,
            record.get("manifest_path") == manifest_path,
            record.get("manifest_scientific_digest")
            == scientific_manifest_payload_digest(manifest_payload),
            record.get("formal_execution_run_lock")
            == dict(formal_execution_lock),
        )
    )
    if manifest_sha256 is not None:
        ready = ready and record.get("manifest_sha256_at_session") == manifest_sha256
    if not ready:
        raise RuntimeError("主方法调度报告与角色化正式产物不一致")


def validate_semantic_watermark_dispatch_artifact(
    dispatch_report: Mapping[str, Any],
    *,
    artifact_role: str,
    summary_path: Path,
    manifest_path: Path,
    repository_root: Path,
    formal_execution_lock: Mapping[str, Any],
    require_prepackaged_manifest_sha256: bool,
) -> None:
    """从仓库文件复验主方法调度报告中的角色化产物绑定."""

    validate_semantic_watermark_dispatch_artifact_snapshot(
        dispatch_report,
        artifact_role=artifact_role,
        summary_path=summary_path.resolve().relative_to(
            repository_root.resolve()
        ).as_posix(),
        summary_sha256=file_sha256(summary_path),
        manifest_path=manifest_path.resolve().relative_to(
            repository_root.resolve()
        ).as_posix(),
        manifest_payload=_load_json_object(manifest_path),
        manifest_sha256=(
            file_sha256(manifest_path)
            if require_prepackaged_manifest_sha256
            else None
        ),
        formal_execution_lock=formal_execution_lock,
    )


def _load_json_object(path: Path) -> dict[str, Any]:
    """读取 JSON 对象, 对缺失文件和非对象根节点立即失败."""

    if not path.is_file():
        raise FileNotFoundError(f"缺少科学执行证据文件: {path}")
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"科学执行证据必须是 JSON 对象: {path}")
    return payload


def normalize_scientific_absolute_path(value: Any) -> str:
    """把 POSIX 或 Windows 绝对路径转换为可跨主机比较的文本."""

    text = str(value or "").strip()
    windows_path = PureWindowsPath(text)
    if windows_path.is_absolute():
        if ".." in windows_path.parts:
            raise RuntimeError("科学执行路径不得包含父目录跳转")
        return windows_path.as_posix().rstrip("/").casefold()
    posix_path = PurePosixPath(text)
    if posix_path.is_absolute():
        if ".." in posix_path.parts:
            raise RuntimeError("科学执行路径不得包含父目录跳转")
        return posix_path.as_posix().rstrip("/")
    raise RuntimeError("科学执行路径必须是绝对路径")


def validate_scientific_command_context_snapshot(
    report: Mapping[str, Any],
    *,
    expected_profile_id: str,
) -> dict[str, str]:
    """复验科学子命令、仓库根与依赖报告来自同一执行上下文."""

    execution = report.get("execution")
    child_argv_tail = report.get("child_argv_tail")
    if not isinstance(execution, dict) or not isinstance(child_argv_tail, list):
        raise RuntimeError("隔离科学执行报告缺少命令上下文")
    repository_root = normalize_scientific_absolute_path(
        report.get("repository_root")
    )
    working_directory = normalize_scientific_absolute_path(
        execution.get("working_directory")
    )
    python_executable = normalize_scientific_absolute_path(
        report.get("python_executable_path")
    )
    environment = execution.get("environment_overrides")
    if not isinstance(environment, dict):
        raise RuntimeError("隔离科学执行报告缺少环境覆盖")
    dependency_environment_path = normalize_scientific_absolute_path(
        environment.get("SLM_WM_ISOLATED_DEPENDENCY_ENVIRONMENT_REPORT_PATH")
    )
    declared_source_dependency_path = normalize_scientific_absolute_path(
        report.get("source_dependency_environment_report_path")
    )
    expected_dependency_environment_path = (
        repository_root
        + "/outputs/dependency_profiles/"
        + expected_profile_id
        + "/"
        + DEPENDENCY_REPORT_FILE_NAME
    )
    formal_lock = _validated_formal_execution_lock(
        report.get("formal_execution_lock")
    )
    expected_environment = {
        "SLM_WM_FORMAL_EXECUTION_COMMIT": formal_lock[
            "formal_execution_commit"
        ],
        "SLM_WM_FORMAL_EXECUTION_LOCK_DIGEST": formal_lock[
            "formal_execution_lock_digest"
        ],
        "SLM_WM_ISOLATED_DEPENDENCY_ENVIRONMENT_REPORT_PATH": str(
            environment.get(
                "SLM_WM_ISOLATED_DEPENDENCY_ENVIRONMENT_REPORT_PATH",
                "",
            )
        ),
        "SLM_WM_ISOLATED_DEPENDENCY_ENVIRONMENT_REPORT_DIGEST": report.get(
            "dependency_environment_report_digest"
        ),
    }
    argv = execution.get("argv")
    if not all(
        (
            report.get("profile_id") == expected_profile_id,
            repository_root == working_directory,
            dependency_environment_path == expected_dependency_environment_path,
            declared_source_dependency_path
            == expected_dependency_environment_path,
            set(environment) == set(expected_environment),
            environment == expected_environment,
            bool(child_argv_tail),
            all(isinstance(token, str) and token for token in child_argv_tail),
            isinstance(argv, list),
            argv
            == [str(report.get("python_executable_path", "")), *child_argv_tail],
        )
    ):
        raise RuntimeError("隔离科学执行报告的命令或上下文身份无效")
    return {
        "repository_root": repository_root,
        "working_directory": working_directory,
        "python_executable": python_executable,
        "source_dependency_environment_report_path": (
            dependency_environment_path
        ),
    }


def _validated_formal_execution_lock(value: Any) -> dict[str, Any]:
    """复用统一 validator 复验完整正式执行锁."""

    try:
        return validate_formal_execution_lock_record(value)
    except FormalExecutionLockError as exc:
        raise RuntimeError("科学执行证据携带的正式执行锁无效") from exc


def validate_dependency_environment_report_snapshot(
    report: Mapping[str, Any],
    *,
    expected_profile_id: str,
    expected_profile_digest: str,
    expected_direct_requirements_digest: str,
    expected_complete_hash_lock_digest: str,
    expected_complete_hash_lock_dependency_count: int,
    expected_python_executable_path: str | Path,
    expected_python_executable_digest: str,
    expected_formal_execution_lock: Mapping[str, Any],
    expected_working_directory: str | Path,
) -> None:
    """复验科学执行报告引用的正式隔离依赖环境快照."""

    dependency_preparation = report.get("dependency_preparation_report")
    if not isinstance(dependency_preparation, Mapping):
        raise RuntimeError("隔离依赖环境报告缺少内层 dependency preparation")
    complete_hash_lock_path = dependency_preparation.get(
        "complete_hash_lock_path"
    )
    if not isinstance(complete_hash_lock_path, str):
        raise RuntimeError("隔离依赖环境报告缺少完整哈希锁路径")
    pytorch_index_url = dependency_preparation.get("pytorch_index_url")
    if pytorch_index_url is not None and not isinstance(pytorch_index_url, str):
        raise RuntimeError("隔离依赖环境报告的 PyTorch index 无效")
    errors = validate_formal_dependency_environment_report(
        report,
        expected_profile_id=expected_profile_id,
        expected_profile_digest=expected_profile_digest,
        expected_direct_requirements_digest=expected_direct_requirements_digest,
        expected_complete_hash_lock_path=complete_hash_lock_path,
        expected_complete_hash_lock_digest=expected_complete_hash_lock_digest,
        expected_complete_hash_lock_dependency_count=(
            expected_complete_hash_lock_dependency_count
        ),
        expected_pytorch_index_url=pytorch_index_url,
        expected_python_executable_path=expected_python_executable_path,
        expected_python_executable_digest=expected_python_executable_digest,
        expected_formal_execution_lock=expected_formal_execution_lock,
        expected_working_directory=expected_working_directory,
    )
    if errors:
        raise RuntimeError(
            "隔离依赖环境报告没有通过完整命令与数量门禁: "
            + ",".join(errors)
        )


def validate_scientific_execution_report(
    report_path: str | Path,
    *,
    expected_profile_id: str,
    require_live_runtime: bool = True,
) -> dict[str, Any]:
    """验证一次成功执行及其依赖报告,解释器和执行锁证据.

    ``require_live_runtime=True`` 用于刚完成的 GPU 会话, 会重新计算临时解释器和
    依赖报告摘要.``False`` 用于 Drive 下载后的离线审计, 此时只核验已复制到
    正式产物目录的报告快照以及执行时已经完成的双向重验证标志.
    """

    resolved_report_path = Path(report_path).resolve()
    report = _load_json_object(resolved_report_path)
    required_true_fields = (
        "execution_completed",
        "dependency_environment_report_valid",
        "python_executable_revalidated_before_child",
        "python_executable_revalidated_after_child",
        "dependency_environment_report_revalidated_before_child",
        "dependency_environment_report_revalidated_after_child",
        "formal_execution_lock_ready",
        "formal_execution_lock_revalidated_before_child",
        "formal_execution_lock_revalidated_after_child",
    )
    execution = report.get("execution")
    if not all(
        (
            report.get("report_schema") == "isolated_scientific_execution_report",
            report.get("schema_version") == 1,
            report.get("operation_kind") == "isolated_scientific_execution",
            report.get("profile_id") == expected_profile_id,
            report.get("decision") == "pass",
            report.get("failure_reasons") == [],
            report.get("supports_paper_claim") is False,
            all(report.get(field_name) is True for field_name in required_true_fields),
            isinstance(execution, dict),
            execution.get("attempted") is True if isinstance(execution, dict) else False,
            execution.get("return_code") == 0 if isinstance(execution, dict) else False,
        )
    ):
        raise RuntimeError("隔离科学执行报告没有通过完整执行门禁")

    report_path_value = Path(str(report.get("execution_report_path", "")))
    if not report_path_value.is_absolute():
        report_path_value = resolved_report_path.parent / report_path_value
    if report_path_value.resolve() != resolved_report_path:
        raise RuntimeError("隔离科学执行报告没有绑定当前报告文件")
    for digest_field in (
        "profile_digest",
        "direct_requirements_digest",
        "complete_hash_lock_digest",
        "dependency_environment_report_digest",
        "python_executable_sha256",
        "formal_execution_lock_digest",
    ):
        digest_value = report.get(digest_field)
        if (
            not isinstance(digest_value, str)
            or len(digest_value) != 64
            or any(character not in "0123456789abcdef" for character in digest_value)
        ):
            raise RuntimeError(f"隔离科学执行报告摘要字段无效: {digest_field}")
    dependency_count = report.get("complete_hash_lock_dependency_count")
    if (
        isinstance(dependency_count, bool)
        or not isinstance(dependency_count, int)
        or dependency_count <= 0
    ):
        raise RuntimeError("隔离科学执行报告未绑定完整依赖锁包集合")

    dependency_path = Path(str(report.get("dependency_environment_report_path", "")))
    if not dependency_path.is_absolute():
        dependency_path = resolved_report_path.parent / dependency_path
    dependency_path = dependency_path.resolve()
    python_path = Path(str(report.get("python_executable_path", "")))
    if not dependency_path.is_file():
        raise RuntimeError("隔离依赖环境报告路径无效")
    if file_sha256(dependency_path) != report.get(
        "dependency_environment_report_digest"
    ):
        raise RuntimeError("隔离依赖环境报告摘要与执行报告不一致")
    python_digest = report["python_executable_sha256"]
    if not python_path.is_absolute():
        raise RuntimeError("隔离科学解释器记录路径无效")
    if require_live_runtime:
        if not python_path.is_absolute() or not python_path.is_file():
            raise RuntimeError("隔离科学解释器路径无效")
        if file_sha256(python_path) != python_digest:
            raise RuntimeError("隔离科学解释器摘要与执行报告不一致")
    formal_lock = _validated_formal_execution_lock(
        report.get("formal_execution_lock")
    )
    if (
        report.get("formal_execution_commit")
        != formal_lock.get("formal_execution_commit")
        or report.get("formal_execution_lock_digest")
        != formal_lock.get("formal_execution_lock_digest")
    ):
        raise RuntimeError("隔离科学执行报告的正式执行锁身份不一致")
    command_context = validate_scientific_command_context_snapshot(
        report,
        expected_profile_id=expected_profile_id,
    )
    dependency_report = _load_json_object(dependency_path)
    validate_dependency_environment_report_snapshot(
        dependency_report,
        expected_profile_id=expected_profile_id,
        expected_profile_digest=str(report["profile_digest"]),
        expected_direct_requirements_digest=str(
            report["direct_requirements_digest"]
        ),
        expected_complete_hash_lock_digest=str(
            report["complete_hash_lock_digest"]
        ),
        expected_complete_hash_lock_dependency_count=dependency_count,
        expected_python_executable_path=python_path,
        expected_python_executable_digest=str(python_digest),
        expected_formal_execution_lock=formal_lock,
        expected_working_directory=command_context["working_directory"],
    )
    return report


def write_scientific_execution_binding(
    artifact_dir: str | Path,
    *,
    artifact_role: str,
    paper_run_name: str,
    summary_file_name: str,
    manifest_file_name: str,
    execution_report_path: str | Path,
    dispatch_report_path: str | Path,
    expected_profile_id: str,
    repository_root: str | Path,
) -> tuple[dict[str, Any], Path]:
    """复制科学执行证据并写出由摘要约束的独立绑定文件.

    绑定文件不修改科学 runner 已经完成的摘要或 manifest.它通过摘要绑定原始文件,
    因而在其他项目中也可复用为"计算产物与隔离环境证据分离保存"的通用结构.
    """

    root = Path(repository_root).resolve()
    resolved_artifact_dir = Path(artifact_dir).resolve()
    resolved_artifact_dir.relative_to((root / "outputs").resolve())
    summary_path = resolved_artifact_dir / summary_file_name
    manifest_path = resolved_artifact_dir / manifest_file_name
    if not summary_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("正式产物摘要或 manifest 缺失, 不得绑定执行证据")

    source_execution_path = Path(execution_report_path).resolve()
    execution_report = validate_scientific_execution_report(
        source_execution_path,
        expected_profile_id=expected_profile_id,
    )
    source_dependency_path = Path(
        str(execution_report["dependency_environment_report_path"])
    ).resolve()
    source_dispatch_path = Path(dispatch_report_path).resolve()
    dispatch_report = _load_json_object(source_dispatch_path)
    if (
        dispatch_report.get("report_schema") != "scientific_command_dispatch_report"
        or dispatch_report.get("schema_version") != 1
        or dispatch_report.get("paper_run_name") != paper_run_name
        or dispatch_report.get("decision") != "pass"
    ):
        raise RuntimeError("科学命令调度报告没有通过绑定门禁")
    child_argv_tail = execution_report.get("child_argv_tail")
    if (
        isinstance(child_argv_tail, list)
        and child_argv_tail[:2]
        == ["-m", "experiments.runtime.semantic_watermark_scientific_session"]
    ):
        validate_semantic_watermark_dispatch_report(
            dispatch_report,
            execution_report,
        )
        validate_semantic_watermark_dispatch_artifact(
            dispatch_report,
            artifact_role=artifact_role,
            summary_path=summary_path,
            manifest_path=manifest_path,
            repository_root=root,
            formal_execution_lock=execution_report["formal_execution_lock"],
            require_prepackaged_manifest_sha256=True,
        )

    local_dependency_path = resolved_artifact_dir / DEPENDENCY_REPORT_FILE_NAME
    local_execution_path = resolved_artifact_dir / EXECUTION_REPORT_FILE_NAME
    local_dispatch_path = resolved_artifact_dir / DISPATCH_REPORT_FILE_NAME
    shutil.copyfile(source_dependency_path, local_dependency_path)
    shutil.copyfile(source_dispatch_path, local_dispatch_path)
    localized_execution_report = dict(execution_report)
    localized_execution_report["dependency_environment_report_path"] = (
        DEPENDENCY_REPORT_FILE_NAME
    )
    localized_execution_report["dependency_environment_report_digest"] = file_sha256(
        local_dependency_path
    )
    localized_execution_report["execution_report_path"] = EXECUTION_REPORT_FILE_NAME
    local_execution_path.write_text(
        json.dumps(
            localized_execution_report,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    validate_scientific_execution_report(
        local_execution_path,
        expected_profile_id=expected_profile_id,
        require_live_runtime=False,
    )

    def relative(path: Path) -> str:
        """把证据路径规范化为仓库相对 POSIX 路径."""

        return path.relative_to(root).as_posix()

    formal_lock = execution_report["formal_execution_lock"]
    binding = {
        "report_schema": BINDING_SCHEMA,
        "schema_version": BINDING_SCHEMA_VERSION,
        "artifact_role": artifact_role,
        "paper_run_name": paper_run_name,
        "profile_id": execution_report["profile_id"],
        "profile_digest": execution_report["profile_digest"],
        "direct_requirements_digest": execution_report[
            "direct_requirements_digest"
        ],
        "complete_hash_lock_digest": execution_report[
            "complete_hash_lock_digest"
        ],
        "scientific_execution_report_path": relative(local_execution_path),
        "scientific_execution_report_digest": file_sha256(local_execution_path),
        "dependency_environment_report_path": relative(local_dependency_path),
        "dependency_environment_report_digest": file_sha256(local_dependency_path),
        "scientific_command_dispatch_report_path": relative(local_dispatch_path),
        "scientific_command_dispatch_report_digest": file_sha256(local_dispatch_path),
        "bound_summary_path": relative(summary_path),
        "bound_summary_digest": file_sha256(summary_path),
        "bound_manifest_path": relative(manifest_path),
        "bound_manifest_scientific_digest": scientific_manifest_payload_digest(
            _load_json_object(manifest_path)
        ),
        "bound_manifest_digest_scope": BOUND_MANIFEST_DIGEST_SCOPE,
        "formal_execution_lock": formal_lock,
        "formal_execution_commit": execution_report["formal_execution_commit"],
        "formal_execution_lock_digest": execution_report[
            "formal_execution_lock_digest"
        ],
        "decision": "pass",
        "supports_paper_claim": False,
    }
    binding_path = resolved_artifact_dir / BINDING_FILE_NAME
    binding_path.write_text(
        json.dumps(binding, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return binding, binding_path


def validate_scientific_execution_binding(
    binding_path: str | Path,
    *,
    expected_artifact_role: str,
    expected_paper_run_name: str,
    repository_root: str | Path,
) -> dict[str, Any]:
    """重新计算全部摘要, 确认绑定文件仍与待打包产物一致."""

    root = Path(repository_root).resolve()
    binding = _load_json_object(Path(binding_path).resolve())
    expected_values = {
        "report_schema": BINDING_SCHEMA,
        "schema_version": BINDING_SCHEMA_VERSION,
        "artifact_role": expected_artifact_role,
        "paper_run_name": expected_paper_run_name,
        "decision": "pass",
        "supports_paper_claim": False,
    }
    if any(binding.get(key) != value for key, value in expected_values.items()):
        raise RuntimeError("科学执行绑定的产物身份或门禁无效")
    path_digest_fields = (
        ("scientific_execution_report_path", "scientific_execution_report_digest"),
        ("dependency_environment_report_path", "dependency_environment_report_digest"),
        (
            "scientific_command_dispatch_report_path",
            "scientific_command_dispatch_report_digest",
        ),
        ("bound_summary_path", "bound_summary_digest"),
    )
    for path_field, digest_field in path_digest_fields:
        evidence_path = (root / str(binding.get(path_field, ""))).resolve()
        evidence_path.relative_to((root / "outputs").resolve())
        if not evidence_path.is_file() or file_sha256(evidence_path) != binding.get(
            digest_field
        ):
            raise RuntimeError(f"科学执行绑定摘要失配: {path_field}")
    bound_manifest_path = (root / str(binding.get("bound_manifest_path", ""))).resolve()
    bound_manifest_path.relative_to((root / "outputs").resolve())
    if (
        not bound_manifest_path.is_file()
        or binding.get("bound_manifest_digest_scope")
        != BOUND_MANIFEST_DIGEST_SCOPE
        or binding.get("bound_manifest_scientific_digest")
        != scientific_manifest_payload_digest(
            _load_json_object(bound_manifest_path)
        )
    ):
        raise RuntimeError("科学执行绑定的科学 manifest 摘要失配")
    execution_report = validate_scientific_execution_report(
        root / binding["scientific_execution_report_path"],
        expected_profile_id=str(binding.get("profile_id", "")),
        require_live_runtime=False,
    )
    identity_fields = (
        "profile_id",
        "profile_digest",
        "direct_requirements_digest",
        "complete_hash_lock_digest",
        "formal_execution_lock",
        "formal_execution_commit",
        "formal_execution_lock_digest",
    )
    if any(
        binding.get(field_name) != execution_report.get(field_name)
        for field_name in identity_fields
    ):
        raise RuntimeError("科学执行绑定与本地执行报告身份不一致")
    binding_lock = _validated_formal_execution_lock(
        binding.get("formal_execution_lock")
    )
    if binding_lock != execution_report.get("formal_execution_lock"):
        raise RuntimeError("科学执行绑定的正式执行锁与执行报告不一致")
    if (
        binding.get("dependency_environment_report_digest")
        != execution_report.get("dependency_environment_report_digest")
    ):
        raise RuntimeError("科学执行绑定与本地依赖报告摘要不一致")
    localized_dependency_path = (
        Path(root / binding["scientific_execution_report_path"]).parent
        / str(execution_report.get("dependency_environment_report_path", ""))
    ).resolve()
    bound_dependency_path = (
        root / str(binding.get("dependency_environment_report_path", ""))
    ).resolve()
    if localized_dependency_path != bound_dependency_path:
        raise RuntimeError("科学执行绑定与执行报告的依赖路径不一致")
    dispatch_report = _load_json_object(
        root / binding["scientific_command_dispatch_report_path"]
    )
    if not all(
        (
            dispatch_report.get("report_schema")
            == "scientific_command_dispatch_report",
            dispatch_report.get("schema_version") == 1,
            dispatch_report.get("paper_run_name") == expected_paper_run_name,
            dispatch_report.get("decision") == "pass",
            dispatch_report.get("failure_reasons", []) == [],
            dispatch_report.get("supports_paper_claim") is False,
        )
    ):
        raise RuntimeError("科学执行绑定的逐命令调度报告无效")
    child_argv_tail = execution_report.get("child_argv_tail")
    if (
        isinstance(child_argv_tail, list)
        and child_argv_tail[:2]
        == ["-m", "experiments.runtime.semantic_watermark_scientific_session"]
    ):
        validate_semantic_watermark_dispatch_report(
            dispatch_report,
            execution_report,
        )
        validate_semantic_watermark_dispatch_artifact(
            dispatch_report,
            artifact_role=expected_artifact_role,
            summary_path=root / binding["bound_summary_path"],
            manifest_path=bound_manifest_path,
            repository_root=root,
            formal_execution_lock=execution_report["formal_execution_lock"],
            require_prepackaged_manifest_sha256=False,
        )
    return binding
