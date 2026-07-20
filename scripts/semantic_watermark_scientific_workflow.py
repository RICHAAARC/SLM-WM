"""调度真实科学算子、仅图像检测、正式消融和参数敏感性的 GPU 会话。

Notebook 所在解释器只运行 workflow 编排. 主方法, 规范 Inception 质量评估和
正式消融与单模型参数敏感性由同一个 ``sd35_method_runtime_gpu`` 子解释器完成。
本模块不依赖 Notebook. 它判断续跑状态, 绑定执行证据并重新打包; 可选的
归档目标目录由更外层 Colab 包装显式注入.
"""

from __future__ import annotations

import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
from typing import Any, Mapping
from zipfile import ZipFile

from experiments.protocol.paper_run_config import build_paper_run_config
from experiments.runtime.dependency_profiles import require_dependency_profile_ready
from experiments.runtime.isolated_scientific_execution import (
    execute_isolated_scientific_command,
)
from experiments.runtime.repository_environment import (
    require_published_formal_execution_lock,
)
from experiments.runtime.scientific_execution_binding import (
    DISPATCH_REPORT_FILE_NAME,
    file_sha256,
    validate_scientific_execution_report,
    write_scientific_execution_binding,
)
from paper_experiments.runners.closure_package_selection import (
    CLOSURE_PACKAGE_FAMILY_SPECS,
    ClosurePackageSelectionError,
    inspect_closure_package,
)


SCIENTIFIC_PROFILE_ID = "sd35_method_runtime_gpu"
SCIENTIFIC_DISPATCH_MODULE = (
    "experiments.runtime.semantic_watermark_scientific_session"
)
_CONTENT_STRENGTH_CANDIDATE_ROLES = {
    "0.75": "content_strength_075",
    "1.0": "content_strength_100",
    "1.25": "content_strength_125",
}


def _content_strength_candidate_role() -> str | None:
    """解析official calibration-only会话的唯一内容倍率角色。"""

    sensitivity = os.environ.get(
        "SLM_WM_CALIBRATION_CONTENT_STRENGTH_SENSITIVITY", ""
    )
    if sensitivity == "":
        return None
    if sensitivity != "1" or os.environ.get("SLM_WM_CALIBRATION_ONLY") != "1":
        raise RuntimeError("内容倍率候选只允许official calibration-only会话")
    role = _CONTENT_STRENGTH_CANDIDATE_ROLES.get(
        os.environ.get("SLM_WM_CONTENT_STRENGTH_COMMON_MULTIPLIER", "")
    )
    if role is None:
        raise RuntimeError("内容倍率候选角色与冻结三候选不一致")
    return role


def _candidate_scoped_directory(path: Path, candidate_role: str | None) -> Path:
    """把隔离执行与dispatch证据置于唯一候选目录。"""

    return path if candidate_role is None else path / candidate_role


def _read_json(path: Path) -> dict[str, Any]:
    """读取一个 JSON 对象, 缺失或格式不符时返回空对象."""

    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _require_repeat_component_ready(
    summary: Mapping[str, Any],
    *,
    artifact_role: str,
) -> None:
    """阻止未完成当前 seed-key 正式证据单元的产物进入闭合."""

    if summary.get("repeat_component_ready") is not True:
        raise RuntimeError(f"{artifact_role} 尚未形成可聚合的重复证据组件")


def _mirror_archive(source_path: Path, destination_dir: Path) -> str:
    """以临时副本、摘要校验和原子 rename 镜像闭合结果包."""

    if not source_path.is_file():
        raise FileNotFoundError(f"缺少待镜像结果包: {source_path}")
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / source_path.name
    source_digest = file_sha256(source_path)
    temporary_destination = destination.with_name(destination.name + ".partial")
    shutil.copyfile(source_path, temporary_destination)
    if file_sha256(temporary_destination) != source_digest:
        temporary_destination.unlink(missing_ok=True)
        raise RuntimeError(f"结果包镜像临时副本摘要不一致: {destination}")
    os.replace(temporary_destination, destination)
    if file_sha256(destination) != source_digest:
        raise RuntimeError(f"结果包镜像发布后摘要不一致: {destination}")
    return str(destination)


def _semantic_package_spec(artifact_role: str) -> Any:
    """返回主方法角色唯一对应的 closure package 规范."""

    return next(
        spec
        for spec in CLOSURE_PACKAGE_FAMILY_SPECS
        if spec.package_family == artifact_role
    )


def _candidate_matches_repository(candidate: Any, root_path: Path) -> bool:
    """把已闭合包的代码锁和科学依赖身份锚定到当前仓库."""

    execution_lock = require_published_formal_execution_lock(root_path)
    if not all(
        (
            candidate.code_version == execution_lock["formal_execution_commit"],
            candidate.formal_execution_run_lock_digest
            == execution_lock["formal_execution_lock_digest"],
            candidate.formal_execution_package_lock_digest
            == execution_lock["formal_execution_lock_digest"],
        )
    ):
        return False
    profile = require_dependency_profile_ready(
        candidate.scientific_profile_id,
        root_path / "configs" / "dependency_profile_registry.json",
    )
    return all(
        (
            candidate.scientific_profile_id == profile.profile_name,
            candidate.scientific_profile_digest == profile.profile_digest,
            candidate.scientific_direct_requirements_digest
            == profile.direct_requirements_digest,
            candidate.scientific_complete_hash_lock_digest
            == profile.complete_hash_lock_digest,
            candidate.scientific_complete_hash_lock_dependency_count
            == profile.complete_hash_lock_dependency_count,
            profile.formal_ready is True,
            profile.readiness_blockers == (),
        )
    )


def _extract_validated_archive(archive_path: Path, root_path: Path) -> None:
    """把已通过 closure 检查的 outputs 成员原子恢复到本地工作区."""

    with ZipFile(archive_path) as archive:
        outputs_root = (root_path / "outputs").resolve()
        members: list[tuple[Any, Path, Path]] = []
        seen_destinations: set[Path] = set()
        for info in archive.infolist():
            member_name = info.filename.rstrip("/") if info.is_dir() else info.filename
            pure_name = PurePosixPath(member_name)
            if (
                not member_name.startswith("outputs/")
                or "\\" in member_name
                or pure_name.is_absolute()
                or any(part in {"", ".", ".."} for part in pure_name.parts)
                or pure_name.as_posix() != member_name
            ):
                raise ValueError(f"闭合包包含非法恢复路径: {info.filename}")
            destination = (root_path / member_name).resolve()
            destination.relative_to(outputs_root)
            if destination in seen_destinations:
                raise ValueError(f"闭合包包含重复恢复路径: {info.filename}")
            seen_destinations.add(destination)
            if info.is_dir():
                continue
            temporary_destination = destination.with_name(
                destination.name + ".partial"
            )
            members.append((info, destination, temporary_destination))
        try:
            for info, destination, temporary_destination in members:
                destination.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(info) as source, temporary_destination.open(
                    "wb"
                ) as target:
                    shutil.copyfileobj(source, target, length=8 * 1024 * 1024)
            for _, destination, temporary_destination in members:
                os.replace(temporary_destination, destination)
        except Exception:
            for _, _, temporary_destination in members:
                temporary_destination.unlink(missing_ok=True)
            raise


def _recover_closed_archives(
    *,
    root_path: Path,
    paper_run_name: str,
    target_fpr: float,
    randomization_repeat_id: str,
    expected_roles: set[str],
    archive_destination_dirs: Mapping[str, str | Path] | None,
) -> dict[str, Any]:
    """从外部持久目录恢复已闭合包, 不读取任何 progress 自声明."""

    if archive_destination_dirs is None:
        return {
            "recovered_roles": [],
            "local_archives": {},
            "mirrored_archives": {},
            "all_expected_roles_recovered": False,
            "supports_paper_claim": False,
        }
    recovered: dict[str, Any] = {}
    for role in sorted(expected_roles):
        destination_dir_value = archive_destination_dirs.get(role)
        if destination_dir_value is None:
            continue
        destination_dir = Path(destination_dir_value).expanduser()
        spec = _semantic_package_spec(role)
        matching_paths = (
            sorted(destination_dir.glob(spec.filename_pattern))
            if destination_dir.is_dir()
            else []
        )
        valid_candidates = []
        validation_errors = []
        for package_path in matching_paths:
            try:
                candidate = inspect_closure_package(
                    package_path,
                    spec=spec,
                    paper_run_name=paper_run_name,
                    target_fpr=target_fpr,
                    randomization_repeat_id=randomization_repeat_id,
                )
                if not _candidate_matches_repository(candidate, root_path):
                    raise ClosurePackageSelectionError(
                        "闭合包与当前仓库代码锁或依赖 profile 不一致"
                    )
                valid_candidates.append(candidate)
            except Exception as error:
                validation_errors.append(
                    f"{package_path.name}:{type(error).__name__}:{error}"
                )
        if matching_paths and not valid_candidates:
            raise RuntimeError(
                f"{role} 外部目录存在结果包但没有可恢复闭合包: "
                + ";".join(validation_errors[:3])
            )
        if not valid_candidates:
            continue
        latest_time = max(candidate.generated_at_utc for candidate in valid_candidates)
        latest = [
            candidate
            for candidate in valid_candidates
            if candidate.generated_at_utc == latest_time
        ]
        if len({candidate.package_sha256 for candidate in latest}) != 1:
            raise RuntimeError(f"{role} 外部目录存在同时间不同内容的闭合包")
        selected = min(
            latest,
            key=lambda candidate: candidate.package_path.as_posix(),
        )
        recovered[role] = {
            "candidate": selected,
            "mirrored_archive_path": selected.package_path,
            "archive_sha256": selected.package_sha256,
            "generated_at": selected.generated_at,
        }
    all_expected_roles_recovered = set(recovered) == expected_roles
    if all_expected_roles_recovered:
        local_dir = (
            root_path
            / "outputs"
            / "semantic_watermark_closed_archives"
            / paper_run_name
        )
        local_dir.mkdir(parents=True, exist_ok=True)
        for role, record in recovered.items():
            selected = record["candidate"]
            _extract_validated_archive(selected.package_path, root_path)
            local_path = local_dir / selected.package_path.name
            temporary_local_path = local_path.with_name(
                local_path.name + ".partial"
            )
            shutil.copyfile(selected.package_path, temporary_local_path)
            if file_sha256(temporary_local_path) != selected.package_sha256:
                temporary_local_path.unlink(missing_ok=True)
                raise RuntimeError(f"{role} 本地恢复副本摘要不一致")
            os.replace(temporary_local_path, local_path)
            record["local_archive_path"] = local_path
    return {
        "recovered_roles": sorted(recovered),
        "local_archives": {
            role: str(record["local_archive_path"])
            for role, record in recovered.items()
            if "local_archive_path" in record
        },
        "mirrored_archives": {
            role: str(record["mirrored_archive_path"])
            for role, record in recovered.items()
        },
        "closed_archive_records": [
            {
                "artifact_role": role,
                "local_archive_path": str(record.get("local_archive_path", "")),
                "mirrored_archive_path": str(record["mirrored_archive_path"]),
                "archive_sha256": record["archive_sha256"],
                "generated_at": record["generated_at"],
                "supports_paper_claim": False,
            }
            for role, record in sorted(recovered.items())
        ],
        "all_expected_roles_recovered": all_expected_roles_recovered,
        "supports_paper_claim": False,
    }


def _scientific_report_evidence(
    report: Mapping[str, Any],
    report_path: Path,
) -> dict[str, Any]:
    """提取所有会话返回分支共享的隔离执行证据."""

    return {
        "scientific_execution_report": dict(report),
        "scientific_execution_report_path": str(report_path),
        "scientific_execution_report_digest": file_sha256(report_path),
        "dependency_environment_report_path": str(
            report["dependency_environment_report_path"]
        ),
        "dependency_environment_report_digest": str(
            report["dependency_environment_report_digest"]
        ),
        "scientific_profile_id": str(report["profile_id"]),
        "scientific_profile_digest": str(report["profile_digest"]),
        "scientific_complete_hash_lock_digest": str(
            report["complete_hash_lock_digest"]
        ),
    }


def _write_bindings(
    *,
    root_path: Path,
    paper_run_name: str,
    execution_report_path: Path,
    dispatch_report_path: Path,
    include_formal_ablation: bool,
) -> dict[str, dict[str, Any]]:
    """为已闭合的主结果, 质量结果和可选消融写入独立执行绑定."""

    specifications = [
        (
            "image_only_dataset_runtime",
            root_path / "outputs" / "image_only_dataset_runtime" / paper_run_name,
            "dataset_runtime_summary.json",
        ),
        (
            "dataset_level_quality",
            root_path / "outputs" / "dataset_level_quality" / paper_run_name,
            "dataset_quality_summary.json",
        ),
    ]
    if include_formal_ablation:
        specifications.extend(
            (
                (
                    "runtime_rerun_ablation",
                    root_path
                    / "outputs"
                    / "formal_mechanism_ablation"
                    / paper_run_name,
                    "ablation_component_summary.json",
                ),
                (
                    "branch_risk_parameter_sensitivity",
                    root_path
                    / "outputs"
                    / "formal_branch_risk_sensitivity"
                    / paper_run_name,
                    "parameter_sensitivity_summary.json",
                ),
            )
        )
    bindings: dict[str, dict[str, Any]] = {}
    for artifact_role, artifact_dir, summary_file_name in specifications:
        _require_repeat_component_ready(
            _read_json(artifact_dir / summary_file_name),
            artifact_role=artifact_role,
        )
        binding, binding_path = write_scientific_execution_binding(
            artifact_dir,
            artifact_role=artifact_role,
            paper_run_name=paper_run_name,
            summary_file_name=summary_file_name,
            manifest_file_name="manifest.local.json",
            execution_report_path=execution_report_path,
            dispatch_report_path=dispatch_report_path,
            expected_profile_id=SCIENTIFIC_PROFILE_ID,
            repository_root=root_path,
        )
        bindings[artifact_role] = {
            "binding_path": str(binding_path),
            "binding_digest": file_sha256(binding_path),
            "binding": binding,
        }
    return bindings


def _run_bound_packaging(
    *,
    root_path: Path,
    paper_run_name: str,
    execution_report: Mapping[str, Any],
    include_formal_ablation: bool,
) -> dict[str, Any]:
    """复用已验证科学解释器重新打包, 不再次准备隔离依赖环境."""

    python_path = Path(str(execution_report["python_executable_path"])).resolve()
    if not python_path.is_file() or file_sha256(python_path) != execution_report.get(
        "python_executable_sha256"
    ):
        raise RuntimeError("重新打包前隔离科学解释器身份发生漂移")
    dependency_path = Path(
        str(execution_report["dependency_environment_report_path"])
    ).resolve()
    if not dependency_path.is_file() or file_sha256(
        dependency_path
    ) != execution_report.get("dependency_environment_report_digest"):
        raise RuntimeError("重新打包前隔离依赖环境报告发生漂移")
    current_lock = require_published_formal_execution_lock(root_path)
    if current_lock != execution_report.get("formal_execution_lock"):
        raise RuntimeError("重新打包前正式执行锁发生漂移")

    argv = [
        str(python_path),
        "-m",
        SCIENTIFIC_DISPATCH_MODULE,
        "--package-bound-outputs",
    ]
    if include_formal_ablation:
        argv.append("--include-formal-ablation")
    environment = os.environ.copy()
    execution = execution_report.get("execution", {})
    if isinstance(execution, Mapping):
        overrides = execution.get("environment_overrides", {})
        if isinstance(overrides, Mapping):
            environment.update({str(key): str(value) for key, value in overrides.items()})
    completed = subprocess.run(
        argv,
        cwd=root_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        shell=False,
    )
    record: dict[str, Any] = {
        "argv": argv,
        "return_code": int(completed.returncode),
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "decision": "pass" if completed.returncode == 0 else "fail",
        "supports_paper_claim": False,
    }
    report_path = (
        root_path
        / "outputs"
        / "scientific_command_execution"
        / paper_run_name
        / "bound_packaging_report.json"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    record["report_path"] = str(report_path)
    record["report_digest"] = file_sha256(report_path)
    if completed.returncode != 0:
        raise RuntimeError(
            "科学执行证据重新打包失败: " + (completed.stderr.strip() or "未知错误")
        )
    output_lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if not output_lines:
        raise RuntimeError("重新打包命令没有返回受治理结果")
    try:
        packaging_result = json.loads(output_lines[-1])
    except json.JSONDecodeError as exc:
        raise RuntimeError("重新打包命令返回值不是有效 JSON") from exc
    if not isinstance(packaging_result, dict) or packaging_result.get("decision") != "pass":
        raise RuntimeError("重新打包命令没有通过结果门禁")
    record["packaging_result"] = packaging_result
    return record


def _archive_paths_from_packaging(
    root_path: Path,
    packaging_execution: Mapping[str, Any],
    *,
    expected_roles: set[str],
) -> dict[str, Path]:
    """解析唯一角色到结果包映射并要求角色集合与当前请求严格一致."""

    result = packaging_execution.get("packaging_result", {})
    archives = result.get("archives", []) if isinstance(result, Mapping) else []
    resolved: dict[str, Path] = {}
    for record in archives:
        if not isinstance(record, Mapping):
            continue
        role = str(record.get("artifact_role", ""))
        if not role or role in resolved:
            raise RuntimeError(f"重新打包结果包含空角色或重复角色: {role}")
        path = (root_path / str(record.get("archive_path", ""))).resolve()
        path.relative_to(root_path)
        if not path.is_file() or file_sha256(path) != record.get("archive_sha256"):
            raise RuntimeError(f"重新打包结果文件或摘要无效: {role}")
        resolved[role] = path
    if set(resolved) != expected_roles:
        missing_roles = sorted(expected_roles - set(resolved))
        unexpected_roles = sorted(set(resolved) - expected_roles)
        raise RuntimeError(
            "重新打包结果角色集合不一致: "
            f"missing={missing_roles},unexpected={unexpected_roles}"
        )
    return resolved


def _validate_packaged_archives(
    archives: Mapping[str, Path],
    *,
    root_path: Path,
    paper_run_name: str,
    target_fpr: float,
    randomization_repeat_id: str,
) -> None:
    """以闭合包生产检查器复验归档, 不信任打包命令的自报状态."""

    for artifact_role, archive_path in archives.items():
        candidate = inspect_closure_package(
            archive_path,
            spec=_semantic_package_spec(artifact_role),
            paper_run_name=paper_run_name,
            target_fpr=target_fpr,
            randomization_repeat_id=randomization_repeat_id,
        )
        if not _candidate_matches_repository(candidate, root_path):
            raise RuntimeError(
                f"{artifact_role} 结果包与当前代码锁或科学依赖身份不一致"
            )


def run_semantic_watermark_image_only_session(
    root: str | Path = ".",
    *,
    run_formal_ablation: bool = False,
    archive_destination_dirs: Mapping[str, str | Path] | None = None,
    resume_checkpoint_dir: str | Path | None = None,
) -> dict[str, Any]:
    """执行一次可中断, 可恢复且只准备一个科学子环境的 GPU 会话."""

    root_path = Path(root).resolve()
    paper_run = build_paper_run_config(root_path)
    paper_run_name = paper_run.run_name
    content_strength_candidate_role = _content_strength_candidate_role()
    if content_strength_candidate_role is not None and run_formal_ablation:
        raise RuntimeError("内容倍率候选不得启动正式消融")
    if resume_checkpoint_dir is not None:
        os.environ["SLM_WM_RESUME_CHECKPOINT_DIR"] = str(
            Path(resume_checkpoint_dir).expanduser().resolve()
        )
    expected_archive_roles = {
        "image_only_dataset_runtime",
        "dataset_level_quality",
    }
    if run_formal_ablation:
        expected_archive_roles.update(
            {
                "runtime_rerun_ablation",
                "branch_risk_parameter_sensitivity",
            }
        )
    closed_archive_recovery = _recover_closed_archives(
        root_path=root_path,
        paper_run_name=paper_run_name,
        target_fpr=paper_run.target_fpr,
        randomization_repeat_id=paper_run.randomization_repeat_id,
        expected_roles=expected_archive_roles,
        archive_destination_dirs=archive_destination_dirs,
    )
    if closed_archive_recovery["all_expected_roles_recovered"] is True:
        return {
            "workflow_decision": "closed_archives_recovered",
            "workflow_completion_state": "repeat_component_complete",
            "session_execution_decision": "pass",
            "paper_run_closed": False,
            "result_closure_ready": False,
            "paper_run_name": paper_run_name,
            "active_workflow": (
                "branch_risk_parameter_sensitivity"
                if run_formal_ablation
                else "image_only_dataset_runtime"
            ),
            "formal_ablation_requested": run_formal_ablation,
            "closed_archive_recovery_ready": True,
            "repeat_component_ready": True,
            **closed_archive_recovery,
        }
    runtime_output_dir = _candidate_scoped_directory(
        root_path / "outputs" / "image_only_dataset_runtime" / paper_run_name,
        content_strength_candidate_role,
    )
    quality_output_dir = root_path / "outputs" / "dataset_level_quality" / paper_run_name
    ablation_output_dir = root_path / "outputs" / "formal_mechanism_ablation" / paper_run_name
    sensitivity_output_dir = (
        root_path
        / "outputs"
        / "formal_branch_risk_sensitivity"
        / paper_run_name
    )
    execution_report_dir = (
        root_path
        / "outputs"
        / "isolated_scientific_execution"
        / SCIENTIFIC_PROFILE_ID
        / paper_run_name
    )
    execution_report_path = (
        _candidate_scoped_directory(
            execution_report_dir,
            content_strength_candidate_role,
        )
        / "semantic_watermark_image_only_session.json"
    )
    child_argv_tail = ["-m", SCIENTIFIC_DISPATCH_MODULE]
    if run_formal_ablation:
        child_argv_tail.append("--run-formal-ablation")
    execution_report, resolved_execution_report_path = (
        execute_isolated_scientific_command(
            SCIENTIFIC_PROFILE_ID,
            child_argv_tail,
            execution_report_path=execution_report_path,
            repository_root=root_path,
        )
    )
    if execution_report.get("decision") != "pass":
        raise RuntimeError(
            "隔离科学子解释器执行失败: "
            + ",".join(str(value) for value in execution_report.get("failure_reasons", []))
        )
    execution_report = validate_scientific_execution_report(
        resolved_execution_report_path,
        expected_profile_id=SCIENTIFIC_PROFILE_ID,
    )
    evidence = _scientific_report_evidence(
        execution_report,
        resolved_execution_report_path,
    )
    dispatch_report_dir = (
        root_path
        / "outputs"
        / "scientific_command_execution"
        / paper_run_name
    )
    dispatch_report_path = (
        _candidate_scoped_directory(
            dispatch_report_dir,
            content_strength_candidate_role,
        )
        / DISPATCH_REPORT_FILE_NAME
    )
    dispatch_report = _read_json(dispatch_report_path)
    if dispatch_report.get("decision") != "pass":
        raise RuntimeError("科学子进程dispatch报告未通过")
    if dispatch_report.get("content_strength_candidate_role", "") != (
        content_strength_candidate_role or ""
    ):
        raise RuntimeError("科学子进程dispatch候选角色漂移")
    artifact_state = dispatch_report.get("artifact_state")
    if not isinstance(artifact_state, dict):
        raise RuntimeError("科学子进程dispatch缺少artifact_state")
    if artifact_state.get("content_strength_candidate_role", "") != (
        content_strength_candidate_role or ""
    ):
        raise RuntimeError("科学子进程artifact候选角色漂移")
    expected_runtime_progress_path = (
        runtime_output_dir / "dataset_runtime_progress.json"
    ).relative_to(root_path).as_posix()
    expected_calibration_summary_path = (
        runtime_output_dir / "calibration_protocol_summary.json"
    ).relative_to(root_path).as_posix()
    if (
        artifact_state.get("runtime_progress_path")
        != expected_runtime_progress_path
        or artifact_state.get("calibration_summary_path")
        != expected_calibration_summary_path
    ):
        raise RuntimeError("科学子进程artifact路径未绑定当前候选")

    if artifact_state.get("calibration_summary_present") is True:
        calibration_summary = _read_json(
            root_path / expected_calibration_summary_path
        )
        if calibration_summary.get("protocol_decision") != "calibration_complete":
            raise RuntimeError("calibration-only intermediate result is invalid")
        return {
            "workflow_decision": "calibration_complete",
            "workflow_completion_state": "calibration_complete",
            "session_execution_decision": "pass",
            "paper_run_closed": False,
            "result_closure_ready": False,
            "paper_run_name": paper_run_name,
            "active_workflow": "image_only_dataset_runtime",
            "calibration_protocol_summary": calibration_summary,
            "content_strength_candidate_role": (
                content_strength_candidate_role or ""
            ),
            "formal_ablation_requested": run_formal_ablation,
            "closed_archive_recovery_ready": False,
            "closed_archive_recovery": closed_archive_recovery,
            "supports_paper_claim": False,
            **evidence,
        }
    if artifact_state.get("runtime_progress_present") is True:
        return {
            "workflow_decision": "resume_required",
            "workflow_completion_state": "resume_required",
            "session_execution_decision": "pass",
            "paper_run_closed": False,
            "result_closure_ready": False,
            "paper_run_name": paper_run_name,
            "active_workflow": "image_only_dataset_runtime",
            "runtime_progress": _read_json(
                root_path / expected_runtime_progress_path
            ),
            "content_strength_candidate_role": (
                content_strength_candidate_role or ""
            ),
            "formal_ablation_requested": run_formal_ablation,
            "closed_archive_recovery_ready": False,
            "closed_archive_recovery": closed_archive_recovery,
            "supports_paper_claim": False,
            **evidence,
        }

    runtime_summary = _read_json(runtime_output_dir / "dataset_runtime_summary.json")
    quality_summary = _read_json(quality_output_dir / "dataset_quality_summary.json")
    if runtime_summary.get("protocol_decision") != "pass":
        raise RuntimeError("仅图像数据集运行未生成通过协议的正式摘要")
    if quality_summary.get("formal_fid_kid_component_ready") is not True:
        raise RuntimeError("数据集运行完成, 但规范 Inception FID/KID 尚未闭合")
    _require_repeat_component_ready(
        runtime_summary,
        artifact_role="image_only_dataset_runtime",
    )
    _require_repeat_component_ready(
        quality_summary,
        artifact_role="dataset_level_quality",
    )

    ablation_progress_path = ablation_output_dir / "runtime_rerun_progress.json"
    ablation_complete = False
    ablation_summary: dict[str, Any] = {}
    if run_formal_ablation and ablation_progress_path.is_file():
        return {
            "workflow_decision": "resume_required",
            "workflow_completion_state": "resume_required",
            "session_execution_decision": "pass",
            "paper_run_closed": False,
            "result_closure_ready": False,
            "paper_run_name": paper_run_name,
            "active_workflow": "runtime_rerun_ablation",
            "runtime_summary": runtime_summary,
            "quality_summary": quality_summary,
            "ablation_progress": _read_json(ablation_progress_path),
            "formal_ablation_requested": True,
            "closed_archive_recovery_ready": False,
            "closed_archive_recovery": closed_archive_recovery,
            "supports_paper_claim": False,
            **evidence,
        }
    if run_formal_ablation and not ablation_progress_path.is_file():
        ablation_summary = _read_json(ablation_output_dir / "ablation_component_summary.json")
        if ablation_summary.get("protocol_decision") != "pass":
            raise RuntimeError("正式机制消融未生成通过协议的摘要")
        _require_repeat_component_ready(
            ablation_summary,
            artifact_role="runtime_rerun_ablation",
        )
        ablation_complete = True

    sensitivity_progress_path = (
        sensitivity_output_dir / "parameter_sensitivity_progress.json"
    )
    sensitivity_complete = False
    sensitivity_summary: dict[str, Any] = {}
    if run_formal_ablation and sensitivity_progress_path.is_file():
        return {
            "workflow_decision": "resume_required",
            "workflow_completion_state": "resume_required",
            "session_execution_decision": "pass",
            "paper_run_closed": False,
            "result_closure_ready": False,
            "paper_run_name": paper_run_name,
            "active_workflow": "branch_risk_parameter_sensitivity",
            "runtime_summary": runtime_summary,
            "quality_summary": quality_summary,
            "ablation_summary": ablation_summary,
            "sensitivity_progress": _read_json(sensitivity_progress_path),
            "formal_ablation_requested": True,
            "closed_archive_recovery_ready": False,
            "closed_archive_recovery": closed_archive_recovery,
            "supports_paper_claim": False,
            **evidence,
        }
    if run_formal_ablation and not sensitivity_progress_path.is_file():
        sensitivity_summary = _read_json(
            sensitivity_output_dir / "parameter_sensitivity_summary.json"
        )
        if sensitivity_summary.get("protocol_decision") != "pass":
            raise RuntimeError("单模型风险参数敏感性未生成通过协议的摘要")
        _require_repeat_component_ready(
            sensitivity_summary,
            artifact_role="branch_risk_parameter_sensitivity",
        )
        sensitivity_complete = True

    bindings = _write_bindings(
        root_path=root_path,
        paper_run_name=paper_run_name,
        execution_report_path=resolved_execution_report_path,
        dispatch_report_path=dispatch_report_path,
        include_formal_ablation=ablation_complete and sensitivity_complete,
    )
    packaging_execution = _run_bound_packaging(
        root_path=root_path,
        paper_run_name=paper_run_name,
        execution_report=execution_report,
        include_formal_ablation=ablation_complete and sensitivity_complete,
    )
    expected_packaged_roles = {
        "image_only_dataset_runtime",
        "dataset_level_quality",
    }
    if ablation_complete:
        expected_packaged_roles.update(
            {
                "runtime_rerun_ablation",
                "branch_risk_parameter_sensitivity",
            }
        )
    archives = _archive_paths_from_packaging(
        root_path,
        packaging_execution,
        expected_roles=expected_packaged_roles,
    )
    _validate_packaged_archives(
        archives,
        root_path=root_path,
        paper_run_name=paper_run_name,
        target_fpr=paper_run.target_fpr,
        randomization_repeat_id=paper_run.randomization_repeat_id,
    )
    local_archives = {role: str(path) for role, path in archives.items()}
    mirrored_archives: dict[str, str] = {}
    if archive_destination_dirs is not None:
        missing_destination_roles = set(archives) - set(archive_destination_dirs)
        if missing_destination_roles:
            raise ValueError(
                "已闭合产物缺少归档目标目录: "
                + ",".join(sorted(missing_destination_roles))
            )
        mirrored_archives = {
            role: _mirror_archive(path, Path(archive_destination_dirs[role]))
            for role, path in archives.items()
        }
    common_result = {
        "paper_run_name": paper_run_name,
        "runtime_summary": runtime_summary,
        "quality_summary": quality_summary,
        "local_archives": local_archives,
        "mirrored_archives": mirrored_archives,
        "scientific_execution_bindings": bindings,
        "bound_packaging_execution": packaging_execution,
        "closed_archive_recovery": closed_archive_recovery,
        **evidence,
    }
    if not run_formal_ablation:
        return {
            "workflow_decision": "dataset_complete",
            "workflow_completion_state": "dataset_component_complete",
            "session_execution_decision": "pass",
            "paper_run_closed": False,
            "result_closure_ready": False,
            "active_workflow": "image_only_dataset_runtime",
            "formal_ablation_requested": False,
            "repeat_component_ready": True,
            "supports_paper_claim": False,
            **common_result,
        }
    return {
        "workflow_decision": "complete",
        "workflow_completion_state": "repeat_component_complete",
        "session_execution_decision": "pass",
        "paper_run_closed": False,
        "result_closure_ready": False,
        "active_workflow": "branch_risk_parameter_sensitivity",
        "ablation_summary": ablation_summary,
        "sensitivity_summary": sensitivity_summary,
        "formal_ablation_requested": True,
        "repeat_component_ready": True,
        "randomization_aggregate_ready": False,
        "supports_paper_claim": False,
        **common_result,
    }
