"""调度真实科学算子,仅图像检测和正式消融的 GPU 会话.

Notebook 所在解释器只运行 workflow 编排.主方法,规范 Inception 质量评估和
正式消融由一次隔离环境准备得到的 ``sd35_method_runtime_gpu`` 子解释器完成.
本模块不依赖 Notebook.它判断续跑状态,绑定执行证据并重新打包; 可选的
归档目标目录由更外层 Colab 包装显式注入.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any, Mapping

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


SCIENTIFIC_PROFILE_ID = "sd35_method_runtime_gpu"
SCIENTIFIC_DISPATCH_MODULE = (
    "experiments.runtime.semantic_watermark_scientific_session"
)


def _read_json(path: Path) -> dict[str, Any]:
    """读取一个 JSON 对象, 缺失或格式不符时返回空对象."""

    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _mirror_archive(source_path: Path, destination_dir: Path) -> str:
    """把本次重新生成的确定结果包复制到 Drive 受治理目录."""

    if not source_path.is_file():
        raise FileNotFoundError(f"缺少待镜像结果包: {source_path}")
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / source_path.name
    shutil.copy2(source_path, destination)
    return str(destination)


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
    """为已闭合的主结果,质量结果和可选消融写入独立执行绑定."""

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
        specifications.append(
            (
                "runtime_rerun_ablation",
                root_path
                / "outputs"
                / "formal_mechanism_ablation"
                / paper_run_name,
                "ablation_claim_summary.json",
            )
        )
    bindings: dict[str, dict[str, Any]] = {}
    for artifact_role, artifact_dir, summary_file_name in specifications:
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


def run_semantic_watermark_image_only_session(
    root: str | Path = ".",
    *,
    run_formal_ablation: bool = False,
    archive_destination_dirs: Mapping[str, str | Path] | None = None,
) -> dict[str, Any]:
    """执行一次可中断,可恢复且只准备一个科学子环境的 GPU 会话."""

    root_path = Path(root).resolve()
    paper_run_name = os.environ.get("SLM_WM_PAPER_RUN_NAME", "pilot_paper")
    runtime_output_dir = root_path / "outputs" / "image_only_dataset_runtime" / paper_run_name
    quality_output_dir = root_path / "outputs" / "dataset_level_quality" / paper_run_name
    ablation_output_dir = root_path / "outputs" / "formal_mechanism_ablation" / paper_run_name
    execution_report_path = (
        root_path
        / "outputs"
        / "isolated_scientific_execution"
        / SCIENTIFIC_PROFILE_ID
        / paper_run_name
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
    dispatch_report_path = (
        root_path
        / "outputs"
        / "scientific_command_execution"
        / paper_run_name
        / DISPATCH_REPORT_FILE_NAME
    )

    runtime_progress_path = runtime_output_dir / "dataset_runtime_progress.json"
    if runtime_progress_path.is_file():
        return {
            "workflow_decision": "resume_required",
            "paper_run_name": paper_run_name,
            "active_workflow": "image_only_dataset_runtime",
            "runtime_progress": _read_json(runtime_progress_path),
            "formal_ablation_requested": run_formal_ablation,
            "supports_paper_claim": False,
            **evidence,
        }

    runtime_summary = _read_json(runtime_output_dir / "dataset_runtime_summary.json")
    quality_summary = _read_json(quality_output_dir / "dataset_quality_summary.json")
    if runtime_summary.get("protocol_decision") != "pass":
        raise RuntimeError("仅图像数据集运行未生成通过协议的正式摘要")
    if quality_summary.get("formal_fid_kid_claim_gate_ready") is not True:
        raise RuntimeError("数据集运行完成, 但规范 Inception FID/KID 尚未闭合")

    ablation_progress_path = ablation_output_dir / "runtime_rerun_progress.json"
    ablation_complete = False
    ablation_summary: dict[str, Any] = {}
    if run_formal_ablation and ablation_progress_path.is_file():
        return {
            "workflow_decision": "resume_required",
            "paper_run_name": paper_run_name,
            "active_workflow": "runtime_rerun_ablation",
            "runtime_summary": runtime_summary,
            "quality_summary": quality_summary,
            "ablation_progress": _read_json(ablation_progress_path),
            "formal_ablation_requested": True,
            "supports_paper_claim": False,
            **evidence,
        }
    if run_formal_ablation and not ablation_progress_path.is_file():
        ablation_summary = _read_json(ablation_output_dir / "ablation_claim_summary.json")
        if ablation_summary.get("protocol_decision") != "pass":
            raise RuntimeError("正式机制消融未生成通过协议的摘要")
        ablation_complete = True

    bindings = _write_bindings(
        root_path=root_path,
        paper_run_name=paper_run_name,
        execution_report_path=resolved_execution_report_path,
        dispatch_report_path=dispatch_report_path,
        include_formal_ablation=ablation_complete,
    )
    packaging_execution = _run_bound_packaging(
        root_path=root_path,
        paper_run_name=paper_run_name,
        execution_report=execution_report,
        include_formal_ablation=ablation_complete,
    )
    expected_archive_roles = {
        "image_only_dataset_runtime",
        "dataset_level_quality",
    }
    if ablation_complete:
        expected_archive_roles.add("runtime_rerun_ablation")
    archives = _archive_paths_from_packaging(
        root_path,
        packaging_execution,
        expected_roles=expected_archive_roles,
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
        **evidence,
    }
    if not run_formal_ablation:
        return {
            "workflow_decision": "dataset_complete",
            "active_workflow": "image_only_dataset_runtime",
            "formal_ablation_requested": False,
            "supports_paper_claim": False,
            **common_result,
        }
    return {
        "workflow_decision": "complete",
        "active_workflow": "runtime_rerun_ablation",
        "ablation_summary": ablation_summary,
        "formal_ablation_requested": True,
        "supports_paper_claim": bool(
            runtime_summary.get("supports_paper_claim", False)
            and quality_summary.get("supports_paper_claim", False)
            and ablation_summary.get("supports_paper_claim", False)
        ),
        **common_result,
    }
