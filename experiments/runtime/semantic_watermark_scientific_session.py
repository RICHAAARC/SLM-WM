"""在同一受验证科学解释器中编排主方法,质量评估和正式消融.

该脚本是可脱离 Notebook 执行的薄命令层.默认模式只运行科学命令并保存逐命令
证据; ``--package-bound-outputs`` 模式只验证外层已经写入的执行绑定并重新打包,
不会再次运行模型或消融.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Sequence
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.runtime.scientific_execution_binding import (
    BINDING_FILE_NAME,
    DEPENDENCY_REPORT_FILE_NAME,
    DISPATCH_REPORT_FILE_NAME,
    EXECUTION_REPORT_FILE_NAME,
    scientific_manifest_payload_digest,
    validate_scientific_execution_binding,
)


DISPATCH_REPORT_SCHEMA = "scientific_command_dispatch_report"
DISPATCH_REPORT_SCHEMA_VERSION = 1
SEMANTIC_ARTIFACT_SPECS = {
    "image_only_dataset_runtime": (
        "outputs/image_only_dataset_runtime/{paper_run_name}",
        "dataset_runtime_summary.json",
    ),
    "dataset_level_quality": (
        "outputs/dataset_level_quality/{paper_run_name}",
        "dataset_quality_summary.json",
    ),
    "runtime_rerun_ablation": (
        "outputs/formal_mechanism_ablation/{paper_run_name}",
        "ablation_claim_summary.json",
    ),
}


def _file_sha256(path: Path) -> str:
    """计算命令输出文件摘要, 避免调度报告只记录不稳定路径."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    """读取必要 JSON 对象, 缺失或根节点错误时直接失败."""

    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON 根节点必须是对象: {path}")
    return payload


def _run_child(command_tail: Sequence[str]) -> dict[str, Any]:
    """用当前隔离解释器执行一个仓库命令并返回独立进程证据."""

    argv = [sys.executable, *command_tail]
    environment = os.environ.copy()
    environment["SLM_WM_DEFER_SCIENTIFIC_PACKAGING"] = "1"
    completed = subprocess.run(
        argv,
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        shell=False,
    )
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, end="", file=sys.stderr)
    return {
        "argv": argv,
        "return_code": int(completed.returncode),
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "packaging_deferred": True,
    }


def _artifact_state(paper_run_name: str) -> dict[str, Any]:
    """读取主方法,质量评估和消融的当前完成或续跑状态."""

    runtime_dir = ROOT / "outputs" / "image_only_dataset_runtime" / paper_run_name
    quality_dir = ROOT / "outputs" / "dataset_level_quality" / paper_run_name
    ablation_dir = ROOT / "outputs" / "formal_mechanism_ablation" / paper_run_name
    runtime_progress_path = runtime_dir / "dataset_runtime_progress.json"
    ablation_progress_path = ablation_dir / "runtime_rerun_progress.json"
    return {
        "runtime_progress_present": runtime_progress_path.is_file(),
        "runtime_progress_path": runtime_progress_path.relative_to(ROOT).as_posix(),
        "runtime_summary_path": (
            runtime_dir / "dataset_runtime_summary.json"
        ).relative_to(ROOT).as_posix(),
        "quality_summary_path": (
            quality_dir / "dataset_quality_summary.json"
        ).relative_to(ROOT).as_posix(),
        "ablation_progress_present": ablation_progress_path.is_file(),
        "ablation_progress_path": ablation_progress_path.relative_to(ROOT).as_posix(),
        "ablation_summary_path": (
            ablation_dir / "ablation_claim_summary.json"
        ).relative_to(ROOT).as_posix(),
    }


def _write_dispatch_report(report_path: Path, report: dict[str, Any]) -> None:
    """稳定写出科学命令调度证据."""

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _closed_artifact_record(
    artifact_role: str,
    paper_run_name: str,
) -> dict[str, Any]:
    """把本会话确认闭合的 summary 与 manifest 写成角色化摘要记录."""

    directory_template, summary_name = SEMANTIC_ARTIFACT_SPECS[artifact_role]
    artifact_dir = ROOT / directory_template.format(paper_run_name=paper_run_name)
    summary_path = artifact_dir / summary_name
    manifest_path = artifact_dir / "manifest.local.json"
    summary = _read_json(summary_path)
    manifest = _read_json(manifest_path)
    formal_execution_run_lock = manifest.get("formal_execution_run_lock")
    if not isinstance(formal_execution_run_lock, dict):
        raise RuntimeError(f"{artifact_role} manifest 缺少正式执行锁")
    return {
        "artifact_role": artifact_role,
        "summary_path": summary_path.relative_to(ROOT).as_posix(),
        "summary_sha256": _file_sha256(summary_path),
        "manifest_path": manifest_path.relative_to(ROOT).as_posix(),
        "manifest_sha256_at_session": _file_sha256(manifest_path),
        "manifest_scientific_digest": scientific_manifest_payload_digest(
            manifest
        ),
        "formal_execution_run_lock": formal_execution_run_lock,
        "summary_protocol_decision": summary.get(
            "protocol_decision",
            "pass"
            if artifact_role == "dataset_level_quality"
            and summary.get("formal_fid_kid_claim_gate_ready") is True
            else "",
        ),
    }


def run_scientific_commands(*, run_formal_ablation: bool) -> dict[str, Any]:
    """顺序运行主方法, 并仅在主证据闭合后运行正式消融."""

    paper_run_name = os.environ.get("SLM_WM_PAPER_RUN_NAME", "pilot_paper")
    report_path = (
        ROOT
        / "outputs"
        / "scientific_command_execution"
        / paper_run_name
        / DISPATCH_REPORT_FILE_NAME
    )
    report: dict[str, Any] = {
        "report_schema": DISPATCH_REPORT_SCHEMA,
        "schema_version": DISPATCH_REPORT_SCHEMA_VERSION,
        "paper_run_name": paper_run_name,
        "formal_ablation_requested": run_formal_ablation,
        "packaging_deferred": True,
        "python_executable": sys.executable,
        "commands": [],
        "artifact_state": {},
        "artifact_records": [],
        "artifact_validation_mode": "completed_or_revalidated_in_current_session",
        "decision": "fail",
        "failure_reasons": [],
        "supports_paper_claim": False,
    }
    try:
        runtime_command = _run_child(
            ("-m", "experiments.runners.image_only_dataset_workload")
        )
        report["commands"].append(
            {"command_role": "image_only_dataset_runtime", **runtime_command}
        )
        if runtime_command["return_code"] != 0:
            raise RuntimeError("image_only_dataset_runtime_command_failed")

        state = _artifact_state(paper_run_name)
        report["artifact_state"] = state
        if state["runtime_progress_present"]:
            report["decision"] = "pass"
            _write_dispatch_report(report_path, report)
            return report

        runtime_summary = _read_json(ROOT / state["runtime_summary_path"])
        quality_summary = _read_json(ROOT / state["quality_summary_path"])
        if runtime_summary.get("protocol_decision") != "pass":
            raise RuntimeError("image_only_dataset_runtime_not_closed")
        if quality_summary.get("formal_fid_kid_claim_gate_ready") is not True:
            raise RuntimeError("dataset_level_quality_not_closed")
        closed_roles = [
            "image_only_dataset_runtime",
            "dataset_level_quality",
        ]

        if run_formal_ablation:
            ablation_command = _run_child(
                ("-m", "experiments.ablations.mechanism_ablation_workload")
            )
            report["commands"].append(
                {"command_role": "runtime_rerun_ablation", **ablation_command}
            )
            if ablation_command["return_code"] != 0:
                raise RuntimeError("runtime_rerun_ablation_command_failed")
            state = _artifact_state(paper_run_name)
            report["artifact_state"] = state
            if not state["ablation_progress_present"]:
                ablation_summary = _read_json(ROOT / state["ablation_summary_path"])
                if ablation_summary.get("protocol_decision") != "pass":
                    raise RuntimeError("runtime_rerun_ablation_not_closed")
                closed_roles.append("runtime_rerun_ablation")

        report["artifact_records"] = [
            _closed_artifact_record(role, paper_run_name)
            for role in closed_roles
        ]

        report["decision"] = "pass"
    except Exception as exc:
        report["failure_reasons"] = [str(exc)]
        report["decision"] = "fail"
        _write_dispatch_report(report_path, report)
        raise
    _write_dispatch_report(report_path, report)
    return report


def _assert_archive_contains_binding(archive_path: Path, artifact_dir: Path) -> None:
    """确认新结果包显式包含三类科学证据和独立绑定."""

    required_names = {
        BINDING_FILE_NAME,
        EXECUTION_REPORT_FILE_NAME,
        DEPENDENCY_REPORT_FILE_NAME,
        DISPATCH_REPORT_FILE_NAME,
    }
    prefix = artifact_dir.relative_to(ROOT).as_posix() + "/"
    with ZipFile(archive_path) as archive:
        names = set(archive.namelist())
    missing = {prefix + name for name in required_names} - names
    if missing:
        raise RuntimeError(f"重新打包结果缺少科学执行证据: {sorted(missing)}")


def package_bound_outputs(*, include_formal_ablation: bool) -> dict[str, Any]:
    """验证外层绑定后重新生成包含执行证据的正式结果包."""

    from experiments.ablations.runtime_rerun import package_runtime_rerun_ablations
    from experiments.artifacts.dataset_level_quality_outputs import (
        package_dataset_level_quality_outputs,
    )
    from experiments.runners.image_only_dataset_runtime import (
        package_image_only_dataset_runtime,
    )

    paper_run_name = os.environ.get("SLM_WM_PAPER_RUN_NAME", "pilot_paper")
    specifications = [
        (
            "image_only_dataset_runtime",
            ROOT / "outputs" / "image_only_dataset_runtime" / paper_run_name,
            package_image_only_dataset_runtime,
        ),
        (
            "dataset_level_quality",
            ROOT / "outputs" / "dataset_level_quality" / paper_run_name,
            package_dataset_level_quality_outputs,
        ),
    ]
    if include_formal_ablation:
        specifications.append(
            (
                "runtime_rerun_ablation",
                ROOT / "outputs" / "formal_mechanism_ablation" / paper_run_name,
                package_runtime_rerun_ablations,
            )
        )

    archive_records = []
    for artifact_role, artifact_dir, package_function in specifications:
        validate_scientific_execution_binding(
            artifact_dir / BINDING_FILE_NAME,
            expected_artifact_role=artifact_role,
            expected_paper_run_name=paper_run_name,
            repository_root=ROOT,
        )
        archive_path = package_function(paper_run_name, root=ROOT)
        # 打包函数可能规范化 manifest; 重新校验可阻止绑定摘要悄然失效.
        validate_scientific_execution_binding(
            artifact_dir / BINDING_FILE_NAME,
            expected_artifact_role=artifact_role,
            expected_paper_run_name=paper_run_name,
            repository_root=ROOT,
        )
        _assert_archive_contains_binding(archive_path, artifact_dir)
        archive_records.append(
            {
                "artifact_role": artifact_role,
                "archive_path": archive_path.relative_to(ROOT).as_posix(),
                "archive_sha256": _file_sha256(archive_path),
            }
        )
    return {
        "paper_run_name": paper_run_name,
        "include_formal_ablation": include_formal_ablation,
        "archives": archive_records,
        "decision": "pass",
        "supports_paper_claim": False,
    }


def _parse_arguments() -> argparse.Namespace:
    """解析科学运行与绑定打包两种互斥模式."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--run-formal-ablation", action="store_true")
    parser.add_argument("--package-bound-outputs", action="store_true")
    parser.add_argument("--include-formal-ablation", action="store_true")
    arguments = parser.parse_args()
    if arguments.include_formal_ablation and not arguments.package_bound_outputs:
        parser.error("--include-formal-ablation 只能与 --package-bound-outputs 一起使用")
    if arguments.run_formal_ablation and arguments.package_bound_outputs:
        parser.error("科学运行与绑定打包模式不能同时启用")
    return arguments


def main() -> None:
    """命令行入口."""

    arguments = _parse_arguments()
    if arguments.package_bound_outputs:
        result = package_bound_outputs(
            include_formal_ablation=arguments.include_formal_ablation
        )
    else:
        result = run_scientific_commands(
            run_formal_ablation=arguments.run_formal_ablation
        )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
