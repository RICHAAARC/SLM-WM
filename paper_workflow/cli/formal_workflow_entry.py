"""在精确 workflow_orchestrator 子解释器中执行正式论文入口.

宿主 launcher 负责解释器与完整哈希锁. 本模块只在该受验证子解释器中发布
正式代码锁、配置当前论文运行层级, 并调用现有 GPU 服务器或 CPU 闭合入口.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import sys
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.runtime.repository_environment import (  # noqa: E402
    build_formal_execution_lock,
    publish_formal_execution_lock,
)
from paper_workflow.colab_utils.paper_run_environment import (  # noqa: E402
    configure_paper_run_environment,
)
from scripts.run_gpu_server_result_closure import (  # noqa: E402
    execute_server_result_closure,
)
from scripts.run_gpu_server_workflow import (  # noqa: E402
    _require_workflow_orchestrator_environment,
    run_workflow,
)


WORKFLOW_CONFIGURATION = {
    "image_only_dataset": ("semantic_watermark_image_only", ""),
    "mechanism_ablation": ("semantic_watermark_image_only", ""),
    "external_baseline_tree_ring": ("external_baseline_method_faithful", "tree_ring"),
    "external_baseline_gaussian_shading": (
        "external_baseline_method_faithful",
        "gaussian_shading",
    ),
    "external_baseline_shallow_diffuse": (
        "external_baseline_method_faithful",
        "shallow_diffuse",
    ),
    "official_reference_t2smark": ("official_reference_t2smark", ""),
    "official_reference_tree_ring": ("official_reference_tree_ring", ""),
    "official_reference_gaussian_shading": (
        "official_reference_gaussian_shading",
        "",
    ),
    "official_reference_shallow_diffuse": (
        "official_reference_shallow_diffuse",
        "",
    ),
}
WORKFLOW_PERSISTENT_ENVIRONMENT_KEYS = {
    "external_baseline_tree_ring": "SLM_WM_EXTERNAL_BASELINE_DRIVE_OUTPUT_DIR",
    "external_baseline_gaussian_shading": "SLM_WM_EXTERNAL_BASELINE_DRIVE_OUTPUT_DIR",
    "external_baseline_shallow_diffuse": "SLM_WM_EXTERNAL_BASELINE_DRIVE_OUTPUT_DIR",
    "official_reference_t2smark": "SLM_WM_T2SMARK_FORMAL_DRIVE_OUTPUT_DIR",
    "official_reference_tree_ring": "SLM_WM_TREE_RING_OFFICIAL_DRIVE_OUTPUT_DIR",
    "official_reference_gaussian_shading": "SLM_WM_GAUSSIAN_SHADING_OFFICIAL_DRIVE_OUTPUT_DIR",
    "official_reference_shallow_diffuse": "SLM_WM_SHALLOW_DIFFUSE_OFFICIAL_DRIVE_OUTPUT_DIR",
}
ORCHESTRATOR_PROFILE_ID = "workflow_orchestrator"
ORCHESTRATOR_PYTHON_VERSION = "3.12.13"
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _file_sha256(path: Path) -> str:
    """流式计算解释器文件摘要, 避免把二进制整体读入内存."""

    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_bootstrap_identity(arguments: argparse.Namespace) -> dict[str, Any]:
    """复验宿主传入的父 profile、锁和当前解释器身份."""

    executable = Path(sys.executable)
    expected_executable = Path(arguments.orchestrator_python_executable)
    lock_digest = str(arguments.orchestrator_lock_digest)
    executable_digest = str(arguments.orchestrator_python_executable_sha256)
    if not all(
        (
            arguments.orchestrator_profile_id == ORCHESTRATOR_PROFILE_ID,
            arguments.orchestrator_python_version == ORCHESTRATOR_PYTHON_VERSION,
            platform.python_implementation() == "CPython",
            platform.python_version() == ORCHESTRATOR_PYTHON_VERSION,
            platform.system().strip().lower() == "linux",
            platform.machine().strip().lower() in {"x86_64", "amd64", "x64"},
            sys.flags.isolated == 1,
            executable.is_file(),
            executable.resolve() == expected_executable.resolve(),
            SHA256_PATTERN.fullmatch(lock_digest) is not None,
            SHA256_PATTERN.fullmatch(executable_digest) is not None,
        )
    ):
        raise RuntimeError("workflow_orchestrator 宿主引导身份无效")
    if _file_sha256(executable) != executable_digest:
        raise RuntimeError("workflow_orchestrator Python executable 摘要不一致")
    return {
        "profile_id": ORCHESTRATOR_PROFILE_ID,
        "python_version": ORCHESTRATOR_PYTHON_VERSION,
        "complete_hash_lock_digest": lock_digest,
        "python_executable": executable.as_posix(),
        "python_executable_sha256": executable_digest,
    }


def _write_result(root: Path, result_path: str | Path, payload: dict[str, Any]) -> Path:
    """把入口结果限制到 outputs/ 并使用原子替换发布."""

    requested = Path(result_path)
    path = requested.resolve() if requested.is_absolute() else (root / requested).resolve()
    outputs_root = (root / "outputs").resolve()
    try:
        path.relative_to(outputs_root)
    except ValueError as exc:
        raise ValueError("正式 workflow result 必须位于 outputs/ 下") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)
    return path


def _gpu_result(arguments: argparse.Namespace, root: Path) -> dict[str, Any]:
    """配置外层 Colab 语义并调用现有 GPU 服务器 workflow."""

    configuration_name, baseline_id = WORKFLOW_CONFIGURATION[arguments.workflow]
    os.environ["SLM_WM_PAPER_RUN_NAME"] = arguments.paper_run_name
    configure_paper_run_environment(
        configuration_name,
        baseline_id=baseline_id,
        repository_root=root,
    )
    persistent_output_dir = arguments.persistent_output_dir
    if not persistent_output_dir:
        if arguments.workflow in {"image_only_dataset", "mechanism_ablation"}:
            persistent_output_dir = os.environ["SLM_WM_DRIVE_RESULT_ROOT"]
        else:
            persistent_output_dir = os.environ[
                WORKFLOW_PERSISTENT_ENVIRONMENT_KEYS[arguments.workflow]
            ]
    return run_workflow(
        arguments.workflow,
        arguments.paper_run_name,
        arguments.repository_commit,
        root,
        persistent_output_dir,
    )


def execute(arguments: argparse.Namespace) -> dict[str, Any]:
    """发布正式执行锁并执行唯一目标入口."""

    root = Path(arguments.root).resolve()
    bootstrap_identity = _validate_bootstrap_identity(arguments)
    execution_lock = publish_formal_execution_lock(
        build_formal_execution_lock(root, arguments.repository_commit)
    )
    if arguments.operation == "gpu":
        result = _gpu_result(arguments, root)
        workflow_name = arguments.workflow
        workflow_summary = result.get("workflow_summary", {})
        archive_record = result.get("archive_record")
        orchestrator_environment = result.get("orchestrator_dependency_environment")
        if not isinstance(orchestrator_environment, dict):
            raise RuntimeError("GPU workflow 缺少父编排依赖环境证据")
    else:
        orchestrator_environment = _require_workflow_orchestrator_environment(root)
        result = execute_server_result_closure(
            root=root,
            paper_run_name=arguments.paper_run_name,
            package_search_root=arguments.package_search_root,
            complete_output_dir=arguments.complete_output_dir,
            repository_commit=arguments.repository_commit,
            dry_run=arguments.dry_run,
        )
        workflow_name = "paper_result_closure"
        workflow_summary = result
        archive_record = None
    if (
        orchestrator_environment.get("profile_id")
        != bootstrap_identity["profile_id"]
        or orchestrator_environment.get("complete_hash_lock_digest")
        != bootstrap_identity["complete_hash_lock_digest"]
    ):
        raise RuntimeError("宿主引导身份与父编排依赖门禁不一致")
    return {
        "report_schema": "formal_workflow_execution_result",
        "schema_version": 1,
        "operation_kind": "exact_orchestrator_workflow_execution",
        "workflow_name": workflow_name,
        "paper_run_name": arguments.paper_run_name,
        "profile_id": "workflow_orchestrator",
        "orchestrator_bootstrap_identity": bootstrap_identity,
        "orchestrator_dependency_environment": orchestrator_environment,
        "formal_execution_lock": execution_lock,
        "workflow_summary": workflow_summary,
        "archive_record": archive_record,
        "decision": "pass",
        "supports_paper_claim": False,
    }


def build_parser() -> argparse.ArgumentParser:
    """构造精确父环境内部入口参数."""

    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=("gpu", "closure"))
    parser.add_argument("--root", required=True)
    parser.add_argument("--repository-commit", required=True)
    parser.add_argument("--paper-run-name", required=True)
    parser.add_argument("--result-path", required=True)
    parser.add_argument("--orchestrator-profile-id", required=True)
    parser.add_argument("--orchestrator-python-version", required=True)
    parser.add_argument("--orchestrator-lock-digest", required=True)
    parser.add_argument("--orchestrator-python-executable", required=True)
    parser.add_argument("--orchestrator-python-executable-sha256", required=True)
    parser.add_argument("--workflow", choices=tuple(WORKFLOW_CONFIGURATION))
    parser.add_argument("--persistent-output-dir", default="")
    parser.add_argument("--package-search-root", default="")
    parser.add_argument("--complete-output-dir", default="")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """执行内部入口并始终写出受治理结果或失败诊断."""

    arguments = build_parser().parse_args(argv)
    root = Path(arguments.root).resolve()
    try:
        if arguments.operation == "gpu" and arguments.workflow is None:
            raise ValueError("GPU 入口必须提供 workflow")
        if arguments.operation == "closure" and (
            not arguments.package_search_root or not arguments.complete_output_dir
        ):
            raise ValueError("CPU 闭合入口缺少输入或输出目录")
        payload = execute(arguments)
        result_path = _write_result(root, arguments.result_path, payload)
        print(
            json.dumps(
                {
                    "decision": payload["decision"],
                    "workflow_name": payload["workflow_name"],
                    "paper_run_name": payload["paper_run_name"],
                    "result_path": result_path.as_posix(),
                    "profile_id": payload["profile_id"],
                    "complete_hash_lock_digest": payload[
                        "orchestrator_bootstrap_identity"
                    ]["complete_hash_lock_digest"],
                    "python_executable_sha256": payload[
                        "orchestrator_bootstrap_identity"
                    ]["python_executable_sha256"],
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0
    except Exception as exc:
        payload = {
            "report_schema": "formal_workflow_execution_result",
            "schema_version": 1,
            "operation_kind": "exact_orchestrator_workflow_execution",
            "workflow_name": arguments.workflow or "paper_result_closure",
            "paper_run_name": arguments.paper_run_name,
            "profile_id": "workflow_orchestrator",
            "orchestrator_bootstrap_identity": {},
            "orchestrator_dependency_environment": {},
            "formal_execution_lock": {},
            "workflow_summary": {},
            "archive_record": None,
            "decision": "fail",
            "failure_reasons": [f"{type(exc).__name__}:{exc}"],
            "supports_paper_claim": False,
        }
        try:
            _write_result(root, arguments.result_path, payload)
        except Exception:
            pass
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
