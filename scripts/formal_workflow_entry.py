"""在精确 workflow_orchestrator 子解释器中执行正式论文入口.

宿主 launcher 负责解释器与完整哈希锁. 本模块只在该受验证子解释器中发布
正式代码锁、配置当前论文运行层级, 并调用 GPU workflow 或单 repeat 证据入口.
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


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.runtime.repository_environment import (  # noqa: E402
    build_formal_execution_lock,
    publish_formal_execution_lock,
)
from experiments.protocol.paper_run_config import build_paper_run_config  # noqa: E402
from paper_experiments.runners.randomization_repeat_evidence import (  # noqa: E402
    write_randomization_repeat_evidence_package,
)
from scripts.run_gpu_server_workflow import (  # noqa: E402
    ACTIVE_REPEAT_GPU_WORKFLOW_NAMES,
    WORKFLOW_ROUTES,
    _require_workflow_orchestrator_environment,
    run_workflow,
)
from scripts.gpu_method_qualification_host_workflow import (  # noqa: E402
    CONTENT_RUNTIME_SMOKE_WORKFLOW_NAME,
    QUALIFICATION_WORKFLOW_NAME,
    run_content_runtime_smoke_host_workflow,
    run_gpu_method_qualification_host_workflow,
)


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
    """调用能够自行完成环境配置的独立 GPU 服务器 workflow."""

    return run_workflow(
        arguments.workflow,
        arguments.paper_run_name,
        arguments.repository_commit,
        root,
        arguments.persistent_output_dir or None,
        randomization_repeat_id=(arguments.randomization_repeat_id or None),
    )


def execute(arguments: argparse.Namespace) -> dict[str, Any]:
    """发布正式执行锁并执行唯一目标入口."""

    root = Path(arguments.root).resolve()
    bootstrap_identity = _validate_bootstrap_identity(arguments)
    execution_lock = publish_formal_execution_lock(
        build_formal_execution_lock(root, arguments.repository_commit)
    )
    decision = "pass"
    failure_reasons: list[str] = []
    if arguments.operation == "gpu":
        result = _gpu_result(arguments, root)
        workflow_name = arguments.workflow
        workflow_summary = result.get("workflow_summary", {})
        workflow_environment = result.get("workflow_environment")
        archive_record = result.get("archive_record")
        orchestrator_environment = result.get("orchestrator_dependency_environment")
        if not isinstance(orchestrator_environment, dict) or not isinstance(
            workflow_environment,
            dict,
        ):
            raise RuntimeError("GPU workflow 缺少父编排或 workflow 环境证据")
        if result.get("randomization_repeat_id", "") != (
            arguments.randomization_repeat_id or ""
        ):
            raise RuntimeError("GPU workflow 返回的 repeat 身份与请求不一致")
        randomization_scope = str(result.get("randomization_scope", ""))
        resolved_repeat_id = str(result.get("randomization_repeat_id", ""))
    elif arguments.operation == "repeat_evidence":
        orchestrator_environment = _require_workflow_orchestrator_environment(root)
        os.environ["SLM_WM_PAPER_RUN_NAME"] = arguments.paper_run_name
        os.environ["SLM_WM_RANDOMIZATION_REPEAT_ID"] = (
            arguments.randomization_repeat_id
        )
        paper_run = build_paper_run_config(root)
        if (
            paper_run.run_name != arguments.paper_run_name
            or paper_run.randomization_repeat_id
            != arguments.randomization_repeat_id
        ):
            raise ValueError("单 repeat 证据入口与当前论文运行配置不一致")
        result = write_randomization_repeat_evidence_package(
            arguments.package_search_root,
            paper_run_name=arguments.paper_run_name,
            target_fpr=paper_run.target_fpr,
            randomization_repeat_id=paper_run.randomization_repeat_id,
            root=root,
        )
        workflow_name = "randomization_repeat_evidence"
        workflow_summary = result
        workflow_environment = {}
        archive_record = result
        randomization_scope = "active_repeat_component"
        resolved_repeat_id = paper_run.randomization_repeat_id
    elif arguments.operation == "qualification":
        orchestrator_environment = _require_workflow_orchestrator_environment(root)
        os.environ["SLM_WM_PAPER_RUN_NAME"] = arguments.paper_run_name
        result = run_gpu_method_qualification_host_workflow(
            root=root,
            repository_commit=arguments.repository_commit,
            paper_run_name=arguments.paper_run_name,
            prompt_id=arguments.prompt_id,
            result_path=arguments.result_path,
            known_answer=arguments.known_answer,
            qualification_output_root=arguments.qualification_output_root,
            registered_budget=(arguments.registered_budget or None),
        )
        workflow_name = QUALIFICATION_WORKFLOW_NAME
        workflow_summary = result.get("workflow_summary", {})
        workflow_environment = result.get("workflow_environment", {})
        archive_record = None
        randomization_scope = "gpu_operator_qualification_only"
        resolved_repeat_id = ""
        decision = str(result.get("decision", "fail"))
        raw_failure_reasons = result.get("failure_reasons", [])
        failure_reasons = (
            [str(reason) for reason in raw_failure_reasons]
            if isinstance(raw_failure_reasons, list)
            else ["qualification_failure_reasons_invalid"]
        )
        if (
            decision not in {"pass", "fail"}
            or result.get("supports_paper_claim") is not False
        ):
            raise RuntimeError("GPU 方法资格化宿主结果身份无效")
    else:
        orchestrator_environment = _require_workflow_orchestrator_environment(root)
        os.environ["SLM_WM_PAPER_RUN_NAME"] = arguments.paper_run_name
        result = run_content_runtime_smoke_host_workflow(
            root=root,
            repository_commit=arguments.repository_commit,
            paper_run_name=arguments.paper_run_name,
            prompt_id=arguments.prompt_id,
            result_path=arguments.result_path,
            smoke_output_root=arguments.smoke_output_root,
            reference_gradient=arguments.reference_gradient,
            reference_response=arguments.reference_response,
            reference_sensitivity=arguments.reference_sensitivity,
        )
        workflow_name = CONTENT_RUNTIME_SMOKE_WORKFLOW_NAME
        workflow_summary = result.get("workflow_summary", {})
        workflow_environment = result.get("workflow_environment", {})
        archive_record = None
        randomization_scope = "single_sample_unqualified_gpu_smoke"
        resolved_repeat_id = ""
        decision = str(result.get("decision", "fail"))
        raw_failure_reasons = result.get("failure_reasons", [])
        failure_reasons = (
            [str(reason) for reason in raw_failure_reasons]
            if isinstance(raw_failure_reasons, list)
            else ["content_runtime_smoke_failure_reasons_invalid"]
        )
        if decision not in {"pass", "fail"} or result.get(
            "supports_paper_claim"
        ) is not False:
            raise RuntimeError("content runtime smoke host result identity invalid")
    if (
        orchestrator_environment.get("profile_id")
        != bootstrap_identity["profile_id"]
        or orchestrator_environment.get("complete_hash_lock_digest")
        != bootstrap_identity["complete_hash_lock_digest"]
    ):
        raise RuntimeError("宿主引导身份与父编排依赖门禁不一致")
    workflow_completion_state = str(
        workflow_summary.get(
            "workflow_completion_state",
            (
                "repeat_component_packaged"
                if workflow_name == "randomization_repeat_evidence"
                and workflow_summary.get("repeat_component_ready") is True
                else "unknown"
            ),
        )
    )
    return {
        "report_schema": "formal_workflow_execution_result",
        "schema_version": 1,
        "operation_kind": "exact_orchestrator_workflow_execution",
        "workflow_name": workflow_name,
        "paper_run_name": arguments.paper_run_name,
        "randomization_scope": randomization_scope,
        "randomization_repeat_id": resolved_repeat_id,
        "profile_id": "workflow_orchestrator",
        "orchestrator_bootstrap_identity": bootstrap_identity,
        "orchestrator_dependency_environment": orchestrator_environment,
        "formal_execution_lock": execution_lock,
        "workflow_summary": workflow_summary,
        "workflow_environment": workflow_environment,
        "archive_record": archive_record,
        "decision": decision,
        "session_execution_decision": decision,
        "workflow_completion_state": workflow_completion_state,
        "paper_run_closed": False,
        "result_closure_ready": False,
        "failure_reasons": failure_reasons,
        "supports_paper_claim": False,
    }


def build_parser() -> argparse.ArgumentParser:
    """构造精确父环境内部入口参数."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "operation",
        choices=("gpu", "repeat_evidence", "qualification", "content_runtime_smoke"),
    )
    parser.add_argument("--root", required=True)
    parser.add_argument("--repository-commit", required=True)
    parser.add_argument("--paper-run-name", required=True)
    parser.add_argument("--result-path", required=True)
    parser.add_argument("--orchestrator-profile-id", required=True)
    parser.add_argument("--orchestrator-python-version", required=True)
    parser.add_argument("--orchestrator-lock-digest", required=True)
    parser.add_argument("--orchestrator-python-executable", required=True)
    parser.add_argument("--orchestrator-python-executable-sha256", required=True)
    parser.add_argument("--workflow", choices=tuple(WORKFLOW_ROUTES))
    parser.add_argument("--persistent-output-dir", default="")
    parser.add_argument("--package-search-root", default="")
    parser.add_argument("--randomization-repeat-id", default="")
    parser.add_argument("--prompt-id", default="")
    parser.add_argument(
        "--known-answer",
        default="configs/keyed_prg_cross_platform_known_answer.json",
    )
    parser.add_argument("--registered-budget", default="")
    parser.add_argument(
        "--qualification-output-root",
        default="outputs/gpu_method_qualification",
    )
    parser.add_argument("--smoke-output-root", default="outputs/content_runtime_smoke")
    parser.add_argument("--reference-gradient", type=float)
    parser.add_argument("--reference-response", type=float)
    parser.add_argument("--reference-sensitivity", type=float)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """执行内部入口并始终写出受治理结果或失败诊断."""

    arguments = build_parser().parse_args(argv)
    root = Path(arguments.root).resolve()
    try:
        if arguments.operation == "gpu":
            if arguments.workflow is None:
                raise ValueError("GPU 入口必须提供 workflow")
            if (
                arguments.workflow in ACTIVE_REPEAT_GPU_WORKFLOW_NAMES
                and not arguments.randomization_repeat_id
            ):
                raise ValueError("活动随机化 GPU 入口必须提供 repeat ID")
            if (
                arguments.workflow not in ACTIVE_REPEAT_GPU_WORKFLOW_NAMES
                and arguments.randomization_repeat_id
            ):
                raise ValueError("跨 repeat 不变 GPU 入口不得提供 repeat ID")
        if (
            arguments.operation == "repeat_evidence"
            and (
                not arguments.package_search_root
                or not arguments.randomization_repeat_id
            )
        ):
            raise ValueError("单 repeat 证据入口缺少 package 搜索目录或 repeat ID")
        if arguments.operation == "qualification" and not arguments.prompt_id:
            raise ValueError("GPU 方法资格化入口必须提供 Prompt ID")
        if arguments.operation == "content_runtime_smoke" and (
            not arguments.prompt_id
            or any(
                value is None
                for value in (
                    arguments.reference_gradient,
                    arguments.reference_response,
                    arguments.reference_sensitivity,
                )
            )
        ):
            raise ValueError("content runtime smoke requires Prompt and three references")
        payload = execute(arguments)
        result_path = _write_result(root, arguments.result_path, payload)
        print(
            json.dumps(
                {
                    "decision": payload["decision"],
                    "session_execution_decision": payload[
                        "session_execution_decision"
                    ],
                    "workflow_completion_state": payload[
                        "workflow_completion_state"
                    ],
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
        return 0 if payload["decision"] == "pass" else 1
    except Exception as exc:
        payload = {
            "report_schema": "formal_workflow_execution_result",
            "schema_version": 1,
            "operation_kind": "exact_orchestrator_workflow_execution",
            "workflow_name": (
                arguments.workflow
                or (
                    QUALIFICATION_WORKFLOW_NAME
                    if arguments.operation == "qualification"
                    else (
                        CONTENT_RUNTIME_SMOKE_WORKFLOW_NAME
                        if arguments.operation == "content_runtime_smoke"
                        else "randomization_repeat_evidence"
                    )
                )
            ),
            "paper_run_name": arguments.paper_run_name,
            "randomization_repeat_id": arguments.randomization_repeat_id,
            "profile_id": "workflow_orchestrator",
            "orchestrator_bootstrap_identity": {},
            "orchestrator_dependency_environment": {},
            "formal_execution_lock": {},
            "workflow_summary": {},
            "workflow_environment": {},
            "archive_record": None,
            "decision": "fail",
            "session_execution_decision": "fail",
            "workflow_completion_state": "failed",
            "paper_run_closed": False,
            "result_closure_ready": False,
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
