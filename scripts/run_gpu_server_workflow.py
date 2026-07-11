"""在无 Notebook 的 CUDA 服务器上调度隔离科学工作流.

该脚本只运行 CPU 父编排逻辑.主方法、正式消融、三个 method-faithful
baseline 和 T2SMark 都必须进入各自受治理的科学子解释器, 宿主解释器不得
直接导入或执行科学 runner.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, Iterator, Mapping, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.runtime.dependency_profiles import (
    WORKFLOW_ORCHESTRATOR_PROFILE_ID,
    inspect_dependency_profile_environment,
    require_dependency_profile_ready,
)
from experiments.runtime.repository_environment import (
    FORMAL_EXECUTION_COMMIT_ENVIRONMENT_KEY,
    FORMAL_EXECUTION_LOCK_DIGEST_ENVIRONMENT_KEY,
    build_formal_execution_lock,
    publish_formal_execution_lock,
)


PAPER_RUN_NAME_ENVIRONMENT_KEY = "SLM_WM_PAPER_RUN_NAME"
PRIMARY_BASELINE_ID_ENVIRONMENT_KEY = "SLM_WM_PRIMARY_BASELINE_ID"
MAIN_METHOD_PROFILE_ID = "sd35_method_runtime_gpu"
T2SMARK_PROFILE_ID = "t2smark_sd35_gpu"
TREE_RING_OFFICIAL_PROFILE_ID = "tree_ring_official_py39_cu117"
GAUSSIAN_SHADING_OFFICIAL_PROFILE_ID = "gaussian_shading_official_py38_cu117"
SHALLOW_DIFFUSE_OFFICIAL_PROFILE_ID = "shallow_diffuse_official_py39_cu117"
MAIN_METHOD_SESSION_MODULE = "experiments.runtime.semantic_watermark_scientific_session"
SHARED_BASELINE_WORKFLOW_NAME = "external_baseline_method_faithful"
SHARED_T2SMARK_WORKFLOW_NAME = "official_reference_t2smark"
RESULT_SCHEMA = "gpu_server_workflow_result"
RESULT_SCHEMA_VERSION = 1
RESULT_OPERATION_KIND = "isolated_gpu_workflow_orchestration"


@dataclass(frozen=True)
class WorkflowRoute:
    """描述一个公开服务器选项如何进入唯一隔离科学执行边界."""

    scientific_profile_id: str
    child_argv_tail: Tuple[str, ...] = ()
    shared_isolated_workflow_name: Optional[str] = None
    official_reference_runner_name: Optional[str] = None
    baseline_id: Optional[str] = None

    @property
    def uses_scientific_command(self) -> bool:
        """判断当前路由是否直接调用通用隔离科学命令执行器."""

        return bool(self.child_argv_tail)

    @property
    def route_kind(self) -> str:
        """返回服务器结果 schema 使用的稳定路由类别."""

        if self.uses_scientific_command:
            return "scientific_session"
        if self.official_reference_runner_name is not None:
            return "official_reference"
        return "shared_isolated_workflow"


WORKFLOW_ROUTES: Dict[str, WorkflowRoute] = {
    "image_only_dataset": WorkflowRoute(
        scientific_profile_id=MAIN_METHOD_PROFILE_ID,
        child_argv_tail=("-m", MAIN_METHOD_SESSION_MODULE),
    ),
    "mechanism_ablation": WorkflowRoute(
        scientific_profile_id=MAIN_METHOD_PROFILE_ID,
        child_argv_tail=(
            "-m",
            MAIN_METHOD_SESSION_MODULE,
            "--run-formal-ablation",
        ),
    ),
    "external_baseline_tree_ring": WorkflowRoute(
        scientific_profile_id=MAIN_METHOD_PROFILE_ID,
        shared_isolated_workflow_name=SHARED_BASELINE_WORKFLOW_NAME,
        baseline_id="tree_ring",
    ),
    "external_baseline_gaussian_shading": WorkflowRoute(
        scientific_profile_id=MAIN_METHOD_PROFILE_ID,
        shared_isolated_workflow_name=SHARED_BASELINE_WORKFLOW_NAME,
        baseline_id="gaussian_shading",
    ),
    "external_baseline_shallow_diffuse": WorkflowRoute(
        scientific_profile_id=MAIN_METHOD_PROFILE_ID,
        shared_isolated_workflow_name=SHARED_BASELINE_WORKFLOW_NAME,
        baseline_id="shallow_diffuse",
    ),
    "official_reference_t2smark": WorkflowRoute(
        scientific_profile_id=T2SMARK_PROFILE_ID,
        shared_isolated_workflow_name=SHARED_T2SMARK_WORKFLOW_NAME,
    ),
    "official_reference_tree_ring": WorkflowRoute(
        scientific_profile_id=TREE_RING_OFFICIAL_PROFILE_ID,
        official_reference_runner_name="tree_ring",
    ),
    "official_reference_gaussian_shading": WorkflowRoute(
        scientific_profile_id=GAUSSIAN_SHADING_OFFICIAL_PROFILE_ID,
        official_reference_runner_name="gaussian_shading",
    ),
    "official_reference_shallow_diffuse": WorkflowRoute(
        scientific_profile_id=SHALLOW_DIFFUSE_OFFICIAL_PROFILE_ID,
        official_reference_runner_name="shallow_diffuse",
    ),
}


def _require_workflow_orchestrator_environment(root_path: Path) -> Dict[str, Any]:
    """要求服务器父解释器精确匹配已提交的 CPU 编排 profile."""

    registry_path = root_path / "configs" / "dependency_profile_registry.json"
    profile = require_dependency_profile_ready(
        WORKFLOW_ORCHESTRATOR_PROFILE_ID,
        registry_path,
    )
    inspection = inspect_dependency_profile_environment(
        WORKFLOW_ORCHESTRATOR_PROFILE_ID,
        path=registry_path,
    )
    expected_inspection = {
        "profile_name": WORKFLOW_ORCHESTRATOR_PROFILE_ID,
        "profile_digest": profile.profile_digest,
        "complete_hash_lock_digest": profile.complete_hash_lock_digest,
        "profile_formal_ready": True,
        "environment_match": True,
        "mismatches": [],
        "readiness_blockers": [],
        "decision": "pass",
    }
    dependency_count = profile.complete_hash_lock_dependency_count
    inspection_digest = inspection.get("inspection_digest")
    if not all(
        (
            profile.execution_role == "workflow_orchestration",
            profile.accelerator_runtime == "cpu",
            profile.pytorch_index_url is None,
            isinstance(dependency_count, int),
            not isinstance(dependency_count, bool),
            dependency_count > 0,
            all(
                inspection.get(field_name) == expected_value
                for field_name, expected_value in expected_inspection.items()
            ),
            isinstance(inspection_digest, str),
            (
                len(inspection_digest) == 64
                and all(
                    character in "0123456789abcdef"
                    for character in inspection_digest
                )
            )
            if isinstance(inspection_digest, str)
            else False,
        )
    ):
        raise RuntimeError("workflow_orchestrator 父解释器未通过正式依赖门禁")
    return {
        "profile_id": profile.profile_name,
        "profile_digest": profile.profile_digest,
        "direct_requirements_digest": profile.direct_requirements_digest,
        "complete_hash_lock_digest": profile.complete_hash_lock_digest,
        "complete_hash_lock_dependency_count": dependency_count,
        "inspection_digest": inspection_digest,
        "inspection": inspection,
    }


def _restore_environment_value(key: str, value: Optional[str]) -> None:
    """恢复父进程环境, 使可复用函数不会污染后续服务器任务."""

    if value is None:
        os.environ.pop(key, None)
    else:
        os.environ[key] = value


@contextmanager
def _published_workflow_environment(
    execution_lock: Mapping[str, Any],
    paper_run_name: str,
    baseline_id: Optional[str],
) -> Iterator[None]:
    """临时发布正式执行锁、论文级别和唯一 baseline 身份.

    该上下文属于通用父子进程写法.隔离执行 API 会在启动前后实时复验锁,
    外部 baseline 包装则通过同一环境继承机制解析唯一 baseline.
    """

    keys = (
        FORMAL_EXECUTION_COMMIT_ENVIRONMENT_KEY,
        FORMAL_EXECUTION_LOCK_DIGEST_ENVIRONMENT_KEY,
        PAPER_RUN_NAME_ENVIRONMENT_KEY,
        PRIMARY_BASELINE_ID_ENVIRONMENT_KEY,
    )
    previous_values = {key: os.environ.get(key) for key in keys}
    try:
        publish_formal_execution_lock(execution_lock)
        os.environ[PAPER_RUN_NAME_ENVIRONMENT_KEY] = paper_run_name
        if baseline_id is None:
            os.environ.pop(PRIMARY_BASELINE_ID_ENVIRONMENT_KEY, None)
        else:
            os.environ[PRIMARY_BASELINE_ID_ENVIRONMENT_KEY] = baseline_id
        yield
    finally:
        for key, value in previous_values.items():
            _restore_environment_value(key, value)


def _run_shared_isolated_workflow(
    *,
    root: Path,
    workflow_name: str,
) -> Mapping[str, Any]:
    """延迟调用完整论文层的 baseline / T2SMark 隔离父包装."""

    from paper_experiments.runners.isolated_scientific_workflow import (
        run_isolated_scientific_workflow,
    )

    return run_isolated_scientific_workflow(
        root=root,
        workflow_name=workflow_name,
    )


def _run_main_method_route(
    *,
    route: WorkflowRoute,
    workflow_name: str,
    paper_run_name: str,
    root_path: Path,
) -> Dict[str, Any]:
    """调用可脱离 Notebook 的完整主方法绑定与打包会话."""

    from scripts.semantic_watermark_scientific_workflow import (
        run_semantic_watermark_image_only_session,
    )

    session = run_semantic_watermark_image_only_session(
        root_path,
        run_formal_ablation=workflow_name == "mechanism_ablation",
    )
    return {
        "workflow_summary": session,
        "archive_record": None,
        "return_code": 0,
        "stdout": "",
        "stderr": "",
    }


def _run_shared_route(
    *,
    route: WorkflowRoute,
    root_path: Path,
    workflow_name: str,
    paper_run_name: str,
) -> Dict[str, Any]:
    """运行外部科学 workflow 并生成可交付结果包."""

    delivery_dir = (
        root_path
        / "outputs"
        / "gpu_server_delivery"
        / paper_run_name
        / workflow_name
    )

    if route.official_reference_runner_name is not None:
        runner_name = route.official_reference_runner_name
        if runner_name == "tree_ring":
            from paper_experiments.runners.tree_ring_official_reference import (
                package_tree_ring_official_reference_outputs,
                run_default_tree_ring_official_reference_plan,
            )

            summary = run_default_tree_ring_official_reference_plan(root=root_path)
            archive_record = package_tree_ring_official_reference_outputs(
                root=root_path,
                drive_output_dir=str(delivery_dir),
            )
        elif runner_name == "gaussian_shading":
            from paper_experiments.runners.gaussian_shading_official_reference import (
                package_gaussian_shading_official_reference_outputs,
                run_default_gaussian_shading_official_reference_plan,
            )

            summary = run_default_gaussian_shading_official_reference_plan(
                root=root_path
            )
            archive_record = package_gaussian_shading_official_reference_outputs(
                root=root_path,
                drive_output_dir=str(delivery_dir),
            )
        else:
            from paper_experiments.runners.shallow_diffuse_official_reference import (
                package_shallow_diffuse_official_reference_outputs,
                run_default_shallow_diffuse_official_reference_plan,
            )

            summary = run_default_shallow_diffuse_official_reference_plan(
                root=root_path
            )
            archive_record = package_shallow_diffuse_official_reference_outputs(
                root=root_path,
                drive_output_dir=str(delivery_dir),
            )
        return {
            "workflow_summary": dict(summary),
            "archive_record": archive_record.to_dict(),
            "return_code": 0,
            "stdout": "",
            "stderr": "",
        }

    isolated_workflow_name = route.shared_isolated_workflow_name
    if isolated_workflow_name is None:
        raise RuntimeError("共享隔离工作流路由缺少内部 workflow 名称")
    summary = _run_shared_isolated_workflow(
        root=root_path,
        workflow_name=isolated_workflow_name,
    )
    if isolated_workflow_name == SHARED_BASELINE_WORKFLOW_NAME:
        from paper_experiments.runners.external_baseline_method_faithful import (
            package_external_baseline_method_faithful_outputs,
        )

        archive_record = package_external_baseline_method_faithful_outputs(
            root=root_path,
            drive_output_dir=str(delivery_dir),
            baseline_id=route.baseline_id,
        )
    else:
        from paper_experiments.runners.isolated_scientific_workflow import (
            package_isolated_scientific_workflow_outputs,
        )

        archive_record = package_isolated_scientific_workflow_outputs(
            root=root_path,
            workflow_name=SHARED_T2SMARK_WORKFLOW_NAME,
            drive_output_dir=str(delivery_dir),
        )
    return {
        "workflow_summary": dict(summary),
        "archive_record": archive_record.to_dict(),
        "return_code": 0,
        "stdout": "",
        "stderr": "",
    }


def _build_workflow_result(
    *,
    workflow_name: str,
    paper_run_name: str,
    route: WorkflowRoute,
    execution_lock: Mapping[str, Any],
    orchestrator_environment: Mapping[str, Any],
    route_result: Mapping[str, Any],
) -> Dict[str, Any]:
    """把9条服务器路由收敛为同一受治理输出 schema."""

    workflow_summary = route_result.get("workflow_summary")
    archive_record = route_result.get("archive_record")
    return_code = route_result.get("return_code")
    stdout = route_result.get("stdout")
    stderr = route_result.get("stderr")
    if not all(
        (
            isinstance(workflow_summary, Mapping),
            archive_record is None or isinstance(archive_record, Mapping),
            isinstance(return_code, int),
            not isinstance(return_code, bool),
            return_code == 0,
            isinstance(stdout, str),
            isinstance(stderr, str),
        )
    ):
        raise RuntimeError("GPU 服务器内部路由没有返回完整统一结果")
    return {
        "report_schema": RESULT_SCHEMA,
        "schema_version": RESULT_SCHEMA_VERSION,
        "operation_kind": RESULT_OPERATION_KIND,
        "workflow_name": workflow_name,
        "paper_run_name": paper_run_name,
        "scientific_profile_id": route.scientific_profile_id,
        "baseline_id": route.baseline_id,
        "route_kind": route.route_kind,
        "child_argv_tail": list(route.child_argv_tail),
        "shared_isolated_workflow_name": route.shared_isolated_workflow_name,
        "official_reference_runner_name": route.official_reference_runner_name,
        "workflow_summary": dict(workflow_summary),
        "archive_record": (
            None if archive_record is None else dict(archive_record)
        ),
        "command": [],
        "return_code": return_code,
        "stdout": stdout,
        "stderr": stderr,
        "orchestrator_dependency_environment": dict(
            orchestrator_environment
        ),
        "formal_execution_lock": dict(execution_lock),
        "decision": "pass",
        "supports_paper_claim": False,
    }


def run_workflow(
    workflow_name: str,
    paper_run_name: str,
    repository_commit: str,
    root: str | Path = ".",
) -> Dict[str, Any]:
    """在 CPU 父入口中发布执行身份并调度一个隔离 GPU 工作流."""

    if workflow_name not in WORKFLOW_ROUTES:
        raise ValueError("未知正式工作流: {0}".format(workflow_name))
    route = WORKFLOW_ROUTES[workflow_name]
    root_path = Path(root).resolve()
    orchestrator_environment = _require_workflow_orchestrator_environment(
        root_path
    )
    execution_lock = build_formal_execution_lock(root_path, repository_commit)
    with _published_workflow_environment(
        execution_lock,
        paper_run_name,
        route.baseline_id,
    ):
        route_result = (
            _run_main_method_route(
                route=route,
                workflow_name=workflow_name,
                paper_run_name=paper_run_name,
                root_path=root_path,
            )
            if route.uses_scientific_command
            else _run_shared_route(
                route=route,
                root_path=root_path,
                workflow_name=workflow_name,
                paper_run_name=paper_run_name,
            )
        )
    return _build_workflow_result(
        workflow_name=workflow_name,
        paper_run_name=paper_run_name,
        route=route,
        execution_lock=execution_lock,
        orchestrator_environment=orchestrator_environment,
        route_result=route_result,
    )


def build_parser() -> argparse.ArgumentParser:
    """构造服务器入口参数."""

    parser = argparse.ArgumentParser(description="运行不依赖 Notebook 的正式 GPU 工作流。")
    parser.add_argument("--workflow", required=True, choices=tuple(WORKFLOW_ROUTES))
    parser.add_argument(
        "--paper-run-name",
        required=True,
        choices=("probe_paper", "pilot_paper", "full_paper"),
    )
    parser.add_argument(
        "--repository-commit",
        required=True,
        help="正式执行使用的精确40位小写 Git SHA.",
    )
    parser.add_argument("--root", default=".")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    """命令行入口."""

    args = build_parser().parse_args(argv)
    print(
        json.dumps(
            run_workflow(
                args.workflow,
                args.paper_run_name,
                args.repository_commit,
                args.root,
            ),
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
