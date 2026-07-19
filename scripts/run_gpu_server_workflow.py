"""在无 Notebook 的 CUDA 服务器上调度隔离科学工作流.

该脚本只运行 CPU 父编排逻辑. 主方法、正式消融、三个 method-faithful
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
from experiments.protocol.formal_randomization import (
    resolve_formal_randomization_repeat,
)
from experiments.protocol.paper_run_config import build_paper_run_config
from scripts.formal_workflow_environment import (
    SEMANTIC_WATERMARK_REPEAT_DRIVE_ROOT_ENVIRONMENT_KEY,
    configure_formal_workflow_environment,
)


PAPER_RUN_NAME_ENVIRONMENT_KEY = "SLM_WM_PAPER_RUN_NAME"
RANDOMIZATION_REPEAT_ID_ENVIRONMENT_KEY = "SLM_WM_RANDOMIZATION_REPEAT_ID"
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


def _is_sha256(value: Any) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_absolute_symlink_free_path(value: str | Path) -> Path:
    requested = Path(value).expanduser()
    if not requested.is_absolute() or ".." in requested.parts:
        raise ValueError("image_only_dataset persistent root must be absolute")
    cursor = Path(requested.anchor)
    for part in requested.parts[1:]:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ValueError(
                "image_only_dataset persistent root must be symlink-free"
            )
    return requested


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
ACTIVE_REPEAT_GPU_WORKFLOW_NAMES = frozenset(
    {
        "image_only_dataset",
        "mechanism_ablation",
        "external_baseline_tree_ring",
        "external_baseline_gaussian_shading",
        "external_baseline_shallow_diffuse",
        "official_reference_t2smark",
    }
)
WORKFLOW_ENVIRONMENT_CONFIGURATION = {
    "image_only_dataset": ("semantic_watermark_image_only", ""),
    "mechanism_ablation": ("semantic_watermark_image_only", ""),
    "external_baseline_tree_ring": (
        "external_baseline_method_faithful",
        "tree_ring",
    ),
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
    "image_only_dataset": SEMANTIC_WATERMARK_REPEAT_DRIVE_ROOT_ENVIRONMENT_KEY,
    "mechanism_ablation": SEMANTIC_WATERMARK_REPEAT_DRIVE_ROOT_ENVIRONMENT_KEY,
    "external_baseline_tree_ring": "SLM_WM_EXTERNAL_BASELINE_DRIVE_OUTPUT_DIR",
    "external_baseline_gaussian_shading": "SLM_WM_EXTERNAL_BASELINE_DRIVE_OUTPUT_DIR",
    "external_baseline_shallow_diffuse": "SLM_WM_EXTERNAL_BASELINE_DRIVE_OUTPUT_DIR",
    "official_reference_t2smark": "SLM_WM_T2SMARK_FORMAL_DRIVE_OUTPUT_DIR",
    "official_reference_tree_ring": "SLM_WM_TREE_RING_OFFICIAL_DRIVE_OUTPUT_DIR",
    "official_reference_gaussian_shading": (
        "SLM_WM_GAUSSIAN_SHADING_OFFICIAL_DRIVE_OUTPUT_DIR"
    ),
    "official_reference_shallow_diffuse": (
        "SLM_WM_SHALLOW_DIFFUSE_OFFICIAL_DRIVE_OUTPUT_DIR"
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
    randomization_repeat_id: str | None,
    baseline_id: Optional[str],
) -> Iterator[None]:
    """临时发布正式执行锁、论文级别和唯一 baseline 身份.

    该上下文属于通用父子进程写法. 隔离执行 API 会在启动前后实时复验锁,
    外部 baseline 包装则通过同一环境继承机制解析唯一 baseline.
    """

    previous_values = {
        key: value
        for key, value in os.environ.items()
        if key.startswith("SLM_WM_") or key == "WANDB_MODE"
    }
    try:
        publish_formal_execution_lock(execution_lock)
        os.environ[PAPER_RUN_NAME_ENVIRONMENT_KEY] = paper_run_name
        if randomization_repeat_id is None:
            os.environ.pop(RANDOMIZATION_REPEAT_ID_ENVIRONMENT_KEY, None)
        else:
            os.environ[RANDOMIZATION_REPEAT_ID_ENVIRONMENT_KEY] = (
                randomization_repeat_id
            )
        if baseline_id is None:
            os.environ.pop(PRIMARY_BASELINE_ID_ENVIRONMENT_KEY, None)
        else:
            os.environ[PRIMARY_BASELINE_ID_ENVIRONMENT_KEY] = baseline_id
        yield
    finally:
        configured_keys = {
            key
            for key in os.environ
            if key.startswith("SLM_WM_") or key == "WANDB_MODE"
        }
        for key in configured_keys | set(previous_values):
            _restore_environment_value(key, previous_values.get(key))


def _configure_workflow_execution_environment(
    *,
    workflow_name: str,
    root_path: Path,
    persistent_output_dir: Optional[str | Path],
) -> tuple[Dict[str, Any], str | Path]:
    """配置独立服务器环境并解析本次持久化根目录."""

    if (
        workflow_name in ACTIVE_REPEAT_GPU_WORKFLOW_NAMES
        and workflow_name != "image_only_dataset"
        and persistent_output_dir is not None
        and str(persistent_output_dir).strip()
    ):
        raise ValueError(
            "活动随机化 GPU workflow 的持久化根必须由受治理 repeat 配置生成, "
            "不得显式覆盖"
        )

    configuration_name, baseline_id = WORKFLOW_ENVIRONMENT_CONFIGURATION[
        workflow_name
    ]
    environment_record = configure_formal_workflow_environment(
        configuration_name,
        baseline_id=baseline_id,
        repository_root=root_path,
    )
    environment_key = WORKFLOW_PERSISTENT_ENVIRONMENT_KEYS[workflow_name]
    resolved_persistent_output = persistent_output_dir
    if not resolved_persistent_output:
        resolved_persistent_output = os.environ.get(environment_key, "")
    if not resolved_persistent_output:
        raise RuntimeError("正式 GPU workflow 缺少持久化输出目录")
    if workflow_name == "image_only_dataset":
        resolved_persistent_output = str(
            _require_absolute_symlink_free_path(resolved_persistent_output)
        )
    os.environ[environment_key] = str(resolved_persistent_output)
    environment_record = {
        **dict(environment_record),
        "persistent_environment_key": environment_key,
        "persistent_output_dir": str(resolved_persistent_output),
    }
    return environment_record, resolved_persistent_output


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
    persistent_output_dir: Optional[str | Path] = None,
    calibration_only: bool = False,
    expected_reference_registry_digest: str = "",
    expected_reference_registry_file_sha256: str = "",
) -> Dict[str, Any]:
    """调用可脱离 Notebook 的完整主方法绑定与打包会话."""

    from scripts.semantic_watermark_scientific_workflow import (
        run_semantic_watermark_image_only_session,
    )

    archive_destination_dirs = None
    resume_checkpoint_dir = None
    if persistent_output_dir is not None:
        persistent_root = Path(persistent_output_dir).expanduser().resolve()
        archive_destination_dirs = {
            "image_only_dataset_runtime": persistent_root / "image_only_dataset_runtime",
            "dataset_level_quality": persistent_root / "dataset_level_quality",
        }
        if workflow_name == "mechanism_ablation":
            archive_destination_dirs["runtime_rerun_ablation"] = (
                persistent_root / "runtime_rerun_ablation"
            )
        resume_checkpoint_dir = persistent_root / "semantic_watermark_resume_checkpoint"
    previous_calibration_only = os.environ.get("SLM_WM_CALIBRATION_ONLY")
    previous_registry_digest = os.environ.get(
        "SLM_WM_CONTENT_ROUTING_REFERENCE_REGISTRY_DIGEST"
    )
    previous_registry_file_sha256 = os.environ.get(
        "SLM_WM_CONTENT_ROUTING_REFERENCE_REGISTRY_FILE_SHA256"
    )
    try:
        if calibration_only:
            os.environ["SLM_WM_CALIBRATION_ONLY"] = "1"
        else:
            os.environ.pop("SLM_WM_CALIBRATION_ONLY", None)
        os.environ["SLM_WM_CONTENT_ROUTING_REFERENCE_REGISTRY_DIGEST"] = (
            expected_reference_registry_digest
        )
        os.environ["SLM_WM_CONTENT_ROUTING_REFERENCE_REGISTRY_FILE_SHA256"] = (
            expected_reference_registry_file_sha256
        )
        session = run_semantic_watermark_image_only_session(
            root_path,
            run_formal_ablation=workflow_name == "mechanism_ablation",
            archive_destination_dirs=archive_destination_dirs,
            resume_checkpoint_dir=resume_checkpoint_dir,
        )
    finally:
        _restore_environment_value(
            "SLM_WM_CALIBRATION_ONLY", previous_calibration_only
        )
        _restore_environment_value(
            "SLM_WM_CONTENT_ROUTING_REFERENCE_REGISTRY_DIGEST",
            previous_registry_digest,
        )
        _restore_environment_value(
            "SLM_WM_CONTENT_ROUTING_REFERENCE_REGISTRY_FILE_SHA256",
            previous_registry_file_sha256,
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
    persistent_output_dir: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """通过共享持久化边界运行外部 workflow 并生成可交付结果包."""

    delivery_dir = (
        Path(persistent_output_dir).expanduser().resolve()
        if persistent_output_dir is not None
        else (
            root_path
            / "outputs"
            / "gpu_server_delivery"
            / paper_run_name
            / workflow_name
        )
    )
    from paper_experiments.runners.persistent_workflow_session import (
        run_persistent_workflow,
    )

    if route.official_reference_runner_name is not None:
        runner_name = route.official_reference_runner_name
        if runner_name == "tree_ring":
            from paper_experiments.runners.tree_ring_official_reference import (
                package_tree_ring_official_reference_outputs,
                run_default_tree_ring_official_reference_plan,
            )

            run_function = run_default_tree_ring_official_reference_plan
            package_function = package_tree_ring_official_reference_outputs
        elif runner_name == "gaussian_shading":
            from paper_experiments.runners.gaussian_shading_official_reference import (
                package_gaussian_shading_official_reference_outputs,
                run_default_gaussian_shading_official_reference_plan,
            )

            run_function = run_default_gaussian_shading_official_reference_plan
            package_function = package_gaussian_shading_official_reference_outputs
        else:
            from paper_experiments.runners.shallow_diffuse_official_reference import (
                package_shallow_diffuse_official_reference_outputs,
                run_default_shallow_diffuse_official_reference_plan,
            )

            run_function = run_default_shallow_diffuse_official_reference_plan
            package_function = package_shallow_diffuse_official_reference_outputs
        summary = run_persistent_workflow(
            root=root_path,
            workflow_name=workflow_name,
            baseline_id=route.baseline_id,
            persistent_output_dir=delivery_dir,
            runner=lambda: run_function(root=root_path),
        )
        archive_record = package_function(
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
    summary = run_persistent_workflow(
        root=root_path,
        workflow_name=isolated_workflow_name,
        baseline_id=route.baseline_id,
        persistent_output_dir=delivery_dir,
        runner=lambda: _run_shared_isolated_workflow(
            root=root_path,
            workflow_name=isolated_workflow_name,
        ),
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
    randomization_repeat_id: str | None,
    route: WorkflowRoute,
    execution_lock: Mapping[str, Any],
    orchestrator_environment: Mapping[str, Any],
    route_result: Mapping[str, Any],
) -> Dict[str, Any]:
    """把9条服务器路由收敛为同一受治理输出 schema."""

    workflow_summary = route_result.get("workflow_summary")
    archive_record = route_result.get("archive_record")
    workflow_environment = route_result.get("workflow_environment")
    return_code = route_result.get("return_code")
    stdout = route_result.get("stdout")
    stderr = route_result.get("stderr")
    if not all(
        (
            isinstance(workflow_summary, Mapping),
            isinstance(workflow_environment, Mapping),
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
        "randomization_scope": (
            "active_repeat_component"
            if randomization_repeat_id is not None
            else "cross_repeat_invariant"
        ),
        "randomization_repeat_id": randomization_repeat_id or "",
        "scientific_profile_id": route.scientific_profile_id,
        "baseline_id": route.baseline_id,
        "route_kind": route.route_kind,
        "child_argv_tail": list(route.child_argv_tail),
        "shared_isolated_workflow_name": route.shared_isolated_workflow_name,
        "official_reference_runner_name": route.official_reference_runner_name,
        "workflow_summary": dict(workflow_summary),
        "workflow_environment": dict(workflow_environment),
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
    persistent_output_dir: Optional[str | Path] = None,
    *,
    randomization_repeat_id: str | None = None,
    calibration_only: bool = False,
    expected_reference_registry_digest: str = "",
    expected_reference_registry_file_sha256: str = "",
) -> Dict[str, Any]:
    """在 CPU 父入口中发布执行身份并调度一个隔离 GPU 工作流."""

    if workflow_name not in WORKFLOW_ROUTES:
        raise ValueError("未知正式工作流: {0}".format(workflow_name))
    active_repeat_workflow = workflow_name in ACTIVE_REPEAT_GPU_WORKFLOW_NAMES
    if active_repeat_workflow:
        if not randomization_repeat_id:
            raise ValueError("活动随机化 GPU workflow 必须显式指定 repeat ID")
        resolved_repeat_id = resolve_formal_randomization_repeat(
            randomization_repeat_id
        ).randomization_repeat_id
        if (
            persistent_output_dir is not None
            and str(persistent_output_dir).strip()
            and workflow_name != "image_only_dataset"
        ):
            raise ValueError(
                "活动随机化 GPU workflow 的持久化根必须由受治理 repeat 配置生成, "
                "不得显式覆盖"
            )
    else:
        if randomization_repeat_id:
            raise ValueError("跨 repeat 不变 GPU workflow 不得绑定活动 repeat ID")
        resolved_repeat_id = None
    if type(calibration_only) is not bool:
        raise TypeError("calibration_only must be an exact bool")
    if calibration_only and workflow_name != "image_only_dataset":
        raise ValueError("calibration-only is valid only for image_only_dataset")
    if workflow_name != "image_only_dataset" and (
        expected_reference_registry_digest
        or expected_reference_registry_file_sha256
    ):
        raise ValueError(
            "fixed content-routing registry identity is valid only for "
            "image_only_dataset"
        )
    if workflow_name == "image_only_dataset" and (
        not _is_sha256(expected_reference_registry_digest)
        or not _is_sha256(expected_reference_registry_file_sha256)
    ):
        raise ValueError("image_only_dataset requires fixed registry dual SHA")
    route = WORKFLOW_ROUTES[workflow_name]
    root_path = Path(root).resolve()
    orchestrator_environment = _require_workflow_orchestrator_environment(
        root_path
    )
    execution_lock = build_formal_execution_lock(root_path, repository_commit)
    with _published_workflow_environment(
        execution_lock,
        paper_run_name,
        resolved_repeat_id,
        route.baseline_id,
    ):
        paper_run = build_paper_run_config(root_path)
        if paper_run.run_name != paper_run_name or (
            active_repeat_workflow
            and paper_run.randomization_repeat_id != resolved_repeat_id
        ):
            raise ValueError("GPU workflow 与当前论文运行及 repeat 配置不一致")
        workflow_environment, resolved_persistent_output_dir = (
            _configure_workflow_execution_environment(
                workflow_name=workflow_name,
                root_path=root_path,
                persistent_output_dir=persistent_output_dir,
            )
        )
        workflow_environment = {
            **workflow_environment,
            "randomization_scope": (
                "active_repeat_component"
                if active_repeat_workflow
                else "cross_repeat_invariant"
            ),
            "randomization_repeat_id": resolved_repeat_id or "",
            "calibration_only": calibration_only,
            "content_routing_reference_registry_digest": (
                expected_reference_registry_digest
            ),
            "content_routing_reference_registry_file_sha256": (
                expected_reference_registry_file_sha256
            ),
        }
        route_result = (
            _run_main_method_route(
                route=route,
                workflow_name=workflow_name,
                paper_run_name=paper_run_name,
                root_path=root_path,
                persistent_output_dir=resolved_persistent_output_dir,
                calibration_only=calibration_only,
                expected_reference_registry_digest=(
                    expected_reference_registry_digest
                ),
                expected_reference_registry_file_sha256=(
                    expected_reference_registry_file_sha256
                ),
            )
            if route.uses_scientific_command
            else _run_shared_route(
                route=route,
                root_path=root_path,
                workflow_name=workflow_name,
                paper_run_name=paper_run_name,
                persistent_output_dir=resolved_persistent_output_dir,
            )
        )
        route_result = {
            **dict(route_result),
            "workflow_environment": workflow_environment,
        }
    return _build_workflow_result(
        workflow_name=workflow_name,
        paper_run_name=paper_run_name,
        randomization_repeat_id=resolved_repeat_id,
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
        "--randomization-repeat-id",
        default="",
        help="活动随机化 workflow 必须显式指定的 seed-key repeat ID.",
    )
    parser.add_argument(
        "--repository-commit",
        required=True,
        help="正式执行使用的精确40位小写 Git SHA.",
    )
    parser.add_argument("--root", default=".")
    parser.add_argument(
        "--persistent-output-dir",
        default=None,
        help=(
            "仅跨 repeat 不变路由可显式指定的持久目录; 活动 repeat 路由必须"
            "使用受治理配置派生的隔离目录."
        ),
    )
    parser.add_argument("--calibration-only", action="store_true")
    parser.add_argument("--expected-reference-registry-digest", default="")
    parser.add_argument(
        "--expected-reference-registry-file-sha256", default=""
    )
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
                args.persistent_output_dir,
                randomization_repeat_id=(
                    args.randomization_repeat_id or None
                ),
                calibration_only=args.calibration_only,
                expected_reference_registry_digest=(
                    args.expected_reference_registry_digest
                ),
                expected_reference_registry_file_sha256=(
                    args.expected_reference_registry_file_sha256
                ),
            ),
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
