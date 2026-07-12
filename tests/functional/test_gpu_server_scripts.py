"""验证无 Notebook GPU 服务器入口只调度隔离科学子进程."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Optional

import pytest

from scripts import run_gpu_server_workflow as workflow
from scripts import semantic_watermark_scientific_workflow as method_session


COMMIT = "a" * 40
LOCK_DIGEST = "b" * 64
ORCHESTRATOR_EVIDENCE = {
    "profile_id": workflow.WORKFLOW_ORCHESTRATOR_PROFILE_ID,
    "profile_digest": "1" * 64,
    "direct_requirements_digest": "2" * 64,
    "complete_hash_lock_digest": "3" * 64,
    "complete_hash_lock_dependency_count": 7,
    "inspection_digest": "4" * 64,
    "inspection": {"decision": "pass"},
}
TEST_PERSISTENT_ROOT = Path("outputs/pytest_server_persistent").resolve()
PUBLIC_WORKFLOW_EXPECTATIONS = {
    "image_only_dataset": (
        workflow.MAIN_METHOD_PROFILE_ID,
        None,
        ("-m", workflow.MAIN_METHOD_SESSION_MODULE),
    ),
    "mechanism_ablation": (
        workflow.MAIN_METHOD_PROFILE_ID,
        None,
        (
            "-m",
            workflow.MAIN_METHOD_SESSION_MODULE,
            "--run-formal-ablation",
        ),
    ),
    "external_baseline_tree_ring": (
        workflow.MAIN_METHOD_PROFILE_ID,
        "tree_ring",
        (),
    ),
    "external_baseline_gaussian_shading": (
        workflow.MAIN_METHOD_PROFILE_ID,
        "gaussian_shading",
        (),
    ),
    "external_baseline_shallow_diffuse": (
        workflow.MAIN_METHOD_PROFILE_ID,
        "shallow_diffuse",
        (),
    ),
    "official_reference_t2smark": (
        workflow.T2SMARK_PROFILE_ID,
        None,
        (),
    ),
    "official_reference_tree_ring": (
        workflow.TREE_RING_OFFICIAL_PROFILE_ID,
        None,
        (),
    ),
    "official_reference_gaussian_shading": (
        workflow.GAUSSIAN_SHADING_OFFICIAL_PROFILE_ID,
        None,
        (),
    ),
    "official_reference_shallow_diffuse": (
        workflow.SHALLOW_DIFFUSE_OFFICIAL_PROFILE_ID,
        None,
        (),
    ),
}


def _patch_formal_lock(monkeypatch: pytest.MonkeyPatch) -> Mapping[str, Any]:
    """用轻量执行锁替代真实 detached Git 查询并模拟统一发布 API."""

    execution_lock = {
        "formal_execution_commit": COMMIT,
        "formal_execution_lock_digest": LOCK_DIGEST,
        "formal_execution_lock_ready": True,
    }

    def fake_publish(record: Mapping[str, Any]) -> Mapping[str, Any]:
        os.environ[workflow.FORMAL_EXECUTION_COMMIT_ENVIRONMENT_KEY] = str(
            record["formal_execution_commit"]
        )
        os.environ[workflow.FORMAL_EXECUTION_LOCK_DIGEST_ENVIRONMENT_KEY] = str(
            record["formal_execution_lock_digest"]
        )
        return record

    monkeypatch.setattr(
        workflow,
        "build_formal_execution_lock",
        lambda root, expected_commit: execution_lock,
    )
    monkeypatch.setattr(workflow, "publish_formal_execution_lock", fake_publish)
    monkeypatch.setattr(
        workflow,
        "_require_workflow_orchestrator_environment",
        lambda root: dict(ORCHESTRATOR_EVIDENCE),
    )

    def fake_configure_environment(
        workflow_name: str,
        *,
        baseline_id: str,
        repository_root: Path,
    ) -> dict[str, Any]:
        for environment_key in set(
            workflow.WORKFLOW_PERSISTENT_ENVIRONMENT_KEYS.values()
        ):
            os.environ[environment_key] = str(TEST_PERSISTENT_ROOT)
        return {
            "workflow_name": workflow_name,
            "paper_run_name": os.environ[workflow.PAPER_RUN_NAME_ENVIRONMENT_KEY],
            "selected_baseline_id": baseline_id,
            "repository_root": repository_root.as_posix(),
        }

    monkeypatch.setattr(
        workflow,
        "configure_formal_workflow_environment",
        fake_configure_environment,
    )
    return execution_lock


@pytest.mark.quick
def test_server_workflow_exposes_complete_isolated_gpu_routes() -> None:
    """服务器入口公开主方法、消融和全部7个 baseline 工作流."""

    assert set(workflow.WORKFLOW_ROUTES) == set(PUBLIC_WORKFLOW_EXPECTATIONS)
    for workflow_name, expected in PUBLIC_WORKFLOW_EXPECTATIONS.items():
        route = workflow.WORKFLOW_ROUTES[workflow_name]
        expected_profile, expected_baseline, expected_tail = expected
        assert route.scientific_profile_id == expected_profile
        assert route.baseline_id == expected_baseline
        assert route.child_argv_tail == expected_tail


@pytest.mark.quick
@pytest.mark.parametrize("workflow_name", tuple(PUBLIC_WORKFLOW_EXPECTATIONS))
def test_server_workflow_configures_inner_environment_and_persistence(
    workflow_name: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """每条服务器路由应在 scripts 层完成配置并发布显式持久化目录."""

    captured: dict[str, Any] = {}

    def fake_configure(
        configuration_name: str,
        *,
        baseline_id: str,
        repository_root: Path,
    ) -> dict[str, Any]:
        captured.update(
            configuration_name=configuration_name,
            baseline_id=baseline_id,
            repository_root=repository_root,
        )
        return {"configuration_ready": True}

    monkeypatch.setattr(
        workflow,
        "configure_formal_workflow_environment",
        fake_configure,
    )
    persistent_root = tmp_path / "persistent"
    environment_key = workflow.WORKFLOW_PERSISTENT_ENVIRONMENT_KEYS[workflow_name]
    monkeypatch.setenv(environment_key, "outer_persistent_value")

    environment_record, resolved_path = (
        workflow._configure_workflow_execution_environment(
            workflow_name=workflow_name,
            root_path=tmp_path,
            persistent_output_dir=persistent_root,
        )
    )

    expected_configuration, expected_baseline = (
        workflow.WORKFLOW_ENVIRONMENT_CONFIGURATION[workflow_name]
    )
    assert captured == {
        "configuration_name": expected_configuration,
        "baseline_id": expected_baseline,
        "repository_root": tmp_path,
    }
    assert resolved_path == persistent_root
    assert os.environ[environment_key] == str(persistent_root)
    assert environment_record["persistent_environment_key"] == environment_key
    assert environment_record["persistent_output_dir"] == str(persistent_root)


@pytest.mark.quick
@pytest.mark.parametrize("workflow_name", tuple(PUBLIC_WORKFLOW_EXPECTATIONS))
def test_server_cli_maps_every_public_workflow_option(workflow_name: str) -> None:
    """CLI 参数必须无歧义映射到全部公开隔离工作流."""

    root = "repository-root"
    arguments = workflow.build_parser().parse_args(
        [
            "--workflow",
            workflow_name,
            "--paper-run-name",
            "probe_paper",
            "--repository-commit",
            COMMIT,
            "--root",
            root,
        ]
    )

    assert arguments.workflow == workflow_name
    assert arguments.paper_run_name == "probe_paper"
    assert arguments.repository_commit == COMMIT
    assert arguments.root == root


@pytest.mark.quick
def test_server_cli_accepts_external_persistent_output_directory() -> None:
    """独立 GPU 服务器必须能够显式选择挂载盘持久化目录."""

    arguments = workflow.build_parser().parse_args(
        [
            "--workflow",
            "official_reference_t2smark",
            "--paper-run-name",
            "probe_paper",
            "--repository-commit",
            COMMIT,
            "--persistent-output-dir",
            "/mnt/persistent/slm_wm/t2smark",
        ]
    )

    assert arguments.persistent_output_dir == "/mnt/persistent/slm_wm/t2smark"


@pytest.mark.quick
def test_orchestrator_gate_accepts_exact_ready_cpu_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """父编排器必须绑定 workflow_orchestrator 完整锁与当前解释器 inspection."""

    profile = SimpleNamespace(
        profile_name=workflow.WORKFLOW_ORCHESTRATOR_PROFILE_ID,
        execution_role="workflow_orchestration",
        accelerator_runtime="cpu",
        pytorch_index_url=None,
        profile_digest="1" * 64,
        direct_requirements_digest="2" * 64,
        complete_hash_lock_digest="3" * 64,
        complete_hash_lock_dependency_count=7,
    )
    inspection = {
        "profile_name": workflow.WORKFLOW_ORCHESTRATOR_PROFILE_ID,
        "profile_digest": profile.profile_digest,
        "complete_hash_lock_digest": profile.complete_hash_lock_digest,
        "profile_formal_ready": True,
        "environment_match": True,
        "mismatches": [],
        "readiness_blockers": [],
        "decision": "pass",
        "inspection_digest": "4" * 64,
    }
    registry_path = tmp_path / "configs/dependency_profile_registry.json"

    def require_profile(profile_id: str, path: Path) -> SimpleNamespace:
        assert profile_id == workflow.WORKFLOW_ORCHESTRATOR_PROFILE_ID
        assert path == registry_path
        return profile

    def inspect_profile(profile_id: str, *, path: Path) -> dict[str, Any]:
        assert profile_id == workflow.WORKFLOW_ORCHESTRATOR_PROFILE_ID
        assert path == registry_path
        return inspection

    monkeypatch.setattr(workflow, "require_dependency_profile_ready", require_profile)
    monkeypatch.setattr(
        workflow,
        "inspect_dependency_profile_environment",
        inspect_profile,
    )

    evidence = workflow._require_workflow_orchestrator_environment(tmp_path)

    assert evidence == {
        "profile_id": workflow.WORKFLOW_ORCHESTRATOR_PROFILE_ID,
        "profile_digest": "1" * 64,
        "direct_requirements_digest": "2" * 64,
        "complete_hash_lock_digest": "3" * 64,
        "complete_hash_lock_dependency_count": 7,
        "inspection_digest": "4" * 64,
        "inspection": inspection,
    }


@pytest.mark.quick
def test_orchestrator_gate_rejects_mismatched_environment_before_dispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """父解释器未通过正式 profile 时不得构造执行锁或进入任一科学路由."""

    profile = SimpleNamespace(
        profile_name=workflow.WORKFLOW_ORCHESTRATOR_PROFILE_ID,
        execution_role="workflow_orchestration",
        accelerator_runtime="cpu",
        pytorch_index_url=None,
        profile_digest="1" * 64,
        direct_requirements_digest="2" * 64,
        complete_hash_lock_digest="3" * 64,
        complete_hash_lock_dependency_count=7,
    )
    monkeypatch.setattr(
        workflow,
        "require_dependency_profile_ready",
        lambda profile_id, path: profile,
    )
    monkeypatch.setattr(
        workflow,
        "inspect_dependency_profile_environment",
        lambda profile_id, *, path: {
            "profile_name": workflow.WORKFLOW_ORCHESTRATOR_PROFILE_ID,
            "profile_digest": profile.profile_digest,
            "complete_hash_lock_digest": profile.complete_hash_lock_digest,
            "profile_formal_ready": True,
            "environment_match": False,
            "mismatches": ["locked_dependency_missing:uv"],
            "readiness_blockers": ["locked_dependency_missing:uv"],
            "decision": "blocked",
            "inspection_digest": "4" * 64,
        },
    )
    monkeypatch.setattr(
        workflow,
        "build_formal_execution_lock",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("门禁失败后不得构造执行锁")
        ),
    )

    with pytest.raises(RuntimeError, match="正式依赖门禁"):
        workflow.run_workflow(
            "image_only_dataset",
            "probe_paper",
            COMMIT,
            tmp_path,
        )


@pytest.mark.quick
def test_all_nine_routes_emit_one_output_schema(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """9条公开路由必须返回完全相同的字段集合与稳定类型."""

    _patch_formal_lock(monkeypatch)

    def fake_main(**kwargs: Any) -> dict[str, Any]:
        return {
            "workflow_summary": {"decision": "pass"},
            "archive_record": None,
            "return_code": 0,
            "stdout": "",
            "stderr": "",
        }

    def fake_shared(**kwargs: Any) -> dict[str, Any]:
        return {
            "workflow_summary": {"decision": "pass"},
            "archive_record": {"archive_path": "archive.zip"},
            "return_code": 0,
            "stdout": "",
            "stderr": "",
        }

    monkeypatch.setattr(workflow, "_run_main_method_route", fake_main)
    monkeypatch.setattr(workflow, "_run_shared_route", fake_shared)
    results = [
        workflow.run_workflow(name, "probe_paper", COMMIT, tmp_path)
        for name in PUBLIC_WORKFLOW_EXPECTATIONS
    ]
    expected_fields = set(results[0])
    assert all(set(result) == expected_fields for result in results)
    assert len(results) == 9
    for result in results:
        route = workflow.WORKFLOW_ROUTES[result["workflow_name"]]
        assert result["report_schema"] == workflow.RESULT_SCHEMA
        assert result["schema_version"] == workflow.RESULT_SCHEMA_VERSION
        assert result["operation_kind"] == workflow.RESULT_OPERATION_KIND
        assert result["route_kind"] == route.route_kind
        assert result["child_argv_tail"] == list(route.child_argv_tail)
        assert result["shared_isolated_workflow_name"] == (
            route.shared_isolated_workflow_name
        )
        assert result["official_reference_runner_name"] == (
            route.official_reference_runner_name
        )
        assert result["workflow_environment"]["persistent_output_dir"] == str(
            TEST_PERSISTENT_ROOT
        )
        assert result["decision"] == "pass"
        assert result["supports_paper_claim"] is False
        assert result["orchestrator_dependency_environment"] == (
            ORCHESTRATOR_EVIDENCE
        )


@pytest.mark.quick
def test_server_rejects_incomplete_internal_route_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """内层路由缺少摘要或返回码类型错误时不得生成表面完整结果."""

    _patch_formal_lock(monkeypatch)
    monkeypatch.setattr(
        workflow,
        "_run_main_method_route",
        lambda **kwargs: {
            "archive_record": None,
            "return_code": False,
            "stdout": "",
            "stderr": "",
        },
    )

    with pytest.raises(RuntimeError, match="统一结果"):
        workflow.run_workflow(
            "image_only_dataset",
            "probe_paper",
            COMMIT,
            tmp_path,
        )


@pytest.mark.quick
@pytest.mark.parametrize(
    ("workflow_name", "expected_tail"),
    (
        (
            "image_only_dataset",
            ("-m", workflow.MAIN_METHOD_SESSION_MODULE),
        ),
        (
            "mechanism_ablation",
            (
                "-m",
                workflow.MAIN_METHOD_SESSION_MODULE,
                "--run-formal-ablation",
            ),
        ),
    ),
)
def test_main_method_routes_use_isolated_scientific_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    workflow_name: str,
    expected_tail: tuple[str, ...],
) -> None:
    """主方法和消融必须复用同一会话脚本并进入受治理子解释器."""

    execution_lock = _patch_formal_lock(monkeypatch)
    monkeypatch.setenv(workflow.PRIMARY_BASELINE_ID_ENVIRONMENT_KEY, "outer_baseline")
    captured: dict[str, Any] = {}

    def fake_session(
        root: Path,
        *,
        run_formal_ablation: bool,
        archive_destination_dirs: object = None,
        resume_checkpoint_dir: object = None,
    ) -> dict[str, Any]:
        captured.update(
            {
                "root": root,
                "run_formal_ablation": run_formal_ablation,
                "archive_destination_dirs": archive_destination_dirs,
                "resume_checkpoint_dir": resume_checkpoint_dir,
                "paper_run_name": os.environ[workflow.PAPER_RUN_NAME_ENVIRONMENT_KEY],
                "baseline_id": os.environ.get(
                    workflow.PRIMARY_BASELINE_ID_ENVIRONMENT_KEY
                ),
                "formal_commit": os.environ[
                    workflow.FORMAL_EXECUTION_COMMIT_ENVIRONMENT_KEY
                ],
                "formal_digest": os.environ[
                    workflow.FORMAL_EXECUTION_LOCK_DIGEST_ENVIRONMENT_KEY
                ],
            }
        )
        return {
            "workflow_decision": "complete",
            "local_archives": {"image_only_dataset_runtime": "archive.zip"},
        }

    def reject_shared_route(**kwargs: Any) -> Mapping[str, Any]:
        raise AssertionError("主方法不得调用 baseline / T2SMark 共享包装")

    monkeypatch.setattr(
        method_session,
        "run_semantic_watermark_image_only_session",
        fake_session,
    )
    monkeypatch.setattr(workflow, "_run_shared_isolated_workflow", reject_shared_route)

    result = workflow.run_workflow(
        workflow_name,
        "probe_paper",
        COMMIT,
        tmp_path,
    )

    assert captured["root"] == tmp_path.resolve()
    assert captured["run_formal_ablation"] is (
        workflow_name == "mechanism_ablation"
    )
    expected_archive_dirs = {
        "image_only_dataset_runtime": (
            TEST_PERSISTENT_ROOT / "image_only_dataset_runtime"
        ),
        "dataset_level_quality": TEST_PERSISTENT_ROOT / "dataset_level_quality",
    }
    if workflow_name == "mechanism_ablation":
        expected_archive_dirs["runtime_rerun_ablation"] = (
            TEST_PERSISTENT_ROOT / "runtime_rerun_ablation"
        )
    assert captured["archive_destination_dirs"] == expected_archive_dirs
    assert captured["resume_checkpoint_dir"] == (
        TEST_PERSISTENT_ROOT / "semantic_watermark_resume_checkpoint"
    )
    assert captured["paper_run_name"] == "probe_paper"
    assert captured["baseline_id"] is None
    assert captured["formal_commit"] == COMMIT
    assert captured["formal_digest"] == LOCK_DIGEST
    assert result["formal_execution_lock"] == execution_lock
    assert result["workflow_environment"]["persistent_output_dir"] == str(
        TEST_PERSISTENT_ROOT
    )
    assert result["orchestrator_dependency_environment"] == ORCHESTRATOR_EVIDENCE
    assert result["scientific_profile_id"] == workflow.MAIN_METHOD_PROFILE_ID
    assert result["child_argv_tail"] == list(expected_tail)
    assert result["command"] == []
    assert result["return_code"] == 0
    assert result["workflow_summary"]["workflow_decision"] == "complete"
    assert os.environ[workflow.PRIMARY_BASELINE_ID_ENVIRONMENT_KEY] == "outer_baseline"


@pytest.mark.quick
def test_main_method_route_maps_persistent_archives_and_resume_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """主方法持久根目录必须同时覆盖完成归档与仅供续跑的 checkpoint."""

    captured: dict[str, Any] = {}

    def fake_session(root: Path, **kwargs: Any) -> dict[str, Any]:
        captured.update(root=root, **kwargs)
        return {"workflow_decision": "complete"}

    monkeypatch.setattr(
        method_session,
        "run_semantic_watermark_image_only_session",
        fake_session,
    )
    persistent_root = tmp_path / "persistent"
    result = workflow._run_main_method_route(
        route=workflow.WORKFLOW_ROUTES["mechanism_ablation"],
        workflow_name="mechanism_ablation",
        paper_run_name="probe_paper",
        root_path=tmp_path,
        persistent_output_dir=persistent_root,
    )

    assert captured["archive_destination_dirs"] == {
        "image_only_dataset_runtime": persistent_root / "image_only_dataset_runtime",
        "dataset_level_quality": persistent_root / "dataset_level_quality",
        "runtime_rerun_ablation": persistent_root / "runtime_rerun_ablation",
    }
    assert captured["resume_checkpoint_dir"] == (
        persistent_root / "semantic_watermark_resume_checkpoint"
    )
    assert result["workflow_summary"]["workflow_decision"] == "complete"


@pytest.mark.quick
@pytest.mark.parametrize(
    ("workflow_name", "expected_profile", "expected_baseline"),
    (
        (
            "external_baseline_tree_ring",
            workflow.MAIN_METHOD_PROFILE_ID,
            "tree_ring",
        ),
        (
            "external_baseline_gaussian_shading",
            workflow.MAIN_METHOD_PROFILE_ID,
            "gaussian_shading",
        ),
        (
            "external_baseline_shallow_diffuse",
            workflow.MAIN_METHOD_PROFILE_ID,
            "shallow_diffuse",
        ),
        (
            "official_reference_t2smark",
            workflow.T2SMARK_PROFILE_ID,
            None,
        ),
        (
            "official_reference_tree_ring",
            workflow.TREE_RING_OFFICIAL_PROFILE_ID,
            None,
        ),
        (
            "official_reference_gaussian_shading",
            workflow.GAUSSIAN_SHADING_OFFICIAL_PROFILE_ID,
            None,
        ),
        (
            "official_reference_shallow_diffuse",
            workflow.SHALLOW_DIFFUSE_OFFICIAL_PROFILE_ID,
            None,
        ),
    ),
)
def test_baseline_and_t2smark_routes_use_shared_isolated_wrapper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    workflow_name: str,
    expected_profile: str,
    expected_baseline: Optional[str],
) -> None:
    """baseline 与 T2SMark 只调用共享隔离包装并绑定唯一 baseline 身份."""

    execution_lock = _patch_formal_lock(monkeypatch)
    monkeypatch.setenv(workflow.PRIMARY_BASELINE_ID_ENVIRONMENT_KEY, "outer_baseline")
    captured: dict[str, Any] = {}

    def fake_shared_route(
        *,
        route: workflow.WorkflowRoute,
        root_path: Path,
        workflow_name: str,
        paper_run_name: str,
        persistent_output_dir: Optional[str | Path] = None,
    ) -> dict[str, Any]:
        captured.update(
            {
                "root": root_path,
                "workflow_name": workflow_name,
                "paper_run_argument": paper_run_name,
                "route": route,
                "paper_run_name": os.environ[workflow.PAPER_RUN_NAME_ENVIRONMENT_KEY],
                "baseline_id": os.environ.get(
                    workflow.PRIMARY_BASELINE_ID_ENVIRONMENT_KEY
                ),
                "formal_commit": os.environ[
                    workflow.FORMAL_EXECUTION_COMMIT_ENVIRONMENT_KEY
                ],
                "formal_digest": os.environ[
                    workflow.FORMAL_EXECUTION_LOCK_DIGEST_ENVIRONMENT_KEY
                ],
                "persistent_output_dir": persistent_output_dir,
            }
        )
        return {
            "workflow_summary": {"decision": "pass"},
            "archive_record": {"archive_path": "archive.zip"},
            "return_code": 0,
            "stdout": "",
            "stderr": "",
        }

    monkeypatch.setattr(workflow, "_run_shared_route", fake_shared_route)

    result = workflow.run_workflow(
        workflow_name,
        "pilot_paper",
        COMMIT,
        tmp_path,
        tmp_path / "persistent_delivery",
    )

    assert captured["root"] == tmp_path.resolve()
    assert captured["workflow_name"] == workflow_name
    assert captured["paper_run_argument"] == "pilot_paper"
    assert captured["paper_run_name"] == "pilot_paper"
    assert captured["baseline_id"] == expected_baseline
    assert captured["formal_commit"] == COMMIT
    assert captured["formal_digest"] == LOCK_DIGEST
    assert captured["persistent_output_dir"] == tmp_path / "persistent_delivery"
    assert captured["route"] == workflow.WORKFLOW_ROUTES[workflow_name]
    assert result["scientific_profile_id"] == expected_profile
    assert result["baseline_id"] == expected_baseline
    assert result["formal_execution_lock"] == execution_lock
    assert result["return_code"] == 0
    assert result["orchestrator_dependency_environment"] == ORCHESTRATOR_EVIDENCE
    assert result["workflow_environment"]["persistent_output_dir"] == str(
        tmp_path / "persistent_delivery"
    )
    assert os.environ[workflow.PRIMARY_BASELINE_ID_ENVIRONMENT_KEY] == "outer_baseline"


@pytest.mark.quick
def test_gpu_server_parent_has_no_notebook_or_direct_scientific_runner_dependency() -> None:
    """服务器父入口只能依赖内层隔离 API, 不能导入 Notebook 或科学 runner."""

    sources = {
        path: Path(path).read_text(encoding="utf-8")
        for path in (
            "scripts/formal_workflow_entry.py",
            "scripts/formal_workflow_environment.py",
            "scripts/run_gpu_server_workflow.py",
        )
    }

    assert all("paper_workflow" not in source for source in sources.values())
    source = sources["scripts/run_gpu_server_workflow.py"]
    assert "run_image_only_dataset_runtime.py" not in source
    assert "run_runtime_rerun_ablations.py" not in source
    assert "subprocess.run" not in source
