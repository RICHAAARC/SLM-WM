"""验证7条外部 GPU workflow 共用同一持久化与重入协议."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
from types import SimpleNamespace
from typing import Any

import pytest

from experiments.protocol.paper_run_config import PaperRunConfig
from experiments.runtime import resume_checkpoint
from experiments.runtime.repository_environment import FORMAL_EXECUTION_LOCK_SCHEMA
from main.core.digest import build_stable_digest
from paper_experiments.runners import persistent_workflow_session as persistence
from paper_workflow.notebook_utils import notebook_entrypoint


COMMIT = "a" * 40


def _formal_lock() -> dict[str, Any]:
    """构造满足正式执行锁严格 schema 的测试记录."""

    payload = {
        "formal_execution_lock_schema": FORMAL_EXECUTION_LOCK_SCHEMA,
        "formal_execution_commit": COMMIT,
        "formal_execution_head_detached": True,
        "formal_execution_worktree_clean": True,
        "formal_execution_lock_ready": True,
    }
    return {
        **payload,
        "formal_execution_lock_digest": build_stable_digest(payload),
    }


def _paper_run() -> PaperRunConfig:
    """构造不依赖真实 Prompt 文件的轻量论文运行配置."""

    return PaperRunConfig(
        run_name="probe_paper",
        protocol_profile="paper_fixed_fpr_0_1",
        prompt_set="probe_paper",
        prompt_file="prompts/probe_paper_prompts.txt",
        prompt_count=70,
        sample_count=70,
        drive_result_root="/drive/probe_paper_results",
        target_fpr=0.1,
        minimum_clean_negative_count=34,
        dataset_level_quality_minimum_count=70,
    )


def _patch_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """避免轻量测试读取或安装真实科学依赖环境."""

    monkeypatch.setattr(
        persistence,
        "require_dependency_profile_ready",
        lambda profile_id, path: SimpleNamespace(
            profile_digest="1" * 64,
            complete_hash_lock_digest="2" * 64,
        ),
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """写出测试所需 JSON object."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_generation_publish_retries_transient_permission_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Windows 短暂目录句柄占用不得破坏不可变 generation 发布."""

    build_directory = tmp_path / ".new-generation"
    generation_directory = tmp_path / "published-generation"
    build_directory.mkdir()
    (build_directory / "checkpoint_manifest.json").write_text(
        "{}\n",
        encoding="utf-8",
    )
    real_replace = persistence.os.replace
    attempt_count = 0

    def transient_replace(source: Path, destination: Path) -> None:
        nonlocal attempt_count
        attempt_count += 1
        if attempt_count <= 2:
            raise PermissionError("模拟 Windows 短暂句柄占用")
        real_replace(source, destination)

    monkeypatch.setattr(persistence.os, "replace", transient_replace)
    monkeypatch.setattr(
        persistence,
        "_GENERATION_PUBLISH_RETRY_DELAYS_SECONDS",
        (0.0, 0.0),
    )

    persistence._publish_generation_directory(
        build_directory,
        generation_directory,
    )

    assert attempt_count == 3
    assert not build_directory.exists()
    assert (generation_directory / "checkpoint_manifest.json").is_file()


def _write_completed_route(
    root: Path,
    route: persistence.PersistentWorkflowRoute,
    *,
    paper_run: PaperRunConfig,
    formal_lock: dict[str, Any],
) -> dict[str, Any]:
    """按真实路由字段写出一个最小完成态文件集合."""

    output_root = route.output_root(root, paper_run.run_name)
    summary = {
        "run_decision": "pass",
        route.ready_field: True,
        route.summary_baseline_field: route.baseline_id,
        route.summary_paper_run_field: paper_run.run_name,
    }
    _write_json(output_root / route.summary_relative_path, summary)
    _write_json(
        output_root / route.manifest_relative_path,
        {
            "code_version": formal_lock["formal_execution_commit"],
            "formal_execution_run_lock": formal_lock,
        },
    )
    for relative_path in route.required_relative_paths:
        path = output_root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_text("{}\n", encoding="utf-8")
    return summary


@pytest.mark.quick
def test_registry_covers_exactly_seven_external_gpu_routes() -> None:
    """三条 method-faithful、T2SMark 和三条 official 必须全部受管."""

    assert persistence.persistent_route_names() == (
        "external_baseline_tree_ring",
        "external_baseline_gaussian_shading",
        "external_baseline_shallow_diffuse",
        "official_reference_t2smark",
        "official_reference_tree_ring",
        "official_reference_gaussian_shading",
        "official_reference_shallow_diffuse",
    )


@pytest.mark.quick
def test_notebook_thin_entry_passes_drive_identity_to_shared_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Notebook helper 只传参, 不维护另一套 checkpoint 实现."""

    captured: dict[str, Any] = {}
    marker = {"run_decision": "pass"}

    def fake_persistent_workflow(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return marker

    monkeypatch.setattr(
        persistence,
        "run_persistent_workflow",
        fake_persistent_workflow,
    )
    drive_dir = tmp_path / "drive"

    result = notebook_entrypoint.run_workflow(
        root=tmp_path,
        workflow_name="external_baseline_method_faithful",
        drive_output_dir=str(drive_dir),
        baseline_id="tree_ring",
        checkpoint_interval_seconds=15,
    )

    assert result is marker
    assert captured["root"] == tmp_path
    assert captured["workflow_name"] == "external_baseline_method_faithful"
    assert captured["baseline_id"] == "tree_ring"
    assert captured["persistent_output_dir"] == str(drive_dir)
    assert captured["checkpoint_interval_seconds"] == 15
    assert callable(captured["runner"])


@pytest.mark.quick
@pytest.mark.parametrize(
    "route_name",
    tuple(persistence.PERSISTENT_WORKFLOW_ROUTES),
)
def test_all_routes_use_their_real_completion_summary_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    route_name: str,
) -> None:
    """完成态重入必须读取每个 runner 的真实 baseline 与论文层级字段."""

    _patch_profile(monkeypatch)
    monkeypatch.setattr(
        persistence,
        "validate_scientific_execution_binding",
        lambda *args, **kwargs: {},
    )
    route = persistence.PERSISTENT_WORKFLOW_ROUTES[route_name]
    paper_run = _paper_run()
    formal_lock = _formal_lock()
    expected_summary = _write_completed_route(
        tmp_path,
        route,
        paper_run=paper_run,
        formal_lock=formal_lock,
    )
    session = persistence.PersistentWorkflowSession(
        repository_root=tmp_path,
        persistent_output_dir=tmp_path / "drive",
        route=route,
        paper_run=paper_run,
        formal_execution_lock=formal_lock,
        interval_seconds=0,
    )

    assert session.validate_completed_output() == expected_summary


@pytest.mark.quick
def test_running_checkpoint_restores_only_owned_method_faithful_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """共享输出 family 恢复时不得覆盖另一条 baseline 的文件."""

    _patch_profile(monkeypatch)
    paper_run = _paper_run()
    formal_lock = _formal_lock()
    route = persistence.PERSISTENT_WORKFLOW_ROUTES[
        "external_baseline_tree_ring"
    ]
    session = persistence.PersistentWorkflowSession(
        repository_root=tmp_path,
        persistent_output_dir=tmp_path / "drive",
        route=route,
        paper_run=paper_run,
        formal_execution_lock=formal_lock,
        interval_seconds=0,
    )
    output_root = route.output_root(tmp_path, paper_run.run_name)
    owned_path = output_root / "run_records/tree_ring/progress.json"
    other_path = output_root / "run_records/gaussian_shading/progress.json"
    owned_path.parent.mkdir(parents=True, exist_ok=True)
    other_path.parent.mkdir(parents=True, exist_ok=True)
    owned_path.write_text("owned", encoding="utf-8")
    other_path.write_text("other", encoding="utf-8")

    manifest = session.snapshot("running")
    owned_path.write_text("stale", encoding="utf-8")
    session.restore_latest()

    assert owned_path.read_text(encoding="utf-8") == "owned"
    assert other_path.read_text(encoding="utf-8") == "other"
    assert manifest["supports_paper_claim"] is False
    assert manifest["checkpoint_state_intermediate"] == "running"
    assert all(
        "gaussian_shading" not in record["path"]
        for record in manifest["checkpoint_file_records_intermediate"]
    )


@pytest.mark.quick
def test_checkpoint_tampering_is_rejected_before_restore(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Drive payload 任一字节变化都必须 fail-closed."""

    _patch_profile(monkeypatch)
    paper_run = _paper_run()
    route = persistence.PERSISTENT_WORKFLOW_ROUTES[
        "official_reference_tree_ring"
    ]
    session = persistence.PersistentWorkflowSession(
        repository_root=tmp_path,
        persistent_output_dir=tmp_path / "drive",
        route=route,
        paper_run=paper_run,
        formal_execution_lock=_formal_lock(),
        interval_seconds=0,
    )
    output_path = route.output_root(
        tmp_path,
        paper_run.run_name,
    ) / "partial.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("original", encoding="utf-8")
    manifest = session.snapshot("interrupted")
    generation = manifest["checkpoint_generation_intermediate"]
    file_record = manifest["checkpoint_file_records_intermediate"][0]
    payload_path = (
        session.generations_directory
        / generation[:16]
        / persistence.CHECKPOINT_PAYLOAD_DIRECTORY_NAME
        / file_record["payload_name_intermediate"]
    )
    payload_path.write_text("tampered", encoding="utf-8")

    with pytest.raises(persistence.WorkflowCheckpointError, match="摘要失配"):
        session.restore_latest()


@pytest.mark.quick
def test_completed_route_reentry_uses_inner_checkpoint_and_skips_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """完成单元由 experiments 层通用原语恢复, 第二次不得重跑科学 runner."""

    _patch_profile(monkeypatch)
    formal_lock = _formal_lock()
    paper_run = _paper_run()
    monkeypatch.setattr(
        persistence,
        "require_published_formal_execution_lock",
        lambda root: formal_lock,
    )
    monkeypatch.setattr(
        resume_checkpoint,
        "require_published_formal_execution_lock",
        lambda root: formal_lock,
    )
    monkeypatch.setattr(
        persistence,
        "build_paper_run_config",
        lambda root: paper_run,
    )
    route = persistence.PERSISTENT_WORKFLOW_ROUTES[
        "official_reference_tree_ring"
    ]
    calls = 0

    def runner() -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return _write_completed_route(
            tmp_path,
            route,
            paper_run=paper_run,
            formal_lock=formal_lock,
        )

    first = persistence.run_persistent_workflow(
        root=tmp_path,
        workflow_name=route.workflow_name,
        runner=runner,
        persistent_output_dir=tmp_path / "drive",
        checkpoint_interval_seconds=0,
    )
    assert first[route.ready_field] is True
    assert calls == 1
    local_output_root = route.output_root(tmp_path, paper_run.run_name)
    shutil.rmtree(local_output_root)
    stale_path = local_output_root / "stale_from_disconnected_session.json"
    stale_path.parent.mkdir(parents=True, exist_ok=True)
    stale_path.write_text("stale", encoding="utf-8")

    second = persistence.run_persistent_workflow(
        root=tmp_path,
        workflow_name=route.workflow_name,
        runner=lambda: (_ for _ in ()).throw(
            AssertionError("完成 checkpoint 重入不得调用 runner")
        ),
        persistent_output_dir=tmp_path / "drive",
        checkpoint_interval_seconds=0,
    )

    assert second == first
    assert calls == 1
    assert route.summary_path(tmp_path, paper_run.run_name).is_file()
    assert not stale_path.exists()


@pytest.mark.quick
def test_tampered_completed_checkpoint_preserves_local_valid_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """外部完成快照损坏时必须在清理本地结果前失败."""

    _patch_profile(monkeypatch)
    formal_lock = _formal_lock()
    paper_run = _paper_run()
    monkeypatch.setattr(
        persistence,
        "require_published_formal_execution_lock",
        lambda root: formal_lock,
    )
    monkeypatch.setattr(
        resume_checkpoint,
        "require_published_formal_execution_lock",
        lambda root: formal_lock,
    )
    monkeypatch.setattr(
        persistence,
        "build_paper_run_config",
        lambda root: paper_run,
    )
    route = persistence.PERSISTENT_WORKFLOW_ROUTES[
        "official_reference_shallow_diffuse"
    ]

    persistence.run_persistent_workflow(
        root=tmp_path,
        workflow_name=route.workflow_name,
        runner=lambda: _write_completed_route(
            tmp_path,
            route,
            paper_run=paper_run,
            formal_lock=formal_lock,
        ),
        persistent_output_dir=tmp_path / "drive",
        checkpoint_interval_seconds=0,
    )
    local_marker = route.output_root(
        tmp_path,
        paper_run.run_name,
    ) / "local_valid_marker.json"
    local_marker.write_text("keep", encoding="utf-8")
    completed_manifest_path = next(
        (tmp_path / "drive" / "cp").rglob(
            resume_checkpoint.CHECKPOINT_MANIFEST_FILE_NAME
        )
    )
    completed_manifest = json.loads(
        completed_manifest_path.read_text(encoding="utf-8")
    )
    payload_name = completed_manifest["entry_records"][0][
        "payload_name_intermediate"
    ]
    (completed_manifest_path.parent / "payload" / payload_name).write_text(
        "tampered",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="payload 摘要不一致"):
        persistence.run_persistent_workflow(
            root=tmp_path,
            workflow_name=route.workflow_name,
            runner=lambda: (_ for _ in ()).throw(
                AssertionError("损坏完成快照不得进入 runner")
            ),
            persistent_output_dir=tmp_path / "drive",
            checkpoint_interval_seconds=0,
        )

    assert local_marker.read_text(encoding="utf-8") == "keep"
    assert route.summary_path(tmp_path, paper_run.run_name).is_file()


@pytest.mark.quick
def test_interrupted_checkpoint_never_skips_runner_or_supports_claim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """中断态只恢复稳定文件, 不得被解释为完成态论文证据."""

    _patch_profile(monkeypatch)
    formal_lock = _formal_lock()
    paper_run = _paper_run()
    monkeypatch.setattr(
        persistence,
        "require_published_formal_execution_lock",
        lambda root: formal_lock,
    )
    monkeypatch.setattr(
        resume_checkpoint,
        "require_published_formal_execution_lock",
        lambda root: formal_lock,
    )
    monkeypatch.setattr(
        persistence,
        "build_paper_run_config",
        lambda root: paper_run,
    )
    route = persistence.PERSISTENT_WORKFLOW_ROUTES[
        "official_reference_gaussian_shading"
    ]
    partial_path = route.output_root(
        tmp_path,
        paper_run.run_name,
    ) / "partial.json"

    def interrupted_runner() -> dict[str, Any]:
        partial_path.parent.mkdir(parents=True, exist_ok=True)
        partial_path.write_text("stable partial", encoding="utf-8")
        raise RuntimeError("模拟 Colab 中断")

    with pytest.raises(RuntimeError, match="模拟 Colab 中断"):
        persistence.run_persistent_workflow(
            root=tmp_path,
            workflow_name=route.workflow_name,
            runner=interrupted_runner,
            persistent_output_dir=tmp_path / "drive",
            checkpoint_interval_seconds=0,
        )
    pointer_paths = tuple(
        (tmp_path / "drive").rglob(persistence.CHECKPOINT_POINTER_FILE_NAME)
    )
    assert len(pointer_paths) == 1
    pointer = json.loads(pointer_paths[0].read_text(encoding="utf-8"))
    assert pointer["checkpoint_state_intermediate"] == "interrupted"
    assert pointer["workflow_completed_intermediate"] is False
    assert pointer["supports_paper_claim"] is False

    completed_calls = 0

    def completed_runner() -> dict[str, Any]:
        nonlocal completed_calls
        completed_calls += 1
        assert partial_path.read_text(encoding="utf-8") == "stable partial"
        return _write_completed_route(
            tmp_path,
            route,
            paper_run=paper_run,
            formal_lock=formal_lock,
        )

    persistence.run_persistent_workflow(
        root=tmp_path,
        workflow_name=route.workflow_name,
        runner=completed_runner,
        persistent_output_dir=tmp_path / "drive",
        checkpoint_interval_seconds=0,
    )
    assert completed_calls == 1
