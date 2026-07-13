"""验证正式 workflow 的活动 repeat 与跨 repeat 不变持久化边界."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import formal_workflow_environment as workflow_environment
from scripts import run_gpu_server_workflow as gpu_workflow


@pytest.fixture(autouse=True)
def _restore_slm_environment_after_test():
    """隔离配置函数写入的全量 SLM_WM 环境, 防止测试顺序改变语义."""

    original = {
        key: value for key, value in os.environ.items() if key.startswith("SLM_WM_")
    }
    yield
    for key in tuple(os.environ):
        if key.startswith("SLM_WM_"):
            os.environ.pop(key, None)
    os.environ.update(original)


@pytest.mark.quick
def test_drive_paths_isolate_active_repeat_and_invariant_evidence_once() -> None:
    """活动路径只出现一次 repeat, 不变路径不得绑定任一 repeat."""

    repeat_id = "seed_01_key_02"
    paper_run = SimpleNamespace(
        drive_result_root="/drive/slm_wm/probe_paper",
        randomization_repeat_id=repeat_id,
        drive_dir=lambda artifact_role: (
            "/drive/slm_wm/probe_paper/randomization_repeats/"
            f"{repeat_id}/{artifact_role}"
        ),
    )

    active_path = workflow_environment._repeat_drive_dir(
        paper_run,
        "image_only_dataset_runtime",
    )
    invariant_path = workflow_environment._invariant_drive_dir(
        paper_run,
        "external_baseline_official_reference",
    )

    assert active_path.count(repeat_id) == 1
    assert active_path.endswith(
        f"randomization_repeats/{repeat_id}/image_only_dataset_runtime"
    )
    assert repeat_id not in invariant_path
    assert invariant_path.endswith(
        "cross_repeat_invariant/external_baseline_official_reference"
    )


@pytest.mark.quick
def test_configured_invariant_workflow_does_not_publish_active_repeat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """不变 official-reference 子进程环境不得泄漏默认或先前 repeat."""

    original_slm_environment = {
        key: value for key, value in os.environ.items() if key.startswith("SLM_WM_")
    }
    monkeypatch.setenv("SLM_WM_PAPER_RUN_NAME", "probe_paper")
    monkeypatch.setenv("SLM_WM_RANDOMIZATION_REPEAT_ID", "seed_01_key_02")
    monkeypatch.setattr(
        workflow_environment,
        "require_published_formal_execution_lock",
        lambda root: {
            "formal_execution_commit": "a" * 40,
            "formal_execution_lock_digest": "b" * 64,
        },
    )

    try:
        result = workflow_environment.configure_formal_workflow_environment(
            "official_reference_tree_ring",
            repository_root=Path("."),
        )

        assert result["randomization_repeat_id"] == ""
        assert "SLM_WM_RANDOMIZATION_REPEAT_ID" not in os.environ
        assert "seed_01_key_02" not in os.environ[
            "SLM_WM_TREE_RING_OFFICIAL_DRIVE_OUTPUT_DIR"
        ]
        assert "/cross_repeat_invariant/" in os.environ[
            "SLM_WM_TREE_RING_OFFICIAL_DRIVE_OUTPUT_DIR"
        ]
    finally:
        for key in tuple(os.environ):
            if key.startswith("SLM_WM_"):
                os.environ.pop(key, None)
        os.environ.update(original_slm_environment)


@pytest.mark.quick
def test_main_route_uses_governed_repeat_root_for_all_persistent_roles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """主方法实际调度必须把三个归档 family 和 checkpoint 写入同一 repeat 根."""

    repeat_id = "seed_01_key_02"
    monkeypatch.setenv("SLM_WM_PAPER_RUN_NAME", "probe_paper")
    monkeypatch.setenv("SLM_WM_RANDOMIZATION_REPEAT_ID", repeat_id)
    monkeypatch.setattr(
        workflow_environment,
        "require_published_formal_execution_lock",
        lambda root: {
            "formal_execution_commit": "a" * 40,
            "formal_execution_lock_digest": "b" * 64,
        },
    )

    environment_record, persistent_root_value = (
        gpu_workflow._configure_workflow_execution_environment(
            workflow_name="mechanism_ablation",
            root_path=Path("."),
            persistent_output_dir=None,
        )
    )
    persistent_root = Path(persistent_root_value)
    resolved_persistent_root = persistent_root.expanduser().resolve()
    expected_suffix = Path("randomization_repeats") / repeat_id
    assert persistent_root.as_posix().endswith(expected_suffix.as_posix())
    assert persistent_root.as_posix().count(repeat_id) == 1
    assert environment_record["persistent_output_dir"] == str(
        persistent_root_value
    )

    captured: dict[str, object] = {}

    def fake_session(*_args, **kwargs):
        captured.update(kwargs)
        return {"workflow_decision": "complete"}

    from scripts import semantic_watermark_scientific_workflow

    monkeypatch.setattr(
        semantic_watermark_scientific_workflow,
        "run_semantic_watermark_image_only_session",
        fake_session,
    )
    gpu_workflow._run_main_method_route(
        route=gpu_workflow.WORKFLOW_ROUTES["mechanism_ablation"],
        workflow_name="mechanism_ablation",
        paper_run_name="probe_paper",
        root_path=Path(".").resolve(),
        persistent_output_dir=persistent_root,
    )

    archive_dirs = captured["archive_destination_dirs"]
    assert archive_dirs == {
        "image_only_dataset_runtime": (
            resolved_persistent_root / "image_only_dataset_runtime"
        ),
        "dataset_level_quality": (
            resolved_persistent_root / "dataset_level_quality"
        ),
        "runtime_rerun_ablation": (
            resolved_persistent_root / "runtime_rerun_ablation"
        ),
    }
    assert captured["resume_checkpoint_dir"] == (
        resolved_persistent_root / "semantic_watermark_resume_checkpoint"
    )
    for path in (*archive_dirs.values(), captured["resume_checkpoint_dir"]):
        assert Path(path).as_posix().count(repeat_id) == 1
