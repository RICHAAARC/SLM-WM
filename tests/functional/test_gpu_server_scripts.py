"""验证无 Notebook GPU 服务器入口。"""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts import run_gpu_server_workflow as workflow


COMMIT = "a" * 40
LOCK_DIGEST = "b" * 64


@pytest.mark.quick
def test_server_workflow_maps_only_current_formal_gpu_jobs() -> None:
    """服务器入口不得重新暴露已移除的分量诊断流程。"""

    assert set(workflow.WORKFLOW_COMMANDS) == {"image_only_dataset", "mechanism_ablation"}


@pytest.mark.quick
def test_server_workflow_passes_paper_run_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """服务器运行必须把论文级别传给与 Colab 相同的仓库脚本。"""

    captured: dict[str, object] = {}

    class Completed:
        returncode = 0
        stdout = "ok"
        stderr = ""

    def fake_run(command: list[str], **kwargs: object) -> Completed:
        captured["command"] = command
        captured.update(kwargs)
        return Completed()

    monkeypatch.setattr(workflow.subprocess, "run", fake_run)
    monkeypatch.setattr(
        workflow,
        "build_formal_execution_lock",
        lambda root, expected_commit: {
            "formal_execution_commit": expected_commit,
            "formal_execution_lock_digest": LOCK_DIGEST,
            "formal_execution_lock_ready": True,
        },
    )
    result = workflow.run_workflow(
        "image_only_dataset",
        "probe_paper",
        COMMIT,
        tmp_path,
    )

    assert result["return_code"] == 0
    assert captured["env"]["SLM_WM_PAPER_RUN_NAME"] == "probe_paper"  # type: ignore[index]
    assert captured["env"]["SLM_WM_FORMAL_EXECUTION_COMMIT"] == COMMIT  # type: ignore[index]
    assert captured["env"]["SLM_WM_FORMAL_EXECUTION_LOCK_DIGEST"] == LOCK_DIGEST  # type: ignore[index]
    assert captured["cwd"] == tmp_path.resolve()
    assert result["formal_execution_lock"]["formal_execution_commit"] == COMMIT
