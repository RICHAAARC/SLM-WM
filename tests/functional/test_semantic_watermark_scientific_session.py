"""验证单一科学子解释器调度器的续跑与消融顺序."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.runtime import semantic_watermark_scientific_session as dispatcher


pytestmark = pytest.mark.quick


def _write_json(path: Path, payload: dict[str, object]) -> None:
    """写入测试调度状态."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _select_probe_run(monkeypatch: pytest.MonkeyPatch) -> None:
    """让调度单元测试只关注科学命令状态机."""

    monkeypatch.setattr(
        dispatcher,
        "build_paper_run_config",
        lambda _root: SimpleNamespace(run_name="probe_paper"),
    )


def test_runtime_progress_prevents_formal_ablation_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """主方法需要续跑时, 同一子解释器不得提前启动正式消融."""

    run_name = "probe_paper"
    _write_json(
        tmp_path
        / "outputs"
        / "image_only_dataset_runtime"
        / run_name
        / "dataset_runtime_progress.json",
        {"protocol_decision": "resume_required"},
    )
    calls = []
    monkeypatch.setattr(dispatcher, "ROOT", tmp_path)
    monkeypatch.setenv("SLM_WM_PAPER_RUN_NAME", run_name)
    _select_probe_run(monkeypatch)

    def run_child(command_tail: tuple[str, ...]) -> dict[str, object]:
        calls.append(command_tail)
        return {"argv": list(command_tail), "return_code": 0, "stdout": "", "stderr": ""}

    monkeypatch.setattr(dispatcher, "_run_child", run_child)

    report = dispatcher.run_scientific_commands(run_formal_ablation=True)

    assert calls == [("-m", "experiments.runners.image_only_dataset_workload")]
    assert report["decision"] == "pass"
    assert report["packaging_deferred"] is True
    assert report["artifact_state"]["runtime_progress_present"] is True


def test_closed_main_runs_requested_ablation_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """主方法和质量均闭合后, 请求的正式消融必须恰好运行一次."""

    run_name = "probe_paper"
    _write_json(
        tmp_path
        / "outputs"
        / "image_only_dataset_runtime"
        / run_name
        / "dataset_runtime_summary.json",
        {"protocol_decision": "pass", "repeat_component_ready": True},
    )
    for relative_path in (
        f"outputs/image_only_dataset_runtime/{run_name}/manifest.local.json",
        f"outputs/dataset_level_quality/{run_name}/manifest.local.json",
        f"outputs/formal_mechanism_ablation/{run_name}/manifest.local.json",
    ):
        _write_json(
            tmp_path / relative_path,
            {"formal_execution_run_lock": {"fixture": "lock"}},
        )
    _write_json(
        tmp_path
        / "outputs"
        / "dataset_level_quality"
        / run_name
        / "dataset_quality_summary.json",
        {
            "formal_fid_kid_component_ready": True,
            "repeat_component_ready": True,
        },
    )
    _write_json(
        tmp_path
        / "outputs"
        / "formal_mechanism_ablation"
        / run_name
        / "ablation_component_summary.json",
        {"protocol_decision": "pass", "repeat_component_ready": True},
    )
    calls = []
    monkeypatch.setattr(dispatcher, "ROOT", tmp_path)
    monkeypatch.setenv("SLM_WM_PAPER_RUN_NAME", run_name)
    _select_probe_run(monkeypatch)

    def run_child(command_tail: tuple[str, ...]) -> dict[str, object]:
        calls.append(command_tail)
        return {"argv": list(command_tail), "return_code": 0, "stdout": "", "stderr": ""}

    monkeypatch.setattr(dispatcher, "_run_child", run_child)

    report = dispatcher.run_scientific_commands(run_formal_ablation=True)

    assert calls == [
        ("-m", "experiments.runners.image_only_dataset_workload"),
        ("-m", "experiments.ablations.mechanism_ablation_workload"),
    ]
    assert report["decision"] == "pass"
    assert [record["command_role"] for record in report["commands"]] == [
        "image_only_dataset_runtime",
        "runtime_rerun_ablation",
    ]
    assert [record["artifact_role"] for record in report["artifact_records"]] == [
        "image_only_dataset_runtime",
        "dataset_level_quality",
        "runtime_rerun_ablation",
    ]


def test_protocol_pass_without_repeat_component_cannot_close(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """仅有协议通过文本时不得把未完成重复单元视为闭合."""

    run_name = "probe_paper"
    _write_json(
        tmp_path
        / "outputs"
        / "image_only_dataset_runtime"
        / run_name
        / "dataset_runtime_summary.json",
        {"protocol_decision": "pass", "repeat_component_ready": False},
    )
    _write_json(
        tmp_path
        / "outputs"
        / "dataset_level_quality"
        / run_name
        / "dataset_quality_summary.json",
        {
            "formal_fid_kid_component_ready": True,
            "repeat_component_ready": True,
        },
    )
    monkeypatch.setattr(dispatcher, "ROOT", tmp_path)
    monkeypatch.setenv("SLM_WM_PAPER_RUN_NAME", run_name)
    _select_probe_run(monkeypatch)
    monkeypatch.setattr(
        dispatcher,
        "_run_child",
        lambda _command: {
            "argv": [],
            "return_code": 0,
            "stdout": "",
            "stderr": "",
        },
    )

    with pytest.raises(RuntimeError, match="重复证据组件"):
        dispatcher.run_scientific_commands(run_formal_ablation=False)


def test_child_command_defers_packaging_until_binding_exists(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """科学命令不得在外层执行绑定生成前调用正式打包器."""

    captured: dict[str, object] = {}

    class Completed:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(argv: list[str], **kwargs: object) -> Completed:
        captured["argv"] = argv
        captured.update(kwargs)
        return Completed()

    monkeypatch.setattr(dispatcher, "ROOT", tmp_path)
    monkeypatch.setattr(dispatcher.subprocess, "run", fake_run)
    result = dispatcher._run_child(("scripts/scientific_task.py",))

    environment = captured["env"]
    assert isinstance(environment, dict)
    assert environment["SLM_WM_DEFER_SCIENTIFIC_PACKAGING"] == "1"
    assert result["packaging_deferred"] is True
