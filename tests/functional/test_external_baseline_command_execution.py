"""验证外部 baseline 命令计划执行链路。"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from paper_experiments.baselines import (
    BaselineCommandSpec,
    build_baseline_execution_manifest,
    run_baseline_commands,
)
from paper_experiments.baselines.command_plan import build_baseline_command_plan_manifest, load_baseline_command_plan
from scripts.build_external_baseline_command_plan import build_parser as build_external_plan_parser
from scripts.build_external_baseline_command_plan import build_plan as build_external_plan

PRIMARY_DIFFUSION_ADAPTERS = (
    "external_baseline/primary/tree_ring/adapter/run_slm_eval.py",
    "external_baseline/primary/gaussian_shading/adapter/run_slm_eval.py",
    "external_baseline/primary/shallow_diffuse/adapter/run_slm_eval.py",
)


@pytest.mark.quick
def test_external_baseline_command_runner_collects_observations(tmp_path: Path) -> None:
    """命令 runner 应执行显式 argv 并读取 adapter 产生的 observation 文件。"""

    adapter_path = tmp_path / "toy_adapter.py"
    output_path = tmp_path / "baseline_observations.json"
    adapter_path.write_text(
        """
import json
import sys
from pathlib import Path
rows = [{"event_id": "event_001", "baseline_id": "toy", "score": 0.8, "threshold": 0.5}]
Path(sys.argv[1]).write_text(json.dumps(rows), encoding="utf-8")
print("toy adapter done")
""",
        encoding="utf-8",
    )
    spec = BaselineCommandSpec(
        baseline_id="toy",
        command=(sys.executable, str(adapter_path), str(output_path)),
        output_path=str(output_path),
        working_directory=str(tmp_path),
        timeout_seconds=30,
    )

    results, rows = run_baseline_commands([spec])

    assert len(results) == 1
    assert results[0].return_code == 0
    assert results[0].observation_count == 1
    assert rows[0]["event_id"] == "event_001"

    manifest = build_baseline_execution_manifest(
        command_specs=[spec.to_dict()],
        command_results=[results[0].to_dict()],
        observation_rows=rows,
        baseline_observations_path=output_path,
        command_results_path=tmp_path / "baseline_command_results.json",
    ).to_dict()
    assert manifest["command_count"] == 1
    assert manifest["observation_count"] == 1
    assert manifest["formal_result_claim"] is False
    assert manifest["failed_command_count"] == 0


@pytest.mark.quick
def test_external_baseline_command_runner_writes_progress_events(tmp_path: Path) -> None:
    """命令 runner 应把子进程细粒度进度写入同一个 JSONL 事件文件。"""

    adapter_path = tmp_path / "toy_progress_adapter.py"
    output_path = tmp_path / "baseline_observations.json"
    progress_path = tmp_path / "progress_events.jsonl"
    adapter_path.write_text(
        """
import json
import os
import sys
from pathlib import Path
progress_path = os.environ.get("SLM_WM_PROGRESS_EVENT_PATH")
if progress_path:
    with Path(progress_path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"desc": "toy adapter", "completed": 1, "total": 2, "profile": "operation=toy"}, ensure_ascii=False) + "\\n")
rows = [{"event_id": "event_001", "baseline_id": "toy", "score": 0.8, "threshold": 0.5}]
Path(sys.argv[1]).write_text(json.dumps(rows), encoding="utf-8")
""",
        encoding="utf-8",
    )
    spec = BaselineCommandSpec(
        baseline_id="toy",
        command=(sys.executable, str(adapter_path), str(output_path)),
        output_path=str(output_path),
        working_directory=str(tmp_path),
        timeout_seconds=30,
    )

    results, rows = run_baseline_commands([spec], progress_event_path=progress_path)

    assert results[0].return_code == 0
    assert rows[0]["event_id"] == "event_001"
    events = [json.loads(line) for line in progress_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert any(event.get("baseline_id") == "toy" and "baseline_command" in str(event.get("profile")) for event in events)
    assert any(event.get("desc") == "toy adapter" for event in events)


@pytest.mark.quick
def test_external_baseline_command_plan_loader_accepts_json_list(tmp_path: Path) -> None:
    """命令计划 loader 应把 JSON 中的 command 列表转为不可变 argv 元组。"""

    plan_path = tmp_path / "baseline_command_plan.json"
    plan_path.write_text(
        json.dumps(
            [
                {
                    "baseline_id": "toy",
                    "command": [sys.executable, "adapter.py"],
                    "output_path": str(tmp_path / "baseline_observations.json"),
                    "timeout_seconds": 7,
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    specs = load_baseline_command_plan(plan_path)
    manifest = build_baseline_command_plan_manifest(specs)

    assert specs[0].command == (sys.executable, "adapter.py")
    assert specs[0].timeout_seconds == 7
    assert manifest["baseline_count"] == 1


@pytest.mark.quick
def test_tree_ring_command_plan_can_select_method_faithful_adapter(tmp_path: Path) -> None:
    """命令计划应能为 Tree-Ring 显式选择 SD3.5 方法忠实 adapter。"""

    root = tmp_path
    adapter_path = root / "external_baseline" / "primary" / "tree_ring" / "adapter" / "run_slm_eval.py"
    adapter_path.parent.mkdir(parents=True)
    adapter_path.write_text("print('tree ring adapter')\n", encoding="utf-8")
    prompt_plan = root / "prompt_plan.json"
    prompt_plan.write_text('[{"prompt_text":"a ceramic fox","prompt_id":"p0"}]\n', encoding="utf-8")
    output_root = root / "outputs" / "test_tree_ring_method_faithful_plan"
    args = build_external_plan_parser().parse_args(
        [
            "--root",
            str(root),
            "--methods",
            "tree_ring",
            "--output-root",
            str(output_root),
            "--prompt-plan",
            str(prompt_plan),
            "--tree-ring-adapter-mode",
            "method_faithful_sd35",
            "--tree-ring-attack-families",
            "jpeg",
            "--max-samples",
            "1",
        ]
    )

    plan = build_external_plan(args)
    command = plan[0]["command"]

    assert plan[0]["baseline_id"] == "tree_ring"
    assert "--adapter-mode" in command
    assert command[command.index("--adapter-mode") + 1] == "method_faithful_sd35"
    assert "--w-radius" in command
    assert "--attack-families" in command


@pytest.mark.quick
def test_diffusion_command_plan_can_select_all_method_faithful_adapters(tmp_path: Path) -> None:
    """命令计划应能为三类扩散主表 baseline 同时选择方法忠实 adapter。"""

    root = tmp_path
    for baseline_id in ("tree_ring", "gaussian_shading", "shallow_diffuse"):
        adapter_path = root / "external_baseline" / "primary" / baseline_id / "adapter" / "run_slm_eval.py"
        adapter_path.parent.mkdir(parents=True, exist_ok=True)
        adapter_path.write_text(f"print('{baseline_id} adapter')\n", encoding="utf-8")
    prompt_plan = root / "prompt_plan.json"
    prompt_plan.write_text('[{"prompt_text":"a ceramic fox","prompt_id":"p0"}]\n', encoding="utf-8")
    output_root = root / "outputs" / "test_diffusion_method_faithful_plan"
    args = build_external_plan_parser().parse_args(
        [
            "--root",
            str(root),
            "--methods",
            "tree_ring,gaussian_shading,shallow_diffuse",
            "--output-root",
            str(output_root),
            "--prompt-plan",
            str(prompt_plan),
            "--tree-ring-adapter-mode",
            "method_faithful_sd35",
            "--gaussian-shading-adapter-mode",
            "method_faithful_sd35",
            "--shallow-diffuse-adapter-mode",
            "method_faithful_sd35",
            "--gaussian-shading-channel-copy",
            "1",
            "--gaussian-shading-hw-copy",
            "8",
            "--shallow-diffuse-edit-fraction",
            "0.25",
            "--max-samples",
            "1",
        ]
    )

    plan = build_external_plan(args)
    commands = {row["baseline_id"]: row["command"] for row in plan}

    assert set(commands) == {"tree_ring", "gaussian_shading", "shallow_diffuse"}
    assert all("--adapter-mode" in command for command in commands.values())
    assert commands["gaussian_shading"][commands["gaussian_shading"].index("--adapter-mode") + 1] == "method_faithful_sd35"
    assert "--channel-copy" in commands["gaussian_shading"]
    assert commands["shallow_diffuse"][commands["shallow_diffuse"].index("--adapter-mode") + 1] == "method_faithful_sd35"
    assert "--edit-fraction" in commands["shallow_diffuse"]


@pytest.mark.quick
@pytest.mark.parametrize("adapter_path", PRIMARY_DIFFUSION_ADAPTERS)
def test_primary_diffusion_adapter_writes_method_faithful_observations(tmp_path: Path, adapter_path: str) -> None:
    """三类扩散 baseline adapter 应在轻量配置下写出 clean 与 positive observation。"""

    prompt_plan = tmp_path / "prompt_plan.json"
    output_path = tmp_path / "baseline_observations.json"
    artifact_root = tmp_path / "artifacts"
    prompt_plan.write_text(
        json.dumps(
            [
                {
                    "prompt_id": "prompt_00000",
                    "split": "method_faithful",
                    "prompt_text": "a small ceramic fox sitting on a wooden desk",
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            adapter_path,
            "--prompt-plan",
            str(prompt_plan),
            "--out",
            str(output_path),
            "--artifact-root",
            str(artifact_root),
            "--height",
            "64",
            "--width",
            "64",
            "--latent-channels",
            "4",
            "--num-inference-steps",
            "2",
            "--num-inversion-steps",
            "2",
            "--max-samples",
            "1",
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=Path.cwd(),
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    rows = json.loads(output_path.read_text(encoding="utf-8"))
    assert len(rows) == 2
    assert {row["sample_role"] for row in rows} == {"clean_negative", "positive_source"}
    assert not any(row["formal_result_claim"] for row in rows)
    assert not any(row["supports_paper_claim"] for row in rows)
    clean_row = next(row for row in rows if row["sample_role"] == "clean_negative")
    positive_row = next(row for row in rows if row["sample_role"] == "positive_source")
    assert clean_row["detection_decision"] is False
    assert positive_row["detection_decision"] is True
    assert artifact_root.exists()

