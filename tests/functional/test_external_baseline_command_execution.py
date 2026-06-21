"""验证外部 baseline 命令计划执行链路。"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

from experiments.baselines import (
    BaselineCommandSpec,
    build_baseline_execution_manifest,
    run_baseline_commands,
)
from experiments.baselines.command_plan import build_baseline_command_plan_manifest, load_baseline_command_plan


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
