"""构建与读取外部 baseline 命令计划。"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import shlex
from typing import Any

from paper_experiments.baselines.command_adapter import BaselineCommandSpec

REQUIRED_COMMAND_PLAN_FIELDS = ("baseline_id", "command", "output_path")
PRIMARY_BASELINE_ADAPTERS = {
    "tree_ring": "external_baseline/primary/tree_ring/adapter/run_slm_eval.py",
    "gaussian_shading": "external_baseline/primary/gaussian_shading/adapter/run_slm_eval.py",
    "shallow_diffuse": "external_baseline/primary/shallow_diffuse/adapter/run_slm_eval.py",
}


def _load_json_or_jsonl(path: Path) -> list[dict[str, Any]]:
    """读取 JSON 或 JSONL 命令计划。"""

    text = path.read_text(encoding="utf-8-sig")
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    payload = json.loads(text)
    if not isinstance(payload, list):
        raise TypeError("baseline command plan JSON 必须是列表")
    return [dict(row) for row in payload]


def _load_csv(path: Path) -> list[dict[str, Any]]:
    """读取 CSV 命令计划。"""

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _command_tuple(value: Any) -> tuple[str, ...]:
    """把命令字段转换为显式 argv 列表。"""

    if isinstance(value, list):
        command = tuple(str(part) for part in value)
    elif isinstance(value, str):
        command = tuple(shlex.split(value))
    else:
        raise TypeError("baseline command 必须是字符串或列表")
    if not command:
        raise ValueError("baseline command 不得为空")
    return command


def load_baseline_command_plan(path: str | Path) -> list[BaselineCommandSpec]:
    """从 JSON / JSONL / CSV 文件读取外部 baseline 命令计划。"""

    input_path = Path(path)
    if input_path.suffix in {".json", ".jsonl"}:
        rows = _load_json_or_jsonl(input_path)
    elif input_path.suffix == ".csv":
        rows = _load_csv(input_path)
    else:
        raise ValueError(f"不支持的 baseline command plan 文件后缀: {input_path.suffix}")
    specs: list[BaselineCommandSpec] = []
    for index, row in enumerate(rows):
        missing = [field for field in REQUIRED_COMMAND_PLAN_FIELDS if field not in row]
        if missing:
            raise ValueError(f"baseline command plan 第 {index} 行缺少字段: {missing}")
        specs.append(
            BaselineCommandSpec(
                baseline_id=str(row["baseline_id"]),
                command=_command_tuple(row["command"]),
                output_path=str(row["output_path"]),
                working_directory=str(row["working_directory"]) if row.get("working_directory") else None,
                timeout_seconds=int(row.get("timeout_seconds", 3600)),
            )
        )
    return specs


def build_baseline_command_plan_manifest(specs: list[BaselineCommandSpec]) -> dict[str, Any]:
    """构建命令计划 manifest, 用于记录外部依赖和执行入口。"""

    return {
        "artifact_name": "baseline_command_plan_manifest.json",
        "baseline_count": len(specs),
        "baselines": [spec.to_dict() for spec in specs],
    }


def selected_primary_baselines(value: str) -> list[str]:
    """解析 common-backbone baseline 列表并拒绝 T2SMark 重复正式入口。"""

    baseline_ids = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [item for item in baseline_ids if item not in PRIMARY_BASELINE_ADAPTERS]
    if unknown:
        raise ValueError(f"未登记的 primary baseline adapter: {unknown}")
    return baseline_ids

