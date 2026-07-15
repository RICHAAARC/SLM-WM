"""从单 Prompt 真实运行产物写出 GPU 方法与资源独立资格化报告."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.protocol.gpu_method_qualification import (
    build_gpu_method_qualification_report,
)
from experiments.runners.image_only_dataset_workload import build_method_config


DEFAULT_KNOWN_ANSWER_PATH = Path(
    "configs/keyed_prg_cross_platform_known_answer.json"
)
DEFAULT_OUTPUT_ROOT = Path("outputs/gpu_method_qualification")


def _read_json(path: Path) -> dict[str, Any]:
    """读取单个 JSON 映射."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON 顶层必须是映射: {path}")
    return payload


def _read_jsonl(path: Path) -> tuple[dict[str, Any], ...]:
    """读取 JSONL 映射记录, 并拒绝空行以外的非映射内容."""

    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"JSONL 第{line_number}行不是映射: {path}")
        rows.append(row)
    return tuple(rows)


def _resolve_input(root: Path, value: str | Path) -> Path:
    """将输入路径相对于仓库根目录解析."""

    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _resolve_output(root: Path, value: str | Path) -> Path:
    """强制持久化输出位于仓库 outputs 目录下."""

    path = _resolve_input(root, value)
    outputs_root = (root / "outputs").resolve()
    try:
        path.relative_to(outputs_root)
    except ValueError as exc:
        raise ValueError("GPU 资格化报告必须写入 outputs/ 下") from exc
    return path


def parse_args() -> argparse.Namespace:
    """解析命令行参数."""

    parser = argparse.ArgumentParser(
        description="写出单 Prompt GPU 方法算子和资源预算独立资格化报告",
    )
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--runtime-result", required=True)
    parser.add_argument("--update-records", required=True)
    parser.add_argument("--detection-records", required=True)
    parser.add_argument(
        "--known-answer",
        default=str(DEFAULT_KNOWN_ANSWER_PATH),
    )
    parser.add_argument("--resource-observation")
    parser.add_argument("--registered-budget")
    parser.add_argument("--qualification-binding", required=True)
    parser.add_argument("--output")
    return parser.parse_args()


def main() -> int:
    """构造报告并返回与方法算子门禁一致的进程状态码."""

    args = parse_args()
    root = Path(args.root).resolve()
    runtime_result = _read_json(_resolve_input(root, args.runtime_result))
    update_records = _read_jsonl(_resolve_input(root, args.update_records))
    detection_records = _read_jsonl(
        _resolve_input(root, args.detection_records)
    )
    resource_observation = (
        _read_json(_resolve_input(root, args.resource_observation))
        if args.resource_observation
        else None
    )
    registered_budget = (
        _read_json(_resolve_input(root, args.registered_budget))
        if args.registered_budget
        else None
    )
    qualification_binding = _read_json(
        _resolve_input(root, args.qualification_binding)
    )
    report = build_gpu_method_qualification_report(
        runtime_result=runtime_result,
        update_records=update_records,
        detection_records=detection_records,
        config=build_method_config(root),
        known_answer_path=_resolve_input(root, args.known_answer),
        resource_observation=resource_observation,
        registered_budget=registered_budget,
        qualification_binding=qualification_binding,
    )
    output = (
        _resolve_output(root, args.output)
        if args.output
        else _resolve_output(
            root,
            DEFAULT_OUTPUT_ROOT
            / str(runtime_result.get("run_id", "unknown_run"))
            / "gpu_method_qualification_report.json",
        )
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(output)
    return 0 if report["gpu_operator_preflight_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
