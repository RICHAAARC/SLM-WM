"""执行外部 baseline 命令计划并汇总 observation 输出。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.baselines.command_adapter import run_baseline_commands
from experiments.baselines.command_plan import build_baseline_command_plan_manifest, load_baseline_command_plan
from experiments.baselines.observation_io import build_baseline_execution_manifest

DEFAULT_OUTPUT_DIR = Path("outputs/external_baseline_execution")


def build_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器。"""

    parser = argparse.ArgumentParser(description="执行外部 baseline 命令计划。")
    parser.add_argument("--plan", required=True, help="baseline 命令计划 JSON / JSONL / CSV。")
    parser.add_argument("--out", default=str(DEFAULT_OUTPUT_DIR), help="输出目录, 必须位于 outputs/ 下。")
    parser.add_argument("--formal-result-claim", action="store_true", help="声明输出可作为正式论文对比结果。")
    parser.add_argument("--evidence-path", action="append", default=[], help="正式运行证据文件路径, 可重复提供。")
    parser.add_argument("--require-pass", action="store_true", help="任一 baseline 命令失败时返回非零退出码。")
    return parser


def _resolve(path: str | Path) -> Path:
    """解析输入路径。"""

    candidate = Path(path)
    return candidate.resolve() if candidate.is_absolute() else (ROOT / candidate).resolve()


def _ensure_output_dir(path: str | Path) -> Path:
    """确保输出目录位于 outputs 下。"""

    output_dir = _resolve(path)
    outputs_root = (ROOT / "outputs").resolve()
    try:
        output_dir.relative_to(outputs_root)
    except ValueError as exc:
        raise ValueError(f"外部 baseline 执行输出必须位于 outputs/ 下: {output_dir}") from exc
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _resolve_evidence_paths(values: list[str]) -> list[str]:
    """解析证据路径并保留绝对路径。"""

    return [str(_resolve(value)) for value in values if str(value).strip()]


def _validate_formal_evidence(*, formal_result_claim: bool, evidence_paths: list[str]) -> None:
    """正式结果声明必须绑定至少一个真实存在的证据文件。"""

    if not formal_result_claim:
        return
    missing = [path for path in evidence_paths if not Path(path).is_file()]
    if not evidence_paths:
        missing.append("formal_result_claim_requires_at_least_one_evidence_path")
    if missing:
        raise FileNotFoundError(f"正式 baseline 证据路径缺失: {missing}")


def main() -> None:
    """CLI 入口。"""

    args = build_parser().parse_args()
    plan_path = _resolve(args.plan)
    specs = load_baseline_command_plan(plan_path)
    evidence_paths = _resolve_evidence_paths(args.evidence_path)
    _validate_formal_evidence(formal_result_claim=args.formal_result_claim, evidence_paths=evidence_paths)

    results, rows = run_baseline_commands(specs)
    output_dir = _ensure_output_dir(args.out)
    command_results_path = output_dir / "baseline_command_results.json"
    observations_path = output_dir / "baseline_observations.json"
    manifest_path = output_dir / "baseline_execution_manifest.json"
    command_plan_manifest_path = output_dir / "baseline_command_plan_manifest.json"

    command_plan_manifest_path.write_text(
        json.dumps(build_baseline_command_plan_manifest(specs), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    result_dicts = [result.to_dict() for result in results]
    command_results_path.write_text(json.dumps(result_dicts, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    observations_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    manifest = build_baseline_execution_manifest(
        command_specs=[spec.to_dict() for spec in specs],
        command_results=result_dicts,
        observation_rows=rows,
        baseline_observations_path=observations_path,
        command_results_path=command_results_path,
        formal_result_claim=args.formal_result_claim,
        evidence_paths=evidence_paths,
    ).to_dict()
    manifest["baseline_command_plan_manifest_path"] = str(command_plan_manifest_path)
    manifest["baseline_command_results_path"] = str(command_results_path)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    failed = [row for row in result_dicts if int(row["return_code"]) != 0]
    print(
        json.dumps(
            {
                "command_count": len(specs),
                "observation_count": len(rows),
                "formal_result_claim": bool(args.formal_result_claim),
                "failed_command_count": len(failed),
                "failed_baseline_ids": [row["baseline_id"] for row in failed],
                "manifest_path": str(manifest_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if args.require_pass and failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
