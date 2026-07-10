"""运行真实生成、攻击和图像盲检机制消融。"""

from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.ablations.runtime_rerun import package_runtime_rerun_ablations, run_runtime_rerun_ablations
from experiments.protocol.paper_run_config import build_paper_run_config
from experiments.protocol.prompts import build_prompt_records, read_prompt_file
from experiments.protocol.splits import apply_split_assignments
from experiments.runners.real_attack_evaluation import clear_strict_ddim_inversion_runtime_cache
from scripts.run_image_only_dataset_runtime import build_method_config


def main() -> None:
    """命令行入口。"""

    paper_run = build_paper_run_config(ROOT)
    records = apply_split_assignments(
        build_prompt_records(
            paper_run.prompt_set,
            read_prompt_file(ROOT / paper_run.prompt_file),
        )
    )
    requested_count = int(os.environ.get("SLM_WM_ABLATION_PROMPT_COUNT", "100"))
    test_records = tuple(record for record in records if record.split == "test")[:requested_count]
    base = build_method_config(ROOT)
    protocol_path = ROOT / "outputs" / "image_only_dataset_runtime" / paper_run.run_name / "frozen_evidence_protocol.json"
    if not protocol_path.is_file():
        raise FileNotFoundError("真实消融必须复用数据集运行冻结的完整 evidence 协议")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    base = replace(
        base,
        content_threshold=float(protocol["content_threshold"]),
        rescue_margin_low=float(protocol["rescue_margin_low"]),
        geometry_score_threshold=float(protocol["geometry_score_threshold"]),
    )
    configs = tuple(
        replace(
            base,
            prompt=record.prompt_text,
            prompt_id=record.prompt_id,
            split=record.split,
            seed=base.seed + record.prompt_index,
        )
        for record in test_records
    )
    output_dir = f"outputs/formal_mechanism_ablation/{paper_run.run_name}"
    summary = run_runtime_rerun_ablations(
        configs,
        root=ROOT,
        output_dir=output_dir,
        minimum_prompt_count=min(requested_count, len(test_records)),
        max_new_runs_per_session=int(os.environ.get("SLM_WM_MAX_NEW_ABLATION_RUNS_PER_SESSION", "0")),
    )
    if summary.get("protocol_decision") == "resume_required":
        clear_strict_ddim_inversion_runtime_cache()
        print(json.dumps({"summary": summary}, ensure_ascii=False, sort_keys=True))
        return
    clear_strict_ddim_inversion_runtime_cache()
    archive_path = package_runtime_rerun_ablations(root=ROOT, output_dir=output_dir)
    print(json.dumps({"summary": summary, "archive_path": str(archive_path)}, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
