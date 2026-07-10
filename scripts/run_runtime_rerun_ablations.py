"""运行完整 Prompt 集的真实机制消融。"""

from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.ablations.runtime_rerun import (
    package_runtime_rerun_ablations,
    run_runtime_rerun_ablations,
)
from experiments.protocol.paper_run_config import build_paper_run_config
from experiments.protocol.prompts import build_prompt_records, read_prompt_file
from experiments.protocol.splits import apply_split_assignments
from scripts.run_image_only_dataset_runtime import build_method_config


def main() -> None:
    """对当前论文级别的全部 Prompt 执行消融并支持断点续跑。"""

    paper_run = build_paper_run_config(ROOT)
    records = apply_split_assignments(
        build_prompt_records(
            paper_run.prompt_set,
            read_prompt_file(ROOT / paper_run.prompt_file),
        )
    )[: paper_run.sample_count]
    if len(records) != paper_run.prompt_count:
        raise RuntimeError("正式消融 Prompt 数量必须与当前论文级别完全一致")
    base = build_method_config(ROOT)
    configs = tuple(
        replace(
            base,
            prompt=record.prompt_text,
            prompt_id=record.prompt_id,
            split=record.split,
            seed=base.seed + record.prompt_index,
            standard_attack_profiles=(base.standard_attack_profiles if record.split == "test" else ()),
            diffusion_attacks_enabled=(
                base.diffusion_attacks_enabled and record.split == "test"
            ),
        )
        for record in records
    )
    output_dir = f"outputs/formal_mechanism_ablation/{paper_run.run_name}"
    summary = run_runtime_rerun_ablations(
        configs,
        target_fpr=paper_run.target_fpr,
        root=ROOT,
        output_dir=output_dir,
        max_new_runs_per_session=int(
            os.environ.get("SLM_WM_MAX_NEW_ABLATION_RUNS_PER_SESSION", "0")
        ),
    )
    if summary.get("protocol_decision") == "resume_required":
        clear_strict_ddim_inversion_runtime_cache()
        print(json.dumps({"summary": summary}, ensure_ascii=False, sort_keys=True))
        return
    if summary.get("protocol_decision") != "pass":
        raise RuntimeError("真实机制消融未通过完整证据门禁")
    clear_strict_ddim_inversion_runtime_cache()
    archive_path = package_runtime_rerun_ablations(root=ROOT, output_dir=output_dir)
    print(
        json.dumps(
            {"summary": summary, "archive_path": str(archive_path)},
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
