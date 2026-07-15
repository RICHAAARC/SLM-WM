"""执行完整 prompt 集的单模型分支风险参数敏感性工作负载。"""

from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path

from experiments.ablations.branch_risk_sensitivity_runtime import (
    package_branch_risk_parameter_sensitivity,
    run_branch_risk_parameter_sensitivity,
)
from experiments.protocol.paper_run_config import build_paper_run_config
from experiments.protocol.prompts import build_prompt_records, read_prompt_file
from experiments.protocol.splits import apply_split_assignments
from experiments.runners.image_only_dataset_workload import build_method_config


ROOT = Path(__file__).resolve().parents[2]


def run_branch_risk_parameter_sensitivity_workload(
    root: str | Path = ROOT,
) -> dict[str, object]:
    """对当前论文层级全部 prompt 执行18项真实参数重运行。"""

    root_path = Path(root).resolve()
    paper_run = build_paper_run_config(root_path)
    records = apply_split_assignments(
        build_prompt_records(
            paper_run.prompt_set,
            read_prompt_file(root_path / paper_run.prompt_file),
        )
    )[: paper_run.sample_count]
    if len(records) != paper_run.prompt_count:
        raise RuntimeError("敏感性实验 prompt 数量必须与当前论文层级完全一致")
    base = build_method_config(root_path)
    configs = tuple(
        replace(
            base,
            prompt=record.prompt_text,
            prompt_id=record.prompt_id,
            split=record.split,
            seed=base.seed + record.prompt_index,
            standard_attack_profiles=(
                base.standard_attack_profiles if record.split == "test" else ()
            ),
            diffusion_attacks_enabled=(
                base.diffusion_attacks_enabled and record.split == "test"
            ),
        )
        for record in records
    )
    summary = run_branch_risk_parameter_sensitivity(
        configs,
        target_fpr=paper_run.target_fpr,
        paper_run_name=paper_run.run_name,
        root=root_path,
        max_new_runs_per_session=int(
            os.environ.get(
                "SLM_WM_MAX_NEW_SENSITIVITY_RUNS_PER_SESSION",
                "0",
            )
        ),
    )
    if summary.get("protocol_decision") == "resume_required":
        return {"summary": summary}
    if summary.get("protocol_decision") != "pass":
        raise RuntimeError("单模型风险参数敏感性未通过完整证据门禁")
    packaging_deferred = (
        os.environ.get("SLM_WM_DEFER_SCIENTIFIC_PACKAGING", "0") == "1"
    )
    archive_path = (
        None
        if packaging_deferred
        else package_branch_risk_parameter_sensitivity(
            paper_run.run_name,
            root=root_path,
        )
    )
    return {
        "summary": summary,
        "archive_path": None if archive_path is None else str(archive_path),
        "packaging_deferred": packaging_deferred,
    }


def main() -> None:
    """命令行入口。"""

    print(
        json.dumps(
            run_branch_risk_parameter_sensitivity_workload(ROOT),
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
