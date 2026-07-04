"""Colab 论文结果闭合命令调度 helper。"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

from paper_workflow.colab_utils.notebook_runtime import write_notebook_runtime_report
from paper_workflow.colab_utils.progress import progress_bar, update_progress

REQUIRED_CLOSURE_PACKAGE_PATTERNS: tuple[tuple[str, str], ...] = (
    ("attention_geometry", "attention_geometry_package_*.zip"),
    ("attention_latent_injection", "attention_latent_injection_package_*.zip"),
    ("aligned_rescoring", "aligned_rescoring_package_*.zip"),
    ("threshold_calibration", "threshold_calibration_package_*.zip"),
    ("real_attack_evaluation", "real_attack_evaluation_package_*.zip"),
    ("conventional_geometric_attack_evaluation", "conventional_geometric_attack_evaluation_package_*.zip"),
    ("dataset_level_quality", "dataset_level_quality_package_*.zip"),
    ("method_faithful_tree_ring", "external_baseline_method_faithful_package_tree_ring_*.zip"),
    ("method_faithful_gaussian_shading", "external_baseline_method_faithful_package_gaussian_shading_*.zip"),
    ("method_faithful_shallow_diffuse", "external_baseline_method_faithful_package_shallow_diffuse_*.zip"),
    ("method_faithful_t2smark", "external_baseline_method_faithful_package_t2smark_*.zip"),
    ("official_reference_tree_ring", "external_baseline_official_reference_package_tree_ring_*.zip"),
    ("official_reference_gaussian_shading", "external_baseline_official_reference_package_gaussian_shading_*.zip"),
    ("official_reference_shallow_diffuse", "external_baseline_official_reference_package_shallow_diffuse_*.zip"),
    ("official_reference_t2smark", "external_baseline_official_reference_package_t2smark_*.zip"),
)


def _short_commit() -> str:
    """读取当前仓库短提交, 用于结果包命名。"""

    return subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _complete_archive_name(paper_run_name: str) -> str:
    """根据当前论文运行层级生成完整结果包名称。"""

    utc_suffix = datetime.now(timezone.utc).strftime("%Y%m%dt%H%M%sz")
    return f"{paper_run_name}_complete_result_package_{utc_suffix}_{_short_commit()}.zip"


def build_paper_result_closure_preflight_report(package_search_root: str) -> dict[str, Any]:
    """检查完整论文结果闭合所需的 Drive 结果包是否齐备。

    该检查属于 workflow 门禁层: 它不替代各个 builder 的 schema 校验, 只在
    长耗时闭合命令启动前发现明显缺失的前序结果包, 避免生成不完整结果包。
    """

    root_path = Path(package_search_root).expanduser()
    package_status: list[dict[str, Any]] = []
    missing_package_families: list[str] = []
    for family_name, pattern in REQUIRED_CLOSURE_PACKAGE_PATTERNS:
        matches = sorted(root_path.rglob(pattern)) if root_path.exists() else []
        package_status.append(
            {
                "package_family": family_name,
                "pattern": pattern,
                "match_count": len(matches),
                "latest_match": str(matches[-1]) if matches else "",
            }
        )
        if not matches:
            missing_package_families.append(family_name)
    return {
        "package_search_root": str(root_path),
        "package_search_root_exists": root_path.exists(),
        "required_package_family_count": len(REQUIRED_CLOSURE_PACKAGE_PATTERNS),
        "missing_package_family_count": len(missing_package_families),
        "missing_package_families": missing_package_families,
        "package_status": package_status,
        "closure_preflight_ready": root_path.exists() and not missing_package_families,
    }


def require_paper_result_closure_inputs(package_search_root: str) -> dict[str, Any]:
    """在闭合命令前强制检查完整结果包输入。"""

    report = build_paper_result_closure_preflight_report(package_search_root)
    print("paper_result_closure_preflight", json.dumps(report, ensure_ascii=False, sort_keys=True))
    if not report["closure_preflight_ready"]:
        missing = ",".join(report["missing_package_families"])
        raise FileNotFoundError(f"pilot_paper_closure_required_packages_missing:{missing}")
    return report


def build_paper_result_closure_commands(
    *,
    package_search_root: str,
    complete_drive_output_dir: str,
    target_fpr: str,
    paper_run_name: str,
) -> list[list[str]]:
    """构造结果闭合所需 repository command 序列。

    该函数只调度已有脚本, 不直接写正式 records、tables、figures 或 reports。
    这样 Notebook 不需要拼接命令、结果包名称和输出路径。
    """

    complete_archive_name = _complete_archive_name(paper_run_name)
    return [
        [sys.executable, "scripts/write_pilot_paper_result_records.py", "--package-search-root", package_search_root, "--materialize-only"],
        [sys.executable, "scripts/write_attack_matrix_outputs.py"],
        [sys.executable, "scripts/write_primary_baseline_result_candidates.py", "--target-fpr-override", target_fpr],
        [sys.executable, "scripts/write_primary_baseline_formal_import_protocol.py"],
        [sys.executable, "scripts/write_external_baseline_comparison_outputs.py"],
        [sys.executable, "scripts/write_internal_ablation_outputs.py"],
        [sys.executable, "scripts/write_pilot_paper_result_records.py", "--require-existing-evidence"],
        [
            sys.executable,
            "scripts/write_pilot_paper_fixed_fpr_common_protocol_outputs.py",
            "--candidate-records-path",
            "outputs/pilot_paper_fixed_fpr_results/pilot_paper_result_records.jsonl",
            "--require-existing-evidence",
        ],
        [
            sys.executable,
            "scripts/write_pilot_paper_complete_result_package.py",
            "--package-search-root",
            package_search_root,
            "--drive-output-dir",
            complete_drive_output_dir,
            "--archive-name",
            complete_archive_name,
            "--skip-package-materialization",
            "--zip-compression",
            "stored",
        ],
    ]


def run_paper_result_closure_commands(
    *,
    package_search_root: str,
    complete_drive_output_dir: str,
    target_fpr: str,
    paper_run_name: str,
) -> dict[str, Any]:
    """运行论文结果闭合命令, 并返回最新 Drive 结果包路径。"""

    preflight_report = require_paper_result_closure_inputs(package_search_root)
    commands = build_paper_result_closure_commands(
        package_search_root=package_search_root,
        complete_drive_output_dir=complete_drive_output_dir,
        target_fpr=target_fpr,
        paper_run_name=paper_run_name,
    )
    with progress_bar(len(commands), desc="paper result closure commands", enabled=True) as command_progress:
        for command_index, command in enumerate(commands, start=1):
            command_name = command[1] if len(command) > 1 else command[0]
            if command_name.endswith("write_pilot_paper_complete_result_package.py"):
                write_notebook_runtime_report(
                    root=".",
                    workflow_name="paper_result_closure",
                    output_dir="outputs/pilot_paper_complete_result_package",
                    drive_output_dir=complete_drive_output_dir,
                    archive_name=command[command.index("--archive-name") + 1] if "--archive-name" in command else "",
                )
            print("run_repository_command", " ".join(command))
            subprocess.run(command, check=True)
            update_progress(
                command_progress,
                profile=f"command={command_name} index={command_index}/{len(commands)}",
            )

    archive_pattern = f"{paper_run_name}_complete_result_package_*.zip"
    complete_archives = sorted(Path(complete_drive_output_dir).glob(archive_pattern))
    if not complete_archives:
        raise FileNotFoundError("未找到写回 Google Drive 的当前论文运行层级完整结果包")
    return {
        "latest_complete_archive": str(complete_archives[-1]),
        "command_count": len(commands),
        "paper_run_name": paper_run_name,
        "target_fpr": target_fpr,
        "preflight_report": preflight_report,
    }
