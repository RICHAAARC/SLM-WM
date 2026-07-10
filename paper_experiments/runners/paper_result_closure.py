"""完整论文实验结果闭合 runner。"""

from __future__ import annotations

from collections.abc import Callable
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

from experiments.runtime.archive_naming import utc_archive_token

REQUIRED_CLOSURE_PACKAGE_PATTERNS: tuple[tuple[str, str], ...] = (
    ("image_only_dataset_runtime", "image_only_dataset_runtime_package_*.zip"),
    ("runtime_rerun_ablation", "runtime_rerun_ablation_package_*.zip"),
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

CommandHook = Callable[[list[str]], None]
ProgressHook = Callable[[int, int, str], None]


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

    return f"{paper_run_name}_complete_result_package_{utc_archive_token()}_{_short_commit()}.zip"


def build_paper_result_closure_preflight_report(package_search_root: str) -> dict[str, Any]:
    """检查完整论文结果闭合所需的结果包是否齐备。

    该检查属于完整论文实验层的门禁: 它不替代各个 builder 的 schema 校验,
    只在长耗时闭合命令启动前发现明显缺失的前序结果包。
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
    这样 Notebook 和服务器入口都不需要重复拼接命令、结果包名称和输出路径。
    """

    complete_archive_name = _complete_archive_name(paper_run_name)
    return [
        [sys.executable, "scripts/write_pilot_paper_result_records.py", "--package-search-root", package_search_root, "--materialize-only"],
        [sys.executable, "scripts/write_primary_baseline_result_candidates.py", "--target-fpr-override", target_fpr],
        [sys.executable, "scripts/write_primary_baseline_formal_import_protocol.py"],
        [sys.executable, "scripts/write_external_baseline_comparison_outputs.py"],
        [sys.executable, "scripts/write_pilot_paper_result_records.py", "--require-existing-evidence"],
        [
            sys.executable,
            "scripts/write_pilot_paper_fixed_fpr_common_protocol_outputs.py",
            "--candidate-records-path",
            "outputs/pilot_paper_fixed_fpr_results/pilot_paper_result_records.jsonl",
            "--require-existing-evidence",
        ],
        [sys.executable, "scripts/write_pilot_paper_result_analysis_outputs.py"],
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
    before_command: CommandHook | None = None,
    progress_hook: ProgressHook | None = None,
) -> dict[str, Any]:
    """运行论文结果闭合命令, 并返回最新完整结果包路径。

    `before_command` 与 `progress_hook` 是可选的入口层扩展点。服务器脚本可以
    不传入任何 hook, Colab 包装层可以用 hook 追加 Notebook runtime 报告和总体进度显示。
    """

    preflight_report = require_paper_result_closure_inputs(package_search_root)
    commands = build_paper_result_closure_commands(
        package_search_root=package_search_root,
        complete_drive_output_dir=complete_drive_output_dir,
        target_fpr=target_fpr,
        paper_run_name=paper_run_name,
    )
    for command_index, command in enumerate(commands, start=1):
        command_name = command[1] if len(command) > 1 else command[0]
        if before_command is not None:
            before_command(command)
        print("run_repository_command", " ".join(command))
        subprocess.run(command, check=True)
        if progress_hook is not None:
            progress_hook(command_index, len(commands), command_name)

    archive_pattern = f"{paper_run_name}_complete_result_package_*.zip"
    complete_archives = sorted(Path(complete_drive_output_dir).glob(archive_pattern))
    if not complete_archives:
        raise FileNotFoundError("未找到写回结果目录的当前论文运行层级完整结果包")
    return {
        "latest_complete_archive": str(complete_archives[-1]),
        "command_count": len(commands),
        "paper_run_name": paper_run_name,
        "target_fpr": target_fpr,
        "preflight_report": preflight_report,
    }
