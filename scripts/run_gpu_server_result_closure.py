"""在汇总服务器执行论文结果闭合。

该脚本面向独立 GPU 服务器体系: 计算服务器把各自结果包上传到本地交换目录,
汇总服务器从该目录递归物化结果包并运行既有 result closure 命令。脚本不挂载
Google Drive, 也不修改 Colab Notebook 的落盘路径。
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.protocol.paper_run_config import RUN_DEFAULTS, build_paper_run_config, normalize_paper_run_name
from paper_workflow.colab_utils.paper_result_closure import (
    build_paper_result_closure_preflight_report,
    run_paper_result_closure_commands,
)


def stable_json_text(value: Any) -> str:
    """把 JSON 兼容对象转为稳定文本。"""

    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def set_env(name: str, value: str | int | float) -> None:
    """写入当前进程环境变量。"""

    os.environ[name] = str(value)


def target_fpr_text(value: float) -> str:
    """生成稳定的 FPR 文本表示。"""

    return f"{float(value):g}"


def configure_closure_environment(
    *,
    root: str | Path,
    paper_run_name: str,
    package_search_root: str | Path,
    target_fpr_override: str,
) -> dict[str, Any]:
    """配置汇总服务器本地结果闭合环境。"""

    root_path = Path(root).resolve()
    normalized_run_name = normalize_paper_run_name(paper_run_name)
    defaults = RUN_DEFAULTS[normalized_run_name]
    resolved_package_root = Path(package_search_root).expanduser()
    if not resolved_package_root.is_absolute():
        resolved_package_root = (root_path / resolved_package_root).resolve()
    set_env("SLM_WM_PAPER_RUN_NAME", normalized_run_name)
    set_env("SLM_WM_PROMPT_SET", defaults["prompt_set"])
    set_env("SLM_WM_PROMPT_FILE", defaults["prompt_file"])
    set_env("SLM_WM_DRIVE_RESULT_ROOT", resolved_package_root.as_posix())
    set_env("SLM_WM_PAPER_RUN_SAMPLE_COUNT", "all")
    set_env("SLM_WM_PAPER_RUN_TARGET_FPR", target_fpr_override or defaults["target_fpr"])
    set_env("SLM_WM_PAPER_RUN_MINIMUM_CLEAN_NEGATIVE_COUNT", os.environ.get("SLM_WM_PAPER_RUN_MINIMUM_CLEAN_NEGATIVE_COUNT", "100"))
    set_env(
        "SLM_WM_PAPER_RUN_DATASET_QUALITY_MINIMUM_COUNT",
        os.environ.get("SLM_WM_PAPER_RUN_DATASET_QUALITY_MINIMUM_COUNT", "100"),
    )
    paper_run = build_paper_run_config(root_path)
    resolved_target_fpr = target_fpr_text(paper_run.target_fpr)
    set_env("SLM_WM_PAPER_RUN_TARGET_FPR", resolved_target_fpr)
    set_env("SLM_WM_PROTOCOL_PROFILE", f"{paper_run.run_name}_fixed_fpr_{resolved_target_fpr.replace('.', '_')}")
    set_env("SLM_WM_PAPER_RUN_EXPECTED_SAMPLE_COUNT", paper_run.sample_count)
    return {
        "root": root_path.as_posix(),
        "paper_run": paper_run.to_dict(),
        "package_search_root": resolved_package_root.as_posix(),
        "target_fpr": resolved_target_fpr,
    }


def execute_server_result_closure(
    *,
    root: str | Path,
    paper_run_name: str,
    package_search_root: str | Path,
    complete_output_dir: str | Path,
    target_fpr_override: str = "",
    dry_run: bool = False,
) -> dict[str, Any]:
    """执行汇总服务器结果闭合。"""

    root_path = Path(root).resolve()
    os.chdir(root_path)
    environment_report = configure_closure_environment(
        root=root_path,
        paper_run_name=paper_run_name,
        package_search_root=package_search_root,
        target_fpr_override=target_fpr_override,
    )
    resolved_complete_output_dir = Path(complete_output_dir).expanduser()
    if not resolved_complete_output_dir.is_absolute():
        resolved_complete_output_dir = (Path(root).resolve() / resolved_complete_output_dir).resolve()
    preflight_report = build_paper_result_closure_preflight_report(environment_report["package_search_root"])
    plan = {
        "server_result_closure_plan_ready": bool(preflight_report["closure_preflight_ready"]),
        "environment_report": environment_report,
        "complete_output_dir": resolved_complete_output_dir.as_posix(),
        "preflight_report": preflight_report,
        "dry_run": dry_run,
    }
    if dry_run:
        return plan
    closure_result = run_paper_result_closure_commands(
        package_search_root=environment_report["package_search_root"],
        complete_drive_output_dir=resolved_complete_output_dir.as_posix(),
        target_fpr=environment_report["target_fpr"],
        paper_run_name=paper_run_name,
    )
    return {
        **plan,
        "closure_result": closure_result,
    }


def build_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器。"""

    parser = argparse.ArgumentParser(description="在汇总服务器执行论文结果闭合。")
    parser.add_argument("--root", default=".", help="仓库根目录。")
    parser.add_argument("--paper-run-name", default="full_paper", choices=sorted(RUN_DEFAULTS), help="论文运行层级。")
    parser.add_argument("--package-search-root", required=True, help="三台计算服务器上传结果包后的本地交换目录。")
    parser.add_argument("--complete-output-dir", required=True, help="完整结果包输出目录, 推荐位于交换目录下。")
    parser.add_argument("--target-fpr", default="", help="可选 fixed-FPR 覆盖值; 为空时使用论文运行层级默认值。")
    parser.add_argument("--dry-run", action="store_true", help="只检查输入包覆盖情况, 不执行闭合命令。")
    return parser


def main() -> None:
    """命令行入口。"""

    args = build_parser().parse_args()
    result = execute_server_result_closure(
        root=args.root,
        paper_run_name=args.paper_run_name,
        package_search_root=args.package_search_root,
        complete_output_dir=args.complete_output_dir,
        target_fpr_override=args.target_fpr,
        dry_run=args.dry_run,
    )
    print(stable_json_text(result), end="")


if __name__ == "__main__":
    main()
