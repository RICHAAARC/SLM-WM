"""从已验证随机化聚合包执行 CPU 论文结果闭合."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.protocol.paper_run_config import (  # noqa: E402
    RUN_DEFAULTS,
    normalize_paper_run_name,
    validate_frozen_paper_run_target_fpr,
)
from experiments.runtime.repository_environment import (  # noqa: E402
    resolve_code_version,
)
from paper_experiments.runners.closure_package_selection import (  # noqa: E402
    normalize_clean_code_version,
)
from paper_experiments.runners.randomization_aggregate_provenance import (  # noqa: E402
    validate_randomization_aggregate_provenance,
)
from scripts.paper_result_closure import (  # noqa: E402
    build_paper_result_closure_commands,
    run_paper_result_closure_commands,
)


def execute_server_result_closure(
    *,
    root: str | Path,
    paper_run_name: str,
    randomization_aggregate_package_path: str | Path,
    complete_output_dir: str | Path,
    repository_commit: str,
    dry_run: bool = False,
) -> dict[str, Any]:
    """核对服务器提交后规划或执行完整 CPU 结果闭合."""

    root_path = Path(root).resolve()
    run_name = normalize_paper_run_name(paper_run_name)
    target_fpr = validate_frozen_paper_run_target_fpr(
        run_name,
        float(RUN_DEFAULTS[run_name]["target_fpr"]),
    )
    expected_commit = normalize_clean_code_version(repository_commit)
    if resolve_code_version(root_path) != expected_commit:
        raise RuntimeError("汇总执行仓库必须位于声明的 clean Git 提交")
    source = validate_randomization_aggregate_provenance(
        randomization_aggregate_package_path,
        paper_run_name=run_name,
        target_fpr=target_fpr,
    )
    if source.common_code_version != expected_commit:
        raise RuntimeError("聚合包代码提交与汇总服务器提交不一致")
    if dry_run:
        commands = build_paper_result_closure_commands(
            randomization_aggregate_package_path=source.package_path,
            paper_run_name=run_name,
            target_fpr=target_fpr,
            root=root_path,
        )
        return {
            "paper_run_name": run_name,
            "target_fpr": target_fpr,
            "repository_commit": expected_commit,
            "randomization_aggregate_package_path": source.package_path.as_posix(),
            "randomization_aggregate_package_sha256": source.package_sha256,
            "randomization_aggregate_digest": source.randomization_aggregate_digest,
            "statistics_commands": commands,
            "statistics_command_count": len(commands),
            "dry_run": True,
            "paper_result_evidence_ready": False,
            "supports_paper_claim": False,
        }
    result = run_paper_result_closure_commands(
        package_search_root=source.package_path,
        complete_drive_output_dir=complete_output_dir,
        paper_run_name=run_name,
        target_fpr=target_fpr,
        root=root_path,
        expected_repository_commit=expected_commit,
    )
    return {
        **result,
        "repository_commit": expected_commit,
        "dry_run": False,
    }


def build_parser() -> argparse.ArgumentParser:
    """构造保留精确运行身份的命令行参数解析器."""

    parser = argparse.ArgumentParser(
        description="从已验证精确随机化聚合包执行服务器论文结果闭合."
    )
    parser.add_argument("--root", default=".", help="仓库根目录.")
    parser.add_argument(
        "--repository-commit",
        required=True,
        help="汇总执行使用的精确40位小写 Git SHA.",
    )
    parser.add_argument(
        "--paper-run-name",
        required=True,
        choices=sorted(RUN_DEFAULTS),
        help="必须显式指定论文运行层级, 避免误用其他统计规模.",
    )
    parser.add_argument(
        "--randomization-aggregate-package-path",
        required=True,
        help="精确随机化聚合来源 ZIP 的显式路径.",
    )
    parser.add_argument(
        "--complete-output-dir",
        required=True,
        help="完整结果包输出目录, 必须位于仓库 outputs/ 下.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只复验聚合来源并输出统计命令计划, 不写派生结果.",
    )
    return parser


def main() -> None:
    """解析参数并执行服务器结果闭合入口."""

    args = build_parser().parse_args()
    result = execute_server_result_closure(
        root=args.root,
        paper_run_name=args.paper_run_name,
        randomization_aggregate_package_path=(
            args.randomization_aggregate_package_path
        ),
        complete_output_dir=args.complete_output_dir,
        repository_commit=args.repository_commit,
        dry_run=args.dry_run,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
