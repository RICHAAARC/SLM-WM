"""提供仓库 intake 检查。"""

from __future__ import annotations

import json
from pathlib import Path
import sys

DIRECTORIES = [
    "configs",
    "docs",
    "tools",
    "tests",
    "main",
    "experiments",
    "paper_experiments",
    "external_baseline",
    "paper_workflow",
    "paper_workflow/notebooks",
    "scripts",
    "outputs",
    ".codex",
]


def inspect_repository(root: str | Path) -> dict:
    """返回当前仓库的基础目录状态。"""
    root_path = Path(root)
    directory_status = {
        name: {"exists": (root_path / name).exists(), "path": str(root_path / name)} for name in DIRECTORIES
    }
    contract_path = root_path / ".codex" / "project_contract.md"
    return {
        "repository_mode": "governed_repository" if contract_path.exists() else "uninitialized_repository",
        "project_contract_exists": contract_path.exists(),
        "directory_status": directory_status,
    }


def main(argv: list[str] | None = None) -> None:
    """命令行入口。"""
    arguments = argv or sys.argv
    root = Path(arguments[1]) if len(arguments) > 1 else Path.cwd()
    print(json.dumps(inspect_repository(root), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main(sys.argv)
