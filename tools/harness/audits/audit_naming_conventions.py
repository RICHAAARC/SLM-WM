"""审计正式文件和目录命名。"""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.harness.lib.file_scanner import should_skip_path
from tools.harness.lib.json_report import build_report, exit_with_report
from tools.harness.lib.naming_rules import (
    has_reserved_progress_marker,
    has_weak_semantic_token,
    is_allowed_directory_name,
    is_allowed_file_name,
)

TEXT_SUFFIXES = {".ipynb", ".md", ".py", ".json", ".toml", ".txt", ".yml", ".yaml"}


def is_docs_path(path: Path) -> bool:
    """判断路径是否位于允许保留规划词的文档目录。"""
    return bool(path.parts) and path.parts[0] == "docs"


def run_audit(root: str | Path) -> dict:
    root_path = Path(root)
    violations = []
    checked_paths = []
    for path in root_path.rglob("*"):
        relative = path.relative_to(root_path)
        if should_skip_path(relative):
            continue
        checked_paths.append(str(relative))
        if path.is_dir():
            if not is_allowed_directory_name(path.name):
                violations.append({"path": str(relative), "reason": "directory_name_not_snake_case"})
        elif path.is_file():
            if not is_allowed_file_name(path.name):
                violations.append({"path": str(relative), "reason": "file_name_not_snake_case"})
        if has_weak_semantic_token(path.stem if path.is_file() else path.name):
            violations.append({"path": str(relative), "reason": "weak_semantic_token"})
        if not is_docs_path(relative):
            if has_reserved_progress_marker(str(relative)):
                violations.append({"path": str(relative), "reason": "reserved_progress_marker"})
            if path.is_file() and path.suffix.lower() in TEXT_SUFFIXES:
                text = path.read_text(encoding="utf-8")
                if has_reserved_progress_marker(text):
                    violations.append({"path": str(relative), "reason": "reserved_progress_marker_in_text"})
    return build_report("audit_naming_conventions", "fail" if violations else "pass", violations, checked_paths)


def main() -> None:
    exit_with_report(run_audit(Path.cwd()))


if __name__ == "__main__":
    main()
