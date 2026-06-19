"""审计基础目录边界是否存在。"""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.harness.lib.json_report import build_report, exit_with_report

REQUIRED_PATHS = [
    ".codex/project_contract.md",
    "docs/file_organization.md",
    "tools/harness/run_all_audits.py",
    "tests/constraints",
]
FORBIDDEN_ROOT_OUTPUT_DIRS = ["audit_reports"]


def run_audit(root: str | Path) -> dict:
    root_path = Path(root)
    violations = []
    checked_paths = []
    for relative in REQUIRED_PATHS:
        checked_paths.append(relative)
        if not (root_path / relative).exists():
            violations.append({"path": relative, "reason": "required_path_missing"})
    for relative in FORBIDDEN_ROOT_OUTPUT_DIRS:
        checked_paths.append(relative)
        if (root_path / relative).exists():
            violations.append({"path": relative, "reason": "root_output_directory_forbidden"})
    return build_report("audit_file_organization_contract", "fail" if violations else "pass", violations, checked_paths)


def main() -> None:
    exit_with_report(run_audit(Path.cwd()))


if __name__ == "__main__":
    main()
