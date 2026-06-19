"""审计测试目录和 pytest marker 基本约束。"""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.harness.lib.json_report import build_report, exit_with_report

REQUIRED_TEST_DIRS = ["tests/constraints", "tests/functional", "tests/integration", "tests/helpers", "tests/fixtures"]
REQUIRED_MARKERS = ["unit", "constraint", "quick", "integration", "smoke", "slow", "formal"]


def run_audit(root: str | Path) -> dict:
    root_path = Path(root)
    violations = []
    checked_paths = []
    for relative in REQUIRED_TEST_DIRS:
        path = root_path / relative
        checked_paths.append(relative)
        if not path.exists():
            violations.append({"path": relative, "reason": "missing_test_directory"})
    tests_root = root_path / "tests"
    if tests_root.exists():
        for path in tests_root.glob("test_*.py"):
            violations.append({"path": str(path.relative_to(root_path)), "reason": "root_level_test_file_forbidden"})
        helper_root = tests_root / "helpers"
        if helper_root.exists():
            for path in helper_root.glob("test_*.py"):
                violations.append({"path": str(path.relative_to(root_path)), "reason": "helper_test_prefix_forbidden"})
    pyproject = root_path / "pyproject.toml"
    checked_paths.append("pyproject.toml")
    if not pyproject.exists():
        violations.append({"path": "pyproject.toml", "reason": "missing_pyproject"})
    else:
        text = pyproject.read_text(encoding="utf-8")
        for marker in REQUIRED_MARKERS:
            if marker not in text:
                violations.append({"path": "pyproject.toml", "reason": "missing_marker", "marker": marker})
    return build_report("audit_test_case_constraints", "fail" if violations else "pass", violations, checked_paths)


def main() -> None:
    exit_with_report(run_audit(Path.cwd()))


if __name__ == "__main__":
    main()
