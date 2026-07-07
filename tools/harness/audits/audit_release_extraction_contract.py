"""审计最小论文附件抽离规则是否存在。"""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.harness.lib.json_report import build_report, exit_with_report

REQUIRED_PROFILE_NAMES = [
    "development_repository",
    "full_experiment_execution_package",
    "paper_artifact_rebuild_package",
    "minimal_method_package",
]

REQUIRED_EXCLUDED_PATHS = [
    ".codex/",
    "tools/harness/",
    "outputs/",
]


def run_audit(root: str | Path) -> dict:
    """检查抽离 profile 文档和抽取脚本是否存在并包含关键边界。"""
    root_path = Path(root)
    violations = []
    checked_paths = ["docs/extraction_profiles.md", "docs/release_boundary.md", "scripts/extract_minimal_paper_package.py"]

    profile_doc = root_path / "docs" / "extraction_profiles.md"
    release_doc = root_path / "docs" / "release_boundary.md"
    extraction_script = root_path / "scripts" / "extract_minimal_paper_package.py"

    if not profile_doc.exists():
        violations.append({"path": "docs/extraction_profiles.md", "reason": "missing_extraction_profiles"})
        profile_text = ""
    else:
        profile_text = profile_doc.read_text(encoding="utf-8")

    if not release_doc.exists():
        violations.append({"path": "docs/release_boundary.md", "reason": "missing_release_boundary"})
        release_text = ""
    else:
        release_text = release_doc.read_text(encoding="utf-8")

    if not extraction_script.exists():
        violations.append({"path": "scripts/extract_minimal_paper_package.py", "reason": "missing_extraction_script"})
        script_text = ""
    else:
        script_text = extraction_script.read_text(encoding="utf-8")

    combined_text = "\n".join([profile_text, release_text, script_text])
    for profile_name in REQUIRED_PROFILE_NAMES:
        if profile_name not in combined_text:
            violations.append({"path": "docs/extraction_profiles.md", "reason": "missing_profile_name", "profile_name": profile_name})
    for excluded_path in REQUIRED_EXCLUDED_PATHS:
        if excluded_path not in combined_text:
            violations.append({"path": "docs/extraction_profiles.md", "reason": "missing_excluded_path", "excluded_path": excluded_path})

    return build_report("audit_release_extraction_contract", "fail" if violations else "pass", violations, checked_paths)


def main() -> None:
    """命令行入口。"""
    exit_with_report(run_audit(Path.cwd()))


if __name__ == "__main__":
    main()
