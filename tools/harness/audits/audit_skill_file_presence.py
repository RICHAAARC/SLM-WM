"""审计必需 skill 文件和章节是否存在。"""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.harness.lib.json_report import build_report, exit_with_report

REQUIRED_SKILLS = {
    "repository_intake.skill.md",
    "naming_governance.skill.md",
    "test_case_governance.skill.md",
    "artifact_rebuild.skill.md",
    "claim_audit.skill.md",
    "placeholder_random_field_governance.skill.md",
}
REQUIRED_SECTIONS = ["## Purpose", "## Scope", "## Blocking Rules", "## Forbidden Changes"]


def run_audit(root: str | Path) -> dict:
    root_path = Path(root)
    skill_root = root_path / ".codex" / "skills"
    violations = []
    checked_paths = []
    for name in sorted(REQUIRED_SKILLS):
        path = skill_root / name
        checked_paths.append(str(path.relative_to(root_path)))
        if not path.exists():
            violations.append({"path": str(path.relative_to(root_path)), "reason": "missing_skill"})
            continue
        text = path.read_text(encoding="utf-8")
        for section in REQUIRED_SECTIONS:
            if section not in text:
                violations.append({"path": str(path.relative_to(root_path)), "reason": "missing_section", "section": section})
    return build_report("audit_skill_file_presence", "fail" if violations else "pass", violations, checked_paths)


def main() -> None:
    exit_with_report(run_audit(Path.cwd()))


if __name__ == "__main__":
    main()
