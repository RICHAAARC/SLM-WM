"""审计仓库三层结构的导入方向。"""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.harness.lib.dependency_rules import (
    FORBIDDEN_IMPORT_PREFIXES_BY_ROOT,
    extract_imported_modules,
    get_boundary_root,
    is_forbidden_import,
)
from tools.harness.lib.file_scanner import should_skip_path
from tools.harness.lib.json_report import build_report, exit_with_report


def run_audit(root: str | Path) -> dict:
    """检查受治理目录的导入方向是否满足发布层边界。"""
    root_path = Path(root)
    violations = []
    checked_paths = []
    required_roots = sorted({boundary_root.split("/", 1)[0] for boundary_root in FORBIDDEN_IMPORT_PREFIXES_BY_ROOT})
    missing_roots = [name for name in required_roots if not (root_path / name).exists()]
    if missing_roots:
        return build_report(
            "audit_dependency_boundaries",
            "fail",
            [{"path": name, "reason": "missing_boundary_root"} for name in missing_roots],
            required_roots,
        )

    for path in root_path.rglob("*.py"):
        relative = path.relative_to(root_path)
        if should_skip_path(relative):
            continue
        boundary_root = get_boundary_root(relative)
        if boundary_root is None:
            continue
        checked_paths.append(str(relative))
        forbidden_prefixes = FORBIDDEN_IMPORT_PREFIXES_BY_ROOT[boundary_root]
        for module_name in extract_imported_modules(path):
            if is_forbidden_import(module_name, forbidden_prefixes):
                violations.append(
                    {
                        "path": str(relative),
                        "reason": "forbidden_import_for_extraction_boundary",
                        "imported_module": module_name,
                        "boundary_root": boundary_root,
                    }
                )
    return build_report("audit_dependency_boundaries", "fail" if violations else "pass", violations, checked_paths)


def main() -> None:
    """命令行入口。"""
    exit_with_report(run_audit(Path.cwd()))


if __name__ == "__main__":
    main()
