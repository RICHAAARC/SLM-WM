"""写出核心包边界的本地审计产物。

该脚本属于仓库辅助命令, 不属于 `main/` 核心方法包。它只读取源码和文档,
并把本地治理输出写入 `outputs/core_method_boundary/`。
"""

from __future__ import annotations

import argparse
import ast
import importlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.artifacts.artifact_manifest import build_artifact_manifest
from experiments.runtime.repository_environment import resolve_code_version
from main.core.digest import build_stable_digest


CONSTRUCTION_UNIT_NAME = "core_method_boundary"
DEFAULT_OUTPUT_DIR = Path("outputs/core_method_boundary")
REQUIRED_PACKAGE_DIRS = (
    "main/core",
    "main/methods",
)
FORBIDDEN_IMPORT_PREFIXES = (
    "google.colab",
    "pydrive",
    "paper_workflow",
    "experiments",
    "scripts",
    "tests",
    "tools",
    "baseline",
    "baselines",
    "external_baseline",
)
FORBIDDEN_TEXT_PATTERNS = (
    "google.colab",
    "pydrive",
    "paper_workflow",
    "experiments/",
    "experiments\\",
    "scripts/",
    "scripts\\",
    "tests/",
    "tests\\",
    "tools/harness",
    "tools\\harness",
    "/content/drive",
    "Google Drive",
    "My Drive",
    "baseline_runner",
    "external_baseline",
    "baselines/",
    "baselines\\",
)


def stable_json_text(value: Any) -> str:
    """把 JSON 兼容对象转为稳定、可读的 UTF-8 文本。"""
    return json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n"


def extract_imported_modules(path: Path) -> list[str]:
    """从 Python 文件中提取导入模块名。"""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.append(node.module)
    return modules


def is_forbidden_import(module_name: str) -> bool:
    """判断模块名是否命中核心包禁止导入前缀。"""
    return any(
        module_name == prefix or module_name.startswith(f"{prefix}.") for prefix in FORBIDDEN_IMPORT_PREFIXES
    )


def iter_main_python_files(root_path: Path) -> list[Path]:
    """列出 `main/` 下参与边界检查的 Python 文件。"""
    main_root = root_path / "main"
    if not main_root.exists():
        return []
    return sorted(path for path in main_root.rglob("*.py") if "__pycache__" not in path.parts)


def relative_text(root_path: Path, path: Path) -> str:
    """返回稳定的 POSIX 风格相对路径。"""
    return path.relative_to(root_path).as_posix()


def build_core_boundary_report(root: str | Path) -> dict[str, Any]:
    """构造核心包边界检查报告。"""
    root_path = Path(root).resolve()
    checked_paths: list[str] = []
    violations: list[dict[str, str]] = []

    for package_dir in REQUIRED_PACKAGE_DIRS:
        checked_paths.append(package_dir)
        if not (root_path / package_dir).is_dir():
            violations.append({"path": package_dir, "reason": "required_package_dir_missing", "value": package_dir})

    for path in iter_main_python_files(root_path):
        relative_path = relative_text(root_path, path)
        checked_paths.append(relative_path)
        for module_name in extract_imported_modules(path):
            if is_forbidden_import(module_name):
                violations.append(
                    {
                        "path": relative_path,
                        "reason": "forbidden_import_in_main_package",
                        "value": module_name,
                    }
                )
        text = path.read_text(encoding="utf-8")
        for pattern in FORBIDDEN_TEXT_PATTERNS:
            if pattern in text:
                violations.append(
                    {
                        "path": relative_path,
                        "reason": "forbidden_runtime_boundary_literal",
                        "value": pattern,
                    }
                )

    return {
        "artifact_id": "core_boundary_report",
        "artifact_type": "boundary_report",
        "construction_unit_name": CONSTRUCTION_UNIT_NAME,
        "decision": "fail" if violations else "pass",
        "checked_paths": checked_paths,
        "violations": violations,
        "metadata": {
            "package_root": "main",
            "required_package_dirs": list(REQUIRED_PACKAGE_DIRS),
            "forbidden_import_prefixes": list(FORBIDDEN_IMPORT_PREFIXES),
            "forbidden_text_patterns": list(FORBIDDEN_TEXT_PATTERNS),
        },
        "summary": {
            "checked_path_count": len(checked_paths),
            "violation_count": len(violations),
        },
    }


def build_core_import_report(root: str | Path) -> dict[str, Any]:
    """构造 `import main` 独立导入检查报告。"""
    root_path = Path(root).resolve()
    checked_paths = ["main/__init__.py"]
    violations: list[dict[str, str]] = []
    try:
        importlib.import_module("main")
    except Exception as exc:  # pragma: no cover - 仅在导入异常时提供诊断。
        violations.append({"path": "main/__init__.py", "reason": "main_import_failed", "value": repr(exc)})

    return {
        "artifact_id": "core_import_report",
        "artifact_type": "import_report",
        "construction_unit_name": CONSTRUCTION_UNIT_NAME,
        "decision": "fail" if violations else "pass",
        "checked_paths": checked_paths,
        "violations": violations,
        "metadata": {
            "package_root": "main",
            "import_target": "main",
            "import_succeeded": not violations,
            "repository_root": str(root_path),
        },
        "summary": {
            "checked_path_count": len(checked_paths),
            "violation_count": len(violations),
        },
    }


def build_core_package_layout(root: str | Path) -> str:
    """生成 `main/` 包结构清单文本。"""
    root_path = Path(root).resolve()
    main_root = root_path / "main"
    lines = ["main/"]
    if not main_root.exists():
        return "\n".join(lines) + "\n"
    for path in sorted(main_root.rglob("*")):
        if "__pycache__" in path.parts:
            continue
        relative_path = path.relative_to(root_path).as_posix()
        suffix = "/" if path.is_dir() else ""
        lines.append(f"{relative_path}{suffix}")
    return "\n".join(lines) + "\n"


def ensure_output_dir_under_outputs(root_path: Path, output_dir: Path) -> Path:
    """确保输出目录位于 `outputs/` 下。"""
    resolved_output_dir = (root_path / output_dir).resolve() if not output_dir.is_absolute() else output_dir.resolve()
    outputs_root = (root_path / "outputs").resolve()
    try:
        resolved_output_dir.relative_to(outputs_root)
    except ValueError as exc:
        raise ValueError("核心包边界输出目录必须位于 outputs/ 下") from exc
    return resolved_output_dir


def write_core_boundary_outputs(root: str | Path, output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    """写出核心包边界报告、包结构清单和 manifest。"""
    root_path = Path(root).resolve()
    resolved_output_dir = ensure_output_dir_under_outputs(root_path, Path(output_dir))
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    boundary_report = build_core_boundary_report(root_path)
    import_report = build_core_import_report(root_path)
    layout_text = build_core_package_layout(root_path)

    report_paths = {
        "core_boundary_report": resolved_output_dir / "core_boundary_report.json",
        "core_import_report": resolved_output_dir / "core_import_report.json",
        "core_package_layout": resolved_output_dir / "core_package_layout.txt",
        "manifest": resolved_output_dir / "manifest.local.json",
    }
    report_paths["core_boundary_report"].write_text(stable_json_text(boundary_report), encoding="utf-8")
    report_paths["core_import_report"].write_text(stable_json_text(import_report), encoding="utf-8")
    report_paths["core_package_layout"].write_text(layout_text, encoding="utf-8")

    generated_at = datetime.now(timezone.utc).isoformat()
    output_paths = tuple(path.relative_to(root_path).as_posix() for path in report_paths.values())
    config = {
        "construction_unit_name": CONSTRUCTION_UNIT_NAME,
        "required_package_dirs": list(REQUIRED_PACKAGE_DIRS),
        "forbidden_import_prefixes": list(FORBIDDEN_IMPORT_PREFIXES),
        "forbidden_text_patterns": list(FORBIDDEN_TEXT_PATTERNS),
        "report_digest": build_stable_digest(
            {
                "boundary_report": boundary_report,
                "import_report": import_report,
                "layout_text": layout_text,
            }
        ),
    }
    manifest = build_artifact_manifest(
        artifact_id="core_package_boundary_manifest",
        artifact_type="local_manifest",
        input_paths=(
            "AGENTS.md",
            ".codex/project_contract.md",
            "docs",
            "main",
            "scripts/write_core_package_boundary_outputs.py",
            "tests/constraints/test_main_boundary_contract.py",
        ),
        output_paths=output_paths,
        config=config,
        code_version=resolve_code_version(root_path),
        rebuild_command="python scripts/write_core_package_boundary_outputs.py",
        metadata={
            "construction_unit_name": CONSTRUCTION_UNIT_NAME,
            "generated_at": generated_at,
            "decision": "fail" if boundary_report["decision"] != "pass" or import_report["decision"] != "pass" else "pass",
        },
    ).to_dict()
    report_paths["manifest"].write_text(stable_json_text(manifest), encoding="utf-8")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器。"""
    parser = argparse.ArgumentParser(description="写出核心包边界本地输出。")
    parser.add_argument("--root", default=".", help="仓库根目录。")
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="输出目录, 必须位于 outputs/ 下。",
    )
    return parser


def main() -> None:
    """命令行入口。"""
    parser = build_parser()
    args = parser.parse_args()
    manifest = write_core_boundary_outputs(args.root, args.output_dir)
    print(stable_json_text(manifest), end="")


if __name__ == "__main__":
    main()
