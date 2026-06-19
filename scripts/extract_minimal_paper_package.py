"""按抽离 profile 生成论文附件候选目录。"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class ExtractionProfile:
    """表示一个可执行的论文附件抽离 profile。"""

    profile_name: str
    include_paths: tuple[str, ...]
    exclude_parts: tuple[str, ...]


PROFILES = {
    "minimal_method_package": ExtractionProfile(
        profile_name="minimal_method_package",
        include_paths=(
            "main/core",
            "main/methods",
            "main/protocol",
            "configs",
            "README.md",
            "pyproject.toml",
        ),
        exclude_parts=(
            ".codex",
            "tools",
            "tests",
            "experiments",
            "scripts",
            "paper_workflow",
            "audit_reports",
            "outputs",
            "__pycache__",
            ".pytest_cache",
        ),
    ),
    "paper_artifact_rebuild_package": ExtractionProfile(
        profile_name="paper_artifact_rebuild_package",
        include_paths=(
            "main",
            "configs",
            "experiments",
            "scripts",
            "docs/artifact_rebuild.md",
            "docs/field_registry.md",
            "docs/file_organization.md",
            "docs/release_boundary.md",
            "docs/extraction_profiles.md",
            "docs/intermediate_state_governance.md",
            "tests/functional",
            "README.md",
            "pyproject.toml",
        ),
        exclude_parts=(
            ".codex",
            "tools",
            "tests/constraints",
            "tests/integration",
            "tests/helpers",
            "paper_workflow",
            "audit_reports",
            "outputs",
            "__pycache__",
            ".pytest_cache",
        ),
    ),
}


def should_skip(relative_path: Path, exclude_parts: Iterable[str]) -> bool:
    """判断相对路径是否应从抽离包中排除。"""
    normalized = relative_path.as_posix()
    parts = set(relative_path.parts)
    for excluded in exclude_parts:
        excluded_normalized = excluded.strip("/").replace("\\", "/")
        if excluded_normalized in parts or normalized == excluded_normalized or normalized.startswith(f"{excluded_normalized}/"):
            return True
    return False


def iter_copy_candidates(root_path: Path, include_path: str, profile: ExtractionProfile) -> Iterable[Path]:
    """遍历某个 include path 下允许复制的文件。"""
    source = root_path / include_path
    if not source.exists():
        return
    if source.is_file():
        relative = source.relative_to(root_path)
        if not should_skip(relative, profile.exclude_parts):
            yield source
        return
    for path in source.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root_path)
        if should_skip(relative, profile.exclude_parts):
            continue
        yield path


def extract_profile(root: str | Path, output: str | Path, profile_name: str, dry_run: bool = False) -> dict:
    """按指定 profile 复制文件, 并返回抽离清单。"""
    root_path = Path(root).resolve()
    output_path = Path(output).resolve()
    if profile_name not in PROFILES:
        raise ValueError(f"不支持的抽离 profile: {profile_name}")
    profile = PROFILES[profile_name]

    copied_files: list[str] = []
    missing_paths: list[str] = []
    for include_path in profile.include_paths:
        source = root_path / include_path
        if not source.exists():
            missing_paths.append(include_path)
            continue
        for source_file in iter_copy_candidates(root_path, include_path, profile):
            relative = source_file.relative_to(root_path)
            copied_files.append(relative.as_posix())
            if dry_run:
                continue
            target_file = output_path / relative
            target_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_file, target_file)

    manifest = {
        "profile_name": profile.profile_name,
        "root_path": str(root_path),
        "output_path": str(output_path),
        "copied_files": sorted(copied_files),
        "missing_paths": missing_paths,
        "excluded_parts": list(profile.exclude_parts),
        "dry_run": dry_run,
    }
    if not dry_run:
        output_path.mkdir(parents=True, exist_ok=True)
        manifest_path = output_path / "extraction_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器。"""
    parser = argparse.ArgumentParser(description="按治理 profile 抽离论文附件候选目录。")
    parser.add_argument(
        "--profile",
        choices=sorted(PROFILES),
        default="minimal_method_package",
        help="选择抽离 profile。",
    )
    parser.add_argument(
        "--root",
        default=".",
        help="仓库根目录。",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="输出目录。建议使用 release_packages/ 下的未提交目录。",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只输出将要复制的文件清单, 不写入文件。",
    )
    return parser


def main() -> None:
    """命令行入口。"""
    parser = build_parser()
    args = parser.parse_args()
    manifest = extract_profile(args.root, args.output, args.profile, args.dry_run)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
