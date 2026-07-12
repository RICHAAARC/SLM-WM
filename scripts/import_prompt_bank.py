"""从冻结外部来源构建或复验可逐字节重建的 Prompt bank."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.protocol.prompt_sources import (
    PROMPT_SELECTION_MANIFEST_PATH,
    PROMPT_SOURCE_REGISTRY_PATH,
    audit_committed_prompt_bank,
    build_prompt_selection_rows,
    build_prompt_source_registry,
    read_selection_manifest,
    selection_manifest_bytes,
    verify_selection_against_sources,
    write_prompt_files_from_selection,
    write_selection_manifest,
)


DEFAULT_COCO_SOURCE = Path("outputs/prompt_sources/captions_train2017.json")
DEFAULT_PARTI_SOURCE = Path("outputs/prompt_sources/PartiPrompts.tsv")
DEFAULT_OUTPUT_ROOT = Path("outputs/prompt_rebuild")


def _stable_report(value: Any) -> str:
    """把命令报告编码为稳定、便于人工检查的 JSON."""

    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)


def _path_sha256(path: Path) -> str:
    """计算输出文件摘要, 使构建报告可直接用于复制前复核."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_persistent_output_root(
    repository_root: str | Path,
    output_root: str | Path,
) -> Path:
    """要求命令持久化产物位于项目 outputs/ 边界内."""

    root_path = Path(repository_root).resolve()
    resolved_output_path = Path(output_root).resolve()
    try:
        resolved_output_path.relative_to((root_path / "outputs").resolve())
    except ValueError as exc:
        raise ValueError("Prompt 构建输出必须位于项目 outputs/ 目录内") from exc
    return resolved_output_path


def build_governed_prompt_bank(
    *,
    coco_source_path: str | Path,
    parti_source_path: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    """从两个冻结来源生成选择清单、三级 Prompt 文件和来源注册表."""

    output_path = Path(output_root)
    config_path = output_path / "configs"
    rows, source_statistics = build_prompt_selection_rows(
        coco_source_path,
        parti_source_path,
    )
    manifest_path = write_selection_manifest(
        rows,
        config_path / PROMPT_SELECTION_MANIFEST_PATH.name,
    )
    prompt_paths = write_prompt_files_from_selection(rows, config_path)
    registry = build_prompt_source_registry(
        rows=rows,
        source_statistics=source_statistics,
        coco_source_path=coco_source_path,
        parti_source_path=parti_source_path,
    )
    registry_path = config_path / PROMPT_SOURCE_REGISTRY_PATH.name
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(
        _stable_report(registry) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    output_files = {
        "selection_manifest": manifest_path,
        "source_registry": registry_path,
        **{f"prompt_{name}": path for name, path in prompt_paths.items()},
    }
    return {
        "operation": "build_governed_prompt_bank",
        "selection_manifest_record_count": len(rows),
        "source_statistics": source_statistics,
        "output_files": {
            name: {
                "path": path.as_posix(),
                "sha256": _path_sha256(path),
                "size": path.stat().st_size,
            }
            for name, path in sorted(output_files.items())
        },
    }


def rebuild_committed_prompt_bank(
    *,
    repository_root: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    """只依赖提交内清单重建全部 Prompt 配置字节."""

    root_path = Path(repository_root).resolve()
    audit_report = audit_committed_prompt_bank(root_path)
    manifest_source_path = root_path / PROMPT_SELECTION_MANIFEST_PATH
    registry_source_path = root_path / PROMPT_SOURCE_REGISTRY_PATH
    rows = read_selection_manifest(manifest_source_path)
    config_path = Path(output_root) / "configs"
    manifest_output_path = write_selection_manifest(
        rows,
        config_path / PROMPT_SELECTION_MANIFEST_PATH.name,
    )
    if manifest_output_path.read_bytes() != selection_manifest_bytes(rows):
        raise RuntimeError("重建的 Prompt 选择清单字节不一致")
    prompt_paths = write_prompt_files_from_selection(rows, config_path)
    registry_output_path = config_path / PROMPT_SOURCE_REGISTRY_PATH.name
    registry_output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(registry_source_path, registry_output_path)
    output_files = {
        "selection_manifest": manifest_output_path,
        "source_registry": registry_output_path,
        **{f"prompt_{name}": path for name, path in prompt_paths.items()},
    }
    return {
        "operation": "rebuild_committed_prompt_bank",
        "audit_report": audit_report,
        "output_files": {
            name: {
                "path": path.as_posix(),
                "sha256": _path_sha256(path),
                "size": path.stat().st_size,
            }
            for name, path in sorted(output_files.items())
        },
    }


def verify_committed_selection_sources(
    *,
    repository_root: str | Path,
    coco_source_path: str | Path,
    parti_source_path: str | Path,
) -> dict[str, Any]:
    """从冻结来源重新选择并要求与提交内清单逐字节一致."""

    root_path = Path(repository_root).resolve()
    manifest_rows = read_selection_manifest(
        root_path / PROMPT_SELECTION_MANIFEST_PATH
    )
    return verify_selection_against_sources(
        manifest_rows,
        coco_source_path=coco_source_path,
        parti_source_path=parti_source_path,
    )


def build_parser() -> argparse.ArgumentParser:
    """构造 Prompt 来源治理命令行参数."""

    parser = argparse.ArgumentParser(
        description="构建、重建或复验可逐字节追溯的 Prompt bank。"
    )
    parser.add_argument(
        "--operation",
        choices=("build", "rebuild", "audit", "source_verify"),
        default="audit",
        help="选择来源构建、提交内重建、轻量审计或完整来源复验。",
    )
    parser.add_argument("--repository-root", default=".", help="项目根目录。")
    parser.add_argument(
        "--coco-source",
        default=DEFAULT_COCO_SOURCE.as_posix(),
        help="冻结 COCO captions JSON 路径。",
    )
    parser.add_argument(
        "--parti-source",
        default=DEFAULT_PARTI_SOURCE.as_posix(),
        help="冻结 PartiPrompts TSV 路径。",
    )
    parser.add_argument(
        "--output-root",
        default=DEFAULT_OUTPUT_ROOT.as_posix(),
        help="构建产物目录, 正式调用必须位于 outputs/。",
    )
    return parser


def main() -> None:
    """执行所选 Prompt 来源治理操作并打印稳定报告."""

    args = build_parser().parse_args()
    if args.operation == "build":
        output_root = _require_persistent_output_root(
            args.repository_root,
            args.output_root,
        )
        report = build_governed_prompt_bank(
            coco_source_path=args.coco_source,
            parti_source_path=args.parti_source,
            output_root=output_root,
        )
    elif args.operation == "rebuild":
        output_root = _require_persistent_output_root(
            args.repository_root,
            args.output_root,
        )
        report = rebuild_committed_prompt_bank(
            repository_root=args.repository_root,
            output_root=output_root,
        )
    elif args.operation == "source_verify":
        report = verify_committed_selection_sources(
            repository_root=args.repository_root,
            coco_source_path=args.coco_source,
            parti_source_path=args.parti_source,
        )
    else:
        report = audit_committed_prompt_bank(args.repository_root)
    print(_stable_report(report))


if __name__ == "__main__":
    main()
