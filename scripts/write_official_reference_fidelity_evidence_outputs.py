"""写出三个官方参考方法忠实度的独立补充证据产物."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.artifacts.artifact_manifest import build_artifact_manifest
from experiments.protocol.paper_run_config import build_paper_run_config
from experiments.runtime.repository_environment import resolve_code_version
from main.core.digest import build_stable_digest
from paper_experiments.baselines.official_reference_fidelity_evidence import (
    OFFICIAL_REFERENCE_BASELINE_IDS,
    OfficialReferenceFidelityEvidenceError,
    audit_exact_official_reference_fidelity_evidence,
    normalize_clean_code_version,
)


DEFAULT_OUTPUT_ROOT = Path("outputs/official_reference_fidelity_evidence")


def stable_json_text(value: Any) -> str:
    """以稳定字段顺序写出 JSON."""

    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def json_lines(rows: list[dict[str, Any]]) -> str:
    """把证据记录写成稳定 JSONL."""

    return "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    )


def _resolve_path(root_path: Path, value: str | Path) -> Path:
    """把相对路径解析到仓库根目录."""

    path = Path(value)
    return path.resolve() if path.is_absolute() else (root_path / path).resolve()


def _output_dir_under_outputs(root_path: Path, value: str | Path) -> Path:
    """确保持久化产物只写入 outputs/."""

    output_dir = _resolve_path(root_path, value)
    try:
        output_dir.relative_to((root_path / "outputs").resolve())
    except ValueError as error:
        raise ValueError(
            f"官方参考方法忠实度证据输出目录必须位于 outputs/ 下: {output_dir.as_posix()}"
        ) from error
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _relative_path(path: Path, root_path: Path) -> str:
    """记录仓库相对 POSIX 路径."""

    return path.resolve().relative_to(root_path.resolve()).as_posix()


def write_official_reference_fidelity_evidence_outputs(
    *,
    root: str | Path = ".",
    output_dir: str | Path | None = None,
    tree_ring_output_dir: str | Path | None = None,
    gaussian_shading_output_dir: str | Path | None = None,
    shallow_diffuse_output_dir: str | Path | None = None,
    repository_code_version: str | None = None,
    require_pass: bool = False,
) -> dict[str, Any]:
    """核验三个已物化 official-reference family 并写出独立证据."""

    root_path = Path(root).resolve()
    paper_run = build_paper_run_config(root_path)
    resolved_output_dir = _output_dir_under_outputs(
        root_path,
        output_dir or DEFAULT_OUTPUT_ROOT / paper_run.run_name,
    )
    explicit_source_values = {
        "tree_ring": tree_ring_output_dir,
        "gaussian_shading": gaussian_shading_output_dir,
        "shallow_diffuse": shallow_diffuse_output_dir,
    }
    provided_values = {
        baseline_id: value
        for baseline_id, value in explicit_source_values.items()
        if value is not None
    }
    if provided_values and set(provided_values) != set(OFFICIAL_REFERENCE_BASELINE_IDS):
        raise ValueError("显式 official-reference 输出目录必须一次提供精确三个 baseline")

    records, summary, input_paths = audit_exact_official_reference_fidelity_evidence(
        root=root_path,
        paper_run_name=paper_run.run_name,
        target_fpr=paper_run.target_fpr,
        source_dirs=provided_values or None,
    )
    current_code_version = normalize_clean_code_version(
        repository_code_version
        if repository_code_version is not None
        else resolve_code_version(root_path)
    )
    if current_code_version != summary["common_code_version"]:
        raise OfficialReferenceFidelityEvidenceError(
            "CPU 审计仓库代码版本必须与三个 official-reference 输入的共同 clean 提交一致"
        )
    summary.update(
        {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "paper_claim_scale": paper_run.run_name,
            "target_fpr": paper_run.target_fpr,
        }
    )
    if require_pass and summary["official_reference_fidelity_evidence_ready"] is not True:
        raise OfficialReferenceFidelityEvidenceError(
            "官方参考方法忠实度证据未通过, 已按 --require-pass 停止"
        )

    records_path = resolved_output_dir / "official_reference_fidelity_evidence_records.jsonl"
    summary_path = resolved_output_dir / "official_reference_fidelity_evidence_summary.json"
    manifest_path = resolved_output_dir / "manifest.local.json"
    records_path.write_text(json_lines(records), encoding="utf-8")
    summary_path.write_text(stable_json_text(summary), encoding="utf-8")
    output_paths = (
        _relative_path(records_path, root_path),
        _relative_path(summary_path, root_path),
        _relative_path(manifest_path, root_path),
    )
    manifest = build_artifact_manifest(
        artifact_id="official_reference_fidelity_evidence_manifest",
        artifact_type="local_manifest",
        input_paths=tuple(_relative_path(path, root_path) for path in input_paths),
        output_paths=output_paths,
        config={
            "paper_run_name": paper_run.run_name,
            "target_fpr": paper_run.target_fpr,
            "expected_official_reference_baseline_ids": list(
                OFFICIAL_REFERENCE_BASELINE_IDS
            ),
            "common_code_version": summary["common_code_version"],
            "official_reference_fidelity_evidence_digest": build_stable_digest(records),
            "summary_digest": build_stable_digest(summary),
        },
        code_version=current_code_version,
        rebuild_command=(
            "python scripts/write_official_reference_fidelity_evidence_outputs.py "
            "--require-pass"
        ),
        metadata=summary,
    ).to_dict()
    manifest_path.write_text(stable_json_text(manifest), encoding="utf-8")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    """构造命令行参数."""

    parser = argparse.ArgumentParser(
        description="核验三个官方参考环境复现并写出独立方法忠实度证据."
    )
    parser.add_argument("--root", default=".", help="仓库根目录.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="输出目录; 默认写入当前论文运行层级的 outputs 子目录.",
    )
    parser.add_argument("--tree-ring-output-dir", default=None)
    parser.add_argument("--gaussian-shading-output-dir", default=None)
    parser.add_argument("--shallow-diffuse-output-dir", default=None)
    parser.add_argument(
        "--require-pass",
        action="store_true",
        help="证据未闭合时返回非零状态.",
    )
    return parser


def main() -> None:
    """命令行入口."""

    args = build_parser().parse_args()
    manifest = write_official_reference_fidelity_evidence_outputs(
        root=args.root,
        output_dir=args.output_dir,
        tree_ring_output_dir=args.tree_ring_output_dir,
        gaussian_shading_output_dir=args.gaussian_shading_output_dir,
        shallow_diffuse_output_dir=args.shallow_diffuse_output_dir,
        require_pass=args.require_pass,
    )
    print(stable_json_text(manifest), end="")


if __name__ == "__main__":
    main()
