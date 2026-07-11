"""写出当前论文运行层级的 fixed-FPR 共同协议产物。"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.protocol.attacks import default_attack_configs
from experiments.protocol.pilot_paper_fixed_fpr import (
    build_attack_matrix_digest,
    build_fixed_fpr_protocol_digest,
    build_paper_fixed_fpr_config,
    build_pilot_paper_attack_matrix_rows,
    build_pilot_paper_common_protocol_summary,
    build_pilot_paper_manifest_config,
    build_pilot_paper_method_registry_rows,
    build_pilot_paper_prompt_split_summary,
    build_pilot_paper_result_import_schema,
    build_pilot_paper_result_import_template_rows,
    validate_pilot_paper_result_import_rows,
)
from experiments.protocol.prompts import build_prompt_records, read_prompt_file
from experiments.artifacts.artifact_manifest import build_artifact_manifest

DEFAULT_OUTPUT_ROOT = Path("outputs/pilot_paper_fixed_fpr_common_protocol")
DEFAULT_RESULT_RECORD_ROOT = Path("outputs/pilot_paper_fixed_fpr_results")


def stable_json_text(value: Any) -> str:
    """把 JSON 兼容对象转换为稳定文本。"""

    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def json_line(value: dict[str, Any]) -> str:
    """把单条记录转换为 JSONL 文本行。"""

    return json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n"


def resolve_code_version(root_path: Path) -> str:
    """读取 Git 短提交标识, 工作区有变更时附加 dirty 标记。"""

    try:
        commit_result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=root_path,
            check=True,
            capture_output=True,
            text=True,
        )
        status_result = subprocess.run(
            ["git", "status", "--short"],
            cwd=root_path,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return "git_version_unavailable"
    commit_id = commit_result.stdout.strip()
    if not commit_id:
        return "git_version_unavailable"
    return f"{commit_id}-dirty" if status_result.stdout.strip() else commit_id


def resolve_input_path(root_path: Path, path: str | Path) -> Path:
    """把输入路径解析为绝对路径。"""

    candidate = Path(path)
    return candidate.resolve() if candidate.is_absolute() else (root_path / candidate).resolve()


def ensure_output_dir_under_outputs(root_path: Path, output_dir: str | Path) -> Path:
    """确保持久化输出目录位于 outputs/ 下。"""

    resolved_output_dir = resolve_input_path(root_path, output_dir)
    outputs_root = (root_path / "outputs").resolve()
    try:
        resolved_output_dir.relative_to(outputs_root)
    except ValueError as exc:
        raise ValueError(f"pilot_paper fixed-FPR 共同协议输出目录必须位于 outputs/ 下: {resolved_output_dir}") from exc
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    return resolved_output_dir


def relative_or_absolute(path: Path, root_path: Path) -> str:
    """优先记录相对仓库根目录的路径。"""

    try:
        return path.resolve().relative_to(root_path.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def read_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    """读取 JSONL 记录, 文件缺失时返回空集合。"""

    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    """按固定字段顺序写出 CSV 表格。"""

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def serializable_attack_rows(rows: tuple[dict[str, Any], ...]) -> list[dict[str, Any]]:
    """把攻击矩阵行转换为 CSV 兼容形式。"""

    return [
        {
            **row,
            "attack_parameters": json.dumps(row["attack_parameters"], ensure_ascii=False, sort_keys=True),
        }
        for row in rows
    ]


def write_pilot_paper_fixed_fpr_common_protocol_outputs(
    *,
    root: str | Path = ".",
    output_dir: str | Path | None = None,
    candidate_records_path: str | Path | None = None,
    require_existing_evidence: bool = False,
) -> dict[str, Any]:
    """写出当前论文运行层级 fixed-FPR 共同协议的运行前治理产物。"""

    root_path = Path(root).resolve()
    config = build_paper_fixed_fpr_config(root_path)
    output_path = ensure_output_dir_under_outputs(
        root_path,
        output_dir or DEFAULT_OUTPUT_ROOT / config.paper_run_name,
    )
    prompt_path = resolve_input_path(root_path, config.prompt_file)
    candidate_path = resolve_input_path(
        root_path,
        candidate_records_path
        or DEFAULT_RESULT_RECORD_ROOT / config.paper_run_name / "pilot_paper_result_records.jsonl",
    )

    prompt_records = build_prompt_records(config.prompt_set, read_prompt_file(prompt_path))
    prompt_summary = build_pilot_paper_prompt_split_summary(prompt_records, config)
    attack_rows = build_pilot_paper_attack_matrix_rows(default_attack_configs(), config)
    attack_matrix_digest = build_attack_matrix_digest(attack_rows)
    fixed_fpr_protocol_digest = build_fixed_fpr_protocol_digest(config)
    method_rows = build_pilot_paper_method_registry_rows(
        prompt_split_digest=prompt_summary["prompt_split_digest"],
        attack_matrix_digest=attack_matrix_digest,
        fixed_fpr_protocol_digest=fixed_fpr_protocol_digest,
        config=config,
    )
    schema = build_pilot_paper_result_import_schema(
        prompt_split_digest=prompt_summary["prompt_split_digest"],
        attack_matrix_digest=attack_matrix_digest,
        fixed_fpr_protocol_digest=fixed_fpr_protocol_digest,
        config=config,
    )
    template_rows = build_pilot_paper_result_import_template_rows(method_rows, attack_rows, config)
    candidate_rows = read_jsonl_rows(candidate_path)
    validation_report = validate_pilot_paper_result_import_rows(
        candidate_rows,
        schema,
        evidence_root=root_path,
        require_existing_evidence=require_existing_evidence,
    )
    summary = build_pilot_paper_common_protocol_summary(
        prompt_summary=prompt_summary,
        attack_rows=attack_rows,
        method_rows=method_rows,
        template_rows=template_rows,
        import_validation_report=validation_report,
        config=config,
    )
    summary["generated_at"] = datetime.now(timezone.utc).isoformat()
    summary["candidate_records_path"] = relative_or_absolute(candidate_path, root_path)
    summary["candidate_record_count"] = len(candidate_rows)

    prompt_summary_path = output_path / "pilot_paper_prompt_split_summary.json"
    attack_matrix_path = output_path / "pilot_paper_attack_matrix.csv"
    method_registry_path = output_path / "pilot_paper_method_registry.csv"
    schema_path = output_path / "pilot_paper_result_import_schema.json"
    template_path = output_path / "pilot_paper_result_import_template.jsonl"
    validation_path = output_path / "pilot_paper_result_import_validation_report.json"
    summary_path = output_path / "pilot_paper_common_protocol_summary.json"
    manifest_path = output_path / "manifest.local.json"

    prompt_summary_path.write_text(stable_json_text(prompt_summary), encoding="utf-8")
    write_csv(
        attack_matrix_path,
        serializable_attack_rows(attack_rows),
        [
            "attack_id",
            "attack_family",
            "attack_name",
            "attack_strength",
            "resource_profile",
            "requires_gpu",
            "attack_parameters",
            "attack_config_digest",
            "result_scope",
            "supports_paper_claim",
        ],
    )
    write_csv(
        method_registry_path,
        list(method_rows),
        [
            "method_id",
            "method_name",
            "method_role",
            "prompt_set",
            "prompt_file",
            "prompt_protocol_name",
            "prompt_split_digest",
            "attack_matrix_digest",
            "fixed_fpr_protocol_digest",
            "target_fpr",
            "confidence_interval_method",
            "confidence_level",
            "result_protocol_name",
            "result_scope",
            "result_claim_scope",
            "governed_import_required",
            "supports_paper_claim",
            "paper_claim_scale",
        ],
    )
    schema_path.write_text(stable_json_text(schema), encoding="utf-8")
    template_path.write_text("".join(json_line(row) for row in template_rows), encoding="utf-8")
    validation_path.write_text(stable_json_text(validation_report), encoding="utf-8")
    summary_path.write_text(stable_json_text(summary), encoding="utf-8")

    output_paths = tuple(
        relative_or_absolute(path, root_path)
        for path in (
            prompt_summary_path,
            attack_matrix_path,
            method_registry_path,
            schema_path,
            template_path,
            validation_path,
            summary_path,
            manifest_path,
        )
    )
    input_paths = [relative_or_absolute(prompt_path, root_path)]
    if candidate_path.exists():
        input_paths.append(relative_or_absolute(candidate_path, root_path))
    manifest = build_artifact_manifest(
        artifact_id="pilot_paper_fixed_fpr_common_protocol_manifest",
        artifact_type="local_manifest",
        input_paths=tuple(input_paths),
        output_paths=output_paths,
        config=build_pilot_paper_manifest_config(
            prompt_summary=prompt_summary,
            attack_rows=attack_rows,
            method_rows=method_rows,
            template_rows=template_rows,
            schema=schema,
            validation_report=validation_report,
            summary=summary,
            config=config,
        ),
        code_version=resolve_code_version(root_path),
        rebuild_command="python scripts/write_pilot_paper_fixed_fpr_common_protocol_outputs.py",
        metadata=summary,
    ).to_dict()
    manifest_path.write_text(stable_json_text(manifest), encoding="utf-8")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器。"""

    parser = argparse.ArgumentParser(description="写出当前论文运行层级 fixed-FPR 共同协议产物。")
    parser.add_argument("--root", default=".", help="仓库根目录。")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="输出目录; 默认写入当前论文运行子目录, 且必须位于 outputs/ 下。",
    )
    parser.add_argument(
        "--candidate-records-path",
        default=None,
        help="待导入论文结果 JSONL 路径; 默认读取当前论文运行子目录。",
    )
    parser.add_argument("--require-existing-evidence", action="store_true", help="校验 evidence_paths 指向的文件存在。")
    return parser


def main() -> None:
    """命令行入口。"""

    args = build_parser().parse_args()
    manifest = write_pilot_paper_fixed_fpr_common_protocol_outputs(
        root=args.root,
        output_dir=args.output_dir,
        candidate_records_path=args.candidate_records_path,
        require_existing_evidence=args.require_existing_evidence,
    )
    print(stable_json_text(manifest), end="")


if __name__ == "__main__":
    main()
