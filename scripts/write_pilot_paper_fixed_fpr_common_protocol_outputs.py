"""写出当前论文运行层级的 fixed-FPR 共同协议产物。"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
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
    build_pilot_paper_result_record_set_digest,
    build_pilot_paper_result_import_schema,
    build_pilot_paper_result_import_template_rows,
    validate_pilot_paper_result_import_rows,
)
from experiments.protocol.prompts import build_prompt_records, read_prompt_file
from experiments.runtime.repository_environment import resolve_code_version
from experiments.artifacts.artifact_manifest import build_artifact_manifest
from paper_experiments.runners.closure_package_selection import (
    load_validated_closure_input_lock,
)

DEFAULT_OUTPUT_ROOT = Path("outputs/pilot_paper_fixed_fpr_common_protocol")
DEFAULT_RESULT_RECORD_ROOT = Path("outputs/pilot_paper_fixed_fpr_results")
DEFAULT_PAIRED_SUPERIORITY_ROOT = Path("outputs/paired_superiority_analysis")


def stable_json_text(value: Any) -> str:
    """把 JSON 兼容对象转换为稳定文本。"""

    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def json_line(value: dict[str, Any]) -> str:
    """把单条记录转换为 JSONL 文本行。"""

    return json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n"


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


def read_json_object(path: Path) -> dict[str, Any]:
    """读取必须存在的 JSON 对象."""

    if not path.is_file():
        raise FileNotFoundError(f"共同协议缺少配对统计输入: {path.as_posix()}")
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise TypeError(f"共同协议配对统计输入必须是 JSON 对象: {path.as_posix()}")
    return dict(payload)


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
    paired_superiority_summary_path: str | Path | None = None,
    paired_superiority_manifest_path: str | Path | None = None,
    require_existing_evidence: bool = False,
) -> dict[str, Any]:
    """写出当前论文运行层级 fixed-FPR 共同协议的运行前治理产物。"""

    root_path = Path(root).resolve()
    config = build_paper_fixed_fpr_config(root_path)
    closure_input_provenance = load_validated_closure_input_lock(
        root_path,
        paper_run_name=config.paper_run_name,
        target_fpr=config.target_fpr,
    )
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
    paired_summary_path = resolve_input_path(
        root_path,
        paired_superiority_summary_path
        or DEFAULT_PAIRED_SUPERIORITY_ROOT
        / config.paper_run_name
        / "paired_superiority_summary.json",
    )
    paired_manifest_path = resolve_input_path(
        root_path,
        paired_superiority_manifest_path
        or DEFAULT_PAIRED_SUPERIORITY_ROOT / config.paper_run_name / "manifest.local.json",
    )
    paired_summary = read_json_object(paired_summary_path)
    paired_manifest = read_json_object(paired_manifest_path)

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
    result_record_set_digest = build_pilot_paper_result_record_set_digest(candidate_rows)
    threshold_values_by_method: dict[str, set[str]] = {}
    for row in candidate_rows:
        threshold_values_by_method.setdefault(str(row.get("method_id", "")), set()).add(
            str(row.get("method_threshold_digest", ""))
        )
    method_threshold_digest_map = {
        method_id: next(iter(values))
        for method_id, values in sorted(threshold_values_by_method.items())
        if len(values) == 1 and len(next(iter(values))) == 64
    }
    if candidate_rows and (
        set(method_threshold_digest_map) != set(schema["method_ids"])
        or any(
        any(character not in "0123456789abcdefABCDEF" for character in digest)
        for digest in method_threshold_digest_map.values()
        )
    ):
        raise ValueError("共同协议要求5个方法各自绑定唯一 SHA-256 阈值摘要")
    schema.update(
        {
            "result_record_set_digest": result_record_set_digest,
            "calibration_prompt_id_digest": prompt_summary[
                "calibration_prompt_id_digest"
            ],
            "test_prompt_id_digest": prompt_summary["test_prompt_id_digest"],
            "method_threshold_digest_map": method_threshold_digest_map,
            "closure_input_lock_digest": closure_input_provenance[
                "closure_input_lock_digest"
            ],
            "common_code_version": closure_input_provenance["common_code_version"],
            "paired_superiority_ready": paired_summary.get(
                "overall_paired_superiority_ready", False
            ),
            "overall_paired_superiority_ready": paired_summary.get(
                "overall_paired_superiority_ready", False
            ),
            "paired_superiority_protocol_digest": paired_summary.get(
                "paired_superiority_protocol_digest", ""
            ),
            "paired_superiority_rows_digest": paired_summary.get(
                "paired_superiority_rows_digest", ""
            ),
            "paired_outcome_set_digest": paired_summary.get(
                "paired_outcome_set_digest", ""
            ),
            "paired_test_prompt_count": paired_summary.get(
                "paired_test_prompt_count", 0
            ),
            "paired_test_prompt_id_digest": paired_summary.get(
                "paired_test_prompt_id_digest", ""
            ),
            "paired_attack_registry_digest": paired_summary.get(
                "paired_attack_registry_digest", ""
            ),
            "method_observation_source_sha256_map": paired_summary.get(
                "method_observation_source_sha256_map", {}
            ),
            "threshold_audit_rows_digest": paired_summary.get(
                "threshold_audit_rows_digest", ""
            ),
            "claim_p_value_method": paired_summary.get(
                "claim_p_value_method", ""
            ),
            "sharp_null_diagnostic_method": paired_summary.get(
                "sharp_null_diagnostic_method", ""
            ),
            "bootstrap_analysis_schema": paired_summary.get(
                "bootstrap_analysis_schema", ""
            ),
            "bootstrap_bit_generator": paired_summary.get(
                "bootstrap_bit_generator", ""
            ),
            "bootstrap_quantile_method": paired_summary.get(
                "bootstrap_quantile_method", ""
            ),
            "bootstrap_resample_count": paired_summary.get(
                "bootstrap_resample_count", 0
            ),
            "confidence_level": paired_summary.get("confidence_level", 0.0),
        }
    )
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
        paired_superiority_summary=paired_summary,
        config=config,
    )
    summary["generated_at"] = datetime.now(timezone.utc).isoformat()
    summary["candidate_records_path"] = relative_or_absolute(candidate_path, root_path)
    summary["candidate_record_count"] = len(candidate_rows)
    summary["result_record_set_digest"] = result_record_set_digest
    summary["method_threshold_digest_map"] = method_threshold_digest_map
    summary["closure_input_lock_digest"] = closure_input_provenance[
        "closure_input_lock_digest"
    ]
    summary["common_code_version"] = closure_input_provenance["common_code_version"]
    paired_metadata = paired_manifest.get("metadata", {})
    if (
        paired_manifest.get("artifact_id") != "paired_superiority_analysis_manifest"
        or paired_manifest.get("code_version") != summary["common_code_version"]
        or not isinstance(paired_metadata, dict)
        or any(
            paired_metadata.get(field_name) != paired_summary.get(field_name)
            for field_name in (
                "paper_claim_scale",
                "target_fpr",
                "paired_superiority_exact_set_ready",
                "overall_paired_superiority_ready",
                "paired_superiority_rows_digest",
                "paired_outcome_set_digest",
                "paired_test_prompt_count",
                "paired_test_prompt_id_digest",
                "paired_attack_registry_digest",
                "method_observation_source_sha256_map",
                "method_observation_source_path_map",
                "method_threshold_digest_map",
                "threshold_audit_rows_digest",
                "claim_p_value_method",
                "sharp_null_diagnostic_method",
                "bootstrap_analysis_schema",
                "bootstrap_bit_generator",
                "bootstrap_quantile_method",
                "bootstrap_resample_count",
                "confidence_level",
            )
        )
    ):
        raise ValueError("共同协议要求配对优势 summary, manifest 与闭合代码版本完全一致")

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
    input_paths.extend(
        (
            relative_or_absolute(paired_summary_path, root_path),
            relative_or_absolute(paired_manifest_path, root_path),
        )
    )
    input_paths.extend(
        relative_or_absolute(closure_input_provenance[field_name], root_path)
        for field_name in (
            "closure_input_lock_path",
            "closure_input_lock_manifest_path",
        )
    )
    manifest_config = build_pilot_paper_manifest_config(
        prompt_summary=prompt_summary,
        attack_rows=attack_rows,
        method_rows=method_rows,
        template_rows=template_rows,
        schema=schema,
        validation_report=validation_report,
        summary=summary,
        config=config,
    )
    manifest_config.update(
        {
            "result_record_set_digest": result_record_set_digest,
            "method_threshold_digest_map": method_threshold_digest_map,
            "closure_input_lock_digest": summary["closure_input_lock_digest"],
            "common_code_version": summary["common_code_version"],
            "paired_test_prompt_count": summary["paired_test_prompt_count"],
            "paired_test_prompt_id_digest": summary[
                "paired_test_prompt_id_digest"
            ],
            "paired_attack_registry_digest": summary[
                "paired_attack_registry_digest"
            ],
            "method_observation_source_sha256_map": summary[
                "method_observation_source_sha256_map"
            ],
            "threshold_audit_rows_digest": summary[
                "threshold_audit_rows_digest"
            ],
            "claim_p_value_method": summary["claim_p_value_method"],
            "sharp_null_diagnostic_method": summary[
                "sharp_null_diagnostic_method"
            ],
            "bootstrap_analysis_schema": summary["bootstrap_analysis_schema"],
            "bootstrap_bit_generator": summary["bootstrap_bit_generator"],
            "bootstrap_quantile_method": summary["bootstrap_quantile_method"],
            "bootstrap_resample_count": summary["bootstrap_resample_count"],
        }
    )
    manifest = build_artifact_manifest(
        artifact_id="pilot_paper_fixed_fpr_common_protocol_manifest",
        artifact_type="local_manifest",
        input_paths=tuple(input_paths),
        output_paths=output_paths,
        config=manifest_config,
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
    parser.add_argument("--paired-superiority-summary-path", default=None)
    parser.add_argument("--paired-superiority-manifest-path", default=None)
    parser.add_argument("--require-existing-evidence", action="store_true", help="校验 evidence_paths 指向的文件存在。")
    return parser


def main() -> None:
    """命令行入口。"""

    args = build_parser().parse_args()
    manifest = write_pilot_paper_fixed_fpr_common_protocol_outputs(
        root=args.root,
        output_dir=args.output_dir,
        candidate_records_path=args.candidate_records_path,
        paired_superiority_summary_path=args.paired_superiority_summary_path,
        paired_superiority_manifest_path=args.paired_superiority_manifest_path,
        require_existing_evidence=args.require_existing_evidence,
    )
    print(stable_json_text(manifest), end="")


if __name__ == "__main__":
    main()
