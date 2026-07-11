"""写出主方法相对4个主表 baseline 的 Prompt-clustered 配对优势证据."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.artifacts.artifact_manifest import build_artifact_manifest
from experiments.protocol.attacks import default_attack_configs
from experiments.protocol.paper_run_config import build_paper_run_config
from experiments.protocol.pilot_paper_fixed_fpr import (
    build_paper_fixed_fpr_config,
    build_pilot_paper_attack_matrix_rows,
)
from experiments.protocol.splits import build_group_split_counts
from experiments.runtime.repository_environment import resolve_code_version
from main.core.digest import build_stable_digest
from paper_experiments.analysis.paired_superiority import (
    BOOTSTRAP_ANALYSIS_SCHEMA,
    BOOTSTRAP_BIT_GENERATOR,
    BOOTSTRAP_QUANTILE_METHOD,
    CLAIM_P_VALUE_METHOD,
    DEFAULT_BOOTSTRAP_RESAMPLE_COUNT,
    DEFAULT_CONFIDENCE_LEVEL,
    PRIMARY_BASELINE_IDS,
    SHARP_NULL_DIAGNOSTIC_METHOD,
    THRESHOLD_AUDIT_METHOD_IDS,
    build_paired_outcome_set_digest,
    build_paired_outcomes,
    build_paired_superiority_manifest_config,
    build_paired_superiority_protocol_digest,
    build_paired_superiority_rows,
    build_paired_superiority_summary,
    build_threshold_audit_binding_maps,
    canonical_attack_registry_rows,
    canonical_threshold_audit_rows,
)
from paper_experiments.analysis.fixed_fpr_threshold_audit import (
    build_fixed_fpr_threshold_manifest_config,
)


DEFAULT_OUTPUT_ROOT = Path("outputs/paired_superiority_analysis")
DEFAULT_IMAGE_ONLY_RUNTIME_ROOT = Path("outputs/image_only_dataset_runtime")
DEFAULT_METHOD_FAITHFUL_ROOT = Path("outputs/external_baseline_method_faithful")
DEFAULT_T2SMARK_ROOT = Path("outputs/t2smark_formal_reproduction")
DEFAULT_THRESHOLD_AUDIT_ROOT = Path("outputs/fixed_fpr_threshold_audit")


def _stable_json_text(value: Any) -> str:
    """把 JSON 兼容值写为确定性文本."""

    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def _resolve(root_path: Path, value: str | Path) -> Path:
    """解析相对仓库根目录或绝对路径."""

    path = Path(value)
    return path.resolve() if path.is_absolute() else (root_path / path).resolve()


def _relative_or_absolute(path: Path, root_path: Path) -> str:
    """优先记录仓库相对路径."""

    try:
        return path.resolve().relative_to(root_path).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _ensure_output_dir(root_path: Path, output_dir: str | Path) -> Path:
    """确保持久化结果只写入 outputs/ 边界."""

    resolved = _resolve(root_path, output_dir)
    outputs_root = (root_path / "outputs").resolve()
    try:
        resolved.relative_to(outputs_root)
    except ValueError as error:
        raise ValueError("配对优势输出目录必须位于 outputs/ 下") from error
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _read_json_object(path: Path) -> dict[str, Any]:
    """读取必须存在的 JSON object."""

    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise TypeError(f"期望 JSON object: {path.as_posix()}")
    return dict(payload)


def _read_json_array(path: Path) -> tuple[dict[str, Any], ...]:
    """读取必须非空的 JSON object 数组."""

    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if (
        not isinstance(payload, list)
        or not payload
        or any(not isinstance(row, dict) for row in payload)
    ):
        raise ValueError(f"期望非空 JSON object 数组: {path.as_posix()}")
    return tuple(dict(row) for row in payload)


def _read_jsonl(path: Path) -> tuple[dict[str, Any], ...]:
    """读取必须非空的 JSONL object 记录."""

    if not path.is_file():
        raise FileNotFoundError(path)
    rows = tuple(
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    )
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"期望非空 JSONL object 记录: {path.as_posix()}")
    return tuple(dict(row) for row in rows)


def _read_csv_rows(path: Path) -> tuple[dict[str, Any], ...]:
    """读取必须非空的 CSV object 记录."""

    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = tuple(dict(row) for row in csv.DictReader(handle))
    if not rows:
        raise ValueError(f"期望非空 CSV object 记录: {path.as_posix()}")
    return rows


def _file_sha256(path: Path) -> str:
    """计算 observation 原始文件字节的 SHA-256."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv(
    path: Path,
    rows: Iterable[dict[str, Any]],
    field_names: tuple[str, ...],
) -> None:
    """按固定字段顺序写出 CSV."""

    materialized = tuple(rows)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=field_names)
        writer.writeheader()
        writer.writerows(materialized)


def _threshold_audit_ready(
    *,
    threshold_report: Mapping[str, Any],
    threshold_manifest: Mapping[str, Any],
    threshold_rows: Iterable[Mapping[str, Any]],
    method_threshold_digest_map: Mapping[str, str],
    method_observation_source_sha256_map: Mapping[str, str],
    paper_run_name: str,
    target_fpr: float,
) -> bool:
    """核验 fixed-FPR 审计的五行, 报告, manifest 与原始文件绑定."""

    canonical_rows = canonical_threshold_audit_rows(threshold_rows)
    report_expected_ids = {
        str(value) for value in threshold_report.get("expected_method_ids", ())
    }
    report_audited_ids = {
        str(value) for value in threshold_report.get("audited_method_ids", ())
    }
    metadata = threshold_manifest.get("metadata", {})
    return bool(
        threshold_report.get("paper_claim_scale") == paper_run_name
        and math.isclose(
            float(threshold_report.get("target_fpr", -1.0)),
            target_fpr,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        and report_expected_ids == set(THRESHOLD_AUDIT_METHOD_IDS)
        and report_audited_ids == set(THRESHOLD_AUDIT_METHOD_IDS)
        and int(threshold_report.get("audited_method_count", 0))
        == len(THRESHOLD_AUDIT_METHOD_IDS)
        and threshold_report.get("method_identity_ready") is True
        and threshold_report.get("all_method_thresholds_ready") is True
        and threshold_report.get("threshold_observation_binding_ready") is True
        and threshold_report.get("fixed_fpr_threshold_audit_ready") is True
        and threshold_report.get("supports_paper_claim") is True
        and all(row["fixed_fpr_threshold_ready"] is True for row in canonical_rows)
        and all(row["detection_decision_ready"] is True for row in canonical_rows)
        and threshold_report.get("method_threshold_digest_map")
        == dict(method_threshold_digest_map)
        and threshold_report.get("method_observation_source_sha256_map")
        == dict(method_observation_source_sha256_map)
        and threshold_report.get("threshold_audit_rows_digest")
        == build_stable_digest(list(canonical_rows))
        and threshold_manifest.get("artifact_id")
        == "fixed_fpr_threshold_audit_manifest"
        and threshold_manifest.get("config_digest")
        == build_stable_digest(
            build_fixed_fpr_threshold_manifest_config(threshold_report)
        )
        and isinstance(metadata, Mapping)
        and metadata.get("method_threshold_digest_map")
        == dict(method_threshold_digest_map)
        and metadata.get("method_observation_source_sha256_map")
        == dict(method_observation_source_sha256_map)
        and metadata.get("threshold_audit_rows_digest")
        == threshold_report.get("threshold_audit_rows_digest")
        and metadata.get("threshold_observation_binding_ready") is True
    )


def write_paired_superiority_outputs(
    *,
    root: str | Path = ".",
    output_dir: str | Path | None = None,
    proposed_records_path: str | Path | None = None,
    method_faithful_root: str | Path | None = None,
    t2smark_observations_path: str | Path | None = None,
    threshold_audit_rows_path: str | Path | None = None,
    threshold_audit_report_path: str | Path | None = None,
    threshold_audit_manifest_path: str | Path | None = None,
    bootstrap_resample_count: int = DEFAULT_BOOTSTRAP_RESAMPLE_COUNT,
    require_pass: bool = False,
) -> dict[str, Any]:
    """重建配对观测, 聚类 CI, 均值检验和 sharp-null 诊断结果."""

    root_path = Path(root).resolve()
    if int(bootstrap_resample_count) != DEFAULT_BOOTSTRAP_RESAMPLE_COUNT:
        raise ValueError("正式配对优势协议固定使用100000次 bootstrap")
    paper_run = build_paper_run_config(root_path)
    run_name = paper_run.run_name
    output_path = _ensure_output_dir(
        root_path,
        output_dir or DEFAULT_OUTPUT_ROOT / run_name,
    )
    proposed_path = _resolve(
        root_path,
        proposed_records_path
        or DEFAULT_IMAGE_ONLY_RUNTIME_ROOT
        / run_name
        / "image_only_detection_records.jsonl",
    )
    collection_root = _resolve(
        root_path,
        method_faithful_root or DEFAULT_METHOD_FAITHFUL_ROOT / run_name,
    )
    t2smark_path = _resolve(
        root_path,
        t2smark_observations_path
        or DEFAULT_T2SMARK_ROOT
        / run_name
        / "t2smark_adapter"
        / "baseline_observations.json",
    )
    threshold_rows_path = _resolve(
        root_path,
        threshold_audit_rows_path
        or DEFAULT_THRESHOLD_AUDIT_ROOT / run_name / "threshold_audit_rows.csv",
    )
    threshold_report_path = _resolve(
        root_path,
        threshold_audit_report_path
        or DEFAULT_THRESHOLD_AUDIT_ROOT / run_name / "threshold_audit_report.json",
    )
    threshold_manifest_path = _resolve(
        root_path,
        threshold_audit_manifest_path
        or DEFAULT_THRESHOLD_AUDIT_ROOT / run_name / "manifest.local.json",
    )

    proposed_rows = _read_jsonl(proposed_path)
    baseline_rows_by_id: dict[str, tuple[dict[str, Any], ...]] = {}
    method_observation_source_paths: dict[str, Path] = {"slm_wm": proposed_path}
    for baseline_id in PRIMARY_BASELINE_IDS[:-1]:
        path = (
            collection_root
            / "split_observations"
            / f"{baseline_id}_baseline_observations.json"
        )
        baseline_rows_by_id[baseline_id] = _read_json_array(path)
        method_observation_source_paths[baseline_id] = path
    baseline_rows_by_id["t2smark"] = _read_json_array(t2smark_path)
    method_observation_source_paths["t2smark"] = t2smark_path
    method_observation_source_sha256_map = {
        method_id: _file_sha256(path)
        for method_id, path in sorted(method_observation_source_paths.items())
    }
    method_observation_source_path_map = {
        method_id: _relative_or_absolute(path, root_path)
        for method_id, path in sorted(method_observation_source_paths.items())
    }

    threshold_rows = canonical_threshold_audit_rows(
        _read_csv_rows(threshold_rows_path)
    )
    threshold_report = _read_json_object(threshold_report_path)
    threshold_manifest = _read_json_object(threshold_manifest_path)
    method_threshold_digest_map, audited_observation_sha256_map = (
        build_threshold_audit_binding_maps(threshold_rows)
    )
    if audited_observation_sha256_map != method_observation_source_sha256_map:
        raise ValueError("配对优势输入 observation 字节摘要与 fixed-FPR 审计不一致")
    if not _threshold_audit_ready(
        threshold_report=threshold_report,
        threshold_manifest=threshold_manifest,
        threshold_rows=threshold_rows,
        method_threshold_digest_map=method_threshold_digest_map,
        method_observation_source_sha256_map=method_observation_source_sha256_map,
        paper_run_name=run_name,
        target_fpr=paper_run.target_fpr,
    ):
        raise ValueError("配对优势统计只接受已通过且绑定原始 observation 的统一 fixed-FPR 审计")
    protocol_digest = build_paired_superiority_protocol_digest(
        threshold_report,
        threshold_rows,
        str(threshold_manifest.get("config_digest", "")),
    )

    fixed_fpr_config = build_paper_fixed_fpr_config(root_path)
    attack_registry = canonical_attack_registry_rows(
        build_pilot_paper_attack_matrix_rows(
            default_attack_configs(),
            fixed_fpr_config,
        )
    )
    paired_outcomes = tuple(
        outcome
        for baseline_id in PRIMARY_BASELINE_IDS
        for outcome in build_paired_outcomes(
            proposed_rows,
            baseline_rows_by_id[baseline_id],
            baseline_id=baseline_id,
            proposed_method_threshold_digest=method_threshold_digest_map["slm_wm"],
            baseline_method_threshold_digest=method_threshold_digest_map[baseline_id],
            attack_registry_rows=attack_registry,
        )
    )
    statistical_rows = build_paired_superiority_rows(
        paired_outcomes,
        protocol_digest=protocol_digest,
        bootstrap_resample_count=int(bootstrap_resample_count),
    )
    expected_test_count = int(build_group_split_counts(paper_run.prompt_count)["test"])
    expected_attack_count = len(attack_registry)
    expected_attack_counts = {
        int(row["paired_attack_count"]) for row in statistical_rows
    }
    expected_prompt_counts = {
        int(row["paired_prompt_count"]) for row in statistical_rows
    }
    scale_ready = (
        expected_prompt_counts == {expected_test_count}
        and expected_attack_counts == {expected_attack_count}
    )
    statistical_summary = build_paired_superiority_summary(
        statistical_rows,
        paired_outcomes=paired_outcomes,
    )
    prompt_identity_ready = (
        int(statistical_summary["paired_test_prompt_count"]) == expected_test_count
    )
    overall_ready = bool(
        scale_ready
        and prompt_identity_ready
        and statistical_summary["overall_paired_superiority_ready"]
    )

    outcomes_path = output_path / "paired_outcomes.jsonl"
    table_path = output_path / "paired_superiority_table.csv"
    summary_path = output_path / "paired_superiority_summary.json"
    manifest_path = output_path / "manifest.local.json"
    outcomes_path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in paired_outcomes
        ),
        encoding="utf-8",
    )
    _write_csv(
        table_path,
        statistical_rows,
        (
            "baseline_id",
            "paired_prompt_count",
            "paired_attack_count",
            "paired_observation_count",
            "mean_paired_true_positive_rate_difference",
            "mean_paired_difference_ci_low",
            "mean_paired_difference_ci_high",
            "positive_prompt_cluster_count",
            "negative_prompt_cluster_count",
            "tied_prompt_cluster_count",
            "one_sided_bounded_hoeffding_mean_p_value",
            "one_sided_exact_prompt_cluster_sign_flip_p_value",
            "exact_prompt_cluster_sign_flip_p_value_is_diagnostic",
            "sharp_null_diagnostic_method",
            "claim_p_value_method",
            "holm_adjusted_p_value",
            "confidence_level",
            "bootstrap_resample_count",
            "bootstrap_seed_digest_random",
            "bootstrap_analysis_schema",
            "bootstrap_bit_generator",
            "bootstrap_quantile_method",
            "proposed_method_threshold_digest",
            "baseline_method_threshold_digest",
            "paired_test_prompt_id_digest",
            "paired_attack_registry_digest",
            "paired_outcome_set_digest",
            "protocol_digest",
            "paired_superiority_ready",
            "supports_paper_claim",
        ),
    )
    paired_outcome_set_digest = build_paired_outcome_set_digest(paired_outcomes)
    summary = {
        "construction_unit_name": "paired_superiority_analysis",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "paper_claim_scale": run_name,
        "target_fpr": paper_run.target_fpr,
        "expected_test_count": expected_test_count,
        "expected_attack_count": expected_attack_count,
        "paired_prompt_counts": sorted(expected_prompt_counts),
        "paired_attack_counts": sorted(expected_attack_counts),
        "paired_outcome_count": len(paired_outcomes),
        "paired_outcome_set_digest": paired_outcome_set_digest,
        "paired_superiority_protocol_digest": protocol_digest,
        "claim_p_value_method": CLAIM_P_VALUE_METHOD,
        "sharp_null_diagnostic_method": SHARP_NULL_DIAGNOSTIC_METHOD,
        "bootstrap_analysis_schema": BOOTSTRAP_ANALYSIS_SCHEMA,
        "bootstrap_bit_generator": BOOTSTRAP_BIT_GENERATOR,
        "bootstrap_quantile_method": BOOTSTRAP_QUANTILE_METHOD,
        "bootstrap_resample_count": int(bootstrap_resample_count),
        "confidence_level": DEFAULT_CONFIDENCE_LEVEL,
        "method_threshold_digest_map": method_threshold_digest_map,
        "method_observation_source_sha256_map": method_observation_source_sha256_map,
        "method_observation_source_path_map": method_observation_source_path_map,
        "threshold_audit_rows_digest": threshold_report[
            "threshold_audit_rows_digest"
        ],
        "paired_attack_registry_digest": build_stable_digest(
            list(attack_registry)
        ),
        "paired_superiority_scale_ready": scale_ready,
        **statistical_summary,
        "overall_paired_superiority_ready": overall_ready,
        "supports_paper_claim": overall_ready,
    }
    summary_path.write_text(_stable_json_text(summary), encoding="utf-8")
    input_paths = (
        *method_observation_source_paths.values(),
        threshold_rows_path,
        threshold_report_path,
        threshold_manifest_path,
    )
    manifest = build_artifact_manifest(
        artifact_id="paired_superiority_analysis_manifest",
        artifact_type="local_manifest",
        input_paths=tuple(
            _relative_or_absolute(path, root_path) for path in input_paths
        ),
        output_paths=tuple(
            _relative_or_absolute(path, root_path)
            for path in (outcomes_path, table_path, summary_path, manifest_path)
        ),
        config=build_paired_superiority_manifest_config(summary),
        code_version=resolve_code_version(root_path),
        rebuild_command="python scripts/write_paired_superiority_outputs.py --require-pass",
        metadata=summary,
    ).to_dict()
    manifest_path.write_text(_stable_json_text(manifest), encoding="utf-8")
    if require_pass and not overall_ready:
        raise RuntimeError("主方法总体配对优势未通过, 不得闭合 superiority claim")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    """构造可脱离 Notebook 执行的命令行参数."""

    parser = argparse.ArgumentParser(description="重建 Prompt-clustered 配对优势统计.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--proposed-records-path", default=None)
    parser.add_argument("--method-faithful-root", default=None)
    parser.add_argument("--t2smark-observations-path", default=None)
    parser.add_argument("--threshold-audit-rows-path", default=None)
    parser.add_argument("--threshold-audit-report-path", default=None)
    parser.add_argument("--threshold-audit-manifest-path", default=None)
    parser.add_argument(
        "--bootstrap-resample-count",
        type=int,
        default=DEFAULT_BOOTSTRAP_RESAMPLE_COUNT,
    )
    parser.add_argument("--require-pass", action="store_true")
    return parser


def main() -> None:
    """执行命令行写出流程."""

    args = build_parser().parse_args()
    manifest = write_paired_superiority_outputs(
        root=args.root,
        output_dir=args.output_dir,
        proposed_records_path=args.proposed_records_path,
        method_faithful_root=args.method_faithful_root,
        t2smark_observations_path=args.t2smark_observations_path,
        threshold_audit_rows_path=args.threshold_audit_rows_path,
        threshold_audit_report_path=args.threshold_audit_report_path,
        threshold_audit_manifest_path=args.threshold_audit_manifest_path,
        bootstrap_resample_count=args.bootstrap_resample_count,
        require_pass=args.require_pass,
    )
    print(_stable_json_text(manifest), end="")


if __name__ == "__main__":
    main()
