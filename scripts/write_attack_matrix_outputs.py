"""写出攻击矩阵、score retention 与 rescue-by-attack 产物。"""

from __future__ import annotations

import argparse
from collections import Counter
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

from experiments.protocol.attacks import (
    AttackConfig,
    AttackEvaluationBoundary,
    build_attack_detection_records,
    default_attack_configs,
    family_metrics,
    rescue_by_attack_rows,
    score_retention_rows,
    strength_curve,
)
from experiments.protocol.pilot_paper_fixed_fpr import PILOT_PAPER_FIXED_FPR
from main.analysis.artifact_manifest import build_artifact_manifest
from main.core.digest import build_stable_digest

CONSTRUCTION_UNIT_NAME = "attack_matrix"
DEFAULT_OUTPUT_DIR = Path("outputs/attack_matrix")
DEFAULT_RESCUE_RECORDS_PATH = Path("outputs/geometric_rescue/aligned_detection_records.jsonl")
DEFAULT_RESCUE_MANIFEST_PATH = Path("outputs/geometric_rescue/manifest.local.json")
DEFAULT_CALIBRATION_THRESHOLDS_PATH = Path("outputs/threshold_calibration/calibration_thresholds.json")
DEFAULT_THRESHOLD_REPORT_PATH = Path("outputs/threshold_calibration/threshold_degeneracy_report.json")
DEFAULT_CALIBRATION_MANIFEST_PATH = Path("outputs/threshold_calibration/manifest.local.json")
DEFAULT_REAL_ATTACK_RECORDS_PATH = Path("outputs/real_attack_evaluation/formal_attack_detection_records.jsonl")
DEFAULT_CONVENTIONAL_GEOMETRIC_ATTACK_RECORDS_PATH = Path(
    "outputs/conventional_geometric_attack_evaluation/formal_attack_detection_records.jsonl"
)
DEFAULT_IMAGE_ATTACK_EVIDENCE_RECORDS_PATH = Path("outputs/image_attack_evidence/formal_attack_detection_records.jsonl")
DEFAULT_MAX_SOURCE_RECORDS: int | None = None
REAL_ATTACK_WATERMARK_RESCORE_METRIC_STATUS = "measured_from_real_attacked_image_watermark_rescore_formal_protocol"
REAL_ATTACK_RETENTION_PROXY_METRIC_STATUS = "measured_from_real_attacked_image_retention_proxy_formal_protocol"
LEGACY_REAL_ATTACK_METRIC_STATUS = "measured_from_real_attacked_image_formal_protocol"
REAL_ATTACK_METRIC_STATUSES = (
    REAL_ATTACK_WATERMARK_RESCORE_METRIC_STATUS,
    REAL_ATTACK_RETENTION_PROXY_METRIC_STATUS,
    LEGACY_REAL_ATTACK_METRIC_STATUS,
)


def stable_json_text(value: Any) -> str:
    """把 JSON 兼容对象转为稳定文本。"""
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def json_line(value: dict[str, Any]) -> str:
    """把字典转为 JSONL 单行。"""
    return json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n"


def read_json(path: Path) -> dict[str, Any]:
    """读取 JSON 文件。"""
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> tuple[dict[str, Any], ...]:
    """读取 JSONL 文件。"""
    return tuple(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def read_optional_jsonl(path: Path) -> tuple[dict[str, Any], ...]:
    """读取可选 JSONL 输入, 文件不存在时返回空记录集。"""
    if not path.exists():
        return ()
    return read_jsonl(path)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    """写出 CSV 表格。"""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def resolve_code_version(root_path: Path) -> str:
    """读取 Git 提交标识, 工作区有变更时附加 dirty 标记。"""
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


def ensure_output_dir_under_outputs(root_path: Path, output_dir: Path) -> Path:
    """确保持久化输出目录位于 outputs 下。"""
    resolved_output_dir = (root_path / output_dir).resolve() if not output_dir.is_absolute() else output_dir.resolve()
    outputs_root = (root_path / "outputs").resolve()
    try:
        resolved_output_dir.relative_to(outputs_root)
    except ValueError as exc:
        raise ValueError("攻击矩阵输出目录必须位于 outputs/ 下") from exc
    return resolved_output_dir


def resolve_input_path(root_path: Path, path: str | Path) -> Path:
    """解析输入路径。"""
    candidate = Path(path)
    return candidate.resolve() if candidate.is_absolute() else (root_path / candidate).resolve()


def relative_or_absolute(path: Path, root_path: Path) -> str:
    """将路径尽量转为相对仓库根目录的字符串。"""
    try:
        return path.resolve().relative_to(root_path).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def full_rescue_records(records: tuple[dict[str, Any], ...]) -> tuple[dict[str, Any], ...]:
    """只保留完整 rescue 协议记录。"""
    return tuple(record for record in records if record.get("rescue_ablation_mode") == "full_rescue")


def formal_real_attack_records(records: tuple[dict[str, Any], ...]) -> tuple[dict[str, Any], ...]:
    """筛选真实 attacked image 闭环产出的正式检测记录。"""
    return tuple(record for record in records if record.get("metric_status") in REAL_ATTACK_METRIC_STATUSES)


def attack_record_key(record: dict[str, Any]) -> tuple[str, str, str]:
    """返回攻击记录在聚合表中的稳定键。"""

    return (
        str(record.get("attack_family", "")),
        str(record.get("attack_name", "")),
        str(record.get("resource_profile", "")),
    )


def split_role_key(record: dict[str, Any]) -> tuple[str, str]:
    """返回攻击记录的 split 与样本角色键。"""

    return (str(record.get("split", "")), str(record.get("sample_role", "")))


def group_records_by_attack_key(records: tuple[dict[str, Any], ...]) -> dict[tuple[str, str, str], list[dict[str, Any]]]:
    """按攻击配置键分组记录。"""

    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(attack_record_key(record), []).append(record)
    return grouped


def formal_coverage_complete(proxy_group: list[dict[str, Any]], formal_group: list[dict[str, Any]]) -> bool:
    """判断真实图像 formal records 是否完整覆盖同一攻击配置的 proxy records。

    该函数属于证据治理层: 只有当 formal records 的总量和 split/sample_role
    分布都覆盖 proxy records 时, 才允许移除同配置 proxy 记录。这样可以避免
    少量真实 attacked image 记录把同攻击配置的大量 proxy 统计整体覆盖。
    """

    performed_formal = [record for record in formal_group if bool(record.get("attack_performed"))]
    if not proxy_group or not performed_formal or len(performed_formal) < len(proxy_group):
        return False
    proxy_role_counts = Counter(split_role_key(record) for record in proxy_group)
    formal_role_counts = Counter(split_role_key(record) for record in performed_formal)
    return all(formal_role_counts[key] >= count for key, count in proxy_role_counts.items())


def serializable_split_role_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    """把 split/sample_role 计数转为 JSON 兼容字典。"""

    counts = Counter(split_role_key(record) for record in records)
    return {f"{split_name}|{sample_role}": count for (split_name, sample_role), count in sorted(counts.items())}


def build_formal_attack_coverage_report(
    proxy_records: tuple[dict[str, Any], ...],
    formal_records: tuple[dict[str, Any], ...],
) -> dict[str, Any]:
    """汇总真实图像 formal records 对 proxy records 的覆盖完整性。"""

    proxy_groups = group_records_by_attack_key(proxy_records)
    formal_groups = group_records_by_attack_key(tuple(record for record in formal_records if bool(record.get("attack_performed"))))
    complete_keys: list[tuple[str, str, str]] = []
    incomplete_rows: list[dict[str, Any]] = []
    for key, proxy_group in sorted(proxy_groups.items()):
        formal_group = formal_groups.get(key, [])
        complete = formal_coverage_complete(proxy_group, formal_group)
        if complete:
            complete_keys.append(key)
        elif formal_group:
            attack_family, attack_name, resource_profile = key
            incomplete_rows.append(
                {
                    "attack_family": attack_family,
                    "attack_name": attack_name,
                    "resource_profile": resource_profile,
                    "proxy_record_count": len(proxy_group),
                    "formal_record_count": len(formal_group),
                    "proxy_split_role_counts": serializable_split_role_counts(proxy_group),
                    "formal_split_role_counts": serializable_split_role_counts(formal_group),
                }
            )
    return {
        "formal_proxy_replacement_complete_keys": complete_keys,
        "formal_proxy_replacement_complete_count": len(complete_keys),
        "formal_proxy_replacement_incomplete_count": len(incomplete_rows),
        "formal_proxy_replacement_incomplete_examples": incomplete_rows[:20],
        "formal_proxy_replacement_requires_complete_split_role_coverage": True,
    }


def deduplicate_formal_attack_records(records: tuple[dict[str, Any], ...]) -> tuple[dict[str, Any], ...]:
    """按正式记录标识去重, 保留多来源真实图像攻击记录的稳定顺序。"""

    seen: set[str] = set()
    deduplicated: list[dict[str, Any]] = []
    for record in records:
        record_id = str(record.get("attack_record_id") or record.get("attack_record_digest") or build_stable_digest(record))
        if record_id in seen:
            continue
        seen.add(record_id)
        deduplicated.append(record)
    return tuple(deduplicated)


def filter_proxy_records_covered_by_formal_records(
    proxy_records: tuple[dict[str, Any], ...],
    formal_records: tuple[dict[str, Any], ...],
    coverage_report: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], ...]:
    """移除已被真实图像级 formal records 覆盖的本地代理记录。

    该实现属于项目特定的证据治理逻辑: 当同一攻击配置已有真实
    attacked image 记录时, 聚合表应优先使用真实记录, 避免 proxy 与
    真实记录混合后掩盖当前攻击簇是否已经完成图像级闭环。
    """

    report = coverage_report or build_formal_attack_coverage_report(proxy_records, formal_records)
    covered_keys = {tuple(key) for key in report["formal_proxy_replacement_complete_keys"]}
    return tuple(record for record in proxy_records if attack_record_key(record) not in covered_keys)


def write_consolidated_formal_attack_records(path: Path, records: tuple[dict[str, Any], ...]) -> None:
    """写出多来源真实图像攻击 formal records 合并文件。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json_line(record) for record in records), encoding="utf-8")


def build_real_attack_ingestion_summary(
    attack_configs: tuple[AttackConfig, ...],
    formal_records: tuple[dict[str, Any], ...],
) -> dict[str, Any]:
    """汇总真实攻击记录覆盖范围, 供 attack matrix 和 paper audit 复用。"""
    required_names = sorted({config.attack_name for config in attack_configs if config.requires_gpu})
    measured_names = sorted(
        {
            str(record.get("attack_name", ""))
            for record in formal_records
            if record.get("attack_family") == "regeneration_attack"
        }
    )
    measured_required_names = sorted(set(required_names).intersection(measured_names))
    real_attacked_image_count = sum(1 for record in formal_records if bool(record.get("attacked_image_available")))
    closed_loop_ready = bool(formal_records) and all(
        bool(record.get("attacked_image_available"))
        and bool(record.get("attacked_image_digest"))
        and bool(record.get("source_image_digest"))
        and bool(record.get("metadata", {}).get("attacked_image_path"))
        and bool(record.get("metadata", {}).get("source_image_path"))
        for record in formal_records
    )
    formal_detection_ready = bool(formal_records) and all(
        bool(record.get("metadata", {}).get("formal_boundary_ready")) for record in formal_records
    )
    required_count = len(required_names)
    measured_count = len(measured_required_names)
    return {
        "formal_real_attack_record_count": len(formal_records),
        "real_attacked_image_count": real_attacked_image_count,
        "real_attacked_image_closed_loop_ready": closed_loop_ready,
        "formal_attack_detection_ready": formal_detection_ready,
        "required_regeneration_attack_count": required_count,
        "measured_regeneration_attack_count": measured_count,
        "regeneration_attack_gpu_validation_ready": bool(required_count and measured_count == required_count),
        "gpu_attack_real_measurement_missing_count": max(0, required_count - measured_count),
        "real_regeneration_attack_names": measured_names,
    }


def build_boundary(thresholds: dict[str, Any], threshold_report: dict[str, Any]) -> AttackEvaluationBoundary:
    """从阈值文件和退化报告构造攻击检测边界。"""
    threshold_value = float(threshold_report.get("calibrated_content_threshold", thresholds["threshold_value"]))
    target_fpr = float(threshold_report.get("target_fpr", thresholds.get("target_fpr", PILOT_PAPER_FIXED_FPR)))
    rescue_margin_low = float(threshold_report.get("rescue_margin_low", -0.05))
    allowed_fail_reasons = tuple(threshold_report.get("allowed_fail_reasons", ("geometry_suspected", "low_confidence")))
    return AttackEvaluationBoundary(
        calibrated_content_threshold=threshold_value,
        target_fpr=target_fpr,
        rescue_margin_low=rescue_margin_low,
        allowed_fail_reasons=allowed_fail_reasons,
        threshold_score_field=str(threshold_report.get("threshold_score_field", "raw_content_score")),
        threshold_score_source_field=str(threshold_report.get("threshold_score_source_field", "raw_content_score")),
        fixed_fpr_control_scope=str(threshold_report.get("fixed_fpr_control_scope", "calibration_clean_negative")),
        fixed_fpr_denominator_role=str(threshold_report.get("fixed_fpr_denominator_role", "clean_negative_only")),
        rescue_control_scope=str(threshold_report.get("rescue_control_scope", "evidence_clean_negative")),
        rescue_changes_fpr_denominator=bool(threshold_report.get("rescue_changes_fpr_denominator", False)),
        attacked_negative_boundary_role=str(
            threshold_report.get(
                "attacked_negative_boundary_role",
                "attack_robustness_diagnostic_not_fpr_denominator",
            )
        ),
        attacked_negative_governs_fixed_fpr=bool(threshold_report.get("attacked_negative_governs_fixed_fpr", False)),
    )


def build_attacked_image_registry_rows(records: tuple[dict[str, Any], ...]) -> list[dict[str, Any]]:
    """构造 attacked image registry。

    registry 同时登记本地代理摘要和已导入的图像级 attacked image 摘要。
    真实图像路径仍保留在 formal record 的 metadata 中, 避免 registry 与
    正式检测记录之间出现重复路径治理逻辑。
    """
    return [
        {
            "attack_record_id": record["attack_record_id"],
            "source_record_id": record["source_record_id"],
            "source_image_digest": record["source_image_digest"],
            "source_image_digest_source": record["source_image_digest_source"],
            "attack_config_digest": record["attack_config_digest"],
            "attacked_image_digest": record["attacked_image_digest"],
            "attacked_image_digest_source": record["attacked_image_digest_source"],
            "attacked_image_available": record["attacked_image_available"],
            "attack_performed": record["attack_performed"],
            "metric_status": record["metric_status"],
            "unsupported_reason": record["unsupported_reason"],
            "supports_paper_claim": False,
        }
        for record in records
    ]


def build_attack_manifest(
    root_path: Path,
    output_dir: Path,
    rescue_records_path: Path,
    rescue_manifest_path: Path,
    calibration_thresholds_path: Path,
    threshold_report_path: Path,
    calibration_manifest_path: Path,
    real_attack_records_path: Path,
    rescue_manifest: dict[str, Any],
    calibration_manifest: dict[str, Any],
    threshold_report: dict[str, Any],
    attack_configs: tuple[AttackConfig, ...],
    attack_records: tuple[dict[str, Any], ...],
    formal_real_records: tuple[dict[str, Any], ...],
    family_rows: list[dict[str, Any]],
    boundary: AttackEvaluationBoundary,
) -> dict[str, Any]:
    """构造攻击矩阵专用 manifest。"""
    real_attack_summary = build_real_attack_ingestion_summary(attack_configs, formal_real_records)
    gpu_unsupported_count = real_attack_summary["gpu_attack_real_measurement_missing_count"]
    performed_count = sum(1 for record in attack_records if record["attack_performed"])
    attack_metrics_ready = bool(performed_count and family_rows)
    aligned_quality = extract_aligned_rescoring_metadata(threshold_report, calibration_manifest)
    return {
        "construction_unit_name": CONSTRUCTION_UNIT_NAME,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_manifests": [
            {
                "manifest_path": relative_or_absolute(rescue_manifest_path, root_path),
                "artifact_id": rescue_manifest.get("artifact_id", "unknown"),
            },
            {
                "manifest_path": relative_or_absolute(calibration_manifest_path, root_path),
                "artifact_id": calibration_manifest.get("artifact_id", "unknown"),
            },
        ],
        "input_records_path": relative_or_absolute(rescue_records_path, root_path),
        "input_thresholds_path": relative_or_absolute(calibration_thresholds_path, root_path),
        "input_threshold_report_path": relative_or_absolute(threshold_report_path, root_path),
        "real_attack_records_path": relative_or_absolute(real_attack_records_path, root_path) if real_attack_records_path.exists() else "",
        "image_attack_evidence_records_path": relative_or_absolute(real_attack_records_path, root_path)
        if real_attack_records_path.exists()
        else "",
        **aligned_quality,
        "attacked_images_dir": relative_or_absolute(output_dir / "attacked_images", root_path),
        "attack_config_count": len(attack_configs),
        "attack_record_count": len(attack_records),
        "attack_family_count": len({config.attack_family for config in attack_configs}),
        "performed_attack_record_count": performed_count,
        "gpu_attack_unsupported_count": gpu_unsupported_count,
        **real_attack_summary,
        "formal_image_attack_record_count": len(formal_real_records),
        "attack_metrics_ready": attack_metrics_ready,
        "resource_profiles": sorted({config.resource_profile for config in attack_configs}),
        "conventional_attack_names": sorted({config.attack_name for config in attack_configs if not config.requires_gpu}),
        "regeneration_attack_names": sorted({config.attack_name for config in attack_configs if config.requires_gpu}),
        "evaluation_boundary": boundary.to_dict(),
        "local_proxy_boundary": "local proxy records are retained only for attack configurations without governed real-image formal records",
        "regeneration_attack_status": "real_gpu_formal_records_available"
        if real_attack_summary["regeneration_attack_gpu_validation_ready"]
        else "unsupported_until_real_gpu_artifacts_exist",
        "full_method_claim_ready": False,
        "supports_paper_claim": False,
    }


def extract_aligned_rescoring_metadata(threshold_report: dict[str, Any], calibration_manifest: dict[str, Any]) -> dict[str, Any]:
    """从阈值校准产物中提取真实 aligned rescoring 向下游传播的摘要。"""
    manifest_metadata = calibration_manifest.get("metadata", {})
    return {
        "aligned_rescoring_package_path": threshold_report.get(
            "aligned_rescoring_package_path",
            manifest_metadata.get("aligned_rescoring_package_path", ""),
        ),
        "aligned_rescoring_package_digest": threshold_report.get(
            "aligned_rescoring_package_digest",
            manifest_metadata.get("aligned_rescoring_package_digest", ""),
        ),
        "aligned_rescoring_quality_metrics_ready": threshold_report.get(
            "aligned_rescoring_quality_metrics_ready",
            manifest_metadata.get("aligned_rescoring_quality_metrics_ready", False),
        ),
        "perceptual_metrics_ready": threshold_report.get(
            "perceptual_metrics_ready",
            manifest_metadata.get("perceptual_metrics_ready", False),
        ),
        "aligned_rescoring_record_count": threshold_report.get(
            "aligned_rescoring_record_count",
            manifest_metadata.get("aligned_rescoring_record_count", 0),
        ),
        "real_aligned_rescore_count": threshold_report.get(
            "real_aligned_rescore_count",
            manifest_metadata.get("real_aligned_rescore_count", 0),
        ),
    }


def write_attack_matrix_outputs(
    root: str | Path = ".",
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    rescue_records_path: str | Path = DEFAULT_RESCUE_RECORDS_PATH,
    rescue_manifest_path: str | Path = DEFAULT_RESCUE_MANIFEST_PATH,
    calibration_thresholds_path: str | Path = DEFAULT_CALIBRATION_THRESHOLDS_PATH,
    threshold_report_path: str | Path = DEFAULT_THRESHOLD_REPORT_PATH,
    calibration_manifest_path: str | Path = DEFAULT_CALIBRATION_MANIFEST_PATH,
    real_attack_records_path: str | Path = DEFAULT_REAL_ATTACK_RECORDS_PATH,
    conventional_geometric_records_path: str | Path = DEFAULT_CONVENTIONAL_GEOMETRIC_ATTACK_RECORDS_PATH,
    image_attack_evidence_records_path: str | Path = DEFAULT_IMAGE_ATTACK_EVIDENCE_RECORDS_PATH,
    max_source_records: int | None = DEFAULT_MAX_SOURCE_RECORDS,
) -> dict[str, Any]:
    """写出攻击矩阵相关产物。"""
    root_path = Path(root).resolve()
    resolved_output_dir = ensure_output_dir_under_outputs(root_path, Path(output_dir))
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    attacked_images_dir = resolved_output_dir / "attacked_images"
    attacked_images_dir.mkdir(parents=True, exist_ok=True)

    resolved_records_path = resolve_input_path(root_path, rescue_records_path)
    resolved_rescue_manifest_path = resolve_input_path(root_path, rescue_manifest_path)
    resolved_thresholds_path = resolve_input_path(root_path, calibration_thresholds_path)
    resolved_threshold_report_path = resolve_input_path(root_path, threshold_report_path)
    resolved_calibration_manifest_path = resolve_input_path(root_path, calibration_manifest_path)
    resolved_real_attack_records_path = resolve_input_path(root_path, real_attack_records_path)
    resolved_conventional_geometric_records_path = resolve_input_path(root_path, conventional_geometric_records_path)
    resolved_image_attack_evidence_records_path = resolve_input_path(root_path, image_attack_evidence_records_path)

    rescue_manifest = read_json(resolved_rescue_manifest_path)
    calibration_manifest = read_json(resolved_calibration_manifest_path)
    thresholds = read_json(resolved_thresholds_path)
    threshold_report = read_json(resolved_threshold_report_path)
    boundary = build_boundary(thresholds, threshold_report)
    source_records = full_rescue_records(read_jsonl(resolved_records_path))
    if max_source_records is not None:
        source_records = source_records[:max_source_records]

    attack_configs = default_attack_configs()
    proxy_attack_records_all = build_attack_detection_records(source_records, attack_configs, boundary)
    formal_input_records = deduplicate_formal_attack_records(
        formal_real_attack_records(read_optional_jsonl(resolved_real_attack_records_path))
        + formal_real_attack_records(read_optional_jsonl(resolved_conventional_geometric_records_path))
    )
    write_consolidated_formal_attack_records(resolved_image_attack_evidence_records_path, formal_input_records)
    formal_coverage_report = build_formal_attack_coverage_report(proxy_attack_records_all, formal_input_records)
    proxy_attack_records = filter_proxy_records_covered_by_formal_records(
        proxy_attack_records_all,
        formal_input_records,
        formal_coverage_report,
    )
    real_attack_records = formal_input_records
    attack_records = tuple(proxy_attack_records) + tuple(real_attack_records)
    family_rows = family_metrics(attack_records)
    strength_rows = strength_curve(attack_records)
    retention_rows = score_retention_rows(attack_records)
    rescue_rows = rescue_by_attack_rows(attack_records)
    registry_rows = build_attacked_image_registry_rows(attack_records)

    records_path = resolved_output_dir / "attack_detection_records.jsonl"
    registry_path = resolved_output_dir / "attacked_image_registry.jsonl"
    attack_manifest_path = resolved_output_dir / "attack_manifest.json"
    family_metrics_path = resolved_output_dir / "attack_family_metrics.csv"
    strength_curve_path = resolved_output_dir / "attack_strength_curve.csv"
    retention_path = resolved_output_dir / "score_retention_by_attack.csv"
    rescue_path = resolved_output_dir / "rescue_by_attack.csv"
    manifest_path = resolved_output_dir / "manifest.local.json"

    records_path.write_text("".join(json_line(record) for record in attack_records), encoding="utf-8")
    registry_path.write_text("".join(json_line(record) for record in registry_rows), encoding="utf-8")
    attack_manifest = build_attack_manifest(
        root_path=root_path,
        output_dir=resolved_output_dir,
        rescue_records_path=resolved_records_path,
        rescue_manifest_path=resolved_rescue_manifest_path,
        calibration_thresholds_path=resolved_thresholds_path,
        threshold_report_path=resolved_threshold_report_path,
        calibration_manifest_path=resolved_calibration_manifest_path,
        real_attack_records_path=resolved_image_attack_evidence_records_path,
        rescue_manifest=rescue_manifest,
        calibration_manifest=calibration_manifest,
        threshold_report=threshold_report,
        attack_configs=attack_configs,
        attack_records=attack_records,
        formal_real_records=real_attack_records,
        family_rows=family_rows,
        boundary=boundary,
    )
    attack_manifest.update(formal_coverage_report)
    attack_manifest_path.write_text(stable_json_text(attack_manifest), encoding="utf-8")
    write_csv(
        family_metrics_path,
        family_rows,
        [
            "attack_family",
            "attack_name",
            "resource_profile",
            "metric_status",
            "attack_record_count",
            "supported_record_count",
            "unsupported_record_count",
            "positive_count",
            "negative_count",
            "true_positive_rate",
            "false_positive_rate",
            "clean_false_positive_rate",
            "attacked_false_positive_rate",
            "quality_score_proxy_mean",
            "score_retention_mean",
            "lf_score_retention_mean",
            "hf_score_retention_mean",
            "attention_consistency_proxy_mean",
            "geometry_reliable_rate",
            "rescue_rate",
            "supports_paper_claim",
        ],
    )
    write_csv(
        strength_curve_path,
        strength_rows,
        [
            "attack_family",
            "attack_name",
            "attack_strength",
            "resource_profile",
            "metric_status",
            "attack_record_count",
            "supported_record_count",
            "true_positive_rate",
            "false_positive_rate",
            "score_retention_mean",
            "quality_score_proxy_mean",
            "supports_paper_claim",
        ],
    )
    write_csv(
        retention_path,
        retention_rows,
        [
            "attack_family",
            "attack_name",
            "attack_strength",
            "resource_profile",
            "metric_status",
            "attack_record_count",
            "supported_record_count",
            "score_retention_mean",
            "score_retention_min",
            "score_retention_max",
            "lf_score_retention_mean",
            "hf_score_retention_mean",
            "supports_paper_claim",
        ],
    )
    write_csv(
        rescue_path,
        rescue_rows,
        [
            "attack_family",
            "attack_name",
            "attack_strength",
            "resource_profile",
            "metric_status",
            "attack_record_count",
            "supported_record_count",
            "rescue_eligible_count",
            "rescue_applied_count",
            "rescue_rate",
            "geometry_reliable_rate",
            "attention_consistency_proxy_mean",
            "supports_paper_claim",
        ],
    )

    output_paths = tuple(
        relative_or_absolute(path, root_path)
        for path in (
            attacked_images_dir,
            attack_manifest_path,
            registry_path,
            records_path,
            family_metrics_path,
            strength_curve_path,
            retention_path,
            rescue_path,
            resolved_image_attack_evidence_records_path,
            manifest_path,
        )
    )
    summary = {
        "attack_manifest": attack_manifest,
        "family_metrics": family_rows,
        "strength_curve": strength_rows,
        "score_retention": retention_rows,
        "rescue_by_attack": rescue_rows,
    }
    aligned_quality = extract_aligned_rescoring_metadata(threshold_report, calibration_manifest)
    manifest = build_artifact_manifest(
        artifact_id="attack_matrix_manifest",
        artifact_type="local_manifest",
        input_paths=(
            relative_or_absolute(resolved_records_path, root_path),
            relative_or_absolute(resolved_rescue_manifest_path, root_path),
            relative_or_absolute(resolved_thresholds_path, root_path),
            relative_or_absolute(resolved_threshold_report_path, root_path),
            relative_or_absolute(resolved_calibration_manifest_path, root_path),
            *(
                (relative_or_absolute(resolved_real_attack_records_path, root_path),)
                if resolved_real_attack_records_path.exists()
                else ()
            ),
            *(
                (relative_or_absolute(resolved_conventional_geometric_records_path, root_path),)
                if resolved_conventional_geometric_records_path.exists()
                else ()
            ),
        ),
        output_paths=output_paths,
        config={
            **boundary.to_dict(),
            "max_source_records": max_source_records,
            "attack_config_digest": build_stable_digest([config.to_dict() for config in attack_configs]),
            "summary_digest": build_stable_digest(summary),
            "formal_image_attack_record_count": len(real_attack_records),
            "image_attack_evidence_records_path": relative_or_absolute(resolved_image_attack_evidence_records_path, root_path),
        },
        code_version=resolve_code_version(root_path),
        rebuild_command="python scripts/write_attack_matrix_outputs.py",
        metadata={
            "construction_unit_name": CONSTRUCTION_UNIT_NAME,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "protocol_decision": "pass" if attack_manifest["attack_metrics_ready"] else "fail",
            "full_method_claim_ready": False,
            "supports_paper_claim": False,
            **aligned_quality,
        },
    ).to_dict()
    manifest_path.write_text(stable_json_text(manifest), encoding="utf-8")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器。"""
    parser = argparse.ArgumentParser(description="写出攻击矩阵、score retention 与 rescue-by-attack 产物。")
    parser.add_argument("--root", default=".", help="仓库根目录。")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="输出目录, 必须位于 outputs/ 下。")
    parser.add_argument("--rescue-records-path", default=str(DEFAULT_RESCUE_RECORDS_PATH), help="几何 rescue 记录路径。")
    parser.add_argument("--rescue-manifest-path", default=str(DEFAULT_RESCUE_MANIFEST_PATH), help="几何 rescue manifest 路径。")
    parser.add_argument(
        "--calibration-thresholds-path",
        default=str(DEFAULT_CALIBRATION_THRESHOLDS_PATH),
        help="fixed-FPR 阈值文件路径。",
    )
    parser.add_argument("--threshold-report-path", default=str(DEFAULT_THRESHOLD_REPORT_PATH), help="阈值边界报告路径。")
    parser.add_argument(
        "--calibration-manifest-path",
        default=str(DEFAULT_CALIBRATION_MANIFEST_PATH),
        help="阈值校准 manifest 路径。",
    )
    parser.add_argument(
        "--real-attack-records-path",
        default=str(DEFAULT_REAL_ATTACK_RECORDS_PATH),
        help="真实 attacked image formal records 路径, 文件不存在时跳过。",
    )
    parser.add_argument(
        "--conventional-geometric-records-path",
        default=str(DEFAULT_CONVENTIONAL_GEOMETRIC_ATTACK_RECORDS_PATH),
        help="常规失真与几何变换 attacked image formal records 路径, 文件不存在时跳过。",
    )
    parser.add_argument(
        "--image-attack-evidence-records-path",
        default=str(DEFAULT_IMAGE_ATTACK_EVIDENCE_RECORDS_PATH),
        help="多来源 formal image attack records 合并输出路径。",
    )
    parser.add_argument("--max-source-records", type=int, default=DEFAULT_MAX_SOURCE_RECORDS, help="最多读取的 full rescue 源记录数。")
    return parser


def main() -> None:
    """命令行入口。"""
    args = build_parser().parse_args()
    manifest = write_attack_matrix_outputs(
        root=args.root,
        output_dir=args.output_dir,
        rescue_records_path=args.rescue_records_path,
        rescue_manifest_path=args.rescue_manifest_path,
        calibration_thresholds_path=args.calibration_thresholds_path,
        threshold_report_path=args.threshold_report_path,
        calibration_manifest_path=args.calibration_manifest_path,
        real_attack_records_path=args.real_attack_records_path,
        conventional_geometric_records_path=args.conventional_geometric_records_path,
        image_attack_evidence_records_path=args.image_attack_evidence_records_path,
        max_source_records=args.max_source_records,
    )
    print(stable_json_text(manifest), end="")


if __name__ == "__main__":
    main()
