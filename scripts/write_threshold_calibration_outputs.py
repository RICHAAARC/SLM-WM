"""写出 fixed-FPR 阈值校准与常规指标产物。"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
from io import StringIO
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.protocol.calibration import (
    FixedFprCalibrationConfig,
    calibrated_records,
    curve_rows,
    empirical_threshold_at_fpr,
    operating_point_metrics,
    score_distribution_rows,
    split_role,
)
from experiments.protocol.pilot_paper_fixed_fpr import PILOT_PAPER_FIXED_FPR
from main.analysis.artifact_manifest import build_artifact_manifest
from main.core.digest import build_stable_digest

CONSTRUCTION_UNIT_NAME = "threshold_calibration"
DEFAULT_OUTPUT_DIR = Path("outputs/threshold_calibration")
DEFAULT_RESCUE_RECORDS_PATH = Path("outputs/geometric_rescue/aligned_detection_records.jsonl")
DEFAULT_RESCUE_AUDIT_PATH = Path("outputs/geometric_rescue/geometry_rescue_audit.json")
DEFAULT_TARGET_FPR = PILOT_PAPER_FIXED_FPR
FIXED_FPR_CONTROL_SCOPE = "calibration_clean_negative"
FIXED_FPR_DENOMINATOR_ROLE = "clean_negative_only"
RESCUE_CONTROL_SCOPE = "evidence_clean_negative"
ATTACKED_NEGATIVE_BOUNDARY_ROLE = "attack_robustness_diagnostic_not_fpr_denominator"
ALIGNED_RESCORING_PACKAGE_ENV = "SLM_WM_ALIGNED_RESCORING_PACKAGE_PATH"
ALIGNED_RESCORING_PACKAGE_PATTERNS = (
    "outputs/aligned_rescoring_package_*.zip",
    "outputs/aligned_rescoring/aligned_rescoring_package_*.zip",
)
ALIGNED_RESCORING_RESULT_MEMBER = "outputs/aligned_rescoring/aligned_rescoring_result.json"
ALIGNED_RESCORING_QUALITY_MEMBER = "outputs/aligned_rescoring/aligned_rescoring_quality_metrics.csv"


def stable_json_text(value: Any) -> str:
    """把 JSON 兼容对象转为稳定文本。"""
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def read_jsonl(path: Path) -> tuple[dict[str, Any], ...]:
    """读取 JSONL 文件。"""
    return tuple(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    """写出 CSV 表格。"""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def file_digest(path: Path) -> str:
    """计算文件 SHA256 摘要。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
        raise ValueError("阈值校准输出目录必须位于 outputs/ 下") from exc
    return resolved_output_dir


def relative_or_absolute(path: Path, root_path: Path) -> str:
    """将路径尽量转为相对仓库根目录的字符串。"""
    try:
        return path.resolve().relative_to(root_path).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def resolve_input_path(root_path: Path, path: str | Path) -> Path:
    """解析外部输入路径, 允许调用方传入绝对路径或相对仓库根目录的路径。"""
    candidate = Path(path)
    return candidate.resolve() if candidate.is_absolute() else (root_path / candidate).resolve()


def full_rescue_records(records: tuple[dict[str, Any], ...]) -> tuple[dict[str, Any], ...]:
    """只保留完整 rescue 协议记录。"""
    return tuple(record for record in records if record.get("rescue_ablation_mode") == "full_rescue")


def read_quality_metrics_from_attention_package(audit: dict[str, Any], root_path: Path) -> dict[str, float | str]:
    """从真实 attention latent injection 包读取 paired quality 指标。"""
    package_path_text = str(audit.get("attention_latent_injection_package_path", ""))
    if not package_path_text:
        return {}
    package_path = Path(package_path_text)
    resolved_package_path = package_path if package_path.is_absolute() else root_path / package_path
    if not resolved_package_path.exists():
        return {}
    with ZipFile(resolved_package_path) as archive:
        result = json.loads(
            archive.read("outputs/attention_latent_injection/attention_latent_injection_result.json").decode("utf-8")
        )
    return {
        "psnr": result.get("psnr", "unsupported"),
        "ssim": result.get("ssim", "unsupported"),
        "mse": result.get("mse", "unsupported"),
        "mean_abs_error": result.get("mean_abs_error", "unsupported"),
    }


def discover_latest_aligned_rescoring_package(root_path: Path) -> Path | None:
    """在 outputs 中选择文件名或写入时间最新的 aligned rescoring 结果包。"""
    candidates: list[Path] = []
    for pattern in ALIGNED_RESCORING_PACKAGE_PATTERNS:
        candidates.extend(path for path in root_path.glob(pattern) if path.is_file())
    if not candidates:
        return None
    return max(candidates, key=lambda path: (path.name, path.stat().st_mtime_ns))


def resolve_aligned_rescoring_package_path(
    root_path: Path,
    aligned_rescoring_package_path: str | Path | None,
) -> Path | None:
    """解析 aligned rescoring 结果包路径, 未显式传入时自动使用 outputs 中最新包。"""
    if aligned_rescoring_package_path:
        resolved_path = resolve_input_path(root_path, aligned_rescoring_package_path)
        if not resolved_path.exists():
            raise FileNotFoundError(f"aligned rescoring 结果包不存在: {resolved_path}")
        return resolved_path
    env_path = os.environ.get(ALIGNED_RESCORING_PACKAGE_ENV, "").strip()
    if env_path:
        resolved_path = resolve_input_path(root_path, env_path)
        if not resolved_path.exists():
            raise FileNotFoundError(f"aligned rescoring 结果包不存在: {resolved_path}")
        return resolved_path
    return discover_latest_aligned_rescoring_package(root_path)


def empty_aligned_rescoring_quality() -> dict[str, Any]:
    """构造没有真实 aligned rescoring 输入时的中性质量摘要。"""
    return {
        "aligned_rescoring_package_path": "",
        "aligned_rescoring_package_digest": "",
        "aligned_rescoring_quality_metrics_ready": False,
        "perceptual_metrics_ready": False,
        "aligned_rescoring_record_count": 0,
        "real_aligned_rescore_count": 0,
        "quality_metrics": {},
    }


def read_quality_metrics_from_aligned_rescoring_package(package_path: Path, root_path: Path) -> dict[str, Any]:
    """从真实 aligned rescoring 结果包读取 LPIPS 与 CLIP 等 pair-level 指标。"""
    with ZipFile(package_path) as archive:
        result = json.loads(archive.read(ALIGNED_RESCORING_RESULT_MEMBER).decode("utf-8"))
        rows = tuple(csv.DictReader(StringIO(archive.read(ALIGNED_RESCORING_QUALITY_MEMBER).decode("utf-8"))))
    first_row = rows[0] if rows else {}
    lpips_status = first_row.get("lpips_status", "unsupported")
    clip_status = first_row.get("clip_score_status", "unsupported")
    perceptual_metrics_ready = bool(result.get("perceptual_metrics_ready"))
    image_metric_status = "measured" if first_row and result.get("image_quality_metrics_ready") else "unsupported"
    return {
        "aligned_rescoring_package_path": relative_or_absolute(package_path, root_path),
        "aligned_rescoring_package_digest": file_digest(package_path),
        "aligned_rescoring_quality_metrics_ready": bool(
            result.get("run_decision") == "pass"
            and result.get("image_quality_metrics_ready")
            and perceptual_metrics_ready
            and lpips_status == "measured"
            and clip_status == "measured"
        ),
        "perceptual_metrics_ready": perceptual_metrics_ready,
        "aligned_rescoring_record_count": result.get("aligned_rescoring_record_count", 0),
        "real_aligned_rescore_count": result.get("real_aligned_rescore_count", 0),
        "quality_metrics": {
            "psnr": {
                "value": first_row.get("psnr", "unsupported"),
                "status": image_metric_status,
            },
            "ssim": {
                "value": first_row.get("ssim", "unsupported"),
                "status": image_metric_status,
            },
            "mse": {
                "value": first_row.get("mse", "unsupported"),
                "status": image_metric_status,
            },
            "mean_abs_error": {
                "value": first_row.get("mean_abs_error", "unsupported"),
                "status": image_metric_status,
            },
            "lpips": {
                "value": first_row.get("lpips", "unsupported"),
                "status": lpips_status,
            },
            "clip_score": {
                "value": first_row.get("clip_score", "unsupported"),
                "status": clip_status,
            },
            "clip_score_clean": {
                "value": first_row.get("clip_score_clean", "unsupported"),
                "status": clip_status,
            },
            "clip_score_aligned": {
                "value": first_row.get("clip_score_aligned", "unsupported"),
                "status": clip_status,
            },
            "clip_score_delta": {
                "value": first_row.get("clip_score_delta", "unsupported"),
                "status": clip_status,
            },
            "fid": {
                "value": first_row.get("fid", "unsupported"),
                "status": first_row.get("fid_status", "unsupported"),
            },
            "kid": {
                "value": first_row.get("kid", "unsupported"),
                "status": first_row.get("kid_status", "unsupported"),
            },
        },
    }


def aligned_rescoring_output_metadata(aligned_quality: dict[str, Any]) -> dict[str, Any]:
    """提取需要写入下游 manifest 与报告的 aligned rescoring 摘要字段。"""
    return {
        "aligned_rescoring_package_path": aligned_quality["aligned_rescoring_package_path"],
        "aligned_rescoring_package_digest": aligned_quality["aligned_rescoring_package_digest"],
        "aligned_rescoring_quality_metrics_ready": aligned_quality["aligned_rescoring_quality_metrics_ready"],
        "perceptual_metrics_ready": aligned_quality["perceptual_metrics_ready"],
        "aligned_rescoring_record_count": aligned_quality["aligned_rescoring_record_count"],
        "real_aligned_rescore_count": aligned_quality["real_aligned_rescore_count"],
    }


def build_quality_metric_rows(audit: dict[str, Any], root_path: Path, aligned_quality: dict[str, Any]) -> list[dict[str, Any]]:
    """构造常规图像质量指标摘要。"""
    measured = read_quality_metrics_from_attention_package(audit, root_path)
    aligned_metrics = aligned_quality["quality_metrics"]
    has_aligned_source = bool(aligned_quality["aligned_rescoring_package_path"])
    rows: list[dict[str, Any]] = []
    for metric_name in ("psnr", "ssim", "mse", "mean_abs_error"):
        aligned_entry = aligned_metrics.get(metric_name, {})
        if has_aligned_source and aligned_entry:
            metric_value = aligned_entry.get("value", "unsupported")
            metric_source = "aligned_rescoring_package"
            metric_status = aligned_entry.get("status", "unsupported")
        else:
            metric_value = measured.get(metric_name, "unsupported")
            metric_source = "attention_latent_injection_package" if metric_name in measured else "missing"
            metric_status = "measured" if metric_name in measured else "unsupported"
        rows.append(
            {
                "quality_metric_name": metric_name,
                "quality_metric_value": metric_value,
                "quality_metric_source": metric_source,
                "metric_status": metric_status,
                "supports_paper_claim": False,
            }
        )
    for metric_name in ("lpips", "clip_score", "clip_score_clean", "clip_score_aligned", "clip_score_delta", "fid", "kid"):
        metric_entry = aligned_metrics.get(metric_name, {})
        rows.append(
            {
                "quality_metric_name": metric_name,
                "quality_metric_value": metric_entry.get("value", "unsupported"),
                "quality_metric_source": "aligned_rescoring_package" if has_aligned_source else "not_computed_in_local_proxy",
                "metric_status": metric_entry.get("status", "unsupported"),
                "supports_paper_claim": False,
            }
        )
    return rows


def build_standard_metric_rows(metrics: dict[str, Any], records: tuple[dict[str, Any], ...]) -> list[dict[str, Any]]:
    """构造常规水印统计指标。"""
    positive_count = sum(1 for record in records if record["sample_role"] == "positive_source")
    clean_count = sum(1 for record in records if record["sample_role"] == "clean_negative")
    agreement_count = sum(
        1
        for record in records
        if (record["sample_role"] == "positive_source" and record["evidence_decision"])
        or (record["sample_role"] == "clean_negative" and not record["evidence_decision"])
    )
    agreement_denominator = positive_count + clean_count
    bit_agreement = agreement_count / agreement_denominator if agreement_denominator else 0.0
    metric_values = {
        "true_positive_rate": metrics["true_positive_rate"],
        "raw_content_clean_fpr": metrics["raw_content_clean_fpr"],
        "evidence_clean_fpr": metrics["evidence_clean_fpr"],
        "evidence_attacked_fpr": metrics["evidence_attacked_fpr"],
        "raw_score_auc": metrics["raw_score_auc"],
        "aligned_score_auc": metrics["aligned_score_auc"],
        "bit_agreement_proxy": bit_agreement,
        "ber_proxy": 1.0 - bit_agreement,
        "ncc_proxy": sum(float(record["raw_content_score"]) for record in records) / len(records) if records else 0.0,
        "correlation_proxy": metrics["raw_score_auc"],
        "score_retention_proxy": metrics["aligned_score_auc"],
        "aligned_score_gain_mean": metrics["aligned_score_gain_mean"],
    }
    return [
        {
            "metric_name": metric_name,
            "metric_value": metric_value,
            "metric_source": "calibrated_rescue_records",
            "metric_status": "proxy" if metric_name.endswith("_proxy") else "measured_from_records",
            "supports_paper_claim": False,
        }
        for metric_name, metric_value in metric_values.items()
    ]


def build_rescue_fpr_rows(metrics: dict[str, Any], threshold_report: dict[str, Any]) -> list[dict[str, Any]]:
    """构造 rescue 前后 FPR 审计表。"""
    row_specs = (
        (
            "raw_content_clean_negative",
            "fixed_fpr_raw_clean_control",
            metrics["raw_content_clean_fpr"],
            True,
        ),
        (
            RESCUE_CONTROL_SCOPE,
            "fixed_fpr_evidence_clean_control",
            metrics["evidence_clean_fpr"],
            True,
        ),
        (
            "evidence_attacked_negative",
            "attack_robustness_diagnostic",
            metrics["evidence_attacked_fpr"],
            False,
        ),
    )
    return [
        {
            "operating_point_id": metrics["operating_point_id"],
            "target_fpr": metrics["target_fpr"],
            "decision_scope": decision_scope,
            "statistical_boundary": statistical_boundary,
            "observed_fpr": observed_fpr,
            "fpr_exceeds_target": observed_fpr > metrics["target_fpr"],
            "governs_fixed_fpr": governs_fixed_fpr,
            "threshold_degenerate": threshold_report["threshold_degenerate"],
            "supports_paper_claim": False,
        }
        for decision_scope, statistical_boundary, observed_fpr, governs_fixed_fpr in row_specs
    ]


def build_threshold_report(
    threshold: Any,
    metrics: dict[str, Any],
    config: FixedFprCalibrationConfig,
    rescue_audit: dict[str, Any],
    aligned_quality: dict[str, Any],
) -> dict[str, Any]:
    """构造阈值退化与 fixed-FPR 主张边界报告。"""
    report = threshold.to_dict()
    clean_fpr_exceeds_target = metrics["evidence_clean_fpr"] > config.target_fpr
    attacked_fpr_diagnostic_exceeds_target = metrics["evidence_attacked_fpr"] > config.target_fpr
    fixed_fpr_and_rescue_boundary_ready = not threshold.threshold_degenerate and not clean_fpr_exceeds_target
    report.update(
        {
            "construction_unit_name": CONSTRUCTION_UNIT_NAME,
            "calibrated_content_threshold": threshold.threshold_value,
            "fixed_fpr_control_scope": FIXED_FPR_CONTROL_SCOPE,
            "fixed_fpr_denominator_role": FIXED_FPR_DENOMINATOR_ROLE,
            "rescue_control_scope": RESCUE_CONTROL_SCOPE,
            "rescue_margin_low": config.rescue_margin_low,
            "allowed_fail_reasons": list(config.allowed_fail_reasons),
            "rescue_window_frozen": True,
            "fail_reason_gate_frozen": True,
            "geometry_gate_source": "geometric_rescue_records",
            "rescue_changes_fpr_denominator": False,
            "attacked_negative_boundary_role": ATTACKED_NEGATIVE_BOUNDARY_ROLE,
            "attacked_negative_governs_fixed_fpr": False,
            "clean_fpr_exceeds_target": clean_fpr_exceeds_target,
            "attacked_fpr_diagnostic_exceeds_target": attacked_fpr_diagnostic_exceeds_target,
            "evidence_fpr_exceeds_target": clean_fpr_exceeds_target,
            "fixed_fpr_boundary_ready": not threshold.threshold_degenerate,
            "rescue_boundary_ready": not clean_fpr_exceeds_target,
            "fixed_fpr_and_rescue_boundary_ready": fixed_fpr_and_rescue_boundary_ready,
            "raw_content_claim_ready": not threshold.threshold_degenerate and metrics["raw_content_clean_fpr"] <= config.target_fpr,
            "full_method_claim_ready": False,
            "unsupported_reason": "aligned_content_score_local_proxy",
            "input_attention_geometry_ready": rescue_audit.get("attention_geometry_ready", False),
            "input_image_quality_metrics_ready": rescue_audit.get("image_quality_metrics_ready", False),
            "supports_paper_claim": False,
        }
    )
    report.update(aligned_rescoring_output_metadata(aligned_quality))
    return report


def write_threshold_calibration_outputs(
    root: str | Path = ".",
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    rescue_records_path: str | Path = DEFAULT_RESCUE_RECORDS_PATH,
    rescue_audit_path: str | Path = DEFAULT_RESCUE_AUDIT_PATH,
    target_fpr: float = DEFAULT_TARGET_FPR,
    aligned_rescoring_package_path: str | Path | None = None,
) -> dict[str, Any]:
    """写出 fixed-FPR 阈值校准与指标产物。"""
    root_path = Path(root).resolve()
    resolved_output_dir = ensure_output_dir_under_outputs(root_path, Path(output_dir))
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    records_path = Path(rescue_records_path)
    audit_path = Path(rescue_audit_path)
    resolved_records_path = records_path if records_path.is_absolute() else root_path / records_path
    resolved_audit_path = audit_path if audit_path.is_absolute() else root_path / audit_path
    rescue_audit = json.loads(resolved_audit_path.read_text(encoding="utf-8"))
    resolved_aligned_package_path = resolve_aligned_rescoring_package_path(root_path, aligned_rescoring_package_path)
    aligned_quality = (
        read_quality_metrics_from_aligned_rescoring_package(resolved_aligned_package_path, root_path)
        if resolved_aligned_package_path and resolved_aligned_package_path.exists()
        else empty_aligned_rescoring_quality()
    )
    config = FixedFprCalibrationConfig(target_fpr=target_fpr)
    records = full_rescue_records(read_jsonl(resolved_records_path))
    calibration_clean_records = split_role(records, config.calibration_split, config.clean_negative_role)
    threshold = empirical_threshold_at_fpr(
        (record["raw_content_score"] for record in calibration_clean_records),
        config,
    )
    calibrated = calibrated_records(records, threshold, config)
    metrics = operating_point_metrics(calibrated, threshold, config)
    threshold_report = build_threshold_report(threshold, metrics, config, rescue_audit, aligned_quality)
    roc_rows, det_rows = curve_rows(calibrated, config)
    distribution_rows = score_distribution_rows(calibrated)
    standard_rows = build_standard_metric_rows(metrics, calibrated)
    quality_rows = build_quality_metric_rows(rescue_audit, root_path, aligned_quality)
    fpr_rows = build_rescue_fpr_rows(metrics, threshold_report)

    thresholds_path = resolved_output_dir / "calibration_thresholds.json"
    operating_points_path = resolved_output_dir / "fixed_fpr_operating_points.csv"
    standard_metrics_path = resolved_output_dir / "standard_watermark_metrics.csv"
    quality_metrics_path = resolved_output_dir / "quality_metrics_summary.csv"
    roc_path = resolved_output_dir / "roc_curve_points.csv"
    det_path = resolved_output_dir / "det_curve_points.csv"
    distribution_path = resolved_output_dir / "score_distribution_table.csv"
    degeneracy_path = resolved_output_dir / "threshold_degeneracy_report.json"
    fpr_audit_path = resolved_output_dir / "rescue_fpr_audit.csv"
    manifest_path = resolved_output_dir / "manifest.local.json"

    thresholds_path.write_text(stable_json_text(threshold.to_dict()), encoding="utf-8")
    write_csv(
        operating_points_path,
        [metrics],
        [
            "operating_point_id",
            "target_fpr",
            "calibrated_content_threshold",
            "threshold_degenerate",
            "positive_count",
            "clean_negative_count",
            "attacked_negative_count",
            "true_positive_rate",
            "raw_content_clean_fpr",
            "evidence_clean_fpr",
            "evidence_attacked_fpr",
            "rescue_applied_rate",
            "aligned_score_gain_mean",
            "raw_score_auc",
            "aligned_score_auc",
            "full_method_claim_ready",
            "supports_paper_claim",
        ],
    )
    write_csv(standard_metrics_path, standard_rows, ["metric_name", "metric_value", "metric_source", "metric_status", "supports_paper_claim"])
    write_csv(
        quality_metrics_path,
        quality_rows,
        ["quality_metric_name", "quality_metric_value", "quality_metric_source", "metric_status", "supports_paper_claim"],
    )
    write_csv(roc_path, roc_rows, ["roc_threshold", "true_positive_rate", "false_positive_rate", "supports_paper_claim"])
    write_csv(det_path, det_rows, ["det_threshold", "det_false_positive_rate", "det_false_negative_rate", "supports_paper_claim"])
    write_csv(
        distribution_path,
        distribution_rows,
        ["sample_role", "score_distribution_bin", "score_count", "score_min", "score_max", "score_mean", "supports_paper_claim"],
    )
    degeneracy_path.write_text(stable_json_text(threshold_report), encoding="utf-8")
    write_csv(
        fpr_audit_path,
        fpr_rows,
        [
            "operating_point_id",
            "target_fpr",
            "decision_scope",
            "statistical_boundary",
            "observed_fpr",
            "fpr_exceeds_target",
            "governs_fixed_fpr",
            "threshold_degenerate",
            "supports_paper_claim",
        ],
    )

    output_paths = tuple(
        relative_or_absolute(path, root_path)
        for path in (
            thresholds_path,
            operating_points_path,
            standard_metrics_path,
            quality_metrics_path,
            roc_path,
            det_path,
            distribution_path,
            degeneracy_path,
            fpr_audit_path,
            manifest_path,
        )
    )
    summary = {
        "operating_point_metrics": metrics,
        "threshold_report": threshold_report,
        "rescue_fpr_audit": fpr_rows,
        "aligned_rescoring_quality": aligned_rescoring_output_metadata(aligned_quality),
    }
    input_paths = [
        relative_or_absolute(resolved_records_path, root_path),
        relative_or_absolute(resolved_audit_path, root_path),
    ]
    rebuild_command = "python scripts/write_threshold_calibration_outputs.py"
    if resolved_aligned_package_path and resolved_aligned_package_path.exists():
        aligned_package_text = relative_or_absolute(resolved_aligned_package_path, root_path)
        input_paths.append(aligned_package_text)
        rebuild_command = (
            "python scripts/write_threshold_calibration_outputs.py "
            f"--aligned-rescoring-package-path {aligned_package_text}"
        )
    manifest = build_artifact_manifest(
        artifact_id="threshold_calibration_manifest",
        artifact_type="local_manifest",
        input_paths=tuple(input_paths),
        output_paths=output_paths,
        config={
            **config.to_dict(),
            "summary_digest": build_stable_digest(summary),
            "threshold_digest": build_stable_digest(threshold.to_dict()),
        },
        code_version=resolve_code_version(root_path),
        rebuild_command=rebuild_command,
        metadata={
            "construction_unit_name": CONSTRUCTION_UNIT_NAME,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "protocol_decision": "pass" if metrics["raw_content_clean_fpr"] <= target_fpr else "fail",
            "full_method_claim_ready": False,
            "supports_paper_claim": False,
            **aligned_rescoring_output_metadata(aligned_quality),
        },
    ).to_dict()
    manifest_path.write_text(stable_json_text(manifest), encoding="utf-8")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器。"""
    parser = argparse.ArgumentParser(description="写出 fixed-FPR 阈值校准与常规指标产物。")
    parser.add_argument("--root", default=".", help="仓库根目录。")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="输出目录, 必须位于 outputs/ 下。")
    parser.add_argument("--rescue-records-path", default=str(DEFAULT_RESCUE_RECORDS_PATH), help="几何 rescue 记录路径。")
    parser.add_argument("--rescue-audit-path", default=str(DEFAULT_RESCUE_AUDIT_PATH), help="几何 rescue 审计路径。")
    parser.add_argument("--target-fpr", type=float, default=DEFAULT_TARGET_FPR, help="目标 false positive rate。")
    parser.add_argument(
        "--aligned-rescoring-package-path",
        default=None,
        help="真实 aligned rescoring 结果包路径; 未传入时自动读取 outputs 中最新包。",
    )
    return parser


def main() -> None:
    """命令行入口。"""
    args = build_parser().parse_args()
    manifest = write_threshold_calibration_outputs(
        root=args.root,
        output_dir=args.output_dir,
        rescue_records_path=args.rescue_records_path,
        rescue_audit_path=args.rescue_audit_path,
        target_fpr=args.target_fpr,
        aligned_rescoring_package_path=args.aligned_rescoring_package_path,
    )
    print(stable_json_text(manifest), end="")


if __name__ == "__main__":
    main()
