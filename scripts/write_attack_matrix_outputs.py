"""从论文级仅图像检测记录重建真实攻击矩阵产物。"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.artifacts.artifact_manifest import build_artifact_manifest
from experiments.protocol.attacks import AttackConfig, attack_config_digest, default_attack_configs
from experiments.protocol.calibration import binomial_rate_upper_confidence_bound
from experiments.protocol.paper_run_config import build_paper_run_config
from experiments.runtime.image_metrics import compute_image_quality_metrics, measured_score_retention
from experiments.runtime.repository_environment import file_digest
from main.core.digest import build_stable_digest

CONSTRUCTION_UNIT_NAME = "attack_matrix"
DEFAULT_OUTPUT_DIR = Path("outputs/attack_matrix")
DEFAULT_IMAGE_ATTACK_EVIDENCE_RECORDS_PATH = Path("outputs/image_attack_evidence/formal_attack_detection_records.jsonl")
FORMAL_METRIC_STATUS = "measured_real_attacked_image_image_only_detection"


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


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: tuple[str, ...]) -> None:
    """按稳定列顺序写出 CSV。"""

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def resolve_code_version(root_path: Path) -> str:
    """读取 Git 提交标识, 工作区有变更时附加 dirty 标记。"""

    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=root_path,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--short"],
            cwd=root_path,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except Exception:
        return "git_version_unavailable"
    return f"{commit}-dirty" if dirty else commit


def resolve_path(root_path: Path, value: str | Path) -> Path:
    """把输入解析为绝对路径。"""

    path = Path(value)
    return path.resolve() if path.is_absolute() else (root_path / path).resolve()


def relative_or_absolute(path: Path, root_path: Path) -> str:
    """优先返回相对仓库根目录的路径。"""

    try:
        return path.resolve().relative_to(root_path).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def ensure_output_path(root_path: Path, value: str | Path) -> Path:
    """确保持久化输出位于 outputs 目录。"""

    path = resolve_path(root_path, value)
    outputs_root = (root_path / "outputs").resolve()
    if path != outputs_root and outputs_root not in path.parents:
        raise ValueError("攻击矩阵输出必须位于 outputs/ 下")
    return path


def _record_key(record: dict[str, Any]) -> tuple[str, str, str]:
    """返回同一运行、Prompt 与样本角色的配对键。"""

    return (
        str(record.get("run_id", "")),
        str(record.get("prompt_id", "")),
        str(record.get("sample_role", "")),
    )


def _ratio(after: Any, before: Any) -> float | None:
    """计算统一真实分数保持率, 缺失值返回空值。"""

    if not isinstance(after, (int, float)) or not isinstance(before, (int, float)):
        return None
    source = float(before)
    evaluated = float(after)
    if not math.isfinite(source) or not math.isfinite(evaluated):
        return None
    return measured_score_retention(source, evaluated)


def _mean(values: Iterable[Any]) -> float | None:
    """计算有限数值均值。"""

    resolved = [float(value) for value in values if isinstance(value, (int, float)) and math.isfinite(float(value))]
    return None if not resolved else sum(resolved) / len(resolved)


def _rate(records: Iterable[dict[str, Any]], field_name: str) -> float:
    """计算布尔字段比例。"""

    rows = tuple(records)
    return 0.0 if not rows else sum(bool(row.get(field_name)) for row in rows) / len(rows)


def _verify_image_record(root_path: Path, record: dict[str, Any]) -> tuple[Path, Path]:
    """验证真实攻击记录中的源图像与攻击图像证据链。"""

    required_fields = (
        "run_id",
        "prompt_id",
        "attack_id",
        "attack_family",
        "attack_name",
        "resource_profile",
        "source_image_path",
        "source_image_digest",
        "attacked_image_path",
        "attacked_image_digest",
        "content_score",
        "formal_evidence_positive",
    )
    missing = [field for field in required_fields if record.get(field) in {None, ""}]
    if missing:
        raise ValueError(f"真实攻击记录缺少字段: {','.join(missing)}")
    if record.get("split") != "test":
        raise ValueError("攻击矩阵正式记录必须来自 test split")
    if record.get("sample_role") not in {"positive_source", "clean_negative"}:
        raise ValueError("攻击矩阵正式记录只接受 positive_source 或 clean_negative")
    if not bool(record.get("attack_performed")):
        raise ValueError("攻击矩阵不得接收未执行攻击的记录")
    if record.get("formal_metric_status") != "measured_image_only_detection":
        raise ValueError("攻击记录必须已经应用 calibration split 冻结的完整检测协议")
    metadata = record.get("metadata", {})
    if not (
        metadata.get("detector_input_access_mode") == "image_key_public_model_only"
        and bool(metadata.get("blind_image_detector"))
        and not bool(metadata.get("generation_latent_trace_required", True))
    ):
        raise ValueError("攻击记录不满足仅图像盲检边界")
    source_path = resolve_path(root_path, str(record["source_image_path"]))
    attacked_path = resolve_path(root_path, str(record["attacked_image_path"]))
    if not source_path.is_file() or not attacked_path.is_file():
        raise FileNotFoundError("真实攻击记录绑定的图像文件不存在")
    if file_digest(source_path) != str(record["source_image_digest"]):
        raise ValueError("source_image_digest 与源图像不一致")
    if file_digest(attacked_path) != str(record["attacked_image_digest"]):
        raise ValueError("attacked_image_digest 与攻击图像不一致")
    return source_path, attacked_path


def build_measured_attack_records(
    root_path: Path,
    detection_records: Iterable[dict[str, Any]],
    attack_configs: tuple[AttackConfig, ...],
    supports_paper_claim: bool,
) -> tuple[dict[str, Any], ...]:
    """把真实图像盲检记录转换为带配对质量与分数保持率的正式攻击记录。"""

    all_records = tuple(detection_records)
    source_records = {
        _record_key(record): record
        for record in all_records
        if record.get("split") == "test" and not record.get("attack_id")
    }
    config_by_id = {config.attack_id: config for config in attack_configs}
    measured: list[dict[str, Any]] = []
    for record in all_records:
        if record.get("split") != "test" or not record.get("attack_id"):
            continue
        attack_id = str(record["attack_id"])
        config = config_by_id.get(attack_id)
        if config is None:
            raise ValueError(f"检测记录包含未登记攻击: {attack_id}")
        source_record = source_records.get(_record_key(record))
        if source_record is None:
            raise ValueError(f"攻击记录缺少同运行同角色的未攻击检测记录: {attack_id}")
        source_path, attacked_path = _verify_image_record(root_path, record)
        from PIL import Image

        with Image.open(source_path) as source_image, Image.open(attacked_path) as attacked_image:
            quality = compute_image_quality_metrics(source_image, attacked_image)
        record_digest = build_stable_digest(
            {
                "detector_digest": record.get("detector_digest"),
                "attack_config_digest": attack_config_digest(config),
                "source_image_digest": record["source_image_digest"],
                "attacked_image_digest": record["attacked_image_digest"],
                "frozen_threshold_digest": record.get("frozen_threshold_digest"),
            }
        )
        measured.append(
            {
                **record,
                "attack_record_id": f"attack_{record_digest[:24]}",
                "attack_record_digest": record_digest,
                "source_record_id": source_record.get("detector_digest", ""),
                "attack_strength": config.attack_strength,
                "requires_gpu": config.requires_gpu,
                "attack_config_digest": attack_config_digest(config),
                "attacked_image_available": True,
                "raw_content_score_before": float(source_record["content_score"]),
                "raw_content_score_after": float(record["content_score"]),
                "aligned_content_score_before": source_record.get("aligned_content_score"),
                "aligned_content_score_after": record.get("aligned_content_score"),
                "lf_score_retention": _ratio(record.get("lf_score"), source_record.get("lf_score")),
                "tail_score_retention": _ratio(
                    record.get("tail_robust_score"),
                    source_record.get("tail_robust_score"),
                ),
                "score_retention": _ratio(record.get("content_score"), source_record.get("content_score")),
                "quality_score": float(quality["ssim"]),
                "quality_ssim": float(quality["ssim"]),
                "quality_psnr": quality["psnr"],
                "quality_mse": float(quality["mse"]),
                "evidence_decision": bool(record["formal_evidence_positive"]),
                "metric_status": FORMAL_METRIC_STATUS,
                "unsupported_reason": "",
                "supports_paper_claim": supports_paper_claim,
            }
        )
    return tuple(measured)


def _group_records(records: Iterable[dict[str, Any]]) -> dict[tuple[str, str, str], list[dict[str, Any]]]:
    """按攻击族、名称与资源档位分组。"""

    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(str(record["attack_family"]), str(record["attack_name"]), str(record["resource_profile"]))].append(record)
    return grouped


def build_family_metrics(
    records: Iterable[dict[str, Any]],
    target_fpr: float,
    supports_paper_claim: bool,
) -> list[dict[str, Any]]:
    """从真实攻击图像检测记录构造攻击级 TPR、FPR、质量和保持率。"""

    rows: list[dict[str, Any]] = []
    for (family, name, profile), group in sorted(_group_records(records).items()):
        positives = [row for row in group if row["sample_role"] == "positive_source"]
        negatives = [row for row in group if row["sample_role"] == "clean_negative"]
        true_positive_count = sum(bool(row["formal_evidence_positive"]) for row in positives)
        false_positive_count = sum(bool(row["formal_evidence_positive"]) for row in negatives)
        false_positive_upper = binomial_rate_upper_confidence_bound(false_positive_count, len(negatives), 0.95)
        rows.append(
            {
                "attack_family": family,
                "attack_name": name,
                "resource_profile": profile,
                "metric_status": FORMAL_METRIC_STATUS,
                "attack_record_count": len(group),
                "supported_record_count": len(group) if supports_paper_claim else 0,
                "unsupported_record_count": 0 if supports_paper_claim else len(group),
                "positive_count": len(positives),
                "negative_count": len(negatives),
                "true_positive_rate": true_positive_count / len(positives),
                "false_positive_rate": false_positive_count / len(negatives),
                "clean_false_positive_rate": false_positive_count / len(negatives),
                "attacked_false_positive_rate": false_positive_count / len(negatives),
                "false_positive_rate_upper_95": false_positive_upper,
                "target_fpr": target_fpr,
                "fixed_fpr_upper_bound_ready": false_positive_upper <= target_fpr,
                "quality_score_mean": _mean(row.get("quality_score") for row in positives),
                "quality_ssim_mean": _mean(row.get("quality_ssim") for row in positives),
                "quality_psnr_mean": _mean(row.get("quality_psnr") for row in positives),
                "attacked_positive_source_to_attacked_ssim_mean": _mean(
                    row.get("quality_ssim") for row in positives
                ),
                "score_retention_mean": _mean(row.get("score_retention") for row in positives),
                "lf_score_retention_mean": _mean(row.get("lf_score_retention") for row in positives),
                "tail_score_retention_mean": _mean(row.get("tail_score_retention") for row in positives),
                "geometry_reliable_rate": _rate(group, "geometry_reliable"),
                "rescue_rate": _rate(group, "formal_rescue_applied"),
                "supports_paper_claim": supports_paper_claim,
            }
        )
    return rows


def build_strength_rows(family_rows: Iterable[dict[str, Any]], config_by_key: dict[tuple[str, str, str], AttackConfig]) -> list[dict[str, Any]]:
    """为真实攻击聚合行补充协议强度。"""

    rows = []
    for row in family_rows:
        key = (str(row["attack_family"]), str(row["attack_name"]), str(row["resource_profile"]))
        rows.append({**row, "attack_strength": config_by_key[key].attack_strength})
    return rows


def build_retention_rows(strength_rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """提取同一检测器内部的分数稳定性诊断表。"""

    fields = (
        "attack_family",
        "attack_name",
        "attack_strength",
        "resource_profile",
        "metric_status",
        "attack_record_count",
        "supported_record_count",
        "score_retention_mean",
        "lf_score_retention_mean",
        "tail_score_retention_mean",
        "supports_paper_claim",
    )
    return [
        {
            **{field: row[field] for field in fields},
            "metric_status": "diagnostic_method_internal_score_stability",
            "supports_paper_claim": False,
        }
        for row in strength_rows
    ]


def build_rescue_rows(strength_rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """提取真实注意力几何救回统计表。"""

    fields = (
        "attack_family",
        "attack_name",
        "attack_strength",
        "resource_profile",
        "metric_status",
        "attack_record_count",
        "supported_record_count",
        "geometry_reliable_rate",
        "rescue_rate",
        "supports_paper_claim",
    )
    return [{field: row[field] for field in fields} for row in strength_rows]


def build_attack_coverage(
    records: Iterable[dict[str, Any]],
    attack_configs: tuple[AttackConfig, ...],
    expected_role_count: int,
) -> dict[str, Any]:
    """核验全部正式攻击配置和两个论文角色均有真实记录。"""

    rows = tuple(records)
    expected_ids = {
        config.attack_id
        for config in attack_configs
        if config.enabled and config.resource_profile in {"full_main", "full_extra"}
    }
    actual_ids = {str(row["attack_id"]) for row in rows}
    role_counts = {
        f"{attack_id}|{sample_role}": sum(
            str(row["attack_id"]) == attack_id and row["sample_role"] == sample_role
            for row in rows
        )
        for attack_id in sorted(expected_ids)
        for sample_role in ("positive_source", "clean_negative")
    }
    if expected_role_count <= 0:
        raise ValueError("attack_prompt_count 必须为正整数")
    ready = actual_ids == expected_ids and all(count == expected_role_count for count in role_counts.values())
    return {
        "expected_attack_ids": sorted(expected_ids),
        "actual_attack_ids": sorted(actual_ids),
        "missing_attack_ids": sorted(expected_ids - actual_ids),
        "unexpected_attack_ids": sorted(actual_ids - expected_ids),
        "attack_split_role_counts": role_counts,
        "expected_attack_split_role_count": expected_role_count,
        "attack_record_coverage_ready": ready,
    }


def build_registry_rows(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """构造只登记真实图像文件与摘要的攻击图像注册表。"""

    fields = (
        "attack_record_id",
        "run_id",
        "prompt_id",
        "split",
        "sample_role",
        "attack_id",
        "attack_family",
        "attack_name",
        "resource_profile",
        "source_image_path",
        "source_image_digest",
        "attacked_image_path",
        "attacked_image_digest",
        "attack_config_digest",
        "metric_status",
        "supports_paper_claim",
    )
    return [{field: record[field] for field in fields} for record in records]


def write_attack_matrix_outputs(
    root: str | Path = ".",
    paper_run_name: str | None = None,
    dataset_runtime_dir: str | Path | None = None,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    image_attack_evidence_records_path: str | Path = DEFAULT_IMAGE_ATTACK_EVIDENCE_RECORDS_PATH,
) -> dict[str, Any]:
    """从当前论文层级的真实数据集运行结果写出攻击矩阵。"""

    root_path = Path(root).resolve()
    paper_run = build_paper_run_config(root_path)
    resolved_run_name = paper_run_name or paper_run.run_name
    runtime_dir = resolve_path(
        root_path,
        dataset_runtime_dir or Path("outputs/image_only_dataset_runtime") / resolved_run_name,
    )
    resolved_output_dir = ensure_output_path(root_path, output_dir)
    evidence_records_path = ensure_output_path(root_path, image_attack_evidence_records_path)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    evidence_records_path.parent.mkdir(parents=True, exist_ok=True)

    detection_records_path = runtime_dir / "image_only_detection_records.jsonl"
    runtime_summary_path = runtime_dir / "dataset_runtime_summary.json"
    frozen_protocol_path = runtime_dir / "frozen_evidence_protocol.json"
    runtime_manifest_path = runtime_dir / "manifest.local.json"
    for path in (detection_records_path, runtime_summary_path, frozen_protocol_path, runtime_manifest_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    runtime_summary = read_json(runtime_summary_path)
    frozen_protocol = read_json(frozen_protocol_path)
    runtime_manifest = read_json(runtime_manifest_path)
    if runtime_summary.get("paper_run_name") != resolved_run_name:
        raise ValueError("数据集运行结果与当前论文层级不一致")
    if runtime_summary.get("protocol_decision") != "pass":
        raise ValueError("攻击矩阵只接受完整通过的数据集运行结果")

    attack_configs = default_attack_configs()
    detection_records = read_jsonl(detection_records_path)
    preliminary_records = build_measured_attack_records(root_path, detection_records, attack_configs, False)
    coverage = build_attack_coverage(
        preliminary_records,
        attack_configs,
        int(runtime_summary.get("attack_prompt_count", 0)),
    )
    evidence_chain_ready = bool(preliminary_records) and all(
        record["attacked_image_available"] and record["source_image_digest"] and record["attacked_image_digest"]
        for record in preliminary_records
    )
    attack_metrics_ready = bool(
        coverage["attack_record_coverage_ready"]
        and evidence_chain_ready
        and runtime_summary.get("formal_attack_detection_ready")
        and runtime_summary.get("attacked_image_evidence_chain_ready")
    )
    supports_paper_claim = bool(runtime_summary.get("full_method_claim_ready") and attack_metrics_ready)
    attack_records = tuple(
        {**record, "supports_paper_claim": supports_paper_claim}
        for record in preliminary_records
    )

    target_fpr = float(frozen_protocol["target_fpr"])
    family_rows = build_family_metrics(attack_records, target_fpr, supports_paper_claim)
    config_by_key = {
        (config.attack_family, config.attack_name, config.resource_profile): config
        for config in attack_configs
        if config.enabled and config.resource_profile in {"full_main", "full_extra"}
    }
    strength_rows = build_strength_rows(family_rows, config_by_key)
    retention_rows = build_retention_rows(strength_rows)
    rescue_rows = build_rescue_rows(strength_rows)
    registry_rows = build_registry_rows(attack_records)

    records_path = resolved_output_dir / "attack_detection_records.jsonl"
    registry_path = resolved_output_dir / "attacked_image_registry.jsonl"
    attack_manifest_path = resolved_output_dir / "attack_manifest.json"
    family_metrics_path = resolved_output_dir / "attack_family_metrics.csv"
    strength_curve_path = resolved_output_dir / "attack_strength_curve.csv"
    retention_path = resolved_output_dir / "score_retention_by_attack.csv"
    rescue_path = resolved_output_dir / "rescue_by_attack.csv"
    manifest_path = resolved_output_dir / "manifest.local.json"

    record_text = "".join(json_line(record) for record in attack_records)
    records_path.write_text(record_text, encoding="utf-8")
    evidence_records_path.write_text(record_text, encoding="utf-8")
    registry_path.write_text("".join(json_line(row) for row in registry_rows), encoding="utf-8")

    evaluation_boundary = {
        "calibrated_content_threshold": frozen_protocol["content_threshold"],
        "geometry_score_threshold": frozen_protocol["geometry_score_threshold"],
        "rescue_margin_low": frozen_protocol["rescue_margin_low"],
        "target_fpr": target_fpr,
        "threshold_digest": frozen_protocol["threshold_digest"],
        "fixed_fpr_control_scope": "calibration_clean_negative",
        "fixed_fpr_denominator_role": "clean_negative_only",
        "attacked_negative_boundary_role": "attack_robustness_diagnostic_not_fpr_denominator",
        "attacked_negative_governs_fixed_fpr": False,
    }
    required_gpu_ids = {
        config.attack_id
        for config in attack_configs
        if config.enabled and config.requires_gpu and config.resource_profile == "full_extra"
    }
    measured_gpu_ids = {str(record["attack_id"]) for record in attack_records if record["requires_gpu"]}
    attack_manifest = {
        "construction_unit_name": CONSTRUCTION_UNIT_NAME,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "paper_run_name": resolved_run_name,
        "source_runtime_manifest_id": runtime_manifest.get("artifact_id", ""),
        "input_records_path": relative_or_absolute(detection_records_path, root_path),
        "image_attack_evidence_records_path": relative_or_absolute(evidence_records_path, root_path),
        "attack_config_count": len(config_by_key),
        "attack_family_count": len({config.attack_family for config in config_by_key.values()}),
        "attack_record_count": len(attack_records),
        "performed_attack_record_count": len(attack_records),
        "formal_real_attack_record_count": len(attack_records),
        "formal_image_attack_record_count": len(attack_records),
        "real_attacked_image_count": len(attack_records),
        "real_attacked_image_closed_loop_ready": evidence_chain_ready,
        "formal_attack_detection_ready": attack_metrics_ready,
        "attack_metrics_ready": attack_metrics_ready,
        "required_real_gpu_attack_count": len(required_gpu_ids),
        "measured_real_gpu_attack_count": len(measured_gpu_ids),
        "real_gpu_attack_validation_ready": measured_gpu_ids == required_gpu_ids,
        "gpu_attack_real_measurement_missing_count": len(required_gpu_ids - measured_gpu_ids),
        "resource_profiles": sorted({config.resource_profile for config in config_by_key.values()}),
        "conventional_attack_names": sorted({config.attack_name for config in config_by_key.values() if not config.requires_gpu}),
        "regeneration_attack_names": sorted({config.attack_name for config in config_by_key.values() if config.requires_gpu}),
        "evaluation_boundary": evaluation_boundary,
        **coverage,
        "full_method_claim_ready": supports_paper_claim,
        "supports_paper_claim": supports_paper_claim,
    }
    attack_manifest_path.write_text(stable_json_text(attack_manifest), encoding="utf-8")

    family_fields = tuple(family_rows[0])
    strength_fields = tuple(strength_rows[0])
    retention_fields = tuple(retention_rows[0])
    rescue_fields = tuple(rescue_rows[0])
    write_csv(family_metrics_path, family_rows, family_fields)
    write_csv(strength_curve_path, strength_rows, strength_fields)
    write_csv(retention_path, retention_rows, retention_fields)
    write_csv(rescue_path, rescue_rows, rescue_fields)

    output_paths = tuple(
        relative_or_absolute(path, root_path)
        for path in (
            attack_manifest_path,
            registry_path,
            records_path,
            family_metrics_path,
            strength_curve_path,
            retention_path,
            rescue_path,
            evidence_records_path,
            manifest_path,
        )
    )
    manifest = build_artifact_manifest(
        artifact_id=f"{resolved_run_name}_attack_matrix_manifest",
        artifact_type="local_manifest",
        input_paths=tuple(
            relative_or_absolute(path, root_path)
            for path in (detection_records_path, runtime_summary_path, frozen_protocol_path, runtime_manifest_path)
        ),
        output_paths=output_paths,
        config={
            "paper_run_name": resolved_run_name,
            "evaluation_boundary": evaluation_boundary,
            "attack_config_digest": build_stable_digest([config.to_dict() for config in config_by_key.values()]),
            "attack_record_digest": build_stable_digest(attack_records),
        },
        code_version=resolve_code_version(root_path),
        rebuild_command="python scripts/write_attack_matrix_outputs.py",
        metadata={
            "construction_unit_name": CONSTRUCTION_UNIT_NAME,
            "protocol_decision": "pass" if attack_metrics_ready else "fail",
            "full_method_claim_ready": supports_paper_claim,
            "supports_paper_claim": supports_paper_claim,
        },
    ).to_dict()
    manifest_path.write_text(stable_json_text(manifest), encoding="utf-8")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器。"""

    parser = argparse.ArgumentParser(description="从真实仅图像检测记录重建攻击矩阵。")
    parser.add_argument("--root", default=".", help="仓库根目录。")
    parser.add_argument("--paper-run-name", default=None, help="论文运行层级, 默认读取统一环境配置。")
    parser.add_argument("--dataset-runtime-dir", default=None, help="真实数据集运行目录。")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="攻击矩阵输出目录。")
    parser.add_argument(
        "--image-attack-evidence-records-path",
        default=str(DEFAULT_IMAGE_ATTACK_EVIDENCE_RECORDS_PATH),
        help="正式真实图像攻击记录合并路径。",
    )
    return parser


def main() -> None:
    """命令行入口。"""

    args = build_parser().parse_args()
    manifest = write_attack_matrix_outputs(
        root=args.root,
        paper_run_name=args.paper_run_name,
        dataset_runtime_dir=args.dataset_runtime_dir,
        output_dir=args.output_dir,
        image_attack_evidence_records_path=args.image_attack_evidence_records_path,
    )
    print(stable_json_text(manifest), end="")


if __name__ == "__main__":
    main()
