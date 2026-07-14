"""从论文级仅图像检测记录重建真实攻击矩阵产物。"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper_experiments.runners.paper_claim_provenance import (
    require_exact9_randomization_aggregate_provenance,
)
from experiments.artifacts.artifact_manifest import build_artifact_manifest
from experiments.artifacts.attack_family_metrics import (
    ATTACK_FAMILY_METRIC_FIELDS,
    build_attack_family_metrics,
)
from experiments.protocol.attacks import (
    AttackConfig,
    attack_config_digest,
    build_attack_record_digest,
    build_attack_matrix_manifest_config,
    default_attack_configs,
)
from experiments.protocol.paper_run_config import build_paper_run_config
from experiments.runtime.image_metrics import compute_image_quality_metrics, measured_score_retention
from experiments.runtime.repository_environment import file_digest, resolve_code_version

CONSTRUCTION_UNIT_NAME = "attack_matrix"
DEFAULT_OUTPUT_ROOT = Path("outputs/attack_matrix")
DEFAULT_IMAGE_ATTACK_EVIDENCE_ROOT = Path("outputs/image_attack_evidence")
IMAGE_ATTACK_EVIDENCE_RECORDS_NAME = "formal_attack_detection_records.jsonl"
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


def _verify_image_record(root_path: Path, record: dict[str, Any]) -> tuple[Path, Path]:
    """验证真实攻击记录中的源图像与攻击图像证据链。"""

    required_fields = (
        "run_id",
        "prompt_id",
        "attack_id",
        "attack_family",
        "attack_name",
        "resource_profile",
        "attack_config_digest",
        "attack_parameters",
        "source_image_path",
        "source_image_digest",
        "attacked_image_path",
        "attacked_image_digest",
        "content_score",
        "formal_evidence_positive",
    )
    missing = [
        field
        for field in required_fields
        if record.get(field) is None or record.get(field) == ""
    ]
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
        expected_identity = {
            "attack_id": config.attack_id,
            "attack_family": config.attack_family,
            "attack_name": config.attack_name,
            "resource_profile": config.resource_profile,
            "attack_config_digest": attack_config_digest(config),
            "attack_parameters": config.attack_parameters,
        }
        actual_identity = {
            field_name: record.get(field_name)
            for field_name in expected_identity
        }
        if actual_identity != expected_identity:
            raise ValueError(f"检测记录攻击身份与正式配置不一致: {attack_id}")
        source_record = source_records.get(_record_key(record))
        if source_record is None:
            raise ValueError(f"攻击记录缺少同运行同角色的未攻击检测记录: {attack_id}")
        source_path, attacked_path = _verify_image_record(root_path, record)
        from PIL import Image

        with Image.open(source_path) as source_image, Image.open(attacked_path) as attacked_image:
            quality = compute_image_quality_metrics(source_image, attacked_image)
        record_digest = build_attack_record_digest(record)
        measured.append(
            {
                **record,
                "attack_record_id": f"attack_{record_digest[:24]}",
                "attack_record_digest": record_digest,
                "source_record_id": source_record.get("measurement_digest", ""),
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
        "attack_id",
        "attack_family",
        "attack_name",
        "attack_strength",
        "resource_profile",
        "attack_config_digest",
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
        "attack_id",
        "attack_family",
        "attack_name",
        "attack_strength",
        "resource_profile",
        "attack_config_digest",
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


def write_attack_matrix_outputs(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """在精确9重复原始记录重算 Writer 就绪前拒绝正式结论物化."""

    require_exact9_randomization_aggregate_provenance()


def build_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器。"""

    parser = argparse.ArgumentParser(description="从真实仅图像检测记录重建攻击矩阵。")
    parser.add_argument("--root", default=".", help="仓库根目录。")
    parser.add_argument("--paper-run-name", default=None, help="论文运行层级, 默认读取统一环境配置。")
    parser.add_argument("--dataset-runtime-dir", default=None, help="真实数据集运行目录。")
    parser.add_argument("--output-dir", default=None, help="攻击矩阵输出目录; 默认写入当前论文运行子目录。")
    parser.add_argument(
        "--image-attack-evidence-records-path",
        default=None,
        help="正式真实图像攻击记录合并路径; 默认写入当前论文运行子目录。",
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
