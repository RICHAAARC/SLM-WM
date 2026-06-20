"""写出同阈值几何恢复重判产物。"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
from typing import Any
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from main.analysis.artifact_manifest import build_artifact_manifest
from main.core.digest import build_stable_digest
from main.methods.detection import (
    RESCUE_ABLATION_MODES,
    SameThresholdRescueConfig,
    decide_same_threshold_geometric_rescue,
    effective_geometry_reliability,
)
from main.methods.geometry.recovery import estimate_aligned_content_score

CONSTRUCTION_UNIT_NAME = "geometric_rescue"
DEFAULT_OUTPUT_DIR = Path("outputs/geometric_rescue")
DEFAULT_CONTENT_RECORDS_PATH = Path("outputs/content_carriers/content_detection_records.jsonl")
DEFAULT_CONTENT_THRESHOLD = 0.75
DEFAULT_RESCUE_MARGIN_LOW = -0.05
DEFAULT_MAX_CONTENT_RECORDS = 96
ATTENTION_INJECTION_PACKAGE_PATTERN = "attention_latent_injection_package_*.zip"


def stable_json_text(value: Any) -> str:
    """把 JSON 兼容对象转为稳定文本。"""
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def json_line(value: Any) -> str:
    """把 JSON 兼容对象转为 JSONL 单行文本。"""
    return json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n"


def read_jsonl_text(text: str) -> tuple[dict[str, Any], ...]:
    """读取 JSONL 文本。"""
    return tuple(json.loads(line) for line in text.splitlines() if line.strip())


def read_jsonl(path: Path) -> tuple[dict[str, Any], ...]:
    """读取 JSONL 文件。"""
    return read_jsonl_text(path.read_text(encoding="utf-8"))


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
        raise ValueError("几何恢复输出目录必须位于 outputs/ 下") from exc
    return resolved_output_dir


def relative_or_absolute(path: Path, root_path: Path) -> str:
    """将路径尽量转为相对仓库根目录的字符串。"""
    try:
        return path.resolve().relative_to(root_path).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def latest_attention_injection_package(root_path: Path) -> Path:
    """查找 outputs 中最新的真实 attention latent injection 包。"""
    candidates = sorted((root_path / "outputs").glob(ATTENTION_INJECTION_PACKAGE_PATTERN), key=lambda path: path.name)
    if not candidates:
        raise FileNotFoundError("attention_latent_injection_package_missing")
    return candidates[-1]


def resolve_attention_injection_package(root_path: Path, package_path: str | Path | None) -> Path:
    """解析真实 attention latent injection 包路径。"""
    if package_path:
        candidate = Path(package_path)
        return candidate.resolve() if candidate.is_absolute() else (root_path / candidate).resolve()
    return latest_attention_injection_package(root_path).resolve()


def load_attention_injection_package(package_path: Path) -> dict[str, Any]:
    """读取真实 attention latent injection 包中的注入、几何与 carrier 证据。"""
    with ZipFile(package_path) as archive:
        result = json.loads(
            archive.read("outputs/attention_latent_injection/attention_latent_injection_result.json").decode("utf-8")
        )
        method_summary = json.loads(
            archive.read("outputs/attention_latent_update/attention_update_summary.json").decode("utf-8")
        )
        carrier_records = read_jsonl_text(
            archive.read("outputs/attention_latent_update/attention_carrier_records.jsonl").decode("utf-8")
        )
        nested_names = [
            name
            for name in archive.namelist()
            if name.startswith("outputs/attention_latent_injection/input_packages/")
            and name.endswith(".zip")
        ]
        if not nested_names:
            raise FileNotFoundError("attention_geometry_input_package_missing")
        nested_bytes = archive.read(sorted(nested_names)[-1])
    with ZipFile(io.BytesIO(nested_bytes)) as nested_archive:
        geometry_records = read_jsonl_text(
            nested_archive.read("outputs/attention_geometry/geometry_evidence_records.jsonl").decode("utf-8")
        )
        geometry_summary = json.loads(
            nested_archive.read("outputs/attention_geometry/geometry_evidence_summary.json").decode("utf-8")
        )
    return {
        "result": result,
        "method_summary": method_summary,
        "carrier_records": carrier_records,
        "geometry_records": geometry_records,
        "geometry_summary": geometry_summary,
        "nested_geometry_package_name": sorted(nested_names)[-1],
    }


def carrier_by_prompt(records: tuple[dict[str, Any], ...]) -> dict[str, dict[str, Any]]:
    """从 active carrier 中建立 prompt_id 到 carrier 的映射。"""
    mapping: dict[str, dict[str, Any]] = {}
    for record in records:
        if record.get("fallback_mode") != "active_update":
            continue
        prompt_id = str(record.get("metadata", {}).get("prompt_id", ""))
        if prompt_id and prompt_id not in mapping:
            mapping[prompt_id] = record
    return mapping


def select_geometry_record(
    content_record: dict[str, Any],
    prompt_carriers: dict[str, dict[str, Any]],
    geometry_by_graph: dict[str, dict[str, Any]],
    fallback_geometries: tuple[dict[str, Any], ...],
) -> dict[str, Any]:
    """为内容记录选择可审计几何证据。"""
    carrier = prompt_carriers.get(str(content_record.get("prompt_id", "")))
    if carrier is not None:
        geometry = geometry_by_graph.get(str(carrier.get("attention_graph_id", "")))
        if geometry is not None:
            return geometry
    index = int(build_stable_digest({"content_detection_record_id": content_record["content_detection_record_id"]})[:8], 16)
    return fallback_geometries[index % len(fallback_geometries)]


def infer_fail_reason(raw_content_margin: float, geometry_reliable: bool, config: SameThresholdRescueConfig) -> str:
    """根据原始内容边界和几何可靠性给出受限失败原因。"""
    if raw_content_margin >= 0.0:
        return "content_positive"
    if raw_content_margin >= config.rescue_margin_low:
        return "geometry_suspected" if geometry_reliable else "low_confidence"
    return "score_below_window"


def aligned_score_for_mode(
    content_record: dict[str, Any],
    geometry_record: dict[str, Any],
    config: SameThresholdRescueConfig,
    rescue_ablation_mode: str,
) -> float:
    """按消融模式估计对齐后内容分数。"""
    raw_score = float(content_record["content_score"])
    if rescue_ablation_mode in {"no_rescue", "no_attention_anchor", "geo_direct_positive_audit"}:
        return raw_score
    if not effective_geometry_reliability(geometry_record, config, rescue_ablation_mode):
        return raw_score
    return estimate_aligned_content_score(
        raw_content_score=raw_score,
        content_threshold=config.content_threshold,
        geometry_evidence=geometry_record,
        sample_role=str(content_record.get("metadata", {}).get("sample_role", "unknown")),
    )


def build_aligned_detection_records(
    content_records: tuple[dict[str, Any], ...],
    package_inputs: dict[str, Any],
    config: SameThresholdRescueConfig,
) -> list[dict[str, Any]]:
    """构造同阈值几何恢复重判记录。"""
    geometry_records = tuple(package_inputs["geometry_records"])
    geometry_by_graph = {record["attention_graph_id"]: record for record in geometry_records}
    prompt_carriers = carrier_by_prompt(tuple(package_inputs["carrier_records"]))
    rows: list[dict[str, Any]] = []
    for content_record in content_records:
        geometry_record = select_geometry_record(content_record, prompt_carriers, geometry_by_graph, geometry_records)
        raw_margin = float(content_record["content_score"]) - config.content_threshold
        for rescue_ablation_mode in RESCUE_ABLATION_MODES:
            geometry_reliable = effective_geometry_reliability(geometry_record, config, rescue_ablation_mode)
            aligned_score = aligned_score_for_mode(content_record, geometry_record, config, rescue_ablation_mode)
            decision = decide_same_threshold_geometric_rescue(
                content_record=content_record,
                geometry_record=geometry_record,
                aligned_content_score=aligned_score,
                config=config,
                fail_reason=infer_fail_reason(raw_margin, geometry_reliable, config),
                rescue_ablation_mode=rescue_ablation_mode,
                metadata={
                    "attention_geometry_ready": package_inputs["method_summary"].get("attention_geometry_ready", False),
                    "image_quality_metrics_ready": package_inputs["result"].get("image_quality_metrics_ready", False),
                    "nested_geometry_package_name": package_inputs["nested_geometry_package_name"],
                    "supports_paper_claim": False,
                },
            )
            rows.append(decision.to_dict())
    return rows


def rate(rows: list[dict[str, Any]], sample_role: str, field_name: str) -> float:
    """计算指定样本角色中某个布尔字段的触发率。"""
    selected = [row for row in rows if row["sample_role"] == sample_role]
    if not selected:
        return 0.0
    return sum(1 for row in selected if bool(row[field_name])) / len(selected)


def mean(values: list[float]) -> float:
    """计算均值, 空输入返回 0。"""
    return sum(values) / len(values) if values else 0.0


def build_metrics_rows(aligned_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """构造 rescue 指标摘要行。"""
    rows: list[dict[str, Any]] = []
    for rescue_ablation_mode in RESCUE_ABLATION_MODES:
        mode_rows = [row for row in aligned_records if row["rescue_ablation_mode"] == rescue_ablation_mode]
        rows.append(
            {
                "rescue_ablation_mode": rescue_ablation_mode,
                "aligned_detection_record_count": len(mode_rows),
                "positive_source_count": sum(1 for row in mode_rows if row["sample_role"] == "positive_source"),
                "clean_negative_count": sum(1 for row in mode_rows if row["sample_role"] == "clean_negative"),
                "attacked_negative_count": sum(1 for row in mode_rows if row["sample_role"] == "attacked_negative"),
                "raw_content_positive_count": sum(1 for row in mode_rows if row["positive_by_content"]),
                "rescue_eligible_count": sum(1 for row in mode_rows if row["rescue_eligible"]),
                "rescue_applied_count": sum(1 for row in mode_rows if row["rescue_applied"]),
                "evidence_positive_count": sum(1 for row in mode_rows if row["evidence_decision"]),
                "raw_content_clean_fpr": rate(mode_rows, "clean_negative", "positive_by_content"),
                "evidence_clean_fpr": rate(mode_rows, "clean_negative", "evidence_decision"),
                "evidence_attacked_fpr": rate(mode_rows, "attacked_negative", "evidence_decision"),
                "geo_direct_positive_audit_rate": rate(mode_rows, "clean_negative", "geo_direct_positive_audit_decision"),
                "rescue_score_gain_mean": mean([float(row["rescue_score_gain"]) for row in mode_rows]),
                "full_method_claim_ready": False,
                "supports_paper_claim": False,
            }
        )
    return rows


def build_failed_subset_rows(aligned_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """构造内容失败子集摘要。"""
    rows: list[dict[str, Any]] = []
    for rescue_ablation_mode in RESCUE_ABLATION_MODES:
        for sample_role in ("positive_source", "clean_negative", "attacked_negative"):
            subset = [
                row
                for row in aligned_records
                if row["rescue_ablation_mode"] == rescue_ablation_mode
                and row["sample_role"] == sample_role
                and not row["positive_by_content"]
            ]
            rows.append(
                {
                    "rescue_ablation_mode": rescue_ablation_mode,
                    "sample_role": sample_role,
                    "raw_content_failed_count": len(subset),
                    "rescue_eligible_count": sum(1 for row in subset if row["rescue_eligible"]),
                    "rescue_applied_count": sum(1 for row in subset if row["rescue_applied"]),
                    "rescue_score_gain_mean": mean([float(row["rescue_score_gain"]) for row in subset]),
                    "supports_paper_claim": False,
                }
            )
    return rows


def build_audit_summary(
    package_path: Path,
    package_inputs: dict[str, Any],
    aligned_records: list[dict[str, Any]],
    metrics_rows: list[dict[str, Any]],
    config: SameThresholdRescueConfig,
    root_path: Path,
) -> dict[str, Any]:
    """构造几何恢复审计摘要。"""
    full_rows = [row for row in aligned_records if row["rescue_ablation_mode"] == "full_rescue"]
    return {
        "construction_unit_name": CONSTRUCTION_UNIT_NAME,
        "attention_latent_injection_package_path": relative_or_absolute(package_path, root_path),
        "attention_latent_injection_package_digest": file_digest(package_path),
        "attention_geometry_ready": bool(package_inputs["method_summary"].get("attention_geometry_ready", False)),
        "image_quality_metrics_ready": bool(package_inputs["result"].get("image_quality_metrics_ready", False)),
        "latent_update_count": int(package_inputs["result"].get("latent_update_count", 0)),
        "content_threshold": config.content_threshold,
        "rescue_margin_low": config.rescue_margin_low,
        "allowed_fail_reasons": list(config.allowed_fail_reasons),
        "aligned_detection_record_count": len(aligned_records),
        "full_rescue_record_count": len(full_rows),
        "full_rescue_applied_count": sum(1 for row in full_rows if row["rescue_applied"]),
        "direct_positive_decision_used": False,
        "geo_direct_positive_audit_formal_method": False,
        "raw_content_clean_fpr": next(
            row["raw_content_clean_fpr"] for row in metrics_rows if row["rescue_ablation_mode"] == "full_rescue"
        ),
        "evidence_clean_fpr": next(
            row["evidence_clean_fpr"] for row in metrics_rows if row["rescue_ablation_mode"] == "full_rescue"
        ),
        "evidence_attacked_fpr": next(
            row["evidence_attacked_fpr"] for row in metrics_rows if row["rescue_ablation_mode"] == "full_rescue"
        ),
        "full_method_claim_ready": False,
        "protocol_decision": "pass" if aligned_records and package_inputs["result"].get("run_decision") == "pass" else "fail",
        "supports_paper_claim": False,
    }


def write_geometric_rescue_outputs(
    root: str | Path = ".",
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    content_records_path: str | Path = DEFAULT_CONTENT_RECORDS_PATH,
    attention_injection_package_path: str | Path | None = None,
    content_threshold: float = DEFAULT_CONTENT_THRESHOLD,
    rescue_margin_low: float = DEFAULT_RESCUE_MARGIN_LOW,
    max_content_records: int | None = DEFAULT_MAX_CONTENT_RECORDS,
) -> dict[str, Any]:
    """写出同阈值几何恢复重判产物。"""
    root_path = Path(root).resolve()
    resolved_output_dir = ensure_output_dir_under_outputs(root_path, Path(output_dir))
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    resolved_content_path = (
        Path(content_records_path).resolve()
        if Path(content_records_path).is_absolute()
        else (root_path / content_records_path).resolve()
    )
    package_path = resolve_attention_injection_package(root_path, attention_injection_package_path)
    config = SameThresholdRescueConfig(
        content_threshold=content_threshold,
        rescue_margin_low=rescue_margin_low,
    )
    content_records = read_jsonl(resolved_content_path)
    if max_content_records is not None:
        content_records = content_records[:max_content_records]
    package_inputs = load_attention_injection_package(package_path)
    aligned_records = build_aligned_detection_records(content_records, package_inputs, config)
    metrics_rows = build_metrics_rows(aligned_records)
    failed_subset_rows = build_failed_subset_rows(aligned_records)
    audit_summary = build_audit_summary(package_path, package_inputs, aligned_records, metrics_rows, config, root_path)

    records_path = resolved_output_dir / "aligned_detection_records.jsonl"
    metrics_path = resolved_output_dir / "rescue_metrics_summary.csv"
    failed_subset_path = resolved_output_dir / "content_failed_subset_summary.csv"
    audit_path = resolved_output_dir / "geometry_rescue_audit.json"
    manifest_path = resolved_output_dir / "manifest.local.json"

    records_path.write_text("".join(json_line(record) for record in aligned_records), encoding="utf-8")
    write_csv(
        metrics_path,
        metrics_rows,
        [
            "rescue_ablation_mode",
            "aligned_detection_record_count",
            "positive_source_count",
            "clean_negative_count",
            "attacked_negative_count",
            "raw_content_positive_count",
            "rescue_eligible_count",
            "rescue_applied_count",
            "evidence_positive_count",
            "raw_content_clean_fpr",
            "evidence_clean_fpr",
            "evidence_attacked_fpr",
            "geo_direct_positive_audit_rate",
            "rescue_score_gain_mean",
            "full_method_claim_ready",
            "supports_paper_claim",
        ],
    )
    write_csv(
        failed_subset_path,
        failed_subset_rows,
        [
            "rescue_ablation_mode",
            "sample_role",
            "raw_content_failed_count",
            "rescue_eligible_count",
            "rescue_applied_count",
            "rescue_score_gain_mean",
            "supports_paper_claim",
        ],
    )
    audit_path.write_text(stable_json_text(audit_summary), encoding="utf-8")

    output_paths = tuple(
        relative_or_absolute(path, root_path)
        for path in (records_path, metrics_path, failed_subset_path, audit_path, manifest_path)
    )
    manifest = build_artifact_manifest(
        artifact_id="geometric_rescue_manifest",
        artifact_type="local_manifest",
        input_paths=(
            relative_or_absolute(resolved_content_path, root_path),
            relative_or_absolute(package_path, root_path),
        ),
        output_paths=output_paths,
        config={
            **config.to_dict(),
            "max_content_records": max_content_records,
            "audit_digest": build_stable_digest(audit_summary),
        },
        code_version=resolve_code_version(root_path),
        rebuild_command=(
            "python scripts/write_geometric_rescue_outputs.py "
            f"--attention-injection-package-path {relative_or_absolute(package_path, root_path)}"
        ),
        metadata={
            "construction_unit_name": CONSTRUCTION_UNIT_NAME,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "protocol_decision": audit_summary["protocol_decision"],
            "supports_paper_claim": False,
        },
    ).to_dict()
    manifest_path.write_text(stable_json_text(manifest), encoding="utf-8")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器。"""
    parser = argparse.ArgumentParser(description="写出同阈值几何恢复重判产物。")
    parser.add_argument("--root", default=".", help="仓库根目录。")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="输出目录, 必须位于 outputs/ 下。")
    parser.add_argument("--content-records-path", default=str(DEFAULT_CONTENT_RECORDS_PATH), help="内容检测记录路径。")
    parser.add_argument("--attention-injection-package-path", default=None, help="真实 attention latent injection 包路径。")
    parser.add_argument("--content-threshold", type=float, default=DEFAULT_CONTENT_THRESHOLD, help="冻结内容阈值。")
    parser.add_argument("--rescue-margin-low", type=float, default=DEFAULT_RESCUE_MARGIN_LOW, help="rescue 窗口下界。")
    parser.add_argument("--max-content-records", type=int, default=DEFAULT_MAX_CONTENT_RECORDS, help="最多读取的内容记录数。")
    return parser


def main() -> None:
    """命令行入口。"""
    args = build_parser().parse_args()
    manifest = write_geometric_rescue_outputs(
        root=args.root,
        output_dir=args.output_dir,
        content_records_path=args.content_records_path,
        attention_injection_package_path=args.attention_injection_package_path,
        content_threshold=args.content_threshold,
        rescue_margin_low=args.rescue_margin_low,
        max_content_records=args.max_content_records,
    )
    print(stable_json_text(manifest), end="")


if __name__ == "__main__":
    main()
