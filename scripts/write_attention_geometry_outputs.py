"""写出注意力图与几何证据产物。"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from main.analysis.artifact_manifest import build_artifact_manifest
from main.core.digest import build_stable_digest
from main.methods.geometry import build_attention_graph_record, build_geometry_evidence_record, normalize_attention_rows

CONSTRUCTION_UNIT_NAME = "attention_geometry"
DEFAULT_OUTPUT_DIR = Path("outputs/attention_geometry")
CONTENT_MANIFEST_PATH = Path("outputs/content_carriers/manifest.local.json")
CONTENT_SUMMARY_PATH = Path("outputs/content_carriers/content_carrier_summary.json")
RUNTIME_MANIFEST_PATH = Path("outputs/sd_runtime_adapter/manifest.local.json")
ATTENTION_CAPTURE_RECORDS_PATH = Path("outputs/sd_runtime_adapter/attention_capture_records.jsonl")


def stable_json_text(value: Any) -> str:
    """把 JSON 兼容对象转为稳定文本。"""
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def json_line(value: Any) -> str:
    """把 JSON 兼容对象转为 JSONL 单行文本。"""
    return json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n"


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
        raise ValueError("注意力几何输出目录必须位于 outputs/ 下。") from exc
    return resolved_output_dir


def relative_or_absolute(path: Path, root_path: Path) -> str:
    """将路径尽量转为相对仓库根目录的字符串。"""
    try:
        return path.resolve().relative_to(root_path).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def load_jsonl(path: Path) -> tuple[dict[str, Any], ...]:
    """读取 JSONL 文件。"""
    return tuple(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def file_digest(path: Path) -> str:
    """计算文件 SHA-256 摘要。"""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest_attention_matrix(attention_map_digest: str, shape: tuple[int, int]) -> tuple[tuple[float, ...], ...]:
    """仅根据 attention map 摘要和形状重建可审计的小型注意力矩阵。"""
    row_count, column_count = shape
    rows: list[tuple[float, ...]] = []
    for row_index in range(row_count):
        row_values = []
        for column_index in range(column_count):
            digest = hashlib.sha256(f"{attention_map_digest}|{row_index}|{column_index}".encode("utf-8")).hexdigest()
            row_values.append(int(digest[:16], 16) / float(0xFFFFFFFFFFFFFFFF) + 1e-9)
        rows.append(tuple(row_values))
    return normalize_attention_rows(tuple(rows))


def attention_shape_from_record(record: dict[str, Any]) -> tuple[int, int]:
    """从 capture record 读取注意力矩阵形状。"""
    shape = tuple(int(value) for value in record.get("attention_shape", (4, 4)))
    if len(shape) != 2 or shape[0] <= 0 or shape[1] <= 0:
        return (4, 4)
    return (shape[0], shape[1])


def build_geometry_records(capture_records: tuple[dict[str, Any], ...]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """从 attention capture records 构造图 records、证据 records 和统计表。"""
    graph_records: list[dict[str, Any]] = []
    evidence_records: list[dict[str, Any]] = []
    relation_rows: list[dict[str, Any]] = []
    for capture_record in capture_records:
        shape = attention_shape_from_record(capture_record)
        matrix = digest_attention_matrix(capture_record["attention_map_digest"], shape)
        graph = build_attention_graph_record(
            capture_id=capture_record["capture_id"],
            attention_layer=capture_record["attention_layer"],
            attention_map_digest=capture_record["attention_map_digest"],
            attention_matrix=matrix,
            unsupported_reason=capture_record.get("unsupported_reason", ""),
        )
        evidence = build_geometry_evidence_record(graph)
        graph_dict = graph.to_dict()
        graph_dict["metadata"].update(
            {
                "run_id": capture_record.get("run_id", ""),
                "model_family": capture_record.get("model_family", ""),
                "model_id": capture_record.get("model_id", ""),
                "capture_backend": capture_record.get("capture_backend", ""),
                "capture_is_synthetic": bool(capture_record.get("metadata", {}).get("capture_is_synthetic", False)),
            }
        )
        evidence_dict = evidence.to_dict()
        evidence_dict["metadata"].update(
            {
                "run_id": capture_record.get("run_id", ""),
                "model_family": capture_record.get("model_family", ""),
                "model_id": capture_record.get("model_id", ""),
                "geometry_source": "attention_graph_record",
            }
        )
        graph_records.append(graph_dict)
        evidence_records.append(evidence_dict)
        relation_rows.append(
            {
                "capture_id": evidence.capture_id,
                "attention_graph_id": evidence.attention_graph_id,
                "attention_layer": capture_record["attention_layer"],
                "attention_relation_consistency": evidence.attention_relation_consistency,
                "anchor_inlier_ratio": evidence.anchor_inlier_ratio,
                "registration_confidence": evidence.registration_confidence,
                "recovered_sync_consistency": evidence.recovered_sync_consistency,
                "alignment_residual": evidence.alignment_residual,
                "geometry_reliable": evidence.geometry_reliable,
                "direct_positive_decision": evidence.direct_positive_decision,
                "unsupported_reason": evidence.unsupported_reason,
            }
        )
    return graph_records, evidence_records, relation_rows


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    """写出 CSV 文件。"""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def metric_mean(rows: list[dict[str, Any]], field_name: str) -> float:
    """计算表格字段均值。"""
    if not rows:
        return 0.0
    return sum(float(row[field_name]) for row in rows) / len(rows)


def existing_input_paths(root_path: Path) -> tuple[str, ...]:
    """登记存在的输入路径。"""
    candidates = (CONTENT_MANIFEST_PATH, CONTENT_SUMMARY_PATH, RUNTIME_MANIFEST_PATH, ATTENTION_CAPTURE_RECORDS_PATH)
    paths = []
    for candidate in candidates:
        path = root_path / candidate
        if path.exists():
            paths.append(relative_or_absolute(path, root_path))
    return tuple(paths)


def write_attention_geometry_outputs(
    root: str | Path = ".",
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    max_records: int | None = None,
) -> dict[str, Any]:
    """写出注意力图与几何证据产物。"""
    root_path = Path(root).resolve()
    resolved_output_dir = ensure_output_dir_under_outputs(root_path, Path(output_dir))
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    capture_records = load_jsonl(root_path / ATTENTION_CAPTURE_RECORDS_PATH)
    if max_records is not None:
        capture_records = capture_records[:max_records]
    graph_records, evidence_records, relation_rows = build_geometry_records(capture_records)

    graph_records_path = resolved_output_dir / "attention_graph_records.jsonl"
    evidence_records_path = resolved_output_dir / "geometry_evidence_records.jsonl"
    relation_table_path = resolved_output_dir / "attention_relation_consistency.csv"
    summary_path = resolved_output_dir / "geometry_evidence_summary.json"
    manifest_path = resolved_output_dir / "manifest.local.json"

    graph_records_path.write_text("".join(json_line(record) for record in graph_records), encoding="utf-8")
    evidence_records_path.write_text("".join(json_line(record) for record in evidence_records), encoding="utf-8")
    write_csv(
        relation_table_path,
        relation_rows,
        [
            "capture_id",
            "attention_graph_id",
            "attention_layer",
            "attention_relation_consistency",
            "anchor_inlier_ratio",
            "registration_confidence",
            "recovered_sync_consistency",
            "alignment_residual",
            "geometry_reliable",
            "direct_positive_decision",
            "unsupported_reason",
        ],
    )
    unsupported_count = sum(1 for record in capture_records if record.get("unsupported_reason"))
    summary = {
        "construction_unit_name": CONSTRUCTION_UNIT_NAME,
        "attention_capture_record_count": len(capture_records),
        "attention_graph_record_count": len(graph_records),
        "geometry_evidence_record_count": len(evidence_records),
        "real_attention_capture_count": len(capture_records) - unsupported_count,
        "unsupported_capture_count": unsupported_count,
        "attention_relation_consistency_mean": metric_mean(relation_rows, "attention_relation_consistency"),
        "anchor_inlier_ratio_mean": metric_mean(relation_rows, "anchor_inlier_ratio"),
        "registration_confidence_mean": metric_mean(relation_rows, "registration_confidence"),
        "recovered_sync_consistency_mean": metric_mean(relation_rows, "recovered_sync_consistency"),
        "alignment_residual_mean": metric_mean(relation_rows, "alignment_residual"),
        "geometry_reliable_count": sum(1 for row in relation_rows if bool(row["geometry_reliable"])),
        "direct_positive_decision_used": any(bool(row["direct_positive_decision"]) for row in relation_rows),
        "attention_geometry_ready": bool(relation_rows) and unsupported_count == 0,
        "protocol_decision": "pass" if relation_rows and not any(bool(row["direct_positive_decision"]) for row in relation_rows) else "fail",
        "supports_paper_claim": False,
    }
    summary_path.write_text(stable_json_text(summary), encoding="utf-8")

    output_paths = tuple(
        relative_or_absolute(path, root_path)
        for path in (graph_records_path, evidence_records_path, relation_table_path, summary_path, manifest_path)
    )
    config = {
        "construction_unit_name": CONSTRUCTION_UNIT_NAME,
        "summary_digest": build_stable_digest(summary),
        "attention_graph_record_count": summary["attention_graph_record_count"],
        "geometry_evidence_record_count": summary["geometry_evidence_record_count"],
        "manifest_digest": file_digest(root_path / CONTENT_MANIFEST_PATH) if (root_path / CONTENT_MANIFEST_PATH).exists() else "missing",
    }
    manifest = build_artifact_manifest(
        artifact_id="attention_geometry_manifest",
        artifact_type="local_manifest",
        input_paths=existing_input_paths(root_path),
        output_paths=output_paths,
        config=config,
        code_version=resolve_code_version(root_path),
        rebuild_command="python scripts/write_attention_geometry_outputs.py",
        metadata={
            "construction_unit_name": CONSTRUCTION_UNIT_NAME,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "protocol_decision": summary["protocol_decision"],
            "attention_geometry_ready": summary["attention_geometry_ready"],
            "supports_paper_claim": False,
        },
    ).to_dict()
    manifest_path.write_text(stable_json_text(manifest), encoding="utf-8")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器。"""
    parser = argparse.ArgumentParser(description="写出注意力图与几何证据产物。")
    parser.add_argument("--root", default=".", help="仓库根目录。")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="输出目录, 必须位于 outputs/ 下。")
    parser.add_argument("--max-records", type=int, default=None, help="调试时限制处理记录数量。")
    return parser


def main() -> None:
    """命令行入口。"""
    args = build_parser().parse_args()
    manifest = write_attention_geometry_outputs(root=args.root, output_dir=args.output_dir, max_records=args.max_records)
    print(stable_json_text(manifest), end="")


if __name__ == "__main__":
    main()
