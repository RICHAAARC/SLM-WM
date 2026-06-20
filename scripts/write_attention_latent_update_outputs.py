"""写出 attention-relative latent update 产物。"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from typing import Any
import zipfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from main.analysis.artifact_manifest import build_artifact_manifest
from main.core.digest import build_stable_digest
from main.methods.carrier import derive_attention_relative_carrier, simulate_attention_update_strengths

CONSTRUCTION_UNIT_NAME = "attention_latent_update"
DEFAULT_OUTPUT_DIR = Path("outputs/attention_latent_update")
SEMANTIC_MANIFEST_PATH = Path("outputs/semantic_subspace/manifest.local.json")
SUBSPACE_RECORDS_PATH = Path("outputs/semantic_subspace/subspace_plan_records.jsonl")
ROUTE_RECORDS_PATH = Path("outputs/semantic_subspace/semantic_route_records.jsonl")
CONTENT_MANIFEST_PATH = Path("outputs/content_carriers/manifest.local.json")
CONTENT_SUMMARY_PATH = Path("outputs/content_carriers/content_carrier_summary.json")
LOCAL_GEOMETRY_DIR = Path("outputs/attention_geometry")
GEOMETRY_PACKAGE_PATTERN = "attention_geometry_package_*.zip"
VECTOR_WIDTH = 8
STRENGTH_SCALES = (0.0, 0.5, 1.0, 1.5)


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
        raise ValueError("attention latent update 输出目录必须位于 outputs/ 下。") from exc
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


def read_json_from_zip(archive: zipfile.ZipFile, name: str) -> dict[str, Any]:
    """从 zip 读取 JSON 文件。"""
    return json.loads(archive.read(name).decode("utf-8"))


def read_jsonl_from_zip(archive: zipfile.ZipFile, name: str) -> tuple[dict[str, Any], ...]:
    """从 zip 读取 JSONL 文件。"""
    return tuple(json.loads(line) for line in archive.read(name).decode("utf-8").splitlines() if line.strip())


def latest_ready_geometry_package(root_path: Path) -> Path | None:
    """查找最新的 ready attention geometry 压缩包。"""
    candidates = sorted((root_path / "outputs").glob(GEOMETRY_PACKAGE_PATTERN), key=lambda path: path.name, reverse=True)
    for candidate in candidates:
        try:
            with zipfile.ZipFile(candidate) as archive:
                summary = read_json_from_zip(archive, "outputs/attention_geometry/geometry_evidence_summary.json")
        except Exception:
            continue
        if bool(summary.get("attention_geometry_ready", False)):
            return candidate
    return None


def load_attention_geometry_from_package(package_path: Path, root_path: Path) -> dict[str, Any]:
    """从真实 attention geometry 压缩包读取图、证据和 summary。"""
    with zipfile.ZipFile(package_path) as archive:
        summary = read_json_from_zip(archive, "outputs/attention_geometry/geometry_evidence_summary.json")
        manifest = read_json_from_zip(archive, "outputs/attention_geometry/manifest.local.json")
        graph_records = read_jsonl_from_zip(archive, "outputs/attention_geometry/attention_graph_records.jsonl")
        evidence_records = read_jsonl_from_zip(archive, "outputs/attention_geometry/geometry_evidence_records.jsonl")
    return {
        "source_kind": "zip_package",
        "source_path": relative_or_absolute(package_path, root_path),
        "summary": summary,
        "manifest": manifest,
        "graph_records": graph_records,
        "evidence_records": evidence_records,
    }


def load_attention_geometry_from_directory(root_path: Path) -> dict[str, Any] | None:
    """从本地 attention geometry 目录读取 ready 产物。"""
    summary_path = root_path / LOCAL_GEOMETRY_DIR / "geometry_evidence_summary.json"
    if not summary_path.exists():
        return None
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not bool(summary.get("attention_geometry_ready", False)):
        return None
    return {
        "source_kind": "local_directory",
        "source_path": relative_or_absolute(root_path / LOCAL_GEOMETRY_DIR, root_path),
        "summary": summary,
        "manifest": json.loads((root_path / LOCAL_GEOMETRY_DIR / "manifest.local.json").read_text(encoding="utf-8")),
        "graph_records": load_jsonl(root_path / LOCAL_GEOMETRY_DIR / "attention_graph_records.jsonl"),
        "evidence_records": load_jsonl(root_path / LOCAL_GEOMETRY_DIR / "geometry_evidence_records.jsonl"),
    }


def load_attention_geometry(root_path: Path, package_path: str | Path | None) -> dict[str, Any]:
    """按显式包、本地目录、最新包的顺序读取 attention geometry 输入。"""
    if package_path:
        resolved_package = Path(package_path)
        if not resolved_package.is_absolute():
            resolved_package = root_path / resolved_package
        return load_attention_geometry_from_package(resolved_package.resolve(), root_path)
    local_bundle = load_attention_geometry_from_directory(root_path)
    if local_bundle is not None:
        return local_bundle
    discovered_package = latest_ready_geometry_package(root_path)
    if discovered_package is None:
        raise RuntimeError("缺少 ready attention geometry 输入。")
    return load_attention_geometry_from_package(discovered_package.resolve(), root_path)


def route_records_by_prompt(route_records: tuple[dict[str, Any], ...]) -> dict[str, dict[str, Any]]:
    """按 prompt_id 建立 route record 索引。"""
    return {record["prompt_id"]: record for record in route_records}


def evidence_records_by_graph(evidence_records: tuple[dict[str, Any], ...]) -> dict[str, dict[str, Any]]:
    """按 attention_graph_id 建立几何证据索引。"""
    return {record["attention_graph_id"]: record for record in evidence_records}


def select_subspace_records(records: tuple[dict[str, Any], ...], max_records: int) -> tuple[dict[str, Any], ...]:
    """选择语义安全子空间记录, 控制本地重建成本。"""
    selected = [record for record in records if record.get("basis_strategy") == "semantic_safe_basis"]
    if not selected:
        selected = list(records)
    return tuple(selected[:max_records])


def build_attention_update_records(
    subspace_records: tuple[dict[str, Any], ...],
    route_lookup: dict[str, dict[str, Any]],
    graph_records: tuple[dict[str, Any], ...],
    evidence_lookup: dict[str, dict[str, Any]],
    vector_width: int,
    embedding_strength: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """构造 attention carrier records 与强度稳定性表。"""
    carrier_records: list[dict[str, Any]] = []
    stability_rows: list[dict[str, Any]] = []
    for subspace_record in subspace_records:
        route_record = route_lookup[subspace_record["prompt_id"]]
        for graph_record in graph_records:
            evidence_record = evidence_lookup[graph_record["attention_graph_id"]]
            carrier = derive_attention_relative_carrier(
                attention_graph=graph_record,
                geometry_evidence=evidence_record,
                subspace_record=subspace_record,
                route_record=route_record,
                vector_width=vector_width,
                embedding_strength=embedding_strength,
            )
            carrier_dict = carrier.to_dict()
            carrier_dict["metadata"].update(
                {
                    "prompt_id": subspace_record.get("prompt_id", ""),
                    "split": subspace_record.get("split", ""),
                    "subspace_plan_id": subspace_record.get("subspace_plan_id", ""),
                    "route_id": route_record.get("route_id", ""),
                }
            )
            carrier_records.append(carrier_dict)
            stability_rows.extend(simulate_attention_update_strengths(carrier, STRENGTH_SCALES))
    return carrier_records, stability_rows


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    """写出 CSV 文件。"""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def metric_mean(rows: list[dict[str, Any]], field_name: str) -> float:
    """计算数值字段均值。"""
    if not rows:
        return 0.0
    return sum(float(row[field_name]) for row in rows) / len(rows)


def build_quality_rows(carrier_records: list[dict[str, Any]], stability_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """构造轻量质量代理指标表。"""
    active_records = [record for record in carrier_records if record["fallback_mode"] == "active_update"]
    stable_rows = [row for row in stability_rows if bool(row["attention_update_stable"])]
    carrier_count = len(carrier_records)
    return [
        {
            "quality_metric_name": "active_update_fraction",
            "quality_metric_value": len(active_records) / carrier_count if carrier_count else 0.0,
            "quality_metric_source": "attention_relation_proxy",
            "supports_paper_claim": False,
        },
        {
            "quality_metric_name": "stable_strength_row_fraction",
            "quality_metric_value": len(stable_rows) / len(stability_rows) if stability_rows else 0.0,
            "quality_metric_source": "attention_relation_proxy",
            "supports_paper_claim": False,
        },
        {
            "quality_metric_name": "quality_proxy_drop_mean",
            "quality_metric_value": metric_mean(stability_rows, "quality_proxy_drop"),
            "quality_metric_source": "attention_relation_proxy",
            "supports_paper_claim": False,
        },
        {
            "quality_metric_name": "image_quality_metrics_ready",
            "quality_metric_value": 0.0,
            "quality_metric_source": "not_measured_in_local_rebuild",
            "supports_paper_claim": False,
        },
    ]


def existing_input_paths(root_path: Path, geometry_bundle: dict[str, Any]) -> tuple[str, ...]:
    """登记存在的输入路径。"""
    candidates = (
        SEMANTIC_MANIFEST_PATH,
        SUBSPACE_RECORDS_PATH,
        ROUTE_RECORDS_PATH,
        CONTENT_MANIFEST_PATH,
        CONTENT_SUMMARY_PATH,
    )
    paths = [geometry_bundle["source_path"]]
    for candidate in candidates:
        path = root_path / candidate
        if path.exists():
            paths.append(relative_or_absolute(path, root_path))
    return tuple(paths)


def write_attention_latent_update_outputs(
    root: str | Path = ".",
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    attention_geometry_package_path: str | Path | None = None,
    max_subspace_records: int = 16,
    vector_width: int = VECTOR_WIDTH,
    embedding_strength: float = 0.08,
) -> dict[str, Any]:
    """写出 attention-relative latent update 产物。"""
    root_path = Path(root).resolve()
    resolved_output_dir = ensure_output_dir_under_outputs(root_path, Path(output_dir))
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    geometry_bundle = load_attention_geometry(root_path, attention_geometry_package_path)
    geometry_summary = geometry_bundle["summary"]
    geometry_ready = bool(geometry_summary.get("attention_geometry_ready", False))
    subspace_records = select_subspace_records(load_jsonl(root_path / SUBSPACE_RECORDS_PATH), max_subspace_records)
    route_lookup = route_records_by_prompt(load_jsonl(root_path / ROUTE_RECORDS_PATH))
    evidence_lookup = evidence_records_by_graph(geometry_bundle["evidence_records"])
    carrier_records, stability_rows = build_attention_update_records(
        subspace_records=subspace_records,
        route_lookup=route_lookup,
        graph_records=geometry_bundle["graph_records"],
        evidence_lookup=evidence_lookup,
        vector_width=vector_width,
        embedding_strength=embedding_strength,
    )
    quality_rows = build_quality_rows(carrier_records, stability_rows)

    carrier_records_path = resolved_output_dir / "attention_carrier_records.jsonl"
    stability_path = resolved_output_dir / "attention_update_stability.csv"
    quality_path = resolved_output_dir / "attention_update_quality_metrics.csv"
    summary_path = resolved_output_dir / "attention_update_summary.json"
    manifest_path = resolved_output_dir / "manifest.local.json"

    carrier_records_path.write_text("".join(json_line(record) for record in carrier_records), encoding="utf-8")
    write_csv(
        stability_path,
        stability_rows,
        [
            "carrier_id",
            "attention_graph_id",
            "capture_id",
            "attention_update_strength",
            "relation_loss_before",
            "relation_loss_after",
            "relation_loss_delta",
            "relation_consistency_before",
            "relation_consistency_after",
            "projected_update_norm",
            "quality_proxy_drop",
            "attention_update_stable",
            "fallback_mode",
            "unsupported_reason",
        ],
    )
    write_csv(quality_path, quality_rows, ["quality_metric_name", "quality_metric_value", "quality_metric_source", "supports_paper_claim"])

    active_update_count = sum(1 for record in carrier_records if record["fallback_mode"] == "active_update")
    stable_carrier_count = sum(1 for record in carrier_records if bool(record["attention_update_stable"]))
    evidence_only_count = sum(1 for record in carrier_records if record["fallback_mode"] == "evidence_only")
    unsupported_reasons = sorted({record["unsupported_reason"] for record in carrier_records if record["unsupported_reason"]})
    summary = {
        "construction_unit_name": CONSTRUCTION_UNIT_NAME,
        "attention_geometry_source_path": geometry_bundle["source_path"],
        "attention_geometry_ready": geometry_ready,
        "subspace_record_count": len(subspace_records),
        "attention_graph_record_count": len(geometry_bundle["graph_records"]),
        "attention_carrier_record_count": len(carrier_records),
        "active_update_count": active_update_count,
        "evidence_only_count": evidence_only_count,
        "attention_update_stable_count": stable_carrier_count,
        "attention_update_stability_row_count": len(stability_rows),
        "quality_metric_count": len(quality_rows),
        "relation_loss_delta_mean": metric_mean(stability_rows, "relation_loss_delta"),
        "quality_proxy_drop_mean": metric_mean(stability_rows, "quality_proxy_drop"),
        "image_quality_metrics_ready": False,
        "full_method_claim_ready": False,
        "unsupported_reasons": unsupported_reasons,
        "protocol_decision": "pass" if geometry_ready and active_update_count > 0 else "fail",
        "supports_paper_claim": False,
    }
    summary_path.write_text(stable_json_text(summary), encoding="utf-8")

    output_paths = tuple(
        relative_or_absolute(path, root_path)
        for path in (carrier_records_path, stability_path, quality_path, summary_path, manifest_path)
    )
    manifest = build_artifact_manifest(
        artifact_id="attention_latent_update_manifest",
        artifact_type="local_manifest",
        input_paths=existing_input_paths(root_path, geometry_bundle),
        output_paths=output_paths,
        config={
            "construction_unit_name": CONSTRUCTION_UNIT_NAME,
            "summary_digest": build_stable_digest(summary),
            "attention_carrier_record_count": summary["attention_carrier_record_count"],
            "active_update_count": active_update_count,
            "embedding_strength": embedding_strength,
            "vector_width": vector_width,
        },
        code_version=resolve_code_version(root_path),
        rebuild_command="python scripts/write_attention_latent_update_outputs.py",
        metadata={
            "construction_unit_name": CONSTRUCTION_UNIT_NAME,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "protocol_decision": summary["protocol_decision"],
            "attention_geometry_ready": geometry_ready,
            "full_method_claim_ready": False,
            "supports_paper_claim": False,
        },
    ).to_dict()
    manifest_path.write_text(stable_json_text(manifest), encoding="utf-8")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器。"""
    parser = argparse.ArgumentParser(description="写出 attention-relative latent update 产物。")
    parser.add_argument("--root", default=".", help="仓库根目录。")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="输出目录, 必须位于 outputs/ 下。")
    parser.add_argument("--attention-geometry-package-path", default=None, help="真实 attention geometry zip 输入路径。")
    parser.add_argument("--max-subspace-records", type=int, default=16, help="限制处理的语义子空间记录数量。")
    parser.add_argument("--vector-width", type=int, default=VECTOR_WIDTH, help="attention update 向量宽度。")
    parser.add_argument("--embedding-strength", type=float, default=0.08, help="attention update 嵌入强度。")
    return parser


def main() -> None:
    """命令行入口。"""
    args = build_parser().parse_args()
    manifest = write_attention_latent_update_outputs(
        root=args.root,
        output_dir=args.output_dir,
        attention_geometry_package_path=args.attention_geometry_package_path,
        max_subspace_records=args.max_subspace_records,
        vector_width=args.vector_width,
        embedding_strength=args.embedding_strength,
    )
    print(stable_json_text(manifest), end="")


if __name__ == "__main__":
    main()
