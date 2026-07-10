"""写出 LF 与高斯幅值尾部截断内容载体诊断产物。"""

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
import zipfile

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from main.analysis.artifact_manifest import build_artifact_manifest
from main.core.digest import build_stable_digest
from experiments.protocol.paper_run_config import DEFAULT_CONTENT_VECTOR_WIDTH, build_paper_run_config
from main.methods.carrier import CONTENT_MODES, compose_content_update, derive_tail_content_carrier, derive_lf_content_carrier
from main.methods.detection import build_content_detection_record, compute_unified_content_score

CONSTRUCTION_UNIT_NAME = "content_carriers"
DEFAULT_OUTPUT_DIR = Path("outputs/content_carriers")
SEMANTIC_MANIFEST_PATH = Path("outputs/semantic_subspace/manifest.local.json")
SUBSPACE_RECORDS_PATH = Path("outputs/semantic_subspace/subspace_plan_records.jsonl")
ROUTE_RECORDS_PATH = Path("outputs/semantic_subspace/semantic_route_records.jsonl")
QUALITY_ARCHIVE_PATH = Path("outputs/minimal_latent_injection_package_20260620t10181781950721z_b2be25c.zip")
VECTOR_WIDTH = DEFAULT_CONTENT_VECTOR_WIDTH
KEY_MATERIAL = "slm_wm_content_carrier_key_v1"
SAMPLE_ROLES = ("positive_source", "clean_negative", "attacked_negative")


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
        raise ValueError("内容载体输出目录必须位于 outputs/ 下。") from exc
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


def digest_signed_values(seed: str, count: int) -> tuple[float, ...]:
    """从稳定摘要派生 [-1, 1] 数值。"""
    values = []
    for index in range(count):
        digest = hashlib.sha256(f"{seed}|{index}".encode("utf-8")).hexdigest()
        unit = int(digest[:16], 16) / float(0xFFFFFFFFFFFFFFFF)
        values.append(unit * 2.0 - 1.0)
    return tuple(values)


def event_digest_for(subspace_record: dict[str, Any], route_record: dict[str, Any], sample_role: str) -> str:
    """为内容载体派生事件摘要。"""
    return build_stable_digest(
        {
            "prompt_id": subspace_record["prompt_id"],
            "subspace_plan_id": subspace_record["subspace_plan_id"],
            "route_digest": route_record["route_digest"],
            "sample_role": sample_role,
        }
    )


def build_observed_values(prompt_id: str, sample_role: str, combined_update: tuple[float, ...]) -> tuple[float, ...]:
    """构造内容检测用的确定性观测向量。"""
    probe = digest_signed_values(f"{prompt_id}|{sample_role}|observed", len(combined_update))
    if sample_role == "positive_source":
        return tuple(value + 0.02 * noise for value, noise in zip(combined_update, probe))
    if sample_role == "attacked_negative":
        return tuple(0.35 * value + 0.08 * noise for value, noise in zip(combined_update, probe))
    return tuple(0.10 * noise for noise in probe)


def build_carrier_bundle(
    subspace_record: dict[str, Any],
    route_record: dict[str, Any],
    sample_role: str,
    vector_width: int = VECTOR_WIDTH,
) -> dict[str, Any]:
    """构造 LF/尾部截断载体、机制开关和内容分数。"""
    selected_indices = tuple(int(value) for value in subspace_record["selected_indices"])
    event_digest = event_digest_for(subspace_record, route_record, sample_role)
    lf_carrier = derive_lf_content_carrier(
        selected_indices=selected_indices,
        basis_digest=subspace_record["basis_digest"],
        route_digest=route_record["route_digest"],
        event_digest=event_digest,
        key_material=KEY_MATERIAL,
        vector_width=vector_width,
    )
    tail_carrier = derive_tail_content_carrier(
        selected_indices=selected_indices,
        basis_digest=subspace_record["basis_digest"],
        route_digest=route_record["route_digest"],
        event_digest=event_digest,
        key_material=KEY_MATERIAL,
        vector_width=vector_width,
    )
    tail_without_truncation = derive_tail_content_carrier(
        selected_indices=selected_indices,
        basis_digest=subspace_record["basis_digest"],
        route_digest=route_record["route_digest"],
        event_digest=event_digest,
        key_material=KEY_MATERIAL,
        vector_width=vector_width,
        tail_truncation_enabled=False,
    )
    full_update = compose_content_update(lf_carrier, tail_carrier, "full_content_chain")
    observed_values = build_observed_values(subspace_record["prompt_id"], sample_role, full_update.combined_update_values)
    updates = {
        "full_content_chain": full_update,
        "lf_only": compose_content_update(lf_carrier, tail_carrier, "lf_only"),
        "tail_only": compose_content_update(lf_carrier, tail_carrier, "tail_only"),
        "no_tail": compose_content_update(lf_carrier, tail_carrier, "no_tail"),
        "no_tail_truncation": compose_content_update(lf_carrier, tail_without_truncation, "no_tail_truncation"),
        "no_lf": compose_content_update(lf_carrier, tail_carrier, "no_lf"),
    }
    scores = {name: compute_unified_content_score(observed_values, update) for name, update in updates.items()}
    return {
        "lf_carrier": lf_carrier,
        "tail_carrier": tail_carrier,
        "tail_without_truncation": tail_without_truncation,
        "observed_values": observed_values,
        "updates": updates,
        "scores": scores,
    }


def build_records(
    subspace_records: tuple[dict[str, Any], ...],
    route_records_by_prompt: dict[str, dict[str, Any]],
    vector_width: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """构造内容检测 records 和 score table 行。"""
    detection_records = []
    score_rows = []
    for subspace_record in subspace_records:
        route_record = route_records_by_prompt[subspace_record["prompt_id"]]
        for sample_role in SAMPLE_ROLES:
            bundle = build_carrier_bundle(subspace_record, route_record, sample_role, vector_width=vector_width)
            full_update = bundle["updates"]["full_content_chain"]
            full_score = bundle["scores"]["full_content_chain"]
            mechanism_scores = {name: score.content_score for name, score in bundle["scores"].items()}
            record = build_content_detection_record(
                prompt_id=subspace_record["prompt_id"],
                split=subspace_record["split"],
                content_update=full_update,
                score=full_score,
                metadata={
                    "sample_role": sample_role,
                    "lf_content_carrier_digest": bundle["lf_carrier"].lf_content_carrier_digest,
                    "tail_content_carrier_digest": bundle["tail_carrier"].tail_content_carrier_digest,
                    "mechanism_scores": mechanism_scores,
                    "used_independent_branch_vote": False,
                    "content_vector_width": vector_width,
                    "supports_paper_claim": False,
                },
            ).to_dict()
            detection_records.append(record)
            score_rows.append(
                {
                    "prompt_id": subspace_record["prompt_id"],
                    "split": subspace_record["split"],
                    "sample_role": sample_role,
                    "content_mode": full_update.content_mode,
                    "lf_score": full_score.lf_score,
                    "tail_score": full_score.tail_score,
                    "combined_score": full_score.combined_score,
                    "lf_tail_fusion_score": full_score.lf_tail_fusion_score,
                    "content_score": full_score.content_score,
                    "lf_enabled": full_update.lf_enabled,
                    "tail_enabled": full_update.tail_enabled,
                    "tail_truncation_enabled": full_update.tail_truncation_enabled,
                    "fixed_fpr_ready": full_score.fixed_fpr_ready,
                    "content_chain_digest": full_update.content_chain_digest,
                }
            )
    return detection_records, score_rows


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    """写出 CSV 文件。"""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def score_distribution_rows(score_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """构造内容分数分布表。"""
    rows = []
    for sample_role in SAMPLE_ROLES:
        role_scores = [float(row["content_score"]) for row in score_rows if row["sample_role"] == sample_role]
        if not role_scores:
            continue
        for lower in [round(-1.0 + 0.2 * index, 1) for index in range(10)]:
            upper = round(lower + 0.2, 1)
            count = sum(1 for score in role_scores if lower <= score < upper or (upper == 1.0 and score <= upper))
            rows.append(
                {
                    "sample_role": sample_role,
                    "score_distribution_bin": f"[{lower:.1f},{upper:.1f}]",
                    "score_count": count,
                    "score_min": min(role_scores),
                    "score_max": max(role_scores),
                    "score_mean": sum(role_scores) / len(role_scores),
                }
            )
    return rows


def write_quality_metrics(root_path: Path, output_path: Path) -> None:
    """从最小注入包复制 paired quality metrics。"""
    archive_path = root_path / QUALITY_ARCHIVE_PATH
    if archive_path.exists():
        with zipfile.ZipFile(archive_path) as archive:
            text = archive.read("sd35_paired_quality_metrics.csv").decode("utf-8")
    else:
        text = "metric_name,metric_value,supports_paper_claim\nquality_source_missing,0,false\n"
    output_path.write_text(text, encoding="utf-8")


def existing_input_paths(root_path: Path) -> tuple[str, ...]:
    """登记存在的输入路径。"""
    candidates = (SEMANTIC_MANIFEST_PATH, SUBSPACE_RECORDS_PATH, ROUTE_RECORDS_PATH, QUALITY_ARCHIVE_PATH)
    paths = []
    for candidate in candidates:
        path = root_path / candidate
        if path.exists():
            paths.append(relative_or_absolute(path, root_path))
    return tuple(paths)


def write_content_carrier_outputs(
    root: str | Path = ".",
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    max_records: int | None = None,
    vector_width: int | None = None,
) -> dict[str, Any]:
    """写出 LF/尾部截断内容载体诊断产物。"""
    root_path = Path(root).resolve()
    resolved_vector_width = int(vector_width or build_paper_run_config(root_path).content_vector_width)
    resolved_output_dir = ensure_output_dir_under_outputs(root_path, Path(output_dir))
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    subspace_records = load_jsonl(root_path / SUBSPACE_RECORDS_PATH)
    if max_records is not None:
        subspace_records = subspace_records[:max_records]
    route_records = load_jsonl(root_path / ROUTE_RECORDS_PATH)
    route_records_by_prompt = {record["prompt_id"]: record for record in route_records}
    detection_records, score_rows = build_records(subspace_records, route_records_by_prompt, resolved_vector_width)
    distribution_rows = score_distribution_rows(score_rows)

    detection_records_path = resolved_output_dir / "content_detection_records.jsonl"
    score_table_path = resolved_output_dir / "lf_tail_score_table.csv"
    quality_metrics_path = resolved_output_dir / "paired_quality_metrics.csv"
    distribution_path = resolved_output_dir / "content_score_distribution.csv"
    summary_path = resolved_output_dir / "content_carrier_summary.json"
    manifest_path = resolved_output_dir / "manifest.local.json"

    detection_records_path.write_text("".join(json_line(record) for record in detection_records), encoding="utf-8")
    write_csv(
        score_table_path,
        score_rows,
        [
            "prompt_id",
            "split",
            "sample_role",
            "content_mode",
            "lf_score",
            "tail_score",
            "combined_score",
            "lf_tail_fusion_score",
            "content_score",
            "lf_enabled",
            "tail_enabled",
            "tail_truncation_enabled",
            "fixed_fpr_ready",
            "content_chain_digest",
        ],
    )
    write_quality_metrics(root_path, quality_metrics_path)
    write_csv(
        distribution_path,
        distribution_rows,
        ["sample_role", "score_distribution_bin", "score_count", "score_min", "score_max", "score_mean"],
    )
    content_scores = [float(row["content_score"]) for row in score_rows]
    summary = {
        "construction_unit_name": CONSTRUCTION_UNIT_NAME,
        "content_detection_record_count": len(detection_records),
        "score_count": len(content_scores),
        "score_min": min(content_scores) if content_scores else 0.0,
        "score_max": max(content_scores) if content_scores else 0.0,
        "score_mean": sum(content_scores) / len(content_scores) if content_scores else 0.0,
        "content_modes": list(CONTENT_MODES),
        "content_vector_width": resolved_vector_width,
        "content_basis_rank": min(
            (len(record.get("selected_indices", ())) for record in subspace_records),
            default=0,
        ),
        "fixed_fpr_ready": all(bool(row["fixed_fpr_ready"]) for row in score_rows),
        "used_independent_branch_vote": False,
        "protocol_decision": "pass" if detection_records and content_scores else "fail",
        "supports_paper_claim": False,
    }
    summary_path.write_text(stable_json_text(summary), encoding="utf-8")
    output_paths = tuple(
        relative_or_absolute(path, root_path)
        for path in (
            detection_records_path,
            score_table_path,
            quality_metrics_path,
            distribution_path,
            summary_path,
            manifest_path,
        )
    )
    manifest = build_artifact_manifest(
        artifact_id="content_carrier_manifest",
        artifact_type="local_manifest",
        input_paths=existing_input_paths(root_path),
        output_paths=output_paths,
        config={
            "construction_unit_name": CONSTRUCTION_UNIT_NAME,
            "summary_digest": build_stable_digest(summary),
            "content_detection_record_count": summary["content_detection_record_count"],
            "score_count": summary["score_count"],
            "content_vector_width": resolved_vector_width,
            "content_basis_rank": summary["content_basis_rank"],
        },
        code_version=resolve_code_version(root_path),
        rebuild_command="python scripts/write_content_carrier_outputs.py",
        metadata={
            "construction_unit_name": CONSTRUCTION_UNIT_NAME,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "protocol_decision": summary["protocol_decision"],
            "supports_paper_claim": False,
        },
    ).to_dict()
    manifest_path.write_text(stable_json_text(manifest), encoding="utf-8")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器。"""
    parser = argparse.ArgumentParser(description="写出 LF/高斯幅值尾部截断内容载体诊断产物。")
    parser.add_argument("--root", default=".", help="仓库根目录。")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="输出目录, 必须位于 outputs/ 下。")
    parser.add_argument("--max-records", type=int, default=None, help="调试时限制处理记录数量。")
    parser.add_argument("--vector-width", type=int, default=None, help="内容载体向量宽度, 默认读取论文运行配置。")
    return parser


def main() -> None:
    """命令行入口。"""
    args = build_parser().parse_args()
    manifest = write_content_carrier_outputs(
        root=args.root,
        output_dir=args.output_dir,
        max_records=args.max_records,
        vector_width=args.vector_width,
    )
    print(stable_json_text(manifest), end="")


if __name__ == "__main__":
    main()

