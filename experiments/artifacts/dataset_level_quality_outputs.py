"""写出数据集级图像质量证据产物的正式实现。"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any
import zipfile

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.protocol import (
    FORMAL_FEATURE_BACKEND,
    build_dataset_quality_image_records,
    build_dataset_quality_metric_rows,
    build_dataset_quality_summary,
)
from main.analysis.artifact_manifest import build_artifact_manifest
from main.core.digest import build_stable_digest

DEFAULT_OUTPUT_DIR = Path("outputs/dataset_level_quality")
DEFAULT_REAL_ATTACK_REGISTRY_PATH = Path("outputs/real_attack_evaluation/real_attacked_image_registry.jsonl")
DEFAULT_FORMAL_MIN_SAMPLE_COUNT = 100
DEFAULT_PROGRESS_INTERVAL_ITEMS = 50


def dataset_quality_io_progress_enabled() -> bool:
    """判断是否输出数据集级质量 I/O 与 proxy 进度."""

    value = os.environ.get("SLM_WM_DATASET_QUALITY_IO_PROGRESS", "1").strip().lower()
    return value not in {"0", "false", "no", "off"}


def progress_interval_items() -> int:
    """读取进度刷新间隔, 避免 Colab 输出区被逐图像刷屏."""

    return max(1, int(os.environ.get("SLM_WM_DATASET_QUALITY_PROGRESS_INTERVAL_ITEMS", str(DEFAULT_PROGRESS_INTERVAL_ITEMS))))


def should_emit_progress(completed: int, total: int) -> bool:
    """判断当前计数是否需要输出进度."""

    interval = progress_interval_items()
    return completed in {0, total} or completed % interval == 0


def emit_dataset_quality_progress(
    *,
    desc: str,
    started_at: float,
    completed: int,
    total: int,
    profile: str,
) -> None:
    """输出数据集级质量重建的总体工作量进度."""

    if not dataset_quality_io_progress_enabled() or not should_emit_progress(completed, total):
        return
    elapsed_seconds = max(0.0, time.monotonic() - started_at)
    percent = completed / max(total, 1) * 100.0
    eta_seconds = elapsed_seconds * max(total - completed, 0) / completed if completed > 0 else 0.0
    print(
        (
            f"工作量进度 | {desc} | {completed}/{total} ({percent:.1f}%) | "
            f"elapsed={elapsed_seconds / 60.0:.1f} min | eta={eta_seconds / 60.0:.1f} min | "
            f"profile={profile}"
        ),
        flush=True,
    )


def stable_json_text(value: Any) -> str:
    """把 JSON 兼容对象转换为稳定文本。"""

    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def json_line(value: dict[str, Any]) -> str:
    """把单条记录转换为 JSONL 行。"""

    return json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n"


def read_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    """读取 JSONL registry 行。"""

    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def read_formal_feature_rows(path: Path) -> list[dict[str, Any]]:
    """读取正式特征导入记录, 文件缺失时返回空集合。"""

    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def path_digest(path: Path) -> str:
    """计算文件 SHA-256 摘要, 用于记录外部 ZIP 与物化图像来源。"""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _numeric_vector(value: Any) -> list[float]:
    """在输入解析层把 feature vector 规整为浮点列表。"""

    if not isinstance(value, list):
        return []
    vector: list[float] = []
    for item in value:
        try:
            vector.append(float(item))
        except (TypeError, ValueError):
            return []
    return vector


def build_formal_feature_import_payload(
    *,
    records: Any,
    feature_rows: list[dict[str, Any]],
    formal_feature_records_path: Path,
    root_path: Path,
    formal_min_sample_count: int,
) -> dict[str, Any]:
    """把外部 Inception 特征记录解析为正式 FID / KID 可消费的矩阵和导入报告。

    该函数属于 schema / 配置解析层: 它集中处理字段缺失、角色不合法和维度不一致问题, 下游指标函数只接收矩阵。
    """

    record_ids = [record.dataset_quality_record_id for record in records]
    expected_pairs = {(record_id, role) for record_id in record_ids for role in ("source", "comparison")}
    vectors_by_key: dict[tuple[str, str], list[float]] = {}
    issues: list[dict[str, Any]] = []
    for row_index, row in enumerate(feature_rows):
        record_id = str(row.get("dataset_quality_record_id", "") or "")
        image_role = str(row.get("dataset_quality_image_role", "") or "")
        feature_backend = str(row.get("feature_backend", "") or "")
        feature_vector = _numeric_vector(row.get("feature_vector"))
        key = (record_id, image_role)
        if key not in expected_pairs:
            issues.append({"row_index": row_index, "field_name": "dataset_quality_record_id", "reason": "record_role_not_expected"})
            continue
        if feature_backend != FORMAL_FEATURE_BACKEND:
            issues.append({"row_index": row_index, "field_name": "feature_backend", "reason": "inception_feature_backend_required"})
            continue
        if not feature_vector:
            issues.append({"row_index": row_index, "field_name": "feature_vector", "reason": "numeric_feature_vector_required"})
            continue
        vectors_by_key[key] = feature_vector

    source_vectors: list[list[float]] = []
    comparison_vectors: list[list[float]] = []
    missing_feature_count = 0
    for record_id in record_ids:
        source_vector = vectors_by_key.get((record_id, "source"))
        comparison_vector = vectors_by_key.get((record_id, "comparison"))
        if source_vector is None or comparison_vector is None:
            missing_feature_count += 1
            continue
        if len(source_vector) != len(comparison_vector):
            issues.append({"row_index": -1, "field_name": "feature_vector", "reason": "source_comparison_feature_dimension_mismatch"})
            continue
        source_vectors.append(source_vector)
        comparison_vectors.append(comparison_vector)

    feature_dimension_values = sorted({len(vector) for vector in (*source_vectors, *comparison_vectors)})
    dimension_consistent = len(feature_dimension_values) <= 1
    if not dimension_consistent:
        issues.append({"row_index": -1, "field_name": "feature_vector", "reason": "feature_dimension_inconsistent"})
        source_vectors = []
        comparison_vectors = []
    accepted_pair_count = min(len(source_vectors), len(comparison_vectors))
    formal_feature_backend_ready = accepted_pair_count > 0 and not issues
    formal_sample_scale_ready = formal_feature_backend_ready and accepted_pair_count >= formal_min_sample_count
    if not feature_rows:
        unsupported_reason = "inception_feature_records_missing"
    elif not formal_feature_backend_ready:
        unsupported_reason = "inception_feature_records_invalid"
    elif not formal_sample_scale_ready:
        unsupported_reason = "requires_full_main_sample_scale"
    else:
        unsupported_reason = ""
    report = {
        "formal_feature_records_path": relative_or_absolute(formal_feature_records_path, root_path),
        "feature_backend": FORMAL_FEATURE_BACKEND,
        "input_feature_record_count": len(feature_rows),
        "accepted_feature_pair_count": accepted_pair_count,
        "missing_feature_pair_count": missing_feature_count,
        "formal_feature_issue_count": len(issues),
        "feature_dimension": feature_dimension_values[0] if dimension_consistent and feature_dimension_values else 0,
        "formal_min_sample_count": formal_min_sample_count,
        "formal_feature_backend_ready": formal_feature_backend_ready,
        "formal_sample_scale_ready": formal_sample_scale_ready,
        "unsupported_reason": unsupported_reason,
        "issues": issues,
        "supports_paper_claim": False,
    }
    return {
        "source_features": source_vectors if formal_feature_backend_ready else None,
        "comparison_features": comparison_vectors if formal_feature_backend_ready else None,
        "report": report,
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    """写出 CSV 表格。"""

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def resolve_path(root_path: Path, path: str | Path) -> Path:
    """将输入路径解析为绝对路径。"""

    candidate = Path(path)
    return candidate.resolve() if candidate.is_absolute() else (root_path / candidate).resolve()


def normalize_path_values(paths: Any) -> tuple[str | Path, ...]:
    """把 CLI 或 Python API 传入的路径参数规整为元组。"""

    if paths is None:
        return ()
    if isinstance(paths, (str, Path)):
        return (paths,)
    return tuple(path for path in paths if str(path))


def relative_or_absolute(path: Path, root_path: Path) -> str:
    """优先返回相对仓库根目录路径。"""

    try:
        return path.resolve().relative_to(root_path).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def ensure_output_dir_under_outputs(root_path: Path, output_dir: str | Path) -> Path:
    """确保持久输出目录位于 outputs/ 下。"""

    resolved = resolve_path(root_path, output_dir)
    outputs_root = (root_path / "outputs").resolve()
    try:
        resolved.relative_to(outputs_root)
    except ValueError as exc:
        raise ValueError("数据集级质量证据输出目录必须位于 outputs/ 下。") from exc
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def expand_input_package_paths(root_path: Path, input_package_paths: Any) -> tuple[Path, ...]:
    """解析显式传入的前序结果 ZIP 路径或 ZIP 目录。"""

    resolved_paths: list[Path] = []
    for raw_path in normalize_path_values(input_package_paths):
        resolved = resolve_path(root_path, raw_path)
        if resolved.is_dir():
            resolved_paths.extend(sorted(path.resolve() for path in resolved.glob("*.zip") if path.is_file()))
        elif resolved.is_file():
            resolved_paths.append(resolved)
    return tuple(dict.fromkeys(resolved_paths))


def requested_image_paths(records: Any) -> tuple[str, ...]:
    """从数据集级质量 records 中收集需要解析的图像相对路径。"""

    values: list[str] = []
    for record in records:
        values.append(record.source_image_path)
        values.append(record.comparison_image_path)
    return tuple(dict.fromkeys(value for value in values if value and not Path(value).is_absolute()))


def _safe_materialized_path(materialized_root: Path, member_name: str) -> Path:
    """把 ZIP member 映射到 outputs 下的受治理物化路径, 并阻断路径穿越。"""

    target_path = (materialized_root / member_name).resolve()
    target_path.relative_to(materialized_root.resolve())
    return target_path


def write_archive_member_with_digest(archive: zipfile.ZipFile, member_name: str, target_path: Path) -> str:
    """把 ZIP member 写入目标路径并同步计算图像摘要.

    该函数属于通用工程写法: 图像物化时已经需要逐块读取 ZIP member,
    因此同时更新 SHA-256 可以避免写盘后再次读取同一图像文件。前序
    ZIP 包摘要则由调用方按包级别缓存, 避免每张图像重复扫描 GB 级包。
    """

    digest = hashlib.sha256()
    with archive.open(member_name) as source_handle, target_path.open("wb") as target_handle:
        for chunk in iter(lambda: source_handle.read(1024 * 1024), b""):
            target_handle.write(chunk)
            digest.update(chunk)
    return digest.hexdigest()


def materialize_images_from_input_packages(
    *,
    records: Any,
    materialized_root: Path,
    input_package_paths: tuple[Path, ...],
) -> tuple[dict[str, Any], ...]:
    """从前序结果 ZIP 中仅解出当前质量指标需要读取的图像文件。"""

    materialized_records: list[dict[str, Any]] = []
    wanted_members = set(requested_image_paths(records))
    if not wanted_members:
        return ()
    materialized_root.mkdir(parents=True, exist_ok=True)
    matched_members_by_package: list[tuple[Path, tuple[str, ...]]] = []
    for package_path in input_package_paths:
        with zipfile.ZipFile(package_path) as archive:
            archive_members = set(archive.namelist())
            matched_members_by_package.append((package_path, tuple(sorted(wanted_members & archive_members))))
    total_materialization_count = sum(len(members) for _, members in matched_members_by_package)
    completed_count = 0
    started_at = time.monotonic()
    emit_dataset_quality_progress(
        desc="dataset-level image materialization",
        started_at=started_at,
        completed=0,
        total=total_materialization_count,
        profile=f"package_count={len(matched_members_by_package)}",
    )
    for package_path, matched_members in matched_members_by_package:
        package_digest = path_digest(package_path)
        with zipfile.ZipFile(package_path) as archive:
            for member_name in matched_members:
                target_path = _safe_materialized_path(materialized_root, member_name)
                target_path.parent.mkdir(parents=True, exist_ok=True)
                resolved_image_digest = write_archive_member_with_digest(archive, member_name, target_path)
                payload = {
                    "requested_image_path": member_name,
                    "resolved_image_path": target_path.as_posix(),
                    "resolved_from_package_path": package_path.as_posix(),
                    "resolved_image_digest": resolved_image_digest,
                    "resolved_from_package_digest": package_digest,
                    "resolution_status": "materialized_from_input_package",
                }
                payload["image_resolution_record_digest"] = build_stable_digest(payload)
                payload["image_resolution_record_id"] = (
                    f"dataset_quality_image_resolution_{payload['image_resolution_record_digest'][:16]}"
                )
                materialized_records.append(payload)
                completed_count += 1
                emit_dataset_quality_progress(
                    desc="dataset-level image materialization",
                    started_at=started_at,
                    completed=completed_count,
                    total=total_materialization_count,
                    profile=f"package={package_path.name} image={Path(member_name).name}",
                )
    return tuple(materialized_records)


def resolve_existing_image_path(
    image_path_text: str,
    root_path: Path,
    image_search_roots: tuple[Path, ...],
) -> Path:
    """在仓库根目录和补充图像根目录中查找可读取图像。"""

    raw_path = Path(image_path_text)
    if raw_path.is_absolute():
        return raw_path.resolve()
    candidates = [(root_path / raw_path).resolve()]
    candidates.extend((search_root / raw_path).resolve() for search_root in image_search_roots)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def build_image_resolution_records(
    *,
    records: Any,
    root_path: Path,
    image_search_roots: tuple[Path, ...],
    materialized_root: Path,
    materialized_records: tuple[dict[str, Any], ...],
) -> tuple[dict[str, Any], ...]:
    """记录每个图像路径最终是否能够被解析。"""

    materialized_by_request = {record["requested_image_path"]: record for record in materialized_records}
    resolution_records: list[dict[str, Any]] = []
    for requested_path in requested_image_paths(records):
        resolved_path = resolve_existing_image_path(requested_path, root_path, image_search_roots)
        materialized_record = materialized_by_request.get(requested_path)
        if resolved_path.is_file() and materialized_record:
            status = "materialized_from_input_package"
            package_path = materialized_record["resolved_from_package_path"]
        elif resolved_path.is_file():
            status = "resolved_existing_image_file"
            package_path = ""
        else:
            status = "image_file_missing"
            package_path = ""
        payload = {
            "requested_image_path": requested_path,
            "resolved_image_path": relative_or_absolute(resolved_path, root_path) if resolved_path.is_file() else "",
            "resolved_from_package_path": package_path,
            "resolution_status": status,
            "resolved_image_digest": path_digest(resolved_path) if resolved_path.is_file() else "",
            "materialized_image_input": bool(
                resolved_path.is_file()
                and materialized_root.resolve() in (resolved_path.resolve(), *resolved_path.resolve().parents)
            ),
            "supports_paper_claim": False,
        }
        payload["image_resolution_record_digest"] = build_stable_digest(payload)
        payload["image_resolution_record_id"] = (
            f"dataset_quality_image_resolution_{payload['image_resolution_record_digest'][:16]}"
        )
        resolution_records.append(payload)
    return tuple(resolution_records)


def resolve_code_version(root_path: Path) -> str:
    """读取当前 Git 短提交标识, 工作区有变更时附加 dirty 标记。"""

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


def write_dataset_level_quality_outputs(
    *,
    root: str | Path = ".",
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    real_attack_registry_path: str | Path = DEFAULT_REAL_ATTACK_REGISTRY_PATH,
    image_search_roots: Any = (),
    input_package_paths: Any = (),
    formal_feature_records_path: str | Path | None = None,
    formal_min_sample_count: int = DEFAULT_FORMAL_MIN_SAMPLE_COUNT,
) -> dict[str, Any]:
    """写出数据集级质量 records、metrics、summary 和 manifest。

    当前实现只生成小样本 pixel feature proxy, 明确保持正式 FID / KID 为 unsupported。
    """

    root_path = Path(root).resolve()
    resolved_output_dir = ensure_output_dir_under_outputs(root_path, output_dir)
    resolved_registry_path = resolve_path(root_path, real_attack_registry_path)
    resolved_formal_feature_records_path = (
        resolve_path(root_path, formal_feature_records_path)
        if formal_feature_records_path
        else resolved_output_dir / "dataset_quality_formal_feature_records.jsonl"
    )
    resolved_input_package_paths = expand_input_package_paths(root_path, input_package_paths)
    explicit_image_search_roots = tuple(
        resolve_path(root_path, path) for path in normalize_path_values(image_search_roots)
    )
    registry_rows = read_jsonl_rows(resolved_registry_path)

    records = build_dataset_quality_image_records(registry_rows, root_path)
    materialized_root = resolved_output_dir / "materialized_image_inputs"
    materialized_records = materialize_images_from_input_packages(
        records=records,
        materialized_root=materialized_root,
        input_package_paths=resolved_input_package_paths,
    )
    materialized_search_roots = (materialized_root,) if materialized_records else ()
    all_image_search_roots = tuple(dict.fromkeys((*explicit_image_search_roots, *materialized_search_roots)))
    image_resolution_records = build_image_resolution_records(
        records=records,
        root_path=root_path,
        image_search_roots=all_image_search_roots,
        materialized_root=materialized_root,
        materialized_records=materialized_records,
    )
    formal_feature_payload = build_formal_feature_import_payload(
        records=records,
        feature_rows=read_formal_feature_rows(resolved_formal_feature_records_path),
        formal_feature_records_path=resolved_formal_feature_records_path,
        root_path=root_path,
        formal_min_sample_count=formal_min_sample_count,
    )
    metric_rows = build_dataset_quality_metric_rows(
        records,
        root_path,
        image_search_roots=all_image_search_roots,
        formal_source_features=formal_feature_payload["source_features"],
        formal_comparison_features=formal_feature_payload["comparison_features"],
        formal_min_sample_count=formal_min_sample_count,
    )

    records_path = resolved_output_dir / "dataset_quality_image_records.jsonl"
    image_resolution_records_path = resolved_output_dir / "dataset_quality_image_resolution_records.jsonl"
    formal_feature_import_report_path = resolved_output_dir / "dataset_quality_formal_feature_import_report.json"
    metrics_path = resolved_output_dir / "dataset_quality_metrics.csv"
    summary_path = resolved_output_dir / "dataset_quality_summary.json"
    manifest_path = resolved_output_dir / "manifest.local.json"
    resolved_image_file_count = sum(1 for record in image_resolution_records if record["resolution_status"] != "image_file_missing")
    materialized_image_input_count = sum(1 for record in image_resolution_records if record["materialized_image_input"])

    summary = {
        **build_dataset_quality_summary(records, metric_rows),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "real_attack_registry_path": relative_or_absolute(resolved_registry_path, root_path),
        "dataset_quality_metrics_path": relative_or_absolute(metrics_path, root_path),
        "dataset_quality_image_resolution_records_path": relative_or_absolute(image_resolution_records_path, root_path),
        "dataset_quality_formal_feature_import_report_path": relative_or_absolute(
            formal_feature_import_report_path,
            root_path,
        ),
        "image_resolution_record_count": len(image_resolution_records),
        "resolved_image_file_count": resolved_image_file_count,
        "missing_image_file_count": len(image_resolution_records) - resolved_image_file_count,
        "materialized_image_input_count": materialized_image_input_count,
        "input_package_count": len(resolved_input_package_paths),
        "formal_feature_backend_ready": formal_feature_payload["report"]["formal_feature_backend_ready"],
        "formal_sample_scale_ready": formal_feature_payload["report"]["formal_sample_scale_ready"],
        "formal_min_sample_count": formal_min_sample_count,
    }

    records_path.write_text("".join(json_line(record.to_dict()) for record in records), encoding="utf-8")
    image_resolution_records_path.write_text(
        "".join(json_line(record) for record in image_resolution_records),
        encoding="utf-8",
    )
    formal_feature_import_report_path.write_text(
        stable_json_text(formal_feature_payload["report"]),
        encoding="utf-8",
    )
    write_csv(
        metrics_path,
        metric_rows,
        [
            "quality_metric_name",
            "quality_metric_value",
            "metric_status",
            "paper_metric_name",
            "feature_backend",
            "source_image_count",
            "comparison_image_count",
            "sample_pair_count",
            "supports_paper_claim",
        ],
    )
    summary_path.write_text(stable_json_text(summary), encoding="utf-8")

    input_paths = [relative_or_absolute(resolved_registry_path, root_path)] if resolved_registry_path.exists() else []
    input_paths.extend(relative_or_absolute(path, root_path) for path in resolved_input_package_paths)
    if resolved_formal_feature_records_path.exists():
        input_paths.append(relative_or_absolute(resolved_formal_feature_records_path, root_path))
    materialized_image_paths = tuple(
        resolve_path(root_path, record["resolved_image_path"])
        for record in image_resolution_records
        if record["materialized_image_input"] and record["resolved_image_path"]
    )
    output_paths = tuple(
        relative_or_absolute(path, root_path)
        for path in (
            records_path,
            image_resolution_records_path,
            formal_feature_import_report_path,
            metrics_path,
            summary_path,
            *materialized_image_paths,
            manifest_path,
        )
    )
    manifest = build_artifact_manifest(
        artifact_id="dataset_level_quality_manifest",
        artifact_type="local_manifest",
        input_paths=tuple(input_paths),
        output_paths=output_paths,
        config={
            "records_digest": build_stable_digest([record.to_dict() for record in records]),
            "image_resolution_records_digest": build_stable_digest(image_resolution_records),
            "formal_feature_import_report_digest": build_stable_digest(formal_feature_payload["report"]),
            "metric_rows_digest": build_stable_digest(metric_rows),
            "summary_digest": build_stable_digest(summary),
        },
        code_version=resolve_code_version(root_path),
        rebuild_command="python scripts/write_dataset_level_quality_outputs.py",
        metadata=summary,
    ).to_dict()
    manifest_path.write_text(stable_json_text(manifest), encoding="utf-8")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器。"""

    parser = argparse.ArgumentParser(description="写出数据集级质量证据产物。")
    parser.add_argument("--root", default=".", help="仓库根目录。")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="输出目录, 必须位于 outputs/ 下。")
    parser.add_argument("--real-attack-registry-path", default=str(DEFAULT_REAL_ATTACK_REGISTRY_PATH), help="真实攻击图像 registry 路径。")
    parser.add_argument(
        "--image-search-root",
        action="append",
        default=[],
        help="补充图像搜索根目录, 可重复传入; 适用于前序 outputs 已解包但不在当前仓库根目录下的情况。",
    )
    parser.add_argument(
        "--input-package-path",
        action="append",
        default=[],
        help="前序结果 ZIP 文件或包含 ZIP 的目录, 可重复传入; 脚本只会物化当前 records 需要的图像。",
    )
    parser.add_argument(
        "--formal-feature-records-path",
        default=None,
        help="可选的 Inception 特征 JSONL 记录路径; 缺失时正式 FID / KID 保持 unsupported。",
    )
    parser.add_argument(
        "--formal-min-sample-count",
        type=int,
        default=DEFAULT_FORMAL_MIN_SAMPLE_COUNT,
        help="正式 FID / KID 所需的最小图像对数量, 默认不适配小样本。",
    )
    return parser


def main() -> None:
    """命令行入口。"""

    args = build_parser().parse_args()
    manifest = write_dataset_level_quality_outputs(
        root=args.root,
        output_dir=args.output_dir,
        real_attack_registry_path=args.real_attack_registry_path,
        image_search_roots=args.image_search_root,
        input_package_paths=args.input_package_path,
        formal_feature_records_path=args.formal_feature_records_path,
        formal_min_sample_count=args.formal_min_sample_count,
    )
    print(stable_json_text(manifest), end="")


if __name__ == "__main__":
    main()
