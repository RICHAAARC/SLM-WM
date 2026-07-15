"""写出数据集级图像质量证据产物的正式实现。"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from typing import Any
import zipfile

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.protocol import (
    FORMAL_DATASET_QUALITY_ATTACK_NAME,
    FORMAL_DATASET_QUALITY_IMAGE_PAIR_ROLE,
    FORMAL_DATASET_QUALITY_METRIC_NAMES,
    FORMAL_FEATURE_BACKEND,
    FORMAL_FEATURE_EXTRACTOR_ID,
    build_dataset_quality_image_records,
    build_dataset_quality_metric_rows,
    build_dataset_quality_summary,
    formal_dataset_quality_metric_protocol,
)
from experiments.protocol.attack_conditioned_quality import (
    attack_quality_dataset_image_records,
    load_attack_conditioned_quality_estimand,
)
from experiments.protocol.attacks import default_attack_configs
from experiments.protocol.paper_run_config import (
    RUN_EXPECTED_PROMPT_COUNTS,
    build_paper_run_config,
    normalize_paper_run_name,
)
from experiments.protocol.splits import (
    apply_split_assignments,
    build_group_split_counts,
)
from experiments.artifacts.paired_quality_outputs import (
    as_dataset_quality_namespaces,
    build_paired_quality_metric_records,
    extract_formal_clip_feature_rows,
    validate_formal_clip_feature_rows,
    validate_paired_quality_metric_records,
)
from experiments.runtime import repository_environment
from experiments.runtime.scientific_execution_binding import (
    validate_scientific_execution_binding,
)
from experiments.runtime.package_input_manifest import (
    collect_exact_package_entries,
    validate_exact_package_archive,
    write_exact_package_input_manifest,
)
from experiments.runtime.resume_checkpoint import (
    clear_progress_checkpoints,
    persist_checkpoint_files,
    persist_progress_checkpoint,
    restore_role_checkpoints,
)
from experiments.runtime.dependency_profiles import (
    require_dependency_profile_ready,
)
from experiments.runtime.scientific_unit_provenance import (
    SCIENTIFIC_UNIT_PROVENANCE_AGGREGATE_FIELDS,
    aggregate_scientific_unit_provenance,
    build_scientific_unit_provenance,
    validate_scientific_unit_provenance,
)
from experiments.protocol.prompts import build_prompt_records, read_prompt_file
from experiments.artifacts.artifact_manifest import build_artifact_manifest
from experiments.artifacts.manifest_schema import manifest_config_digest_ready
from main.core.digest import build_stable_digest
from experiments.runtime.archive_naming import utc_archive_token

DEFAULT_PROGRESS_INTERVAL_ITEMS = 50
PACKAGE_INPUT_MANIFEST_FILE_NAME = "dataset_quality_package_input_manifest.json"
FORMAL_FEATURE_DIMENSION = int(
    formal_dataset_quality_metric_protocol()["feature_dimension"]
)
IMAGE_RESOLUTION_RECORD_FIELDNAMES = frozenset(
    {
        "requested_image_path",
        "resolved_image_path",
        "resolved_from_package_path",
        "resolution_status",
        "resolved_image_digest",
        "materialized_image_input",
        "supports_paper_claim",
        "image_resolution_record_digest",
        "image_resolution_record_id",
    }
)
DATASET_QUALITY_IMAGE_RECORD_FIELDNAMES = frozenset(
    {
        "dataset_quality_record_id",
        "dataset_quality_record_digest",
        "run_id",
        "prompt_id",
        "attack_name",
        "image_pair_index",
        "image_pair_role",
        "source_image_path",
        "source_image_digest",
        "comparison_image_path",
        "comparison_image_digest",
        "feature_backend",
        "supports_paper_claim",
    }
)


def dataset_quality_io_progress_enabled() -> bool:
    """判断是否输出数据集级质量 I/O 与正式特征进度。"""

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
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            return []
        normalized_item = float(item)
        if not math.isfinite(normalized_item):
            return []
        vector.append(normalized_item)
    return vector


def _inception_batch_config_digest(
    item_identity: list[dict[str, Any]],
) -> str:
    """绑定一个 Inception batch 实际消费的图像身份与算子配置."""

    return build_stable_digest(
        {
            "feature_backend": FORMAL_FEATURE_BACKEND,
            "feature_extractor_id": FORMAL_FEATURE_EXTRACTOR_ID,
            "item_identity": item_identity,
        }
    )


def _empty_scientific_unit_provenance_summary() -> dict[str, Any]:
    """返回缺少逐完成单元来源时的明确阻断摘要."""

    return {
        "scientific_unit_provenance_reference_count": 0,
        "scientific_unit_provenance_record_count": 0,
        "scientific_unit_provenance_records_digest": "",
        "scientific_unit_ids": [],
        "scientific_unit_config_digests": [],
        "scientific_execution_environment_digests": [],
        "scientific_dependency_profile_ids": [],
        "scientific_dependency_profile_digests": [],
        "scientific_complete_hash_lock_digests": [],
        "scientific_formal_execution_commits": [],
        "scientific_formal_execution_lock_digests": [],
        "scientific_torch_versions": [],
        "scientific_torch_cuda_versions": [],
        "scientific_execution_device_names": [],
        "scientific_cuda_device_names": [],
        "scientific_random_identity_digests_random": [],
        "scientific_unit_provenance_ready": False,
    }


def validate_inception_feature_provenance_groups(
    feature_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """把每个 feature batch 的来源配置绑定到组内精确图像集合."""

    grouped_rows: dict[str, list[dict[str, Any]]] = {}
    for row in feature_rows:
        provenance = row.get("scientific_unit_provenance")
        if not isinstance(provenance, dict):
            raise TypeError("Inception 特征行缺少科学完成单元来源")
        unit_id = str(provenance.get("scientific_unit_id", ""))
        grouped_rows.setdefault(unit_id, []).append(row)

    validated_references: list[dict[str, Any]] = []
    for unit_id, rows in grouped_rows.items():
        item_identity = [
            {
                "dataset_quality_record_id": row[
                    "dataset_quality_record_id"
                ],
                "dataset_quality_image_role": row[
                    "dataset_quality_image_role"
                ],
                "image_path": row["image_path"],
                "image_digest": row["image_digest"],
            }
            for row in rows
        ]
        batch_identity_digest = build_stable_digest(
            [
                (
                    identity["dataset_quality_record_id"],
                    identity["dataset_quality_image_role"],
                )
                for identity in item_identity
            ]
        )
        expected_unit_id = f"feature_batch_{batch_identity_digest[:16]}"
        if unit_id != expected_unit_id:
            raise ValueError("Inception feature batch 标识与组内图像身份不一致")
        expected_config_digest = _inception_batch_config_digest(item_identity)
        group_provenance = [
            validate_scientific_unit_provenance(
                row["scientific_unit_provenance"],
                expected_unit_id=expected_unit_id,
                expected_config_digest=expected_config_digest,
            )
            for row in rows
        ]
        if any(record != group_provenance[0] for record in group_provenance[1:]):
            raise ValueError("同一 Inception feature batch 包含冲突来源")
        validated_references.extend(group_provenance)
    return validated_references


def _is_sha256(value: Any) -> bool:
    """判断字段是否为规范小写 SHA-256."""

    text = str(value)
    return len(text) == 64 and all(
        character in "0123456789abcdef" for character in text
    )


def inspect_dataset_quality_image_resolution_identity(
    *,
    records: Any,
    image_resolution_records: Any,
    root_path: Path,
) -> tuple[
    dict[tuple[str, str], tuple[str, str]],
    list[dict[str, Any]],
]:
    """核验图像解析记录并返回 feature 行必须使用的实际路径与 SHA.

    通用写法是在特征导入前把声明路径解析为实际文件, 并重新读取文件计算
    SHA-256。这样 feature 行不能只复述图像 registry 中的声明摘要。项目特定
    约束是 source/comparison 两种角色必须精确覆盖每个正式质量记录。
    """

    materialized_records = tuple(records)
    materialized_resolutions = tuple(
        dict(record) for record in image_resolution_records
    )
    issues: list[dict[str, Any]] = []
    declared_by_key: dict[tuple[str, str], tuple[str, str]] = {}
    declared_digest_by_path: dict[str, str] = {}
    source_file_identities: set[str] = set()
    comparison_file_identities: set[str] = set()
    run_ids: set[str] = set()
    for record_index, record in enumerate(materialized_records):
        record_values = vars(record)
        record_id = str(getattr(record, "dataset_quality_record_id", ""))
        record_digest = str(
            getattr(record, "dataset_quality_record_digest", "")
        )
        record_payload = {
            field_name: getattr(record, field_name, None)
            for field_name in DATASET_QUALITY_IMAGE_RECORD_FIELDNAMES
            if field_name
            not in {
                "dataset_quality_record_id",
                "dataset_quality_record_digest",
            }
        }
        run_id = str(getattr(record, "run_id", ""))
        if (
            set(record_values) != DATASET_QUALITY_IMAGE_RECORD_FIELDNAMES
            or not _is_sha256(record_digest)
            or build_stable_digest(record_payload) != record_digest
            or record_id != f"dataset_quality_record_{record_digest[:16]}"
            or not run_id
            or run_id in run_ids
            or getattr(record, "attack_name", None)
            != FORMAL_DATASET_QUALITY_ATTACK_NAME
            or getattr(record, "image_pair_role", None)
            != FORMAL_DATASET_QUALITY_IMAGE_PAIR_ROLE
            or getattr(record, "supports_paper_claim", None) is not False
        ):
            issues.append(
                {
                    "row_index": record_index,
                    "field_name": "image_pair_role",
                    "reason": "formal_clean_to_watermarked_pair_required",
                }
            )
        run_ids.add(run_id)
        role_path_identities: dict[str, str] = {}
        for role in ("source", "comparison"):
            requested_path = str(
                getattr(record, f"{role}_image_path", "")
            )
            requested_file = (
                resolve_path(root_path, requested_path)
                if requested_path
                else None
            )
            requested_file_identity = (
                os.path.normcase(str(requested_file.resolve()))
                if requested_file is not None
                else ""
            )
            role_path_identities[role] = requested_file_identity
            declared_digest = str(
                getattr(record, f"{role}_image_digest", "")
            )
            key = (record_id, role)
            if (
                not record_id
                or not requested_path
                or not _is_sha256(declared_digest)
                or key in declared_by_key
            ):
                issues.append(
                    {
                        "row_index": record_index,
                        "field_name": f"{role}_image_path",
                        "reason": "dataset_quality_image_identity_invalid",
                    }
                )
                continue
            previous_digest = declared_digest_by_path.setdefault(
                requested_path,
                declared_digest,
            )
            if previous_digest != declared_digest:
                issues.append(
                    {
                        "row_index": record_index,
                        "field_name": f"{role}_image_digest",
                        "reason": "requested_image_path_digest_conflict",
                    }
                )
            declared_by_key[key] = (requested_path, declared_digest)
        source_identity = role_path_identities.get("source", "")
        comparison_identity = role_path_identities.get("comparison", "")
        if (
            not source_identity
            or not comparison_identity
            or source_identity == comparison_identity
            or source_identity in source_file_identities
            or comparison_identity in comparison_file_identities
            or source_identity in comparison_file_identities
            or comparison_identity in source_file_identities
        ):
            issues.append(
                {
                    "row_index": record_index,
                    "field_name": "source_image_path",
                    "reason": "independent_pair_file_identity_required",
                }
            )
        source_file_identities.add(source_identity)
        comparison_file_identities.add(comparison_identity)

    resolution_by_requested_path: dict[str, tuple[str, str]] = {}
    resolved_file_identities: set[str] = set()
    for row_index, resolution in enumerate(materialized_resolutions):
        requested_path = str(resolution.get("requested_image_path", ""))
        resolved_path = str(resolution.get("resolved_image_path", ""))
        declared_digest = str(resolution.get("resolved_image_digest", ""))
        digest_payload = {
            field_name: resolution[field_name]
            for field_name in IMAGE_RESOLUTION_RECORD_FIELDNAMES
            if field_name
            not in {
                "image_resolution_record_digest",
                "image_resolution_record_id",
            }
            and field_name in resolution
        }
        record_digest = str(
            resolution.get("image_resolution_record_digest", "")
        )
        resolved_file = resolve_path(root_path, resolved_path) if resolved_path else None
        actual_digest = (
            path_digest(resolved_file)
            if resolved_file is not None and resolved_file.is_file()
            else ""
        )
        resolved_file_identity = (
            os.path.normcase(str(resolved_file.resolve()))
            if resolved_file is not None
            else ""
        )
        if (
            set(resolution) != IMAGE_RESOLUTION_RECORD_FIELDNAMES
            or not requested_path
            or requested_path in resolution_by_requested_path
            or not resolved_path
            or resolution.get("resolution_status")
            not in {
                "resolved_existing_image_file",
                "materialized_from_input_package",
            }
            or resolution.get("supports_paper_claim") is not False
            or not isinstance(resolution.get("materialized_image_input"), bool)
            or not _is_sha256(declared_digest)
            or declared_digest != actual_digest
            or not resolved_file_identity
            or resolved_file_identity in resolved_file_identities
            or not _is_sha256(record_digest)
            or build_stable_digest(digest_payload) != record_digest
            or resolution.get("image_resolution_record_id")
            != f"dataset_quality_image_resolution_{record_digest[:16]}"
        ):
            issues.append(
                {
                    "row_index": row_index,
                    "field_name": "image_resolution_record_digest",
                    "reason": "image_resolution_record_or_actual_sha_invalid",
                }
            )
            continue
        resolution_by_requested_path[requested_path] = (
            resolved_path,
            declared_digest,
        )
        resolved_file_identities.add(resolved_file_identity)

    if (
        len(materialized_resolutions) != len(materialized_records) * 2
        or len(resolution_by_requested_path) != len(materialized_records) * 2
        or set(resolution_by_requested_path) != set(declared_digest_by_path)
    ):
        issues.append(
            {
                "row_index": -1,
                "field_name": "requested_image_path",
                "reason": "image_resolution_exact_path_coverage_required",
            }
        )

    expected_feature_identity: dict[
        tuple[str, str], tuple[str, str]
    ] = {}
    for key, (requested_path, declared_digest) in declared_by_key.items():
        resolution_identity = resolution_by_requested_path.get(requested_path)
        if (
            resolution_identity is None
            or resolution_identity[1] != declared_digest
        ):
            issues.append(
                {
                    "row_index": -1,
                    "field_name": "resolved_image_digest",
                    "reason": "image_resolution_declared_digest_mismatch",
                }
            )
            continue
        expected_feature_identity[key] = resolution_identity
    return expected_feature_identity, issues


def build_formal_feature_import_payload(
    *,
    records: Any,
    image_resolution_records: Any,
    feature_rows: list[dict[str, Any]],
    formal_feature_records_path: Path,
    root_path: Path,
    formal_min_sample_count: int,
    formal_feature_records_sha256: str,
) -> dict[str, Any]:
    """把外部 Inception 特征记录解析为正式 FID / KID 可消费的矩阵和导入报告。

    该函数属于 schema / 配置解析层: 它集中处理字段缺失、角色不合法和维度不一致问题, 下游指标函数只接收矩阵。
    """

    materialized_records = tuple(records)
    record_ids = [
        record.dataset_quality_record_id for record in materialized_records
    ]
    expected_pairs = {(record_id, role) for record_id in record_ids for role in ("source", "comparison")}
    expected_image_identity, resolution_issues = (
        inspect_dataset_quality_image_resolution_identity(
            records=materialized_records,
            image_resolution_records=image_resolution_records,
            root_path=root_path,
        )
    )
    image_resolution_identity_ready = bool(
        not resolution_issues
        and set(expected_image_identity) == expected_pairs
    )
    issues = list(resolution_issues)
    vectors_by_key: dict[tuple[str, str], list[float]] = {}
    try:
        provenance_references = validate_inception_feature_provenance_groups(
            feature_rows
        )
    except (KeyError, TypeError, ValueError) as error:
        provenance_references = []
        issues.append(
            {
                "row_index": -1,
                "field_name": "scientific_unit_provenance",
                "reason": f"scientific_unit_provenance_group_invalid:{error}",
            }
        )
    for row_index, row in enumerate(feature_rows):
        record_id = str(row.get("dataset_quality_record_id", "") or "")
        image_role = str(row.get("dataset_quality_image_role", "") or "")
        feature_backend = str(row.get("feature_backend", "") or "")
        feature_vector = _numeric_vector(row.get("feature_vector"))
        key = (record_id, image_role)
        expected_path, expected_digest = expected_image_identity.get(
            key,
            ("", ""),
        )
        if (
            str(row.get("image_path", "")) != expected_path
            or row.get("image_digest") != expected_digest
        ):
            issues.append(
                {
                    "row_index": row_index,
                    "field_name": "image_digest",
                    "reason": "feature_image_identity_mismatch",
                }
            )
            continue
        if key not in expected_pairs:
            issues.append({"row_index": row_index, "field_name": "dataset_quality_record_id", "reason": "record_role_not_expected"})
            continue
        if feature_backend != FORMAL_FEATURE_BACKEND:
            issues.append({"row_index": row_index, "field_name": "feature_backend", "reason": "inception_feature_backend_required"})
            continue
        if row.get("feature_extractor_id") != FORMAL_FEATURE_EXTRACTOR_ID:
            issues.append(
                {
                    "row_index": row_index,
                    "field_name": "feature_extractor_id",
                    "reason": "canonical_feature_extractor_required",
                }
            )
            continue
        if (
            row.get("feature_dimension") != FORMAL_FEATURE_DIMENSION
            or len(feature_vector) != FORMAL_FEATURE_DIMENSION
        ):
            issues.append(
                {
                    "row_index": row_index,
                    "field_name": "feature_dimension",
                    "reason": "formal_feature_dimension_required",
                }
            )
            continue
        if row.get("supports_paper_claim") is not False:
            issues.append(
                {
                    "row_index": row_index,
                    "field_name": "supports_paper_claim",
                    "reason": "raw_feature_must_not_claim_paper_support",
                }
            )
            continue
        if key in vectors_by_key:
            issues.append(
                {
                    "row_index": row_index,
                    "field_name": "dataset_quality_record_id",
                    "reason": "duplicate_record_role_feature",
                }
            )
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
    expected_pair_count = len(record_ids)
    try:
        provenance_summary = aggregate_scientific_unit_provenance(
            provenance_references,
            expected_reference_count=len(feature_rows),
        )
    except (TypeError, ValueError) as error:
        issues.append(
            {
                "row_index": -1,
                "field_name": "scientific_unit_provenance",
                "reason": f"scientific_unit_provenance_aggregate_invalid:{error}",
            }
        )
        provenance_summary = _empty_scientific_unit_provenance_summary()
    formal_feature_backend_ready = (
        accepted_pair_count == expected_pair_count
        and missing_feature_count == 0
        and len(feature_rows) == expected_pair_count * 2
        and provenance_summary["scientific_unit_provenance_ready"]
        and not issues
    )
    formal_sample_scale_ready = (
        formal_feature_backend_ready
        and accepted_pair_count == formal_min_sample_count
    )
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
        "formal_feature_records_sha256": formal_feature_records_sha256,
        "feature_backend": FORMAL_FEATURE_BACKEND,
        "formal_feature_record_count": len(feature_rows),
        "expected_feature_pair_count": expected_pair_count,
        "accepted_feature_pair_count": accepted_pair_count,
        "missing_feature_pair_count": missing_feature_count,
        "feature_issue_count": len(issues),
        "feature_dimension": feature_dimension_values[0] if dimension_consistent and feature_dimension_values else 0,
        "image_resolution_identity_ready": image_resolution_identity_ready,
        "formal_min_sample_count": formal_min_sample_count,
        "formal_feature_backend_ready": formal_feature_backend_ready,
        "formal_sample_scale_ready": formal_sample_scale_ready,
        "unsupported_reason": unsupported_reason,
        "issues": issues,
        **provenance_summary,
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


def canonical_prompt_ids_for_paper_run(
    *,
    root_path: Path,
    prompt_set: str,
    prompt_file: str | Path,
) -> tuple[str, ...]:
    """从当前论文配置读取唯一受治理 Prompt 标识集合。"""

    requested_path = resolve_path(root_path, prompt_file)
    packaged_path = (ROOT / prompt_file).resolve()
    resolved_path = requested_path if requested_path.is_file() else packaged_path
    prompt_texts = read_prompt_file(resolved_path)
    return tuple(
        record.prompt_id for record in build_prompt_records(prompt_set, prompt_texts)
    )


def dataset_quality_prompt_contract(
    registry_rows: list[dict[str, Any]],
    canonical_prompt_ids: tuple[str, ...],
) -> dict[str, Any]:
    """核验质量 registry 对当前 Prompt 集的一对一精确覆盖。"""

    actual_prompt_ids = tuple(str(row.get("prompt_id", "") or "") for row in registry_rows)
    canonical_set = set(canonical_prompt_ids)
    actual_set = set(actual_prompt_ids)
    duplicate_count = len(actual_prompt_ids) - len(actual_set)
    missing_ids = sorted(canonical_set - actual_set)
    unexpected_ids = sorted(actual_set - canonical_set)
    exact_ready = (
        len(actual_prompt_ids) == len(canonical_prompt_ids)
        and duplicate_count == 0
        and not missing_ids
        and not unexpected_ids
        and "" not in actual_set
    )
    return {
        "expected_prompt_count": len(canonical_prompt_ids),
        "registry_prompt_count": len(actual_prompt_ids),
        "duplicate_registry_prompt_id_count": duplicate_count,
        "missing_registry_prompt_id_count": len(missing_ids),
        "unexpected_registry_prompt_id_count": len(unexpected_ids),
        "canonical_prompt_id_digest": build_stable_digest(sorted(canonical_prompt_ids)),
        "registry_prompt_id_digest": build_stable_digest(sorted(actual_prompt_ids)),
        "prompt_registry_exact_set_ready": exact_ready,
    }


def write_canonical_formal_feature_records(
    path: Path,
    rows: list[dict[str, Any]],
) -> str:
    """把提取或导入的特征统一写入当前 run 的规范 JSONL 文件。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json_line(row) for row in rows), encoding="utf-8")
    return path_digest(path)


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


def extract_formal_inception_feature_rows(
    *,
    records: Any,
    root_path: Path,
    image_search_roots: tuple[Path, ...],
    output_path: Path,
    device_name: str | None = None,
    batch_size: int = 32,
) -> list[dict[str, Any]]:
    """直接从受治理图像提取 torch-fidelity Inception v3 兼容特征。

    该函数实现正式 FID / KID 所需的真实特征算子。它使用 torch-fidelity
    的 `inception-v3-compat` 和 2048 维特征层; 该实现使用从官方
    TensorFlow 权重转换的参数、TensorFlow 兼容双线性缩放和 uint8 输入。
    它不读取 pixel histogram 诊断特征, 生成的逐图像记录可单独审计和复算。
    """

    if batch_size <= 0:
        raise ValueError("Inception 特征 batch_size 必须为正整数")
    items: list[tuple[Any, str, Path, str]] = []
    image_digest_by_path: dict[Path, str] = {}
    for record in records:
        for image_role, path_text, image_digest in (
            ("source", record.source_image_path, record.source_image_digest),
            ("comparison", record.comparison_image_path, record.comparison_image_digest),
        ):
            resolved_path = resolve_existing_image_path(path_text, root_path, image_search_roots)
            if not resolved_path.is_file():
                raise FileNotFoundError(f"正式 Inception 特征缺少图像文件: {path_text}")
            resolved_path = resolved_path.resolve()
            actual_image_digest = image_digest_by_path.get(resolved_path)
            if actual_image_digest is None:
                actual_image_digest = path_digest(resolved_path)
                image_digest_by_path[resolved_path] = actual_image_digest
            if image_digest and image_digest != actual_image_digest:
                raise RuntimeError(
                    f"正式 Inception 特征图像摘要与实际文件不一致: {path_text}"
                )
            items.append(
                (record, image_role, resolved_path, actual_image_digest)
            )

    checkpoint_dir = output_path.parent / "inception_feature_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_path.parent / "inception_feature_progress.json"
    item_identity = [
        {
            "dataset_quality_record_id": record.dataset_quality_record_id,
            "dataset_quality_image_role": image_role,
            "image_path": relative_or_absolute(image_path, root_path),
            "image_digest": image_digest,
        }
        for record, image_role, image_path, image_digest in items
    ]
    context_path = checkpoint_dir / "feature_checkpoint_context.json"
    context = {
        "report_schema": "inception_feature_checkpoint_context",
        "schema_version": 1,
        "feature_backend": FORMAL_FEATURE_BACKEND,
        "feature_extractor_id": FORMAL_FEATURE_EXTRACTOR_ID,
        "item_count": len(item_identity),
        "item_identity_digest": build_stable_digest(item_identity),
        "formal_execution_lock": (
            repository_environment.require_published_formal_execution_lock(
                root_path
            )
        ),
        "evidence_eligibility": "intermediate_state_only",
        "supports_paper_claim": False,
    }
    if context_path.is_file():
        existing_context = json.loads(
            context_path.read_text(encoding="utf-8-sig")
        )
        if existing_context != context:
            raise RuntimeError("正式 Inception 特征检查点身份与当前运行不一致")
    else:
        temporary_context_path = context_path.with_name(
            context_path.name + ".partial"
        )
        temporary_context_path.write_text(
            stable_json_text(context),
            encoding="utf-8",
        )
        temporary_context_path.replace(context_path)

    expected_by_key = {
        (
            identity["dataset_quality_record_id"],
            identity["dataset_quality_image_role"],
        ): identity
        for identity in item_identity
    }
    rows_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for shard_path in sorted(checkpoint_dir.glob("feature_batch_*.jsonl")):
        shard_rows = [
            json.loads(line)
            for line in shard_path.read_text(encoding="utf-8-sig").splitlines()
            if line.strip()
        ]
        shard_item_identity: list[dict[str, Any]] = []
        for row in shard_rows:
            if not isinstance(row, dict):
                raise TypeError("正式 Inception 特征检查点行必须是 JSON object")
            key = (
                str(row.get("dataset_quality_record_id", "")),
                str(row.get("dataset_quality_image_role", "")),
            )
            expected = expected_by_key.get(key)
            feature_vector = row.get("feature_vector")
            normalized_feature_vector = _numeric_vector(feature_vector)
            ready = bool(
                expected
                and row.get("feature_backend") == FORMAL_FEATURE_BACKEND
                and row.get("feature_extractor_id")
                == FORMAL_FEATURE_EXTRACTOR_ID
                and row.get("feature_dimension") == 2048
                and isinstance(feature_vector, list)
                and len(normalized_feature_vector) == 2048
                and row.get("image_path") == expected["image_path"]
                and row.get("image_digest") == expected["image_digest"]
                and isinstance(row.get("scientific_unit_provenance"), dict)
                and row.get("supports_paper_claim") is False
            )
            if not ready:
                raise RuntimeError("正式 Inception 特征检查点内容未通过身份校验")
            existing = rows_by_key.get(key)
            if existing is not None and existing != row:
                raise RuntimeError("正式 Inception 特征检查点包含冲突记录")
            rows_by_key[key] = row
            shard_item_identity.append(
                {
                    "dataset_quality_record_id": row[
                        "dataset_quality_record_id"
                    ],
                    "dataset_quality_image_role": row[
                        "dataset_quality_image_role"
                    ],
                    "image_path": row["image_path"],
                    "image_digest": row["image_digest"],
                }
            )
        expected_batch_config_digest = _inception_batch_config_digest(
            shard_item_identity
        )
        expected_unit_id = shard_path.stem
        validated_provenance_records = [
            validate_scientific_unit_provenance(
                row["scientific_unit_provenance"],
                expected_unit_id=expected_unit_id,
                expected_config_digest=expected_batch_config_digest,
            )
            for row in shard_rows
        ]
        if (
            not validated_provenance_records
            or any(
                record != validated_provenance_records[0]
                for record in validated_provenance_records[1:]
            )
        ):
            raise RuntimeError("正式 Inception batch 来源记录不唯一")

    remaining_items = [
        item
        for item in items
        if (
            item[0].dataset_quality_record_id,
            item[1],
        )
        not in rows_by_key
    ]
    if remaining_items:
        import torch
        from PIL import Image
        from torch_fidelity.feature_extractor_inceptionv3 import (
            FeatureExtractorInceptionV3,
        )

        resolved_device = device_name or os.environ.get(
            "SLM_WM_INCEPTION_DEVICE",
            "cuda" if torch.cuda.is_available() else "cpu",
        )
        if resolved_device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("正式 Inception 特征提取要求的 CUDA 设备不可用")
        if not resolved_device.startswith("cuda"):
            raise RuntimeError("正式 Inception 特征提取必须在 CUDA 设备执行")
        feature_runtime_environment = (
            repository_environment.build_runtime_environment_report(
                "sd35_method_runtime_gpu",
                torch_module=torch,
                verified_formal_execution_lock=context[
                    "formal_execution_lock"
                ],
                repository_root=root_path,
            )
        )
        if feature_runtime_environment["dependency_environment_ready"] is not True:
            blockers = ",".join(
                feature_runtime_environment["dependency_readiness_blockers"]
            )
            raise RuntimeError(
                f"正式 Inception 特征依赖环境未通过门禁:{blockers}"
            )
        model = FeatureExtractorInceptionV3(
            "inception-v3-compat",
            ["2048"],
            verbose=True,
        ).to(resolved_device).eval()

    if remaining_items:
        with torch.inference_mode():
            for start in range(0, len(remaining_items), batch_size):
                batch_items = remaining_items[start : start + batch_size]
                tensors = []
                for _, _, image_path, _ in batch_items:
                    with Image.open(image_path) as image:
                        rgb = image.convert("RGB")
                        width, height = rgb.size
                        tensor = torch.frombuffer(bytearray(rgb.tobytes()), dtype=torch.uint8)
                        tensors.append(tensor.reshape(height, width, 3).permute(2, 0, 1))
                batch_shapes = {tuple(tensor.shape) for tensor in tensors}
                if len(batch_shapes) != 1:
                    raise ValueError("同一 Inception batch 中的图像尺寸必须一致")
                features = model(torch.stack(tensors).to(resolved_device))[0].float().cpu()
                if features.ndim != 2:
                    features = features.reshape(features.shape[0], -1)
                batch_rows = []
                batch_item_identity = [
                    {
                        "dataset_quality_record_id": record.dataset_quality_record_id,
                        "dataset_quality_image_role": image_role,
                        "image_path": relative_or_absolute(image_path, root_path),
                        "image_digest": image_digest,
                    }
                    for record, image_role, image_path, image_digest in batch_items
                ]
                batch_identity_digest = build_stable_digest(
                    [
                        (
                            identity["dataset_quality_record_id"],
                            identity["dataset_quality_image_role"],
                        )
                        for identity in batch_item_identity
                    ]
                )
                scientific_unit_id = (
                    f"feature_batch_{batch_identity_digest[:16]}"
                )
                scientific_unit_provenance = build_scientific_unit_provenance(
                    scientific_unit_id=scientific_unit_id,
                    scientific_unit_config_digest=_inception_batch_config_digest(
                        batch_item_identity
                    ),
                    runtime_environment=feature_runtime_environment,
                    execution_device_name=resolved_device,
                    torch_module=torch,
                    random_identity_random={
                        "feature_extraction_seed_random": (
                            "not_used_deterministic_eval"
                        )
                    },
                )
                for (record, image_role, image_path, image_digest), feature in zip(
                    batch_items,
                    features,
                ):
                    row = {
                        "dataset_quality_record_id": record.dataset_quality_record_id,
                        "dataset_quality_image_role": image_role,
                        "feature_backend": FORMAL_FEATURE_BACKEND,
                        "feature_extractor_id": FORMAL_FEATURE_EXTRACTOR_ID,
                        "feature_dimension": int(feature.numel()),
                        "image_path": relative_or_absolute(image_path, root_path),
                        "image_digest": image_digest,
                        "feature_vector": [float(value) for value in feature.tolist()],
                        "scientific_unit_provenance": scientific_unit_provenance,
                        "supports_paper_claim": False,
                    }
                    batch_rows.append(row)
                    rows_by_key[
                        (record.dataset_quality_record_id, image_role)
                    ] = row
                shard_path = (
                    checkpoint_dir
                    / f"feature_batch_{batch_identity_digest[:16]}.jsonl"
                )
                temporary_shard_path = shard_path.with_name(
                    shard_path.name + ".partial"
                )
                temporary_shard_path.write_text(
                    "".join(json_line(row) for row in batch_rows),
                    encoding="utf-8",
                )
                temporary_shard_path.replace(shard_path)
                persist_checkpoint_files(
                    repository_root=root_path,
                    artifact_role="dataset_level_quality",
                    paper_run_name=os.environ.get(
                        "SLM_WM_PAPER_RUN_NAME",
                        output_path.parent.name,
                    ),
                    checkpoint_kind="feature_batches",
                    checkpoint_id=shard_path.stem,
                    paths=(context_path, shard_path),
                )
                progress = {
                    "report_schema": "inception_feature_progress",
                    "schema_version": 1,
                    "expected_feature_record_count": len(items),
                    "completed_feature_record_count": len(rows_by_key),
                    "remaining_feature_record_count": len(items) - len(rows_by_key),
                    "protocol_decision": "resume_required",
                    "evidence_eligibility": "intermediate_state_only",
                    "supports_paper_claim": False,
                }
                temporary_progress_path = progress_path.with_name(
                    progress_path.name + ".partial"
                )
                temporary_progress_path.write_text(
                    stable_json_text(progress),
                    encoding="utf-8",
                )
                temporary_progress_path.replace(progress_path)
                persist_progress_checkpoint(
                    progress_path,
                    repository_root=root_path,
                    artifact_role="dataset_level_quality",
                    paper_run_name=os.environ.get(
                        "SLM_WM_PAPER_RUN_NAME",
                        output_path.parent.name,
                    ),
                )

    rows = [
        rows_by_key[
            (record.dataset_quality_record_id, image_role)
        ]
        for record, image_role, _, _ in items
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_output_path = output_path.with_name(output_path.name + ".partial")
    temporary_output_path.write_text(
        "".join(json_line(row) for row in rows),
        encoding="utf-8",
    )
    temporary_output_path.replace(output_path)
    progress_path.unlink(missing_ok=True)
    clear_progress_checkpoints(
        artifact_role="dataset_level_quality",
        paper_run_name=os.environ.get(
            "SLM_WM_PAPER_RUN_NAME",
            output_path.parent.name,
        ),
    )
    return rows


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
    all_requested_paths = tuple(
        dict.fromkeys(
            path_text
            for record in records
            for path_text in (
                record.source_image_path,
                record.comparison_image_path,
            )
            if path_text
        )
    )
    for requested_path in all_requested_paths:
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


def write_dataset_level_quality_outputs(
    *,
    paper_run_name: str,
    target_fpr: float,
    root: str | Path = ".",
    quality_image_registry_path: str | Path | None = None,
    attack_quality_registry_path: str | Path | None = None,
    image_search_roots: Any = (),
    input_package_paths: Any = (),
    formal_feature_records_path: str | Path | None = None,
    formal_min_sample_count: int | None = None,
    auto_extract_formal_features: bool = False,
    inception_device_name: str | None = None,
    inception_batch_size: int = 32,
    clip_device_name: str | None = None,
    clip_batch_size: int = 32,
) -> dict[str, Any]:
    """写出正式 Inception FID / KID records、metrics、summary 和 manifest。

    该入口只接受真实图像对和正式 Inception 特征。特征、样本规模或数值
    后端不满足协议时, 产物保留明确阻断状态而不生成替代指标。
    """

    root_path = Path(root).resolve()
    formal_execution_run_lock = (
        repository_environment.require_published_formal_execution_lock(root_path)
    )
    resolved_paper_run_name = normalize_paper_run_name(paper_run_name)
    resolved_target_fpr = float(target_fpr)
    if not 0.0 < resolved_target_fpr < 1.0:
        raise ValueError("target_fpr 必须位于 (0, 1)")
    paper_run = build_paper_run_config(root_path)
    if paper_run.run_name != resolved_paper_run_name or abs(
        resolved_target_fpr - paper_run.target_fpr
    ) > 1e-12:
        raise ValueError("数据集级质量身份必须与当前论文运行配置一致")
    expected_prompt_count = RUN_EXPECTED_PROMPT_COUNTS[resolved_paper_run_name]
    if paper_run.prompt_count != expected_prompt_count:
        raise ValueError("当前论文运行 Prompt 文件数量不符合70/700/7000精确协议")
    required_formal_sample_count = paper_run.dataset_level_quality_minimum_count
    resolved_formal_min_sample_count = (
        required_formal_sample_count
        if formal_min_sample_count is None
        else int(formal_min_sample_count)
    )
    if resolved_formal_min_sample_count != required_formal_sample_count:
        raise ValueError("formal_min_sample_count 必须等于当前论文层级完整 Prompt 数量")
    resolved_output_dir = ensure_output_dir_under_outputs(
        root_path,
        Path("outputs") / "dataset_level_quality" / resolved_paper_run_name,
    )
    restore_role_checkpoints(
        repository_root=root_path,
        artifact_role="dataset_level_quality",
        paper_run_name=resolved_paper_run_name,
        allowed_output_prefix=(
            f"outputs/dataset_level_quality/{resolved_paper_run_name}"
        ),
    )
    registry_path = quality_image_registry_path or (
        Path("outputs")
        / "image_only_dataset_runtime"
        / resolved_paper_run_name
        / "watermark_quality_image_registry.jsonl"
    )
    resolved_registry_path = resolve_path(root_path, registry_path)
    resolved_attack_quality_registry_path = resolve_path(
        root_path,
        attack_quality_registry_path
        or (
            Path("outputs")
            / "image_only_dataset_runtime"
            / resolved_paper_run_name
            / "attack_conditioned_quality_image_records.jsonl"
        ),
    )
    formal_feature_source_path = (
        resolve_path(root_path, formal_feature_records_path)
        if formal_feature_records_path
        else resolved_output_dir / "dataset_quality_formal_feature_records.jsonl"
    )
    resolved_formal_feature_records_path = (
        resolved_output_dir / "dataset_quality_formal_feature_records.jsonl"
    )
    resolved_input_package_paths = expand_input_package_paths(root_path, input_package_paths)
    explicit_image_search_roots = tuple(
        resolve_path(root_path, path) for path in normalize_path_values(image_search_roots)
    )
    registry_rows = read_jsonl_rows(resolved_registry_path)
    attack_quality_registry_rows = read_jsonl_rows(
        resolved_attack_quality_registry_path
    )
    canonical_prompt_ids = canonical_prompt_ids_for_paper_run(
        root_path=root_path,
        prompt_set=paper_run.prompt_set,
        prompt_file=paper_run.prompt_file,
    )
    prompt_contract = dataset_quality_prompt_contract(
        registry_rows,
        canonical_prompt_ids,
    )
    if not prompt_contract["prompt_registry_exact_set_ready"]:
        raise ValueError("数据集级质量 registry 必须一对一精确覆盖当前受治理 Prompt 集")

    records = build_dataset_quality_image_records(registry_rows, root_path)
    if len(records) != expected_prompt_count:
        raise ValueError("数据集级质量 records 数量必须恰好等于当前 Prompt 数量")
    attack_quality_pair_rows = attack_quality_dataset_image_records(
        attack_quality_registry_rows
    )
    attack_quality_pair_records = as_dataset_quality_namespaces(
        attack_quality_pair_rows
    )
    expected_attack_quality_record_count = (
        int(build_group_split_counts(expected_prompt_count)["test"])
        * sum(attack.enabled for attack in default_attack_configs())
    )
    if attack_quality_registry_rows and (
        len(attack_quality_registry_rows) != expected_attack_quality_record_count
        or len(attack_quality_pair_records)
        != expected_attack_quality_record_count
    ):
        raise ValueError("逐攻击四图质量记录未精确覆盖完整 test Prompt 与攻击集合")
    materialized_root = resolved_output_dir / "materialized_image_inputs"
    materialized_records = materialize_images_from_input_packages(
        records=(*records, *attack_quality_pair_records),
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
    formal_feature_rows = read_formal_feature_rows(formal_feature_source_path)
    formal_features_generated = False
    if auto_extract_formal_features and not formal_feature_rows:
        formal_feature_rows = extract_formal_inception_feature_rows(
            records=records,
            root_path=root_path,
            image_search_roots=all_image_search_roots,
            output_path=resolved_formal_feature_records_path,
            device_name=inception_device_name,
            batch_size=inception_batch_size,
        )
        formal_features_generated = True
    formal_feature_records_sha256 = write_canonical_formal_feature_records(
        resolved_formal_feature_records_path,
        formal_feature_rows,
    )
    formal_feature_payload = build_formal_feature_import_payload(
        records=records,
        image_resolution_records=image_resolution_records,
        feature_rows=formal_feature_rows,
        formal_feature_records_path=resolved_formal_feature_records_path,
        root_path=root_path,
        formal_min_sample_count=resolved_formal_min_sample_count,
        formal_feature_records_sha256=formal_feature_records_sha256,
    )
    formal_feature_payload["report"].update(
        {
            "paper_run_name": resolved_paper_run_name,
            "target_fpr": resolved_target_fpr,
            "canonical_prompt_id_digest": prompt_contract[
                "canonical_prompt_id_digest"
            ],
            "registry_prompt_id_digest": prompt_contract[
                "registry_prompt_id_digest"
            ],
            "prompt_registry_exact_set_ready": prompt_contract[
                "prompt_registry_exact_set_ready"
            ],
        }
    )
    scientific_unit_provenance_identity_ready = False
    if formal_feature_payload["report"].get(
        "scientific_unit_provenance_ready"
    ) is True:
        scientific_profile = require_dependency_profile_ready(
            "sd35_method_runtime_gpu",
            root_path / "configs" / "dependency_profile_registry.json",
        )
        scientific_unit_provenance_identity_ready = all(
            (
                formal_feature_payload["report"].get(
                    "scientific_dependency_profile_ids"
                )
                == ["sd35_method_runtime_gpu"],
                formal_feature_payload["report"].get(
                    "scientific_dependency_profile_digests"
                )
                == [scientific_profile.profile_digest],
                formal_feature_payload["report"].get(
                    "scientific_complete_hash_lock_digests"
                )
                == [scientific_profile.complete_hash_lock_digest],
                formal_feature_payload["report"].get(
                    "scientific_formal_execution_commits"
                )
                == [formal_execution_run_lock["formal_execution_commit"]],
                formal_feature_payload["report"].get(
                    "scientific_formal_execution_lock_digests"
                )
                == [formal_execution_run_lock["formal_execution_lock_digest"]],
            )
        )
    formal_feature_payload["report"][
        "scientific_unit_provenance_identity_ready"
    ] = scientific_unit_provenance_identity_ready
    if not scientific_unit_provenance_identity_ready:
        formal_feature_payload["source_features"] = None
        formal_feature_payload["comparison_features"] = None
        formal_feature_payload["report"]["formal_feature_backend_ready"] = False
        formal_feature_payload["report"]["formal_sample_scale_ready"] = False
        formal_feature_payload["report"][
            "unsupported_reason"
        ] = "scientific_unit_provenance_identity_invalid"
    metric_rows = build_dataset_quality_metric_rows(
        records,
        root_path,
        image_search_roots=all_image_search_roots,
        formal_source_features=formal_feature_payload["source_features"],
        formal_comparison_features=formal_feature_payload["comparison_features"],
        formal_min_sample_count=resolved_formal_min_sample_count,
    )

    attack_quality_output_dir = (
        resolved_output_dir / "attack_conditioned_quality"
    )
    attack_quality_output_dir.mkdir(parents=True, exist_ok=True)
    attack_quality_records_path = (
        attack_quality_output_dir
        / "attack_conditioned_quality_image_records.jsonl"
    )
    attack_quality_pair_records_path = (
        attack_quality_output_dir
        / "attack_conditioned_quality_pair_records.jsonl"
    )
    attack_quality_inception_feature_records_path = (
        attack_quality_output_dir
        / "attack_conditioned_quality_inception_feature_records.jsonl"
    )
    paired_quality_clip_feature_records_path = (
        attack_quality_output_dir / "paired_quality_clip_feature_records.jsonl"
    )
    paired_quality_metric_records_path = (
        attack_quality_output_dir / "paired_quality_metric_records.jsonl"
    )
    attack_quality_inception_feature_rows = read_formal_feature_rows(
        attack_quality_inception_feature_records_path
    )
    if (
        auto_extract_formal_features
        and attack_quality_pair_records
        and not attack_quality_inception_feature_rows
    ):
        attack_quality_inception_feature_rows = (
            extract_formal_inception_feature_rows(
                records=attack_quality_pair_records,
                root_path=root_path,
                image_search_roots=all_image_search_roots,
                output_path=attack_quality_inception_feature_records_path,
                device_name=inception_device_name,
                batch_size=inception_batch_size,
            )
        )
    write_canonical_formal_feature_records(
        attack_quality_inception_feature_records_path,
        attack_quality_inception_feature_rows,
    )
    paired_quality_clip_feature_rows = read_formal_feature_rows(
        paired_quality_clip_feature_records_path
    )
    all_paired_quality_records = (*records, *attack_quality_pair_records)
    if (
        auto_extract_formal_features
        and all_paired_quality_records
        and not paired_quality_clip_feature_rows
    ):
        paired_quality_clip_feature_rows = extract_formal_clip_feature_rows(
            records=all_paired_quality_records,
            root_path=root_path,
            image_search_roots=all_image_search_roots,
            output_path=paired_quality_clip_feature_records_path,
            device_name=clip_device_name,
            batch_size=clip_batch_size,
        )
    if paired_quality_clip_feature_rows:
        paired_quality_clip_feature_rows = list(
            validate_formal_clip_feature_rows(
                paired_quality_clip_feature_rows,
                expected_record_ids=(
                    record.dataset_quality_record_id
                    for record in all_paired_quality_records
                ),
                expected_code_version=formal_execution_run_lock[
                    "formal_execution_commit"
                ],
            )
        )
        paired_quality_metric_records = build_paired_quality_metric_records(
            records,
            attack_quality_pair_records,
            paired_quality_clip_feature_rows,
            randomization_repeat_id=paper_run.randomization_repeat_id,
            root_path=root_path,
            image_search_roots=all_image_search_roots,
        )
        paired_quality_metric_records = (
            validate_paired_quality_metric_records(
                paired_quality_metric_records,
                base_records=records,
                attack_records=attack_quality_pair_records,
                expected_randomization_repeat_id=(
                    paper_run.randomization_repeat_id
                ),
            )
        )
    else:
        paired_quality_metric_records = ()
    attack_quality_records_path.write_text(
        "".join(json_line(row) for row in attack_quality_registry_rows),
        encoding="utf-8",
    )
    attack_quality_pair_records_path.write_text(
        "".join(json_line(row) for row in attack_quality_pair_rows),
        encoding="utf-8",
    )
    paired_quality_clip_feature_records_path.write_text(
        "".join(json_line(row) for row in paired_quality_clip_feature_rows),
        encoding="utf-8",
    )
    paired_quality_metric_records_path.write_text(
        "".join(json_line(row) for row in paired_quality_metric_records),
        encoding="utf-8",
    )
    if attack_quality_inception_feature_rows:
        validate_inception_feature_provenance_groups(
            attack_quality_inception_feature_rows
        )
    attack_feature_keys = {
        (
            str(row.get("dataset_quality_record_id", "")),
            str(row.get("dataset_quality_image_role", "")),
        )
        for row in attack_quality_inception_feature_rows
    }
    expected_attack_feature_keys = {
        (record.dataset_quality_record_id, role)
        for record in attack_quality_pair_records
        for role in ("source", "comparison")
    }
    attack_conditioned_quality_component_ready = bool(
        attack_quality_registry_rows
        and len(attack_quality_registry_rows)
        == expected_attack_quality_record_count
        and attack_feature_keys == expected_attack_feature_keys
        and len(paired_quality_metric_records)
        == expected_prompt_count + expected_attack_quality_record_count
        and len(paired_quality_clip_feature_rows)
        == 2 * (expected_prompt_count + expected_attack_quality_record_count)
    )


    records_path = resolved_output_dir / "dataset_quality_image_records.jsonl"
    image_resolution_records_path = resolved_output_dir / "dataset_quality_image_resolution_records.jsonl"
    formal_feature_import_report_path = resolved_output_dir / "dataset_quality_formal_feature_import_report.json"
    metrics_path = resolved_output_dir / "dataset_quality_metrics.csv"
    summary_path = resolved_output_dir / "dataset_quality_summary.json"
    manifest_path = resolved_output_dir / "manifest.local.json"
    resolved_image_file_count = sum(1 for record in image_resolution_records if record["resolution_status"] != "image_file_missing")
    materialized_image_input_count = sum(1 for record in image_resolution_records if record["materialized_image_input"])
    image_resolution_identity_ready = bool(
        formal_feature_payload["report"][
            "image_resolution_identity_ready"
        ]
        and resolved_image_file_count == len(image_resolution_records)
    )

    base_summary = build_dataset_quality_summary(records, metric_rows)
    formal_feature_extractor_ids = sorted(
        {
            str(row.get("feature_extractor_id"))
            for row in formal_feature_rows
            if row.get("feature_extractor_id")
        }
    )
    canonical_formal_feature_extractor_ready = formal_feature_extractor_ids == [FORMAL_FEATURE_EXTRACTOR_ID]
    formal_component_ready = bool(
        base_summary.get("formal_fid_kid_component_ready", False)
        and canonical_formal_feature_extractor_ready
        and prompt_contract["prompt_registry_exact_set_ready"]
        and scientific_unit_provenance_identity_ready
        and image_resolution_identity_ready
        and formal_feature_payload["report"]["accepted_feature_pair_count"]
        == expected_prompt_count
        and formal_feature_payload["report"]["missing_feature_pair_count"] == 0
        and formal_feature_payload["report"]["feature_issue_count"] == 0
        and formal_feature_payload["report"]["formal_feature_record_count"]
        == expected_prompt_count * 2
    )
    summary = {
        **base_summary,
        **prompt_contract,
        **{
            field_name: formal_feature_payload["report"][field_name]
            for field_name in (
                "formal_feature_record_count",
                "expected_feature_pair_count",
                "accepted_feature_pair_count",
                "missing_feature_pair_count",
                "feature_issue_count",
                "formal_feature_records_sha256",
            )
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "paper_run_name": resolved_paper_run_name,
        "target_fpr": resolved_target_fpr,
        "randomization_repeat_identity": {
            "randomization_repeat_id": paper_run.randomization_repeat_id,
            "generation_seed_index": paper_run.generation_seed_index,
            "generation_seed_offset": paper_run.generation_seed_offset,
            "watermark_key_index": paper_run.watermark_key_index,
            "formal_randomization_protocol_digest": (
                paper_run.formal_randomization_protocol_digest
            ),
        },
        "quality_image_registry_path": relative_or_absolute(
            resolved_registry_path,
            root_path,
        ),
        "attack_conditioned_quality_registry_path": relative_or_absolute(
            resolved_attack_quality_registry_path,
            root_path,
        ),
        "attack_conditioned_quality_image_records_path": (
            relative_or_absolute(attack_quality_records_path, root_path)
        ),
        "attack_conditioned_quality_pair_records_path": (
            relative_or_absolute(
                attack_quality_pair_records_path,
                root_path,
            )
        ),
        "attack_conditioned_quality_inception_feature_records_path": (
            relative_or_absolute(
                attack_quality_inception_feature_records_path,
                root_path,
            )
        ),
        "paired_quality_clip_feature_records_path": relative_or_absolute(
            paired_quality_clip_feature_records_path,
            root_path,
        ),
        "paired_quality_metric_records_path": relative_or_absolute(
            paired_quality_metric_records_path,
            root_path,
        ),
        "attack_conditioned_quality_estimand": (
            load_attack_conditioned_quality_estimand()
        ),
        "expected_attack_conditioned_quality_record_count": (
            expected_attack_quality_record_count
        ),
        "attack_conditioned_quality_record_count": len(
            attack_quality_registry_rows
        ),
        "attack_conditioned_quality_feature_record_count": len(
            attack_quality_inception_feature_rows
        ),
        "paired_quality_clip_feature_record_count": len(
            paired_quality_clip_feature_rows
        ),
        "paired_quality_metric_record_count": len(
            paired_quality_metric_records
        ),
        "attack_conditioned_quality_component_ready": (
            attack_conditioned_quality_component_ready
        ),
        "dataset_quality_metrics_path": relative_or_absolute(metrics_path, root_path),
        "dataset_quality_formal_metrics_path": relative_or_absolute(metrics_path, root_path),
        "dataset_quality_image_resolution_records_path": relative_or_absolute(image_resolution_records_path, root_path),
        "dataset_quality_formal_feature_import_report_path": relative_or_absolute(
            formal_feature_import_report_path,
            root_path,
        ),
        "dataset_quality_formal_feature_records_path": relative_or_absolute(
            resolved_formal_feature_records_path,
            root_path,
        ),
        "image_resolution_record_count": len(image_resolution_records),
        "resolved_image_file_count": resolved_image_file_count,
        "missing_image_file_count": len(image_resolution_records) - resolved_image_file_count,
        "materialized_image_input_count": materialized_image_input_count,
        "image_resolution_identity_ready": image_resolution_identity_ready,
        "input_package_count": len(resolved_input_package_paths),
        "formal_feature_backend_ready": formal_feature_payload["report"]["formal_feature_backend_ready"],
        "formal_sample_scale_ready": formal_feature_payload["report"]["formal_sample_scale_ready"],
        "formal_min_sample_count": resolved_formal_min_sample_count,
        "formal_feature_origin": (
            "direct_torch_fidelity_inception_v3_compat_extraction"
            if formal_features_generated
            else "governed_feature_record_import"
        ),
        "formal_feature_extractor_ids": formal_feature_extractor_ids,
        "canonical_formal_feature_extractor_ready": canonical_formal_feature_extractor_ready,
        "scientific_unit_provenance_identity_ready": (
            scientific_unit_provenance_identity_ready
        ),
        **{
            field_name: formal_feature_payload["report"][field_name]
            for field_name in SCIENTIFIC_UNIT_PROVENANCE_AGGREGATE_FIELDS
        },
        "formal_fid_kid_component_ready": formal_component_ready,
        "formal_fid_kid_component_blocker": (
            ""
            if formal_component_ready
            else (
                "requires_canonical_inception_feature_extractor"
                if base_summary.get("formal_fid_kid_ready", False)
                else base_summary.get("formal_fid_kid_component_blocker", "formal_fid_kid_not_measured")
            )
        ),
        "repeat_component_ready": bool(
            formal_component_ready
            and attack_conditioned_quality_component_ready
        ),
        "randomization_aggregate_ready": False,
        "supports_paper_claim": False,
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
    if resolved_attack_quality_registry_path.exists():
        input_paths.append(
            relative_or_absolute(
                resolved_attack_quality_registry_path,
                root_path,
            )
        )
    input_paths.extend(relative_or_absolute(path, root_path) for path in resolved_input_package_paths)
    if formal_feature_source_path.exists() and formal_feature_source_path != resolved_formal_feature_records_path:
        input_paths.append(relative_or_absolute(formal_feature_source_path, root_path))
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
            resolved_formal_feature_records_path,
            attack_quality_records_path,
            attack_quality_pair_records_path,
            attack_quality_inception_feature_records_path,
            paired_quality_clip_feature_records_path,
            paired_quality_metric_records_path,
            metrics_path,
            summary_path,
            *materialized_image_paths,
            manifest_path,
        )
    )
    formal_execution_run_lock = repository_environment.validate_formal_execution_lock_pair(
        formal_execution_run_lock,
        repository_environment.require_published_formal_execution_lock(root_path),
        formal_execution_run_lock["formal_execution_commit"],
    )
    manifest = build_artifact_manifest(
        artifact_id="dataset_level_quality_manifest",
        artifact_type="local_manifest",
        input_paths=tuple(input_paths),
        output_paths=output_paths,
        config={
            "paper_run_name": resolved_paper_run_name,
            "target_fpr": resolved_target_fpr,
            "randomization_repeat_id": paper_run.randomization_repeat_id,
            "generation_seed_index": paper_run.generation_seed_index,
            "generation_seed_offset": paper_run.generation_seed_offset,
            "watermark_key_index": paper_run.watermark_key_index,
            "formal_randomization_protocol_digest": (
                paper_run.formal_randomization_protocol_digest
            ),
            "randomization_repeat_identity": {
                "randomization_repeat_id": paper_run.randomization_repeat_id,
                "generation_seed_index": paper_run.generation_seed_index,
                "generation_seed_offset": paper_run.generation_seed_offset,
                "watermark_key_index": paper_run.watermark_key_index,
                "formal_randomization_protocol_digest": (
                    paper_run.formal_randomization_protocol_digest
                ),
            },
            "records_digest": build_stable_digest([record.to_dict() for record in records]),
            "attack_conditioned_quality_records_digest": build_stable_digest(
                attack_quality_registry_rows
            ),
            "attack_conditioned_quality_pair_records_digest": (
                build_stable_digest(attack_quality_pair_rows)
            ),
            "attack_conditioned_quality_inception_feature_records_digest": (
                build_stable_digest(
                    attack_quality_inception_feature_rows
                )
            ),
            "paired_quality_clip_feature_records_digest": build_stable_digest(
                paired_quality_clip_feature_rows
            ),
            "paired_quality_metric_records_digest": build_stable_digest(
                paired_quality_metric_records
            ),
            "attack_conditioned_quality_component_ready": (
                attack_conditioned_quality_component_ready
            ),
            "image_resolution_records_digest": build_stable_digest(image_resolution_records),
            "formal_feature_import_report_digest": build_stable_digest(formal_feature_payload["report"]),
            "formal_feature_records_sha256": formal_feature_records_sha256,
            "metric_rows_digest": build_stable_digest(metric_rows),
            "summary_digest": build_stable_digest(summary),
            **prompt_contract,
            "accepted_feature_pair_count": formal_feature_payload["report"]["accepted_feature_pair_count"],
            "missing_feature_pair_count": formal_feature_payload["report"]["missing_feature_pair_count"],
            "feature_issue_count": formal_feature_payload["report"]["feature_issue_count"],
            "formal_feature_record_count": formal_feature_payload["report"]["formal_feature_record_count"],
            "auto_extract_formal_features": auto_extract_formal_features,
            "formal_features_generated": formal_features_generated,
        },
        code_version=formal_execution_run_lock["formal_execution_commit"],
        rebuild_command=(
            "python -m experiments.artifacts.dataset_level_quality_outputs"
        ),
        metadata=summary,
    ).to_dict()
    manifest["formal_execution_run_lock"] = formal_execution_run_lock
    manifest_path.write_text(stable_json_text(manifest), encoding="utf-8")
    return manifest


def package_dataset_level_quality_outputs(
    paper_run_name: str,
    root: str | Path = ".",
) -> Path:
    """打包当前论文运行层级的正式 FID / KID 证据。"""

    root_path = Path(root).resolve()
    formal_execution_package_lock = (
        repository_environment.require_published_formal_execution_lock(root_path)
    )
    resolved_paper_run_name = normalize_paper_run_name(paper_run_name)
    paper_run = build_paper_run_config(root_path)
    if paper_run.run_name != resolved_paper_run_name:
        raise ValueError("数据集级质量打包层级必须与当前论文配置一致")
    source_dir = (
        root_path / "outputs" / "dataset_level_quality" / resolved_paper_run_name
    )
    required_paths = tuple(
        source_dir / filename
        for filename in (
            "dataset_quality_image_records.jsonl",
            "dataset_quality_image_resolution_records.jsonl",
            "dataset_quality_formal_feature_records.jsonl",
            "dataset_quality_formal_feature_import_report.json",
            "dataset_quality_metrics.csv",
            "dataset_quality_summary.json",
            "attack_conditioned_quality/attack_conditioned_quality_image_records.jsonl",
            "attack_conditioned_quality/attack_conditioned_quality_pair_records.jsonl",
            "attack_conditioned_quality/attack_conditioned_quality_inception_feature_records.jsonl",
            "attack_conditioned_quality/paired_quality_clip_feature_records.jsonl",
            "attack_conditioned_quality/paired_quality_metric_records.jsonl",
            "manifest.local.json",
        )
    )
    if any(not path.is_file() for path in required_paths):
        raise FileNotFoundError("当前论文运行层级的数据集级质量输出不完整, 不得打包")
    summary = json.loads((source_dir / "dataset_quality_summary.json").read_text(encoding="utf-8-sig"))
    feature_report = json.loads(
        (source_dir / "dataset_quality_formal_feature_import_report.json").read_text(
            encoding="utf-8-sig"
        )
    )
    manifest = json.loads((source_dir / "manifest.local.json").read_text(encoding="utf-8-sig"))
    formal_execution_run_lock = repository_environment.validate_formal_execution_lock_pair(
        manifest.get("formal_execution_run_lock"),
        formal_execution_package_lock,
        manifest.get("code_version"),
    )
    with (source_dir / "dataset_quality_metrics.csv").open(
        encoding="utf-8-sig",
        newline="",
    ) as stream:
        metric_rows = list(csv.DictReader(stream))
    expected_prompt_ids = canonical_prompt_ids_for_paper_run(
        root_path=root_path,
        prompt_set=paper_run.prompt_set,
        prompt_file=paper_run.prompt_file,
    )
    expected_prompt_count = RUN_EXPECTED_PROMPT_COUNTS[resolved_paper_run_name]
    expected_prompt_digest = build_stable_digest(sorted(expected_prompt_ids))
    feature_records_path = source_dir / "dataset_quality_formal_feature_records.jsonl"
    feature_records_sha256 = path_digest(feature_records_path)
    packaged_feature_rows = read_formal_feature_rows(feature_records_path)
    validate_inception_feature_provenance_groups(packaged_feature_rows)
    packaged_quality_records = tuple(
        SimpleNamespace(**row)
        for row in read_jsonl_rows(
            source_dir / "dataset_quality_image_records.jsonl"
        )
    )
    packaged_image_resolution_records = read_jsonl_rows(
        source_dir / "dataset_quality_image_resolution_records.jsonl"
    )
    revalidated_feature_payload = build_formal_feature_import_payload(
        records=packaged_quality_records,
        image_resolution_records=packaged_image_resolution_records,
        feature_rows=packaged_feature_rows,
        formal_feature_records_path=feature_records_path,
        root_path=root_path,
        formal_min_sample_count=expected_prompt_count,
        formal_feature_records_sha256=feature_records_sha256,
    )
    packaged_scientific_unit_provenance = aggregate_scientific_unit_provenance(
        (
            row["scientific_unit_provenance"]
            for row in packaged_feature_rows
        ),
        expected_reference_count=expected_prompt_count * 2,
    )
    scientific_unit_provenance_summary_bound = all(
        summary.get(field_name)
        == packaged_scientific_unit_provenance[field_name]
        and feature_report.get(field_name)
        == packaged_scientific_unit_provenance[field_name]
        for field_name in SCIENTIFIC_UNIT_PROVENANCE_AGGREGATE_FIELDS
    )
    manifest_config = manifest.get("config", {})
    attack_quality_dir = source_dir / "attack_conditioned_quality"
    packaged_attack_quality_records = read_jsonl_rows(
        attack_quality_dir / "attack_conditioned_quality_image_records.jsonl"
    )
    packaged_attack_quality_pair_rows = read_jsonl_rows(
        attack_quality_dir / "attack_conditioned_quality_pair_records.jsonl"
    )
    rebuilt_attack_quality_pair_rows = attack_quality_dataset_image_records(
        packaged_attack_quality_records
    )
    packaged_attack_quality_pairs = as_dataset_quality_namespaces(
        packaged_attack_quality_pair_rows
    )
    packaged_attack_inception_rows = read_formal_feature_rows(
        attack_quality_dir
        / "attack_conditioned_quality_inception_feature_records.jsonl"
    )
    packaged_clip_rows = read_formal_feature_rows(
        attack_quality_dir / "paired_quality_clip_feature_records.jsonl"
    )
    packaged_paired_metric_rows = read_jsonl_rows(
        attack_quality_dir / "paired_quality_metric_records.jsonl"
    )
    expected_test_prompt_ids = tuple(
        record.prompt_id
        for record in apply_split_assignments(
            build_prompt_records(
                paper_run.prompt_set,
                read_prompt_file(root_path / paper_run.prompt_file),
            )
        )
        if record.split == "test"
    )
    registered_attack_ids = tuple(
        sorted(
            attack.attack_id
            for attack in default_attack_configs()
            if attack.enabled
        )
    )
    expected_attack_keys = {
        (prompt_id, attack_id)
        for prompt_id in expected_test_prompt_ids
        for attack_id in registered_attack_ids
    }
    actual_attack_keys = {
        (str(row.get("prompt_id", "")), str(row.get("attack_id", "")))
        for row in packaged_attack_quality_pair_rows
    }
    expected_attack_feature_keys = {
        (str(row["dataset_quality_record_id"]), role)
        for row in packaged_attack_quality_pair_rows
        for role in ("source", "comparison")
    }
    actual_attack_feature_keys = {
        (
            str(row.get("dataset_quality_record_id", "")),
            str(row.get("dataset_quality_image_role", "")),
        )
        for row in packaged_attack_inception_rows
    }
    if packaged_attack_inception_rows:
        validate_inception_feature_provenance_groups(
            packaged_attack_inception_rows
        )
    packaged_clip_rows = list(
        validate_formal_clip_feature_rows(
            packaged_clip_rows,
            expected_record_ids=(
                record.dataset_quality_record_id
                for record in (
                    *packaged_quality_records,
                    *packaged_attack_quality_pairs,
                )
            ),
            expected_code_version=str(manifest.get("code_version", "")),
        )
    )
    packaged_paired_metric_rows = list(
        validate_paired_quality_metric_records(
            packaged_paired_metric_rows,
            base_records=packaged_quality_records,
            attack_records=packaged_attack_quality_pairs,
            expected_randomization_repeat_id=(
                paper_run.randomization_repeat_id
            ),
        )
    )
    attack_quality_contract_ready = all(
        (
            packaged_attack_quality_pair_rows
            == list(rebuilt_attack_quality_pair_rows),
            actual_attack_keys == expected_attack_keys,
            actual_attack_feature_keys == expected_attack_feature_keys,
            len(packaged_attack_quality_records) == len(expected_attack_keys),
            len(packaged_paired_metric_rows)
            == expected_prompt_count + len(expected_attack_keys),
            len(packaged_clip_rows)
            == 2 * (expected_prompt_count + len(expected_attack_keys)),
            summary.get("attack_conditioned_quality_component_ready") is True,
            manifest_config.get("attack_conditioned_quality_component_ready")
            is True,
            manifest_config.get("attack_conditioned_quality_records_digest")
            == build_stable_digest(packaged_attack_quality_records),
            manifest_config.get(
                "attack_conditioned_quality_pair_records_digest"
            )
            == build_stable_digest(packaged_attack_quality_pair_rows),
            manifest_config.get(
                "attack_conditioned_quality_inception_feature_records_digest"
            )
            == build_stable_digest(packaged_attack_inception_rows),
            manifest_config.get("paired_quality_clip_feature_records_digest")
            == build_stable_digest(packaged_clip_rows),
            manifest_config.get("paired_quality_metric_records_digest")
            == build_stable_digest(packaged_paired_metric_rows),
        )
    )
    expected_metric_protocol = formal_dataset_quality_metric_protocol()
    expected_kid_effective_subset_size = min(
        int(expected_metric_protocol["kid_subset_size"]),
        expected_prompt_count,
    )
    image_resolution_records_digest = build_stable_digest(
        packaged_image_resolution_records
    )
    image_resolution_contract_ready = bool(
        packaged_image_resolution_records
        and summary.get("image_resolution_identity_ready") is True
        and int(summary.get("image_resolution_record_count", -1))
        == len(packaged_image_resolution_records)
        and int(summary.get("resolved_image_file_count", -1))
        == len(packaged_image_resolution_records)
        and int(summary.get("missing_image_file_count", -1)) == 0
        and manifest_config.get("image_resolution_records_digest")
        == image_resolution_records_digest
        and revalidated_feature_payload["report"][
            "image_resolution_identity_ready"
        ]
        is True
    )
    metric_contract_ready = (
        len(metric_rows) == len(FORMAL_DATASET_QUALITY_METRIC_NAMES)
        and [row.get("quality_metric_name") for row in metric_rows]
        == list(FORMAL_DATASET_QUALITY_METRIC_NAMES)
        and [row.get("paper_metric_name") for row in metric_rows]
        == list(FORMAL_DATASET_QUALITY_METRIC_NAMES)
        and all(row.get("metric_status") == "measured" for row in metric_rows)
        and all(
            float(row.get("quality_metric_value", -1.0)) >= 0.0
            for row in metric_rows
            if row.get("quality_metric_name") in {"fid", "kid_std"}
        )
        and all(
            int(row.get(field_name, -1)) == expected_prompt_count
            for row in metric_rows
            for field_name in (
                "source_image_count",
                "comparison_image_count",
                "sample_pair_count",
            )
        )
    )
    feature_contract_ready = all(
        (
            paper_run.prompt_count == expected_prompt_count,
            summary.get("prompt_registry_exact_set_ready") is True,
            summary.get("canonical_prompt_id_digest") == expected_prompt_digest,
            summary.get("registry_prompt_id_digest") == expected_prompt_digest,
            int(summary.get("expected_prompt_count", -1)) == expected_prompt_count,
            int(summary.get("registry_prompt_count", -1)) == expected_prompt_count,
            int(summary.get("sample_pair_count", -1)) == expected_prompt_count,
            int(summary.get("accepted_feature_pair_count", -1)) == expected_prompt_count,
            int(summary.get("missing_feature_pair_count", -1)) == 0,
            int(summary.get("feature_issue_count", -1)) == 0,
            int(summary.get("formal_feature_record_count", -1)) == expected_prompt_count * 2,
            summary.get("formal_feature_records_sha256") == feature_records_sha256,
            feature_report.get("paper_run_name") == resolved_paper_run_name,
            feature_report.get("prompt_registry_exact_set_ready") is True,
            feature_report.get("canonical_prompt_id_digest") == expected_prompt_digest,
            feature_report.get("registry_prompt_id_digest") == expected_prompt_digest,
            int(feature_report.get("expected_feature_pair_count", -1)) == expected_prompt_count,
            int(feature_report.get("accepted_feature_pair_count", -1)) == expected_prompt_count,
            int(feature_report.get("missing_feature_pair_count", -1)) == 0,
            int(feature_report.get("feature_issue_count", -1)) == 0,
            int(feature_report.get("formal_feature_record_count", -1)) == expected_prompt_count * 2,
            feature_report.get("formal_feature_records_sha256") == feature_records_sha256,
            manifest_config.get("canonical_prompt_id_digest") == expected_prompt_digest,
            manifest_config.get("registry_prompt_id_digest") == expected_prompt_digest,
            manifest_config.get("prompt_registry_exact_set_ready") is True,
            manifest_config.get("formal_feature_records_sha256") == feature_records_sha256,
            metric_contract_ready,
        )
    )
    metric_protocol_ready = bool(
        summary.get("formal_metric_protocol") == expected_metric_protocol
        and summary.get("formal_metric_protocol_digest")
        == expected_metric_protocol["formal_metric_protocol_digest"]
        and int(summary.get("kid_effective_subset_size", -1))
        == expected_kid_effective_subset_size
    )
    if not all(
        (
            summary.get("paper_run_name") == resolved_paper_run_name,
            abs(float(summary.get("target_fpr", -1.0)) - paper_run.target_fpr) <= 1e-12,
            bool(summary.get("generated_at")),
            summary.get("formal_feature_backend_ready") is True,
            summary.get("formal_sample_scale_ready") is True,
            summary.get("canonical_formal_feature_extractor_ready") is True,
            summary.get("scientific_unit_provenance_ready") is True,
            summary.get("scientific_unit_provenance_identity_ready") is True,
            summary.get("image_resolution_identity_ready") is True,
            int(summary.get("image_resolution_record_count", -1)) > 0,
            int(summary.get("resolved_image_file_count", -1))
            == int(summary.get("image_resolution_record_count", -1)),
            int(summary.get("missing_image_file_count", -1)) == 0,
            summary.get("scientific_unit_provenance_reference_count")
            == paper_run.prompt_count * 2,
            bool(summary.get("scientific_unit_provenance_records_digest")),
            scientific_unit_provenance_summary_bound,
            revalidated_feature_payload["report"][
                "formal_feature_backend_ready"
            ]
            is True,
            revalidated_feature_payload["report"][
                "formal_sample_scale_ready"
            ]
            is True,
            summary.get("formal_fid_kid_component_ready") is True,
            summary.get("repeat_component_ready") is True,
            summary.get("randomization_aggregate_ready") is False,
            summary.get("supports_paper_claim") is False,
            manifest.get("artifact_id") == "dataset_level_quality_manifest",
            manifest_config_digest_ready(manifest),
            feature_contract_ready,
            image_resolution_contract_ready,
            metric_protocol_ready,
            attack_quality_contract_ready,
        )
    ):
        raise RuntimeError("数据集级质量身份、精确 Prompt/特征覆盖或 ready 门禁未通过")
    manifest["formal_execution_package_lock"] = formal_execution_package_lock
    (source_dir / "manifest.local.json").write_text(
        stable_json_text(manifest),
        encoding="utf-8",
    )
    validate_scientific_execution_binding(
        source_dir / "scientific_execution_binding.json",
        expected_artifact_role="dataset_level_quality",
        expected_paper_run_name=resolved_paper_run_name,
        repository_root=root_path,
    )
    code_version = formal_execution_package_lock["formal_execution_commit"]
    archive_path = source_dir / (
        f"dataset_level_quality_package_{utc_archive_token()}_{code_version[:7]}.zip"
    )
    package_input_manifest_path = source_dir / PACKAGE_INPUT_MANIFEST_FILE_NAME
    package_input_manifest_path.unlink(missing_ok=True)
    entries = collect_exact_package_entries(
        repository_root=root_path,
        source_dir=source_dir,
        artifact_manifest=manifest,
        scientific_binding_path=source_dir / "scientific_execution_binding.json",
    )
    if not set(required_paths).issubset(entries):
        raise RuntimeError("artifact manifest 未精确声明全部数据集质量必要产物")
    write_exact_package_input_manifest(
        package_input_manifest_path,
        repository_root=root_path,
        package_family="dataset_level_quality",
        paper_run_name=resolved_paper_run_name,
        target_fpr=paper_run.target_fpr,
        randomization_repeat_identity={
            "randomization_repeat_id": paper_run.randomization_repeat_id,
            "generation_seed_index": paper_run.generation_seed_index,
            "generation_seed_offset": paper_run.generation_seed_offset,
            "watermark_key_index": paper_run.watermark_key_index,
        },
        formal_randomization_protocol_digest=(
            paper_run.formal_randomization_protocol_digest
        ),
        entries=entries,
        formal_execution_run_lock=formal_execution_run_lock,
        formal_execution_package_lock=formal_execution_package_lock,
    )
    entries = (*entries, package_input_manifest_path)
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        for path in entries:
            archive.write(path, path.relative_to(root_path).as_posix())
    try:
        validate_exact_package_archive(
            archive_path,
            repository_root=root_path,
            package_input_manifest_path=package_input_manifest_path,
        )
        final_package_lock = (
            repository_environment.require_published_formal_execution_lock(root_path)
        )
        repository_environment.validate_formal_execution_lock_pair(
            formal_execution_package_lock,
            final_package_lock,
            code_version,
        )
    except Exception:
        archive_path.unlink(missing_ok=True)
        raise
    return archive_path


def build_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器。"""

    parser = argparse.ArgumentParser(description="写出数据集级质量证据产物。")
    parser.add_argument("--root", default=".", help="仓库根目录。")
    parser.add_argument(
        "--quality-image-registry-path",
        default=None,
        help="clean/watermarked 质量图像 registry 路径; 默认读取当前论文层级主方法 registry。",
    )
    parser.add_argument(
        "--attack-quality-registry-path",
        default=None,
        help="四图逐攻击质量 registry 路径; 默认读取当前论文层级主方法记录。",
    )
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
        default=None,
        help="正式 FID / KID 所需最小图像对数量; 默认等于当前论文层级完整 Prompt 数量, 显式值不得更低。",
    )
    parser.add_argument(
        "--auto-extract-formal-features",
        action="store_true",
        help="特征记录缺失时直接运行 torch-fidelity Inception v3 兼容特征提取。",
    )
    parser.add_argument("--inception-device-name", default=None, help="正式 Inception 特征提取设备。")
    parser.add_argument("--inception-batch-size", type=int, default=32, help="正式 Inception 特征 batch 大小。")
    parser.add_argument("--clip-device-name", default=None, help="正式 CLIP 特征提取设备。")
    parser.add_argument("--clip-batch-size", type=int, default=32, help="正式 CLIP 特征 batch 大小。")
    return parser


def main() -> None:
    """命令行入口。"""

    args = build_parser().parse_args()
    paper_run = build_paper_run_config(args.root)
    manifest = write_dataset_level_quality_outputs(
        paper_run_name=paper_run.run_name,
        target_fpr=paper_run.target_fpr,
        root=args.root,
        quality_image_registry_path=args.quality_image_registry_path,
        attack_quality_registry_path=args.attack_quality_registry_path,
        image_search_roots=args.image_search_root,
        input_package_paths=args.input_package_path,
        formal_feature_records_path=args.formal_feature_records_path,
        formal_min_sample_count=args.formal_min_sample_count,
        auto_extract_formal_features=args.auto_extract_formal_features,
        inception_device_name=args.inception_device_name,
        inception_batch_size=args.inception_batch_size,
        clip_device_name=args.clip_device_name,
        clip_batch_size=args.clip_batch_size,
    )
    print(stable_json_text(manifest), end="")


if __name__ == "__main__":
    main()
