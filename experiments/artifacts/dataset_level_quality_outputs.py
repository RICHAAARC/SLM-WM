"""写出数据集级图像质量证据产物的正式实现。"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
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
from experiments.protocol.paper_run_config import (
    RUN_EXPECTED_PROMPT_COUNTS,
    build_paper_run_config,
    normalize_paper_run_name,
)
from experiments.runtime import repository_environment
from experiments.protocol.prompts import build_prompt_records, read_prompt_file
from experiments.artifacts.artifact_manifest import build_artifact_manifest
from main.core.digest import build_stable_digest
from experiments.runtime.archive_naming import utc_archive_token

DEFAULT_PROGRESS_INTERVAL_ITEMS = 50
FORMAL_FEATURE_EXTRACTOR_ID = "torch_fidelity_0_4_0_inception_v3_compat_2048"


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
    formal_feature_records_sha256: str,
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
    formal_feature_backend_ready = (
        accepted_pair_count == expected_pair_count
        and missing_feature_count == 0
        and len(feature_rows) == expected_pair_count * 2
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
    import torch
    from PIL import Image
    from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3

    resolved_device = device_name or os.environ.get(
        "SLM_WM_INCEPTION_DEVICE",
        "cuda" if torch.cuda.is_available() else "cpu",
    )
    if resolved_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("正式 Inception 特征提取要求的 CUDA 设备不可用")
    model = FeatureExtractorInceptionV3(
        "inception-v3-compat",
        ["2048"],
        verbose=True,
    ).to(resolved_device).eval()
    items: list[tuple[Any, str, Path, str]] = []
    for record in records:
        for image_role, path_text, image_digest in (
            ("source", record.source_image_path, record.source_image_digest),
            ("comparison", record.comparison_image_path, record.comparison_image_digest),
        ):
            resolved_path = resolve_existing_image_path(path_text, root_path, image_search_roots)
            if not resolved_path.is_file():
                raise FileNotFoundError(f"正式 Inception 特征缺少图像文件: {path_text}")
            items.append((record, image_role, resolved_path, image_digest))
    rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for start in range(0, len(items), batch_size):
            batch_items = items[start : start + batch_size]
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
            for (record, image_role, image_path, image_digest), feature in zip(batch_items, features):
                rows.append(
                    {
                        "dataset_quality_record_id": record.dataset_quality_record_id,
                        "dataset_quality_image_role": image_role,
                        "feature_backend": FORMAL_FEATURE_BACKEND,
                        "feature_extractor_id": FORMAL_FEATURE_EXTRACTOR_ID,
                        "feature_dimension": int(feature.numel()),
                        "image_path": relative_or_absolute(image_path, root_path),
                        "image_digest": image_digest or path_digest(image_path),
                        "feature_vector": [float(value) for value in feature.tolist()],
                        "supports_paper_claim": False,
                    }
                )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(json_line(row) for row in rows), encoding="utf-8")
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


def write_dataset_level_quality_outputs(
    *,
    paper_run_name: str,
    target_fpr: float,
    root: str | Path = ".",
    real_attack_registry_path: str | Path | None = None,
    image_search_roots: Any = (),
    input_package_paths: Any = (),
    formal_feature_records_path: str | Path | None = None,
    formal_min_sample_count: int | None = None,
    auto_extract_formal_features: bool = False,
    inception_device_name: str | None = None,
    inception_batch_size: int = 32,
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
    registry_path = real_attack_registry_path or (
        Path("outputs")
        / "image_only_dataset_runtime"
        / resolved_paper_run_name
        / "watermark_quality_image_registry.jsonl"
    )
    resolved_registry_path = resolve_path(root_path, registry_path)
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
    metric_rows = build_dataset_quality_metric_rows(
        records,
        root_path,
        image_search_roots=all_image_search_roots,
        formal_source_features=formal_feature_payload["source_features"],
        formal_comparison_features=formal_feature_payload["comparison_features"],
        formal_min_sample_count=resolved_formal_min_sample_count,
    )


    records_path = resolved_output_dir / "dataset_quality_image_records.jsonl"
    image_resolution_records_path = resolved_output_dir / "dataset_quality_image_resolution_records.jsonl"
    formal_feature_import_report_path = resolved_output_dir / "dataset_quality_formal_feature_import_report.json"
    metrics_path = resolved_output_dir / "dataset_quality_metrics.csv"
    summary_path = resolved_output_dir / "dataset_quality_summary.json"
    manifest_path = resolved_output_dir / "manifest.local.json"
    resolved_image_file_count = sum(1 for record in image_resolution_records if record["resolution_status"] != "image_file_missing")
    materialized_image_input_count = sum(1 for record in image_resolution_records if record["materialized_image_input"])

    base_summary = build_dataset_quality_summary(records, metric_rows)
    formal_feature_extractor_ids = sorted(
        {
            str(row.get("feature_extractor_id"))
            for row in formal_feature_rows
            if row.get("feature_extractor_id")
        }
    )
    canonical_formal_feature_extractor_ready = formal_feature_extractor_ids == [FORMAL_FEATURE_EXTRACTOR_ID]
    formal_claim_gate_ready = bool(
        base_summary.get("formal_fid_kid_claim_gate_ready", False)
        and canonical_formal_feature_extractor_ready
        and prompt_contract["prompt_registry_exact_set_ready"]
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
        "real_attack_registry_path": relative_or_absolute(resolved_registry_path, root_path),
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
        "formal_fid_kid_claim_gate_ready": formal_claim_gate_ready,
        "formal_fid_kid_claim_blocker": (
            ""
            if formal_claim_gate_ready
            else (
                "requires_canonical_inception_feature_extractor"
                if base_summary.get("formal_fid_kid_ready", False)
                else base_summary.get("formal_fid_kid_claim_blocker", "formal_fid_kid_not_measured")
            )
        ),
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
            "records_digest": build_stable_digest([record.to_dict() for record in records]),
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
        rebuild_command="python scripts/write_dataset_level_quality_outputs.py",
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
    repository_environment.validate_formal_execution_lock_pair(
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
    manifest_config = manifest.get("config", {})
    metric_contract_ready = (
        len(metric_rows) == 2
        and [row.get("quality_metric_name") for row in metric_rows] == ["fid", "kid"]
        and all(row.get("metric_status") == "measured" for row in metric_rows)
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
    if not all(
        (
            summary.get("paper_run_name") == resolved_paper_run_name,
            abs(float(summary.get("target_fpr", -1.0)) - paper_run.target_fpr) <= 1e-12,
            bool(summary.get("generated_at")),
            summary.get("formal_feature_backend_ready") is True,
            summary.get("formal_sample_scale_ready") is True,
            summary.get("canonical_formal_feature_extractor_ready") is True,
            summary.get("formal_fid_kid_claim_gate_ready") is True,
            manifest.get("artifact_id") == "dataset_level_quality_manifest",
            feature_contract_ready,
        )
    ):
        raise RuntimeError("数据集级质量身份、精确 Prompt/特征覆盖或 ready 门禁未通过")
    manifest["formal_execution_package_lock"] = formal_execution_package_lock
    (source_dir / "manifest.local.json").write_text(
        stable_json_text(manifest),
        encoding="utf-8",
    )
    code_version = formal_execution_package_lock["formal_execution_commit"]
    archive_path = source_dir / (
        f"dataset_level_quality_package_{utc_archive_token()}_{code_version[:7]}.zip"
    )
    entries = tuple(
        path for path in sorted(source_dir.rglob("*")) if path.is_file() and path.suffix.lower() != ".zip"
    )
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        for path in entries:
            archive.write(path, path.relative_to(root_path).as_posix())
    try:
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
        "--real-attack-registry-path",
        default=None,
        help="真实攻击图像 registry 路径; 默认读取当前论文层级主方法 registry。",
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
    return parser


def main() -> None:
    """命令行入口。"""

    args = build_parser().parse_args()
    paper_run = build_paper_run_config(args.root)
    manifest = write_dataset_level_quality_outputs(
        paper_run_name=paper_run.run_name,
        target_fpr=paper_run.target_fpr,
        root=args.root,
        real_attack_registry_path=args.real_attack_registry_path,
        image_search_roots=args.image_search_root,
        input_package_paths=args.input_package_path,
        formal_feature_records_path=args.formal_feature_records_path,
        formal_min_sample_count=args.formal_min_sample_count,
        auto_extract_formal_features=args.auto_extract_formal_features,
        inception_device_name=args.inception_device_name,
        inception_batch_size=args.inception_batch_size,
    )
    print(stable_json_text(manifest), end="")


if __name__ == "__main__":
    main()
