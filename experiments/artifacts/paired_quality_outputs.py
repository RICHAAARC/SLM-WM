"""从受治理图像对生产真实 SSIM,CLIP 和特征身份记录."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from experiments.protocol.attack_conditioned_quality import (
    ATTACK_CONDITIONED_IMAGE_PAIR_ROLE,
    load_attack_conditioned_quality_estimand,
)
from experiments.runtime.image_metrics import compute_image_quality_metrics
from experiments.runtime import repository_environment
from experiments.runtime.diffusion.semantic_model_loader import load_clip_vision_model
from experiments.runtime.resume_checkpoint import (
    clear_progress_checkpoints,
    persist_checkpoint_files,
    persist_progress_checkpoint,
)
from experiments.runtime.scientific_unit_provenance import (
    build_scientific_unit_provenance,
    validate_scientific_unit_provenance,
)
from main.core.digest import build_stable_digest


FORMAL_CLIP_FEATURE_BACKEND = "clip_projected_image_embedding"
FORMAL_CLIP_FEATURE_DIMENSION = 512
PAIRED_QUALITY_METRIC_RECORD_SCHEMA = "paired_quality_metric_record_v1"


def _stable_json_text(value: Any) -> str:
    """返回排序稳定且带结尾换行的 JSON 文本."""

    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"


def _path_digest(path: Path) -> str:
    """流式计算文件 SHA-256."""

    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_or_absolute(path: Path, root_path: Path) -> str:
    """优先返回相对仓库根目录的可移植路径."""

    try:
        return path.resolve().relative_to(root_path.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _resolve_existing_image_path(
    path_text: str,
    root_path: Path,
    image_search_roots: tuple[Path, ...],
) -> Path:
    """从原始路径和显式搜索根目录解析已持久化图像."""

    path = Path(path_text)
    candidates = [path if path.is_absolute() else root_path / path]
    candidates.extend(search_root / path_text for search_root in image_search_roots)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return candidates[0].resolve()


def _json_line(value: Mapping[str, Any]) -> str:
    """返回排序稳定的 JSONL 行."""

    return json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n"


def _clip_batch_config_digest(
    item_identity: Iterable[Mapping[str, Any]],
    protocol_digest: str,
) -> str:
    """绑定 CLIP batch 中的图像顺序,模型和 estimand 身份."""

    return build_stable_digest(
        {
            "feature_backend": FORMAL_CLIP_FEATURE_BACKEND,
            "feature_dimension": FORMAL_CLIP_FEATURE_DIMENSION,
            "item_identity": list(item_identity),
            "quality_estimand_protocol_digest": protocol_digest,
        }
    )


def extract_formal_clip_feature_rows(
    *,
    records: Iterable[Any],
    root_path: Path,
    image_search_roots: tuple[Path, ...],
    output_path: Path,
    device_name: str | None = None,
    batch_size: int = 32,
) -> list[dict[str, Any]]:
    """使用冻结 CLIP revision 提取逐图像投影 embedding 并支持断点续算."""

    if batch_size <= 0:
        raise ValueError("CLIP 特征 batch_size 必须为正整数")
    protocol = load_attack_conditioned_quality_estimand()
    clip_protocol = dict(protocol["clip_image_similarity"])
    items: list[tuple[Any, str, Path, str]] = []
    digest_by_path: dict[Path, str] = {}
    for record in records:
        for image_role, path_text, image_digest in (
            ("source", record.source_image_path, record.source_image_digest),
            (
                "comparison",
                record.comparison_image_path,
                record.comparison_image_digest,
            ),
        ):
            image_path = _resolve_existing_image_path(
                path_text,
                root_path,
                image_search_roots,
            ).resolve()
            if not image_path.is_file():
                raise FileNotFoundError(f"正式 CLIP 特征缺少图像文件: {path_text}")
            actual_digest = digest_by_path.get(image_path)
            if actual_digest is None:
                actual_digest = _path_digest(image_path)
                digest_by_path[image_path] = actual_digest
            if image_digest != actual_digest:
                raise RuntimeError(f"正式 CLIP 图像摘要与实际文件不一致: {path_text}")
            items.append((record, image_role, image_path, actual_digest))
    if not items:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("", encoding="utf-8")
        return []

    checkpoint_dir = output_path.parent / "clip_feature_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_path.parent / "clip_feature_progress.json"
    item_identity = [
        {
            "dataset_quality_record_id": record.dataset_quality_record_id,
            "dataset_quality_image_role": image_role,
            "image_path": _relative_or_absolute(image_path, root_path),
            "image_digest": image_digest,
        }
        for record, image_role, image_path, image_digest in items
    ]
    formal_execution_lock = (
        repository_environment.require_published_formal_execution_lock(
            root_path
        )
    )
    context = {
        "report_schema": "clip_feature_checkpoint_context",
        "schema_version": 1,
        "feature_backend": FORMAL_CLIP_FEATURE_BACKEND,
        "model_id": clip_protocol["model_id"],
        "model_revision": clip_protocol["model_revision"],
        "quality_estimand_protocol_digest": protocol[
            "quality_estimand_protocol_digest"
        ],
        "item_count": len(item_identity),
        "item_identity_digest": build_stable_digest(item_identity),
        "formal_execution_lock": formal_execution_lock,
        "evidence_eligibility": "intermediate_state_only",
        "supports_paper_claim": False,
    }
    context_path = checkpoint_dir / "feature_checkpoint_context.json"
    if context_path.is_file():
        if json.loads(context_path.read_text(encoding="utf-8-sig")) != context:
            raise RuntimeError("正式 CLIP 检查点身份与当前运行不一致")
    else:
        context_path.write_text(_stable_json_text(context), encoding="utf-8")

    expected_by_key = {
        (
            identity["dataset_quality_record_id"],
            identity["dataset_quality_image_role"],
        ): identity
        for identity in item_identity
    }
    rows_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for shard_path in sorted(checkpoint_dir.glob("clip_batch_*.jsonl")):
        shard_rows = [
            json.loads(line)
            for line in shard_path.read_text(encoding="utf-8-sig").splitlines()
            if line.strip()
        ]
        shard_identity: list[dict[str, Any]] = []
        for row in shard_rows:
            key = (
                str(row.get("dataset_quality_record_id", "")),
                str(row.get("dataset_quality_image_role", "")),
            )
            expected = expected_by_key.get(key)
            vector = row.get("feature_vector")
            if not all(
                (
                    expected is not None,
                    row.get("feature_backend") == FORMAL_CLIP_FEATURE_BACKEND,
                    row.get("feature_dimension") == FORMAL_CLIP_FEATURE_DIMENSION,
                    isinstance(vector, list),
                    len(vector) == FORMAL_CLIP_FEATURE_DIMENSION,
                    all(isinstance(value, (int, float)) and math.isfinite(value) for value in vector),
                    row.get("image_path") == expected["image_path"],
                    row.get("image_digest") == expected["image_digest"],
                    row.get("supports_paper_claim") is False,
                )
            ):
                raise RuntimeError("正式 CLIP 检查点内容未通过身份校验")
            existing = rows_by_key.get(key)
            if existing is not None and existing != row:
                raise RuntimeError("正式 CLIP 检查点包含冲突记录")
            rows_by_key[key] = row
            shard_identity.append(dict(expected))
        unit_id = shard_path.stem
        expected_config_digest = _clip_batch_config_digest(
            shard_identity,
            protocol["quality_estimand_protocol_digest"],
        )
        validated = [
            validate_scientific_unit_provenance(
                row["scientific_unit_provenance"],
                expected_unit_id=unit_id,
                expected_config_digest=expected_config_digest,
            )
            for row in shard_rows
        ]
        if not validated or any(record != validated[0] for record in validated[1:]):
            raise RuntimeError("正式 CLIP batch 来源记录不唯一")

    remaining_items = [
        item
        for item in items
        if (item[0].dataset_quality_record_id, item[1]) not in rows_by_key
    ]
    if remaining_items:
        import torch
        from PIL import Image
        from transformers import CLIPImageProcessor

        resolved_device = device_name or os.environ.get(
            "SLM_WM_CLIP_DEVICE",
            "cuda" if torch.cuda.is_available() else "cpu",
        )
        if not resolved_device.startswith("cuda") or not torch.cuda.is_available():
            raise RuntimeError("正式 CLIP 特征提取必须在可用 CUDA 设备执行")
        runtime_environment = repository_environment.build_runtime_environment_report(
            "sd35_method_runtime_gpu",
            torch_module=torch,
            verified_formal_execution_lock=formal_execution_lock,
            repository_root=root_path,
        )
        if runtime_environment["dependency_environment_ready"] is not True:
            blockers = ",".join(
                runtime_environment["dependency_readiness_blockers"]
            )
            raise RuntimeError(f"正式 CLIP 特征依赖环境未通过门禁:{blockers}")
        model = load_clip_vision_model(
            clip_protocol["model_id"],
            clip_protocol["model_revision"],
            resolved_device,
            "float32",
        )
        processor = CLIPImageProcessor.from_pretrained(
            clip_protocol["model_id"],
            revision=clip_protocol["model_revision"],
        )
        with torch.inference_mode():
            for start in range(0, len(remaining_items), batch_size):
                batch_items = remaining_items[start : start + batch_size]
                images = []
                for _, _, image_path, _ in batch_items:
                    with Image.open(image_path) as image:
                        images.append(image.convert("RGB").copy())
                pixels = processor(images=images, return_tensors="pt")[
                    "pixel_values"
                ].to(device=resolved_device, dtype=torch.float32)
                features = model(pixel_values=pixels).image_embeds.float()
                features = torch.nn.functional.normalize(features, dim=-1).cpu()
                if features.ndim != 2 or features.shape[1] != FORMAL_CLIP_FEATURE_DIMENSION:
                    raise RuntimeError("正式 CLIP 投影特征维度不符合冻结协议")
                batch_identity = [
                    {
                        "dataset_quality_record_id": record.dataset_quality_record_id,
                        "dataset_quality_image_role": image_role,
                        "image_path": _relative_or_absolute(image_path, root_path),
                        "image_digest": image_digest,
                    }
                    for record, image_role, image_path, image_digest in batch_items
                ]
                batch_digest = build_stable_digest(
                    [
                        (
                            identity["dataset_quality_record_id"],
                            identity["dataset_quality_image_role"],
                        )
                        for identity in batch_identity
                    ]
                )
                unit_id = f"clip_batch_{batch_digest[:16]}"
                provenance = build_scientific_unit_provenance(
                    scientific_unit_id=unit_id,
                    scientific_unit_config_digest=_clip_batch_config_digest(
                        batch_identity,
                        protocol["quality_estimand_protocol_digest"],
                    ),
                    runtime_environment=runtime_environment,
                    execution_device_name=resolved_device,
                    torch_module=torch,
                    random_identity_random={
                        "feature_extraction_seed_random": (
                            "not_used_deterministic_eval"
                        )
                    },
                )
                batch_rows = []
                for item, feature in zip(batch_items, features, strict=True):
                    record, image_role, image_path, image_digest = item
                    row = {
                        "dataset_quality_record_id": record.dataset_quality_record_id,
                        "dataset_quality_image_role": image_role,
                        "feature_backend": FORMAL_CLIP_FEATURE_BACKEND,
                        "feature_extractor_id": (
                            f"{clip_protocol['model_id']}@{clip_protocol['model_revision']}"
                        ),
                        "feature_dimension": FORMAL_CLIP_FEATURE_DIMENSION,
                        "image_path": _relative_or_absolute(image_path, root_path),
                        "image_digest": image_digest,
                        "feature_vector": [float(value) for value in feature.tolist()],
                        "quality_estimand_protocol_digest": protocol[
                            "quality_estimand_protocol_digest"
                        ],
                        "scientific_unit_provenance": provenance,
                        "supports_paper_claim": False,
                    }
                    key = (record.dataset_quality_record_id, image_role)
                    rows_by_key[key] = row
                    batch_rows.append(row)
                shard_path = checkpoint_dir / f"{unit_id}.jsonl"
                temporary_path = shard_path.with_name(shard_path.name + ".partial")
                temporary_path.write_text(
                    "".join(_json_line(row) for row in batch_rows),
                    encoding="utf-8",
                )
                temporary_path.replace(shard_path)
                persist_checkpoint_files(
                    repository_root=root_path,
                    artifact_role="dataset_level_quality",
                    paper_run_name=os.environ.get(
                        "SLM_WM_PAPER_RUN_NAME",
                        output_path.parent.name,
                    ),
                    checkpoint_kind="clip_feature_batches",
                    checkpoint_id=unit_id,
                    paths=(context_path, shard_path),
                )
                progress = {
                    "report_schema": "clip_feature_progress",
                    "schema_version": 1,
                    "expected_feature_record_count": len(items),
                    "completed_feature_record_count": len(rows_by_key),
                    "remaining_feature_record_count": len(items) - len(rows_by_key),
                    "protocol_decision": "resume_required",
                    "evidence_eligibility": "intermediate_state_only",
                    "supports_paper_claim": False,
                }
                progress_path.write_text(
                    _stable_json_text(progress),
                    encoding="utf-8",
                )
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
        rows_by_key[(record.dataset_quality_record_id, image_role)]
        for record, image_role, _, _ in items
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_output_path = output_path.with_name(output_path.name + ".partial")
    temporary_output_path.write_text(
        "".join(_json_line(row) for row in rows),
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


def _record_value(record: Any, field_name: str, default: Any = None) -> Any:
    """同时读取 dataclass,namespace 和 mapping 字段."""

    if isinstance(record, Mapping):
        return record.get(field_name, default)
    return getattr(record, field_name, default)


def build_paired_quality_metric_records(
    base_records: Iterable[Any],
    attack_records: Iterable[Any],
    clip_feature_rows: Iterable[Mapping[str, Any]],
    *,
    randomization_repeat_id: str,
    root_path: Path,
    image_search_roots: tuple[Path, ...],
) -> tuple[dict[str, Any], ...]:
    """从真实图像和 CLIP embedding 形成 base 及逐攻击配对指标记录."""

    feature_by_key: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in clip_feature_rows:
        key = (
            str(row.get("dataset_quality_record_id", "")),
            str(row.get("dataset_quality_image_role", "")),
        )
        if key in feature_by_key:
            raise ValueError("CLIP feature 身份重复")
        feature_by_key[key] = row
    observations = [
        ("base", record) for record in base_records
    ] + [("registered_attack", record) for record in attack_records]
    records: list[dict[str, Any]] = []
    for estimand_scope, record in observations:
        dataset_record_id = str(
            _record_value(record, "dataset_quality_record_id", "")
        )
        source_feature = feature_by_key.get((dataset_record_id, "source"))
        comparison_feature = feature_by_key.get(
            (dataset_record_id, "comparison")
        )
        if source_feature is None or comparison_feature is None:
            raise ValueError("配对质量记录缺少完整 CLIP feature 对")
        source_path = _resolve_existing_image_path(
            str(_record_value(record, "source_image_path", "")),
            root_path,
            image_search_roots,
        )
        comparison_path = _resolve_existing_image_path(
            str(_record_value(record, "comparison_image_path", "")),
            root_path,
            image_search_roots,
        )
        from PIL import Image

        with Image.open(source_path) as source_image, Image.open(
            comparison_path
        ) as comparison_image:
            quality = compute_image_quality_metrics(
                source_image.convert("RGB"),
                comparison_image.convert("RGB"),
            )
        source_vector = np.asarray(
            source_feature["feature_vector"],
            dtype=np.float64,
        )
        comparison_vector = np.asarray(
            comparison_feature["feature_vector"],
            dtype=np.float64,
        )
        denominator = float(
            np.linalg.norm(source_vector) * np.linalg.norm(comparison_vector)
        )
        clip_cosine = (
            float(np.dot(source_vector, comparison_vector) / denominator)
            if denominator > 0.0
            else math.nan
        )
        if not math.isfinite(clip_cosine):
            raise ValueError("CLIP cosine 不能由零范数或非有限特征形成")
        attack_id = (
            "none"
            if estimand_scope == "base"
            else str(_record_value(record, "attack_id", ""))
        )
        pair_role = str(_record_value(record, "image_pair_role", ""))
        if (
            estimand_scope == "registered_attack"
            and pair_role != ATTACK_CONDITIONED_IMAGE_PAIR_ROLE
        ):
            raise ValueError("逐攻击质量记录使用了错误配对角色")
        core = {
            "record_schema": PAIRED_QUALITY_METRIC_RECORD_SCHEMA,
            "dataset_quality_record_id": dataset_record_id,
            "dataset_quality_record_digest": str(
                _record_value(record, "dataset_quality_record_digest", "")
            ),
            "attack_quality_record_id": str(
                _record_value(record, "attack_quality_record_id", "")
            ),
            "randomization_repeat_id": str(randomization_repeat_id),
            "prompt_id": str(_record_value(record, "prompt_id", "")),
            "estimand_scope": estimand_scope,
            "sample_role": (
                "base_quality_pair"
                if estimand_scope == "base"
                else "matched_attack_quality_pair"
            ),
            "attack_id": attack_id,
            "attack_config_digest": str(
                _record_value(record, "attack_config_digest", "")
            ),
            "attack_seed_random": _record_value(
                record,
                "attack_seed_random",
            ),
            "image_pair_role": pair_role,
            "source_image_digest": str(
                _record_value(record, "source_image_digest", "")
            ),
            "comparison_image_digest": str(
                _record_value(record, "comparison_image_digest", "")
            ),
            "paired_ssim": float(quality["ssim"]),
            "clip_cosine": clip_cosine,
            "clip_source_feature_digest": build_stable_digest(
                source_feature["feature_vector"]
            ),
            "clip_comparison_feature_digest": build_stable_digest(
                comparison_feature["feature_vector"]
            ),
            "quality_estimand_protocol_digest": (
                load_attack_conditioned_quality_estimand()[
                    "quality_estimand_protocol_digest"
                ]
            ),
            "supports_paper_claim": False,
        }
        record_digest = build_stable_digest(core)
        records.append(
            {
                "paired_quality_metric_record_id": (
                    f"paired_quality_metric_{record_digest[:16]}"
                ),
                "paired_quality_metric_record_digest": record_digest,
                **core,
            }
        )
    return tuple(records)


def validate_formal_clip_feature_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    expected_record_ids: Iterable[str],
    expected_code_version: str,
) -> tuple[dict[str, Any], ...]:
    """验证 CLIP 特征精确覆盖每个图像对并绑定同一正式提交."""

    protocol = load_attack_conditioned_quality_estimand()
    expected_ids = tuple(str(record_id) for record_id in expected_record_ids)
    expected_keys = {
        (record_id, role)
        for record_id in expected_ids
        for role in ("source", "comparison")
    }
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for raw_row in rows:
        row = dict(raw_row)
        key = (
            str(row.get("dataset_quality_record_id", "")),
            str(row.get("dataset_quality_image_role", "")),
        )
        vector = row.get("feature_vector")
        provenance = row.get("scientific_unit_provenance")
        if (
            key not in expected_keys
            or key in by_key
            or row.get("feature_backend") != FORMAL_CLIP_FEATURE_BACKEND
            or row.get("feature_dimension") != FORMAL_CLIP_FEATURE_DIMENSION
            or not isinstance(vector, list)
            or len(vector) != FORMAL_CLIP_FEATURE_DIMENSION
            or any(
                not isinstance(value, (int, float)) or not math.isfinite(value)
                for value in vector
            )
            or not isinstance(provenance, Mapping)
            or row.get("quality_estimand_protocol_digest")
            != protocol["quality_estimand_protocol_digest"]
            or row.get("supports_paper_claim") is not False
        ):
            raise ValueError("正式 CLIP 特征集合,维度或协议身份无效")
        validated = validate_scientific_unit_provenance(provenance)
        environment = validated["scientific_execution_environment"]
        if (
            environment.get("formal_execution_commit")
            != expected_code_version
            or environment.get("dependency_profile_id")
            != "sd35_method_runtime_gpu"
        ):
            raise ValueError("正式 CLIP 特征未绑定当前提交和 GPU 依赖 profile")
        norm = float(np.linalg.norm(np.asarray(vector, dtype=np.float64)))
        if not math.isclose(norm, 1.0, rel_tol=1e-5, abs_tol=1e-5):
            raise ValueError("正式 CLIP 特征未按冻结协议完成 L2 归一化")
        by_key[key] = row
    if set(by_key) != expected_keys:
        raise ValueError("正式 CLIP 特征未精确覆盖全部图像对")
    return tuple(
        by_key[(record_id, role)]
        for record_id in expected_ids
        for role in ("source", "comparison")
    )


def validate_paired_quality_metric_records(
    rows: Iterable[Mapping[str, Any]],
    *,
    base_records: Iterable[Any],
    attack_records: Iterable[Any],
    expected_randomization_repeat_id: str,
) -> tuple[dict[str, Any], ...]:
    """验证每个 base/attack 图像对恰好对应一个可复算指标记录."""

    expected: dict[str, tuple[str, Any]] = {}
    for scope, records in (
        ("base", base_records),
        ("registered_attack", attack_records),
    ):
        for record in records:
            record_id = str(
                _record_value(record, "dataset_quality_record_id", "")
            )
            if not record_id or record_id in expected:
                raise ValueError("配对质量图像记录身份缺失或重复")
            expected[record_id] = (scope, record)
    by_record_id: dict[str, dict[str, Any]] = {}
    for raw_row in rows:
        row = dict(raw_row)
        record_id = str(row.get("dataset_quality_record_id", ""))
        expected_entry = expected.get(record_id)
        digest = str(row.get("paired_quality_metric_record_digest", ""))
        digest_payload = {
            key: value
            for key, value in row.items()
            if key
            not in {
                "paired_quality_metric_record_id",
                "paired_quality_metric_record_digest",
            }
        }
        if (
            expected_entry is None
            or record_id in by_record_id
            or row.get("record_schema") != PAIRED_QUALITY_METRIC_RECORD_SCHEMA
            or row.get("randomization_repeat_id")
            != expected_randomization_repeat_id
            or not isinstance(row.get("paired_ssim"), (int, float))
            or not math.isfinite(float(row["paired_ssim"]))
            or not -1.0 <= float(row["paired_ssim"]) <= 1.0
            or not isinstance(row.get("clip_cosine"), (int, float))
            or not math.isfinite(float(row["clip_cosine"]))
            or not -1.0 <= float(row["clip_cosine"]) <= 1.0
            or build_stable_digest(digest_payload) != digest
            or row.get("paired_quality_metric_record_id")
            != f"paired_quality_metric_{digest[:16]}"
        ):
            raise ValueError("配对质量指标身份,数值或自摘要无效")
        scope, record = expected_entry
        expected_attack_id = (
            "none"
            if scope == "base"
            else str(_record_value(record, "attack_id", ""))
        )
        if not all(
            (
                row.get("estimand_scope") == scope,
                row.get("prompt_id")
                == _record_value(record, "prompt_id", ""),
                row.get("attack_id") == expected_attack_id,
                row.get("source_image_digest")
                == _record_value(record, "source_image_digest", ""),
                row.get("comparison_image_digest")
                == _record_value(record, "comparison_image_digest", ""),
                row.get("supports_paper_claim") is False,
            )
        ):
            raise ValueError("配对质量指标未绑定对应图像和 estimand")
        by_record_id[record_id] = row
    if set(by_record_id) != set(expected):
        raise ValueError("配对质量指标未精确覆盖 base 和逐攻击图像对")
    return tuple(by_record_id[record_id] for record_id in expected)


def as_dataset_quality_namespaces(
    records: Iterable[Mapping[str, Any]],
) -> tuple[SimpleNamespace, ...]:
    """将通用映射转换为现有 Inception/CLIP 提取器可消费对象."""

    return tuple(SimpleNamespace(**dict(record)) for record in records)


__all__ = [
    "FORMAL_CLIP_FEATURE_BACKEND",
    "FORMAL_CLIP_FEATURE_DIMENSION",
    "PAIRED_QUALITY_METRIC_RECORD_SCHEMA",
    "as_dataset_quality_namespaces",
    "build_paired_quality_metric_records",
    "extract_formal_clip_feature_rows",
    "validate_formal_clip_feature_rows",
    "validate_paired_quality_metric_records",
]
