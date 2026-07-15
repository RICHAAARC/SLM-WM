"""从真实持久化图像提取冻结 DINOv2 独立语义质量特征."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

from experiments.protocol.independent_semantic_quality import (
    INDEPENDENT_SEMANTIC_FEATURE_BACKEND,
    INDEPENDENT_SEMANTIC_FEATURE_DIMENSION,
    load_independent_semantic_quality_evaluator,
)
from experiments.runtime import repository_environment
from experiments.runtime.scientific_unit_provenance import (
    build_scientific_unit_provenance,
    validate_scientific_unit_provenance,
)
from main.core.digest import build_stable_digest


def _path_sha256(path: Path) -> str:
    """流式计算真实图像文件的 SHA-256."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _record_value(record: Any, field_name: str) -> Any:
    """同时读取 mapping 与 dataclass 风格记录字段."""

    if isinstance(record, Mapping):
        return record.get(field_name)
    return getattr(record, field_name)


def _resolve_image_path(
    path_text: str,
    root_path: Path,
    image_search_roots: tuple[Path, ...],
) -> Path:
    """只解析已经持久化的图像路径, 不允许构造替代图像."""

    source = Path(path_text)
    candidates = [source if source.is_absolute() else root_path / source]
    candidates.extend(search_root / path_text for search_root in image_search_roots)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"独立语义评估缺少真实持久化图像: {path_text}")


def _portable_path(path: Path, root_path: Path) -> str:
    """优先保存相对仓库根目录的可迁移图像路径."""

    try:
        return path.relative_to(root_path.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _validate_processor_identity(processor: Any, protocol: Mapping[str, Any]) -> None:
    """复验运行时预处理器与冻结 revision 中登记的像素变换一致."""

    preprocessing = protocol["preprocessing_contract"]
    size = getattr(processor, "size", {})
    crop = getattr(processor, "crop_size", {})
    if not all(
        (
            processor.__class__.__name__ == preprocessing["image_processor_class"],
            int(size.get("shortest_edge", -1))
            == preprocessing["resize_shortest_edge"],
            int(crop.get("height", -1))
            == preprocessing["center_crop_height"],
            int(crop.get("width", -1))
            == preprocessing["center_crop_width"],
            list(getattr(processor, "image_mean", ()))
            == preprocessing["image_mean"],
            list(getattr(processor, "image_std", ()))
            == preprocessing["image_std"],
            math.isclose(
                float(getattr(processor, "rescale_factor", math.nan)),
                float(preprocessing["rescale_factor"]),
                rel_tol=0.0,
                abs_tol=1e-15,
            ),
        )
    ):
        raise RuntimeError("独立语义评估器预处理身份与冻结协议不一致")


def extract_independent_semantic_feature_rows(
    *,
    records: Iterable[Any],
    root_path: Path,
    image_search_roots: tuple[Path, ...],
    output_path: Path,
    device_name: str | None = None,
    batch_size: int = 32,
) -> list[dict[str, Any]]:
    """使用冻结 DINOv2 对真实图像提取 CLS 特征, 并原子写出原始记录."""

    if batch_size <= 0:
        raise ValueError("独立语义特征 batch_size 必须为正整数")
    root = root_path.resolve()
    protocol = load_independent_semantic_quality_evaluator()
    model_contract = protocol["model_contract"]
    items: list[tuple[Any, str, Path, str]] = []
    for record in records:
        for image_role, path_field, digest_field in (
            ("source", "source_image_path", "source_image_digest"),
            ("comparison", "comparison_image_path", "comparison_image_digest"),
        ):
            image_path = _resolve_image_path(
                str(_record_value(record, path_field)),
                root,
                image_search_roots,
            )
            actual_digest = _path_sha256(image_path)
            if actual_digest != str(_record_value(record, digest_field)):
                raise RuntimeError("独立语义评估图像摘要与真实文件不一致")
            items.append((record, image_role, image_path, actual_digest))
    if not items:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("", encoding="utf-8")
        return []

    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, Dinov2Model

    resolved_device = device_name or os.environ.get(
        "SLM_WM_INDEPENDENT_SEMANTIC_DEVICE",
        "cuda" if torch.cuda.is_available() else "cpu",
    )
    if not resolved_device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("正式独立语义特征提取必须在真实 CUDA 设备执行")
    formal_execution_lock = repository_environment.require_published_formal_execution_lock(
        root
    )
    runtime_environment = repository_environment.build_runtime_environment_report(
        protocol["dependency_contract"]["dependency_profile_id"],
        torch_module=torch,
        verified_formal_execution_lock=formal_execution_lock,
        repository_root=root,
    )
    if runtime_environment["dependency_environment_ready"] is not True:
        blockers = ",".join(runtime_environment["dependency_readiness_blockers"])
        raise RuntimeError(f"独立语义评估依赖环境未通过门禁:{blockers}")

    processor = AutoImageProcessor.from_pretrained(
        model_contract["model_id"],
        revision=model_contract["model_revision"],
        use_fast=True,
    )
    _validate_processor_identity(processor, protocol)
    model = Dinov2Model.from_pretrained(
        model_contract["model_id"],
        revision=model_contract["model_revision"],
        torch_dtype=torch.float32,
    )
    model.requires_grad_(False).eval().to(resolved_device)
    if model.__class__.__name__ != model_contract["model_class"]:
        raise RuntimeError("独立语义评估模型类与冻结协议不一致")

    rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for start in range(0, len(items), batch_size):
            batch_items = items[start : start + batch_size]
            images = []
            for _record, _role, image_path, _digest in batch_items:
                with Image.open(image_path) as image:
                    images.append(image.convert("RGB").copy())
            pixel_values = processor(images=images, return_tensors="pt")[
                "pixel_values"
            ].to(device=resolved_device, dtype=torch.float32)
            outputs = model(pixel_values=pixel_values)
            features = outputs.last_hidden_state[:, 0, :].float()
            features = torch.nn.functional.normalize(features, dim=-1).cpu()
            if (
                features.ndim != 2
                or features.shape[1] != INDEPENDENT_SEMANTIC_FEATURE_DIMENSION
            ):
                raise RuntimeError("独立语义 CLS 特征维度与冻结协议不一致")
            item_identity = [
                {
                    "dataset_quality_record_id": str(
                        _record_value(record, "dataset_quality_record_id")
                    ),
                    "dataset_quality_image_role": role,
                    "image_path": _portable_path(image_path, root),
                    "image_digest": image_digest,
                }
                for record, role, image_path, image_digest in batch_items
            ]
            batch_digest = build_stable_digest(item_identity)
            provenance = build_scientific_unit_provenance(
                scientific_unit_id=f"independent_semantic_batch_{batch_digest[:16]}",
                scientific_unit_config_digest=build_stable_digest(
                    {
                        "item_identity": item_identity,
                        "independent_semantic_quality_protocol_digest": protocol[
                            "independent_semantic_quality_protocol_digest"
                        ],
                    }
                ),
                runtime_environment=runtime_environment,
                execution_device_name=resolved_device,
                torch_module=torch,
                random_identity_random={
                    "feature_extraction_seed_random": "not_used_deterministic_eval"
                },
            )
            for identity, feature in zip(item_identity, features, strict=True):
                vector = [float(value) for value in feature.tolist()]
                rows.append(
                    {
                        **identity,
                        "feature_backend": INDEPENDENT_SEMANTIC_FEATURE_BACKEND,
                        "feature_extractor_id": (
                            f"{model_contract['model_id']}@"
                            f"{model_contract['model_revision']}"
                        ),
                        "feature_dimension": INDEPENDENT_SEMANTIC_FEATURE_DIMENSION,
                        "feature_layer": protocol["feature_contract"]["feature_layer"],
                        "feature_normalization": "l2",
                        "feature_vector": vector,
                        "feature_vector_digest": build_stable_digest(vector),
                        "independent_semantic_quality_protocol_digest": protocol[
                            "independent_semantic_quality_protocol_digest"
                        ],
                        "scientific_unit_provenance": provenance,
                        "supports_paper_claim": False,
                    }
                )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(output_path.name + ".partial")
    temporary_path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    temporary_path.replace(output_path)
    return rows


def validate_independent_semantic_feature_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    expected_record_ids: Iterable[str],
    expected_code_version: str,
) -> tuple[dict[str, Any], ...]:
    """验证真实独立特征精确覆盖图像对并绑定同一提交和依赖 profile."""

    protocol = load_independent_semantic_quality_evaluator()
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
        if not all(
            (
                key in expected_keys,
                key not in by_key,
                row.get("feature_backend") == INDEPENDENT_SEMANTIC_FEATURE_BACKEND,
                row.get("feature_extractor_id")
                == (
                    f"{protocol['model_contract']['model_id']}@"
                    f"{protocol['model_contract']['model_revision']}"
                ),
                row.get("feature_dimension")
                == INDEPENDENT_SEMANTIC_FEATURE_DIMENSION,
                row.get("feature_layer") == "last_hidden_state_cls_token",
                row.get("feature_normalization") == "l2",
                isinstance(vector, list),
                len(vector) == INDEPENDENT_SEMANTIC_FEATURE_DIMENSION,
                all(
                    isinstance(value, (int, float)) and math.isfinite(value)
                    for value in vector
                ),
                row.get("feature_vector_digest") == build_stable_digest(vector),
                row.get("independent_semantic_quality_protocol_digest")
                == protocol["independent_semantic_quality_protocol_digest"],
                isinstance(provenance, Mapping),
                row.get("supports_paper_claim") is False,
            )
        ):
            raise ValueError("独立语义特征集合、维度或协议身份无效")
        reference = validate_scientific_unit_provenance(provenance)
        environment = reference["scientific_execution_environment"]
        norm = float(np.linalg.norm(np.asarray(vector, dtype=np.float64)))
        if not all(
            (
                environment.get("formal_execution_commit") == expected_code_version,
                environment.get("dependency_profile_id")
                == protocol["dependency_contract"]["dependency_profile_id"],
                environment.get("dependency_profile_digest")
                == protocol["dependency_profile_digest"],
                environment.get("complete_hash_lock_digest")
                == protocol["complete_hash_lock_digest"],
                math.isclose(norm, 1.0, rel_tol=1e-5, abs_tol=1e-5),
            )
        ):
            raise ValueError("独立语义特征未绑定提交、依赖锁或 L2 归一化")
        by_key[key] = row
    if set(by_key) != expected_keys:
        raise ValueError("独立语义特征未精确覆盖全部图像对")
    return tuple(
        by_key[(record_id, role)]
        for record_id in expected_ids
        for role in ("source", "comparison")
    )


__all__ = [
    "extract_independent_semantic_feature_rows",
    "validate_independent_semantic_feature_rows",
]
