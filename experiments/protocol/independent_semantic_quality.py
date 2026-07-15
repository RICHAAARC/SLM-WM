"""冻结与方法条件编码器不同源的视觉语义质量评估协议."""

from __future__ import annotations

from collections.abc import Mapping
import json
import math
from pathlib import Path
from typing import Any

from experiments.runtime.dependency_profiles import get_dependency_profile
from main.core.digest import build_stable_digest


DEFAULT_INDEPENDENT_SEMANTIC_QUALITY_EVALUATOR_PATH = (
    Path(__file__).resolve().parents[2]
    / "configs"
    / "independent_semantic_quality_evaluator.json"
)
INDEPENDENT_SEMANTIC_FEATURE_BACKEND = "dinov2_cls_image_embedding"
INDEPENDENT_SEMANTIC_FEATURE_DIMENSION = 768


class IndependentSemanticQualityProtocolError(ValueError):
    """表示独立语义评估器的模型、预处理或依赖身份发生漂移."""


def _exact_direct_dependency_versions(profile: Any) -> dict[str, str]:
    """从冻结 profile 的直接依赖恢复包名与精确版本."""

    versions: dict[str, str] = {}
    for requirement in profile.direct_requirements:
        package_name, version = requirement.split("==", maxsplit=1)
        versions[package_name.lower()] = version
    return versions


def load_independent_semantic_quality_evaluator(
    path: str | Path = DEFAULT_INDEPENDENT_SEMANTIC_QUALITY_EVALUATOR_PATH,
) -> dict[str, Any]:
    """读取冻结评估器, 并复验模型族隔离、特征规则和完整依赖锁."""

    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise IndependentSemanticQualityProtocolError(
            "独立语义质量评估器配置无法读取"
        ) from exc
    model = payload.get("model_contract")
    feature = payload.get("feature_contract")
    preprocessing = payload.get("preprocessing_contract")
    independence = payload.get("independence_contract")
    dependency = payload.get("dependency_contract")
    if not all(
        isinstance(value, Mapping)
        for value in (model, feature, preprocessing, independence, dependency)
    ):
        raise IndependentSemanticQualityProtocolError(
            "独立语义评估器组件必须是对象"
        )
    if not all(
        (
            payload.get("protocol_schema")
            == "independent_semantic_quality_evaluator_v1",
            payload.get("evaluator_id")
            == "independent_visual_semantic_dinov2_base",
            model.get("model_id") == "facebook/dinov2-base",
            model.get("model_revision")
            == "f9e44c814b77203eaa57a6bdbbd535f21ede1415",
            model.get("model_class") == "Dinov2Model",
            model.get("model_family")
            == "dinov2_self_supervised_vision_transformer",
            feature.get("feature_layer") == "last_hidden_state_cls_token",
            feature.get("feature_dimension")
            == INDEPENDENT_SEMANTIC_FEATURE_DIMENSION,
            feature.get("feature_dtype") == "float32",
            feature.get("normalization") == "l2",
            feature.get("similarity_metric") == "cosine_similarity",
            preprocessing.get("image_processor_class") == "BitImageProcessor",
            preprocessing.get("color_mode") == "RGB",
            preprocessing.get("resize_shortest_edge") == 256,
            preprocessing.get("center_crop_height") == 224,
            preprocessing.get("center_crop_width") == 224,
            preprocessing.get("interpolation") == "bicubic",
            math.isclose(
                float(preprocessing.get("rescale_factor", math.nan)),
                1.0 / 255.0,
                rel_tol=0.0,
                abs_tol=1e-15,
            ),
            preprocessing.get("image_mean") == [0.485, 0.456, 0.406],
            preprocessing.get("image_std") == [0.229, 0.224, 0.225],
            independence.get("quality_evidence_role")
            == "independent_semantic_preservation_primary",
            independence.get("optimization_or_detection_usage") == "forbidden",
            independence.get("quality_evaluator_family")
            != independence.get("method_condition_encoder_family"),
            dependency.get("dependency_profile_id") == "sd35_method_runtime_gpu",
            dependency.get("complete_hash_lock_path")
            == "configs/dependency_profiles/sd35_method_runtime_gpu_lock.txt",
        )
    ):
        raise IndependentSemanticQualityProtocolError(
            "独立语义评估器冻结身份发生漂移"
        )
    profile = get_dependency_profile(str(dependency["dependency_profile_id"]))
    versions = _exact_direct_dependency_versions(profile)
    required = {
        str(name).lower(): str(version)
        for name, version in dependency.get(
            "required_direct_dependencies", {}
        ).items()
    }
    if (
        not profile.formal_ready
        or not profile.complete_hash_lock_present
        or not profile.complete_hash_lock_digest
        or profile.complete_hash_lock_path != dependency["complete_hash_lock_path"]
        or any(versions.get(name) != version for name, version in required.items())
    ):
        raise IndependentSemanticQualityProtocolError(
            "独立语义评估器未绑定有效完整依赖锁"
        )
    resolved = dict(payload)
    resolved["dependency_profile_digest"] = profile.profile_digest
    resolved["complete_hash_lock_digest"] = profile.complete_hash_lock_digest
    resolved["independent_semantic_quality_protocol_digest"] = build_stable_digest(
        {
            **payload,
            "dependency_profile_digest": profile.profile_digest,
            "complete_hash_lock_digest": profile.complete_hash_lock_digest,
        }
    )
    return resolved


__all__ = [
    "INDEPENDENT_SEMANTIC_FEATURE_BACKEND",
    "INDEPENDENT_SEMANTIC_FEATURE_DIMENSION",
    "load_independent_semantic_quality_evaluator",
]
