"""为测试夹具构造可独立重算的正式检测记录."""

from __future__ import annotations

import math
from typing import Any, Mapping

from experiments.protocol.method_runtime_config import (
    FORMAL_METHOD_PACKAGE_ROOT,
    load_formal_method_runtime_config,
)
from experiments.runners.image_only_dataset_runtime import (
    formal_low_frequency_carrier_protocol_record,
)
from main.core.digest import build_stable_digest
from main.methods.carrier.high_frequency_tail import (
    HIGH_FREQUENCY_TAIL_PROTOCOL_DIGEST,
)
from main.methods.detection import (
    ImageOnlyMeasurementConfig,
    image_only_measurement_config_identity_record,
    recompute_image_only_measurement_digest_payload,
)
from main.methods.geometry import (
    ATTENTION_COORDINATE_CONVENTION,
    ATTENTION_GRID_ALIGN_CORNERS,
    DIRECT_QK_RELATION_SOURCE,
    attention_alignment_gate_record,
    attention_relation_component_protocol,
    recompute_attention_alignment_digest_payload,
)


_FORMAL_METHOD_CONFIG = load_formal_method_runtime_config(
    FORMAL_METHOD_PACKAGE_ROOT
)
_LF_PROTOCOL = formal_low_frequency_carrier_protocol_record()
def _synthetic_alignment_record(
    alignment: Mapping[str, Any],
    *,
    alignment_gate: Mapping[str, int | float],
    relation_sync_score: float,
    registration_confidence: float,
    registration_geometry_reliable: bool,
    component_protocol: Mapping[str, Any],
) -> dict[str, Any]:
    """为测试构造满足真实 decision 公式的最小 alignment 记录。"""

    if "layer_name" in alignment:
        resolved = dict(alignment)
        resolved["registration_geometry_reliable"] = (
            registration_geometry_reliable
        )
        return resolved

    token_count = 16
    anchor_count = int(alignment_gate["attention_anchor_count"])
    component_names = list(
        component_protocol["attention_relation_component_names"]
    )
    component_weights = list(
        component_protocol["attention_relation_component_weights"]
    )
    component_scores = {
        name: relation_sync_score for name in component_names
    }
    default_affine_transform = (
        ((1.0, 0.0, 0.01), (0.0, 1.0, 0.0))
        if registration_geometry_reliable
        else ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
    )
    affine_transform = alignment.get(
        "affine_transform",
        default_affine_transform,
    )
    objective_margin = 0.1 if registration_geometry_reliable else 0.0
    registration_objective_score = registration_confidence
    identity_registration_objective_score = (
        registration_objective_score - objective_margin
    )
    record = {
        "layer_name": _FORMAL_METHOD_CONFIG.attention_module_names[0],
        "token_indices": list(range(token_count)),
        "affine_transform": [list(row) for row in affine_transform],
        "expected_anchor_indices": list(range(anchor_count)),
        "observed_anchor_indices": list(range(anchor_count)),
        "inlier_mask": [True] * anchor_count,
        **alignment_gate,
        "inlier_ratio": 1.0,
        "mean_inlier_residual": 0.0,
        "relation_sync_score": relation_sync_score,
        "relation_component_scores": dict(component_scores),
        "observation_relation_score": max(
            registration_confidence,
            0.1,
        ),
        "observation_relation_component_scores": dict(component_scores),
        "identity_observation_relation_score": 0.0,
        "identity_observation_relation_component_scores": {
            name: 0.0 for name in component_names
        },
        "registration_alignment_gain": max(
            registration_confidence,
            0.1,
        ),
        "bidirectional_relation_score": registration_confidence,
        "registration_objective_score": registration_objective_score,
        "identity_registration_objective_score": (
            identity_registration_objective_score
        ),
        "registration_objective_margin": objective_margin,
        "registration_coverage_penalty": 0.0,
        "canonical_coverage_ratio": 1.0,
        "observation_coverage_ratio": 1.0,
        "canonical_unique_ratio": 1.0,
        "observation_unique_ratio": 1.0,
        "canonical_token_weights": [1.0] * token_count,
        "stable_pair_weight_identity_digest": "1" * 64,
        "observed_pair_weight_realization_digest": "2" * 64,
        "canonical_pair_weight_realization_digest": "3" * 64,
        "attention_relation_source": DIRECT_QK_RELATION_SOURCE,
        "attention_relation_active_component_names": list(
            component_protocol[
                "attention_relation_active_component_names"
            ]
        ),
        "attention_relation_component_weights": component_weights,
        "attention_relation_component_protocol_digest": (
            component_protocol[
                "attention_relation_component_protocol_digest"
            ]
        ),
        "attention_relation_component_identity_digest": "4" * 64,
        "attention_relation_keyed_projection_digest": "5" * 64,
        "attention_relation_qk_operator_metadata_digest": "6" * 64,
        "attention_relation_qk_operator_metadata_ready": True,
        "attention_relation_qk_atomic_content_digest": "7" * 64,
        "attention_relation_qk_atomic_content_ready": True,
        "registration_confidence": registration_confidence,
        "geometry_reliable": registration_geometry_reliable,
        "metadata": {
            "attention_alignment_gate": dict(alignment_gate),
            **alignment_gate,
            "coordinate_convention": ATTENTION_COORDINATE_CONVENTION,
            "grid_align_corners": ATTENTION_GRID_ALIGN_CORNERS,
        },
        "registration_geometry_reliable": (
            registration_geometry_reliable
        ),
    }
    record["alignment_digest"] = build_stable_digest(
        recompute_attention_alignment_digest_payload(record)
    )
    return record


def bind_formal_detection_record(
    record: Mapping[str, Any],
    *,
    method_role: str = "full_dual_chain",
    lf_weight: float = _FORMAL_METHOD_CONFIG.lf_detection_score_weight,
    tail_robust_weight: float = (
        _FORMAL_METHOD_CONFIG.tail_robust_detection_score_weight
    ),
    tail_fraction: float = _FORMAL_METHOD_CONFIG.tail_fraction,
) -> dict[str, Any]:
    """补齐与 detector digest 同构的记录级科学正文.

    该 helper 只服务测试夹具. 正式运行仍必须由核心检测器直接产生记录,
    不得在业务路径中调用本函数补写证据.
    """

    if (
        type(lf_weight) is not float
        or type(tail_robust_weight) is not float
        or not math.isclose(
            lf_weight + tail_robust_weight,
            1.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or type(tail_fraction) is not float
        or tail_fraction != _FORMAL_METHOD_CONFIG.tail_fraction
    ):
        raise ValueError("测试内容载体权重或 tail_fraction 无效")
    resolved = dict(record)
    metadata = dict(resolved.get("metadata", {}))
    alignment_gate = attention_alignment_gate_record(
        _FORMAL_METHOD_CONFIG.attention_anchor_count,
        _FORMAL_METHOD_CONFIG.attention_residual_threshold,
        _FORMAL_METHOD_CONFIG.attention_minimum_inlier_ratio,
    )
    metadata.update(alignment_gate)
    metadata["attention_alignment_gate"] = dict(alignment_gate)
    component_protocol = attention_relation_component_protocol(
        _FORMAL_METHOD_CONFIG.attention_relation_component_weights
    )
    alignment = resolved.get("alignment")
    supplied_geometry_score = resolved.get("attention_geometry_score")
    supplied_raw_geometry_score = resolved.get(
        "raw_attention_geometry_score",
        supplied_geometry_score,
    )
    attention_geometry_enabled = bool(
        metadata.get(
            "attention_geometry_enabled",
            alignment is not None or supplied_raw_geometry_score is not None,
        )
    )
    image_alignment_enabled = bool(
        metadata.get("image_alignment_enabled", alignment is not None)
    )
    if isinstance(alignment, Mapping):
        relation_sync_score_value = alignment.get(
            "relation_sync_score",
            supplied_geometry_score,
        )
        if relation_sync_score_value is None:
            relation_sync_score_value = 0.0
        relation_sync_score = float(relation_sync_score_value)
        raw_geometry_score = float(
            relation_sync_score
            if supplied_raw_geometry_score is None
            else supplied_raw_geometry_score
        )
        registration_confidence_value = alignment.get(
            "registration_confidence",
            resolved.get("registration_confidence"),
        )
        if registration_confidence_value is None:
            registration_confidence_value = 0.0
        registration_confidence = float(
            registration_confidence_value
        )
        registration_geometry_reliable = alignment.get(
            "registration_geometry_reliable",
            alignment.get("geometry_reliable"),
        )
        if type(registration_geometry_reliable) is not bool:
            requested_geometry_reliable = resolved.get(
                "geometry_reliable"
            )
            registration_geometry_reliable = (
                requested_geometry_reliable
                if type(requested_geometry_reliable) is bool
                else False
            )
        alignment = _synthetic_alignment_record(
            alignment,
            alignment_gate=alignment_gate,
            relation_sync_score=relation_sync_score,
            registration_confidence=registration_confidence,
            registration_geometry_reliable=(
                registration_geometry_reliable
            ),
            component_protocol=component_protocol,
        )
        resolved["alignment"] = alignment
        resolved["attention_geometry_score"] = relation_sync_score
        resolved["raw_attention_geometry_score"] = raw_geometry_score
        resolved["registration_confidence"] = registration_confidence
    elif alignment is None:
        resolved["attention_geometry_score"] = None
        resolved["raw_attention_geometry_score"] = (
            float(supplied_raw_geometry_score)
            if attention_geometry_enabled
            and supplied_raw_geometry_score is not None
            else None
        )
        resolved["registration_confidence"] = None
        resolved["attention_sync_score"] = None
    supplied_content_score = float(resolved["content_score"])
    lf_score = float(resolved.get("lf_score", supplied_content_score))
    tail_score = float(
        resolved.get("tail_robust_score", supplied_content_score)
    )
    content_score = lf_weight * lf_score + tail_robust_weight * tail_score
    aligned_score_value = resolved.get("aligned_content_score")
    aligned_score = (
        None if aligned_score_value is None else float(aligned_score_value)
    )
    if not image_alignment_enabled:
        aligned_score = None
    aligned_lf_score = (
        None
        if aligned_score is None
        else float(resolved.get("aligned_lf_score", aligned_score))
    )
    aligned_tail_score = (
        None
        if aligned_score is None
        else float(
            resolved.get("aligned_tail_robust_score", aligned_score)
        )
    )
    aligned_score = (
        None
        if aligned_lf_score is None or aligned_tail_score is None
        else (
            lf_weight * aligned_lf_score
            + tail_robust_weight * aligned_tail_score
        )
    )
    model_id = str(metadata.get("model_id", _FORMAL_METHOD_CONFIG.model_id))
    attention_stable_token_fraction = float(
        metadata.get(
            "attention_stable_token_fraction",
            _FORMAL_METHOD_CONFIG.attention_stable_token_fraction,
        )
    )
    attention_unstable_pair_weight = float(
        metadata.get(
            "attention_unstable_pair_weight",
            _FORMAL_METHOD_CONFIG.attention_unstable_pair_weight,
        )
    )
    lf_template_sha256 = str(
        resolved.get(
            "lf_template_content_sha256",
            metadata.get("lf_template_content_sha256", "a" * 64),
        )
    )
    protocol_digest = _LF_PROTOCOL["lf_carrier_protocol_digest"]
    if tail_fraction != _FORMAL_METHOD_CONFIG.tail_fraction:
        raise ValueError("测试记录必须使用正式20%高频尾部比例")
    tail_protocol_digest = HIGH_FREQUENCY_TAIL_PROTOCOL_DIGEST
    tail_template_sha256 = str(
        resolved.get(
            "tail_template_content_sha256",
            metadata.get("tail_template_content_sha256", "c" * 64),
        )
    )
    tail_template_shape_value = resolved.get(
        "tail_template_shape",
        metadata.get("tail_template_shape", [1, 1, 1, 10]),
    )
    if (
        not isinstance(tail_template_shape_value, (list, tuple))
        or len(tail_template_shape_value) != 4
        or any(
            type(value) is not int or value <= 0
            for value in tail_template_shape_value
        )
    ):
        raise ValueError("测试 tail 模板形状必须是4维正整数序列")
    tail_template_shape = list(tail_template_shape_value)
    tail_template_element_count = (
        math.prod(tail_template_shape) if tail_template_shape else 0
    )
    tail_selected_element_count = (
        max(1, (tail_template_element_count + 4) // 5)
        if tail_template_element_count
        else 0
    )
    tail_threshold = 1.0
    tail_retained_fraction = (
        tail_selected_element_count / tail_template_element_count
        if tail_template_element_count
        else 0.0
    )
    component_weights = list(
        _FORMAL_METHOD_CONFIG.attention_relation_component_weights
    )
    metadata.update(
        {
            "attention_geometry_enabled": attention_geometry_enabled,
            "image_alignment_enabled": image_alignment_enabled,
        }
    )
    requested_geometry_reliable = resolved.get("geometry_reliable")
    registration_geometry_reliable = False
    if isinstance(alignment, Mapping):
        registration_geometry_reliable = alignment.get(
            "registration_geometry_reliable",
            alignment.get("geometry_reliable"),
        )
        if type(registration_geometry_reliable) is not bool:
            registration_geometry_reliable = False
    default_stable_pair_identity_ready = (
        requested_geometry_reliable
        if type(requested_geometry_reliable) is bool
        else registration_geometry_reliable
    )
    for field_name, default_value in {
        "stable_token_selection_digest": "",
        "stable_pair_weight_identity_digest": "",
        "observed_pair_weight_realization_digest": "",
        "aligned_pair_weight_realization_digest": "",
        "stable_pair_weight_identity_ready": (
            default_stable_pair_identity_ready
        ),
        "attention_relation_source": "",
        "attention_relation_component_identity_digest": "",
        "attention_relation_keyed_projection_digest": "",
        "attention_relation_qk_operator_metadata_digest": "",
        "detection_qk_atomic_content_digest": "",
    }.items():
        metadata.setdefault(field_name, default_value)
    for field_name in (
        "raw_content_margin",
        "aligned_content_margin",
        "positive_by_content",
        "geometry_reliable",
        "content_failure_reason",
        "rescue_eligible",
        "rescue_applied",
        "evidence_positive",
    ):
        resolved.pop(field_name, None)
    for field_name in (
        "content_threshold",
        "geometry_score_threshold",
        "registration_confidence_threshold",
        "attention_sync_score_threshold",
        "rescue_margin_low",
    ):
        metadata.pop(field_name, None)
    resolved.update(
        {
            "lf_score": lf_score,
            "tail_robust_score": tail_score,
            "content_score": content_score,
            "lf_weight": lf_weight,
            "tail_robust_weight": tail_robust_weight,
            "tail_fraction": tail_fraction,
            "lf_carrier_protocol_digest": protocol_digest,
            "lf_template_content_sha256": lf_template_sha256,
            "tail_carrier_protocol_digest": tail_protocol_digest,
            "tail_template_content_sha256": tail_template_sha256,
            "tail_template_shape": list(tail_template_shape),
            "tail_template_element_count": tail_template_element_count,
            "tail_selected_element_count": tail_selected_element_count,
            "tail_threshold": tail_threshold,
            "tail_retained_fraction": tail_retained_fraction,
            "aligned_lf_score": aligned_lf_score,
            "aligned_tail_robust_score": aligned_tail_score,
            "aligned_content_score": aligned_score,
            "raw_attention_geometry_score": resolved.get(
                "raw_attention_geometry_score",
                resolved.get("attention_geometry_score"),
            ),
            "metadata": metadata,
        }
    )
    measurement_config = ImageOnlyMeasurementConfig(
        model_id=model_id,
        model_revision=_FORMAL_METHOD_CONFIG.model_revision,
        vae_class_name=_FORMAL_METHOD_CONFIG.vae_class_name,
        transformer_class_name=_FORMAL_METHOD_CONFIG.transformer_class_name,
        scheduler_class_name=_FORMAL_METHOD_CONFIG.scheduler_class_name,
        vae_scaling_factor=_FORMAL_METHOD_CONFIG.vae_scaling_factor,
        vae_shift_factor=_FORMAL_METHOD_CONFIG.vae_shift_factor,
        latent_torch_dtype=_FORMAL_METHOD_CONFIG.latent_torch_dtype,
        width=_FORMAL_METHOD_CONFIG.width,
        height=_FORMAL_METHOD_CONFIG.height,
        inference_steps=_FORMAL_METHOD_CONFIG.inference_steps,
        public_detection_schedule_index=(
            _FORMAL_METHOD_CONFIG.public_detection_schedule_index
        ),
        public_detection_noise_prg_protocol=(
            _FORMAL_METHOD_CONFIG.public_detection_noise_prg_protocol
        ),
        public_detection_noise_domain=(
            _FORMAL_METHOD_CONFIG.public_detection_noise_domain
        ),
        public_detection_conditioning_protocol=(
            _FORMAL_METHOD_CONFIG.public_detection_conditioning_protocol
        ),
        public_detection_condition_text=(
            _FORMAL_METHOD_CONFIG.public_detection_condition_text
        ),
        max_attention_tokens=_FORMAL_METHOD_CONFIG.max_attention_tokens,
        attention_coordinate_convention=(
            _FORMAL_METHOD_CONFIG.attention_coordinate_convention
        ),
        attention_grid_align_corners=(
            _FORMAL_METHOD_CONFIG.attention_grid_align_corners
        ),
        attention_module_names=_FORMAL_METHOD_CONFIG.attention_module_names,
        attention_anchor_count=_FORMAL_METHOD_CONFIG.attention_anchor_count,
        attention_residual_threshold=(
            _FORMAL_METHOD_CONFIG.attention_residual_threshold
        ),
        attention_minimum_inlier_ratio=(
            _FORMAL_METHOD_CONFIG.attention_minimum_inlier_ratio
        ),
        low_frequency_config=(
            _FORMAL_METHOD_CONFIG.low_frequency_carrier_config
        ),
        lf_weight=lf_weight,
        tail_robust_weight=tail_robust_weight,
        tail_fraction=tail_fraction,
        keyed_prg_version=_FORMAL_METHOD_CONFIG.keyed_prg_version,
        attention_stable_token_fraction=attention_stable_token_fraction,
        attention_unstable_pair_weight=attention_unstable_pair_weight,
        attention_relation_component_weights=tuple(component_weights),
        method_role=method_role,
    )
    measurement_config_identity = image_only_measurement_config_identity_record(
        measurement_config,
        attention_geometry_enabled=metadata["attention_geometry_enabled"],
        image_alignment_enabled=metadata["image_alignment_enabled"],
    )
    measurement_config_digest = measurement_config_identity[
        "image_only_measurement_config_digest"
    ]
    metadata["image_only_measurement_config_digest"] = measurement_config_digest
    resolved["image_only_measurement_config_digest"] = measurement_config_digest
    resolved["measurement_digest"] = build_stable_digest(
        recompute_image_only_measurement_digest_payload(resolved)
    )
    return resolved


__all__ = ["bind_formal_detection_record"]
