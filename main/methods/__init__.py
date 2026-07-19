"""SLM-WM 正式内容双链方法公开接口，按需加载科学依赖。"""

from importlib import import_module
from typing import Any


__all__ = [
    "ATTENTION_ALIGNMENT_LAYER_SELECTION_RULE",
    "ATTENTION_IMAGE_PADDING_MODE",
    "ATTENTION_IMAGE_QUANTIZATION_PROTOCOL",
    "ATTENTION_IMAGE_RESAMPLING_MODE",
    "ATTENTION_RELATION_COMPONENT_NAMES",
    "ATTENTION_RELATION_COMPONENT_WEIGHTS",
    "ATTENTION_RELATION_NUMERICAL_EPSILON",
    "DIRECT_QK_RELATION_SOURCE",
    "FROZEN_SD35_ATTENTION_MODULE_NAMES",
    "HIGH_FREQUENCY_TAIL_PROTOCOL_DIGEST",
    "IMAGE_ONLY_EXTRACTION_PROFILE_SCHEMA",
    "IMAGE_ONLY_IMAGE_PREPROCESSING_PROTOCOL",
    "IMAGE_ONLY_MEASUREMENT_CONFIG_SCHEMA",
    "IMAGE_ONLY_VAE_ENCODING_PROTOCOL",
    "KEYED_PRG_VERSION",
    "METHOD_DEFINITION_SCHEMA",
    "AttentionAlignmentResult",
    "AttentionGeometryGradient",
    "AttentionRelationDescriptor",
    "BlindContentScore",
    "ContentCarrierUpdateResult",
    "ContentObservationRuntimeResult",
    "DifferentiableAttentionRecorder",
    "DualChainWriteBudget",
    "DualChainWriteResult",
    "GeometrySyncUpdate",
    "HighFrequencyTailCarrierTemplate",
    "ImageOnlyMeasurementConfig",
    "ImageOnlyMeasurementResult",
    "LowFrequencyCarrierTemplate",
    "QKAttentionRelation",
    "attention_geometry_score",
    "attention_relation_component_protocol",
    "build_attention_geometry_sync_update",
    "build_attention_relation_descriptor",
    "build_content_carrier_update",
    "build_content_observation_routing",
    "build_high_frequency_tail_template",
    "build_low_frequency_template",
    "compose_dual_chain_update_once",
    "compute_attention_geometry_gradient",
    "compute_blind_content_score",
    "formal_dual_chain_write_budget",
    "image_only_measurement_config_identity_record",
    "measure_image_only_watermark",
    "project_image_only_measurement_record",
    "qk_operator_metadata_records_digest",
    "qk_operator_metadata_records_ready",
    "qk_self_attention",
    "recompute_attention_alignment_digest_payload",
    "recompute_image_only_measurement_digest_payload",
    "recover_attention_affine_alignment",
    "resample_attention_aligned_rgb_uint8",
    "select_image_only_alignment_candidate",
    "semantic_conditioned_latent_method_definition",
    "semantic_conditioned_latent_method_definition_digest",
    "validate_attention_alignment_record",
    "validate_attention_relation_component_weights",
    "validate_image_only_measurement_digest_record",
    "validate_image_only_measurement_projection_record",
]


_EXPORT_MODULES = {
    **{
        name: "main.methods.carrier"
        for name in (
            "BlindContentScore",
            "HIGH_FREQUENCY_TAIL_PROTOCOL_DIGEST",
            "HighFrequencyTailCarrierTemplate",
            "KEYED_PRG_VERSION",
            "LowFrequencyCarrierTemplate",
            "build_high_frequency_tail_template",
            "build_low_frequency_template",
            "compute_blind_content_score",
        )
    },
    **{
        name: "main.methods.carrier.content_update"
        for name in ("ContentCarrierUpdateResult", "build_content_carrier_update")
    },
    **{
        name: "main.methods.content.runtime_adapter"
        for name in (
            "ContentObservationRuntimeResult",
            "build_content_observation_routing",
        )
    },
    **{
        name: "main.methods.detection"
        for name in (
            "ATTENTION_ALIGNMENT_LAYER_SELECTION_RULE",
            "IMAGE_ONLY_EXTRACTION_PROFILE_SCHEMA",
            "IMAGE_ONLY_IMAGE_PREPROCESSING_PROTOCOL",
            "IMAGE_ONLY_MEASUREMENT_CONFIG_SCHEMA",
            "IMAGE_ONLY_VAE_ENCODING_PROTOCOL",
            "ImageOnlyMeasurementConfig",
            "ImageOnlyMeasurementResult",
            "image_only_measurement_config_identity_record",
            "measure_image_only_watermark",
            "project_image_only_measurement_record",
            "recompute_image_only_measurement_digest_payload",
            "select_image_only_alignment_candidate",
            "validate_image_only_measurement_digest_record",
            "validate_image_only_measurement_projection_record",
        )
    },
    **{
        name: "main.methods.geometry"
        for name in (
            "ATTENTION_IMAGE_PADDING_MODE",
            "ATTENTION_IMAGE_QUANTIZATION_PROTOCOL",
            "ATTENTION_IMAGE_RESAMPLING_MODE",
            "ATTENTION_RELATION_COMPONENT_NAMES",
            "ATTENTION_RELATION_COMPONENT_WEIGHTS",
            "ATTENTION_RELATION_NUMERICAL_EPSILON",
            "DIRECT_QK_RELATION_SOURCE",
            "FROZEN_SD35_ATTENTION_MODULE_NAMES",
            "AttentionAlignmentResult",
            "AttentionGeometryGradient",
            "AttentionRelationDescriptor",
            "DifferentiableAttentionRecorder",
            "QKAttentionRelation",
            "attention_geometry_score",
            "attention_relation_component_protocol",
            "build_attention_relation_descriptor",
            "compute_attention_geometry_gradient",
            "qk_operator_metadata_records_digest",
            "qk_operator_metadata_records_ready",
            "qk_self_attention",
            "recompute_attention_alignment_digest_payload",
            "recover_attention_affine_alignment",
            "resample_attention_aligned_rgb_uint8",
            "validate_attention_alignment_record",
            "validate_attention_relation_component_weights",
        )
    },
    **{
        name: "main.methods.geometry.sync_update"
        for name in ("GeometrySyncUpdate", "build_attention_geometry_sync_update")
    },
    **{
        name: "main.methods.method_definition"
        for name in (
            "METHOD_DEFINITION_SCHEMA",
            "semantic_conditioned_latent_method_definition",
            "semantic_conditioned_latent_method_definition_digest",
        )
    },
    **{
        name: "main.methods.update_composition"
        for name in (
            "DualChainWriteBudget",
            "DualChainWriteResult",
            "compose_dual_chain_update_once",
            "formal_dual_chain_write_budget",
        )
    },
}


def __getattr__(name: str) -> Any:
    """仅在公开符号被实际消费时加载对应科学模块。"""

    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """向交互式调用者公开稳定接口集合。"""

    return sorted((*globals(), *__all__))
