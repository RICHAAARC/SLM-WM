"""运行真实语义安全子空间嵌入和仅图像检测闭环。

该 runner 属于核心方法复现层, 在真实 SD3/SD3.5 latent 上计算分支风险、
完整特征 JVP/VJP Null Space、安全投影、真实 Q/K 注意力梯度和最终图像盲检。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
import math
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

from experiments.protocol.method_runtime_config import (
    FORMAL_METHOD_PACKAGE_ROOT,
    FormalBranchRiskConfig,
    load_formal_method_runtime_config,
)
from experiments.protocol.formal_randomization import (
    DEFAULT_FORMAL_RANDOMIZATION_REPEAT_ID,
    build_canonical_sd35_base_latent,
    formal_randomization_sample_reference,
    formal_random_trace_fields,
    formal_randomization_protocol_record,
    resolve_formal_randomization_repeat,
)
from experiments.protocol.detection_key_identity import (
    REGISTERED_WATERMARK_KEY_ROLE,
    REGISTERED_WRONG_KEY_ROLE,
    build_detection_key_plan_record,
    resolve_detection_key_material_and_identity,
)
from experiments.protocol.image_only_evidence import (
    FrozenEvidenceProtocol,
    decision_equivalent_score,
    validate_frozen_evidence_protocol_integrity,
)
from experiments.runtime.diffusion.prompt_saliency_model_loader import (
    load_prompt_saliency_clip_runtime,
)
from experiments.protocol.content_routing_reference_quantile import (
    ContentRoutingReferenceScalars,
)
from experiments.protocol.attacks import (
    attack_config_digest,
    default_attack_configs,
    formal_attack_seed_protocol_record,
    formal_attack_seed_random,
)
from experiments.runtime.image_attacks import apply_standard_image_attack
from experiments.runtime.diffusion.sd3_pipeline_runtime import load_pipeline, tensor_norm
from experiments.runtime.image_metrics import compute_image_quality_metrics
from experiments.runtime.model_sources import (
    require_registered_model_reference,
)
from experiments.runtime.repository_environment import file_digest, resolve_code_version
from experiments.runtime.resume_checkpoint import (
    persist_completed_unit_from_manifest,
)
from experiments.runtime.scientific_unit_provenance import (
    build_scientific_unit_provenance,
    validate_scientific_unit_provenance,
)
from experiments.runtime.scientific_content_binding import (
    SCIENTIFIC_CONTENT_BINDING_SCHEMA,
    build_scientific_content_binding_record,
    canonical_rgb_uint8_content_record,
    read_canonical_rgb_uint8_content_record,
    recompute_scientific_content_binding_digest,
)
from experiments.artifacts.artifact_manifest import build_artifact_manifest
from main.core.digest import (
    TENSOR_CONTENT_DIGEST_VERSION,
    build_stable_digest,
    tensor_content_sha256,
)
from main.core.keyed_prg import (
    KEYED_PRG_VERSION,
    build_keyed_gaussian_tensor,
    keyed_prg_protocol_record,
    require_supported_keyed_prg_version,
)
from main.methods.carrier.keyed_tensor import (
    LowFrequencyCarrierConfig,
)
from main.methods.carrier.high_frequency_tail import (
    HIGH_FREQUENCY_TAIL_PROTOCOL_DIGEST,
)
from main.methods.detection import (
    ImageOnlyMeasurementConfig,
    image_only_measurement_config_identity_record,
    measure_image_only_watermark,
    recompute_image_only_measurement_digest_payload,
)
from main.methods.geometry import (
    ATTENTION_COORDINATE_CONVENTION,
    ATTENTION_GRID_ALIGN_CORNERS,
    ATTENTION_RELATION_COMPONENT_NAMES,
    DIRECT_QK_RELATION_SOURCE,
    DifferentiableAttentionRecorder,
    StableAttentionTokenSelection,
    attention_geometry_component_scores,
    attention_geometry_score,
    attention_relation_component_protocol,
    attention_relation_stability_map,
    build_attention_relation_graph_identity,
    build_stable_attention_pair_weights,
    compute_attention_geometry_gradient,
    qk_atomic_evaluation_records_digest,
    qk_atomic_evaluation_records_ready,
    qk_operator_metadata_records_digest,
    qk_operator_metadata_records_ready,
    resample_attention_aligned_rgb_uint8,
    select_stable_attention_tokens,
    validate_attention_alignment_gate,
    validate_attention_relation_component_weights,
)
from main.methods.method_definition import (
    semantic_conditioned_latent_method_definition,
    semantic_conditioned_latent_method_definition_digest,
)
from main.methods.update_composition import (
    build_quantized_composition_candidate,
    build_risk_bounded_update,
    iter_quantized_composition_candidates,
    rescale_risk_bounded_update,
    compose_dual_chain_update_once,
    formal_dual_chain_write_budget,
)
from main.methods.carrier.low_frequency import (
    build_low_frequency_template as build_formal_low_frequency_template,
)
from main.methods.carrier.high_frequency_tail import (
    build_high_frequency_tail_template,
)
from main.methods.carrier.content_update import build_content_carrier_update
from main.methods.content.local_sensitivity import build_public_probe_identity
from main.methods.content.runtime_adapter import build_content_observation_routing
from main.methods.geometry.sync_update import (
    _build_attention_geometry_sync_update_with_evidence,
    _evaluate_post_write_geometry_relation,
)


_FORMAL_METHOD_CONFIG = load_formal_method_runtime_config(
    FORMAL_METHOD_PACKAGE_ROOT
)
_FORMAL_RANDOMIZATION_CONFIG = formal_randomization_protocol_record()
_FORMAL_ATTENTION_COMPONENT_WEIGHT_PROTOCOLS = (
    _FORMAL_METHOD_CONFIG.attention_relation_component_weights,
    (0.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
    (1.0 / 3.0, 0.0, 1.0 / 3.0, 1.0 / 3.0),
    (1.0 / 3.0, 1.0 / 3.0, 0.0, 1.0 / 3.0),
    (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 0.0),
)
@dataclass(frozen=True)
class SemanticWatermarkRuntimeConfig:
    """定义一次真实方法嵌入和仅图像检测运行。"""

    model_family: str = _FORMAL_METHOD_CONFIG.model_family
    model_id: str = _FORMAL_METHOD_CONFIG.model_id
    model_revision: str = _FORMAL_METHOD_CONFIG.model_revision
    vision_model_id: str = _FORMAL_METHOD_CONFIG.vision_model_id
    vision_model_revision: str = _FORMAL_METHOD_CONFIG.vision_model_revision
    formal_method_config_digest: str = (
        _FORMAL_METHOD_CONFIG.formal_method_config_digest
    )
    pipeline_class_name: str = _FORMAL_METHOD_CONFIG.pipeline_class_name
    vae_class_name: str = _FORMAL_METHOD_CONFIG.vae_class_name
    transformer_class_name: str = _FORMAL_METHOD_CONFIG.transformer_class_name
    scheduler_class_name: str = _FORMAL_METHOD_CONFIG.scheduler_class_name
    vae_scaling_factor: float = _FORMAL_METHOD_CONFIG.vae_scaling_factor
    vae_shift_factor: float = _FORMAL_METHOD_CONFIG.vae_shift_factor
    latent_torch_dtype: str = _FORMAL_METHOD_CONFIG.latent_torch_dtype
    device_name: str = "cuda"
    torch_dtype: str = _FORMAL_METHOD_CONFIG.latent_torch_dtype
    vision_torch_dtype: str = _FORMAL_METHOD_CONFIG.vision_torch_dtype
    hf_token_env: str = "HF_TOKEN"
    prompt: str = _FORMAL_METHOD_CONFIG.prompt
    prompt_id: str = "runtime_prompt"
    split: str = "dev"
    negative_prompt: str = _FORMAL_METHOD_CONFIG.negative_prompt
    key_material: str = "slm_wm_runtime_key"
    seed: int = _FORMAL_METHOD_CONFIG.seed
    randomization_repeat_id: str = DEFAULT_FORMAL_RANDOMIZATION_REPEAT_ID
    generation_seed_index: int = 0
    generation_seed_offset: int = 0
    watermark_key_index: int = 0
    watermark_key_seed_random: int = 0
    formal_randomization_protocol_digest: str = _FORMAL_RANDOMIZATION_CONFIG[
        "formal_randomization_protocol_digest"
    ]
    width: int = _FORMAL_METHOD_CONFIG.width
    height: int = _FORMAL_METHOD_CONFIG.height
    inference_steps: int = _FORMAL_METHOD_CONFIG.inference_steps
    guidance_scale: float = _FORMAL_METHOD_CONFIG.guidance_scale
    risk_signal_calibration_protocol: str = (
        _FORMAL_METHOD_CONFIG.risk_signal_calibration_protocol
    )
    risk_image_signal_interpolation_mode: str = (
        _FORMAL_METHOD_CONFIG.risk_image_signal_interpolation_mode
    )
    risk_image_signal_align_corners: bool = (
        _FORMAL_METHOD_CONFIG.risk_image_signal_align_corners
    )
    risk_attention_signal_interpolation_mode: str = (
        _FORMAL_METHOD_CONFIG.risk_attention_signal_interpolation_mode
    )
    risk_attention_signal_align_corners: bool = (
        _FORMAL_METHOD_CONFIG.risk_attention_signal_align_corners
    )
    risk_neutral_texture_value: float = (
        _FORMAL_METHOD_CONFIG.risk_neutral_texture_value
    )
    risk_eligibility_comparison: str = (
        _FORMAL_METHOD_CONFIG.risk_eligibility_comparison
    )
    risk_budget_broadcast_protocol: str = (
        _FORMAL_METHOD_CONFIG.risk_budget_broadcast_protocol
    )
    risk_zero_support_protocol: str = (
        _FORMAL_METHOD_CONFIG.risk_zero_support_protocol
    )
    risk_parameter_protocol: str = "formal_reference"
    risk_bounded_scale_protocol: str = (
        _FORMAL_METHOD_CONFIG.risk_bounded_scale_protocol
    )
    risk_bounded_scale_direction_epsilon: float = (
        _FORMAL_METHOD_CONFIG.risk_bounded_scale_direction_epsilon
    )
    lf_content_risk_config: FormalBranchRiskConfig = (
        _FORMAL_METHOD_CONFIG.lf_content_risk_config
    )
    tail_robust_risk_config: FormalBranchRiskConfig = (
        _FORMAL_METHOD_CONFIG.tail_robust_risk_config
    )
    attention_geometry_risk_config: FormalBranchRiskConfig = (
        _FORMAL_METHOD_CONFIG.attention_geometry_risk_config
    )
    attention_injection_steps: tuple[int, ...] = (
        _FORMAL_METHOD_CONFIG.injection_step_indices
    )
    injection_step_indices: tuple[int, ...] = _FORMAL_METHOD_CONFIG.injection_step_indices
    lf_relative_strength: float = _FORMAL_METHOD_CONFIG.lf_relative_strength
    tail_relative_strength: float = _FORMAL_METHOD_CONFIG.tail_relative_strength
    attention_relative_strength: float = _FORMAL_METHOD_CONFIG.attention_relative_strength
    lf_kernel_size: int = _FORMAL_METHOD_CONFIG.lf_kernel_size
    lf_stride: int = _FORMAL_METHOD_CONFIG.lf_stride
    lf_padding: int = _FORMAL_METHOD_CONFIG.lf_padding
    lf_boundary_mode: str = _FORMAL_METHOD_CONFIG.lf_boundary_mode
    lf_ceil_mode: bool = _FORMAL_METHOD_CONFIG.lf_ceil_mode
    lf_count_include_pad: bool = _FORMAL_METHOD_CONFIG.lf_count_include_pad
    lf_divisor_override: int | None = (
        _FORMAL_METHOD_CONFIG.lf_divisor_override
    )
    lf_detection_score_weight: float = (
        _FORMAL_METHOD_CONFIG.lf_detection_score_weight
    )
    tail_robust_detection_score_weight: float = (
        _FORMAL_METHOD_CONFIG.tail_robust_detection_score_weight
    )
    attention_stable_token_fraction: float = (
        _FORMAL_METHOD_CONFIG.attention_stable_token_fraction
    )
    attention_unstable_pair_weight: float = (
        _FORMAL_METHOD_CONFIG.attention_unstable_pair_weight
    )
    attention_relation_component_weights: tuple[float, ...] = (
        _FORMAL_METHOD_CONFIG.attention_relation_component_weights
    )
    attention_anchor_count: int = (
        _FORMAL_METHOD_CONFIG.attention_anchor_count
    )
    attention_residual_threshold: float = (
        _FORMAL_METHOD_CONFIG.attention_residual_threshold
    )
    attention_minimum_inlier_ratio: float = (
        _FORMAL_METHOD_CONFIG.attention_minimum_inlier_ratio
    )
    attention_backtracking_factor: float = (
        _FORMAL_METHOD_CONFIG.attention_backtracking_factor
    )
    attention_backtracking_maximum_steps: int = (
        _FORMAL_METHOD_CONFIG.attention_backtracking_maximum_steps
    )
    minimum_final_image_attention_score_gain: float = (
        _FORMAL_METHOD_CONFIG.minimum_final_image_attention_score_gain
    )
    tail_fraction: float = _FORMAL_METHOD_CONFIG.tail_fraction
    keyed_prg_version: str = _FORMAL_METHOD_CONFIG.keyed_prg_version
    quantized_branch_composition_protocol: str = (
        _FORMAL_METHOD_CONFIG.quantized_branch_composition_protocol
    )
    quantized_branch_composition_order: tuple[str, ...] = (
        _FORMAL_METHOD_CONFIG.quantized_branch_composition_order
    )
    combined_budget_envelope_rule: str = (
        _FORMAL_METHOD_CONFIG.combined_budget_envelope_rule
    )
    quantized_budget_envelope_absolute_tolerance: float = (
        _FORMAL_METHOD_CONFIG.quantized_budget_envelope_absolute_tolerance
    )
    quantized_budget_envelope_backtracking_factor: float = (
        _FORMAL_METHOD_CONFIG.quantized_budget_envelope_backtracking_factor
    )
    quantized_budget_envelope_backtracking_maximum_steps: int = (
        _FORMAL_METHOD_CONFIG.quantized_budget_envelope_backtracking_maximum_steps
    )
    minimum_semantic_preservation_cosine: float = (
        _FORMAL_METHOD_CONFIG.minimum_semantic_preservation_cosine
    )
    maximum_handcrafted_structure_feature_relative_drift: float = (
        _FORMAL_METHOD_CONFIG.maximum_handcrafted_structure_feature_relative_drift
    )
    attention_operator_schedule_index: int = (
        _FORMAL_METHOD_CONFIG.attention_operator_schedule_index
    )
    public_detection_schedule_index: int = (
        _FORMAL_METHOD_CONFIG.public_detection_schedule_index
    )
    public_detection_noise_prg_protocol: str = (
        _FORMAL_METHOD_CONFIG.public_detection_noise_prg_protocol
    )
    public_detection_noise_domain: str = (
        _FORMAL_METHOD_CONFIG.public_detection_noise_domain
    )
    public_detection_conditioning_protocol: str = (
        _FORMAL_METHOD_CONFIG.public_detection_conditioning_protocol
    )
    public_detection_condition_text: str = (
        _FORMAL_METHOD_CONFIG.public_detection_condition_text
    )
    max_attention_tokens: int = _FORMAL_METHOD_CONFIG.max_attention_tokens
    attention_module_names: tuple[str, ...] = (
        _FORMAL_METHOD_CONFIG.attention_module_names
    )
    attention_alignment_layer_selection_rule: str = (
        _FORMAL_METHOD_CONFIG.attention_alignment_layer_selection_rule
    )
    image_alignment_resampling_mode: str = (
        _FORMAL_METHOD_CONFIG.image_alignment_resampling_mode
    )
    image_alignment_padding_mode: str = (
        _FORMAL_METHOD_CONFIG.image_alignment_padding_mode
    )
    image_alignment_quantization_protocol: str = (
        _FORMAL_METHOD_CONFIG.image_alignment_quantization_protocol
    )
    attention_coordinate_convention: str = (
        _FORMAL_METHOD_CONFIG.attention_coordinate_convention
    )
    attention_grid_align_corners: bool = (
        _FORMAL_METHOD_CONFIG.attention_grid_align_corners
    )
    semantic_routing_enabled: bool = True
    branch_risk_mode: str = "branch_specific"
    lf_enabled: bool = True
    tail_robust_enabled: bool = True
    tail_truncation_enabled: bool = True
    attention_geometry_enabled: bool = True
    image_alignment_enabled: bool = True
    standard_attack_profiles: tuple[str, ...] = ("full_main",)
    diffusion_attacks_enabled: bool = _FORMAL_METHOD_CONFIG.diffusion_attacks_enabled
    detector_guided_attack_threshold_protocol: dict[str, Any] | None = None
    output_dir: str = "outputs/semantic_watermark_runtime"

    def __post_init__(self) -> None:
        """集中校验重型运行配置。"""

        if self.detector_guided_attack_threshold_protocol is not None:
            if type(self.detector_guided_attack_threshold_protocol) is not dict:
                raise TypeError(
                    "detector_guided_attack_threshold_protocol 必须为 dict 或 None"
                )
            try:
                attack_protocol = FrozenEvidenceProtocol(
                    **self.detector_guided_attack_threshold_protocol
                )
                validate_frozen_evidence_protocol_integrity(attack_protocol)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "detector-guided attack 必须绑定完整可重建的冻结协议"
                ) from exc

        require_registered_model_reference(
            self.model_id,
            self.model_revision,
            required_usage_role="primary_diffusion_model",
        )
        require_registered_model_reference(
            self.vision_model_id,
            self.vision_model_revision,
            required_usage_role="semantic_condition_encoder",
        )
        if (
            not math.isfinite(self.minimum_final_image_attention_score_gain)
            or self.minimum_final_image_attention_score_gain <= 0.0
        ):
            raise ValueError(
                "minimum_final_image_attention_score_gain 必须为正有限数"
            )
        validate_attention_alignment_gate(
            self.attention_anchor_count,
            self.attention_residual_threshold,
            self.attention_minimum_inlier_ratio,
        )
        expected_method_settings = _FORMAL_METHOD_CONFIG.paper_method_settings()
        actual_method_settings: dict[str, Any] = {}
        for field_name in expected_method_settings:
            field_value = getattr(self, field_name)
            actual_method_settings[field_name] = (
                asdict(field_value)
                if isinstance(field_value, FormalBranchRiskConfig)
                else field_value
            )
        drifted_method_fields = tuple(
            field_name
            for field_name, expected_value in expected_method_settings.items()
            if actual_method_settings[field_name] != expected_value
            and not (
                field_name == "attention_relation_component_weights"
                and self.attention_relation_component_weights
                in _FORMAL_ATTENTION_COMPONENT_WEIGHT_PROTOCOLS
            )
        )
        risk_config_fields = {
            "lf_content_risk_config",
            "tail_robust_risk_config",
            "attention_geometry_risk_config",
        }
        if self.risk_parameter_protocol not in {
            "formal_reference",
            "single_model_internal_sensitivity",
        }:
            raise ValueError("risk_parameter_protocol 不是受治理的参数协议")
        sensitivity_override_ready = (
            self.risk_parameter_protocol
            == "single_model_internal_sensitivity"
            and bool(drifted_method_fields)
            and set(drifted_method_fields) <= risk_config_fields
        )
        if drifted_method_fields and not sensitivity_override_ready:
            raise ValueError(
                "方法运行设置必须精确继承 configs/model_sd35.yaml: "
                + ", ".join(drifted_method_fields)
            )
        if (
            self.torch_dtype != self.latent_torch_dtype
            or self.injection_step_indices != self.attention_injection_steps
        ):
            raise ValueError("方法运行别名字段必须与唯一正式配置保持一致")
        if self.device_name != "cuda":
            raise ValueError("正式真实方法运行要求 CUDA 设备")
        if self.injection_step_indices != (10,):
            raise ValueError("正式 callback 必须只在索引10执行一次写回")
        if type(self.tail_fraction) is not float or not 0.0 < self.tail_fraction <= 1.0:
            raise ValueError("tail_fraction 必须为 (0, 1] 内的精确 float")
        self.low_frequency_carrier_config
        if (
            type(self.lf_detection_score_weight) is not float
            or type(self.tail_robust_detection_score_weight) is not float
        ):
            raise TypeError("正式内容检测分支权重必须为精确 float")
        require_supported_keyed_prg_version(self.keyed_prg_version)
        if not 0.0 < self.attention_stable_token_fraction <= 1.0:
            raise ValueError(
                "attention_stable_token_fraction 必须位于 (0, 1]"
            )
        if not 0.0 <= self.attention_unstable_pair_weight < 1.0:
            raise ValueError(
                "attention_unstable_pair_weight 必须位于 [0, 1)"
            )
        validate_attention_relation_component_weights(
            self.attention_relation_component_weights
        )
        if not 0.0 < self.minimum_semantic_preservation_cosine <= 1.0:
            raise ValueError(
                "minimum_semantic_preservation_cosine 必须位于 (0, 1]"
            )
        if not (
            0.0
            <= self.maximum_handcrafted_structure_feature_relative_drift
            <= 1.0
        ):
            raise ValueError(
                "maximum_handcrafted_structure_feature_relative_drift 必须位于 [0, 1]"
            )
        if self.branch_risk_mode not in {"branch_specific", "shared_global"}:
            raise ValueError(
                "branch_risk_mode 必须为 branch_specific 或 shared_global"
            )
        if not self.lf_enabled and not self.tail_robust_enabled:
            raise ValueError("正式内容检测至少需要启用一个内容载体分支")
        if self.image_alignment_enabled and not self.attention_geometry_enabled:
            raise ValueError("图像配准要求启用真实注意力几何")
        if len(self.attention_module_names) < 2:
            raise ValueError("真实注意力关系稳定度至少需要两个 Q/K 注意力层")
        if len(set(self.attention_module_names)) != len(
            self.attention_module_names
        ):
            raise ValueError("attention_module_names 不得包含重复层名")
        if self.attention_module_names != (
            _FORMAL_METHOD_CONFIG.attention_module_names
        ):
            raise ValueError("正式运行不得改变冻结的精确注意力层集合")
        if (
            self.attention_coordinate_convention
            != ATTENTION_COORDINATE_CONVENTION
            or self.attention_grid_align_corners
            is not ATTENTION_GRID_ALIGN_CORNERS
        ):
            raise ValueError("注意力 token 与图像坐标约定必须匹配核心算子")
        if self.max_attention_tokens < 4:
            raise ValueError("max_attention_tokens 至少为 4")
        if self.split not in {"dev", "calibration", "test"}:
            raise ValueError("split 必须为 dev、calibration 或 test")
        repeat = resolve_formal_randomization_repeat(
            self.randomization_repeat_id
        )
        if (
            self.generation_seed_index != repeat.generation_seed_index
            or self.generation_seed_offset != repeat.generation_seed_offset
            or self.watermark_key_index != repeat.watermark_key_index
            or self.formal_randomization_protocol_digest
            != _FORMAL_RANDOMIZATION_CONFIG[
                "formal_randomization_protocol_digest"
            ]
        ):
            raise ValueError("方法运行随机化身份未匹配正式重复注册表")

    @property
    def low_frequency_carrier_config(self) -> LowFrequencyCarrierConfig:
        """返回嵌入和仅图像检测共同消费的冻结 LF 协议对象."""

        return LowFrequencyCarrierConfig(
            kernel_size=self.lf_kernel_size,
            stride=self.lf_stride,
            padding=self.lf_padding,
            boundary_mode=self.lf_boundary_mode,
            ceil_mode=self.lf_ceil_mode,
            count_include_pad=self.lf_count_include_pad,
            divisor_override=self.lf_divisor_override,
        )

    @property
    def carrier_model_reference(self) -> str:
        """返回同时绑定仓库和精确 revision 的公开载体标识."""

        return f"{self.model_id}@{self.model_revision}"


def _require_full_content_runtime_config(
    config: SemanticWatermarkRuntimeConfig,
) -> None:
    """在重型运行前拒绝尚未迁移到新链的消融配置。"""

    if type(config) is not SemanticWatermarkRuntimeConfig:
        raise TypeError("config 必须为精确 SemanticWatermarkRuntimeConfig")
    required_values = {
        "semantic_routing_enabled": True,
        "branch_risk_mode": "branch_specific",
        "lf_enabled": True,
        "tail_robust_enabled": True,
        "tail_truncation_enabled": True,
        "attention_geometry_enabled": True,
        "image_alignment_enabled": True,
        "risk_parameter_protocol": "formal_reference",
        "attention_relation_component_weights": (
            _FORMAL_METHOD_CONFIG.attention_relation_component_weights
        ),
    }
    drifted = tuple(
        field_name
        for field_name, expected in required_values.items()
        if getattr(config, field_name) != expected
    )
    if drifted:
        raise RuntimeError(
            "当前新正式 runtime 只允许 full_dual_chain；尚未迁移的消融配置必须失败关闭: "
            + ",".join(drifted)
        )


@dataclass(frozen=True)
class SemanticWatermarkRuntimeResult:
    """保存真实嵌入、图像输出和仅图像检测摘要。"""

    run_id: str
    run_decision: str
    clean_image_path: str
    watermarked_image_path: str
    update_record_path: str
    detection_record_path: str
    manifest_path: str
    update_count: int
    elapsed_seconds: float
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """转为可写入 JSON 的字典。"""

        return asdict(self)


@dataclass
class SemanticWatermarkRuntimeContext:
    """保存可跨 Prompt 复用的正式内容双链运行组件。"""

    pipeline: Any
    prompt_saliency_runtime: Any
    attention_modules: tuple[tuple[str, Any], ...]
    unconditional_prompt: Any
    unconditional_pooled: Any
    runtime_versions: dict[str, Any]


def load_semantic_watermark_runtime_context(
    config: SemanticWatermarkRuntimeConfig,
    *,
    verified_formal_execution_lock: Mapping[str, Any],
    repository_root: str | Path,
) -> SemanticWatermarkRuntimeContext:
    """只加载 SD3.5、Prompt-saliency、正式 Q/K 层与检测条件。"""

    _require_full_content_runtime_config(config)
    components = _load_content_runtime_smoke_components(
        config,
        verified_formal_execution_lock=verified_formal_execution_lock,
        repository_root=repository_root,
    )
    return SemanticWatermarkRuntimeContext(
        pipeline=components.pipeline,
        prompt_saliency_runtime=components.prompt_saliency_runtime,
        attention_modules=components.attention_modules,
        unconditional_prompt=components.unconditional_prompt,
        unconditional_pooled=components.unconditional_pooled,
        runtime_versions=components.runtime_versions,
    )


@dataclass(frozen=True)
class _ContentRuntimeSmokeComponents:
    """Minimal real components for the new content/QK smoke path."""

    pipeline: Any
    prompt_saliency_runtime: Any
    attention_modules: tuple[tuple[str, Any], ...]
    unconditional_prompt: Any
    unconditional_pooled: Any
    runtime_versions: dict[str, Any]


def _load_content_runtime_smoke_components(
    config: SemanticWatermarkRuntimeConfig,
    *,
    verified_formal_execution_lock: Mapping[str, Any],
    repository_root: str | Path,
) -> _ContentRuntimeSmokeComponents:
    """Load only SD3.5, formal Q/K layers, and the Prompt-saliency towers."""

    _require_full_content_runtime_config(config)
    pipeline, runtime_versions = load_pipeline(config)
    from diffusers.models.attention_processor import AttnProcessor

    pipeline.vae.set_attn_processor(AttnProcessor())
    for parameter in pipeline.transformer.parameters():
        parameter.requires_grad_(False)
    attention_modules = _attention_modules(pipeline, config.attention_module_names)
    unconditional_prompt, unconditional_pooled = _unconditional_embeddings(
        pipeline,
        pipeline._execution_device,
        config.public_detection_conditioning_protocol,
        config.public_detection_condition_text,
    )
    prompt_saliency_runtime = load_prompt_saliency_clip_runtime(
        "openai/clip-vit-base-patch32",
        "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268",
        config.device_name,
        local_files_only=True,
        verified_formal_execution_lock=verified_formal_execution_lock,
        repository_root=repository_root,
    )
    runtime_versions = {
        **runtime_versions,
        "content_runtime_operator": "formal_content_qk_single_write_v1",
        "prompt_saliency_model_identity_digest": (
            prompt_saliency_runtime.model_identity_digest
        ),
        "attention_module_names": list(config.attention_module_names),
    }
    if "semantic_feature_operator_contract" in json.dumps(
        runtime_versions,
        ensure_ascii=False,
        sort_keys=True,
    ):
        raise RuntimeError("content smoke must not contain the legacy semantic operator")
    return _ContentRuntimeSmokeComponents(
        pipeline=pipeline,
        prompt_saliency_runtime=prompt_saliency_runtime,
        attention_modules=attention_modules,
        unconditional_prompt=unconditional_prompt,
        unconditional_pooled=unconditional_pooled,
        runtime_versions=runtime_versions,
    )


def _decode_content_runtime_latent(pipeline: Any, latent: Any) -> Any:
    """Decode one SD3.5 scheduler latent to an RGB [0,1] Tensor."""

    import torch

    vae_dtype = next(pipeline.vae.parameters()).dtype
    scaled = latent.to(dtype=vae_dtype) / pipeline.vae.config.scaling_factor
    scaled = scaled + pipeline.vae.config.shift_factor
    decoded = pipeline.vae.decode(scaled, return_dict=False)[0]
    image = pipeline.image_processor.postprocess(decoded, output_type="pt")
    if not isinstance(image, torch.Tensor):
        raise RuntimeError("SD3.5 VAE postprocess must return an RGB Tensor")
    return image.to(device=latent.device, dtype=torch.float32)


def _content_runtime_prompt_embeddings(
    pipeline: Any,
    prompt: str,
) -> tuple[Any, Any]:
    """Build the actual Prompt condition used by the Q/K geometry forward."""

    prompt_embeds, _, pooled_prompt_embeds, _ = pipeline.encode_prompt(
        prompt=prompt,
        prompt_2=prompt,
        prompt_3=prompt,
        device=pipeline._execution_device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=False,
    )
    return prompt_embeds, pooled_prompt_embeds


def _run_content_runtime_generation(
    config: SemanticWatermarkRuntimeConfig,
    references: ContentRoutingReferenceScalars,
    *,
    components: _ContentRuntimeSmokeComponents,
    include_clean: bool,
) -> tuple[Any | None, Any, dict[str, Any]]:
    """以既有正式组件运行 clean 与索引10单写回生成。"""

    _require_full_content_runtime_config(config)
    import torch

    if type(config) is not SemanticWatermarkRuntimeConfig:
        raise TypeError("config must be an exact SemanticWatermarkRuntimeConfig")
    if type(references) is not ContentRoutingReferenceScalars:
        raise TypeError("references must be exact ContentRoutingReferenceScalars")
    if type(config.key_material) is not str or not config.key_material:
        raise ValueError("content smoke requires explicit non-empty key material")
    pipeline = components.pipeline
    base_latent_shape = (
        1,
        int(pipeline.transformer.config.in_channels),
        int(config.height) // int(pipeline.vae_scale_factor),
        int(config.width) // int(pipeline.vae_scale_factor),
    )
    base_latent, base_identity = build_canonical_sd35_base_latent(
        shape=base_latent_shape,
        generation_seed_random=int(config.seed),
        model_id=config.model_id,
        model_revision=config.model_revision,
        device=pipeline._execution_device,
        dtype=pipeline.transformer.dtype,
    )
    prompt_embeds, pooled_prompt_embeds = _content_runtime_prompt_embeddings(
        pipeline,
        config.prompt,
    )
    model_identity_digest = build_stable_digest(
        {"model_id": config.model_id, "model_revision": config.model_revision}
    )
    public_probe_identity = build_public_probe_identity(config.model_revision)
    captured_z9: Any | None = None
    captured_z9_count = 0
    callback_count = 0
    current_image_decode_count = 0
    public_probe_additional_decode_count = 0
    actual_dtype_single_write_count = 0
    diagnostic: dict[str, Any] = {}

    def callback(
        pipe: Any,
        step_index: int,
        timestep: Any,
        callback_kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        nonlocal captured_z9, captured_z9_count, callback_count
        nonlocal current_image_decode_count, public_probe_additional_decode_count
        nonlocal actual_dtype_single_write_count, diagnostic
        latent = callback_kwargs.get("latents")
        if latent is None:
            raise RuntimeError("content smoke callback requires latents")
        if step_index == 9:
            captured_z9_count += 1
            if captured_z9_count != 1:
                raise RuntimeError("content smoke must capture index 9 exactly once")
            captured_z9 = latent.detach().clone()
            return callback_kwargs
        if step_index != 10:
            return callback_kwargs
        if captured_z9 is None:
            raise RuntimeError("index-10 content smoke requires captured z9")
        if captured_z9_count != 1:
            raise RuntimeError("index-10 content smoke requires one index-9 capture")
        callback_count += 1
        if callback_count != 1:
            raise RuntimeError("index-10 content smoke callback must execute once")
        z10 = latent
        current_image_decode_count += 1
        if current_image_decode_count != 1:
            raise RuntimeError("content smoke must decode current x10 exactly once")
        decoded_x10 = _decode_content_runtime_latent(pipe, z10)

        def vae_decoder(candidate: Any) -> Any:
            nonlocal public_probe_additional_decode_count
            public_probe_additional_decode_count += 1
            if public_probe_additional_decode_count != 1:
                raise RuntimeError("public probe must perform one additional VAE decode")
            return _decode_content_runtime_latent(pipe, candidate)

        observations = build_content_observation_routing(
            previous_scheduler_latent=captured_z9,
            current_scheduler_latent=z10,
            decoded_current_image=decoded_x10,
            prompt=config.prompt,
            saliency_runtime=components.prompt_saliency_runtime,
            vae_decoder=vae_decoder,
            public_probe_identity=public_probe_identity,
            reference_gradient=references.reference_gradient,
            reference_response=references.reference_response,
            reference_sensitivity=references.reference_sensitivity,
        )
        if public_probe_additional_decode_count != 1:
            raise RuntimeError("public probe must perform one additional VAE decode")
        lf_template = build_formal_low_frequency_template(
            z10,
            config.key_material,
            model_identity_digest,
            prg_version=KEYED_PRG_VERSION,
        )
        hf_template = build_high_frequency_tail_template(
            z10,
            config.key_material,
            model_identity_digest,
            prg_version=KEYED_PRG_VERSION,
        )
        content_update = build_content_carrier_update(
            current_scheduler_latent=z10,
            routing=observations.routing,
            lf_template=lf_template,
            hf_tail_template=hf_template,
            method_role="full_dual_chain",
        )
        transformer_forward = _transformer_forward_function(
            pipe,
            timestep,
            prompt_embeds,
            pooled_prompt_embeds,
        )
        recorder = DifferentiableAttentionRecorder(
            components.attention_modules,
            max_tokens=config.max_attention_tokens,
        )
        try:
            geometry, geometry_evidence = (
                _build_attention_geometry_sync_update_with_evidence(
                    current_scheduler_latent=z10,
                    content_update=content_update,
                    transformer_forward=transformer_forward,
                    recorder=recorder,
                    key_material=config.key_material,
                    prg_version=KEYED_PRG_VERSION,
                )
            )
            write_budget = formal_dual_chain_write_budget()
            write_result = compose_dual_chain_update_once(
                z10,
                content_update.lf_update,
                content_update.hf_tail_update,
                geometry.geometry_update,
                write_budget,
                method_role=content_update.method_role,
            )
            same_gamma_content = (
                z10.detach().float()
                + content_update.lf_update * write_result.accepted_common_scale
            )
            same_gamma_content = (
                same_gamma_content
                + content_update.hf_tail_update
                * write_result.accepted_common_scale
            ).to(dtype=z10.dtype)
            content_score, content_qk_digest = (
                _evaluate_post_write_geometry_relation(
                    written_latent=same_gamma_content,
                    transformer_forward=transformer_forward,
                    recorder=recorder,
                    key_material=config.key_material,
                    runtime_evidence=geometry_evidence,
                )
            )
            final_score, final_qk_digest = _evaluate_post_write_geometry_relation(
                written_latent=write_result.written_latent,
                transformer_forward=transformer_forward,
                recorder=recorder,
                key_material=config.key_material,
                runtime_evidence=geometry_evidence,
            )
        finally:
            recorder.close()
        if not final_score > content_score:
            raise RuntimeError("post-write Q/K gate did not strictly improve")
        z10_float32 = z10.detach().float()
        latent_l2 = torch.linalg.vector_norm(z10_float32.reshape(-1))
        combined_limit = latent_l2 * z10_float32.new_tensor(
            write_budget.combined_relative_l2_limit
        )
        if not all(
            value > 0.0
            for value in (
                write_result.lf_effective_l2,
                write_result.hf_tail_effective_l2,
                write_result.geometry_effective_l2,
                write_result.combined_effective_l2,
            )
        ):
            raise RuntimeError("full_dual_chain smoke requires three effective branches")
        combined_ready = (
            math.isfinite(write_result.combined_effective_l2)
            and bool(
                z10_float32.new_tensor(write_result.combined_effective_l2)
                <= combined_limit
            )
        )
        if not combined_ready:
            raise RuntimeError("content smoke combined actual write exceeds its budget")
        callback_kwargs["latents"] = write_result.written_latent
        actual_dtype_single_write_count += 1
        if actual_dtype_single_write_count != 1:
            raise RuntimeError("content smoke must write actual dtype exactly once")
        diagnostic = {
            "method_role": content_update.method_role,
            "callback_write_index": 10,
            "callback_write_count": callback_count,
            "captured_previous_index": 9,
            "captured_previous_count": captured_z9_count,
            "current_image_decode_count": current_image_decode_count,
            "public_probe_additional_decode_count": (
                public_probe_additional_decode_count
            ),
            "actual_dtype_single_write_count": actual_dtype_single_write_count,
            "common_gamma": write_result.accepted_common_scale,
            "lf_effective_l2": write_result.lf_effective_l2,
            "hf_tail_effective_l2": write_result.hf_tail_effective_l2,
            "geometry_effective_l2": write_result.geometry_effective_l2,
            "combined_effective_l2": write_result.combined_effective_l2,
            "combined_effective_l2_limit": float(combined_limit.item()),
            "combined_effective_l2_ready": combined_ready,
            "actual_dtype_single_write_digest": (
                write_result.actual_dtype_write_digest
            ),
            "content_only_postwrite_qk_score": content_score,
            "final_postwrite_qk_score": final_score,
            "post_write_qk_strict_ready": True,
            "content_only_postwrite_qk_digest": content_qk_digest,
            "final_postwrite_qk_digest": final_qk_digest,
            "routing_identity_digest": observations.routing.routing_identity_digest,
            "geometry_qk_atomic_records_digest": (
                geometry.qk_atomic_records_digest
            ),
            "geometry_update_digest": geometry.geometry_update_digest,
        }
        return callback_kwargs

    common_kwargs = {
        "prompt": config.prompt,
        "negative_prompt": config.negative_prompt,
        "width": config.width,
        "height": config.height,
        "num_inference_steps": config.inference_steps,
        "guidance_scale": config.guidance_scale,
        "output_type": "pil",
    }
    clean_image = None
    if include_clean:
        with torch.no_grad():
            clean_image = pipeline(
                latents=base_latent.detach().clone(),
                **common_kwargs,
            ).images[0]
    output = pipeline(
        latents=base_latent,
        callback_on_step_end=callback,
        callback_on_step_end_tensor_inputs=["latents"],
        **common_kwargs,
    )
    if (
        captured_z9_count != 1
        or callback_count != 1
        or current_image_decode_count != 1
        or public_probe_additional_decode_count != 1
        or actual_dtype_single_write_count != 1
        or not diagnostic
    ):
        raise RuntimeError("content smoke did not execute the unique index-10 write")
    return clean_image, output.images[0], {
        **diagnostic,
        "base_latent_content_digest_random": base_identity[
            "base_latent_content_digest_random"
        ],
        "base_latent_identity_digest_random": base_identity[
            "base_latent_identity_digest_random"
        ],
        "runtime_versions": components.runtime_versions,
        "prompt_saliency_model_identity_digest": (
            components.prompt_saliency_runtime.model_identity_digest
        ),
        "legacy_semantic_feature_operator_present": False,
    }


def run_content_runtime_smoke(
    config: SemanticWatermarkRuntimeConfig,
    references: ContentRoutingReferenceScalars,
    *,
    verified_formal_execution_lock: Mapping[str, Any],
    repository_root: str | Path,
) -> tuple[Any, dict[str, Any]]:
    """运行一个真实索引10内容/QK单写回样本。"""

    _require_full_content_runtime_config(config)
    components = _load_content_runtime_smoke_components(
        config,
        verified_formal_execution_lock=verified_formal_execution_lock,
        repository_root=repository_root,
    )
    _, watermarked_image, diagnostic = _run_content_runtime_generation(
        config,
        references,
        components=components,
        include_clean=False,
    )
    return watermarked_image, diagnostic


def _content_runtime_formal_randomization_reference(
    config: SemanticWatermarkRuntimeConfig,
    diagnostic: Mapping[str, Any],
) -> dict[str, Any]:
    """从新单写回链实际使用的seed、key和base latent重建样本随机身份。"""

    identity = {
        "randomization_repeat_id": config.randomization_repeat_id,
        "generation_seed_index": int(config.generation_seed_index),
        "generation_seed_offset": int(config.generation_seed_offset),
        "watermark_key_index": int(config.watermark_key_index),
        "generation_seed_random": int(config.seed),
        "watermark_key_seed_random": int(config.watermark_key_seed_random),
        "formal_randomization_protocol_digest": (
            config.formal_randomization_protocol_digest
        ),
        "watermark_key_material_digest_random": build_stable_digest(
            {"key_material": config.key_material}
        ),
    }
    identity["formal_randomization_identity_digest_random"] = (
        build_stable_digest(identity)
    )
    return formal_randomization_sample_reference(
        identity,
        base_latent_identity={
            "base_latent_content_digest_random": diagnostic[
                "base_latent_content_digest_random"
            ],
            "base_latent_identity_digest_random": diagnostic[
                "base_latent_identity_digest_random"
            ],
        },
    )


def _stable_json(value: Any) -> str:
    """生成稳定 JSON 文本。"""

    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"


def _is_sha256_hex(value: Any) -> bool:
    """判断字段是否为规范的小写 SHA-256 十六进制文本。"""

    text = str(value)
    return len(text) == 64 and all(
        character in "0123456789abcdef" for character in text
    )


def semantic_watermark_runtime_config_payload(
    config: SemanticWatermarkRuntimeConfig,
) -> dict[str, Any]:
    """返回隐藏密钥原文但保留精确科学身份的运行配置."""

    payload = asdict(config)
    payload.pop("key_material")
    payload["key_material_digest_random"] = build_stable_digest(
        {"key_material": config.key_material}
    )
    payload["injection_step_indices"] = list(config.injection_step_indices)
    payload["attention_injection_steps"] = list(
        config.attention_injection_steps
    )
    payload["attention_module_names"] = list(config.attention_module_names)
    payload["attention_relation_component_weights"] = list(
        config.attention_relation_component_weights
    )
    payload["standard_attack_profiles"] = list(config.standard_attack_profiles)
    payload["quantized_branch_composition_order"] = list(
        config.quantized_branch_composition_order
    )
    payload["lf_carrier_protocol_digest"] = (
        config.low_frequency_carrier_config.protocol_digest
    )
    payload["tail_carrier_protocol_digest"] = (
        HIGH_FREQUENCY_TAIL_PROTOCOL_DIGEST
    )
    payload["method_definition"] = semantic_conditioned_latent_method_definition()
    payload["method_definition_digest"] = (
        semantic_conditioned_latent_method_definition_digest()
    )
    return payload


def semantic_watermark_runtime_config_digest(
    config: SemanticWatermarkRuntimeConfig,
) -> str:
    """计算单个 Prompt 或消融完成单元的配置摘要."""

    return build_stable_digest(semantic_watermark_runtime_config_payload(config))


def build_semantic_watermark_run_id(config: SemanticWatermarkRuntimeConfig) -> str:
    """根据完整运行配置生成稳定标识."""

    return f"semantic_watermark_{semantic_watermark_runtime_config_digest(config)[:16]}"


def validate_semantic_watermark_runtime_result_provenance(
    result_payload: Mapping[str, Any],
    *,
    expected_config: SemanticWatermarkRuntimeConfig | None = None,
    unit_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """用顶层 manifest 配置把结果、run id 与科学来源精确绑定."""

    payload = dict(result_payload)
    run_id = str(payload.get("run_id", ""))
    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping):
        raise TypeError("语义水印完成结果缺少 metadata")
    if expected_config is not None:
        resolved_unit_config = semantic_watermark_runtime_config_payload(
            expected_config
        )
        if unit_config is not None and dict(unit_config) != resolved_unit_config:
            raise ValueError("顶层 manifest 配置与当前请求不一致")
    elif isinstance(unit_config, Mapping):
        resolved_unit_config = dict(unit_config)
    else:
        raise TypeError("结果复验必须提供顶层 manifest 逐单元配置")
    current_method_definition = semantic_conditioned_latent_method_definition()
    current_method_definition_digest = (
        semantic_conditioned_latent_method_definition_digest()
    )
    if (
        resolved_unit_config.get("method_definition")
        != current_method_definition
        or resolved_unit_config.get("method_definition_digest")
        != current_method_definition_digest
    ):
        raise ValueError("语义水印逐单元配置未绑定当前方法定义")
    config_digest = build_stable_digest(resolved_unit_config)
    if metadata.get("scientific_unit_config_digest") != config_digest:
        raise ValueError("语义水印结果未引用顶层逐单元配置摘要")
    if run_id != f"semantic_watermark_{config_digest[:16]}":
        raise ValueError("语义水印 run id 与逐单元配置摘要不一致")
    provenance = metadata.get("scientific_unit_provenance")
    if not isinstance(provenance, Mapping):
        raise TypeError("语义水印完成结果缺少科学运行来源记录")
    return validate_scientific_unit_provenance(
        provenance,
        expected_unit_id=run_id,
        expected_config_digest=config_digest,
    )


def _resolve_repository_output_path(
    root_path: Path,
    relative_path: str,
) -> Path | None:
    """把产物相对路径限制在仓库根目录内。"""

    resolved = (root_path / relative_path).resolve()
    if resolved != root_path and root_path not in resolved.parents:
        return None
    return resolved


def _read_jsonl_object_records(path: Path) -> list[dict[str, Any]]:
    """读取 JSONL 对象记录, 供全部持久化证据重建复用。"""

    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise TypeError("持久化原子必须是 JSON 对象")
        records.append(payload)
    return records


def _persisted_image_content_record(
    path: Path,
    root_path: Path,
) -> dict[str, Any]:
    """从已写出的图像重建文件字节与规范 RGB 像素双重身份。"""

    return {
        "image_path": path.relative_to(root_path).as_posix(),
        "image_file_sha256": file_digest(path),
        **read_canonical_rgb_uint8_content_record(path),
    }


def _bind_final_image_qk_to_pixels(
    observability: Mapping[str, Any],
    final_image_records: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """把最终三图的规范像素身份逐角色绑定到已测得的 Q/K 原子。"""

    resolved = dict(observability)
    qk_records = resolved.get("final_image_qk_atomic_content_records")
    if not isinstance(qk_records, list) or len(qk_records) != 3:
        raise RuntimeError("最终图像 Q/K 记录必须覆盖冻结三角色")
    role_pairs = (
        ("final_clean_image", "clean_image"),
        ("final_carrier_only_image", "carrier_only_image"),
        ("final_watermarked_image", "watermarked_image"),
    )
    bindings = []
    for qk_record, (qk_role, image_role) in zip(qk_records, role_pairs):
        if qk_record.get("qk_evaluation_role") != qk_role:
            raise RuntimeError("最终图像 Q/K 角色顺序与冻结协议不一致")
        image_record = final_image_records[image_role]
        bindings.append(
            {
                "qk_evaluation_role": qk_role,
                "evaluation_image_rgb_uint8_content_sha256": (
                    image_record["image_rgb_uint8_content_sha256"]
                ),
                "qk_atomic_content_digest": qk_record[
                    "qk_atomic_content_digest"
                ],
                "public_detection_noise_content_sha256": qk_record[
                    "public_detection_noise_content_sha256"
                ],
                "public_detection_noise_prg_identity_digest": qk_record[
                    "public_detection_noise_prg_identity_digest"
                ],
                "public_detection_noise_evaluation_index": qk_record[
                    "public_detection_noise_evaluation_index"
                ],
            }
        )
    resolved["final_image_qk_image_content_bindings"] = bindings
    resolved["final_image_qk_image_content_binding_digest"] = (
        build_stable_digest(
            {"final_image_qk_image_content_bindings": bindings}
        )
    )
    return resolved


def _bind_detection_qk_to_pixels(
    record: Mapping[str, Any],
    evaluated_image: Any,
) -> dict[str, Any]:
    """把 raw/aligned 实际评价图像逐次绑定到检测 Q/K 与公开噪声。"""

    resolved = dict(record)
    metadata_value = resolved.get("metadata")
    if not isinstance(metadata_value, Mapping):
        raise RuntimeError("检测记录缺少 Q/K metadata")
    metadata = dict(metadata_value)
    qk_records = metadata.get("detection_qk_atomic_content_records")
    if not isinstance(qk_records, list) or not 1 <= len(qk_records) <= 2:
        raise RuntimeError("检测 Q/K 记录必须包含 raw 或 raw/aligned 评价")
    evaluated_content = canonical_rgb_uint8_content_record(evaluated_image)
    image_contents = {
        "raw_detection_image": evaluated_content[
            "image_rgb_uint8_content_sha256"
        ]
    }
    if len(qk_records) == 2:
        alignment = resolved.get("alignment")
        if not isinstance(alignment, Mapping):
            raise RuntimeError("aligned 检测 Q/K 缺少仿射恢复记录")
        aligned_image = _align_image(
            evaluated_image,
            SimpleNamespace(
                affine_transform=alignment.get("affine_transform")
            ),
        )
        image_contents["aligned_detection_image"] = (
            canonical_rgb_uint8_content_record(aligned_image)[
                "image_rgb_uint8_content_sha256"
            ]
        )
    expected_roles = (
        ("raw_detection_image",)
        if len(qk_records) == 1
        else ("raw_detection_image", "aligned_detection_image")
    )
    if tuple(
        str(item.get("qk_evaluation_role", ""))
        for item in qk_records
    ) != expected_roles:
        raise RuntimeError("检测 Q/K 记录没有保留冻结 raw/aligned 顺序")
    bindings = []
    for qk_record in qk_records:
        role = str(qk_record.get("qk_evaluation_role", ""))
        if role not in image_contents:
            raise RuntimeError("检测 Q/K 记录包含未定义的图像评价角色")
        bindings.append(
            {
                "qk_evaluation_role": role,
                "evaluation_image_rgb_uint8_content_sha256": (
                    image_contents[role]
                ),
                "qk_atomic_content_digest": qk_record[
                    "qk_atomic_content_digest"
                ],
                "public_detection_noise_evaluation_index": qk_record[
                    "public_detection_noise_evaluation_index"
                ],
            }
        )
    metadata["detection_qk_image_content_bindings"] = bindings
    metadata["detection_qk_image_content_binding_digest"] = (
        build_stable_digest(
            {"detection_qk_image_content_bindings": bindings}
        )
    )
    resolved["metadata"] = metadata
    return resolved


def _carrier_only_counterfactual_artifact_binding_ready(
    result_payload: Mapping[str, Any],
    manifest: Mapping[str, Any],
    root_path: Path,
    config: SemanticWatermarkRuntimeConfig | Mapping[str, Any],
) -> bool:
    """复验反事实原子、图像、保持记录、Q/K 记录与 manifest 绑定。"""

    attention_geometry_enabled = (
        config.attention_geometry_enabled
        if isinstance(config, SemanticWatermarkRuntimeConfig)
        else config.get("attention_geometry_enabled") is True
    )
    if not attention_geometry_enabled:
        return True
    metadata = result_payload.get("metadata")
    manifest_metadata = manifest.get("metadata")
    if not isinstance(metadata, Mapping) or not isinstance(
        manifest_metadata,
        Mapping,
    ):
        return False
    observability = metadata.get("final_image_attention_observability")
    preservation = metadata.get("carrier_only_final_image_preservation")
    final_preservation = metadata.get("final_image_preservation")
    counterfactual = metadata.get("carrier_only_counterfactual")
    if not isinstance(observability, Mapping) or not isinstance(
        preservation,
        Mapping,
    ) or not isinstance(final_preservation, Mapping) or not isinstance(
        counterfactual,
        Mapping,
    ):
        return False
    identity_digest = str(
        counterfactual.get("carrier_only_counterfactual_identity_digest", "")
    )
    image_path = str(
        counterfactual.get("carrier_only_counterfactual_image_path", "")
    )
    image_digest = str(
        counterfactual.get("carrier_only_counterfactual_image_digest", "")
    )
    atom_path = str(
        counterfactual.get("carrier_only_counterfactual_atom_path", "")
    )
    atom_file_sha256 = str(
        counterfactual.get(
            "carrier_only_counterfactual_atom_file_sha256",
            "",
        )
    )
    atom_content_digest = str(
        counterfactual.get(
            "carrier_only_counterfactual_atom_content_digest",
            "",
        )
    )
    output_paths = tuple(str(path) for path in manifest.get("output_paths", ()))
    if not (
        _is_sha256_hex(identity_digest)
        and _is_sha256_hex(image_digest)
        and _is_sha256_hex(atom_file_sha256)
        and _is_sha256_hex(atom_content_digest)
        and image_path
        and atom_path
        and final_preservation.get("final_image_preservation_gate_ready") is True
        and preservation.get(
            "carrier_only_final_image_preservation_gate_ready"
        )
        is True
        and preservation.get(
            "carrier_only_to_full_final_image_preservation_gate_ready"
        )
        is True
        and preservation.get(
            "carrier_only_counterfactual_three_way_preservation_gate_ready"
        )
        is True
        and all(
            record.get("carrier_only_counterfactual_identity_digest")
            == identity_digest
            and record.get("carrier_only_counterfactual_image_path")
            == image_path
            and record.get("carrier_only_counterfactual_image_digest")
            == image_digest
            and record.get("carrier_only_counterfactual_atom_path")
            == atom_path
            and record.get("carrier_only_counterfactual_atom_file_sha256")
            == atom_file_sha256
            and record.get("carrier_only_counterfactual_atom_content_digest")
            == atom_content_digest
            for record in (observability, preservation)
        )
        and manifest_metadata.get("carrier_only_counterfactual_identity_digest")
        == identity_digest
        and manifest_metadata.get("carrier_only_counterfactual_image_digest")
        == image_digest
        and manifest_metadata.get("carrier_only_counterfactual_atom_path")
        == atom_path
        and manifest_metadata.get(
            "carrier_only_counterfactual_atom_file_sha256"
        )
        == atom_file_sha256
        and manifest_metadata.get(
            "carrier_only_counterfactual_atom_content_digest"
        )
        == atom_content_digest
        and image_path in output_paths
        and atom_path in output_paths
    ):
        return False

    resolved_image_path = _resolve_repository_output_path(
        root_path,
        image_path,
    )
    resolved_atom_path = _resolve_repository_output_path(
        root_path,
        atom_path,
    )
    full_record_path_text = str(result_payload.get("update_record_path", ""))
    resolved_full_record_path = _resolve_repository_output_path(
        root_path,
        full_record_path_text,
    )
    if (
        resolved_image_path is None
        or resolved_atom_path is None
        or resolved_full_record_path is None
        or full_record_path_text not in output_paths
    ):
        return False
    if not (
        resolved_image_path.is_file()
        and resolved_atom_path.is_file()
        and resolved_full_record_path.is_file()
        and file_digest(resolved_image_path) == image_digest
        and file_digest(resolved_atom_path) == atom_file_sha256
    ):
        return False

    try:
        full_records = _read_jsonl_object_records(
            resolved_full_record_path
        )
        carrier_records = _read_jsonl_object_records(resolved_atom_path)
        carrier_config = (
            replace(config, attention_geometry_enabled=False)
            if isinstance(config, SemanticWatermarkRuntimeConfig)
            else {
                **dict(config),
                "attention_geometry_enabled": False,
            }
        )
        rebuilt_identity = _carrier_only_counterfactual_identity(
            config,
            carrier_config,
            full_records,
            carrier_records,
        )
    except (OSError, TypeError, ValueError, RuntimeError, json.JSONDecodeError):
        return False
    return bool(
        build_stable_digest(carrier_records) == atom_content_digest
        and rebuilt_identity["carrier_only_counterfactual_identity_digest"]
        == identity_digest
        and rebuilt_identity[
            "full_method_counterfactual_update_records_digest"
        ]
        == counterfactual.get(
            "full_method_counterfactual_update_records_digest"
        )
        and rebuilt_identity[
            "carrier_only_counterfactual_update_records_digest"
        ]
        == counterfactual.get(
            "carrier_only_counterfactual_update_records_digest"
        )
        and rebuilt_identity[
            "full_method_initial_latent_content_sha256"
        ]
        == counterfactual.get("full_method_initial_latent_content_sha256")
        and rebuilt_identity[
            "carrier_only_initial_latent_content_sha256"
        ]
        == counterfactual.get("carrier_only_initial_latent_content_sha256")
    )


def _scientific_content_binding_validation_parameters(
    config: SemanticWatermarkRuntimeConfig | Mapping[str, Any],
) -> dict[str, Any]:
    """解析运行配置或持久化脱敏配置, 供同一内容消费者复用。"""

    if isinstance(config, SemanticWatermarkRuntimeConfig):
        attention_geometry_enabled = config.attention_geometry_enabled
        return {
            "attention_geometry_enabled": attention_geometry_enabled,
            "image_alignment_enabled": config.image_alignment_enabled,
            "semantic_routing_enabled": config.semantic_routing_enabled,
            "null_space_enabled": config.null_space_enabled,
            "full_active_branches": list(
                _active_carrier_branch_names(config)
            ),
            "carrier_only_active_branches": list(
                _active_carrier_branch_names(
                    replace(config, attention_geometry_enabled=False)
                )
                if attention_geometry_enabled
                else ()
            ),
            "scientific_unit_config_digest": (
                semantic_watermark_runtime_config_digest(config)
            ),
            "expected_steps": list(config.injection_step_indices),
        }
    if not isinstance(config, Mapping):
        raise TypeError("科学内容复验配置必须为运行配置或持久化配置对象")
    payload = dict(config)
    if (
        payload.get("method_definition_digest")
        != semantic_conditioned_latent_method_definition_digest()
        or payload.get("method_definition")
        != semantic_conditioned_latent_method_definition()
    ):
        raise ValueError("持久化配置的方法定义与当前核心实现不一致")
    bool_fields = (
        "lf_enabled",
        "tail_robust_enabled",
        "attention_geometry_enabled",
        "image_alignment_enabled",
        "semantic_routing_enabled",
        "null_space_enabled",
    )
    if any(not isinstance(payload.get(field_name), bool) for field_name in bool_fields):
        raise ValueError("持久化配置的分支与 Null Space 开关必须为布尔值")
    active_branches = [
        branch_name
        for branch_name, field_name in (
            ("lf_content", "lf_enabled"),
            ("tail_robust", "tail_robust_enabled"),
            ("attention_geometry", "attention_geometry_enabled"),
        )
        if payload[field_name]
    ]
    expected_steps = payload.get("injection_step_indices")
    if (
        not active_branches
        or not isinstance(expected_steps, list)
        or not expected_steps
        or any(
            isinstance(step, bool) or not isinstance(step, int)
            for step in expected_steps
        )
        or len(expected_steps) != len(set(expected_steps))
    ):
        raise ValueError("持久化配置缺少有效活动分支或注入步骤")
    attention_geometry_enabled = payload["attention_geometry_enabled"]
    return {
        "attention_geometry_enabled": attention_geometry_enabled,
        "image_alignment_enabled": payload["image_alignment_enabled"],
        "semantic_routing_enabled": payload["semantic_routing_enabled"],
        "null_space_enabled": payload["null_space_enabled"],
        "full_active_branches": active_branches,
        "carrier_only_active_branches": (
            [
                branch_name
                for branch_name in ("lf_content", "tail_robust")
                if branch_name in active_branches
            ]
            if attention_geometry_enabled
            else []
        ),
        "scientific_unit_config_digest": build_stable_digest(payload),
        "expected_steps": list(expected_steps),
    }


def _scientific_content_binding_artifact_ready(
    result_payload: Mapping[str, Any],
    manifest: Mapping[str, Any],
    root_path: Path,
    config: SemanticWatermarkRuntimeConfig | Mapping[str, Any],
) -> bool:
    """从持久化记录和图像文件重建总科学内容证据。"""

    metadata = result_payload.get("metadata")
    manifest_metadata = manifest.get("metadata")
    if not isinstance(metadata, Mapping) or not isinstance(
        manifest_metadata,
        Mapping,
    ):
        return False
    binding_record = metadata.get("scientific_content_binding_record")
    if not isinstance(binding_record, Mapping):
        return False
    binding_record = dict(binding_record)
    supplied_digest = metadata.get("scientific_content_binding_digest")
    if (
        metadata.get("scientific_content_binding_schema")
        != SCIENTIFIC_CONTENT_BINDING_SCHEMA
        or supplied_digest
        != binding_record.get("scientific_content_binding_digest")
        or manifest_metadata.get("scientific_content_binding_schema")
        != SCIENTIFIC_CONTENT_BINDING_SCHEMA
        or manifest_metadata.get("scientific_content_binding_digest")
        != supplied_digest
    ):
        return False
    try:
        if recompute_scientific_content_binding_digest(
            binding_record
        ) != supplied_digest:
            return False
    except (TypeError, ValueError):
        return False
    try:
        validation_parameters = (
            _scientific_content_binding_validation_parameters(config)
        )
    except (TypeError, ValueError):
        return False
    attention_geometry_enabled = validation_parameters[
        "attention_geometry_enabled"
    ]
    image_alignment_enabled = validation_parameters[
        "image_alignment_enabled"
    ]
    semantic_routing_enabled = validation_parameters[
        "semantic_routing_enabled"
    ]
    null_space_enabled = validation_parameters["null_space_enabled"]
    full_active_branches = validation_parameters["full_active_branches"]
    carrier_only_active_branches = validation_parameters[
        "carrier_only_active_branches"
    ]
    scientific_unit_config_digest = validation_parameters[
        "scientific_unit_config_digest"
    ]
    expected_steps = validation_parameters["expected_steps"]
    output_paths = tuple(
        str(path) for path in manifest.get("output_paths", ())
    )
    path_fields = {
        "full_update": str(result_payload.get("update_record_path", "")),
        "detection": str(result_payload.get("detection_record_path", "")),
        "clean_image": str(result_payload.get("clean_image_path", "")),
        "watermarked_image": str(
            result_payload.get("watermarked_image_path", "")
        ),
    }
    counterfactual = metadata.get("carrier_only_counterfactual")
    if attention_geometry_enabled:
        if not isinstance(counterfactual, Mapping):
            return False
        path_fields["carrier_update"] = str(
            counterfactual.get("carrier_only_counterfactual_atom_path", "")
        )
        path_fields["carrier_only_image"] = str(
            counterfactual.get(
                "carrier_only_counterfactual_image_path",
                "",
            )
        )
    if any(
        not relative_path or relative_path not in output_paths
        for relative_path in path_fields.values()
    ):
        return False
    resolved_paths = {
        role: _resolve_repository_output_path(root_path, relative_path)
        for role, relative_path in path_fields.items()
    }
    if any(
        path is None or not path.is_file()
        for path in resolved_paths.values()
    ):
        return False
    try:
        full_records = _read_jsonl_object_records(
            resolved_paths["full_update"]
        )
        detection_records = _read_jsonl_object_records(
            resolved_paths["detection"]
        )
        carrier_records = (
            _read_jsonl_object_records(resolved_paths["carrier_update"])
            if attention_geometry_enabled
            else []
        )
        if [record.get("step_index") for record in full_records] != (
            expected_steps
        ):
            return False
        if attention_geometry_enabled and [
            record.get("step_index") for record in carrier_records
        ] != expected_steps:
            return False
        for detection_record in detection_records:
            for path_field, digest_field, pixel_field in (
                (
                    "source_image_path",
                    "source_image_digest",
                    "source_image_rgb_uint8_content_sha256",
                ),
                (
                    "evaluated_image_path",
                    "evaluated_image_digest",
                    "evaluated_image_rgb_uint8_content_sha256",
                ),
            ):
                relative_path = str(
                    detection_record.get(path_field, "")
                )
                resolved_image_path = _resolve_repository_output_path(
                    root_path,
                    relative_path,
                )
                persisted_pixel_identity = (
                    None
                    if resolved_image_path is None
                    or not resolved_image_path.is_file()
                    else read_canonical_rgb_uint8_content_record(
                        resolved_image_path
                    )
                )
                dimension_prefix = path_field.removesuffix("_path")
                if (
                    not relative_path
                    or relative_path not in output_paths
                    or resolved_image_path is None
                    or not resolved_image_path.is_file()
                    or persisted_pixel_identity is None
                    or file_digest(resolved_image_path)
                    != detection_record.get(digest_field)
                    or persisted_pixel_identity[
                        "image_rgb_uint8_content_sha256"
                    ]
                    != detection_record.get(pixel_field)
                    or persisted_pixel_identity["image_width"]
                    != detection_record.get(
                        f"{dimension_prefix}_width"
                    )
                    or persisted_pixel_identity["image_height"]
                    != detection_record.get(
                        f"{dimension_prefix}_height"
                    )
                ):
                    return False
            if attention_geometry_enabled:
                detection_metadata = detection_record.get("metadata")
                if not isinstance(detection_metadata, Mapping):
                    return False
                qk_bindings = detection_metadata.get(
                    "detection_qk_image_content_bindings"
                )
                if not isinstance(qk_bindings, list) or not qk_bindings:
                    return False
                evaluated_image_path = _resolve_repository_output_path(
                    root_path,
                    str(detection_record.get("evaluated_image_path", "")),
                )
                if evaluated_image_path is None:
                    return False
                from PIL import Image

                with Image.open(evaluated_image_path) as evaluated_image:
                    if len(qk_bindings) == 2:
                        alignment = detection_record.get("alignment")
                        if not isinstance(alignment, Mapping):
                            return False
                        aligned_image = _align_image(
                            evaluated_image,
                            SimpleNamespace(
                                affine_transform=alignment.get(
                                    "affine_transform"
                                )
                            ),
                        )
                        aligned_digest = (
                            canonical_rgb_uint8_content_record(
                                aligned_image
                            )["image_rgb_uint8_content_sha256"]
                        )
                        if qk_bindings[1].get(
                            "evaluation_image_rgb_uint8_content_sha256"
                        ) != aligned_digest:
                            return False
        final_image_records = {
            "clean_image": _persisted_image_content_record(
                resolved_paths["clean_image"],
                root_path,
            ),
            **(
                {
                    "carrier_only_image": (
                        _persisted_image_content_record(
                            resolved_paths["carrier_only_image"],
                            root_path,
                        )
                    )
                }
                if attention_geometry_enabled
                else {}
            ),
            "watermarked_image": _persisted_image_content_record(
                resolved_paths["watermarked_image"],
                root_path,
            ),
        }
        rebuilt_record = build_scientific_content_binding_record(
            run_id=str(result_payload.get("run_id", "")),
            method_definition_digest=(
                semantic_conditioned_latent_method_definition_digest()
            ),
            scientific_unit_config_digest=scientific_unit_config_digest,
            full_update_records=full_records,
            carrier_only_update_records=carrier_records,
            detection_records=detection_records,
            detection_key_plan=manifest_metadata.get(
                "detection_key_plan",
                {},
            ),
            final_image_records=final_image_records,
            final_image_attention_observability=metadata.get(
                "final_image_attention_observability"
            ),
            final_image_preservation=metadata.get(
                "final_image_preservation"
            ),
            carrier_only_final_image_preservation=metadata.get(
                "carrier_only_final_image_preservation"
            ),
            carrier_only_counterfactual=counterfactual,
            attention_geometry_enabled=attention_geometry_enabled,
            image_alignment_enabled=image_alignment_enabled,
            semantic_routing_enabled=semantic_routing_enabled,
            null_space_enabled=null_space_enabled,
            full_active_branches=full_active_branches,
            carrier_only_active_branches=carrier_only_active_branches,
        )
    except (
        OSError,
        OverflowError,
        RuntimeError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ):
        return False
    return rebuilt_record == binding_record


def load_completed_semantic_watermark_runtime_result(
    config: SemanticWatermarkRuntimeConfig,
    root: str | Path = ".",
) -> SemanticWatermarkRuntimeResult | None:
    """读取同代码版本、同配置且文件完整的已完成运行。

    该函数用于 Colab 跨会话续跑。缓存只在运行决策为 pass、配置摘要一致、
    Git 代码版本一致且 manifest 中全部输出文件仍存在时复用, 避免把半写入
    目录或旧算法结果混入当前正式记录。
    """

    root_path = Path(root).resolve()
    run_id = build_semantic_watermark_run_id(config)
    run_dir = (root_path / config.output_dir / run_id).resolve()
    manifest_path = run_dir / "manifest.local.json"
    result_path = run_dir / "runtime_result.json"
    detection_path = run_dir / "image_only_detection_records.jsonl"
    if not manifest_path.is_file() or not result_path.is_file() or not detection_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        result_payload = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    expected_config_digest = semantic_watermark_runtime_config_digest(config)
    manifest_config = manifest.get("config")
    if (
        not isinstance(manifest_config, Mapping)
        or manifest_config.get("scientific_unit_config_digest")
        != expected_config_digest
        or manifest.get("config_digest")
        != build_stable_digest(manifest_config)
    ):
        return None
    if manifest.get("code_version") != resolve_code_version(root_path):
        return None
    if result_payload.get("run_decision") != "pass":
        return None
    try:
        validate_semantic_watermark_runtime_result_provenance(
            result_payload,
            expected_config=config,
        )
    except (TypeError, ValueError):
        return None
    output_paths = tuple(str(path) for path in manifest.get("output_paths", ()))
    if not output_paths or not all((root_path / path).is_file() for path in output_paths):
        return None
    if not _carrier_only_counterfactual_artifact_binding_ready(
        result_payload,
        manifest,
        root_path,
        config,
    ):
        return None
    if not _scientific_content_binding_artifact_ready(
        result_payload,
        manifest,
        root_path,
        config,
    ):
        return None
    try:
        return SemanticWatermarkRuntimeResult(**result_payload)
    except TypeError:
        return None


def _attention_modules(
    pipeline: Any,
    layer_names: tuple[str, ...],
) -> tuple[tuple[str, Any], ...]:
    """按配置中的精确层名解析真实 Q/K 注意力模块."""

    available = dict(pipeline.transformer.named_modules())
    resolved = []
    for layer_name in layer_names:
        module = available.get(layer_name)
        if module is None:
            raise RuntimeError(
                f"冻结注意力层不存在: {layer_name}"
            )
        if not all(
            hasattr(module, attribute)
            for attribute in ("to_q", "to_k", "heads")
        ):
            raise RuntimeError(
                f"冻结注意力层不满足公开 Q/K 协议: {layer_name}"
            )
        resolved.append((layer_name, module))
    return tuple(resolved)


def _unconditional_embeddings(
    pipeline: Any,
    device: Any,
    conditioning_protocol: str,
    condition_text: str,
) -> tuple[Any, Any]:
    """构造嵌入端和检测端都可复现的空文本条件。"""

    if (
        conditioning_protocol
        != _FORMAL_METHOD_CONFIG.public_detection_conditioning_protocol
        or condition_text != ""
    ):
        raise ValueError("公开检测条件必须使用冻结的 SD3 三路空文本编码")
    prompt_embeds, _, pooled_prompt_embeds, _ = pipeline.encode_prompt(
        prompt=condition_text,
        prompt_2=condition_text,
        prompt_3=condition_text,
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=False,
    )
    return prompt_embeds, pooled_prompt_embeds


def _transformer_forward_function(
    pipeline: Any,
    timestep: Any,
    prompt_embeds: Any,
    pooled_prompt_embeds: Any,
) -> Any:
    """构造以 latent 为唯一变量的真实 Transformer 前向函数。"""

    transformer_dtype = next(pipeline.transformer.parameters()).dtype

    def forward(latent: Any) -> Any:
        timestep_batch = timestep.expand(latent.shape[0])
        return pipeline.transformer(
            hidden_states=latent.to(dtype=transformer_dtype),
            timestep=timestep_batch,
            encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled_prompt_embeds,
            joint_attention_kwargs=None,
            return_dict=False,
        )[0]

    return forward


def _branch_budget(
    latent: Any,
    branch_field: Any,
    *,
    semantic_routing_enabled: bool,
    budget_ceiling: float,
) -> Any:
    """构造一次并复用的 contiguous NCHW 有效风险预算 Tensor。"""

    import torch

    if latent.ndim != 4:
        raise ValueError("分支风险预算要求 NCHW latent")
    batch, channels, height, width = (
        int(value) for value in latent.shape
    )
    values = (
        branch_field.effective_budget_values
        if semantic_routing_enabled
        else tuple(
            float(budget_ceiling)
            for _ in branch_field.effective_budget_values
        )
    )
    if len(values) != batch * height * width:
        raise RuntimeError("分支风险预算与逐样本 latent 空间网格不一致")
    spatial = torch.tensor(
        values,
        device=latent.device,
        dtype=torch.float32,
    ).reshape(batch, 1, height, width)
    return spatial.expand(batch, channels, height, width).contiguous()


def _constant_branch_budget(latent: Any, budget_ceiling: float) -> Any:
    """为移除风险路由的消融直接构造全支持常量预算."""

    import torch

    if latent.ndim != 4:
        raise ValueError("常量分支预算要求 NCHW latent")
    return torch.full_like(
        latent,
        float(budget_ceiling),
        dtype=torch.float32,
        memory_format=torch.contiguous_format,
    )


def _active_carrier_branch_names(
    config: SemanticWatermarkRuntimeConfig,
) -> tuple[str, ...]:
    """返回当前机制配置中实际参与嵌入的载体分支。"""

    enabled = {
        "lf_content": config.lf_enabled,
        "tail_robust": config.tail_robust_enabled,
        "attention_geometry": config.attention_geometry_enabled,
    }
    return tuple(branch_name for branch_name, is_enabled in enabled.items() if is_enabled)


def _required_branch_risk_eligibility(
    config: SemanticWatermarkRuntimeConfig,
) -> tuple[str, ...]:
    """返回需要通过风险资格门禁的活动载体分支."""

    if not config.semantic_routing_enabled:
        return ()
    return _active_carrier_branch_names(config)


def _branch_risk_configs(
    config: SemanticWatermarkRuntimeConfig,
) -> dict[str, BranchRiskConfig]:
    """返回完整方法或共享全局风险正式消融使用的风险配置。

    完整方法逐项转换唯一 YAML 已解析配置, 不在核心层保留第二份默认常量。
    ``shared_global`` 把同一个预注册全局风险复制到三个载体分支, 用于直接
    检验分支特定风险语义是否优于单一共享路由。该对照仍执行真实风险门控,
    不等价于完全移除路由。
    """

    from main.methods.semantic.branch_risk import BranchRiskConfig

    if config.branch_risk_mode == "branch_specific":
        formal_configs = {
            "lf_content": config.lf_content_risk_config,
            "tail_robust": config.tail_robust_risk_config,
            "attention_geometry": config.attention_geometry_risk_config,
        }
        return {
            branch_name: BranchRiskConfig(**asdict(formal_config))
            for branch_name, formal_config in formal_configs.items()
        }
    shared_config = BranchRiskConfig(
        local_contrast_risk_weight=0.25,
        semantic_weight=0.25,
        texture_weight=0.20,
        adjacent_step_instability_weight=0.20,
        attention_instability_weight=0.10,
        texture_preference="avoid",
        eligibility_threshold=0.55,
        budget_floor=0.05,
        budget_ceiling=1.0,
        budget_gain=0.70,
    )
    return {
        branch_name: shared_config
        for branch_name in (
            "lf_content",
            "tail_robust",
            "attention_geometry",
        )
    }


def _branch_risk_record(branch_field: Any) -> dict[str, Any]:
    """生成带精确风险、预算与资格 mask 内容摘要的分支记录."""

    return {
        "branch_name": branch_field.branch_name,
        "risk_field_digest": branch_field.risk_field_digest,
        "risk_values_content_sha256": (
            branch_field.risk_values_content_sha256
        ),
        "budget_values_content_sha256": (
            branch_field.budget_values_content_sha256
        ),
        "eligible_mask_content_sha256": (
            branch_field.eligible_mask_content_sha256
        ),
        "risk_value_mean": sum(branch_field.risk_values) / len(branch_field.risk_values),
        "budget_value_mean": sum(branch_field.budget_values) / len(branch_field.budget_values),
        "effective_budget_value_mean": (
            sum(branch_field.effective_budget_values)
            / len(branch_field.effective_budget_values)
        ),
        "eligible_position_count": len(branch_field.eligible_indices),
        "risk_field_position_count": len(branch_field.risk_values),
        "metadata": branch_field.metadata,
    }


_RISK_SIGNAL_CONTENT_FIELDS = (
    "current_decoded_rgb_content_sha256",
    "previous_step_decoded_rgb_content_sha256",
    "clip_patch_tokens_content_sha256",
    "clip_cls_token_content_sha256",
    "semantic_risk_signal_content_sha256",
    "texture_risk_signal_content_sha256",
    "local_contrast_risk_signal_content_sha256",
    "adjacent_step_stability_signal_content_sha256",
    "attention_stability_signal_content_sha256",
)
_BRANCH_RISK_BASE_CONTENT_FIELDS = (
    "risk_values_content_sha256",
    "budget_values_content_sha256",
    "eligible_mask_content_sha256",
)
_BRANCH_RISK_ENVELOPE_CONTENT_FIELDS = (
    "effective_budget_values_content_sha256",
    "branch_unit_direction_content_sha256",
    "branch_budget_envelope_content_sha256",
    "branch_written_update_content_sha256",
)
_BRANCH_RISK_POST_NULL_CONTENT_FIELDS = (
    "branch_post_risk_direction_content_sha256",
    "branch_post_risk_reference_direction_content_sha256",
    "branch_post_risk_response_content_sha256",
    "branch_post_risk_reference_response_content_sha256",
)


def _branch_risk_content_evidence(
    risk_signal_source: Mapping[str, Any],
    branch_risk_records: Mapping[str, Mapping[str, Any]],
    *,
    semantic_routing_enabled: bool,
    null_space_enabled: bool,
    active_branch_names: tuple[str, ...],
) -> dict[str, Any]:
    """按唯一角色集合构造可重算的风险输入与分支内容证据。"""

    if not semantic_routing_enabled:
        if risk_signal_source or branch_risk_records:
            raise ValueError("风险路由关闭时不得保留风险信号或分支风险原子")
        return {
            "semantic_routing_enabled": False,
            "active_carrier_branches": list(active_branch_names),
            "risk_signal_content_records": {},
            "branch_risk_content_records": {},
        }
    if set(branch_risk_records) != set(active_branch_names):
        raise ValueError("branch_risk_records 必须精确覆盖活动载体分支")
    risk_signal_content_records = {
        field_name: risk_signal_source.get(field_name)
        for field_name in _RISK_SIGNAL_CONTENT_FIELDS
        if risk_signal_source.get(field_name) not in {None, ""}
    }
    branch_risk_content_records: dict[str, dict[str, Any]] = {}
    for branch_name in active_branch_names:
        branch_record = branch_risk_records[branch_name]
        content_record = {
            field_name: branch_record.get(field_name)
            for field_name in _BRANCH_RISK_BASE_CONTENT_FIELDS
        }
        if any(
            field_name not in branch_record
            for field_name in _BRANCH_RISK_ENVELOPE_CONTENT_FIELDS
        ):
            raise ValueError("活动分支缺少完整风险包络内容字段")
        content_record.update(
            {
                field_name: branch_record[field_name]
                for field_name in _BRANCH_RISK_ENVELOPE_CONTENT_FIELDS
            }
        )
        post_null_fields_present = tuple(
            field_name in branch_record
            for field_name in _BRANCH_RISK_POST_NULL_CONTENT_FIELDS
        )
        if null_space_enabled:
            if not all(post_null_fields_present):
                raise ValueError("Null Space 活动分支缺少 post-risk JVP 内容字段")
            content_record.update(
                {
                    field_name: branch_record[field_name]
                    for field_name in _BRANCH_RISK_POST_NULL_CONTENT_FIELDS
                }
            )
        elif any(post_null_fields_present):
            raise ValueError("Null Space 关闭时不得保留 post-risk JVP 内容字段")
        branch_risk_content_records[branch_name] = content_record
    return {
        "semantic_routing_enabled": True,
        "active_carrier_branches": list(active_branch_names),
        "risk_signal_content_records": risk_signal_content_records,
        "branch_risk_content_records": branch_risk_content_records,
    }



def _feature_preservation_values(
    semantic_before: Any,
    structure_before: Any,
    semantic_after: Any,
    structure_after: Any,
    config: SemanticWatermarkRuntimeConfig,
) -> tuple[float, float, bool]:
    """统一计算完整 CLIP 与手工结构统计的保持指标和门禁."""

    import torch
    import torch.nn.functional as functional

    semantic_before_flat = semantic_before.float().reshape(-1)
    semantic_after_flat = semantic_after.float().reshape(-1)
    structure_before_flat = structure_before.float().reshape(-1)
    structure_after_flat = structure_after.float().reshape(-1)
    if semantic_before_flat.shape != semantic_after_flat.shape:
        raise RuntimeError("写回前后完整 CLIP 特征宽度不一致")
    if structure_before_flat.shape != structure_after_flat.shape:
        raise RuntimeError("写回前后手工结构统计特征宽度不一致")
    semantic_cosine = float(
        functional.cosine_similarity(
            semantic_before_flat,
            semantic_after_flat,
            dim=0,
            eps=config.null_space_numerical_epsilon,
        ).item()
    )
    structure_relative_drift = float(
        torch.linalg.norm(structure_after_flat - structure_before_flat).item()
        / max(
            float(torch.linalg.norm(structure_before_flat).item()),
            config.null_space_numerical_epsilon,
        )
    )
    ready = bool(
        math.isfinite(semantic_cosine)
        and math.isfinite(structure_relative_drift)
        and semantic_cosine >= config.minimum_semantic_preservation_cosine
        and structure_relative_drift <= config.maximum_handcrafted_structure_feature_relative_drift
    )
    return semantic_cosine, structure_relative_drift, ready


def _quantized_write_jacobian_response_record(
    feature_function: Any | None,
    latent: Any,
    injected: Any,
    maximum_relative_response: float,
    numerical_epsilon: float,
) -> dict[str, Any]:
    """复验实际量化写回 Tensor 的完整特征 Jacobian 响应。

    Null Space 基底在 float32 中求解, 但扩散 latent 通常使用 float16。该函数
    先按真实 latent dtype 完成加法, 再以 ``written_latent - latent`` 恢复实际
    写入增量。相对响应以当前完整特征向量二范数归一化, 因而直接表示一阶
    特征变化相对于当前语义与视觉状态的比例。该门禁验证量化后的写回对象,
    不能由量化前的分支方向或有限更新保持记录替代。
    """

    import torch
    from main.methods.subspace.jacobian_nullspace import exact_jvp

    if not 0.0 < maximum_relative_response <= 1.0:
        raise ValueError("实际写回 Jacobian 相对响应阈值必须位于 (0, 1]")
    if not math.isfinite(numerical_epsilon) or numerical_epsilon <= 0.0:
        raise ValueError("实际写回 Jacobian 数值 epsilon 必须为有限正数")
    if tuple(latent.shape) != tuple(injected.shape):
        raise ValueError("实际写回前后的 latent 形状必须一致")
    quantized_latent = injected.detach().to(dtype=latent.dtype)
    quantized_update = quantized_latent - latent.detach()
    update_norm = float(torch.linalg.norm(quantized_update.float()).item())
    base_record = {
        "quantized_write_update_content_sha256": tensor_content_sha256(
            quantized_update
        ),
        "quantized_write_update_dtype": str(quantized_update.dtype),
        "quantized_write_update_shape": [
            int(value) for value in quantized_update.shape
        ],
        "quantized_write_update_norm": update_norm,
        "maximum_quantized_write_relative_jacobian_response": (
            maximum_relative_response
        ),
    }
    if feature_function is None:
        return {
            **base_record,
            "quantized_write_jacobian_gate_applicable": False,
            "quantized_write_jacobian_response_norm": None,
            "quantized_write_reference_feature_norm": None,
            "quantized_write_relative_jacobian_response": None,
            "quantized_write_jacobian_gate_ready": False,
            "quantized_write_jacobian_status": (
                "not_applicable_jacobian_null_space_disabled"
            ),
        }
    primal, response = exact_jvp(
        feature_function,
        latent.detach().float(),
        quantized_update.float(),
    )
    response = response.detach().float()
    response_norm = float(torch.linalg.norm(response).item())
    reference_feature_norm = float(
        torch.linalg.norm(primal.detach().float()).item()
    )
    relative_response = response_norm / max(
        reference_feature_norm,
        numerical_epsilon,
    )
    ready = bool(
        math.isfinite(update_norm)
        and update_norm > 0.0
        and math.isfinite(response_norm)
        and math.isfinite(reference_feature_norm)
        and math.isfinite(relative_response)
        and relative_response <= maximum_relative_response
    )
    return {
        **base_record,
        "quantized_write_reference_feature_content_sha256": (
            tensor_content_sha256(primal.detach().float())
        ),
        "quantized_write_jacobian_response_content_sha256": (
            tensor_content_sha256(response)
        ),
        "quantized_write_jacobian_gate_applicable": True,
        "quantized_write_jacobian_response_norm": response_norm,
        "quantized_write_reference_feature_norm": reference_feature_norm,
        "quantized_write_relative_jacobian_response": relative_response,
        "quantized_write_jacobian_gate_ready": ready,
        "quantized_write_jacobian_status": (
            "measured_from_actual_quantized_latent_delta"
        ),
    }


def _quantized_write_update_nonzero(record: dict[str, Any]) -> bool:
    """验证实际 dtype 写回非零, 该条件不依赖 Null Space 是否启用。"""

    value = record.get("quantized_write_update_norm")
    return bool(
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and float(value) > 0.0
    )


def _post_risk_direction_jacobian_record(
    feature_function: Any,
    latent: Any,
    branch_name: str,
    actual_unit_direction: Any,
    reference_direction: Any,
    maximum_relative_response: float,
    numerical_epsilon: float,
) -> dict[str, Any]:
    """对风险支持修正后的实际单位方向重新执行精确 JVP。

    Null Space 求解器验证的是其输出基底。风险硬包络还会清除零预算位置的
    数值泄漏并重新单位化，因此必须对最终实际方向单独复验，不能依赖三分支
    联合更新可能发生的响应抵消。参考分母使用投影前的真实载体或 Q/K 梯度
    方向响应，与 Null Space 的低响应比例定义保持一致。
    """

    import torch
    from main.methods.semantic.branch_risk import BRANCH_NAMES
    from main.methods.subspace.jacobian_nullspace import exact_jvp

    if branch_name not in BRANCH_NAMES:
        raise ValueError("branch_name 不是冻结的三分支角色")
    if not 0.0 < maximum_relative_response <= 1.0:
        raise ValueError("逐分支 JVP 相对响应阈值必须位于 (0, 1]")
    if not math.isfinite(numerical_epsilon) or numerical_epsilon <= 0.0:
        raise ValueError("逐分支 JVP numerical_epsilon 必须为正有限数")
    actual = torch.as_tensor(
        actual_unit_direction,
        device=latent.device,
        dtype=torch.float32,
    )
    reference = torch.as_tensor(
        reference_direction,
        device=latent.device,
        dtype=torch.float32,
    )
    if actual.shape != latent.shape or reference.shape != latent.shape:
        raise ValueError("逐分支 JVP 方向必须与 latent 形状一致")
    if not bool(torch.isfinite(actual).all()) or not bool(
        torch.isfinite(reference).all()
    ):
        raise ValueError("逐分支 JVP 方向必须全部有限")
    actual_norm = torch.linalg.vector_norm(actual)
    reference_norm = torch.linalg.vector_norm(reference)
    if (
        float(actual_norm.item()) <= numerical_epsilon
        or float(reference_norm.item()) <= numerical_epsilon
    ):
        raise RuntimeError("逐分支 JVP 方向不得为零")
    if not math.isclose(
        float(actual_norm.item()),
        1.0,
        rel_tol=1e-6,
        abs_tol=1e-6,
    ):
        raise RuntimeError("风险支持修正后的实际方向必须已经单位化")
    reference = reference / reference_norm
    _, actual_response = exact_jvp(
        feature_function,
        latent.detach().float(),
        actual,
    )
    _, reference_response = exact_jvp(
        feature_function,
        latent.detach().float(),
        reference,
    )
    actual_response = actual_response.detach().float()
    reference_response = reference_response.detach().float()
    response_norm = float(torch.linalg.vector_norm(actual_response).item())
    reference_response_norm = float(
        torch.linalg.vector_norm(reference_response).item()
    )
    relative_response = response_norm / max(
        reference_response_norm,
        numerical_epsilon,
    )
    if not bool(
        math.isfinite(response_norm)
        and math.isfinite(reference_response_norm)
        and reference_response_norm > numerical_epsilon
        and math.isfinite(relative_response)
        and relative_response <= maximum_relative_response
    ):
        raise RuntimeError(
            f"{branch_name} 风险支持修正后的实际方向未通过精确 JVP 门禁"
        )
    return {
        "branch_post_risk_direction_content_sha256": (
            tensor_content_sha256(actual)
        ),
        "branch_post_risk_reference_direction_content_sha256": (
            tensor_content_sha256(reference)
        ),
        "branch_post_risk_response_content_sha256": (
            tensor_content_sha256(actual_response)
        ),
        "branch_post_risk_reference_response_content_sha256": (
            tensor_content_sha256(reference_response)
        ),
        "branch_post_risk_response_norm": response_norm,
        "branch_post_risk_reference_response_norm": (
            reference_response_norm
        ),
        "branch_post_risk_relative_response_residual": relative_response,
        "branch_post_risk_jacobian_gate_ready": True,
        "branch_post_risk_jvp_mode": (
            "torch_autograd_exact_jvp_vjp_reexecution"
        ),
    }


def _combined_update_preservation_record(
    feature_runtime: DifferentiableSemanticFeatureRuntime,
    latent: Any,
    injected: Any,
    config: SemanticWatermarkRuntimeConfig,
) -> dict[str, Any]:
    """验证一次实际写回 latent 的完整特征有限更新保持性。"""

    import torch

    with torch.no_grad():
        semantic_before, structure_before = feature_runtime.joint_features(
            latent.detach().float()
        )
        semantic_after, structure_after = feature_runtime.joint_features(
            injected.detach().float()
        )
    semantic_cosine, structure_relative_drift, ready = _feature_preservation_values(
        semantic_before,
        structure_before,
        semantic_after,
        structure_after,
        config,
    )
    return {
        "full_semantic_cosine_similarity": semantic_cosine,
        "full_handcrafted_structure_feature_relative_drift": structure_relative_drift,
        "minimum_semantic_preservation_cosine": (
            config.minimum_semantic_preservation_cosine
        ),
        "maximum_handcrafted_structure_feature_relative_drift": (
            config.maximum_handcrafted_structure_feature_relative_drift
        ),
        "semantic_preservation_gate_ready": ready,
        "preservation_validation_scope": (
            "actual_combined_latent_full_clip_and_handcrafted_structure_features"
        ),
    }


def _final_image_preservation_record(
    pipeline: Any,
    feature_runtime: DifferentiableSemanticFeatureRuntime,
    clean_image: Any,
    watermarked_image: Any,
    config: SemanticWatermarkRuntimeConfig,
) -> dict[str, Any]:
    """在最终成图上验证累计 CLIP 与手工结构统计保持性."""

    clean_features, watermarked_features = _final_image_joint_features(
        pipeline,
        feature_runtime,
        clean_image,
        watermarked_image,
    )
    semantic_cosine, structure_relative_drift, ready = _feature_preservation_values(
        clean_features[0],
        clean_features[1],
        watermarked_features[0],
        watermarked_features[1],
        config,
    )
    return {
        "final_image_semantic_cosine_similarity": semantic_cosine,
        "final_image_handcrafted_structure_feature_relative_drift": structure_relative_drift,
        "minimum_semantic_preservation_cosine": (
            config.minimum_semantic_preservation_cosine
        ),
        "maximum_handcrafted_structure_feature_relative_drift": (
            config.maximum_handcrafted_structure_feature_relative_drift
        ),
        "final_image_preservation_gate_ready": ready,
        "preservation_validation_scope": (
            "paired_final_images_full_clip_and_handcrafted_structure_features"
        ),
    }


def _final_image_joint_features(
    pipeline: Any,
    feature_runtime: DifferentiableSemanticFeatureRuntime,
    *images: Any,
) -> tuple[tuple[Any, Any], ...]:
    """一次性提取最终成图的完整 CLIP 与手工结构统计."""

    import torch

    device = pipeline._execution_device

    def image_tensor(image: Any) -> Any:
        """把最终 PIL 成图转换为 [0, 1] 模型输入 tensor。"""

        pixels = pipeline.image_processor.preprocess(image).to(
            device=device,
            dtype=torch.float32,
        )
        return (pixels / 2.0 + 0.5).clamp(0.0, 1.0)

    with torch.no_grad():
        return tuple(
            feature_runtime.joint_image_features(image_tensor(image))
            for image in images
        )


def _three_way_final_image_preservation_records(
    pipeline: Any,
    feature_runtime: DifferentiableSemanticFeatureRuntime,
    clean_image: Any,
    carrier_only_image: Any,
    watermarked_image: Any,
    config: SemanticWatermarkRuntimeConfig,
    carrier_only_counterfactual: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """验证 clean、carrier-only 与完整方法最终成图的三边保持性。"""

    identity_digest = str(
        carrier_only_counterfactual.get(
            "carrier_only_counterfactual_identity_digest",
            "",
        )
    )
    if (
        carrier_only_counterfactual.get("carrier_only_counterfactual_ready")
        is not True
        or len(identity_digest) != 64
    ):
        raise RuntimeError("carrier-only 最终保持门禁缺少反事实身份")
    clean_features, carrier_features, watermarked_features = (
        _final_image_joint_features(
            pipeline,
            feature_runtime,
            clean_image,
            carrier_only_image,
            watermarked_image,
        )
    )
    clean_full_values = _feature_preservation_values(
        clean_features[0],
        clean_features[1],
        watermarked_features[0],
        watermarked_features[1],
        config,
    )
    clean_carrier_values = _feature_preservation_values(
        clean_features[0],
        clean_features[1],
        carrier_features[0],
        carrier_features[1],
        config,
    )
    carrier_full_values = _feature_preservation_values(
        carrier_features[0],
        carrier_features[1],
        watermarked_features[0],
        watermarked_features[1],
        config,
    )
    final_record = {
        "final_image_semantic_cosine_similarity": clean_full_values[0],
        "final_image_handcrafted_structure_feature_relative_drift": clean_full_values[1],
        "minimum_semantic_preservation_cosine": (
            config.minimum_semantic_preservation_cosine
        ),
        "maximum_handcrafted_structure_feature_relative_drift": (
            config.maximum_handcrafted_structure_feature_relative_drift
        ),
        "final_image_preservation_gate_ready": clean_full_values[2],
        "preservation_validation_scope": (
            "clean_to_full_final_images_full_clip_and_handcrafted_structure_features"
        ),
    }
    counterfactual_record = {
        "carrier_only_final_image_preservation_applicable": True,
        "carrier_only_final_image_semantic_cosine_similarity": (
            clean_carrier_values[0]
        ),
        "carrier_only_final_image_handcrafted_structure_feature_relative_drift": (
            clean_carrier_values[1]
        ),
        "carrier_only_final_image_preservation_gate_ready": (
            clean_carrier_values[2]
        ),
        "carrier_only_to_full_final_image_semantic_cosine_similarity": (
            carrier_full_values[0]
        ),
        "carrier_only_to_full_final_image_handcrafted_structure_feature_relative_drift": (
            carrier_full_values[1]
        ),
        "carrier_only_to_full_final_image_preservation_gate_ready": (
            carrier_full_values[2]
        ),
        "carrier_only_counterfactual_three_way_preservation_gate_ready": (
            clean_full_values[2]
            and clean_carrier_values[2]
            and carrier_full_values[2]
        ),
        "minimum_semantic_preservation_cosine": (
            config.minimum_semantic_preservation_cosine
        ),
        "maximum_handcrafted_structure_feature_relative_drift": (
            config.maximum_handcrafted_structure_feature_relative_drift
        ),
        "carrier_only_counterfactual_identity_digest": identity_digest,
        "carrier_only_final_image_preservation_status": (
            "measured_from_clean_carrier_only_and_full_final_images"
        ),
        "preservation_validation_scope": (
            "three_pair_final_images_full_clip_and_handcrafted_structure_features"
        ),
    }
    return final_record, counterfactual_record


def _final_image_attention_attribution_gate_ready(
    *,
    blind_attribution_gain: float,
    frozen_pair_attribution_gain: float,
    minimum_gain: float,
    measured_values: tuple[float, ...],
    relation_identity_ready: bool,
) -> bool:
    """要求盲选择与冻结 carrier pair 两条归因证据同时通过。"""

    return bool(
        relation_identity_ready
        and all(math.isfinite(value) for value in measured_values)
        and blind_attribution_gain > minimum_gain
        and frozen_pair_attribution_gain > minimum_gain
    )


def _final_image_attention_observability_record(
    image_attention_extractor: Any | None,
    clean_image: Any,
    carrier_only_image: Any | None,
    watermarked_image: Any,
    config: SemanticWatermarkRuntimeConfig,
    *,
    carrier_only_counterfactual: Mapping[str, Any] | None = None,
    require_gpu_execution: bool = True,
) -> dict[str, Any]:
    """以 carrier-only 反事实验证最终成图中的 attention 因果增益。

    clean 只保留为总体水印对照。正式 attention 归因比较同 seed、同 scheduler、
    同 LF/tail 配置与算子的 carrier-only 图像与完整方法图像, 估计含下游交互的
    总机制效应, 不假设两侧已经实现相同 carrier。盲分数允许两张图分别执行自身
    稳定 token 选择; 配对分数则冻结 carrier-only 的 pair 权重。两种归因增益都
    必须严格超过正下界, 且不得使用中间 latent 分数替代最终成图 Q/K。
    """

    source = "image_reencoded_public_noise_real_qk"
    if not config.attention_geometry_enabled:
        return {
            "final_image_attention_observability_applicable": False,
            "final_image_attention_observability_gate_ready": False,
            "final_image_attention_observability_source": source,
            "final_image_attention_observability_requires_gpu": True,
            "final_image_attention_observability_gpu_execution_verified": False,
            "minimum_final_image_attention_score_gain": (
                config.minimum_final_image_attention_score_gain
            ),
            "final_clean_blind_attention_score": None,
            "final_carrier_only_blind_attention_score": None,
            "final_watermarked_blind_attention_score": None,
            "final_image_blind_attention_score_gain": None,
            "final_image_attention_blind_attribution_gain": None,
            "final_clean_paired_attention_score": None,
            "final_carrier_only_paired_attention_score": None,
            "final_watermarked_carrier_paired_attention_score": None,
            "final_watermarked_paired_attention_score": None,
            "final_image_paired_attention_score_gain": None,
            "final_image_attention_carrier_paired_attribution_gain": None,
            "final_clean_pair_weight_identity_digest": "",
            "final_carrier_only_pair_weight_identity_digest": "",
            "final_watermarked_pair_weight_identity_digest": "",
            "final_paired_pair_weight_identity_digest": "",
            "final_image_attention_record_schema_digest": "",
            "attention_relation_component_names": [],
            "attention_relation_active_component_names": [],
            "attention_relation_component_weights": [],
            "attention_relation_component_protocol_digest": "",
            "attention_relation_source": "",
            "attention_relation_direct_qk_source_ready": False,
            "attention_relation_component_identity_digest": "",
            "attention_relation_keyed_projection_digest": "",
            "attention_relation_qk_operator_metadata_records": [],
            "attention_relation_qk_operator_metadata_digest": "",
            "attention_relation_qk_operator_metadata_ready": False,
            "final_image_qk_atomic_content_records": [],
            "final_image_qk_atomic_content_digest": "",
            "final_image_qk_atomic_content_ready": False,
            "final_image_public_detection_noise_evidence_records": [],
            "final_image_public_detection_noise_evidence_digest": "",
            "final_image_public_detection_noise_content_sha256": "",
            "final_image_public_detection_noise_prg_identity_digest": "",
            "final_image_public_detection_noise_evidence_ready": False,
            "attention_module_names": list(config.attention_module_names),
            "attention_coordinate_convention": (
                config.attention_coordinate_convention
            ),
            "attention_grid_align_corners": (
                config.attention_grid_align_corners
            ),
            "final_carrier_only_paired_attention_component_scores": {},
            "final_watermarked_carrier_paired_attention_component_scores": {},
            "final_image_attention_carrier_paired_component_gains": {},
            "carrier_only_counterfactual_ready": False,
            "observability_status": "not_applicable_attention_geometry_disabled",
        }
    if image_attention_extractor is None:
        raise RuntimeError("最终成图注意力可观测性门禁缺少真实 Q/K 提取器")
    if carrier_only_image is None or not carrier_only_counterfactual:
        raise RuntimeError("最终成图注意力归因缺少 carrier-only 反事实")
    if carrier_only_counterfactual.get("carrier_only_counterfactual_ready") is not True:
        raise RuntimeError("carrier-only 反事实身份没有通过同种子同调度门禁")

    public_noise_cursor = _public_detection_noise_evidence_cursor(
        image_attention_extractor
    )
    if public_noise_cursor != 0:
        raise RuntimeError("最终三图 Q/K 必须占用公开噪声全局索引0至2")
    clean_records = tuple(image_attention_extractor(clean_image))
    carrier_only_records = tuple(image_attention_extractor(carrier_only_image))
    watermarked_records = tuple(image_attention_extractor(watermarked_image))
    if any(
        len(records) < 2
        for records in (clean_records, carrier_only_records, watermarked_records)
    ):
        raise RuntimeError("最终成图注意力可观测性要求至少两个真实 Q/K 层")

    def record_schema(records: tuple[Any, ...]) -> tuple[Any, ...]:
        """返回 Q/K 层名称与二维 token 网格的共同身份。"""

        return tuple(
            (layer_name, tuple(token_indices))
            for layer_name, _, token_indices in records
        )

    clean_record_schema = record_schema(clean_records)
    carrier_only_record_schema = record_schema(carrier_only_records)
    watermarked_record_schema = record_schema(watermarked_records)
    if not (
        clean_record_schema
        == carrier_only_record_schema
        == watermarked_record_schema
    ):
        raise RuntimeError("最终三图 Q/K 层身份或二维网格不一致")
    if tuple(name for name, _ in clean_record_schema) != (
        config.attention_module_names
    ):
        raise RuntimeError("最终成图 Q/K 记录没有使用配置冻结的精确层名")
    relation_identities = tuple(
        build_attention_relation_graph_identity(
            records,
            config.key_material,
            prg_version=config.keyed_prg_version,
            component_weights=config.attention_relation_component_weights,
        )
        for records in (
            clean_records,
            carrier_only_records,
            watermarked_records,
        )
    )
    relation_identity = relation_identities[0]
    relation_identity_ready = all(
        identity.relation_source == DIRECT_QK_RELATION_SOURCE
        and identity.qk_operator_metadata_ready
        and identity.qk_atomic_content_ready
        and identity.component_names == relation_identity.component_names
        and identity.active_component_names
        == relation_identity.active_component_names
        and identity.component_weights == relation_identity.component_weights
        and identity.component_protocol_digest
        == relation_identity.component_protocol_digest
        and identity.component_identity_digest
        == relation_identity.component_identity_digest
        and identity.keyed_projection_digest
        == relation_identity.keyed_projection_digest
        and identity.qk_operator_metadata_digest
        == relation_identity.qk_operator_metadata_digest
        for identity in relation_identities
    )
    if not relation_identity_ready:
        raise RuntimeError("最终三图没有共享直接 Q/K 四分量关系图身份")
    public_noise_records = [
        dict(record)
        for record in getattr(
            image_attention_extractor,
            "public_detection_noise_evidence_records",
            (),
        )[public_noise_cursor:]
    ]
    if (
        len(public_noise_records) != 3
        or [
            record.get("public_detection_noise_evaluation_index")
            for record in public_noise_records
        ]
        != [0, 1, 2]
        or len(
            {
                record.get("public_detection_noise_content_sha256")
                for record in public_noise_records
            }
        )
        != 1
        or len(
            {
                record.get(
                    "public_detection_noise_prg_identity_digest"
                )
                for record in public_noise_records
            }
        )
        != 1
    ):
        raise RuntimeError("最终三图 Q/K 缺少唯一且连续的公开噪声证据")
    final_image_qk_atomic_content_records = tuple(
        {
            "qk_evaluation_role": evaluation_role,
            "qk_atomic_content_records": list(
                identity.qk_atomic_content_records
            ),
            "qk_atomic_content_digest": identity.qk_atomic_content_digest,
            "qk_atomic_content_ready": identity.qk_atomic_content_ready,
            "public_detection_noise_content_sha256": noise_record[
                "public_detection_noise_content_sha256"
            ],
            "public_detection_noise_prg_identity_digest": noise_record[
                "public_detection_noise_prg_identity_digest"
            ],
            "public_detection_noise_evaluation_index": noise_record[
                "public_detection_noise_evaluation_index"
            ],
        }
        for evaluation_role, identity, noise_record in zip(
            (
                "final_clean_image",
                "final_carrier_only_image",
                "final_watermarked_image",
            ),
            relation_identities,
            public_noise_records,
        )
    )
    all_records = clean_records + carrier_only_records + watermarked_records
    gpu_verified = all(
        getattr(attention, "device", None) is not None
        and attention.device.type == "cuda"
        for _, attention, _ in all_records
    )
    if require_gpu_execution and not gpu_verified:
        raise RuntimeError("最终成图真实 Q/K 可观测性必须在 CUDA 上执行")

    clean_selection = select_stable_attention_tokens(
        clean_records,
        stable_token_fraction=config.attention_stable_token_fraction,
    )
    carrier_only_selection = select_stable_attention_tokens(
        carrier_only_records,
        stable_token_fraction=config.attention_stable_token_fraction,
    )
    watermarked_selection = select_stable_attention_tokens(
        watermarked_records,
        stable_token_fraction=config.attention_stable_token_fraction,
    )
    clean_pair_weights = build_stable_attention_pair_weights(
        clean_records,
        clean_selection,
        unstable_pair_weight=config.attention_unstable_pair_weight,
    )
    carrier_only_pair_weights = build_stable_attention_pair_weights(
        carrier_only_records,
        carrier_only_selection,
        unstable_pair_weight=config.attention_unstable_pair_weight,
    )
    watermarked_pair_weights = build_stable_attention_pair_weights(
        watermarked_records,
        watermarked_selection,
        unstable_pair_weight=config.attention_unstable_pair_weight,
    )

    def score(records: tuple[Any, ...], pair_weights: Any) -> float:
        """以显式冻结 pair 权重计算最终成图真实 Q/K 分数。"""

        value = attention_geometry_score(
            records,
            config.key_material,
            prg_version=config.keyed_prg_version,
            stable_pair_weights=pair_weights,
            component_weights=config.attention_relation_component_weights,
        )
        return float(value.detach().item())

    clean_blind_score = score(clean_records, clean_pair_weights)
    carrier_only_blind_score = score(
        carrier_only_records,
        carrier_only_pair_weights,
    )
    watermarked_blind_score = score(
        watermarked_records,
        watermarked_pair_weights,
    )
    clean_paired_score = score(clean_records, clean_pair_weights)
    watermarked_paired_score = score(watermarked_records, clean_pair_weights)
    carrier_only_paired_score = score(
        carrier_only_records,
        carrier_only_pair_weights,
    )
    watermarked_carrier_paired_score = score(
        watermarked_records,
        carrier_only_pair_weights,
    )
    carrier_only_paired_components = attention_geometry_component_scores(
        carrier_only_records,
        config.key_material,
        carrier_only_pair_weights,
        prg_version=config.keyed_prg_version,
        component_weights=config.attention_relation_component_weights,
    )
    watermarked_carrier_paired_components = attention_geometry_component_scores(
        watermarked_records,
        config.key_material,
        carrier_only_pair_weights,
        prg_version=config.keyed_prg_version,
        component_weights=config.attention_relation_component_weights,
    )
    carrier_paired_component_gains = (
        watermarked_carrier_paired_components - carrier_only_paired_components
    )
    clean_control_blind_gain = watermarked_blind_score - clean_blind_score
    clean_control_paired_gain = watermarked_paired_score - clean_paired_score
    blind_attribution_gain = watermarked_blind_score - carrier_only_blind_score
    carrier_paired_attribution_gain = (
        watermarked_carrier_paired_score - carrier_only_paired_score
    )
    values = (
        clean_blind_score,
        carrier_only_blind_score,
        watermarked_blind_score,
        clean_paired_score,
        carrier_only_paired_score,
        watermarked_carrier_paired_score,
        watermarked_paired_score,
        clean_control_blind_gain,
        clean_control_paired_gain,
        blind_attribution_gain,
        carrier_paired_attribution_gain,
    )
    ready = _final_image_attention_attribution_gate_ready(
        blind_attribution_gain=blind_attribution_gain,
        frozen_pair_attribution_gain=carrier_paired_attribution_gain,
        minimum_gain=config.minimum_final_image_attention_score_gain,
        measured_values=values,
        relation_identity_ready=relation_identity_ready,
    )
    return {
        **dict(carrier_only_counterfactual),
        "final_image_attention_observability_applicable": True,
        "final_image_attention_observability_gate_ready": ready,
        "final_image_attention_observability_source": source,
        "final_image_attention_observability_requires_gpu": True,
        "final_image_attention_observability_gpu_execution_verified": gpu_verified,
        "minimum_final_image_attention_score_gain": (
            config.minimum_final_image_attention_score_gain
        ),
        "final_clean_blind_attention_score": clean_blind_score,
        "final_carrier_only_blind_attention_score": carrier_only_blind_score,
        "final_watermarked_blind_attention_score": watermarked_blind_score,
        "final_image_blind_attention_score_gain": clean_control_blind_gain,
        "final_image_attention_blind_attribution_gain": blind_attribution_gain,
        "final_clean_paired_attention_score": clean_paired_score,
        "final_carrier_only_paired_attention_score": carrier_only_paired_score,
        "final_watermarked_carrier_paired_attention_score": (
            watermarked_carrier_paired_score
        ),
        "final_watermarked_paired_attention_score": watermarked_paired_score,
        "final_image_paired_attention_score_gain": clean_control_paired_gain,
        "final_image_attention_carrier_paired_attribution_gain": (
            carrier_paired_attribution_gain
        ),
        "final_clean_pair_weight_identity_digest": (
            clean_pair_weights.pair_weight_identity_digest
        ),
        "final_carrier_only_pair_weight_identity_digest": (
            carrier_only_pair_weights.pair_weight_identity_digest
        ),
        "final_watermarked_pair_weight_identity_digest": (
            watermarked_pair_weights.pair_weight_identity_digest
        ),
        "final_paired_pair_weight_identity_digest": (
            clean_pair_weights.pair_weight_identity_digest
        ),
        "final_image_attention_record_schema_digest": build_stable_digest(
            {"attention_record_schema": clean_record_schema}
        ),
        "attention_relation_component_names": list(
            relation_identity.component_names
        ),
        "attention_relation_active_component_names": list(
            relation_identity.active_component_names
        ),
        "attention_relation_component_weights": list(
            relation_identity.component_weights
        ),
        "attention_relation_component_protocol_digest": (
            relation_identity.component_protocol_digest
        ),
        "attention_relation_source": relation_identity.relation_source,
        "attention_relation_direct_qk_source_ready": relation_identity_ready,
        "attention_relation_probability_scope": (
            "sampled_image_token_qk_relation_probability"
        ),
        "attention_relation_component_identity_digest": (
            relation_identity.component_identity_digest
        ),
        "attention_relation_keyed_projection_digest": (
            relation_identity.keyed_projection_digest
        ),
        "attention_relation_qk_operator_metadata_records": list(
            relation_identity.qk_operator_metadata_records
        ),
        "attention_relation_qk_operator_metadata_digest": (
            relation_identity.qk_operator_metadata_digest
        ),
        "attention_relation_qk_operator_metadata_ready": (
            relation_identity.qk_operator_metadata_ready
        ),
        "final_image_qk_atomic_content_records": list(
            final_image_qk_atomic_content_records
        ),
        "final_image_qk_atomic_content_digest": (
            qk_atomic_evaluation_records_digest(
                final_image_qk_atomic_content_records,
                "final_image_qk_atomic_content_records",
            )
        ),
        "final_image_qk_atomic_content_ready": all(
            bool(record["qk_atomic_content_ready"])
            for record in final_image_qk_atomic_content_records
        ),
        "final_image_public_detection_noise_evidence_records": (
            public_noise_records
        ),
        "final_image_public_detection_noise_evidence_digest": (
            build_stable_digest(
                {
                    "final_image_public_detection_noise_evidence_records": (
                        public_noise_records
                    )
                }
            )
        ),
        "final_image_public_detection_noise_content_sha256": (
            public_noise_records[0][
                "public_detection_noise_content_sha256"
            ]
        ),
        "final_image_public_detection_noise_prg_identity_digest": (
            public_noise_records[0][
                "public_detection_noise_prg_identity_digest"
            ]
        ),
        "final_image_public_detection_noise_evidence_ready": True,
        "attention_module_names": list(config.attention_module_names),
        "attention_coordinate_convention": (
            config.attention_coordinate_convention
        ),
        "attention_grid_align_corners": (
            config.attention_grid_align_corners
        ),
        "final_carrier_only_paired_attention_component_scores": {
            name: float(value)
            for name, value in zip(
                ATTENTION_RELATION_COMPONENT_NAMES,
                carrier_only_paired_components.detach().cpu().tolist(),
            )
        },
        "final_watermarked_carrier_paired_attention_component_scores": {
            name: float(value)
            for name, value in zip(
                ATTENTION_RELATION_COMPONENT_NAMES,
                watermarked_carrier_paired_components.detach().cpu().tolist(),
            )
        },
        "final_image_attention_carrier_paired_component_gains": {
            name: float(value)
            for name, value in zip(
                ATTENTION_RELATION_COMPONENT_NAMES,
                carrier_paired_component_gains.detach().cpu().tolist(),
            )
        },
        "observability_status": "measured_from_carrier_only_counterfactual_real_qk",
    }


class _FullLatentSpace:
    """在 Null Space 消融中提供不改变方向的完整空间投影。"""

    solver_digest = build_stable_digest(
        {"solver_role": "full_latent_space_ablation"}
    )

    @staticmethod
    def project(tensor: Any) -> Any:
        """原样返回 tensor。"""

        return tensor


def _encode_image_latent(pipeline: Any, image: Any) -> Any:
    """仅从待检图像执行 VAE 编码, 不读取生成轨迹。"""

    import torch

    dtype = next(pipeline.vae.parameters()).dtype
    pixels = pipeline.image_processor.preprocess(image).to(device=pipeline._execution_device, dtype=dtype)
    with torch.no_grad():
        encoded = pipeline.vae.encode(pixels).latent_dist.mode()
    shift_factor = float(pipeline.vae.config.shift_factor)
    scaling_factor = float(pipeline.vae.config.scaling_factor)
    return (encoded - shift_factor) * scaling_factor


def _public_detection_noise_prg_identity(
    config: SemanticWatermarkRuntimeConfig,
    latent_shape: tuple[int, ...],
) -> dict[str, Any]:
    """返回公开检测噪声的完整逐字节 PRG 调用身份。"""

    domain_fields = {
        "operator": config.public_detection_noise_domain,
        "model_id": config.model_id,
        "model_revision": config.model_revision,
        "width": config.width,
        "height": config.height,
        "inference_steps": config.inference_steps,
        "public_detection_schedule_index": (
            config.public_detection_schedule_index
        ),
        "latent_shape": latent_shape,
    }
    payload: dict[str, Any] = {
        "public_detection_noise_prg_protocol": (
            config.public_detection_noise_prg_protocol
        ),
        "keyed_prg_protocol_digest": keyed_prg_protocol_record(
            config.public_detection_noise_prg_protocol
        )["keyed_prg_protocol_digest"],
        "key_material": config.public_detection_noise_domain,
        "domain_fields": domain_fields,
        "shape": latent_shape,
    }
    return {
        **payload,
        "public_detection_noise_prg_identity_digest": build_stable_digest(
            payload
        ),
    }


def _public_detection_noise_seed(config: SemanticWatermarkRuntimeConfig) -> int:
    """为记录层生成不参与 PRG 字节流的公开随机追踪编号。"""

    identity_digest = build_stable_digest(
        {
            "public_detection_noise_prg_protocol": (
                config.public_detection_noise_prg_protocol
            ),
            "public_detection_noise_domain": (
                config.public_detection_noise_domain
            ),
            "model_id": config.model_id,
            "model_revision": config.model_revision,
            "width": config.width,
            "height": config.height,
            "inference_steps": config.inference_steps,
            "public_detection_schedule_index": (
                config.public_detection_schedule_index
            ),
        }
    )
    return int(identity_digest[:16], 16) % (2**63 - 1)


def _public_detection_noise_tensor(
    latent: Any,
    config: SemanticWatermarkRuntimeConfig,
) -> Any:
    """在 CPU 规范域生成公开高斯噪声, 再搬运到 latent 的设备与 dtype。"""

    shape = tuple(int(value) for value in latent.shape)
    identity = _public_detection_noise_prg_identity(config, shape)
    noise_cpu = build_keyed_gaussian_tensor(
        shape,
        key_material=config.public_detection_noise_domain,
        domain_fields=identity["domain_fields"],
        prg_version=config.public_detection_noise_prg_protocol,
    )
    typed_noise_cpu = noise_cpu.to(device="cpu", dtype=latent.dtype)
    return typed_noise_cpu.to(device=latent.device)


def _image_attention_extractor(
    pipeline: Any,
    config: SemanticWatermarkRuntimeConfig,
    modules: tuple[tuple[str, Any], ...],
    prompt_embeds: Any,
    pooled_prompt_embeds: Any,
) -> Any:
    """构造只使用图像 VAE latent 的确定性注意力提取器。"""

    import torch

    detection_index = config.public_detection_schedule_index
    noise_evidence_records: list[dict[str, Any]] = []

    def extract(image: Any) -> tuple[tuple[str, Any, tuple[int, ...]], ...]:
        """从任意待检图像提取全部冻结层的公开固定噪声 Q/K 关系。"""

        # img2img, 反演和再生成攻击会与主 pipeline 共享 scheduler 实例并改写
        # timesteps, begin_index 和 step_index. 每次检测都重新建立正式检测日程,
        # 保证 scale_noise 使用的 sigma 与 Transformer 前向 timestep 属于同一日程.
        pipeline.scheduler.set_timesteps(
            config.inference_steps,
            device=pipeline._execution_device,
        )
        timestep = pipeline.scheduler.timesteps[detection_index]
        scale_noise = getattr(pipeline.scheduler, "scale_noise", None)
        if not callable(scale_noise):
            raise RuntimeError(
                "正式仅图像 Q/K 提取要求 scheduler 提供可调用的 scale_noise"
            )
        latent = _encode_image_latent(pipeline, image)
        noise = _public_detection_noise_tensor(latent, config)
        noise_identity = _public_detection_noise_prg_identity(
            config,
            tuple(int(value) for value in latent.shape),
        )
        noise_evidence_records.append(
            {
                "public_detection_noise_evaluation_index": len(
                    noise_evidence_records
                ),
                "tensor_content_digest_version": (
                    TENSOR_CONTENT_DIGEST_VERSION
                ),
                "public_detection_noise_content_sha256": (
                    tensor_content_sha256(noise)
                ),
                "public_detection_noise_prg_identity_digest": (
                    noise_identity[
                        "public_detection_noise_prg_identity_digest"
                    ]
                ),
                "public_detection_noise_prg_identity": noise_identity,
                "public_detection_noise_shape": [
                    int(value) for value in noise.shape
                ],
                "public_detection_noise_dtype": str(noise.dtype),
            }
        )
        timestep_batch = timestep.reshape(1).expand(latent.shape[0])
        noisy_latent = scale_noise(latent, timestep_batch, noise)
        with DifferentiableAttentionRecorder(modules, max_tokens=config.max_attention_tokens) as recorder:
            with torch.no_grad():
                _transformer_forward_function(
                    pipeline,
                    timestep,
                    prompt_embeds,
                    pooled_prompt_embeds,
                )(noisy_latent)
            if not recorder.records:
                raise RuntimeError("图像盲检没有捕获到真实 Q/K attention")
            records = tuple(
                (layer_name, attention.detach(), token_indices)
                for layer_name, attention, token_indices in recorder.records
            )
        return records

    setattr(
        extract,
        "public_detection_noise_evidence_records",
        noise_evidence_records,
    )
    return extract


def _build_image_only_measurement_config(
    config: SemanticWatermarkRuntimeConfig,
) -> ImageOnlyMeasurementConfig:
    """把正式运行配置映射为阈值无关的核心盲检测量配置."""

    lf_weight = (
        config.lf_detection_score_weight
        if config.lf_enabled and config.tail_robust_enabled
        else (1.0 if config.lf_enabled else 0.0)
    )
    return ImageOnlyMeasurementConfig(
        model_id=config.model_id,
        model_revision=config.model_revision,
        vae_class_name=config.vae_class_name,
        transformer_class_name=config.transformer_class_name,
        scheduler_class_name=config.scheduler_class_name,
        vae_scaling_factor=config.vae_scaling_factor,
        vae_shift_factor=config.vae_shift_factor,
        latent_torch_dtype=config.latent_torch_dtype,
        width=config.width,
        height=config.height,
        inference_steps=config.inference_steps,
        public_detection_schedule_index=(
            config.public_detection_schedule_index
        ),
        public_detection_noise_prg_protocol=(
            config.public_detection_noise_prg_protocol
        ),
        public_detection_noise_domain=(
            config.public_detection_noise_domain
        ),
        public_detection_conditioning_protocol=(
            config.public_detection_conditioning_protocol
        ),
        public_detection_condition_text=(
            config.public_detection_condition_text
        ),
        max_attention_tokens=config.max_attention_tokens,
        attention_coordinate_convention=(
            config.attention_coordinate_convention
        ),
        attention_grid_align_corners=(
            config.attention_grid_align_corners
        ),
        attention_module_names=config.attention_module_names,
        low_frequency_config=config.low_frequency_carrier_config,
        keyed_prg_version=config.keyed_prg_version,
        lf_weight=lf_weight,
        tail_robust_weight=(
            config.tail_robust_detection_score_weight
            if config.lf_enabled and config.tail_robust_enabled
            else 1.0 - lf_weight
        ),
        tail_fraction=(
            config.tail_fraction if config.tail_truncation_enabled else 1.0
        ),
        attention_stable_token_fraction=(
            config.attention_stable_token_fraction
        ),
        attention_unstable_pair_weight=(
            config.attention_unstable_pair_weight
        ),
        attention_relation_component_weights=(
            config.attention_relation_component_weights
        ),
        attention_anchor_count=config.attention_anchor_count,
        attention_residual_threshold=config.attention_residual_threshold,
        attention_minimum_inlier_ratio=(
            config.attention_minimum_inlier_ratio
        ),
        method_role=(
            "lf_only_content"
            if config.lf_enabled and not config.tail_robust_enabled
            else "hf_tail_only_content"
            if config.tail_robust_enabled and not config.lf_enabled
            else "full_dual_chain"
        ),
    )


def _public_detection_noise_evidence_cursor(extractor: Any | None) -> int:
    """返回 extractor 当前公开检测噪声证据条数。"""

    records = getattr(
        extractor,
        "public_detection_noise_evidence_records",
        (),
    )
    return len(records)


def _discard_public_detection_noise_evidence_since(
    extractor: Any | None,
    cursor: int,
) -> None:
    """丢弃只服务攻击优化目标且不会持久化的临时检测证据。"""

    records = getattr(
        extractor,
        "public_detection_noise_evidence_records",
        None,
    )
    if isinstance(records, list):
        del records[cursor:]


def _bind_public_detection_noise_qk_evidence(
    record: dict[str, Any],
    extractor: Any | None,
    cursor: int,
) -> None:
    """把本次公开噪声实际 Tensor 摘要绑定到对应检测 Q/K 证据。"""

    records = getattr(
        extractor,
        "public_detection_noise_evidence_records",
        (),
    )
    evidence_records = [dict(item) for item in records[cursor:]]
    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        raise RuntimeError("检测记录缺少可绑定公开噪声的 metadata")
    qk_records_value = metadata.get("detection_qk_atomic_content_records")
    if not isinstance(qk_records_value, list) or len(qk_records_value) != len(
        evidence_records
    ):
        raise RuntimeError("公开检测噪声证据数量与 Q/K 评价次数不一致")
    if not evidence_records:
        raise RuntimeError("真实 Q/K 检测缺少公开噪声 Tensor 内容摘要")
    qk_records = [dict(item) for item in qk_records_value]
    for qk_record, noise_record in zip(qk_records, evidence_records):
        qk_record.update(
            {
                "public_detection_noise_content_sha256": noise_record[
                    "public_detection_noise_content_sha256"
                ],
                "public_detection_noise_prg_identity_digest": noise_record[
                    "public_detection_noise_prg_identity_digest"
                ],
                "public_detection_noise_evaluation_index": noise_record[
                    "public_detection_noise_evaluation_index"
                ],
            }
        )
    qk_digest = qk_atomic_evaluation_records_digest(
        qk_records,
        "detection_qk_atomic_content_records",
    )
    expected_roles_by_count = {
        1: ("raw_detection_image",),
        2: ("raw_detection_image", "aligned_detection_image"),
    }
    expected_roles = expected_roles_by_count.get(len(qk_records))
    if expected_roles is None:
        raise RuntimeError("检测 Q/K 公开噪声证据包含未定义的评价次数")
    first_atoms = qk_records[0].get("qk_atomic_content_records")
    if not isinstance(first_atoms, list):
        raise RuntimeError("检测 Q/K 证据缺少逐层原子记录")
    expected_layer_names = tuple(
        str(item["record_layer_name"]) for item in first_atoms
    )
    qk_ready = qk_atomic_evaluation_records_ready(
        qk_records,
        qk_digest,
        aggregate_field_name="detection_qk_atomic_content_records",
        expected_roles=expected_roles,
        expected_layer_names=expected_layer_names,
    )
    content_digests = {
        str(item["public_detection_noise_content_sha256"])
        for item in evidence_records
    }
    identity_digests = {
        str(item["public_detection_noise_prg_identity_digest"])
        for item in evidence_records
    }
    if len(content_digests) != 1 or len(identity_digests) != 1 or not qk_ready:
        raise RuntimeError("公开检测噪声没有与 Q/K 证据形成唯一稳定绑定")
    noise_evidence_digest = build_stable_digest(
        {"public_detection_noise_evidence_records": evidence_records}
    )
    content_digest = next(iter(content_digests))
    identity_digest = next(iter(identity_digests))
    metadata.update(
        {
            "public_detection_noise_evidence_records": evidence_records,
            "public_detection_noise_evidence_digest": noise_evidence_digest,
            "public_detection_noise_content_sha256": content_digest,
            "public_detection_noise_prg_identity_digest": identity_digest,
            "public_detection_noise_evidence_ready": True,
            "detection_qk_atomic_content_records": qk_records,
            "detection_qk_atomic_content_digest": qk_digest,
            "detection_qk_atomic_content_ready": True,
        }
    )
    record.update(
        {
            "public_detection_noise_content_sha256": content_digest,
            "public_detection_noise_prg_identity_digest": identity_digest,
            "public_detection_noise_evidence_digest": noise_evidence_digest,
        }
    )
    record["measurement_digest"] = build_stable_digest(
        recompute_image_only_measurement_digest_payload(record)
    )


def _runtime_public_detection_noise_identity(
    detections: Sequence[Mapping[str, Any]],
    *,
    attention_geometry_enabled: bool,
) -> tuple[dict[str, Any] | None, str]:
    """仅在注意力检测实际执行时提取公开噪声 PRG 身份."""

    if not attention_geometry_enabled:
        return None, ""
    if not detections:
        raise RuntimeError("注意力检测缺少可提取公开噪声身份的记录")
    first_detection = detections[0]
    metadata = first_detection.get("metadata")
    if not isinstance(metadata, Mapping):
        raise RuntimeError("注意力检测缺少公开噪声 metadata")
    evidence_records = metadata.get("public_detection_noise_evidence_records")
    if not isinstance(evidence_records, list) or not evidence_records:
        raise RuntimeError("注意力检测缺少公开噪声证据")
    identity = evidence_records[0].get("public_detection_noise_prg_identity")
    digest = first_detection.get("public_detection_noise_prg_identity_digest")
    if not isinstance(identity, Mapping) or not _is_sha256_hex(digest):
        raise RuntimeError("注意力检测的公开噪声 PRG 身份无效")
    resolved_identity = dict(identity)
    nested_digest = resolved_identity.pop(
        "public_detection_noise_prg_identity_digest",
        None,
    )
    if nested_digest != digest or build_stable_digest(resolved_identity) != digest:
        raise RuntimeError("注意力检测的公开噪声 PRG 身份不能独立重建")
    return dict(identity), str(digest)


def _align_image(image: Any, alignment: Any) -> Any:
    """依据恢复的仿射参考系对待检图像执行可复现重采样。"""

    import torch
    from PIL import Image

    rgb = image.convert("RGB")
    width, height = rgb.size
    pixels = torch.frombuffer(bytearray(rgb.tobytes()), dtype=torch.uint8).reshape(height, width, 3)
    aligned = resample_attention_aligned_rgb_uint8(
        pixels.permute(2, 0, 1).unsqueeze(0),
        alignment.affine_transform,
    )
    output = aligned[0].permute(1, 2, 0).numpy()
    return Image.fromarray(output, mode="RGB")


def _carrier_only_counterfactual_identity(
    full_config: SemanticWatermarkRuntimeConfig | Mapping[str, Any],
    carrier_only_config: SemanticWatermarkRuntimeConfig | Mapping[str, Any],
    full_update_records: list[dict[str, Any]],
    carrier_only_update_records: list[dict[str, Any]],
) -> dict[str, Any]:
    """核验仅关闭 attention geometry 的总机制效应反事实身份。"""

    full_payload = (
        semantic_watermark_runtime_config_payload(full_config)
        if isinstance(full_config, SemanticWatermarkRuntimeConfig)
        else dict(full_config)
    )
    carrier_payload = (
        semantic_watermark_runtime_config_payload(carrier_only_config)
        if isinstance(
            carrier_only_config,
            SemanticWatermarkRuntimeConfig,
        )
        else dict(carrier_only_config)
    )
    changed_fields = tuple(
        sorted(
            field_name
            for field_name in full_payload
            if full_payload[field_name] != carrier_payload[field_name]
        )
    )
    if changed_fields != ("attention_geometry_enabled",):
        raise RuntimeError(
            "carrier-only 反事实必须只关闭 attention_geometry_enabled"
        )
    if (
        full_payload.get("attention_geometry_enabled") is not True
        or carrier_payload.get("attention_geometry_enabled") is not False
        or full_payload.get("seed") != carrier_payload.get("seed")
    ):
        raise RuntimeError("carrier-only 反事实的 attention 开关或生成种子不一致")

    expected_steps = tuple(
        int(value)
        for value in full_payload.get("injection_step_indices", ())
    )
    expected_full_branches = tuple(
        branch_name
        for branch_name, field_name in (
            ("lf_content", "lf_enabled"),
            ("tail_robust", "tail_robust_enabled"),
            ("attention_geometry", "attention_geometry_enabled"),
        )
        if full_payload.get(field_name) is True
    )
    expected_carrier_branches = tuple(
        branch_name
        for branch_name, field_name in (
            ("lf_content", "lf_enabled"),
            ("tail_robust", "tail_robust_enabled"),
            ("attention_geometry", "attention_geometry_enabled"),
        )
        if carrier_payload.get(field_name) is True
    )
    keyed_prg_version = str(full_payload.get("keyed_prg_version", ""))
    attention_operator_schedule_index = full_payload.get(
        "attention_operator_schedule_index"
    )
    if (
        type(attention_operator_schedule_index) is not int
        or attention_operator_schedule_index < 0
    ):
        raise RuntimeError("carrier-only 反事实缺少合法的固定注意力算子索引")
    attention_module_names = tuple(
        str(name) for name in full_payload.get("attention_module_names", ())
    )
    null_space_enabled = full_payload.get("null_space_enabled") is True
    maximum_quantized_response = float(
        full_payload.get(
            "maximum_quantized_write_relative_jacobian_response",
            -1.0,
        )
    )
    attention_component_weights = tuple(
        float(value)
        for value in full_payload.get(
            "attention_relation_component_weights",
            (),
        )
    )
    if (
        len(full_update_records) != len(expected_steps)
        or len(carrier_only_update_records) != len(expected_steps)
    ):
        raise RuntimeError("完整方法与 carrier-only 必须精确覆盖全部注入步")

    required_content_sha256_fields = (
        "latent_content_sha256_before",
        "latent_content_sha256_after",
        "combined_update_content_sha256",
        "quantized_write_update_content_sha256",
        "adjacent_step_reference_latent_content_sha256",
    )

    def validate_common_record(
        record: Mapping[str, Any],
        *,
        execution_role: str,
        attention_enabled: bool,
        expected_branches: tuple[str, ...],
    ) -> None:
        """验证反事实两侧更新原子的共同内容与分支身份。"""

        metadata = record.get("metadata")
        null_space_records = record.get("null_space_records")
        if not isinstance(metadata, Mapping) or not isinstance(
            null_space_records,
            Mapping,
        ):
            raise RuntimeError("反事实更新原子缺少 metadata 或 Null Space 记录")
        if (
            metadata.get("injection_execution_role") != execution_role
            or metadata.get("attention_geometry_enabled") is not attention_enabled
            or record.get("active_carrier_branches") != list(expected_branches)
            or set(null_space_records) != set(expected_branches)
        ):
            raise RuntimeError("反事实更新原子的执行角色或活动分支身份不一致")
        if record.get("tensor_content_digest_version") != (
            TENSOR_CONTENT_DIGEST_VERSION
        ):
            raise RuntimeError("反事实更新原子的 Tensor 内容摘要版本无效")
        for field_name in required_content_sha256_fields:
            value = str(record.get(field_name, ""))
            if len(value) != 64 or any(
                character not in "0123456789abcdef" for character in value
            ):
                raise RuntimeError("反事实更新原子缺少完整 tensor 内容 SHA-256")
        for branch_name, field_name in (
            ("lf_content", "lf_update_content_sha256"),
            ("tail_robust", "tail_robust_update_content_sha256"),
            ("attention_geometry", "attention_geometry_update_content_sha256"),
        ):
            value = record.get(field_name)
            if branch_name in expected_branches:
                if not _is_sha256_hex(value):
                    raise RuntimeError("活动分支缺少真实更新 Tensor 摘要")
            elif value != "":
                raise RuntimeError("已禁用分支仍保留更新 Tensor 原子")
        branch_update_content_records = {
            "lf_content": record.get("lf_update_content_sha256"),
            "tail_robust": record.get("tail_robust_update_content_sha256"),
            "attention_geometry": record.get(
                "attention_geometry_update_content_sha256"
            ),
        }
        if record.get("branch_updates_content_digest") != build_stable_digest(
            branch_update_content_records
        ):
            raise RuntimeError("反事实更新原子的三分支 Tensor 摘要不一致")
        branch_risk_records = record.get("branch_risk_records")
        semantic_routing_enabled = metadata.get("semantic_routing_enabled") is True
        expected_risk_branches = (
            set(expected_branches) if semantic_routing_enabled else set()
        )
        if not isinstance(branch_risk_records, Mapping) or set(
            branch_risk_records
        ) != expected_risk_branches:
            raise RuntimeError("反事实更新原子的活动风险记录集合无效")
        try:
            branch_risk_content_evidence = _branch_risk_content_evidence(
                record,
                branch_risk_records,
                semantic_routing_enabled=semantic_routing_enabled,
                null_space_enabled=null_space_enabled,
                active_branch_names=expected_branches,
            )
        except ValueError as error:
            raise RuntimeError(
                "反事实更新原子的分支风险 Tensor 证据结构无效"
            ) from error
        if (
            any(
                not _is_sha256_hex(value)
                for content_group in (
                    branch_risk_content_evidence[
                        "risk_signal_content_records"
                    ].values(),
                    *(
                        content_record.values()
                        for content_record in branch_risk_content_evidence[
                            "branch_risk_content_records"
                        ].values()
                    ),
                )
                for value in content_group
            )
            or record.get("branch_risk_content_digest")
            != build_stable_digest(branch_risk_content_evidence)
        ):
            raise RuntimeError("反事实更新原子的分支风险 Tensor 摘要不一致")
        step_index = record.get("step_index")
        if (
            not isinstance(step_index, int)
            or record.get("adjacent_step_reference_index") != step_index - 1
            or record.get("post_step_schedule_index") != step_index + 1
            or record.get("attention_operator_schedule_index")
            != attention_operator_schedule_index
            or not isinstance(
                record.get("post_step_schedule_timestep"),
                (int, float),
            )
            or not math.isfinite(
                float(record["post_step_schedule_timestep"])
            )
            or not isinstance(
                record.get("attention_operator_timestep"),
                (int, float),
            )
            or not math.isfinite(
                float(record["attention_operator_timestep"])
            )
            or record.get("adjacent_step_stability_status")
            != (
                "measured_from_immediately_previous_scheduler_step"
                if semantic_routing_enabled
                else "not_applicable_semantic_routing_disabled"
            )
        ):
            raise RuntimeError("反事实更新原子的相邻调度步稳定度身份无效")
        expected_prg_digest = keyed_prg_protocol_record(
            keyed_prg_version
        )["keyed_prg_protocol_digest"]
        if (
            record.get("keyed_prg_version") != keyed_prg_version
            or record.get("keyed_prg_protocol_digest")
            != expected_prg_digest
        ):
            raise RuntimeError("反事实更新原子的密钥 PRG 协议身份无效")
        if (
            record.get("attention_module_names")
            != list(attention_module_names)
            or record.get("attention_coordinate_convention")
            != ATTENTION_COORDINATE_CONVENTION
            or record.get("attention_grid_align_corners")
            is not ATTENTION_GRID_ALIGN_CORNERS
        ):
            raise RuntimeError("反事实更新原子的注意力层或坐标身份无效")
        quantized_gate_applicable = record.get(
            "quantized_write_jacobian_gate_applicable"
        )
        quantized_gate_ready = record.get(
            "quantized_write_jacobian_gate_ready"
        )
        if not _quantized_write_update_nonzero(dict(record)):
            raise RuntimeError("反事实更新原子的实际 dtype 写回不得为0")
        if null_space_enabled:
            relative_response = record.get(
                "quantized_write_relative_jacobian_response"
            )
            if (
                quantized_gate_applicable is not True
                or quantized_gate_ready is not True
                or not isinstance(relative_response, (int, float))
                or not math.isfinite(float(relative_response))
                or float(relative_response)
                > maximum_quantized_response
            ):
                raise RuntimeError("反事实更新原子的实际量化写回 Jacobian 门禁无效")
        elif (
            quantized_gate_applicable is not False
            or quantized_gate_ready is not False
            or record.get("quantized_write_jacobian_status")
            != "not_applicable_jacobian_null_space_disabled"
        ):
            raise RuntimeError("Null Space 消融错误声明实际量化写回 Jacobian 门禁")

    for full_record in full_update_records:
        validate_common_record(
            full_record,
            execution_role="full_method",
            attention_enabled=True,
            expected_branches=expected_full_branches,
        )
        if full_record.get("metadata", {}).get("attention_source") != (
            "real_qk_projection"
        ):
            raise RuntimeError("完整方法更新原子缺少真实 Q/K attention 来源")
        component_protocol = attention_relation_component_protocol(
            attention_component_weights
        )
        if (
            full_record.get("attention_relation_component_names")
            != list(ATTENTION_RELATION_COMPONENT_NAMES)
            or full_record.get("attention_relation_active_component_names")
            != list(
                component_protocol[
                    "attention_relation_active_component_names"
                ]
            )
            or full_record.get("attention_relation_component_weights")
            != list(attention_component_weights)
            or full_record.get(
                "attention_relation_component_protocol_digest"
            )
            != component_protocol[
                "attention_relation_component_protocol_digest"
            ]
        ):
            raise RuntimeError("完整方法更新原子的四分量权重协议无效")
        qk_operator_records = full_record.get(
            "attention_relation_qk_operator_metadata_records"
        )
        if (
            full_record.get("attention_relation_qk_operator_metadata_ready")
            is not True
            or not isinstance(qk_operator_records, list)
            or not qk_operator_metadata_records_ready(
                qk_operator_records,
                attention_module_names,
            )
            or full_record.get(
                "attention_relation_qk_operator_metadata_digest"
            )
            != qk_operator_metadata_records_digest(qk_operator_records)
        ):
            raise RuntimeError("完整方法更新原子的 Q/K 算子元数据无效")
        if (
            full_record.get("attention_qk_atomic_content_ready") is not True
            or not qk_atomic_evaluation_records_ready(
                full_record.get("attention_qk_atomic_content_records"),
                full_record.get("attention_qk_atomic_content_digest"),
                aggregate_field_name="attention_qk_atomic_content_records",
                expected_roles=(
                    "latent_before",
                    "optimization_content_base_latent",
                    "accepted_attention_candidate",
                    "actual_written_content_base_latent",
                    "actual_written_combined_latent",
                ),
                expected_layer_names=attention_module_names,
                require_evaluation_identity=True,
            )
        ):
            raise RuntimeError("完整方法更新原子缺少真实 Q/K 原子内容摘要")

    carrier_none_fields = (
        "attention_score_before",
        "attention_content_base_score",
        "attention_score_after",
        "attention_actual_written_content_base_score",
        "attention_final_combined_score",
        "attention_score_gain",
        "attention_applied_update_strength",
        "attention_backtracking_step_count",
    )
    carrier_empty_string_fields = (
        "attention_update_digest",
        "attention_update_content_sha256",
        "attention_update_unit_direction_content_sha256",
        "stable_token_selection_digest",
        "stable_pair_weight_identity_digest",
        "stable_pair_weight_realization_digest",
        "attention_relation_source",
        "attention_relation_probability_scope",
        "attention_relation_component_identity_digest",
        "attention_relation_keyed_projection_digest",
        "attention_relation_qk_operator_metadata_digest",
        "attention_relation_component_protocol_digest",
        "attention_qk_atomic_content_digest",
    )
    carrier_empty_list_fields = (
        "stable_token_indices",
        "attention_relation_component_names",
        "attention_relation_active_component_names",
        "attention_relation_component_weights",
        "attention_relation_qk_operator_metadata_records",
        "attention_qk_atomic_content_records",
    )
    for carrier_record in carrier_only_update_records:
        validate_common_record(
            carrier_record,
            execution_role="carrier_only_counterfactual",
            attention_enabled=False,
            expected_branches=expected_carrier_branches,
        )
        carrier_metadata = carrier_record["metadata"]
        if carrier_metadata.get("attention_source") != (
            "disabled_attention_geometry"
        ):
            raise RuntimeError("carrier-only 更新原子错误声明真实 Q/K attention 来源")
        if any(carrier_record.get(field_name) is not None for field_name in carrier_none_fields):
            raise RuntimeError("carrier-only 更新原子仍包含 attention 数值")
        if any(carrier_record.get(field_name) != "" for field_name in carrier_empty_string_fields):
            raise RuntimeError("carrier-only 更新原子仍包含 attention 或 pair 身份")
        if any(carrier_record.get(field_name) != [] for field_name in carrier_empty_list_fields):
            raise RuntimeError("carrier-only 更新原子仍包含 attention 关系集合")
        if carrier_record.get("attention_relation_direct_qk_source_ready") is not False:
            raise RuntimeError("carrier-only 更新原子错误声明直接 Q/K 来源")
        if carrier_record.get("attention_relation_qk_operator_metadata_ready") is not False:
            raise RuntimeError("carrier-only 更新原子错误声明 Q/K 算子元数据完整")
        if carrier_record.get("attention_qk_atomic_content_ready") is not False:
            raise RuntimeError("carrier-only 更新原子错误声明 Q/K 原子内容完整")

    def scheduler_trace(records: list[dict[str, Any]]) -> tuple[dict[str, Any], ...]:
        """提取足以核验相同 scheduler 轨迹的冻结字段。"""

        return tuple(
            {
                "step_index": int(record["step_index"]),
                "scheduler_step_timestep": float(
                    record["scheduler_step_timestep"]
                ),
                "post_step_schedule_index": int(
                    record["post_step_schedule_index"]
                ),
                "post_step_schedule_timestep": float(
                    record["post_step_schedule_timestep"]
                ),
                "attention_operator_schedule_index": int(
                    record["attention_operator_schedule_index"]
                ),
                "attention_operator_timestep": float(
                    record["attention_operator_timestep"]
                ),
            }
            for record in records
        )

    full_trace = scheduler_trace(full_update_records)
    carrier_trace = scheduler_trace(carrier_only_update_records)
    if (
        tuple(item["step_index"] for item in full_trace) != expected_steps
        or tuple(item["step_index"] for item in carrier_trace) != expected_steps
        or full_trace != carrier_trace
        or {
            item["attention_operator_schedule_index"]
            for item in full_trace
        }
        != {attention_operator_schedule_index}
        or len(
            {
                item["attention_operator_timestep"]
                for item in full_trace
            }
        )
        != 1
    ):
        raise RuntimeError("carrier-only 反事实没有复现完整方法的 scheduler 轨迹")
    full_initial_latent_sha256 = str(
        full_update_records[0]["latent_content_sha256_before"]
    )
    carrier_initial_latent_sha256 = str(
        carrier_only_update_records[0]["latent_content_sha256_before"]
    )
    if full_initial_latent_sha256 != carrier_initial_latent_sha256:
        raise RuntimeError("完整方法与 carrier-only 的首个注入前 latent 不一致")

    record = {
        "carrier_only_counterfactual_changed_fields": list(changed_fields),
        "carrier_only_counterfactual_generation_seed_random": int(
            full_payload["seed"]
        ),
        "carrier_only_counterfactual_config_digest": build_stable_digest(
            carrier_payload
        ),
        "full_method_counterfactual_update_count": len(full_update_records),
        "carrier_only_counterfactual_update_count": len(
            carrier_only_update_records
        ),
        "full_method_counterfactual_update_records_digest": (
            build_stable_digest(full_update_records)
        ),
        "carrier_only_counterfactual_update_records_digest": (
            build_stable_digest(carrier_only_update_records)
        ),
        "carrier_only_counterfactual_atom_content_digest": (
            build_stable_digest(carrier_only_update_records)
        ),
        "full_method_initial_latent_content_sha256": (
            full_initial_latent_sha256
        ),
        "carrier_only_initial_latent_content_sha256": (
            carrier_initial_latent_sha256
        ),
        "carrier_only_counterfactual_initial_latent_identity_ready": True,
        "carrier_only_counterfactual_scheduler_trace": list(full_trace),
        "carrier_only_counterfactual_scheduler_trace_digest": (
            build_stable_digest(full_trace)
        ),
        "carrier_only_counterfactual_scheduler_identity_ready": True,
        "carrier_only_counterfactual_attention_geometry_enabled": False,
        "full_method_counterfactual_carrier_branches": list(
            expected_full_branches
        ),
        "carrier_only_counterfactual_carrier_branches": list(
            expected_carrier_branches
        ),
        "carrier_only_counterfactual_effect_scope": (
            "attention_geometry_switch_total_mechanism_effect"
        ),
        "carrier_only_counterfactual_realized_carrier_equality_assumed": False,
        "carrier_only_counterfactual_downstream_interactions_included": True,
    }
    record["carrier_only_counterfactual_identity_digest"] = build_stable_digest(
        record
    )
    record["carrier_only_counterfactual_ready"] = True
    return record


def _resolve_injection_schedule_times(
    scheduler_timesteps: Any,
    *,
    step_index: int,
    attention_operator_schedule_index: int,
) -> tuple[int, Any, Any]:
    """分离当前写回状态与固定 Q/K 科学算子的调度时刻。"""

    if (
        type(step_index) is not int
        or type(attention_operator_schedule_index) is not int
    ):
        raise TypeError("调度索引必须为精确 int")
    post_step_schedule_index = step_index + 1
    if (
        post_step_schedule_index >= len(scheduler_timesteps)
        or attention_operator_schedule_index < 0
        or attention_operator_schedule_index >= len(scheduler_timesteps)
    ):
        raise RuntimeError("注入写回或冻结注意力算子索引超出真实调度序列")
    return (
        post_step_schedule_index,
        scheduler_timesteps[post_step_schedule_index],
        scheduler_timesteps[attention_operator_schedule_index],
    )


def _legacy_semantic_watermark_runtime(
    config: SemanticWatermarkRuntimeConfig,
    runtime_context: SemanticWatermarkRuntimeContext | None = None,
) -> tuple[
    SemanticWatermarkRuntimeResult,
    tuple[dict[str, Any], ...],
    tuple[dict[str, Any], ...],
    tuple[dict[str, Any], ...],
    Any,
    Any,
    Any | None,
    dict[str, Any],
]:
    """保留迁移审计所需的旧实现体；正式入口不得调用。"""

    if config.image_alignment_enabled and not config.attention_geometry_enabled:
        raise ValueError("图像配准必须以真实注意力几何测量为前提")

    import torch
    from main.methods.carrier.keyed_tensor import (
        build_low_frequency_template,
        build_tail_robust_template,
        project_canonical_template,
        tail_robust_carrier_protocol_record,
    )
    from main.methods.geometry.differentiable_attention import (
        optimize_attention_geometry_update,
    )
    from main.methods.semantic.branch_risk import (
        build_active_branch_risk_fields,
    )
    from experiments.runtime.diffusion.regeneration_attacks import (
        default_diffusion_attack_specs,
    )
    from main.methods.subspace.jacobian_nullspace import (
        build_exact_jacobian_linearization,
    )
    from main.methods.subspace.semantic_projection import (
        solve_semantic_branch_subspace,
    )

    started_at = time.time()
    run_id = build_semantic_watermark_run_id(config)
    context = runtime_context or load_semantic_watermark_runtime_context(config)
    pipeline = context.pipeline
    runtime_versions = context.runtime_versions
    feature_runtime = context.feature_runtime
    attention_modules = context.attention_modules
    unconditional_prompt = context.unconditional_prompt
    unconditional_pooled = context.unconditional_pooled
    diffusion_attack_runtime = context.diffusion_attack_runtime

    common_kwargs = {
        "prompt": config.prompt,
        "negative_prompt": config.negative_prompt,
        "width": config.width,
        "height": config.height,
        "num_inference_steps": config.inference_steps,
        "guidance_scale": config.guidance_scale,
        "output_type": "pil",
    }
    base_latent_shape = (
        1,
        int(pipeline.transformer.config.in_channels),
        int(config.height) // int(pipeline.vae_scale_factor),
        int(config.width) // int(pipeline.vae_scale_factor),
    )
    base_latent, base_latent_identity = build_canonical_sd35_base_latent(
        shape=base_latent_shape,
        generation_seed_random=int(config.seed),
        model_id=config.model_id,
        model_revision=config.model_revision,
        device=pipeline._execution_device,
        dtype=pipeline.transformer.dtype,
    )
    formal_randomization_identity = {
        "randomization_repeat_id": config.randomization_repeat_id,
        "generation_seed_index": int(config.generation_seed_index),
        "generation_seed_offset": int(config.generation_seed_offset),
        "watermark_key_index": int(config.watermark_key_index),
        "generation_seed_random": int(config.seed),
        "watermark_key_seed_random": int(config.watermark_key_seed_random),
        "formal_randomization_protocol_digest": (
            config.formal_randomization_protocol_digest
        ),
        "watermark_key_material_digest_random": build_stable_digest(
            {"key_material": config.key_material}
        ),
    }
    formal_randomization_identity[
        "formal_randomization_identity_digest_random"
    ] = build_stable_digest(formal_randomization_identity)
    sample_randomization_reference = formal_randomization_sample_reference(
        formal_randomization_identity,
        base_latent_identity=base_latent_identity,
    )
    attack_seed_protocol = formal_attack_seed_protocol_record()
    with torch.no_grad():
        clean_image = pipeline(
            latents=base_latent.detach().clone(),
            **common_kwargs,
        ).images[0]

    update_records: list[dict[str, Any]] = []
    active_update_records = update_records
    active_injection_config = config
    injection_execution_role = "full_method"
    previous_step_latent: Any | None = None

    def inject(pipe: Any, step_index: int, timestep: Any, callback_kwargs: dict[str, Any]) -> dict[str, Any]:
        nonlocal previous_step_latent
        latent = callback_kwargs.get("latents")
        if latent is None:
            return callback_kwargs
        if step_index not in active_injection_config.injection_step_indices:
            previous_step_latent = latent.detach().clone()
            return callback_kwargs
        if previous_step_latent is None:
            raise RuntimeError(
                "分支风险缺少紧邻上一 scheduler 步的真实 latent"
            )
        adjacent_step_reference_sha256 = tensor_content_sha256(
            previous_step_latent
        )
        scheduler_timesteps = pipe.scheduler.timesteps
        attention_operator_schedule_index = (
            active_injection_config.attention_operator_schedule_index
        )
        (
            post_step_index,
            post_step_schedule_timestep,
            attention_operator_timestep,
        ) = _resolve_injection_schedule_times(
            scheduler_timesteps,
            step_index=step_index,
            attention_operator_schedule_index=(
                attention_operator_schedule_index
            ),
        )
        with torch.enable_grad():
            active_branch_names = _active_carrier_branch_names(
                active_injection_config
            )
            active_branch_name_set = set(active_branch_names)
            branch_risk_configs = _branch_risk_configs(
                active_injection_config
            )
            active_branch_risk_configs = {
                branch_name: branch_risk_configs[branch_name]
                for branch_name in active_branch_names
            }
            effective_tail_fraction = (
                active_injection_config.tail_fraction
                if active_injection_config.tail_truncation_enabled
                else 1.0
            )
            lf_template = (
                build_low_frequency_template(
                    latent,
                    active_injection_config.key_material,
                    active_injection_config.carrier_model_reference,
                    active_injection_config.low_frequency_carrier_config,
                    prg_version=active_injection_config.keyed_prg_version,
                )
                if active_injection_config.lf_enabled
                else None
            )
            tail_template = None
            tail_threshold = 0.0
            retained_fraction = 0.0
            if active_injection_config.tail_robust_enabled:
                (
                    tail_template,
                    tail_threshold,
                    retained_fraction,
                ) = build_tail_robust_template(
                    latent,
                    active_injection_config.key_material,
                    active_injection_config.carrier_model_reference,
                    effective_tail_fraction,
                    prg_version=active_injection_config.keyed_prg_version,
                )
            tail_carrier_protocol = tail_robust_carrier_protocol_record(
                effective_tail_fraction,
                prg_version=active_injection_config.keyed_prg_version,
            )
            attention_gradient = None
            attention_stability = None
            risk_requires_attention = bool(
                active_injection_config.semantic_routing_enabled
                and any(
                    risk_config.attention_instability_weight > 0.0
                    for risk_config in active_branch_risk_configs.values()
                )
            )
            attention_work_required = bool(
                active_injection_config.attention_geometry_enabled
                or risk_requires_attention
            )
            transformer_forward = None
            if attention_work_required:
                transformer_forward = _transformer_forward_function(
                    pipeline,
                    attention_operator_timestep,
                    unconditional_prompt,
                    unconditional_pooled,
                )
                with DifferentiableAttentionRecorder(
                    attention_modules,
                    max_tokens=active_injection_config.max_attention_tokens,
                ) as recorder:
                    if active_injection_config.attention_geometry_enabled:
                        attention_gradient = compute_attention_geometry_gradient(
                            latent,
                            transformer_forward,
                            recorder,
                            active_injection_config.key_material,
                            prg_version=(
                                active_injection_config.keyed_prg_version
                            ),
                            stable_token_fraction=(
                                active_injection_config.attention_stable_token_fraction
                            ),
                            unstable_pair_weight=(
                                active_injection_config.attention_unstable_pair_weight
                            ),
                            component_weights=(
                                active_injection_config.attention_relation_component_weights
                            ),
                        )
                    elif risk_requires_attention:
                        recorder.clear()
                        with torch.no_grad():
                            transformer_forward(latent.detach().float())
                    if risk_requires_attention:
                        attention_stability = attention_relation_stability_map(
                            recorder.records,
                            tuple(int(value) for value in latent.shape[-2:]),
                        ).detach()
            risk_signal_content_records: dict[str, str] = {}
            branch_fields: dict[str, Any] = {}
            if active_injection_config.semantic_routing_enabled:
                signals = feature_runtime.branch_signal_maps(
                    latent.float(),
                    previous_step_latent.float(),
                )
                risk_signal_content_records = {
                    "current_decoded_rgb_content_sha256": tensor_content_sha256(
                        signals["current_decoded_rgb"]
                    ),
                    "previous_step_decoded_rgb_content_sha256": tensor_content_sha256(
                        signals["previous_step_decoded_rgb"]
                    ),
                    "clip_patch_tokens_content_sha256": tensor_content_sha256(
                        signals["clip_patch_tokens"]
                    ),
                    "clip_cls_token_content_sha256": tensor_content_sha256(
                        signals["clip_cls_token"]
                    ),
                    "semantic_risk_signal_content_sha256": tensor_content_sha256(
                        signals["semantic"]
                    ),
                    "texture_risk_signal_content_sha256": tensor_content_sha256(
                        signals["texture"]
                    ),
                    "local_contrast_risk_signal_content_sha256": tensor_content_sha256(
                        signals["local_contrast_risk"]
                    ),
                    "adjacent_step_stability_signal_content_sha256": tensor_content_sha256(
                        signals["adjacent_step_stability"]
                    ),
                }
                if attention_stability is not None:
                    risk_signal_content_records[
                        "attention_stability_signal_content_sha256"
                    ] = tensor_content_sha256(attention_stability)
                branch_fields = build_active_branch_risk_fields(
                    semantic_values=signals["semantic"].reshape(-1).cpu().tolist(),
                    texture_values=signals["texture"].reshape(-1).cpu().tolist(),
                    adjacent_step_stability_values=signals[
                        "adjacent_step_stability"
                    ].reshape(-1).cpu().tolist(),
                    local_contrast_risk_values=signals[
                        "local_contrast_risk"
                    ].reshape(-1).cpu().tolist(),
                    attention_stability_values=(
                        None
                        if attention_stability is None
                        else attention_stability.reshape(-1).cpu().tolist()
                    ),
                    configs=active_branch_risk_configs,
                    risk_neutral_texture_value=(
                        active_injection_config.risk_neutral_texture_value
                    ),
                    required_eligible_branches=(
                        _required_branch_risk_eligibility(
                            active_injection_config
                        )
                    ),
                )
                risk_bundle_digest = build_stable_digest(
                    {
                        branch_name: branch_field.risk_field_digest
                        for branch_name, branch_field in branch_fields.items()
                    }
                )
                effective_branch_budgets = {
                    branch_name: _branch_budget(
                        latent,
                        branch_fields[branch_name],
                        semantic_routing_enabled=True,
                        budget_ceiling=(
                            branch_risk_configs[branch_name].budget_ceiling
                        ),
                    )
                    for branch_name in active_branch_names
                }
            else:
                risk_bundle_digest = build_stable_digest(
                    {
                        "semantic_routing_enabled": False,
                        "active_carrier_branches": active_branch_names,
                    }
                )
                effective_branch_budgets = {
                    branch_name: _constant_branch_budget(
                        latent,
                        branch_risk_configs[branch_name].budget_ceiling,
                    )
                    for branch_name in active_branch_names
                }
            preferred_directions = {
                "lf_content": () if lf_template is None else (lf_template,),
                "tail_robust": () if tail_template is None else (tail_template,),
                "attention_geometry": (
                    () if attention_gradient is None else (attention_gradient.gradient,)
                ),
            }
            effective_branch_budget_content_records = {
                branch_name: tensor_content_sha256(effective_budget)
                for branch_name, effective_budget in (
                    effective_branch_budgets.items()
                )
            }
            if active_injection_config.null_space_enabled:
                linearized_latent = latent.float()
                joint_feature_linearization = build_exact_jacobian_linearization(
                    feature_runtime.full_joint_feature_vector,
                    linearized_latent,
                )
                subspaces = {
                    branch_name: solve_semantic_branch_subspace(
                        latent=linearized_latent,
                        feature_runtime=feature_runtime,
                        key_material=active_injection_config.key_material,
                        branch_name=branch_name,
                        axis_budget=effective_branch_budgets[branch_name],
                        candidate_count=active_injection_config.candidate_count,
                        null_rank=active_injection_config.null_rank,
                        joint_feature_linearization=joint_feature_linearization,
                        preferred_directions=preferred_directions[branch_name],
                        maximum_relative_response_residual=(
                            active_injection_config.maximum_relative_response_residual
                        ),
                        minimum_projection_energy_retention=(
                            active_injection_config.minimum_projection_energy_retention
                        ),
                        cg_maximum_iterations=(
                            active_injection_config.null_space_cg_max_iterations
                        ),
                        cg_relative_tolerance=(
                            active_injection_config.null_space_cg_relative_tolerance
                        ),
                        numerical_epsilon=(
                            active_injection_config.null_space_numerical_epsilon
                        ),
                        maximum_qr_condition_number=(
                            active_injection_config.maximum_qr_condition_number
                        ),
                        maximum_orthogonality_error=(
                            active_injection_config.maximum_orthogonality_error
                        ),
                        qr_reference_solve_protocol=(
                            active_injection_config.qr_reference_solve_protocol
                        ),
                        prg_version=active_injection_config.keyed_prg_version,
                    )
                    for branch_name in active_branch_names
                }
                # Null Space 基底已经物化; 立即释放 JVP/VJP 图, 为 Q/K 回溯腾出显存.
                del joint_feature_linearization
            else:
                subspaces = {
                    branch_name: _FullLatentSpace()
                    for branch_name in active_branch_names
                }
            jvp_modes = tuple(
                sorted(
                    {
                        str(result.metadata.get("jvp_mode"))
                        for result in subspaces.values()
                        if hasattr(result, "metadata") and result.metadata.get("jvp_mode")
                    }
                )
            )
            lf_carrier = (
                project_canonical_template(
                    "lf_content",
                    lf_template,
                    subspaces["lf_content"],
                    active_injection_config.minimum_projection_energy_retention,
                    carrier_protocol_digest=(
                        active_injection_config.low_frequency_carrier_config.protocol_digest
                    ),
                    prg_version=active_injection_config.keyed_prg_version,
                )
                if active_injection_config.lf_enabled
                else None
            )
            tail_carrier = (
                project_canonical_template(
                    "tail_robust",
                    tail_template,
                    subspaces["tail_robust"],
                    active_injection_config.minimum_projection_energy_retention,
                    carrier_protocol_digest=tail_carrier_protocol[
                        "tail_carrier_protocol_digest"
                    ],
                    prg_version=active_injection_config.keyed_prg_version,
                )
                if active_injection_config.tail_robust_enabled
                else None
            )
            latent_norms = torch.linalg.vector_norm(
                latent.detach().float().flatten(1),
                dim=1,
            )
            bounded_branch_updates: dict[str, Any] = {}
            post_risk_reference_directions: dict[str, Any] = {}
            if lf_carrier is not None:
                bounded_branch_updates["lf_content"] = (
                    build_risk_bounded_update(
                        branch_name="lf_content",
                        direction=lf_carrier.embedded_direction,
                        effective_budget=effective_branch_budgets["lf_content"],
                        nominal_strength=(
                            active_injection_config.lf_relative_strength
                            * latent_norms
                        ),
                        budget_ceiling=(
                            branch_risk_configs["lf_content"].budget_ceiling
                        ),
                        direction_epsilon=(
                            active_injection_config.risk_bounded_scale_direction_epsilon
                        ),
                        numerical_epsilon=(
                            active_injection_config.null_space_numerical_epsilon
                        ),
                    )
                )
                post_risk_reference_directions["lf_content"] = (
                    lf_carrier.canonical_template
                )
            if tail_carrier is not None:
                bounded_branch_updates["tail_robust"] = (
                    build_risk_bounded_update(
                        branch_name="tail_robust",
                        direction=tail_carrier.embedded_direction,
                        effective_budget=effective_branch_budgets["tail_robust"],
                        nominal_strength=(
                            active_injection_config.tail_relative_strength
                            * latent_norms
                        ),
                        budget_ceiling=(
                            branch_risk_configs["tail_robust"].budget_ceiling
                        ),
                        direction_epsilon=(
                            active_injection_config.risk_bounded_scale_direction_epsilon
                        ),
                        numerical_epsilon=(
                            active_injection_config.null_space_numerical_epsilon
                        ),
                    )
                )
                post_risk_reference_directions["tail_robust"] = (
                    tail_carrier.canonical_template
                )
            content_branch_updates = {
                branch_name: bounded_update
                for branch_name, bounded_update in bounded_branch_updates.items()
                if branch_name in {"lf_content", "tail_robust"}
            }
            content_base_candidate = build_quantized_composition_candidate(
                original_latent=latent,
                branch_updates=content_branch_updates,
                common_scale=1.0,
                backtracking_factor=(
                    active_injection_config.quantized_budget_envelope_backtracking_factor
                ),
                backtracking_step_count=0,
                absolute_tolerance=(
                    active_injection_config.quantized_budget_envelope_absolute_tolerance
                ),
            )
            attention_update = None
            if attention_gradient is not None:
                if int(latent.shape[0]) != 1:
                    raise RuntimeError("正式注意力几何更新要求单图像 batch")
                content_base_gradient = compute_attention_geometry_gradient(
                    content_base_candidate.candidate_latent,
                    transformer_forward,
                    recorder,
                    active_injection_config.key_material,
                    prg_version=(
                        active_injection_config.keyed_prg_version
                    ),
                    stable_token_fraction=(
                        attention_gradient.stable_token_fraction
                    ),
                    unstable_pair_weight=(
                        attention_gradient.unstable_pair_weight
                    ),
                    stable_token_selection=StableAttentionTokenSelection(
                        token_positions=(
                            attention_gradient.stable_token_positions
                        ),
                        token_indices=(
                            attention_gradient.stable_token_indices
                        ),
                        stable_token_fraction=(
                            attention_gradient.stable_token_fraction
                        ),
                        selection_digest=(
                            attention_gradient.stable_token_selection_digest
                        ),
                    ),
                    component_weights=(
                        active_injection_config.attention_relation_component_weights
                    ),
                )
                attention_safe_direction = subspaces[
                    "attention_geometry"
                ].project(content_base_gradient.gradient.float())
                maximum_attention_bound = build_risk_bounded_update(
                    branch_name="attention_geometry",
                    direction=attention_safe_direction,
                    effective_budget=effective_branch_budgets[
                        "attention_geometry"
                    ],
                    nominal_strength=(
                        active_injection_config.attention_relative_strength
                        * latent_norms
                    ),
                    budget_ceiling=(
                        branch_risk_configs[
                            "attention_geometry"
                        ].budget_ceiling
                    ),
                    direction_epsilon=(
                        active_injection_config.risk_bounded_scale_direction_epsilon
                    ),
                    numerical_epsilon=(
                        active_injection_config.null_space_numerical_epsilon
                    ),
                )
                attention_update = optimize_attention_geometry_update(
                    latent=latent,
                    transformer_forward=transformer_forward,
                    recorder=recorder,
                    key_material=active_injection_config.key_material,
                    safe_subspace=subspaces["attention_geometry"],
                    risk_bounded_update=maximum_attention_bound,
                    backtracking_factor=(
                        active_injection_config.attention_backtracking_factor
                    ),
                    maximum_backtracking_steps=(
                        active_injection_config.attention_backtracking_maximum_steps
                    ),
                    precomputed_gradient=attention_gradient,
                    precomputed_content_base_gradient=(
                        content_base_gradient
                    ),
                    prg_version=(
                        active_injection_config.keyed_prg_version
                    ),
                    base_update=(
                        content_base_candidate.float32_combined_update
                    ),
                    stable_token_fraction=(
                        active_injection_config.attention_stable_token_fraction
                    ),
                    unstable_pair_weight=(
                        active_injection_config.attention_unstable_pair_weight
                    ),
                    component_weights=(
                        active_injection_config.attention_relation_component_weights
                    ),
                )
                if attention_update.unit_update_content_sha256 != (
                    tensor_content_sha256(
                        maximum_attention_bound.unit_direction
                    )
                ):
                    raise RuntimeError(
                        "注意力单调回溯与风险包络没有复用同一单位方向 Tensor"
                    )
                bounded_branch_updates["attention_geometry"] = (
                    rescale_risk_bounded_update(
                        maximum_attention_bound,
                        attention_update.applied_update_strength,
                    )
                )
                if attention_update.update_content_sha256 != (
                    tensor_content_sha256(
                        bounded_branch_updates[
                            "attention_geometry"
                        ].update
                    )
                ):
                    raise RuntimeError(
                        "attention 接受候选与风险有界分支写回不是同一 update Tensor"
                    )
                post_risk_reference_directions[
                    "attention_geometry"
                ] = content_base_gradient.gradient

            post_risk_direction_records = {
                branch_name: _post_risk_direction_jacobian_record(
                    feature_runtime.full_joint_feature_vector,
                    latent,
                    branch_name,
                    bounded_update.unit_direction,
                    post_risk_reference_directions[branch_name],
                    active_injection_config.maximum_relative_response_residual,
                    active_injection_config.null_space_numerical_epsilon,
                )
                for branch_name, bounded_update in (
                    bounded_branch_updates.items()
                )
            } if active_injection_config.null_space_enabled else {}

            accepted_composition = None
            quantized_write_jacobian_record = None
            preservation_record = None
            accepted_attention_record = None
            for quantized_step, composition_candidate in enumerate(
                iter_quantized_composition_candidates(
                    original_latent=latent,
                    branch_updates=bounded_branch_updates,
                    backtracking_factor=(
                        active_injection_config.quantized_budget_envelope_backtracking_factor
                    ),
                    maximum_steps=(
                        active_injection_config.quantized_budget_envelope_backtracking_maximum_steps
                    ),
                    absolute_tolerance=(
                        active_injection_config.quantized_budget_envelope_absolute_tolerance
                    ),
                )
            ):
                if not composition_candidate.envelope_ready:
                    continue
                candidate_jacobian_record = (
                    _quantized_write_jacobian_response_record(
                        (
                            feature_runtime.full_joint_feature_vector
                            if active_injection_config.null_space_enabled
                            else None
                        ),
                        latent,
                        composition_candidate.candidate_latent,
                        active_injection_config.maximum_quantized_write_relative_jacobian_response,
                        active_injection_config.null_space_numerical_epsilon,
                    )
                )
                if not _quantized_write_update_nonzero(
                    candidate_jacobian_record
                ):
                    continue
                if (
                    active_injection_config.null_space_enabled
                    and not candidate_jacobian_record[
                        "quantized_write_jacobian_gate_ready"
                    ]
                ):
                    continue
                candidate_preservation_record = (
                    _combined_update_preservation_record(
                        feature_runtime,
                        latent,
                        composition_candidate.candidate_latent,
                        config,
                    )
                )
                if (
                    active_injection_config.null_space_enabled
                    and not candidate_preservation_record[
                        "semantic_preservation_gate_ready"
                    ]
                ):
                    continue

                candidate_attention_record = None
                if attention_update is not None:
                    content_candidate_at_scale = (
                        build_quantized_composition_candidate(
                            original_latent=latent,
                            branch_updates=content_branch_updates,
                            common_scale=composition_candidate.common_scale,
                            backtracking_factor=(
                                active_injection_config.quantized_budget_envelope_backtracking_factor
                            ),
                            backtracking_step_count=quantized_step,
                            absolute_tolerance=(
                                active_injection_config.quantized_budget_envelope_absolute_tolerance
                            ),
                        )
                    )
                    if not content_candidate_at_scale.envelope_ready:
                        continue
                    recorder.clear()
                    with torch.no_grad():
                        transformer_forward(
                            content_candidate_at_scale.candidate_latent.detach().float()
                        )
                        content_score_tensor = attention_geometry_score(
                            recorder.records,
                            active_injection_config.key_material,
                            prg_version=(
                                active_injection_config.keyed_prg_version
                            ),
                            stable_pair_weights=(
                                attention_gradient.stable_pair_weights
                            ),
                            component_weights=(
                                active_injection_config.attention_relation_component_weights
                            ),
                        )
                        written_content_qk_identity = (
                            build_attention_relation_graph_identity(
                                recorder.records,
                                active_injection_config.key_material,
                                prg_version=(
                                    active_injection_config.keyed_prg_version
                                ),
                                component_weights=(
                                    active_injection_config.attention_relation_component_weights
                                ),
                            )
                        )
                        recorder.clear()
                        transformer_forward(
                            composition_candidate.candidate_latent.detach().float()
                        )
                        final_score_tensor = attention_geometry_score(
                            recorder.records,
                            active_injection_config.key_material,
                            prg_version=(
                                active_injection_config.keyed_prg_version
                            ),
                            stable_pair_weights=(
                                attention_gradient.stable_pair_weights
                            ),
                            component_weights=(
                                active_injection_config.attention_relation_component_weights
                            ),
                        )
                        written_qk_identity = (
                            build_attention_relation_graph_identity(
                                recorder.records,
                                active_injection_config.key_material,
                                prg_version=(
                                    active_injection_config.keyed_prg_version
                                ),
                                component_weights=(
                                    active_injection_config.attention_relation_component_weights
                                ),
                            )
                        )
                    content_score = float(
                        content_score_tensor.detach().item()
                    )
                    final_score = float(final_score_tensor.detach().item())
                    required_score = max(
                        attention_update.score_before,
                        content_score,
                    )
                    if (
                        not math.isfinite(content_score)
                        or not math.isfinite(final_score)
                        or final_score <= required_score
                    ):
                        continue
                    if (
                        not written_content_qk_identity.qk_atomic_content_ready
                        or not written_qk_identity.qk_atomic_content_ready
                    ):
                        continue
                    if (
                        written_qk_identity.component_protocol_digest
                        != attention_update.attention_relation_component_protocol_digest
                        or written_content_qk_identity.component_protocol_digest
                        != attention_update.attention_relation_component_protocol_digest
                        or written_qk_identity.component_weights
                        != attention_update.attention_relation_component_weights
                        or written_content_qk_identity.component_weights
                        != attention_update.attention_relation_component_weights
                        or written_qk_identity.component_identity_digest
                        != attention_update.attention_relation_component_identity_digest
                        or written_content_qk_identity.component_identity_digest
                        != attention_update.attention_relation_component_identity_digest
                        or written_qk_identity.keyed_projection_digest
                        != attention_update.attention_relation_keyed_projection_digest
                        or written_content_qk_identity.keyed_projection_digest
                        != attention_update.attention_relation_keyed_projection_digest
                        or written_qk_identity.qk_operator_metadata_digest
                        != attention_update.attention_relation_qk_operator_metadata_digest
                        or written_content_qk_identity.qk_operator_metadata_digest
                        != attention_update.attention_relation_qk_operator_metadata_digest
                    ):
                        continue
                    qk_atomic_evaluation_records = (
                        attention_update.qk_atomic_evaluation_records[0],
                        attention_update.qk_atomic_evaluation_records[1],
                        attention_update.qk_atomic_evaluation_records[2],
                        {
                            "qk_evaluation_role": (
                                "actual_written_content_base_latent"
                            ),
                            "evaluation_latent_content_sha256": (
                                tensor_content_sha256(
                                    content_candidate_at_scale.candidate_latent.detach().float()
                                )
                            ),
                            "evaluation_score": content_score,
                            "qk_atomic_content_records": list(
                                written_content_qk_identity.qk_atomic_content_records
                            ),
                            "qk_atomic_content_digest": (
                                written_content_qk_identity.qk_atomic_content_digest
                            ),
                            "qk_atomic_content_ready": (
                                written_content_qk_identity.qk_atomic_content_ready
                            ),
                        },
                        {
                            "qk_evaluation_role": "actual_written_combined_latent",
                            "evaluation_latent_content_sha256": (
                                tensor_content_sha256(
                                    composition_candidate.candidate_latent.detach().float()
                                )
                            ),
                            "evaluation_score": final_score,
                            "qk_atomic_content_records": list(
                                written_qk_identity.qk_atomic_content_records
                            ),
                            "qk_atomic_content_digest": (
                                written_qk_identity.qk_atomic_content_digest
                            ),
                            "qk_atomic_content_ready": (
                                written_qk_identity.qk_atomic_content_ready
                            ),
                        },
                    )
                    qk_evaluation_digest = (
                        qk_atomic_evaluation_records_digest(
                            qk_atomic_evaluation_records,
                            "attention_qk_atomic_content_records",
                        )
                    )
                    qk_evaluation_ready = (
                        qk_atomic_evaluation_records_ready(
                            qk_atomic_evaluation_records,
                            qk_evaluation_digest,
                            aggregate_field_name=(
                                "attention_qk_atomic_content_records"
                            ),
                            expected_roles=(
                                "latent_before",
                                "optimization_content_base_latent",
                                "accepted_attention_candidate",
                                "actual_written_content_base_latent",
                                "actual_written_combined_latent",
                            ),
                            expected_layer_names=(
                                active_injection_config.attention_module_names
                            ),
                            require_evaluation_identity=True,
                        )
                    )
                    if not qk_evaluation_ready:
                        continue
                    candidate_attention_record = {
                        "attention_score_before": attention_update.score_before,
                        "attention_content_base_score": (
                            attention_update.content_base_score
                        ),
                        "attention_score_after": attention_update.score_after,
                        "attention_actual_written_content_base_score": (
                            content_score
                        ),
                        "attention_final_combined_score": final_score,
                        "attention_score_gain": (
                            final_score - attention_update.score_before
                        ),
                        "attention_applied_update_strength": (
                            attention_update.applied_update_strength
                        ),
                        "attention_backtracking_step_count": (
                            attention_update.backtracking_step_count
                        ),
                        "attention_update_digest": attention_update.update_digest,
                        "attention_update_content_sha256": (
                            attention_update.update_content_sha256
                        ),
                        "attention_update_unit_direction_content_sha256": (
                            attention_update.unit_update_content_sha256
                        ),
                        "stable_token_indices": list(
                            attention_update.stable_token_indices
                        ),
                        "stable_token_selection_digest": (
                            attention_update.stable_token_selection_digest
                        ),
                        "stable_pair_weight_identity_digest": (
                            attention_update.stable_pair_weight_identity_digest
                        ),
                        "stable_pair_weight_realization_digest": (
                            attention_update.stable_pair_weight_realization_digest
                        ),
                        "attention_relation_component_names": list(
                            attention_update.attention_relation_component_names
                        ),
                        "attention_relation_active_component_names": list(
                            attention_update.attention_relation_active_component_names
                        ),
                        "attention_relation_component_weights": list(
                            attention_update.attention_relation_component_weights
                        ),
                        "attention_relation_component_protocol_digest": (
                            attention_update.attention_relation_component_protocol_digest
                        ),
                        "attention_relation_source": (
                            attention_update.attention_relation_source
                        ),
                        "attention_relation_direct_qk_source_ready": (
                            attention_update.attention_relation_source
                            == DIRECT_QK_RELATION_SOURCE
                        ),
                        "attention_relation_probability_scope": (
                            "sampled_image_token_qk_relation_probability"
                        ),
                        "attention_relation_component_identity_digest": (
                            attention_update.attention_relation_component_identity_digest
                        ),
                        "attention_relation_keyed_projection_digest": (
                            attention_update.attention_relation_keyed_projection_digest
                        ),
                        "attention_relation_qk_operator_metadata_records": list(
                            attention_update.attention_relation_qk_operator_metadata_records
                        ),
                        "attention_relation_qk_operator_metadata_digest": (
                            attention_update.attention_relation_qk_operator_metadata_digest
                        ),
                        "attention_relation_qk_operator_metadata_ready": (
                            attention_update.attention_relation_qk_operator_metadata_ready
                        ),
                        "attention_qk_atomic_content_records": list(
                            qk_atomic_evaluation_records
                        ),
                        "attention_qk_atomic_content_digest": (
                            qk_evaluation_digest
                        ),
                        "attention_qk_atomic_content_ready": True,
                    }

                composition_record = composition_candidate.to_record()
                for shared_field in (
                    "quantized_write_update_content_sha256",
                    "quantized_write_update_dtype",
                    "quantized_write_update_shape",
                ):
                    if candidate_jacobian_record[shared_field] != (
                        composition_record[shared_field]
                    ):
                        raise RuntimeError(
                            "实际 JVP 写回记录与唯一量化合成记录的 Tensor 身份不一致: "
                            f"{shared_field}"
                        )
                accepted_composition = composition_candidate
                quantized_write_jacobian_record = {
                    **candidate_jacobian_record,
                    **composition_record,
                }
                preservation_record = candidate_preservation_record
                accepted_attention_record = candidate_attention_record
                break

            if (
                accepted_composition is None
                or quantized_write_jacobian_record is None
                or preservation_record is None
            ):
                raise RuntimeError(
                    "共同缩放候选未同时通过预算、JVP、有限特征和 Q/K 门禁"
                )
            injected = accepted_composition.candidate_latent
            combined_update = accepted_composition.float32_combined_update
            lf_update = (
                bounded_branch_updates["lf_content"].update
                if "lf_content" in bounded_branch_updates
                else torch.zeros_like(latent, dtype=torch.float32)
            )
            tail_update = (
                bounded_branch_updates["tail_robust"].update
                if "tail_robust" in bounded_branch_updates
                else torch.zeros_like(latent, dtype=torch.float32)
            )
            attention_tensor = (
                bounded_branch_updates["attention_geometry"].update
                if "attention_geometry" in bounded_branch_updates
                else torch.zeros_like(latent, dtype=torch.float32)
            )
            attention_record = (
                accepted_attention_record
                if accepted_attention_record is not None
                else {
                    "attention_score_before": None,
                    "attention_content_base_score": None,
                    "attention_score_after": None,
                    "attention_actual_written_content_base_score": None,
                    "attention_final_combined_score": None,
                    "attention_score_gain": None,
                    "attention_applied_update_strength": None,
                    "attention_backtracking_step_count": None,
                    "attention_update_digest": "",
                    "attention_update_content_sha256": "",
                    "attention_update_unit_direction_content_sha256": "",
                    "stable_token_indices": [],
                    "stable_token_selection_digest": "",
                    "stable_pair_weight_identity_digest": "",
                    "stable_pair_weight_realization_digest": "",
                    "attention_relation_component_names": [],
                    "attention_relation_active_component_names": [],
                    "attention_relation_component_weights": [],
                    "attention_relation_component_protocol_digest": "",
                    "attention_relation_source": "",
                    "attention_relation_direct_qk_source_ready": False,
                    "attention_relation_probability_scope": "",
                    "attention_relation_component_identity_digest": "",
                    "attention_relation_keyed_projection_digest": "",
                    "attention_relation_qk_operator_metadata_records": [],
                    "attention_relation_qk_operator_metadata_digest": "",
                    "attention_relation_qk_operator_metadata_ready": False,
                    "attention_qk_atomic_content_records": [],
                    "attention_qk_atomic_content_digest": "",
                    "attention_qk_atomic_content_ready": False,
                }
            )
            bounded_branch_update_records = {
                branch_name: {
                    **bounded_update.to_record(),
                    **post_risk_direction_records.get(branch_name, {}),
                }
                for branch_name, bounded_update in (
                    bounded_branch_updates.items()
                )
            }
        branch_update_content_records = {
            "lf_content": (
                tensor_content_sha256(lf_update)
                if "lf_content" in active_branch_name_set
                else ""
            ),
            "tail_robust": (
                tensor_content_sha256(tail_update)
                if "tail_robust" in active_branch_name_set
                else ""
            ),
            "attention_geometry": (
                tensor_content_sha256(attention_tensor)
                if "attention_geometry" in active_branch_name_set
                else ""
            ),
        }
        branch_risk_records = {
            name: _branch_risk_record(branch_field)
            for name, branch_field in branch_fields.items()
        }
        for branch_name, bounded_record in (
            bounded_branch_update_records.items()
        ):
            if bounded_record[
                "effective_budget_values_content_sha256"
            ] != effective_branch_budget_content_records[branch_name]:
                raise RuntimeError(
                    "Null Space 与风险写回没有复用同一有效预算 Tensor"
                )
            if active_injection_config.semantic_routing_enabled:
                branch_risk_records[branch_name].update(bounded_record)
        branch_risk_content_evidence = _branch_risk_content_evidence(
            risk_signal_content_records,
            branch_risk_records,
            semantic_routing_enabled=(
                active_injection_config.semantic_routing_enabled
            ),
            null_space_enabled=active_injection_config.null_space_enabled,
            active_branch_names=active_branch_names,
        )
        combined_update_content_sha256 = tensor_content_sha256(
            combined_update
        )
        if quantized_write_jacobian_record[
            "combined_update_content_sha256"
        ] != combined_update_content_sha256:
            raise RuntimeError(
                "最终分支更新与量化合成记录的 combined update 身份不一致"
            )
        if quantized_write_jacobian_record[
            "tensor_content_digest_version"
        ] != TENSOR_CONTENT_DIGEST_VERSION:
            raise RuntimeError("量化合成记录使用了错误的 Tensor 摘要版本")
        active_update_records.append(
            {
                "run_id": run_id,
                "prompt_id": active_injection_config.prompt_id,
                "split": active_injection_config.split,
                **sample_randomization_reference,
                "step_index": int(step_index),
                "scheduler_step_timestep": float(timestep.detach().float().item()),
                "post_step_schedule_index": int(post_step_index),
                "post_step_schedule_timestep": float(
                    post_step_schedule_timestep.detach().float().item()
                ),
                "attention_operator_schedule_index": int(
                    attention_operator_schedule_index
                ),
                "attention_operator_timestep": float(
                    attention_operator_timestep.detach().float().item()
                ),
                "adjacent_step_reference_index": int(step_index - 1),
                "adjacent_step_reference_latent_content_sha256": (
                    adjacent_step_reference_sha256
                ),
                "adjacent_step_stability_status": (
                    "measured_from_immediately_previous_scheduler_step"
                    if active_injection_config.semantic_routing_enabled
                    else "not_applicable_semantic_routing_disabled"
                ),
                "latent_content_sha256_before": tensor_content_sha256(latent),
                "latent_content_sha256_after": tensor_content_sha256(injected),
                "combined_update_content_sha256": (
                    combined_update_content_sha256
                ),
                "lf_update_content_sha256": (
                    branch_update_content_records["lf_content"]
                ),
                "tail_robust_update_content_sha256": (
                    branch_update_content_records["tail_robust"]
                ),
                "attention_geometry_update_content_sha256": (
                    branch_update_content_records["attention_geometry"]
                ),
                "branch_updates_content_digest": build_stable_digest(
                    branch_update_content_records
                ),
                "tensor_content_digest_version": (
                    TENSOR_CONTENT_DIGEST_VERSION
                ),
                "relative_update_norm": tensor_norm(combined_update) / max(tensor_norm(latent), 1e-12),
                "active_carrier_branches": list(active_branch_names),
                "branch_risk_bundle_digest": risk_bundle_digest,
                **risk_signal_content_records,
                "branch_risk_records": branch_risk_records,
                "branch_risk_content_digest": build_stable_digest(
                    branch_risk_content_evidence
                ),
                "null_space_records": {
                    name: (
                        result.to_record()
                        if hasattr(result, "to_record")
                        else {"branch_name": name, "solver": result.solver_digest}
                    )
                    for name, result in subspaces.items()
                },
                "lf_projection_energy_retention": (
                    None if lf_carrier is None else lf_carrier.projection_energy_retention
                ),
                "lf_carrier_protocol_digest": (
                    active_injection_config.low_frequency_carrier_config.protocol_digest
                ),
                "lf_template_content_sha256": (
                    tensor_content_sha256(lf_template)
                    if lf_template is not None
                    else ""
                ),
                "lf_template_digest": (
                    "" if lf_carrier is None else lf_carrier.template_digest
                ),
                "lf_template_shape": (
                    [int(value) for value in lf_template.shape]
                    if lf_template is not None
                    else []
                ),
                "tail_template_digest": (
                    ""
                    if tail_carrier is None
                    else tail_carrier.template_digest
                ),
                "tail_carrier_protocol_digest": tail_carrier_protocol[
                    "tail_carrier_protocol_digest"
                ],
                "tail_fraction": effective_tail_fraction,
                "tail_template_content_sha256": (
                    tensor_content_sha256(tail_template)
                    if tail_template is not None
                    else ""
                ),
                "tail_template_shape": [
                    int(value) for value in tail_template.shape
                ] if tail_template is not None else [],
                "tail_template_element_count": (
                    int(tail_template.numel())
                    if tail_template is not None
                    else 0
                ),
                "tail_selected_element_count": (
                    int(
                        math.ceil(
                            tail_template.numel() * effective_tail_fraction
                        )
                    )
                    if tail_template is not None
                    else 0
                ),
                "tail_projection_energy_retention": (
                    None if tail_carrier is None else tail_carrier.projection_energy_retention
                ),
                "tail_threshold": tail_threshold,
                "tail_retained_fraction": retained_fraction,
                "keyed_prg_version": active_injection_config.keyed_prg_version,
                "keyed_prg_protocol_digest": keyed_prg_protocol_record(
                    active_injection_config.keyed_prg_version
                )["keyed_prg_protocol_digest"],
                "attention_module_names": list(
                    active_injection_config.attention_module_names
                ),
                "attention_coordinate_convention": (
                    active_injection_config.attention_coordinate_convention
                ),
                "attention_grid_align_corners": (
                    active_injection_config.attention_grid_align_corners
                ),
                **attention_record,
                **quantized_write_jacobian_record,
                **preservation_record,
                "metadata": {
                    "jvp_mode": jvp_modes[0] if len(jvp_modes) == 1 else "disabled_or_mixed",
                    "jvp_modes": list(jvp_modes),
                    "basis_solver": "matrix_free_full_jacobian_psd_cg",
                    "attention_source": (
                        "real_qk_projection"
                        if active_injection_config.attention_geometry_enabled
                        else "disabled_attention_geometry"
                    ),
                    "detector_requires_generation_trace": False,
                    "semantic_routing_enabled": active_injection_config.semantic_routing_enabled,
                    "branch_risk_mode": active_injection_config.branch_risk_mode,
                    "null_space_enabled": active_injection_config.null_space_enabled,
                    "lf_enabled": active_injection_config.lf_enabled,
                    "tail_robust_enabled": active_injection_config.tail_robust_enabled,
                    "tail_truncation_enabled": active_injection_config.tail_truncation_enabled,
                    "attention_geometry_enabled": active_injection_config.attention_geometry_enabled,
                    "injection_execution_role": injection_execution_role,
                    "supports_paper_claim": False,
                },
            }
        )
        previous_step_latent = injected.detach().clone()
        callback_kwargs["latents"] = injected.detach().to(dtype=latent.dtype)
        return callback_kwargs

    watermarked_image = pipeline(
        latents=base_latent.detach().clone(),
        callback_on_step_end=inject,
        callback_on_step_end_tensor_inputs=["latents"],
        **common_kwargs,
    ).images[0]
    carrier_only_image: Any | None = None
    carrier_only_counterfactual: dict[str, Any] | None = None
    carrier_only_final_image_preservation: dict[str, Any] | None = None
    final_image_preservation: dict[str, Any] | None = None
    carrier_only_update_records: list[dict[str, Any]] = []
    if config.attention_geometry_enabled:
        carrier_only_config = replace(
            config,
            attention_geometry_enabled=False,
        )
        active_update_records = carrier_only_update_records
        active_injection_config = carrier_only_config
        injection_execution_role = "carrier_only_counterfactual"
        previous_step_latent = None
        carrier_only_image = pipeline(
            latents=base_latent.detach().clone(),
            callback_on_step_end=inject,
            callback_on_step_end_tensor_inputs=["latents"],
            **common_kwargs,
        ).images[0]
        carrier_only_counterfactual = _carrier_only_counterfactual_identity(
            config,
            carrier_only_config,
            update_records,
            carrier_only_update_records,
        )
        active_update_records = update_records
        active_injection_config = config
        injection_execution_role = "full_method"
        previous_step_latent = None
        (
            final_image_preservation,
            carrier_only_final_image_preservation,
        ) = _three_way_final_image_preservation_records(
            pipeline,
            feature_runtime,
            clean_image,
            carrier_only_image,
            watermarked_image,
            config,
            carrier_only_counterfactual,
        )
        if not carrier_only_final_image_preservation[
            "carrier_only_counterfactual_three_way_preservation_gate_ready"
        ]:
            raise RuntimeError(
                "最终 clean、carrier-only 与完整方法成图未通过三边特征保持门禁"
            )
    attention_extractor = (
        _image_attention_extractor(
            pipeline,
            config,
            attention_modules,
            unconditional_prompt,
            unconditional_pooled,
        )
        if config.attention_geometry_enabled
        else None
    )
    final_image_attention_observability = (
        _final_image_attention_observability_record(
            attention_extractor,
            clean_image,
            carrier_only_image,
            watermarked_image,
            config,
            carrier_only_counterfactual=carrier_only_counterfactual,
            require_gpu_execution=True,
        )
    )
    if (
        config.attention_geometry_enabled
        and not final_image_attention_observability[
            "final_image_attention_observability_gate_ready"
        ]
    ):
        raise RuntimeError(
            "最终 carrier-only/完整方法成图未通过真实 Q/K 双归因门禁"
        )
    if final_image_preservation is None:
        final_image_preservation = _final_image_preservation_record(
            pipeline,
            feature_runtime,
            clean_image,
            watermarked_image,
            config,
        )
    if (
        config.null_space_enabled
        and not final_image_preservation["final_image_preservation_gate_ready"]
    ):
        raise RuntimeError("最终 clean/watermarked 成图未通过累计完整特征保持门禁")
    paired_quality = compute_image_quality_metrics(clean_image, watermarked_image)

    measurement_config = _build_image_only_measurement_config(config)
    measurement_config_identity = image_only_measurement_config_identity_record(
        measurement_config,
        attention_geometry_enabled=config.attention_geometry_enabled,
        image_alignment_enabled=config.image_alignment_enabled,
    )
    attack_threshold_protocol = (
        None
        if config.detector_guided_attack_threshold_protocol is None
        else FrozenEvidenceProtocol(
            **config.detector_guided_attack_threshold_protocol
        )
    )
    if (
        attack_threshold_protocol is not None
        and attack_threshold_protocol.image_only_measurement_config_digest
        != measurement_config_identity["image_only_measurement_config_digest"]
    ):
        raise RuntimeError(
            "detector-guided attack 协议与当前图像测量配置身份不一致"
        )

    def adversarial_detection_score(candidate: Any) -> float:
        """返回与最终内容主判和几何对齐救回一致的连续攻击目标。"""

        noise_evidence_cursor = _public_detection_noise_evidence_cursor(
            attention_extractor
        )
        evaluated = measure_image_only_watermark(
            image=candidate,
            key_material=config.key_material,
            config=measurement_config,
            image_latent_encoder=lambda image: _encode_image_latent(pipeline, image),
            image_attention_extractor=(
                attention_extractor if config.attention_geometry_enabled else None
            ),
            image_aligner=_align_image if config.image_alignment_enabled else None,
        )
        _discard_public_detection_noise_evidence_since(
            attention_extractor,
            noise_evidence_cursor,
        )
        if attack_threshold_protocol is None:
            raise RuntimeError(
                "detector-guided attack 不得使用未冻结的临时检测器"
            )
        return decision_equivalent_score(
            evaluated.to_record(),
            geometry_rescue_enabled=(
                attack_threshold_protocol.geometry_rescue_enabled
            ),
            rescue_margin_low=attack_threshold_protocol.rescue_margin_low,
            geometry_score_threshold=(
                attack_threshold_protocol.geometry_score_threshold
            ),
            registration_confidence_threshold=(
                attack_threshold_protocol.registration_confidence_threshold
            ),
            attention_sync_score_threshold=(
                attack_threshold_protocol.attention_sync_score_threshold
            ),
        )

    detection_key_plan = build_detection_key_plan_record(
        config.key_material
    )
    detections = []
    for sample_role, image, detection_key_role in (
        (
            "clean_negative",
            clean_image,
            REGISTERED_WATERMARK_KEY_ROLE,
        ),
        (
            "positive_source",
            watermarked_image,
            REGISTERED_WATERMARK_KEY_ROLE,
        ),
        (
            "wrong_key_negative",
            watermarked_image,
            REGISTERED_WRONG_KEY_ROLE,
        ),
    ):
        detection_key, detection_key_identity = (
            resolve_detection_key_material_and_identity(
                config.key_material,
                detection_key_role,
            )
        )
        noise_evidence_cursor = _public_detection_noise_evidence_cursor(
            attention_extractor
        )
        detection = measure_image_only_watermark(
            image=image,
            key_material=detection_key,
            config=measurement_config,
            image_latent_encoder=lambda candidate: _encode_image_latent(pipeline, candidate),
            image_attention_extractor=attention_extractor if config.attention_geometry_enabled else None,
            image_aligner=_align_image if config.image_alignment_enabled else None,
        )
        record = detection.to_record()
        if config.attention_geometry_enabled:
            _bind_public_detection_noise_qk_evidence(
                record,
                attention_extractor,
                noise_evidence_cursor,
            )
        record["run_id"] = run_id
        record["prompt_id"] = config.prompt_id
        record["split"] = config.split
        record["sample_role"] = sample_role
        record.update(detection_key_identity)
        record["embedding_pair_ssim"] = float(paired_quality["ssim"])
        record.update(sample_randomization_reference)
        record["metadata"] = {
            **record["metadata"],
            "supports_paper_claim": False,
            "measurement_status": "threshold_independent_image_only_evidence",
        }
        detections.append(record)

    _, registered_detection_key_identity = (
        resolve_detection_key_material_and_identity(
            config.key_material,
            REGISTERED_WATERMARK_KEY_ROLE,
        )
    )
    attacked_images: dict[str, Any] = {}
    attack_configs = tuple(
        attack
        for attack in default_attack_configs()
        if attack.enabled
        and not attack.requires_gpu
        and attack.resource_profile in set(config.standard_attack_profiles)
    )
    for sample_role, source_image in (("clean_negative", clean_image), ("positive_source", watermarked_image)):
        for attack_config in attack_configs:
            attack_seed_random = formal_attack_seed_random(
                int(config.seed),
                attack_config.attack_id,
            )
            attacked_image = apply_standard_image_attack(
                source_image,
                attack_config,
                seed=attack_seed_random,
            )
            image_key = f"{sample_role}_{attack_config.attack_id}"
            attacked_images[image_key] = attacked_image
            noise_evidence_cursor = _public_detection_noise_evidence_cursor(
                attention_extractor
            )
            detection = measure_image_only_watermark(
                image=attacked_image,
                key_material=config.key_material,
                config=measurement_config,
                image_latent_encoder=lambda candidate: _encode_image_latent(pipeline, candidate),
                image_attention_extractor=attention_extractor if config.attention_geometry_enabled else None,
                image_aligner=_align_image if config.image_alignment_enabled else None,
            )
            record = detection.to_record()
            if config.attention_geometry_enabled:
                _bind_public_detection_noise_qk_evidence(
                    record,
                    attention_extractor,
                    noise_evidence_cursor,
                )
            record.update(
                {
                    "run_id": run_id,
                    "prompt_id": config.prompt_id,
                    "split": config.split,
                    "sample_role": sample_role,
                    **registered_detection_key_identity,
                    "embedding_pair_ssim": float(paired_quality["ssim"]),
                    **sample_randomization_reference,
                    "attack_id": attack_config.attack_id,
                    "attack_family": attack_config.attack_family,
                    "attack_name": attack_config.attack_name,
                    "resource_profile": attack_config.resource_profile,
                    "attack_config_digest": attack_config_digest(attack_config),
                    "attack_seed_random": attack_seed_random,
                    "formal_attack_seed_protocol_digest": (
                        attack_seed_protocol[
                            "formal_attack_seed_protocol_digest"
                        ]
                    ),
                    "attack_parameters": attack_config.attack_parameters,
                    "attack_performed": True,
                    "attacked_image_key": image_key,
                }
            )
            record["metadata"] = {
                **record["metadata"],
                "metric_status": "measured_from_real_attacked_image",
                "supports_paper_claim": False,
                "measurement_status": "threshold_independent_image_only_evidence",
            }
            detections.append(record)

    if config.diffusion_attacks_enabled:
        if diffusion_attack_runtime is None:
            raise RuntimeError("diffusion_attacks_enabled 要求共享再扩散攻击运行时")
        if attack_threshold_protocol is None:
            raise RuntimeError(
                "detector-guided 再扩散攻击必须在 calibration 协议冻结后运行"
            )
        formal_attack_configs_by_id = {
            attack.attack_id: attack for attack in default_attack_configs()
        }
        for sample_role, source_image in (("clean_negative", clean_image), ("positive_source", watermarked_image)):
            for attack_spec in default_diffusion_attack_specs():
                attack_seed_random = formal_attack_seed_random(
                    int(config.seed),
                    attack_spec.attack_id,
                )
                attack_execution = diffusion_attack_runtime.apply(
                    source_image,
                    attack_spec,
                    seed=attack_seed_random,
                    prompt_text=config.prompt,
                    detection_score=adversarial_detection_score,
                )
                attacked_image = attack_execution.image
                image_key = f"{sample_role}_{attack_spec.attack_id}"
                attacked_images[image_key] = attacked_image
                noise_evidence_cursor = _public_detection_noise_evidence_cursor(
                    attention_extractor
                )
                detection = measure_image_only_watermark(
                    image=attacked_image,
                    key_material=config.key_material,
                    config=measurement_config,
                    image_latent_encoder=lambda candidate: _encode_image_latent(pipeline, candidate),
                    image_attention_extractor=attention_extractor if config.attention_geometry_enabled else None,
                    image_aligner=_align_image if config.image_alignment_enabled else None,
                )
                record = detection.to_record()
                if config.attention_geometry_enabled:
                    _bind_public_detection_noise_qk_evidence(
                        record,
                        attention_extractor,
                        noise_evidence_cursor,
                    )
                record.update(
                    {
                        "run_id": run_id,
                        "prompt_id": config.prompt_id,
                        "split": config.split,
                        "sample_role": sample_role,
                        **registered_detection_key_identity,
                        "embedding_pair_ssim": float(paired_quality["ssim"]),
                        **sample_randomization_reference,
                        "attack_id": attack_spec.attack_id,
                        "attack_family": attack_spec.attack_family,
                        "attack_name": attack_spec.attack_name,
                        "resource_profile": "full_extra",
                        "attack_config_digest": attack_config_digest(
                            formal_attack_configs_by_id[attack_spec.attack_id]
                        ),
                        "attack_seed_random": attack_seed_random,
                        "formal_attack_seed_protocol_digest": (
                            attack_seed_protocol[
                                "formal_attack_seed_protocol_digest"
                            ]
                        ),
                        "attack_parameters": attack_spec.attack_parameters,
                        "attack_implementation": attack_spec.attack_implementation,
                        "attack_execution": attack_execution.to_record(),
                        "attack_performed": True,
                        "attacked_image_key": image_key,
                        "detector_guided_attack_threshold_digest": (
                            attack_threshold_protocol.threshold_digest
                        ),
                    }
                )
                record["metadata"] = {
                    **record["metadata"],
                    "metric_status": "measured_from_real_diffusion_attacked_image",
                    "supports_paper_claim": False,
                    "measurement_status": "threshold_independent_image_only_evidence",
                }
                detections.append(record)

    elapsed_seconds = time.time() - started_at
    random_identity_random = {
        **formal_random_trace_fields(formal_randomization_identity),
        **formal_random_trace_fields(base_latent_identity),
        "public_detection_seed_random": int(_public_detection_noise_seed(config)),
        "key_material_digest_random": build_stable_digest(
            {"key_material": config.key_material}
        ),
        "detection_key_plan_digest_random": (
            registered_detection_key_identity[
                "detection_key_plan_digest_random"
            ]
        ),
        "registered_wrong_key_negative_digest_random": (
            detection_key_plan[
                "registered_wrong_key_negative_digest_random"
            ]
        ),
        "standard_attack_seeds_random": {
            attack.attack_id: formal_attack_seed_random(
                int(config.seed),
                attack.attack_id,
            )
            for attack in attack_configs
        },
        "diffusion_attack_seeds_random": {
            attack.attack_id: formal_attack_seed_random(
                int(config.seed),
                attack.attack_id,
            )
            for attack in default_diffusion_attack_specs()
        }
        if config.diffusion_attacks_enabled
        else {},
    }
    scientific_unit_provenance = build_scientific_unit_provenance(
        scientific_unit_id=run_id,
        scientific_unit_config_digest=semantic_watermark_runtime_config_digest(config),
        runtime_environment=runtime_versions["runtime_environment"],
        execution_device_name=str(pipeline._execution_device),
        torch_module=torch,
        random_identity_random=random_identity_random,
    )
    (
        public_detection_noise_prg_identity,
        public_detection_noise_prg_identity_digest,
    ) = _runtime_public_detection_noise_identity(
        detections,
        attention_geometry_enabled=config.attention_geometry_enabled,
    )
    result = SemanticWatermarkRuntimeResult(
        run_id=run_id,
        run_decision="pass" if update_records else "fail",
        clean_image_path="",
        watermarked_image_path="",
        update_record_path="",
        detection_record_path="",
        manifest_path="",
        update_count=len(update_records),
        elapsed_seconds=elapsed_seconds,
        metadata={
            **runtime_versions,
            "method_runtime": "real_scientific_operators",
            "formal_method_config_digest": (
                config.formal_method_config_digest
            ),
            "method_definition": semantic_conditioned_latent_method_definition(),
            "method_definition_digest": (
                semantic_conditioned_latent_method_definition_digest()
            ),
            "formal_randomization_reference": (
                sample_randomization_reference
            ),
            "detector_input_access_mode": "image_key_public_model_only",
            "public_detection_noise_prg_identity": (
                public_detection_noise_prg_identity
            ),
            "public_detection_noise_prg_identity_digest": (
                public_detection_noise_prg_identity_digest
            ),
            "supports_paper_claim": False,
            "paired_quality": paired_quality,
            "final_image_preservation": final_image_preservation,
            "carrier_only_final_image_preservation": (
                carrier_only_final_image_preservation
            ),
            "carrier_only_counterfactual": carrier_only_counterfactual,
            "final_image_attention_observability": (
                final_image_attention_observability
            ),
            "scientific_unit_config_digest": (
                semantic_watermark_runtime_config_digest(config)
            ),
            "scientific_unit_provenance": scientific_unit_provenance,
        },
    )
    return (
        result,
        tuple(update_records),
        tuple(carrier_only_update_records),
        tuple(detections),
        clean_image,
        watermarked_image,
        carrier_only_image,
        attacked_images,
    )


def run_semantic_watermark_runtime(
    config: SemanticWatermarkRuntimeConfig,
    *,
    references: ContentRoutingReferenceScalars,
    verified_formal_execution_lock: Mapping[str, Any],
    repository_root: str | Path,
    runtime_context: SemanticWatermarkRuntimeContext | None = None,
) -> tuple[
    SemanticWatermarkRuntimeResult,
    tuple[dict[str, Any], ...],
    tuple[dict[str, Any], ...],
    tuple[dict[str, Any], ...],
    Any,
    Any,
    Any | None,
    dict[str, Any],
]:
    """执行正式索引10单写回生成与仅图像盲检测。"""

    _require_full_content_runtime_config(config)
    import torch

    if type(config) is not SemanticWatermarkRuntimeConfig:
        raise TypeError("config 必须为精确 SemanticWatermarkRuntimeConfig")
    if type(references) is not ContentRoutingReferenceScalars:
        raise TypeError("正式 runtime 必须显式接收 ContentRoutingReferenceScalars")
    if config.injection_step_indices != (10,):
        raise ValueError("正式 runtime 只允许索引10单次写回")
    if config.standard_attack_profiles or config.diffusion_attacks_enabled:
        raise RuntimeError(
            "未资格化 reference 的正式单样本链不得启动攻击或实验矩阵"
        )

    started_at = time.time()
    context = runtime_context or load_semantic_watermark_runtime_context(
        config,
        verified_formal_execution_lock=verified_formal_execution_lock,
        repository_root=repository_root,
    )
    serialized_runtime_versions = json.dumps(
        context.runtime_versions,
        ensure_ascii=False,
        sort_keys=True,
    )
    for forbidden in (
        "semantic_feature_operator_contract",
        "DifferentiableSemanticFeatureRuntime",
        "complete_716",
        "exact_jvp",
        "exact_vjp",
        "psd_cg",
    ):
        if forbidden in serialized_runtime_versions:
            raise RuntimeError("正式 runtime 仍携带旧716/JVP/VJP/PSD-CG身份")
    components = _ContentRuntimeSmokeComponents(
        pipeline=context.pipeline,
        prompt_saliency_runtime=context.prompt_saliency_runtime,
        attention_modules=context.attention_modules,
        unconditional_prompt=context.unconditional_prompt,
        unconditional_pooled=context.unconditional_pooled,
        runtime_versions=context.runtime_versions,
    )
    clean_image, watermarked_image, diagnostic = (
        _run_content_runtime_generation(
            config,
            references,
            components=components,
            include_clean=True,
        )
    )
    if clean_image is None:
        raise RuntimeError("正式 runtime 必须生成共享base latent的clean图像")

    measurement_config = _build_image_only_measurement_config(config)
    attention_extractor = _image_attention_extractor(
        context.pipeline,
        config,
        context.attention_modules,
        context.unconditional_prompt,
        context.unconditional_pooled,
    )
    run_id = build_semantic_watermark_run_id(config)
    paired_quality = compute_image_quality_metrics(clean_image, watermarked_image)
    detections: list[dict[str, Any]] = []
    detection_key_plan = build_detection_key_plan_record(config.key_material)
    for sample_role, image, detection_key_role in (
        ("clean_negative", clean_image, REGISTERED_WATERMARK_KEY_ROLE),
        ("positive_source", watermarked_image, REGISTERED_WATERMARK_KEY_ROLE),
        ("wrong_key_negative", watermarked_image, REGISTERED_WRONG_KEY_ROLE),
    ):
        detection_key, detection_key_identity = (
            resolve_detection_key_material_and_identity(
                config.key_material,
                detection_key_role,
            )
        )
        noise_cursor = _public_detection_noise_evidence_cursor(
            attention_extractor
        )
        detection = measure_image_only_watermark(
            image=image,
            key_material=detection_key,
            config=measurement_config,
            image_latent_encoder=lambda candidate: _encode_image_latent(
                context.pipeline,
                candidate,
            ),
            image_attention_extractor=attention_extractor,
            image_aligner=_align_image if config.image_alignment_enabled else None,
        )
        record = detection.to_record()
        _bind_public_detection_noise_qk_evidence(
            record,
            attention_extractor,
            noise_cursor,
        )
        record.update(
            {
                "run_id": run_id,
                "prompt_id": config.prompt_id,
                "split": config.split,
                "sample_role": sample_role,
                **detection_key_identity,
                "embedding_pair_ssim": float(paired_quality["ssim"]),
                "attack_id": None,
            }
        )
        record["metadata"] = {
            **record["metadata"],
            "method_role": measurement_config.method_role,
            "measurement_status": "threshold_independent_image_only_evidence",
            "reference_source": "explicit_smoke_only_unqualified",
            "supports_paper_claim": False,
        }
        detections.append(record)

    update_record = {
        "run_id": run_id,
        "step_index": 10,
        "method_role": diagnostic["method_role"],
        "captured_previous_index": diagnostic["captured_previous_index"],
        "captured_previous_count": diagnostic["captured_previous_count"],
        "callback_write_index": diagnostic["callback_write_index"],
        "callback_write_count": diagnostic["callback_write_count"],
        "actual_dtype_single_write_count": diagnostic[
            "actual_dtype_single_write_count"
        ],
        "current_image_decode_count": diagnostic["current_image_decode_count"],
        "public_probe_additional_decode_count": diagnostic[
            "public_probe_additional_decode_count"
        ],
        "lf_effective_l2": diagnostic["lf_effective_l2"],
        "hf_tail_effective_l2": diagnostic["hf_tail_effective_l2"],
        "geometry_effective_l2": diagnostic["geometry_effective_l2"],
        "combined_effective_l2": diagnostic["combined_effective_l2"],
        "combined_effective_l2_limit": diagnostic[
            "combined_effective_l2_limit"
        ],
        "combined_effective_l2_ready": diagnostic[
            "combined_effective_l2_ready"
        ],
        "common_gamma": diagnostic["common_gamma"],
        "content_only_postwrite_qk_score": diagnostic[
            "content_only_postwrite_qk_score"
        ],
        "final_postwrite_qk_score": diagnostic["final_postwrite_qk_score"],
        "post_write_qk_strict_ready": diagnostic[
            "post_write_qk_strict_ready"
        ],
        "actual_dtype_single_write_digest": diagnostic[
            "actual_dtype_single_write_digest"
        ],
        "routing_identity_digest": diagnostic["routing_identity_digest"],
        "geometry_update_digest": diagnostic["geometry_update_digest"],
        "geometry_qk_atomic_records_digest": diagnostic[
            "geometry_qk_atomic_records_digest"
        ],
        "content_only_postwrite_qk_digest": diagnostic[
            "content_only_postwrite_qk_digest"
        ],
        "final_postwrite_qk_digest": diagnostic[
            "final_postwrite_qk_digest"
        ],
        "attention_module_names": list(config.attention_module_names),
        "reference_source": "explicit_smoke_only_unqualified",
        "supports_paper_claim": False,
    }
    if not all(
        float(update_record[field_name]) > 0.0
        for field_name in (
            "lf_effective_l2",
            "hf_tail_effective_l2",
            "geometry_effective_l2",
            "combined_effective_l2",
        )
    ):
        raise RuntimeError("full_dual_chain 三个正式分支必须均产生非零实际写回")
    if (
        update_record["combined_effective_l2_ready"] is not True
        or not float(update_record["combined_effective_l2"])
        <= float(update_record["combined_effective_l2_limit"])
        or update_record["post_write_qk_strict_ready"] is not True
        or not float(update_record["final_postwrite_qk_score"])
        > float(update_record["content_only_postwrite_qk_score"])
    ):
        raise RuntimeError("正式单写回预算或post-write Q/K门禁未闭合")

    runtime_versions = dict(context.runtime_versions)
    forbidden_runtime_modules = (
        "main.methods.subspace.jacobian_nullspace",
        "main.methods.semantic.runtime",
    )
    legacy_runtime_dependency_absence_ready = not any(
        module_name in sys.modules for module_name in forbidden_runtime_modules
    )
    random_identity_random = {
        "generation_seed_random": int(config.seed),
        "watermark_key_material_digest_random": build_stable_digest(
            {"key_material": config.key_material}
        ),
        "detection_key_plan_digest_random": detection_key_plan[
            "detection_key_plan_digest_random"
        ],
    }
    provenance = build_scientific_unit_provenance(
        scientific_unit_id=run_id,
        scientific_unit_config_digest=semantic_watermark_runtime_config_digest(
            config
        ),
        runtime_environment=runtime_versions["runtime_environment"],
        execution_device_name=str(context.pipeline._execution_device),
        torch_module=torch,
        random_identity_random=random_identity_random,
    )
    formal_randomization_reference = (
        _content_runtime_formal_randomization_reference(config, diagnostic)
    )
    result = SemanticWatermarkRuntimeResult(
        run_id=run_id,
        run_decision="pass",
        clean_image_path="",
        watermarked_image_path="",
        update_record_path="",
        detection_record_path="",
        manifest_path="",
        update_count=1,
        elapsed_seconds=time.time() - started_at,
        metadata={
            **runtime_versions,
            "method_runtime": "formal_content_dual_chain_single_write",
            "formal_method_config_digest": config.formal_method_config_digest,
            "method_definition": semantic_conditioned_latent_method_definition(),
            "method_definition_digest": (
                semantic_conditioned_latent_method_definition_digest()
            ),
            "formal_randomization_reference": formal_randomization_reference,
            "detector_input_access_mode": "image_key_public_model_only",
            "threshold_free_blind_measurement_ready": bool(detections),
            "formal_blind_detection_ready": False,
            "legacy_runtime_dependency_absence_ready": (
                legacy_runtime_dependency_absence_ready
            ),
            "forbidden_runtime_modules": list(forbidden_runtime_modules),
            "paired_quality": paired_quality,
            "content_runtime_diagnostic": diagnostic,
            "reference_source": "explicit_smoke_only_unqualified",
            "supports_paper_claim": False,
            "scientific_unit_config_digest": (
                semantic_watermark_runtime_config_digest(config)
            ),
            "scientific_unit_provenance": provenance,
        },
    )
    return (
        result,
        (update_record,),
        (),
        tuple(detections),
        clean_image,
        watermarked_image,
        None,
        {},
    )


def write_semantic_watermark_runtime_outputs(
    config: SemanticWatermarkRuntimeConfig,
    root: str | Path = ".",
    *,
    references: ContentRoutingReferenceScalars,
    verified_formal_execution_lock: Mapping[str, Any],
    runtime_context: SemanticWatermarkRuntimeContext | None = None,
) -> SemanticWatermarkRuntimeResult:
    """运行真实方法并把全部持久化产物写入 outputs。"""

    _require_full_content_runtime_config(config)
    root_path = Path(root).resolve()
    output_dir = (root_path / config.output_dir).resolve()
    outputs_root = (root_path / "outputs").resolve()
    if output_dir != outputs_root and outputs_root not in output_dir.parents:
        raise ValueError("真实方法输出必须位于 outputs 目录")
    output_dir.mkdir(parents=True, exist_ok=True)
    (
        result,
        update_records,
        carrier_only_update_records,
        detections,
        clean_image,
        watermarked_image,
        carrier_only_image,
        attacked_images,
    ) = run_semantic_watermark_runtime(
        config,
        references=references,
        verified_formal_execution_lock=verified_formal_execution_lock,
        repository_root=root_path,
        runtime_context=runtime_context,
    )
    detection_key_plan = build_detection_key_plan_record(
        config.key_material
    )
    run_dir = output_dir / result.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    clean_image_path = run_dir / "clean_image.png"
    watermarked_image_path = run_dir / "watermarked_image.png"
    carrier_only_image_path = run_dir / "carrier_only_image.png"
    clean_image.save(clean_image_path)
    watermarked_image.save(watermarked_image_path)
    if carrier_only_image is not None:
        carrier_only_image.save(carrier_only_image_path)
    attacked_image_dir = run_dir / "attacked_images"
    attacked_image_dir.mkdir(parents=True, exist_ok=True)
    attacked_image_paths = []
    attacked_image_path_by_key: dict[str, Path] = {}
    for image_key, attacked_image in sorted(attacked_images.items()):
        attacked_path = attacked_image_dir / f"{image_key}.png"
        attacked_image.save(attacked_path)
        attacked_image_path_by_key[image_key] = attacked_path
        attacked_image_paths.append(attacked_path.relative_to(root_path).as_posix())
    update_path = run_dir / "latent_update_records.jsonl"
    carrier_only_update_path = run_dir / "carrier_only_update_records.jsonl"
    detection_path = run_dir / "image_only_detection_records.jsonl"
    result_path = run_dir / "runtime_result.json"
    governed_detections = []
    for detection in detections:
        record = dict(detection)
        sample_role = str(record.get("sample_role", ""))
        source_path = clean_image_path if sample_role == "clean_negative" else watermarked_image_path
        source_image = clean_image if sample_role == "clean_negative" else watermarked_image
        attacked_image_key = str(record.get("attacked_image_key", ""))
        evaluated_path = attacked_image_path_by_key.get(attacked_image_key, source_path)
        evaluated_image = attacked_images.get(attacked_image_key, source_image)
        source_to_evaluated_quality = compute_image_quality_metrics(source_image, evaluated_image)
        source_pixel_identity = canonical_rgb_uint8_content_record(
            source_image
        )
        evaluated_pixel_identity = canonical_rgb_uint8_content_record(
            evaluated_image
        )
        record.update(
            {
                "run_id": result.run_id,
                "source_image_path": source_path.relative_to(root_path).as_posix(),
                "source_image_digest": file_digest(source_path),
                "source_image_rgb_uint8_content_sha256": (
                    source_pixel_identity[
                        "image_rgb_uint8_content_sha256"
                    ]
                ),
                "source_image_width": source_pixel_identity[
                    "image_width"
                ],
                "source_image_height": source_pixel_identity[
                    "image_height"
                ],
                "evaluated_image_path": evaluated_path.relative_to(root_path).as_posix(),
                "evaluated_image_digest": file_digest(evaluated_path),
                "evaluated_image_rgb_uint8_content_sha256": (
                    evaluated_pixel_identity[
                        "image_rgb_uint8_content_sha256"
                    ]
                ),
                "evaluated_image_width": evaluated_pixel_identity[
                    "image_width"
                ],
                "evaluated_image_height": evaluated_pixel_identity[
                    "image_height"
                ],
                "attacked_image_path": (
                    evaluated_path.relative_to(root_path).as_posix() if attacked_image_key else ""
                ),
                "attacked_image_digest": file_digest(evaluated_path) if attacked_image_key else "",
                "source_to_evaluated_ssim": float(source_to_evaluated_quality["ssim"]),
                "source_to_evaluated_psnr": source_to_evaluated_quality["psnr"],
                "source_to_evaluated_mse": float(source_to_evaluated_quality["mse"]),
            }
        )
        if config.attention_geometry_enabled:
            record = _bind_detection_qk_to_pixels(
                record,
                evaluated_image,
            )
        governed_detections.append(record)
    update_path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in update_records), encoding="utf-8")
    if carrier_only_image is not None:
        carrier_only_update_path.write_text(
            "".join(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                for row in carrier_only_update_records
            ),
            encoding="utf-8",
        )
    detection_path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in governed_detections), encoding="utf-8")
    manifest_path = run_dir / "manifest.local.json"
    resolved_result = SemanticWatermarkRuntimeResult(
        **{
            **result.to_dict(),
            "clean_image_path": clean_image_path.relative_to(
                root_path
            ).as_posix(),
            "watermarked_image_path": watermarked_image_path.relative_to(
                root_path
            ).as_posix(),
            "update_record_path": update_path.relative_to(
                root_path
            ).as_posix(),
            "detection_record_path": detection_path.relative_to(
                root_path
            ).as_posix(),
            "manifest_path": manifest_path.relative_to(root_path).as_posix(),
        }
    )
    result_path.write_text(
        _stable_json(resolved_result.to_dict()),
        encoding="utf-8",
    )
    output_paths = (
        update_path.relative_to(root_path).as_posix(),
        detection_path.relative_to(root_path).as_posix(),
        result_path.relative_to(root_path).as_posix(),
        clean_image_path.relative_to(root_path).as_posix(),
        watermarked_image_path.relative_to(root_path).as_posix(),
        *(
            (carrier_only_image_path.relative_to(root_path).as_posix(),)
            if carrier_only_image is not None
            else ()
        ),
        *(
            (carrier_only_update_path.relative_to(root_path).as_posix(),)
            if carrier_only_image is not None
            else ()
        ),
        *attacked_image_paths,
        manifest_path.relative_to(root_path).as_posix(),
    )
    manifest = build_artifact_manifest(
        artifact_id=f"{result.run_id}_manifest",
        artifact_type="local_manifest",
        input_paths=(),
        output_paths=output_paths,
        config={
            "scientific_unit_config_digest": (
                semantic_watermark_runtime_config_digest(config)
            ),
            "formal_randomization_reference": result.metadata[
                "formal_randomization_reference"
            ],
        },
        code_version=resolve_code_version(root_path),
        rebuild_command=(
            "调用 experiments.runners.semantic_watermark_runtime."
            "write_semantic_watermark_runtime_outputs"
        ),
        metadata={
            "run_id": result.run_id,
            "protocol_decision": result.run_decision,
            "detector_input_access_mode": "image_key_public_model_only",
            "detection_key_plan": detection_key_plan,
            "supports_paper_claim": False,
            "formal_runtime_chain": (
                "content_routing_lf_hf_qk_common_gamma_single_write"
            ),
            "formal_detection_chain": (
                "threshold_free_lf_hf_tail_measurement_pending_frozen_evidence"
            ),
        },
    ).to_dict()
    manifest_path.write_text(_stable_json(manifest), encoding="utf-8")
    output_parts = Path(config.output_dir).parts
    checkpoint_roles = {
        "image_only_dataset_runtime": "image_only_dataset_runtime",
        "formal_mechanism_ablation": "runtime_rerun_ablation",
    }
    if (
        len(output_parts) >= 3
        and output_parts[0] == "outputs"
        and output_parts[1] in checkpoint_roles
    ):
        persist_completed_unit_from_manifest(
            manifest_path,
            repository_root=root_path,
            artifact_role=checkpoint_roles[output_parts[1]],
            paper_run_name=output_parts[2],
        )
    return resolved_result
