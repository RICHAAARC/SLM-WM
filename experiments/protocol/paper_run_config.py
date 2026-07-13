"""集中描述论文运行配置。

该模块的作用是把论文运行层级、prompt 文件、Drive 结果根目录和样本规模收敛到一个配置解析层。
这样 Notebook 与 Colab helper 不需要各自硬编码 120, 128 或固定的 Drive 子目录. 无显式
输入时统一从 probe_paper 开始, 后续切换到 pilot_paper 或 full_paper 只需要设置
`SLM_WM_PAPER_RUN_NAME` 或相关环境变量.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

from experiments.protocol.method_runtime_config import (
    FORMAL_METHOD_PACKAGE_ROOT,
    load_formal_method_runtime_config,
)
from experiments.protocol.formal_randomization import (
    DEFAULT_FORMAL_RANDOMIZATION_REPEAT_ID,
    build_formal_randomization_identity,
    formal_randomization_protocol_record,
    formal_randomization_repeats,
    resolve_formal_randomization_repeat,
)
from experiments.protocol.prompts import PROMPT_FILES, read_prompt_file
from experiments.protocol.prompt_sources import audit_governed_prompt_set
from experiments.protocol.splits import build_group_split_counts
from main.core.keyed_prg import require_supported_keyed_prg_version
from main.methods.carrier import LowFrequencyCarrierConfig
from main.methods.geometry import validate_attention_alignment_gate

PILOT_PAPER_RUN_NAME = "pilot_paper"
PROBE_PAPER_RUN_NAME = "probe_paper"
FULL_PAPER_RUN_NAME = "full_paper"
DEFAULT_TARGET_FPR = 0.1
DEFAULT_MINIMUM_CLEAN_NEGATIVE_COUNT = 34
DEFAULT_DATASET_LEVEL_QUALITY_MINIMUM_COUNT = 70
DEFAULT_DRIVE_ROOT = "/content/drive/MyDrive/SLM"
_FORMAL_METHOD_DEFAULTS = load_formal_method_runtime_config(
    FORMAL_METHOD_PACKAGE_ROOT
)
_FORMAL_RANDOMIZATION_DEFAULTS = formal_randomization_protocol_record()
DEFAULT_FORMAL_RANDOMIZATION_PROTOCOL_DIGEST = (
    _FORMAL_RANDOMIZATION_DEFAULTS["formal_randomization_protocol_digest"]
)
DEFAULT_FORMAL_RANDOMIZATION_REPEAT_COUNT = int(
    _FORMAL_RANDOMIZATION_DEFAULTS["crossed_repeat_count"]
)
DEFAULT_FORMAL_METHOD_CONFIG_DIGEST = (
    _FORMAL_METHOD_DEFAULTS.formal_method_config_digest
)
DEFAULT_PIPELINE_CLASS_NAME = _FORMAL_METHOD_DEFAULTS.pipeline_class_name
DEFAULT_VAE_CLASS_NAME = _FORMAL_METHOD_DEFAULTS.vae_class_name
DEFAULT_TRANSFORMER_CLASS_NAME = _FORMAL_METHOD_DEFAULTS.transformer_class_name
DEFAULT_SCHEDULER_CLASS_NAME = _FORMAL_METHOD_DEFAULTS.scheduler_class_name
DEFAULT_VAE_SCALING_FACTOR = _FORMAL_METHOD_DEFAULTS.vae_scaling_factor
DEFAULT_VAE_SHIFT_FACTOR = _FORMAL_METHOD_DEFAULTS.vae_shift_factor
DEFAULT_LATENT_TORCH_DTYPE = _FORMAL_METHOD_DEFAULTS.latent_torch_dtype
DEFAULT_VISION_TORCH_DTYPE = _FORMAL_METHOD_DEFAULTS.vision_torch_dtype
DEFAULT_INFERENCE_STEPS = _FORMAL_METHOD_DEFAULTS.inference_steps
DEFAULT_GUIDANCE_SCALE = _FORMAL_METHOD_DEFAULTS.guidance_scale
DEFAULT_RISK_SIGNAL_CALIBRATION_PROTOCOL = (
    _FORMAL_METHOD_DEFAULTS.risk_signal_calibration_protocol
)
DEFAULT_RISK_IMAGE_SIGNAL_INTERPOLATION_MODE = (
    _FORMAL_METHOD_DEFAULTS.risk_image_signal_interpolation_mode
)
DEFAULT_RISK_IMAGE_SIGNAL_ALIGN_CORNERS = (
    _FORMAL_METHOD_DEFAULTS.risk_image_signal_align_corners
)
DEFAULT_RISK_ATTENTION_SIGNAL_INTERPOLATION_MODE = (
    _FORMAL_METHOD_DEFAULTS.risk_attention_signal_interpolation_mode
)
DEFAULT_RISK_ATTENTION_SIGNAL_ALIGN_CORNERS = (
    _FORMAL_METHOD_DEFAULTS.risk_attention_signal_align_corners
)
DEFAULT_RISK_NEUTRAL_TEXTURE_VALUE = (
    _FORMAL_METHOD_DEFAULTS.risk_neutral_texture_value
)
DEFAULT_RISK_ELIGIBILITY_COMPARISON = (
    _FORMAL_METHOD_DEFAULTS.risk_eligibility_comparison
)
DEFAULT_RISK_BUDGET_BROADCAST_PROTOCOL = (
    _FORMAL_METHOD_DEFAULTS.risk_budget_broadcast_protocol
)
DEFAULT_RISK_ZERO_SUPPORT_PROTOCOL = (
    _FORMAL_METHOD_DEFAULTS.risk_zero_support_protocol
)
DEFAULT_RISK_BOUNDED_SCALE_PROTOCOL = (
    _FORMAL_METHOD_DEFAULTS.risk_bounded_scale_protocol
)
DEFAULT_RISK_BOUNDED_SCALE_DIRECTION_EPSILON = (
    _FORMAL_METHOD_DEFAULTS.risk_bounded_scale_direction_epsilon
)
DEFAULT_LF_CONTENT_RISK_CONFIG = asdict(
    _FORMAL_METHOD_DEFAULTS.lf_content_risk_config
)
DEFAULT_TAIL_ROBUST_RISK_CONFIG = asdict(
    _FORMAL_METHOD_DEFAULTS.tail_robust_risk_config
)
DEFAULT_ATTENTION_GEOMETRY_RISK_CONFIG = asdict(
    _FORMAL_METHOD_DEFAULTS.attention_geometry_risk_config
)
DEFAULT_ATTENTION_INJECTION_STEPS = _FORMAL_METHOD_DEFAULTS.injection_step_indices
DEFAULT_JACOBIAN_CANDIDATE_COUNT = _FORMAL_METHOD_DEFAULTS.jacobian_candidate_count
DEFAULT_NULL_SPACE_RANK = _FORMAL_METHOD_DEFAULTS.null_space_rank
DEFAULT_NULL_SPACE_NUMERICAL_EPSILON = (
    _FORMAL_METHOD_DEFAULTS.null_space_numerical_epsilon
)
DEFAULT_MAXIMUM_QR_CONDITION_NUMBER = (
    _FORMAL_METHOD_DEFAULTS.maximum_qr_condition_number
)
DEFAULT_MAXIMUM_ORTHOGONALITY_ERROR = (
    _FORMAL_METHOD_DEFAULTS.maximum_orthogonality_error
)
DEFAULT_QR_REFERENCE_SOLVE_PROTOCOL = (
    _FORMAL_METHOD_DEFAULTS.qr_reference_solve_protocol
)
DEFAULT_LF_RELATIVE_STRENGTH = _FORMAL_METHOD_DEFAULTS.lf_relative_strength
DEFAULT_TAIL_RELATIVE_STRENGTH = _FORMAL_METHOD_DEFAULTS.tail_relative_strength
DEFAULT_ATTENTION_RELATIVE_STRENGTH = _FORMAL_METHOD_DEFAULTS.attention_relative_strength
DEFAULT_LF_KERNEL_SIZE = _FORMAL_METHOD_DEFAULTS.lf_kernel_size
DEFAULT_LF_STRIDE = _FORMAL_METHOD_DEFAULTS.lf_stride
DEFAULT_LF_PADDING = _FORMAL_METHOD_DEFAULTS.lf_padding
DEFAULT_LF_BOUNDARY_MODE = _FORMAL_METHOD_DEFAULTS.lf_boundary_mode
DEFAULT_LF_CEIL_MODE = _FORMAL_METHOD_DEFAULTS.lf_ceil_mode
DEFAULT_LF_COUNT_INCLUDE_PAD = _FORMAL_METHOD_DEFAULTS.lf_count_include_pad
DEFAULT_LF_DIVISOR_OVERRIDE = _FORMAL_METHOD_DEFAULTS.lf_divisor_override
DEFAULT_LF_DETECTION_SCORE_WEIGHT = (
    _FORMAL_METHOD_DEFAULTS.lf_detection_score_weight
)
DEFAULT_TAIL_ROBUST_DETECTION_SCORE_WEIGHT = (
    _FORMAL_METHOD_DEFAULTS.tail_robust_detection_score_weight
)
DEFAULT_ATTENTION_STABLE_TOKEN_FRACTION = (
    _FORMAL_METHOD_DEFAULTS.attention_stable_token_fraction
)
DEFAULT_ATTENTION_UNSTABLE_PAIR_WEIGHT = (
    _FORMAL_METHOD_DEFAULTS.attention_unstable_pair_weight
)
DEFAULT_ATTENTION_RELATION_COMPONENT_WEIGHTS = (
    _FORMAL_METHOD_DEFAULTS.attention_relation_component_weights
)
DEFAULT_ATTENTION_ANCHOR_COUNT = (
    _FORMAL_METHOD_DEFAULTS.attention_anchor_count
)
DEFAULT_ATTENTION_RESIDUAL_THRESHOLD = (
    _FORMAL_METHOD_DEFAULTS.attention_residual_threshold
)
DEFAULT_ATTENTION_MINIMUM_INLIER_RATIO = (
    _FORMAL_METHOD_DEFAULTS.attention_minimum_inlier_ratio
)
DEFAULT_ATTENTION_BACKTRACKING_FACTOR = (
    _FORMAL_METHOD_DEFAULTS.attention_backtracking_factor
)
DEFAULT_ATTENTION_BACKTRACKING_MAXIMUM_STEPS = (
    _FORMAL_METHOD_DEFAULTS.attention_backtracking_maximum_steps
)
DEFAULT_MINIMUM_FINAL_IMAGE_ATTENTION_SCORE_GAIN = (
    _FORMAL_METHOD_DEFAULTS.minimum_final_image_attention_score_gain
)
DEFAULT_TAIL_FRACTION = _FORMAL_METHOD_DEFAULTS.tail_fraction
DEFAULT_KEYED_PRG_VERSION = _FORMAL_METHOD_DEFAULTS.keyed_prg_version
DEFAULT_MINIMUM_PROJECTION_ENERGY_RETENTION = _FORMAL_METHOD_DEFAULTS.minimum_projection_energy_retention
DEFAULT_MAXIMUM_RELATIVE_RESPONSE_RESIDUAL = _FORMAL_METHOD_DEFAULTS.maximum_relative_response_residual
DEFAULT_MAXIMUM_QUANTIZED_WRITE_RELATIVE_JACOBIAN_RESPONSE = (
    _FORMAL_METHOD_DEFAULTS.maximum_quantized_write_relative_jacobian_response
)
DEFAULT_QUANTIZED_BRANCH_COMPOSITION_PROTOCOL = (
    _FORMAL_METHOD_DEFAULTS.quantized_branch_composition_protocol
)
DEFAULT_QUANTIZED_BRANCH_COMPOSITION_ORDER = (
    _FORMAL_METHOD_DEFAULTS.quantized_branch_composition_order
)
DEFAULT_COMBINED_BUDGET_ENVELOPE_RULE = (
    _FORMAL_METHOD_DEFAULTS.combined_budget_envelope_rule
)
DEFAULT_QUANTIZED_BUDGET_ENVELOPE_ABSOLUTE_TOLERANCE = (
    _FORMAL_METHOD_DEFAULTS.quantized_budget_envelope_absolute_tolerance
)
DEFAULT_QUANTIZED_BUDGET_ENVELOPE_BACKTRACKING_FACTOR = (
    _FORMAL_METHOD_DEFAULTS.quantized_budget_envelope_backtracking_factor
)
DEFAULT_QUANTIZED_BUDGET_ENVELOPE_BACKTRACKING_MAXIMUM_STEPS = (
    _FORMAL_METHOD_DEFAULTS.quantized_budget_envelope_backtracking_maximum_steps
)
DEFAULT_NULL_SPACE_CG_MAX_ITERATIONS = (
    _FORMAL_METHOD_DEFAULTS.null_space_cg_max_iterations
)
DEFAULT_NULL_SPACE_CG_RELATIVE_TOLERANCE = (
    _FORMAL_METHOD_DEFAULTS.null_space_cg_relative_tolerance
)
DEFAULT_MINIMUM_SEMANTIC_PRESERVATION_COSINE = (
    _FORMAL_METHOD_DEFAULTS.minimum_semantic_preservation_cosine
)
DEFAULT_MAXIMUM_HANDCRAFTED_STRUCTURE_FEATURE_RELATIVE_DRIFT = (
    _FORMAL_METHOD_DEFAULTS.maximum_handcrafted_structure_feature_relative_drift
)
DEFAULT_PUBLIC_DETECTION_SCHEDULE_INDEX = (
    _FORMAL_METHOD_DEFAULTS.public_detection_schedule_index
)
DEFAULT_PUBLIC_DETECTION_NOISE_PRG_PROTOCOL = (
    _FORMAL_METHOD_DEFAULTS.public_detection_noise_prg_protocol
)
DEFAULT_PUBLIC_DETECTION_NOISE_DOMAIN = (
    _FORMAL_METHOD_DEFAULTS.public_detection_noise_domain
)
DEFAULT_PUBLIC_DETECTION_CONDITIONING_PROTOCOL = (
    _FORMAL_METHOD_DEFAULTS.public_detection_conditioning_protocol
)
DEFAULT_PUBLIC_DETECTION_CONDITION_TEXT = (
    _FORMAL_METHOD_DEFAULTS.public_detection_condition_text
)
DEFAULT_MAX_ATTENTION_TOKENS = _FORMAL_METHOD_DEFAULTS.max_attention_tokens
DEFAULT_ATTENTION_MODULE_NAMES = _FORMAL_METHOD_DEFAULTS.attention_module_names
DEFAULT_ATTENTION_ALIGNMENT_LAYER_SELECTION_RULE = (
    _FORMAL_METHOD_DEFAULTS.attention_alignment_layer_selection_rule
)
DEFAULT_IMAGE_ALIGNMENT_RESAMPLING_MODE = (
    _FORMAL_METHOD_DEFAULTS.image_alignment_resampling_mode
)
DEFAULT_IMAGE_ALIGNMENT_PADDING_MODE = (
    _FORMAL_METHOD_DEFAULTS.image_alignment_padding_mode
)
DEFAULT_IMAGE_ALIGNMENT_QUANTIZATION_PROTOCOL = (
    _FORMAL_METHOD_DEFAULTS.image_alignment_quantization_protocol
)
DEFAULT_ATTENTION_COORDINATE_CONVENTION = (
    _FORMAL_METHOD_DEFAULTS.attention_coordinate_convention
)
DEFAULT_ATTENTION_GRID_ALIGN_CORNERS = (
    _FORMAL_METHOD_DEFAULTS.attention_grid_align_corners
)
UNBOUNDED_LIMIT_TOKENS = {"", "all", "none", "unlimited"}
SHARED_METHOD_SETTING_FIELDS = tuple(
    _FORMAL_METHOD_DEFAULTS.paper_method_settings()
)
SHARED_EXPERIMENT_SETTING_FIELDS = (
    "randomization_repeat_id",
    "generation_seed_index",
    "generation_seed_offset",
    "watermark_key_index",
    "formal_randomization_protocol_digest",
    "formal_randomization_repeat_count",
)

RUN_DEFAULTS: dict[str, dict[str, Any]] = {
    PROBE_PAPER_RUN_NAME: {
        "prompt_set": PROBE_PAPER_RUN_NAME,
        "prompt_file": PROMPT_FILES[PROBE_PAPER_RUN_NAME].as_posix(),
        "drive_result_root": f"{DEFAULT_DRIVE_ROOT}/probe_paper_results",
        "protocol_profile": "paper_fixed_fpr_0_1",
        "target_fpr": 0.1,
        "sample_count": "all",
    },
    PILOT_PAPER_RUN_NAME: {
        "prompt_set": PILOT_PAPER_RUN_NAME,
        "prompt_file": PROMPT_FILES[PILOT_PAPER_RUN_NAME].as_posix(),
        "drive_result_root": f"{DEFAULT_DRIVE_ROOT}/pilot_paper_results",
        "protocol_profile": "paper_fixed_fpr_0_1",
        "target_fpr": 0.1,
        "sample_count": "all",
    },
    FULL_PAPER_RUN_NAME: {
        "prompt_set": FULL_PAPER_RUN_NAME,
        "prompt_file": PROMPT_FILES[FULL_PAPER_RUN_NAME].as_posix(),
        "drive_result_root": f"{DEFAULT_DRIVE_ROOT}/full_paper_results",
        "protocol_profile": "paper_fixed_fpr_0_1",
        "target_fpr": 0.1,
        "sample_count": "all",
    },
}
RUN_EXPECTED_PROMPT_COUNTS = {
    PROBE_PAPER_RUN_NAME: 70,
    PILOT_PAPER_RUN_NAME: 700,
    FULL_PAPER_RUN_NAME: 7000,
}


@dataclass(frozen=True)
class PaperRunConfig:
    """保存当前论文运行层级的统一配置。"""

    run_name: str
    protocol_profile: str
    prompt_set: str
    prompt_file: str
    prompt_count: int
    sample_count: int
    drive_result_root: str
    target_fpr: float = DEFAULT_TARGET_FPR
    minimum_clean_negative_count: int = DEFAULT_MINIMUM_CLEAN_NEGATIVE_COUNT
    dataset_level_quality_minimum_count: int = DEFAULT_DATASET_LEVEL_QUALITY_MINIMUM_COUNT
    randomization_repeat_id: str = DEFAULT_FORMAL_RANDOMIZATION_REPEAT_ID
    generation_seed_index: int = 0
    generation_seed_offset: int = 0
    watermark_key_index: int = 0
    formal_randomization_protocol_digest: str = (
        DEFAULT_FORMAL_RANDOMIZATION_PROTOCOL_DIGEST
    )
    formal_randomization_repeat_count: int = (
        DEFAULT_FORMAL_RANDOMIZATION_REPEAT_COUNT
    )
    formal_method_config_digest: str = DEFAULT_FORMAL_METHOD_CONFIG_DIGEST
    pipeline_class_name: str = DEFAULT_PIPELINE_CLASS_NAME
    vae_class_name: str = DEFAULT_VAE_CLASS_NAME
    transformer_class_name: str = DEFAULT_TRANSFORMER_CLASS_NAME
    scheduler_class_name: str = DEFAULT_SCHEDULER_CLASS_NAME
    vae_scaling_factor: float = DEFAULT_VAE_SCALING_FACTOR
    vae_shift_factor: float = DEFAULT_VAE_SHIFT_FACTOR
    latent_torch_dtype: str = DEFAULT_LATENT_TORCH_DTYPE
    vision_torch_dtype: str = DEFAULT_VISION_TORCH_DTYPE
    inference_steps: int = DEFAULT_INFERENCE_STEPS
    guidance_scale: float = DEFAULT_GUIDANCE_SCALE
    risk_signal_calibration_protocol: str = (
        DEFAULT_RISK_SIGNAL_CALIBRATION_PROTOCOL
    )
    risk_image_signal_interpolation_mode: str = (
        DEFAULT_RISK_IMAGE_SIGNAL_INTERPOLATION_MODE
    )
    risk_image_signal_align_corners: bool = (
        DEFAULT_RISK_IMAGE_SIGNAL_ALIGN_CORNERS
    )
    risk_attention_signal_interpolation_mode: str = (
        DEFAULT_RISK_ATTENTION_SIGNAL_INTERPOLATION_MODE
    )
    risk_attention_signal_align_corners: bool = (
        DEFAULT_RISK_ATTENTION_SIGNAL_ALIGN_CORNERS
    )
    risk_neutral_texture_value: float = DEFAULT_RISK_NEUTRAL_TEXTURE_VALUE
    risk_eligibility_comparison: str = DEFAULT_RISK_ELIGIBILITY_COMPARISON
    risk_budget_broadcast_protocol: str = (
        DEFAULT_RISK_BUDGET_BROADCAST_PROTOCOL
    )
    risk_zero_support_protocol: str = DEFAULT_RISK_ZERO_SUPPORT_PROTOCOL
    risk_bounded_scale_protocol: str = DEFAULT_RISK_BOUNDED_SCALE_PROTOCOL
    risk_bounded_scale_direction_epsilon: float = (
        DEFAULT_RISK_BOUNDED_SCALE_DIRECTION_EPSILON
    )
    lf_content_risk_config: dict[str, Any] = field(
        default_factory=lambda: dict(DEFAULT_LF_CONTENT_RISK_CONFIG)
    )
    tail_robust_risk_config: dict[str, Any] = field(
        default_factory=lambda: dict(DEFAULT_TAIL_ROBUST_RISK_CONFIG)
    )
    attention_geometry_risk_config: dict[str, Any] = field(
        default_factory=lambda: dict(DEFAULT_ATTENTION_GEOMETRY_RISK_CONFIG)
    )
    attention_injection_steps: tuple[int, ...] = DEFAULT_ATTENTION_INJECTION_STEPS
    jacobian_candidate_count: int = DEFAULT_JACOBIAN_CANDIDATE_COUNT
    null_space_rank: int = DEFAULT_NULL_SPACE_RANK
    null_space_numerical_epsilon: float = DEFAULT_NULL_SPACE_NUMERICAL_EPSILON
    maximum_qr_condition_number: float = DEFAULT_MAXIMUM_QR_CONDITION_NUMBER
    maximum_orthogonality_error: float = DEFAULT_MAXIMUM_ORTHOGONALITY_ERROR
    qr_reference_solve_protocol: str = DEFAULT_QR_REFERENCE_SOLVE_PROTOCOL
    lf_relative_strength: float = DEFAULT_LF_RELATIVE_STRENGTH
    tail_relative_strength: float = DEFAULT_TAIL_RELATIVE_STRENGTH
    attention_relative_strength: float = DEFAULT_ATTENTION_RELATIVE_STRENGTH
    lf_kernel_size: int = DEFAULT_LF_KERNEL_SIZE
    lf_stride: int = DEFAULT_LF_STRIDE
    lf_padding: int = DEFAULT_LF_PADDING
    lf_boundary_mode: str = DEFAULT_LF_BOUNDARY_MODE
    lf_ceil_mode: bool = DEFAULT_LF_CEIL_MODE
    lf_count_include_pad: bool = DEFAULT_LF_COUNT_INCLUDE_PAD
    lf_divisor_override: int | None = DEFAULT_LF_DIVISOR_OVERRIDE
    lf_detection_score_weight: float = DEFAULT_LF_DETECTION_SCORE_WEIGHT
    tail_robust_detection_score_weight: float = (
        DEFAULT_TAIL_ROBUST_DETECTION_SCORE_WEIGHT
    )
    attention_stable_token_fraction: float = (
        DEFAULT_ATTENTION_STABLE_TOKEN_FRACTION
    )
    attention_unstable_pair_weight: float = (
        DEFAULT_ATTENTION_UNSTABLE_PAIR_WEIGHT
    )
    attention_relation_component_weights: tuple[float, ...] = (
        DEFAULT_ATTENTION_RELATION_COMPONENT_WEIGHTS
    )
    attention_anchor_count: int = DEFAULT_ATTENTION_ANCHOR_COUNT
    attention_residual_threshold: float = (
        DEFAULT_ATTENTION_RESIDUAL_THRESHOLD
    )
    attention_minimum_inlier_ratio: float = (
        DEFAULT_ATTENTION_MINIMUM_INLIER_RATIO
    )
    attention_backtracking_factor: float = DEFAULT_ATTENTION_BACKTRACKING_FACTOR
    attention_backtracking_maximum_steps: int = (
        DEFAULT_ATTENTION_BACKTRACKING_MAXIMUM_STEPS
    )
    minimum_final_image_attention_score_gain: float = (
        DEFAULT_MINIMUM_FINAL_IMAGE_ATTENTION_SCORE_GAIN
    )
    tail_fraction: float = DEFAULT_TAIL_FRACTION
    keyed_prg_version: str = DEFAULT_KEYED_PRG_VERSION
    minimum_projection_energy_retention: float = DEFAULT_MINIMUM_PROJECTION_ENERGY_RETENTION
    maximum_relative_response_residual: float = DEFAULT_MAXIMUM_RELATIVE_RESPONSE_RESIDUAL
    maximum_quantized_write_relative_jacobian_response: float = (
        DEFAULT_MAXIMUM_QUANTIZED_WRITE_RELATIVE_JACOBIAN_RESPONSE
    )
    quantized_branch_composition_protocol: str = (
        DEFAULT_QUANTIZED_BRANCH_COMPOSITION_PROTOCOL
    )
    quantized_branch_composition_order: tuple[str, ...] = (
        DEFAULT_QUANTIZED_BRANCH_COMPOSITION_ORDER
    )
    combined_budget_envelope_rule: str = DEFAULT_COMBINED_BUDGET_ENVELOPE_RULE
    quantized_budget_envelope_absolute_tolerance: float = (
        DEFAULT_QUANTIZED_BUDGET_ENVELOPE_ABSOLUTE_TOLERANCE
    )
    quantized_budget_envelope_backtracking_factor: float = (
        DEFAULT_QUANTIZED_BUDGET_ENVELOPE_BACKTRACKING_FACTOR
    )
    quantized_budget_envelope_backtracking_maximum_steps: int = (
        DEFAULT_QUANTIZED_BUDGET_ENVELOPE_BACKTRACKING_MAXIMUM_STEPS
    )
    null_space_cg_max_iterations: int = DEFAULT_NULL_SPACE_CG_MAX_ITERATIONS
    null_space_cg_relative_tolerance: float = (
        DEFAULT_NULL_SPACE_CG_RELATIVE_TOLERANCE
    )
    minimum_semantic_preservation_cosine: float = (
        DEFAULT_MINIMUM_SEMANTIC_PRESERVATION_COSINE
    )
    maximum_handcrafted_structure_feature_relative_drift: float = (
        DEFAULT_MAXIMUM_HANDCRAFTED_STRUCTURE_FEATURE_RELATIVE_DRIFT
    )
    public_detection_schedule_index: int = (
        DEFAULT_PUBLIC_DETECTION_SCHEDULE_INDEX
    )
    public_detection_noise_prg_protocol: str = (
        DEFAULT_PUBLIC_DETECTION_NOISE_PRG_PROTOCOL
    )
    public_detection_noise_domain: str = (
        DEFAULT_PUBLIC_DETECTION_NOISE_DOMAIN
    )
    public_detection_conditioning_protocol: str = (
        DEFAULT_PUBLIC_DETECTION_CONDITIONING_PROTOCOL
    )
    public_detection_condition_text: str = (
        DEFAULT_PUBLIC_DETECTION_CONDITION_TEXT
    )
    max_attention_tokens: int = DEFAULT_MAX_ATTENTION_TOKENS
    attention_module_names: tuple[str, ...] = DEFAULT_ATTENTION_MODULE_NAMES
    attention_alignment_layer_selection_rule: str = (
        DEFAULT_ATTENTION_ALIGNMENT_LAYER_SELECTION_RULE
    )
    image_alignment_resampling_mode: str = (
        DEFAULT_IMAGE_ALIGNMENT_RESAMPLING_MODE
    )
    image_alignment_padding_mode: str = DEFAULT_IMAGE_ALIGNMENT_PADDING_MODE
    image_alignment_quantization_protocol: str = (
        DEFAULT_IMAGE_ALIGNMENT_QUANTIZATION_PROTOCOL
    )
    attention_coordinate_convention: str = (
        DEFAULT_ATTENTION_COORDINATE_CONVENTION
    )
    attention_grid_align_corners: bool = (
        DEFAULT_ATTENTION_GRID_ALIGN_CORNERS
    )

    def __post_init__(self) -> None:
        """集中校验内容载体维度边界。

        content_basis_rank 是检测统计的有效自由度。该值必须显著大于早期
        诊断用 4 维稀疏设置, 否则 clean negative 的随机高分尾部会抬高
        fixed-FPR 阈值, 造成真实 positive 难以越过阈值。
        """

        validate_attention_alignment_gate(
            self.attention_anchor_count,
            self.attention_residual_threshold,
            self.attention_minimum_inlier_ratio,
        )
        LowFrequencyCarrierConfig(
            kernel_size=self.lf_kernel_size,
            stride=self.lf_stride,
            padding=self.lf_padding,
            boundary_mode=self.lf_boundary_mode,
            ceil_mode=self.lf_ceil_mode,
            count_include_pad=self.lf_count_include_pad,
            divisor_override=self.lf_divisor_override,
        )
        if (
            type(self.lf_detection_score_weight) is not float
            or type(self.tail_robust_detection_score_weight) is not float
        ):
            raise TypeError("论文内容检测分支权重必须为精确 float")
        expected_method_settings = (
            _FORMAL_METHOD_DEFAULTS.paper_method_settings()
        )
        actual_method_settings = {
            field_name: getattr(self, field_name)
            for field_name in SHARED_METHOD_SETTING_FIELDS
        }
        drifted_method_fields = tuple(
            field_name
            for field_name in SHARED_METHOD_SETTING_FIELDS
            if actual_method_settings[field_name]
            != expected_method_settings[field_name]
        )
        if drifted_method_fields:
            raise ValueError(
                "论文运行方法设置必须精确继承 configs/model_sd35.yaml: "
                + ", ".join(drifted_method_fields)
            )
        if self.jacobian_candidate_count < self.null_space_rank or self.null_space_rank <= 0:
            raise ValueError("jacobian_candidate_count 必须不小于正的 null_space_rank")
        if type(self.tail_fraction) is not float or not 0.0 < self.tail_fraction <= 1.0:
            raise ValueError("tail_fraction 必须为 (0, 1] 内的精确 float")
        require_supported_keyed_prg_version(self.keyed_prg_version)
        if not 0.0 < self.attention_stable_token_fraction <= 1.0:
            raise ValueError(
                "attention_stable_token_fraction 必须位于 (0, 1]"
            )
        if not 0.0 <= self.attention_unstable_pair_weight < 1.0:
            raise ValueError(
                "attention_unstable_pair_weight 必须位于 [0, 1)"
            )
        if (
            self.attention_relation_component_weights
            != DEFAULT_ATTENTION_RELATION_COMPONENT_WEIGHTS
        ):
            raise ValueError("论文运行必须使用四个等权注意力关系分量")
        if (
            not math.isfinite(self.minimum_final_image_attention_score_gain)
            or self.minimum_final_image_attention_score_gain <= 0.0
        ):
            raise ValueError(
                "minimum_final_image_attention_score_gain 必须为正有限数"
            )
        if not 0.0 < self.minimum_projection_energy_retention <= 1.0:
            raise ValueError("minimum_projection_energy_retention 必须位于 (0, 1]")
        if not 0.0 < self.maximum_relative_response_residual <= 1.0:
            raise ValueError("maximum_relative_response_residual 必须位于 (0, 1]")
        if not 0.0 < self.maximum_quantized_write_relative_jacobian_response <= 1.0:
            raise ValueError(
                "maximum_quantized_write_relative_jacobian_response 必须位于 (0, 1]"
            )
        if self.null_space_cg_max_iterations <= 0:
            raise ValueError("null_space_cg_max_iterations 必须为正整数")
        if not 0.0 < self.null_space_cg_relative_tolerance < 1.0:
            raise ValueError("null_space_cg_relative_tolerance 必须位于 (0, 1)")
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
        if self.max_attention_tokens < 4 or len(self.attention_module_names) < 2:
            raise ValueError("注意力几何配置不能退化为单层或过短 token 近似")
        if (
            self.attention_module_names != DEFAULT_ATTENTION_MODULE_NAMES
            or self.attention_alignment_layer_selection_rule
            != DEFAULT_ATTENTION_ALIGNMENT_LAYER_SELECTION_RULE
            or self.image_alignment_resampling_mode
            != DEFAULT_IMAGE_ALIGNMENT_RESAMPLING_MODE
            or self.image_alignment_padding_mode
            != DEFAULT_IMAGE_ALIGNMENT_PADDING_MODE
            or self.image_alignment_quantization_protocol
            != DEFAULT_IMAGE_ALIGNMENT_QUANTIZATION_PROTOCOL
            or self.attention_coordinate_convention
            != DEFAULT_ATTENTION_COORDINATE_CONVENTION
            or self.attention_grid_align_corners
            is not DEFAULT_ATTENTION_GRID_ALIGN_CORNERS
        ):
            raise ValueError("论文运行必须使用统一冻结的注意力层与坐标约定")
        repeat = resolve_formal_randomization_repeat(
            self.randomization_repeat_id
        )
        protocol = formal_randomization_protocol_record()
        if (
            self.generation_seed_index != repeat.generation_seed_index
            or self.generation_seed_offset != repeat.generation_seed_offset
            or self.watermark_key_index != repeat.watermark_key_index
            or self.formal_randomization_protocol_digest
            != protocol["formal_randomization_protocol_digest"]
            or self.formal_randomization_repeat_count
            != int(protocol["crossed_repeat_count"])
        ):
            raise ValueError("论文运行随机化重复未精确匹配正式注册表")

    def to_dict(self) -> dict[str, Any]:
        """转换为 JSON 兼容字典, 便于写入 manifest 或 Notebook 日志。"""

        return asdict(self)

    def drive_dir(self, child_name: str) -> str:
        """根据统一 Drive 根目录生成某个 workflow 的输出目录。"""

        return (
            f"{self.drive_result_root.rstrip('/')}/randomization_repeats/"
            f"{self.randomization_repeat_id}/{child_name.strip('/')}"
        )

    def formal_randomization_identity(
        self,
        *,
        base_seed: int,
        prompt_index: int,
        root_key_material: str,
    ) -> dict[str, Any]:
        """返回当前 Prompt 在活动交叉重复中的随机身份."""

        repeat = resolve_formal_randomization_repeat(
            self.randomization_repeat_id
        )
        return build_formal_randomization_identity(
            base_seed=base_seed,
            prompt_index=prompt_index,
            root_key_material=root_key_material,
            repeat=repeat,
        )


@dataclass(frozen=True)
class PaperRunPromptContract:
    """显式声明测试注入或正式注册表约束的 Prompt 文件身份。"""

    run_name: str
    prompt_file: str
    expected_prompt_count: int
    prompt_file_sha256: str

    def __post_init__(self) -> None:
        """在配置边界校验数量和 SHA-256 身份。"""

        if self.expected_prompt_count <= 0:
            raise ValueError("expected_prompt_count 必须为正整数")
        if len(self.prompt_file_sha256) != 64 or any(
            character not in "0123456789abcdef"
            for character in self.prompt_file_sha256
        ):
            raise ValueError("prompt_file_sha256 必须是小写 SHA-256")


def normalize_paper_run_name(value: str | None) -> str:
    """解析论文运行层级名称。"""

    resolved = (value or PROBE_PAPER_RUN_NAME).strip()
    if resolved not in RUN_DEFAULTS:
        raise ValueError(f"未知论文运行层级: {resolved}")
    return resolved


def validate_frozen_paper_run_target_fpr(
    paper_run_name: str,
    target_fpr: float,
) -> float:
    """返回运行层级冻结的 FPR, 并拒绝底层 API 改写统计工作点."""

    run_name = normalize_paper_run_name(paper_run_name)
    if isinstance(target_fpr, bool) or not isinstance(target_fpr, (int, float)):
        raise TypeError("target_fpr 必须是有限数值")
    resolved = float(target_fpr)
    expected = float(RUN_DEFAULTS[run_name]["target_fpr"])
    if not math.isfinite(resolved) or not math.isclose(
        resolved,
        expected,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            f"{run_name} 的 target_fpr 必须使用冻结值 {expected}"
        )
    return expected


def _file_sha256(path: Path) -> str:
    """计算 Prompt 文件的字节级 SHA-256。"""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _formal_prompt_contract(
    root: str | Path,
    run_name: str,
) -> PaperRunPromptContract:
    """从受治理注册表加载正式 Prompt 数量和文件摘要。

    ``root`` 在产物构建测试中可能只表示隔离输出根目录, 并不包含项目配置。
    因此该函数与正式方法 YAML 的解析规则保持一致: 目标根目录存在注册表时
    必须使用目标注册表; 目标根目录没有注册表时, 使用当前代码包内随提交固定
    的注册表。该回退不会接受外部同名文件, 因为后续仍会核验规范路径、数量和
    字节级 SHA-256。
    """

    root_path = Path(root).resolve()
    package_root = Path(__file__).resolve().parents[2]
    requested_registry_path = (
        root_path / "configs" / "prompt_source_registry.json"
    )
    registry_path = (
        requested_registry_path
        if requested_registry_path.is_file()
        else package_root / "configs" / "prompt_source_registry.json"
    )
    if not registry_path.is_file():
        raise FileNotFoundError("正式运行缺少 configs/prompt_source_registry.json")
    registry_root = registry_path.parent.parent
    audit_governed_prompt_set(registry_root, run_name)
    registry = json.loads(registry_path.read_text(encoding="utf-8-sig"))
    record = registry.get("prompt_sets", {}).get(run_name)
    if not isinstance(record, dict):
        raise ValueError("Prompt 注册表缺少当前论文运行层级")
    expected_prompt_count = RUN_EXPECTED_PROMPT_COUNTS[run_name]
    if record.get("result_count") != expected_prompt_count:
        raise ValueError("Prompt 注册表数量与论文运行层级不一致")
    if record.get("prompt_file") != RUN_DEFAULTS[run_name]["prompt_file"]:
        raise ValueError("Prompt 注册表路径与论文运行层级不一致")
    return PaperRunPromptContract(
        run_name=run_name,
        prompt_file=str(RUN_DEFAULTS[run_name]["prompt_file"]),
        expected_prompt_count=expected_prompt_count,
        prompt_file_sha256=str(record.get("prompt_file_sha256", "")),
    )


def _validate_prompt_contract(
    root: str | Path,
    contract: PaperRunPromptContract,
) -> tuple[str, int]:
    """要求 Prompt 路径、数量和字节摘要同时精确匹配。

    调用方根目录包含规范路径时必须核验该文件; 仅将 ``root`` 用作隔离产物
    根目录且没有 Prompt 文件时, 才核验当前代码包内随提交固定的规范文件。
    这一解析次序可以让通用产物构建器脱离仓库根目录复用, 同时保证任何显式
    提供的同名文件都不能绕过摘要校验。
    """

    root_path = Path(root).resolve()
    package_root = Path(__file__).resolve().parents[2]
    prompt_path = Path(contract.prompt_file)
    if prompt_path.is_absolute():
        resolved_path = prompt_path.resolve()
        try:
            resolved_path.relative_to(root_path)
        except ValueError as exc:
            raise ValueError("Prompt 文件必须位于显式配置根目录内") from exc
    else:
        requested_path = (root_path / prompt_path).resolve()
        packaged_path = (package_root / prompt_path).resolve()
        resolved_path = requested_path if requested_path.is_file() else packaged_path
    if not resolved_path.is_file():
        raise FileNotFoundError(f"Prompt 文件不存在: {contract.prompt_file}")
    prompt_count = len(read_prompt_file(resolved_path))
    if prompt_count != contract.expected_prompt_count:
        raise ValueError("Prompt 文件实际数量与受治理数量不一致")
    if _file_sha256(resolved_path) != contract.prompt_file_sha256:
        raise ValueError("Prompt 文件 SHA-256 与受治理摘要不一致")
    return contract.prompt_file, prompt_count


def parse_record_limit(value: str | int | None, *, prompt_count: int, default_value: str | int | None = "all") -> int:
    """解析样本或记录上限。

    `all`、`none`、`unlimited` 和空字符串表示使用当前 prompt 文件的全部数量。
    该函数属于配置解析层, 用于避免业务函数内部重复实现同类边界处理。
    """

    raw_value = default_value if value is None else value
    if isinstance(raw_value, str):
        normalized = raw_value.strip().lower()
        if normalized in UNBOUNDED_LIMIT_TOKENS:
            return int(prompt_count)
        resolved = int(normalized)
    else:
        resolved = int(raw_value)
    if resolved <= 0:
        return int(prompt_count)
    return resolved


def derive_minimum_clean_negative_count(prompt_count: int, target_fpr: float) -> int:
    """从 Prompt 总量派生完整 test split 的 clean negative 门禁。

    该函数属于配置解析层。probe_paper、pilot_paper 与 full_paper 不再各自硬编码
    clean negative 门禁, 而是统一要求完整 test split。70、700、7000个 Prompt
    分别对应34、340、3400个 test 样本。三个运行层级使用同一 FPR=0.1
    工作点; 更大样本只提高统计强度, 不改变 detector 决策协议。
    """

    if prompt_count <= 0:
        raise ValueError("prompt_count 必须为正整数")
    if not 0.0 < target_fpr < 1.0:
        raise ValueError("target_fpr 必须位于 (0, 1)")
    test_split_count = max(1, int(build_group_split_counts(prompt_count)["test"]))
    return test_split_count


def derive_dataset_level_quality_minimum_count(prompt_count: int) -> int:
    """要求正式 FID/KID 覆盖当前运行层级的全部 Prompt 图像对。"""

    if prompt_count <= 0:
        raise ValueError("prompt_count 必须为正整数")
    return int(prompt_count)


def build_paper_run_config(
    root: str | Path = ".",
    *,
    prompt_contract: PaperRunPromptContract | None = None,
) -> PaperRunConfig:
    """从运行规模环境变量和唯一方法 YAML 构建论文配置。"""

    run_name = normalize_paper_run_name(os.environ.get("SLM_WM_PAPER_RUN_NAME"))
    defaults = RUN_DEFAULTS[run_name]
    # 方法配置属于当前可执行代码包, 不属于可替换的结果根目录。
    method_settings = _FORMAL_METHOD_DEFAULTS.paper_method_settings()
    prompt_set = os.environ.get("SLM_WM_PROMPT_SET", str(defaults["prompt_set"]))
    resolved_prompt_contract = prompt_contract or _formal_prompt_contract(
        root,
        run_name,
    )
    if resolved_prompt_contract.run_name != run_name:
        raise ValueError("Prompt contract 必须与当前论文运行层级一致")
    if prompt_contract is None and Path(
        resolved_prompt_contract.prompt_file
    ).as_posix() != Path(str(defaults["prompt_file"])).as_posix():
        raise ValueError("正式 Prompt contract 路径不是规范运行路径")
    prompt_file = os.environ.get(
        "SLM_WM_PROMPT_FILE",
        resolved_prompt_contract.prompt_file,
    )
    if prompt_set != str(defaults["prompt_set"]):
        raise ValueError("SLM_WM_PROMPT_SET 必须与 SLM_WM_PAPER_RUN_NAME 对应的论文运行层级一致")
    if Path(prompt_file).as_posix() != Path(
        resolved_prompt_contract.prompt_file
    ).as_posix():
        raise ValueError("SLM_WM_PROMPT_FILE 必须精确匹配受治理 Prompt 路径")
    prompt_file, prompt_count = _validate_prompt_contract(
        root,
        resolved_prompt_contract,
    )
    sample_count = parse_record_limit(
        os.environ.get("SLM_WM_PAPER_RUN_SAMPLE_COUNT", str(defaults["sample_count"])),
        prompt_count=prompt_count,
        default_value=str(defaults["sample_count"]),
    )
    target_fpr = float(defaults.get("target_fpr", DEFAULT_TARGET_FPR))
    expected_prompt_count = RUN_EXPECTED_PROMPT_COUNTS[run_name]
    derived_minimum_clean_negative_count = derive_minimum_clean_negative_count(expected_prompt_count, target_fpr)
    derived_dataset_level_quality_minimum_count = derive_dataset_level_quality_minimum_count(expected_prompt_count)
    repeat = resolve_formal_randomization_repeat(
        os.environ.get(
            "SLM_WM_RANDOMIZATION_REPEAT_ID",
            DEFAULT_FORMAL_RANDOMIZATION_REPEAT_ID,
        )
    )
    randomization_protocol = formal_randomization_protocol_record()
    return PaperRunConfig(
        run_name=run_name,
        protocol_profile=str(defaults["protocol_profile"]),
        prompt_set=prompt_set,
        prompt_file=prompt_file,
        prompt_count=prompt_count,
        sample_count=sample_count,
        drive_result_root=os.environ.get("SLM_WM_DRIVE_RESULT_ROOT", str(defaults["drive_result_root"])),
        target_fpr=target_fpr,
        minimum_clean_negative_count=derived_minimum_clean_negative_count,
        dataset_level_quality_minimum_count=derived_dataset_level_quality_minimum_count,
        randomization_repeat_id=repeat.randomization_repeat_id,
        generation_seed_index=repeat.generation_seed_index,
        generation_seed_offset=repeat.generation_seed_offset,
        watermark_key_index=repeat.watermark_key_index,
        formal_randomization_protocol_digest=randomization_protocol[
            "formal_randomization_protocol_digest"
        ],
        formal_randomization_repeat_count=len(formal_randomization_repeats()),
        **method_settings,
    )


def resolve_count_from_environment(
    env_name: str,
    *,
    root: str | Path = ".",
    default_value: str | int | None = None,
    prompt_contract: PaperRunPromptContract | None = None,
) -> int:
    """按当前论文运行配置解析某个计数环境变量。"""

    paper_run = build_paper_run_config(
        root,
        prompt_contract=prompt_contract,
    )
    return parse_record_limit(
        os.environ.get(env_name),
        prompt_count=paper_run.prompt_count,
        default_value=paper_run.sample_count if default_value is None else default_value,
    )


def shared_method_settings(config: PaperRunConfig) -> dict[str, Any]:
    """返回应在各论文运行层级间保持一致的方法级设置。

    fixed-FPR 门禁需要的最小样本数属于协议规模约束, 不属于方法机制本身。
    probe_paper、pilot_paper 与 full_paper 均使用同一方法设置和 FPR=0.1
    工作点; 三者只通过 Prompt 数量表达不同统计强度。
    """

    payload = config.to_dict()
    return {field_name: payload[field_name] for field_name in SHARED_METHOD_SETTING_FIELDS}


def shared_experiment_settings(config: PaperRunConfig) -> dict[str, Any]:
    """返回三个论文层级必须共享的随机化实验设置."""

    payload = config.to_dict()
    return {
        field_name: payload[field_name]
        for field_name in SHARED_EXPERIMENT_SETTING_FIELDS
    }
