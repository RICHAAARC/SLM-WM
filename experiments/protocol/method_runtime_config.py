"""加载正式 SLM-WM 方法配置并约束配置来源唯一性。

该模块只解析方法与模型身份, 不解析论文运行规模、输出目录或设备选择。
因此 probe_paper、pilot_paper 和 full_paper 可以共享同一份方法配置, 同时由
``paper_run_config`` 独立控制 Prompt 数量与 fixed-FPR 统计强度。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
from typing import Any

import yaml

from experiments.runtime.model_sources import require_registered_model_reference
from main.core.digest import build_stable_digest
from main.core.keyed_prg import require_supported_keyed_prg_version
from main.methods.carrier.keyed_tensor import LowFrequencyCarrierConfig
from main.methods.detection import ATTENTION_ALIGNMENT_LAYER_SELECTION_RULE
from main.methods.geometry import (
    ATTENTION_ALIGNMENT_ANCHOR_COUNT,
    ATTENTION_ALIGNMENT_MINIMUM_INLIER_RATIO,
    ATTENTION_ALIGNMENT_RESIDUAL_THRESHOLD,
    ATTENTION_COORDINATE_CONVENTION,
    ATTENTION_GRID_ALIGN_CORNERS,
    ATTENTION_OPERATOR_SCHEDULE_INDEX,
    ATTENTION_IMAGE_PADDING_MODE,
    ATTENTION_IMAGE_QUANTIZATION_PROTOCOL,
    ATTENTION_IMAGE_RESAMPLING_MODE,
    ATTENTION_RELATION_COMPONENT_WEIGHTS,
    FROZEN_SD35_ATTENTION_MODULE_NAMES,
    validate_attention_alignment_gate,
    validate_attention_relation_component_weights,
)


FORMAL_METHOD_CONFIG_RELATIVE_PATH = Path("configs/model_sd35.yaml")
FORMAL_METHOD_PACKAGE_ROOT = Path(__file__).resolve().parents[2]
FORMAL_METHOD_CONFIG_SCHEMA = "slm_wm_formal_method_runtime_config"
FORMAL_SD35_PIPELINE_CLASS_NAME = (
    "diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3."
    "StableDiffusion3Pipeline"
)
FORMAL_SD35_VAE_CLASS_NAME = (
    "diffusers.models.autoencoders.autoencoder_kl.AutoencoderKL"
)
FORMAL_SD35_TRANSFORMER_CLASS_NAME = (
    "diffusers.models.transformers.transformer_sd3.SD3Transformer2DModel"
)
FORMAL_SD35_SCHEDULER_CLASS_NAME = (
    "diffusers.schedulers.scheduling_flow_match_euler_discrete."
    "FlowMatchEulerDiscreteScheduler"
)
FORMAL_PUBLIC_DETECTION_NOISE_DOMAIN = (
    "public_image_only_qk_detection_noise"
)
FORMAL_PUBLIC_DETECTION_CONDITIONING_PROTOCOL = (
    "sd3_empty_text_triplet_without_cfg"
)


@dataclass(frozen=True)
class FormalBranchRiskConfig:
    """保存一个正式载体分支的风险公式常量。"""

    local_contrast_risk_weight: float
    semantic_weight: float
    texture_weight: float
    adjacent_step_instability_weight: float
    attention_instability_weight: float
    texture_preference: str
    eligibility_threshold: float
    budget_floor: float
    budget_ceiling: float
    budget_gain: float

    def __post_init__(self) -> None:
        """集中校验风险权重、严格资格阈值和绝对预算范围。"""

        weights = (
            self.local_contrast_risk_weight,
            self.semantic_weight,
            self.texture_weight,
            self.adjacent_step_instability_weight,
            self.attention_instability_weight,
        )
        values = (
            *weights,
            self.eligibility_threshold,
            self.budget_floor,
            self.budget_ceiling,
            self.budget_gain,
        )
        if any(not math.isfinite(value) for value in values):
            raise ValueError("正式分支风险配置必须全部为有限数")
        if any(value < 0.0 for value in weights) or sum(weights) <= 0.0:
            raise ValueError("正式分支风险权重必须非负且至少一个大于 0")
        if self.texture_preference not in {"avoid", "prefer", "neutral"}:
            raise ValueError(
                "正式 texture_preference 必须为 avoid、prefer 或 neutral"
            )
        if not 0.0 <= self.eligibility_threshold <= 1.0:
            raise ValueError("正式 eligibility_threshold 必须位于 [0, 1]")
        if not 0.0 <= self.budget_floor < self.budget_ceiling <= 1.0:
            raise ValueError("正式风险预算必须满足 0 <= floor < ceiling <= 1")
        if self.budget_gain < 0.0:
            raise ValueError("正式 budget_gain 必须非负")


FORMAL_LF_CONTENT_RISK_CONFIG = FormalBranchRiskConfig(
    local_contrast_risk_weight=0.30,
    semantic_weight=0.30,
    texture_weight=0.20,
    adjacent_step_instability_weight=0.20,
    attention_instability_weight=0.0,
    texture_preference="avoid",
    eligibility_threshold=0.55,
    budget_floor=0.05,
    budget_ceiling=1.0,
    budget_gain=0.70,
)
FORMAL_TAIL_ROBUST_RISK_CONFIG = FormalBranchRiskConfig(
    local_contrast_risk_weight=0.25,
    semantic_weight=0.25,
    texture_weight=0.30,
    adjacent_step_instability_weight=0.20,
    attention_instability_weight=0.0,
    texture_preference="prefer",
    eligibility_threshold=0.55,
    budget_floor=0.05,
    budget_ceiling=1.0,
    budget_gain=0.70,
)
FORMAL_ATTENTION_GEOMETRY_RISK_CONFIG = FormalBranchRiskConfig(
    local_contrast_risk_weight=0.20,
    semantic_weight=0.25,
    texture_weight=0.05,
    adjacent_step_instability_weight=0.20,
    attention_instability_weight=0.30,
    texture_preference="neutral",
    eligibility_threshold=0.55,
    budget_floor=0.05,
    budget_ceiling=1.0,
    budget_gain=0.70,
)

@dataclass(frozen=True)
class FormalMethodRuntimeConfig:
    """保存由 YAML 唯一确定的正式方法参数。"""

    model_family: str
    model_id: str
    model_revision: str
    vision_model_id: str
    vision_model_revision: str
    pipeline_class_name: str
    vae_class_name: str
    transformer_class_name: str
    scheduler_class_name: str
    vae_scaling_factor: float
    vae_shift_factor: float
    latent_torch_dtype: str
    vision_torch_dtype: str
    backend_mode: str
    prompt: str
    negative_prompt: str
    seed: int
    width: int
    height: int
    inference_steps: int
    guidance_scale: float
    detector_input_access_mode: str
    risk_signal_calibration_protocol: str
    risk_image_signal_interpolation_mode: str
    risk_image_signal_align_corners: bool
    risk_attention_signal_interpolation_mode: str
    risk_attention_signal_align_corners: bool
    risk_neutral_texture_value: float
    risk_eligibility_comparison: str
    risk_budget_broadcast_protocol: str
    risk_zero_support_protocol: str
    risk_bounded_scale_protocol: str
    risk_bounded_scale_direction_epsilon: float
    lf_content_risk_config: FormalBranchRiskConfig
    tail_robust_risk_config: FormalBranchRiskConfig
    attention_geometry_risk_config: FormalBranchRiskConfig
    lf_relative_strength: float
    tail_relative_strength: float
    attention_relative_strength: float
    lf_kernel_size: int
    lf_stride: int
    lf_padding: int
    lf_boundary_mode: str
    lf_ceil_mode: bool
    lf_count_include_pad: bool
    lf_divisor_override: int | None
    lf_detection_score_weight: float
    tail_robust_detection_score_weight: float
    attention_stable_token_fraction: float
    attention_unstable_pair_weight: float
    attention_relation_component_weights: tuple[float, ...]
    attention_anchor_count: int
    attention_residual_threshold: float
    attention_minimum_inlier_ratio: float
    attention_backtracking_factor: float
    attention_backtracking_maximum_steps: int
    minimum_final_image_attention_score_gain: float
    tail_fraction: float
    keyed_prg_version: str
    quantized_branch_composition_protocol: str
    quantized_branch_composition_order: tuple[str, ...]
    combined_budget_envelope_rule: str
    quantized_budget_envelope_absolute_tolerance: float
    quantized_budget_envelope_backtracking_factor: float
    quantized_budget_envelope_backtracking_maximum_steps: int
    minimum_semantic_preservation_cosine: float
    maximum_handcrafted_structure_feature_relative_drift: float
    injection_step_indices: tuple[int, ...]
    attention_operator_schedule_index: int
    public_detection_schedule_index: int
    public_detection_noise_prg_protocol: str
    public_detection_noise_domain: str
    public_detection_conditioning_protocol: str
    public_detection_condition_text: str
    max_attention_tokens: int
    attention_module_names: tuple[str, ...]
    attention_alignment_layer_selection_rule: str
    image_alignment_resampling_mode: str
    image_alignment_padding_mode: str
    image_alignment_quantization_protocol: str
    attention_coordinate_convention: str
    attention_grid_align_corners: bool
    diffusion_attacks_enabled: bool

    def __post_init__(self) -> None:
        """集中校验模型身份和关键方法边界。"""

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
            self.pipeline_class_name != FORMAL_SD35_PIPELINE_CLASS_NAME
            or self.vae_class_name != FORMAL_SD35_VAE_CLASS_NAME
            or self.transformer_class_name
            != FORMAL_SD35_TRANSFORMER_CLASS_NAME
            or self.scheduler_class_name != FORMAL_SD35_SCHEDULER_CLASS_NAME
        ):
            raise ValueError("正式 SD3.5 pipeline 组件类身份发生漂移")
        if (
            not math.isfinite(self.vae_scaling_factor)
            or not math.isfinite(self.vae_shift_factor)
            or self.vae_scaling_factor != 1.5305
            or self.vae_shift_factor != 0.0609
        ):
            raise ValueError("正式 SD3.5 VAE scaling_factor 或 shift_factor 发生漂移")
        if (
            self.latent_torch_dtype != "float16"
            or self.vision_torch_dtype != "float32"
        ):
            raise ValueError("正式 latent 与视觉编码 dtype 必须分别为 float16 和 float32")
        if self.model_family != "sd35" or self.backend_mode != "real_diffusion":
            raise ValueError("正式方法配置必须使用 sd35 real_diffusion 后端")
        if self.detector_input_access_mode != "image_key_public_model_only":
            raise ValueError("正式检测器必须保持仅图像盲检输入制度")
        if self.risk_signal_calibration_protocol != (
            "analytic_bounded_branch_signals"
        ):
            raise ValueError("正式风险输入必须使用冻结解析范围协议")
        if (
            self.risk_image_signal_interpolation_mode != "bilinear"
            or self.risk_image_signal_align_corners is not False
            or self.risk_attention_signal_interpolation_mode != "bilinear"
            or self.risk_attention_signal_align_corners is not True
        ):
            raise ValueError("正式风险图插值模式与 align_corners 约定不匹配")
        if (
            not math.isfinite(self.risk_neutral_texture_value)
            or self.risk_neutral_texture_value != 0.5
        ):
            raise ValueError("正式 neutral texture 风险项必须固定为 0.5")
        if self.risk_eligibility_comparison != "strict_less_than":
            raise ValueError("正式风险资格集合必须使用严格小于阈值")
        if self.risk_budget_broadcast_protocol != (
            "per_sample_hw_repeat_channels_nchw"
        ):
            raise ValueError("正式风险预算必须逐样本沿通道重复且不得混合 batch")
        if self.risk_zero_support_protocol != (
            "exact_zero_direction_or_fail_closed"
        ):
            raise ValueError("正式零预算支持必须对应精确零方向或直接失败")
        if self.risk_bounded_scale_protocol != (
            "direction_peak_frozen_budget_ceiling_box"
        ):
            raise ValueError("正式风险写回必须使用冻结的 RiskBoundedScale 协议")
        if (
            not math.isfinite(self.risk_bounded_scale_direction_epsilon)
            or self.risk_bounded_scale_direction_epsilon != 1e-12
        ):
            raise ValueError("正式 RiskBoundedScale 方向 epsilon 必须固定为 1e-12")
        expected_risk_configs = (
            FORMAL_LF_CONTENT_RISK_CONFIG,
            FORMAL_TAIL_ROBUST_RISK_CONFIG,
            FORMAL_ATTENTION_GEOMETRY_RISK_CONFIG,
        )
        if (
            self.lf_content_risk_config,
            self.tail_robust_risk_config,
            self.attention_geometry_risk_config,
        ) != expected_risk_configs:
            raise ValueError("正式三分支风险权重、阈值或预算常量发生漂移")
        if self.width <= 0 or self.height <= 0 or self.inference_steps <= 0:
            raise ValueError("图像尺寸和推理步数必须为正整数")
        if (
            self.injection_step_indices != (10,)
            or self.attention_operator_schedule_index
            != ATTENTION_OPERATOR_SCHEDULE_INDEX
            or self.public_detection_schedule_index
            != self.attention_operator_schedule_index
            or self.public_detection_schedule_index >= self.inference_steps
        ):
            raise ValueError(
                "正式生成必须只在索引10写回，公开检测保持冻结索引7"
            )
        if type(self.tail_fraction) is not float or not 0.0 < self.tail_fraction <= 1.0:
            raise ValueError("tail_fraction 必须为 (0, 1] 内的精确 float")
        # 核心协议对象执行精确类型校验, 防止 5.0 或 bool 通过 Python 等值比较.
        self.low_frequency_carrier_config
        if (
            self.lf_kernel_size != 5
            or self.lf_stride != 1
            or self.lf_padding != 2
            or self.lf_boundary_mode != "zero_padding"
            or self.lf_ceil_mode is not False
            or self.lf_count_include_pad is not True
            or self.lf_divisor_override is not None
        ):
            raise ValueError("正式 LF 二维平均低通核、步幅、填充或边界协议发生漂移")
        if (
            type(self.lf_detection_score_weight) is not float
            or type(self.tail_robust_detection_score_weight) is not float
            or not math.isfinite(self.lf_detection_score_weight)
            or not math.isfinite(self.tail_robust_detection_score_weight)
            or self.lf_detection_score_weight != 0.70
            or self.tail_robust_detection_score_weight != 0.30
        ):
            raise ValueError("正式内容检测分支权重必须固定为 0.70 和 0.30")
        require_supported_keyed_prg_version(self.keyed_prg_version)
        require_supported_keyed_prg_version(
            self.public_detection_noise_prg_protocol
        )
        if (
            self.public_detection_noise_prg_protocol != self.keyed_prg_version
            or self.public_detection_noise_domain
            != FORMAL_PUBLIC_DETECTION_NOISE_DOMAIN
            or self.public_detection_conditioning_protocol
            != FORMAL_PUBLIC_DETECTION_CONDITIONING_PROTOCOL
            or self.public_detection_condition_text != ""
        ):
            raise ValueError("公开仅图像检测的 PRG domain 或空文本条件发生漂移")
        if not 0.0 < self.attention_stable_token_fraction <= 1.0:
            raise ValueError(
                "attention_stable_token_fraction 必须位于 (0, 1]"
            )
        if not 0.0 <= self.attention_unstable_pair_weight < 1.0:
            raise ValueError(
                "attention_unstable_pair_weight 必须位于 [0, 1)"
            )
        component_weights = validate_attention_relation_component_weights(
            self.attention_relation_component_weights
        )
        if component_weights != ATTENTION_RELATION_COMPONENT_WEIGHTS:
            raise ValueError("正式完整方法必须启用四个等权注意力关系分量")
        validate_attention_alignment_gate(
            self.attention_anchor_count,
            self.attention_residual_threshold,
            self.attention_minimum_inlier_ratio,
        )
        if (
            self.attention_anchor_count != ATTENTION_ALIGNMENT_ANCHOR_COUNT
            or self.attention_residual_threshold
            != ATTENTION_ALIGNMENT_RESIDUAL_THRESHOLD
            or self.attention_minimum_inlier_ratio
            != ATTENTION_ALIGNMENT_MINIMUM_INLIER_RATIO
        ):
            raise ValueError("正式注意力配准锚点、残差或内点门禁发生漂移")
        if (
            self.attention_backtracking_factor != 0.5
            or self.attention_backtracking_maximum_steps != 8
        ):
            raise ValueError("正式注意力回溯必须固定为最多8次二分缩减")
        if (
            not math.isfinite(self.minimum_final_image_attention_score_gain)
            or self.minimum_final_image_attention_score_gain <= 0.0
        ):
            raise ValueError(
                "minimum_final_image_attention_score_gain 必须为正有限数"
            )
        if self.quantized_branch_composition_protocol != (
            "float32_ordered_branch_sum_add_float32_latent_single_cast"
        ):
            raise ValueError("正式三分支实际 dtype 合成协议发生漂移")
        if self.quantized_branch_composition_order != (
            "lf_content",
            "tail_robust",
            "attention_geometry",
        ):
            raise ValueError("正式三分支 actual-dtype 合成顺序发生漂移")
        if self.combined_budget_envelope_rule != "sum_active_branch_envelopes":
            raise ValueError("正式联合风险包络必须等于活动分支包络之和")
        if (
            not math.isfinite(self.quantized_budget_envelope_absolute_tolerance)
            or self.quantized_budget_envelope_absolute_tolerance != 0.0
        ):
            raise ValueError("正式量化写回风险包络不允许正的绝对超限容差")
        if (
            self.quantized_budget_envelope_backtracking_factor != 0.5
            or self.quantized_budget_envelope_backtracking_maximum_steps != 24
        ):
            raise ValueError("正式量化包络回溯必须固定为最多24次二分缩减")
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
        if self.max_attention_tokens < 4:
            raise ValueError("注意力几何配置不能退化为单层或过短 token 近似")
        if self.attention_module_names != FROZEN_SD35_ATTENTION_MODULE_NAMES:
            raise ValueError("attention_module_names 必须等于冻结 SD3.5 层顺序")
        if (
            self.attention_alignment_layer_selection_rule
            != ATTENTION_ALIGNMENT_LAYER_SELECTION_RULE
            or self.image_alignment_resampling_mode
            != ATTENTION_IMAGE_RESAMPLING_MODE
            or self.image_alignment_padding_mode
            != ATTENTION_IMAGE_PADDING_MODE
            or self.image_alignment_quantization_protocol
            != ATTENTION_IMAGE_QUANTIZATION_PROTOCOL
        ):
            raise ValueError("注意力跨层裁决或 aligned 图像重采样协议发生漂移")
        if (
            self.attention_coordinate_convention
            != ATTENTION_COORDINATE_CONVENTION
            or self.attention_grid_align_corners
            is not ATTENTION_GRID_ALIGN_CORNERS
        ):
            raise ValueError("注意力 token 与图像坐标约定必须匹配核心算子")
        if not self.diffusion_attacks_enabled:
            raise ValueError("正式方法配置必须启用真实扩散攻击协议")

    @property
    def low_frequency_carrier_config(self) -> LowFrequencyCarrierConfig:
        """把 YAML 七字段转换为核心嵌入与检测共享的 LF 协议对象."""

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
    def formal_method_config_digest(self) -> str:
        """返回与 YAML 排版和仓库绝对路径无关的正式配置摘要。"""

        return formal_method_config_digest(self)

    def paper_method_settings(self) -> dict[str, Any]:
        """返回需要写入各论文运行层级的共享方法字段。"""

        return {
            "formal_method_config_digest": self.formal_method_config_digest,
            "pipeline_class_name": self.pipeline_class_name,
            "vae_class_name": self.vae_class_name,
            "transformer_class_name": self.transformer_class_name,
            "scheduler_class_name": self.scheduler_class_name,
            "vae_scaling_factor": self.vae_scaling_factor,
            "vae_shift_factor": self.vae_shift_factor,
            "latent_torch_dtype": self.latent_torch_dtype,
            "vision_torch_dtype": self.vision_torch_dtype,
            "inference_steps": self.inference_steps,
            "guidance_scale": self.guidance_scale,
            "risk_signal_calibration_protocol": (
                self.risk_signal_calibration_protocol
            ),
            "risk_image_signal_interpolation_mode": (
                self.risk_image_signal_interpolation_mode
            ),
            "risk_image_signal_align_corners": (
                self.risk_image_signal_align_corners
            ),
            "risk_attention_signal_interpolation_mode": (
                self.risk_attention_signal_interpolation_mode
            ),
            "risk_attention_signal_align_corners": (
                self.risk_attention_signal_align_corners
            ),
            "risk_neutral_texture_value": self.risk_neutral_texture_value,
            "risk_eligibility_comparison": self.risk_eligibility_comparison,
            "risk_budget_broadcast_protocol": (
                self.risk_budget_broadcast_protocol
            ),
            "risk_zero_support_protocol": self.risk_zero_support_protocol,
            "risk_bounded_scale_protocol": self.risk_bounded_scale_protocol,
            "risk_bounded_scale_direction_epsilon": (
                self.risk_bounded_scale_direction_epsilon
            ),
            "lf_content_risk_config": asdict(self.lf_content_risk_config),
            "tail_robust_risk_config": asdict(
                self.tail_robust_risk_config
            ),
            "attention_geometry_risk_config": asdict(
                self.attention_geometry_risk_config
            ),
            "attention_injection_steps": self.injection_step_indices,
            "lf_relative_strength": self.lf_relative_strength,
            "tail_relative_strength": self.tail_relative_strength,
            "attention_relative_strength": self.attention_relative_strength,
            "lf_kernel_size": self.lf_kernel_size,
            "lf_stride": self.lf_stride,
            "lf_padding": self.lf_padding,
            "lf_boundary_mode": self.lf_boundary_mode,
            "lf_ceil_mode": self.lf_ceil_mode,
            "lf_count_include_pad": self.lf_count_include_pad,
            "lf_divisor_override": self.lf_divisor_override,
            "lf_detection_score_weight": self.lf_detection_score_weight,
            "tail_robust_detection_score_weight": (
                self.tail_robust_detection_score_weight
            ),
            "attention_stable_token_fraction": (
                self.attention_stable_token_fraction
            ),
            "attention_unstable_pair_weight": (
                self.attention_unstable_pair_weight
            ),
            "attention_relation_component_weights": (
                self.attention_relation_component_weights
            ),
            "attention_anchor_count": self.attention_anchor_count,
            "attention_residual_threshold": (
                self.attention_residual_threshold
            ),
            "attention_minimum_inlier_ratio": (
                self.attention_minimum_inlier_ratio
            ),
            "attention_backtracking_factor": (
                self.attention_backtracking_factor
            ),
            "attention_backtracking_maximum_steps": (
                self.attention_backtracking_maximum_steps
            ),
            "minimum_final_image_attention_score_gain": (
                self.minimum_final_image_attention_score_gain
            ),
            "tail_fraction": self.tail_fraction,
            "keyed_prg_version": self.keyed_prg_version,
            "quantized_branch_composition_protocol": (
                self.quantized_branch_composition_protocol
            ),
            "quantized_branch_composition_order": (
                self.quantized_branch_composition_order
            ),
            "combined_budget_envelope_rule": (
                self.combined_budget_envelope_rule
            ),
            "quantized_budget_envelope_absolute_tolerance": (
                self.quantized_budget_envelope_absolute_tolerance
            ),
            "quantized_budget_envelope_backtracking_factor": (
                self.quantized_budget_envelope_backtracking_factor
            ),
            "quantized_budget_envelope_backtracking_maximum_steps": (
                self.quantized_budget_envelope_backtracking_maximum_steps
            ),
            "minimum_semantic_preservation_cosine": (
                self.minimum_semantic_preservation_cosine
            ),
            "maximum_handcrafted_structure_feature_relative_drift": (
                self.maximum_handcrafted_structure_feature_relative_drift
            ),
            "attention_operator_schedule_index": (
                self.attention_operator_schedule_index
            ),
            "public_detection_schedule_index": (
                self.public_detection_schedule_index
            ),
            "public_detection_noise_prg_protocol": (
                self.public_detection_noise_prg_protocol
            ),
            "public_detection_noise_domain": (
                self.public_detection_noise_domain
            ),
            "public_detection_conditioning_protocol": (
                self.public_detection_conditioning_protocol
            ),
            "public_detection_condition_text": (
                self.public_detection_condition_text
            ),
            "max_attention_tokens": self.max_attention_tokens,
            "attention_module_names": self.attention_module_names,
            "attention_alignment_layer_selection_rule": (
                self.attention_alignment_layer_selection_rule
            ),
            "image_alignment_resampling_mode": (
                self.image_alignment_resampling_mode
            ),
            "image_alignment_padding_mode": (
                self.image_alignment_padding_mode
            ),
            "image_alignment_quantization_protocol": (
                self.image_alignment_quantization_protocol
            ),
            "attention_coordinate_convention": (
                self.attention_coordinate_convention
            ),
            "attention_grid_align_corners": (
                self.attention_grid_align_corners
            ),
        }


def formal_method_config_payload(
    config: FormalMethodRuntimeConfig,
) -> dict[str, Any]:
    """返回不含文件路径和派生摘要的规范正式方法配置。"""

    return {
        "formal_method_config_schema": FORMAL_METHOD_CONFIG_SCHEMA,
        "formal_method_config": asdict(config),
    }


def formal_method_config_digest(config: FormalMethodRuntimeConfig) -> str:
    """计算只由完整配置值决定的稳定 SHA-256 摘要。"""

    return build_stable_digest(formal_method_config_payload(config))


def resolve_formal_method_config_path(root: str | Path = ".") -> Path:
    """解析目标仓库内的唯一正式配置, 缺失时直接失败。"""

    requested_path = (Path(root) / FORMAL_METHOD_CONFIG_RELATIVE_PATH).resolve()
    if not requested_path.is_file():
        raise FileNotFoundError(f"正式方法配置不存在: {requested_path}")
    return requested_path


def _required_payload(payload: Any, path: Path) -> dict[str, Any]:
    """确认 YAML 根节点与正式字段完整, 避免缺失字段被隐式默认。"""

    if not isinstance(payload, dict):
        raise ValueError(f"正式方法配置必须是 YAML 映射: {path}")
    required_fields = tuple(FormalMethodRuntimeConfig.__dataclass_fields__)
    missing_fields = tuple(field for field in required_fields if field not in payload)
    if missing_fields:
        raise ValueError(f"正式方法配置缺少字段 {missing_fields}: {path}")
    unknown_fields = tuple(sorted(set(payload) - set(required_fields)))
    if unknown_fields:
        raise ValueError(f"正式方法配置包含未消费字段 {unknown_fields}: {path}")
    return payload


def load_formal_method_runtime_config(root: str | Path = ".") -> FormalMethodRuntimeConfig:
    """从 ``configs/model_sd35.yaml`` 构造经过完整校验的方法配置。"""

    path = resolve_formal_method_config_path(root)
    payload = _required_payload(yaml.safe_load(path.read_text(encoding="utf-8")), path)
    normalized = dict(payload)
    normalized["injection_step_indices"] = tuple(int(value) for value in payload["injection_step_indices"])
    normalized["attention_module_names"] = tuple(
        str(value) for value in payload["attention_module_names"]
    )
    normalized["attention_relation_component_weights"] = tuple(
        float(value)
        for value in payload["attention_relation_component_weights"]
    )
    normalized["quantized_branch_composition_order"] = tuple(
        str(value) for value in payload["quantized_branch_composition_order"]
    )
    for field_name in (
        "lf_content_risk_config",
        "tail_robust_risk_config",
        "attention_geometry_risk_config",
    ):
        field_payload = payload[field_name]
        if not isinstance(field_payload, dict):
            raise ValueError(f"正式 {field_name} 必须是 YAML 映射")
        normalized[field_name] = FormalBranchRiskConfig(**field_payload)
    return FormalMethodRuntimeConfig(**normalized)


def require_formal_method_environment_consistency(config: FormalMethodRuntimeConfig) -> None:
    """拒绝残留环境变量改变 YAML 中冻结的方法身份或超参数。

    允许环境变量重复声明相同值, 便于 Notebook 显示当前配置; 任何不同值都会
    在模型加载前失败, 从而避免不同 Colab session 形成未登记的方法分叉。
    """

    expected_values = {
        "SLM_WM_MODEL_FAMILY": config.model_family,
        "SLM_WM_MODEL_ID": config.model_id,
        "SLM_WM_MODEL_REVISION": config.model_revision,
        "SLM_WM_VISION_MODEL_ID": config.vision_model_id,
        "SLM_WM_VISION_MODEL_REVISION": config.vision_model_revision,
        "SLM_WM_PIPELINE_CLASS_NAME": config.pipeline_class_name,
        "SLM_WM_VAE_CLASS_NAME": config.vae_class_name,
        "SLM_WM_TRANSFORMER_CLASS_NAME": config.transformer_class_name,
        "SLM_WM_SCHEDULER_CLASS_NAME": config.scheduler_class_name,
        "SLM_WM_VAE_SCALING_FACTOR": str(config.vae_scaling_factor),
        "SLM_WM_VAE_SHIFT_FACTOR": str(config.vae_shift_factor),
        "SLM_WM_TORCH_DTYPE": config.latent_torch_dtype,
        "SLM_WM_VISION_TORCH_DTYPE": config.vision_torch_dtype,
        "SLM_WM_FORMAL_METHOD_CONFIG_DIGEST": (
            config.formal_method_config_digest
        ),
        "SLM_WM_SEED": str(config.seed),
        "SLM_WM_IMAGE_WIDTH": str(config.width),
        "SLM_WM_IMAGE_HEIGHT": str(config.height),
        "SLM_WM_INFERENCE_STEPS": str(config.inference_steps),
        "SLM_WM_GUIDANCE_SCALE": str(config.guidance_scale),
        "SLM_WM_RISK_SIGNAL_CALIBRATION_PROTOCOL": (
            config.risk_signal_calibration_protocol
        ),
        "SLM_WM_RISK_IMAGE_SIGNAL_INTERPOLATION_MODE": (
            config.risk_image_signal_interpolation_mode
        ),
        "SLM_WM_RISK_IMAGE_SIGNAL_ALIGN_CORNERS": (
            "1" if config.risk_image_signal_align_corners else "0"
        ),
        "SLM_WM_RISK_ATTENTION_SIGNAL_INTERPOLATION_MODE": (
            config.risk_attention_signal_interpolation_mode
        ),
        "SLM_WM_RISK_ATTENTION_SIGNAL_ALIGN_CORNERS": (
            "1" if config.risk_attention_signal_align_corners else "0"
        ),
        "SLM_WM_RISK_NEUTRAL_TEXTURE_VALUE": str(
            config.risk_neutral_texture_value
        ),
        "SLM_WM_RISK_ELIGIBILITY_COMPARISON": (
            config.risk_eligibility_comparison
        ),
        "SLM_WM_RISK_BUDGET_BROADCAST_PROTOCOL": (
            config.risk_budget_broadcast_protocol
        ),
        "SLM_WM_RISK_ZERO_SUPPORT_PROTOCOL": (
            config.risk_zero_support_protocol
        ),
        "SLM_WM_RISK_BOUNDED_SCALE_PROTOCOL": (
            config.risk_bounded_scale_protocol
        ),
        "SLM_WM_RISK_BOUNDED_SCALE_DIRECTION_EPSILON": str(
            config.risk_bounded_scale_direction_epsilon
        ),
        "SLM_WM_LF_CONTENT_RISK_CONFIG": json.dumps(
            asdict(config.lf_content_risk_config),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
        "SLM_WM_TAIL_ROBUST_RISK_CONFIG": json.dumps(
            asdict(config.tail_robust_risk_config),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
        "SLM_WM_ATTENTION_GEOMETRY_RISK_CONFIG": json.dumps(
            asdict(config.attention_geometry_risk_config),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
        "SLM_WM_ATTENTION_INJECTION_STEPS": ",".join(str(value) for value in config.injection_step_indices),
        "SLM_WM_LF_RELATIVE_STRENGTH": str(config.lf_relative_strength),
        "SLM_WM_TAIL_RELATIVE_STRENGTH": str(config.tail_relative_strength),
        "SLM_WM_ATTENTION_RELATIVE_STRENGTH": str(config.attention_relative_strength),
        "SLM_WM_LF_KERNEL_SIZE": str(config.lf_kernel_size),
        "SLM_WM_LF_STRIDE": str(config.lf_stride),
        "SLM_WM_LF_PADDING": str(config.lf_padding),
        "SLM_WM_LF_BOUNDARY_MODE": config.lf_boundary_mode,
        "SLM_WM_LF_CEIL_MODE": "1" if config.lf_ceil_mode else "0",
        "SLM_WM_LF_COUNT_INCLUDE_PAD": (
            "1" if config.lf_count_include_pad else "0"
        ),
        "SLM_WM_LF_DIVISOR_OVERRIDE": (
            "none"
            if config.lf_divisor_override is None
            else str(config.lf_divisor_override)
        ),
        "SLM_WM_LF_DETECTION_SCORE_WEIGHT": str(
            config.lf_detection_score_weight
        ),
        "SLM_WM_TAIL_ROBUST_DETECTION_SCORE_WEIGHT": str(
            config.tail_robust_detection_score_weight
        ),
        "SLM_WM_ATTENTION_STABLE_TOKEN_FRACTION": str(
            config.attention_stable_token_fraction
        ),
        "SLM_WM_ATTENTION_UNSTABLE_PAIR_WEIGHT": str(
            config.attention_unstable_pair_weight
        ),
        "SLM_WM_ATTENTION_RELATION_COMPONENT_WEIGHTS": ",".join(
            str(value)
            for value in config.attention_relation_component_weights
        ),
        "SLM_WM_ATTENTION_ANCHOR_COUNT": str(
            config.attention_anchor_count
        ),
        "SLM_WM_ATTENTION_RESIDUAL_THRESHOLD": str(
            config.attention_residual_threshold
        ),
        "SLM_WM_ATTENTION_MINIMUM_INLIER_RATIO": str(
            config.attention_minimum_inlier_ratio
        ),
        "SLM_WM_ATTENTION_BACKTRACKING_FACTOR": str(
            config.attention_backtracking_factor
        ),
        "SLM_WM_ATTENTION_BACKTRACKING_MAXIMUM_STEPS": str(
            config.attention_backtracking_maximum_steps
        ),
        "SLM_WM_MINIMUM_FINAL_IMAGE_ATTENTION_SCORE_GAIN": str(
            config.minimum_final_image_attention_score_gain
        ),
        "SLM_WM_TAIL_FRACTION": str(config.tail_fraction),
        "SLM_WM_KEYED_PRG_VERSION": config.keyed_prg_version,
        "SLM_WM_QUANTIZED_BRANCH_COMPOSITION_PROTOCOL": (
            config.quantized_branch_composition_protocol
        ),
        "SLM_WM_QUANTIZED_BRANCH_COMPOSITION_ORDER": ",".join(
            config.quantized_branch_composition_order
        ),
        "SLM_WM_COMBINED_BUDGET_ENVELOPE_RULE": (
            config.combined_budget_envelope_rule
        ),
        "SLM_WM_QUANTIZED_BUDGET_ENVELOPE_ABSOLUTE_TOLERANCE": str(
            config.quantized_budget_envelope_absolute_tolerance
        ),
        "SLM_WM_QUANTIZED_BUDGET_ENVELOPE_BACKTRACKING_FACTOR": str(
            config.quantized_budget_envelope_backtracking_factor
        ),
        "SLM_WM_QUANTIZED_BUDGET_ENVELOPE_BACKTRACKING_MAXIMUM_STEPS": str(
            config.quantized_budget_envelope_backtracking_maximum_steps
        ),
        "SLM_WM_MINIMUM_SEMANTIC_PRESERVATION_COSINE": str(
            config.minimum_semantic_preservation_cosine
        ),
        "SLM_WM_MAXIMUM_HANDCRAFTED_STRUCTURE_FEATURE_RELATIVE_DRIFT": str(
            config.maximum_handcrafted_structure_feature_relative_drift
        ),
        "SLM_WM_MAX_ATTENTION_TOKENS": str(config.max_attention_tokens),
        "SLM_WM_ATTENTION_OPERATOR_SCHEDULE_INDEX": str(
            config.attention_operator_schedule_index
        ),
        "SLM_WM_PUBLIC_DETECTION_SCHEDULE_INDEX": str(
            config.public_detection_schedule_index
        ),
        "SLM_WM_PUBLIC_DETECTION_NOISE_PRG_PROTOCOL": (
            config.public_detection_noise_prg_protocol
        ),
        "SLM_WM_PUBLIC_DETECTION_NOISE_DOMAIN": (
            config.public_detection_noise_domain
        ),
        "SLM_WM_PUBLIC_DETECTION_CONDITIONING_PROTOCOL": (
            config.public_detection_conditioning_protocol
        ),
        "SLM_WM_PUBLIC_DETECTION_CONDITION_TEXT": (
            config.public_detection_condition_text
        ),
        "SLM_WM_ATTENTION_MODULE_NAMES": ",".join(
            config.attention_module_names
        ),
        "SLM_WM_ATTENTION_COORDINATE_CONVENTION": (
            config.attention_coordinate_convention
        ),
        "SLM_WM_ATTENTION_GRID_ALIGN_CORNERS": (
            "1" if config.attention_grid_align_corners else "0"
        ),
        "SLM_WM_ENABLE_DIFFUSION_ATTACKS": "1" if config.diffusion_attacks_enabled else "0",
    }
    drift = {
        name: {"expected": expected, "actual": os.environ[name]}
        for name, expected in expected_values.items()
        if name in os.environ and os.environ[name].strip() != expected
    }
    if drift:
        raise ValueError(f"正式方法环境变量与 configs/model_sd35.yaml 不一致: {drift}")
