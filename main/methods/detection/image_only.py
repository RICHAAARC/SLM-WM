"""实现不读取生成轨迹的仅图像水印检测接口。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from main.core.digest import build_stable_digest, tensor_content_sha256
from main.methods.carrier.keyed_tensor import (
    BlindContentScore,
    LowFrequencyCarrierConfig,
    build_low_frequency_template,
    build_tail_robust_template,
    compute_blind_content_score,
    tail_robust_carrier_protocol_record,
    validate_low_frequency_carrier_protocol_record,
    validate_tail_robust_carrier_protocol_record,
)
from main.core.keyed_prg import (
    KEYED_PRG_VERSION,
    keyed_prg_protocol_record,
    require_supported_keyed_prg_version,
)
from main.methods.geometry.attention_alignment import (
    ATTENTION_IMAGE_PADDING_MODE,
    ATTENTION_IMAGE_QUANTIZATION_PROTOCOL,
    ATTENTION_IMAGE_RESAMPLING_MODE,
    AttentionAlignmentResult,
    attention_alignment_gate_record,
    recover_attention_affine_alignment,
    validate_attention_alignment_gate,
    validate_attention_alignment_record,
)
from main.methods.geometry.differentiable_attention import (
    ATTENTION_COORDINATE_CONVENTION,
    ATTENTION_GRID_ALIGN_CORNERS,
    ATTENTION_OPERATOR_SCHEDULE_INDEX,
    DIRECT_QK_RELATION_SOURCE,
    FROZEN_SD35_ATTENTION_MODULE_NAMES,
    attention_geometry_score,
    attention_relation_component_protocol,
    build_attention_relation_graph_identity,
    build_stable_attention_pair_weights,
    qk_atomic_evaluation_records_digest,
    qk_atomic_evaluation_records_ready,
    restore_transported_stable_attention_pair_weights,
    select_stable_attention_tokens,
    validate_attention_relation_component_weights,
)

ImageLatentEncoder = Callable[[Any], Any]
AttentionRecord = tuple[str, Any, tuple[int, ...]]
ImageAttentionExtractor = Callable[[Any], tuple[AttentionRecord, ...]]
ImageAligner = Callable[[Any, AttentionAlignmentResult], Any]
IMAGE_ONLY_MEASUREMENT_CONFIG_SCHEMA = (
    "slm_wm_image_only_measurement_config"
)
IMAGE_ONLY_EXTRACTION_PROFILE_SCHEMA = (
    "slm_wm_image_only_extraction_profile"
)
IMAGE_ONLY_IMAGE_PREPROCESSING_PROTOCOL = (
    "diffusers_sd3_image_processor_rgb_preprocess"
)
IMAGE_ONLY_VAE_ENCODING_PROTOCOL = (
    "sd3_vae_latent_dist_mode_shift_then_scale"
)
ATTENTION_ALIGNMENT_LAYER_SELECTION_RULE = (
    "lexicographic_objective_observation_confidence_then_frozen_layer_order"
)


@dataclass(frozen=True)
class ImageOnlyMeasurementConfig:
    """定义仅图像盲检科学证据测量所需的公开参数。

    该配置只控制图像到连续证据的测量过程, 不包含任何由 calibration
    数据决定的阈值或 rescue 窗口。最终判定由实验协议层冻结并应用,
    从类型边界上避免原始测量路径形成第二套临时检测器。
    """

    model_id: str
    attention_module_names: tuple[str, ...]
    attention_anchor_count: int
    attention_residual_threshold: float
    attention_minimum_inlier_ratio: float
    low_frequency_config: LowFrequencyCarrierConfig
    lf_weight: float
    tail_robust_weight: float
    tail_fraction: float
    keyed_prg_version: str
    attention_stable_token_fraction: float
    attention_unstable_pair_weight: float
    attention_relation_component_weights: tuple[float, ...]
    model_revision: str
    vae_class_name: str
    transformer_class_name: str
    scheduler_class_name: str
    vae_scaling_factor: float
    vae_shift_factor: float
    latent_torch_dtype: str
    width: int
    height: int
    inference_steps: int
    public_detection_schedule_index: int
    public_detection_noise_prg_protocol: str
    public_detection_noise_domain: str
    public_detection_conditioning_protocol: str
    public_detection_condition_text: str
    max_attention_tokens: int
    attention_coordinate_convention: str
    attention_grid_align_corners: bool

    def __post_init__(self) -> None:
        """集中校验阈值无关的测量协议。"""

        if type(self.model_id) is not str or not self.model_id:
            raise ValueError("model_id 必须为非空精确 str")
        for field_name in (
            "model_revision",
            "vae_class_name",
            "transformer_class_name",
            "scheduler_class_name",
            "latent_torch_dtype",
            "public_detection_noise_prg_protocol",
            "public_detection_noise_domain",
            "public_detection_conditioning_protocol",
            "attention_coordinate_convention",
        ):
            value = getattr(self, field_name)
            if type(value) is not str or not value:
                raise ValueError(f"{field_name} 必须为非空精确 str")
        if type(self.public_detection_condition_text) is not str:
            raise ValueError(
                "public_detection_condition_text 必须为精确 str"
            )
        if (
            type(self.vae_scaling_factor) is not float
            or type(self.vae_shift_factor) is not float
            or not math.isfinite(self.vae_scaling_factor)
            or not math.isfinite(self.vae_shift_factor)
            or self.vae_scaling_factor <= 0.0
        ):
            raise ValueError("VAE 缩放与平移参数必须为有效精确 float")
        for field_name in (
            "width",
            "height",
            "inference_steps",
            "max_attention_tokens",
        ):
            value = getattr(self, field_name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{field_name} 必须为正整数")
        if self.width % 8 != 0 or self.height % 8 != 0:
            raise ValueError("盲检图像宽高必须能被8整除")
        if (
            type(self.public_detection_schedule_index) is not int
            or self.public_detection_schedule_index
            != ATTENTION_OPERATOR_SCHEDULE_INDEX
            or self.public_detection_schedule_index >= self.inference_steps
        ):
            raise ValueError("公开检测 schedule 索引必须等于冻结注意力算子索引")
        if (
            self.attention_coordinate_convention
            != ATTENTION_COORDINATE_CONVENTION
            or self.attention_grid_align_corners
            is not ATTENTION_GRID_ALIGN_CORNERS
        ):
            raise ValueError("注意力坐标与 align_corners 必须等于冻结方法约定")
        if self.attention_module_names != FROZEN_SD35_ATTENTION_MODULE_NAMES:
            raise ValueError("attention_module_names 必须等于冻结 SD3.5 层顺序")
        for field_name, value in (
            (
                "attention_stable_token_fraction",
                self.attention_stable_token_fraction,
            ),
            (
                "attention_unstable_pair_weight",
                self.attention_unstable_pair_weight,
            ),
        ):
            if type(value) is not float or not math.isfinite(value):
                raise ValueError(f"{field_name} 必须为有限精确 float")
        if not isinstance(
            self.low_frequency_config,
            LowFrequencyCarrierConfig,
        ):
            raise TypeError(
                "low_frequency_config 必须为 LowFrequencyCarrierConfig"
            )
        if (
            type(self.lf_weight) is not float
            or type(self.tail_robust_weight) is not float
            or not 0.0 <= self.lf_weight <= 1.0
            or not 0.0 <= self.tail_robust_weight <= 1.0
        ):
            raise ValueError("内容分支权重必须为 [0, 1] 内的精确 float")
        if type(self.tail_fraction) is not float or not 0.0 < self.tail_fraction <= 1.0:
            raise ValueError("tail_fraction 必须位于 (0, 1]")
        if (
            type(self.attention_residual_threshold) is not float
            or type(self.attention_minimum_inlier_ratio) is not float
        ):
            raise ValueError("注意力配准连续门禁必须为精确 float")
        if (
            type(self.attention_relation_component_weights) is not tuple
            or any(
                type(value) is not float
                for value in self.attention_relation_component_weights
            )
        ):
            raise ValueError("注意力关系分量权重必须为精确 float 元组")
        validate_attention_alignment_gate(
            self.attention_anchor_count,
            self.attention_residual_threshold,
            self.attention_minimum_inlier_ratio,
        )
        require_supported_keyed_prg_version(self.keyed_prg_version)
        require_supported_keyed_prg_version(
            self.public_detection_noise_prg_protocol
        )
        if abs(self.lf_weight + self.tail_robust_weight - 1.0) > 1e-9:
            raise ValueError("内容分支权重之和必须为 1")
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


def select_image_only_alignment_candidate(
    candidates: tuple[AttentionAlignmentResult, ...],
    attention_module_names: tuple[str, ...],
) -> AttentionAlignmentResult:
    """按冻结跨层字典序选择唯一配准候选.

    每个冻结层先独立完成层内仿射搜索. 本函数随后依次比较注册目标、观测关系
    分和注册置信度；三者完全相同时选择冻结层顺序中更靠前的候选. 该显式
    裁决保证不同运行环境不会依赖容器遍历顺序产生不同 rescue 数值.
    """

    if attention_module_names != FROZEN_SD35_ATTENTION_MODULE_NAMES:
        raise ValueError("跨层配准选择必须使用冻结 SD3.5 层顺序")
    if (
        type(candidates) is not tuple
        or len(candidates) != len(attention_module_names)
        or any(
            not isinstance(candidate, AttentionAlignmentResult)
            for candidate in candidates
        )
        or tuple(candidate.layer_name for candidate in candidates)
        != attention_module_names
    ):
        raise ValueError("跨层配准候选必须与冻结层顺序逐项一致")
    return max(
        enumerate(candidates),
        key=lambda indexed: (
            indexed[1].registration_objective_score,
            indexed[1].observation_relation_score,
            indexed[1].registration_confidence,
            -indexed[0],
        ),
    )[1]


def image_only_measurement_config_identity_record(
    config: ImageOnlyMeasurementConfig,
    *,
    attention_geometry_enabled: bool,
    image_alignment_enabled: bool,
) -> dict[str, Any]:
    """构造可由原始记录独立重建的阈值无关测量配置身份."""

    if not isinstance(config, ImageOnlyMeasurementConfig):
        raise TypeError("config 必须为 ImageOnlyMeasurementConfig")
    if (
        type(attention_geometry_enabled) is not bool
        or type(image_alignment_enabled) is not bool
    ):
        raise TypeError("检测机制开关必须为精确 bool")
    if image_alignment_enabled and not attention_geometry_enabled:
        raise ValueError("图像配准必须以真实注意力几何测量为前提")
    relation_protocol = attention_relation_component_protocol(
        config.attention_relation_component_weights
    )
    prg_protocol = keyed_prg_protocol_record(config.keyed_prg_version)
    tail_protocol = tail_robust_carrier_protocol_record(
        config.tail_fraction,
        prg_version=config.keyed_prg_version,
    )
    payload = {
        "image_only_measurement_config_schema": (
            IMAGE_ONLY_MEASUREMENT_CONFIG_SCHEMA
        ),
        "model_id": config.model_id,
        "model_revision": config.model_revision,
        "image_only_extraction_profile_schema": (
            IMAGE_ONLY_EXTRACTION_PROFILE_SCHEMA
        ),
        "image_preprocessing_protocol": (
            IMAGE_ONLY_IMAGE_PREPROCESSING_PROTOCOL
        ),
        "vae_encoding_protocol": IMAGE_ONLY_VAE_ENCODING_PROTOCOL,
        "vae_class_name": config.vae_class_name,
        "transformer_class_name": config.transformer_class_name,
        "scheduler_class_name": config.scheduler_class_name,
        "vae_scaling_factor": config.vae_scaling_factor,
        "vae_shift_factor": config.vae_shift_factor,
        "latent_torch_dtype": config.latent_torch_dtype,
        "width": config.width,
        "height": config.height,
        "inference_steps": config.inference_steps,
        "public_detection_schedule_index": (
            config.public_detection_schedule_index
        ),
        "public_detection_noise_prg_protocol": (
            config.public_detection_noise_prg_protocol
        ),
        "public_detection_noise_domain": (
            config.public_detection_noise_domain
        ),
        "public_detection_conditioning_protocol": (
            config.public_detection_conditioning_protocol
        ),
        "public_detection_condition_text": (
            config.public_detection_condition_text
        ),
        "max_attention_tokens": config.max_attention_tokens,
        "attention_coordinate_convention": (
            config.attention_coordinate_convention
        ),
        "attention_grid_align_corners": (
            config.attention_grid_align_corners
        ),
        "attention_module_names": list(config.attention_module_names),
        "attention_alignment_layer_selection_rule": (
            ATTENTION_ALIGNMENT_LAYER_SELECTION_RULE
        ),
        "image_alignment_resampling_mode": ATTENTION_IMAGE_RESAMPLING_MODE,
        "image_alignment_padding_mode": ATTENTION_IMAGE_PADDING_MODE,
        "image_alignment_quantization_protocol": (
            ATTENTION_IMAGE_QUANTIZATION_PROTOCOL
        ),
        "lf_carrier_protocol_digest": (
            config.low_frequency_config.protocol_digest
        ),
        "tail_carrier_protocol_digest": tail_protocol[
            "tail_carrier_protocol_digest"
        ],
        "lf_weight": config.lf_weight,
        "tail_robust_weight": config.tail_robust_weight,
        "tail_fraction": config.tail_fraction,
        "keyed_prg_version": config.keyed_prg_version,
        "keyed_prg_protocol_digest": prg_protocol[
            "keyed_prg_protocol_digest"
        ],
        "attention_alignment_gate": attention_alignment_gate_record(
            config.attention_anchor_count,
            config.attention_residual_threshold,
            config.attention_minimum_inlier_ratio,
        ),
        "attention_stable_token_fraction": (
            config.attention_stable_token_fraction
        ),
        "attention_unstable_pair_weight": (
            config.attention_unstable_pair_weight
        ),
        "attention_relation_component_weights": list(
            config.attention_relation_component_weights
        ),
        "attention_relation_component_protocol_digest": relation_protocol[
            "attention_relation_component_protocol_digest"
        ],
        "attention_geometry_enabled": attention_geometry_enabled,
        "image_alignment_enabled": image_alignment_enabled,
    }
    return {
        **payload,
        "image_only_measurement_config_digest": build_stable_digest(payload),
    }


@dataclass(frozen=True)
class ImageOnlyMeasurementResult:
    """保存仅图像盲检产生的阈值无关连续证据。"""

    content: BlindContentScore
    lf_carrier_protocol_digest: str
    lf_template_content_sha256: str
    tail_carrier_protocol_digest: str
    tail_fraction: float
    tail_template_content_sha256: str
    tail_template_shape: list[int]
    tail_template_element_count: int
    tail_selected_element_count: int
    tail_threshold: float
    tail_retained_fraction: float
    aligned_lf_score: float | None
    aligned_tail_robust_score: float | None
    aligned_content_score: float | None
    attention_geometry_score: float | None
    raw_attention_geometry_score: float | None
    attention_sync_score: float | None
    registration_confidence: float | None
    alignment: AttentionAlignmentResult | None
    image_only_measurement_config_digest: str
    measurement_digest: str
    metadata: dict[str, Any]

    def to_record(self) -> dict[str, Any]:
        """转换成可序列化检测记录。"""

        alignment_record = None
        if self.alignment is not None:
            alignment_record = {
                **self.alignment.__dict__,
                "registration_geometry_reliable": self.alignment.geometry_reliable,
            }
        return {
            "lf_score": self.content.lf_score,
            "tail_robust_score": self.content.tail_robust_score,
            "content_score": self.content.content_score,
            "lf_weight": self.content.lf_weight,
            "tail_robust_weight": self.content.tail_robust_weight,
            "tail_fraction": self.tail_fraction,
            "lf_carrier_protocol_digest": self.lf_carrier_protocol_digest,
            "lf_template_content_sha256": self.lf_template_content_sha256,
            "tail_carrier_protocol_digest": (
                self.tail_carrier_protocol_digest
            ),
            "tail_template_content_sha256": (
                self.tail_template_content_sha256
            ),
            "tail_template_shape": list(self.tail_template_shape),
            "tail_template_element_count": (
                self.tail_template_element_count
            ),
            "tail_selected_element_count": (
                self.tail_selected_element_count
            ),
            "tail_threshold": self.tail_threshold,
            "tail_retained_fraction": self.tail_retained_fraction,
            "aligned_lf_score": self.aligned_lf_score,
            "aligned_tail_robust_score": self.aligned_tail_robust_score,
            "aligned_content_score": self.aligned_content_score,
            "attention_geometry_score": self.attention_geometry_score,
            "raw_attention_geometry_score": self.raw_attention_geometry_score,
            "attention_sync_score": self.attention_sync_score,
            "registration_confidence": self.registration_confidence,
            "alignment": alignment_record,
            "image_only_measurement_config_digest": (
                self.image_only_measurement_config_digest
            ),
            "measurement_digest": self.measurement_digest,
            "metadata": self.metadata,
        }


def recompute_image_only_measurement_digest_payload(
    record: Mapping[str, Any],
) -> dict[str, Any]:
    """从样本级连续证据重建紧凑测量记录摘要正文.

    完整测量配置只保存在顶层运行配置中. 样本记录通过配置摘要引用该配置,
    此处只绑定会改变样本分数或配准证据的字段. calibration 派生参数和
    最终判定不属于原始测量记录, 因而不得进入该摘要正文.
    """

    if not isinstance(record, Mapping):
        raise TypeError("检测记录必须为 mapping")
    metadata = record.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("检测记录缺少 metadata")
    alignment = record.get("alignment")
    return {
        "image_only_measurement_config_digest": record.get(
            "image_only_measurement_config_digest"
        ),
        "lf_carrier_protocol_digest": record.get(
            "lf_carrier_protocol_digest"
        ),
        "tail_carrier_protocol_digest": record.get(
            "tail_carrier_protocol_digest"
        ),
        "lf_template_content_sha256": record.get(
            "lf_template_content_sha256"
        ),
        "tail_template_content_sha256": record.get(
            "tail_template_content_sha256"
        ),
        "tail_template_shape": list(record.get("tail_template_shape", [])),
        "tail_template_element_count": record.get(
            "tail_template_element_count"
        ),
        "tail_selected_element_count": record.get(
            "tail_selected_element_count"
        ),
        "tail_threshold": record.get("tail_threshold"),
        "tail_retained_fraction": record.get("tail_retained_fraction"),
        "lf_score": record.get("lf_score"),
        "tail_robust_score": record.get("tail_robust_score"),
        "content_score": record.get("content_score"),
        "lf_weight": record.get("lf_weight"),
        "tail_robust_weight": record.get("tail_robust_weight"),
        "tail_fraction": record.get("tail_fraction"),
        "aligned_lf_score": record.get("aligned_lf_score"),
        "aligned_tail_robust_score": record.get(
            "aligned_tail_robust_score"
        ),
        "aligned_content_score": record.get("aligned_content_score"),
        "attention_geometry_score": record.get("attention_geometry_score"),
        "raw_attention_geometry_score": record.get(
            "raw_attention_geometry_score"
        ),
        "attention_sync_score": record.get("attention_sync_score"),
        "registration_confidence": record.get("registration_confidence"),
        "alignment_digest": (
            None
            if alignment is None
            else dict(alignment).get("alignment_digest")
        ),
        "stable_pair_weight_identity_ready": metadata.get(
            "stable_pair_weight_identity_ready"
        ),
        "stable_pair_weight_identity_digest": metadata.get(
            "stable_pair_weight_identity_digest"
        ),
        "observed_pair_weight_realization_digest": metadata.get(
            "observed_pair_weight_realization_digest"
        ),
        "aligned_pair_weight_realization_digest": metadata.get(
            "aligned_pair_weight_realization_digest"
        ),
        "detection_qk_atomic_content_digest": metadata.get(
            "detection_qk_atomic_content_digest"
        ),
    }


def _finite_number(value: Any) -> bool:
    """判断值是否为非 bool 的有限实数."""

    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def _sha256_text(value: Any) -> bool:
    """判断值是否为规范小写 SHA-256 文本."""

    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


_DECISION_MATERIALIZATION_FIELDS = frozenset(
    {
        "frozen_content_threshold",
        "frozen_rescue_margin_low",
        "frozen_geometry_score_threshold",
        "frozen_registration_confidence_threshold",
        "frozen_attention_sync_score_threshold",
        "frozen_threshold_digest",
        "frozen_image_only_measurement_config_digest",
        "frozen_attention_geometry_enabled",
        "frozen_image_alignment_enabled",
        "frozen_geometry_rescue_enabled",
        "formal_raw_content_margin",
        "formal_aligned_content_margin",
        "formal_positive_by_content",
        "formal_geometry_reliable",
        "formal_content_failure_reason",
        "formal_rescue_eligible",
        "formal_rescue_applied",
        "formal_evidence_positive",
        "formal_metric_status",
    }
)


def _validate_image_only_measurement_digest_record(
    record: Mapping[str, Any],
    *,
    require_threshold_free: bool,
) -> dict[str, Any]:
    """复算连续证据；可选择强制原始记录尚未物化决策。"""

    metadata = record.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("检测记录缺少 metadata")
    for field_name in (
        "image_only_measurement_config_digest",
        "lf_carrier_protocol_digest",
        "tail_carrier_protocol_digest",
    ):
        if not _sha256_text(record.get(field_name)):
            raise ValueError(f"{field_name} 必须为规范 SHA-256")

    lf_weight = record.get("lf_weight")
    tail_weight = record.get("tail_robust_weight")
    tail_fraction = record.get("tail_fraction")
    raw_scores = (
        record.get("lf_score"),
        record.get("tail_robust_score"),
        record.get("content_score"),
    )
    if (
        type(lf_weight) is not float
        or type(tail_weight) is not float
        or type(tail_fraction) is not float
        or not math.isclose(lf_weight + tail_weight, 1.0, abs_tol=1e-12)
        or not 0.0 < tail_fraction <= 1.0
        or not all(_finite_number(value) for value in raw_scores)
        or not math.isclose(
            float(raw_scores[2]),
            lf_weight * float(raw_scores[0])
            + tail_weight * float(raw_scores[1]),
            abs_tol=1e-7,
        )
    ):
        raise ValueError("检测记录的 LF/tail 加权内容分数不能重建")
    if (
        (lf_weight > 0.0 and not _sha256_text(record.get("lf_template_content_sha256")))
        or (lf_weight == 0.0 and record.get("lf_template_content_sha256") != "")
        or (tail_weight > 0.0 and not _sha256_text(record.get("tail_template_content_sha256")))
        or (tail_weight == 0.0 and record.get("tail_template_content_sha256") != "")
        or (lf_weight == 0.0 and float(raw_scores[0]) != 0.0)
        or (tail_weight == 0.0 and float(raw_scores[1]) != 0.0)
    ):
        raise ValueError("已禁用内容分支仍保留模板原子或非零分数")

    shape = record.get("tail_template_shape")
    element_count = record.get("tail_template_element_count")
    selected_count = record.get("tail_selected_element_count")
    retained_fraction = record.get("tail_retained_fraction")
    threshold = record.get("tail_threshold")
    if tail_weight == 0.0:
        if not (
            shape == []
            and element_count == 0
            and selected_count == 0
            and retained_fraction == 0.0
            and threshold == 0.0
        ):
            raise ValueError("已禁用尾部分支不得保留模板 shape 或选择原子")
    elif (
        not isinstance(shape, list)
        or len(shape) != 4
        or any(type(value) is not int or value <= 0 for value in shape)
        or type(element_count) is not int
        or element_count != math.prod(shape)
        or type(selected_count) is not int
        or selected_count != math.ceil(element_count * tail_fraction)
        or not _finite_number(retained_fraction)
        or not math.isclose(
            float(retained_fraction),
            selected_count / element_count,
            abs_tol=1e-12,
        )
        or not _finite_number(threshold)
        or float(threshold) < 0.0
    ):
        raise ValueError("检测记录的尾部比例、shape 或选择计数不能重建")

    aligned_score = record.get("aligned_content_score")
    if aligned_score is None:
        if any(
            record.get(field_name) is not None
            for field_name in (
                "aligned_lf_score",
                "aligned_tail_robust_score",
            )
        ):
            raise ValueError("无 aligned 分数时不得保留部分分支分数")
    else:
        aligned_lf = record.get("aligned_lf_score")
        aligned_tail = record.get("aligned_tail_robust_score")
        if (
            not all(
                _finite_number(value)
                for value in (aligned_lf, aligned_tail, aligned_score)
            )
            or (lf_weight == 0.0 and float(aligned_lf) != 0.0)
            or (tail_weight == 0.0 and float(aligned_tail) != 0.0)
            or not math.isclose(
                float(aligned_score),
                lf_weight * float(aligned_lf)
                + tail_weight * float(aligned_tail),
                abs_tol=1e-7,
            )
        ):
            raise ValueError("aligned LF/tail 加权分数不能重建")

    alignment = record.get("alignment")
    geometry_score = record.get("attention_geometry_score")
    raw_geometry_score = record.get("raw_attention_geometry_score")
    sync_score = record.get("attention_sync_score")
    registration_confidence = record.get("registration_confidence")
    stable_pair_ready = metadata.get("stable_pair_weight_identity_ready")
    if type(stable_pair_ready) is not bool:
        raise ValueError("stable pair 权重身份就绪字段必须为 bool")
    if alignment is None:
        if (
            aligned_score is not None
            or geometry_score is not None
            or registration_confidence is not None
            or sync_score is not None
            or (
                metadata.get("attention_geometry_enabled") is True
                and not _finite_number(raw_geometry_score)
            )
            or (
                metadata.get("attention_geometry_enabled") is not True
                and raw_geometry_score is not None
            )
        ):
            raise ValueError("无 alignment 时注意力原始分数或对齐分数状态无效")
    else:
        if not isinstance(alignment, Mapping):
            raise ValueError("alignment 必须为 mapping 或 None")
        if aligned_score is None:
            raise ValueError("存在 alignment 时必须提供完整 aligned 内容分数")
        validate_attention_alignment_record(alignment)
        if alignment.get("layer_name") not in FROZEN_SD35_ATTENTION_MODULE_NAMES:
            raise ValueError("alignment 所选层不属于冻结 SD3.5 注意力层")
        if (
            not _finite_number(geometry_score)
            or not _finite_number(raw_geometry_score)
            or not _finite_number(registration_confidence)
            or not _finite_number(sync_score)
            or not math.isclose(
                float(geometry_score),
                float(alignment.get("relation_sync_score")),
                abs_tol=1e-12,
            )
            or not math.isclose(
                float(registration_confidence),
                float(alignment.get("registration_confidence")),
                abs_tol=1e-12,
            )
        ):
            raise ValueError("注意力分数、同步分数或注册置信度与 alignment 不一致")

    forbidden_decision_fields = {
        "raw_content_margin",
        "aligned_content_margin",
        "positive_by_content",
        "geometry_reliable",
        "content_failure_reason",
        "rescue_eligible",
        "rescue_applied",
        "evidence_positive",
    }
    if require_threshold_free and (
        forbidden_decision_fields.intersection(record)
        or _DECISION_MATERIALIZATION_FIELDS.intersection(record)
    ):
        raise ValueError("原始测量记录不得包含 calibration 决策字段")
    forbidden_metadata_fields = {
        "content_threshold",
        "geometry_score_threshold",
        "registration_confidence_threshold",
        "attention_sync_score_threshold",
        "rescue_margin_low",
    }
    if forbidden_metadata_fields.intersection(metadata):
        raise ValueError("原始测量 metadata 不得包含 calibration 参数")

    payload = recompute_image_only_measurement_digest_payload(record)
    if record.get("measurement_digest") != build_stable_digest(payload):
        raise ValueError("measurement digest 不能由样本级连续证据独立重建")
    return payload


def validate_image_only_measurement_digest_record(
    record: Mapping[str, Any],
) -> dict[str, Any]:
    """严格复算尚未经过 calibration/Apply 的阈值无关测量记录。"""

    return _validate_image_only_measurement_digest_record(
        record,
        require_threshold_free=True,
    )


def validate_image_only_measurement_projection_record(
    record: Mapping[str, Any],
) -> dict[str, Any]:
    """从最终记录投影并复算原始连续测量原子。

    该函数只验证 Measure 产生的原子是否仍可重建，不把最终判定记录重新
    解释为 calibration 输入。Calibrate 与 Apply 必须调用严格版本。
    """

    return _validate_image_only_measurement_digest_record(
        record,
        require_threshold_free=False,
    )


def project_image_only_measurement_record(
    record: Mapping[str, Any],
) -> dict[str, Any]:
    """从最终记录显式投影出可重新校准的阈值无关测量记录。

    该函数先复验连续测量原子，再只移除由 Apply 物化的决策字段。调用方必须
    显式执行此投影，Calibrate 本身仍拒绝最终记录，从而保持测量与判定边界清晰。
    """

    validate_image_only_measurement_projection_record(record)
    projected = {
        field_name: value
        for field_name, value in record.items()
        if field_name not in _DECISION_MATERIALIZATION_FIELDS
    }
    validate_image_only_measurement_digest_record(projected)
    return projected


def measure_image_only_watermark(
    image: Any,
    key_material: str,
    config: ImageOnlyMeasurementConfig,
    image_latent_encoder: ImageLatentEncoder,
    image_attention_extractor: ImageAttentionExtractor | None = None,
    image_aligner: ImageAligner | None = None,
) -> ImageOnlyMeasurementResult:
    """仅从待检图像、密钥和公开模型配置测量水印连续证据。

    函数签名故意不接受原始 latent、生成轨迹、原始图像、prompt 或样本级安全
    基底, 从接口层阻止检测路径重新依赖生成端私有状态。
    """

    measurement_config_identity = image_only_measurement_config_identity_record(
        config,
        attention_geometry_enabled=image_attention_extractor is not None,
        image_alignment_enabled=image_aligner is not None,
    )
    measurement_config_digest = measurement_config_identity[
        "image_only_measurement_config_digest"
    ]
    observed_latent = image_latent_encoder(image)
    alignment_gate = attention_alignment_gate_record(
        config.attention_anchor_count,
        config.attention_residual_threshold,
        config.attention_minimum_inlier_ratio,
    )
    component_protocol = attention_relation_component_protocol(
        config.attention_relation_component_weights
    )
    component_weights = tuple(
        component_protocol["attention_relation_component_weights"]
    )
    lf_active = config.lf_weight > 0.0
    tail_active = config.tail_robust_weight > 0.0
    tail_carrier_protocol = tail_robust_carrier_protocol_record(
        config.tail_fraction,
        prg_version=config.keyed_prg_version,
    )
    lf_template = (
        build_low_frequency_template(
            observed_latent,
            key_material,
            config.model_id,
            config.low_frequency_config,
            prg_version=config.keyed_prg_version,
        )
        if lf_active
        else None
    )
    lf_template_content_sha256 = (
        tensor_content_sha256(lf_template) if lf_template is not None else ""
    )
    tail_template = None
    tail_threshold = 0.0
    retained_fraction = 0.0
    tail_template_content_sha256 = ""
    tail_template_shape: list[int] = []
    tail_template_element_count = 0
    tail_selected_element_count = 0
    if tail_active:
        (
            tail_template,
            tail_threshold,
            retained_fraction,
        ) = build_tail_robust_template(
            observed_latent,
            key_material,
            config.model_id,
            config.tail_fraction,
            prg_version=config.keyed_prg_version,
        )
        tail_template_content_sha256 = tensor_content_sha256(tail_template)
        tail_template_shape = [int(value) for value in tail_template.shape]
        tail_template_element_count = int(tail_template.numel())
        tail_selected_element_count = math.ceil(
            tail_template_element_count * config.tail_fraction
        )
        actual_selected_element_count = int(
            (tail_template != 0).sum().item()
        )
        if (
            actual_selected_element_count != tail_selected_element_count
            or not math.isclose(
                retained_fraction,
                tail_selected_element_count / tail_template_element_count,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            raise RuntimeError("高斯幅值尾部模板的保留计数与冻结协议不一致")
    content = compute_blind_content_score(
        observed_latent,
        lf_template,
        tail_template,
        config.lf_weight,
        config.tail_robust_weight,
    )
    geometry_score: float | None = None
    raw_geometry_score: float | None = None
    sync_score: float | None = None
    registration_confidence: float | None = None
    alignment: AttentionAlignmentResult | None = None
    aligned_image: Any | None = None
    stable_token_selection_digest = ""
    stable_pair_weight_identity_digest = ""
    observed_pair_weight_realization_digest = ""
    aligned_pair_weight_realization_digest = ""
    stable_pair_weight_identity_ready = False
    attention_relation_source = ""
    attention_relation_component_identity_digest = ""
    attention_relation_keyed_projection_digest = ""
    attention_relation_qk_operator_metadata_digest = ""
    qk_atomic_content_records: list[dict[str, Any]] = []
    qk_atomic_layer_names: tuple[str, ...] = ()
    if image_attention_extractor is not None:
        attention_records = image_attention_extractor(image)
        if not attention_records:
            raise RuntimeError("图像盲检没有返回真实 Q/K attention")
        attention_record_schema = tuple(
            (layer_name, tuple(token_indices))
            for layer_name, _, token_indices in attention_records
        )
        if tuple(
            layer_name for layer_name, _, _ in attention_records
        ) != config.attention_module_names:
            raise RuntimeError("图像盲检 Q/K 层顺序与冻结检测配置不一致")
        relation_identity = build_attention_relation_graph_identity(
            attention_records,
            key_material,
            prg_version=config.keyed_prg_version,
            component_weights=component_weights,
        )
        if (
            relation_identity.relation_source != DIRECT_QK_RELATION_SOURCE
            or not relation_identity.qk_operator_metadata_ready
            or not relation_identity.qk_atomic_content_ready
        ):
            raise RuntimeError("正式图像盲检注意力必须绑定完整真实 Q/K 算子与内容")
        qk_atomic_content_records.append(
            {
                "qk_evaluation_role": "raw_detection_image",
                "qk_atomic_content_records": list(
                    relation_identity.qk_atomic_content_records
                ),
                "qk_atomic_content_digest": (
                    relation_identity.qk_atomic_content_digest
                ),
                "qk_atomic_content_ready": (
                    relation_identity.qk_atomic_content_ready
                ),
            }
        )
        qk_atomic_layer_names = tuple(
            record["record_layer_name"]
            for record in relation_identity.qk_atomic_content_records
        )
        attention_relation_source = relation_identity.relation_source
        attention_relation_component_identity_digest = (
            relation_identity.component_identity_digest
        )
        attention_relation_keyed_projection_digest = (
            relation_identity.keyed_projection_digest
        )
        attention_relation_qk_operator_metadata_digest = (
            relation_identity.qk_operator_metadata_digest
        )
        stable_selection = select_stable_attention_tokens(
            attention_records,
            stable_token_fraction=config.attention_stable_token_fraction,
        )
        stable_token_selection_digest = stable_selection.selection_digest
        stable_pair_weights = build_stable_attention_pair_weights(
            attention_records,
            stable_selection,
            unstable_pair_weight=config.attention_unstable_pair_weight,
        )
        stable_pair_weight_identity_digest = (
            stable_pair_weights.pair_weight_identity_digest
        )
        observed_pair_weight_realization_digest = (
            stable_pair_weights.pair_weight_realization_digest
        )
        score_tensor = attention_geometry_score(
            attention_records,
            key_material,
            prg_version=config.keyed_prg_version,
            stable_pair_weights=stable_pair_weights,
            component_weights=component_weights,
        )
        raw_geometry_score = float(score_tensor.detach().item())
        if image_aligner is not None:
            alignment_candidates = tuple(
                recover_attention_affine_alignment(
                    attention,
                    key_material,
                    layer_name,
                    token_indices,
                    stable_pair_weights,
                    prg_version=config.keyed_prg_version,
                    anchor_count=config.attention_anchor_count,
                    residual_threshold=config.attention_residual_threshold,
                    minimum_inlier_ratio=config.attention_minimum_inlier_ratio,
                    component_weights=component_weights,
                )
                for layer_name, attention, token_indices in attention_records
            )
            alignment = select_image_only_alignment_candidate(
                alignment_candidates,
                config.attention_module_names,
            )
            geometry_score = alignment.relation_sync_score
            registration_confidence = alignment.registration_confidence
            aligned_image = image_aligner(image, alignment)
            aligned_attention_records = image_attention_extractor(aligned_image)
            if not aligned_attention_records:
                raise RuntimeError("对齐图像没有返回真实 Q/K attention")
            aligned_attention_record_schema = tuple(
                (layer_name, tuple(token_indices))
                for layer_name, _, token_indices in aligned_attention_records
            )
            if aligned_attention_record_schema != attention_record_schema:
                raise RuntimeError("对齐前后真实 Q/K 层身份或二维网格不一致")
            aligned_relation_identity = build_attention_relation_graph_identity(
                aligned_attention_records,
                key_material,
                prg_version=config.keyed_prg_version,
                component_weights=component_weights,
            )
            if (
                aligned_relation_identity.relation_source
                != DIRECT_QK_RELATION_SOURCE
                or aligned_relation_identity.component_identity_digest
                != attention_relation_component_identity_digest
                or aligned_relation_identity.keyed_projection_digest
                != attention_relation_keyed_projection_digest
                or not aligned_relation_identity.qk_operator_metadata_ready
                or aligned_relation_identity.qk_operator_metadata_digest
                != attention_relation_qk_operator_metadata_digest
                or not aligned_relation_identity.qk_atomic_content_ready
            ):
                raise RuntimeError("对齐前后没有共享同一四分量 Q/K 关系图身份")
            qk_atomic_content_records.append(
                {
                    "qk_evaluation_role": "aligned_detection_image",
                    "qk_atomic_content_records": list(
                        aligned_relation_identity.qk_atomic_content_records
                    ),
                    "qk_atomic_content_digest": (
                        aligned_relation_identity.qk_atomic_content_digest
                    ),
                    "qk_atomic_content_ready": (
                        aligned_relation_identity.qk_atomic_content_ready
                    ),
                }
            )
            aligned_pair_weights = restore_transported_stable_attention_pair_weights(
                stable_pair_weights,
                alignment.canonical_token_weights,
                coordinate_space="registered_canonical_qk_grid",
                expected_realization_digest=(
                    alignment.canonical_pair_weight_realization_digest
                ),
            )
            aligned_pair_weight_realization_digest = (
                aligned_pair_weights.pair_weight_realization_digest
            )
            sync_score_tensor = attention_geometry_score(
                aligned_attention_records,
                key_material,
                prg_version=config.keyed_prg_version,
                stable_pair_weights=aligned_pair_weights,
                component_weights=component_weights,
            )
            sync_score = float(sync_score_tensor.detach().item())
            stable_pair_weight_identity_ready = (
                alignment.stable_pair_weight_identity_digest
                == stable_pair_weights.pair_weight_identity_digest
                == aligned_pair_weights.pair_weight_identity_digest
                and alignment.observed_pair_weight_realization_digest
                == stable_pair_weights.pair_weight_realization_digest
                and alignment.canonical_pair_weight_realization_digest
                == aligned_pair_weights.pair_weight_realization_digest
            )
    alignment_measurement_available = (
        alignment is not None and aligned_image is not None
    )
    aligned_lf_score: float | None = None
    aligned_tail_robust_score: float | None = None
    aligned_content_score: float | None = None
    aligned_lf_template_content_sha256: str | None = None
    aligned_tail_template_content_sha256: str | None = None
    aligned_tail_threshold: float | None = None
    aligned_tail_retained_fraction: float | None = None
    if (
        alignment_measurement_available
        and alignment is not None
        and aligned_image is not None
    ):
        aligned_latent = image_latent_encoder(aligned_image)
        if tuple(aligned_latent.shape) != tuple(observed_latent.shape):
            raise RuntimeError("配准前后的编码 latent 形状必须完全一致")
        aligned_lf_template = (
            build_low_frequency_template(
                aligned_latent,
                key_material,
                config.model_id,
                config.low_frequency_config,
                prg_version=config.keyed_prg_version,
            )
            if lf_active
            else None
        )
        aligned_lf_template_content_sha256 = (
            tensor_content_sha256(aligned_lf_template)
            if aligned_lf_template is not None
            else ""
        )
        if (
            lf_active
            and aligned_lf_template_content_sha256
            != lf_template_content_sha256
        ):
            raise RuntimeError("配准前后的 LF 固定模板身份必须完全一致")
        aligned_tail_template = None
        if tail_active:
            (
                aligned_tail_template,
                aligned_tail_threshold,
                aligned_tail_retained_fraction,
            ) = build_tail_robust_template(
                aligned_latent,
                key_material,
                config.model_id,
                config.tail_fraction,
                prg_version=config.keyed_prg_version,
            )
            aligned_tail_template_content_sha256 = tensor_content_sha256(
                aligned_tail_template
            )
            if (
                aligned_tail_template_content_sha256
                != tail_template_content_sha256
                or aligned_tail_threshold != tail_threshold
                or aligned_tail_retained_fraction != retained_fraction
            ):
                raise RuntimeError("配准前后的高斯幅值尾部模板身份必须完全一致")
        else:
            aligned_tail_template_content_sha256 = ""
            aligned_tail_threshold = 0.0
            aligned_tail_retained_fraction = 0.0
        aligned_content = compute_blind_content_score(
            aligned_latent,
            aligned_lf_template,
            aligned_tail_template,
            config.lf_weight,
            config.tail_robust_weight,
        )
        aligned_lf_score = aligned_content.lf_score
        aligned_tail_robust_score = aligned_content.tail_robust_score
        aligned_content_score = aligned_content.content_score
    qk_atomic_content_digest = (
        qk_atomic_evaluation_records_digest(
            qk_atomic_content_records,
            "detection_qk_atomic_content_records",
        )
        if qk_atomic_content_records
        else ""
    )
    metadata = {
        "detector_input_access_mode": "image_key_public_model_only",
        "blind_image_detector": True,
        "generation_latent_trace_required": False,
        "source_image_required": False,
        "prompt_required": False,
        "image_only_measurement_config_digest": measurement_config_digest,
        "attention_geometry_enabled": image_attention_extractor is not None,
        "image_alignment_enabled": image_aligner is not None,
        "attention_alignment_gate": dict(alignment_gate),
        **alignment_gate,
        "attention_sync_source": "aligned_image_reextracted_real_qk",
        "stable_token_selection_digest": stable_token_selection_digest,
        "stable_pair_weight_identity_digest": (
            stable_pair_weight_identity_digest
        ),
        "observed_pair_weight_realization_digest": (
            observed_pair_weight_realization_digest
        ),
        "aligned_pair_weight_realization_digest": (
            aligned_pair_weight_realization_digest
        ),
        "stable_pair_weight_identity_ready": stable_pair_weight_identity_ready,
        "attention_relation_source": attention_relation_source,
        "attention_relation_direct_qk_source_ready": (
            attention_relation_source == DIRECT_QK_RELATION_SOURCE
        ),
        "attention_relation_component_identity_digest": (
            attention_relation_component_identity_digest
        ),
        "attention_relation_keyed_projection_digest": (
            attention_relation_keyed_projection_digest
        ),
        "attention_relation_qk_operator_metadata_digest": (
            attention_relation_qk_operator_metadata_digest
        ),
        "attention_relation_qk_operator_metadata_ready": (
            bool(image_attention_extractor is not None)
            and relation_identity.qk_operator_metadata_ready
        ),
        "detection_qk_atomic_content_records": qk_atomic_content_records,
        "detection_qk_atomic_content_digest": qk_atomic_content_digest,
        "detection_qk_atomic_content_ready": (
            qk_atomic_evaluation_records_ready(
                qk_atomic_content_records,
                qk_atomic_content_digest,
                aggregate_field_name="detection_qk_atomic_content_records",
                expected_roles=(
                    "raw_detection_image",
                    *(
                        ("aligned_detection_image",)
                        if image_aligner is not None
                        else ()
                    ),
                ),
                expected_layer_names=qk_atomic_layer_names,
            )
            if qk_atomic_content_records
            else False
        ),
    }
    result_kwargs = {
        "content": content,
        "lf_carrier_protocol_digest": (
            config.low_frequency_config.protocol_digest
        ),
        "lf_template_content_sha256": lf_template_content_sha256,
        "tail_carrier_protocol_digest": tail_carrier_protocol[
            "tail_carrier_protocol_digest"
        ],
        "tail_fraction": config.tail_fraction,
        "tail_template_content_sha256": tail_template_content_sha256,
        "tail_template_shape": tail_template_shape,
        "tail_template_element_count": tail_template_element_count,
        "tail_selected_element_count": tail_selected_element_count,
        "tail_threshold": tail_threshold,
        "tail_retained_fraction": retained_fraction,
        "aligned_lf_score": aligned_lf_score,
        "aligned_tail_robust_score": aligned_tail_robust_score,
        "aligned_content_score": aligned_content_score,
        "attention_geometry_score": geometry_score,
        "raw_attention_geometry_score": raw_geometry_score,
        "attention_sync_score": sync_score,
        "registration_confidence": registration_confidence,
        "alignment": alignment,
        "image_only_measurement_config_digest": measurement_config_digest,
        "metadata": metadata,
    }
    unsigned_result = ImageOnlyMeasurementResult(
        **result_kwargs,
        measurement_digest="",
    )
    measurement_digest = build_stable_digest(
        recompute_image_only_measurement_digest_payload(
            unsigned_result.to_record()
        )
    )
    return ImageOnlyMeasurementResult(
        **result_kwargs,
        measurement_digest=measurement_digest,
    )
