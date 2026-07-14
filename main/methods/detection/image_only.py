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
IMAGE_ONLY_DETECTOR_CONFIG_SCHEMA = "slm_wm_image_only_detector_config_v3"
ATTENTION_ALIGNMENT_LAYER_SELECTION_RULE = (
    "lexicographic_objective_observation_confidence_then_frozen_layer_order_v1"
)


@dataclass(frozen=True)
class ImageOnlyDetectionConfig:
    """定义仅图像检测允许使用的公开参数和冻结阈值。"""

    model_id: str
    attention_module_names: tuple[str, ...]
    content_threshold: float
    geometry_score_threshold: float
    attention_anchor_count: int
    attention_residual_threshold: float
    attention_minimum_inlier_ratio: float
    low_frequency_config: LowFrequencyCarrierConfig
    lf_weight: float
    tail_robust_weight: float
    tail_fraction: float
    keyed_prg_version: str
    registration_confidence_threshold: float
    attention_sync_score_threshold: float
    rescue_margin_low: float
    attention_stable_token_fraction: float
    attention_unstable_pair_weight: float
    attention_relation_component_weights: tuple[float, ...]

    def __post_init__(self) -> None:
        """集中校验冻结检测协议。"""

        if type(self.model_id) is not str or not self.model_id:
            raise ValueError("model_id 必须为非空精确 str")
        if self.attention_module_names != FROZEN_SD35_ATTENTION_MODULE_NAMES:
            raise ValueError("attention_module_names 必须等于冻结 SD3.5 层顺序")
        for field_name, value in (
            ("content_threshold", self.content_threshold),
            ("geometry_score_threshold", self.geometry_score_threshold),
            (
                "registration_confidence_threshold",
                self.registration_confidence_threshold,
            ),
            (
                "attention_sync_score_threshold",
                self.attention_sync_score_threshold,
            ),
            ("rescue_margin_low", self.rescue_margin_low),
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
        if self.rescue_margin_low >= 0.0:
            raise ValueError("rescue_margin_low 必须小于 0")
        if abs(self.lf_weight + self.tail_robust_weight - 1.0) > 1e-9:
            raise ValueError("内容分支权重之和必须为 1")
        if not -1.0 <= self.geometry_score_threshold <= 1.0:
            raise ValueError("geometry_score_threshold 必须位于 [-1, 1]")
        if not 0.0 <= self.registration_confidence_threshold <= 1.0:
            raise ValueError("registration_confidence_threshold 必须位于 [0, 1]")
        if not -1.0 <= self.attention_sync_score_threshold <= 1.0:
            raise ValueError("attention_sync_score_threshold 必须位于 [-1, 1]")
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


def image_only_detector_config_identity_record(
    config: ImageOnlyDetectionConfig,
    *,
    attention_geometry_enabled: bool,
    image_alignment_enabled: bool,
) -> dict[str, Any]:
    """构造可由检测记录独立重建的完整检测器配置身份."""

    if not isinstance(config, ImageOnlyDetectionConfig):
        raise TypeError("config 必须为 ImageOnlyDetectionConfig")
    if (
        type(attention_geometry_enabled) is not bool
        or type(image_alignment_enabled) is not bool
    ):
        raise TypeError("检测机制开关必须为精确 bool")
    relation_protocol = attention_relation_component_protocol(
        config.attention_relation_component_weights
    )
    prg_protocol = keyed_prg_protocol_record(config.keyed_prg_version)
    tail_protocol = tail_robust_carrier_protocol_record(
        config.tail_fraction,
        prg_version=config.keyed_prg_version,
    )
    payload = {
        "image_only_detector_config_schema": (
            IMAGE_ONLY_DETECTOR_CONFIG_SCHEMA
        ),
        "model_id": config.model_id,
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
        "content_threshold": config.content_threshold,
        "geometry_score_threshold": config.geometry_score_threshold,
        "registration_confidence_threshold": (
            config.registration_confidence_threshold
        ),
        "attention_sync_score_threshold": (
            config.attention_sync_score_threshold
        ),
        "rescue_margin_low": config.rescue_margin_low,
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
        "image_only_detector_config_digest": build_stable_digest(payload),
    }


@dataclass(frozen=True)
class ImageOnlyDetectionResult:
    """保存内容主判、注意力几何救回和完整 evidence 判定。"""

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
    raw_content_margin: float
    aligned_lf_score: float | None
    aligned_tail_robust_score: float | None
    aligned_content_score: float | None
    aligned_content_margin: float | None
    positive_by_content: bool
    attention_geometry_score: float | None
    raw_attention_geometry_score: float | None
    attention_sync_score: float | None
    registration_confidence: float | None
    alignment: AttentionAlignmentResult | None
    geometry_reliable: bool
    content_failure_reason: str
    rescue_eligible: bool
    rescue_applied: bool
    evidence_positive: bool
    image_only_detector_config_digest: str
    detector_digest: str
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
            "raw_content_margin": self.raw_content_margin,
            "aligned_lf_score": self.aligned_lf_score,
            "aligned_tail_robust_score": self.aligned_tail_robust_score,
            "aligned_content_score": self.aligned_content_score,
            "aligned_content_margin": self.aligned_content_margin,
            "positive_by_content": self.positive_by_content,
            "attention_geometry_score": self.attention_geometry_score,
            "raw_attention_geometry_score": self.raw_attention_geometry_score,
            "attention_sync_score": self.attention_sync_score,
            "registration_confidence": self.registration_confidence,
            "alignment": alignment_record,
            "geometry_reliable": self.geometry_reliable,
            "content_failure_reason": self.content_failure_reason,
            "rescue_eligible": self.rescue_eligible,
            "rescue_applied": self.rescue_applied,
            "evidence_positive": self.evidence_positive,
            "image_only_detector_config_digest": (
                self.image_only_detector_config_digest
            ),
            "detector_digest": self.detector_digest,
            "metadata": self.metadata,
        }


def recompute_image_only_detection_digest_payload(
    record: Mapping[str, Any],
) -> dict[str, Any]:
    """从样本级决策字段重建紧凑检测记录摘要正文.

    完整检测器配置只保存在顶层运行配置中. 样本记录通过配置摘要引用该配置,
    此处只绑定会改变样本分数、配准救回或最终判定的字段.
    """

    if not isinstance(record, Mapping):
        raise TypeError("检测记录必须为 mapping")
    metadata = record.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("检测记录缺少 metadata")
    alignment = record.get("alignment")
    return {
        "image_only_detector_config_digest": record.get(
            "image_only_detector_config_digest"
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
        "raw_content_margin": record.get("raw_content_margin"),
        "aligned_lf_score": record.get("aligned_lf_score"),
        "aligned_tail_robust_score": record.get(
            "aligned_tail_robust_score"
        ),
        "aligned_content_score": record.get("aligned_content_score"),
        "aligned_content_margin": record.get("aligned_content_margin"),
        "content_threshold": metadata.get("content_threshold"),
        "geometry_score_threshold": metadata.get(
            "geometry_score_threshold"
        ),
        "registration_confidence_threshold": metadata.get(
            "registration_confidence_threshold"
        ),
        "attention_sync_score_threshold": metadata.get(
            "attention_sync_score_threshold"
        ),
        "rescue_margin_low": metadata.get("rescue_margin_low"),
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
        "content_failure_reason": record.get("content_failure_reason"),
        "positive_by_content": record.get("positive_by_content"),
        "geometry_reliable": record.get("geometry_reliable"),
        "rescue_eligible": record.get("rescue_eligible"),
        "rescue_applied": record.get("rescue_applied"),
        "evidence_positive": record.get("evidence_positive"),
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


def validate_image_only_detection_digest_record(
    record: Mapping[str, Any],
) -> dict[str, Any]:
    """复算样本分数、配准救回、最终判定和紧凑记录摘要."""

    metadata = record.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("检测记录缺少 metadata")
    for field_name in (
        "image_only_detector_config_digest",
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

    thresholds = tuple(
        metadata.get(field_name)
        for field_name in (
            "content_threshold",
            "geometry_score_threshold",
            "registration_confidence_threshold",
            "attention_sync_score_threshold",
            "rescue_margin_low",
        )
    )
    if not all(type(value) is float and math.isfinite(value) for value in thresholds):
        raise ValueError("检测记录缺少显式冻结阈值")
    (
        content_threshold,
        geometry_score_threshold,
        registration_confidence_threshold,
        attention_sync_score_threshold,
        rescue_margin_low,
    ) = thresholds
    raw_margin = record.get("raw_content_margin")
    if not _finite_number(raw_margin) or not math.isclose(
        float(raw_margin),
        float(raw_scores[2]) - content_threshold,
        abs_tol=1e-7,
    ):
        raise ValueError("原图内容 margin 不能由分数和阈值重建")

    aligned_score = record.get("aligned_content_score")
    aligned_margin = record.get("aligned_content_margin")
    if aligned_score is None:
        if any(
            record.get(field_name) is not None
            for field_name in (
                "aligned_lf_score",
                "aligned_tail_robust_score",
                "aligned_content_margin",
            )
        ):
            raise ValueError("无 aligned 分数时不得保留部分分支分数")
    else:
        aligned_lf = record.get("aligned_lf_score")
        aligned_tail = record.get("aligned_tail_robust_score")
        if (
            not all(
                _finite_number(value)
                for value in (aligned_lf, aligned_tail, aligned_score, aligned_margin)
            )
            or (lf_weight == 0.0 and float(aligned_lf) != 0.0)
            or (tail_weight == 0.0 and float(aligned_tail) != 0.0)
            or not math.isclose(
                float(aligned_score),
                lf_weight * float(aligned_lf)
                + tail_weight * float(aligned_tail),
                abs_tol=1e-7,
            )
            or not math.isclose(
                float(aligned_margin),
                float(aligned_score) - content_threshold,
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
    registration_geometry_reliable = False
    if alignment is None:
        if (
            geometry_score is not None
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
        validate_attention_alignment_record(alignment)
        if alignment.get("layer_name") not in FROZEN_SD35_ATTENTION_MODULE_NAMES:
            raise ValueError("alignment 所选层不属于冻结 SD3.5 注意力层")
        registration_geometry_reliable = alignment.get("geometry_reliable") is True
        if (
            not _finite_number(geometry_score)
            or not _finite_number(raw_geometry_score)
            or not _finite_number(registration_confidence)
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
            raise ValueError("注意力分数或注册置信度与 alignment 不一致")

    expected_geometry_reliable = bool(
        alignment is not None
        and registration_geometry_reliable
        and stable_pair_ready
        and _finite_number(geometry_score)
        and float(geometry_score) >= geometry_score_threshold
        and _finite_number(registration_confidence)
        and float(registration_confidence) >= registration_confidence_threshold
        and _finite_number(sync_score)
        and float(sync_score) >= attention_sync_score_threshold
    )
    positive_by_content = float(raw_margin) >= 0.0
    failure_reason = (
        "content_positive"
        if positive_by_content
        else (
            "geometry_suspected"
            if rescue_margin_low <= float(raw_margin) < 0.0
            and expected_geometry_reliable
            else (
                "low_confidence"
                if rescue_margin_low <= float(raw_margin) < 0.0
                else "content_evidence_absent"
            )
        )
    )
    rescue_eligible = bool(
        rescue_margin_low <= float(raw_margin) < 0.0
        and expected_geometry_reliable
        and aligned_score is not None
    )
    rescue_applied = bool(
        rescue_eligible
        and aligned_margin is not None
        and float(aligned_margin) >= 0.0
    )
    expected = {
        "positive_by_content": positive_by_content,
        "geometry_reliable": expected_geometry_reliable,
        "content_failure_reason": failure_reason,
        "rescue_eligible": rescue_eligible,
        "rescue_applied": rescue_applied,
        "evidence_positive": positive_by_content or rescue_applied,
    }
    if any(record.get(name) != value for name, value in expected.items()):
        raise ValueError("检测记录的配准救回或最终判定不能重建")

    payload = recompute_image_only_detection_digest_payload(record)
    if record.get("detector_digest") != build_stable_digest(payload):
        raise ValueError("detector digest 不能由样本级决策字段独立重建")
    return payload


def detect_image_only_watermark(
    image: Any,
    key_material: str,
    config: ImageOnlyDetectionConfig,
    image_latent_encoder: ImageLatentEncoder,
    image_attention_extractor: ImageAttentionExtractor | None = None,
    image_aligner: ImageAligner | None = None,
) -> ImageOnlyDetectionResult:
    """仅从待检图像、密钥和公开模型配置计算水印判定。

    函数签名故意不接受原始 latent、生成轨迹、原始图像、prompt 或样本级安全
    基底, 从接口层阻止检测路径重新依赖生成端私有状态。
    """

    detector_config_identity = image_only_detector_config_identity_record(
        config,
        attention_geometry_enabled=image_attention_extractor is not None,
        image_alignment_enabled=image_aligner is not None,
    )
    detector_config_digest = detector_config_identity[
        "image_only_detector_config_digest"
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
    margin = content.content_score - config.content_threshold
    positive_by_content = margin >= 0.0

    geometry_score: float | None = None
    raw_geometry_score: float | None = None
    sync_score: float | None = None
    registration_confidence: float | None = None
    alignment: AttentionAlignmentResult | None = None
    geometry_reliable = False
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
        geometry_reliable = bool(
            alignment is not None
            and alignment.geometry_reliable
            and stable_pair_weight_identity_ready
            and geometry_score is not None
            and geometry_score >= config.geometry_score_threshold
            and registration_confidence is not None
            and registration_confidence >= config.registration_confidence_threshold
            and sync_score is not None
            and sync_score >= config.attention_sync_score_threshold
        )
    alignment_available = (
        alignment is not None
        and alignment.geometry_reliable
        and aligned_image is not None
    )
    if positive_by_content:
        content_failure_reason = "content_positive"
    elif config.rescue_margin_low <= margin < 0.0 and geometry_reliable:
        content_failure_reason = "geometry_suspected"
    elif config.rescue_margin_low <= margin < 0.0:
        content_failure_reason = "low_confidence"
    else:
        content_failure_reason = "content_evidence_absent"
    rescue_eligible = (
        config.rescue_margin_low <= margin < 0.0
        and alignment_available
        and geometry_reliable
        and content_failure_reason in {"geometry_suspected", "low_confidence"}
    )
    aligned_lf_score: float | None = None
    aligned_tail_robust_score: float | None = None
    aligned_content_score: float | None = None
    aligned_content_margin: float | None = None
    aligned_lf_template_content_sha256: str | None = None
    aligned_tail_template_content_sha256: str | None = None
    aligned_tail_threshold: float | None = None
    aligned_tail_retained_fraction: float | None = None
    if alignment_available and alignment is not None and aligned_image is not None:
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
        aligned_content_margin = aligned_content_score - config.content_threshold
    rescue_applied = (
        rescue_eligible
        and aligned_content_margin is not None
        and aligned_content_margin >= 0.0
    )
    evidence_positive = positive_by_content or rescue_applied
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
        "image_only_detector_config_digest": detector_config_digest,
        "attention_geometry_enabled": image_attention_extractor is not None,
        "image_alignment_enabled": image_aligner is not None,
        "content_threshold": config.content_threshold,
        "rescue_margin_low": config.rescue_margin_low,
        "geometry_score_threshold": config.geometry_score_threshold,
        "registration_confidence_threshold": (
            config.registration_confidence_threshold
        ),
        "attention_sync_score_threshold": config.attention_sync_score_threshold,
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
        "raw_content_margin": margin,
        "aligned_lf_score": aligned_lf_score,
        "aligned_tail_robust_score": aligned_tail_robust_score,
        "aligned_content_score": aligned_content_score,
        "aligned_content_margin": aligned_content_margin,
        "positive_by_content": positive_by_content,
        "attention_geometry_score": geometry_score,
        "raw_attention_geometry_score": raw_geometry_score,
        "attention_sync_score": sync_score,
        "registration_confidence": registration_confidence,
        "alignment": alignment,
        "geometry_reliable": geometry_reliable,
        "content_failure_reason": content_failure_reason,
        "rescue_eligible": rescue_eligible,
        "rescue_applied": rescue_applied,
        "evidence_positive": evidence_positive,
        "image_only_detector_config_digest": detector_config_digest,
        "metadata": metadata,
    }
    unsigned_result = ImageOnlyDetectionResult(
        **result_kwargs,
        detector_digest="",
    )
    detector_digest = build_stable_digest(
        recompute_image_only_detection_digest_payload(
            unsigned_result.to_record()
        )
    )
    return ImageOnlyDetectionResult(
        **result_kwargs,
        detector_digest=detector_digest,
    )
