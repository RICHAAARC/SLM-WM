"""实现不读取生成轨迹的仅图像水印检测接口。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from main.core.digest import build_stable_digest
from main.methods.carrier.keyed_tensor import (
    BlindContentScore,
    build_low_frequency_template,
    build_tail_robust_template,
    compute_blind_content_score,
)
from main.methods.geometry.attention_alignment import AttentionAlignmentResult, recover_attention_affine_alignment
from main.methods.geometry.differentiable_attention import attention_geometry_score

ImageLatentEncoder = Callable[[Any], Any]
AttentionRecord = tuple[str, Any, tuple[int, ...]]
ImageAttentionExtractor = Callable[[Any], tuple[AttentionRecord, ...]]
ImageAligner = Callable[[Any, AttentionAlignmentResult], Any]


@dataclass(frozen=True)
class ImageOnlyDetectionConfig:
    """定义仅图像检测允许使用的公开参数和冻结阈值。"""

    model_id: str
    content_threshold: float
    geometry_score_threshold: float
    registration_confidence_threshold: float = 0.0
    attention_sync_score_threshold: float = 0.0
    rescue_margin_low: float = -0.05
    lf_weight: float = 0.70
    tail_robust_weight: float = 0.30
    tail_fraction: float = 0.20
    attention_anchor_count: int = 12
    attention_residual_threshold: float = 0.20
    attention_minimum_inlier_ratio: float = 0.50

    def __post_init__(self) -> None:
        """集中校验冻结检测协议。"""

        if not 0.0 < self.tail_fraction <= 1.0:
            raise ValueError("tail_fraction 必须位于 (0, 1]")
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


@dataclass(frozen=True)
class ImageOnlyDetectionResult:
    """保存内容主判、注意力几何救回和完整 evidence 判定。"""

    content: BlindContentScore
    raw_content_margin: float
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
    detector_digest: str
    metadata: dict[str, Any]

    def to_record(self) -> dict[str, Any]:
        """转换成可序列化检测记录。"""

        alignment_record = None
        if self.alignment is not None:
            alignment_record = {
                **self.alignment.__dict__,
                "registration_geometry_reliable": self.alignment.geometry_reliable,
                # 外层冻结协议读取该字段时必须看到已经包含恢复后 sync 的最终门禁。
                "geometry_reliable": self.geometry_reliable,
            }
        return {
            "lf_score": self.content.lf_score,
            "tail_robust_score": self.content.tail_robust_score,
            "content_score": self.content.content_score,
            "raw_content_margin": self.raw_content_margin,
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
            "detector_digest": self.detector_digest,
            "metadata": self.metadata,
        }


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

    observed_latent = image_latent_encoder(image)
    lf_template = build_low_frequency_template(observed_latent, key_material, config.model_id)
    tail_template, tail_threshold, retained_fraction = build_tail_robust_template(
        observed_latent,
        key_material,
        config.model_id,
        config.tail_fraction,
    )
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
    if image_attention_extractor is not None:
        attention_records = image_attention_extractor(image)
        if not attention_records:
            raise RuntimeError("图像盲检没有返回真实 Q/K attention")
        score_tensor = attention_geometry_score(attention_records, key_material)
        raw_geometry_score = float(score_tensor.detach().item())
        alignment_candidates = tuple(
            recover_attention_affine_alignment(
                attention,
                key_material,
                layer_name,
                token_indices,
                anchor_count=config.attention_anchor_count,
                residual_threshold=config.attention_residual_threshold,
                minimum_inlier_ratio=config.attention_minimum_inlier_ratio,
            )
            for layer_name, attention, token_indices in attention_records
        )
        alignment = max(
            alignment_candidates,
            key=lambda candidate: (
                candidate.relation_sync_score,
                candidate.registration_confidence,
            ),
        )
        geometry_score = alignment.relation_sync_score
        registration_confidence = alignment.registration_confidence
        if image_aligner is not None:
            aligned_image = image_aligner(image, alignment)
            aligned_attention_records = image_attention_extractor(aligned_image)
            if not aligned_attention_records:
                raise RuntimeError("对齐图像没有返回真实 Q/K attention")
            sync_score_tensor = attention_geometry_score(
                aligned_attention_records,
                key_material,
            )
            sync_score = float(sync_score_tensor.detach().item())
        geometry_reliable = (
            alignment.geometry_reliable
            and geometry_score >= config.geometry_score_threshold
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
    aligned_content_score: float | None = None
    aligned_content_margin: float | None = None
    if alignment_available and alignment is not None and aligned_image is not None:
        aligned_latent = image_latent_encoder(aligned_image)
        aligned_content = compute_blind_content_score(
            aligned_latent,
            build_low_frequency_template(aligned_latent, key_material, config.model_id),
            build_tail_robust_template(
                aligned_latent,
                key_material,
                config.model_id,
                config.tail_fraction,
            )[0],
            config.lf_weight,
            config.tail_robust_weight,
        )
        aligned_content_score = aligned_content.content_score
        aligned_content_margin = aligned_content_score - config.content_threshold
    rescue_applied = rescue_eligible and aligned_content_margin is not None and aligned_content_margin >= 0.0
    evidence_positive = positive_by_content or rescue_applied
    payload = {
        "content_score_digest": content.score_digest,
        "content_threshold": config.content_threshold,
        "raw_content_margin": round(margin, 12),
        "aligned_content_margin": None if aligned_content_margin is None else round(aligned_content_margin, 12),
        "attention_geometry_score": None if geometry_score is None else round(geometry_score, 12),
        "raw_attention_geometry_score": (
            None if raw_geometry_score is None else round(raw_geometry_score, 12)
        ),
        "attention_sync_score": None if sync_score is None else round(sync_score, 12),
        "registration_confidence": (
            None if registration_confidence is None else round(registration_confidence, 12)
        ),
        "alignment_digest": None if alignment is None else alignment.alignment_digest,
        "content_failure_reason": content_failure_reason,
        "rescue_applied": rescue_applied,
        "evidence_positive": evidence_positive,
    }
    return ImageOnlyDetectionResult(
        content=content,
        raw_content_margin=margin,
        aligned_content_score=aligned_content_score,
        aligned_content_margin=aligned_content_margin,
        positive_by_content=positive_by_content,
        attention_geometry_score=geometry_score,
        raw_attention_geometry_score=raw_geometry_score,
        attention_sync_score=sync_score,
        registration_confidence=registration_confidence,
        alignment=alignment,
        geometry_reliable=geometry_reliable,
        content_failure_reason=content_failure_reason,
        rescue_eligible=rescue_eligible,
        rescue_applied=rescue_applied,
        evidence_positive=evidence_positive,
        detector_digest=build_stable_digest(payload),
        metadata={
            "detector_input_access_mode": "image_key_public_model_only",
            "blind_image_detector": True,
            "generation_latent_trace_required": False,
            "source_image_required": False,
            "prompt_required": False,
            "tail_threshold": tail_threshold,
            "tail_retained_fraction": retained_fraction,
            "geometry_score_threshold": config.geometry_score_threshold,
            "registration_confidence_threshold": config.registration_confidence_threshold,
            "attention_sync_score_threshold": config.attention_sync_score_threshold,
            "attention_sync_source": "aligned_image_reextracted_real_qk",
        },
    )
