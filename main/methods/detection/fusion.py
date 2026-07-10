"""内容检测记录融合。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from main.core.digest import build_stable_digest
from main.methods.carrier.compose import ContentUpdate
from main.methods.detection.scores import ContentScore


RESCUE_ABLATION_MODES = (
    "full_rescue",
    "no_rescue",
    "no_attention_anchor",
    "fft_sync_only",
    "image_registration_only",
    "geo_direct_positive_audit",
)


@dataclass(frozen=True)
class ContentDetectionRecord:
    """内容载体检测记录。"""

    content_detection_record_id: str
    prompt_id: str
    split: str
    content_mode: str
    lf_enabled: bool
    tail_enabled: bool
    tail_truncation_enabled: bool
    lf_score: float
    tail_score: float
    combined_score: float
    lf_tail_fusion_score: float
    content_score: float
    fixed_fpr_ready: bool
    content_update_digest: str
    content_chain_digest: str
    score_digest: str
    supports_paper_claim: bool
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """转为可写入 JSON 的字典。"""
        return asdict(self)


def build_content_detection_record(
    prompt_id: str,
    split: str,
    content_update: ContentUpdate,
    score: ContentScore,
    metadata: dict[str, Any] | None = None,
) -> ContentDetectionRecord:
    """合并内容 update 和内容分数, 形成检测记录。"""
    record_payload = {
        "prompt_id": prompt_id,
        "split": split,
        "content_mode": content_update.content_mode,
        "content_chain_digest": content_update.content_chain_digest,
        "score_digest": score.score_digest,
    }
    record_id = build_stable_digest(record_payload)[:24]
    return ContentDetectionRecord(
        content_detection_record_id=record_id,
        prompt_id=prompt_id,
        split=split,
        content_mode=content_update.content_mode,
        lf_enabled=content_update.lf_enabled,
        tail_enabled=content_update.tail_enabled,
        tail_truncation_enabled=content_update.tail_truncation_enabled,
        lf_score=score.lf_score,
        tail_score=score.tail_score,
        combined_score=score.combined_score,
        lf_tail_fusion_score=score.lf_tail_fusion_score,
        content_score=score.content_score,
        fixed_fpr_ready=score.fixed_fpr_ready,
        content_update_digest=content_update.content_update_digest,
        content_chain_digest=content_update.content_chain_digest,
        score_digest=score.score_digest,
        supports_paper_claim=False,
        metadata=metadata or {},
    )


@dataclass(frozen=True)
class SameThresholdRescueConfig:
    """描述同阈值几何恢复重判协议的冻结配置。

    该配置属于通用工程写法: 将阈值、窗口和允许的失败原因集中到
    dataclass 构造时校验, 使业务函数只表达核心判定逻辑。
    """

    content_threshold: float
    rescue_margin_low: float
    allowed_fail_reasons: tuple[str, ...] = ("geometry_suspected", "low_confidence")
    registration_threshold: float = 0.70
    sync_threshold: float = 0.65
    residual_threshold: float = 0.40
    attestation_required: bool = False

    def __post_init__(self) -> None:
        """集中校验 rescue 协议边界, 避免业务路径重复防御式校验。"""
        if self.rescue_margin_low >= 0.0:
            raise ValueError("rescue_margin_low 必须小于 0")
        if not self.allowed_fail_reasons:
            raise ValueError("allowed_fail_reasons 不得为空")

    def to_dict(self) -> dict[str, Any]:
        """转为可写入 JSON 的字典。"""
        return asdict(self)


@dataclass(frozen=True)
class GeometricRescueDecisionRecord:
    """记录同阈值几何恢复后的内容重判结果。

    该对象体现项目特定机制: 几何证据只允许恢复参考系, 不能直接判正;
    正式 evidence 判定仍必须由同一个内容阈值上的内容分数完成。
    """

    aligned_detection_record_id: str
    content_detection_record_id: str
    prompt_id: str
    split: str
    sample_role: str
    rescue_ablation_mode: str
    raw_content_score: float
    aligned_content_score: float
    content_threshold: float
    raw_content_margin: float
    aligned_content_margin: float
    rescue_score_gain: float
    fail_reason: str
    rescue_margin_low: float
    geometry_evidence_record_id: str
    attention_graph_id: str
    capture_id: str
    geometry_reliable: bool
    positive_by_content: bool
    rescue_eligible: bool
    rescue_applied: bool
    evidence_decision: bool
    final_decision: bool
    direct_positive_decision: bool
    geo_direct_positive_audit_decision: bool
    supports_paper_claim: bool
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """转为可写入 JSON 的字典。"""
        return asdict(self)


def _field_value(record: Any, field_name: str, default: Any = None) -> Any:
    """从 dataclass 或字典中读取字段值。

    该辅助函数属于通用工程写法, 用于让核心判定函数同时复用已有
    dataclass 记录和脚本读取的 JSON 字典。
    """
    if isinstance(record, dict):
        return record.get(field_name, default)
    return getattr(record, field_name, default)


def effective_geometry_reliability(
    geometry_record: Any,
    config: SameThresholdRescueConfig,
    rescue_ablation_mode: str = "full_rescue",
) -> bool:
    """按消融模式计算几何恢复是否可用于同阈值重判。

    项目特定写法在于: `geo_direct_positive_audit` 只生成反例审计字段,
    不进入正式 evidence decision。
    """
    if rescue_ablation_mode == "no_attention_anchor":
        return False
    if rescue_ablation_mode == "fft_sync_only":
        return float(_field_value(geometry_record, "recovered_sync_consistency", 0.0)) >= config.sync_threshold
    if rescue_ablation_mode == "image_registration_only":
        return (
            float(_field_value(geometry_record, "registration_confidence", 0.0)) >= config.registration_threshold
            and float(_field_value(geometry_record, "alignment_residual", 1.0)) <= config.residual_threshold
        )
    return bool(_field_value(geometry_record, "geometry_reliable", False))


def decide_same_threshold_geometric_rescue(
    content_record: ContentDetectionRecord | dict[str, Any],
    geometry_record: dict[str, Any],
    aligned_content_score: float,
    config: SameThresholdRescueConfig,
    fail_reason: str,
    rescue_ablation_mode: str = "full_rescue",
    attestation_pass: bool = True,
    metadata: dict[str, Any] | None = None,
) -> GeometricRescueDecisionRecord:
    """执行同阈值几何恢复重判。

    该函数的作用是把内容分数、几何可靠性和 rescue 边界窗口合并为
    可审计记录。它不重新估计阈值, 也不允许几何链直接产生正式 positive。
    """
    raw_content_score = float(_field_value(content_record, "content_score"))
    raw_margin = raw_content_score - config.content_threshold
    aligned_margin = aligned_content_score - config.content_threshold
    positive_by_content = raw_margin >= 0.0
    geometry_reliable = effective_geometry_reliability(geometry_record, config, rescue_ablation_mode)
    rescue_enabled = rescue_ablation_mode not in {"no_rescue", "geo_direct_positive_audit"}
    rescue_eligible = (
        rescue_enabled
        and config.rescue_margin_low <= raw_margin < 0.0
        and geometry_reliable
        and fail_reason in config.allowed_fail_reasons
    )
    rescue_applied = rescue_eligible and aligned_margin >= 0.0
    evidence_decision = positive_by_content or rescue_applied
    final_decision = evidence_decision and (attestation_pass or not config.attestation_required)
    geo_direct_positive_audit_decision = positive_by_content or geometry_reliable
    payload = {
        "content_detection_record_id": _field_value(content_record, "content_detection_record_id"),
        "geometry_evidence_record_id": geometry_record["geometry_evidence_record_id"],
        "rescue_ablation_mode": rescue_ablation_mode,
        "raw_content_score": round(raw_content_score, 12),
        "aligned_content_score": round(aligned_content_score, 12),
        "content_threshold": config.content_threshold,
        "fail_reason": fail_reason,
    }
    record_id = build_stable_digest(payload)[:24]
    return GeometricRescueDecisionRecord(
        aligned_detection_record_id=f"aligned_detection_{record_id}",
        content_detection_record_id=str(_field_value(content_record, "content_detection_record_id")),
        prompt_id=str(_field_value(content_record, "prompt_id")),
        split=str(_field_value(content_record, "split")),
        sample_role=str(_field_value(content_record, "metadata", {}).get("sample_role", "unknown")),
        rescue_ablation_mode=rescue_ablation_mode,
        raw_content_score=raw_content_score,
        aligned_content_score=aligned_content_score,
        content_threshold=config.content_threshold,
        raw_content_margin=raw_margin,
        aligned_content_margin=aligned_margin,
        rescue_score_gain=aligned_content_score - raw_content_score,
        fail_reason=fail_reason,
        rescue_margin_low=config.rescue_margin_low,
        geometry_evidence_record_id=str(geometry_record["geometry_evidence_record_id"]),
        attention_graph_id=str(geometry_record["attention_graph_id"]),
        capture_id=str(geometry_record["capture_id"]),
        geometry_reliable=geometry_reliable,
        positive_by_content=positive_by_content,
        rescue_eligible=rescue_eligible,
        rescue_applied=rescue_applied,
        evidence_decision=evidence_decision,
        final_decision=final_decision,
        direct_positive_decision=False,
        geo_direct_positive_audit_decision=geo_direct_positive_audit_decision,
        supports_paper_claim=False,
        metadata=metadata or {},
    )
