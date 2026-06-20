"""内容检测记录融合。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from main.core.digest import build_stable_digest
from main.methods.carrier.compose import ContentUpdate
from main.methods.detection.scores import ContentScore


@dataclass(frozen=True)
class ContentDetectionRecord:
    """内容载体检测记录。"""

    content_detection_record_id: str
    prompt_id: str
    split: str
    content_mode: str
    lf_enabled: bool
    hf_enabled: bool
    tail_truncation_enabled: bool
    lf_score: float
    hf_score: float
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
        hf_enabled=content_update.hf_enabled,
        tail_truncation_enabled=content_update.tail_truncation_enabled,
        lf_score=score.lf_score,
        hf_score=score.hf_score,
        content_score=score.content_score,
        fixed_fpr_ready=score.fixed_fpr_ready,
        content_update_digest=content_update.content_update_digest,
        content_chain_digest=content_update.content_chain_digest,
        score_digest=score.score_digest,
        supports_paper_claim=False,
        metadata=metadata or {},
    )
