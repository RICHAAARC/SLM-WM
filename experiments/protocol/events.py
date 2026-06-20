"""由 prompt 协议记录构建事件协议。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

from main.core.digest import build_stable_digest
from experiments.protocol.prompts import PromptProtocolRecord
from experiments.protocol.splits import SAMPLE_ROLES, apply_split_assignments


@dataclass(frozen=True)
class EventProtocolRecord:
    """描述后续实验可执行的单个事件。"""

    event_id: str
    prompt_id: str
    prompt_set: str
    split: str
    sample_role: str
    event_family: str
    attack_family: str
    protocol_decision: str
    supports_paper_claim: bool

    def to_dict(self) -> dict[str, Any]:
        """转为可写入 JSONL 的事件记录。"""
        return asdict(self)


def event_family_for_role(sample_role: str) -> str:
    """将 sample role 映射到事件族。"""
    if sample_role == "positive_source":
        return "watermark_embedding"
    if sample_role == "clean_negative":
        return "clean_generation"
    return "attack_generation"


def attack_family_for_role(sample_role: str) -> str:
    """将 sample role 映射到攻击族。"""
    return "geometric_shift" if sample_role == "attacked_negative" else "none"


def build_event_id(prompt_id: str, split: str, sample_role: str, event_family: str, attack_family: str) -> str:
    """生成稳定 event_id。"""
    digest = build_stable_digest(
        {
            "attack_family": attack_family,
            "event_family": event_family,
            "prompt_id": prompt_id,
            "sample_role": sample_role,
            "split": split,
        }
    )
    return f"event_{digest[:16]}"


def build_event_records(prompt_records: Iterable[PromptProtocolRecord]) -> tuple[EventProtocolRecord, ...]:
    """为全部 prompt 与 sample role 构造事件协议记录。"""
    events: list[EventProtocolRecord] = []
    for prompt_record in apply_split_assignments(prompt_records):
        for sample_role in SAMPLE_ROLES:
            event_family = event_family_for_role(sample_role)
            attack_family = attack_family_for_role(sample_role)
            events.append(
                EventProtocolRecord(
                    event_id=build_event_id(
                        prompt_record.prompt_id,
                        prompt_record.split,
                        sample_role,
                        event_family,
                        attack_family,
                    ),
                    prompt_id=prompt_record.prompt_id,
                    prompt_set=prompt_record.prompt_set,
                    split=prompt_record.split,
                    sample_role=sample_role,
                    event_family=event_family,
                    attack_family=attack_family,
                    protocol_decision="scheduled",
                    supports_paper_claim=False,
                )
            )
    return tuple(events)
