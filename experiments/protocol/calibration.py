"""实验协议的 split 与统计校验。"""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable

from experiments.protocol.events import EventProtocolRecord
from experiments.protocol.prompts import PromptProtocolRecord
from experiments.protocol.splits import assert_disjoint_calibration_and_test, group_prompt_ids_by_split


def build_prompt_statistics(
    prompt_records: Iterable[PromptProtocolRecord],
    event_records: Iterable[EventProtocolRecord],
) -> dict[str, Any]:
    """构造 prompt 与事件协议统计。"""
    prompt_tuple = tuple(prompt_records)
    event_tuple = tuple(event_records)
    split_groups = group_prompt_ids_by_split(prompt_tuple)
    split_counts = {name: len(ids) for name, ids in split_groups.items()}
    sample_role_counts = Counter(event.sample_role for event in event_tuple)
    prompt_set_counts = Counter(prompt.prompt_set for prompt in prompt_tuple)
    calibration_test_disjoint = assert_disjoint_calibration_and_test(split_groups)
    return {
        "protocol_decision": "pass" if calibration_test_disjoint else "fail",
        "prompt_count": len(prompt_tuple),
        "event_count": len(event_tuple),
        "split_counts": split_counts,
        "sample_role_counts": dict(sorted(sample_role_counts.items())),
        "prompt_set_counts": dict(sorted(prompt_set_counts.items())),
        "calibration_test_disjoint": calibration_test_disjoint,
        "supports_paper_claim": False,
    }
