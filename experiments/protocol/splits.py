"""定义实验 split 与 sample role 协议。"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from typing import Iterable

from experiments.protocol.prompts import PromptProtocolRecord

SPLIT_NAMES = ("dev", "calibration", "test")
SAMPLE_ROLES = ("positive_source", "clean_negative", "attacked_negative")
UNASSIGNED_SPLIT = "unassigned"
LARGE_SCALE_PROMPT_COUNT = 1000
SMALL_SCALE_DEV_RATIO = 0.10
LARGE_SCALE_DEV_RATIO = 0.09


def dev_ratio_for_prompt_count(prompt_count: int) -> float:
    """按样本规模选择 dev 占比。

    full_paper 目标会推进到 fixed-FPR=0.001。若仍使用 10% dev,
    6000 个 prompt 在分层 split 后的 calibration clean negative 数量会略低于
    95% 单侧置信边界所需数量。这里仅在大规模运行中将 dev 占比收缩到 9%,
    保持 calibration 与 test 有足够样本支撑低误报结论。
    """

    return LARGE_SCALE_DEV_RATIO if prompt_count >= LARGE_SCALE_PROMPT_COUNT else SMALL_SCALE_DEV_RATIO


def build_group_split_counts(prompt_count: int) -> dict[str, int]:
    """根据语义桶大小计算开发、校准和测试数量。"""
    if prompt_count <= 0:
        return {name: 0 for name in SPLIT_NAMES}
    dev_count = round(prompt_count * dev_ratio_for_prompt_count(prompt_count)) if prompt_count >= 10 else 0
    remaining_count = prompt_count - dev_count
    calibration_count = remaining_count // 2
    test_count = remaining_count - calibration_count
    return {
        "dev": dev_count,
        "calibration": calibration_count,
        "test": test_count,
    }


def build_split_assignments(records: Iterable[PromptProtocolRecord]) -> dict[str, str]:
    """按 prompt set 与 risk profile 分层后, 为 prompt_id 分配稳定 split。"""
    grouped_records: dict[tuple[str, str], list[PromptProtocolRecord]] = defaultdict(list)
    for record in records:
        grouped_records[(record.prompt_set, record.risk_profile)].append(record)

    assignments: dict[str, str] = {}
    for _, group_records in sorted(grouped_records.items()):
        sorted_records = sorted(group_records, key=lambda record: (record.prompt_digest, record.prompt_id))
        counts = build_group_split_counts(len(sorted_records))
        split_sequence = (
            ["dev"] * counts["dev"]
            + ["calibration"] * counts["calibration"]
            + ["test"] * counts["test"]
        )
        for record, split_name in zip(sorted_records, split_sequence):
            assignments[record.prompt_id] = split_name
    return assignments


def apply_split_assignments(records: Iterable[PromptProtocolRecord]) -> tuple[PromptProtocolRecord, ...]:
    """返回带有稳定 split 的 prompt records。"""
    record_tuple = tuple(records)
    assignments = build_split_assignments(record_tuple)
    return tuple(replace(record, split=assignments[record.prompt_id]) for record in record_tuple)


def group_prompt_ids_by_split(records: Iterable[PromptProtocolRecord]) -> dict[str, tuple[str, ...]]:
    """按 split 聚合 prompt_id。"""
    assigned_records = apply_split_assignments(records)
    grouped: dict[str, list[str]] = defaultdict(list)
    for record in assigned_records:
        grouped[record.split].append(record.prompt_id)
    return {name: tuple(sorted(grouped.get(name, []))) for name in SPLIT_NAMES}


def assert_disjoint_calibration_and_test(split_groups: dict[str, tuple[str, ...]]) -> bool:
    """检查 calibration 与 test 的 prompt_id 是否无交叉。"""
    calibration_ids = set(split_groups.get("calibration", ()))
    test_ids = set(split_groups.get("test", ()))
    return calibration_ids.isdisjoint(test_ids)
