"""定义实验 split 与 sample role 协议。"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
import math
from typing import Iterable

from experiments.protocol.prompts import PromptProtocolRecord

SPLIT_NAMES = ("dev", "calibration", "test")
SAMPLE_ROLES = ("positive_source", "clean_negative", "attacked_negative")
UNASSIGNED_SPLIT = "unassigned"
# 70 / 700 / 7000 三级规模共享 3:33:34 比例, 分别得到
# dev 3/30/300、calibration 33/330/3300 和 test 34/340/3400。
DEV_RATIO = 3.0 / 70.0
CALIBRATION_RATIO = 33.0 / 70.0
SPLIT_ASSIGNMENT_BLOCK_SIZE = 70


def dev_ratio_for_prompt_count(prompt_count: int) -> float:
    """返回统一 dev 占比。

    三类论文运行层级使用同一 3:33:34 比例, 避免不同运行规模出现不同的
    样本划分语义。calibration 负责冻结阈值, test 提供独立固定 FPR 置信上界。
    """

    return DEV_RATIO


def build_group_split_counts(prompt_count: int) -> dict[str, int]:
    """根据语义桶大小计算开发、校准和测试数量。"""
    if prompt_count <= 0:
        return {name: 0 for name in SPLIT_NAMES}
    dev_count = round(prompt_count * DEV_RATIO) if prompt_count >= 10 else 0
    remaining_count = prompt_count - dev_count
    calibration_count = round(prompt_count * CALIBRATION_RATIO)
    if dev_count + calibration_count > prompt_count:
        calibration_count = prompt_count - dev_count
    test_count = remaining_count - calibration_count
    return {
        "dev": dev_count,
        "calibration": calibration_count,
        "test": test_count,
    }


def allocate_stratified_counts(
    group_sizes: tuple[int, ...],
    ratio: float,
    capacities: tuple[int, ...] | None = None,
) -> tuple[int, ...]:
    """按全局目标比例向各语义桶分摊整数样本数。

    该函数属于通用的分层抽样写法。先按每个桶的比例下取整, 再把余数
    按小数部分从大到小补回, 可避免每个桶独立四舍五入造成全局比例漂移。
    """

    total_count = sum(group_sizes)
    resolved_capacities = capacities or group_sizes
    target_count = min(round(total_count * ratio), sum(resolved_capacities))
    allocated = [min(math.floor(size * ratio), resolved_capacities[index]) for index, size in enumerate(group_sizes)]
    remaining = target_count - sum(allocated)
    ranked_indices = sorted(
        range(len(group_sizes)),
        key=lambda index: (group_sizes[index] * ratio - allocated[index], group_sizes[index], -index),
        reverse=True,
    )
    for index in ranked_indices:
        if remaining <= 0:
            break
        if allocated[index] < resolved_capacities[index]:
            allocated[index] += 1
            remaining -= 1
    return tuple(allocated)


def build_split_assignments(records: Iterable[PromptProtocolRecord]) -> dict[str, str]:
    """在固定70条前缀块内按风险类型分层分配稳定 split.

    每个完整块精确产生3个 dev、33个 calibration 和34个 test 记录。因为三级
    Prompt 文件是同一清单前缀, 同一 Prompt 在 probe、pilot 与 full 中不会改变
    split。最后一个不足70条的块仍按相同比例分配, 供轻量测试和通用调用复用。
    """

    grouped_blocks: dict[tuple[str, int], list[PromptProtocolRecord]] = defaultdict(list)
    for record in records:
        block_index = record.prompt_index // SPLIT_ASSIGNMENT_BLOCK_SIZE
        grouped_blocks[(record.prompt_set, block_index)].append(record)

    assignments: dict[str, str] = {}
    for _, block_records in sorted(grouped_blocks.items()):
        risk_groups: dict[str, list[PromptProtocolRecord]] = defaultdict(list)
        for record in block_records:
            risk_groups[record.risk_profile].append(record)
        group_items = tuple(sorted(risk_groups.items()))
        group_sizes = tuple(len(group_records) for _, group_records in group_items)
        dev_counts = allocate_stratified_counts(group_sizes, DEV_RATIO)
        calibration_capacities = tuple(
            size - dev_count
            for size, dev_count in zip(group_sizes, dev_counts)
        )
        calibration_counts = allocate_stratified_counts(
            group_sizes,
            CALIBRATION_RATIO,
            calibration_capacities,
        )
        for (_, group_records), dev_count, calibration_count in zip(
            group_items,
            dev_counts,
            calibration_counts,
        ):
            sorted_records = sorted(
                group_records,
                key=lambda record: (
                    record.prompt_digest,
                    record.prompt_index,
                    record.prompt_id,
                ),
            )
            split_sequence = (
                ["dev"] * dev_count
                + ["calibration"] * calibration_count
                + ["test"]
                * (len(sorted_records) - dev_count - calibration_count)
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
