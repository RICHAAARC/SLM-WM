"""统一阻断缺少精确9重复聚合来源的正式论文结论写入入口.

单重复证据组件只能作为跨重复聚合的原子输入. 版本化聚合包构造器与独立
验证器已经实现; 在各正式 claim Writer 显式接受不可变来源对象并传播其摘要
前, 仍不得从历史文件或手工输入恢复正向结论. 该边界是失败即关闭协议,
不是统计结果的代理实现.
"""

from __future__ import annotations

from typing import NoReturn


PAPER_CLAIM_AGGREGATE_REQUIRED_MESSAGE = (
    "正式论文结论写入必须先通过版本化精确9重复聚合证据验证"
)


class PaperClaimAggregateRequiredError(RuntimeError):
    """表示正式结论入口缺少可独立复验的跨重复聚合来源."""


def require_exact9_randomization_aggregate_provenance() -> NoReturn:
    """在调用入口尚未绑定不可变聚合来源对象时统一拒绝结论物化."""

    raise PaperClaimAggregateRequiredError(
        PAPER_CLAIM_AGGREGATE_REQUIRED_MESSAGE
    )


__all__ = [
    "PAPER_CLAIM_AGGREGATE_REQUIRED_MESSAGE",
    "PaperClaimAggregateRequiredError",
    "require_exact9_randomization_aggregate_provenance",
]
