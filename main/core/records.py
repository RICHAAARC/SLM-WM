"""定义论文候选实验 records 的最小通用结构。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ExperimentRecord:
    """表示一条可被论文产物重建流程消费的实验记录。"""

    record_id: str
    run_id: str
    split: str
    method_name: str
    metric_name: str
    metric_value: float
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """转为普通字典, 便于写入 JSONL、测试 fixture 或 artifact builder。"""
        return asdict(self)


REQUIRED_RECORD_FIELDS = (
    "record_id",
    "run_id",
    "split",
    "method_name",
    "metric_name",
    "metric_value",
)


def validate_record(record: dict[str, Any]) -> list[str]:
    """返回缺失的最小 record 字段列表。"""
    return [field_name for field_name in REQUIRED_RECORD_FIELDS if field_name not in record]
