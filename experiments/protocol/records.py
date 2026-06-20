"""实验协议 records 的轻量校验与序列化。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from experiments.protocol.events import EventProtocolRecord
from experiments.protocol.prompts import PromptProtocolRecord


def json_line(value: Any) -> str:
    """将 JSON 兼容对象转换为稳定 JSONL 单行文本。"""
    return json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n"


def write_prompt_records(path: str | Path, records: Iterable[PromptProtocolRecord]) -> None:
    """写出 prompt records JSONL。"""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(json_line(record.to_dict()) for record in records), encoding="utf-8")


def write_event_records(path: str | Path, records: Iterable[EventProtocolRecord]) -> None:
    """写出 event records JSONL。"""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(json_line(record.to_dict()) for record in records), encoding="utf-8")


def validate_unique_ids(records: Iterable[dict[str, Any]], field_name: str) -> bool:
    """检查记录集合中的稳定标识是否唯一。"""
    values = [record[field_name] for record in records]
    return len(values) == len(set(values))
