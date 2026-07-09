"""正式论文证据边界的通用判定工具。"""

from __future__ import annotations

from typing import Any, Mapping

NONFORMAL_RESULT_MARKERS = ("proxy", "placeholder", "fallback", "synthetic", "formal_null")


def contains_nonformal_marker(value: Any) -> bool:
    """判断字段值是否含有不能进入正式论文结果包的诊断标记。

    该函数属于协议 schema 边界。probe_paper、pilot_paper 与 full_paper
    都只允许真实测量结果进入共同协议结果记录, 因此含有诊断标记的
    状态值、来源值或证据路径会被统一拦截。
    """

    if isinstance(value, Mapping):
        return any(contains_nonformal_marker(item) for item in value.values())
    if isinstance(value, list | tuple | set):
        return any(contains_nonformal_marker(item) for item in value)
    if isinstance(value, str):
        lowered = value.lower()
        return any(marker in lowered for marker in NONFORMAL_RESULT_MARKERS)
    return False
