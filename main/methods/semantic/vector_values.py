"""提供风险场输入使用的有限数值向量工具。"""

from __future__ import annotations

import math
from numbers import Real
from typing import Any, Iterable, Sequence

NumberLike = int | float
VectorInput = NumberLike | Sequence["VectorInput"]


def _is_sequence(value: object) -> bool:
    """判断对象是否为可递归展开的数值序列。"""

    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def as_float_vector(values: VectorInput | Iterable[NumberLike], field_name: str) -> tuple[float, ...]:
    """将数值或嵌套数值序列展平为有限浮点向量。"""

    flattened: list[float] = []

    def visit(candidate: object) -> None:
        """递归收集一个数值或嵌套序列。"""

        if isinstance(candidate, Real):
            flattened.append(float(candidate))
            return
        if _is_sequence(candidate):
            for item in candidate:
                visit(item)
            return
        raise TypeError(f"{field_name} 必须是数值或数值序列")

    visit(values)
    if not flattened:
        raise ValueError(f"{field_name} 不得为空")
    if any(not math.isfinite(value) for value in flattened):
        raise ValueError(f"{field_name} 必须只包含有限数值")
    return tuple(flattened)


def ensure_equal_length(named_vectors: dict[str, Sequence[Any]]) -> int:
    """集中校验多个向量具有相同且非零的长度。"""

    lengths = {name: len(values) for name, values in named_vectors.items()}
    unique_lengths = set(lengths.values())
    if len(unique_lengths) != 1:
        raise ValueError(f"字段长度不一致: {lengths}")
    length = unique_lengths.pop()
    if length <= 0:
        raise ValueError("向量长度必须大于 0")
    return length


def clip_unit(value: float) -> float:
    """将数值裁剪到闭区间 [0, 1]。"""

    return min(max(value, 0.0), 1.0)
