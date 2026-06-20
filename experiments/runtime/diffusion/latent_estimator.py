"""提供轻量 latent 估计与摘要函数。

该模块属于实验运行层, 用于在没有真实模型权重的环境中构造可复现的 synthetic
latent 轨迹。它只产生工程测试记录, 不支持正式论文主张。
"""

from __future__ import annotations

import hashlib
import math
from numbers import Real
from typing import Iterable, Sequence

from main.core.digest import build_stable_digest

NumberLike = int | float


def _as_float_vector(values: Iterable[NumberLike], field_name: str) -> tuple[float, ...]:
    """将外部向量输入收敛为有限 float 元组。"""
    vector = tuple(float(value) for value in values)
    if not vector:
        raise ValueError(f"{field_name} 不得为空")
    if any(not math.isfinite(value) for value in vector):
        raise ValueError(f"{field_name} 必须只包含有限数值")
    return vector


def stable_unit_values(parts: Sequence[str], count: int) -> tuple[float, ...]:
    """根据文本材料生成稳定的 [-1, 1] 数值模板。"""
    if count <= 0:
        raise ValueError("count 必须为正数")
    values: list[float] = []
    for index in range(count):
        payload = "|".join((*parts, str(index))).encode("utf-8")
        digest = hashlib.sha256(payload).hexdigest()
        unit = int(digest[:16], 16) / float(0xFFFFFFFFFFFFFFFF)
        values.append(unit * 2.0 - 1.0)
    return tuple(values)


def vector_digest(values: Iterable[NumberLike]) -> str:
    """生成向量稳定摘要, 供 records 与 manifest 使用。"""
    vector = _as_float_vector(values, "values")
    return build_stable_digest([round(value, 12) for value in vector])


def latent_statistics(values: Iterable[NumberLike]) -> dict[str, float]:
    """计算 synthetic latent 的基础统计量。"""
    vector = _as_float_vector(values, "values")
    mean_value = sum(vector) / len(vector)
    variance = sum((value - mean_value) ** 2 for value in vector) / len(vector)
    return {
        "latent_mean": mean_value,
        "latent_std": math.sqrt(variance),
        "latent_min": min(vector),
        "latent_max": max(vector),
    }


def build_initial_latent(model_id: str, prompt: str, seed: int, latent_width: int) -> tuple[float, ...]:
    """构造 synthetic 初始 latent, 用于无模型环境下的适配层测试。"""
    return stable_unit_values((model_id, prompt, str(seed), "initial_latent"), latent_width)


def build_prompt_delta(prompt: str, negative_prompt: str, latent_width: int) -> tuple[float, ...]:
    """构造由 prompt 摘要驱动的 synthetic 采样方向。"""
    return stable_unit_values((prompt, negative_prompt, "prompt_delta"), latent_width)


def estimate_image_digest(latent_values: Iterable[NumberLike], model_id: str, seed: int) -> str:
    """根据最终 latent 摘要构造 synthetic 图像摘要。"""
    return build_stable_digest(
        {
            "model_id": model_id,
            "seed": seed,
            "latent_digest": vector_digest(latent_values),
            "image_source": "synthetic_latent_adapter",
        }
    )


def estimate_quality_score(latent_values: Iterable[NumberLike]) -> float:
    """给 synthetic 图像生成一个稳定质量分数, 仅用于工程测试排序。"""
    stats = latent_statistics(latent_values)
    spread_penalty = min(stats["latent_std"], 1.0) * 0.2
    center_penalty = min(abs(stats["latent_mean"]), 1.0) * 0.1
    return max(0.0, min(1.0, 0.95 - spread_penalty - center_penalty))
