"""提供设备无关、版本化的密钥伪随机 Tensor 原语.

该模块只负责从密钥化 SHA-256 计数器字节流构造规范 CPU float32 Tensor.
调用方可以把结果搬运到 CUDA, 但设备、PyTorch RNG 和设备特定随机算法不会
进入随机值定义.高斯载体、Jacobian 候选方向和注意力关系符号共享该原语.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any, Mapping

from main.core.digest import build_stable_digest, stable_json_dumps


KEYED_PRG_VERSION = "sha256_counter_box_muller_float32_v1"
_PRG_COUNTER_BYTES = 16
_PRG_UNIFORM_BITS = 53


def _torch() -> Any:
    """延迟导入 PyTorch, 保持治理工具的轻量导入边界."""

    import torch

    return torch


def require_supported_keyed_prg_version(prg_version: str) -> None:
    """拒绝未登记的密钥 PRG 版本, 防止科学算子随机身份漂移."""

    if prg_version != KEYED_PRG_VERSION:
        raise ValueError(f"keyed_prg_version 必须为 {KEYED_PRG_VERSION}")


def keyed_prg_protocol_record(
    prg_version: str = KEYED_PRG_VERSION,
) -> dict[str, Any]:
    """返回不含密钥和样本输入的公开 PRG 算法身份."""

    require_supported_keyed_prg_version(prg_version)
    payload = {
        "keyed_prg_version": prg_version,
        "domain_serialization": "stable_json_utf8_then_sha256",
        "counter_stream": "sha256(domain_digest||counter_uint128_be)",
        "counter_initial_value": 0,
        "counter_bytes": _PRG_COUNTER_BYTES,
        "sha256_block_bytes": 32,
        "word_bytes": 8,
        "word_byte_order": "big",
        "word_offsets": [0, 8, 16, 24],
        "uniform_bits": _PRG_UNIFORM_BITS,
        "uniform_word_rule": "high_53_bits_of_uint64_be",
        "uniform_mapping": "(mantissa+1)/(2^53+2)",
        "uniform_interval": "strict_open_unit_interval",
        "normal_transform": "box_muller_float64_then_float32",
        "normal_pair_order": "radius_cos_then_radius_sin",
        "canonical_generation_device": "cpu",
        "canonical_output_dtype": "float32",
    }
    return {
        **payload,
        "keyed_prg_protocol_digest": build_stable_digest(payload),
    }


def _prg_domain(
    shape: tuple[int, ...],
    key_material: str,
    domain_fields: Mapping[str, Any],
    prg_version: str,
) -> bytes:
    """把密钥、算子 domain 和输出 shape 绑定为固定长度摘要."""

    require_supported_keyed_prg_version(prg_version)
    if not key_material:
        raise ValueError("密钥 PRG 的 key_material 不能为空")
    if not domain_fields:
        raise ValueError("密钥 PRG 的 domain_fields 不能为空")
    if not shape or any(value <= 0 for value in shape):
        raise ValueError("密钥 PRG 的 Tensor shape 必须全部为正整数")
    payload = {
        "keyed_prg_version": prg_version,
        "key_material": key_material,
        "domain_fields": dict(domain_fields),
        "shape": shape,
    }
    return hashlib.sha256(stable_json_dumps(payload).encode("utf-8")).digest()


def _open_unit_interval(word: int) -> float:
    """把 SHA-256 的高53位映射到严格位于 (0, 1) 的双精度数."""

    mantissa = word >> (64 - _PRG_UNIFORM_BITS)
    return (float(mantissa) + 1.0) / float((1 << _PRG_UNIFORM_BITS) + 2)


def _uniform_values(
    element_count: int,
    domain: bytes,
) -> list[float]:
    """按大端计数器顺序展开规范均匀数流."""

    values: list[float] = []
    counter = 0
    while len(values) < element_count:
        block = hashlib.sha256(
            domain + counter.to_bytes(_PRG_COUNTER_BYTES, "big")
        ).digest()
        values.extend(
            _open_unit_interval(
                int.from_bytes(block[offset : offset + 8], "big")
            )
            for offset in range(0, len(block), 8)
        )
        counter += 1
    return values[:element_count]


def build_keyed_uniform_tensor(
    shape: tuple[int, ...],
    key_material: str,
    domain_fields: Mapping[str, Any],
    prg_version: str = KEYED_PRG_VERSION,
) -> Any:
    """在 CPU 上构造开区间均匀分布的规范 float32 Tensor."""

    torch = _torch()
    normalized_shape = tuple(int(value) for value in shape)
    domain = _prg_domain(
        normalized_shape,
        key_material,
        domain_fields,
        prg_version,
    )
    values = _uniform_values(math.prod(normalized_shape), domain)
    return torch.tensor(values, dtype=torch.float32, device="cpu").reshape(
        normalized_shape
    )


def build_keyed_gaussian_tensor(
    shape: tuple[int, ...],
    key_material: str,
    domain_fields: Mapping[str, Any],
    prg_version: str = KEYED_PRG_VERSION,
) -> Any:
    """在 CPU 上通过 Box-Muller 构造规范高斯 float32 Tensor.

    该函数先生成同一 domain 的均匀数对, 再在 Python float64 中执行
    Box-Muller, 最后统一取整为 CPU float32.其他项目可以直接复用该函数生成
    跨 CPU/CUDA 一致的密钥方向, 然后按自身算子需要搬运和 reshape.
    """

    torch = _torch()
    normalized_shape = tuple(int(value) for value in shape)
    element_count = math.prod(normalized_shape)
    domain = _prg_domain(
        normalized_shape,
        key_material,
        domain_fields,
        prg_version,
    )
    uniform_values = _uniform_values(2 * math.ceil(element_count / 2), domain)
    normal_values: list[float] = []
    for index in range(0, len(uniform_values), 2):
        radius = math.sqrt(-2.0 * math.log(uniform_values[index]))
        angle = 2.0 * math.pi * uniform_values[index + 1]
        normal_values.append(radius * math.cos(angle))
        if len(normal_values) < element_count:
            normal_values.append(radius * math.sin(angle))
    return torch.tensor(
        normal_values,
        dtype=torch.float32,
        device="cpu",
    ).reshape(normalized_shape)
