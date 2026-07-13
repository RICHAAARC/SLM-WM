"""提供设备无关、版本化的密钥伪随机 Tensor 原语.

该模块只负责从密钥化 SHA-256 计数器字节流构造规范 CPU float32 Tensor.
调用方可以把结果搬运到 CUDA, 但设备、PyTorch RNG 和设备特定随机算法不会
进入随机值定义. 高斯载体, Jacobian 候选方向和注意力关系符号共享该原语.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any, Mapping

from main.core.digest import build_stable_digest, stable_json_dumps
from main.core.normal_quantile_table import (
    NORMAL_QUANTILE_COUNT,
    NORMAL_QUANTILE_INDEX_BITS,
    standard_normal_quantile_float32_table,
    standard_normal_quantile_table_record,
)


KEYED_PRG_VERSION = "sha256_counter_normal_icdf_table20_float32_v2"
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
    normal_table_record = standard_normal_quantile_table_record()
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
        "normal_index_bits": NORMAL_QUANTILE_INDEX_BITS,
        "normal_counter_block_bits": 256,
        "normal_bitstream_order": "sha256_blocks_then_msb_first_bits",
        "normal_index_rule": (
            "consecutive_20bit_words_across_counter_block_boundaries"
        ),
        "normal_transform": (
            "frozen_midpoint_inverse_normal_cdf_table20_float32"
        ),
        **normal_table_record,
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


def _normal_quantile_indices(
    element_count: int,
    domain: bytes,
) -> list[int]:
    """从连续 SHA-256 大端位流提取跨块20位分位数表索引."""

    indices: list[int] = []
    counter = 0
    bit_buffer = 0
    available_bits = 0
    index_mask = (1 << NORMAL_QUANTILE_INDEX_BITS) - 1
    while len(indices) < element_count:
        block = hashlib.sha256(
            domain + counter.to_bytes(_PRG_COUNTER_BYTES, "big")
        ).digest()
        counter += 1
        bit_buffer = (bit_buffer << 256) | int.from_bytes(block, "big")
        available_bits += 256
        while (
            available_bits >= NORMAL_QUANTILE_INDEX_BITS
            and len(indices) < element_count
        ):
            available_bits -= NORMAL_QUANTILE_INDEX_BITS
            indices.append(
                (bit_buffer >> available_bits) & index_mask
            )
            bit_buffer &= (
                (1 << available_bits) - 1 if available_bits else 0
            )
    return indices[:element_count]


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
    """在 CPU 上通过冻结逆 CDF 表构造规范高斯 float32 Tensor.

    该函数从同一 SHA-256 domain 的连续大端位流提取20位索引, 再查询
    1048576格标准正态中点分位数表. 运行时不调用平台数学库, 因而其他
    项目可以直接复用该函数生成跨操作系统, CPU 和 CUDA 一致的密钥方向.
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
    quantile_table = standard_normal_quantile_float32_table()
    quantile_indices = _normal_quantile_indices(element_count, domain)
    if len(quantile_table) != NORMAL_QUANTILE_COUNT:
        raise RuntimeError("标准正态分位数表数量发生漂移")
    normal_values = [quantile_table[index] for index in quantile_indices]
    return torch.tensor(
        normal_values,
        dtype=torch.float32,
        device="cpu",
    ).reshape(normalized_shape)
