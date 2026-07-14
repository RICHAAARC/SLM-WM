"""提供稳定摘要能力, 用于记录可重建输入和配置。"""

from __future__ import annotations

import hashlib
import json
from typing import Any


TENSOR_CONTENT_DIGEST_VERSION = "slm_wm_tensor_content"


def stable_json_dumps(value: Any) -> str:
    """将 JSON 兼容对象转为稳定字符串, 便于跨平台生成一致摘要。"""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def build_stable_digest(value: Any) -> str:
    """根据 JSON 兼容对象生成 SHA-256 摘要。"""
    encoded = stable_json_dumps(value).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def tensor_content_sha256(tensor: Any) -> str:
    """计算可跨运行复验的 Tensor 内容 SHA-256.

    摘要同时绑定精度、形状和连续原始字节, 因而相同数值使用不同 dtype 或
    不同 shape 时不会被误认为同一个科学原子. 该函数可复用于风险、基底、
    分支更新和 Q/K 内容记录.
    """

    import torch

    values = tensor.detach().cpu().contiguous()
    raw_bytes = (
        values.reshape(-1)
        .contiguous()
        .view(torch.uint8)
        .numpy()
        .tobytes(order="C")
    )
    digest = hashlib.sha256()
    digest.update(TENSOR_CONTENT_DIGEST_VERSION.encode("ascii"))
    digest.update(b"\0")
    digest.update(str(values.dtype).encode("utf-8"))
    digest.update(b"\0")
    digest.update(
        json.dumps(
            [int(value) for value in values.shape],
            separators=(",", ":"),
        ).encode("ascii")
    )
    digest.update(b"\0")
    digest.update(raw_bytes)
    return digest.hexdigest()


def tensor_content_identity(tensor: Any) -> dict[str, Any]:
    """返回可读 dtype、shape 与内容摘要组成的 Tensor 身份."""

    values = tensor.detach()
    return {
        "tensor_content_digest_version": TENSOR_CONTENT_DIGEST_VERSION,
        "tensor_dtype": str(values.dtype),
        "tensor_shape": [int(value) for value in values.shape],
        "tensor_content_sha256": tensor_content_sha256(values),
    }
