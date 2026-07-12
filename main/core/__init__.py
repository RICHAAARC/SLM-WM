"""核心方法使用的稳定摘要和密钥 PRG 工具。"""

from main.core.digest import (
    TENSOR_CONTENT_DIGEST_VERSION,
    build_stable_digest,
    tensor_content_identity,
    tensor_content_sha256,
)
from main.core.keyed_prg import (
    KEYED_PRG_VERSION,
    build_keyed_gaussian_tensor,
    build_keyed_uniform_tensor,
    keyed_prg_protocol_record,
    require_supported_keyed_prg_version,
)
__all__ = [
    "KEYED_PRG_VERSION",
    "TENSOR_CONTENT_DIGEST_VERSION",
    "build_keyed_gaussian_tensor",
    "build_keyed_uniform_tensor",
    "build_stable_digest",
    "keyed_prg_protocol_record",
    "require_supported_keyed_prg_version",
    "tensor_content_identity",
    "tensor_content_sha256",
]
