"""构造冻结二维低通、逐样本中心化的正式 LF 密钥载体。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from main.core.digest import build_stable_digest, tensor_content_sha256
from main.core.keyed_prg import (
    build_keyed_gaussian_tensor,
    require_supported_keyed_prg_version,
)
from main.methods.carrier.keyed_tensor import (
    LOW_FREQUENCY_BOUNDARY_MODE,
    LOW_FREQUENCY_CEIL_MODE,
    LOW_FREQUENCY_COUNT_INCLUDE_PAD,
    LOW_FREQUENCY_DIVISOR_OVERRIDE,
    LOW_FREQUENCY_KERNEL_SIZE,
    LOW_FREQUENCY_PADDING,
    LOW_FREQUENCY_STRIDE,
    LowFrequencyCarrierConfig,
)


__all__ = [
    "LowFrequencyCarrierTemplate",
    "build_low_frequency_template",
]


_LF_PRG_DOMAIN = "lf_content"


def _torch() -> Any:
    """延迟导入 PyTorch，保持模块导入边界轻量。"""

    import torch

    return torch


def _require_sha256(value: Any, *, label: str) -> str:
    """要求规范的64位小写十六进制 SHA-256。"""

    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} 必须为规范 SHA-256")
    return value


def _validate_reference_latent(reference_latent: Any) -> tuple[int, int, int, int]:
    """在调用密钥 PRG 前闭合正式 latent 的输入边界。"""

    torch = _torch()
    if not isinstance(reference_latent, torch.Tensor):
        raise TypeError("reference_latent 必须为 Tensor")
    if not reference_latent.dtype.is_floating_point:
        raise TypeError("reference_latent 必须使用真实浮点 dtype")
    if reference_latent.device.type == "meta":
        raise ValueError("reference_latent 必须是已物化 Tensor")
    if reference_latent.ndim != 4:
        raise ValueError("reference_latent 必须具有 [1, C, H, W] 形状")
    shape = tuple(int(value) for value in reference_latent.shape)
    if shape[0] != 1 or any(value <= 0 for value in shape[1:]):
        raise ValueError("reference_latent 必须具有正尺寸 [1, C, H, W] 形状")
    if not bool(torch.isfinite(reference_latent).all()):
        raise ValueError("reference_latent 必须全部有限")
    return shape


def _frozen_low_frequency_config() -> LowFrequencyCarrierConfig:
    """从唯一现有冻结常量重建 LF 滤波协议身份。"""

    return LowFrequencyCarrierConfig(
        kernel_size=LOW_FREQUENCY_KERNEL_SIZE,
        stride=LOW_FREQUENCY_STRIDE,
        padding=LOW_FREQUENCY_PADDING,
        boundary_mode=LOW_FREQUENCY_BOUNDARY_MODE,
        ceil_mode=LOW_FREQUENCY_CEIL_MODE,
        count_include_pad=LOW_FREQUENCY_COUNT_INCLUDE_PAD,
        divisor_override=LOW_FREQUENCY_DIVISOR_OVERRIDE,
    )


def _validate_canonical_prg_tensor(raw: Any, shape: tuple[int, int, int, int]) -> Any:
    """验证密钥 PRG 的规范 CPU float32 输出。"""

    torch = _torch()
    if not isinstance(raw, torch.Tensor):
        raise TypeError("密钥 PRG 必须返回 Tensor")
    if raw.device.type != "cpu":
        raise ValueError("密钥 PRG 必须返回规范 CPU Tensor")
    if raw.dtype != torch.float32:
        raise TypeError("密钥 PRG 必须返回规范 float32 Tensor")
    if tuple(int(value) for value in raw.shape) != shape:
        raise ValueError("密钥 PRG 返回形状与 reference_latent 不一致")
    if not bool(torch.isfinite(raw).all()):
        raise ValueError("密钥 PRG 返回值必须全部有限")
    return raw


def _paired_low_pass(raw: Any, config: LowFrequencyCarrierConfig) -> Any:
    """只沿 H/W 应用冻结且参数完全显式的二维平均低通。"""

    import torch.nn.functional as functional

    if config.boundary_mode != "zero_padding":
        raise RuntimeError("LF 边界协议必须为 zero_padding")
    return functional.avg_pool2d(
        raw,
        kernel_size=config.kernel_size,
        stride=config.stride,
        padding=config.padding,
        ceil_mode=config.ceil_mode,
        count_include_pad=config.count_include_pad,
        divisor_override=config.divisor_override,
    )


def _center_and_l2_normalize_per_sample(low_pass: Any) -> Any:
    """在正式 B=1 输入域内逐样本中心化并执行 float32 L2。"""

    torch = _torch()
    if not bool(torch.isfinite(low_pass).all()):
        raise RuntimeError("LF 低通结果必须全部有限")
    centered = low_pass - low_pass.mean(dim=(1, 2, 3), keepdim=True)
    flat = centered.reshape(centered.shape[0], -1)
    norms = torch.linalg.vector_norm(flat, dim=1)
    if not bool(torch.isfinite(norms).all()):
        raise RuntimeError("LF 中心化 L2 能量必须有限")
    if bool((norms == 0.0).any()):
        raise RuntimeError("LF 中心化模板没有可归一化的非零能量")
    normalized = centered / norms.reshape(-1, 1, 1, 1)
    if not bool(torch.isfinite(normalized).all()):
        raise RuntimeError("LF L2 归一化产生非有限值")
    return normalized


@dataclass(frozen=True)
class LowFrequencyCarrierTemplate:
    """保存正式 LF 标准载体及其可重建身份。"""

    template: Any
    latent_shape: tuple[int, int, int, int]
    scoring_key_identity_digest: str
    model_identity_digest: str
    prg_version: str
    prg_domain: Literal["lf_content"]
    filter_identity_digest: str
    template_digest: str


def build_low_frequency_template(
    reference_latent: Any,
    key_material: str,
    model_identity_digest: str,
    *,
    prg_version: str,
) -> LowFrequencyCarrierTemplate:
    """按参考 latent 的精确 NCHW 形状构造二维低通 LF 密钥载体。"""

    shape = _validate_reference_latent(reference_latent)
    if type(key_material) is not str or not key_material:
        raise ValueError("key_material 必须为非空精确 str")
    normalized_model_identity_digest = _require_sha256(
        model_identity_digest,
        label="model_identity_digest",
    )
    require_supported_keyed_prg_version(prg_version)
    config = _frozen_low_frequency_config()

    raw = _validate_canonical_prg_tensor(
        build_keyed_gaussian_tensor(
            shape,
            key_material,
            {
                "operator": "latent_carrier_template",
                "branch_name": _LF_PRG_DOMAIN,
                "model_identity_digest": normalized_model_identity_digest,
            },
            prg_version=prg_version,
        ),
        shape,
    )
    low_pass = _paired_low_pass(raw, config)
    if tuple(int(value) for value in low_pass.shape) != shape:
        raise RuntimeError("冻结 LF 低通必须保持 Tensor 形状")
    canonical_template = _center_and_l2_normalize_per_sample(low_pass)

    scoring_key_identity_digest = build_stable_digest(
        {"key_material": key_material}
    )
    filter_identity_digest = config.protocol_digest
    template_content_sha256 = tensor_content_sha256(canonical_template)
    template_digest = build_stable_digest(
        {
            "carrier_template": "low_frequency",
            "latent_shape": list(shape),
            "scoring_key_identity_digest": scoring_key_identity_digest,
            "model_identity_digest": normalized_model_identity_digest,
            "prg_version": prg_version,
            "prg_domain": _LF_PRG_DOMAIN,
            "filter_identity_digest": filter_identity_digest,
            "template_content_sha256": template_content_sha256,
        }
    )
    template = canonical_template.to(
        device=reference_latent.device,
        dtype=_torch().float32,
    )
    return LowFrequencyCarrierTemplate(
        template=template,
        latent_shape=shape,
        scoring_key_identity_digest=scoring_key_identity_digest,
        model_identity_digest=normalized_model_identity_digest,
        prg_version=prg_version,
        prg_domain=_LF_PRG_DOMAIN,
        filter_identity_digest=filter_identity_digest,
        template_digest=template_digest,
    )
