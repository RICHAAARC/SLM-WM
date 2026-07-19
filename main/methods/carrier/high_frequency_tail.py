"""构造先二维高通、再稳定截取幅值尾部的正式密钥载体。"""

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
)


__all__ = [
    "HIGH_FREQUENCY_TAIL_PROTOCOL_DIGEST",
    "HighFrequencyTailCarrierTemplate",
    "build_high_frequency_tail_template",
]


_HF_TAIL_PRG_DOMAIN = "hf_tail_robust"


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
    """在调用密钥 PRG 前闭合正式 latent 的静态与内容边界。"""

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


def _paired_low_pass(raw: Any) -> Any:
    """使用与正式 LF 配对的冻结二维低通算子。"""

    import torch.nn.functional as functional

    if LOW_FREQUENCY_BOUNDARY_MODE != "zero_padding":
        raise RuntimeError("LF 边界协议必须为 zero_padding")
    return functional.avg_pool2d(
        raw,
        kernel_size=LOW_FREQUENCY_KERNEL_SIZE,
        stride=LOW_FREQUENCY_STRIDE,
        padding=LOW_FREQUENCY_PADDING,
        ceil_mode=LOW_FREQUENCY_CEIL_MODE,
        count_include_pad=LOW_FREQUENCY_COUNT_INCLUDE_PAD,
        divisor_override=LOW_FREQUENCY_DIVISOR_OVERRIDE,
    )


def _stable_high_frequency_tail(high_pass: Any) -> tuple[Any, int]:
    """按绝对值降序、flat index升序稳定保留每样本20%坐标。"""

    torch = _torch()
    flat = high_pass.reshape(-1)
    flat_values = flat.tolist()
    element_count = len(flat_values)
    selected_element_count = max(1, (element_count + 4) // 5)
    ranked_indices = sorted(
        range(element_count),
        key=lambda index: (-abs(flat_values[index]), index),
    )
    selected_indices = ranked_indices[:selected_element_count]
    selected_index_tensor = torch.tensor(
        selected_indices,
        dtype=torch.long,
        device="cpu",
    )
    sparse_flat = torch.zeros_like(flat)
    sparse_flat[selected_index_tensor] = flat[selected_index_tensor]
    sparse = sparse_flat.reshape(high_pass.shape)
    norm = sparse.norm()
    if not bool(torch.isfinite(norm)):
        raise RuntimeError("HF-tail L2 能量必须有限")
    if float(norm.item()) == 0.0:
        raise RuntimeError("HF-tail 没有可归一化的非零能量")
    normalized = sparse / norm
    if not bool(torch.isfinite(normalized).all()):
        raise RuntimeError("HF-tail L2 归一化产生非有限值")
    return normalized, selected_element_count


def _high_pass_identity_digest() -> str:
    """绑定正式高通、稳定尾部选择和归一化协议。"""

    return build_stable_digest(
        {
            "low_pass_kernel_size": LOW_FREQUENCY_KERNEL_SIZE,
            "low_pass_stride": LOW_FREQUENCY_STRIDE,
            "low_pass_padding": LOW_FREQUENCY_PADDING,
            "low_pass_boundary_mode": LOW_FREQUENCY_BOUNDARY_MODE,
            "low_pass_ceil_mode": LOW_FREQUENCY_CEIL_MODE,
            "low_pass_count_include_pad": LOW_FREQUENCY_COUNT_INCLUDE_PAD,
            "low_pass_divisor_override": LOW_FREQUENCY_DIVISOR_OVERRIDE,
            "high_pass_rule": "input_minus_paired_low_pass",
            "tail_selection_scope": "per_sample_flatten_channel_height_width",
            "tail_fraction_numerator": 1,
            "tail_fraction_denominator": 5,
            "selected_element_count_rule": "max_one_integer_ceil_one_fifth",
            "tail_order": "absolute_value_descending_then_flat_index_ascending",
            "unselected_value": 0.0,
            "normalization": "per_sample_float32_l2_without_centering",
        }
    )


HIGH_FREQUENCY_TAIL_PROTOCOL_DIGEST = _high_pass_identity_digest()


@dataclass(frozen=True)
class HighFrequencyTailCarrierTemplate:
    """保存正式 HF-tail 标准载体及其可重建身份。"""

    template: Any
    latent_shape: tuple[int, int, int, int]
    scoring_key_identity_digest: str
    model_identity_digest: str
    prg_version: str
    prg_domain: Literal["hf_tail_robust"]
    high_pass_identity_digest: str
    selected_element_count: int
    template_digest: str


def build_high_frequency_tail_template(
    reference_latent: Any,
    key_material: str,
    model_identity_digest: str,
    *,
    prg_version: str,
) -> HighFrequencyTailCarrierTemplate:
    """按参考 latent 形状先高通，再稳定保留20%幅值 tail。"""

    shape = _validate_reference_latent(reference_latent)
    if type(key_material) is not str or not key_material:
        raise ValueError("key_material 必须为非空精确 str")
    normalized_model_identity_digest = _require_sha256(
        model_identity_digest,
        label="model_identity_digest",
    )
    require_supported_keyed_prg_version(prg_version)

    raw = _validate_canonical_prg_tensor(
        build_keyed_gaussian_tensor(
            shape,
            key_material,
            {
                "operator": "latent_carrier_template",
                "branch_name": _HF_TAIL_PRG_DOMAIN,
                "model_identity_digest": normalized_model_identity_digest,
            },
            prg_version=prg_version,
        ),
        shape,
    )
    low_pass = _paired_low_pass(raw)
    if tuple(int(value) for value in low_pass.shape) != shape:
        raise RuntimeError("配对低通必须保持 HF-tail Tensor 形状")
    high_pass = raw - low_pass
    if not bool(_torch().isfinite(high_pass).all()):
        raise RuntimeError("HF-tail 高通残差必须全部有限")
    canonical_template, selected_element_count = _stable_high_frequency_tail(
        high_pass
    )

    scoring_key_identity_digest = build_stable_digest(
        {"key_material": key_material}
    )
    high_pass_identity_digest = HIGH_FREQUENCY_TAIL_PROTOCOL_DIGEST
    template_content_sha256 = tensor_content_sha256(canonical_template)
    template_digest = build_stable_digest(
        {
            "carrier_template": "high_frequency_tail",
            "latent_shape": list(shape),
            "scoring_key_identity_digest": scoring_key_identity_digest,
            "model_identity_digest": normalized_model_identity_digest,
            "prg_version": prg_version,
            "prg_domain": _HF_TAIL_PRG_DOMAIN,
            "high_pass_identity_digest": high_pass_identity_digest,
            "selected_element_count": selected_element_count,
            "template_content_sha256": template_content_sha256,
        }
    )
    template = canonical_template.to(
        device=reference_latent.device,
        dtype=_torch().float32,
    )
    return HighFrequencyTailCarrierTemplate(
        template=template,
        latent_shape=shape,
        scoring_key_identity_digest=scoring_key_identity_digest,
        model_identity_digest=normalized_model_identity_digest,
        prg_version=prg_version,
        prg_domain=_HF_TAIL_PRG_DOMAIN,
        high_pass_identity_digest=high_pass_identity_digest,
        selected_element_count=selected_element_count,
        template_digest=template_digest,
    )
