"""计算正式 LF 与 HF-tail 密钥载体的盲内容相关分数。"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING, Any, Literal

from main.core.digest import build_stable_digest, tensor_content_sha256
from main.core.keyed_prg import require_supported_keyed_prg_version
from main.methods.carrier.high_frequency_tail import (
    HighFrequencyTailCarrierTemplate,
)
from main.methods.carrier.low_frequency import LowFrequencyCarrierTemplate


if TYPE_CHECKING:
    from torch import Tensor


__all__ = [
    "BlindContentScore",
    "compute_blind_content_score",
]


_FORMAL_METHOD_ROLES = (
    "full_dual_chain",
    "uniform_content_routing",
    "lf_only_content",
    "hf_tail_only_content",
    "content_chain_only",
    "geometry_recovery_without_embedded_sync",
)
_LF_PRG_DOMAIN = "lf_content"
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


def _role_weights(method_role: Any) -> tuple[float, float]:
    """最先解析六个正式方法角色对应的冻结分支权重。"""

    if type(method_role) is not str or method_role not in _FORMAL_METHOD_ROLES:
        raise ValueError("method_role 必须为登记的精确方法角色")
    if method_role == "lf_only_content":
        return 1.0, 0.0
    if method_role == "hf_tail_only_content":
        return 0.0, 1.0
    return 0.70, 0.30


def _observed_shape(observed_latent: Any) -> tuple[int, int, int, int]:
    """只读取元数据并验证正式观测 latent 的静态边界。"""

    torch = _torch()
    if not isinstance(observed_latent, torch.Tensor):
        raise TypeError("observed_latent 必须为 Tensor")
    if not observed_latent.dtype.is_floating_point:
        raise TypeError("observed_latent 必须使用真实浮点 dtype")
    if observed_latent.device.type == "meta":
        raise ValueError("observed_latent 必须是已物化 Tensor")
    if observed_latent.ndim != 4:
        raise ValueError("observed_latent 必须具有 [1, C, H, W] 形状")
    shape = tuple(int(value) for value in observed_latent.shape)
    if shape[0] != 1 or any(value <= 0 for value in shape[1:]):
        raise ValueError("observed_latent 必须具有正尺寸 [1, C, H, W] 形状")
    return shape


def _require_latent_shape(
    value: Any,
    *,
    label: str,
    expected: tuple[int, int, int, int],
) -> None:
    """验证模板记录的 exact tuple 形状身份，显式拒绝 bool。"""

    if type(value) is not tuple or len(value) != 4:
        raise TypeError(f"{label} 必须为精确四元 tuple")
    if any(type(member) is not int for member in value):
        raise TypeError(f"{label} 成员必须为精确 int")
    if value[0] != 1 or any(member <= 0 for member in value[1:]):
        raise ValueError(f"{label} 必须描述正尺寸 [1, C, H, W]")
    if value != expected:
        raise ValueError(f"{label} 与 observed_latent 形状不一致")


def _require_template_tensor_metadata(
    value: Any,
    *,
    label: str,
    expected_shape: tuple[int, int, int, int],
    expected_device: Any,
) -> None:
    """只读取元数据并验证正式模板 Tensor 边界。"""

    torch = _torch()
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{label} 必须为 Tensor")
    if value.dtype != torch.float32:
        raise TypeError(f"{label} 必须使用 float32")
    if value.device.type == "meta":
        raise ValueError(f"{label} 必须是已物化 Tensor")
    if tuple(int(member) for member in value.shape) != expected_shape:
        raise ValueError(f"{label} 形状与 observed_latent 不一致")
    if value.device != expected_device:
        raise ValueError(f"{label} device 与 observed_latent 不一致")


def _validate_static_inputs_and_identity(
    observed_latent: Any,
    lf_template: Any,
    hf_tail_template: Any,
) -> tuple[str, str, str]:
    """在任何 Tensor 内容读取前闭合形状与跨模板身份。"""

    shape = _observed_shape(observed_latent)
    if type(lf_template) is not LowFrequencyCarrierTemplate:
        raise TypeError("lf_template 必须为正式 LowFrequencyCarrierTemplate")
    if type(hf_tail_template) is not HighFrequencyTailCarrierTemplate:
        raise TypeError(
            "hf_tail_template 必须为正式 HighFrequencyTailCarrierTemplate"
        )

    _require_latent_shape(
        lf_template.latent_shape,
        label="lf_template.latent_shape",
        expected=shape,
    )
    _require_latent_shape(
        hf_tail_template.latent_shape,
        label="hf_tail_template.latent_shape",
        expected=shape,
    )
    _require_template_tensor_metadata(
        lf_template.template,
        label="lf_template.template",
        expected_shape=shape,
        expected_device=observed_latent.device,
    )
    _require_template_tensor_metadata(
        hf_tail_template.template,
        label="hf_tail_template.template",
        expected_shape=shape,
        expected_device=observed_latent.device,
    )

    if lf_template.prg_domain != _LF_PRG_DOMAIN:
        raise ValueError("lf_template.prg_domain 必须为 lf_content")
    if hf_tail_template.prg_domain != _HF_TAIL_PRG_DOMAIN:
        raise ValueError(
            "hf_tail_template.prg_domain 必须为 hf_tail_robust"
        )

    lf_model_digest = _require_sha256(
        lf_template.model_identity_digest,
        label="lf_template.model_identity_digest",
    )
    hf_model_digest = _require_sha256(
        hf_tail_template.model_identity_digest,
        label="hf_tail_template.model_identity_digest",
    )
    if lf_model_digest != hf_model_digest:
        raise ValueError("LF 与 HF-tail 模板模型身份不一致")

    lf_key_digest = _require_sha256(
        lf_template.scoring_key_identity_digest,
        label="lf_template.scoring_key_identity_digest",
    )
    hf_key_digest = _require_sha256(
        hf_tail_template.scoring_key_identity_digest,
        label="hf_tail_template.scoring_key_identity_digest",
    )
    if lf_key_digest != hf_key_digest:
        raise ValueError("LF 与 HF-tail 模板评分密钥身份不一致")

    _require_sha256(lf_template.template_digest, label="lf_template.template_digest")
    _require_sha256(
        hf_tail_template.template_digest,
        label="hf_tail_template.template_digest",
    )
    if lf_template.prg_version != hf_tail_template.prg_version:
        raise ValueError("LF 与 HF-tail 模板 PRG 版本不一致")
    require_supported_keyed_prg_version(lf_template.prg_version)
    return lf_model_digest, lf_key_digest, lf_template.prg_version


def _require_finite_tensor(value: Any, *, label: str) -> None:
    """在全部静态与身份门禁闭合后验证 Tensor 内容有限。"""

    if not bool(_torch().isfinite(value).all()):
        raise ValueError(f"{label} 必须全部有限")


def _normalized_correlation(left: Any, right: Any) -> float:
    """分别中心化并 L2 后计算未裁剪、未舍入的 float32 内积。"""

    torch = _torch()
    left_flat = left.detach().to(dtype=torch.float32).reshape(-1)
    right_flat = right.detach().to(dtype=torch.float32).reshape(-1)
    if left_flat.numel() != right_flat.numel():
        raise ValueError("盲相关输入元素数必须一致")
    left_centered = left_flat - left_flat.mean()
    right_centered = right_flat - right_flat.mean()
    left_norm = torch.linalg.vector_norm(left_centered)
    right_norm = torch.linalg.vector_norm(right_centered)
    if not bool(torch.isfinite(left_norm)) or not bool(torch.isfinite(right_norm)):
        raise ValueError("盲相关中心化 L2 能量必须有限")
    if float(left_norm.item()) == 0.0 or float(right_norm.item()) == 0.0:
        raise ValueError("盲相关输入必须具有非零中心化能量")
    score_tensor = torch.dot(
        left_centered / left_norm,
        right_centered / right_norm,
    )
    score = float(score_tensor.item())
    if not math.isfinite(score):
        raise ValueError("盲相关分数必须有限")
    return score


@dataclass(frozen=True)
class BlindContentScore:
    """保存正式 LF、HF-tail 与角色加权总内容分数。"""

    blind_lf_score: float
    blind_hf_tail_score: float
    blind_content_score: float
    lf_weight: float
    hf_tail_weight: float
    method_role: str
    scoring_key_identity_digest: str
    score_identity_digest: str


def compute_blind_content_score(
    observed_latent: Tensor,
    lf_template: LowFrequencyCarrierTemplate,
    hf_tail_template: HighFrequencyTailCarrierTemplate,
    method_role: Literal[
        "full_dual_chain",
        "uniform_content_routing",
        "lf_only_content",
        "hf_tail_only_content",
        "content_chain_only",
        "geometry_recovery_without_embedded_sync",
    ],
) -> BlindContentScore:
    """按方法角色的冻结权重返回 LF、HF-tail 和总内容分数。"""

    lf_weight, hf_tail_weight = _role_weights(method_role)
    model_identity_digest, scoring_key_identity_digest, prg_version = (
        _validate_static_inputs_and_identity(
            observed_latent,
            lf_template,
            hf_tail_template,
        )
    )

    _require_finite_tensor(observed_latent, label="observed_latent")
    _require_finite_tensor(lf_template.template, label="lf_template.template")
    _require_finite_tensor(
        hf_tail_template.template,
        label="hf_tail_template.template",
    )
    blind_lf_score = _normalized_correlation(
        observed_latent,
        lf_template.template,
    )
    blind_hf_tail_score = _normalized_correlation(
        observed_latent,
        hf_tail_template.template,
    )
    blind_content_score = (
        lf_weight * blind_lf_score + hf_tail_weight * blind_hf_tail_score
    )
    if not math.isfinite(blind_content_score):
        raise ValueError("blind_content_score 必须有限")

    observed_latent_content_sha256 = tensor_content_sha256(observed_latent)
    lf_template_content_sha256 = tensor_content_sha256(lf_template.template)
    hf_tail_template_content_sha256 = tensor_content_sha256(
        hf_tail_template.template
    )
    score_identity_digest = build_stable_digest(
        {
            "observed_latent_content_sha256": observed_latent_content_sha256,
            "lf_template_digest": lf_template.template_digest,
            "lf_template_content_sha256": lf_template_content_sha256,
            "hf_tail_template_digest": hf_tail_template.template_digest,
            "hf_tail_template_content_sha256": hf_tail_template_content_sha256,
            "model_identity_digest": model_identity_digest,
            "prg_version": prg_version,
            "scoring_key_identity_digest": scoring_key_identity_digest,
            "method_role": method_role,
            "lf_weight": lf_weight,
            "hf_tail_weight": hf_tail_weight,
            "blind_lf_score": blind_lf_score,
            "blind_hf_tail_score": blind_hf_tail_score,
            "blind_content_score": blind_content_score,
        }
    )
    return BlindContentScore(
        blind_lf_score=blind_lf_score,
        blind_hf_tail_score=blind_hf_tail_score,
        blind_content_score=blind_content_score,
        lf_weight=lf_weight,
        hf_tail_weight=hf_tail_weight,
        method_role=method_role,
        scoring_key_identity_digest=scoring_key_identity_digest,
        score_identity_digest=score_identity_digest,
    )
