"""按正式内容路由构造 LF 与 HF-tail 的 float32 名义更新。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from main.core.keyed_prg import require_supported_keyed_prg_version
from main.methods.carrier.high_frequency_tail import (
    HighFrequencyTailCarrierTemplate,
)
from main.methods.carrier.low_frequency import LowFrequencyCarrierTemplate
from main.methods.content.routing import ContentRoutingResult


__all__ = [
    "ContentCarrierUpdateResult",
    "build_content_carrier_update",
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
_TEMPLATE_UNIT_NORM_RTOL = 1.0e-5
_TEMPLATE_UNIT_NORM_ATOL = 1.0e-6
_LF_RELATIVE_STRENGTH = 0.0025
_HF_TAIL_RELATIVE_STRENGTH = 0.0015
_CONTENT_STRENGTH_COMMON_MULTIPLIERS = (0.75, 1.0, 1.25)


def _torch() -> Any:
    """延迟导入 PyTorch，保持模块导入边界轻量。"""

    import torch

    return torch


def _role_activity(method_role: Any) -> tuple[bool, bool, bool]:
    """最先解析角色的 LF/HF 活动状态与统一路由消融。"""

    if type(method_role) is not str or method_role not in _FORMAL_METHOD_ROLES:
        raise ValueError("method_role 必须为登记的精确方法角色")
    return (
        method_role != "hf_tail_only_content",
        method_role != "lf_only_content",
        method_role == "uniform_content_routing",
    )


def _require_sha256(value: Any, *, label: str) -> str:
    """要求规范的64位小写十六进制 SHA-256。"""

    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} 必须为规范 SHA-256")
    return value


def _latent_metadata(value: Any) -> tuple[int, int, int, int]:
    """只读取元数据并验证 callback 索引10 latent 的静态边界。"""

    torch = _torch()
    if not isinstance(value, torch.Tensor):
        raise TypeError("current_scheduler_latent 必须为 Tensor")
    if not value.dtype.is_floating_point:
        raise TypeError("current_scheduler_latent 必须使用真实浮点 dtype")
    if value.device.type == "meta":
        raise ValueError("current_scheduler_latent 必须是已物化 Tensor")
    if value.ndim != 4:
        raise ValueError("current_scheduler_latent 必须具有 [1, C, H, W] 形状")
    shape = tuple(int(member) for member in value.shape)
    if shape[0] != 1 or any(member <= 0 for member in shape[1:]):
        raise ValueError(
            "current_scheduler_latent 必须具有正尺寸 [1, C, H, W] 形状"
        )
    return shape


def _require_latent_shape(
    value: Any,
    *,
    label: str,
    expected: tuple[int, int, int, int],
) -> None:
    """验证模板记录的 exact tuple 形状身份并拒绝 bool。"""

    if type(value) is not tuple or len(value) != 4:
        raise TypeError(f"{label} 必须为精确四元 tuple")
    if any(type(member) is not int for member in value):
        raise TypeError(f"{label} 成员必须为精确 int")
    if value[0] != 1 or any(member <= 0 for member in value[1:]):
        raise ValueError(f"{label} 必须描述正尺寸 [1, C, H, W]")
    if value != expected:
        raise ValueError(f"{label} 与 current_scheduler_latent 形状不一致")


def _require_float32_tensor_metadata(
    value: Any,
    *,
    label: str,
    expected_shape: tuple[int, ...],
    expected_device: Any,
) -> None:
    """只读取元数据并验证正式 float32 Tensor 边界。"""

    torch = _torch()
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{label} 必须为 Tensor")
    if value.dtype != torch.float32:
        raise TypeError(f"{label} 必须使用 float32")
    if value.device.type == "meta":
        raise ValueError(f"{label} 必须是已物化 Tensor")
    if tuple(int(member) for member in value.shape) != expected_shape:
        raise ValueError(f"{label} 形状与正式 latent 边界不一致")
    if value.device != expected_device:
        raise ValueError(f"{label} device 与 current_scheduler_latent 不一致")


def _validate_static_inputs_and_identity(
    current_scheduler_latent: Any,
    routing: Any,
    lf_template: Any,
    hf_tail_template: Any,
) -> tuple[int, int, int, int]:
    """在任何 Tensor 内容读取前闭合跨组件静态与身份门禁。"""

    shape = _latent_metadata(current_scheduler_latent)
    if type(routing) is not ContentRoutingResult:
        raise TypeError("routing 必须为正式 ContentRoutingResult")
    spatial_shape = (1, 1, shape[2], shape[3])
    for label, value in (
        ("routing.writable_capacity_map", routing.writable_capacity_map),
        ("routing.lf_mask", routing.lf_mask),
        ("routing.hf_tail_mask", routing.hf_tail_mask),
    ):
        _require_float32_tensor_metadata(
            value,
            label=label,
            expected_shape=spatial_shape,
            expected_device=current_scheduler_latent.device,
        )
    _require_sha256(
        routing.routing_identity_digest,
        label="routing.routing_identity_digest",
    )

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
    _require_float32_tensor_metadata(
        lf_template.template,
        label="lf_template.template",
        expected_shape=shape,
        expected_device=current_scheduler_latent.device,
    )
    _require_float32_tensor_metadata(
        hf_tail_template.template,
        label="hf_tail_template.template",
        expected_shape=shape,
        expected_device=current_scheduler_latent.device,
    )

    if lf_template.prg_domain != _LF_PRG_DOMAIN:
        raise ValueError("lf_template.prg_domain 必须为 lf_content")
    if hf_tail_template.prg_domain != _HF_TAIL_PRG_DOMAIN:
        raise ValueError("hf_tail_template.prg_domain 必须为 hf_tail_robust")

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
    _require_sha256(
        lf_template.filter_identity_digest,
        label="lf_template.filter_identity_digest",
    )
    _require_sha256(
        hf_tail_template.high_pass_identity_digest,
        label="hf_tail_template.high_pass_identity_digest",
    )
    _require_sha256(
        lf_template.template_digest,
        label="lf_template.template_digest",
    )
    _require_sha256(
        hf_tail_template.template_digest,
        label="hf_tail_template.template_digest",
    )
    if (
        type(hf_tail_template.selected_element_count) is not int
        or hf_tail_template.selected_element_count <= 0
    ):
        raise ValueError("hf_tail_template.selected_element_count 必须为正精确 int")
    if lf_template.prg_version != hf_tail_template.prg_version:
        raise ValueError("LF 与 HF-tail 模板 PRG 版本不一致")
    require_supported_keyed_prg_version(lf_template.prg_version)
    return shape


def _validate_routing_contents(routing: ContentRoutingResult) -> None:
    """验证三个实际路由图的有限与闭区间内容边界。"""

    torch = _torch()
    for label, value in (
        ("routing.writable_capacity_map", routing.writable_capacity_map),
        ("routing.lf_mask", routing.lf_mask),
        ("routing.hf_tail_mask", routing.hf_tail_mask),
    ):
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"{label} 必须全部有限")
        if bool(((value < 0.0) | (value > 1.0)).any()):
            raise ValueError(f"{label} 必须位于 [0, 1]")


def _validate_template_contents(value: Any, *, label: str) -> None:
    """以冻结float32容差验证正式模板近单位L2，不再归一化。"""

    torch = _torch()
    detached = value.detach()
    if not bool(torch.isfinite(detached).all()):
        raise ValueError(f"{label} 必须全部有限")
    norm = torch.linalg.vector_norm(detached.reshape(-1))
    if not bool(torch.isfinite(norm)):
        raise ValueError(f"{label} L2 必须有限")
    if not bool(
        torch.allclose(
            norm,
            torch.ones_like(norm),
            rtol=_TEMPLATE_UNIT_NORM_RTOL,
            atol=_TEMPLATE_UNIT_NORM_ATOL,
        )
    ):
        raise ValueError(f"{label} 必须在冻结float32容差内具有单位L2")


def _build_float32_updates(
    current_scheduler_latent: Any,
    routing: ContentRoutingResult,
    lf_template: LowFrequencyCarrierTemplate,
    hf_tail_template: HighFrequencyTailCarrierTemplate,
    *,
    lf_active: bool,
    hf_tail_active: bool,
    uniform_routing: bool,
    content_strength_common_multiplier: float,
) -> tuple[Any, Any, Any, Any, Any, Any, Any, Any, Any]:
    """按冻结float32顺序形成方向、名义更新与唯一内容基底。"""

    torch = _torch()
    latent_float32 = current_scheduler_latent.detach().to(dtype=torch.float32)
    if not bool(torch.isfinite(latent_float32).all()):
        raise ValueError("current_scheduler_latent 必须全部有限")
    latent_l2_tensor = torch.linalg.vector_norm(latent_float32.reshape(-1))
    if not bool(torch.isfinite(latent_l2_tensor)) or bool(latent_l2_tensor <= 0.0):
        raise ValueError("current_scheduler_latent float32 L2 必须有限且严格为正")

    writable_capacity = routing.writable_capacity_map.detach()
    lf_mask = routing.lf_mask.detach()
    hf_tail_mask = routing.hf_tail_mask.detach()
    if uniform_routing:
        writable_capacity = torch.ones_like(writable_capacity)
        lf_mask = torch.ones_like(lf_mask)
        hf_tail_mask = torch.ones_like(hf_tail_mask)

    lf_direction = lf_mask * lf_template.template.detach()
    hf_tail_direction = hf_tail_mask * hf_tail_template.template.detach()
    if not bool(torch.isfinite(lf_direction).all()) or not bool(
        torch.isfinite(hf_tail_direction).all()
    ):
        raise ValueError("内容载体方向必须全部有限")

    multiplier_tensor = latent_float32.new_tensor(
        content_strength_common_multiplier
    )
    lf_relative_strength_tensor = (
        latent_float32.new_tensor(_LF_RELATIVE_STRENGTH) * multiplier_tensor
    )
    hf_tail_relative_strength_tensor = (
        latent_float32.new_tensor(_HF_TAIL_RELATIVE_STRENGTH)
        * multiplier_tensor
    )
    lf_nominal_strength_tensor = (
        latent_l2_tensor * lf_relative_strength_tensor
    )
    hf_tail_nominal_strength_tensor = (
        latent_l2_tensor * hf_tail_relative_strength_tensor
    )
    lf_update = (
        lf_direction * lf_nominal_strength_tensor
        if lf_active
        else torch.zeros_like(lf_direction)
    )
    hf_tail_update = (
        hf_tail_direction * hf_tail_nominal_strength_tensor
        if hf_tail_active
        else torch.zeros_like(hf_tail_direction)
    )
    content_only_latent_float32 = latent_float32 + lf_update
    content_only_latent_float32 = content_only_latent_float32 + hf_tail_update
    for label, value in (
        ("lf_update", lf_update),
        ("hf_tail_update", hf_tail_update),
        ("content_only_latent_float32", content_only_latent_float32),
    ):
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"{label} 必须全部有限")
    return (
        writable_capacity,
        lf_direction,
        hf_tail_direction,
        lf_update,
        hf_tail_update,
        content_only_latent_float32,
        latent_l2_tensor,
        lf_nominal_strength_tensor,
        hf_tail_nominal_strength_tensor,
    )


@dataclass(frozen=True)
class ContentCarrierUpdateResult:
    """保存角色解析后的正式内容方向、名义更新与float32基底。"""

    geometry_capacity_map: Any
    lf_direction: Any
    hf_tail_direction: Any
    lf_update: Any
    hf_tail_update: Any
    content_only_latent_float32: Any
    latent_l2: float
    lf_nominal_strength: float
    hf_tail_nominal_strength: float
    method_role: str


def build_content_carrier_update(
    *,
    current_scheduler_latent: Any,
    routing: ContentRoutingResult,
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
    content_strength_common_multiplier: float = 1.0,
) -> ContentCarrierUpdateResult:
    """以冻结角色、掩码和名义强度构造几何同步前内容基底。"""

    lf_active, hf_tail_active, uniform_routing = _role_activity(method_role)
    if (
        type(content_strength_common_multiplier) is not float
        or content_strength_common_multiplier
        not in _CONTENT_STRENGTH_COMMON_MULTIPLIERS
    ):
        raise ValueError(
            "content_strength_common_multiplier 必须为 0.75、1.0 或 1.25"
        )
    _validate_static_inputs_and_identity(
        current_scheduler_latent,
        routing,
        lf_template,
        hf_tail_template,
    )
    _validate_routing_contents(routing)
    _validate_template_contents(lf_template.template, label="lf_template.template")
    _validate_template_contents(
        hf_tail_template.template,
        label="hf_tail_template.template",
    )
    (
        geometry_capacity_map,
        lf_direction,
        hf_tail_direction,
        lf_update,
        hf_tail_update,
        content_only_latent_float32,
        latent_l2_tensor,
        lf_nominal_strength_tensor,
        hf_tail_nominal_strength_tensor,
    ) = _build_float32_updates(
        current_scheduler_latent,
        routing,
        lf_template,
        hf_tail_template,
        lf_active=lf_active,
        hf_tail_active=hf_tail_active,
        uniform_routing=uniform_routing,
        content_strength_common_multiplier=content_strength_common_multiplier,
    )
    return ContentCarrierUpdateResult(
        geometry_capacity_map=geometry_capacity_map,
        lf_direction=lf_direction,
        hf_tail_direction=hf_tail_direction,
        lf_update=lf_update,
        hf_tail_update=hf_tail_update,
        content_only_latent_float32=content_only_latent_float32,
        latent_l2=float(latent_l2_tensor.item()),
        lf_nominal_strength=float(lf_nominal_strength_tensor.item()),
        hf_tail_nominal_strength=float(hf_tail_nominal_strength_tensor.item()),
        method_role=method_role,
    )
