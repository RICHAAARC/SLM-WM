"""构造受分支风险硬包络约束的更新与实际 dtype 合成候选。"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterator, Mapping, Sequence

from main.core.digest import (
    TENSOR_CONTENT_DIGEST_VERSION,
    build_stable_digest,
    tensor_content_sha256,
)


_DUAL_CHAIN_LF_RELATIVE_STRENGTH = 0.0025
_DUAL_CHAIN_HF_RELATIVE_STRENGTH = 0.0015
_DUAL_CHAIN_GEOMETRY_RELATIVE_STRENGTH = 0.0010
_DUAL_CHAIN_COMBINED_RELATIVE_L2_LIMIT = 0.0050
_DUAL_CHAIN_BACKTRACKING_FACTOR = 0.5
_DUAL_CHAIN_MAXIMUM_STEPS = 24
_DUAL_CHAIN_METHOD_ROLE_ACTIVITY = {
    "full_dual_chain": (True, True, True),
    "uniform_content_routing": (True, True, True),
    "lf_only_content": (True, False, True),
    "hf_tail_only_content": (False, True, True),
    "content_chain_only": (True, True, False),
    "geometry_recovery_without_embedded_sync": (True, True, False),
}


def _dual_chain_budget_payload() -> dict[str, Any]:
    """Return the single formal common-backtracking budget identity."""

    return {
        "lf_relative_strength": _DUAL_CHAIN_LF_RELATIVE_STRENGTH,
        "hf_tail_relative_strength": _DUAL_CHAIN_HF_RELATIVE_STRENGTH,
        "geometry_relative_strength": _DUAL_CHAIN_GEOMETRY_RELATIVE_STRENGTH,
        "combined_relative_l2_limit": _DUAL_CHAIN_COMBINED_RELATIVE_L2_LIMIT,
        "common_backtracking_factor": _DUAL_CHAIN_BACKTRACKING_FACTOR,
        "common_backtracking_maximum_steps": _DUAL_CHAIN_MAXIMUM_STEPS,
    }


@dataclass(frozen=True)
class DualChainWriteBudget:
    """Hold the fixed three-branch common-backtracking budget."""

    lf_relative_strength: float
    hf_tail_relative_strength: float
    geometry_relative_strength: float
    combined_relative_l2_limit: float
    common_backtracking_factor: float
    common_backtracking_maximum_steps: int
    budget_identity_digest: str

    def __post_init__(self) -> None:
        expected = _dual_chain_budget_payload()
        actual = {
            key: getattr(self, key)
            for key in expected
        }
        if any(type(value) is not type(expected[key]) for key, value in actual.items()):
            raise TypeError("dual-chain budget fields must use exact governed types")
        if actual != expected:
            raise ValueError("dual-chain budget fields must equal the formal constants")
        if self.budget_identity_digest != build_stable_digest(expected):
            raise ValueError("budget_identity_digest does not bind the formal budget")


@dataclass(frozen=True)
class DualChainWriteResult:
    """Hold the first quantization-safe common-scale single write."""

    written_latent: Any
    lf_update_digest: str
    hf_tail_update_digest: str
    geometry_update_digest: str
    lf_effective_l2: float
    hf_tail_effective_l2: float
    geometry_effective_l2: float
    combined_update_digest: str
    combined_effective_l2: float
    accepted_common_scale: float
    actual_dtype_write_digest: str
    write_identity_digest: str


def formal_dual_chain_write_budget() -> DualChainWriteBudget:
    """Construct the only formal three-branch write budget."""

    payload = _dual_chain_budget_payload()
    return DualChainWriteBudget(
        **payload,
        budget_identity_digest=build_stable_digest(payload),
    )


def _formal_branch_activity(method_role: Any) -> tuple[bool, bool, bool]:
    """Resolve the exact LF/HF/geometry activity registered by the method role."""

    if (
        type(method_role) is not str
        or method_role not in _DUAL_CHAIN_METHOD_ROLE_ACTIVITY
    ):
        raise ValueError("method_role must be an exact formal method role")
    return _DUAL_CHAIN_METHOD_ROLE_ACTIVITY[method_role]


def compose_dual_chain_update_once(
    latent: Any,
    lf_update: Any,
    hf_tail_update: Any,
    geometry_update: Any,
    budget: DualChainWriteBudget,
    *,
    method_role: str,
) -> DualChainWriteResult:
    """Backtrack all active branches together and cast/write exactly once."""

    active = _formal_branch_activity(method_role)
    torch = _torch()
    if type(budget) is not DualChainWriteBudget:
        raise TypeError("budget must be an exact DualChainWriteBudget")
    # Re-run the frozen constructor invariant for forged dataclass instances.
    budget.__post_init__()
    if not torch.is_tensor(latent) or not latent.dtype.is_floating_point:
        raise TypeError("latent must be a real floating Tensor")
    if latent.ndim != 4 or latent.shape[0] != 1 or latent.numel() == 0:
        raise ValueError("latent must have non-empty [1,C,H,W] shape")
    if latent.device.type == "meta":
        raise ValueError("latent must be materialized")
    shape = tuple(latent.shape)
    for label, update in (
        ("lf_update", lf_update),
        ("hf_tail_update", hf_tail_update),
        ("geometry_update", geometry_update),
    ):
        if (
            not torch.is_tensor(update)
            or update.dtype != torch.float32
            or tuple(update.shape) != shape
            or update.device != latent.device
        ):
            raise ValueError(f"{label} must be same-shape/device float32")
    latent_float32 = latent.detach().float()
    if not bool(torch.isfinite(latent_float32).all()):
        raise ValueError("latent must be finite")
    updates = (lf_update.detach(), hf_tail_update.detach(), geometry_update.detach())
    if any(not bool(torch.isfinite(update).all()) for update in updates):
        raise ValueError("branch updates must be finite")
    if any(
        not enabled and bool(torch.count_nonzero(update).item())
        for enabled, update in zip(active, updates)
    ):
        raise ValueError("disabled formal branches must be exact zero tensors")
    latent_l2 = torch.linalg.vector_norm(latent_float32.reshape(-1))
    if not bool(torch.isfinite(latent_l2)) or bool(latent_l2 <= 0.0):
        raise ValueError("latent float32 L2 must be positive and finite")
    combined_limit = latent_l2 * latent_float32.new_tensor(
        budget.combined_relative_l2_limit
    )

    accepted: tuple[Any, ...] | None = None
    for step in range(budget.common_backtracking_maximum_steps + 1):
        scale = latent_float32.new_tensor(
            budget.common_backtracking_factor
        ).pow(step)
        scaled = tuple(update * scale for update in updates)
        # These casts are branch quantization gates only; the accepted latent is
        # still produced by the single combined cast below.
        effective = tuple(
            (latent_float32 + branch).to(dtype=latent.dtype).detach().float()
            - latent_float32
            for branch in scaled
        )
        effective_l2 = tuple(
            torch.linalg.vector_norm(branch.reshape(-1)) for branch in effective
        )
        if any(
            enabled
            and (
                not bool(torch.isfinite(norm))
                or float(norm.item()) <= 0.0
            )
            for enabled, norm in zip(active, effective_l2)
        ):
            continue
        combined = scaled[0] + scaled[1]
        combined = combined + scaled[2]
        candidate = (latent_float32 + combined).to(dtype=latent.dtype)
        actual_delta = candidate.detach().float() - latent_float32
        actual_l2 = torch.linalg.vector_norm(actual_delta.reshape(-1))
        if (
            not bool(torch.isfinite(candidate).all())
            or not bool(torch.isfinite(actual_delta).all())
            or not bool(torch.isfinite(actual_l2))
            or float(actual_l2.item()) <= 0.0
            or bool(actual_l2 > combined_limit)
        ):
            continue
        accepted = (scale, scaled, effective_l2, combined, candidate, actual_delta, actual_l2)
        break
    if accepted is None:
        raise ValueError("common gamma did not produce a quantization-safe single write")

    scale, scaled, effective_l2, combined, candidate, actual_delta, actual_l2 = accepted
    branch_digests = tuple(tensor_content_sha256(value) for value in scaled)
    combined_digest = tensor_content_sha256(combined)
    write_digest = tensor_content_sha256(actual_delta)
    identity = {
        "method_role": method_role,
        "formal_branch_activity": active,
        "budget_identity_digest": budget.budget_identity_digest,
        "original_latent_content_sha256": tensor_content_sha256(latent_float32),
        "branch_update_content_sha256": branch_digests,
        "combined_update_content_sha256": combined_digest,
        "actual_dtype_write_content_sha256": write_digest,
        "accepted_common_scale": float(scale.item()),
        "combined_effective_l2": float(actual_l2.item()),
        "write_dtype": str(latent.dtype),
        "write_shape": list(shape),
    }
    return DualChainWriteResult(
        written_latent=candidate,
        lf_update_digest=branch_digests[0],
        hf_tail_update_digest=branch_digests[1],
        geometry_update_digest=branch_digests[2],
        lf_effective_l2=float(effective_l2[0].item()) if active[0] else 0.0,
        hf_tail_effective_l2=float(effective_l2[1].item()) if active[1] else 0.0,
        geometry_effective_l2=float(effective_l2[2].item()) if active[2] else 0.0,
        combined_update_digest=combined_digest,
        combined_effective_l2=float(actual_l2.item()),
        accepted_common_scale=float(scale.item()),
        actual_dtype_write_digest=write_digest,
        write_identity_digest=build_stable_digest(identity),
    )


QUANTIZED_COMPOSITION_ORDER = (
    "lf_content",
    "tail_robust",
    "attention_geometry",
)

QUANTIZED_COMPOSITION_EVIDENCE_VERSION = (
    "slm_wm_quantized_composition_evidence"
)


def _torch() -> Any:
    """延迟导入 PyTorch, 保持治理工具的轻量导入边界。"""

    import torch

    return torch


def _batch_values(value: Any) -> float | list[float]:
    """把逐样本 Tensor 转换为 batch=1 标量或多样本列表。"""

    values = value.detach().cpu().reshape(-1).tolist()
    return float(values[0]) if len(values) == 1 else [float(item) for item in values]


def _content_sha256(value: Any, *, field_name: str) -> str:
    """验证记录中的 Tensor 内容摘要, 防止任意说明字符串进入证据载荷。"""

    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field_name} 必须为小写 SHA-256 摘要")
    return value


def _finite_record_number(value: Any, *, field_name: str) -> float:
    """把 JSON 记录数值规范化为有限 float, 供稳定摘要跨进程重算。"""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} 必须为有限数值")
    resolved = float(value)
    if not math.isfinite(resolved):
        raise ValueError(f"{field_name} 必须为有限数值")
    return resolved


def _maximum_envelope_ratio_record(value: Any) -> float | list[float]:
    """规范化标量或逐样本最大包络比例, 同时拒绝非有限自证值。"""

    if isinstance(value, list):
        if not value:
            raise ValueError("quantized_write_maximum_envelope_ratio 不得为空")
        resolved = [
            _finite_record_number(
                item,
                field_name="quantized_write_maximum_envelope_ratio",
            )
            for item in value
        ]
        if any(item < 0.0 for item in resolved):
            raise ValueError("quantized_write_maximum_envelope_ratio 必须非负")
        return resolved
    resolved_scalar = _finite_record_number(
        value,
        field_name="quantized_write_maximum_envelope_ratio",
    )
    if resolved_scalar < 0.0:
        raise ValueError("quantized_write_maximum_envelope_ratio 必须非负")
    return resolved_scalar


def _backtracking_trace(
    *,
    common_scale: Any,
    backtracking_factor: Any,
    backtracking_step_count: Any,
) -> tuple[float, float, int]:
    """验证共同缩放必须精确来自冻结回溯因子的整数次幂。"""

    resolved_scale = _finite_record_number(
        common_scale,
        field_name="quantized_write_common_scale",
    )
    resolved_factor = _finite_record_number(
        backtracking_factor,
        field_name="quantized_write_backtracking_factor",
    )
    if not 0.0 < resolved_scale <= 1.0:
        raise ValueError("quantized_write_common_scale 必须位于 (0, 1]")
    if not 0.0 < resolved_factor < 1.0:
        raise ValueError("quantized_write_backtracking_factor 必须位于 (0, 1)")
    if (
        isinstance(backtracking_step_count, bool)
        or not isinstance(backtracking_step_count, int)
        or backtracking_step_count < 0
    ):
        raise ValueError("quantized_write_backtracking_step_count 必须为非负整数")
    expected_scale = resolved_factor**backtracking_step_count
    if resolved_scale != expected_scale:
        raise ValueError(
            "quantized_write_common_scale 必须精确等于 "
            "quantized_write_backtracking_factor 的 step 次幂"
        )
    return resolved_scale, resolved_factor, backtracking_step_count


def _infer_backtracking_step(*, common_scale: float, backtracking_factor: float) -> int:
    """为直接构造调用恢复唯一整数步数, 不允许任意共同缩放绕过公式。"""

    if common_scale == 1.0:
        return 0
    approximate_step = math.log(common_scale) / math.log(backtracking_factor)
    resolved_step = int(round(approximate_step))
    if resolved_step < 0 or backtracking_factor**resolved_step != common_scale:
        raise ValueError(
            "common_scale 必须能够由 backtracking_factor 的非负整数次幂精确表示"
        )
    return resolved_step


def recompute_quantized_composition_evidence_digest(
    record: Mapping[str, Any],
) -> str:
    """仅根据记录字段重算量化合成证据摘要, 不读取运行时 Tensor。"""

    if not isinstance(record, Mapping):
        raise ValueError("量化合成证据必须为映射记录")
    if (
        record.get("quantized_composition_evidence_version")
        != QUANTIZED_COMPOSITION_EVIDENCE_VERSION
    ):
        raise ValueError("量化合成证据版本不受支持")
    if record.get("tensor_content_digest_version") != TENSOR_CONTENT_DIGEST_VERSION:
        raise ValueError("Tensor 内容摘要版本不受支持")

    composition_order = record.get("quantized_write_composition_order")
    if composition_order != list(QUANTIZED_COMPOSITION_ORDER):
        raise ValueError("quantized_write_composition_order 必须等于冻结三分支顺序")
    active_branch_order = record.get("quantized_write_active_branch_order")
    if not isinstance(active_branch_order, list) or not active_branch_order:
        raise ValueError("quantized_write_active_branch_order 必须为非空列表")
    if any(not isinstance(role, str) for role in active_branch_order):
        raise ValueError("quantized_write_active_branch_order 只能包含分支名称")
    expected_active_order = [
        role for role in QUANTIZED_COMPOSITION_ORDER if role in active_branch_order
    ]
    if (
        active_branch_order != expected_active_order
        or len(set(active_branch_order)) != len(active_branch_order)
    ):
        raise ValueError("活动分支必须是冻结三分支顺序的无重复子序列")

    branch_identities = record.get("quantized_write_branch_content_identities")
    if not isinstance(branch_identities, Mapping) or set(branch_identities) != set(
        active_branch_order
    ):
        raise ValueError("活动分支顺序与分支 Tensor 身份不一致")
    canonical_branch_identities: dict[str, dict[str, str]] = {}
    for role in active_branch_order:
        identity = branch_identities[role]
        if not isinstance(identity, Mapping):
            raise ValueError("每个活动分支必须提供 update 与 envelope Tensor 身份")
        canonical_branch_identities[role] = {
            "branch_written_update_content_sha256": _content_sha256(
                identity.get("branch_written_update_content_sha256"),
                field_name=(
                    f"{role}.branch_written_update_content_sha256"
                ),
            ),
            "branch_budget_envelope_content_sha256": _content_sha256(
                identity.get("branch_budget_envelope_content_sha256"),
                field_name=(
                    f"{role}.branch_budget_envelope_content_sha256"
                ),
            ),
        }

    write_shape = record.get("quantized_write_update_shape")
    if (
        not isinstance(write_shape, list)
        or len(write_shape) != 4
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in write_shape
        )
    ):
        raise ValueError("quantized_write_update_shape 必须为正整数 NCHW 形状")
    write_dtype = record.get("quantized_write_update_dtype")
    if not isinstance(write_dtype, str) or not write_dtype:
        raise ValueError("quantized_write_update_dtype 必须为非空字符串")
    common_scale, backtracking_factor, backtracking_step_count = _backtracking_trace(
        common_scale=record.get("quantized_write_common_scale"),
        backtracking_factor=record.get("quantized_write_backtracking_factor"),
        backtracking_step_count=record.get(
            "quantized_write_backtracking_step_count"
        ),
    )
    envelope_ready = record.get("quantized_write_budget_envelope_ready")
    if not isinstance(envelope_ready, bool):
        raise ValueError("quantized_write_budget_envelope_ready 必须为布尔值")

    payload = {
        "quantized_composition_evidence_version": (
            QUANTIZED_COMPOSITION_EVIDENCE_VERSION
        ),
        "tensor_content_digest_version": TENSOR_CONTENT_DIGEST_VERSION,
        "quantized_write_original_latent_content_sha256": _content_sha256(
            record.get("quantized_write_original_latent_content_sha256"),
            field_name="quantized_write_original_latent_content_sha256",
        ),
        "quantized_write_candidate_latent_content_sha256": _content_sha256(
            record.get("quantized_write_candidate_latent_content_sha256"),
            field_name="quantized_write_candidate_latent_content_sha256",
        ),
        "quantized_write_update_content_sha256": _content_sha256(
            record.get("quantized_write_update_content_sha256"),
            field_name="quantized_write_update_content_sha256",
        ),
        "quantized_write_update_dtype": write_dtype,
        "quantized_write_update_shape": list(write_shape),
        "quantized_write_composition_order": list(QUANTIZED_COMPOSITION_ORDER),
        "quantized_write_active_branch_order": list(active_branch_order),
        "quantized_write_branch_content_identities": canonical_branch_identities,
        "combined_update_content_sha256": _content_sha256(
            record.get("combined_update_content_sha256"),
            field_name="combined_update_content_sha256",
        ),
        "combined_budget_envelope_content_sha256": _content_sha256(
            record.get("combined_budget_envelope_content_sha256"),
            field_name="combined_budget_envelope_content_sha256",
        ),
        "quantized_write_common_scale": common_scale,
        "quantized_write_backtracking_factor": backtracking_factor,
        "quantized_write_backtracking_step_count": backtracking_step_count,
        "quantized_write_maximum_envelope_ratio": (
            _maximum_envelope_ratio_record(
                record.get("quantized_write_maximum_envelope_ratio")
            )
        ),
        "quantized_write_budget_envelope_ready": envelope_ready,
    }
    return build_stable_digest(payload)


def _spatial_budget(direction: Any, effective_budget: Any) -> Any:
    """把每样本 HxW 预算只沿 channel 轴扩展到 NCHW。"""

    torch = _torch()
    if direction.ndim != 4:
        raise ValueError("风险有界更新要求 direction 具有 [batch, channel, height, width] 形状")
    batch, channel, height, width = (int(value) for value in direction.shape)
    budget = torch.as_tensor(
        effective_budget,
        device=direction.device,
        dtype=torch.float32,
    )
    if tuple(budget.shape) == tuple(direction.shape):
        if (
            not torch.is_tensor(effective_budget)
            or effective_budget.device != direction.device
            or effective_budget.dtype != torch.float32
        ):
            raise ValueError("同形 NCHW effective_budget 必须是同设备 float32 Tensor")
        if not budget.is_contiguous():
            raise ValueError("同形 NCHW effective_budget 必须连续存储")
        if not torch.equal(budget, budget[:, :1].expand_as(budget)):
            raise ValueError("NCHW effective_budget 的 channel 副本必须逐值相同")
        return budget
    if tuple(budget.shape) == (batch, height, width):
        spatial = budget.unsqueeze(1)
    elif tuple(budget.shape) == (batch, 1, height, width):
        spatial = budget
    elif batch == 1 and tuple(budget.shape) == (height, width):
        spatial = budget.reshape(1, 1, height, width)
    elif budget.ndim == 1 and budget.numel() == batch * height * width:
        spatial = budget.reshape(batch, 1, height, width)
    else:
        raise ValueError("effective_budget 必须按每样本 HxW 定义且不得包含独立 channel 预算")
    return spatial.expand(batch, channel, height, width).clone()


def _per_sample_strength(
    value: float | Sequence[float] | Any,
    *,
    batch_size: int,
    device: Any,
    field_name: str,
    allow_zero: bool,
) -> Any:
    """把标量或逐样本强度解析为长度等于 batch 的 float32 Tensor。"""

    torch = _torch()
    resolved = torch.as_tensor(value, device=device, dtype=torch.float32).reshape(-1)
    if resolved.numel() == 1:
        resolved = resolved.expand(batch_size).clone()
    if resolved.numel() != batch_size:
        raise ValueError(f"{field_name} 必须为标量或每样本一个值")
    lower_bound_invalid = resolved < 0.0 if allow_zero else resolved <= 0.0
    if not bool(torch.isfinite(resolved).all()) or bool(lower_bound_invalid.any()):
        qualifier = "非负" if allow_zero else "正"
        raise ValueError(f"{field_name} 必须为{qualifier}有限数")
    return resolved


def _envelope_measurement(
    update: Any,
    envelope: Any,
    *,
    absolute_tolerance: float,
) -> tuple[Any, bool]:
    """逐样本测量实际更新相对硬包络的最大比例。"""

    torch = _torch()
    if update.shape != envelope.shape or update.ndim < 2:
        raise ValueError("update 与 envelope 必须具有相同形状并保留 batch 轴")
    update_abs = update.detach().float().abs()
    resolved_envelope = envelope.detach().float()
    finite = bool(torch.isfinite(update_abs).all()) and bool(
        torch.isfinite(resolved_envelope).all()
    )
    if not finite or bool((resolved_envelope < 0.0).any()):
        ratios = torch.full(
            (update.shape[0],),
            float("inf"),
            device=update.device,
            dtype=torch.float32,
        )
        return ratios, False
    positive = resolved_envelope > 0.0
    ratio_values = torch.zeros_like(update_abs)
    ratio_values[positive] = update_abs[positive] / resolved_envelope[positive]
    zero_leak = (~positive) & (update_abs > absolute_tolerance)
    ratios = ratio_values.reshape(update.shape[0], -1).amax(dim=1)
    ratios = torch.where(
        zero_leak.reshape(update.shape[0], -1).any(dim=1),
        torch.full_like(ratios, float("inf")),
        ratios,
    )
    ready = bool(
        finite
        and not bool(zero_leak.any())
        and bool((update_abs <= resolved_envelope + absolute_tolerance).all())
    )
    return ratios, ready


def _materialize_float32_bounded_update(
    unit_direction: Any,
    applied_strength: Any,
    envelope: Any,
) -> tuple[Any, Any, Any]:
    """选择不因 float32 乘法舍入越过零容差包络的最大可表示步长。"""

    torch = _torch()
    resolved_strength = applied_strength
    batch_size = int(unit_direction.shape[0])
    strength_shape = (batch_size,) + (1,) * (unit_direction.ndim - 1)
    for _ in range(4):
        update = unit_direction * resolved_strength.reshape(strength_shape)
        violation = (update.abs() > envelope).reshape(batch_size, -1).any(dim=1)
        if not bool(violation.any()):
            ratios, ready = _envelope_measurement(
                update,
                envelope,
                absolute_tolerance=0.0,
            )
            if not ready:
                raise RuntimeError("分支实际更新没有形成有限风险包络测量")
            return resolved_strength, update, ratios
        reduced = torch.nextafter(
            resolved_strength,
            torch.zeros_like(resolved_strength),
        )
        resolved_strength = torch.where(violation, reduced, resolved_strength)
    raise RuntimeError("float32 分支步长无法在零容差下满足风险硬包络")


@dataclass(frozen=True)
class RiskBoundedUpdate:
    """保存一个分支的单位方向、硬包络和实际 float32 更新。"""

    branch_name: str
    unit_direction: Any
    effective_budget: Any
    amplitude_envelope: Any
    update: Any
    nominal_strength: Any
    applied_strength: Any
    risk_scale_factor: Any
    maximum_envelope_ratio: Any
    budget_ceiling: float
    direction_epsilon: float
    numerical_epsilon: float

    def to_record(self) -> dict[str, Any]:
        """从实际 Tensor 物化分支风险写回证据字段。"""

        return {
            "branch_name": self.branch_name,
            "tensor_content_digest_version": TENSOR_CONTENT_DIGEST_VERSION,
            "effective_budget_values_content_sha256": tensor_content_sha256(
                self.effective_budget
            ),
            "branch_unit_direction_content_sha256": tensor_content_sha256(
                self.unit_direction
            ),
            "branch_budget_envelope_content_sha256": tensor_content_sha256(
                self.amplitude_envelope
            ),
            "branch_written_update_content_sha256": tensor_content_sha256(
                self.update
            ),
            "branch_nominal_strength": _batch_values(self.nominal_strength),
            "branch_applied_strength": _batch_values(self.applied_strength),
            "branch_risk_scale_factor": _batch_values(self.risk_scale_factor),
            "branch_budget_ceiling": self.budget_ceiling,
            "branch_direction_epsilon": self.direction_epsilon,
            "branch_numerical_epsilon": self.numerical_epsilon,
            "branch_written_update_maximum_envelope_ratio": _batch_values(
                self.maximum_envelope_ratio
            ),
        }


def build_risk_bounded_update(
    *,
    branch_name: str,
    direction: Any,
    effective_budget: Any,
    nominal_strength: float | Sequence[float] | Any,
    budget_ceiling: float,
    direction_epsilon: float = 1e-12,
    numerical_epsilon: float = 1e-12,
) -> RiskBoundedUpdate:
    """沿固定安全方向构造满足逐位置风险硬包络的分支更新。"""

    torch = _torch()
    if branch_name not in QUANTIZED_COMPOSITION_ORDER:
        raise ValueError("branch_name 不是冻结的三分支角色")
    if not math.isfinite(budget_ceiling) or budget_ceiling <= 0.0:
        raise ValueError("budget_ceiling 必须为正有限数")
    if not math.isfinite(direction_epsilon) or direction_epsilon <= 0.0:
        raise ValueError("direction_epsilon 必须为正有限数")
    if not math.isfinite(numerical_epsilon) or numerical_epsilon <= 0.0:
        raise ValueError("numerical_epsilon 必须为正有限数")
    values = direction.detach().float()
    if values.ndim != 4 or not bool(torch.isfinite(values).all()):
        raise ValueError("direction 必须为有限 NCHW Tensor")
    budget = _spatial_budget(values, effective_budget)
    if (
        not bool(torch.isfinite(budget).all())
        or bool((budget < 0.0).any())
        or bool((budget > budget_ceiling).any())
    ):
        raise ValueError("effective_budget 必须位于 [0, budget_ceiling]")
    batch_size = int(values.shape[0])
    original_direction_norm = torch.linalg.vector_norm(values.flatten(1), dim=1)
    if bool((original_direction_norm <= numerical_epsilon).any()):
        raise RuntimeError("每个样本的原始安全方向都必须具有非零能量")
    normalized_direction = values / original_direction_norm.reshape(
        batch_size,
        1,
        1,
        1,
    )
    zero_budget = budget == 0.0
    if bool(
        (
            zero_budget
            & (normalized_direction.abs() > direction_epsilon)
        ).any()
    ):
        raise RuntimeError("零预算位置存在方向泄漏")
    cleaned_direction = torch.where(
        zero_budget,
        torch.zeros_like(normalized_direction),
        normalized_direction,
    )
    cleaned_direction_norm = torch.linalg.vector_norm(
        cleaned_direction.flatten(1),
        dim=1,
    )
    if bool((cleaned_direction_norm <= numerical_epsilon).any()):
        raise RuntimeError("零预算支持清理后没有可单位化的安全方向")
    unit_direction = cleaned_direction / cleaned_direction_norm.reshape(
        batch_size,
        1,
        1,
        1,
    )
    active_direction = unit_direction.abs() > direction_epsilon
    if bool((~active_direction.flatten(1).any(dim=1)).any()):
        raise RuntimeError("每个样本都必须具有超过 direction_epsilon 的活动方向坐标")
    nominal = _per_sample_strength(
        nominal_strength,
        batch_size=batch_size,
        device=values.device,
        field_name="nominal_strength",
        allow_zero=False,
    )
    direction_peak = unit_direction.abs().flatten(1).amax(dim=1)
    envelope = (
        nominal.reshape(batch_size, 1, 1, 1)
        * direction_peak.reshape(batch_size, 1, 1, 1)
        * (budget / budget_ceiling)
    )
    safe_direction_abs = torch.where(
        active_direction,
        unit_direction.abs(),
        torch.ones_like(unit_direction),
    )
    coordinate_bounds = torch.where(
        active_direction,
        envelope / safe_direction_abs,
        torch.full_like(envelope, float("inf")),
    )
    feasible_strength = coordinate_bounds.flatten(1).amin(dim=1)
    applied_bound = torch.minimum(nominal, feasible_strength)
    if not bool(torch.isfinite(applied_bound).all()) or bool((applied_bound < 0.0).any()):
        raise RuntimeError("风险硬包络没有形成有限可行步长")
    if bool((applied_bound <= numerical_epsilon).any()):
        raise RuntimeError("风险硬包络最终步长不得小于等于 numerical_epsilon")
    applied, update, ratios = _materialize_float32_bounded_update(
        unit_direction,
        applied_bound,
        envelope,
    )
    if bool((applied <= numerical_epsilon).any()):
        raise RuntimeError("float32 物化后的最终步长不得小于等于 numerical_epsilon")
    return RiskBoundedUpdate(
        branch_name=branch_name,
        unit_direction=unit_direction,
        effective_budget=budget,
        amplitude_envelope=envelope,
        update=update,
        nominal_strength=nominal,
        applied_strength=applied,
        risk_scale_factor=applied / nominal,
        maximum_envelope_ratio=ratios,
        budget_ceiling=float(budget_ceiling),
        direction_epsilon=float(direction_epsilon),
        numerical_epsilon=float(numerical_epsilon),
    )


def rescale_risk_bounded_update(
    result: RiskBoundedUpdate,
    applied_strength: float | Sequence[float] | Any,
) -> RiskBoundedUpdate:
    """只沿原单位方向缩小分支步长并重新物化包络测量。"""

    batch_size = int(result.update.shape[0])
    resolved = _per_sample_strength(
        applied_strength,
        batch_size=batch_size,
        device=result.update.device,
        field_name="applied_strength",
        allow_zero=False,
    )
    if bool((resolved > result.applied_strength).any()):
        raise ValueError("rescale_risk_bounded_update 不允许放大原分支步长")
    if bool((resolved <= result.numerical_epsilon).any()):
        raise ValueError("缩放后的分支步长不得小于等于 numerical_epsilon")
    resolved, update, ratios = _materialize_float32_bounded_update(
        result.unit_direction,
        resolved,
        result.amplitude_envelope,
    )
    if bool((resolved <= result.numerical_epsilon).any()):
        raise RuntimeError("float32 物化后的分支步长不得小于等于 numerical_epsilon")
    return RiskBoundedUpdate(
        branch_name=result.branch_name,
        unit_direction=result.unit_direction,
        effective_budget=result.effective_budget,
        amplitude_envelope=result.amplitude_envelope,
        update=update,
        nominal_strength=result.nominal_strength,
        applied_strength=resolved,
        risk_scale_factor=resolved / result.nominal_strength,
        maximum_envelope_ratio=ratios,
        budget_ceiling=result.budget_ceiling,
        direction_epsilon=result.direction_epsilon,
        numerical_epsilon=result.numerical_epsilon,
    )


@dataclass(frozen=True)
class QuantizedCompositionCandidate:
    """保存固定顺序合成并单次 cast 后的实际写回候选。"""

    composition_order: tuple[str, ...]
    active_branch_order: tuple[str, ...]
    branch_content_identities: tuple[tuple[str, str, str], ...]
    common_scale: float
    backtracking_factor: float
    backtracking_step_count: int
    original_latent_content_sha256: str
    float32_combined_update: Any
    combined_envelope: Any
    candidate_latent: Any
    written_update: Any
    maximum_envelope_ratio: Any
    envelope_ready: bool

    def to_record(self) -> dict[str, Any]:
        """从合成 Tensor 和实际 dtype 增量物化正式证据字段。"""

        record = {
            "quantized_composition_evidence_version": (
                QUANTIZED_COMPOSITION_EVIDENCE_VERSION
            ),
            "tensor_content_digest_version": TENSOR_CONTENT_DIGEST_VERSION,
            "quantized_write_original_latent_content_sha256": (
                self.original_latent_content_sha256
            ),
            "quantized_write_candidate_latent_content_sha256": (
                tensor_content_sha256(self.candidate_latent)
            ),
            "combined_update_content_sha256": tensor_content_sha256(
                self.float32_combined_update
            ),
            "combined_budget_envelope_content_sha256": tensor_content_sha256(
                self.combined_envelope
            ),
            "quantized_write_update_content_sha256": tensor_content_sha256(
                self.written_update
            ),
            "quantized_write_update_dtype": str(self.written_update.dtype),
            "quantized_write_update_shape": [
                int(value) for value in self.written_update.shape
            ],
            "quantized_write_composition_order": list(self.composition_order),
            "quantized_write_active_branch_order": list(
                self.active_branch_order
            ),
            "quantized_write_branch_content_identities": {
                role: {
                    "branch_written_update_content_sha256": update_digest,
                    "branch_budget_envelope_content_sha256": envelope_digest,
                }
                for role, update_digest, envelope_digest in (
                    self.branch_content_identities
                )
            },
            "quantized_write_common_scale": self.common_scale,
            "quantized_write_backtracking_factor": self.backtracking_factor,
            "quantized_write_backtracking_step_count": (
                self.backtracking_step_count
            ),
            "quantized_write_maximum_envelope_ratio": _batch_values(
                self.maximum_envelope_ratio
            ),
            "quantized_write_budget_envelope_ready": self.envelope_ready,
        }
        record["quantized_composition_evidence_digest"] = (
            recompute_quantized_composition_evidence_digest(record)
        )
        return record


def _validated_branch_updates(
    original_latent: Any,
    branch_updates: Mapping[str, RiskBoundedUpdate],
) -> dict[str, RiskBoundedUpdate]:
    """验证非空分支子集的角色、设备和 Tensor 形状。"""

    if not isinstance(branch_updates, Mapping) or not branch_updates:
        raise ValueError("branch_updates 必须是冻结三分支的非空子集")
    unknown = set(branch_updates) - set(QUANTIZED_COMPOSITION_ORDER)
    if unknown:
        raise ValueError("branch_updates 包含未知分支角色")
    resolved = dict(branch_updates)
    for role, result in resolved.items():
        if not isinstance(result, RiskBoundedUpdate) or result.branch_name != role:
            raise ValueError("branch_updates 的映射角色与 RiskBoundedUpdate 身份不一致")
        for tensor in (result.update, result.amplitude_envelope):
            if tensor.shape != original_latent.shape or tensor.device != original_latent.device:
                raise ValueError("分支更新必须与 original_latent 具有相同形状和设备")
            if tensor.dtype != _torch().float32:
                raise ValueError("分支更新和硬包络必须保持 float32")
    return resolved


def compose_ordered_float32_update_once(
    *,
    original_latent: Any,
    branch_update_tensors: Mapping[str, Any],
    common_scale: float,
) -> tuple[Any, Any, Any]:
    """按冻结顺序完成 float32 合成, 再对候选 latent 只执行一次 cast。"""

    torch = _torch()
    if original_latent.numel() == 0 or not bool(
        torch.is_floating_point(original_latent)
    ):
        raise ValueError("original_latent 必须为非空浮点 Tensor")
    if not isinstance(branch_update_tensors, Mapping) or not branch_update_tensors:
        raise ValueError("branch_update_tensors 必须是冻结三分支的非空子集")
    unknown = set(branch_update_tensors) - set(QUANTIZED_COMPOSITION_ORDER)
    if unknown:
        raise ValueError("branch_update_tensors 包含未知分支角色")
    if not math.isfinite(common_scale) or not 0.0 < common_scale <= 1.0:
        raise ValueError("common_scale 必须位于 (0, 1]")

    combined_update = torch.zeros_like(original_latent, dtype=torch.float32)
    for role in QUANTIZED_COMPOSITION_ORDER:
        if role not in branch_update_tensors:
            continue
        update = branch_update_tensors[role]
        if (
            not torch.is_tensor(update)
            or update.shape != original_latent.shape
            or update.device != original_latent.device
            or update.dtype != torch.float32
        ):
            raise ValueError(
                "每个分支 update 必须是与 original_latent 同形同设备的 float32 Tensor"
            )
        combined_update = combined_update + update
    combined_update = combined_update * float(common_scale)
    candidate_latent = (
        original_latent.detach().float() + combined_update
    ).to(dtype=original_latent.dtype)
    written_update = candidate_latent - original_latent.detach()
    return combined_update, candidate_latent, written_update


def build_quantized_composition_candidate(
    *,
    original_latent: Any,
    branch_updates: Mapping[str, RiskBoundedUpdate],
    common_scale: float = 1.0,
    backtracking_factor: float = 0.5,
    backtracking_step_count: int | None = None,
    absolute_tolerance: float = 0.0,
) -> QuantizedCompositionCandidate:
    """按冻结顺序合成分支并对 original latent 执行一次实际 dtype cast。"""

    torch = _torch()
    if original_latent.ndim != 4 or not bool(torch.is_floating_point(original_latent)):
        raise ValueError("original_latent 必须为浮点 NCHW Tensor")
    if not math.isfinite(common_scale) or not 0.0 < common_scale <= 1.0:
        raise ValueError("common_scale 必须位于 (0, 1]")
    if not math.isfinite(backtracking_factor) or not 0.0 < backtracking_factor < 1.0:
        raise ValueError("backtracking_factor 必须位于 (0, 1)")
    resolved_step = (
        _infer_backtracking_step(
            common_scale=float(common_scale),
            backtracking_factor=float(backtracking_factor),
        )
        if backtracking_step_count is None
        else backtracking_step_count
    )
    common_scale, backtracking_factor, resolved_step = _backtracking_trace(
        common_scale=common_scale,
        backtracking_factor=backtracking_factor,
        backtracking_step_count=resolved_step,
    )
    if not math.isfinite(absolute_tolerance) or absolute_tolerance < 0.0:
        raise ValueError("absolute_tolerance 必须为非负有限数")
    resolved = _validated_branch_updates(original_latent, branch_updates)
    active_branch_order = tuple(
        role for role in QUANTIZED_COMPOSITION_ORDER if role in resolved
    )
    branch_content_identities = tuple(
        (
            role,
            tensor_content_sha256(resolved[role].update),
            tensor_content_sha256(resolved[role].amplitude_envelope),
        )
        for role in active_branch_order
    )
    original_latent_content_sha256 = tensor_content_sha256(original_latent)
    combined_update, candidate_latent, written_update = (
        compose_ordered_float32_update_once(
            original_latent=original_latent,
            branch_update_tensors={
                role: resolved[role].update for role in active_branch_order
            },
            common_scale=common_scale,
        )
    )
    combined_envelope = torch.zeros_like(original_latent, dtype=torch.float32)
    for role in QUANTIZED_COMPOSITION_ORDER:
        branch = resolved.get(role)
        if branch is None:
            continue
        combined_envelope = combined_envelope + branch.amplitude_envelope
    combined_envelope = combined_envelope * common_scale
    ratios, ready = _envelope_measurement(
        written_update,
        combined_envelope,
        absolute_tolerance=absolute_tolerance,
    )
    return QuantizedCompositionCandidate(
        composition_order=QUANTIZED_COMPOSITION_ORDER,
        active_branch_order=active_branch_order,
        branch_content_identities=branch_content_identities,
        common_scale=float(common_scale),
        backtracking_factor=float(backtracking_factor),
        backtracking_step_count=resolved_step,
        original_latent_content_sha256=original_latent_content_sha256,
        float32_combined_update=combined_update,
        combined_envelope=combined_envelope,
        candidate_latent=candidate_latent,
        written_update=written_update,
        maximum_envelope_ratio=ratios,
        envelope_ready=ready,
    )


def iter_quantized_composition_candidates(
    *,
    original_latent: Any,
    branch_updates: Mapping[str, RiskBoundedUpdate],
    backtracking_factor: float = 0.5,
    maximum_steps: int = 24,
    absolute_tolerance: float = 0.0,
) -> Iterator[QuantizedCompositionCandidate]:
    """按共同缩放序列产出候选, 由调用方执行全部科学门禁后决定接受。"""

    if not math.isfinite(backtracking_factor) or not 0.0 < backtracking_factor < 1.0:
        raise ValueError("backtracking_factor 必须位于 (0, 1)")
    if maximum_steps < 0:
        raise ValueError("maximum_steps 必须为非负整数")
    for step in range(maximum_steps + 1):
        yield build_quantized_composition_candidate(
            original_latent=original_latent,
            branch_updates=branch_updates,
            common_scale=backtracking_factor**step,
            backtracking_factor=backtracking_factor,
            backtracking_step_count=step,
            absolute_tolerance=absolute_tolerance,
        )


__all__ = [
    "DualChainWriteBudget",
    "DualChainWriteResult",
    "QUANTIZED_COMPOSITION_EVIDENCE_VERSION",
    "QUANTIZED_COMPOSITION_ORDER",
    "QuantizedCompositionCandidate",
    "RiskBoundedUpdate",
    "build_quantized_composition_candidate",
    "build_risk_bounded_update",
    "compose_dual_chain_update_once",
    "compose_ordered_float32_update_once",
    "iter_quantized_composition_candidates",
    "formal_dual_chain_write_budget",
    "recompute_quantized_composition_evidence_digest",
    "rescale_risk_bounded_update",
]
