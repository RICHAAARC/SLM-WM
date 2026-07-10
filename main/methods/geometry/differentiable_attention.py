"""从真实 Transformer 注意力模块构造可微几何载体。

本模块不使用 hidden-state 相似度代理。它直接调用注意力模块的 `to_q` 和
`to_k` 投影, 得到真实 Q/K 自注意力矩阵, 并通过 autograd 计算几何签名分数
对 latent 的梯度。调用方负责提供一次真实 Transformer 前向函数。
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any, Callable, Iterable

from main.core.digest import build_stable_digest
from main.methods.subspace.jacobian_nullspace import JacobianNullSpaceResult


def _torch() -> Any:
    """延迟导入 PyTorch。"""

    import torch

    return torch


def _seed(key_material: str, layer_name: str, token_count: int) -> int:
    """为每个注意力层生成稳定密钥种子。"""

    payload = f"{key_material}|{layer_name}|{token_count}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63 - 1)


def _first_tensor(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any | None:
    """从注意力模块输入中读取图像 token hidden states。"""

    candidate = kwargs.get("hidden_states")
    if candidate is not None and hasattr(candidate, "shape"):
        return candidate
    for value in args:
        if hasattr(value, "shape") and getattr(value, "ndim", 0) >= 3:
            return value
    return None


def qk_self_attention(module: Any, hidden_states: Any, max_tokens: int = 256) -> tuple[Any, tuple[int, ...]]:
    """在真实二维图像 token 网格上抽样并计算 Q/K 自注意力矩阵。

    抽样沿原始网格的两个空间轴分别等距进行, 不把一维序号等距抽样误解释为
    二维网格。返回的 `token_indices` 始终指向原始图像 token 网格, 供检测端恢复
    真实空间坐标。该结构可以复用于任何公开 `to_q`、`to_k` 和 `heads` 的
    Transformer 注意力模块。
    """

    torch = _torch()
    if not hasattr(module, "to_q") or not hasattr(module, "to_k"):
        raise TypeError("注意力模块必须公开 to_q 和 to_k 投影")
    if hidden_states.ndim != 3:
        raise ValueError("hidden_states 必须具有 [batch, token, channel] 形状")
    token_count = int(hidden_states.shape[1])
    source_side = int(round(math.sqrt(token_count)))
    if source_side * source_side != token_count:
        raise ValueError("真实注意力几何要求图像 token 构成方形二维网格")
    if max_tokens < 4:
        raise ValueError("max_tokens 至少为 4, 以保留二维几何结构")
    sampled_side = min(source_side, int(math.sqrt(max_tokens)))
    axis_indices = tuple(
        round(index * (source_side - 1) / (sampled_side - 1))
        for index in range(sampled_side)
    )
    token_indices = tuple(
        row * source_side + column
        for row in axis_indices
        for column in axis_indices
    )
    bounded_count = len(token_indices)
    index_tensor = torch.tensor(token_indices, device=hidden_states.device)
    query = module.to_q(hidden_states)
    key = module.to_k(hidden_states)
    heads = int(getattr(module, "heads", 1))
    if query.shape[-1] % heads != 0:
        raise ValueError("Q 投影宽度必须能被注意力头数整除")
    head_width = int(query.shape[-1] // heads)
    query = query.reshape(query.shape[0], token_count, heads, head_width).transpose(1, 2)
    key = key.reshape(key.shape[0], token_count, heads, head_width).transpose(1, 2)
    norm_q = getattr(module, "norm_q", None)
    norm_k = getattr(module, "norm_k", None)
    if norm_q is not None:
        query = norm_q(query)
    if norm_k is not None:
        key = norm_k(key)
    query = query.index_select(2, index_tensor)
    key = key.index_select(2, index_tensor)
    logits = query.float() @ key.float().transpose(-1, -2) / math.sqrt(head_width)
    attention = torch.softmax(logits, dim=-1).mean(dim=1)
    return attention, token_indices


def attention_relation_stability_map(
    records: Iterable[tuple[str, Any, tuple[int, ...]]],
    spatial_size: tuple[int, int],
) -> Any:
    """由多个真实 Q/K 层的一致性构造注意力关系稳定图。

    每个空间位置的稳定度定义为不同层中对应注意力关系行的平均余弦相似度。
    该值来自模型实际 Q/K 关系, 不是语义图、纹理图或隐藏状态相似度的替代量。
    """

    import torch.nn.functional as functional

    torch = _torch()
    resolved_records = tuple(records)
    if len(resolved_records) < 2:
        raise ValueError("注意力关系稳定图至少需要两个真实 Q/K 层")
    reference_indices = resolved_records[0][2]
    if any(token_indices != reference_indices for _, _, token_indices in resolved_records[1:]):
        raise ValueError("用于稳定度计算的 Q/K 层必须共享同一二维 token 抽样网格")
    sampled_side = int(round(math.sqrt(len(reference_indices))))
    if sampled_side * sampled_side != len(reference_indices):
        raise ValueError("注意力稳定度要求抽样 token 构成方形二维网格")
    normalized_rows = []
    for _, attention, _ in resolved_records:
        matrix = attention.float()
        centered = matrix - matrix.mean(dim=-1, keepdim=True)
        normalized_rows.append(functional.normalize(centered, dim=-1, eps=1e-12))
    pair_scores = []
    for left_index in range(len(normalized_rows) - 1):
        for right_index in range(left_index + 1, len(normalized_rows)):
            pair_scores.append(
                (normalized_rows[left_index] * normalized_rows[right_index]).sum(dim=-1)
            )
    stability = (torch.stack(pair_scores).mean(dim=0) + 1.0) * 0.5
    sampled_map = stability.clamp(0.0, 1.0).reshape(
        stability.shape[0],
        1,
        sampled_side,
        sampled_side,
    )
    return functional.interpolate(
        sampled_map,
        size=spatial_size,
        mode="bilinear",
        align_corners=False,
    )[:, 0]


class DifferentiableAttentionRecorder:
    """在指定真实注意力模块上记录保持计算图的 Q/K attention。"""

    def __init__(self, modules: Iterable[tuple[str, Any]], max_tokens: int = 256) -> None:
        """注册 forward pre-hook, 记录对象需要在使用后显式关闭。"""

        self.max_tokens = max_tokens
        self.records: list[tuple[str, Any, tuple[int, ...]]] = []
        self.handles: list[Any] = []
        for layer_name, module in modules:
            self.handles.append(
                module.register_forward_pre_hook(self._hook(layer_name), with_kwargs=True)
            )

    def _hook(self, layer_name: str) -> Callable[..., None]:
        """构造一个保留计算图的模块前向钩子。"""

        def capture(module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
            hidden_states = _first_tensor(args, kwargs)
            if hidden_states is None:
                return
            attention, token_indices = qk_self_attention(module, hidden_states, self.max_tokens)
            self.records.append((layer_name, attention, token_indices))

        return capture

    def clear(self) -> None:
        """清除上一次前向记录, 不移除钩子。"""

        self.records.clear()

    def close(self) -> None:
        """移除全部钩子。"""

        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def __enter__(self) -> "DifferentiableAttentionRecorder":
        """支持上下文管理器写法。"""

        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        """退出上下文时确保钩子被移除。"""

        self.close()


def keyed_relation_signs(attention: Any, key_material: str, layer_name: str) -> Any:
    """为注意力关系生成零对角、近似均衡的密钥符号矩阵。"""

    torch = _torch()
    token_count = int(attention.shape[-1])
    device_name = attention.device.type if attention.device.type in {"cpu", "cuda"} else "cpu"
    generator = torch.Generator(device=device_name).manual_seed(_seed(key_material, layer_name, token_count))
    random_values = torch.rand(
        token_count,
        token_count,
        generator=generator,
        device=attention.device,
        dtype=torch.float32,
    )
    signs = torch.where(random_values >= 0.5, 1.0, -1.0)
    signs = torch.triu(signs, diagonal=1)
    signs = signs + signs.transpose(0, 1)
    return signs


def attention_geometry_score(
    records: Iterable[tuple[str, Any, tuple[int, ...]]],
    key_material: str,
) -> Any:
    """计算真实 Q/K attention 与密钥关系签名的一致性分数。"""

    torch = _torch()
    layer_scores = []
    for layer_name, attention, _ in records:
        relation_signs = keyed_relation_signs(attention, key_material, layer_name)
        token_count = int(attention.shape[-1])
        off_diagonal = 1.0 - torch.eye(token_count, device=attention.device, dtype=attention.dtype)
        centered = attention - attention.mean(dim=-1, keepdim=True)
        numerator = (centered * relation_signs * off_diagonal).sum(dim=(-1, -2))
        denominator = (
            (centered * off_diagonal).square().sum(dim=(-1, -2)).sqrt()
            * (relation_signs * off_diagonal).square().sum().sqrt()
        ).clamp_min(1e-12)
        layer_scores.append((numerator / denominator).mean())
    if not layer_scores:
        raise RuntimeError("真实 Transformer 前向没有产生 Q/K attention 记录")
    return torch.stack(layer_scores).mean()


@dataclass
class AttentionGeometryGradient:
    """保存真实 Q/K 目标在当前 latent 点的梯度。"""

    gradient: Any
    score_before: float
    gradient_norm: float
    layer_names: tuple[str, ...]


def compute_attention_geometry_gradient(
    latent: Any,
    transformer_forward: Callable[[Any], Any],
    recorder: DifferentiableAttentionRecorder,
    key_material: str,
) -> AttentionGeometryGradient:
    """对真实 Transformer Q/K 几何分数计算 latent 梯度。"""

    torch = _torch()
    with torch.enable_grad():
        differentiable_latent = latent.detach().float().requires_grad_(True)
        recorder.clear()
        transformer_forward(differentiable_latent)
        score_before_tensor = attention_geometry_score(recorder.records, key_material)
        layer_names = tuple(dict.fromkeys(layer_name for layer_name, _, _ in recorder.records))
        gradient = torch.autograd.grad(score_before_tensor, differentiable_latent, retain_graph=False)[0]
    return AttentionGeometryGradient(
        gradient=gradient.detach(),
        score_before=float(score_before_tensor.detach().item()),
        gradient_norm=float(gradient.detach().float().norm().item()),
        layer_names=layer_names,
    )


@dataclass
class AttentionGeometryUpdate:
    """保存真实注意力几何目标产生的 latent 更新。"""

    update: Any
    score_before: float
    score_after: float
    score_gain: float
    gradient_norm: float
    projected_gradient_norm: float
    applied_update_strength: float
    backtracking_step_count: int
    layer_names: tuple[str, ...]
    update_digest: str
    metadata: dict[str, Any]


def optimize_attention_geometry_update(
    latent: Any,
    transformer_forward: Callable[[Any], Any],
    recorder: DifferentiableAttentionRecorder,
    key_material: str,
    safe_subspace: JacobianNullSpaceResult,
    update_strength: float,
    precomputed_gradient: AttentionGeometryGradient | None = None,
) -> AttentionGeometryUpdate:
    """通过真实 Transformer 前向和 autograd 生成注意力几何更新。"""

    torch = _torch()
    if update_strength <= 0.0:
        raise ValueError("update_strength 必须为正数")
    gradient_evidence = precomputed_gradient or compute_attention_geometry_gradient(
        latent,
        transformer_forward,
        recorder,
        key_material,
    )
    with torch.enable_grad():
        differentiable_latent = latent.detach().float()
        score_before = gradient_evidence.score_before
        gradient = gradient_evidence.gradient.to(device=latent.device, dtype=torch.float32)
        projected = safe_subspace.project(gradient)
        projected_norm = projected.float().norm()
        if float(projected_norm.item()) <= 1e-12:
            raise RuntimeError("注意力梯度在安全子空间中的投影为零")
        unit_update = projected / projected_norm
        applied_strength = float(update_strength)
        score_after_tensor = torch.tensor(score_before, device=latent.device, dtype=torch.float32)
        accepted = False
        backtracking_step_count = 0
        for backtracking_step_count in range(9):
            update = unit_update * applied_strength
            candidate = differentiable_latent.detach() + update.detach().float()
            recorder.clear()
            transformer_forward(candidate)
            score_after_tensor = attention_geometry_score(recorder.records, key_material)
            if bool(torch.isfinite(score_after_tensor)) and float(score_after_tensor.detach().item()) > score_before:
                accepted = True
                break
            applied_strength *= 0.5
        if not accepted:
            raise RuntimeError("注意力几何更新在回溯搜索后仍未提高真实 Q/K 目标")
    score_after = float(score_after_tensor.detach().item())
    layer_names = gradient_evidence.layer_names
    payload = {
        "score_before": round(score_before, 12),
        "score_after": round(score_after, 12),
        "gradient_norm": round(float(gradient.norm().item()), 12),
        "projected_gradient_norm": round(float(projected.norm().item()), 12),
        "applied_update_strength": round(applied_strength, 12),
        "backtracking_step_count": backtracking_step_count,
        "layer_names": layer_names,
        "safe_subspace_digest": safe_subspace.solver_digest,
    }
    return AttentionGeometryUpdate(
        update=update.to(dtype=latent.dtype),
        score_before=score_before,
        score_after=score_after,
        score_gain=score_after - score_before,
        gradient_norm=gradient_evidence.gradient_norm,
        projected_gradient_norm=float(projected.norm().item()),
        applied_update_strength=applied_strength,
        backtracking_step_count=backtracking_step_count,
        layer_names=layer_names,
        update_digest=build_stable_digest(payload),
        metadata={
            "attention_source": "real_qk_projection",
            "gradient_source": "torch_autograd",
            "safe_projection": "jacobian_null_space",
            "update_search": "monotonic_backtracking",
        },
    )
