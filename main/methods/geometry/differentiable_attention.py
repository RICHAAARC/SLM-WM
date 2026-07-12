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


@dataclass(frozen=True)
class StableAttentionTokenSelection:
    """保存跨层 Q/K 稳定性与 attention 中心性共同选出的 token。"""

    token_positions: tuple[int, ...]
    token_indices: tuple[int, ...]
    stable_token_fraction: float
    selection_digest: str


def select_stable_attention_tokens(
    records: Iterable[tuple[str, Any, tuple[int, ...]]],
    stable_token_fraction: float = 0.5,
) -> StableAttentionTokenSelection:
    """从真实多层 Q/K 关系选择固定比例的稳定且显著 token。

    稳定度使用不同冻结层对应关系行的平均余弦一致性, 显著度使用 token 在
    全部关系图中接收的平均 attention 质量。两者乘积决定排序, 并以原始 token
    索引作为确定性并列规则。选择结果在一次注入的梯度、回溯和最终写回复验
    中保持冻结, 避免优化过程通过改变选择集合抬高自身目标。
    """

    import torch.nn.functional as functional

    torch = _torch()
    resolved_records = tuple(records)
    if len(resolved_records) < 2:
        raise ValueError("稳定 token 选择至少需要两个真实 Q/K 层")
    reference_indices = resolved_records[0][2]
    token_count = len(reference_indices)
    if token_count < 4 or any(
        token_indices != reference_indices
        for _, _, token_indices in resolved_records[1:]
    ):
        raise ValueError("稳定 token 选择要求共享且不少于4点的二维抽样网格")
    if not 0.0 < stable_token_fraction <= 1.0:
        raise ValueError("stable_token_fraction 必须位于 (0, 1]")

    normalized_rows = []
    centrality_rows = []
    for _, attention, _ in resolved_records:
        matrix = attention.float()
        centered = matrix - matrix.mean(dim=-1, keepdim=True)
        normalized_rows.append(functional.normalize(centered, dim=-1, eps=1e-12))
        centrality_rows.append(matrix.mean(dim=-2).mean(dim=0))
    pair_scores = [
        (normalized_rows[left] * normalized_rows[right]).sum(dim=-1).mean(dim=0)
        for left in range(len(normalized_rows) - 1)
        for right in range(left + 1, len(normalized_rows))
    ]
    stability = ((torch.stack(pair_scores).mean(dim=0) + 1.0) * 0.5).clamp(
        0.0,
        1.0,
    )
    centrality = torch.stack(centrality_rows).mean(dim=0)
    centrality = centrality / centrality.sum().clamp_min(1e-12)
    selection_scores = (stability * centrality).detach().cpu().tolist()
    selected_count = min(
        token_count,
        max(4, int(math.ceil(token_count * stable_token_fraction))),
    )
    ordered_positions = sorted(
        range(token_count),
        key=lambda position: (
            -float(selection_scores[position]),
            int(reference_indices[position]),
        ),
    )
    token_positions = tuple(sorted(ordered_positions[:selected_count]))
    token_indices = tuple(reference_indices[position] for position in token_positions)
    payload = {
        "selection_rule": "cross_layer_relation_stability_times_incoming_attention",
        "stable_token_fraction": float(stable_token_fraction),
        "token_positions": token_positions,
        "token_indices": token_indices,
        "selection_scores": [round(float(value), 12) for value in selection_scores],
    }
    return StableAttentionTokenSelection(
        token_positions=token_positions,
        token_indices=token_indices,
        stable_token_fraction=float(stable_token_fraction),
        selection_digest=build_stable_digest(payload),
    )


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
    *,
    stable_token_positions: tuple[int, ...] | None = None,
    stable_token_fraction: float = 0.5,
    unstable_pair_weight: float = 0.25,
) -> Any:
    """计算稳定 token 加权的真实 Q/K 密钥关系一致性分数。

    稳定 token 对使用权重1, 其余规则网格关系保留较小的冻结权重。保留完整
    网格可为盲检仿射注册提供连续二维采样支撑, 同时保证稳定 token 集合真实
    改变嵌入目标而不是只作为日志字段。
    """

    torch = _torch()
    resolved_records = tuple(records)
    if not 0.0 <= unstable_pair_weight < 1.0:
        raise ValueError("unstable_pair_weight 必须位于 [0, 1)")
    if stable_token_positions is None:
        selection = select_stable_attention_tokens(
            resolved_records,
            stable_token_fraction=stable_token_fraction,
        )
        stable_token_positions = selection.token_positions
    if not stable_token_positions:
        raise ValueError("stable_token_positions 不得为空")
    layer_scores = []
    for layer_name, attention, _ in resolved_records:
        relation_signs = keyed_relation_signs(attention, key_material, layer_name)
        token_count = int(attention.shape[-1])
        if any(
            position < 0 or position >= token_count
            for position in stable_token_positions
        ):
            raise ValueError("stable_token_positions 超出 attention 宽度")
        off_diagonal = 1.0 - torch.eye(token_count, device=attention.device, dtype=attention.dtype)
        token_weights = torch.full(
            (token_count,),
            float(unstable_pair_weight),
            device=attention.device,
            dtype=attention.dtype,
        )
        token_weights[list(stable_token_positions)] = 1.0
        pair_weights = token_weights[:, None] * token_weights[None, :] * off_diagonal
        row_weight = pair_weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        centered = attention - (
            attention * pair_weights
        ).sum(dim=-1, keepdim=True) / row_weight
        square_root_weights = pair_weights.sqrt()
        weighted_centered = centered * square_root_weights
        weighted_reference = relation_signs * square_root_weights
        numerator = (weighted_centered * weighted_reference).sum(dim=(-1, -2))
        denominator = (
            weighted_centered.square().sum(dim=(-1, -2)).sqrt()
            * weighted_reference.square().sum().sqrt()
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
    stable_token_positions: tuple[int, ...]
    stable_token_indices: tuple[int, ...]
    stable_token_selection_digest: str
    stable_token_fraction: float
    unstable_pair_weight: float


def compute_attention_geometry_gradient(
    latent: Any,
    transformer_forward: Callable[[Any], Any],
    recorder: DifferentiableAttentionRecorder,
    key_material: str,
    stable_token_fraction: float = 0.5,
    unstable_pair_weight: float = 0.25,
    stable_token_selection: StableAttentionTokenSelection | None = None,
) -> AttentionGeometryGradient:
    """对真实 Transformer Q/K 几何分数计算 latent 梯度。"""

    torch = _torch()
    with torch.enable_grad():
        differentiable_latent = latent.detach().float().requires_grad_(True)
        recorder.clear()
        transformer_forward(differentiable_latent)
        selection = stable_token_selection or select_stable_attention_tokens(
            recorder.records,
            stable_token_fraction=stable_token_fraction,
        )
        if stable_token_selection is not None:
            current_indices = recorder.records[0][2]
            if tuple(
                current_indices[position]
                for position in selection.token_positions
            ) != selection.token_indices:
                raise RuntimeError(
                    "冻结稳定 token 选择与当前 Q/K 二维抽样网格不一致"
                )
        score_before_tensor = attention_geometry_score(
            recorder.records,
            key_material,
            stable_token_positions=selection.token_positions,
            stable_token_fraction=stable_token_fraction,
            unstable_pair_weight=unstable_pair_weight,
        )
        layer_names = tuple(dict.fromkeys(layer_name for layer_name, _, _ in recorder.records))
        gradient = torch.autograd.grad(score_before_tensor, differentiable_latent, retain_graph=False)[0]
    return AttentionGeometryGradient(
        gradient=gradient.detach(),
        score_before=float(score_before_tensor.detach().item()),
        gradient_norm=float(gradient.detach().float().norm().item()),
        layer_names=layer_names,
        stable_token_positions=selection.token_positions,
        stable_token_indices=selection.token_indices,
        stable_token_selection_digest=selection.selection_digest,
        stable_token_fraction=selection.stable_token_fraction,
        unstable_pair_weight=float(unstable_pair_weight),
    )


@dataclass
class AttentionGeometryUpdate:
    """保存真实注意力几何目标产生的 latent 更新。"""

    update: Any
    score_before: float
    content_base_score: float
    score_after: float
    score_gain: float
    gradient_norm: float
    projected_gradient_norm: float
    applied_update_strength: float
    backtracking_step_count: int
    layer_names: tuple[str, ...]
    stable_token_indices: tuple[int, ...]
    stable_token_selection_digest: str
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
    base_update: Any | None = None,
    stable_token_fraction: float = 0.5,
    unstable_pair_weight: float = 0.25,
) -> AttentionGeometryUpdate:
    """在固定内容更新基底上生成并验证真实注意力几何更新。

    ``base_update`` 用于传入已经确定的 LF 与尾部载体更新。回溯搜索对
    ``latent + base_update + attention_update`` 这一真正写回的候选 latent 评分,
    避免只验证 attention-only 中间状态后再被内容分支抵消。
    """

    torch = _torch()
    if update_strength <= 0.0:
        raise ValueError("update_strength 必须为正数")
    original_evidence = precomputed_gradient or compute_attention_geometry_gradient(
        latent,
        transformer_forward,
        recorder,
        key_material,
        stable_token_fraction=stable_token_fraction,
        unstable_pair_weight=unstable_pair_weight,
    )
    resolved_base_update = (
        torch.zeros_like(latent)
        if base_update is None
        else torch.as_tensor(base_update, device=latent.device, dtype=latent.dtype)
    )
    if resolved_base_update.shape != latent.shape:
        raise ValueError("base_update 必须与 latent 形状一致")
    content_base_latent = (latent.detach() + resolved_base_update.detach()).to(dtype=latent.dtype)
    content_base_evidence = compute_attention_geometry_gradient(
        content_base_latent,
        transformer_forward,
        recorder,
        key_material,
        stable_token_fraction=original_evidence.stable_token_fraction,
        unstable_pair_weight=original_evidence.unstable_pair_weight,
        stable_token_selection=StableAttentionTokenSelection(
            token_positions=original_evidence.stable_token_positions,
            token_indices=original_evidence.stable_token_indices,
            stable_token_fraction=original_evidence.stable_token_fraction,
            selection_digest=original_evidence.stable_token_selection_digest,
        ),
    )
    with torch.enable_grad():
        score_before = original_evidence.score_before
        content_base_score = content_base_evidence.score_before
        minimum_accepted_score = max(score_before, content_base_score)
        gradient = content_base_evidence.gradient.to(device=latent.device, dtype=torch.float32)
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
            candidate = (
                content_base_latent.detach()
                + update.detach().to(dtype=content_base_latent.dtype)
            ).float()
            recorder.clear()
            transformer_forward(candidate)
            score_after_tensor = attention_geometry_score(
                recorder.records,
                key_material,
                stable_token_positions=(
                    content_base_evidence.stable_token_positions
                ),
                stable_token_fraction=(
                    content_base_evidence.stable_token_fraction
                ),
                unstable_pair_weight=(
                    content_base_evidence.unstable_pair_weight
                ),
            )
            if (
                bool(torch.isfinite(score_after_tensor))
                and float(score_after_tensor.detach().item()) > minimum_accepted_score
            ):
                accepted = True
                break
            applied_strength *= 0.5
        if not accepted:
            raise RuntimeError("注意力几何更新在回溯搜索后仍未提高真实 Q/K 目标")
    score_after = float(score_after_tensor.detach().item())
    layer_names = content_base_evidence.layer_names
    payload = {
        "score_before": round(score_before, 12),
        "content_base_score": round(content_base_score, 12),
        "score_after": round(score_after, 12),
        "gradient_norm": round(float(gradient.norm().item()), 12),
        "projected_gradient_norm": round(float(projected.norm().item()), 12),
        "applied_update_strength": round(applied_strength, 12),
        "backtracking_step_count": backtracking_step_count,
        "layer_names": layer_names,
        "stable_token_indices": content_base_evidence.stable_token_indices,
        "stable_token_selection_digest": (
            content_base_evidence.stable_token_selection_digest
        ),
        "safe_subspace_digest": safe_subspace.solver_digest,
    }
    return AttentionGeometryUpdate(
        update=update.to(dtype=latent.dtype),
        score_before=score_before,
        content_base_score=content_base_score,
        score_after=score_after,
        score_gain=score_after - score_before,
        gradient_norm=content_base_evidence.gradient_norm,
        projected_gradient_norm=float(projected.norm().item()),
        applied_update_strength=applied_strength,
        backtracking_step_count=backtracking_step_count,
        layer_names=layer_names,
        stable_token_indices=content_base_evidence.stable_token_indices,
        stable_token_selection_digest=(
            content_base_evidence.stable_token_selection_digest
        ),
        update_digest=build_stable_digest(payload),
        metadata={
            "attention_source": "real_qk_projection",
            "gradient_source": "torch_autograd",
            "safe_projection": "jacobian_null_space",
            "update_search": "monotonic_backtracking",
            "optimization_base": "fixed_lf_and_tail_update",
            "verified_candidate": "actual_combined_latent",
            "stable_token_fraction": content_base_evidence.stable_token_fraction,
            "unstable_pair_weight": content_base_evidence.unstable_pair_weight,
            "stable_token_selection_rule": (
                "cross_layer_relation_stability_times_incoming_attention"
            ),
        },
    )
