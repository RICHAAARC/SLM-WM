"""从真实 Transformer 注意力模块构造可微几何载体。

本模块不使用 hidden-state 相似度代理。它直接调用注意力模块的 `to_q` 和
`to_k` 投影, 在冻结二维图像 token 抽样集合上构造中心化 logits、可微 rank、
关系概率和公开距离四分量图, 并通过 autograd 计算几何签名分数对 latent 的
梯度。调用方负责提供一次真实 Transformer 前向函数。
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import math
from typing import Any, Callable, Iterable

from main.core.digest import build_stable_digest
from main.methods.subspace.jacobian_nullspace import JacobianNullSpaceResult


ATTENTION_RELATION_COMPONENT_NAMES = (
    "centered_qk_logit",
    "differentiable_row_rank",
    "row_normalized_attention_probability",
    "distance_modulated_centered_attention_probability",
)
ATTENTION_RELATION_SOFT_RANK_TEMPERATURE = 0.25
ATTENTION_RELATION_SOFT_RANK_CHUNK_SIZE = 32
ATTENTION_RELATION_COMPONENT_WEIGHTS = (0.25, 0.25, 0.25, 0.25)
ATTENTION_RELATION_COMPONENT_POLARITIES = (1.0, -1.0, 1.0, 1.0)
DIRECT_QK_RELATION_SOURCE = "direct_qk_centered_logits_and_probabilities"
PROBABILITY_INVERSE_RELATION_SOURCE = "probability_log_inverse"


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


@dataclass(frozen=True)
class QKAttentionRelation:
    """保存同一次真实 Q/K 计算产生的中心化 logits 与 attention 概率。

    ``centered_logits`` 直接来自 ``QK^T / sqrt(d)`` 并移除每行均值。
    ``probabilities`` 来自同一 logits 的 row-wise softmax。两个张量共享计算图,
    因而四分量关系算子可以同时消费线性强度、序关系和非线性概率, 而不把
    数学上相同的概率重复登记成两个分量。
    """

    centered_logits: Any
    probabilities: Any
    relation_source: str = DIRECT_QK_RELATION_SOURCE
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def shape(self) -> Any:
        """返回 attention 概率张量形状。"""

        return self.probabilities.shape

    @property
    def ndim(self) -> int:
        """返回 attention 概率张量维数。"""

        return int(self.probabilities.ndim)

    @property
    def device(self) -> Any:
        """返回两个关系张量共同所在设备。"""

        return self.probabilities.device

    @property
    def dtype(self) -> Any:
        """返回 attention 概率张量精度。"""

        return self.probabilities.dtype

    def detach(self) -> "QKAttentionRelation":
        """同时切断 logits 与概率的计算图, 供盲检推理复用。"""

        return QKAttentionRelation(
            centered_logits=self.centered_logits.detach(),
            probabilities=self.probabilities.detach(),
            relation_source=self.relation_source,
            metadata=dict(self.metadata),
        )

    def clone(self) -> "QKAttentionRelation":
        """同时复制 logits 与概率张量, 保持关系来源身份不变。"""

        return QKAttentionRelation(
            centered_logits=self.centered_logits.clone(),
            probabilities=self.probabilities.clone(),
            relation_source=self.relation_source,
            metadata=dict(self.metadata),
        )


@dataclass(frozen=True)
class AttentionRelationDescriptor:
    """保存 attention-relative graph 的四分量边描述。"""

    values: Any
    token_indices: tuple[int, ...]
    component_names: tuple[str, ...]
    soft_rank_temperature: float
    soft_rank_scale: float
    relative_distance_scale: float
    relation_source: str
    component_identity_digest: str


@dataclass(frozen=True)
class KeyedAttentionRelationProjection:
    """保存密钥为四个关系分量生成的逐边符号投影。"""

    values: Any
    component_names: tuple[str, ...]
    component_weights: tuple[float, ...]
    component_polarities: tuple[float, ...]
    component_identity_digest: str
    projection_digest: str


@dataclass(frozen=True)
class AttentionRelationGraphIdentity:
    """保存一次多层四分量关系评分的科学算子身份。"""

    component_names: tuple[str, ...]
    relation_source: str
    component_identity_digest: str
    keyed_projection_digest: str
    soft_rank_temperature: float
    soft_rank_scale: float
    relative_distance_scale: float
    qk_operator_metadata_records: tuple[dict[str, Any], ...]
    qk_operator_metadata_digest: str
    qk_operator_metadata_ready: bool


def attention_probability(attention: Any) -> Any:
    """统一读取真实 Q/K 关系对象或概率矩阵中的 attention 概率。"""

    if isinstance(attention, QKAttentionRelation):
        return attention.probabilities
    return attention


def centered_qk_logits(attention: Any) -> tuple[Any, str]:
    """读取真实中心化 Q/K logits, 或从概率恢复行常数不变的 logits。

    正式运行由 ``QKAttentionRelation`` 提供直接 Q/K logits。仅传入概率矩阵
    时, ``log(A)`` 与原始 logits 只相差逐行常数, 再次中心化后得到同一关系
    坐标。该输入形式用于轻量算法测试和通用函数调用, 不支持正式运行来源门禁。
    """

    torch = _torch()
    if isinstance(attention, QKAttentionRelation):
        return attention.centered_logits, attention.relation_source
    probability = attention_probability(attention).float()
    if probability.ndim not in {2, 3}:
        raise ValueError("attention 必须是二维或三维方形概率矩阵")
    minimum = torch.finfo(probability.dtype).tiny
    recovered = probability.clamp_min(minimum).log()
    return (
        recovered - recovered.mean(dim=-1, keepdim=True),
        PROBABILITY_INVERSE_RELATION_SOURCE,
    )


def public_token_grid_coordinates(token_indices: tuple[int, ...], device: Any) -> Any:
    """由公开原始 token 索引恢复归一化二维坐标。"""

    torch = _torch()
    if len(token_indices) < 4 or len(set(token_indices)) != len(token_indices):
        raise ValueError("二维关系图要求至少4个不重复的原始 token 索引")
    source_token_count = max(token_indices) + 1
    source_side = int(round(math.sqrt(source_token_count)))
    if source_side * source_side != source_token_count:
        raise ValueError("原始 token 索引无法还原为方形图像网格")
    if source_side < 2:
        raise ValueError("二维关系图边长至少为2")
    coordinates = []
    for token_index in token_indices:
        row, column = divmod(token_index, source_side)
        coordinates.append(
            (
                -1.0 + 2.0 * column / (source_side - 1),
                -1.0 + 2.0 * row / (source_side - 1),
            )
        )
    return torch.tensor(coordinates, device=device, dtype=torch.float32)


def _differentiable_row_rank(centered_logits: Any) -> Any:
    """以冻结温度计算降序 soft-rank, 并用 ``1 / token_count`` 缩放。"""

    torch = _torch()
    matrix = centered_logits.unsqueeze(0) if centered_logits.ndim == 2 else centered_logits
    token_count = int(matrix.shape[-1])
    identity = torch.eye(
        token_count,
        device=matrix.device,
        dtype=matrix.dtype,
    )
    rank_chunks = []
    for start in range(0, token_count, ATTENTION_RELATION_SOFT_RANK_CHUNK_SIZE):
        stop = min(token_count, start + ATTENTION_RELATION_SOFT_RANK_CHUNK_SIZE)
        selected = matrix[..., start:stop]
        comparisons = torch.sigmoid(
            (
                matrix.unsqueeze(-2)
                - selected.unsqueeze(-1)
            )
            / ATTENTION_RELATION_SOFT_RANK_TEMPERATURE
        )
        comparison_mask = 1.0 - identity[start:stop]
        ranks = 1.0 + (
            comparisons
            * comparison_mask.reshape(
                *((1,) * (comparisons.ndim - 2)),
                stop - start,
                token_count,
            )
        ).sum(dim=-1)
        rank_chunks.append(ranks / float(token_count))
    result = torch.cat(rank_chunks, dim=-1)
    return result[0] if centered_logits.ndim == 2 else result


def build_attention_relation_descriptor(
    attention: Any,
    token_indices: tuple[int, ...],
) -> AttentionRelationDescriptor:
    """构造非冗余四分量 attention-relative graph 边描述。

    四个分量依次为真实中心化 Q/K logit、基于该 logit 的可微降序 soft-rank、
    row-normalized attention 概率, 以及逐行中心化概率与逐行中心化公开二维
    token 相对距离的乘积。soft-rank 温度固定为0.25个 logit 单位, 输出按
    ``1 / token_count`` 缩放；距离按方形归一化网格最大欧氏距离
    ``2 * sqrt(2)`` 缩放。第四分量在均匀 attention 下严格为0, 因而公开坐标
    不能脱离真实 Q/K 内容形成密钥相关。该结构可直接复用于嵌入、盲检和双边
    仿射注册。
    """

    torch = _torch()
    probability = attention_probability(attention).float()
    logits, relation_source = centered_qk_logits(attention)
    logits = logits.float()
    if probability.ndim == 2:
        probability = probability.unsqueeze(0)
    if logits.ndim == 2:
        logits = logits.unsqueeze(0)
    if (
        probability.ndim != 3
        or logits.shape != probability.shape
        or probability.shape[-2] != probability.shape[-1]
    ):
        raise ValueError("Q/K logits 与 attention 概率必须具有相同方形关系图形状")
    token_count = int(probability.shape[-1])
    if len(token_indices) != token_count:
        raise ValueError("token_indices 数量必须与 Q/K 关系图宽度一致")
    row_probability = probability / probability.sum(
        dim=-1,
        keepdim=True,
    ).clamp_min(1e-12)
    soft_rank = _differentiable_row_rank(logits)
    coordinates = public_token_grid_coordinates(token_indices, probability.device)
    relative_distance_scale = 1.0 / (2.0 * math.sqrt(2.0))
    distances = torch.cdist(coordinates, coordinates) * relative_distance_scale
    distances = distances.reshape(1, token_count, token_count).expand(
        probability.shape[0],
        -1,
        -1,
    )
    centered_probability = row_probability - row_probability.mean(
        dim=-1,
        keepdim=True,
    )
    centered_distance = distances - distances.mean(dim=-1, keepdim=True)
    distance_weighted_probability = centered_probability * centered_distance
    values = torch.stack(
        (
            logits,
            soft_rank,
            row_probability,
            distance_weighted_probability,
        ),
        dim=-1,
    )
    soft_rank_scale = 1.0 / float(token_count)
    identity_payload = {
        "component_names": ATTENTION_RELATION_COMPONENT_NAMES,
        "soft_rank_definition": "descending_pairwise_sigmoid_rank",
        "soft_rank_temperature": ATTENTION_RELATION_SOFT_RANK_TEMPERATURE,
        "soft_rank_scale": soft_rank_scale,
        "row_probability_definition": "attention_probability_divided_by_row_sum",
        "relative_distance_definition": (
            "row_centered_attention_probability_times_row_centered_public_2d_distance"
        ),
        "relative_distance_scale": relative_distance_scale,
        "token_indices": token_indices,
        "relation_source": relation_source,
    }
    return AttentionRelationDescriptor(
        values=values,
        token_indices=token_indices,
        component_names=ATTENTION_RELATION_COMPONENT_NAMES,
        soft_rank_temperature=ATTENTION_RELATION_SOFT_RANK_TEMPERATURE,
        soft_rank_scale=soft_rank_scale,
        relative_distance_scale=relative_distance_scale,
        relation_source=relation_source,
        component_identity_digest=build_stable_digest(identity_payload),
    )


def qk_self_attention(
    module: Any,
    hidden_states: Any,
    max_tokens: int = 256,
    *,
    layer_name: str = "",
) -> tuple[Any, tuple[int, ...]]:
    """在真实二维图像 token 网格上抽样并计算 Q/K 关系对象。

    抽样沿原始网格的两个空间轴分别等距进行, 不把一维序号等距抽样误解释为
    二维网格。返回的 `token_indices` 始终指向原始图像 token 网格, 供检测端恢复
    真实空间坐标。该结构可以复用于任何公开 `to_q`、`to_k` 和 `heads` 的
    Transformer 注意力模块。返回概率是抽样图像 token 集合上的项目关系概率,
    不表示包含文本 token 与未抽样图像 token 的完整模块 attention 权重。多头
    概率先逐头 softmax 再平均, 不等同于平均 logits 的 softmax。
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
    expected_attention_scale = 1.0 / math.sqrt(head_width)
    module_scale = getattr(module, "scale", None)
    attention_scale = (
        expected_attention_scale
        if module_scale is None
        else float(module_scale)
    )
    if not math.isfinite(attention_scale) or not math.isclose(
        attention_scale,
        expected_attention_scale,
        rel_tol=1e-6,
        abs_tol=1e-12,
    ):
        raise RuntimeError("注意力模块 scale 与 1 / sqrt(head_width) 不一致")
    per_head_logits = (
        query.float() @ key.float().transpose(-1, -2)
    ) * attention_scale
    centered_logits = per_head_logits - per_head_logits.mean(dim=-1, keepdim=True)
    attention = torch.softmax(per_head_logits, dim=-1)
    relation = QKAttentionRelation(
        centered_logits=centered_logits.mean(dim=1),
        probabilities=attention.mean(dim=1),
        metadata={
            "module_layer_name": (
                layer_name or module.__class__.__qualname__
            ),
            "module_class_name": (
                f"{module.__class__.__module__}.{module.__class__.__qualname__}"
            ),
            "head_count": heads,
            "head_width": head_width,
            "attention_scale": attention_scale,
            "attention_scale_source": (
                "module_scale"
                if module_scale is not None
                else "inverse_sqrt_head_width"
            ),
            "q_normalization_applied": norm_q is not None,
            "k_normalization_applied": norm_k is not None,
            "q_normalization_class": (
                ""
                if norm_q is None
                else f"{norm_q.__class__.__module__}.{norm_q.__class__.__qualname__}"
            ),
            "k_normalization_class": (
                ""
                if norm_k is None
                else f"{norm_k.__class__.__module__}.{norm_k.__class__.__qualname__}"
            ),
            "source_token_count": token_count,
            "source_grid_side": source_side,
            "sampled_token_count": len(token_indices),
            "sampled_grid_side": sampled_side,
            "sampled_token_indices": list(token_indices),
            "centered_logit_aggregation": (
                "mean_of_per_head_row_centered_sampled_qk_logits"
            ),
            "relation_probability_aggregation": (
                "mean_of_per_head_sampled_image_token_probabilities"
            ),
            "mean_probability_is_softmax_of_mean_logits": False,
        },
    )
    return relation, token_indices


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
        matrix = attention_probability(attention).float()
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


@dataclass(frozen=True)
class StableAttentionPairWeights:
    """保存稳定 token 对权重的科学身份和当前坐标实现。

    ``pair_weight_identity_digest`` 绑定稳定 token 选择、非稳定权重和外积规则,
    因而在嵌入、仿射注册与恢复后复验之间保持不变。``token_weights`` 可以随
    坐标变换进行双线性传递, ``pair_weight_realization_digest`` 则区分观测网格
    与规范网格上的数值实现。该区分避免把重新选择 token 误写成同一组权重。
    """

    token_weights: tuple[float, ...]
    grid_token_indices: tuple[int, ...]
    stable_token_positions: tuple[int, ...]
    stable_token_indices: tuple[int, ...]
    stable_token_fraction: float
    unstable_pair_weight: float
    pair_weight_identity_digest: str
    pair_weight_realization_digest: str
    coordinate_space: str

    def pair_tensor(self, reference_attention: Any) -> Any:
        """在目标 attention 的设备和精度上恢复零对角 pair 权重矩阵。"""

        torch = _torch()
        token_count = int(reference_attention.shape[-1])
        if len(self.token_weights) != token_count:
            raise ValueError("稳定 token 权重宽度必须与 attention 宽度一致")
        dtype = (
            reference_attention.dtype
            if reference_attention.dtype.is_floating_point
            else torch.float32
        )
        token_weights = torch.tensor(
            self.token_weights,
            device=reference_attention.device,
            dtype=dtype,
        )
        off_diagonal = 1.0 - torch.eye(
            token_count,
            device=reference_attention.device,
            dtype=dtype,
        )
        return token_weights[:, None] * token_weights[None, :] * off_diagonal


def _pair_weight_realization_digest(
    pair_weight_identity_digest: str,
    token_weights: tuple[float, ...],
    coordinate_space: str,
) -> str:
    """记录同一权重身份在指定坐标系中的数值实现。"""

    return build_stable_digest(
        {
            "pair_weight_identity_digest": pair_weight_identity_digest,
            "coordinate_space": coordinate_space,
            "token_weights": [round(float(value), 12) for value in token_weights],
            "pair_rule": "outer_product_token_weights_off_diagonal",
        }
    )


def build_stable_attention_pair_weights(
    records: Iterable[tuple[str, Any, tuple[int, ...]]],
    selection: StableAttentionTokenSelection,
    unstable_pair_weight: float = 0.25,
) -> StableAttentionPairWeights:
    """由一次冻结选择构造可跨嵌入、注册和检测复用的 pair 权重。

    该函数属于通用工程写法: 所有使用稳定 token 的科学算子都只接收这一对象,
    不在各自内部重复解释权重规则。项目特定设计是稳定 token 权重为1, 其余
    token 权重为 ``unstable_pair_weight``, pair 权重由两个端点权重的外积得到。
    """

    resolved_records = tuple(records)
    if not resolved_records:
        raise ValueError("构造稳定 token 权重至少需要一层 Q/K 记录")
    reference_indices = tuple(int(value) for value in resolved_records[0][2])
    token_count = len(reference_indices)
    if token_count < 4 or any(
        tuple(token_indices) != reference_indices
        for _, _, token_indices in resolved_records[1:]
    ):
        raise ValueError("稳定 token 权重要求共享且不少于4点的 Q/K 网格")
    if not 0.0 <= unstable_pair_weight < 1.0:
        raise ValueError("unstable_pair_weight 必须位于 [0, 1)")
    if not selection.token_positions:
        raise ValueError("稳定 token 选择不得为空")
    if any(position < 0 or position >= token_count for position in selection.token_positions):
        raise ValueError("稳定 token 位置超出 Q/K 网格")
    selected_indices = tuple(
        reference_indices[position] for position in selection.token_positions
    )
    if selected_indices != selection.token_indices:
        raise ValueError("稳定 token 选择与 Q/K 原始索引不一致")
    token_weights = [float(unstable_pair_weight)] * token_count
    for position in selection.token_positions:
        token_weights[position] = 1.0
    resolved_token_weights = tuple(token_weights)
    identity_digest = build_stable_digest(
        {
            "selection_digest": selection.selection_digest,
            "stable_token_positions": selection.token_positions,
            "stable_token_indices": selection.token_indices,
            "grid_token_indices": reference_indices,
            "stable_token_fraction": selection.stable_token_fraction,
            "unstable_pair_weight": float(unstable_pair_weight),
            "token_count": token_count,
            "pair_rule": "outer_product_token_weights_off_diagonal",
        }
    )
    coordinate_space = "selected_qk_observation_grid"
    return StableAttentionPairWeights(
        token_weights=resolved_token_weights,
        grid_token_indices=reference_indices,
        stable_token_positions=selection.token_positions,
        stable_token_indices=selection.token_indices,
        stable_token_fraction=selection.stable_token_fraction,
        unstable_pair_weight=float(unstable_pair_weight),
        pair_weight_identity_digest=identity_digest,
        pair_weight_realization_digest=_pair_weight_realization_digest(
            identity_digest,
            resolved_token_weights,
            coordinate_space,
        ),
        coordinate_space=coordinate_space,
    )


def transport_stable_attention_pair_weights(
    pair_weights: StableAttentionPairWeights,
    sampling_weights: Any,
    valid_positions: Any,
    *,
    coordinate_space: str,
) -> StableAttentionPairWeights:
    """使用与关系图相同的双线性采样矩阵传递稳定 token 权重。

    ``sampling_weights`` 的每一行描述目标 token 如何从源 token 插值。算子先
    计算单点权重场 ``a' = W a``, 再计算 ``P' = a' a'^T`` 并清零对角线。
    该定义不是 ``P' = W P W^T``；后者会传播源 pair 的零对角约束并产生不同
    数值。无覆盖位置权重置零并由注册覆盖门禁处理, 不重新选择 token。
    """

    torch = _torch()
    if sampling_weights.ndim != 2 or sampling_weights.shape[0] != sampling_weights.shape[1]:
        raise ValueError("sampling_weights 必须是方形二维矩阵")
    token_count = int(sampling_weights.shape[0])
    if len(pair_weights.token_weights) != token_count:
        raise ValueError("采样矩阵宽度必须与稳定 token 权重宽度一致")
    valid = torch.as_tensor(
        valid_positions,
        device=sampling_weights.device,
        dtype=torch.bool,
    )
    if valid.shape != (token_count,):
        raise ValueError("valid_positions 必须与目标 token 数量一致")
    source_weights = torch.tensor(
        pair_weights.token_weights,
        device=sampling_weights.device,
        dtype=torch.float32,
    )
    transported = sampling_weights.float() @ source_weights
    transported = transported * valid.to(dtype=transported.dtype)
    values = tuple(float(value) for value in transported.detach().cpu().tolist())
    return StableAttentionPairWeights(
        token_weights=values,
        grid_token_indices=pair_weights.grid_token_indices,
        stable_token_positions=pair_weights.stable_token_positions,
        stable_token_indices=pair_weights.stable_token_indices,
        stable_token_fraction=pair_weights.stable_token_fraction,
        unstable_pair_weight=pair_weights.unstable_pair_weight,
        pair_weight_identity_digest=pair_weights.pair_weight_identity_digest,
        pair_weight_realization_digest=_pair_weight_realization_digest(
            pair_weights.pair_weight_identity_digest,
            values,
            coordinate_space,
        ),
        coordinate_space=coordinate_space,
    )


def restore_transported_stable_attention_pair_weights(
    source_pair_weights: StableAttentionPairWeights,
    token_weights: tuple[float, ...],
    *,
    coordinate_space: str,
    expected_realization_digest: str,
) -> StableAttentionPairWeights:
    """从注册结果恢复同一身份的规范网格权重并校验数值摘要。"""

    if len(token_weights) != len(source_pair_weights.token_weights):
        raise ValueError("恢复后的稳定 token 权重宽度不一致")
    realization_digest = _pair_weight_realization_digest(
        source_pair_weights.pair_weight_identity_digest,
        token_weights,
        coordinate_space,
    )
    if realization_digest != expected_realization_digest:
        raise RuntimeError("注册结果中的稳定 token pair 权重实现摘要不一致")
    return StableAttentionPairWeights(
        token_weights=token_weights,
        grid_token_indices=source_pair_weights.grid_token_indices,
        stable_token_positions=source_pair_weights.stable_token_positions,
        stable_token_indices=source_pair_weights.stable_token_indices,
        stable_token_fraction=source_pair_weights.stable_token_fraction,
        unstable_pair_weight=source_pair_weights.unstable_pair_weight,
        pair_weight_identity_digest=(
            source_pair_weights.pair_weight_identity_digest
        ),
        pair_weight_realization_digest=realization_digest,
        coordinate_space=coordinate_space,
    )


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
        matrix = attention_probability(attention).float()
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
            attention, token_indices = qk_self_attention(
                module,
                hidden_states,
                self.max_tokens,
                layer_name=layer_name,
            )
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


def keyed_attention_relation_projection(
    descriptor: AttentionRelationDescriptor,
    key_material: str,
    layer_name: str,
) -> KeyedAttentionRelationProjection:
    """为四分量关系描述构造共享密钥图和冻结分量极性的投影。"""

    torch = _torch()
    base_signs = keyed_relation_signs(
        descriptor.values[..., 0],
        key_material,
        layer_name,
    )
    polarities = torch.tensor(
        ATTENTION_RELATION_COMPONENT_POLARITIES,
        device=descriptor.values.device,
        dtype=torch.float32,
    )
    values = base_signs.unsqueeze(-1) * polarities.reshape(1, 1, -1)
    projection_payload = {
        "component_identity_digest": descriptor.component_identity_digest,
        "component_names": descriptor.component_names,
        "component_weights": ATTENTION_RELATION_COMPONENT_WEIGHTS,
        "component_polarities": ATTENTION_RELATION_COMPONENT_POLARITIES,
        "key_material_digest": hashlib.sha256(
            key_material.encode("utf-8")
        ).hexdigest(),
        "layer_name": layer_name,
        "token_indices": descriptor.token_indices,
        "projection_rule": "shared_symmetric_pair_sign_with_component_polarity",
    }
    return KeyedAttentionRelationProjection(
        values=values,
        component_names=descriptor.component_names,
        component_weights=ATTENTION_RELATION_COMPONENT_WEIGHTS,
        component_polarities=ATTENTION_RELATION_COMPONENT_POLARITIES,
        component_identity_digest=descriptor.component_identity_digest,
        projection_digest=build_stable_digest(projection_payload),
    )


def build_attention_relation_graph_identity(
    records: Iterable[tuple[str, Any, tuple[int, ...]]],
    key_material: str,
) -> AttentionRelationGraphIdentity:
    """重建多层关系分量和密钥投影的共同身份。"""

    resolved_records = tuple(records)
    if not resolved_records:
        raise RuntimeError("关系图身份要求至少一层真实 Q/K 记录")
    descriptors = tuple(
        build_attention_relation_descriptor(attention, token_indices)
        for _, attention, token_indices in resolved_records
    )
    reference = descriptors[0]
    if any(
        descriptor.component_identity_digest
        != reference.component_identity_digest
        for descriptor in descriptors[1:]
    ):
        raise RuntimeError("多层 Q/K 记录没有共享同一四分量关系算子身份")
    projections = tuple(
        keyed_attention_relation_projection(
            descriptor,
            key_material,
            layer_name,
        )
        for (layer_name, _, _), descriptor in zip(resolved_records, descriptors)
    )
    projection_digest = build_stable_digest(
        {
            "component_identity_digest": reference.component_identity_digest,
            "layer_projection_digests": [
                (layer_name, projection.projection_digest)
                for (layer_name, _, _), projection in zip(
                    resolved_records,
                    projections,
                )
            ],
        }
    )
    operator_records = []
    operator_ready = True
    for layer_name, attention, token_indices in resolved_records:
        metadata = (
            dict(attention.metadata)
            if isinstance(attention, QKAttentionRelation)
            else {}
        )
        record = {"record_layer_name": layer_name, **metadata}
        operator_records.append(record)
        head_width = metadata.get("head_width")
        attention_scale = metadata.get("attention_scale")
        sampled_indices = metadata.get("sampled_token_indices")
        sampled_grid_side = metadata.get("sampled_grid_side")
        source_grid_side = metadata.get("source_grid_side")
        source_token_count = metadata.get("source_token_count")
        q_normalized = metadata.get("q_normalization_applied")
        k_normalized = metadata.get("k_normalization_applied")
        operator_ready = operator_ready and bool(
            isinstance(attention, QKAttentionRelation)
            and metadata.get("module_layer_name") == layer_name
            and isinstance(metadata.get("module_class_name"), str)
            and bool(metadata.get("module_class_name"))
            and isinstance(metadata.get("head_count"), int)
            and int(metadata.get("head_count", 0)) > 0
            and isinstance(head_width, int)
            and int(head_width or 0) > 0
            and isinstance(attention_scale, (int, float))
            and math.isfinite(float(attention_scale or 0.0))
            and math.isclose(
                float(attention_scale or 0.0),
                1.0 / math.sqrt(int(head_width or 1)),
                rel_tol=1e-6,
                abs_tol=1e-12,
            )
            and metadata.get("attention_scale_source")
            in {"module_scale", "inverse_sqrt_head_width"}
            and isinstance(q_normalized, bool)
            and isinstance(k_normalized, bool)
            and (
                bool(metadata.get("q_normalization_class"))
                == q_normalized
            )
            and (
                bool(metadata.get("k_normalization_class"))
                == k_normalized
            )
            and metadata.get("sampled_token_count") == len(token_indices)
            and isinstance(sampled_grid_side, int)
            and sampled_grid_side**2 == len(token_indices)
            and sampled_indices == list(token_indices)
            and isinstance(source_grid_side, int)
            and isinstance(source_token_count, int)
            and source_grid_side**2 == source_token_count
            and metadata.get("centered_logit_aggregation")
            == "mean_of_per_head_row_centered_sampled_qk_logits"
            and metadata.get("relation_probability_aggregation")
            == "mean_of_per_head_sampled_image_token_probabilities"
            and metadata.get("mean_probability_is_softmax_of_mean_logits")
            is False
        )
    qk_operator_metadata_records = tuple(operator_records)
    qk_operator_metadata_digest = build_stable_digest(
        {"qk_operator_metadata_records": qk_operator_metadata_records}
    )
    return AttentionRelationGraphIdentity(
        component_names=reference.component_names,
        relation_source=reference.relation_source,
        component_identity_digest=reference.component_identity_digest,
        keyed_projection_digest=projection_digest,
        soft_rank_temperature=reference.soft_rank_temperature,
        soft_rank_scale=reference.soft_rank_scale,
        relative_distance_scale=reference.relative_distance_scale,
        qk_operator_metadata_records=qk_operator_metadata_records,
        qk_operator_metadata_digest=qk_operator_metadata_digest,
        qk_operator_metadata_ready=operator_ready,
    )


def attention_relation_component_scores(
    relation_values: Any,
    projection_values: Any,
    pair_weights: Any,
    valid_positions: Any | None = None,
) -> Any:
    """逐行加权中心化并归一化四个关系分量, 返回分量级相关分数。

    该算子同时服务嵌入评分和仿射注册。每个分量先独立计算逐行加权相关,
    再在具有非零关系方差的有效行上取平均, 避免 logits、概率、rank 与距离
    的数值尺度差异改变冻结的等权组合。
    """

    torch = _torch()
    relation = relation_values
    if relation.ndim == 3:
        relation = relation.unsqueeze(0)
    if relation.ndim != 4 or int(relation.shape[-1]) != len(
        ATTENTION_RELATION_COMPONENT_NAMES
    ):
        raise ValueError("relation_values 必须具有 [batch, token, token, 4] 形状")
    projection = projection_values
    if projection.ndim == 3:
        projection = projection.unsqueeze(0)
    if projection.shape[0] == 1 and relation.shape[0] != 1:
        projection = projection.expand(relation.shape[0], -1, -1, -1)
    if projection.shape != relation.shape:
        raise ValueError("projection_values 必须与 relation_values 具有相同关系图形状")
    weights = pair_weights
    if weights.ndim == 2:
        weights = weights.unsqueeze(0)
    if weights.shape[0] == 1 and relation.shape[0] != 1:
        weights = weights.expand(relation.shape[0], -1, -1)
    if weights.shape != relation.shape[:-1]:
        raise ValueError("pair_weights 必须与四分量关系图的 token 轴一致")
    token_count = int(relation.shape[-2])
    if valid_positions is None:
        valid = torch.ones(
            relation.shape[0],
            token_count,
            device=relation.device,
            dtype=torch.bool,
        )
    else:
        valid = torch.as_tensor(
            valid_positions,
            device=relation.device,
            dtype=torch.bool,
        )
        if valid.ndim == 1:
            valid = valid.unsqueeze(0)
        if valid.shape[0] == 1 and relation.shape[0] != 1:
            valid = valid.expand(relation.shape[0], -1)
        if valid.shape != relation.shape[:2]:
            raise ValueError("valid_positions 必须与关系图 batch 和 token 轴一致")
    pair_valid = valid.unsqueeze(-1) & valid.unsqueeze(-2)
    off_diagonal = ~torch.eye(
        token_count,
        device=relation.device,
        dtype=torch.bool,
    ).unsqueeze(0)
    effective_weights = (
        weights.to(dtype=relation.dtype)
        * (pair_valid & off_diagonal).to(dtype=relation.dtype)
    )
    row_weight = effective_weights.sum(dim=-1, keepdim=True)
    safe_row_weight = row_weight.clamp_min(1e-12)
    expanded_weights = effective_weights.unsqueeze(-1)
    relation_mean = (
        relation * expanded_weights
    ).sum(dim=-2, keepdim=True) / safe_row_weight.unsqueeze(-1)
    projection_mean = (
        projection.to(dtype=relation.dtype) * expanded_weights
    ).sum(dim=-2, keepdim=True) / safe_row_weight.unsqueeze(-1)
    relation_centered = relation - relation_mean
    projection_centered = projection.to(dtype=relation.dtype) - projection_mean
    row_numerator = (
        relation_centered * projection_centered * expanded_weights
    ).sum(dim=-2)
    relation_energy = (
        relation_centered.square() * expanded_weights
    ).sum(dim=-2)
    projection_energy = (
        projection_centered.square() * expanded_weights
    ).sum(dim=-2)
    row_energy_product = relation_energy * projection_energy
    row_denominator = row_energy_product.clamp_min(1e-24).sqrt()
    row_valid = (row_weight.squeeze(-1) > 0.0).unsqueeze(-1) & (
        row_energy_product > 1e-24
    )
    row_scores = torch.where(
        row_valid,
        row_numerator / row_denominator.clamp_min(1e-12),
        torch.zeros_like(row_numerator),
    )
    valid_row_count = row_valid.sum(dim=-2).clamp_min(1)
    return (row_scores * row_valid.to(dtype=row_scores.dtype)).sum(
        dim=-2
    ) / valid_row_count


def combine_attention_relation_component_scores(component_scores: Any) -> Any:
    """按冻结的四分量等权协议组合逐分量关系分数。"""

    torch = _torch()
    if component_scores.shape[-1] != len(ATTENTION_RELATION_COMPONENT_WEIGHTS):
        raise ValueError("component_scores 最后一维必须覆盖全部四个关系分量")
    weights = torch.tensor(
        ATTENTION_RELATION_COMPONENT_WEIGHTS,
        device=component_scores.device,
        dtype=component_scores.dtype,
    )
    return (component_scores * weights).sum(dim=-1)


def attention_geometry_component_scores(
    records: Iterable[tuple[str, Any, tuple[int, ...]]],
    key_material: str,
    stable_pair_weights: StableAttentionPairWeights,
) -> Any:
    """使用同一 pair 权重计算多层四分量几何分数。"""

    torch = _torch()
    layer_component_scores = []
    for layer_name, attention, token_indices in tuple(records):
        token_count = int(attention.shape[-1])
        if len(token_indices) != token_count:
            raise ValueError("Q/K token_indices 数量必须与 attention 宽度一致")
        if tuple(token_indices) != stable_pair_weights.grid_token_indices:
            raise ValueError("稳定 token pair 权重与当前 Q/K 抽样网格不一致")
        descriptor = build_attention_relation_descriptor(attention, token_indices)
        projection = keyed_attention_relation_projection(
            descriptor,
            key_material,
            layer_name,
        )
        component_scores = attention_relation_component_scores(
            descriptor.values,
            projection.values,
            stable_pair_weights.pair_tensor(attention),
        )
        layer_component_scores.append(component_scores.mean(dim=0))
    if not layer_component_scores:
        raise RuntimeError("真实 Transformer 前向没有产生 Q/K attention 记录")
    return torch.stack(layer_component_scores).mean(dim=0)


def attention_geometry_score(
    records: Iterable[tuple[str, Any, tuple[int, ...]]],
    key_material: str,
    *,
    stable_token_positions: tuple[int, ...] | None = None,
    stable_token_fraction: float = 0.5,
    unstable_pair_weight: float = 0.25,
    stable_pair_weights: StableAttentionPairWeights | None = None,
) -> Any:
    """计算稳定 token 加权的四分量 Q/K 密钥关系一致性分数。

    稳定 token 对使用权重1, 其余规则网格关系保留较小的冻结权重。保留完整
    网格可为盲检仿射注册提供连续二维采样支撑, 同时保证稳定 token 集合真实
    改变嵌入目标而不是只作为日志字段。四分量分别逐行中心化、归一化后等权,
    防止不同物理量的数值尺度主导最终分数。
    """

    resolved_records = tuple(records)
    if stable_pair_weights is not None and stable_token_positions is not None:
        raise ValueError("不得同时传入 stable_pair_weights 和 stable_token_positions")
    if stable_pair_weights is None and stable_token_positions is None:
        selection = select_stable_attention_tokens(
            resolved_records,
            stable_token_fraction=stable_token_fraction,
        )
        stable_pair_weights = build_stable_attention_pair_weights(
            resolved_records,
            selection,
            unstable_pair_weight=unstable_pair_weight,
        )
    elif stable_pair_weights is None:
        if not 0.0 <= unstable_pair_weight < 1.0:
            raise ValueError("unstable_pair_weight 必须位于 [0, 1)")
        if not stable_token_positions:
            raise ValueError("stable_token_positions 不得为空")
        reference_indices = resolved_records[0][2]
        stable_pair_weights = build_stable_attention_pair_weights(
            resolved_records,
            StableAttentionTokenSelection(
                token_positions=stable_token_positions,
                token_indices=tuple(
                    reference_indices[position] for position in stable_token_positions
                ),
                stable_token_fraction=float(stable_token_fraction),
                selection_digest=build_stable_digest(
                    {
                        "selection_rule": "explicit_frozen_token_positions",
                        "stable_token_positions": stable_token_positions,
                        "stable_token_fraction": float(stable_token_fraction),
                    }
                ),
            ),
            unstable_pair_weight=unstable_pair_weight,
        )
    component_scores = attention_geometry_component_scores(
        resolved_records,
        key_material,
        stable_pair_weights,
    )
    return combine_attention_relation_component_scores(component_scores)


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
    stable_pair_weight_identity_digest: str
    stable_pair_weight_realization_digest: str
    attention_relation_component_names: tuple[str, ...]
    attention_relation_source: str
    attention_relation_component_identity_digest: str
    attention_relation_keyed_projection_digest: str
    attention_relation_soft_rank_temperature: float
    attention_relation_soft_rank_scale: float
    attention_relation_relative_distance_scale: float
    attention_relation_qk_operator_metadata_records: tuple[dict[str, Any], ...]
    attention_relation_qk_operator_metadata_digest: str
    attention_relation_qk_operator_metadata_ready: bool
    stable_token_fraction: float
    unstable_pair_weight: float
    stable_pair_weights: StableAttentionPairWeights


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
        pair_weights = build_stable_attention_pair_weights(
            recorder.records,
            selection,
            unstable_pair_weight=unstable_pair_weight,
        )
        score_before_tensor = attention_geometry_score(
            recorder.records,
            key_material,
            stable_pair_weights=pair_weights,
        )
        relation_identity = build_attention_relation_graph_identity(
            recorder.records,
            key_material,
        )
        if (
            relation_identity.relation_source != DIRECT_QK_RELATION_SOURCE
            or not relation_identity.qk_operator_metadata_ready
        ):
            raise RuntimeError("正式注意力梯度必须绑定完整真实 Q/K 算子元数据")
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
        stable_pair_weight_identity_digest=(
            pair_weights.pair_weight_identity_digest
        ),
        stable_pair_weight_realization_digest=(
            pair_weights.pair_weight_realization_digest
        ),
        attention_relation_component_names=(
            relation_identity.component_names
        ),
        attention_relation_source=relation_identity.relation_source,
        attention_relation_component_identity_digest=(
            relation_identity.component_identity_digest
        ),
        attention_relation_keyed_projection_digest=(
            relation_identity.keyed_projection_digest
        ),
        attention_relation_soft_rank_temperature=(
            relation_identity.soft_rank_temperature
        ),
        attention_relation_soft_rank_scale=relation_identity.soft_rank_scale,
        attention_relation_relative_distance_scale=(
            relation_identity.relative_distance_scale
        ),
        attention_relation_qk_operator_metadata_records=(
            relation_identity.qk_operator_metadata_records
        ),
        attention_relation_qk_operator_metadata_digest=(
            relation_identity.qk_operator_metadata_digest
        ),
        attention_relation_qk_operator_metadata_ready=(
            relation_identity.qk_operator_metadata_ready
        ),
        stable_token_fraction=selection.stable_token_fraction,
        unstable_pair_weight=float(unstable_pair_weight),
        stable_pair_weights=pair_weights,
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
    stable_pair_weight_identity_digest: str
    stable_pair_weight_realization_digest: str
    attention_relation_component_names: tuple[str, ...]
    attention_relation_source: str
    attention_relation_component_identity_digest: str
    attention_relation_keyed_projection_digest: str
    attention_relation_qk_operator_metadata_records: tuple[dict[str, Any], ...]
    attention_relation_qk_operator_metadata_digest: str
    attention_relation_qk_operator_metadata_ready: bool
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
    if (
        content_base_evidence.stable_pair_weight_identity_digest
        != original_evidence.stable_pair_weight_identity_digest
        or content_base_evidence.stable_pair_weight_realization_digest
        != original_evidence.stable_pair_weight_realization_digest
        or content_base_evidence.attention_relation_component_identity_digest
        != original_evidence.attention_relation_component_identity_digest
        or content_base_evidence.attention_relation_keyed_projection_digest
        != original_evidence.attention_relation_keyed_projection_digest
        or content_base_evidence.attention_relation_qk_operator_metadata_digest
        != original_evidence.attention_relation_qk_operator_metadata_digest
    ):
        raise RuntimeError("注意力梯度、内容基底与回溯没有共享同一关系图身份")
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
                stable_pair_weights=content_base_evidence.stable_pair_weights,
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
        "stable_pair_weight_identity_digest": (
            content_base_evidence.stable_pair_weight_identity_digest
        ),
        "stable_pair_weight_realization_digest": (
            content_base_evidence.stable_pair_weight_realization_digest
        ),
        "attention_relation_component_names": (
            content_base_evidence.attention_relation_component_names
        ),
        "attention_relation_source": (
            content_base_evidence.attention_relation_source
        ),
        "attention_relation_component_identity_digest": (
            content_base_evidence.attention_relation_component_identity_digest
        ),
        "attention_relation_keyed_projection_digest": (
            content_base_evidence.attention_relation_keyed_projection_digest
        ),
        "attention_relation_qk_operator_metadata_digest": (
            content_base_evidence.attention_relation_qk_operator_metadata_digest
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
        stable_pair_weight_identity_digest=(
            content_base_evidence.stable_pair_weight_identity_digest
        ),
        stable_pair_weight_realization_digest=(
            content_base_evidence.stable_pair_weight_realization_digest
        ),
        attention_relation_component_names=(
            content_base_evidence.attention_relation_component_names
        ),
        attention_relation_source=(
            content_base_evidence.attention_relation_source
        ),
        attention_relation_component_identity_digest=(
            content_base_evidence.attention_relation_component_identity_digest
        ),
        attention_relation_keyed_projection_digest=(
            content_base_evidence.attention_relation_keyed_projection_digest
        ),
        attention_relation_qk_operator_metadata_records=(
            content_base_evidence.attention_relation_qk_operator_metadata_records
        ),
        attention_relation_qk_operator_metadata_digest=(
            content_base_evidence.attention_relation_qk_operator_metadata_digest
        ),
        attention_relation_qk_operator_metadata_ready=(
            content_base_evidence.attention_relation_qk_operator_metadata_ready
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
            "stable_pair_weight_identity_digest": (
                content_base_evidence.stable_pair_weight_identity_digest
            ),
            "stable_pair_weight_realization_digest": (
                content_base_evidence.stable_pair_weight_realization_digest
            ),
            "attention_relation_component_names": list(
                content_base_evidence.attention_relation_component_names
            ),
            "attention_relation_source": (
                content_base_evidence.attention_relation_source
            ),
            "attention_relation_direct_qk_source_ready": (
                content_base_evidence.attention_relation_source
                == DIRECT_QK_RELATION_SOURCE
            ),
            "attention_relation_probability_scope": (
                "sampled_image_token_qk_relation_probability"
            ),
            "attention_relation_component_identity_digest": (
                content_base_evidence.attention_relation_component_identity_digest
            ),
            "attention_relation_keyed_projection_digest": (
                content_base_evidence.attention_relation_keyed_projection_digest
            ),
            "attention_relation_qk_operator_metadata_records": list(
                content_base_evidence.attention_relation_qk_operator_metadata_records
            ),
            "attention_relation_qk_operator_metadata_digest": (
                content_base_evidence.attention_relation_qk_operator_metadata_digest
            ),
            "attention_relation_qk_operator_metadata_ready": (
                content_base_evidence.attention_relation_qk_operator_metadata_ready
            ),
            "attention_relation_soft_rank_temperature": (
                content_base_evidence.attention_relation_soft_rank_temperature
            ),
            "attention_relation_soft_rank_scale": (
                content_base_evidence.attention_relation_soft_rank_scale
            ),
            "attention_relation_relative_distance_scale": (
                content_base_evidence.attention_relation_relative_distance_scale
            ),
            "attention_relation_component_weights": list(
                ATTENTION_RELATION_COMPONENT_WEIGHTS
            ),
        },
    )
