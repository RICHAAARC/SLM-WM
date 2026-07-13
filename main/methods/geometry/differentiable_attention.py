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

from main.core.digest import (
    TENSOR_CONTENT_DIGEST_VERSION,
    build_stable_digest,
    tensor_content_sha256,
)
from main.core.keyed_prg import (
    build_keyed_uniform_tensor,
    keyed_prg_protocol_record,
)
from main.methods.subspace.jacobian_nullspace import JacobianNullSpaceResult
from main.methods.update_composition import (
    RiskBoundedUpdate,
    compose_ordered_float32_update_once,
    rescale_risk_bounded_update,
)


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
ATTENTION_RELATION_NUMERICAL_EPSILON = 1e-12
DIRECT_QK_RELATION_SOURCE = "direct_qk_centered_logits_and_probabilities"
ATTENTION_COORDINATE_CONVENTION = (
    "normalized_xy_token_centers_corner_endpoints_v1"
)
ATTENTION_GRID_ALIGN_CORNERS = True
FROZEN_SD35_ATTENTION_MODULE_NAMES = (
    "transformer_blocks.0.attn",
    "transformer_blocks.23.attn",
)
QK_ATOMIC_CONTENT_SHA256_FIELDS = (
    "sampled_query_content_sha256",
    "sampled_key_content_sha256",
    "centered_qk_logits_content_sha256",
    "qk_probabilities_content_sha256",
    "sampled_token_indices_content_sha256",
)
QK_OPERATOR_METADATA_FIELD_NAMES = (
    "module_layer_name",
    "module_class_name",
    "head_count",
    "head_width",
    "attention_scale",
    "attention_scale_source",
    "q_normalization_applied",
    "k_normalization_applied",
    "q_normalization_class",
    "k_normalization_class",
    "source_token_count",
    "source_grid_side",
    "sampled_token_count",
    "sampled_grid_side",
    "sampled_token_indices",
    "coordinate_convention",
    "grid_align_corners",
    "centered_logit_aggregation",
    "relation_probability_aggregation",
    "mean_probability_is_softmax_of_mean_logits",
)


def qk_operator_metadata_records_digest(
    records: Iterable[dict[str, Any]],
) -> str:
    """从完整冻结 Q/K 算子元数据重建聚合摘要。"""

    return build_stable_digest(
        {"qk_operator_metadata_records": tuple(records)}
    )


def qk_operator_metadata_records_ready(
    records: Iterable[dict[str, Any]],
    expected_layer_names: Iterable[str],
) -> bool:
    """完整复验冻结 Q/K 层、头布局、归一化和二维抽样协议。"""

    resolved_records = tuple(records)
    resolved_layers = tuple(expected_layer_names)
    if len(resolved_records) != len(resolved_layers) or not resolved_records:
        return False
    for layer_name, record in zip(resolved_layers, resolved_records):
        if not isinstance(record, dict) or set(record) != {
            "record_layer_name",
            *QK_OPERATOR_METADATA_FIELD_NAMES,
        }:
            return False
        head_count = record.get("head_count")
        head_width = record.get("head_width")
        attention_scale = record.get("attention_scale")
        source_token_count = record.get("source_token_count")
        source_grid_side = record.get("source_grid_side")
        sampled_token_count = record.get("sampled_token_count")
        sampled_grid_side = record.get("sampled_grid_side")
        sampled_indices = record.get("sampled_token_indices")
        q_normalized = record.get("q_normalization_applied")
        k_normalized = record.get("k_normalization_applied")
        integer_values = (
            head_count,
            head_width,
            source_token_count,
            source_grid_side,
            sampled_token_count,
            sampled_grid_side,
        )
        if (
            record.get("record_layer_name") != layer_name
            or record.get("module_layer_name") != layer_name
            or not isinstance(record.get("module_class_name"), str)
            or not record.get("module_class_name")
            or not all(
                isinstance(value, int)
                and not isinstance(value, bool)
                and value > 0
                for value in integer_values
            )
            or isinstance(attention_scale, bool)
            or not isinstance(attention_scale, (int, float))
            or not math.isfinite(float(attention_scale))
            or not math.isclose(
                float(attention_scale),
                1.0 / math.sqrt(int(head_width)),
                rel_tol=1e-6,
                abs_tol=ATTENTION_RELATION_NUMERICAL_EPSILON,
            )
            or record.get("attention_scale_source")
            not in {"module_scale", "inverse_sqrt_head_width"}
            or not isinstance(q_normalized, bool)
            or not isinstance(k_normalized, bool)
            or bool(record.get("q_normalization_class")) != q_normalized
            or bool(record.get("k_normalization_class")) != k_normalized
            or int(source_grid_side) ** 2 != int(source_token_count)
            or int(sampled_grid_side) ** 2 != int(sampled_token_count)
            or not isinstance(sampled_indices, list)
            or len(sampled_indices) != int(sampled_token_count)
            or len(set(sampled_indices)) != len(sampled_indices)
            or not all(
                isinstance(value, int)
                and not isinstance(value, bool)
                and 0 <= value < int(source_token_count)
                for value in sampled_indices
            )
            or record.get("coordinate_convention")
            != ATTENTION_COORDINATE_CONVENTION
            or record.get("grid_align_corners")
            is not ATTENTION_GRID_ALIGN_CORNERS
            or record.get("centered_logit_aggregation")
            != "mean_of_per_head_row_centered_sampled_qk_logits"
            or record.get("relation_probability_aggregation")
            != "mean_of_per_head_sampled_image_token_probabilities"
            or record.get("mean_probability_is_softmax_of_mean_logits")
            is not False
        ):
            return False
    return True


def validate_attention_relation_component_weights(
    component_weights: Iterable[float],
) -> tuple[float, ...]:
    """校验并规范化四分量评分协议的冻结权重.

    完整方法使用四个等权分量. 正式留一消融把被移除分量权重置零, 其余
    三个分量各取三分之一. 所有权重必须非负且精确归一化, 避免运行端通过
    未登记的整体缩放改变注意力阈值和回溯目标.
    """

    resolved = tuple(float(value) for value in component_weights)
    if len(resolved) != len(ATTENTION_RELATION_COMPONENT_NAMES):
        raise ValueError("注意力关系权重必须精确覆盖四个冻结分量")
    if any(not math.isfinite(value) or value < 0.0 for value in resolved):
        raise ValueError("注意力关系权重必须为非负有限数")
    if not math.isclose(
        sum(resolved),
        1.0,
        rel_tol=0.0,
        abs_tol=ATTENTION_RELATION_NUMERICAL_EPSILON,
    ):
        raise ValueError("注意力关系权重之和必须为 1")
    if not any(value > 0.0 for value in resolved):
        raise ValueError("注意力关系协议至少需要一个活动分量")
    return resolved


def attention_relation_component_protocol(
    component_weights: Iterable[float] = ATTENTION_RELATION_COMPONENT_WEIGHTS,
) -> dict[str, Any]:
    """返回四分量名称、活动集合、权重和稳定协议摘要."""

    resolved_weights = validate_attention_relation_component_weights(
        component_weights
    )
    active_names = tuple(
        name
        for name, weight in zip(
            ATTENTION_RELATION_COMPONENT_NAMES,
            resolved_weights,
        )
        if weight > 0.0
    )
    payload = {
        "attention_relation_component_names": (
            ATTENTION_RELATION_COMPONENT_NAMES
        ),
        "attention_relation_active_component_names": active_names,
        "attention_relation_component_weights": resolved_weights,
        "attention_relation_component_combination_rule": (
            "normalized_nonnegative_weighted_sum"
        ),
    }
    return {
        **payload,
        "attention_relation_component_protocol_digest": (
            build_stable_digest(payload)
        ),
    }


def _torch() -> Any:
    """延迟导入 PyTorch。"""

    import torch

    return torch


def _is_sha256_hex(value: Any) -> bool:
    """判断内容摘要是否为规范小写 SHA-256 文本."""

    text = str(value)
    return len(text) == 64 and all(
        character in "0123456789abcdef" for character in text
    )


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
    coordinate_convention: str
    grid_align_corners: bool
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
    active_component_names: tuple[str, ...]
    component_weights: tuple[float, ...]
    component_protocol_digest: str
    relation_source: str
    component_identity_digest: str
    keyed_projection_digest: str
    soft_rank_temperature: float
    soft_rank_scale: float
    relative_distance_scale: float
    coordinate_convention: str
    grid_align_corners: bool
    qk_operator_metadata_records: tuple[dict[str, Any], ...]
    qk_operator_metadata_digest: str
    qk_operator_metadata_ready: bool
    qk_atomic_content_records: tuple[dict[str, Any], ...]
    qk_atomic_content_digest: str
    qk_atomic_content_ready: bool


def attention_probability(attention: Any) -> Any:
    """只读取具有完整直接 Q/K 内容和算子身份的关系概率。"""

    if (
        not isinstance(attention, QKAttentionRelation)
        or attention.relation_source != DIRECT_QK_RELATION_SOURCE
    ):
        raise ValueError("注意力概率必须来自具有冻结身份的直接 Q/K 关系")
    metadata = attention.metadata
    if not isinstance(metadata, dict):
        raise ValueError("直接 Q/K 关系缺少可核验的算子元数据")
    layer_name = metadata.get("record_layer_name")
    token_indices = metadata.get("sampled_token_indices")
    if (
        not isinstance(layer_name, str)
        or not layer_name
        or not isinstance(token_indices, list)
    ):
        raise ValueError("直接 Q/K 关系缺少冻结层或 token 索引身份")
    operator_record = {
        "record_layer_name": layer_name,
        **{
            field_name: metadata.get(field_name)
            for field_name in QK_OPERATOR_METADATA_FIELD_NAMES
        },
    }
    token_index_tensor = _torch().tensor(token_indices, dtype=_torch().int64)
    atom_payload = {
        "record_layer_name": layer_name,
        "tensor_content_digest_version": TENSOR_CONTENT_DIGEST_VERSION,
        "sampled_query_content_sha256": metadata.get(
            "sampled_query_content_sha256"
        ),
        "sampled_key_content_sha256": metadata.get(
            "sampled_key_content_sha256"
        ),
        "centered_qk_logits_content_sha256": tensor_content_sha256(
            attention.centered_logits
        ),
        "qk_probabilities_content_sha256": tensor_content_sha256(
            attention.probabilities
        ),
        "sampled_token_indices_content_sha256": tensor_content_sha256(
            token_index_tensor
        ),
    }
    atom_record = {
        **atom_payload,
        "qk_atom_content_digest": build_stable_digest(atom_payload),
    }
    if (
        not qk_operator_metadata_records_ready(
            (operator_record,),
            (layer_name,),
        )
        or not qk_atomic_content_records_ready((atom_record,))
        or any(
            metadata.get(field_name) != atom_record[field_name]
            for field_name in (
                *QK_ATOMIC_CONTENT_SHA256_FIELDS,
                "qk_atom_content_digest",
            )
        )
    ):
        raise ValueError("直接 Q/K 关系的算子或原子内容身份不完整")
    return attention.probabilities


def _require_qk_record_identity(
    layer_name: str,
    attention: Any,
    token_indices: tuple[int, ...],
) -> Any:
    """绑定外层记录的层名和 token 索引到关系对象内部身份。"""

    probability = attention_probability(attention)
    metadata = attention.metadata
    if (
        metadata.get("record_layer_name") != layer_name
        or tuple(metadata.get("sampled_token_indices", ()))
        != tuple(token_indices)
    ):
        raise ValueError("Q/K 记录的外层层名或 token 索引与内部身份不一致")
    return probability


def _require_qk_record_identities(
    records: Iterable[tuple[str, Any, tuple[int, ...]]],
) -> tuple[tuple[str, Any, tuple[int, ...]], ...]:
    """集中复验一组有序 Q/K 记录的唯一层身份和内外绑定。"""

    resolved = tuple(records)
    layer_names = tuple(layer_name for layer_name, _, _ in resolved)
    if len(set(layer_names)) != len(layer_names):
        raise ValueError("Q/K 记录不得用同一层身份重复冒充多层关系")
    for layer_name, attention, token_indices in resolved:
        _require_qk_record_identity(
            layer_name,
            attention,
            tuple(token_indices),
        )
    return resolved


def centered_qk_logits(attention: Any) -> tuple[Any, str]:
    """读取真实中心化 Q/K logits, 拒绝从概率矩阵反推几何关系。"""

    attention_probability(attention)
    return attention.centered_logits, attention.relation_source


def public_token_grid_coordinates(token_indices: tuple[int, ...], device: Any) -> Any:
    """按角点中心落在 -1 和 1 的约定恢复归一化二维坐标."""

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


def build_qk_atomic_content_metadata(
    layer_name: str,
    sampled_query: Any,
    sampled_key: Any,
    centered_logits: Any,
    probabilities: Any,
    token_indices: tuple[int, ...],
) -> dict[str, Any]:
    """绑定一次真实 Q/K 计算的输入、输出与二维索引内容摘要."""

    torch = _torch()
    payload = {
        "record_layer_name": layer_name,
        "tensor_content_digest_version": TENSOR_CONTENT_DIGEST_VERSION,
        "sampled_query_content_sha256": tensor_content_sha256(sampled_query),
        "sampled_key_content_sha256": tensor_content_sha256(sampled_key),
        "centered_qk_logits_content_sha256": tensor_content_sha256(
            centered_logits
        ),
        "qk_probabilities_content_sha256": tensor_content_sha256(
            probabilities
        ),
        "sampled_token_indices_content_sha256": tensor_content_sha256(
            torch.tensor(token_indices, dtype=torch.int64)
        ),
    }
    return {
        **payload,
        "qk_atom_content_digest": build_stable_digest(payload),
    }


def qk_atomic_content_records_digest(
    records: Iterable[dict[str, Any]],
) -> str:
    """计算一组有序 Q/K 层原子内容记录的联合摘要."""

    resolved = tuple(dict(record) for record in records)
    return build_stable_digest({"qk_atomic_content_records": resolved})


def qk_atomic_content_records_ready(records: Any) -> bool:
    """复验每层 Q/K 原子字段和自摘要是否完整一致."""

    if (
        not isinstance(records, (list, tuple))
        or not records
        or any(not isinstance(record, dict) for record in records)
    ):
        return False
    resolved = tuple(dict(record) for record in records)
    for record in resolved:
        payload = {
            "record_layer_name": record.get("record_layer_name"),
            "tensor_content_digest_version": record.get(
                "tensor_content_digest_version"
            ),
            **{
                field_name: record.get(field_name)
                for field_name in QK_ATOMIC_CONTENT_SHA256_FIELDS
            },
        }
        if (
            not isinstance(payload["record_layer_name"], str)
            or not payload["record_layer_name"]
            or payload["tensor_content_digest_version"]
            != TENSOR_CONTENT_DIGEST_VERSION
            or not all(
                _is_sha256_hex(payload[field_name])
                for field_name in QK_ATOMIC_CONTENT_SHA256_FIELDS
            )
            or record.get("qk_atom_content_digest")
            != build_stable_digest(payload)
        ):
            return False
    return True


def qk_atomic_evaluation_records_digest(
    records: Iterable[dict[str, Any]],
    aggregate_field_name: str,
) -> str:
    """计算多次 Q/K 评价记录的有序联合摘要."""

    resolved = tuple(dict(record) for record in records)
    return build_stable_digest({aggregate_field_name: resolved})


def qk_atomic_evaluation_records_ready(
    records: Any,
    aggregate_digest: Any,
    *,
    aggregate_field_name: str,
    expected_roles: tuple[str, ...],
    expected_layer_names: tuple[str, ...],
    require_evaluation_identity: bool = False,
) -> bool:
    """复验多次 Q/K 评价的角色、逐层内容与联合摘要."""

    if (
        not isinstance(records, list)
        or len(records) != len(expected_roles)
        or any(not isinstance(record, dict) for record in records)
        or tuple(record.get("qk_evaluation_role") for record in records)
        != expected_roles
    ):
        return False
    for evaluation_record in records:
        atom_records = evaluation_record.get("qk_atomic_content_records")
        if (
            evaluation_record.get("qk_atomic_content_ready") is not True
            or not isinstance(atom_records, list)
            or any(not isinstance(record, dict) for record in atom_records)
            or tuple(
                atom_record.get("record_layer_name")
                for atom_record in atom_records
            )
            != expected_layer_names
            or not qk_atomic_content_records_ready(atom_records)
            or evaluation_record.get("qk_atomic_content_digest")
            != qk_atomic_content_records_digest(atom_records)
        ):
            return False
        if require_evaluation_identity and (
            not _is_sha256_hex(
                evaluation_record.get(
                    "evaluation_latent_content_sha256"
                )
            )
            or isinstance(evaluation_record.get("evaluation_score"), bool)
            or not isinstance(
                evaluation_record.get("evaluation_score"),
                (int, float),
            )
            or not math.isfinite(
                float(evaluation_record.get("evaluation_score"))
            )
        ):
            return False
    if require_evaluation_identity:
        for left_index, left_record in enumerate(records):
            for right_record in records[left_index + 1 :]:
                same_qk_content = (
                    left_record.get("qk_atomic_content_digest")
                    == right_record.get("qk_atomic_content_digest")
                )
                same_latent = (
                    left_record.get("evaluation_latent_content_sha256")
                    == right_record.get("evaluation_latent_content_sha256")
                )
                same_score = math.isclose(
                    float(left_record.get("evaluation_score")),
                    float(right_record.get("evaluation_score")),
                    rel_tol=ATTENTION_RELATION_NUMERICAL_EPSILON,
                    abs_tol=ATTENTION_RELATION_NUMERICAL_EPSILON,
                )
                if (same_qk_content and not same_score) or (
                    same_latent and (not same_qk_content or not same_score)
                ):
                    return False
    return aggregate_digest == qk_atomic_evaluation_records_digest(
        records,
        aggregate_field_name,
    )


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
    internal_layer_name = (
        attention.metadata.get("record_layer_name", "")
        if isinstance(attention, QKAttentionRelation)
        and isinstance(attention.metadata, dict)
        else ""
    )
    probability = _require_qk_record_identity(
        internal_layer_name,
        attention,
        tuple(token_indices),
    ).float()
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
    ).clamp_min(ATTENTION_RELATION_NUMERICAL_EPSILON)
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
        "coordinate_convention": ATTENTION_COORDINATE_CONVENTION,
        "grid_align_corners": ATTENTION_GRID_ALIGN_CORNERS,
    }
    return AttentionRelationDescriptor(
        values=values,
        token_indices=token_indices,
        component_names=ATTENTION_RELATION_COMPONENT_NAMES,
        soft_rank_temperature=ATTENTION_RELATION_SOFT_RANK_TEMPERATURE,
        soft_rank_scale=soft_rank_scale,
        relative_distance_scale=relative_distance_scale,
        relation_source=relation_source,
        coordinate_convention=ATTENTION_COORDINATE_CONVENTION,
        grid_align_corners=ATTENTION_GRID_ALIGN_CORNERS,
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
    heads_value = getattr(module, "heads", None)
    if (
        isinstance(heads_value, bool)
        or not isinstance(heads_value, int)
        or heads_value <= 0
    ):
        raise TypeError("注意力模块必须公开正整数 heads")
    heads = heads_value
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
        abs_tol=ATTENTION_RELATION_NUMERICAL_EPSILON,
    ):
        raise RuntimeError("注意力模块 scale 与 1 / sqrt(head_width) 不一致")
    per_head_logits = (
        query.float() @ key.float().transpose(-1, -2)
    ) * attention_scale
    centered_logits = per_head_logits - per_head_logits.mean(dim=-1, keepdim=True)
    attention = torch.softmax(per_head_logits, dim=-1)
    resolved_layer_name = layer_name or module.__class__.__qualname__
    centered_relation = centered_logits.mean(dim=1)
    probability_relation = attention.mean(dim=1)
    qk_atom_metadata = build_qk_atomic_content_metadata(
        resolved_layer_name,
        query,
        key,
        centered_relation,
        probability_relation,
        token_indices,
    )
    relation = QKAttentionRelation(
        centered_logits=centered_relation,
        probabilities=probability_relation,
        metadata={
            "module_layer_name": resolved_layer_name,
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
            "coordinate_convention": ATTENTION_COORDINATE_CONVENTION,
            "grid_align_corners": ATTENTION_GRID_ALIGN_CORNERS,
            **qk_atom_metadata,
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
    resolved_records = _require_qk_record_identities(records)
    if len(resolved_records) < 2:
        raise ValueError("注意力关系稳定图至少需要两个真实 Q/K 层")
    reference_indices = resolved_records[0][2]
    if any(token_indices != reference_indices for _, _, token_indices in resolved_records[1:]):
        raise ValueError("用于稳定度计算的 Q/K 层必须共享同一二维 token 抽样网格")
    sampled_side = int(round(math.sqrt(len(reference_indices))))
    if sampled_side * sampled_side != len(reference_indices):
        raise ValueError("注意力稳定度要求抽样 token 构成方形二维网格")
    normalized_rows = []
    for layer_name, attention, token_indices in resolved_records:
        matrix = _require_qk_record_identity(
            layer_name,
            attention,
            token_indices,
        ).float()
        centered = matrix - matrix.mean(dim=-1, keepdim=True)
        normalized_rows.append(
            functional.normalize(
                centered,
                dim=-1,
                eps=ATTENTION_RELATION_NUMERICAL_EPSILON,
            )
        )
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
        align_corners=ATTENTION_GRID_ALIGN_CORNERS,
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

    resolved_records = _require_qk_record_identities(records)
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
    resolved_records = _require_qk_record_identities(records)
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
    for layer_name, attention, token_indices in resolved_records:
        matrix = _require_qk_record_identity(
            layer_name,
            attention,
            token_indices,
        ).float()
        centered = matrix - matrix.mean(dim=-1, keepdim=True)
        normalized_rows.append(
            functional.normalize(
                centered,
                dim=-1,
                eps=ATTENTION_RELATION_NUMERICAL_EPSILON,
            )
        )
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
    centrality = centrality / centrality.sum().clamp_min(
        ATTENTION_RELATION_NUMERICAL_EPSILON
    )
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
                raise RuntimeError(
                    f"冻结注意力层 {layer_name} 没有提供可核验的 hidden_states Tensor"
                )
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


def keyed_relation_signs(
    attention: Any,
    key_material: str,
    layer_name: str,
    prg_version: str,
) -> Any:
    """为注意力关系生成零对角、近似均衡的密钥符号矩阵。"""

    torch = _torch()
    token_count = int(attention.shape[-1])
    random_values = build_keyed_uniform_tensor(
        (token_count, token_count),
        key_material,
        {
            "operator": "attention_relation_signs",
            "layer_name": layer_name,
            "token_count": token_count,
        },
        prg_version=prg_version,
    ).to(device=attention.device, dtype=torch.float32)
    signs = torch.where(random_values >= 0.5, 1.0, -1.0)
    signs = torch.triu(signs, diagonal=1)
    signs = signs + signs.transpose(0, 1)
    return signs


def keyed_attention_relation_projection(
    descriptor: AttentionRelationDescriptor,
    key_material: str,
    layer_name: str,
    prg_version: str,
    component_weights: Iterable[float] = ATTENTION_RELATION_COMPONENT_WEIGHTS,
) -> KeyedAttentionRelationProjection:
    """为四分量关系描述构造共享密钥图和冻结分量极性的投影。"""

    torch = _torch()
    resolved_component_weights = validate_attention_relation_component_weights(
        component_weights
    )
    base_signs = keyed_relation_signs(
        descriptor.values[..., 0],
        key_material,
        layer_name,
        prg_version,
    )
    prg_record = keyed_prg_protocol_record(prg_version)
    polarities = torch.tensor(
        ATTENTION_RELATION_COMPONENT_POLARITIES,
        device=descriptor.values.device,
        dtype=torch.float32,
    )
    values = base_signs.unsqueeze(-1) * polarities.reshape(1, 1, -1)
    projection_payload = {
        "component_identity_digest": descriptor.component_identity_digest,
        "component_names": descriptor.component_names,
        "component_weights": resolved_component_weights,
        "component_polarities": ATTENTION_RELATION_COMPONENT_POLARITIES,
        "key_material_digest": hashlib.sha256(
            key_material.encode("utf-8")
        ).hexdigest(),
        "layer_name": layer_name,
        "token_indices": descriptor.token_indices,
        "projection_rule": "shared_symmetric_pair_sign_with_component_polarity",
        "keyed_prg_version": prg_version,
        "keyed_prg_protocol_digest": prg_record[
            "keyed_prg_protocol_digest"
        ],
    }
    return KeyedAttentionRelationProjection(
        values=values,
        component_names=descriptor.component_names,
        component_weights=resolved_component_weights,
        component_polarities=ATTENTION_RELATION_COMPONENT_POLARITIES,
        component_identity_digest=descriptor.component_identity_digest,
        projection_digest=build_stable_digest(projection_payload),
    )


def build_attention_relation_graph_identity(
    records: Iterable[tuple[str, Any, tuple[int, ...]]],
    key_material: str,
    *,
    prg_version: str,
    component_weights: Iterable[float] = ATTENTION_RELATION_COMPONENT_WEIGHTS,
) -> AttentionRelationGraphIdentity:
    """重建多层关系分量和密钥投影的共同身份。"""

    resolved_records = _require_qk_record_identities(records)
    component_protocol = attention_relation_component_protocol(
        component_weights
    )
    resolved_component_weights = tuple(
        component_protocol["attention_relation_component_weights"]
    )
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
            prg_version,
            component_weights=resolved_component_weights,
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
    atomic_content_records = []
    operator_ready = all(
        isinstance(attention, QKAttentionRelation)
        and attention.relation_source == DIRECT_QK_RELATION_SOURCE
        for _, attention, _ in resolved_records
    )
    atomic_content_ready = True
    for layer_name, attention, token_indices in resolved_records:
        metadata = (
            dict(attention.metadata)
            if isinstance(attention, QKAttentionRelation)
            else {}
        )
        record = {
            "record_layer_name": layer_name,
            **{
                field_name: metadata.get(field_name)
                for field_name in QK_OPERATOR_METADATA_FIELD_NAMES
            },
        }
        operator_records.append(record)
        probability = attention_probability(attention)
        logits, _ = centered_qk_logits(attention)
        token_index_tensor = _torch().tensor(
            token_indices,
            dtype=_torch().int64,
        )
        atom_payload = {
            "record_layer_name": layer_name,
            "tensor_content_digest_version": TENSOR_CONTENT_DIGEST_VERSION,
            "sampled_query_content_sha256": str(
                metadata.get("sampled_query_content_sha256", "")
            ),
            "sampled_key_content_sha256": str(
                metadata.get("sampled_key_content_sha256", "")
            ),
            "centered_qk_logits_content_sha256": tensor_content_sha256(
                logits
            ),
            "qk_probabilities_content_sha256": tensor_content_sha256(
                probability
            ),
            "sampled_token_indices_content_sha256": tensor_content_sha256(
                token_index_tensor
            ),
        }
        qk_atom_content_digest = build_stable_digest(atom_payload)
        atom_record = {
            **atom_payload,
            "qk_atom_content_digest": qk_atom_content_digest,
        }
        atomic_content_records.append(atom_record)
        atomic_content_ready = atomic_content_ready and bool(
            isinstance(attention, QKAttentionRelation)
            and attention.relation_source == DIRECT_QK_RELATION_SOURCE
            and metadata.get("record_layer_name") == layer_name
            and _is_sha256_hex(atom_payload["sampled_query_content_sha256"])
            and _is_sha256_hex(atom_payload["sampled_key_content_sha256"])
            and metadata.get("centered_qk_logits_content_sha256")
            == atom_payload["centered_qk_logits_content_sha256"]
            and metadata.get("qk_probabilities_content_sha256")
            == atom_payload["qk_probabilities_content_sha256"]
            and metadata.get("sampled_token_indices_content_sha256")
            == atom_payload["sampled_token_indices_content_sha256"]
            and metadata.get("qk_atom_content_digest")
            == qk_atom_content_digest
        )
    qk_operator_metadata_records = tuple(operator_records)
    operator_ready = operator_ready and qk_operator_metadata_records_ready(
        qk_operator_metadata_records,
        (layer_name for layer_name, _, _ in resolved_records),
    )
    qk_operator_metadata_digest = qk_operator_metadata_records_digest(
        qk_operator_metadata_records
    )
    qk_atomic_content_records = tuple(atomic_content_records)
    qk_atomic_content_digest = qk_atomic_content_records_digest(
        qk_atomic_content_records
    )
    return AttentionRelationGraphIdentity(
        component_names=reference.component_names,
        active_component_names=tuple(
            component_protocol["attention_relation_active_component_names"]
        ),
        component_weights=resolved_component_weights,
        component_protocol_digest=str(
            component_protocol[
                "attention_relation_component_protocol_digest"
            ]
        ),
        relation_source=reference.relation_source,
        component_identity_digest=reference.component_identity_digest,
        keyed_projection_digest=projection_digest,
        soft_rank_temperature=reference.soft_rank_temperature,
        soft_rank_scale=reference.soft_rank_scale,
        relative_distance_scale=reference.relative_distance_scale,
        coordinate_convention=reference.coordinate_convention,
        grid_align_corners=reference.grid_align_corners,
        qk_operator_metadata_records=qk_operator_metadata_records,
        qk_operator_metadata_digest=qk_operator_metadata_digest,
        qk_operator_metadata_ready=operator_ready,
        qk_atomic_content_records=qk_atomic_content_records,
        qk_atomic_content_digest=qk_atomic_content_digest,
        qk_atomic_content_ready=(
            atomic_content_ready
            and qk_atomic_content_records_ready(qk_atomic_content_records)
        ),
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
    的数值尺度差异改变冻结的分量权重语义。
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
    numerical_epsilon = ATTENTION_RELATION_NUMERICAL_EPSILON
    safe_row_weight = row_weight.clamp_min(numerical_epsilon)
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
    energy_threshold = numerical_epsilon**2
    row_valid = (
        (row_weight.squeeze(-1) > 0.0).unsqueeze(-1)
        & (relation_energy > energy_threshold)
        & (projection_energy > energy_threshold)
    )
    safe_relation_energy = torch.where(
        relation_energy > energy_threshold,
        relation_energy,
        torch.ones_like(relation_energy),
    )
    safe_projection_energy = torch.where(
        projection_energy > energy_threshold,
        projection_energy,
        torch.ones_like(projection_energy),
    )
    safe_row_denominator = (
        safe_relation_energy * safe_projection_energy
    ).sqrt()
    row_scores = torch.where(
        row_valid,
        row_numerator / safe_row_denominator,
        torch.zeros_like(row_numerator),
    )
    valid_row_count = row_valid.sum(dim=-2).clamp_min(1)
    return (row_scores * row_valid.to(dtype=row_scores.dtype)).sum(
        dim=-2
    ) / valid_row_count


def combine_attention_relation_component_scores(
    component_scores: Any,
    component_weights: Iterable[float] = ATTENTION_RELATION_COMPONENT_WEIGHTS,
) -> Any:
    """按冻结权重协议组合逐分量关系分数."""

    torch = _torch()
    if component_scores.shape[-1] != len(ATTENTION_RELATION_COMPONENT_WEIGHTS):
        raise ValueError("component_scores 最后一维必须覆盖全部四个关系分量")
    resolved_weights = validate_attention_relation_component_weights(
        component_weights
    )
    weights = torch.tensor(
        resolved_weights,
        device=component_scores.device,
        dtype=component_scores.dtype,
    )
    return (component_scores * weights).sum(dim=-1)


def attention_geometry_component_scores(
    records: Iterable[tuple[str, Any, tuple[int, ...]]],
    key_material: str,
    stable_pair_weights: StableAttentionPairWeights,
    *,
    prg_version: str,
    component_weights: Iterable[float] = ATTENTION_RELATION_COMPONENT_WEIGHTS,
) -> Any:
    """使用同一 pair 权重计算多层四分量几何分数。"""

    torch = _torch()
    layer_component_scores = []
    for layer_name, attention, token_indices in _require_qk_record_identities(
        records
    ):
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
            prg_version,
            component_weights=component_weights,
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
    prg_version: str,
    stable_token_positions: tuple[int, ...] | None = None,
    stable_token_fraction: float = 0.5,
    unstable_pair_weight: float = 0.25,
    stable_pair_weights: StableAttentionPairWeights | None = None,
    component_weights: Iterable[float] = ATTENTION_RELATION_COMPONENT_WEIGHTS,
) -> Any:
    """计算稳定 token 加权的四分量 Q/K 密钥关系一致性分数。

    稳定 token 对使用权重1, 其余规则网格关系保留较小的冻结权重。保留完整
    网格可为盲检仿射注册提供连续二维采样支撑, 同时保证稳定 token 集合真实
    改变嵌入目标而不是只作为日志字段。四分量分别逐行中心化、归一化后按
    冻结非负权重组合, 防止不同物理量的数值尺度主导最终分数。
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
        prg_version=prg_version,
        component_weights=component_weights,
    )
    return combine_attention_relation_component_scores(
        component_scores,
        component_weights,
    )


@dataclass
class AttentionGeometryGradient:
    """保存真实 Q/K 目标在当前 latent 点的梯度。"""

    gradient: Any
    evaluation_latent_content_sha256: str
    score_before: float
    gradient_norm: float
    layer_names: tuple[str, ...]
    stable_token_positions: tuple[int, ...]
    stable_token_indices: tuple[int, ...]
    stable_token_selection_digest: str
    stable_pair_weight_identity_digest: str
    stable_pair_weight_realization_digest: str
    attention_relation_component_names: tuple[str, ...]
    attention_relation_active_component_names: tuple[str, ...]
    attention_relation_component_weights: tuple[float, ...]
    attention_relation_component_protocol_digest: str
    attention_relation_source: str
    attention_relation_component_identity_digest: str
    attention_relation_keyed_projection_digest: str
    attention_relation_soft_rank_temperature: float
    attention_relation_soft_rank_scale: float
    attention_relation_relative_distance_scale: float
    attention_relation_qk_operator_metadata_records: tuple[dict[str, Any], ...]
    attention_relation_qk_operator_metadata_digest: str
    attention_relation_qk_operator_metadata_ready: bool
    qk_atomic_content_records: tuple[dict[str, Any], ...]
    qk_atomic_content_digest: str
    qk_atomic_content_ready: bool
    stable_token_fraction: float
    unstable_pair_weight: float
    stable_pair_weights: StableAttentionPairWeights


def compute_attention_geometry_gradient(
    latent: Any,
    transformer_forward: Callable[[Any], Any],
    recorder: DifferentiableAttentionRecorder,
    key_material: str,
    *,
    prg_version: str,
    stable_token_fraction: float = 0.5,
    unstable_pair_weight: float = 0.25,
    stable_token_selection: StableAttentionTokenSelection | None = None,
    component_weights: Iterable[float] = ATTENTION_RELATION_COMPONENT_WEIGHTS,
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
            prg_version=prg_version,
            stable_pair_weights=pair_weights,
            component_weights=component_weights,
        )
        relation_identity = build_attention_relation_graph_identity(
            recorder.records,
            key_material,
            prg_version=prg_version,
            component_weights=component_weights,
        )
        if (
            relation_identity.relation_source != DIRECT_QK_RELATION_SOURCE
            or not relation_identity.qk_operator_metadata_ready
            or not relation_identity.qk_atomic_content_ready
        ):
            raise RuntimeError("正式注意力梯度必须绑定完整真实 Q/K 算子与内容")
        layer_names = tuple(dict.fromkeys(layer_name for layer_name, _, _ in recorder.records))
        gradient = torch.autograd.grad(score_before_tensor, differentiable_latent, retain_graph=False)[0]
    return AttentionGeometryGradient(
        gradient=gradient.detach(),
        evaluation_latent_content_sha256=tensor_content_sha256(
            differentiable_latent.detach()
        ),
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
        attention_relation_active_component_names=(
            relation_identity.active_component_names
        ),
        attention_relation_component_weights=(
            relation_identity.component_weights
        ),
        attention_relation_component_protocol_digest=(
            relation_identity.component_protocol_digest
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
        qk_atomic_content_records=(
            relation_identity.qk_atomic_content_records
        ),
        qk_atomic_content_digest=relation_identity.qk_atomic_content_digest,
        qk_atomic_content_ready=relation_identity.qk_atomic_content_ready,
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
    attention_relation_active_component_names: tuple[str, ...]
    attention_relation_component_weights: tuple[float, ...]
    attention_relation_component_protocol_digest: str
    attention_relation_source: str
    attention_relation_component_identity_digest: str
    attention_relation_keyed_projection_digest: str
    attention_relation_qk_operator_metadata_records: tuple[dict[str, Any], ...]
    attention_relation_qk_operator_metadata_digest: str
    attention_relation_qk_operator_metadata_ready: bool
    qk_atomic_evaluation_records: tuple[dict[str, Any], ...]
    qk_atomic_evaluation_digest: str
    qk_atomic_content_ready: bool
    unit_update_content_sha256: str
    update_content_sha256: str
    update_digest: str
    metadata: dict[str, Any]


def optimize_attention_geometry_update(
    latent: Any,
    transformer_forward: Callable[[Any], Any],
    recorder: DifferentiableAttentionRecorder,
    key_material: str,
    safe_subspace: JacobianNullSpaceResult,
    risk_bounded_update: RiskBoundedUpdate,
    precomputed_gradient: AttentionGeometryGradient,
    precomputed_content_base_gradient: AttentionGeometryGradient,
    *,
    prg_version: str,
    backtracking_factor: float = 0.5,
    maximum_backtracking_steps: int = 8,
    base_update: Any | None = None,
    stable_token_fraction: float = 0.5,
    unstable_pair_weight: float = 0.25,
    component_weights: Iterable[float] = ATTENTION_RELATION_COMPONENT_WEIGHTS,
) -> AttentionGeometryUpdate:
    """在固定内容更新基底上生成并验证真实注意力几何更新。

    ``risk_bounded_update`` 必须是 attention 分支风险硬包络从真实 Q/K 安全
    投影构造的单样本结果。内部每个回溯候选都通过同一对象重新物化, 因而
    接受候选和最终分支写回共享完全相同的单位方向、步长与 update Tensor。
    ``base_update`` 用于传入已经确定的 LF 与尾部载体更新。每个候选都先在
    float32 中合成 original latent、内容更新和注意力更新, 再只向实际 latent
    dtype 转换一次；回溯评分不得分别量化任一分支。

    ``precomputed_gradient`` 与 ``precomputed_content_base_gradient`` 分别绑定
    原始 latent 和实际量化后的内容基底 latent。该函数只消费这两个已经完成
    的真实 Q/K 求值, 不在优化内部重算或用数值近似替代其来源身份。
    """

    torch = _torch()
    if not isinstance(risk_bounded_update, RiskBoundedUpdate) or (
        risk_bounded_update.branch_name != "attention_geometry"
        or risk_bounded_update.update.shape != latent.shape
        or risk_bounded_update.applied_strength.numel() != 1
    ):
        raise ValueError("risk_bounded_update 必须是单样本 attention_geometry 分支")
    if (
        not math.isfinite(backtracking_factor)
        or not 0.0 < backtracking_factor < 1.0
    ):
        raise ValueError("backtracking_factor 必须为 (0, 1) 内的有限数")
    if (
        isinstance(maximum_backtracking_steps, bool)
        or not isinstance(maximum_backtracking_steps, int)
        or maximum_backtracking_steps < 0
    ):
        raise ValueError("maximum_backtracking_steps 必须为非负整数")
    original_evidence = precomputed_gradient
    for evidence in (
        precomputed_gradient,
        precomputed_content_base_gradient,
    ):
        evidence_gradient = torch.as_tensor(
            evidence.gradient,
            device=latent.device,
            dtype=torch.float32,
        )
        if (
            evidence_gradient.shape != latent.shape
            or not bool(torch.isfinite(evidence_gradient).all())
            or not math.isfinite(float(evidence.gradient_norm))
            or float(evidence.gradient_norm)
            <= risk_bounded_update.numerical_epsilon
            or evidence.attention_relation_source
            != DIRECT_QK_RELATION_SOURCE
            or evidence.attention_relation_qk_operator_metadata_ready
            is not True
            or evidence.qk_atomic_content_ready is not True
        ):
            raise RuntimeError("预计算注意力梯度缺少完整直接 Q/K 有限证据")
    original_latent_content_sha256 = tensor_content_sha256(
        latent.detach().float()
    )
    if (
        original_evidence.evaluation_latent_content_sha256
        != original_latent_content_sha256
    ):
        raise RuntimeError("预计算注意力梯度未绑定当前原始 latent")
    resolved_component_weights = validate_attention_relation_component_weights(
        component_weights
    )
    if (
        original_evidence.attention_relation_component_weights
        != resolved_component_weights
        or precomputed_content_base_gradient.attention_relation_component_weights
        != resolved_component_weights
    ):
        raise RuntimeError("预计算注意力梯度与当前四分量权重协议不一致")
    resolved_base_update = (
        torch.zeros_like(latent, dtype=torch.float32)
        if base_update is None
        else torch.as_tensor(
            base_update,
            device=latent.device,
            dtype=torch.float32,
        )
    )
    if resolved_base_update.shape != latent.shape:
        raise ValueError("base_update 必须与 latent 形状一致")
    _, content_base_latent, _ = compose_ordered_float32_update_once(
        original_latent=latent,
        branch_update_tensors={"lf_content": resolved_base_update},
        common_scale=1.0,
    )
    content_base_evidence = precomputed_content_base_gradient
    content_base_latent_content_sha256 = tensor_content_sha256(
        content_base_latent.detach().float()
    )
    if (
        content_base_evidence.evaluation_latent_content_sha256
        != content_base_latent_content_sha256
    ):
        raise RuntimeError("预计算注意力梯度未绑定当前内容基底 latent")
    if (
        content_base_evidence.layer_names != original_evidence.layer_names
        or content_base_evidence.stable_pair_weight_identity_digest
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
        if (
            float(projected_norm.item())
            <= risk_bounded_update.numerical_epsilon
        ):
            raise RuntimeError("注意力梯度在安全子空间中的投影为零")
        projected_unit_update = (projected / projected_norm).to(
            dtype=torch.float32
        )
        resolved_unit_update = torch.as_tensor(
            risk_bounded_update.unit_direction,
            device=latent.device,
            dtype=torch.float32,
        )
        if (
            resolved_unit_update.shape != latent.shape
            or not bool(torch.isfinite(resolved_unit_update).all())
            or not torch.allclose(
                resolved_unit_update,
                projected_unit_update,
                rtol=1e-5,
                atol=1e-6,
            )
        ):
            raise RuntimeError(
                "风险有界单位方向与真实 Q/K 安全投影方向不一致"
            )
        unit_update = resolved_unit_update.detach()
        maximum_update_strength = float(
            risk_bounded_update.applied_strength.item()
        )
        score_after_tensor = torch.tensor(
            score_before,
            device=latent.device,
            dtype=torch.float32,
        )
        accepted = False
        backtracking_step_count = 0
        update = torch.zeros_like(latent, dtype=torch.float32)
        for backtracking_step_count in range(maximum_backtracking_steps + 1):
            proposed_strength = maximum_update_strength * (
                float(backtracking_factor) ** backtracking_step_count
            )
            bounded_candidate = rescale_risk_bounded_update(
                risk_bounded_update,
                proposed_strength,
            )
            applied_strength = float(
                bounded_candidate.applied_strength.item()
            )
            update = bounded_candidate.update
            _, candidate, _ = compose_ordered_float32_update_once(
                original_latent=latent,
                branch_update_tensors={
                    "lf_content": resolved_base_update,
                    "attention_geometry": update,
                },
                common_scale=1.0,
            )
            recorder.clear()
            transformer_forward(candidate.detach())
            score_after_tensor = attention_geometry_score(
                recorder.records,
                key_material,
                prg_version=prg_version,
                stable_pair_weights=content_base_evidence.stable_pair_weights,
                component_weights=resolved_component_weights,
            )
            if (
                bool(torch.isfinite(score_after_tensor))
                and float(score_after_tensor.detach().item())
                > minimum_accepted_score
            ):
                accepted = True
                break
        if not accepted:
            raise RuntimeError("注意力几何更新在回溯搜索后仍未提高真实 Q/K 目标")
    accepted_relation_identity = build_attention_relation_graph_identity(
        recorder.records,
        key_material,
        prg_version=prg_version,
        component_weights=resolved_component_weights,
    )
    if not accepted_relation_identity.qk_atomic_content_ready:
        raise RuntimeError("接受的注意力候选缺少真实 Q/K 原子内容摘要")
    qk_atomic_evaluation_records = (
        {
            "qk_evaluation_role": "latent_before",
            "evaluation_latent_content_sha256": (
                original_latent_content_sha256
            ),
            "evaluation_score": score_before,
            "qk_atomic_content_records": list(
                original_evidence.qk_atomic_content_records
            ),
            "qk_atomic_content_digest": (
                original_evidence.qk_atomic_content_digest
            ),
            "qk_atomic_content_ready": (
                original_evidence.qk_atomic_content_ready
            ),
        },
        {
            "qk_evaluation_role": "optimization_content_base_latent",
            "evaluation_latent_content_sha256": (
                content_base_latent_content_sha256
            ),
            "evaluation_score": content_base_score,
            "qk_atomic_content_records": list(
                content_base_evidence.qk_atomic_content_records
            ),
            "qk_atomic_content_digest": (
                content_base_evidence.qk_atomic_content_digest
            ),
            "qk_atomic_content_ready": (
                content_base_evidence.qk_atomic_content_ready
            ),
        },
        {
            "qk_evaluation_role": "accepted_attention_candidate",
            "evaluation_latent_content_sha256": tensor_content_sha256(
                candidate.detach().float()
            ),
            "evaluation_score": float(score_after_tensor.detach().item()),
            "qk_atomic_content_records": list(
                accepted_relation_identity.qk_atomic_content_records
            ),
            "qk_atomic_content_digest": (
                accepted_relation_identity.qk_atomic_content_digest
            ),
            "qk_atomic_content_ready": (
                accepted_relation_identity.qk_atomic_content_ready
            ),
        },
    )
    qk_atomic_evaluation_digest = qk_atomic_evaluation_records_digest(
        qk_atomic_evaluation_records,
        "qk_atomic_evaluation_records",
    )
    score_after = float(score_after_tensor.detach().item())
    layer_names = content_base_evidence.layer_names
    payload = {
        "score_before": round(score_before, 12),
        "content_base_score": round(content_base_score, 12),
        "score_after": round(score_after, 12),
        "gradient_norm": round(float(gradient.norm().item()), 12),
        "projected_gradient_norm": round(float(projected.norm().item()), 12),
        "maximum_update_strength": round(maximum_update_strength, 12),
        "applied_update_strength": round(applied_strength, 12),
        "backtracking_factor": round(float(backtracking_factor), 12),
        "maximum_backtracking_steps": maximum_backtracking_steps,
        "backtracking_step_count": backtracking_step_count,
        "candidate_composition_protocol": (
            "ordered_float32_branch_sum_then_latent_add_single_cast_v1"
        ),
        "returned_update_dtype": "float32",
        "original_evaluation_latent_content_sha256": (
            original_latent_content_sha256
        ),
        "content_base_evaluation_latent_content_sha256": (
            content_base_latent_content_sha256
        ),
        "unit_update_content_sha256": tensor_content_sha256(unit_update),
        "update_content_sha256": tensor_content_sha256(update),
        "risk_bounded_update_consumed": True,
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
        "attention_relation_active_component_names": (
            content_base_evidence.attention_relation_active_component_names
        ),
        "attention_relation_component_weights": (
            content_base_evidence.attention_relation_component_weights
        ),
        "attention_relation_component_protocol_digest": (
            content_base_evidence.attention_relation_component_protocol_digest
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
        "qk_atomic_evaluation_digest": qk_atomic_evaluation_digest,
    }
    return AttentionGeometryUpdate(
        update=update.detach().to(dtype=torch.float32),
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
        attention_relation_active_component_names=(
            content_base_evidence.attention_relation_active_component_names
        ),
        attention_relation_component_weights=(
            content_base_evidence.attention_relation_component_weights
        ),
        attention_relation_component_protocol_digest=(
            content_base_evidence.attention_relation_component_protocol_digest
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
        qk_atomic_evaluation_records=qk_atomic_evaluation_records,
        qk_atomic_evaluation_digest=qk_atomic_evaluation_digest,
        qk_atomic_content_ready=all(
            bool(record["qk_atomic_content_ready"])
            for record in qk_atomic_evaluation_records
        ),
        unit_update_content_sha256=tensor_content_sha256(unit_update),
        update_content_sha256=tensor_content_sha256(update),
        update_digest=build_stable_digest(payload),
        metadata={
            "attention_source": "real_qk_projection",
            "gradient_source": "torch_autograd",
            "safe_projection": "jacobian_null_space",
            "update_search": "monotonic_backtracking",
            "optimization_base": "fixed_lf_and_tail_update",
            "verified_candidate": "actual_combined_latent",
            "maximum_update_strength": maximum_update_strength,
            "backtracking_factor": float(backtracking_factor),
            "maximum_backtracking_steps": maximum_backtracking_steps,
            "candidate_composition_protocol": (
                "ordered_float32_branch_sum_then_latent_add_single_cast_v1"
            ),
            "returned_update_dtype": "float32",
            "original_evaluation_latent_content_sha256": (
                original_latent_content_sha256
            ),
            "content_base_evaluation_latent_content_sha256": (
                content_base_latent_content_sha256
            ),
            "risk_bounded_update_consumed": True,
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
            "attention_relation_active_component_names": list(
                content_base_evidence.attention_relation_active_component_names
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
                content_base_evidence.attention_relation_component_weights
            ),
            "attention_relation_component_protocol_digest": (
                content_base_evidence.attention_relation_component_protocol_digest
            ),
        },
    )
