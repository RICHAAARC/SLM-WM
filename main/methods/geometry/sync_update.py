"""构造正式内容基底上的直接 Q/K 几何同步更新。"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable

from main.core.digest import build_stable_digest, tensor_content_sha256
from main.core.keyed_prg import require_supported_keyed_prg_version
from main.methods.carrier.content_update import ContentCarrierUpdateResult
from main.methods.geometry.differentiable_attention import (
    ATTENTION_COORDINATE_CONVENTION,
    ATTENTION_GRID_ALIGN_CORNERS,
    ATTENTION_RELATION_COMPONENT_NAMES,
    ATTENTION_RELATION_COMPONENT_WEIGHTS,
    DIRECT_QK_RELATION_SOURCE,
    FROZEN_SD35_ATTENTION_MODULE_NAMES,
    AttentionGeometryGradient,
    AttentionRelationGraphIdentity,
    DifferentiableAttentionRecorder,
    StableAttentionPairWeights,
    attention_geometry_score,
    build_attention_relation_graph_identity,
    compute_attention_geometry_gradient,
    qk_atomic_content_records_digest,
    qk_atomic_content_records_ready,
    qk_atomic_evaluation_records_digest,
    qk_atomic_evaluation_records_ready,
    qk_operator_metadata_records_digest,
    qk_operator_metadata_records_ready,
)


__all__ = [
    "GeometrySyncUpdate",
    "build_attention_geometry_sync_update",
]


_GEOMETRY_ENABLED_ROLES = (
    "full_dual_chain",
    "uniform_content_routing",
    "lf_only_content",
    "hf_tail_only_content",
)
_GEOMETRY_DISABLED_ROLES = (
    "content_chain_only",
    "geometry_recovery_without_embedded_sync",
)
_GEOMETRY_RELATIVE_STRENGTH = 0.0010
_DIRECTION_DENOMINATOR_EPSILON = 1.0e-12
_MAXIMUM_BACKTRACKING_INDEX = 8
_STABLE_TOKEN_FRACTION = 0.5
_UNSTABLE_PAIR_WEIGHT = 0.25
_QK_EVALUATION_ROLES = (
    "gradient_content_base_float32",
    "actual_dtype_content_baseline",
    "accepted_actual_dtype_candidate",
)
_QK_EVALUATION_RECORD_KEYS = {
    "qk_evaluation_role",
    "evaluation_latent_content_sha256",
    "evaluation_score",
    "qk_atomic_content_records",
    "qk_atomic_content_digest",
    "qk_atomic_content_ready",
}
_RELATION_TEMPLATE_IDENTITY_KEYS = {
    "formal_layer_names",
    "stable_token_positions",
    "stable_token_indices",
    "stable_token_fraction",
    "stable_token_selection_digest",
    "unstable_pair_weight",
    "stable_pair_weight_identity_digest",
    "stable_pair_weight_realization_digest",
    "attention_relation_component_names",
    "attention_relation_active_component_names",
    "attention_relation_component_weights",
    "attention_relation_component_protocol_digest",
    "attention_relation_source",
    "attention_relation_component_identity_digest",
    "attention_relation_keyed_projection_digest",
    "attention_relation_soft_rank_temperature",
    "attention_relation_soft_rank_scale",
    "attention_relation_relative_distance_scale",
    "attention_coordinate_convention",
    "attention_grid_align_corners",
    "attention_relation_qk_operator_metadata_digest",
    "prg_version",
}
_GEOMETRY_UPDATE_DIGEST_KEYS = {
    "geometry_update_rule",
    "z10_float32_content_sha256",
    "content_base_float32_content_sha256",
    "actual_dtype_content_baseline_content_sha256",
    "geometry_capacity_map_content_sha256",
    "geometry_gradient_content_sha256",
    "geometry_gradient_norm",
    "geometry_direction_content_sha256",
    "geometry_update_content_sha256",
    "actual_dtype_candidate_content_sha256",
    "actual_dtype_delta_content_sha256",
    "actual_dtype_delta_l2",
    "relative_strength",
    "accepted_scale",
    "backtracking_index",
    "l2_budget",
    "relation_score_before",
    "relation_score_after",
    "qk_atomic_records_digest",
    "relation_template_identity_digest",
    "prg_version",
}
_GEOMETRY_UPDATE_RULE = (
    "capacity_masked_direct_qk_monotonic_backtracking_v1"
)


def _torch() -> Any:
    """延迟导入 PyTorch，保持模块导入边界轻量。"""

    import torch

    return torch


def _require_sha256(value: Any, *, label: str) -> str:
    """要求规范小写 SHA-256。"""

    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} 必须为规范 SHA-256")
    return value


def _require_finite_number(value: Any, *, label: str) -> float:
    """读取非 bool 的有限 Python 数值。"""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} 必须为有限数值")
    resolved = float(value)
    if not math.isfinite(resolved):
        raise ValueError(f"{label} 必须为有限数值")
    return resolved


def _tensor_metadata(
    value: Any,
    *,
    label: str,
    expected_shape: tuple[int, int, int, int] | None = None,
    expected_device: Any | None = None,
    require_float32: bool = False,
) -> tuple[int, int, int, int]:
    """只读取 Tensor 元数据并闭合正式单样本边界。"""

    torch = _torch()
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{label} 必须为 Tensor")
    if not value.dtype.is_floating_point:
        raise TypeError(f"{label} 必须使用真实浮点 dtype")
    if value.device.type == "meta":
        raise ValueError(f"{label} 必须为已物化 Tensor")
    if value.ndim != 4:
        raise ValueError(f"{label} 必须具有 [1,C,H,W] 形状")
    shape = tuple(int(member) for member in value.shape)
    if shape[0] != 1 or any(member <= 0 for member in shape[1:]):
        raise ValueError(f"{label} 必须具有正尺寸 [1,C,H,W] 形状")
    if expected_shape is not None and shape != expected_shape:
        raise ValueError(f"{label} 形状与 z10 不一致")
    if expected_device is not None and value.device != expected_device:
        raise ValueError(f"{label} device 与 z10 不一致")
    if require_float32 and value.dtype != torch.float32:
        raise TypeError(f"{label} 必须为 float32")
    return shape


def _validate_static_inputs(
    current_scheduler_latent: Any,
    content_update: Any,
    transformer_forward: Any,
    recorder: Any,
    key_material: Any,
    prg_version: Any,
) -> tuple[int, int, int, int]:
    """在内容读取前闭合角色、元数据与调用身份。"""

    if type(content_update) is not ContentCarrierUpdateResult:
        raise TypeError("content_update 必须为精确 ContentCarrierUpdateResult")
    if type(content_update.method_role) is not str:
        raise TypeError("method_role 必须为精确 str")
    if content_update.method_role in _GEOMETRY_DISABLED_ROLES:
        raise ValueError("当前 method_role 禁止嵌入几何同步更新")
    if content_update.method_role not in _GEOMETRY_ENABLED_ROLES:
        raise ValueError("method_role 必须为登记的精确方法角色")

    z10_shape = _tensor_metadata(
        current_scheduler_latent,
        label="current_scheduler_latent",
    )
    z10_device = current_scheduler_latent.device
    for label in (
        "lf_update",
        "hf_tail_update",
        "content_only_latent_float32",
    ):
        _tensor_metadata(
            getattr(content_update, label),
            label=f"content_update.{label}",
            expected_shape=z10_shape,
            expected_device=z10_device,
            require_float32=True,
        )
    capacity_shape = _tensor_metadata(
        content_update.geometry_capacity_map,
        label="content_update.geometry_capacity_map",
        expected_device=z10_device,
        require_float32=True,
    )
    if capacity_shape != (z10_shape[0], 1, z10_shape[2], z10_shape[3]):
        raise ValueError(
            "geometry_capacity_map 必须具有与 z10 对齐的 [1,1,H,W] 形状"
        )
    if type(content_update.latent_l2) is not float:
        raise TypeError("content_update.latent_l2 必须为精确 float")
    for label in ("lf_nominal_strength", "hf_tail_nominal_strength"):
        if type(getattr(content_update, label)) is not float:
            raise TypeError(f"content_update.{label} 必须为精确 float")
    if type(key_material) is not str or not key_material:
        raise ValueError("key_material 必须为非空精确 str")
    if not callable(transformer_forward):
        raise TypeError("transformer_forward 必须可调用")
    if type(recorder) is not DifferentiableAttentionRecorder:
        raise TypeError("recorder 必须为精确 DifferentiableAttentionRecorder")
    if type(prg_version) is not str:
        raise TypeError("prg_version 必须为精确 str")
    require_supported_keyed_prg_version(prg_version)
    return z10_shape


def _validate_content_update_formula(
    current_scheduler_latent: Any,
    content_update: ContentCarrierUpdateResult,
) -> tuple[Any, Any]:
    """从权威 z10 重建范数和固定 LF→HF 内容基底。"""

    torch = _torch()
    tensors = (
        ("current_scheduler_latent", current_scheduler_latent),
        ("content_update.geometry_capacity_map", content_update.geometry_capacity_map),
        ("content_update.lf_update", content_update.lf_update),
        ("content_update.hf_tail_update", content_update.hf_tail_update),
        (
            "content_update.content_only_latent_float32",
            content_update.content_only_latent_float32,
        ),
    )
    for label, value in tensors:
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"{label} 必须全部有限")
    if not bool(
        (
            (content_update.geometry_capacity_map >= 0.0)
            & (content_update.geometry_capacity_map <= 1.0)
        ).all()
    ):
        raise ValueError("geometry_capacity_map 必须位于 [0,1]")

    z10_float32 = current_scheduler_latent.detach().to(dtype=torch.float32)
    latent_l2_tensor = torch.linalg.vector_norm(z10_float32.reshape(-1))
    if (
        not bool(torch.isfinite(latent_l2_tensor))
        or float(latent_l2_tensor.item()) <= 0.0
    ):
        raise ValueError("float32(z10) 必须具有正有限 L2 范数")
    if float(latent_l2_tensor.item()) != content_update.latent_l2:
        raise ValueError("content_update.latent_l2 未绑定权威 float32(z10) 范数")

    expected_content_only = (
        z10_float32
        + content_update.lf_update
        + content_update.hf_tail_update
    )
    if not torch.equal(
        content_update.content_only_latent_float32,
        expected_content_only,
    ):
        raise ValueError("content_only_latent_float32 未按固定 LF→HF 顺序闭合")
    for label in ("lf_nominal_strength", "hf_tail_nominal_strength"):
        value = _require_finite_number(
            getattr(content_update, label),
            label=f"content_update.{label}",
        )
        if value <= 0.0:
            raise ValueError(f"content_update.{label} 必须为正")
    return z10_float32, latent_l2_tensor


def _validate_pair_weights(
    evidence: AttentionGeometryGradient,
    pair_weights: Any,
) -> StableAttentionPairWeights:
    """闭合稳定 token 对权重身份与当前实现。"""

    if type(pair_weights) is not StableAttentionPairWeights:
        raise TypeError("stable_pair_weights 必须为精确 StableAttentionPairWeights")
    if (
        pair_weights.stable_token_positions != evidence.stable_token_positions
        or pair_weights.stable_token_indices != evidence.stable_token_indices
        or pair_weights.stable_token_fraction != evidence.stable_token_fraction
        or pair_weights.unstable_pair_weight != evidence.unstable_pair_weight
        or pair_weights.pair_weight_identity_digest
        != evidence.stable_pair_weight_identity_digest
        or pair_weights.pair_weight_realization_digest
        != evidence.stable_pair_weight_realization_digest
    ):
        raise ValueError("stable_pair_weights 与梯度证据身份不一致")
    _require_sha256(
        pair_weights.pair_weight_identity_digest,
        label="stable_pair_weight_identity_digest",
    )
    _require_sha256(
        pair_weights.pair_weight_realization_digest,
        label="stable_pair_weight_realization_digest",
    )
    if (
        type(pair_weights.token_weights) is not tuple
        or not pair_weights.token_weights
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in pair_weights.token_weights
        )
        or type(pair_weights.grid_token_indices) is not tuple
        or any(type(value) is not int for value in pair_weights.grid_token_indices)
        or type(pair_weights.coordinate_space) is not str
        or not pair_weights.coordinate_space
    ):
        raise ValueError("stable_pair_weights 数值实现不完整")
    return pair_weights


def _validate_gradient_evidence(
    evidence: Any,
    content_base: Any,
) -> tuple[AttentionGeometryGradient, Any, StableAttentionPairWeights]:
    """复验唯一 gradient forward 的直接 Q/K 科学证据。"""

    torch = _torch()
    if type(evidence) is not AttentionGeometryGradient:
        raise TypeError("gradient evidence 必须为精确 AttentionGeometryGradient")
    gradient = evidence.gradient
    if (
        not isinstance(gradient, torch.Tensor)
        or gradient.dtype != torch.float32
        or gradient.shape != content_base.shape
        or gradient.device != content_base.device
        or not bool(torch.isfinite(gradient).all())
    ):
        raise ValueError("geometry gradient 必须为同形同设备有限 float32 Tensor")
    expected_content_digest = tensor_content_sha256(content_base)
    if evidence.evaluation_latent_content_sha256 != expected_content_digest:
        raise ValueError("gradient evidence 未绑定 content-only float32 latent")
    gradient_norm = torch.linalg.vector_norm(gradient.reshape(-1))
    if (
        not bool(torch.isfinite(gradient_norm))
        or float(gradient_norm.item()) <= 0.0
        or float(gradient_norm.item()) != evidence.gradient_norm
    ):
        raise ValueError("gradient_norm 与实际 float32 gradient 不一致")
    if evidence.layer_names != FROZEN_SD35_ATTENTION_MODULE_NAMES:
        raise ValueError("gradient evidence 层名必须精确匹配冻结 SD3.5 层")
    if (
        type(evidence.stable_token_positions) is not tuple
        or type(evidence.stable_token_indices) is not tuple
        or not evidence.stable_token_positions
        or len(evidence.stable_token_positions) != len(evidence.stable_token_indices)
        or any(type(value) is not int for value in evidence.stable_token_positions)
        or any(type(value) is not int for value in evidence.stable_token_indices)
        or evidence.stable_token_fraction != _STABLE_TOKEN_FRACTION
        or evidence.unstable_pair_weight != _UNSTABLE_PAIR_WEIGHT
    ):
        raise ValueError("gradient evidence 稳定 token 选择身份不完整")
    _require_sha256(
        evidence.stable_token_selection_digest,
        label="stable_token_selection_digest",
    )
    pair_weights = _validate_pair_weights(evidence, evidence.stable_pair_weights)

    if (
        evidence.attention_relation_component_names
        != ATTENTION_RELATION_COMPONENT_NAMES
        or evidence.attention_relation_active_component_names
        != ATTENTION_RELATION_COMPONENT_NAMES
        or evidence.attention_relation_component_weights
        != ATTENTION_RELATION_COMPONENT_WEIGHTS
        or evidence.attention_relation_source != DIRECT_QK_RELATION_SOURCE
    ):
        raise ValueError("gradient evidence 四分量关系协议不一致")
    for label, value in (
        (
            "attention_relation_component_protocol_digest",
            evidence.attention_relation_component_protocol_digest,
        ),
        (
            "attention_relation_component_identity_digest",
            evidence.attention_relation_component_identity_digest,
        ),
        (
            "attention_relation_keyed_projection_digest",
            evidence.attention_relation_keyed_projection_digest,
        ),
        (
            "attention_relation_qk_operator_metadata_digest",
            evidence.attention_relation_qk_operator_metadata_digest,
        ),
        ("qk_atomic_content_digest", evidence.qk_atomic_content_digest),
    ):
        _require_sha256(value, label=label)
    for label, value in (
        (
            "attention_relation_soft_rank_temperature",
            evidence.attention_relation_soft_rank_temperature,
        ),
        (
            "attention_relation_soft_rank_scale",
            evidence.attention_relation_soft_rank_scale,
        ),
        (
            "attention_relation_relative_distance_scale",
            evidence.attention_relation_relative_distance_scale,
        ),
        ("score_before", evidence.score_before),
    ):
        _require_finite_number(value, label=label)
    if (
        evidence.attention_relation_qk_operator_metadata_ready is not True
        or type(evidence.attention_relation_qk_operator_metadata_records)
        is not tuple
        or not qk_operator_metadata_records_ready(
            evidence.attention_relation_qk_operator_metadata_records,
            FROZEN_SD35_ATTENTION_MODULE_NAMES,
        )
        or evidence.attention_relation_qk_operator_metadata_digest
        != qk_operator_metadata_records_digest(
            evidence.attention_relation_qk_operator_metadata_records
        )
    ):
        raise ValueError("gradient evidence Q/K 算子身份不完整")
    if (
        evidence.qk_atomic_content_ready is not True
        or type(evidence.qk_atomic_content_records) is not tuple
        or tuple(
            record.get("record_layer_name")
            for record in evidence.qk_atomic_content_records
        )
        != FROZEN_SD35_ATTENTION_MODULE_NAMES
        or not qk_atomic_content_records_ready(evidence.qk_atomic_content_records)
        or evidence.qk_atomic_content_digest
        != qk_atomic_content_records_digest(evidence.qk_atomic_content_records)
    ):
        raise ValueError("gradient evidence Q/K 原子内容身份不完整")
    return evidence, gradient_norm, pair_weights


def _validate_relation_identity(
    identity: Any,
    evidence: AttentionGeometryGradient,
) -> AttentionRelationGraphIdentity:
    """要求一次actual-dtype求值保持评分模板身份。"""

    if type(identity) is not AttentionRelationGraphIdentity:
        raise TypeError("relation identity 必须为精确 AttentionRelationGraphIdentity")
    expected_pairs = (
        (identity.component_names, evidence.attention_relation_component_names),
        (
            identity.active_component_names,
            evidence.attention_relation_active_component_names,
        ),
        (identity.component_weights, evidence.attention_relation_component_weights),
        (
            identity.component_protocol_digest,
            evidence.attention_relation_component_protocol_digest,
        ),
        (identity.relation_source, evidence.attention_relation_source),
        (
            identity.component_identity_digest,
            evidence.attention_relation_component_identity_digest,
        ),
        (
            identity.keyed_projection_digest,
            evidence.attention_relation_keyed_projection_digest,
        ),
        (
            identity.soft_rank_temperature,
            evidence.attention_relation_soft_rank_temperature,
        ),
        (identity.soft_rank_scale, evidence.attention_relation_soft_rank_scale),
        (
            identity.relative_distance_scale,
            evidence.attention_relation_relative_distance_scale,
        ),
        (
            identity.qk_operator_metadata_records,
            evidence.attention_relation_qk_operator_metadata_records,
        ),
        (
            identity.qk_operator_metadata_digest,
            evidence.attention_relation_qk_operator_metadata_digest,
        ),
        (
            identity.qk_operator_metadata_ready,
            evidence.attention_relation_qk_operator_metadata_ready,
        ),
    )
    if any(actual != expected for actual, expected in expected_pairs):
        raise ValueError("actual-dtype Q/K 求值与 gradient 评分模板身份不一致")
    if (
        identity.coordinate_convention != ATTENTION_COORDINATE_CONVENTION
        or identity.grid_align_corners is not ATTENTION_GRID_ALIGN_CORNERS
        or identity.qk_operator_metadata_ready is not True
        or identity.qk_atomic_content_ready is not True
        or tuple(
            record.get("record_layer_name")
            for record in identity.qk_atomic_content_records
        )
        != FROZEN_SD35_ATTENTION_MODULE_NAMES
        or not qk_atomic_content_records_ready(identity.qk_atomic_content_records)
        or identity.qk_atomic_content_digest
        != qk_atomic_content_records_digest(identity.qk_atomic_content_records)
    ):
        raise ValueError("actual-dtype Q/K 求值缺少完整直接原子身份")
    return identity


def _evaluate_actual_dtype_relation(
    latent: Any,
    transformer_forward: Callable[[Any], Any],
    recorder: DifferentiableAttentionRecorder,
    key_material: str,
    prg_version: str,
    evidence: AttentionGeometryGradient,
    pair_weights: StableAttentionPairWeights,
) -> tuple[float, AttentionRelationGraphIdentity]:
    """清空记录后执行一次actual-dtype Q/K身份与评分。"""

    torch = _torch()
    recorder.clear()
    transformer_forward(latent)
    records = tuple(recorder.records)
    if tuple(layer_name for layer_name, _, _ in records) != (
        FROZEN_SD35_ATTENTION_MODULE_NAMES
    ):
        raise ValueError("每次 Q/K evaluation 必须只产生冻结层有序记录")
    identity = _validate_relation_identity(
        build_attention_relation_graph_identity(
            records,
            key_material,
            prg_version=prg_version,
            component_weights=ATTENTION_RELATION_COMPONENT_WEIGHTS,
        ),
        evidence,
    )
    score_tensor = attention_geometry_score(
        records,
        key_material,
        prg_version=prg_version,
        stable_pair_weights=pair_weights,
        component_weights=ATTENTION_RELATION_COMPONENT_WEIGHTS,
    )
    if (
        not isinstance(score_tensor, torch.Tensor)
        or score_tensor.numel() != 1
        or not bool(torch.isfinite(score_tensor).all())
    ):
        raise ValueError("attention geometry score 必须为有限标量 Tensor")
    _validate_pair_weights(evidence, pair_weights)
    return float(score_tensor.detach().item()), identity


def _evaluation_record(
    role: str,
    latent_float32: Any,
    score: float,
    atomic_records: Any,
    atomic_digest: str,
) -> dict[str, Any]:
    """构造exact六键Q/K evaluation记录。"""

    record = {
        "qk_evaluation_role": role,
        "evaluation_latent_content_sha256": tensor_content_sha256(latent_float32),
        "evaluation_score": score,
        "qk_atomic_content_records": [dict(item) for item in atomic_records],
        "qk_atomic_content_digest": atomic_digest,
        "qk_atomic_content_ready": True,
    }
    if set(record) != _QK_EVALUATION_RECORD_KEYS:
        raise RuntimeError("Q/K evaluation record 字段集合漂移")
    return record


def _qk_evaluation_digest(
    records: list[dict[str, Any]],
) -> str:
    """以现有ready helper复验精确三角色联合证据。"""

    if (
        type(records) is not list
        or len(records) != 3
        or tuple(record.get("qk_evaluation_role") for record in records)
        != _QK_EVALUATION_ROLES
        or any(type(record) is not dict for record in records)
        or any(set(record) != _QK_EVALUATION_RECORD_KEYS for record in records)
    ):
        raise ValueError("Q/K evaluation records 必须为精确三角色六键list")
    digest = qk_atomic_evaluation_records_digest(
        records,
        "qk_atomic_evaluation_records",
    )
    if not qk_atomic_evaluation_records_ready(
        records,
        digest,
        aggregate_field_name="qk_atomic_evaluation_records",
        expected_roles=_QK_EVALUATION_ROLES,
        expected_layer_names=FROZEN_SD35_ATTENTION_MODULE_NAMES,
        require_evaluation_identity=True,
    ):
        raise ValueError("Q/K evaluation records 联合证据不完整")
    return digest


def _relation_template_digest(
    evidence: AttentionGeometryGradient,
    prg_version: str,
) -> str:
    """绑定不随evaluation latent变化的评分模板身份。"""

    payload = {
        "formal_layer_names": FROZEN_SD35_ATTENTION_MODULE_NAMES,
        "stable_token_positions": evidence.stable_token_positions,
        "stable_token_indices": evidence.stable_token_indices,
        "stable_token_fraction": evidence.stable_token_fraction,
        "stable_token_selection_digest": evidence.stable_token_selection_digest,
        "unstable_pair_weight": evidence.unstable_pair_weight,
        "stable_pair_weight_identity_digest": (
            evidence.stable_pair_weight_identity_digest
        ),
        "stable_pair_weight_realization_digest": (
            evidence.stable_pair_weight_realization_digest
        ),
        "attention_relation_component_names": (
            evidence.attention_relation_component_names
        ),
        "attention_relation_active_component_names": (
            evidence.attention_relation_active_component_names
        ),
        "attention_relation_component_weights": (
            evidence.attention_relation_component_weights
        ),
        "attention_relation_component_protocol_digest": (
            evidence.attention_relation_component_protocol_digest
        ),
        "attention_relation_source": evidence.attention_relation_source,
        "attention_relation_component_identity_digest": (
            evidence.attention_relation_component_identity_digest
        ),
        "attention_relation_keyed_projection_digest": (
            evidence.attention_relation_keyed_projection_digest
        ),
        "attention_relation_soft_rank_temperature": (
            evidence.attention_relation_soft_rank_temperature
        ),
        "attention_relation_soft_rank_scale": (
            evidence.attention_relation_soft_rank_scale
        ),
        "attention_relation_relative_distance_scale": (
            evidence.attention_relation_relative_distance_scale
        ),
        "attention_coordinate_convention": ATTENTION_COORDINATE_CONVENTION,
        "attention_grid_align_corners": ATTENTION_GRID_ALIGN_CORNERS,
        "attention_relation_qk_operator_metadata_digest": (
            evidence.attention_relation_qk_operator_metadata_digest
        ),
        "prg_version": prg_version,
    }
    if set(payload) != _RELATION_TEMPLATE_IDENTITY_KEYS:
        raise RuntimeError("relation template identity payload 字段集合漂移")
    return build_stable_digest(payload)


@dataclass(frozen=True)
class GeometrySyncUpdate:
    """保存正式直接Q/K几何同步分支的接受结果与身份。"""

    geometry_update: Any
    accepted_scale: float
    backtracking_index: int
    relative_strength: float
    l2_budget: float
    relation_score_before: float
    relation_score_after: float
    qk_atomic_records_digest: str
    relation_template_identity_digest: str
    geometry_update_digest: str


@dataclass(frozen=True)
class _GeometrySyncRuntimeEvidence:
    """Carry the exact scoring template needed by the post-write gate."""

    gradient_evidence: AttentionGeometryGradient
    stable_pair_weights: StableAttentionPairWeights
    relation_template_identity_digest: str
    prg_version: str


def _build_attention_geometry_sync_update_with_evidence(
    *,
    current_scheduler_latent: Any,
    content_update: ContentCarrierUpdateResult,
    transformer_forward: Callable[[Any], Any],
    recorder: DifferentiableAttentionRecorder,
    key_material: str,
    prg_version: str,
) -> tuple[GeometrySyncUpdate, _GeometrySyncRuntimeEvidence]:
    """在正式content-only基底构造单调改善的直接Q/K几何更新。"""

    torch = _torch()
    _validate_static_inputs(
        current_scheduler_latent,
        content_update,
        transformer_forward,
        recorder,
        key_material,
        prg_version,
    )
    z10_float32, latent_l2_tensor = _validate_content_update_formula(
        current_scheduler_latent,
        content_update,
    )
    content_base = content_update.content_only_latent_float32
    evidence, gradient_norm, pair_weights = _validate_gradient_evidence(
        compute_attention_geometry_gradient(
            content_base,
            transformer_forward,
            recorder,
            key_material,
            prg_version=prg_version,
            stable_token_fraction=_STABLE_TOKEN_FRACTION,
            unstable_pair_weight=_UNSTABLE_PAIR_WEIGHT,
            component_weights=ATTENTION_RELATION_COMPONENT_WEIGHTS,
        ),
        content_base,
    )

    direction = (
        content_update.geometry_capacity_map
        * evidence.gradient
        / (gradient_norm + z10_float32.new_tensor(_DIRECTION_DENOMINATOR_EPSILON))
    )
    direction_norm = torch.linalg.vector_norm(direction.reshape(-1))
    if (
        not bool(torch.isfinite(direction).all())
        or not bool(torch.isfinite(direction_norm))
        or float(direction_norm.item()) <= 0.0
    ):
        raise ValueError("geometry capacity mask 后方向必须具有正有限能量")

    baseline_actual = content_base.detach().to(
        dtype=current_scheduler_latent.dtype
    )
    baseline_score, baseline_identity = _evaluate_actual_dtype_relation(
        baseline_actual,
        transformer_forward,
        recorder,
        key_material,
        prg_version,
        evidence,
        pair_weights,
    )
    relative_strength_tensor = z10_float32.new_tensor(
        _GEOMETRY_RELATIVE_STRENGTH
    )
    nominal_budget = latent_l2_tensor * relative_strength_tensor

    accepted: tuple[int, Any, Any, Any, Any, float, AttentionRelationGraphIdentity] | None = None
    for backtracking_index in range(_MAXIMUM_BACKTRACKING_INDEX + 1):
        scale_tensor = z10_float32.new_tensor(2.0).pow(-backtracking_index)
        accepted_budget = nominal_budget * scale_tensor
        theoretical_update = accepted_budget * direction
        candidate_actual = (content_base + theoretical_update).to(
            dtype=current_scheduler_latent.dtype
        )
        actual_delta = (
            candidate_actual.detach().float()
            - baseline_actual.detach().float()
        )
        actual_delta_l2 = torch.linalg.vector_norm(actual_delta.reshape(-1))
        if (
            not bool(torch.isfinite(candidate_actual).all())
            or not bool(torch.isfinite(actual_delta).all())
            or not bool(torch.isfinite(actual_delta_l2))
            or float(actual_delta_l2.item()) <= 0.0
            or bool(actual_delta_l2 > accepted_budget)
        ):
            continue
        candidate_score, candidate_identity = _evaluate_actual_dtype_relation(
            candidate_actual,
            transformer_forward,
            recorder,
            key_material,
            prg_version,
            evidence,
            pair_weights,
        )
        if candidate_score > baseline_score:
            accepted = (
                backtracking_index,
                scale_tensor,
                accepted_budget,
                theoretical_update,
                actual_delta,
                candidate_score,
                candidate_identity,
            )
            break
    if accepted is None:
        raise ValueError("9项几何回溯均未产生actual-dtype严格关系分数改善")

    (
        backtracking_index,
        scale_tensor,
        accepted_budget,
        geometry_update,
        actual_delta,
        candidate_score,
        candidate_identity,
    ) = accepted
    candidate_actual = (content_base + geometry_update).to(
        dtype=current_scheduler_latent.dtype
    )
    actual_delta_l2 = torch.linalg.vector_norm(actual_delta.reshape(-1))

    evaluation_records = [
        _evaluation_record(
            _QK_EVALUATION_ROLES[0],
            content_base,
            float(evidence.score_before),
            evidence.qk_atomic_content_records,
            evidence.qk_atomic_content_digest,
        ),
        _evaluation_record(
            _QK_EVALUATION_ROLES[1],
            baseline_actual.detach().float(),
            baseline_score,
            baseline_identity.qk_atomic_content_records,
            baseline_identity.qk_atomic_content_digest,
        ),
        _evaluation_record(
            _QK_EVALUATION_ROLES[2],
            candidate_actual.detach().float(),
            candidate_score,
            candidate_identity.qk_atomic_content_records,
            candidate_identity.qk_atomic_content_digest,
        ),
    ]
    qk_records_digest = _qk_evaluation_digest(evaluation_records)
    relation_template_digest = _relation_template_digest(evidence, prg_version)
    update_payload = {
        "geometry_update_rule": _GEOMETRY_UPDATE_RULE,
        "z10_float32_content_sha256": tensor_content_sha256(z10_float32),
        "content_base_float32_content_sha256": tensor_content_sha256(content_base),
        "actual_dtype_content_baseline_content_sha256": tensor_content_sha256(
            baseline_actual.detach().float()
        ),
        "geometry_capacity_map_content_sha256": tensor_content_sha256(
            content_update.geometry_capacity_map
        ),
        "geometry_gradient_content_sha256": tensor_content_sha256(
            evidence.gradient
        ),
        "geometry_gradient_norm": float(gradient_norm.item()),
        "geometry_direction_content_sha256": tensor_content_sha256(direction),
        "geometry_update_content_sha256": tensor_content_sha256(geometry_update),
        "actual_dtype_candidate_content_sha256": tensor_content_sha256(
            candidate_actual.detach().float()
        ),
        "actual_dtype_delta_content_sha256": tensor_content_sha256(actual_delta),
        "actual_dtype_delta_l2": float(actual_delta_l2.item()),
        "relative_strength": float(relative_strength_tensor.item()),
        "accepted_scale": float(scale_tensor.item()),
        "backtracking_index": backtracking_index,
        "l2_budget": float(accepted_budget.item()),
        "relation_score_before": baseline_score,
        "relation_score_after": candidate_score,
        "qk_atomic_records_digest": qk_records_digest,
        "relation_template_identity_digest": relation_template_digest,
        "prg_version": prg_version,
    }
    if set(update_payload) != _GEOMETRY_UPDATE_DIGEST_KEYS:
        raise RuntimeError("geometry update digest payload 字段集合漂移")
    result = GeometrySyncUpdate(
        geometry_update=geometry_update,
        accepted_scale=float(scale_tensor.item()),
        backtracking_index=backtracking_index,
        relative_strength=float(relative_strength_tensor.item()),
        l2_budget=float(accepted_budget.item()),
        relation_score_before=baseline_score,
        relation_score_after=candidate_score,
        qk_atomic_records_digest=qk_records_digest,
        relation_template_identity_digest=relation_template_digest,
        geometry_update_digest=build_stable_digest(update_payload),
    )
    return result, _GeometrySyncRuntimeEvidence(
        gradient_evidence=evidence,
        stable_pair_weights=pair_weights,
        relation_template_identity_digest=relation_template_digest,
        prg_version=prg_version,
    )


def _evaluate_post_write_geometry_relation(
    *,
    written_latent: Any,
    transformer_forward: Callable[[Any], Any],
    recorder: DifferentiableAttentionRecorder,
    key_material: str,
    runtime_evidence: _GeometrySyncRuntimeEvidence,
) -> tuple[float, str]:
    """Evaluate the final single-write latent with the same Q/K template."""

    if type(runtime_evidence) is not _GeometrySyncRuntimeEvidence:
        raise TypeError("runtime_evidence must be exact geometry evidence")
    evidence = runtime_evidence.gradient_evidence
    pair_weights = _validate_pair_weights(
        evidence,
        runtime_evidence.stable_pair_weights,
    )
    score, identity = _evaluate_actual_dtype_relation(
        written_latent,
        transformer_forward,
        recorder,
        key_material,
        runtime_evidence.prg_version,
        evidence,
        pair_weights,
    )
    if runtime_evidence.relation_template_identity_digest != _relation_template_digest(
        evidence,
        runtime_evidence.prg_version,
    ):
        raise ValueError("post-write relation template identity drifted")
    return score, build_stable_digest(
        {
            "evaluation_role": "post_common_gamma_actual_dtype_write",
            "evaluation_latent_content_sha256": tensor_content_sha256(
                written_latent.detach().float()
            ),
            "evaluation_score": score,
            "qk_atomic_content_digest": identity.qk_atomic_content_digest,
            "relation_template_identity_digest": (
                runtime_evidence.relation_template_identity_digest
            ),
            "prg_version": runtime_evidence.prg_version,
        }
    )


def build_attention_geometry_sync_update(
    *,
    current_scheduler_latent: Any,
    content_update: ContentCarrierUpdateResult,
    transformer_forward: Callable[[Any], Any],
    recorder: DifferentiableAttentionRecorder,
    key_material: str,
    prg_version: str,
) -> GeometrySyncUpdate:
    """Build the public geometry update while keeping runtime evidence private."""

    result, _evidence = _build_attention_geometry_sync_update_with_evidence(
        current_scheduler_latent=current_scheduler_latent,
        content_update=content_update,
        transformer_forward=transformer_forward,
        recorder=recorder,
        key_material=key_material,
        prg_version=prg_version,
    )
    return result
