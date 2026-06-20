"""实现 SLM-WM 的纯算法原语。

本模块只使用 Python 标准库和 `main/core` 中的稳定摘要能力, 不接入 SD3、
SD3.5、Colab、Drive、Notebook 或真实 Self-Attention 运行时。该实现的定位是
算法原语的 synthetic / tensor 闭环, 用于冻结方法数据流和统计边界。

通用工程写法: 重复的向量形状校验集中在 dataclass 构造和少量私有辅助函数中。
项目特定写法: LF、HF 与 attention synthetic stub 均从同一语义安全子空间导出,
几何链只允许辅助 rescue, attestation 只影响 final-level。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
from numbers import Real
from typing import Any, Iterable, Sequence

from main.core.digest import build_stable_digest

NumberLike = int | float
TensorLike = NumberLike | Sequence["TensorLike"]


def _is_sequence(value: object) -> bool:
    """判断对象是否为可展开序列, 字符串和字节串不视为 tensor。"""
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _as_vector(values: TensorLike | Iterable[NumberLike], field_name: str) -> tuple[float, ...]:
    """将 synthetic tensor 展平为一维 float 向量。

    该函数是本模块的统一输入边界。业务函数只调用它一次, 避免在每个算法步骤中
    重复书写相同的类型检查和错误信息。
    """
    flattened: list[float] = []

    def visit(candidate: object) -> None:
        if isinstance(candidate, Real):
            flattened.append(float(candidate))
            return
        if _is_sequence(candidate):
            for item in candidate:
                visit(item)
            return
        raise TypeError(f"{field_name} 必须是数值或数值序列")

    visit(values)
    if not flattened:
        raise ValueError(f"{field_name} 不得为空")
    if any(not math.isfinite(value) for value in flattened):
        raise ValueError(f"{field_name} 必须只包含有限数值")
    return tuple(flattened)


def _ensure_equal_lengths(field_values: dict[str, Sequence[Any]]) -> int:
    """集中校验多个字段长度一致, 并返回共同长度。"""
    lengths = {field_name: len(values) for field_name, values in field_values.items()}
    unique_lengths = set(lengths.values())
    if len(unique_lengths) != 1:
        raise ValueError(f"字段长度不一致: {lengths}")
    return unique_lengths.pop()


def _clip(value: float, lower: float, upper: float) -> float:
    """将数值裁剪到闭区间。"""
    return min(max(value, lower), upper)


def _vector_norm(values: Sequence[float]) -> float:
    """计算向量二范数。"""
    return math.sqrt(sum(value * value for value in values))


def _correlation(left: Sequence[float], right: Sequence[float]) -> float:
    """计算归一化相关分数, 零向量时返回 0。"""
    _ensure_equal_lengths({"left": left, "right": right})
    denominator = _vector_norm(left) * _vector_norm(right)
    if denominator == 0:
        return 0.0
    return sum(left_value * right_value for left_value, right_value in zip(left, right)) / denominator


def _resample_vector(values: Sequence[float], target_length: int) -> tuple[float, ...]:
    """用最近邻策略把一维 mask 投影到目标 latent 长度。"""
    if target_length <= 0:
        raise ValueError("target_length 必须为正数")
    if len(values) == target_length:
        return tuple(values)
    if len(values) == 1:
        return tuple(values[0] for _ in range(target_length))
    source_last = len(values) - 1
    target_last = target_length - 1
    return tuple(values[round(index * source_last / target_last)] for index in range(target_length))


def _stable_unit_values(parts: Sequence[str], count: int) -> tuple[float, ...]:
    """根据密钥材料生成稳定伪随机 [-1, 1] 模板。"""
    if count <= 0:
        raise ValueError("count 必须为正数")
    values: list[float] = []
    for index in range(count):
        payload = "|".join((*parts, str(index))).encode("utf-8")
        digest = hashlib.sha256(payload).hexdigest()
        integer = int(digest[:16], 16)
        unit = integer / float(0xFFFFFFFFFFFFFFFF)
        values.append(unit * 2.0 - 1.0)
    return tuple(values)


def _quantile(values: Sequence[float], quantile: float) -> float:
    """计算简单确定性分位数, 用于 synthetic HF tail truncation。"""
    if not values:
        raise ValueError("values 不得为空")
    bounded_quantile = _clip(quantile, 0.0, 1.0)
    ordered = sorted(values)
    index = int(math.floor(bounded_quantile * (len(ordered) - 1)))
    return ordered[index]


def _one_hot_basis(length: int, indices: Sequence[int], weights: Sequence[float]) -> tuple[tuple[float, ...], ...]:
    """根据索引构造加权 one-hot 基底。"""
    basis_rows: list[tuple[float, ...]] = []
    for index in indices:
        row = [0.0 for _ in range(length)]
        row[index] = weights[index]
        basis_rows.append(tuple(row))
    return tuple(basis_rows)


def _basis_weighted_update(
    basis: Sequence[Sequence[float]],
    coefficients: Sequence[float],
    strength: float,
) -> tuple[float, ...]:
    """把基底和模板系数组合成 latent update。"""
    _ensure_equal_lengths({"basis": basis, "coefficients": coefficients})
    if not basis:
        return ()
    width = _ensure_equal_lengths({f"basis_{index}": row for index, row in enumerate(basis)})
    update = [0.0 for _ in range(width)]
    for row, coefficient in zip(basis, coefficients):
        for index, value in enumerate(row):
            update[index] += strength * coefficient * value
    return tuple(update)


def _basis_digest(basis: Sequence[Sequence[float]]) -> str:
    """生成基底稳定摘要。"""
    rounded_basis = [[round(value, 12) for value in row] for row in basis]
    return build_stable_digest(rounded_basis)


@dataclass(frozen=True)
class RiskFieldParameters:
    """语义风险场参数。

    参数校验集中在构造时, 后续业务函数只读取已合法的参数对象。
    """

    eta_saliency: float = 0.35
    eta_semantic: float = 0.25
    eta_texture: float = 0.20
    eta_stability: float = 0.20
    budget_min: float = 0.05
    budget_max: float = 1.00
    budget_gain: float = 0.50
    texture_threshold: float = 0.50
    risk_threshold: float = 0.55
    stability_threshold: float = 0.45
    attention_threshold: float = 0.50

    def __post_init__(self) -> None:
        """验证阈值与预算边界。"""
        if self.budget_min < 0 or self.budget_max <= self.budget_min:
            raise ValueError("budget_min 与 budget_max 必须形成有效预算区间")
        threshold_values = (
            self.texture_threshold,
            self.risk_threshold,
            self.stability_threshold,
            self.attention_threshold,
        )
        if any(value < 0.0 or value > 1.0 for value in threshold_values):
            raise ValueError("语义路由阈值必须位于 [0, 1]")


@dataclass(frozen=True)
class SemanticRiskField:
    """语义风险场与 LF/HF/attention 路由 mask。"""

    risk_values: tuple[float, ...]
    budget_values: tuple[float, ...]
    lf_mask: tuple[bool, ...]
    hf_mask: tuple[bool, ...]
    attention_mask: tuple[bool, ...]
    route_digest: str
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        """集中校验风险场各路由长度一致。"""
        _ensure_equal_lengths(
            {
                "risk_values": self.risk_values,
                "budget_values": self.budget_values,
                "lf_mask": self.lf_mask,
                "hf_mask": self.hf_mask,
                "attention_mask": self.attention_mask,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        """转为可写入 JSON 的字典。"""
        return asdict(self)


@dataclass(frozen=True)
class LatentMaskProjection:
    """图像域或外部 mask 投影到 latent 长度后的结果。"""

    mask_values: tuple[float, ...]
    masked_latent_values: tuple[float, ...]
    source_length: int
    target_length: int
    projection_digest: str
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        """集中校验 mask 与 latent 长度。"""
        _ensure_equal_lengths({"mask_values": self.mask_values, "masked_latent_values": self.masked_latent_values})
        if self.target_length != len(self.mask_values):
            raise ValueError("target_length 必须等于投影后 mask 长度")

    def to_dict(self) -> dict[str, Any]:
        """转为可写入 JSON 的字典。"""
        return asdict(self)


@dataclass(frozen=True)
class SafeBasisEstimate:
    """语义条件安全子空间的 synthetic 基底估计。"""

    safe_basis: tuple[tuple[float, ...], ...]
    lf_basis: tuple[tuple[float, ...], ...]
    hf_basis: tuple[tuple[float, ...], ...]
    attention_basis: tuple[tuple[float, ...], ...]
    selected_indices: tuple[int, ...]
    basis_digest: str
    route_digest: str
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        """校验所有基底行共享同一 latent 宽度。"""
        rows = [*self.safe_basis, *self.lf_basis, *self.hf_basis, *self.attention_basis]
        if rows:
            _ensure_equal_lengths({f"basis_row_{index}": row for index, row in enumerate(rows)})

    def to_dict(self) -> dict[str, Any]:
        """转为可写入 JSON 的字典。"""
        return asdict(self)


@dataclass(frozen=True)
class CarrierPrimitive:
    """LF、HF 或 attention synthetic stub 载体。"""

    carrier_id: str
    carrier_family: str
    frequency_band: str
    key_digest: str
    template_values: tuple[float, ...]
    update_values: tuple[float, ...]
    embedding_strength: float
    carrier_digest: str
    tail_threshold: float | None
    retained_fraction: float
    synthetic_stub: bool
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        """校验载体模板和 update 的最小边界。"""
        if not self.template_values:
            raise ValueError("template_values 不得为空")
        if not self.update_values:
            raise ValueError("update_values 不得为空")
        if self.retained_fraction < 0.0 or self.retained_fraction > 1.0:
            raise ValueError("retained_fraction 必须位于 [0, 1]")

    def to_dict(self) -> dict[str, Any]:
        """转为可写入 JSON 的字典。"""
        return asdict(self)


@dataclass(frozen=True)
class LatentUpdateComposition:
    """三个载体分量相加后的 latent update。"""

    lf_update_values: tuple[float, ...]
    hf_update_values: tuple[float, ...]
    attention_update_values: tuple[float, ...]
    combined_update_values: tuple[float, ...]
    update_digest: str
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        """校验三个 update 分量长度一致。"""
        _ensure_equal_lengths(
            {
                "lf_update_values": self.lf_update_values,
                "hf_update_values": self.hf_update_values,
                "attention_update_values": self.attention_update_values,
                "combined_update_values": self.combined_update_values,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        """转为可写入 JSON 的字典。"""
        return asdict(self)


@dataclass(frozen=True)
class ContentScoreResult:
    """LF/HF 融合后的内容分数。"""

    lf_score: float
    hf_score: float
    content_score: float
    lambda_lf: float
    lambda_hf: float
    used_independent_branch_vote: bool
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        """校验融合权重边界。"""
        if not math.isclose(self.lambda_lf + self.lambda_hf, 1.0, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError("lambda_lf 与 lambda_hf 之和必须为 1")
        if self.lambda_lf <= self.lambda_hf:
            raise ValueError("lambda_lf 必须大于 lambda_hf")

    def to_dict(self) -> dict[str, Any]:
        """转为可写入 JSON 的字典。"""
        return asdict(self)


@dataclass(frozen=True)
class GeometryReliabilityResult:
    """几何参考系恢复可靠性结果。

    `direct_positive_decision` 固定为 False, 表示几何链不能直接给出 positive。
    """

    registration_confidence: float
    anchor_inlier_ratio: float
    recovered_sync_consistency: float
    alignment_residual: float
    geometry_reliable: bool
    direct_positive_decision: bool
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """转为可写入 JSON 的字典。"""
        return asdict(self)


@dataclass(frozen=True)
class EvidenceDecisionResult:
    """evidence-level、attestation 和 final-level 判定结果。"""

    raw_content_score: float
    aligned_content_score: float
    content_threshold: float
    raw_content_margin: float
    aligned_content_margin: float
    fail_reason: str
    rescue_margin_low: float
    positive_by_content: bool
    rescue_eligible: bool
    rescue_applied: bool
    evidence_level: bool
    attestation_pass: bool
    final_level: bool
    final_label: str
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """转为可写入 JSON 的字典。"""
        return asdict(self)


def build_semantic_risk_field(
    semantic_values: TensorLike | Iterable[NumberLike],
    texture_values: TensorLike | Iterable[NumberLike],
    stability_values: TensorLike | Iterable[NumberLike],
    saliency_values: TensorLike | Iterable[NumberLike],
    attention_stability_values: TensorLike | Iterable[NumberLike] | None = None,
    parameters: RiskFieldParameters | None = None,
) -> SemanticRiskField:
    """构造语义风险场、承载预算和三个路由 mask。"""
    config = parameters or RiskFieldParameters()
    semantic_vector = _as_vector(semantic_values, "semantic_values")
    texture_vector = _as_vector(texture_values, "texture_values")
    stability_vector = _as_vector(stability_values, "stability_values")
    saliency_vector = _as_vector(saliency_values, "saliency_values")
    if attention_stability_values is None:
        attention_vector = stability_vector
    else:
        attention_vector = _as_vector(attention_stability_values, "attention_stability_values")
    length = _ensure_equal_lengths(
        {
            "semantic_values": semantic_vector,
            "texture_values": texture_vector,
            "stability_values": stability_vector,
            "saliency_values": saliency_vector,
            "attention_stability_values": attention_vector,
        }
    )

    risk_values: list[float] = []
    budget_values: list[float] = []
    lf_mask: list[bool] = []
    hf_mask: list[bool] = []
    attention_mask: list[bool] = []
    for index in range(length):
        risk = (
            config.eta_saliency * saliency_vector[index]
            + config.eta_semantic * semantic_vector[index]
            - config.eta_texture * texture_vector[index]
            - config.eta_stability * stability_vector[index]
        )
        bounded_risk = _clip(risk, 0.0, 1.0)
        budget = _clip(
            config.budget_min + config.budget_gain * (1.0 - bounded_risk),
            config.budget_min,
            config.budget_max,
        )
        risk_values.append(bounded_risk)
        budget_values.append(budget)
        lf_mask.append(texture_vector[index] < config.texture_threshold and bounded_risk < config.risk_threshold)
        hf_mask.append(texture_vector[index] >= config.texture_threshold and stability_vector[index] >= config.stability_threshold)
        attention_mask.append(
            attention_vector[index] >= config.attention_threshold and stability_vector[index] >= config.stability_threshold
        )

    route_payload = {
        "lf_mask": lf_mask,
        "hf_mask": hf_mask,
        "attention_mask": attention_mask,
        "risk_values": [round(value, 12) for value in risk_values],
    }
    return SemanticRiskField(
        risk_values=tuple(risk_values),
        budget_values=tuple(budget_values),
        lf_mask=tuple(lf_mask),
        hf_mask=tuple(hf_mask),
        attention_mask=tuple(attention_mask),
        route_digest=build_stable_digest(route_payload),
        metadata={"primitive_name": "semantic_risk_field", "field_length": length},
    )


def project_latent_mask(
    latent_values: TensorLike | Iterable[NumberLike],
    mask_values: TensorLike | Iterable[NumberLike],
) -> LatentMaskProjection:
    """将外部 mask 投影到 latent 向量长度并应用到 latent。"""
    latent_vector = _as_vector(latent_values, "latent_values")
    source_mask = _as_vector(mask_values, "mask_values")
    projected_mask = tuple(_clip(value, 0.0, 1.0) for value in _resample_vector(source_mask, len(latent_vector)))
    masked_latent = tuple(value * mask for value, mask in zip(latent_vector, projected_mask))
    return LatentMaskProjection(
        mask_values=projected_mask,
        masked_latent_values=masked_latent,
        source_length=len(source_mask),
        target_length=len(latent_vector),
        projection_digest=build_stable_digest(
            {
                "mask_values": [round(value, 12) for value in projected_mask],
                "masked_latent_values": [round(value, 12) for value in masked_latent],
            }
        ),
        metadata={"primitive_name": "latent_mask_projection"},
    )


def estimate_safe_basis(
    latent_values: TensorLike | Iterable[NumberLike],
    mask_projection: LatentMaskProjection,
    risk_field: SemanticRiskField,
    basis_rank: int = 4,
) -> SafeBasisEstimate:
    """估计 semantic-conditioned safe basis 的 synthetic one-hot 近似。"""
    latent_vector = _as_vector(latent_values, "latent_values")
    length = _ensure_equal_lengths(
        {
            "latent_values": latent_vector,
            "mask_values": mask_projection.mask_values,
            "risk_values": risk_field.risk_values,
        }
    )
    rank = max(1, min(basis_rank, length))
    response_scores = []
    for index in range(length):
        response = risk_field.risk_values[index] + (1.0 - risk_field.budget_values[index])
        mask_bonus = 1.0 - mask_projection.mask_values[index]
        response_scores.append((response + 0.05 * mask_bonus, index))
    selected_indices = tuple(index for _, index in sorted(response_scores)[:rank])

    def route_indices(route_mask: Sequence[bool]) -> tuple[int, ...]:
        routed_scores = [(score, index) for score, index in response_scores if route_mask[index]]
        routed = tuple(index for _, index in sorted(routed_scores)[:rank])
        return routed or selected_indices

    safe_basis = _one_hot_basis(length, selected_indices, risk_field.budget_values)
    lf_basis = _one_hot_basis(length, route_indices(risk_field.lf_mask), risk_field.budget_values)
    hf_basis = _one_hot_basis(length, route_indices(risk_field.hf_mask), risk_field.budget_values)
    attention_basis = _one_hot_basis(length, route_indices(risk_field.attention_mask), risk_field.budget_values)
    basis_payload = {
        "safe_basis": safe_basis,
        "lf_basis": lf_basis,
        "hf_basis": hf_basis,
        "attention_basis": attention_basis,
    }
    return SafeBasisEstimate(
        safe_basis=safe_basis,
        lf_basis=lf_basis,
        hf_basis=hf_basis,
        attention_basis=attention_basis,
        selected_indices=selected_indices,
        basis_digest=_basis_digest(safe_basis),
        route_digest=build_stable_digest({"route_digest": risk_field.route_digest, "basis_payload": basis_payload}),
        metadata={"primitive_name": "safe_basis_estimate", "basis_rank": rank},
    )


def derive_lf_carrier(
    safe_basis: SafeBasisEstimate,
    key: str,
    event_digest: str,
    embedding_strength: float = 0.20,
) -> CarrierPrimitive:
    """从 LF 子基底导出低频主证据载体。"""
    template = _stable_unit_values((key, event_digest, safe_basis.basis_digest, safe_basis.route_digest, "lf"), len(safe_basis.lf_basis))
    if len(template) == 1:
        encoded_template = template
    else:
        encoded_template = tuple(
            (template[max(index - 1, 0)] + template[index] + template[min(index + 1, len(template) - 1)]) / 3.0
            for index in range(len(template))
        )
    update = _basis_weighted_update(safe_basis.lf_basis, encoded_template, embedding_strength)
    carrier_payload = {
        "carrier_family": "latent_frequency",
        "frequency_band": "low_frequency",
        "template_values": [round(value, 12) for value in encoded_template],
        "update_values": [round(value, 12) for value in update],
    }
    return CarrierPrimitive(
        carrier_id=f"lf_{build_stable_digest(carrier_payload)[:12]}",
        carrier_family="latent_frequency",
        frequency_band="low_frequency",
        key_digest=build_stable_digest({"key": key, "event_digest": event_digest, "branch": "lf"}),
        template_values=encoded_template,
        update_values=update,
        embedding_strength=embedding_strength,
        carrier_digest=build_stable_digest(carrier_payload),
        tail_threshold=None,
        retained_fraction=1.0,
        synthetic_stub=False,
        metadata={"primitive_name": "lf_carrier"},
    )


def derive_hf_carrier(
    safe_basis: SafeBasisEstimate,
    risk_field: SemanticRiskField,
    key: str,
    event_digest: str,
    embedding_strength: float = 0.12,
    tail_fraction: float = 0.50,
) -> CarrierPrimitive:
    """从 HF 子基底导出带 tail truncation 的高频补充载体。"""
    template = _stable_unit_values((key, event_digest, safe_basis.basis_digest, safe_basis.route_digest, "hf"), len(safe_basis.hf_basis))
    weights = []
    for row in safe_basis.hf_basis:
        active_index = max(range(len(row)), key=lambda index: abs(row[index]))
        weights.append(risk_field.budget_values[active_index] + (1.0 if risk_field.hf_mask[active_index] else 0.0))
    tail_scores = [abs(value) * weight for value, weight in zip(template, weights)]
    if tail_fraction >= 1.0:
        threshold = min(tail_scores) - 1e-12
    elif tail_fraction <= 0.0:
        threshold = max(tail_scores) + 1e-12
    else:
        threshold = _quantile(tail_scores, 1.0 - tail_fraction)
    truncated_template = tuple(value if score >= threshold else 0.0 for value, score in zip(template, tail_scores))
    retained_count = sum(1 for value in truncated_template if value != 0.0)
    if retained_count == 0:
        strongest_index = max(range(len(tail_scores)), key=lambda index: tail_scores[index])
        truncated_template = tuple(value if index == strongest_index else 0.0 for index, value in enumerate(template))
        retained_count = 1
    update = _basis_weighted_update(safe_basis.hf_basis, truncated_template, embedding_strength)
    carrier_payload = {
        "carrier_family": "latent_frequency",
        "frequency_band": "high_frequency",
        "template_values": [round(value, 12) for value in truncated_template],
        "update_values": [round(value, 12) for value in update],
        "tail_threshold": round(threshold, 12),
    }
    return CarrierPrimitive(
        carrier_id=f"hf_{build_stable_digest(carrier_payload)[:12]}",
        carrier_family="latent_frequency",
        frequency_band="high_frequency",
        key_digest=build_stable_digest({"key": key, "event_digest": event_digest, "branch": "hf"}),
        template_values=truncated_template,
        update_values=update,
        embedding_strength=embedding_strength,
        carrier_digest=build_stable_digest(carrier_payload),
        tail_threshold=threshold,
        retained_fraction=retained_count / len(truncated_template),
        synthetic_stub=False,
        metadata={"primitive_name": "hf_carrier", "tail_fraction": tail_fraction},
    )


def derive_attention_carrier_stub(
    safe_basis: SafeBasisEstimate,
    key: str,
    event_digest: str,
    embedding_strength: float = 0.05,
) -> CarrierPrimitive:
    """导出 attention-relative 几何载体的 synthetic stub。

    该函数不读取真实 Self-Attention map, 也不声称已接入真实 attention carrier。
    它只提供与 LF/HF 同源的几何 update 形状, 便于后续运行单元替换为真实运行时实现。
    """
    template = _stable_unit_values(
        (key, event_digest, safe_basis.basis_digest, safe_basis.route_digest, "attention_synthetic_stub"),
        len(safe_basis.attention_basis),
    )
    update = _basis_weighted_update(safe_basis.attention_basis, template, embedding_strength)
    carrier_payload = {
        "carrier_family": "attention_relative_geometry",
        "frequency_band": "attention_synthetic_stub",
        "template_values": [round(value, 12) for value in template],
        "update_values": [round(value, 12) for value in update],
    }
    return CarrierPrimitive(
        carrier_id=f"attention_{build_stable_digest(carrier_payload)[:12]}",
        carrier_family="attention_relative_geometry",
        frequency_band="attention_synthetic_stub",
        key_digest=build_stable_digest({"key": key, "event_digest": event_digest, "branch": "attention_stub"}),
        template_values=template,
        update_values=update,
        embedding_strength=embedding_strength,
        carrier_digest=build_stable_digest(carrier_payload),
        tail_threshold=None,
        retained_fraction=1.0,
        synthetic_stub=True,
        metadata={"primitive_name": "attention_carrier_stub", "attention_runtime": "not_connected"},
    )


def compose_latent_update(
    lf_carrier: CarrierPrimitive,
    hf_carrier: CarrierPrimitive,
    attention_carrier: CarrierPrimitive,
) -> LatentUpdateComposition:
    """组合三个载体分量, 返回 `Delta z_t = LF + HF + A`。"""
    length = _ensure_equal_lengths(
        {
            "lf_update_values": lf_carrier.update_values,
            "hf_update_values": hf_carrier.update_values,
            "attention_update_values": attention_carrier.update_values,
        }
    )
    combined = tuple(
        lf_carrier.update_values[index] + hf_carrier.update_values[index] + attention_carrier.update_values[index]
        for index in range(length)
    )
    return LatentUpdateComposition(
        lf_update_values=lf_carrier.update_values,
        hf_update_values=hf_carrier.update_values,
        attention_update_values=attention_carrier.update_values,
        combined_update_values=combined,
        update_digest=build_stable_digest({"combined_update_values": [round(value, 12) for value in combined]}),
        metadata={"primitive_name": "latent_update_composition"},
    )


def compute_content_score(
    observed_values: TensorLike | Iterable[NumberLike],
    lf_carrier: CarrierPrimitive,
    hf_carrier: CarrierPrimitive,
    lambda_lf: float = 0.70,
    lambda_hf: float = 0.30,
) -> ContentScoreResult:
    """计算内容分数 `s_c = lambda_lf * s_lf + lambda_hf * s_hf`。"""
    observed_vector = _as_vector(observed_values, "observed_values")
    _ensure_equal_lengths(
        {
            "observed_values": observed_vector,
            "lf_update_values": lf_carrier.update_values,
            "hf_update_values": hf_carrier.update_values,
        }
    )
    lf_score = _correlation(observed_vector, lf_carrier.update_values)
    hf_raw_score = _correlation(observed_vector, hf_carrier.update_values)
    hf_score = hf_raw_score * hf_carrier.retained_fraction
    content_score = lambda_lf * lf_score + lambda_hf * hf_score
    return ContentScoreResult(
        lf_score=lf_score,
        hf_score=hf_score,
        content_score=content_score,
        lambda_lf=lambda_lf,
        lambda_hf=lambda_hf,
        used_independent_branch_vote=False,
        metadata={"primitive_name": "content_score"},
    )


def evaluate_geometry_reliability(
    registration_confidence: float,
    anchor_inlier_ratio: float,
    recovered_sync_consistency: float,
    alignment_residual: float,
    registration_threshold: float = 0.70,
    inlier_threshold: float = 0.60,
    sync_threshold: float = 0.65,
    residual_threshold: float = 0.40,
) -> GeometryReliabilityResult:
    """评估几何恢复是否可信, 但不输出直接 positive 判定。"""
    geometry_reliable = (
        registration_confidence >= registration_threshold
        and anchor_inlier_ratio >= inlier_threshold
        and recovered_sync_consistency >= sync_threshold
        and alignment_residual <= residual_threshold
    )
    return GeometryReliabilityResult(
        registration_confidence=registration_confidence,
        anchor_inlier_ratio=anchor_inlier_ratio,
        recovered_sync_consistency=recovered_sync_consistency,
        alignment_residual=alignment_residual,
        geometry_reliable=geometry_reliable,
        direct_positive_decision=False,
        metadata={
            "primitive_name": "geometry_reliability",
            "registration_threshold": registration_threshold,
            "inlier_threshold": inlier_threshold,
            "sync_threshold": sync_threshold,
            "residual_threshold": residual_threshold,
        },
    )


def decide_evidence_and_final(
    raw_content_score: float,
    aligned_content_score: float,
    content_threshold: float,
    geometry: GeometryReliabilityResult,
    fail_reason: str,
    attestation_pass: bool,
    rescue_margin_low: float = -0.05,
) -> EvidenceDecisionResult:
    """执行 evidence-level rescue 与 final-level attestation 判定。"""
    allowed_fail_reasons = {"geometry_suspected", "low_confidence"}
    raw_margin = raw_content_score - content_threshold
    aligned_margin = aligned_content_score - content_threshold
    positive_by_content = raw_margin >= 0.0
    rescue_eligible = (
        rescue_margin_low <= raw_margin < 0.0
        and geometry.geometry_reliable
        and fail_reason in allowed_fail_reasons
    )
    rescue_applied = rescue_eligible and aligned_margin >= 0.0
    evidence_level = positive_by_content or rescue_applied
    final_level = evidence_level and attestation_pass
    if final_level:
        final_label = "final_positive"
    elif evidence_level:
        final_label = "evidence_positive_but_unattested"
    else:
        final_label = "evidence_negative"
    return EvidenceDecisionResult(
        raw_content_score=raw_content_score,
        aligned_content_score=aligned_content_score,
        content_threshold=content_threshold,
        raw_content_margin=raw_margin,
        aligned_content_margin=aligned_margin,
        fail_reason=fail_reason,
        rescue_margin_low=rescue_margin_low,
        positive_by_content=positive_by_content,
        rescue_eligible=rescue_eligible,
        rescue_applied=rescue_applied,
        evidence_level=evidence_level,
        attestation_pass=attestation_pass,
        final_level=final_level,
        final_label=final_label,
        metadata={"primitive_name": "evidence_and_final_decision"},
    )
