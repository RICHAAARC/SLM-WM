"""为不同水印载体构造独立的语义风险场。

通用工程写法: 每个风险项都先转换为 [0, 1] 区间内的风险量, 再执行非负
加权平均。这样配置权重只表达重要性, 不再同时承担正负号语义。

项目特定写法: LF 载体回避高纹理区域, 尾部截断鲁棒载体偏好稳定纹理区域,
注意力几何载体还要求注意力关系稳定。三个分支因此不能共享一个标量风险场。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Iterable, Mapping

from main.core.digest import (
    TENSOR_CONTENT_DIGEST_VERSION,
    build_stable_digest,
    tensor_content_sha256,
)
from main.methods.semantic.vector_values import (
    NumberLike,
    VectorInput,
    as_float_vector,
    clip_unit,
    ensure_equal_length,
)

BRANCH_NAMES = ("lf_content", "tail_robust", "attention_geometry")
NEUTRAL_TEXTURE_RISK_VALUE = 0.5


@dataclass(frozen=True)
class BranchRiskConfig:
    """描述一个载体分支如何解释局部对比度、纹理和真实稳定性.

    `texture_preference` 取值为 `avoid`、`prefer` 或 `neutral`。`avoid` 表示纹理
    越高风险越高, `prefer` 表示纹理越低风险越高。该枚举避免使用容易产生
    符号歧义的负权重。
    """

    local_contrast_risk_weight: float
    semantic_weight: float
    texture_weight: float
    adjacent_step_instability_weight: float
    attention_instability_weight: float
    texture_preference: str
    eligibility_threshold: float
    budget_floor: float
    budget_ceiling: float
    budget_gain: float

    def __post_init__(self) -> None:
        """集中校验风险配置, 业务函数只处理已合法参数。"""

        weights = (
            self.local_contrast_risk_weight,
            self.semantic_weight,
            self.texture_weight,
            self.adjacent_step_instability_weight,
            self.attention_instability_weight,
        )
        scalar_values = (
            *weights,
            self.eligibility_threshold,
            self.budget_floor,
            self.budget_ceiling,
            self.budget_gain,
        )
        if any(not math.isfinite(value) for value in scalar_values):
            raise ValueError("分支风险配置必须全部为有限数")
        if any(value < 0.0 for value in weights) or sum(weights) <= 0.0:
            raise ValueError("分支风险权重必须非负且至少一个权重大于 0")
        if self.texture_preference not in {"avoid", "prefer", "neutral"}:
            raise ValueError("texture_preference 必须为 avoid、prefer 或 neutral")
        if not 0.0 <= self.eligibility_threshold <= 1.0:
            raise ValueError("eligibility_threshold 必须位于 [0, 1]")
        if (
            self.budget_floor < 0.0
            or self.budget_ceiling <= self.budget_floor
            or self.budget_ceiling > 1.0
        ):
            raise ValueError("预算上下界必须形成有效区间")
        if self.budget_gain < 0.0:
            raise ValueError("budget_gain 必须为非负数")


@dataclass(frozen=True)
class CarrierRiskField:
    """保存单个载体分支的风险、预算和可承载位置。"""

    branch_name: str
    risk_values: tuple[float, ...]
    budget_values: tuple[float, ...]
    effective_budget_values: tuple[float, ...]
    eligible_indices: tuple[int, ...]
    risk_values_content_sha256: str
    budget_values_content_sha256: str
    eligible_mask_content_sha256: str
    risk_field_digest: str
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        """校验风险和预算宽度一致。"""

        ensure_equal_length(
            {
                "risk_values": self.risk_values,
                "budget_values": self.budget_values,
                "effective_budget_values": self.effective_budget_values,
            }
        )
        if self.branch_name not in BRANCH_NAMES:
            raise ValueError("未知载体分支")

    def to_dict(self) -> dict[str, Any]:
        """转为可写入 JSON 的字典。"""

        return asdict(self)


@dataclass(frozen=True)
class BranchRiskFieldBundle:
    """保存 LF、尾部鲁棒和注意力几何三个独立风险场。"""

    lf_content: CarrierRiskField
    tail_robust: CarrierRiskField
    attention_geometry: CarrierRiskField
    bundle_digest: str
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        """保证三个分支在同一 latent 网格上定义。"""

        ensure_equal_length(
            {
                "lf_content": self.lf_content.risk_values,
                "tail_robust": self.tail_robust.risk_values,
                "attention_geometry": self.attention_geometry.risk_values,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        """转为可写入 JSON 的字典。"""

        return asdict(self)


def _texture_risk(
    texture: float,
    preference: str,
    neutral_texture_risk_value: float,
) -> float:
    """把纹理偏好转换成统一的风险量。"""

    if preference == "avoid":
        return texture
    if preference == "prefer":
        return 1.0 - texture
    return neutral_texture_risk_value


def _require_unit_vector(
    values: VectorInput | Iterable[NumberLike],
    field_name: str,
) -> tuple[float, ...]:
    """读取有限风险信号并拒绝超出冻结解析范围的输入。"""

    resolved = as_float_vector(values, field_name)
    if any(value < 0.0 or value > 1.0 for value in resolved):
        raise ValueError(f"{field_name} 必须全部位于 [0, 1]")
    return resolved


def _build_single_branch(
    branch_name: str,
    semantic_values: tuple[float, ...],
    texture_values: tuple[float, ...],
    adjacent_step_stability_values: tuple[float, ...],
    local_contrast_risk_values: tuple[float, ...],
    attention_stability_values: tuple[float, ...],
    config: BranchRiskConfig,
    neutral_texture_risk_value: float,
    require_eligible_position: bool,
) -> CarrierRiskField:
    """根据统一输入构造一个分支风险场。"""

    weight_sum = (
        config.local_contrast_risk_weight
        + config.semantic_weight
        + config.texture_weight
        + config.adjacent_step_instability_weight
        + config.attention_instability_weight
    )
    risks: list[float] = []
    budgets: list[float] = []
    for (
        semantic,
        texture,
        adjacent_step_stability,
        local_contrast_risk,
        attention_stability,
    ) in zip(
        semantic_values,
        texture_values,
        adjacent_step_stability_values,
        local_contrast_risk_values,
        attention_stability_values,
    ):
        raw_risk = (
            config.local_contrast_risk_weight * local_contrast_risk
            + config.semantic_weight * semantic
            + config.texture_weight
            * _texture_risk(
                texture,
                config.texture_preference,
                neutral_texture_risk_value,
            )
            + config.adjacent_step_instability_weight
            * (1.0 - adjacent_step_stability)
            + config.attention_instability_weight * (1.0 - attention_stability)
        ) / weight_sum
        risk = clip_unit(raw_risk)
        budget = max(
            config.budget_floor,
            min(
                config.budget_ceiling,
                config.budget_floor + config.budget_gain * (1.0 - risk),
            ),
        )
        risks.append(risk)
        budgets.append(budget)
    eligible = tuple(
        index
        for index, risk in enumerate(risks)
        if risk < config.eligibility_threshold
    )
    if require_eligible_position and not eligible:
        raise RuntimeError(
            f"{branch_name} 分支没有满足冻结风险阈值的可承载位置"
        )
    eligible_set = set(eligible)
    effective_budgets = tuple(
        budget if index in eligible_set else 0.0
        for index, budget in enumerate(budgets)
    )
    payload = {
        "branch_name": branch_name,
        "risk_values": [round(value, 12) for value in risks],
        "budget_values": [round(value, 12) for value in budgets],
        "effective_budget_values": [
            round(value, 12) for value in effective_budgets
        ],
        "eligible_indices": eligible,
        "config": asdict(config),
        "require_eligible_position": require_eligible_position,
        "neutral_texture_risk_value": neutral_texture_risk_value,
    }
    import torch

    risk_values_content_sha256 = tensor_content_sha256(
        torch.tensor(risks, dtype=torch.float64)
    )
    budget_values_content_sha256 = tensor_content_sha256(
        torch.tensor(budgets, dtype=torch.float64)
    )
    eligible_mask = torch.zeros(len(risks), dtype=torch.bool)
    if eligible:
        eligible_mask[list(eligible)] = True
    eligible_mask_content_sha256 = tensor_content_sha256(eligible_mask)
    payload["risk_values_content_sha256"] = risk_values_content_sha256
    payload["budget_values_content_sha256"] = budget_values_content_sha256
    payload["eligible_mask_content_sha256"] = eligible_mask_content_sha256
    payload["tensor_content_digest_version"] = TENSOR_CONTENT_DIGEST_VERSION
    return CarrierRiskField(
        branch_name=branch_name,
        risk_values=tuple(risks),
        budget_values=tuple(budgets),
        effective_budget_values=effective_budgets,
        eligible_indices=eligible,
        risk_values_content_sha256=risk_values_content_sha256,
        budget_values_content_sha256=budget_values_content_sha256,
        eligible_mask_content_sha256=eligible_mask_content_sha256,
        risk_field_digest=build_stable_digest(payload),
        metadata={
            "risk_definition": "branch_specific_nonnegative_risk_terms",
            "local_contrast_risk_definition": (
                "decoded_grayscale_absolute_deviation_from_5x5_local_mean"
            ),
            "adjacent_step_stability_definition": (
                "one_minus_mean_absolute_decoded_rgb_change_from_previous_scheduler_step"
            ),
            "attention_stability_definition": (
                "cross_frozen_layer_direct_qk_relation_consistency"
            ),
            "tensor_content_digest_version": TENSOR_CONTENT_DIGEST_VERSION,
            "texture_preference": config.texture_preference,
            "eligibility_threshold": config.eligibility_threshold,
            "eligibility_comparison": "strict_less_than",
            "neutral_texture_risk_value": neutral_texture_risk_value,
        },
    )


def build_branch_risk_fields(
    semantic_values: VectorInput | Iterable[NumberLike],
    texture_values: VectorInput | Iterable[NumberLike],
    adjacent_step_stability_values: VectorInput | Iterable[NumberLike],
    local_contrast_risk_values: VectorInput | Iterable[NumberLike],
    attention_stability_values: VectorInput | Iterable[NumberLike],
    *,
    configs: Mapping[str, BranchRiskConfig],
    risk_neutral_texture_value: float,
    required_eligible_branches: Iterable[str] | None = None,
) -> BranchRiskFieldBundle:
    """构造三个语义不同且宽度一致的分支风险场。

    该函数是正式方法路径唯一使用的风险入口。三个分支必须分别构造风险语义,
    不能通过单一共享标量或仅用于日志的路由记录替代。

    ``required_eligible_branches`` 指定必须至少存在一个合格位置的活动分支.
    ``None`` 表示完整方法的三个分支都必须通过门禁. 空集合用于移除风险路由的
    正式消融, 此时仍可记录风险诊断值, 但风险阈值不得决定样本能否继续运行.
    ``configs`` 与 ``risk_neutral_texture_value`` 必须由唯一方法配置显式传入,
    核心方法不保存会绕过配置身份的正式默认参数.
    """

    semantic = _require_unit_vector(semantic_values, "semantic_values")
    texture = _require_unit_vector(texture_values, "texture_values")
    adjacent_step_stability = _require_unit_vector(
        adjacent_step_stability_values,
        "adjacent_step_stability_values",
    )
    local_contrast_risk = _require_unit_vector(
        local_contrast_risk_values,
        "local_contrast_risk_values",
    )
    if attention_stability_values is None:
        raise ValueError("attention_stability_values 必须来自真实跨层 Q/K 关系")
    attention_stability = _require_unit_vector(
        attention_stability_values,
        "attention_stability_values",
    )
    length = ensure_equal_length(
        {
            "semantic_values": semantic,
            "texture_values": texture,
            "adjacent_step_stability_values": adjacent_step_stability,
            "local_contrast_risk_values": local_contrast_risk,
            "attention_stability_values": attention_stability,
        }
    )
    if (
        not math.isfinite(risk_neutral_texture_value)
        or risk_neutral_texture_value != NEUTRAL_TEXTURE_RISK_VALUE
    ):
        raise ValueError("risk_neutral_texture_value 必须精确等于 0.5")
    resolved_configs = dict(configs)
    if set(resolved_configs) != set(BRANCH_NAMES):
        raise ValueError("configs 必须完整定义三个载体分支")
    if any(
        not isinstance(resolved_configs[name], BranchRiskConfig)
        for name in BRANCH_NAMES
    ):
        raise TypeError("configs 的每个分支都必须使用 BranchRiskConfig")
    required_branches = (
        set(BRANCH_NAMES)
        if required_eligible_branches is None
        else set(required_eligible_branches)
    )
    if not required_branches <= set(BRANCH_NAMES):
        raise ValueError("required_eligible_branches 包含未知载体分支")
    fields = {
        name: _build_single_branch(
            name,
            semantic,
            texture,
            adjacent_step_stability,
            local_contrast_risk,
            attention_stability,
            resolved_configs[name],
            risk_neutral_texture_value,
            name in required_branches,
        )
        for name in BRANCH_NAMES
    }
    bundle_digest = build_stable_digest(
        {name: field.risk_field_digest for name, field in fields.items()}
    )
    return BranchRiskFieldBundle(
        lf_content=fields["lf_content"],
        tail_robust=fields["tail_robust"],
        attention_geometry=fields["attention_geometry"],
        bundle_digest=bundle_digest,
        metadata={"risk_field_length": length, "branch_names": BRANCH_NAMES},
    )
