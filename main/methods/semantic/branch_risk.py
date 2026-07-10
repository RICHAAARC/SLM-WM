"""为不同水印载体构造独立的语义风险场。

通用工程写法: 每个风险项都先转换为 [0, 1] 区间内的风险量, 再执行非负
加权平均。这样配置权重只表达重要性, 不再同时承担正负号语义。

项目特定写法: LF 载体回避高纹理区域, 尾部截断鲁棒载体偏好稳定纹理区域,
注意力几何载体还要求注意力关系稳定。三个分支因此不能共享一个标量风险场。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

from main.core.digest import build_stable_digest
from main.methods.semantic.risk_field import NumberLike, VectorInput, as_float_vector, clip_unit, ensure_equal_length

BRANCH_NAMES = ("lf_content", "tail_robust", "attention_geometry")


@dataclass(frozen=True)
class BranchRiskConfig:
    """描述一个载体分支如何解释纹理和稳定性。

    `texture_preference` 取值为 `avoid`、`prefer` 或 `neutral`。`avoid` 表示纹理
    越高风险越高, `prefer` 表示纹理越低风险越高。该枚举避免使用容易产生
    符号歧义的负权重。
    """

    saliency_weight: float
    semantic_weight: float
    texture_weight: float
    instability_weight: float
    attention_instability_weight: float = 0.0
    texture_preference: str = "neutral"
    eligibility_threshold: float = 0.55
    budget_floor: float = 0.05
    budget_ceiling: float = 1.0
    budget_gain: float = 0.70

    def __post_init__(self) -> None:
        """集中校验风险配置, 业务函数只处理已合法参数。"""

        weights = (
            self.saliency_weight,
            self.semantic_weight,
            self.texture_weight,
            self.instability_weight,
            self.attention_instability_weight,
        )
        if any(value < 0.0 for value in weights) or sum(weights) <= 0.0:
            raise ValueError("分支风险权重必须非负且至少一个权重大于 0")
        if self.texture_preference not in {"avoid", "prefer", "neutral"}:
            raise ValueError("texture_preference 必须为 avoid、prefer 或 neutral")
        if not 0.0 <= self.eligibility_threshold <= 1.0:
            raise ValueError("eligibility_threshold 必须位于 [0, 1]")
        if self.budget_floor < 0.0 or self.budget_ceiling <= self.budget_floor:
            raise ValueError("预算上下界必须形成有效区间")


@dataclass(frozen=True)
class CarrierRiskField:
    """保存单个载体分支的风险、预算和可承载位置。"""

    branch_name: str
    risk_values: tuple[float, ...]
    budget_values: tuple[float, ...]
    eligible_indices: tuple[int, ...]
    risk_field_digest: str
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        """校验风险和预算宽度一致。"""

        ensure_equal_length({"risk_values": self.risk_values, "budget_values": self.budget_values})
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


DEFAULT_BRANCH_CONFIGS = {
    "lf_content": BranchRiskConfig(
        saliency_weight=0.30,
        semantic_weight=0.30,
        texture_weight=0.20,
        instability_weight=0.20,
        texture_preference="avoid",
    ),
    "tail_robust": BranchRiskConfig(
        saliency_weight=0.25,
        semantic_weight=0.25,
        texture_weight=0.30,
        instability_weight=0.20,
        texture_preference="prefer",
    ),
    "attention_geometry": BranchRiskConfig(
        saliency_weight=0.20,
        semantic_weight=0.25,
        texture_weight=0.05,
        instability_weight=0.20,
        attention_instability_weight=0.30,
        texture_preference="neutral",
    ),
}


def _texture_risk(texture: float, preference: str) -> float:
    """把纹理偏好转换成统一的风险量。"""

    if preference == "avoid":
        return texture
    if preference == "prefer":
        return 1.0 - texture
    return 0.0


def _build_single_branch(
    branch_name: str,
    semantic_values: tuple[float, ...],
    texture_values: tuple[float, ...],
    stability_values: tuple[float, ...],
    saliency_values: tuple[float, ...],
    attention_stability_values: tuple[float, ...],
    config: BranchRiskConfig,
) -> CarrierRiskField:
    """根据统一输入构造一个分支风险场。"""

    weight_sum = (
        config.saliency_weight
        + config.semantic_weight
        + config.texture_weight
        + config.instability_weight
        + config.attention_instability_weight
    )
    risks: list[float] = []
    budgets: list[float] = []
    for semantic, texture, stability, saliency, attention_stability in zip(
        semantic_values,
        texture_values,
        stability_values,
        saliency_values,
        attention_stability_values,
    ):
        raw_risk = (
            config.saliency_weight * saliency
            + config.semantic_weight * semantic
            + config.texture_weight * _texture_risk(texture, config.texture_preference)
            + config.instability_weight * (1.0 - stability)
            + config.attention_instability_weight * (1.0 - attention_stability)
        ) / weight_sum
        risk = clip_unit(raw_risk)
        budget = min(
            config.budget_ceiling,
            config.budget_floor + config.budget_gain * (1.0 - risk),
        )
        risks.append(risk)
        budgets.append(budget)
    eligible = tuple(index for index, risk in enumerate(risks) if risk <= config.eligibility_threshold)
    if not eligible:
        eligible = (min(range(len(risks)), key=risks.__getitem__),)
    payload = {
        "branch_name": branch_name,
        "risk_values": [round(value, 12) for value in risks],
        "budget_values": [round(value, 12) for value in budgets],
        "eligible_indices": eligible,
        "config": asdict(config),
    }
    return CarrierRiskField(
        branch_name=branch_name,
        risk_values=tuple(risks),
        budget_values=tuple(budgets),
        eligible_indices=eligible,
        risk_field_digest=build_stable_digest(payload),
        metadata={
            "risk_definition": "branch_specific_nonnegative_risk_terms",
            "texture_preference": config.texture_preference,
            "eligibility_threshold": config.eligibility_threshold,
        },
    )


def build_branch_risk_fields(
    semantic_values: VectorInput | Iterable[NumberLike],
    texture_values: VectorInput | Iterable[NumberLike],
    stability_values: VectorInput | Iterable[NumberLike],
    saliency_values: VectorInput | Iterable[NumberLike],
    attention_stability_values: VectorInput | Iterable[NumberLike] | None = None,
    configs: dict[str, BranchRiskConfig] | None = None,
) -> BranchRiskFieldBundle:
    """构造三个语义不同且宽度一致的分支风险场。

    该函数是正式方法路径使用的风险入口。旧的 `build_risk_field` 继续保留给历史
    记录重建, 但不再作为真实载体路由的共同风险定义。
    """

    semantic = tuple(clip_unit(value) for value in as_float_vector(semantic_values, "semantic_values"))
    texture = tuple(clip_unit(value) for value in as_float_vector(texture_values, "texture_values"))
    stability = tuple(clip_unit(value) for value in as_float_vector(stability_values, "stability_values"))
    saliency = tuple(clip_unit(value) for value in as_float_vector(saliency_values, "saliency_values"))
    attention_stability = (
        stability
        if attention_stability_values is None
        else tuple(
            clip_unit(value)
            for value in as_float_vector(attention_stability_values, "attention_stability_values")
        )
    )
    length = ensure_equal_length(
        {
            "semantic_values": semantic,
            "texture_values": texture,
            "stability_values": stability,
            "saliency_values": saliency,
            "attention_stability_values": attention_stability,
        }
    )
    resolved_configs = configs or DEFAULT_BRANCH_CONFIGS
    if set(resolved_configs) != set(BRANCH_NAMES):
        raise ValueError("configs 必须完整定义三个载体分支")
    fields = {
        name: _build_single_branch(
            name,
            semantic,
            texture,
            stability,
            saliency,
            attention_stability,
            resolved_configs[name],
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
