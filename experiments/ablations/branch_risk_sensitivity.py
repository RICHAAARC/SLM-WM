"""单模型内部的分支风险函数参数敏感性协议。

该实验固定模型、prompt 划分、密钥、随机种子、攻击矩阵和检测协议, 每次只改变
一个手工风险函数参数。它用于回答参数选择是否依赖狭窄取值, 不替代机制必要性
消融, 也不提供跨模型泛化证据。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any, Iterable

from experiments.protocol.method_runtime_config import FormalBranchRiskConfig
from experiments.runners.semantic_watermark_runtime import SemanticWatermarkRuntimeConfig
from main.core.digest import build_stable_digest


_RISK_CONFIG_FIELD_NAMES = (
    "lf_content_risk_config",
    "tail_robust_risk_config",
    "attention_geometry_risk_config",
)
_RISK_WEIGHT_FIELD_NAMES = (
    "local_contrast_risk_weight",
    "semantic_weight",
    "texture_weight",
    "adjacent_step_instability_weight",
    "attention_instability_weight",
)


@dataclass(frozen=True)
class BranchRiskSensitivitySpec:
    """描述一个单参数变化, 其余科学身份保持不变。"""

    sensitivity_id: str
    parameter_name: str
    variation: str
    operation: str
    numeric_value: float

    def __post_init__(self) -> None:
        """拒绝未登记参数和会同时改变多个语义的操作。"""

        supported_parameters = {
            *_RISK_WEIGHT_FIELD_NAMES,
            "eligibility_threshold",
            "budget_floor",
            "budget_ceiling",
            "budget_gain",
            "formal_reference",
        }
        if self.parameter_name not in supported_parameters:
            raise ValueError("敏感性参数不属于手工分支风险函数")
        if self.variation not in {"reference", "low", "high"}:
            raise ValueError("敏感性 variation 必须为 reference、low 或 high")
        if self.operation not in {"identity", "multiply", "replace"}:
            raise ValueError("敏感性 operation 不是受治理的单参数操作")
        if self.parameter_name == "formal_reference":
            if self.variation != "reference" or self.operation != "identity":
                raise ValueError("参考设置必须保持正式参数不变")
        elif self.operation == "identity":
            raise ValueError("非参考设置不得使用 identity 操作")

    def _modify_branch_config(
        self,
        config: FormalBranchRiskConfig,
    ) -> FormalBranchRiskConfig:
        """只修改一个数值字段, 并复用正式配置类型的边界校验。"""

        if self.parameter_name == "formal_reference":
            return config
        current_value = float(getattr(config, self.parameter_name))
        resolved_value = (
            current_value * self.numeric_value
            if self.operation == "multiply"
            else self.numeric_value
        )
        return replace(config, **{self.parameter_name: resolved_value})

    def resolved_risk_configs(
        self,
        config: SemanticWatermarkRuntimeConfig,
    ) -> dict[str, FormalBranchRiskConfig]:
        """返回三个分支的完整参数对象, 便于 manifest 审计真实变化。"""

        return {
            field_name: self._modify_branch_config(getattr(config, field_name))
            for field_name in _RISK_CONFIG_FIELD_NAMES
        }

    def to_dict(
        self,
        reference_config: SemanticWatermarkRuntimeConfig | None = None,
    ) -> dict[str, Any]:
        """返回协议记录, 可选附加三个分支解析后的完整参数。"""

        payload: dict[str, Any] = asdict(self)
        if reference_config is not None:
            payload["resolved_branch_risk_configs"] = {
                name: asdict(value)
                for name, value in self.resolved_risk_configs(
                    reference_config
                ).items()
            }
        return payload

    def apply(
        self,
        config: SemanticWatermarkRuntimeConfig,
        output_root: str,
    ) -> SemanticWatermarkRuntimeConfig:
        """构造真实重新生成所需配置, 不改变模型和随机化身份。"""

        risk_configs = self.resolved_risk_configs(config)
        return replace(
            config,
            **risk_configs,
            risk_parameter_protocol="single_model_internal_sensitivity",
            output_dir=f"{output_root}/runs/{self.sensitivity_id}",
        )


FORMAL_BRANCH_RISK_SENSITIVITY_SPECS = (
    BranchRiskSensitivitySpec(
        "formal_reference",
        "formal_reference",
        "reference",
        "identity",
        1.0,
    ),
    *tuple(
        BranchRiskSensitivitySpec(
            f"{parameter_name}_{variation}",
            parameter_name,
            variation,
            "multiply",
            multiplier,
        )
        for parameter_name in _RISK_WEIGHT_FIELD_NAMES
        for variation, multiplier in (("low", 0.5), ("high", 1.5))
    ),
    BranchRiskSensitivitySpec(
        "eligibility_threshold_low",
        "eligibility_threshold",
        "low",
        "replace",
        0.45,
    ),
    BranchRiskSensitivitySpec(
        "eligibility_threshold_high",
        "eligibility_threshold",
        "high",
        "replace",
        0.65,
    ),
    BranchRiskSensitivitySpec(
        "budget_floor_low",
        "budget_floor",
        "low",
        "replace",
        0.025,
    ),
    BranchRiskSensitivitySpec(
        "budget_floor_high",
        "budget_floor",
        "high",
        "replace",
        0.10,
    ),
    BranchRiskSensitivitySpec(
        "budget_ceiling_low",
        "budget_ceiling",
        "low",
        "replace",
        0.75,
    ),
    BranchRiskSensitivitySpec(
        "budget_gain_low",
        "budget_gain",
        "low",
        "replace",
        0.35,
    ),
    BranchRiskSensitivitySpec(
        "budget_gain_high",
        "budget_gain",
        "high",
        "replace",
        1.05,
    ),
)
FORMAL_BRANCH_RISK_SENSITIVITY_IDS = tuple(
    spec.sensitivity_id for spec in FORMAL_BRANCH_RISK_SENSITIVITY_SPECS
)
FORMAL_BRANCH_RISK_SENSITIVITY_SPEC_DIGEST = build_stable_digest(
    [spec.to_dict() for spec in FORMAL_BRANCH_RISK_SENSITIVITY_SPECS]
)


def default_branch_risk_sensitivity_specs() -> tuple[BranchRiskSensitivitySpec, ...]:
    """返回唯一受治理的18项单模型内部敏感性规范。"""

    return FORMAL_BRANCH_RISK_SENSITIVITY_SPECS


def branch_risk_sensitivity_contract(
    specs: Iterable[BranchRiskSensitivitySpec],
) -> dict[str, Any]:
    """核验设置集合完整、无重复且顺序固定。"""

    resolved = tuple(specs)
    ids = tuple(spec.sensitivity_id for spec in resolved)
    digest = build_stable_digest([spec.to_dict() for spec in resolved])
    exact_set_ready = (
        ids == FORMAL_BRANCH_RISK_SENSITIVITY_IDS
        and len(set(ids)) == len(ids)
        and digest == FORMAL_BRANCH_RISK_SENSITIVITY_SPEC_DIGEST
    )
    return {
        "sensitivity_protocol": "single_model_one_parameter_at_a_time",
        "sensitivity_model_scope": "registered_primary_diffusion_model_only",
        "sensitivity_fixed_identity_fields": [
            "model_id",
            "model_revision",
            "vision_model_id",
            "vision_model_revision",
            "prompt_split_digest",
            "randomization_repeat_id",
            "generation_seed_random",
            "watermark_key_seed_random",
            "attack_matrix_digest",
            "fixed_fpr_protocol_digest",
        ],
        "sensitivity_setting_ids": list(ids),
        "sensitivity_setting_count": len(resolved),
        "sensitivity_spec_digest": digest,
        "sensitivity_exact_set_ready": exact_set_ready,
        "cross_model_evidence_provided": False,
        "supports_paper_claim": False,
    }
