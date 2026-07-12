"""验证分支风险资格集合严格遵守冻结阈值。"""

from __future__ import annotations

from dataclasses import replace

import pytest

from experiments.ablations.runtime_rerun import default_runtime_rerun_ablation_specs
from experiments.runners.semantic_watermark_runtime import (
    SemanticWatermarkRuntimeConfig,
    _required_branch_risk_eligibility,
)
from main.methods.semantic import BranchRiskConfig, build_branch_risk_fields


def _strict_configs() -> dict[str, BranchRiskConfig]:
    """构造不会接受高风险位置的三分支配置。"""

    return {
        branch_name: BranchRiskConfig(
            local_contrast_risk_weight=1.0,
            semantic_weight=0.0,
            texture_weight=0.0,
            adjacent_step_instability_weight=0.0,
            eligibility_threshold=0.1,
        )
        for branch_name in (
            "lf_content",
            "tail_robust",
            "attention_geometry",
        )
    }


@pytest.mark.quick
def test_branch_risk_rejects_empty_frozen_eligibility_set() -> None:
    """正式风险场不得把最低风险单点冒充满足冻结阈值的位置。"""

    with pytest.raises(RuntimeError, match="没有满足冻结风险阈值"):
        build_branch_risk_fields(
            semantic_values=(0.0, 0.0),
            texture_values=(0.0, 0.0),
            adjacent_step_stability_values=(1.0, 1.0),
            local_contrast_risk_values=(0.8, 0.9),
            attention_stability_values=(1.0, 1.0),
            configs=_strict_configs(),
        )


@pytest.mark.quick
def test_branch_risk_requires_independent_attention_stability() -> None:
    """缺少真实跨层 Q/K 稳定度时必须失败, 不得复用扩散稳定度."""

    with pytest.raises(ValueError, match="真实跨层 Q/K"):
        build_branch_risk_fields(
            semantic_values=(0.1, 0.2),
            texture_values=(0.3, 0.4),
            adjacent_step_stability_values=(0.5, 0.6),
            local_contrast_risk_values=(0.7, 0.8),
            attention_stability_values=None,  # type: ignore[arg-type]
        )


@pytest.mark.quick
def test_without_branch_risk_routing_does_not_filter_formal_ablation_samples() -> None:
    """移除风险路由的正式消融不得继续用风险阈值筛掉样本."""

    ablation = next(
        spec
        for spec in default_runtime_rerun_ablation_specs()
        if spec.ablation_id == "without_branch_risk_routing"
    )
    config = ablation.apply(
        SemanticWatermarkRuntimeConfig(),
        "outputs/formal_mechanism_ablation/probe_paper",
    )
    required_branches = _required_branch_risk_eligibility(config)

    fields = build_branch_risk_fields(
        semantic_values=(0.0, 0.0),
        texture_values=(0.0, 0.0),
        adjacent_step_stability_values=(1.0, 1.0),
        local_contrast_risk_values=(0.8, 0.9),
        attention_stability_values=(1.0, 1.0),
        configs=_strict_configs(),
        required_eligible_branches=required_branches,
    )

    assert required_branches == ()
    assert fields.lf_content.eligible_indices == ()
    assert fields.tail_robust.eligible_indices == ()
    assert fields.attention_geometry.eligible_indices == ()


@pytest.mark.quick
def test_disabled_attention_branch_does_not_apply_eligibility_gate() -> None:
    """载体消融只对仍参与嵌入的活动分支执行 fail-closed 门禁."""

    ablation = next(
        spec
        for spec in default_runtime_rerun_ablation_specs()
        if spec.ablation_id == "without_attention_geometry"
    )
    config = ablation.apply(
        SemanticWatermarkRuntimeConfig(),
        "outputs/formal_mechanism_ablation/probe_paper",
    )
    configs = _strict_configs()
    configs["lf_content"] = replace(
        configs["lf_content"],
        eligibility_threshold=1.0,
    )
    configs["tail_robust"] = replace(
        configs["tail_robust"],
        eligibility_threshold=1.0,
    )
    required_branches = _required_branch_risk_eligibility(config)

    fields = build_branch_risk_fields(
        semantic_values=(0.0, 0.0),
        texture_values=(0.0, 0.0),
        adjacent_step_stability_values=(1.0, 1.0),
        local_contrast_risk_values=(0.8, 0.9),
        attention_stability_values=(1.0, 1.0),
        configs=configs,
        required_eligible_branches=required_branches,
    )

    assert required_branches == ("lf_content", "tail_robust")
    assert fields.lf_content.eligible_indices == (0, 1)
    assert fields.tail_robust.eligible_indices == (0, 1)
    assert fields.attention_geometry.eligible_indices == ()
