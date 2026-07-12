"""独立验证分支风险输入、风险公式和严格资格预算。"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as functional

from experiments.runtime.diffusion.semantic_features import (
    DifferentiableSemanticFeatureRuntime,
)
from main.methods.semantic import BranchRiskConfig, build_branch_risk_fields


def _risk_config(
    *,
    active_term: str,
    texture_preference: str = "neutral",
    eligibility_threshold: float = 0.55,
) -> BranchRiskConfig:
    """构造只有一个活动风险项的完整配置, 便于逐项验证公式。"""

    weights = {
        "local": 0.0,
        "semantic": 0.0,
        "texture": 0.0,
        "adjacent": 0.0,
        "attention": 0.0,
    }
    weights[active_term] = 1.0
    return BranchRiskConfig(
        local_contrast_risk_weight=weights["local"],
        semantic_weight=weights["semantic"],
        texture_weight=weights["texture"],
        adjacent_step_instability_weight=weights["adjacent"],
        attention_instability_weight=weights["attention"],
        texture_preference=texture_preference,
        eligibility_threshold=eligibility_threshold,
        budget_floor=0.05,
        budget_ceiling=1.0,
        budget_gain=0.70,
    )


def _formula_configs() -> dict[str, BranchRiskConfig]:
    """构造分别覆盖 avoid、prefer 和 neutral 纹理语义的三分支配置。"""

    return {
        "lf_content": _risk_config(
            active_term="texture",
            texture_preference="avoid",
        ),
        "tail_robust": _risk_config(
            active_term="texture",
            texture_preference="prefer",
        ),
        "attention_geometry": _risk_config(
            active_term="texture",
            texture_preference="neutral",
        ),
    }


@pytest.mark.quick
def test_branch_risk_formula_uses_frozen_texture_directions_and_neutral_value() -> None:
    """三个分支必须分别使用 q、1-q 和常数0.5作为纹理风险。"""

    fields = build_branch_risk_fields(
        semantic_values=(0.0, 0.0),
        texture_values=(0.2, 0.8),
        adjacent_step_stability_values=(1.0, 1.0),
        local_contrast_risk_values=(0.0, 0.0),
        attention_stability_values=(1.0, 1.0),
        configs=_formula_configs(),
        risk_neutral_texture_value=0.5,
        required_eligible_branches=(),
    )

    assert fields.lf_content.risk_values == pytest.approx((0.2, 0.8))
    assert fields.tail_robust.risk_values == pytest.approx((0.8, 0.2))
    assert fields.attention_geometry.risk_values == pytest.approx((0.5, 0.5))
    assert fields.lf_content.budget_values == pytest.approx((0.61, 0.19))
    assert fields.tail_robust.budget_values == pytest.approx((0.19, 0.61))
    assert fields.attention_geometry.budget_values == pytest.approx((0.40, 0.40))
    assert fields.lf_content.effective_budget_values == pytest.approx((0.61, 0.0))
    assert fields.tail_robust.effective_budget_values == pytest.approx((0.0, 0.61))
    assert fields.attention_geometry.effective_budget_values == pytest.approx((0.40, 0.40))


@pytest.mark.quick
def test_branch_risk_threshold_equality_is_ineligible() -> None:
    """风险恰好等于阈值时必须不合格, 不能退回小于等于比较。"""

    semantic_config = _risk_config(
        active_term="semantic",
        eligibility_threshold=0.55,
    )
    fields = build_branch_risk_fields(
        semantic_values=(0.55, 0.549),
        texture_values=(0.0, 0.0),
        adjacent_step_stability_values=(1.0, 1.0),
        local_contrast_risk_values=(0.0, 0.0),
        attention_stability_values=(1.0, 1.0),
        configs={name: semantic_config for name in _formula_configs()},
        risk_neutral_texture_value=0.5,
        required_eligible_branches=(),
    )

    assert fields.lf_content.eligible_indices == (1,)
    assert fields.lf_content.effective_budget_values[0] == 0.0
    assert fields.lf_content.effective_budget_values[1] > 0.0


@pytest.mark.quick
def test_branch_risk_builder_rejects_implicit_or_incomplete_method_configuration() -> None:
    """正式 builder 必须显式接收完整三分支配置和冻结 neutral 值。"""

    inputs = {
        "semantic_values": (0.2,),
        "texture_values": (0.2,),
        "adjacent_step_stability_values": (1.0,),
        "local_contrast_risk_values": (0.0,),
        "attention_stability_values": (1.0,),
    }
    with pytest.raises(TypeError):
        build_branch_risk_fields(**inputs)  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="完整定义三个载体分支"):
        build_branch_risk_fields(
            **inputs,
            configs={"lf_content": _formula_configs()["lf_content"]},
            risk_neutral_texture_value=0.5,
        )
    with pytest.raises(ValueError, match="精确等于 0.5"):
        build_branch_risk_fields(
            **inputs,
            configs=_formula_configs(),
            risk_neutral_texture_value=0.0,
        )


@pytest.mark.quick
@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    (
        ("semantic_values", -0.01),
        ("texture_values", 1.01),
        ("adjacent_step_stability_values", -0.01),
        ("local_contrast_risk_values", 1.01),
        ("attention_stability_values", float("inf")),
    ),
)
def test_branch_risk_builder_rejects_nonfinite_or_out_of_range_signals(
    field_name: str,
    invalid_value: float,
) -> None:
    """核心风险公式不得把非法上游信号静默裁剪成合法风险值。"""

    inputs = {
        "semantic_values": (0.2,),
        "texture_values": (0.2,),
        "adjacent_step_stability_values": (1.0,),
        "local_contrast_risk_values": (0.0,),
        "attention_stability_values": (1.0,),
    }
    inputs[field_name] = (invalid_value,)
    with pytest.raises(ValueError):
        build_branch_risk_fields(
            **inputs,
            configs=_formula_configs(),
            risk_neutral_texture_value=0.5,
        )


class _BatchSemanticVision(torch.nn.Module):
    """按每个 batch 样本构造正向或反向的常量 patch 语义。"""

    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))

    def forward(
        self,
        pixel_values: torch.Tensor,
        output_hidden_states: bool,
    ) -> SimpleNamespace:
        """返回一个 CLS token 和2x2 patch token 网格。"""

        del output_hidden_states
        batch = pixel_values.shape[0]
        cls = torch.tensor(
            (1.0, 0.0),
            dtype=torch.float32,
            device=pixel_values.device,
        ).reshape(1, 1, 2).expand(batch, -1, -1)
        direction = torch.where(
            pixel_values.mean(dim=(1, 2, 3)) > 0.5,
            1.0,
            -1.0,
        )
        patches = torch.stack(
            (direction, torch.zeros_like(direction)),
            dim=-1,
        ).reshape(batch, 1, 2).expand(-1, 4, -1)
        return SimpleNamespace(last_hidden_state=torch.cat((cls, patches), dim=1))


def _signal_runtime() -> DifferentiableSemanticFeatureRuntime:
    """构造不改变输入图像的轻量风险信号运行时。"""

    runtime = DifferentiableSemanticFeatureRuntime(
        vae=torch.nn.Identity(),
        vision_model=_BatchSemanticVision(),
    )
    runtime.decode_latent = lambda latent: latent  # type: ignore[method-assign]
    runtime.clip_pixels = lambda image: image  # type: ignore[method-assign]
    return runtime


@pytest.mark.quick
def test_constant_semantic_maps_keep_analytic_values_without_batch_mixing() -> None:
    """常量语义图必须保持0或1, 且一个样本不得改变另一个样本。"""

    current = torch.stack(
        (
            torch.ones((3, 5, 5), dtype=torch.float32),
            torch.zeros((3, 5, 5), dtype=torch.float32),
        )
    )
    signals = _signal_runtime().branch_signal_maps(current, current.clone())

    assert torch.equal(signals["semantic"][0], torch.ones((5, 5)))
    assert torch.equal(signals["semantic"][1], torch.zeros((5, 5)))
    assert torch.equal(signals["texture"], torch.zeros((2, 5, 5)))
    assert torch.equal(
        signals["adjacent_step_stability"],
        torch.ones((2, 5, 5)),
    )


@pytest.mark.quick
def test_image_risk_signals_match_analytic_texture_contrast_and_adjacent_formulas() -> None:
    """纹理、局部对比和相邻稳定度必须直接使用冻结解析公式。"""

    gray = torch.arange(25, dtype=torch.float32).reshape(1, 1, 5, 5) / 24.0
    current = gray.expand(-1, 3, -1, -1).clone()
    previous = current * 0.5
    signals = _signal_runtime().branch_signal_maps(current, previous)

    horizontal = functional.pad(
        (gray[:, :, :, 1:] - gray[:, :, :, :-1]).abs(),
        (0, 1, 0, 0),
    )
    vertical = functional.pad(
        (gray[:, :, 1:, :] - gray[:, :, :-1, :]).abs(),
        (0, 0, 0, 1),
    )
    local_mean = functional.avg_pool2d(
        functional.pad(gray, (2, 2, 2, 2), mode="reflect"),
        kernel_size=5,
        stride=1,
    )
    expected_texture = ((horizontal + vertical) * 0.5)[:, 0]
    expected_contrast = (gray - local_mean).abs()[:, 0]
    expected_stability = 1.0 - (current - previous).abs().mean(dim=1)

    assert torch.allclose(signals["texture"], expected_texture)
    assert torch.allclose(signals["local_contrast_risk"], expected_contrast)
    assert torch.allclose(signals["adjacent_step_stability"], expected_stability)
