"""验证载体归一化、分支风险硬包络和实际 dtype 合成性质。"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import MISSING, fields
import inspect
import math

import pytest
import torch
import torch.nn.functional as functional

from main.core.digest import tensor_content_sha256
from main.core.keyed_prg import KEYED_PRG_VERSION, build_keyed_gaussian_tensor
from main.methods.carrier.keyed_tensor import (
    LowFrequencyCarrierConfig,
    build_low_frequency_template,
    build_tail_robust_template,
    compute_blind_content_score,
    project_canonical_template,
)
from main.methods.update_composition import (
    QUANTIZED_COMPOSITION_EVIDENCE_VERSION,
    QUANTIZED_COMPOSITION_ORDER,
    build_quantized_composition_candidate,
    build_risk_bounded_update,
    compose_ordered_float32_update_once,
    iter_quantized_composition_candidates,
    recompute_quantized_composition_evidence_digest,
    rescale_risk_bounded_update,
)


FORMAL_LOW_FREQUENCY_CONFIG = LowFrequencyCarrierConfig(
    kernel_size=5,
    stride=1,
    padding=2,
    boundary_mode="zero_padding",
    ceil_mode=False,
    count_include_pad=True,
    divisor_override=None,
)


@pytest.mark.quick
def test_low_frequency_and_content_score_protocols_have_no_defaults() -> None:
    """核心 LF 协议和内容权重必须由调用方完整显式提供."""

    for field in fields(LowFrequencyCarrierConfig):
        assert field.default is MISSING
        assert field.default_factory is MISSING
    template_signature = inspect.signature(build_low_frequency_template)
    for parameter_name in ("low_frequency_config", "prg_version"):
        assert (
            template_signature.parameters[parameter_name].default
            is inspect.Parameter.empty
        )
    score_signature = inspect.signature(compute_blind_content_score)
    for parameter_name in ("lf_weight", "tail_robust_weight"):
        assert (
            score_signature.parameters[parameter_name].default
            is inspect.Parameter.empty
        )
    tail_signature = inspect.signature(build_tail_robust_template)
    for parameter_name in ("tail_fraction", "prg_version"):
        assert (
            tail_signature.parameters[parameter_name].default
            is inspect.Parameter.empty
        )
    projection_signature = inspect.signature(project_canonical_template)
    for parameter_name in (
        "minimum_energy_retention",
        "carrier_protocol_digest",
        "prg_version",
    ):
        assert (
            projection_signature.parameters[parameter_name].default
            is inspect.Parameter.empty
        )


@pytest.mark.quick
@pytest.mark.parametrize("tail_fraction", (True, 1, 0.0, 1.1))
def test_tail_template_rejects_fraction_type_or_range_drift(
    tail_fraction: object,
) -> None:
    """尾部模板必须独立拒绝布尔、整数和越界保留比例."""

    with pytest.raises(ValueError, match="精确 float"):
        build_tail_robust_template(
            torch.zeros(1, 1, 4, 4),
            "tail-key",
            "tail-model",
            tail_fraction,
            prg_version=KEYED_PRG_VERSION,
        )


@pytest.mark.quick
def test_tail_template_requires_nchw_reference() -> None:
    """尾部模板不得把非 NCHW Tensor 解释为正式 latent."""

    with pytest.raises(ValueError, match="batch, channel, height, width"):
        build_tail_robust_template(
            torch.zeros(1, 4, 4),
            "tail-key",
            "tail-model",
            0.20,
            prg_version=KEYED_PRG_VERSION,
        )


@pytest.mark.quick
@pytest.mark.parametrize(
    ("lf_weight", "tail_weight", "error_type"),
    (
        (-1.0, 2.0, ValueError),
        (2.0, -1.0, ValueError),
        (0, 1.0, TypeError),
        (0.5, True, TypeError),
    ),
)
def test_blind_content_score_rejects_invalid_weight_protocol(
    lf_weight: object,
    tail_weight: object,
    error_type: type[Exception],
) -> None:
    """核心内容分数算子必须独立拒绝越界权重和类型伪装."""

    observed = torch.zeros(1, 1, 4, 4)
    template = torch.ones_like(observed)
    with pytest.raises(error_type):
        compute_blind_content_score(
            observed,
            template,
            template,
            lf_weight,
            tail_weight,
        )


@pytest.mark.quick
@pytest.mark.parametrize(
    ("field_name", "invalid_value", "error_type"),
    (
        ("kernel_size", 5.0, TypeError),
        ("stride", True, TypeError),
        ("padding", 2.0, TypeError),
        ("boundary_mode", None, TypeError),
        ("ceil_mode", 0, TypeError),
        ("count_include_pad", 1, TypeError),
        ("divisor_override", False, TypeError),
        ("kernel_size", 7, ValueError),
        ("stride", 2, ValueError),
        ("padding", 1, ValueError),
        ("boundary_mode", "reflect", ValueError),
        ("ceil_mode", True, ValueError),
        ("count_include_pad", False, ValueError),
        ("divisor_override", 9, TypeError),
    ),
)
def test_low_frequency_config_rejects_exact_type_and_protocol_drift(
    field_name: str,
    invalid_value: object,
    error_type: type[Exception],
) -> None:
    """七个离散字段不得接受 Python 等值伪装或正式协议漂移."""

    values = {
        "kernel_size": 5,
        "stride": 1,
        "padding": 2,
        "boundary_mode": "zero_padding",
        "ceil_mode": False,
        "count_include_pad": True,
        "divisor_override": None,
    }
    values[field_name] = invalid_value
    with pytest.raises(error_type):
        LowFrequencyCarrierConfig(**values)


@pytest.mark.quick
def test_low_frequency_template_consumes_frozen_spatial_pooling_parameters() -> None:
    """LF 必须只在 H/W 上执行显式零填充平均池化并全局归一化。"""

    reference = torch.zeros((1, 2, 7, 7), dtype=torch.float32)
    template = build_low_frequency_template(
        reference,
        "lf-key",
        "model@revision",
        FORMAL_LOW_FREQUENCY_CONFIG,
        prg_version=KEYED_PRG_VERSION,
    )
    raw = build_keyed_gaussian_tensor(
        tuple(reference.shape),
        "lf-key",
        {
            "operator": "latent_carrier_template",
            "model_id": "model@revision",
            "branch_name": "lf_content",
        },
    )
    pooled = functional.avg_pool2d(
        raw,
        kernel_size=5,
        stride=1,
        padding=2,
        ceil_mode=False,
        count_include_pad=True,
        divisor_override=None,
    )
    expected = (pooled - pooled.mean()) / (pooled - pooled.mean()).norm()

    assert torch.equal(template, expected)
    assert float(template.mean().item()) == pytest.approx(0.0, abs=1e-7)
    assert float(template.norm().item()) == pytest.approx(1.0, abs=1e-7)


@pytest.mark.quick
def test_tail_template_preserves_exact_sparse_support_without_centering() -> None:
    """tail 分支截断后只能 L2 归一化, 未选择坐标必须保持精确零。"""

    reference = torch.zeros((1, 2, 4, 4), dtype=torch.float32)
    tail_fraction = 0.25
    template, threshold, retained_fraction = build_tail_robust_template(
        reference,
        "tail-key",
        "model@revision",
        tail_fraction,
        prg_version=KEYED_PRG_VERSION,
    )
    raw = build_keyed_gaussian_tensor(
        tuple(reference.shape),
        "tail-key",
        {
            "operator": "latent_carrier_template",
            "model_id": "model@revision",
            "branch_name": "tail_robust",
        },
    )
    flat = raw.reshape(-1)
    retained_count = math.ceil(flat.numel() * tail_fraction)
    selected = sorted(
        range(flat.numel()),
        key=lambda index: (-abs(float(flat[index].item())), index),
    )[:retained_count]
    expected_flat = torch.zeros_like(flat)
    expected_flat[selected] = flat[selected]
    expected = (expected_flat / expected_flat.norm()).reshape(reference.shape)

    assert torch.equal(template, expected)
    assert torch.count_nonzero(template).item() == retained_count
    assert torch.equal(
        template.reshape(-1)[expected_flat == 0.0],
        torch.zeros_like(expected_flat[expected_flat == 0.0]),
    )
    assert float(template.norm().item()) == pytest.approx(1.0, abs=1e-7)
    assert threshold == pytest.approx(abs(float(flat[selected[-1]].item())))
    assert retained_fraction == pytest.approx(tail_fraction)


@pytest.mark.quick
def test_carrier_templates_remain_canonical_float32_for_float16_latent() -> None:
    """扩散 latent dtype 不得在安全投影前量化规范载体模板。"""

    reference_float16 = torch.zeros((1, 2, 4, 4), dtype=torch.float16)
    reference_float32 = reference_float16.float()
    lf_float16 = build_low_frequency_template(
        reference_float16,
        "canonical-lf-key",
        "canonical-model",
        FORMAL_LOW_FREQUENCY_CONFIG,
        prg_version=KEYED_PRG_VERSION,
    )
    lf_float32 = build_low_frequency_template(
        reference_float32,
        "canonical-lf-key",
        "canonical-model",
        FORMAL_LOW_FREQUENCY_CONFIG,
        prg_version=KEYED_PRG_VERSION,
    )
    tail_float16 = build_tail_robust_template(
        reference_float16,
        "canonical-tail-key",
        "canonical-model",
        0.20,
        prg_version=KEYED_PRG_VERSION,
    )[0]
    tail_float32 = build_tail_robust_template(
        reference_float32,
        "canonical-tail-key",
        "canonical-model",
        0.20,
        prg_version=KEYED_PRG_VERSION,
    )[0]

    assert lf_float16.dtype == torch.float32
    assert tail_float16.dtype == torch.float32
    assert torch.equal(lf_float16, lf_float32)
    assert torch.equal(tail_float16, tail_float32)
    assert tensor_content_sha256(lf_float16) == tensor_content_sha256(
        lf_float32
    )
    assert tensor_content_sha256(tail_float16) == tensor_content_sha256(
        tail_float32
    )


@pytest.mark.quick
def test_risk_budget_monotonically_bounds_update_and_zero_support() -> None:
    """增大预算只能放宽步长, 零预算位置不得出现方向或更新泄漏。"""

    direction = torch.tensor([[[[1.0, 0.0], [2.0, 0.0]]]])
    low = build_risk_bounded_update(
        branch_name="lf_content",
        direction=direction,
        effective_budget=torch.tensor([[[0.25, 0.0], [0.50, 0.0]]]),
        nominal_strength=1.0,
        budget_ceiling=1.0,
    )
    high = build_risk_bounded_update(
        branch_name="lf_content",
        direction=direction,
        effective_budget=torch.tensor([[[0.50, 0.0], [1.00, 0.0]]]),
        nominal_strength=1.0,
        budget_ceiling=1.0,
    )

    assert float(high.applied_strength.item()) >= float(low.applied_strength.item())
    assert torch.all(high.amplitude_envelope >= low.amplitude_envelope)
    assert torch.count_nonzero(low.update[low.effective_budget == 0.0]).item() == 0
    assert float(low.maximum_envelope_ratio.item()) <= 1.0
    record = low.to_record()
    assert record["effective_budget_values_content_sha256"] == tensor_content_sha256(
        low.effective_budget
    )
    assert record["branch_budget_envelope_content_sha256"] == tensor_content_sha256(
        low.amplitude_envelope
    )
    assert record["branch_written_update_content_sha256"] == tensor_content_sha256(
        low.update
    )
    assert record["branch_direction_epsilon"] == 1e-12
    assert record["branch_numerical_epsilon"] == 1e-12

    with pytest.raises(RuntimeError, match="零预算位置存在方向泄漏"):
        build_risk_bounded_update(
            branch_name="lf_content",
            direction=torch.ones_like(direction),
            effective_budget=torch.tensor([[[1.0, 0.0], [1.0, 1.0]]]),
            nominal_strength=1.0,
            budget_ceiling=1.0,
        )

    epsilon = 1e-6
    tiny_direction = torch.tensor([[[[1.0, 0.5 * epsilon], [2.0, 0.0]]]])
    tiny = build_risk_bounded_update(
        branch_name="lf_content",
        direction=tiny_direction,
        effective_budget=torch.tensor([[[1.0, 0.0], [1.0, 0.0]]]),
        nominal_strength=1.0,
        budget_ceiling=1.0,
        direction_epsilon=epsilon,
        numerical_epsilon=1e-12,
    )
    assert tiny.unit_direction[0, 0, 0, 1].item() == 0.0
    with pytest.raises(RuntimeError, match="零预算位置存在方向泄漏"):
        build_risk_bounded_update(
            branch_name="lf_content",
            direction=torch.tensor(
                [[[[1.0, 3.0 * epsilon], [2.0, 0.0]]]]
            ),
            effective_budget=torch.tensor([[[1.0, 0.0], [1.0, 0.0]]]),
            nominal_strength=1.0,
            budget_ceiling=1.0,
            direction_epsilon=epsilon,
            numerical_epsilon=1e-12,
        )


@pytest.mark.quick
def test_risk_bounded_update_separates_direction_and_numerical_epsilon() -> None:
    """方向活动阈值和最终步长阈值必须分别控制坐标支持与 alpha。"""

    direction = torch.tensor([[[[0.05, 1.0], [0.10, 0.0]]]])
    result = build_risk_bounded_update(
        branch_name="lf_content",
        direction=direction,
        effective_budget=torch.tensor([[[0.0, 1.0], [1.0, 0.0]]]),
        nominal_strength=1.0,
        budget_ceiling=1.0,
        direction_epsilon=0.1,
        numerical_epsilon=0.01,
    )

    assert result.unit_direction[0, 0, 0, 0].item() == 0.0
    assert result.unit_direction[0, 0, 1, 0].item() != 0.0
    assert result.update[0, 0, 0, 0].item() == 0.0
    assert result.update[0, 0, 1, 0].item() != 0.0
    assert result.direction_epsilon == 0.1
    assert result.numerical_epsilon == 0.01

    with pytest.raises(RuntimeError, match="原始安全方向"):
        build_risk_bounded_update(
            branch_name="lf_content",
            direction=torch.zeros((1, 1, 1, 2)),
            effective_budget=torch.ones((1, 1, 2)),
            nominal_strength=1.0,
            budget_ceiling=1.0,
            direction_epsilon=0.1,
            numerical_epsilon=0.01,
        )
    with pytest.raises(RuntimeError, match="零预算位置存在方向泄漏"):
        build_risk_bounded_update(
            branch_name="lf_content",
            direction=torch.tensor([[[[0.2, 1.0]]]]),
            effective_budget=torch.tensor([[[0.0, 1.0]]]),
            nominal_strength=1.0,
            budget_ceiling=1.0,
            direction_epsilon=0.1,
            numerical_epsilon=0.01,
        )
    with pytest.raises(RuntimeError, match="活动方向坐标"):
        build_risk_bounded_update(
            branch_name="lf_content",
            direction=torch.full((1, 1, 1, 2), 0.1),
            effective_budget=torch.ones((1, 1, 2)),
            nominal_strength=1.0,
            budget_ceiling=1.0,
            direction_epsilon=0.8,
            numerical_epsilon=0.01,
        )
    with pytest.raises(RuntimeError, match="最终步长"):
        build_risk_bounded_update(
            branch_name="lf_content",
            direction=torch.ones((1, 1, 1, 1)),
            effective_budget=torch.tensor([[[0.01]]]),
            nominal_strength=1.0,
            budget_ceiling=1.0,
            direction_epsilon=1e-12,
            numerical_epsilon=0.01,
        )


@pytest.mark.quick
def test_risk_bounded_direction_support_is_invariant_to_input_scaling() -> None:
    """活动集必须由单位方向定义, 原始方向等比例缩放不得改变结果。"""

    direction = torch.tensor([[[[0.2, 1.0], [0.01, -0.4]]]])
    budget = torch.tensor([[[1.0, 0.5], [0.0, 0.75]]])
    results = [
        build_risk_bounded_update(
            branch_name="lf_content",
            direction=direction * scale,
            effective_budget=budget,
            nominal_strength=0.5,
            budget_ceiling=1.0,
            direction_epsilon=0.05,
            numerical_epsilon=1e-12,
        )
        for scale in (0.25, 64.0)
    ]

    torch.testing.assert_close(
        results[0].unit_direction,
        results[1].unit_direction,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        results[0].amplitude_envelope,
        results[1].amplitude_envelope,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        results[0].update,
        results[1].update,
        rtol=0.0,
        atol=0.0,
    )
    assert results[0].unit_direction[0, 0, 1, 0].item() == 0.0


@pytest.mark.quick
def test_direction_epsilon_only_clears_zero_budget_leakage() -> None:
    """正预算小坐标保留在单位方向中, 但不进入 alpha 活动集合。"""

    result = build_risk_bounded_update(
        branch_name="lf_content",
        direction=torch.tensor([[[[1.0, 0.01]]]]),
        effective_budget=torch.ones((1, 1, 2)),
        nominal_strength=1.0,
        budget_ceiling=1.0,
        direction_epsilon=0.05,
        numerical_epsilon=1e-12,
    )

    assert result.unit_direction[0, 0, 0, 1].item() != 0.0
    assert torch.all(result.update.abs() <= result.amplitude_envelope)


@pytest.mark.quick
def test_risk_bounded_update_is_batch_isolated_and_reuses_nchw_budget() -> None:
    """每个样本必须独立归一化和限幅, NCHW 预算不得跨样本或通道漂移。"""

    direction = torch.tensor(
        [
            [[[1.0, 0.0], [2.0, 0.0]], [[0.5, 0.0], [1.0, 0.0]]],
            [[[0.0, 3.0], [0.0, 1.0]], [[0.0, 1.5], [0.0, 0.5]]],
        ],
        dtype=torch.float32,
    )
    spatial_budget = torch.tensor(
        [
            [[0.5, 0.0], [1.0, 0.0]],
            [[0.0, 1.0], [0.0, 0.25]],
        ],
        dtype=torch.float32,
    )
    nchw_budget = spatial_budget.unsqueeze(1).repeat(1, 2, 1, 1).contiguous()
    together = build_risk_bounded_update(
        branch_name="tail_robust",
        direction=direction,
        effective_budget=nchw_budget,
        nominal_strength=torch.tensor([1.0, 2.0]),
        budget_ceiling=1.0,
    )

    assert together.effective_budget is nchw_budget
    for sample_index, nominal in enumerate((1.0, 2.0)):
        separate = build_risk_bounded_update(
            branch_name="tail_robust",
            direction=direction[sample_index : sample_index + 1],
            effective_budget=spatial_budget[sample_index : sample_index + 1],
            nominal_strength=nominal,
            budget_ceiling=1.0,
        )
        assert torch.equal(
            together.unit_direction[sample_index : sample_index + 1],
            separate.unit_direction,
        )
        assert torch.equal(
            together.amplitude_envelope[sample_index : sample_index + 1],
            separate.amplitude_envelope,
        )
        assert torch.equal(
            together.update[sample_index : sample_index + 1],
            separate.update,
        )


@pytest.mark.quick
def test_rescale_risk_bounded_update_only_allows_smaller_fixed_direction_step() -> None:
    """注意力回溯只能缩小原风险允许步长, 不能改变方向或放大。"""

    result = build_risk_bounded_update(
        branch_name="attention_geometry",
        direction=torch.ones((1, 1, 2, 2)),
        effective_budget=torch.ones((1, 2, 2)),
        nominal_strength=1.0,
        budget_ceiling=1.0,
    )
    smaller = rescale_risk_bounded_update(
        result,
        result.applied_strength * 0.25,
    )

    assert smaller.unit_direction is result.unit_direction
    assert torch.equal(smaller.update, result.update * 0.25)
    assert torch.equal(smaller.risk_scale_factor, result.risk_scale_factor * 0.25)
    assert float(smaller.maximum_envelope_ratio.item()) <= float(
        result.maximum_envelope_ratio.item()
    )
    with pytest.raises(ValueError, match="不允许放大"):
        rescale_risk_bounded_update(
            result,
            result.applied_strength * 1.01,
        )
    with pytest.raises(ValueError, match="numerical_epsilon"):
        rescale_risk_bounded_update(
            result,
            torch.full_like(result.applied_strength, result.numerical_epsilon),
        )


@pytest.mark.quick
@pytest.mark.parametrize("seed", (7, 19, 31, 46, 47, 48))
def test_random_risk_bound_remains_exact_under_float32_rounding(seed: int) -> None:
    """随机方向在正式零容差下也不得因 float32 乘法舍入越过包络。"""

    generator = torch.Generator(device="cpu").manual_seed(seed)
    direction = torch.randn((3, 4, 8, 8), generator=generator)
    budget = torch.rand((3, 8, 8), generator=generator)
    zero_mask = budget < 0.1
    budget[zero_mask] = 0.0
    direction = direction.masked_fill(zero_mask.unsqueeze(1), 0.0)
    result = build_risk_bounded_update(
        branch_name="lf_content",
        direction=direction,
        effective_budget=budget,
        nominal_strength=torch.rand((3,), generator=generator) + 0.1,
        budget_ceiling=1.0,
    )

    assert torch.all(result.update.abs() <= result.amplitude_envelope)
    assert torch.all(result.maximum_envelope_ratio <= 1.0)


def _full_budget_update(branch_name: str, strength: float) -> object:
    """构造用于合成性质测试的单坐标风险有界更新。"""

    return build_risk_bounded_update(
        branch_name=branch_name,
        direction=torch.ones((1, 1, 1, 1)),
        effective_budget=torch.ones((1, 1, 1)),
        nominal_strength=strength,
        budget_ceiling=1.0,
    )


@pytest.mark.quick
def test_quantized_composition_accepts_ordered_nonempty_branch_subset() -> None:
    """消融可省略分支, 但合成顺序记录必须保持完整冻结顺序。"""

    original = torch.zeros((1, 1, 1, 1), dtype=torch.float32)
    lf_update = _full_budget_update("lf_content", 0.2)
    attention_update = _full_budget_update("attention_geometry", 0.1)
    candidate = build_quantized_composition_candidate(
        original_latent=original,
        branch_updates={
            "lf_content": lf_update,
            "attention_geometry": attention_update,
        },
        common_scale=0.5,
        backtracking_factor=0.5,
        backtracking_step_count=1,
    )

    expected_update = 0.5 * (lf_update.update + attention_update.update)
    expected_envelope = 0.5 * (
        lf_update.amplitude_envelope + attention_update.amplitude_envelope
    )
    assert candidate.composition_order == QUANTIZED_COMPOSITION_ORDER
    assert torch.equal(candidate.float32_combined_update, expected_update)
    assert torch.equal(candidate.combined_envelope, expected_envelope)
    assert torch.equal(candidate.candidate_latent, original + expected_update)
    assert candidate.envelope_ready is True
    record = candidate.to_record()
    assert record["combined_update_content_sha256"] == tensor_content_sha256(
        candidate.float32_combined_update
    )
    assert record["combined_budget_envelope_content_sha256"] == (
        tensor_content_sha256(candidate.combined_envelope)
    )
    assert record["quantized_write_update_content_sha256"] == tensor_content_sha256(
        candidate.written_update
    )
    assert record["quantized_write_composition_order"] == list(
        QUANTIZED_COMPOSITION_ORDER
    )
    assert record["quantized_write_active_branch_order"] == [
        "lf_content",
        "attention_geometry",
    ]
    assert record["quantized_write_backtracking_factor"] == 0.5
    assert record["quantized_write_backtracking_step_count"] == 1
    assert record["quantized_composition_evidence_digest"] == (
        recompute_quantized_composition_evidence_digest(record)
    )

    with pytest.raises(ValueError, match="非空子集"):
        build_quantized_composition_candidate(
            original_latent=original,
            branch_updates={},
        )
    with pytest.raises(ValueError, match="未知分支"):
        build_quantized_composition_candidate(
            original_latent=original,
            branch_updates={"unknown": lf_update},
        )
    with pytest.raises(ValueError, match="映射角色"):
        build_quantized_composition_candidate(
            original_latent=original,
            branch_updates={"lf_content": attention_update},
        )


@pytest.mark.quick
def test_quantized_composition_evidence_binds_real_three_branch_tensors() -> None:
    """合成证据必须绑定三分支真实 Tensor、单次 cast 结果和回溯轨迹。"""

    original = torch.tensor(
        [[[[1.0, -1.0], [0.5, -0.5]]]],
        dtype=torch.float16,
    )
    branch_directions = {
        "lf_content": torch.tensor([[[[1.0, 0.0], [0.0, 0.0]]]]),
        "tail_robust": torch.tensor([[[[0.0, -1.0], [0.0, 0.0]]]]),
        "attention_geometry": torch.tensor([[[[0.0, 0.0], [1.0, 0.0]]]]),
    }
    branch_strengths = {
        "lf_content": 0.125,
        "tail_robust": 0.0625,
        "attention_geometry": 0.03125,
    }
    branches = {
        role: build_risk_bounded_update(
            branch_name=role,
            direction=direction,
            effective_budget=torch.ones((1, 2, 2)),
            nominal_strength=branch_strengths[role],
            budget_ceiling=1.0,
        )
        for role, direction in branch_directions.items()
    }
    candidate = build_quantized_composition_candidate(
        original_latent=original,
        branch_updates=branches,
        common_scale=0.25,
        backtracking_factor=0.5,
        backtracking_step_count=2,
    )
    record = candidate.to_record()

    assert record["quantized_composition_evidence_version"] == (
        QUANTIZED_COMPOSITION_EVIDENCE_VERSION
    )
    assert record["quantized_write_original_latent_content_sha256"] == (
        tensor_content_sha256(original)
    )
    assert record["quantized_write_candidate_latent_content_sha256"] == (
        tensor_content_sha256(candidate.candidate_latent)
    )
    assert record["quantized_write_update_content_sha256"] == (
        tensor_content_sha256(candidate.written_update)
    )
    assert record["quantized_write_update_dtype"] == str(
        candidate.written_update.dtype
    )
    assert record["quantized_write_update_shape"] == [1, 1, 2, 2]
    assert record["quantized_write_active_branch_order"] == list(
        QUANTIZED_COMPOSITION_ORDER
    )
    for role in QUANTIZED_COMPOSITION_ORDER:
        identity = record["quantized_write_branch_content_identities"][role]
        assert identity["branch_written_update_content_sha256"] == (
            tensor_content_sha256(branches[role].update)
        )
        assert identity["branch_budget_envelope_content_sha256"] == (
            tensor_content_sha256(branches[role].amplitude_envelope)
        )
    assert record["combined_update_content_sha256"] == tensor_content_sha256(
        candidate.float32_combined_update
    )
    assert record["combined_budget_envelope_content_sha256"] == (
        tensor_content_sha256(candidate.combined_envelope)
    )
    assert record["quantized_write_common_scale"] == 0.25
    assert record["quantized_write_backtracking_factor"] == 0.5
    assert record["quantized_write_backtracking_step_count"] == 2
    assert record["quantized_write_maximum_envelope_ratio"] == pytest.approx(
        float(candidate.maximum_envelope_ratio.item())
    )
    assert record["quantized_composition_evidence_digest"] == (
        recompute_quantized_composition_evidence_digest(record)
    )


@pytest.mark.quick
def test_float32_composition_casts_once_instead_of_associative_latent_writes() -> None:
    """三个更新必须先在 float32 合成, 不得分两次写入低精度 latent。"""

    original = torch.ones((1, 1, 1, 1), dtype=torch.float16)
    updates = {
        "attention_geometry": torch.full((1, 1, 1, 1), 0.0004),
        "tail_robust": torch.full((1, 1, 1, 1), 0.0002),
        "lf_content": torch.full((1, 1, 1, 1), 0.0002),
    }
    combined, candidate, written = compose_ordered_float32_update_once(
        original_latent=original,
        branch_update_tensors=updates,
        common_scale=1.0,
    )

    base_update = updates["lf_content"] + updates["tail_robust"]
    wrong_base_latent = (
        original.float() + base_update
    ).to(dtype=original.dtype)
    wrong_sequential_write_candidate = (
        wrong_base_latent.float() + updates["attention_geometry"]
    ).to(dtype=original.dtype)

    assert combined.item() == pytest.approx(0.0008)
    assert candidate.item() == pytest.approx(1.0009765625)
    assert written.item() == pytest.approx(0.0009765625)
    assert wrong_sequential_write_candidate.item() == pytest.approx(1.0)
    assert not torch.equal(candidate, wrong_sequential_write_candidate)


@pytest.mark.quick
def test_float32_composition_uses_frozen_role_order_not_mapping_order() -> None:
    """float32 非结合反例必须暴露分支重排, 即使输入映射顺序相反。"""

    original = torch.zeros((1, 1, 1, 1), dtype=torch.float32)
    updates = {
        "attention_geometry": torch.full((1, 1, 1, 1), 1.0),
        "tail_robust": torch.full((1, 1, 1, 1), -1e20),
        "lf_content": torch.full((1, 1, 1, 1), 1e20),
    }
    combined, candidate, written = compose_ordered_float32_update_once(
        original_latent=original,
        branch_update_tensors=updates,
        common_scale=1.0,
    )
    reordered = updates["lf_content"] + (
        updates["tail_robust"] + updates["attention_geometry"]
    )

    assert combined.item() == pytest.approx(1.0)
    assert candidate.item() == pytest.approx(1.0)
    assert written.item() == pytest.approx(1.0)
    assert reordered.item() == pytest.approx(0.0)


@pytest.mark.quick
@pytest.mark.parametrize(
    ("field_name", "replacement"),
    (
        ("quantized_write_original_latent_content_sha256", "0" * 64),
        ("quantized_write_candidate_latent_content_sha256", "1" * 64),
        ("quantized_write_update_content_sha256", "2" * 64),
        ("quantized_write_update_dtype", "torch.float64"),
        ("quantized_write_update_shape", [1, 1, 1, 2]),
        ("combined_update_content_sha256", "3" * 64),
        ("combined_budget_envelope_content_sha256", "4" * 64),
        ("quantized_write_maximum_envelope_ratio", 0.125),
    ),
)
def test_quantized_composition_evidence_detects_record_tampering(
    field_name: str,
    replacement: object,
) -> None:
    """任一被绑定的记录字段变化后都不得继续匹配原证据摘要。"""

    candidate = build_quantized_composition_candidate(
        original_latent=torch.zeros((1, 1, 1, 1), dtype=torch.float32),
        branch_updates={"lf_content": _full_budget_update("lf_content", 0.25)},
        common_scale=0.5,
        backtracking_factor=0.5,
        backtracking_step_count=1,
    )
    record = candidate.to_record()
    original_digest = record["quantized_composition_evidence_digest"]
    tampered = deepcopy(record)
    tampered[field_name] = replacement

    assert recompute_quantized_composition_evidence_digest(tampered) != (
        original_digest
    )


@pytest.mark.quick
def test_quantized_composition_evidence_detects_branch_tensor_tampering() -> None:
    """活动分支 update 或 envelope 摘要变化后必须破坏合成证据。"""

    candidate = build_quantized_composition_candidate(
        original_latent=torch.zeros((1, 1, 1, 1), dtype=torch.float32),
        branch_updates={"lf_content": _full_budget_update("lf_content", 0.25)},
        common_scale=1.0,
        backtracking_factor=0.5,
        backtracking_step_count=0,
    )
    record = candidate.to_record()
    original_digest = record["quantized_composition_evidence_digest"]
    for field_name, replacement in (
        ("branch_written_update_content_sha256", "5" * 64),
        ("branch_budget_envelope_content_sha256", "6" * 64),
    ):
        tampered = deepcopy(record)
        tampered["quantized_write_branch_content_identities"]["lf_content"][
            field_name
        ] = replacement
        assert recompute_quantized_composition_evidence_digest(tampered) != (
            original_digest
        )


@pytest.mark.quick
def test_quantized_composition_evidence_binds_trace_and_active_branch_protocol() -> None:
    """合法但不同的活动分支、缩放轨迹或就绪结论必须产生不同摘要。"""

    candidate = build_quantized_composition_candidate(
        original_latent=torch.zeros((1, 1, 1, 1), dtype=torch.float32),
        branch_updates={
            "lf_content": _full_budget_update("lf_content", 0.25),
            "tail_robust": _full_budget_update("tail_robust", 0.125),
        },
        common_scale=0.5,
        backtracking_factor=0.5,
        backtracking_step_count=1,
    )
    record = candidate.to_record()
    original_digest = record["quantized_composition_evidence_digest"]

    reduced_branches = deepcopy(record)
    reduced_branches["quantized_write_active_branch_order"] = ["lf_content"]
    del reduced_branches["quantized_write_branch_content_identities"][
        "tail_robust"
    ]
    assert recompute_quantized_composition_evidence_digest(
        reduced_branches
    ) != original_digest

    different_trace = deepcopy(record)
    different_trace["quantized_write_common_scale"] = 0.25
    different_trace["quantized_write_backtracking_step_count"] = 2
    assert recompute_quantized_composition_evidence_digest(
        different_trace
    ) != original_digest

    different_factor = deepcopy(record)
    different_factor["quantized_write_common_scale"] = 0.25
    different_factor["quantized_write_backtracking_factor"] = 0.25
    assert recompute_quantized_composition_evidence_digest(
        different_factor
    ) != original_digest

    different_readiness = deepcopy(record)
    different_readiness["quantized_write_budget_envelope_ready"] = False
    assert recompute_quantized_composition_evidence_digest(
        different_readiness
    ) != original_digest

    for field_name, replacement in (
        ("quantized_composition_evidence_version", "unsupported"),
        ("tensor_content_digest_version", "unsupported"),
        ("quantized_write_composition_order", ["tail_robust", "lf_content"]),
        ("quantized_write_active_branch_order", ["tail_robust", "lf_content"]),
    ):
        invalid = deepcopy(record)
        invalid[field_name] = replacement
        with pytest.raises(ValueError):
            recompute_quantized_composition_evidence_digest(invalid)


@pytest.mark.quick
def test_quantized_composition_rejects_inconsistent_backtracking_trace() -> None:
    """构造器与纯记录重算都必须拒绝缩放因子和步数不一致。"""

    original = torch.zeros((1, 1, 1, 1), dtype=torch.float32)
    branch = _full_budget_update("lf_content", 0.25)
    with pytest.raises(ValueError, match="必须精确等于"):
        build_quantized_composition_candidate(
            original_latent=original,
            branch_updates={"lf_content": branch},
            common_scale=0.5,
            backtracking_factor=0.5,
            backtracking_step_count=2,
        )
    with pytest.raises(ValueError, match="非负整数次幂"):
        build_quantized_composition_candidate(
            original_latent=original,
            branch_updates={"lf_content": branch},
            common_scale=0.3,
            backtracking_factor=0.5,
        )

    record = build_quantized_composition_candidate(
        original_latent=original,
        branch_updates={"lf_content": branch},
        common_scale=0.5,
        backtracking_factor=0.5,
        backtracking_step_count=1,
    ).to_record()
    record["quantized_write_backtracking_step_count"] = 2
    with pytest.raises(ValueError, match="必须精确等于"):
        recompute_quantized_composition_evidence_digest(record)


@pytest.mark.quick
def test_quantized_overshoot_requires_common_backtracking_candidate() -> None:
    """float16 舍入越过包络时首候选必须失败, 共同缩小后才可形成候选。"""

    original = torch.ones((1, 1, 1, 1), dtype=torch.float16)
    branch = _full_budget_update("lf_content", 0.0005)
    first = build_quantized_composition_candidate(
        original_latent=original,
        branch_updates={"lf_content": branch},
    )
    first_only = tuple(
        iter_quantized_composition_candidates(
            original_latent=original,
            branch_updates={"lf_content": branch},
            maximum_steps=0,
        )
    )
    candidates = tuple(
        iter_quantized_composition_candidates(
            original_latent=original,
            branch_updates={"lf_content": branch},
            backtracking_factor=0.5,
            maximum_steps=1,
        )
    )

    assert first.envelope_ready is False
    assert math.isfinite(float(first.maximum_envelope_ratio.item()))
    assert float(first.maximum_envelope_ratio.item()) > 1.0
    assert len(first_only) == 1 and first_only[0].envelope_ready is False
    assert candidates[0].envelope_ready is False
    assert candidates[1].common_scale == pytest.approx(0.5)
    assert candidates[1].envelope_ready is True
    assert torch.count_nonzero(candidates[1].written_update).item() == 0
