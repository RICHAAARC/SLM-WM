"""验证正式内容载体更新核的纯 CPU 公式、角色与失败关闭边界。"""

from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError, fields, replace
import inspect
from pathlib import Path
import struct
from typing import Any

import pytest
import torch

from main.core.keyed_prg import KEYED_PRG_VERSION
from main.methods.carrier.content_update import (
    ContentCarrierUpdateResult,
    build_content_carrier_update,
)
from main.methods.carrier.high_frequency_tail import (
    HighFrequencyTailCarrierTemplate,
    build_high_frequency_tail_template,
)
from main.methods.carrier.low_frequency import (
    LowFrequencyCarrierTemplate,
    build_low_frequency_template,
)
from main.methods.content.routing import ContentRoutingResult, route_content_carriers
import main.methods.carrier as carrier_package
import main.methods.carrier.content_update as update_module


pytestmark = pytest.mark.unit

_MODEL_DIGEST = "a" * 64
_KEY_DIGEST = "b" * 64
_ROUTING_DIGEST = "c" * 64
_LF_TEMPLATE_DIGEST = "d" * 64
_HF_TEMPLATE_DIGEST = "e" * 64
_SHAPE = (1, 2, 2, 3)
_ROLES = (
    "full_dual_chain",
    "uniform_content_routing",
    "lf_only_content",
    "hf_tail_only_content",
    "content_chain_only",
    "geometry_recovery_without_embedded_sync",
)


def _unit(value: torch.Tensor) -> torch.Tensor:
    """以float32完整CHW范数构造测试用近单位模板。"""

    resolved = value.to(dtype=torch.float32)
    return resolved / torch.linalg.vector_norm(resolved.reshape(-1))


def _manual_templates(
    *,
    shape: tuple[int, int, int, int] = _SHAPE,
    device: torch.device | str = "cpu",
) -> tuple[LowFrequencyCarrierTemplate, HighFrequencyTailCarrierTemplate]:
    """构造字段完整但不冒充真实builder身份的最小模板。"""

    element_count = shape[1] * shape[2] * shape[3]
    lf_values = torch.arange(1, element_count + 1, dtype=torch.float32)
    lf = _unit(lf_values.reshape(shape)).to(device=device)
    hf_values = torch.arange(element_count, 0, -1, dtype=torch.float32)
    signs = torch.where(
        torch.arange(element_count) % 2 == 0,
        torch.tensor(1.0),
        torch.tensor(-1.0),
    )
    hf = _unit((hf_values * signs).reshape(shape)).to(device=device)
    return (
        LowFrequencyCarrierTemplate(
            template=lf,
            latent_shape=shape,
            scoring_key_identity_digest=_KEY_DIGEST,
            model_identity_digest=_MODEL_DIGEST,
            prg_version=KEYED_PRG_VERSION,
            prg_domain="lf_content",
            filter_identity_digest="f" * 64,
            template_digest=_LF_TEMPLATE_DIGEST,
        ),
        HighFrequencyTailCarrierTemplate(
            template=hf,
            latent_shape=shape,
            scoring_key_identity_digest=_KEY_DIGEST,
            model_identity_digest=_MODEL_DIGEST,
            prg_version=KEYED_PRG_VERSION,
            prg_domain="hf_tail_robust",
            high_pass_identity_digest="1" * 64,
            selected_element_count=max(1, (element_count + 4) // 5),
            template_digest=_HF_TEMPLATE_DIGEST,
        ),
    )


def _routing(
    *,
    spatial_shape: tuple[int, int] = (_SHAPE[2], _SHAPE[3]),
) -> ContentRoutingResult:
    """通过正式路由公式构造非平凡分数掩码。"""

    height, width = spatial_shape
    base = torch.arange(height * width, dtype=torch.float32).reshape(
        1, 1, height, width
    )
    denominator = float(max(1, height * width - 1))
    saliency = 0.10 + 0.20 * base / denominator
    texture = 0.15 + 0.65 * torch.flip(base, dims=(-1,)) / denominator
    response = 0.05 + 0.15 * base / denominator
    sensitivity = 0.08 + 0.10 * torch.flip(base, dims=(-2,)) / denominator
    return route_content_carriers(saliency, texture, response, sensitivity)


def _inputs(
    *,
    dtype: torch.dtype = torch.float32,
) -> tuple[
    torch.Tensor,
    ContentRoutingResult,
    LowFrequencyCarrierTemplate,
    HighFrequencyTailCarrierTemplate,
]:
    """返回正常公式测试使用的全部显式输入。"""

    latent = torch.tensor(
        [
            [
                [[0.25, -0.50, 1.00], [0.75, -1.25, 0.40]],
                [[-0.30, 0.80, -0.90], [1.10, 0.20, -0.60]],
            ]
        ],
        dtype=dtype,
    )
    lf, hf = _manual_templates()
    return latent, _routing(), lf, hf


def _build(
    latent: torch.Tensor,
    routing: ContentRoutingResult,
    lf: LowFrequencyCarrierTemplate,
    hf: HighFrequencyTailCarrierTemplate,
    role: str = "full_dual_chain",
    multiplier: float = 1.0,
) -> ContentCarrierUpdateResult:
    """使用全部keyword-only正式参数调用公开构造器。"""

    return build_content_carrier_update(
        current_scheduler_latent=latent,
        routing=routing,
        lf_template=lf,
        hf_tail_template=hf,
        method_role=role,
        content_strength_common_multiplier=multiplier,
    )


def test_public_contract_is_frozen_exact_and_not_package_exported() -> None:
    """公开字段、keyword-only签名和隔离模块边界不得扩张。"""

    assert update_module.__all__ == [
        "ContentCarrierUpdateResult",
        "build_content_carrier_update",
    ]
    assert [field.name for field in fields(ContentCarrierUpdateResult)] == [
        "geometry_capacity_map",
        "lf_direction",
        "hf_tail_direction",
        "lf_update",
        "hf_tail_update",
        "content_only_latent_float32",
        "latent_l2",
        "lf_nominal_strength",
        "hf_tail_nominal_strength",
        "method_role",
    ]
    signature = inspect.signature(build_content_carrier_update)
    assert list(signature.parameters) == [
        "current_scheduler_latent",
        "routing",
        "lf_template",
        "hf_tail_template",
        "method_role",
        "content_strength_common_multiplier",
    ]
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY
        for parameter in signature.parameters.values()
    )
    assert not hasattr(carrier_package, "ContentCarrierUpdateResult")
    assert not hasattr(carrier_package, "build_content_carrier_update")
    assert update_module._TEMPLATE_UNIT_NORM_RTOL == 1.0e-5
    assert update_module._TEMPLATE_UNIT_NORM_ATOL == 1.0e-6

    latent, routing, lf, hf = _inputs()
    result = _build(latent, routing, lf, hf)
    with pytest.raises(FrozenInstanceError):
        result.method_role = "lf_only_content"  # type: ignore[misc]


@pytest.mark.parametrize("multiplier", (0.75, 1.0, 1.25))
def test_common_multiplier_scales_only_two_nominal_content_strengths(
    multiplier: float,
) -> None:
    """冻结三候选只共同缩放LF/HF名义强度且不改变0.70/0.30职责。"""

    latent, routing, lf, hf = _inputs()
    result = _build(latent, routing, lf, hf, multiplier=multiplier)
    latent_float32 = latent.detach().to(dtype=torch.float32)
    latent_l2 = torch.linalg.vector_norm(latent_float32.reshape(-1))
    multiplier_tensor = latent_float32.new_tensor(multiplier)
    expected_lf = latent_l2 * (
        latent_float32.new_tensor(0.0025) * multiplier_tensor
    )
    expected_hf = latent_l2 * (
        latent_float32.new_tensor(0.0015) * multiplier_tensor
    )

    assert result.lf_nominal_strength == float(expected_lf.item())
    assert result.hf_tail_nominal_strength == float(expected_hf.item())
    assert torch.equal(result.lf_update, result.lf_direction * expected_lf)
    assert torch.equal(
        result.hf_tail_update,
        result.hf_tail_direction * expected_hf,
    )


def test_one_point_two_five_multiplier_uses_device_float32_bit_order() -> None:
    """1.25候选不得先在Python binary64中预乘再转换为float32。"""

    latent, routing, lf, hf = _inputs()
    result = _build(latent, routing, lf, hf, multiplier=1.25)
    latent_float32 = latent.detach().to(dtype=torch.float32)
    latent_l2 = torch.linalg.vector_norm(latent_float32.reshape(-1))
    multiplier = latent_float32.new_tensor(1.25)
    lf_relative = latent_float32.new_tensor(0.0025) * multiplier
    hf_relative = latent_float32.new_tensor(0.0015) * multiplier
    binary64_first_lf = latent_float32.new_tensor(0.0025 * 1.25)
    binary64_first_hf = latent_float32.new_tensor(0.0015 * 1.25)

    assert struct.pack(">f", float(lf_relative.item())) != struct.pack(
        ">f", float(binary64_first_lf.item())
    )
    assert struct.pack(">f", float(hf_relative.item())) != struct.pack(
        ">f", float(binary64_first_hf.item())
    )
    assert struct.pack(
        ">f", result.lf_nominal_strength
    ) == struct.pack(">f", float((latent_l2 * lf_relative).item()))
    assert struct.pack(
        ">f", result.hf_tail_nominal_strength
    ) == struct.pack(">f", float((latent_l2 * hf_relative).item()))


@pytest.mark.parametrize("invalid", (True, 1, 0.5, 1.5, float("nan")))
def test_common_multiplier_rejects_values_outside_frozen_candidates(
    invalid: Any,
) -> None:
    """业务入口不得把任意倍率或bool解释为正式敏感性候选。"""

    latent, routing, lf, hf = _inputs()
    with pytest.raises(ValueError, match="0.75、1.0 或 1.25"):
        build_content_carrier_update(
            current_scheduler_latent=latent,
            routing=routing,
            lf_template=lf,
            hf_tail_template=hf,
            method_role="full_dual_chain",
            content_strength_common_multiplier=invalid,
        )


@pytest.mark.parametrize(
    ("role", "lf_active", "hf_active", "uniform"),
    (
        ("full_dual_chain", True, True, False),
        ("uniform_content_routing", True, True, True),
        ("lf_only_content", True, False, False),
        ("hf_tail_only_content", False, True, False),
        ("content_chain_only", True, True, False),
        ("geometry_recovery_without_embedded_sync", True, True, False),
    ),
)
def test_six_roles_follow_independent_float32_formula(
    role: str,
    lf_active: bool,
    hf_active: bool,
    uniform: bool,
) -> None:
    """六角色只门控update，统一路由精确令A/LF/HF掩码全一。"""

    latent, routing, lf, hf = _inputs(dtype=torch.float64)
    result = _build(latent, routing, lf, hf, role)
    latent_float32 = latent.detach().to(dtype=torch.float32)
    latent_l2 = torch.linalg.vector_norm(latent_float32.reshape(-1))
    lf_strength = latent_l2 * latent_float32.new_tensor(0.0025)
    hf_strength = latent_l2 * latent_float32.new_tensor(0.0015)
    capacity = (
        torch.ones_like(routing.writable_capacity_map)
        if uniform
        else routing.writable_capacity_map
    )
    lf_mask = torch.ones_like(routing.lf_mask) if uniform else routing.lf_mask
    hf_mask = (
        torch.ones_like(routing.hf_tail_mask)
        if uniform
        else routing.hf_tail_mask
    )
    expected_lf_direction = lf_mask * lf.template
    expected_hf_direction = hf_mask * hf.template
    expected_lf_update = (
        expected_lf_direction * lf_strength
        if lf_active
        else torch.zeros_like(expected_lf_direction)
    )
    expected_hf_update = (
        expected_hf_direction * hf_strength
        if hf_active
        else torch.zeros_like(expected_hf_direction)
    )
    expected_content = latent_float32 + expected_lf_update
    expected_content = expected_content + expected_hf_update

    for actual, expected in (
        (result.geometry_capacity_map, capacity),
        (result.lf_direction, expected_lf_direction),
        (result.hf_tail_direction, expected_hf_direction),
        (result.lf_update, expected_lf_update),
        (result.hf_tail_update, expected_hf_update),
        (result.content_only_latent_float32, expected_content),
    ):
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
        assert actual.dtype == torch.float32
        assert actual.device == latent.device
        assert bool(torch.isfinite(actual).all())
    assert result.latent_l2 == float(latent_l2.item())
    assert result.lf_nominal_strength == float(lf_strength.item())
    assert result.hf_tail_nominal_strength == float(hf_strength.item())
    assert result.method_role == role
    if not lf_active:
        assert torch.count_nonzero(result.lf_update).item() == 0
    if not hf_active:
        assert torch.count_nonzero(result.hf_tail_update).item() == 0


def test_fractional_masks_broadcast_without_post_mask_normalization() -> None:
    """空间掩码跨channel广播且不被整体重新单位化。"""

    latent, routing, lf, hf = _inputs()
    result = _build(latent, routing, lf, hf)
    expected_lf = routing.lf_mask * lf.template
    expected_hf = routing.hf_tail_mask * hf.template
    torch.testing.assert_close(result.lf_direction, expected_lf, rtol=0.0, atol=0.0)
    torch.testing.assert_close(result.hf_tail_direction, expected_hf, rtol=0.0, atol=0.0)
    assert float(torch.linalg.vector_norm(result.lf_direction).item()) < 1.0
    assert float(torch.linalg.vector_norm(result.hf_tail_direction).item()) < 1.0
    assert not torch.allclose(
        torch.linalg.vector_norm(result.lf_direction),
        torch.ones((), dtype=torch.float32),
        rtol=1.0e-5,
        atol=1.0e-6,
    )


def test_zero_masks_are_valid_and_produce_exact_zero_updates() -> None:
    """零容量是合法内容事实，不在共同写回前伪造非零失败。"""

    latent, routing, lf, hf = _inputs()
    zeros = torch.zeros_like(routing.lf_mask)
    zero_routing = ContentRoutingResult(
        writable_capacity_map=zeros,
        lf_mask=zeros,
        hf_tail_mask=zeros,
        routing_identity_digest=_ROUTING_DIGEST,
    )
    result = _build(latent, zero_routing, lf, hf)
    for value in (
        result.geometry_capacity_map,
        result.lf_direction,
        result.hf_tail_direction,
        result.lf_update,
        result.hf_tail_update,
    ):
        assert torch.count_nonzero(value).item() == 0
    torch.testing.assert_close(
        result.content_only_latent_float32,
        latent.float(),
        rtol=0.0,
        atol=0.0,
    )


def test_real_formal_builders_pass_frozen_unit_norm_tolerance() -> None:
    """真实正式builder的65536元素float32舍入必须被consumer接受。"""

    shape = (1, 16, 64, 64)
    reference = torch.zeros(shape, dtype=torch.float32)
    lf = build_low_frequency_template(
        reference,
        "registered-key",
        _MODEL_DIGEST,
        prg_version=KEYED_PRG_VERSION,
    )
    hf = build_high_frequency_tail_template(
        reference,
        "registered-key",
        _MODEL_DIGEST,
        prg_version=KEYED_PRG_VERSION,
    )
    ones = torch.ones((1, 1, 64, 64), dtype=torch.float32)
    routing = ContentRoutingResult(ones, ones, ones, _ROUTING_DIGEST)
    result = _build(torch.ones(shape), routing, lf, hf)
    tolerance_upper_bound = 1.0 + 1.0e-6 + 1.0e-5
    for template, direction in (
        (lf.template, result.lf_direction),
        (hf.template, result.hf_tail_direction),
    ):
        norm = torch.linalg.vector_norm(template.detach().reshape(-1))
        assert torch.allclose(
            norm,
            torch.ones_like(norm),
            rtol=1.0e-5,
            atol=1.0e-6,
        )
        assert float(torch.linalg.vector_norm(direction).item()) <= (
            tolerance_upper_bound
        )

    for changed_lf, changed_hf in (
        (replace(lf, template=lf.template * 1.001), hf),
        (lf, replace(hf, template=hf.template * 1.001)),
    ):
        with pytest.raises(ValueError, match="单位L2"):
            _build(torch.ones(shape), routing, changed_lf, changed_hf)


def test_strengths_use_device_local_float32_before_python_item() -> None:
    """强度先按float32 scalar Tensor计算，不允许binary64中间乘法。"""

    latent = torch.tensor(
        [[[[9.0 / 37.0]], [[9.0 / 53.0]], [[1.0]]]],
        dtype=torch.float32,
    )
    shape = tuple(latent.shape)
    lf_tensor = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32).reshape(shape)
    hf_tensor = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32).reshape(shape)
    lf, hf = _manual_templates(shape=shape)
    lf = replace(lf, template=lf_tensor)
    hf = replace(hf, template=hf_tensor)
    ones = torch.ones((1, 1, 1, 1), dtype=torch.float32)
    routing = ContentRoutingResult(ones, ones, ones, _ROUTING_DIGEST)
    result = _build(latent, routing, lf, hf)

    latent_l2 = torch.linalg.vector_norm(latent.detach().float().reshape(-1))
    lf_strength = latent_l2 * latent.new_tensor(0.0025)
    hf_strength = latent_l2 * latent.new_tensor(0.0015)
    python_double_then_float32 = torch.tensor(
        float(latent_l2.item()) * 0.0025,
        dtype=torch.float32,
    )
    assert lf_strength.view(torch.int32).item() != (
        python_double_then_float32.view(torch.int32).item()
    )
    assert result.latent_l2 == float(latent_l2.item())
    assert result.lf_nominal_strength == float(lf_strength.item())
    assert result.hf_tail_nominal_strength == float(hf_strength.item())
    assert result.lf_update[0, 0, 0, 0].view(torch.int32).item() == (
        lf_strength.view(torch.int32).item()
    )
    assert result.hf_tail_update[0, 1, 0, 0].view(torch.int32).item() == (
        hf_strength.view(torch.int32).item()
    )


def test_common_unsupported_prg_fails_before_all_content_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """共同unsupported版本必须由共享validator在内容读取前拒绝。"""

    latent, routing, lf, hf = _inputs()
    lf = replace(lf, prg_version="unsupported-prg")
    hf = replace(hf, prg_version="unsupported-prg")
    calls: list[str] = []

    def reject(version: str) -> None:
        calls.append(version)
        raise ValueError("unsupported")

    monkeypatch.setattr(update_module, "require_supported_keyed_prg_version", reject)
    for name in (
        "_validate_routing_contents",
        "_validate_template_contents",
        "_build_float32_updates",
    ):
        monkeypatch.setattr(
            update_module,
            name,
            lambda *args, _name=name, **kwargs: pytest.fail(
                f"{_name} must not run"
            ),
        )
    with pytest.raises(ValueError, match="unsupported"):
        _build(latent, routing, lf, hf)
    assert calls == ["unsupported-prg"]


def test_mismatched_prg_versions_fail_before_shared_validator_or_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """跨模板版本不一致先于shared validator失败。"""

    latent, routing, lf, hf = _inputs()
    hf = replace(hf, prg_version="different-prg")
    monkeypatch.setattr(
        update_module,
        "require_supported_keyed_prg_version",
        lambda version: pytest.fail("shared validator must not run"),
    )
    monkeypatch.setattr(
        update_module,
        "_validate_routing_contents",
        lambda routing: pytest.fail("content must not be read"),
    )
    with pytest.raises(ValueError, match="PRG 版本不一致"):
        _build(latent, routing, lf, hf)


def test_supported_prg_is_validated_exactly_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """有效路径只调用一次共享PRG版本边界。"""

    latent, routing, lf, hf = _inputs()
    calls: list[str] = []
    shared_validator = update_module.require_supported_keyed_prg_version

    def record(version: str) -> None:
        calls.append(version)
        shared_validator(version)

    monkeypatch.setattr(update_module, "require_supported_keyed_prg_version", record)
    _build(latent, routing, lf, hf)
    assert calls == [KEYED_PRG_VERSION]


@pytest.mark.parametrize(
    "case",
    (
        "routing_digest",
        "routing_dtype",
        "lf_latent_shape_bool",
        "hf_domain",
        "model_mismatch",
        "key_mismatch",
        "filter_digest",
        "high_pass_digest",
        "selected_count_bool",
        "template_dtype",
    ),
)
def test_static_metadata_and_identity_fail_before_content(
    case: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """支持但职责错误的静态身份不得触发内容扫描。"""

    latent, routing, lf, hf = _inputs()
    if case == "routing_digest":
        routing = replace(routing, routing_identity_digest="A" * 64)
    elif case == "routing_dtype":
        routing = replace(routing, lf_mask=routing.lf_mask.double())
    elif case == "lf_latent_shape_bool":
        lf = replace(lf, latent_shape=(1, 2, 2, True))
    elif case == "hf_domain":
        hf = replace(hf, prg_domain="lf_content")  # type: ignore[arg-type]
    elif case == "model_mismatch":
        hf = replace(hf, model_identity_digest="2" * 64)
    elif case == "key_mismatch":
        hf = replace(hf, scoring_key_identity_digest="3" * 64)
    elif case == "filter_digest":
        lf = replace(lf, filter_identity_digest="invalid")
    elif case == "high_pass_digest":
        hf = replace(hf, high_pass_identity_digest="invalid")
    elif case == "selected_count_bool":
        hf = replace(hf, selected_element_count=True)
    elif case == "template_dtype":
        hf = replace(hf, template=hf.template.double())
    else:  # pragma: no cover - exhaustive parameter guard
        raise AssertionError(case)

    monkeypatch.setattr(
        update_module,
        "_validate_routing_contents",
        lambda routing: pytest.fail("routing content must not be read"),
    )
    monkeypatch.setattr(
        update_module,
        "_validate_template_contents",
        lambda *args, **kwargs: pytest.fail("template content must not be read"),
    )
    with pytest.raises((TypeError, ValueError)):
        _build(latent, routing, lf, hf)


@pytest.mark.parametrize("case", ("nonfinite", "below_zero", "above_one"))
def test_invalid_routing_contents_fail_closed(case: str) -> None:
    """三个实际路由图必须全部finite且位于闭区间。"""

    latent, routing, lf, hf = _inputs()
    changed = routing.lf_mask.clone()
    if case == "nonfinite":
        changed.reshape(-1)[0] = float("nan")
    elif case == "below_zero":
        changed.reshape(-1)[0] = -0.01
    else:
        changed.reshape(-1)[0] = 1.01
    with pytest.raises(ValueError):
        _build(latent, replace(routing, lf_mask=changed), lf, hf)


@pytest.mark.parametrize("case", ("latent_nonfinite", "latent_zero", "template_nonfinite"))
def test_invalid_tensor_contents_fail_closed(case: str) -> None:
    """非有限内容、零latent能量和非有限模板均失败关闭。"""

    latent, routing, lf, hf = _inputs()
    if case == "latent_nonfinite":
        latent = latent.clone()
        latent.reshape(-1)[0] = float("inf")
    elif case == "latent_zero":
        latent = torch.zeros_like(latent)
    else:
        changed = lf.template.clone()
        changed.reshape(-1)[0] = float("nan")
        lf = replace(lf, template=changed)
    with pytest.raises(ValueError):
        _build(latent, routing, lf, hf)


@pytest.mark.parametrize("dtype", (torch.float16, torch.float32, torch.float64))
def test_latent_dtypes_produce_float32_deterministic_results(
    dtype: torch.dtype,
) -> None:
    """合法latent dtype只影响输入，全部公式输出保持float32。"""

    latent, routing, lf, hf = _inputs(dtype=dtype)
    first = _build(latent, routing, lf, hf)
    second = _build(latent, routing, lf, hf)
    for name in (
        "geometry_capacity_map",
        "lf_direction",
        "hf_tail_direction",
        "lf_update",
        "hf_tail_update",
        "content_only_latent_float32",
    ):
        left = getattr(first, name)
        right = getattr(second, name)
        assert left.dtype == torch.float32
        torch.testing.assert_close(left, right, rtol=0.0, atol=0.0)
    assert (
        first.latent_l2,
        first.lf_nominal_strength,
        first.hf_tail_nominal_strength,
        first.method_role,
    ) == (
        second.latent_l2,
        second.lf_nominal_strength,
        second.hf_tail_nominal_strength,
        second.method_role,
    )


def test_noncontiguous_requires_grad_inputs_remain_unchanged() -> None:
    """只读消费不得改写输入内容、shape、stride或grad状态。"""

    latent_base = torch.arange(1, 25, dtype=torch.float32).reshape(1, 2, 3, 4)
    latent = latent_base.transpose(2, 3).requires_grad_()
    shape = tuple(latent.shape)
    lf, hf = _manual_templates(shape=shape)
    lf_value = lf.template.transpose(2, 3).contiguous().transpose(2, 3)
    hf_value = hf.template.transpose(2, 3).contiguous().transpose(2, 3)
    lf_value.requires_grad_()
    hf_value.requires_grad_()
    lf = replace(lf, template=lf_value)
    hf = replace(hf, template=hf_value)
    routing = _routing(spatial_shape=(shape[2], shape[3]))
    before = {
        "latent": (latent.detach().clone(), latent.shape, latent.stride(), latent.requires_grad, latent.grad),
        "lf": (lf_value.detach().clone(), lf_value.shape, lf_value.stride(), lf_value.requires_grad, lf_value.grad),
        "hf": (hf_value.detach().clone(), hf_value.shape, hf_value.stride(), hf_value.requires_grad, hf_value.grad),
    }
    result = _build(latent, routing, lf, hf)
    for label, value in (("latent", latent), ("lf", lf_value), ("hf", hf_value)):
        content, shape_before, stride_before, requires_grad, grad = before[label]
        torch.testing.assert_close(value.detach(), content, rtol=0.0, atol=0.0)
        assert value.shape == shape_before
        assert value.stride() == stride_before
        assert value.requires_grad is requires_grad
        assert value.grad is grad
    assert not result.content_only_latent_float32.requires_grad


def test_invalid_role_is_the_first_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    """method_role必须先于任何输入或共享身份解析。"""

    monkeypatch.setattr(
        update_module,
        "_validate_static_inputs_and_identity",
        lambda *args: pytest.fail("static inputs must not be read"),
    )
    with pytest.raises(ValueError, match="method_role"):
        build_content_carrier_update(
            current_scheduler_latent=object(),
            routing=object(),  # type: ignore[arg-type]
            lf_template=object(),  # type: ignore[arg-type]
            hf_tail_template=object(),  # type: ignore[arg-type]
            method_role="unsupported",  # type: ignore[arg-type]
        )


def test_source_has_no_legacy_runtime_geometry_digest_or_gpu_dependencies() -> None:
    """隔离更新核不得回接旧合成、模型、治理摘要或GPU路径。"""

    source = Path(update_module.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    forbidden_imports = {
        "main.methods.update_composition",
        "main.methods.carrier.keyed_tensor",
        "main.methods.carrier.blind_content_score",
        "main.methods.geometry",
        "experiments",
    }
    assert imported_modules.isdisjoint(forbidden_imports)
    for forbidden in (
        "compose_ordered_float32_update_once",
        "build_low_frequency_template",
        "build_tail_robust_template",
        "build_high_frequency_tail_template",
        "compute_blind_content_score",
        "Jacobian",
        "jvp",
        "vjp",
        "build_stable_digest",
        "tensor_content_sha256",
        "cuda",
    ):
        assert forbidden not in source
    item_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "item"
    ]
    assert len(item_calls) == 3
    public_builder = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "build_content_carrier_update"
    )
    update_call_line = next(
        node.lineno
        for node in ast.walk(public_builder)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_build_float32_updates"
    )
    assert all(node.lineno > update_call_line for node in item_calls)
