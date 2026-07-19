"""验证正式二维 HF-tail 密钥载体的纯 CPU 数学与身份边界。"""

from __future__ import annotations

from dataclasses import FrozenInstanceError, fields
import inspect
from typing import Any

import pytest
import torch
import torch.nn.functional as functional

from main.core.digest import build_stable_digest, tensor_content_sha256
from main.core.keyed_prg import KEYED_PRG_VERSION
from main.methods.carrier.high_frequency_tail import (
    HighFrequencyTailCarrierTemplate,
    build_high_frequency_tail_template,
)
import main.methods.carrier.high_frequency_tail as hf_tail_module


MODEL_IDENTITY_DIGEST = "a" * 64


def _independent_expected(raw: torch.Tensor) -> tuple[torch.Tensor, int]:
    """只按权威字面量独立重建高通、稳定选择与L2。"""

    low_pass = functional.avg_pool2d(
        raw,
        kernel_size=5,
        stride=1,
        padding=2,
        ceil_mode=False,
        count_include_pad=True,
        divisor_override=None,
    )
    high_pass = raw - low_pass
    flat = high_pass.reshape(-1)
    values = flat.tolist()
    selected_count = max(1, (len(values) + 4) // 5)
    ordered = list(range(len(values)))
    ordered.sort(key=lambda index: (-abs(values[index]), index))
    expected_flat = torch.zeros_like(flat)
    for index in ordered[:selected_count]:
        expected_flat[index] = flat[index]
    expected = expected_flat.reshape(raw.shape)
    return expected / torch.linalg.vector_norm(expected), selected_count


def _patch_prg(
    monkeypatch: pytest.MonkeyPatch,
    raw: torch.Tensor,
) -> list[dict[str, Any]]:
    """用规范CPU fixture替代PRG并记录唯一调用正文。"""

    calls: list[dict[str, Any]] = []

    def fake_prg(
        shape: tuple[int, ...],
        key_material: str,
        domain_fields: dict[str, Any],
        prg_version: str,
    ) -> torch.Tensor:
        calls.append(
            {
                "shape": shape,
                "key_material": key_material,
                "domain_fields": dict(domain_fields),
                "prg_version": prg_version,
            }
        )
        return raw.clone()

    monkeypatch.setattr(
        hf_tail_module,
        "build_keyed_gaussian_tensor",
        fake_prg,
    )
    return calls


def _build(
    reference: torch.Tensor,
    *,
    key_material: str = "registered-key",
) -> HighFrequencyTailCarrierTemplate:
    """以全部显式正式输入调用公开构造器。"""

    return build_high_frequency_tail_template(
        reference,
        key_material,
        MODEL_IDENTITY_DIGEST,
        prg_version=KEYED_PRG_VERSION,
    )


@pytest.mark.quick
def test_public_contract_is_frozen_and_exact() -> None:
    """公开类型、字段、签名和模块导出不得扩张。"""

    assert hf_tail_module.__all__ == [
        "HIGH_FREQUENCY_TAIL_PROTOCOL_DIGEST",
        "HighFrequencyTailCarrierTemplate",
        "build_high_frequency_tail_template",
    ]
    assert [field.name for field in fields(HighFrequencyTailCarrierTemplate)] == [
        "template",
        "latent_shape",
        "scoring_key_identity_digest",
        "model_identity_digest",
        "prg_version",
        "prg_domain",
        "high_pass_identity_digest",
        "selected_element_count",
        "template_digest",
    ]
    signature = inspect.signature(build_high_frequency_tail_template)
    assert list(signature.parameters) == [
        "reference_latent",
        "key_material",
        "model_identity_digest",
        "prg_version",
    ]
    assert signature.parameters["prg_version"].kind is inspect.Parameter.KEYWORD_ONLY
    assert signature.parameters["prg_version"].default is inspect.Parameter.empty

    raw = torch.arange(1, 7, dtype=torch.float32).reshape(1, 1, 2, 3)
    original = HighFrequencyTailCarrierTemplate(
        template=raw,
        latent_shape=(1, 1, 2, 3),
        scoring_key_identity_digest="b" * 64,
        model_identity_digest=MODEL_IDENTITY_DIGEST,
        prg_version=KEYED_PRG_VERSION,
        prg_domain="hf_tail_robust",
        high_pass_identity_digest="c" * 64,
        selected_element_count=2,
        template_digest="d" * 64,
    )
    with pytest.raises(FrozenInstanceError):
        original.selected_element_count = 3  # type: ignore[misc]


@pytest.mark.quick
def test_formula_matches_independent_non_square_multichannel_rebuild(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """公开结果必须逐值等于独立高通、稳定20%选择与L2公式。"""

    raw = torch.tensor(
        [
            [
                [[1.0, -2.0, 3.0], [4.0, -5.0, 6.0]],
                [[-7.0, 8.0, -9.0], [10.0, -11.0, 12.0]],
            ]
        ],
        dtype=torch.float32,
    )
    calls = _patch_prg(monkeypatch, raw)
    reference = torch.zeros_like(raw, dtype=torch.float64)
    result = _build(reference)
    expected, selected_count = _independent_expected(raw)

    torch.testing.assert_close(result.template, expected, rtol=0.0, atol=0.0)
    assert result.selected_element_count == selected_count == 3
    assert result.latent_shape == (1, 2, 2, 3)
    assert result.template.dtype == torch.float32
    assert result.template.device == reference.device
    assert bool(torch.isfinite(result.template).all())
    assert float(torch.linalg.vector_norm(result.template).item()) == pytest.approx(1.0)
    assert calls == [
        {
            "shape": (1, 2, 2, 3),
            "key_material": "registered-key",
            "domain_fields": {
                "operator": "latent_carrier_template",
                "branch_name": "hf_tail_robust",
                "model_identity_digest": MODEL_IDENTITY_DIGEST,
            },
            "prg_version": KEYED_PRG_VERSION,
        }
    ]
    assert calls[0]["domain_fields"]["branch_name"] != "lf_content"


@pytest.mark.quick
@pytest.mark.parametrize(
    ("shape", "expected_count"),
    (
        ((1, 1, 1, 1), 1),
        ((1, 1, 1, 4), 1),
        ((1, 1, 1, 5), 1),
        ((1, 1, 1, 6), 2),
    ),
)
def test_selected_count_uses_exact_one_fifth_integer_ceil(
    monkeypatch: pytest.MonkeyPatch,
    shape: tuple[int, int, int, int],
    expected_count: int,
) -> None:
    """n=1/4/5/6必须使用max(1,(n+4)//5)，不能floor或round。"""

    raw = torch.arange(1, torch.tensor(shape).prod().item() + 1, dtype=torch.float32).reshape(shape)
    _patch_prg(monkeypatch, raw)
    result = _build(torch.zeros(shape, dtype=torch.float32))
    assert result.selected_element_count == expected_count
    assert int(torch.count_nonzero(result.template).item()) == expected_count


@pytest.mark.quick
def test_stable_tie_prefers_ascending_flat_index_and_does_not_center(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """并列绝对值必须选更小flat index，稀疏结果不得被中心化。"""

    raw = torch.tensor([[[[1.0, -1.0, 0.0, -1.0, 1.0]]]], dtype=torch.float32)
    _patch_prg(monkeypatch, raw)
    result = _build(torch.zeros_like(raw))

    nonzero_indices = torch.nonzero(result.template.reshape(-1), as_tuple=False).reshape(-1)
    assert nonzero_indices.tolist() == [0]
    assert result.template.reshape(-1)[0].item() > 0.0
    assert result.template.mean().item() > 0.0
    assert result.selected_element_count == 1


@pytest.mark.quick
def test_selected_values_keep_high_pass_sign_and_unselected_values_are_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """稳定tail只改变支持集和共同L2比例，不得丢失符号。"""

    raw = torch.tensor([[[[-9.0, 1.0, 2.0, 3.0, 4.0, 8.0]]]], dtype=torch.float32)
    _patch_prg(monkeypatch, raw)
    result = _build(torch.zeros_like(raw))
    expected, selected_count = _independent_expected(raw)

    assert selected_count == 2
    torch.testing.assert_close(result.template, expected, rtol=0.0, atol=0.0)
    support = result.template != 0
    assert int(support.sum().item()) == 2
    assert bool((torch.sign(result.template[support]) == torch.sign(expected[support])).all())
    assert bool((result.template[~support] == 0.0).all())


@pytest.mark.quick
def test_reference_dtype_and_layout_do_not_change_canonical_float32_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """float16/float64与noncontiguous reference只决定形状和最终设备。"""

    raw = torch.arange(1, 13, dtype=torch.float32).reshape(1, 2, 2, 3)
    calls = _patch_prg(monkeypatch, raw)
    base = torch.zeros((1, 2, 3, 2), dtype=torch.float64)
    noncontiguous = base.transpose(-1, -2)
    assert not noncontiguous.is_contiguous()
    before = noncontiguous.clone()
    before_stride = noncontiguous.stride()
    first = _build(noncontiguous)

    _patch_prg(monkeypatch, raw)
    second = _build(torch.zeros(raw.shape, dtype=torch.float16))

    torch.testing.assert_close(first.template, second.template, rtol=0.0, atol=0.0)
    torch.testing.assert_close(noncontiguous, before, rtol=0.0, atol=0.0)
    assert noncontiguous.dtype == torch.float64
    assert noncontiguous.stride() == before_stride
    assert first.template.dtype == second.template.dtype == torch.float32
    assert len(calls) == 1


@pytest.mark.quick
def test_different_keys_change_the_single_prg_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """评分密钥必须实际进入唯一HF-tail PRG调用和模板身份。"""

    calls: list[str] = []

    def fake_prg(
        shape: tuple[int, ...],
        key_material: str,
        domain_fields: dict[str, Any],
        prg_version: str,
    ) -> torch.Tensor:
        del domain_fields, prg_version
        calls.append(key_material)
        offset = 0.0 if key_material == "first-key" else 7.0
        return (torch.arange(1, 7, dtype=torch.float32) + offset).reshape(shape)

    monkeypatch.setattr(hf_tail_module, "build_keyed_gaussian_tensor", fake_prg)
    reference = torch.zeros((1, 1, 2, 3), dtype=torch.float32)
    first = _build(reference, key_material="first-key")
    second = _build(reference, key_material="second-key")

    assert calls == ["first-key", "second-key"]
    assert not torch.equal(first.template, second.template)
    assert first.scoring_key_identity_digest != second.scoring_key_identity_digest
    assert first.template_digest != second.template_digest


@pytest.mark.quick
def test_all_three_digests_match_independent_literal_payloads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """评分密钥、高通协议和模板摘要职责必须逐字可重建。"""

    raw = torch.arange(1, 7, dtype=torch.float32).reshape(1, 1, 2, 3)
    _patch_prg(monkeypatch, raw)
    result = _build(torch.zeros_like(raw))
    expected_scoring_key_digest = build_stable_digest(
        {"key_material": "registered-key"}
    )
    expected_high_pass_digest = build_stable_digest(
        {
            "low_pass_kernel_size": 5,
            "low_pass_stride": 1,
            "low_pass_padding": 2,
            "low_pass_boundary_mode": "zero_padding",
            "low_pass_ceil_mode": False,
            "low_pass_count_include_pad": True,
            "low_pass_divisor_override": None,
            "high_pass_rule": "input_minus_paired_low_pass",
            "tail_selection_scope": "per_sample_flatten_channel_height_width",
            "tail_fraction_numerator": 1,
            "tail_fraction_denominator": 5,
            "selected_element_count_rule": "max_one_integer_ceil_one_fifth",
            "tail_order": "absolute_value_descending_then_flat_index_ascending",
            "unselected_value": 0.0,
            "normalization": "per_sample_float32_l2_without_centering",
        }
    )
    expected_template_digest = build_stable_digest(
        {
            "carrier_template": "high_frequency_tail",
            "latent_shape": [1, 1, 2, 3],
            "scoring_key_identity_digest": expected_scoring_key_digest,
            "model_identity_digest": MODEL_IDENTITY_DIGEST,
            "prg_version": KEYED_PRG_VERSION,
            "prg_domain": "hf_tail_robust",
            "high_pass_identity_digest": expected_high_pass_digest,
            "selected_element_count": 2,
            "template_content_sha256": tensor_content_sha256(result.template),
        }
    )

    assert result.scoring_key_identity_digest == expected_scoring_key_digest
    assert result.high_pass_identity_digest == expected_high_pass_digest
    assert result.template_digest == expected_template_digest
    assert "registered-key" not in repr(result)


@pytest.mark.quick
@pytest.mark.parametrize(
    "reference",
    (
        object(),
        torch.zeros((1, 1, 2, 3), dtype=torch.int64),
        torch.zeros((1, 1, 2, 3), dtype=torch.bool),
        torch.zeros((1, 1, 2, 3), dtype=torch.complex64),
        torch.zeros((1, 2, 3), dtype=torch.float32),
        torch.zeros((2, 1, 2, 3), dtype=torch.float32),
        torch.zeros((1, 0, 2, 3), dtype=torch.float32),
        torch.tensor([[[[float("nan")]]]], dtype=torch.float32),
        torch.tensor([[[[float("inf")]]]], dtype=torch.float32),
    ),
)
def test_invalid_reference_fails_before_prg(
    monkeypatch: pytest.MonkeyPatch,
    reference: object,
) -> None:
    """reference类型、shape、dtype和内容门禁必须先于密钥PRG。"""

    calls = 0

    def fail_prg(*args: object, **kwargs: object) -> torch.Tensor:
        nonlocal calls
        calls += 1
        raise AssertionError((args, kwargs))

    monkeypatch.setattr(hf_tail_module, "build_keyed_gaussian_tensor", fail_prg)
    with pytest.raises((TypeError, ValueError)):
        build_high_frequency_tail_template(
            reference,
            "registered-key",
            MODEL_IDENTITY_DIGEST,
            prg_version=KEYED_PRG_VERSION,
        )
    assert calls == 0


@pytest.mark.quick
def test_meta_reference_fails_before_any_content_read_or_prg(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """meta设备必须在isfinite等内容读取前失败关闭。"""

    reference = torch.empty((1, 1, 2, 3), dtype=torch.float32, device="meta")
    monkeypatch.setattr(
        hf_tail_module,
        "build_keyed_gaussian_tensor",
        lambda *args, **kwargs: pytest.fail((args, kwargs)),
    )
    with pytest.raises(ValueError, match="已物化"):
        _build(reference)


@pytest.mark.quick
@pytest.mark.parametrize(
    ("key_material", "model_digest", "prg_version"),
    (
        ("", MODEL_IDENTITY_DIGEST, KEYED_PRG_VERSION),
        (b"key", MODEL_IDENTITY_DIGEST, KEYED_PRG_VERSION),
        ("key", "A" * 64, KEYED_PRG_VERSION),
        ("key", "a" * 63, KEYED_PRG_VERSION),
        ("key", MODEL_IDENTITY_DIGEST, "unsupported_prg"),
    ),
)
def test_identity_drift_fails_before_prg(
    monkeypatch: pytest.MonkeyPatch,
    key_material: object,
    model_digest: str,
    prg_version: str,
) -> None:
    """密钥、模型和PRG身份漂移不得触发随机张量构造。"""

    calls = 0

    def fail_prg(*args: object, **kwargs: object) -> torch.Tensor:
        nonlocal calls
        calls += 1
        raise AssertionError((args, kwargs))

    monkeypatch.setattr(hf_tail_module, "build_keyed_gaussian_tensor", fail_prg)
    with pytest.raises(ValueError):
        build_high_frequency_tail_template(
            torch.zeros((1, 1, 2, 3), dtype=torch.float32),
            key_material,  # type: ignore[arg-type]
            model_digest,
            prg_version=prg_version,
        )
    assert calls == 0


@pytest.mark.quick
@pytest.mark.parametrize(
    "raw",
    (
        object(),
        torch.zeros((1, 1, 2, 3), dtype=torch.float64),
        torch.zeros((1, 1, 2, 2), dtype=torch.float32),
        torch.tensor([[[[1.0, float("nan"), 2.0], [3.0, 4.0, 5.0]]]]),
    ),
)
def test_invalid_prg_output_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    raw: object,
) -> None:
    """PRG输出必须是同shape的有限CPU float32 Tensor。"""

    calls = 0

    def fake_prg(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        return raw

    monkeypatch.setattr(hf_tail_module, "build_keyed_gaussian_tensor", fake_prg)
    with pytest.raises((TypeError, ValueError)):
        _build(torch.zeros((1, 1, 2, 3), dtype=torch.float32))
    assert calls == 1


@pytest.mark.quick
def test_meta_prg_output_fails_before_content_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """非CPU PRG结果必须在finite读取前失败。"""

    raw = torch.empty((1, 1, 2, 3), dtype=torch.float32, device="meta")
    monkeypatch.setattr(
        hf_tail_module,
        "build_keyed_gaussian_tensor",
        lambda *args, **kwargs: raw,
    )
    with pytest.raises(ValueError, match="CPU"):
        _build(torch.zeros((1, 1, 2, 3), dtype=torch.float32))


@pytest.mark.quick
def test_zero_high_pass_energy_fails_without_epsilon_or_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """全零高通不得通过epsilon、clamp或随机回退伪造方向。"""

    raw = torch.zeros((1, 1, 2, 3), dtype=torch.float32)
    _patch_prg(monkeypatch, raw)
    with pytest.raises(RuntimeError, match="非零能量"):
        _build(torch.zeros_like(raw))


@pytest.mark.quick
def test_source_has_no_old_builder_or_unstable_selection_backdoor() -> None:
    """正式HF-tail不得调用旧载体或后端相关选择原语。"""

    source = inspect.getsource(hf_tail_module)
    for forbidden in (
        "build_low_frequency_template",
        "build_tail_robust_template",
        "torch.topk",
        "torch.kthvalue",
        "torch.quantile",
        ".topk(",
        ".kthvalue(",
        ".quantile(",
        "cuda",
        "transformers",
    ):
        assert forbidden not in source
    assert "key=lambda index: (-abs(flat_values[index]), index)" in source
    assert "(element_count + 4) // 5" in source
