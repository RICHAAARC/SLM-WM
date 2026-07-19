"""验证正式二维 LF 密钥载体的纯 CPU 数学与身份边界。"""

from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError, fields
import inspect
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn.functional as functional

from main.core.digest import build_stable_digest, tensor_content_sha256
from main.core.keyed_prg import KEYED_PRG_VERSION
from main.methods.carrier.low_frequency import (
    LowFrequencyCarrierTemplate,
    build_low_frequency_template,
)
import main.methods.carrier.low_frequency as lf_module


MODEL_IDENTITY_DIGEST = "a" * 64


def _independent_expected(raw: torch.Tensor) -> torch.Tensor:
    """只按权威字面量独立重建二维低通、逐样本中心化与L2。"""

    low_pass = functional.avg_pool2d(
        raw,
        kernel_size=5,
        stride=1,
        padding=2,
        ceil_mode=False,
        count_include_pad=True,
        divisor_override=None,
    )
    centered = low_pass - low_pass.mean(dim=(1, 2, 3), keepdim=True)
    norm = torch.linalg.vector_norm(centered.reshape(1, -1), dim=1)
    return centered / norm.reshape(1, 1, 1, 1)


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

    monkeypatch.setattr(lf_module, "build_keyed_gaussian_tensor", fake_prg)
    return calls


def _build(
    reference: torch.Tensor,
    *,
    key_material: str = "registered-key",
) -> LowFrequencyCarrierTemplate:
    """以全部显式正式输入调用公开构造器。"""

    return build_low_frequency_template(
        reference,
        key_material,
        MODEL_IDENTITY_DIGEST,
        prg_version=KEYED_PRG_VERSION,
    )


@pytest.mark.quick
def test_public_contract_is_frozen_and_exact() -> None:
    """公开类型、字段、签名和模块导出不得扩张。"""

    assert lf_module.__all__ == [
        "LowFrequencyCarrierTemplate",
        "build_low_frequency_template",
    ]
    assert [field.name for field in fields(LowFrequencyCarrierTemplate)] == [
        "template",
        "latent_shape",
        "scoring_key_identity_digest",
        "model_identity_digest",
        "prg_version",
        "prg_domain",
        "filter_identity_digest",
        "template_digest",
    ]
    signature = inspect.signature(build_low_frequency_template)
    assert list(signature.parameters) == [
        "reference_latent",
        "key_material",
        "model_identity_digest",
        "prg_version",
    ]
    assert signature.parameters["prg_version"].kind is inspect.Parameter.KEYWORD_ONLY
    assert signature.parameters["prg_version"].default is inspect.Parameter.empty

    template = torch.arange(1, 7, dtype=torch.float32).reshape(1, 1, 2, 3)
    result = LowFrequencyCarrierTemplate(
        template=template,
        latent_shape=(1, 1, 2, 3),
        scoring_key_identity_digest="b" * 64,
        model_identity_digest=MODEL_IDENTITY_DIGEST,
        prg_version=KEYED_PRG_VERSION,
        prg_domain="lf_content",
        filter_identity_digest="c" * 64,
        template_digest="d" * 64,
    )
    with pytest.raises(FrozenInstanceError):
        result.filter_identity_digest = "e" * 64  # type: ignore[misc]


@pytest.mark.quick
def test_formula_matches_independent_non_square_multichannel_rebuild(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """公开结果必须逐值等于独立二维低通、中心化和L2公式。"""

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
    expected = _independent_expected(raw)

    torch.testing.assert_close(result.template, expected, rtol=0.0, atol=0.0)
    assert result.latent_shape == (1, 2, 2, 3)
    assert result.template.dtype == torch.float32
    assert result.template.device == reference.device
    assert bool(torch.isfinite(result.template).all())
    assert abs(float(result.template.mean().item())) < 2e-7
    assert float(torch.linalg.vector_norm(result.template).item()) == pytest.approx(1.0)
    assert calls == [
        {
            "shape": (1, 2, 2, 3),
            "key_material": "registered-key",
            "domain_fields": {
                "operator": "latent_carrier_template",
                "branch_name": "lf_content",
                "model_identity_digest": MODEL_IDENTITY_DIGEST,
            },
            "prg_version": KEYED_PRG_VERSION,
        }
    ]


@pytest.mark.quick
def test_avg_pool_consumes_every_frozen_parameter_explicitly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """二维低通不得依赖任何后端默认参数。"""

    raw = torch.arange(1, 13, dtype=torch.float32).reshape(1, 2, 2, 3)
    _patch_prg(monkeypatch, raw)
    real_avg_pool2d = functional.avg_pool2d
    calls: list[dict[str, Any]] = []

    def recording_avg_pool2d(input_tensor: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        calls.append(dict(kwargs))
        return real_avg_pool2d(input_tensor, **kwargs)

    monkeypatch.setattr(functional, "avg_pool2d", recording_avg_pool2d)
    result = _build(torch.zeros_like(raw))
    assert result.template.shape == raw.shape
    assert calls == [
        {
            "kernel_size": 5,
            "stride": 1,
            "padding": 2,
            "ceil_mode": False,
            "count_include_pad": True,
            "divisor_override": None,
        }
    ]


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
def test_real_prg_is_deterministic_and_key_isolated() -> None:
    """真实既有CPU PRG必须重复逐字相同且不同key改变正式LF身份。"""

    reference = torch.zeros((1, 2, 3, 5), dtype=torch.float32)
    first = _build(reference, key_material="first-key")
    repeated = _build(reference, key_material="first-key")
    changed = _build(reference, key_material="second-key")

    assert torch.equal(first.template, repeated.template)
    assert first.template_digest == repeated.template_digest
    assert first.scoring_key_identity_digest == repeated.scoring_key_identity_digest
    assert not torch.equal(first.template, changed.template)
    assert first.scoring_key_identity_digest != changed.scoring_key_identity_digest
    assert first.template_digest != changed.template_digest
    assert first.template.shape == (1, 2, 3, 5)
    assert first.template.dtype == torch.float32
    assert float(torch.linalg.vector_norm(first.template).item()) == pytest.approx(1.0)


@pytest.mark.quick
def test_all_three_digests_match_independent_literal_payloads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """评分密钥、冻结滤波和模板摘要职责必须逐字可重建。"""

    raw = torch.arange(1, 16, dtype=torch.float32).reshape(1, 1, 3, 5)
    _patch_prg(monkeypatch, raw)
    result = _build(torch.zeros_like(raw))
    expected_scoring_key_digest = build_stable_digest(
        {"key_material": "registered-key"}
    )
    expected_filter_digest = build_stable_digest(
        {
            "lf_carrier_protocol_schema": "slm_wm_low_frequency_carrier_protocol",
            "lf_kernel_size": 5,
            "lf_stride": 1,
            "lf_padding": 2,
            "lf_boundary_mode": "zero_padding",
            "lf_ceil_mode": False,
            "lf_count_include_pad": True,
            "lf_divisor_override": None,
            "lf_pooling_axes": "height_width_only",
            "lf_batch_channel_isolation": True,
            "lf_normalization_scope": "global_tensor_center_then_l2",
        }
    )
    expected_template_digest = build_stable_digest(
        {
            "carrier_template": "low_frequency",
            "latent_shape": [1, 1, 3, 5],
            "scoring_key_identity_digest": expected_scoring_key_digest,
            "model_identity_digest": MODEL_IDENTITY_DIGEST,
            "prg_version": KEYED_PRG_VERSION,
            "prg_domain": "lf_content",
            "filter_identity_digest": expected_filter_digest,
            "template_content_sha256": tensor_content_sha256(result.template),
        }
    )

    assert result.scoring_key_identity_digest == expected_scoring_key_digest
    assert result.filter_identity_digest == expected_filter_digest
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

    monkeypatch.setattr(lf_module, "build_keyed_gaussian_tensor", fail_prg)
    with pytest.raises((TypeError, ValueError)):
        build_low_frequency_template(
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
        lf_module,
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

    monkeypatch.setattr(lf_module, "build_keyed_gaussian_tensor", fail_prg)
    with pytest.raises(ValueError):
        build_low_frequency_template(
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

    monkeypatch.setattr(lf_module, "build_keyed_gaussian_tensor", fake_prg)
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
        lf_module,
        "build_keyed_gaussian_tensor",
        lambda *args, **kwargs: raw,
    )
    with pytest.raises(ValueError, match="CPU"):
        _build(torch.zeros((1, 1, 2, 3), dtype=torch.float32))


@pytest.mark.quick
def test_nonfinite_or_zero_low_pass_energy_fails_without_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """低通非有限或中心化零能量不得以epsilon/fallback伪造方向。"""

    raw = torch.zeros((1, 1, 2, 3), dtype=torch.float32)
    _patch_prg(monkeypatch, raw)
    with pytest.raises(RuntimeError, match="非零能量"):
        _build(torch.zeros_like(raw))

    _patch_prg(monkeypatch, torch.ones_like(raw))
    monkeypatch.setattr(
        functional,
        "avg_pool2d",
        lambda *args, **kwargs: torch.full_like(raw, float("inf")),
    )
    with pytest.raises(RuntimeError, match="全部有限"):
        _build(torch.zeros_like(raw))


@pytest.mark.quick
def test_source_does_not_import_or_call_legacy_builder_or_runtime() -> None:
    """同名正式API不得暗中委托旧LF、tail、Null Space或模型路径。"""

    source = inspect.getsource(lf_module)
    tree = ast.parse(source)
    keyed_tensor_imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module == "main.methods.carrier.keyed_tensor"
        for alias in node.names
    }
    called_names = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "build_low_frequency_template" not in keyed_tensor_imports
    assert "build_low_frequency_template" not in called_names
    for forbidden in (
        "build_tail_robust_template",
        "build_high_frequency_tail_template",
        "JacobianNullSpaceResult",
        "project_canonical_template",
        "transformers",
        "cuda",
        "requests",
    ):
        assert forbidden not in source
    assert Path(lf_module.__file__).name == "low_frequency.py"
