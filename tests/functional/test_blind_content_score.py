"""验证正式 BlindContentScore 的纯 CPU 数学、身份与失败关闭边界。"""

from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError, fields, replace
import inspect
import math
from pathlib import Path
from typing import Any, Callable

import pytest
import torch

from main.core.digest import build_stable_digest, tensor_content_sha256
from main.core.keyed_prg import KEYED_PRG_VERSION
from main.methods.carrier.blind_content_score import (
    BlindContentScore,
    compute_blind_content_score,
)
from main.methods.carrier.high_frequency_tail import (
    HighFrequencyTailCarrierTemplate,
    build_high_frequency_tail_template,
)
from main.methods.carrier.low_frequency import (
    LowFrequencyCarrierTemplate,
    build_low_frequency_template,
)
import main.methods.carrier as carrier_package
import main.methods.carrier.blind_content_score as score_module


MODEL_IDENTITY_DIGEST = "a" * 64
SCORING_KEY_IDENTITY_DIGEST = "b" * 64
LF_TEMPLATE_DIGEST = "c" * 64
HF_TEMPLATE_DIGEST = "d" * 64
SHAPE = (1, 2, 2, 3)
FORMAL_ROLES = (
    "full_dual_chain",
    "uniform_content_routing",
    "lf_only_content",
    "hf_tail_only_content",
    "content_chain_only",
    "geometry_recovery_without_embedded_sync",
)


def _base_tensors(
    *,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """返回三张具有非零中心化能量的非方形多通道 Tensor。"""

    observed = torch.tensor(
        [
            [
                [[1.0, -2.0, 3.0], [4.0, 0.5, -1.0]],
                [[-3.0, 2.0, 5.0], [1.5, -4.0, 0.25]],
            ]
        ],
        dtype=dtype,
    )
    lf = torch.tensor(
        [
            [
                [[-1.0, 0.0, 1.0], [2.0, 3.0, 4.0]],
                [[4.0, 2.0, 0.0], [-2.0, -3.0, -4.0]],
            ]
        ],
        dtype=torch.float32,
    )
    hf = torch.tensor(
        [
            [
                [[3.0, -1.0, -2.0], [0.0, 4.0, -3.0]],
                [[-5.0, 1.0, 2.0], [3.0, -4.0, 6.0]],
            ]
        ],
        dtype=torch.float32,
    )
    return observed, lf, hf


def _templates(
    lf_tensor: torch.Tensor | None = None,
    hf_tensor: torch.Tensor | None = None,
) -> tuple[LowFrequencyCarrierTemplate, HighFrequencyTailCarrierTemplate]:
    """构造只承载正式字段的最小已冻结模板 fixture。"""

    _observed, default_lf, default_hf = _base_tensors()
    lf_value = default_lf if lf_tensor is None else lf_tensor
    hf_value = default_hf if hf_tensor is None else hf_tensor
    lf = LowFrequencyCarrierTemplate(
        template=lf_value,
        latent_shape=SHAPE,
        scoring_key_identity_digest=SCORING_KEY_IDENTITY_DIGEST,
        model_identity_digest=MODEL_IDENTITY_DIGEST,
        prg_version=KEYED_PRG_VERSION,
        prg_domain="lf_content",
        filter_identity_digest="e" * 64,
        template_digest=LF_TEMPLATE_DIGEST,
    )
    hf = HighFrequencyTailCarrierTemplate(
        template=hf_value,
        latent_shape=SHAPE,
        scoring_key_identity_digest=SCORING_KEY_IDENTITY_DIGEST,
        model_identity_digest=MODEL_IDENTITY_DIGEST,
        prg_version=KEYED_PRG_VERSION,
        prg_domain="hf_tail_robust",
        high_pass_identity_digest="f" * 64,
        selected_element_count=3,
        template_digest=HF_TEMPLATE_DIGEST,
    )
    return lf, hf


def _independent_correlation(left: torch.Tensor, right: torch.Tensor) -> float:
    """按权威字面公式独立重建 float32 中心化归一化内积。"""

    left_flat = left.detach().float().reshape(-1)
    right_flat = right.detach().float().reshape(-1)
    left_centered = left_flat - left_flat.mean()
    right_centered = right_flat - right_flat.mean()
    return float(
        torch.dot(
            left_centered / torch.linalg.vector_norm(left_centered),
            right_centered / torch.linalg.vector_norm(right_centered),
        ).item()
    )


def _compute(
    observed: torch.Tensor,
    lf: LowFrequencyCarrierTemplate,
    hf: HighFrequencyTailCarrierTemplate,
    role: str = "full_dual_chain",
) -> BlindContentScore:
    """以全部显式正式输入调用公开评分器。"""

    return compute_blind_content_score(observed, lf, hf, role)


@pytest.mark.quick
def test_public_contract_is_frozen_exact_and_package_exported() -> None:
    """公开字段、签名与正式 carrier 导出必须保持一致。"""

    assert score_module.__all__ == [
        "BlindContentScore",
        "compute_blind_content_score",
    ]
    assert [field.name for field in fields(BlindContentScore)] == [
        "blind_lf_score",
        "blind_hf_tail_score",
        "blind_content_score",
        "lf_weight",
        "hf_tail_weight",
        "method_role",
        "scoring_key_identity_digest",
        "score_identity_digest",
    ]
    signature = inspect.signature(compute_blind_content_score)
    assert list(signature.parameters) == [
        "observed_latent",
        "lf_template",
        "hf_tail_template",
        "method_role",
    ]
    assert all(
        parameter.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
        for parameter in signature.parameters.values()
    )
    assert all(
        parameter.default is inspect.Parameter.empty
        for parameter in signature.parameters.values()
    )
    assert signature.parameters["observed_latent"].annotation == "Tensor"
    assert signature.return_annotation == "BlindContentScore"

    instance = BlindContentScore(
        blind_lf_score=0.1,
        blind_hf_tail_score=-0.2,
        blind_content_score=0.01,
        lf_weight=0.7,
        hf_tail_weight=0.3,
        method_role="full_dual_chain",
        scoring_key_identity_digest=SCORING_KEY_IDENTITY_DIGEST,
        score_identity_digest="9" * 64,
    )
    with pytest.raises(FrozenInstanceError):
        instance.blind_content_score = 1.0  # type: ignore[misc]

    assert carrier_package.BlindContentScore is BlindContentScore
    assert (
        carrier_package.compute_blind_content_score
        is compute_blind_content_score
    )


@pytest.mark.quick
@pytest.mark.parametrize(
    ("role", "expected_weights"),
    [
        ("full_dual_chain", (0.70, 0.30)),
        ("uniform_content_routing", (0.70, 0.30)),
        ("lf_only_content", (1.0, 0.0)),
        ("hf_tail_only_content", (0.0, 1.0)),
        ("content_chain_only", (0.70, 0.30)),
        ("geometry_recovery_without_embedded_sync", (0.70, 0.30)),
    ],
)
def test_all_roles_apply_only_the_frozen_total_weights(
    role: str,
    expected_weights: tuple[float, float],
) -> None:
    """六角色都计算两条raw相关，只有总分应用冻结权重。"""

    observed, lf_tensor, hf_tensor = _base_tensors(dtype=torch.float64)
    lf, hf = _templates(lf_tensor, hf_tensor)
    result = _compute(observed, lf, hf, role)
    expected_lf = _independent_correlation(observed, lf_tensor)
    expected_hf = _independent_correlation(observed, hf_tensor)
    lf_weight, hf_weight = expected_weights

    assert result.blind_lf_score == expected_lf
    assert result.blind_hf_tail_score == expected_hf
    assert result.lf_weight == lf_weight
    assert result.hf_tail_weight == hf_weight
    assert result.blind_content_score == (
        lf_weight * expected_lf + hf_weight * expected_hf
    )
    assert result.method_role == role
    assert result.scoring_key_identity_digest == SCORING_KEY_IDENTITY_DIGEST


@pytest.mark.quick
@pytest.mark.parametrize("role", FORMAL_ROLES)
def test_every_role_computes_both_correlations_once(
    monkeypatch: pytest.MonkeyPatch,
    role: str,
) -> None:
    """零权重分支也必须保留一次raw盲相关观测。"""

    observed, lf_tensor, hf_tensor = _base_tensors()
    lf, hf = _templates(lf_tensor, hf_tensor)
    calls: list[tuple[torch.Tensor, torch.Tensor]] = []
    values = iter((0.25, -0.5))

    def fake_correlation(left: torch.Tensor, right: torch.Tensor) -> float:
        calls.append((left, right))
        return next(values)

    monkeypatch.setattr(score_module, "_normalized_correlation", fake_correlation)
    result = _compute(observed, lf, hf, role)

    assert calls == [(observed, lf_tensor), (observed, hf_tensor)]
    assert result.blind_lf_score == 0.25
    assert result.blind_hf_tail_score == -0.5
    assert result.blind_content_score == (
        result.lf_weight * 0.25 + result.hf_tail_weight * -0.5
    )


@pytest.mark.quick
def test_digest_binds_exact_unrounded_scores_and_each_content_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """score摘要复用三次唯一内容摘要并绑定未round的实际分数。"""

    observed, lf_tensor, hf_tensor = _base_tensors()
    lf, hf = _templates(lf_tensor, hf_tensor)
    content_calls: list[torch.Tensor] = []
    digest_by_identity = {
        id(observed): "1" * 64,
        id(lf_tensor): "2" * 64,
        id(hf_tensor): "3" * 64,
    }

    def fake_tensor_digest(value: torch.Tensor) -> str:
        content_calls.append(value)
        return digest_by_identity[id(value)]

    score_values = iter((0.12345678901234567, -0.9876543210987654))
    monkeypatch.setattr(score_module, "tensor_content_sha256", fake_tensor_digest)
    monkeypatch.setattr(
        score_module,
        "_normalized_correlation",
        lambda _left, _right: next(score_values),
    )
    result = _compute(observed, lf, hf, "uniform_content_routing")
    expected_total = 0.70 * result.blind_lf_score + 0.30 * result.blind_hf_tail_score
    expected_payload = {
        "observed_latent_content_sha256": "1" * 64,
        "lf_template_digest": LF_TEMPLATE_DIGEST,
        "lf_template_content_sha256": "2" * 64,
        "hf_tail_template_digest": HF_TEMPLATE_DIGEST,
        "hf_tail_template_content_sha256": "3" * 64,
        "model_identity_digest": MODEL_IDENTITY_DIGEST,
        "prg_version": KEYED_PRG_VERSION,
        "scoring_key_identity_digest": SCORING_KEY_IDENTITY_DIGEST,
        "method_role": "uniform_content_routing",
        "lf_weight": 0.70,
        "hf_tail_weight": 0.30,
        "blind_lf_score": 0.12345678901234567,
        "blind_hf_tail_score": -0.9876543210987654,
        "blind_content_score": expected_total,
    }

    assert content_calls == [observed, lf_tensor, hf_tensor]
    assert result.blind_content_score == expected_total
    assert result.score_identity_digest == build_stable_digest(expected_payload)


@pytest.mark.quick
def test_real_digest_matches_independent_literal_payload() -> None:
    """不替换摘要实现时，结果仍精确等于独立字面payload。"""

    observed, lf_tensor, hf_tensor = _base_tensors(dtype=torch.float16)
    lf, hf = _templates(lf_tensor, hf_tensor)
    result = _compute(observed, lf, hf, "content_chain_only")
    expected = build_stable_digest(
        {
            "observed_latent_content_sha256": tensor_content_sha256(observed),
            "lf_template_digest": LF_TEMPLATE_DIGEST,
            "lf_template_content_sha256": tensor_content_sha256(lf_tensor),
            "hf_tail_template_digest": HF_TEMPLATE_DIGEST,
            "hf_tail_template_content_sha256": tensor_content_sha256(hf_tensor),
            "model_identity_digest": MODEL_IDENTITY_DIGEST,
            "prg_version": KEYED_PRG_VERSION,
            "scoring_key_identity_digest": SCORING_KEY_IDENTITY_DIGEST,
            "method_role": "content_chain_only",
            "lf_weight": 0.70,
            "hf_tail_weight": 0.30,
            "blind_lf_score": result.blind_lf_score,
            "blind_hf_tail_score": result.blind_hf_tail_score,
            "blind_content_score": result.blind_content_score,
        }
    )
    assert result.score_identity_digest == expected


class _RoleSubclass(str):
    """用于证明角色门禁拒绝str子类。"""


@pytest.mark.quick
@pytest.mark.parametrize("invalid_role", [None, True, 1, _RoleSubclass("lf_only_content"), "other"])
def test_role_gate_is_first_and_exact(
    monkeypatch: pytest.MonkeyPatch,
    invalid_role: Any,
) -> None:
    """非法角色必须在触碰任何输入或内容前失败关闭。"""

    monkeypatch.setattr(
        score_module,
        "_validate_static_inputs_and_identity",
        lambda *_args: pytest.fail("非法角色不得进入输入验证"),
    )
    with pytest.raises(ValueError):
        compute_blind_content_score(object(), object(), object(), invalid_role)


Mutation = Callable[
    [torch.Tensor, LowFrequencyCarrierTemplate, HighFrequencyTailCarrierTemplate],
    tuple[Any, Any, Any],
]


def _replace_lf_tensor(
    lf: LowFrequencyCarrierTemplate,
    value: Any,
) -> LowFrequencyCarrierTemplate:
    return replace(lf, template=value)


def _replace_hf_tensor(
    hf: HighFrequencyTailCarrierTemplate,
    value: Any,
) -> HighFrequencyTailCarrierTemplate:
    return replace(hf, template=value)


STATIC_MUTATIONS: tuple[tuple[str, Mutation], ...] = (
    ("observed_type", lambda _o, lf, hf: (object(), lf, hf)),
    (
        "observed_dtype",
        lambda o, lf, hf: (o.to(torch.int64), lf, hf),
    ),
    (
        "observed_meta",
        lambda o, lf, hf: (torch.empty(o.shape, device="meta"), lf, hf),
    ),
    (
        "observed_ndim",
        lambda o, lf, hf: (o.reshape(1, -1), lf, hf),
    ),
    (
        "observed_batch",
        lambda o, lf, hf: (o.repeat(2, 1, 1, 1), lf, hf),
    ),
    ("lf_type", lambda o, _lf, hf: (o, object(), hf)),
    ("hf_type", lambda o, lf, _hf: (o, lf, object())),
    (
        "lf_shape_container",
        lambda o, lf, hf: (o, replace(lf, latent_shape=list(SHAPE)), hf),
    ),
    (
        "hf_shape_bool",
        lambda o, lf, hf: (o, lf, replace(hf, latent_shape=(True, 2, 2, 3))),
    ),
    (
        "lf_shape_value",
        lambda o, lf, hf: (o, replace(lf, latent_shape=(1, 2, 3, 2)), hf),
    ),
    (
        "lf_tensor_type",
        lambda o, lf, hf: (o, _replace_lf_tensor(lf, object()), hf),
    ),
    (
        "lf_tensor_dtype",
        lambda o, lf, hf: (o, _replace_lf_tensor(lf, lf.template.double()), hf),
    ),
    (
        "hf_tensor_meta",
        lambda o, lf, hf: (
            o,
            lf,
            _replace_hf_tensor(hf, torch.empty(SHAPE, device="meta")),
        ),
    ),
    (
        "hf_tensor_shape",
        lambda o, lf, hf: (o, lf, _replace_hf_tensor(hf, hf.template[..., :2])),
    ),
    (
        "lf_domain",
        lambda o, lf, hf: (o, replace(lf, prg_domain="tail_robust"), hf),
    ),
    (
        "hf_domain",
        lambda o, lf, hf: (o, lf, replace(hf, prg_domain="hf_content")),
    ),
    (
        "lf_model_sha",
        lambda o, lf, hf: (o, replace(lf, model_identity_digest="A" * 64), hf),
    ),
    (
        "model_mismatch",
        lambda o, lf, hf: (o, lf, replace(hf, model_identity_digest="1" * 64)),
    ),
    (
        "lf_key_sha",
        lambda o, lf, hf: (
            o,
            replace(lf, scoring_key_identity_digest="short"),
            hf,
        ),
    ),
    (
        "key_mismatch",
        lambda o, lf, hf: (
            o,
            lf,
            replace(hf, scoring_key_identity_digest="1" * 64),
        ),
    ),
    (
        "lf_template_digest",
        lambda o, lf, hf: (o, replace(lf, template_digest="bad"), hf),
    ),
    (
        "hf_template_digest",
        lambda o, lf, hf: (o, lf, replace(hf, template_digest="G" * 64)),
    ),
    (
        "prg_mismatch",
        lambda o, lf, hf: (o, lf, replace(hf, prg_version="other")),
    ),
    (
        "unsupported_common_prg",
        lambda o, lf, hf: (
            o,
            replace(lf, prg_version="other"),
            replace(hf, prg_version="other"),
        ),
    ),
)


@pytest.mark.quick
@pytest.mark.parametrize(("_label", "mutate"), STATIC_MUTATIONS)
def test_static_and_identity_failures_precede_all_content_reads(
    monkeypatch: pytest.MonkeyPatch,
    _label: str,
    mutate: Mutation,
) -> None:
    """任一静态或跨模板身份错误都不得触发finite/相关/摘要。"""

    observed, lf_tensor, hf_tensor = _base_tensors()
    lf, hf = _templates(lf_tensor, hf_tensor)
    changed_observed, changed_lf, changed_hf = mutate(observed, lf, hf)
    monkeypatch.setattr(
        score_module,
        "_require_finite_tensor",
        lambda *_args, **_kwargs: pytest.fail("静态错误不得读取Tensor内容"),
    )
    monkeypatch.setattr(
        score_module,
        "tensor_content_sha256",
        lambda *_args: pytest.fail("静态错误不得计算内容摘要"),
    )
    with pytest.raises((TypeError, ValueError)):
        compute_blind_content_score(
            changed_observed,
            changed_lf,
            changed_hf,
            "full_dual_chain",
        )


@pytest.mark.quick
def test_prg_version_is_validated_once_after_cross_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """共同PRG版本只在两模板一致后调用现有validator一次。"""

    observed, lf_tensor, hf_tensor = _base_tensors()
    lf, hf = _templates(lf_tensor, hf_tensor)
    calls: list[str] = []
    monkeypatch.setattr(
        score_module,
        "require_supported_keyed_prg_version",
        lambda value: calls.append(value),
    )
    _compute(observed, lf, hf)
    assert calls == [KEYED_PRG_VERSION]


@pytest.mark.quick
@pytest.mark.parametrize("target", ["observed", "lf", "hf"])
@pytest.mark.parametrize("invalid_value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_content_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    target: str,
    invalid_value: float,
) -> None:
    """三张Tensor任一非有限都必须在相关和摘要前失败。"""

    observed, lf_tensor, hf_tensor = _base_tensors()
    values = {"observed": observed, "lf": lf_tensor, "hf": hf_tensor}
    values[target] = values[target].clone()
    values[target].reshape(-1)[0] = invalid_value
    lf, hf = _templates(values["lf"], values["hf"])
    monkeypatch.setattr(
        score_module,
        "_normalized_correlation",
        lambda *_args: pytest.fail("非有限输入不得计算相关"),
    )
    monkeypatch.setattr(
        score_module,
        "tensor_content_sha256",
        lambda *_args: pytest.fail("非有限输入不得计算摘要"),
    )
    with pytest.raises(ValueError):
        _compute(values["observed"], lf, hf)


@pytest.mark.quick
@pytest.mark.parametrize("target", ["observed", "lf", "hf"])
def test_zero_centered_energy_fails_closed(target: str) -> None:
    """观测或任一模板为常数时不得以epsilon伪造相关。"""

    observed, lf_tensor, hf_tensor = _base_tensors()
    values = {"observed": observed, "lf": lf_tensor, "hf": hf_tensor}
    values[target] = torch.ones_like(values[target])
    lf, hf = _templates(values["lf"], values["hf"])
    with pytest.raises(ValueError, match="非零中心化能量"):
        _compute(values["observed"], lf, hf)


@pytest.mark.quick
def test_noncontiguous_requires_grad_inputs_are_not_modified() -> None:
    """评分只detach读取，不改变内容、shape、stride、grad或requires_grad。"""

    observed_source = torch.arange(1, 73, dtype=torch.float64).reshape(1, 2, 3, 12)
    lf_source = torch.arange(-60, 12, dtype=torch.float32).reshape(1, 2, 3, 12)
    hf_source = torch.arange(30, 102, dtype=torch.float32).reshape(1, 2, 3, 12)
    observed = observed_source[..., ::4].requires_grad_()
    lf_tensor = lf_source[..., ::4].requires_grad_()
    hf_tensor = hf_source[..., ::4].flip(-1).requires_grad_()
    shape = tuple(observed.shape)
    lf, hf = _templates(lf_tensor, hf_tensor)
    lf = replace(lf, latent_shape=shape)
    hf = replace(hf, latent_shape=shape)
    snapshots = [
        (value.detach().clone(), value.shape, value.stride(), value.requires_grad, value.grad)
        for value in (observed, lf_tensor, hf_tensor)
    ]

    result = _compute(observed, lf, hf)
    assert math.isfinite(result.blind_content_score)
    for value, snapshot in zip((observed, lf_tensor, hf_tensor), snapshots, strict=True):
        content, original_shape, original_stride, requires_grad, grad = snapshot
        torch.testing.assert_close(value.detach(), content, rtol=0.0, atol=0.0)
        assert value.shape == original_shape
        assert value.stride() == original_stride
        assert value.requires_grad is requires_grad
        assert value.grad is grad


@pytest.mark.quick
def test_real_formal_builders_interoperate_without_model_or_gpu() -> None:
    """两类正式builder的真实CPU输出可直接进入盲评分器。"""

    reference = torch.zeros((1, 2, 3, 5), dtype=torch.float64)
    lf = build_low_frequency_template(
        reference,
        "registered-score-key",
        MODEL_IDENTITY_DIGEST,
        prg_version=KEYED_PRG_VERSION,
    )
    hf = build_high_frequency_tail_template(
        reference,
        "registered-score-key",
        MODEL_IDENTITY_DIGEST,
        prg_version=KEYED_PRG_VERSION,
    )
    observed = (0.8 * lf.template + 0.2 * hf.template).to(torch.float64)
    result = _compute(observed, lf, hf)

    assert result.scoring_key_identity_digest == lf.scoring_key_identity_digest
    assert result.scoring_key_identity_digest == hf.scoring_key_identity_digest
    assert result.blind_lf_score == _independent_correlation(observed, lf.template)
    assert result.blind_hf_tail_score == _independent_correlation(
        observed,
        hf.template,
    )
    assert len(result.score_identity_digest) == 64


@pytest.mark.quick
def test_source_has_no_legacy_score_or_runtime_dependencies() -> None:
    """隔离模块不得回接旧评分、Jacobian、路由、模型或CUDA路径。"""

    source_path = Path(score_module.__file__)
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    called_names = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    called_attributes = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }

    assert "main.methods.carrier.keyed_tensor" not in source
    assert "normalized_correlation" not in imports
    assert "normalized_correlation" not in called_names
    assert "compute_blind_content_score" not in called_names
    assert not {
        "JacobianNullSpaceResult",
        "project_canonical_template",
        "jvp",
        "vjp",
        "topk",
        "kthvalue",
        "quantile",
    } & (imports | called_names | called_attributes)
    for forbidden in (
        "lf_mask",
        "hf_tail_mask",
        "saliency_map",
        "texture_map",
        "response_map",
        "local_sensitivity_map",
        "transformers",
        "diffusers",
        "cuda",
        "outputs/",
    ):
        assert forbidden not in source.lower()
