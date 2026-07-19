"""验证正式Q/K几何同步更新核的CPU性质，不冒充GPU qualification。"""

from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError, fields, replace
import inspect
from pathlib import Path
from typing import Any, Callable

import pytest
import torch

from main.core.digest import build_stable_digest, tensor_content_sha256
from main.core.keyed_prg import KEYED_PRG_VERSION
from main.methods.carrier.content_update import ContentCarrierUpdateResult
import main.methods.geometry as geometry_package
from main.methods.geometry.differentiable_attention import (
    ATTENTION_COORDINATE_CONVENTION,
    ATTENTION_GRID_ALIGN_CORNERS,
    ATTENTION_RELATION_COMPONENT_NAMES,
    ATTENTION_RELATION_COMPONENT_WEIGHTS,
    FROZEN_SD35_ATTENTION_MODULE_NAMES,
    AttentionGeometryGradient,
    DifferentiableAttentionRecorder,
    compute_attention_geometry_gradient,
)
from main.methods.geometry.sync_update import (
    GeometrySyncUpdate,
    build_attention_geometry_sync_update,
)
import main.methods.geometry.sync_update as sync_module


pytestmark = pytest.mark.unit

_SHAPE = (1, 4, 2, 2)
_KEY = "cpu_property_only_geometry_key"
_ENABLED_ROLES = (
    "full_dual_chain",
    "uniform_content_routing",
    "lf_only_content",
    "hf_tail_only_content",
)
_DISABLED_ROLES = (
    "content_chain_only",
    "geometry_recovery_without_embedded_sync",
)


class _ToyAttention(torch.nn.Module):
    """提供正式Q/K hook所需的轻量CPU投影。"""

    def __init__(self, dtype: torch.dtype) -> None:
        super().__init__()
        self.to_q = torch.nn.Linear(4, 4, bias=False, dtype=dtype)
        self.to_k = torch.nn.Linear(4, 4, bias=False, dtype=dtype)
        self.heads = 1
        with torch.no_grad():
            torch.nn.init.eye_(self.to_q.weight)
            torch.nn.init.eye_(self.to_k.weight)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """返回输入，Q/K由正式pre-hook直接构造。"""

        return hidden_states


def _content_update(
    dtype: torch.dtype = torch.float32,
    *,
    role: str = "full_dual_chain",
    capacity: float = 0.5,
) -> tuple[torch.Tensor, ContentCarrierUpdateResult]:
    """构造满足固定LF→HF公式的进程内上游结果。"""

    z10 = torch.tensor(
        [
            [
                [[0.30, 0.20], [-0.10, 0.40]],
                [[0.10, -0.20], [0.50, 0.30]],
                [[-0.40, 0.20], [0.10, 0.60]],
                [[0.20, 0.70], [-0.30, 0.10]],
            ]
        ],
        dtype=dtype,
    )
    lf_update = torch.full(_SHAPE, 1.0e-4, dtype=torch.float32)
    hf_update = torch.full(_SHAPE, -5.0e-5, dtype=torch.float32)
    if role == "lf_only_content":
        hf_update = torch.zeros_like(hf_update)
    elif role == "hf_tail_only_content":
        lf_update = torch.zeros_like(lf_update)
    content_only = z10.detach().float() + lf_update + hf_update
    result = ContentCarrierUpdateResult(
        geometry_capacity_map=torch.full((1, 1, 2, 2), capacity),
        lf_direction=torch.ones_like(content_only),
        hf_tail_direction=torch.ones_like(content_only),
        lf_update=lf_update,
        hf_tail_update=hf_update,
        content_only_latent_float32=content_only,
        latent_l2=float(torch.linalg.vector_norm(z10.float().reshape(-1)).item()),
        lf_nominal_strength=0.0025,
        hf_tail_nominal_strength=0.0015,
        method_role=role,
    )
    return z10, result


def _runtime(
    dtype: torch.dtype,
) -> tuple[
    DifferentiableAttentionRecorder,
    Callable[[torch.Tensor], torch.Tensor],
    list[tuple[str, torch.dtype]],
]:
    """构造使用正式层名的CPU ToyAttention recorder与forward。"""

    modules = (_ToyAttention(dtype), _ToyAttention(dtype))
    recorder = DifferentiableAttentionRecorder(
        tuple(zip(FROZEN_SD35_ATTENTION_MODULE_NAMES, modules)),
        max_tokens=4,
    )
    calls: list[tuple[str, torch.dtype]] = []

    def forward(latent: torch.Tensor) -> torch.Tensor:
        calls.append(("forward", latent.dtype))
        hidden_states = latent.flatten(2).transpose(1, 2).to(dtype=dtype)
        for module in modules:
            module(hidden_states)
        return latent

    return recorder, forward, calls


def _build(
    z10: torch.Tensor,
    content_update: ContentCarrierUpdateResult,
    *,
    score_spy: Callable[..., Any] | None = None,
) -> tuple[GeometrySyncUpdate, list[tuple[str, torch.dtype]], list[int]]:
    """执行一次CPU性质构造并返回forward/clear轨迹。"""

    recorder, forward, calls = _runtime(z10.dtype)
    clear_sizes: list[int] = []
    original_clear = recorder.clear

    def clear() -> None:
        clear_sizes.append(len(recorder.records))
        original_clear()

    recorder.clear = clear  # type: ignore[method-assign]
    original_score = sync_module.attention_geometry_score
    if score_spy is not None:
        sync_module.attention_geometry_score = score_spy
    try:
        result = build_attention_geometry_sync_update(
            current_scheduler_latent=z10,
            content_update=content_update,
            transformer_forward=forward,
            recorder=recorder,
            key_material=_KEY,
            prg_version=KEYED_PRG_VERSION,
        )
    finally:
        sync_module.attention_geometry_score = original_score
        recorder.close()
    return result, calls, clear_sizes


def _gradient_evidence(
    z10: torch.Tensor,
    content_update: ContentCarrierUpdateResult,
) -> AttentionGeometryGradient:
    """以真实CPU autograd构造可变异的合法gradient evidence。"""

    recorder, forward, _ = _runtime(z10.dtype)
    try:
        return compute_attention_geometry_gradient(
            content_update.content_only_latent_float32,
            forward,
            recorder,
            _KEY,
            prg_version=KEYED_PRG_VERSION,
            stable_token_fraction=0.5,
            unstable_pair_weight=0.25,
            component_weights=ATTENTION_RELATION_COMPONENT_WEIGHTS,
        )
    finally:
        recorder.close()


def test_public_contract_is_exact_frozen_and_isolated() -> None:
    """10字段、keyword-only接口与无package导出边界必须固定。"""

    assert sync_module.__all__ == [
        "GeometrySyncUpdate",
        "build_attention_geometry_sync_update",
    ]
    assert [field.name for field in fields(GeometrySyncUpdate)] == [
        "geometry_update",
        "accepted_scale",
        "backtracking_index",
        "relative_strength",
        "l2_budget",
        "relation_score_before",
        "relation_score_after",
        "qk_atomic_records_digest",
        "relation_template_identity_digest",
        "geometry_update_digest",
    ]
    signature = inspect.signature(build_attention_geometry_sync_update)
    assert list(signature.parameters) == [
        "current_scheduler_latent",
        "content_update",
        "transformer_forward",
        "recorder",
        "key_material",
        "prg_version",
    ]
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY
        for parameter in signature.parameters.values()
    )
    assert not hasattr(geometry_package, "GeometrySyncUpdate")
    assert not hasattr(geometry_package, "build_attention_geometry_sync_update")

    z10, content = _content_update()
    result, _, _ = _build(z10, content)
    with pytest.raises(FrozenInstanceError):
        result.backtracking_index = 4  # type: ignore[misc]


@pytest.mark.parametrize("dtype", (torch.float32, torch.float16))
def test_cpu_property_only_real_qk_autograd_formula(dtype: torch.dtype) -> None:
    """CPU ToyAttention只证明正式公式性质，不声称真实GPU资格化。"""

    z10, content = _content_update(dtype)
    z10_before = z10.clone()
    content_before = content.content_only_latent_float32.clone()
    result, calls, clear_sizes = _build(z10, content)

    z10_norm = torch.linalg.vector_norm(z10.detach().float().reshape(-1))
    expected_nominal = z10_norm * z10.float().new_tensor(0.0010)
    expected_budget = expected_nominal * z10.float().new_tensor(2.0).pow(
        -result.backtracking_index
    )
    assert result.relative_strength == float(z10.float().new_tensor(0.0010).item())
    assert result.accepted_scale == 2.0 ** (-result.backtracking_index)
    assert result.l2_budget == float(expected_budget.item())
    assert result.relation_score_after > result.relation_score_before
    assert result.geometry_update.dtype == torch.float32
    assert result.geometry_update.shape == _SHAPE
    assert torch.isfinite(result.geometry_update).all()
    assert len(result.qk_atomic_records_digest) == 64
    assert len(result.relation_template_identity_digest) == 64
    assert len(result.geometry_update_digest) == 64
    assert len(calls) == 2 + result.backtracking_index + 1
    assert len(clear_sizes) == len(calls)
    assert all(size in (0, 2) for size in clear_sizes)
    assert torch.equal(z10, z10_before)
    assert torch.equal(content.content_only_latent_float32, content_before)


def test_budget_uses_z10_norm_not_content_only_norm() -> None:
    """几何预算只能消费权威z10范数，不能改用更新后基底范数。"""

    z10, content = _content_update()
    result, _, _ = _build(z10, content)
    z10_budget = float((z10.float().norm() * z10.new_tensor(0.0010)).item())
    content_budget = float(
        (
            content.content_only_latent_float32.norm()
            * content.content_only_latent_float32.new_tensor(0.0010)
        ).item()
    )
    assert z10_budget != content_budget
    assert result.l2_budget == z10_budget * result.accepted_scale


@pytest.mark.parametrize("role", _ENABLED_ROLES)
def test_exact_geometry_enabled_roles_execute(role: str) -> None:
    """四个正式geometry-enabled角色都消费同一几何核。"""

    z10, content = _content_update(role=role)
    result, calls, _ = _build(z10, content)
    assert result.relation_score_after > result.relation_score_before
    assert len(calls) >= 3


@pytest.mark.parametrize("role", _DISABLED_ROLES)
def test_exact_geometry_disabled_roles_fail_before_qk(
    role: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """两个禁用角色必须在clear/forward/gradient前失败。"""

    z10, content = _content_update(role=role)
    recorder, forward, calls = _runtime(z10.dtype)
    gradient_calls = 0

    def forbidden_gradient(*args: Any, **kwargs: Any) -> Any:
        nonlocal gradient_calls
        gradient_calls += 1
        raise AssertionError("gradient不得调用")

    monkeypatch.setattr(sync_module, "compute_attention_geometry_gradient", forbidden_gradient)
    try:
        with pytest.raises(ValueError, match="禁止嵌入"):
            build_attention_geometry_sync_update(
                current_scheduler_latent=z10,
                content_update=content,
                transformer_forward=forward,
                recorder=recorder,
                key_material=_KEY,
                prg_version=KEYED_PRG_VERSION,
            )
    finally:
        recorder.close()
    assert gradient_calls == 0
    assert calls == []
    assert recorder.records == []


def test_capacity_map_is_single_channel_and_broadcasts() -> None:
    """A精确为[1,1,H,W]并按C通道广播。"""

    z10, content = _content_update()
    result, _, _ = _build(z10, content)
    assert content.geometry_capacity_map.shape == (1, 1, 2, 2)
    assert result.geometry_update.shape == (1, 4, 2, 2)

    invalid = replace(
        content,
        geometry_capacity_map=torch.ones(_SHAPE, dtype=torch.float32),
    )
    recorder, forward, calls = _runtime(z10.dtype)
    try:
        with pytest.raises(ValueError, match=r"\[1,1,H,W\]"):
            build_attention_geometry_sync_update(
                current_scheduler_latent=z10,
                content_update=invalid,
                transformer_forward=forward,
                recorder=recorder,
                key_material=_KEY,
                prg_version=KEYED_PRG_VERSION,
            )
    finally:
        recorder.close()
    assert calls == []


@pytest.mark.parametrize(
    "mutation",
    ("latent_l2", "content_only", "capacity", "nonfinite_z10"),
)
def test_content_formula_failures_precede_gradient(
    mutation: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """z10/content-update内容与公式必须在gradient前闭合。"""

    z10, content = _content_update()
    if mutation == "latent_l2":
        content = replace(content, latent_l2=content.latent_l2 + 1.0)
    elif mutation == "content_only":
        changed = content.content_only_latent_float32.clone()
        changed[0, 0, 0, 0] += 0.25
        content = replace(content, content_only_latent_float32=changed)
    elif mutation == "capacity":
        content = replace(content, geometry_capacity_map=torch.full((1, 1, 2, 2), 1.1))
    else:
        z10 = z10.clone()
        z10[0, 0, 0, 0] = float("nan")

    gradient_calls = 0

    def forbidden_gradient(*args: Any, **kwargs: Any) -> Any:
        nonlocal gradient_calls
        gradient_calls += 1
        raise AssertionError("gradient不得调用")

    monkeypatch.setattr(sync_module, "compute_attention_geometry_gradient", forbidden_gradient)
    recorder, forward, calls = _runtime(z10.dtype)
    try:
        with pytest.raises(ValueError):
            build_attention_geometry_sync_update(
                current_scheduler_latent=z10,
                content_update=content,
                transformer_forward=forward,
                recorder=recorder,
                key_material=_KEY,
                prg_version=KEYED_PRG_VERSION,
            )
    finally:
        recorder.close()
    assert gradient_calls == 0
    assert calls == []


def test_unsupported_prg_precedes_tensor_content_and_qk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """本核只验证调用方PRG受支持，并在内容读取前失败。"""

    z10, content = _content_update()
    content = replace(
        content,
        content_only_latent_float32=torch.full(_SHAPE, float("nan")),
    )
    recorder, forward, calls = _runtime(z10.dtype)
    try:
        with pytest.raises(ValueError, match="keyed_prg_version"):
            build_attention_geometry_sync_update(
                current_scheduler_latent=z10,
                content_update=content,
                transformer_forward=forward,
                recorder=recorder,
                key_material=_KEY,
                prg_version="unsupported",
            )
    finally:
        recorder.close()
    assert calls == []


@pytest.mark.parametrize(
    "mutation",
    (
        "content_digest",
        "gradient_norm",
        "layer_order",
        "active_components",
        "operator_ready",
        "atomic_ready",
        "pair_identity",
    ),
)
def test_gradient_evidence_drift_fails_before_baseline_forward(
    mutation: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """gradient evidence关键身份漂移不得进入actual baseline求值。"""

    z10, content = _content_update()
    evidence = _gradient_evidence(z10, content)
    if mutation == "content_digest":
        evidence = replace(evidence, evaluation_latent_content_sha256="0" * 64)
    elif mutation == "gradient_norm":
        evidence = replace(evidence, gradient_norm=evidence.gradient_norm + 1.0)
    elif mutation == "layer_order":
        evidence = replace(evidence, layer_names=tuple(reversed(evidence.layer_names)))
    elif mutation == "active_components":
        evidence = replace(
            evidence,
            attention_relation_active_component_names=ATTENTION_RELATION_COMPONENT_NAMES[:-1],
        )
    elif mutation == "operator_ready":
        evidence = replace(evidence, attention_relation_qk_operator_metadata_ready=False)
    elif mutation == "atomic_ready":
        evidence = replace(evidence, qk_atomic_content_ready=False)
    else:
        evidence = replace(evidence, stable_pair_weight_identity_digest="0" * 64)

    monkeypatch.setattr(
        sync_module,
        "compute_attention_geometry_gradient",
        lambda *args, **kwargs: evidence,
    )
    recorder, forward, calls = _runtime(z10.dtype)
    try:
        with pytest.raises((TypeError, ValueError)):
            build_attention_geometry_sync_update(
                current_scheduler_latent=z10,
                content_update=content,
                transformer_forward=forward,
                recorder=recorder,
                key_material=_KEY,
                prg_version=KEYED_PRG_VERSION,
            )
    finally:
        recorder.close()
    assert calls == []


def test_same_stable_pair_object_and_evaluation_order_are_used(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """baseline/candidate评分显式复用gradient产生的同一pair对象。"""

    z10, content = _content_update()
    original_gradient = sync_module.compute_attention_geometry_gradient
    original_score = sync_module.attention_geometry_score
    pair_ids: list[int] = []
    evidence_holder: list[AttentionGeometryGradient] = []

    def gradient_spy(*args: Any, **kwargs: Any) -> AttentionGeometryGradient:
        evidence = original_gradient(*args, **kwargs)
        evidence_holder.append(evidence)
        return evidence

    def score_spy(*args: Any, **kwargs: Any) -> Any:
        pair_ids.append(id(kwargs["stable_pair_weights"]))
        return original_score(*args, **kwargs)

    monkeypatch.setattr(sync_module, "compute_attention_geometry_gradient", gradient_spy)
    result, calls, clear_sizes = _build(z10, content, score_spy=score_spy)
    assert len(evidence_holder) == 1
    assert pair_ids == [id(evidence_holder[0].stable_pair_weights)] * (
        1 + result.backtracking_index + 1
    )
    assert len(calls) == len(clear_sizes) == 2 + result.backtracking_index + 1
    assert clear_sizes[0] == 0
    assert all(size == 2 for size in clear_sizes[1:])


@pytest.mark.parametrize(
    "field_name",
    (
        "component_identity_digest",
        "keyed_projection_digest",
        "qk_operator_metadata_digest",
        "coordinate_convention",
        "qk_atomic_content_ready",
    ),
)
def test_actual_dtype_relation_identity_drift_fails_at_baseline(
    field_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """baseline/candidate relation identity不得漂移评分模板或冻结常量。"""

    z10, content = _content_update()
    original_builder = sync_module.build_attention_relation_graph_identity

    def identity_spy(*args: Any, **kwargs: Any) -> Any:
        identity = original_builder(*args, **kwargs)
        replacement: Any
        if field_name == "coordinate_convention":
            replacement = "different_coordinate_convention"
        elif field_name == "qk_atomic_content_ready":
            replacement = False
        else:
            replacement = "0" * 64
        return replace(identity, **{field_name: replacement})

    monkeypatch.setattr(
        sync_module,
        "build_attention_relation_graph_identity",
        identity_spy,
    )
    recorder, forward, calls = _runtime(z10.dtype)
    try:
        with pytest.raises(ValueError):
            build_attention_geometry_sync_update(
                current_scheduler_latent=z10,
                content_update=content,
                transformer_forward=forward,
                recorder=recorder,
                key_material=_KEY,
                prg_version=KEYED_PRG_VERSION,
            )
    finally:
        recorder.close()
    assert len(calls) == 2


def test_supported_prg_validator_is_called_exactly_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """本核只在自身边界验证一次调用方PRG版本。"""

    z10, content = _content_update()
    original_validator = sync_module.require_supported_keyed_prg_version
    calls: list[str] = []

    def validator_spy(prg_version: str) -> None:
        calls.append(prg_version)
        original_validator(prg_version)

    monkeypatch.setattr(
        sync_module,
        "require_supported_keyed_prg_version",
        validator_spy,
    )
    _build(z10, content)
    assert calls == [KEYED_PRG_VERSION]


def test_zero_capacity_cannot_produce_success() -> None:
    """A全零必须在baseline前因方向零能量失败。"""

    z10, content = _content_update(capacity=0.0)
    recorder, forward, calls = _runtime(z10.dtype)
    try:
        with pytest.raises(ValueError, match="正有限能量"):
            build_attention_geometry_sync_update(
                current_scheduler_latent=z10,
                content_update=content,
                transformer_forward=forward,
                recorder=recorder,
                key_material=_KEY,
                prg_version=KEYED_PRG_VERSION,
            )
    finally:
        recorder.close()
    assert len(calls) == 1


def test_backtracking_accepts_first_strict_improvement_and_counts_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """回溯只接受首个严格改善，且每个实际评分candidate前clear。"""

    z10, content = _content_update()
    original_score = sync_module.attention_geometry_score
    score_call_index = 0

    def score_spy(*args: Any, **kwargs: Any) -> torch.Tensor:
        nonlocal score_call_index
        real = original_score(*args, **kwargs)
        if score_call_index == 0:
            resolved = real
        elif score_call_index < 3:
            resolved = real - 1.0
        else:
            resolved = real + 1.0
        score_call_index += 1
        return resolved

    result, calls, clear_sizes = _build(z10, content, score_spy=score_spy)
    assert result.backtracking_index == 2
    assert result.accepted_scale == 0.25
    assert result.relation_score_after > result.relation_score_before
    assert len(calls) == 5
    assert len(clear_sizes) == 5


def test_all_nine_scored_candidates_without_improvement_fail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """九项回溯均无严格改善时不得返回候选。"""

    z10, content = _content_update()
    original_score = sync_module.attention_geometry_score
    score_call_index = 0

    def score_spy(*args: Any, **kwargs: Any) -> torch.Tensor:
        nonlocal score_call_index
        real = original_score(*args, **kwargs)
        score_call_index += 1
        return real if score_call_index == 1 else real - 1.0

    recorder, forward, calls = _runtime(z10.dtype)
    original_clear = recorder.clear
    clear_count = 0

    def clear() -> None:
        nonlocal clear_count
        clear_count += 1
        original_clear()

    recorder.clear = clear  # type: ignore[method-assign]
    monkeypatch.setattr(sync_module, "attention_geometry_score", score_spy)
    try:
        with pytest.raises(ValueError, match="9项"):
            build_attention_geometry_sync_update(
                current_scheduler_latent=z10,
                content_update=content,
                transformer_forward=forward,
                recorder=recorder,
                key_material=_KEY,
                prg_version=KEYED_PRG_VERSION,
            )
    finally:
        recorder.close()
    assert len(calls) == 11
    assert clear_count == 11


def test_actual_dtype_zero_delta_candidates_do_not_forward() -> None:
    """未通过actual-dtype非零门禁的candidate不得执行Q/K forward。"""

    z10, content = _content_update(torch.float16, capacity=1.0e-12)
    recorder, forward, calls = _runtime(z10.dtype)
    try:
        with pytest.raises(ValueError, match="9项"):
            build_attention_geometry_sync_update(
                current_scheduler_latent=z10,
                content_update=content,
                transformer_forward=forward,
                recorder=recorder,
                key_material=_KEY,
                prg_version=KEYED_PRG_VERSION,
            )
    finally:
        recorder.close()
    assert len(calls) == 2


def test_qk_evaluation_records_use_exact_roles_keys_and_real_helper_signature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """三项联合证据必须按existing ready helper真实签名验证。"""

    z10, content = _content_update()
    original_ready = sync_module.qk_atomic_evaluation_records_ready
    captured: list[tuple[Any, ...]] = []

    def ready_spy(records: Any, digest: Any, **kwargs: Any) -> bool:
        captured.append((records, digest, kwargs))
        return original_ready(records, digest, **kwargs)

    monkeypatch.setattr(sync_module, "qk_atomic_evaluation_records_ready", ready_spy)
    result, _, _ = _build(z10, content)
    assert len(captured) == 1
    records, digest, kwargs = captured[0]
    assert type(records) is list
    assert tuple(record["qk_evaluation_role"] for record in records) == (
        "gradient_content_base_float32",
        "actual_dtype_content_baseline",
        "accepted_actual_dtype_candidate",
    )
    assert all(type(record) is dict for record in records)
    assert all(set(record) == sync_module._QK_EVALUATION_RECORD_KEYS for record in records)
    assert digest == result.qk_atomic_records_digest
    assert kwargs == {
        "aggregate_field_name": "qk_atomic_evaluation_records",
        "expected_roles": sync_module._QK_EVALUATION_ROLES,
        "expected_layer_names": FROZEN_SD35_ATTENTION_MODULE_NAMES,
        "require_evaluation_identity": True,
    }


def test_exact_relation_and_update_digest_payloads_are_independently_rebuilt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """两个摘要只能消费冻结22键与21键exact payload。"""

    z10, content = _content_update()
    original_digest = sync_module.build_stable_digest
    captured: list[dict[str, Any]] = []

    def digest_spy(payload: Any) -> str:
        captured.append(dict(payload))
        return original_digest(payload)

    monkeypatch.setattr(sync_module, "build_stable_digest", digest_spy)
    result, _, _ = _build(z10, content)
    assert len(captured) == 2
    relation_payload, update_payload = captured
    assert set(relation_payload) == sync_module._RELATION_TEMPLATE_IDENTITY_KEYS
    assert set(update_payload) == sync_module._GEOMETRY_UPDATE_DIGEST_KEYS
    assert "stable_token_positions" in relation_payload
    assert "stable_token_indices" in relation_payload
    assert relation_payload["formal_layer_names"] == FROZEN_SD35_ATTENTION_MODULE_NAMES
    assert relation_payload["attention_coordinate_convention"] == ATTENTION_COORDINATE_CONVENTION
    assert relation_payload["attention_grid_align_corners"] is ATTENTION_GRID_ALIGN_CORNERS
    assert "key_material" not in relation_payload
    assert "key_material" not in update_payload
    assert update_payload["geometry_update_rule"] == (
        "capacity_masked_direct_qk_monotonic_backtracking_v1"
    )
    assert update_payload["actual_dtype_delta_l2"] <= update_payload["l2_budget"]
    assert build_stable_digest(relation_payload) == result.relation_template_identity_digest
    assert build_stable_digest(update_payload) == result.geometry_update_digest
    assert update_payload["z10_float32_content_sha256"] == tensor_content_sha256(z10.float())
    assert update_payload["geometry_update_content_sha256"] == tensor_content_sha256(
        result.geometry_update
    )


def test_evaluation_record_extra_field_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """联合摘要不得接受改变exact六键结构的extra字段。"""

    z10, content = _content_update()
    original_record = sync_module._evaluation_record

    def record_with_extra(*args: Any, **kwargs: Any) -> dict[str, Any]:
        record = original_record(*args, **kwargs)
        record["extra"] = True
        return record

    monkeypatch.setattr(sync_module, "_evaluation_record", record_with_extra)
    with pytest.raises(ValueError, match="六键"):
        _build(z10, content)


def test_private_runtime_evidence_reuses_same_pair_for_postwrite_gate() -> None:
    """The integration gate must score final actual dtype with the same pair object."""

    z10, content = _content_update(torch.float16)
    recorder, forward, _ = _runtime(z10.dtype)
    try:
        result, evidence = sync_module._build_attention_geometry_sync_update_with_evidence(
            current_scheduler_latent=z10,
            content_update=content,
            transformer_forward=forward,
            recorder=recorder,
            key_material=_KEY,
            prg_version=KEYED_PRG_VERSION,
        )
        candidate = (content.content_only_latent_float32 + result.geometry_update).to(
            dtype=z10.dtype
        )
        score, digest = sync_module._evaluate_post_write_geometry_relation(
            written_latent=candidate,
            transformer_forward=forward,
            recorder=recorder,
            key_material=_KEY,
            runtime_evidence=evidence,
        )
    finally:
        recorder.close()
    assert score == pytest.approx(result.relation_score_after)
    assert len(digest) == 64
    assert evidence.stable_pair_weights is evidence.gradient_evidence.stable_pair_weights


def test_source_has_no_legacy_geometry_or_gpu_dependencies() -> None:
    """新核不得回退旧投影/组合路径或引入模型与CUDA执行。"""

    source_path = Path(sync_module.__file__)
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    forbidden = {
        "JacobianNullSpaceResult",
        "exact_jvp",
        "exact_vjp",
        "safe_subspace",
        "optimize_attention_geometry_update",
        "compose_ordered_float32_update_once",
        "transformers",
        "cuda",
    }
    names = {
        node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
    }
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert forbidden.isdisjoint(names | imports)
