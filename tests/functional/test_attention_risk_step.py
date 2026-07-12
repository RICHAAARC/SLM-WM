"""独立验证注意力风险最大步长、回溯和单次量化语义。"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch

import main.methods.geometry.differentiable_attention as attention_module


def _risk_bounded_fixture(
    direction: torch.Tensor,
    strength: float,
) -> attention_module.RiskBoundedUpdate:
    """构造与解析梯度同形的最小风险有界对象。"""

    unit_direction = direction.detach().float()
    unit_direction = unit_direction / unit_direction.norm()
    applied_strength = torch.tensor([strength], dtype=torch.float32)
    nominal_strength = applied_strength.clone()
    update = unit_direction * strength
    envelope = torch.full_like(update, update.abs().max())
    return attention_module.RiskBoundedUpdate(
        branch_name="attention_geometry",
        unit_direction=unit_direction,
        effective_budget=torch.ones_like(update),
        amplitude_envelope=envelope,
        update=update,
        nominal_strength=nominal_strength,
        applied_strength=applied_strength,
        risk_scale_factor=torch.ones_like(applied_strength),
        maximum_envelope_ratio=torch.ones_like(applied_strength),
        budget_ceiling=1.0,
        direction_epsilon=1e-12,
        numerical_epsilon=1e-12,
    )


class _CandidateRecorder:
    """保存测试前向收到的实际候选 Tensor。"""

    def __init__(self) -> None:
        self.records: tuple[Any, ...] = ()
        self.candidates: list[torch.Tensor] = []

    def clear(self) -> None:
        """清除上一候选的关系记录。"""

        self.records = ()

    def forward(self, candidate: torch.Tensor) -> torch.Tensor:
        """记录候选并提供可供身份构造消费的层记录。"""

        self.candidates.append(candidate.detach().clone())
        self.records = (("layer_a", candidate, (0, 1, 2, 3)),)
        return candidate


def _gradient_evidence(
    latent: torch.Tensor,
    score: float,
) -> SimpleNamespace:
    """构造只包含回溯协议所需字段的冻结 Q/K 梯度证据。"""

    digest = "a" * 64
    component_names = attention_module.ATTENTION_RELATION_COMPONENT_NAMES
    return SimpleNamespace(
        gradient=torch.ones_like(latent, dtype=torch.float32),
        evaluation_latent_content_sha256=(
            attention_module.tensor_content_sha256(latent.detach().float())
        ),
        score_before=float(score),
        gradient_norm=float(torch.ones_like(latent).float().norm().item()),
        layer_names=("layer_a", "layer_b"),
        stable_token_positions=(0, 1, 2, 3),
        stable_token_indices=(0, 1, 2, 3),
        stable_token_selection_digest=digest,
        stable_pair_weight_identity_digest=digest,
        stable_pair_weight_realization_digest=digest,
        attention_relation_component_names=component_names,
        attention_relation_active_component_names=component_names,
        attention_relation_component_weights=(0.25, 0.25, 0.25, 0.25),
        attention_relation_component_protocol_digest=digest,
        attention_relation_source=attention_module.DIRECT_QK_RELATION_SOURCE,
        attention_relation_component_identity_digest=digest,
        attention_relation_keyed_projection_digest=digest,
        attention_relation_soft_rank_temperature=0.25,
        attention_relation_soft_rank_scale=0.25,
        attention_relation_relative_distance_scale=2.0 * (2.0**0.5),
        attention_relation_qk_operator_metadata_records=({"layer_name": "layer_a"},),
        attention_relation_qk_operator_metadata_digest=digest,
        attention_relation_qk_operator_metadata_ready=True,
        qk_atomic_content_records=({"layer_name": "layer_a"},),
        qk_atomic_content_digest=digest,
        qk_atomic_content_ready=True,
        stable_token_fraction=0.5,
        unstable_pair_weight=0.25,
        stable_pair_weights=SimpleNamespace(identity=digest),
    )


def _install_risk_step_fixture(
    monkeypatch: pytest.MonkeyPatch,
    *,
    latent: torch.Tensor,
    original_score: float,
    content_base_score: float,
    candidate_scores: list[float],
    base_update: torch.Tensor | None = None,
) -> tuple[_CandidateRecorder, list[torch.Tensor], SimpleNamespace]:
    """安装最小真实协议边界, 只隔离昂贵 Transformer 数值。"""

    recorder = _CandidateRecorder()
    content_base_inputs: list[torch.Tensor] = []
    original_evidence = _gradient_evidence(latent, original_score)
    resolved_base_update = (
        torch.zeros_like(latent, dtype=torch.float32)
        if base_update is None
        else base_update
    )
    _, content_base_latent, _ = (
        attention_module.compose_ordered_float32_update_once(
            original_latent=latent,
            branch_update_tensors={"lf_content": resolved_base_update},
            common_scale=1.0,
        )
    )
    content_base_evidence = _gradient_evidence(
        content_base_latent,
        content_base_score,
    )
    scores = iter(candidate_scores)

    def compute_content_base(
        candidate: torch.Tensor,
        *_: Any,
        **__: Any,
    ) -> SimpleNamespace:
        content_base_inputs.append(candidate.detach().clone())
        return content_base_evidence

    def candidate_score(*_: Any, **__: Any) -> torch.Tensor:
        return torch.tensor(next(scores), dtype=torch.float32)

    accepted_identity = SimpleNamespace(
        qk_atomic_content_ready=True,
        qk_atomic_content_records=({"layer_name": "layer_a"},),
        qk_atomic_content_digest="b" * 64,
    )
    monkeypatch.setattr(
        attention_module,
        "compute_attention_geometry_gradient",
        compute_content_base,
    )
    monkeypatch.setattr(
        attention_module,
        "attention_geometry_score",
        candidate_score,
    )
    monkeypatch.setattr(
        attention_module,
        "build_attention_relation_graph_identity",
        lambda *_args, **_kwargs: accepted_identity,
    )
    monkeypatch.setattr(
        attention_module,
        "qk_atomic_evaluation_records_digest",
        lambda *_args, **_kwargs: "c" * 64,
    )
    safe_subspace = SimpleNamespace(
        project=lambda value: value,
        solver_digest="d" * 64,
    )
    return recorder, content_base_inputs, SimpleNamespace(
        original_evidence=original_evidence,
        content_base_evidence=content_base_evidence,
        content_base_latent=content_base_latent,
        safe_subspace=safe_subspace,
    )


@pytest.mark.quick
def test_attention_risk_step_starts_at_maximum_and_consumes_factor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """首个候选必须使用风险上界, 后续强度只按冻结因子递减。"""

    latent = torch.zeros((1, 1), dtype=torch.float32)
    recorder, _, fixture = _install_risk_step_fixture(
        monkeypatch,
        latent=latent,
        original_score=0.4,
        content_base_score=0.5,
        candidate_scores=[0.5, 0.6],
    )

    result = attention_module.optimize_attention_geometry_update(
        latent=latent,
        transformer_forward=recorder.forward,
        recorder=recorder,
        key_material="risk-step-key",
        safe_subspace=fixture.safe_subspace,
        risk_bounded_update=_risk_bounded_fixture(latent + 1.0, 0.8),
        backtracking_factor=0.25,
        maximum_backtracking_steps=3,
        precomputed_gradient=fixture.original_evidence,
        precomputed_content_base_gradient=fixture.content_base_evidence,
        base_update=torch.zeros_like(latent),
    )

    strengths = [float(candidate.item()) for candidate in recorder.candidates]
    assert strengths == pytest.approx([0.8, 0.2])
    assert max(strengths) <= 0.8 + 1e-6
    assert result.applied_update_strength == pytest.approx(0.2)
    assert result.backtracking_step_count == 1
    assert result.metadata["maximum_update_strength"] == pytest.approx(0.8)
    assert result.metadata["backtracking_factor"] == pytest.approx(0.25)
    assert result.metadata["maximum_backtracking_steps"] == 3
    assert result.score_after > result.score_before
    assert result.score_after > result.content_base_score


@pytest.mark.quick
def test_attention_optimizer_consumes_exact_risk_bounded_direction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Q/K 回溯必须直接消费风险包络冻结的同一单位方向 Tensor。"""

    latent = torch.zeros((1, 2), dtype=torch.float32)
    recorder, _, fixture = _install_risk_step_fixture(
        monkeypatch,
        latent=latent,
        original_score=0.0,
        content_base_score=0.0,
        candidate_scores=[1.0],
    )
    risk_direction = torch.full_like(latent, 2.0**-0.5)
    risk_bound = _risk_bounded_fixture(risk_direction, 0.5)

    result = attention_module.optimize_attention_geometry_update(
        latent=latent,
        transformer_forward=recorder.forward,
        recorder=recorder,
        key_material="risk-direction-key",
        safe_subspace=fixture.safe_subspace,
        risk_bounded_update=risk_bound,
        maximum_backtracking_steps=0,
        precomputed_gradient=fixture.original_evidence,
        precomputed_content_base_gradient=fixture.content_base_evidence,
        base_update=torch.zeros_like(latent),
    )

    assert result.unit_update_content_sha256 == (
        attention_module.tensor_content_sha256(risk_bound.unit_direction)
    )
    assert torch.equal(result.update, risk_bound.update)


@pytest.mark.quick
def test_attention_optimizer_rejects_different_risk_direction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """风险方向与真实 Q/K 安全投影不同即失败, 不得只在写回端替换。"""

    latent = torch.zeros((1, 2), dtype=torch.float32)
    recorder, _, fixture = _install_risk_step_fixture(
        monkeypatch,
        latent=latent,
        original_score=0.0,
        content_base_score=0.0,
        candidate_scores=[1.0],
    )

    with pytest.raises(RuntimeError, match="风险有界单位方向"):
        attention_module.optimize_attention_geometry_update(
            latent=latent,
            transformer_forward=recorder.forward,
            recorder=recorder,
            key_material="risk-direction-mismatch-key",
            safe_subspace=fixture.safe_subspace,
            risk_bounded_update=_risk_bounded_fixture(
                torch.tensor([[1.0, -1.0]]),
                0.5,
            ),
            maximum_backtracking_steps=0,
            precomputed_gradient=fixture.original_evidence,
            precomputed_content_base_gradient=fixture.content_base_evidence,
            base_update=torch.zeros_like(latent),
        )


@pytest.mark.quick
@pytest.mark.parametrize(
    ("evidence_role", "message"),
    (
        ("original", "当前原始 latent"),
        ("content_base", "当前内容基底 latent"),
    ),
)
def test_attention_optimizer_rejects_gradient_latent_identity_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    evidence_role: str,
    message: str,
) -> None:
    """预计算梯度与实际求值 latent 不一致时必须立即失败。"""

    latent = torch.zeros((1, 2), dtype=torch.float32)
    recorder, content_base_inputs, fixture = _install_risk_step_fixture(
        monkeypatch,
        latent=latent,
        original_score=0.0,
        content_base_score=0.0,
        candidate_scores=[1.0],
    )
    if evidence_role == "original":
        fixture.original_evidence.evaluation_latent_content_sha256 = "f" * 64
    else:
        fixture.content_base_evidence.evaluation_latent_content_sha256 = "f" * 64

    with pytest.raises(RuntimeError, match=message):
        attention_module.optimize_attention_geometry_update(
            latent=latent,
            transformer_forward=recorder.forward,
            recorder=recorder,
            key_material="gradient-latent-identity-key",
            safe_subspace=fixture.safe_subspace,
            risk_bounded_update=_risk_bounded_fixture(latent + 1.0, 0.5),
            precomputed_gradient=fixture.original_evidence,
            precomputed_content_base_gradient=fixture.content_base_evidence,
            maximum_backtracking_steps=0,
            base_update=torch.zeros_like(latent),
        )

    assert content_base_inputs == []
    assert recorder.candidates == []


@pytest.mark.quick
def test_attention_risk_step_uses_one_actual_dtype_cast(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """候选必须先合成 float32 三项, 再只量化一次并返回 float32 更新。"""

    latent = torch.tensor([[1.0]], dtype=torch.float16)
    base_update = torch.tensor([[0.0004]], dtype=torch.float32)
    recorder, content_base_inputs, fixture = _install_risk_step_fixture(
        monkeypatch,
        latent=latent,
        original_score=0.0,
        content_base_score=0.0,
        candidate_scores=[1.0],
        base_update=base_update,
    )

    result = attention_module.optimize_attention_geometry_update(
        latent=latent,
        transformer_forward=recorder.forward,
        recorder=recorder,
        key_material="single-cast-key",
        safe_subspace=fixture.safe_subspace,
        risk_bounded_update=_risk_bounded_fixture(
            torch.ones_like(latent),
            0.0004,
        ),
        backtracking_factor=0.5,
        maximum_backtracking_steps=0,
        precomputed_gradient=fixture.original_evidence,
        precomputed_content_base_gradient=fixture.content_base_evidence,
        base_update=base_update,
    )

    expected_content_base = (latent.float() + base_update).to(latent.dtype)
    _, expected_candidate, _ = (
        attention_module.compose_ordered_float32_update_once(
            original_latent=latent,
            branch_update_tensors={
                "lf_content": base_update,
                "attention_geometry": result.update.float(),
            },
            common_scale=1.0,
        )
    )
    separately_quantized = (
        latent
        + base_update.to(latent.dtype)
        + result.update.to(latent.dtype)
    )

    assert content_base_inputs == []
    assert fixture.content_base_latent.dtype == latent.dtype
    assert torch.equal(fixture.content_base_latent, expected_content_base)
    assert recorder.candidates[0].dtype == latent.dtype
    assert torch.equal(recorder.candidates[0], expected_candidate)
    assert not torch.equal(recorder.candidates[0], separately_quantized)
    assert result.update.dtype == torch.float32
    assert result.metadata["candidate_composition_protocol"] == (
        "ordered_float32_branch_sum_then_latent_add_single_cast_v1"
    )
    assert result.metadata["returned_update_dtype"] == "float32"


@pytest.mark.quick
def test_attention_risk_step_rejects_non_monotonic_candidates_at_step_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """初始候选及允许回溯次数全部不单调时必须失败。"""

    latent = torch.zeros((1, 1), dtype=torch.float32)
    recorder, _, fixture = _install_risk_step_fixture(
        monkeypatch,
        latent=latent,
        original_score=0.4,
        content_base_score=0.5,
        candidate_scores=[0.5, 0.5, 0.5],
    )

    with pytest.raises(RuntimeError, match="仍未提高真实 Q/K 目标"):
        attention_module.optimize_attention_geometry_update(
            latent=latent,
            transformer_forward=recorder.forward,
            recorder=recorder,
            key_material="step-limit-key",
            safe_subspace=fixture.safe_subspace,
            risk_bounded_update=_risk_bounded_fixture(latent + 1.0, 0.8),
            backtracking_factor=0.5,
            maximum_backtracking_steps=2,
            precomputed_gradient=fixture.original_evidence,
            precomputed_content_base_gradient=fixture.content_base_evidence,
            base_update=torch.zeros_like(latent),
        )

    assert [float(candidate.item()) for candidate in recorder.candidates] == (
        pytest.approx([0.8, 0.4, 0.2])
    )


@pytest.mark.quick
@pytest.mark.parametrize(
    ("backtracking_factor", "maximum_backtracking_steps", "message"),
    (
        (1.0, 1, "backtracking_factor"),
        (0.5, -1, "maximum_backtracking_steps"),
    ),
)
def test_attention_risk_step_rejects_invalid_search_constants(
    backtracking_factor: float,
    maximum_backtracking_steps: int,
    message: str,
) -> None:
    """核心函数必须拒绝无法形成收缩序列的搜索常量。"""

    latent = torch.zeros((1, 1), dtype=torch.float32)
    recorder = _CandidateRecorder()
    safe_subspace = SimpleNamespace(project=lambda value: value, solver_digest="d" * 64)

    with pytest.raises(ValueError, match=message):
        attention_module.optimize_attention_geometry_update(
            latent=latent,
            transformer_forward=recorder.forward,
            recorder=recorder,
            key_material="invalid-search-key",
            safe_subspace=safe_subspace,
            risk_bounded_update=_risk_bounded_fixture(latent + 1.0, 0.8),
            precomputed_gradient=_gradient_evidence(latent, 0.0),
            precomputed_content_base_gradient=_gradient_evidence(latent, 0.0),
            backtracking_factor=backtracking_factor,
            maximum_backtracking_steps=maximum_backtracking_steps,
        )
