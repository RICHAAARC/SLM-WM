"""验证正式几何攻击的双边 Q/K 关系图注册."""

from __future__ import annotations

import pytest
import torch

from main.methods.geometry.attention_alignment import (
    _affine_matrix,
    _apply_affine,
    _grid_coordinates,
    _invert_affine,
    _sampling_weights,
    recover_attention_affine_alignment,
)
from main.methods.geometry.differentiable_attention import keyed_relation_signs


_TOKEN_COUNT = 64
_TOKEN_INDICES = tuple(range(_TOKEN_COUNT))
_KEY_MATERIAL = "formal_affine_protocol_key"
_LAYER_NAME = "formal_affine_protocol_layer"


def _continuous_protocol_attention(
    rotation_degrees: float,
    crop_ratio: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """按 Q/K 关系图的双空间几何等变式生成确定性 attention.

    该构造直接在关系 logits 的查询空间和键空间应用同一个双线性仿射算子,
    再执行真实 attention 使用的 row-wise softmax. 它验证的是注册算子自身,
    不以图像相似度或隐藏状态相关性替代 Q/K 关系.
    """

    coordinates = _grid_coordinates(_TOKEN_INDICES, torch.device("cpu"))
    relation_signs = keyed_relation_signs(
        torch.zeros(1, _TOKEN_COUNT, _TOKEN_COUNT),
        _KEY_MATERIAL,
        _LAYER_NAME,
    )
    expected_transform = _affine_matrix(
        rotation_degrees,
        1.0 / crop_ratio,
        0.0,
        0.0,
        device="cpu",
    )
    inverse_transform = _invert_affine(expected_transform)
    canonical_coordinates_at_observation = _apply_affine(
        coordinates,
        inverse_transform,
    )
    observation_weights, _, _ = _sampling_weights(
        canonical_coordinates_at_observation.unsqueeze(0),
        coordinates,
    )
    relation_logits = 2.0 * relation_signs
    observed_logits = (
        observation_weights[0]
        @ relation_logits
        @ observation_weights[0].transpose(0, 1)
    )
    return torch.softmax(observed_logits, dim=-1).unsqueeze(0), expected_transform


@pytest.mark.quick
@pytest.mark.parametrize(
    ("rotation_degrees", "crop_ratio"),
    (
        (5.0, 1.0),
        (-5.0, 1.0),
        (7.0, 1.0),
        (-7.0, 1.0),
        (0.0, 0.82),
        (0.0, 0.80),
        (0.0, 0.78),
        (7.0, 0.78),
    ),
)
def test_formal_affine_protocol_recovers_exact_relation_transform(
    rotation_degrees: float,
    crop_ratio: float,
) -> None:
    """正式旋转和中心裁剪必须恢复生成观测关系图的精确候选."""

    observed_attention, expected_transform = _continuous_protocol_attention(
        rotation_degrees,
        crop_ratio,
    )

    result = recover_attention_affine_alignment(
        observed_attention,
        _KEY_MATERIAL,
        _LAYER_NAME,
        _TOKEN_INDICES,
    )

    recovered_transform = torch.tensor(result.affine_transform)
    assert torch.allclose(recovered_transform, expected_transform, atol=1e-6, rtol=0.0)
    assert result.geometry_reliable is True
    assert result.observation_relation_score > 0.85
    assert result.registration_alignment_gain > 0.25
    assert result.registration_objective_margin > 0.0
    assert result.canonical_coverage_ratio >= 0.45
    assert result.observation_coverage_ratio >= 0.45
    assert result.canonical_unique_ratio > 0.0
    assert result.observation_unique_ratio > 0.0

    metadata = result.metadata
    expected_bidirectional_score = (
        metadata["canonical_relation_weight"] * result.relation_sync_score
        + metadata["observation_relation_weight"]
        * result.observation_relation_score
    )
    expected_coverage_penalty = metadata[
        "registration_coverage_penalty_weight"
    ] * (
        (1.0 - result.canonical_coverage_ratio)
        + (1.0 - result.observation_coverage_ratio)
        + (1.0 - result.canonical_unique_ratio)
        + (1.0 - result.observation_unique_ratio)
    )
    assert result.bidirectional_relation_score == pytest.approx(
        expected_bidirectional_score,
        abs=1e-7,
    )
    assert result.registration_coverage_penalty == pytest.approx(
        expected_coverage_penalty,
        abs=1e-7,
    )
    assert result.registration_objective_score == pytest.approx(
        result.bidirectional_relation_score - result.registration_coverage_penalty,
        abs=1e-7,
    )


@pytest.mark.quick
def test_relation_registration_rejects_signature_free_attention() -> None:
    """没有密钥关系信息的均匀 attention 不得通过结构注册门禁."""

    uniform_attention = torch.full(
        (1, _TOKEN_COUNT, _TOKEN_COUNT),
        1.0 / _TOKEN_COUNT,
    )

    result = recover_attention_affine_alignment(
        uniform_attention,
        _KEY_MATERIAL,
        _LAYER_NAME,
        _TOKEN_INDICES,
    )

    assert result.observation_relation_score == pytest.approx(0.0, abs=1e-12)
    assert result.geometry_reliable is False
