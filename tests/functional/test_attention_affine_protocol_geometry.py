"""验证攻击配置无关的连续 Q/K 关系图仿射注册。"""

from __future__ import annotations

from dataclasses import replace
import math
import random

import pytest
import torch

import main.methods.geometry.attention_alignment as alignment_module
from main.methods.geometry.attention_alignment import (
    recover_attention_affine_alignment,
)
from main.methods.geometry.differentiable_attention import (
    ATTENTION_COORDINATE_CONVENTION,
    ATTENTION_GRID_ALIGN_CORNERS,
    ATTENTION_RELATION_COMPONENT_NAMES,
    QKAttentionRelation,
    StableAttentionTokenSelection,
    build_attention_relation_descriptor,
    build_qk_atomic_content_metadata,
    build_stable_attention_pair_weights,
    keyed_relation_signs,
    public_token_grid_coordinates,
    select_stable_attention_tokens,
    transport_stable_attention_pair_weights,
)


_TOKEN_COUNT = 64
_TOKEN_INDICES = tuple(range(_TOKEN_COUNT))
_KEY_MATERIAL = "held_out_affine_relation_key"
_LAYER_NAME = "held_out_affine_relation_layer"
_RESERVED_ROTATION_MAGNITUDES = (5.0, 7.0)
_RESERVED_SCALE_VALUES = (
    0.75,
    0.78,
    0.80,
    0.82,
    1.0 / 0.75,
    1.0 / 0.78,
    1.0 / 0.80,
    1.0 / 0.82,
)


def _qk_operator_metadata(
    layer_name: str,
    centered_logits: torch.Tensor,
    probabilities: torch.Tensor,
) -> dict[str, object]:
    """构造直接 Q/K 关系测试使用的完整算子元数据。"""

    return {
        "module_layer_name": layer_name,
        "module_class_name": "tests.DirectQKRelation",
        "head_count": 1,
        "head_width": 1,
        "attention_scale": 1.0,
        "attention_scale_source": "inverse_sqrt_head_width",
        "q_normalization_applied": False,
        "k_normalization_applied": False,
        "q_normalization_class": "",
        "k_normalization_class": "",
        "source_token_count": _TOKEN_COUNT,
        "source_grid_side": 8,
        "sampled_token_count": _TOKEN_COUNT,
        "sampled_grid_side": 8,
        "sampled_token_indices": list(_TOKEN_INDICES),
        "coordinate_convention": ATTENTION_COORDINATE_CONVENTION,
        "grid_align_corners": ATTENTION_GRID_ALIGN_CORNERS,
        **build_qk_atomic_content_metadata(
            layer_name,
            centered_logits,
            probabilities,
            centered_logits,
            probabilities,
            _TOKEN_INDICES,
        ),
        "centered_logit_aggregation": (
            "mean_of_per_head_row_centered_sampled_qk_logits"
        ),
        "relation_probability_aggregation": (
            "mean_of_per_head_sampled_image_token_probabilities"
        ),
        "mean_probability_is_softmax_of_mean_logits": False,
    }


def _grid_coordinates() -> torch.Tensor:
    """独立构造8x8归一化网格, 不调用被测配准实现。"""

    axis = torch.linspace(-1.0, 1.0, 8)
    return torch.tensor(
        [(float(x), float(y)) for y in axis for x in axis],
        dtype=torch.float32,
    )


@pytest.mark.quick
def test_public_token_coordinates_use_corner_center_endpoints() -> None:
    """公开 token 坐标必须与 align_corners=True 的角点中心完全一致."""

    coordinates = public_token_grid_coordinates((0, 2, 6, 8), "cpu")

    assert ATTENTION_GRID_ALIGN_CORNERS is True
    assert ATTENTION_COORDINATE_CONVENTION == (
        "normalized_xy_token_centers_corner_endpoints_v1"
    )
    assert torch.equal(
        coordinates,
        torch.tensor(
            (
                (-1.0, -1.0),
                (1.0, -1.0),
                (-1.0, 1.0),
                (1.0, 1.0),
            )
        ),
    )


def _affine_matrix(
    rotation_degrees: float,
    scale: float,
    translation_x: float,
    translation_y: float,
) -> torch.Tensor:
    """独立构造规范坐标到观测坐标的相似仿射矩阵。"""

    angle = math.radians(rotation_degrees)
    cosine = math.cos(angle) * scale
    sine = math.sin(angle) * scale
    return torch.tensor(
        ((cosine, -sine, translation_x), (sine, cosine, translation_y)),
        dtype=torch.float32,
    )


def _invert_affine(transform: torch.Tensor) -> torch.Tensor:
    """以齐次矩阵形式独立求逆。"""

    homogeneous = torch.eye(3)
    homogeneous[:2] = transform
    return torch.linalg.inv(homogeneous)[:2]


def _apply_affine(points: torch.Tensor, transform: torch.Tensor) -> torch.Tensor:
    """独立应用二维仿射变换。"""

    homogeneous = torch.cat((points, torch.ones(points.shape[0], 1)), dim=1)
    return homogeneous @ transform.transpose(0, 1)


def _independent_bilinear_weights(target_coordinates: torch.Tensor) -> torch.Tensor:
    """以显式邻点公式构造双线性采样矩阵。

    此函数不导入被测模块的坐标、采样或仿射辅助函数, 因而可以发现实现与
    测试共同复用同一错误公式的问题。
    """

    side = 8
    step = 2.0 / float(side - 1)
    weights = torch.zeros(_TOKEN_COUNT, _TOKEN_COUNT)
    for target_index, (x_value, y_value) in enumerate(target_coordinates.tolist()):
        if not (-1.0 <= x_value <= 1.0 and -1.0 <= y_value <= 1.0):
            continue
        x_grid = (x_value + 1.0) / step
        y_grid = (y_value + 1.0) / step
        x_lower = min(side - 2, max(0, int(math.floor(x_grid))))
        y_lower = min(side - 2, max(0, int(math.floor(y_grid))))
        x_fraction = min(1.0, max(0.0, x_grid - x_lower))
        y_fraction = min(1.0, max(0.0, y_grid - y_lower))
        neighbors = (
            (y_lower, x_lower, (1.0 - x_fraction) * (1.0 - y_fraction)),
            (y_lower, x_lower + 1, x_fraction * (1.0 - y_fraction)),
            (y_lower + 1, x_lower, (1.0 - x_fraction) * y_fraction),
            (y_lower + 1, x_lower + 1, x_fraction * y_fraction),
        )
        for row, column, weight in neighbors:
            weights[target_index, row * side + column] += float(weight)
    return weights


def _continuous_observed_attention(
    rotation_degrees: float,
    scale: float,
    translation_x: float,
    translation_y: float,
) -> tuple[QKAttentionRelation, torch.Tensor]:
    """在查询轴和键轴上独立生成连续仿射后的 Q/K 关系图。"""

    coordinates = _grid_coordinates()
    expected_transform = _affine_matrix(
        rotation_degrees,
        scale,
        translation_x,
        translation_y,
    )
    inverse_transform = _invert_affine(expected_transform)
    canonical_coordinates_at_observation = _apply_affine(
        coordinates,
        inverse_transform,
    )
    observation_weights = _independent_bilinear_weights(
        canonical_coordinates_at_observation
    )
    relation_signs = keyed_relation_signs(
        torch.zeros(1, _TOKEN_COUNT, _TOKEN_COUNT),
        _KEY_MATERIAL,
        _LAYER_NAME,
    )
    observed_logits = (
        observation_weights
        @ (2.0 * relation_signs)
        @ observation_weights.transpose(0, 1)
    )
    batched_logits = observed_logits.unsqueeze(0)
    centered_logits = batched_logits - batched_logits.mean(
        dim=-1,
        keepdim=True,
    )
    probabilities = torch.softmax(batched_logits, dim=-1)
    return (
        QKAttentionRelation(
            centered_logits=centered_logits,
            probabilities=probabilities,
            metadata=_qk_operator_metadata(
                _LAYER_NAME,
                centered_logits,
                probabilities,
            ),
        ),
        expected_transform,
    )


def _stable_pair_weights(observed_attention: object):
    """为注册构造与盲检评分共享的稳定 token pair 权重。"""

    records = (
        (_LAYER_NAME, observed_attention, _TOKEN_INDICES),
        (f"{_LAYER_NAME}_replicate", observed_attention.clone(), _TOKEN_INDICES),
    )
    selection = select_stable_attention_tokens(records, stable_token_fraction=0.5)
    return build_stable_attention_pair_weights(
        records,
        selection,
        unstable_pair_weight=0.25,
    )


def _deterministic_random_held_out_affines() -> tuple[tuple[float, ...], ...]:
    """从连续定义域生成远离正式攻击取值的确定性随机仿射集合。"""

    generator = random.Random(260712)
    cases: list[tuple[float, ...]] = []
    while len(cases) < 8:
        rotation_degrees = generator.uniform(-27.0, 27.0)
        scale = math.exp(
            generator.uniform(
                math.log(0.88),
                math.log(1.16),
            )
        )
        translation_x = generator.uniform(-0.14, 0.14)
        translation_y = generator.uniform(-0.14, 0.14)
        if min(
            abs(abs(rotation_degrees) - value)
            for value in _RESERVED_ROTATION_MAGNITUDES
        ) < 3.0:
            continue
        if min(abs(scale - value) for value in _RESERVED_SCALE_VALUES) < 0.055:
            continue
        cases.append(
            (
                rotation_degrees,
                scale,
                translation_x,
                translation_y,
            )
        )
    return tuple(cases)


@pytest.mark.quick
@pytest.mark.parametrize(
    ("rotation_degrees", "scale", "translation_x", "translation_y"),
    _deterministic_random_held_out_affines(),
)
def test_generic_hierarchical_search_recovers_random_continuous_affine(
    rotation_degrees: float,
    scale: float,
    translation_x: float,
    translation_y: float,
) -> None:
    """远离正式攻击取值的随机连续变换仍应由通用层级搜索恢复。"""

    observed_attention, expected_transform = _continuous_observed_attention(
        rotation_degrees,
        scale,
        translation_x,
        translation_y,
    )
    pair_weights = _stable_pair_weights(observed_attention)
    result = recover_attention_affine_alignment(
        observed_attention,
        _KEY_MATERIAL,
        _LAYER_NAME,
        _TOKEN_INDICES,
        pair_weights,
    )

    recovered_transform = torch.tensor(result.affine_transform)
    coordinates = _grid_coordinates()
    coordinate_error = (
        _apply_affine(coordinates, recovered_transform)
        - _apply_affine(coordinates, expected_transform)
    ).norm(dim=-1).mean()
    identity_error = (
        coordinates - _apply_affine(coordinates, expected_transform)
    ).norm(dim=-1).mean()
    assert float(coordinate_error) < 0.08
    assert float(coordinate_error) < float(identity_error)
    assert not torch.allclose(
        recovered_transform,
        torch.tensor(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0))),
        atol=1e-3,
        rtol=0.0,
    )
    assert result.geometry_reliable is True
    assert result.observation_relation_score > 0.55
    assert result.registration_alignment_gain > 0.02
    assert result.registration_objective_margin > 0.0
    assert result.canonical_coverage_ratio >= 0.45
    assert result.observation_coverage_ratio >= 0.45
    assert result.stable_pair_weight_identity_digest == (
        pair_weights.pair_weight_identity_digest
    )
    assert result.metadata["stable_pair_weight_identity_ready"] is True
    assert len(result.canonical_token_weights) == _TOKEN_COUNT
    assert result.metadata["robust_estimator"] == (
        "attack_independent_hierarchical_affine_relation_search"
    )
    assert set(result.relation_component_scores) == set(
        ATTENTION_RELATION_COMPONENT_NAMES
    )
    assert set(result.observation_relation_component_scores) == set(
        ATTENTION_RELATION_COMPONENT_NAMES
    )
    assert result.metadata["attention_relation_direct_qk_source_ready"] is True
    assert result.metadata["attention_relation_qk_operator_metadata_ready"] is True
    assert len(result.attention_relation_qk_operator_metadata_digest) == 64


@pytest.mark.quick
def test_hierarchical_search_grid_does_not_encode_reserved_attack_values() -> None:
    """粗网格和全部局部尺度路径不得精确复述正式攻击或其逆尺度。"""

    coarse = alignment_module._coarse_affine_candidates(torch.device("cpu"))
    schedule = alignment_module._local_affine_refinement_schedule()
    rotation_scale_pairs: set[tuple[float, float]] = set()
    for transform in coarse:
        linear = transform[:, :2]
        determinant = float(torch.det(linear))
        scale = math.sqrt(abs(determinant))
        rotation = math.degrees(
            math.atan2(float(linear[1, 0]), float(linear[0, 0]))
        )
        rotation_scale_pairs.add((rotation, scale))

    for rotation_delta, log_scale_delta, _ in schedule:
        rotation_scale_pairs = {
            (
                rotation + rotation_offset,
                scale * math.exp(log_scale_offset),
            )
            for rotation, scale in rotation_scale_pairs
            for rotation_offset in (-rotation_delta, 0.0, rotation_delta)
            for log_scale_offset in (-log_scale_delta, 0.0, log_scale_delta)
        }

    assert all(
        not math.isclose(
            scale,
            reserved,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        for _, scale in rotation_scale_pairs
        for reserved in _RESERVED_SCALE_VALUES
    )
    assert all(
        not math.isclose(
            abs(rotation),
            reserved,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        for rotation, _ in rotation_scale_pairs
        for reserved in _RESERVED_ROTATION_MAGNITUDES
    )


@pytest.mark.quick
def test_all_multi_round_local_candidates_remain_inside_public_domain() -> None:
    """粗搜索和多轮局部细化的每个候选都必须满足严格公开定义域。"""

    coarse = alignment_module._coarse_affine_candidates(torch.device("cpu"))
    assert bool(alignment_module._bounded_similarity_candidate_mask(coarse).all())
    current_transforms = coarse
    for rotation_delta, log_scale_delta, translation_delta in (
        alignment_module._local_affine_refinement_schedule()
    ):
        next_transforms = []
        for current in current_transforms[::17]:
            local = alignment_module._local_affine_candidates(
                current,
                rotation_delta_degrees=rotation_delta,
                log_scale_delta=log_scale_delta,
                translation_delta=translation_delta,
            )
            assert bool(
                alignment_module._bounded_similarity_candidate_mask(local).all()
            )
            extremal_index = local[:, :, 2].abs().sum(dim=-1).argmax()
            next_transforms.append(local[int(extremal_index.item())])
        current_transforms = torch.stack(next_transforms)

    valid_boundary = alignment_module._affine_matrix(
        32.0,
        math.sqrt(2.0),
        0.28,
        -0.28,
        device="cpu",
    )
    outside = torch.stack(
        (
            alignment_module._affine_matrix(
                33.0,
                1.0,
                0.0,
                0.0,
                device="cpu",
            ),
            alignment_module._affine_matrix(
                0.0,
                1.50,
                0.0,
                0.0,
                device="cpu",
            ),
            alignment_module._affine_matrix(
                0.0,
                1.0,
                0.281,
                0.0,
                device="cpu",
            ),
        )
    )
    assert bool(
        alignment_module._bounded_similarity_candidate_mask(valid_boundary)[0]
    )
    assert not bool(alignment_module._bounded_similarity_candidate_mask(outside).any())


@pytest.mark.quick
def test_relation_registration_rejects_signature_free_attention() -> None:
    """没有密钥关系信息的均匀 attention 不得通过结构注册门禁。"""

    zero_logits = torch.zeros(1, _TOKEN_COUNT, _TOKEN_COUNT)
    uniform_probabilities = torch.softmax(zero_logits, dim=-1)
    uniform_attention = QKAttentionRelation(
        centered_logits=zero_logits,
        probabilities=uniform_probabilities,
        metadata=_qk_operator_metadata(
            _LAYER_NAME,
            zero_logits,
            uniform_probabilities,
        ),
    )
    descriptor = build_attention_relation_descriptor(
        uniform_attention,
        _TOKEN_INDICES,
    )
    assert torch.count_nonzero(descriptor.values[..., 3]).item() == 0
    pair_weights = _stable_pair_weights(uniform_attention)
    result = recover_attention_affine_alignment(
        uniform_attention,
        _KEY_MATERIAL,
        _LAYER_NAME,
        _TOKEN_INDICES,
        pair_weights,
    )

    for component_name in ATTENTION_RELATION_COMPONENT_NAMES:
        assert result.observation_relation_component_scores[
            component_name
        ] == pytest.approx(0.0, abs=1e-12)
    assert result.observation_relation_score == pytest.approx(0.0, abs=1e-12)
    assert result.geometry_reliable is False


@pytest.mark.quick
def test_probability_only_relation_is_rejected_before_formal_alignment() -> None:
    """只有关系概率的输入不得通过反推 logits 进入核心几何算子。"""

    direct_relation, _ = _continuous_observed_attention(
        18.0,
        0.95,
        0.08,
        -0.05,
    )
    probability_only = direct_relation.probabilities
    with pytest.raises(ValueError, match="必须直接提供冻结层 Q/K"):
        recover_attention_affine_alignment(
            probability_only,
            _KEY_MATERIAL,
            _LAYER_NAME,
            _TOKEN_INDICES,
            _stable_pair_weights(probability_only),
        )


@pytest.mark.quick
def test_distance_modulated_component_changes_registration_objective(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """注册必须真实消费第4通道, 将其清零应改变分量分数和候选目标。"""

    relation, _ = _continuous_observed_attention(18.0, 0.95, 0.08, -0.05)
    pair_weights = _stable_pair_weights(relation)
    baseline = recover_attention_affine_alignment(
        relation,
        _KEY_MATERIAL,
        _LAYER_NAME,
        _TOKEN_INDICES,
        pair_weights,
    )
    original_builder = alignment_module.build_attention_relation_descriptor

    def without_distance_modulation(attention: object, token_indices: tuple[int, ...]):
        """仅清零第4通道, 保留其余三通道和组件协议身份。"""

        descriptor = original_builder(attention, token_indices)
        values = descriptor.values.clone()
        values[..., 3] = 0.0
        return replace(descriptor, values=values)

    monkeypatch.setattr(
        alignment_module,
        "build_attention_relation_descriptor",
        without_distance_modulation,
    )
    changed = recover_attention_affine_alignment(
        relation,
        _KEY_MATERIAL,
        _LAYER_NAME,
        _TOKEN_INDICES,
        pair_weights,
    )
    component_name = "distance_modulated_centered_attention_probability"

    assert baseline.observation_relation_component_scores[
        component_name
    ] != pytest.approx(
        changed.observation_relation_component_scores[component_name],
        abs=1e-7,
    )
    assert baseline.registration_objective_score != pytest.approx(
        changed.registration_objective_score,
        abs=1e-7,
    )


@pytest.mark.quick
def test_pair_transport_interpolates_token_field_before_outer_product() -> None:
    """规范 pair 必须由 ``a'=W a`` 外积构造, 不能使用 ``W P W^T``。"""

    attention = torch.full(
        (1, _TOKEN_COUNT, _TOKEN_COUNT),
        1.0 / _TOKEN_COUNT,
    )
    records = (
        (_LAYER_NAME, attention, _TOKEN_INDICES),
        (f"{_LAYER_NAME}_replicate", attention.clone(), _TOKEN_INDICES),
    )
    selection = StableAttentionTokenSelection(
        token_positions=(0, 2, 4, 6),
        token_indices=(0, 2, 4, 6),
        stable_token_fraction=4.0 / _TOKEN_COUNT,
        selection_digest="explicit_transport_test_selection",
    )
    source = build_stable_attention_pair_weights(
        records,
        selection,
        unstable_pair_weight=0.25,
    )
    sampling = torch.eye(_TOKEN_COUNT)
    sampling[0].zero_()
    sampling[0, 0] = 0.25
    sampling[0, 1] = 0.75
    transported = transport_stable_attention_pair_weights(
        source,
        sampling,
        torch.ones(_TOKEN_COUNT, dtype=torch.bool),
        coordinate_space="transport_test_grid",
    )

    source_token_weights = torch.tensor(source.token_weights)
    expected_token_weights = sampling @ source_token_weights
    expected_pair_weights = (
        expected_token_weights[:, None]
        * expected_token_weights[None, :]
        * (1.0 - torch.eye(_TOKEN_COUNT))
    )
    actual_pair_weights = transported.pair_tensor(attention)
    source_pair_weights = source.pair_tensor(attention)
    matrix_transport = sampling @ source_pair_weights @ sampling.transpose(0, 1)

    assert torch.allclose(
        torch.tensor(transported.token_weights),
        expected_token_weights,
    )
    assert torch.allclose(actual_pair_weights, expected_pair_weights)
    assert not torch.allclose(actual_pair_weights, matrix_transport)
