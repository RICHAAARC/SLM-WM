"""根据密钥注意力关系图恢复图像 token 网格的仿射参考系。

几何变换会同时重排四分量关系张量的查询轴和键轴。若观测关系张量为 ``R_obs``,
注册目标对四个通道分别计算 ``W R_obs,c W^T`` 的规范拉回一致性, 并计算
``V (pi_c S_key) V^T`` 的观测前推一致性, 最后显式记录双向覆盖惩罚。
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, Mapping

from main.core.digest import build_stable_digest
from main.methods.geometry.differentiable_attention import (
    ATTENTION_COORDINATE_CONVENTION,
    ATTENTION_GRID_ALIGN_CORNERS,
    ATTENTION_RELATION_COMPONENT_NAMES,
    ATTENTION_RELATION_COMPONENT_WEIGHTS,
    DIRECT_QK_RELATION_SOURCE,
    StableAttentionPairWeights,
    attention_probability,
    attention_relation_component_scores,
    build_attention_relation_descriptor,
    build_attention_relation_graph_identity,
    combine_attention_relation_component_scores,
    keyed_attention_relation_projection,
    public_token_grid_coordinates,
    transport_stable_attention_pair_weights,
    validate_attention_relation_component_weights,
)


_CANONICAL_RELATION_WEIGHT = 0.10
_OBSERVATION_RELATION_WEIGHT = 0.90
_COVERAGE_PENALTY_WEIGHT = 0.01
_MINIMUM_REGISTRATION_COVERAGE = 0.45
ATTENTION_ALIGNMENT_ANCHOR_COUNT = 12
ATTENTION_ALIGNMENT_RESIDUAL_THRESHOLD = 0.20
ATTENTION_ALIGNMENT_MINIMUM_INLIER_RATIO = 0.50
ATTENTION_IMAGE_RESAMPLING_MODE = "bilinear"
ATTENTION_IMAGE_PADDING_MODE = "border"
ATTENTION_IMAGE_QUANTIZATION_PROTOCOL = (
    "clamp_0_1_multiply_255_floor_uint8_rgb_v1"
)
_SIMILARITY_ROTATION_BOUND_DEGREES = 32.0
_SIMILARITY_ROTATION_INTERVAL_COUNT = 4
_SIMILARITY_LOG_SCALE_ANCHOR_BOUND = math.log(math.sqrt(2.0))
_SIMILARITY_TRANSLATION_ANCHOR_BOUND = 0.28
_SIMILARITY_LOCAL_REFINEMENT_ROUNDS = 3
_SIMILARITY_LOCAL_REFINEMENT_RATIO = 3.0
_DIHEDRAL_LINEAR_PARTS = (
    ((1.0, 0.0), (0.0, 1.0)),
    ((-1.0, 0.0), (0.0, 1.0)),
    ((1.0, 0.0), (0.0, -1.0)),
    ((-1.0, 0.0), (0.0, -1.0)),
    ((0.0, -1.0), (1.0, 0.0)),
    ((0.0, 1.0), (-1.0, 0.0)),
    ((0.0, 1.0), (1.0, 0.0)),
    ((0.0, -1.0), (-1.0, 0.0)),
)


def _torch() -> Any:
    """延迟导入 PyTorch。"""

    import torch

    return torch


def resample_attention_aligned_rgb_uint8(
    image_rgb_uint8: Any,
    affine_transform: Any,
) -> Any:
    """按冻结图像协议恢复参考系并返回 RGB uint8 Tensor.

    输入必须是 ``[1, 3, H, W]`` 的 RGB uint8 Tensor. 仿射网格和图像采样
    统一使用 ``align_corners=True``；边界复制避免几何恢复引入额外黑边。输出先
    截断到 ``[0, 1]``，乘以255后执行 floor, 从而把连续采样结果唯一映射回
    仅图像检测重新编码所消费的 RGB 字节。
    """

    torch = _torch()
    import torch.nn.functional as functional

    if (
        not isinstance(image_rgb_uint8, torch.Tensor)
        or image_rgb_uint8.dtype != torch.uint8
        or image_rgb_uint8.ndim != 4
        or tuple(image_rgb_uint8.shape[:2]) != (1, 3)
        or int(image_rgb_uint8.shape[-2]) < 2
        or int(image_rgb_uint8.shape[-1]) < 2
    ):
        raise ValueError("图像配准输入必须为 [1, 3, H, W] RGB uint8 Tensor")
    tensor = image_rgb_uint8.to(dtype=torch.float32) / 255.0
    theta = torch.as_tensor(
        affine_transform,
        dtype=tensor.dtype,
        device=tensor.device,
    )
    if theta.shape != (2, 3) or not bool(torch.isfinite(theta).all().item()):
        raise ValueError("图像配准仿射矩阵必须为有限的 [2, 3] Tensor")
    grid = functional.affine_grid(
        theta.unsqueeze(0),
        tensor.shape,
        align_corners=ATTENTION_GRID_ALIGN_CORNERS,
    )
    aligned = functional.grid_sample(
        tensor,
        grid,
        mode=ATTENTION_IMAGE_RESAMPLING_MODE,
        padding_mode=ATTENTION_IMAGE_PADDING_MODE,
        align_corners=ATTENTION_GRID_ALIGN_CORNERS,
    )
    return torch.floor(aligned.clamp(0.0, 1.0) * 255.0).to(
        dtype=torch.uint8
    )


def validate_attention_alignment_gate(
    anchor_count: int,
    residual_threshold: float,
    minimum_inlier_ratio: float,
) -> tuple[int, float, float]:
    """集中校验可持久化的注意力配准结构门禁参数."""

    if type(anchor_count) is not int or anchor_count < 3:
        raise ValueError("attention_anchor_count 必须为不小于3的整数")
    if (
        isinstance(residual_threshold, bool)
        or not isinstance(residual_threshold, (int, float))
        or not math.isfinite(residual_threshold)
        or residual_threshold <= 0.0
    ):
        raise ValueError("attention_residual_threshold 必须为正有限数")
    if not (
        not isinstance(minimum_inlier_ratio, bool)
        and isinstance(minimum_inlier_ratio, (int, float))
        and math.isfinite(minimum_inlier_ratio)
        and 0.0 < minimum_inlier_ratio <= 1.0
    ):
        raise ValueError(
            "attention_minimum_inlier_ratio 必须位于 (0, 1]"
        )
    return anchor_count, float(residual_threshold), float(minimum_inlier_ratio)


def attention_alignment_gate_record(
    anchor_count: int,
    residual_threshold: float,
    minimum_inlier_ratio: float,
) -> dict[str, int | float]:
    """返回经过集中校验且可进入摘要的注意力配准门禁记录."""

    resolved_gate = validate_attention_alignment_gate(
        anchor_count,
        residual_threshold,
        minimum_inlier_ratio,
    )
    return {
        "attention_anchor_count": resolved_gate[0],
        "attention_residual_threshold": resolved_gate[1],
        "attention_minimum_inlier_ratio": resolved_gate[2],
    }


def _grid_coordinates(token_indices: tuple[int, ...], device: Any) -> Any:
    """把原始图像 token 索引转换成归一化二维网格坐标。"""

    return public_token_grid_coordinates(token_indices, device)


def _apply_affine(points: Any, transform: Any) -> Any:
    """将一个或一批 2x3 仿射矩阵应用到二维点。"""

    torch = _torch()
    if transform.ndim == 2:
        ones = torch.ones(points.shape[0], 1, device=points.device, dtype=points.dtype)
        return torch.cat((points, ones), dim=1) @ transform.transpose(0, 1)
    if transform.ndim != 3:
        raise ValueError("transform 必须具有 [2, 3] 或 [candidate, 2, 3] 形状")
    expanded = points.unsqueeze(0).expand(transform.shape[0], -1, -1)
    ones = torch.ones(
        transform.shape[0],
        points.shape[0],
        1,
        device=points.device,
        dtype=points.dtype,
    )
    homogeneous = torch.cat((expanded, ones), dim=-1)
    return torch.bmm(homogeneous, transform.transpose(1, 2))


def _invert_affine(transforms: Any) -> Any:
    """求一批可逆 2x3 仿射矩阵的逆变换."""

    torch = _torch()
    resolved = transforms.unsqueeze(0) if transforms.ndim == 2 else transforms
    homogeneous = torch.eye(
        3,
        device=resolved.device,
        dtype=resolved.dtype,
    ).unsqueeze(0).repeat(resolved.shape[0], 1, 1)
    homogeneous[:, :2, :] = resolved
    inverted = torch.linalg.inv(homogeneous)[:, :2, :]
    return inverted[0] if transforms.ndim == 2 else inverted


def _anchor_indices(token_count: int, anchor_count: int) -> tuple[int, ...]:
    """在完整 token 网格上稳定选择锚点。"""

    if anchor_count == token_count:
        return tuple(range(token_count))
    return tuple(
        round(index * (token_count - 1) / (anchor_count - 1))
        for index in range(anchor_count)
    )


def _affine_matrix(
    rotation_degrees: float,
    scale: float,
    translation_x: float,
    translation_y: float,
    *,
    device: Any,
) -> Any:
    """构造从规范坐标到观测坐标的二维相似仿射矩阵。"""

    torch = _torch()
    angle = math.radians(rotation_degrees)
    cosine = math.cos(angle) * scale
    sine = math.sin(angle) * scale
    return torch.tensor(
        ((cosine, -sine, translation_x), (sine, cosine, translation_y)),
        device=device,
        dtype=torch.float32,
    )


def _bounded_similarity_candidate_mask(transforms: Any) -> Any:
    """校验候选相对某个方形二面体基元的严格相似变换边界。"""

    torch = _torch()
    resolved = transforms.unsqueeze(0) if transforms.ndim == 2 else transforms
    if resolved.ndim != 3 or resolved.shape[-2:] != (2, 3):
        raise ValueError("候选仿射必须具有 [candidate, 2, 3] 形状")
    dihedral = torch.tensor(
        _DIHEDRAL_LINEAR_PARTS,
        device=resolved.device,
        dtype=resolved.dtype,
    )
    residual = torch.matmul(
        resolved[:, None, :, :2],
        dihedral.transpose(-1, -2)[None, :, :, :],
    )
    determinant = torch.linalg.det(residual)
    scale = determinant.clamp_min(0.0).sqrt()
    rotation = torch.rad2deg(torch.atan2(residual[..., 1, 0], residual[..., 0, 0]))
    first_column_norm = residual[..., :, 0].norm(dim=-1)
    second_column_norm = residual[..., :, 1].norm(dim=-1)
    column_inner_product = (
        residual[..., :, 0] * residual[..., :, 1]
    ).sum(dim=-1)
    similarity_ready = (
        determinant > 0.0
    ) & (
        (first_column_norm - second_column_norm).abs() <= 1e-5
    ) & (
        column_inner_product.abs() <= 1e-5
    )
    residual_ready = (
        similarity_ready
        & (rotation.abs() <= _SIMILARITY_ROTATION_BOUND_DEGREES + 1e-6)
        & (scale >= math.exp(-_SIMILARITY_LOG_SCALE_ANCHOR_BOUND) - 1e-6)
        & (scale <= math.exp(_SIMILARITY_LOG_SCALE_ANCHOR_BOUND) + 1e-6)
    )
    translation = resolved[:, :, 2]
    translation_ready = (
        translation.abs() <= _SIMILARITY_TRANSLATION_ANCHOR_BOUND + 1e-6
    ).all(dim=-1)
    return residual_ready.any(dim=1) & translation_ready


def _coarse_affine_candidates(device: Any) -> Any:
    """从通用相似变换定义域构造冻结的低成本搜索集合。

    旋转锚点由对称角度定义域等分得到, 尺度锚点在 log-scale 上对称分布,
    位移锚点由归一化图像坐标定义域给出。所有数值只由公开的连续变换容量
    确定, 不读取攻击注册表、攻击角度或裁剪比例。方形二面体变换作为离散
    对称性另行加入。
    """

    torch = _torch()
    rotation_step = (
        2.0 * _SIMILARITY_ROTATION_BOUND_DEGREES
        / float(_SIMILARITY_ROTATION_INTERVAL_COUNT)
    )
    rotations = tuple(
        -_SIMILARITY_ROTATION_BOUND_DEGREES + rotation_step * index
        for index in range(_SIMILARITY_ROTATION_INTERVAL_COUNT + 1)
    )
    scales = tuple(
        math.exp(log_scale)
        for log_scale in (
            -_SIMILARITY_LOG_SCALE_ANCHOR_BOUND,
            0.0,
            _SIMILARITY_LOG_SCALE_ANCHOR_BOUND,
        )
    )
    translations = (
        -_SIMILARITY_TRANSLATION_ANCHOR_BOUND,
        0.0,
        _SIMILARITY_TRANSLATION_ANCHOR_BOUND,
    )
    candidates = [
        _affine_matrix(rotation, scale, tx, ty, device=device)
        for rotation in rotations
        for scale in scales
        for tx in translations
        for ty in translations
    ]
    candidates.extend(
        torch.tensor(
            ((left[0], left[1], 0.0), (right[0], right[1], 0.0)),
            device=device,
            dtype=torch.float32,
        )
        for left, right in _DIHEDRAL_LINEAR_PARTS
    )
    stacked = torch.stack(candidates)
    return stacked[_bounded_similarity_candidate_mask(stacked)]


def _local_affine_candidates(
    best_transform: Any,
    *,
    rotation_delta_degrees: float,
    log_scale_delta: float,
    translation_delta: float,
) -> Any:
    """在 rotation、log-scale 和 translation 上执行对称局部细化。"""

    torch = _torch()
    device = best_transform.device
    candidates = []
    for rotation in (-rotation_delta_degrees, 0.0, rotation_delta_degrees):
        for scale in (
            math.exp(-log_scale_delta),
            1.0,
            math.exp(log_scale_delta),
        ):
            delta = _affine_matrix(rotation, scale, 0.0, 0.0, device=device)
            linear = delta[:, :2] @ best_transform[:, :2]
            base_translation = best_transform[:, 2]
            for tx in (-translation_delta, 0.0, translation_delta):
                for ty in (-translation_delta, 0.0, translation_delta):
                    transform = torch.cat(
                        (
                            linear,
                            (base_translation + torch.tensor((tx, ty), device=device)).reshape(2, 1),
                        ),
                        dim=1,
                    )
                    candidates.append(transform)
    stacked = torch.stack(candidates)
    bounded = stacked[_bounded_similarity_candidate_mask(stacked)]
    if bounded.shape[0] == 0:
        raise RuntimeError("局部相似变换细化没有产生定义域内候选")
    return bounded


def _local_affine_refinement_schedule() -> tuple[tuple[float, float, float], ...]:
    """由粗网格单元宽度生成与攻击参数无关的三分层级细化日程。

    第一轮在每个粗网格单元的半宽处搜索, 后续轮次按固定三分比例缩小。
    因此日程只依赖连续变换定义域和网格分辨率, 可以复用于不同攻击协议。
    """

    coarse_rotation_step = (
        2.0 * _SIMILARITY_ROTATION_BOUND_DEGREES
        / float(_SIMILARITY_ROTATION_INTERVAL_COUNT)
    )
    initial_deltas = (
        coarse_rotation_step * 0.5,
        _SIMILARITY_LOG_SCALE_ANCHOR_BOUND * 0.5,
        _SIMILARITY_TRANSLATION_ANCHOR_BOUND * 0.5,
    )
    return tuple(
        tuple(
            delta / (_SIMILARITY_LOCAL_REFINEMENT_RATIO**round_index)
            for delta in initial_deltas
        )
        for round_index in range(_SIMILARITY_LOCAL_REFINEMENT_ROUNDS)
    )


def _sampling_weights(target_coordinates: Any, observed_coordinates: Any) -> tuple[Any, Any, Any]:
    """为一批仿射候选构造规范坐标到观测坐标的插值矩阵。

    返回的 ``W`` 满足 ``A_canonical = W A_observed W^T``。这一步同时处理
    attention 的查询轴和键轴, 是 ``P A P^T`` 等变注册的核心。
    """

    torch = _torch()
    candidate_count, token_count, _ = target_coordinates.shape
    x_axis = torch.unique(observed_coordinates[:, 0], sorted=True)
    y_axis = torch.unique(observed_coordinates[:, 1], sorted=True)
    if x_axis.numel() * y_axis.numel() != token_count:
        raise ValueError("token_indices 必须形成完整的规则二维抽样网格")
    x_positions = torch.searchsorted(
        x_axis,
        observed_coordinates[:, 0].contiguous(),
    ).long()
    y_positions = torch.searchsorted(
        y_axis,
        observed_coordinates[:, 1].contiguous(),
    ).long()
    lookup = torch.full(
        (y_axis.numel(), x_axis.numel()),
        -1,
        device=observed_coordinates.device,
        dtype=torch.long,
    )
    lookup[y_positions, x_positions] = torch.arange(
        token_count,
        device=observed_coordinates.device,
    )
    if bool((lookup < 0).any()):
        raise ValueError("token_indices 的二维抽样网格存在缺失坐标")

    target_x = target_coordinates[..., 0]
    target_y = target_coordinates[..., 1]
    valid = (
        (target_x >= x_axis[0] - 1e-6)
        & (target_x <= x_axis[-1] + 1e-6)
        & (target_y >= y_axis[0] - 1e-6)
        & (target_y <= y_axis[-1] + 1e-6)
    )

    def axis_neighbors(values: Any, axis: Any) -> tuple[Any, Any, Any]:
        """在真实抽样轴上计算双线性插值邻点与权重。"""

        upper = torch.searchsorted(axis, values.contiguous(), right=True)
        upper = upper.clamp(1, axis.numel() - 1)
        lower = upper - 1
        lower_value = axis.index_select(0, lower.reshape(-1)).reshape(lower.shape)
        upper_value = axis.index_select(0, upper.reshape(-1)).reshape(upper.shape)
        fraction = (values - lower_value) / (upper_value - lower_value).clamp_min(1e-12)
        return lower, upper, fraction.clamp(0.0, 1.0)

    x_lower, x_upper, x_fraction = axis_neighbors(target_x, x_axis)
    y_lower, y_upper, y_fraction = axis_neighbors(target_y, y_axis)
    neighbor_indices = torch.stack(
        (
            lookup[y_lower, x_lower],
            lookup[y_lower, x_upper],
            lookup[y_upper, x_lower],
            lookup[y_upper, x_upper],
        ),
        dim=-1,
    )
    neighbor_weights = torch.stack(
        (
            (1.0 - x_fraction) * (1.0 - y_fraction),
            x_fraction * (1.0 - y_fraction),
            (1.0 - x_fraction) * y_fraction,
            x_fraction * y_fraction,
        ),
        dim=-1,
    )
    neighbor_weights = neighbor_weights * valid.unsqueeze(-1)
    weights = torch.zeros(
        candidate_count,
        token_count,
        token_count,
        device=target_coordinates.device,
        dtype=torch.float32,
    )
    weights.scatter_add_(2, neighbor_indices, neighbor_weights)
    neighbor_coordinates = observed_coordinates.index_select(
        0,
        neighbor_indices.reshape(-1),
    ).reshape(candidate_count, token_count, 4, 2)
    nearest_residuals = (
        neighbor_coordinates - target_coordinates.unsqueeze(-2)
    ).norm(dim=-1).min(dim=-1).values
    return weights, valid, nearest_residuals


def _relation_scores(
    aligned: Any,
    relation_projection: Any,
    valid: Any,
    pair_weights: Any,
    component_weights: tuple[float, ...],
) -> tuple[Any, Any]:
    """以统一四分量算子计算一批关系图的分量分数与协议总分."""

    torch = _torch()
    component_scores = attention_relation_component_scores(
        aligned,
        relation_projection,
        pair_weights,
        valid,
        component_weights,
    )
    scores = combine_attention_relation_component_scores(
        component_scores,
        component_weights,
    )
    finite_scores = torch.where(
        torch.isfinite(scores),
        scores,
        torch.full_like(scores, -1.0),
    )
    finite_components = torch.where(
        torch.isfinite(component_scores),
        component_scores,
        torch.full_like(component_scores, -1.0),
    )
    return finite_scores, finite_components


def _unique_sampling_ratios(weights: Any, valid: Any) -> Any:
    """计算每个候选覆盖的唯一观测位置比例."""

    torch = _torch()
    observed_indices = weights.argmax(dim=-1)
    observed_counts = torch.zeros(
        weights.shape[0],
        weights.shape[-1],
        device=weights.device,
        dtype=torch.float32,
    )
    observed_counts.scatter_add_(
        1,
        observed_indices,
        valid.to(dtype=torch.float32),
    )
    return (observed_counts > 0).float().sum(dim=-1) / valid.float().sum(
        dim=-1
    ).clamp_min(1.0)


@dataclass(frozen=True)
class _CandidateEvaluation:
    """保存一批仿射候选的双边关系图评分。"""

    transforms: Any
    scores: Any
    relation_component_scores: Any
    observation_relation_scores: Any
    observation_relation_component_scores: Any
    bidirectional_relation_scores: Any
    objectives: Any
    coverage_penalties: Any
    weights: Any
    valid: Any
    nearest_residuals: Any
    canonical_coverage_ratios: Any
    observation_coverage_ratios: Any
    canonical_unique_ratios: Any
    observation_unique_ratios: Any


def _evaluate_candidates(
    relation_values: Any,
    relation_projection: Any,
    coordinates: Any,
    transforms: Any,
    stable_pair_weights: StableAttentionPairWeights,
    component_weights: tuple[float, ...],
) -> _CandidateEvaluation:
    """计算规范拉回和观测前推一致性共同约束的注册目标."""

    torch = _torch()
    target_coordinates = _apply_affine(coordinates, transforms)
    weights, valid, nearest_residuals = _sampling_weights(target_coordinates, coordinates)
    aligned = torch.einsum(
        "bik,klc,bjl->bijc",
        weights,
        relation_values.float(),
        weights,
    )
    source_token_weights = torch.tensor(
        stable_pair_weights.token_weights,
        device=relation_values.device,
        dtype=torch.float32,
    )
    canonical_token_weights = torch.bmm(
        weights,
        source_token_weights.reshape(1, -1, 1).expand(
            transforms.shape[0],
            -1,
            -1,
        ),
    ).squeeze(-1)
    canonical_token_weights = canonical_token_weights * valid.to(
        dtype=canonical_token_weights.dtype
    )
    off_diagonal = 1.0 - torch.eye(
        relation_values.shape[-2],
        device=relation_values.device,
        dtype=torch.float32,
    )
    canonical_pair_weights = (
        canonical_token_weights.unsqueeze(-1)
        * canonical_token_weights.unsqueeze(-2)
        * off_diagonal.unsqueeze(0)
    )
    scores, relation_component_scores = _relation_scores(
        aligned,
        relation_projection,
        valid,
        canonical_pair_weights,
        component_weights,
    )
    inverse_transforms = _invert_affine(transforms)
    canonical_coordinates_at_observation = _apply_affine(
        coordinates,
        inverse_transforms,
    )
    observation_weights, observation_valid, _ = _sampling_weights(
        canonical_coordinates_at_observation,
        coordinates,
    )
    expected_observation = torch.einsum(
        "bik,klc,bjl->bijc",
        observation_weights,
        relation_projection.float(),
        observation_weights,
    )
    expanded_relation_values = relation_values.float().unsqueeze(0).expand(
        transforms.shape[0],
        -1,
        -1,
        -1,
    )
    (
        observation_relation_scores,
        observation_relation_component_scores,
    ) = _relation_scores(
        expanded_relation_values,
        expected_observation,
        observation_valid,
        stable_pair_weights.pair_tensor(relation_values[..., 0]).float(),
        component_weights,
    )
    canonical_coverage_ratios = valid.float().mean(dim=-1)
    observation_coverage_ratios = observation_valid.float().mean(dim=-1)
    canonical_unique_ratios = _unique_sampling_ratios(weights, valid)
    observation_unique_ratios = _unique_sampling_ratios(
        observation_weights,
        observation_valid,
    )
    bidirectional_relation_scores = (
        _CANONICAL_RELATION_WEIGHT * scores
        + _OBSERVATION_RELATION_WEIGHT * observation_relation_scores
    )
    coverage_penalties = _COVERAGE_PENALTY_WEIGHT * (
        (1.0 - canonical_coverage_ratios)
        + (1.0 - observation_coverage_ratios)
        + (1.0 - canonical_unique_ratios)
        + (1.0 - observation_unique_ratios)
    )
    objectives = bidirectional_relation_scores - coverage_penalties
    return _CandidateEvaluation(
        transforms=transforms,
        scores=scores,
        relation_component_scores=relation_component_scores,
        observation_relation_scores=observation_relation_scores,
        observation_relation_component_scores=(
            observation_relation_component_scores
        ),
        bidirectional_relation_scores=bidirectional_relation_scores,
        objectives=objectives,
        coverage_penalties=coverage_penalties,
        weights=weights,
        valid=valid,
        nearest_residuals=nearest_residuals,
        canonical_coverage_ratios=canonical_coverage_ratios,
        observation_coverage_ratios=observation_coverage_ratios,
        canonical_unique_ratios=canonical_unique_ratios,
        observation_unique_ratios=observation_unique_ratios,
    )


def _combine_evaluations(evaluations: Iterable[_CandidateEvaluation]) -> _CandidateEvaluation:
    """连接粗搜索与局部细化结果。"""

    torch = _torch()
    resolved = tuple(evaluations)
    return _CandidateEvaluation(
        transforms=torch.cat(tuple(item.transforms for item in resolved)),
        scores=torch.cat(tuple(item.scores for item in resolved)),
        relation_component_scores=torch.cat(
            tuple(item.relation_component_scores for item in resolved)
        ),
        observation_relation_scores=torch.cat(
            tuple(item.observation_relation_scores for item in resolved)
        ),
        observation_relation_component_scores=torch.cat(
            tuple(
                item.observation_relation_component_scores
                for item in resolved
            )
        ),
        bidirectional_relation_scores=torch.cat(
            tuple(item.bidirectional_relation_scores for item in resolved)
        ),
        objectives=torch.cat(tuple(item.objectives for item in resolved)),
        coverage_penalties=torch.cat(
            tuple(item.coverage_penalties for item in resolved)
        ),
        weights=torch.cat(tuple(item.weights for item in resolved)),
        valid=torch.cat(tuple(item.valid for item in resolved)),
        nearest_residuals=torch.cat(tuple(item.nearest_residuals for item in resolved)),
        canonical_coverage_ratios=torch.cat(
            tuple(item.canonical_coverage_ratios for item in resolved)
        ),
        observation_coverage_ratios=torch.cat(
            tuple(item.observation_coverage_ratios for item in resolved)
        ),
        canonical_unique_ratios=torch.cat(
            tuple(item.canonical_unique_ratios for item in resolved)
        ),
        observation_unique_ratios=torch.cat(
            tuple(item.observation_unique_ratios for item in resolved)
        ),
    )


@dataclass(frozen=True)
class AttentionAlignmentResult:
    """保存双边关系图注册和仿射恢复结果。"""

    layer_name: str
    token_indices: tuple[int, ...]
    affine_transform: tuple[tuple[float, float, float], tuple[float, float, float]]
    expected_anchor_indices: tuple[int, ...]
    observed_anchor_indices: tuple[int, ...]
    inlier_mask: tuple[bool, ...]
    attention_anchor_count: int
    attention_residual_threshold: float
    attention_minimum_inlier_ratio: float
    inlier_ratio: float
    mean_inlier_residual: float
    relation_sync_score: float
    relation_component_scores: dict[str, float]
    observation_relation_score: float
    observation_relation_component_scores: dict[str, float]
    identity_observation_relation_score: float
    identity_observation_relation_component_scores: dict[str, float]
    registration_alignment_gain: float
    bidirectional_relation_score: float
    registration_objective_score: float
    identity_registration_objective_score: float
    registration_objective_margin: float
    registration_coverage_penalty: float
    canonical_coverage_ratio: float
    observation_coverage_ratio: float
    canonical_unique_ratio: float
    observation_unique_ratio: float
    canonical_token_weights: tuple[float, ...]
    stable_pair_weight_identity_digest: str
    observed_pair_weight_realization_digest: str
    canonical_pair_weight_realization_digest: str
    attention_relation_source: str
    attention_relation_active_component_names: tuple[str, ...]
    attention_relation_component_weights: tuple[float, ...]
    attention_relation_component_protocol_digest: str
    attention_relation_component_identity_digest: str
    attention_relation_keyed_projection_digest: str
    attention_relation_qk_operator_metadata_digest: str
    attention_relation_qk_operator_metadata_ready: bool
    attention_relation_qk_atomic_content_digest: str
    attention_relation_qk_atomic_content_ready: bool
    registration_confidence: float
    geometry_reliable: bool
    alignment_digest: str
    metadata: dict[str, Any]


def _alignment_float(
    record: Mapping[str, Any],
    field_name: str,
    *,
    allow_positive_infinity: bool = False,
) -> float:
    """读取 alignment 决策字段并拒绝隐式数值类型。"""

    value = record.get(field_name)
    if type(value) is not float or (
        not math.isfinite(value)
        and not (allow_positive_infinity and value == math.inf)
    ):
        raise ValueError(f"alignment 字段 {field_name} 必须为有限精确 float")
    return value


def recompute_attention_alignment_digest_payload(
    record: Mapping[str, Any],
) -> dict[str, Any]:
    """从一次 alignment 记录重建影响配准和 rescue 的规范摘要正文。

    该函数只覆盖方法数值、结构门禁和检测决策实际使用的字段, 不承担通用
    metadata 防篡改职责。检测层附加的注册可靠性别名不进入核心摘要, 由
    ``validate_attention_alignment_record`` 单独核对。
    """

    if not isinstance(record, Mapping):
        raise TypeError("alignment 记录必须为 mapping")
    metadata = record.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("alignment 记录缺少 metadata")

    layer_name = record.get("layer_name")
    token_indices = record.get("token_indices")
    if type(layer_name) is not str or not layer_name:
        raise ValueError("alignment layer_name 必须为非空精确 str")
    if not isinstance(token_indices, (list, tuple)) or any(
        type(value) is not int or value < 0 for value in token_indices
    ):
        raise ValueError("alignment token_indices 必须为非负整数序列")

    def integer_sequence(field_name: str) -> list[int]:
        """读取固定顺序的整数索引序列。"""

        value = record.get(field_name)
        if not isinstance(value, (list, tuple)) or any(
            type(item) is not int for item in value
        ):
            raise ValueError(f"alignment 字段 {field_name} 必须为整数序列")
        return list(value)

    def float_sequence(
        field_name: str,
        *,
        round_values: bool = True,
    ) -> list[float]:
        """读取并规范化有限浮点序列。"""

        value = record.get(field_name)
        if not isinstance(value, (list, tuple)) or any(
            type(item) is not float or not math.isfinite(item)
            for item in value
        ):
            raise ValueError(f"alignment 字段 {field_name} 必须为有限 float 序列")
        return [
            round(item, 12) if round_values else item
            for item in value
        ]

    def component_scores(field_name: str) -> dict[str, float]:
        """读取四分量关系得分并固定数值精度。"""

        value = record.get(field_name)
        if not isinstance(value, Mapping) or any(
            type(item) is not float or not math.isfinite(item)
            for item in value.values()
        ):
            raise ValueError(f"alignment 字段 {field_name} 必须为有限 float mapping")
        return {
            str(name): round(item, 12)
            for name, item in value.items()
        }

    inlier_mask = record.get("inlier_mask")
    affine_transform = record.get("affine_transform")
    active_component_names = record.get(
        "attention_relation_active_component_names"
    )
    if not isinstance(inlier_mask, (list, tuple)) or any(
        type(value) is not bool for value in inlier_mask
    ):
        raise ValueError("alignment inlier_mask 必须为 bool 序列")
    if (
        not isinstance(affine_transform, (list, tuple))
        or len(affine_transform) != 2
        or any(
            not isinstance(row, (list, tuple)) or len(row) != 3
            for row in affine_transform
        )
        or any(
            type(value) is not float or not math.isfinite(value)
            for row in affine_transform
            for value in row
        )
    ):
        raise ValueError("alignment affine_transform 必须为有限 float 2x3 矩阵")
    if not isinstance(active_component_names, (list, tuple)) or any(
        type(value) is not str for value in active_component_names
    ):
        raise ValueError("alignment 活动分量名称必须为 str 序列")

    gate = attention_alignment_gate_record(
        record.get("attention_anchor_count"),
        record.get("attention_residual_threshold"),
        record.get("attention_minimum_inlier_ratio"),
    )
    return {
        "layer_name": layer_name,
        "token_indices": list(token_indices),
        **gate,
        "expected_anchor_indices": integer_sequence(
            "expected_anchor_indices"
        ),
        "observed_anchor_indices": integer_sequence(
            "observed_anchor_indices"
        ),
        "inlier_mask": list(inlier_mask),
        "affine_transform": [list(row) for row in affine_transform],
        "inlier_ratio": round(_alignment_float(record, "inlier_ratio"), 12),
        "mean_inlier_residual": round(
            _alignment_float(
                record,
                "mean_inlier_residual",
                allow_positive_infinity=True,
            ),
            12,
        ),
        "relation_sync_score": round(
            _alignment_float(record, "relation_sync_score"),
            12,
        ),
        "relation_component_scores": component_scores(
            "relation_component_scores"
        ),
        "observation_relation_score": round(
            _alignment_float(record, "observation_relation_score"),
            12,
        ),
        "observation_relation_component_scores": component_scores(
            "observation_relation_component_scores"
        ),
        "identity_observation_relation_score": round(
            _alignment_float(
                record,
                "identity_observation_relation_score",
            ),
            12,
        ),
        "identity_observation_relation_component_scores": component_scores(
            "identity_observation_relation_component_scores"
        ),
        "registration_alignment_gain": round(
            _alignment_float(record, "registration_alignment_gain"),
            12,
        ),
        "bidirectional_relation_score": round(
            _alignment_float(record, "bidirectional_relation_score"),
            12,
        ),
        "registration_objective_score": round(
            _alignment_float(record, "registration_objective_score"),
            12,
        ),
        "identity_registration_objective_score": round(
            _alignment_float(
                record,
                "identity_registration_objective_score",
            ),
            12,
        ),
        "registration_objective_margin": round(
            _alignment_float(record, "registration_objective_margin"),
            12,
        ),
        "registration_coverage_penalty": round(
            _alignment_float(record, "registration_coverage_penalty"),
            12,
        ),
        "canonical_coverage_ratio": round(
            _alignment_float(record, "canonical_coverage_ratio"),
            12,
        ),
        "observation_coverage_ratio": round(
            _alignment_float(record, "observation_coverage_ratio"),
            12,
        ),
        "canonical_unique_ratio": round(
            _alignment_float(record, "canonical_unique_ratio"),
            12,
        ),
        "observation_unique_ratio": round(
            _alignment_float(record, "observation_unique_ratio"),
            12,
        ),
        "canonical_token_weights": float_sequence(
            "canonical_token_weights"
        ),
        "stable_pair_weight_identity_digest": record.get(
            "stable_pair_weight_identity_digest"
        ),
        "observed_pair_weight_realization_digest": record.get(
            "observed_pair_weight_realization_digest"
        ),
        "canonical_pair_weight_realization_digest": record.get(
            "canonical_pair_weight_realization_digest"
        ),
        "attention_relation_source": record.get(
            "attention_relation_source"
        ),
        "attention_relation_active_component_names": list(
            active_component_names
        ),
        "attention_relation_component_weights": float_sequence(
            "attention_relation_component_weights",
            round_values=False,
        ),
        "attention_relation_component_protocol_digest": record.get(
            "attention_relation_component_protocol_digest"
        ),
        "attention_relation_component_identity_digest": record.get(
            "attention_relation_component_identity_digest"
        ),
        "attention_relation_keyed_projection_digest": record.get(
            "attention_relation_keyed_projection_digest"
        ),
        "attention_relation_qk_operator_metadata_digest": record.get(
            "attention_relation_qk_operator_metadata_digest"
        ),
        "attention_relation_qk_operator_metadata_ready": record.get(
            "attention_relation_qk_operator_metadata_ready"
        ),
        "attention_relation_qk_atomic_content_digest": record.get(
            "attention_relation_qk_atomic_content_digest"
        ),
        "attention_relation_qk_atomic_content_ready": record.get(
            "attention_relation_qk_atomic_content_ready"
        ),
        "coordinate_convention": metadata.get("coordinate_convention"),
        "grid_align_corners": metadata.get("grid_align_corners"),
        "registration_confidence": round(
            _alignment_float(record, "registration_confidence"),
            12,
        ),
        "geometry_reliable": record.get("geometry_reliable"),
    }


def validate_attention_alignment_record(
    record: Mapping[str, Any],
) -> dict[str, Any]:
    """复验影响检测 rescue 的 alignment 数值、门禁和规范摘要。"""

    payload = recompute_attention_alignment_digest_payload(record)
    alignment_digest = record.get("alignment_digest")
    if (
        type(alignment_digest) is not str
        or len(alignment_digest) != 64
        or any(
            character not in "0123456789abcdef"
            for character in alignment_digest
        )
        or build_stable_digest(payload) != alignment_digest
    ):
        raise ValueError("alignment 正文与 alignment_digest 不一致")

    metadata = record["metadata"]
    gate = {
        field_name: payload[field_name]
        for field_name in (
            "attention_anchor_count",
            "attention_residual_threshold",
            "attention_minimum_inlier_ratio",
        )
    }
    if (
        metadata.get("attention_alignment_gate") != gate
        or any(metadata.get(name) != value for name, value in gate.items())
        or payload["coordinate_convention"]
        != ATTENTION_COORDINATE_CONVENTION
        or payload["grid_align_corners"] is not ATTENTION_GRID_ALIGN_CORNERS
    ):
        raise ValueError("alignment 结构门禁或坐标约定不一致")

    inlier_ratio = _alignment_float(record, "inlier_ratio")
    mean_residual = _alignment_float(
        record,
        "mean_inlier_residual",
        allow_positive_infinity=True,
    )
    bidirectional_score = _alignment_float(
        record,
        "bidirectional_relation_score",
    )
    canonical_coverage = _alignment_float(
        record,
        "canonical_coverage_ratio",
    )
    observation_coverage = _alignment_float(
        record,
        "observation_coverage_ratio",
    )
    if (
        not 0.0 <= inlier_ratio <= 1.0
        or mean_residual < 0.0
        or not 0.0 <= canonical_coverage <= 1.0
        or not 0.0 <= observation_coverage <= 1.0
    ):
        raise ValueError("alignment 内点率、残差或覆盖率超出定义域")
    expected_confidence = (
        max(0.0, bidirectional_score)
        * inlier_ratio
        * math.exp(-mean_residual)
        * min(canonical_coverage, observation_coverage)
    )
    registration_confidence = _alignment_float(
        record,
        "registration_confidence",
    )
    if (
        not 0.0 <= registration_confidence <= 1.0
        or not math.isclose(
            registration_confidence,
            expected_confidence,
            rel_tol=0.0,
            abs_tol=2e-12,
        )
    ):
        raise ValueError("registration_confidence 与配准公式不一致")

    registration_objective_score = _alignment_float(
        record,
        "registration_objective_score",
    )
    identity_registration_objective_score = _alignment_float(
        record,
        "identity_registration_objective_score",
    )
    registration_objective_margin = _alignment_float(
        record,
        "registration_objective_margin",
    )
    expected_objective_margin = (
        registration_objective_score
        - identity_registration_objective_score
    )
    if (
        registration_objective_margin < 0.0
        or not math.isclose(
            registration_objective_margin,
            expected_objective_margin,
            rel_tol=0.0,
            abs_tol=2e-12,
        )
    ):
        raise ValueError("registration_objective_margin 与 identity 目标差公式不一致")
    affine_transform = payload["affine_transform"]
    identity_transform = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
    affine_is_identity = all(
        math.isclose(
            float(affine_transform[row_index][column_index]),
            identity_transform[row_index][column_index],
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        for row_index in range(2)
        for column_index in range(3)
    )
    if affine_is_identity and (
        registration_objective_margin != 0.0
        or not math.isclose(
            registration_objective_score,
            identity_registration_objective_score,
            rel_tol=0.0,
            abs_tol=2e-12,
        )
    ):
        raise ValueError("identity 配准不得声明正注册目标增益")

    expected_reliable = bool(
        _alignment_float(record, "relation_sync_score") >= -1.0
        and _alignment_float(record, "observation_relation_score") > 0.0
        and bidirectional_score > 0.0
        and inlier_ratio >= gate["attention_minimum_inlier_ratio"]
        and mean_residual <= gate["attention_residual_threshold"]
        and canonical_coverage >= _MINIMUM_REGISTRATION_COVERAGE
        and observation_coverage >= _MINIMUM_REGISTRATION_COVERAGE
        and registration_objective_margin > 0.0
        and record.get("attention_relation_source")
        == DIRECT_QK_RELATION_SOURCE
        and record.get("attention_relation_qk_operator_metadata_ready")
        is True
        and record.get("attention_relation_qk_atomic_content_ready") is True
    )
    geometry_reliable = record.get("geometry_reliable")
    wrapper_reliable = record.get(
        "registration_geometry_reliable",
        geometry_reliable,
    )
    if (
        type(geometry_reliable) is not bool
        or geometry_reliable is not expected_reliable
        or type(wrapper_reliable) is not bool
        or wrapper_reliable is not geometry_reliable
    ):
        raise ValueError("alignment 注册可靠性与核心门禁不一致")
    return payload


def recover_attention_affine_alignment(
    attention: Any,
    key_material: str,
    layer_name: str,
    token_indices: tuple[int, ...],
    stable_pair_weights: StableAttentionPairWeights,
    prg_version: str,
    anchor_count: int,
    residual_threshold: float,
    minimum_inlier_ratio: float,
    component_weights: tuple[float, ...] = ATTENTION_RELATION_COMPONENT_WEIGHTS,
) -> AttentionAlignmentResult:
    """通过双边重采样密钥关系图恢复仿射参考系。

    对每个公开候选仿射变换构造规范拉回矩阵 ``W`` 和观测前推矩阵 ``V``.
    注册目标逐通道比较 ``W R_observed,c W^T`` 与四通道密钥投影, 以及真实
    观测关系图与 ``V (pi_c S_key) V^T``。两个方向都对查询轴和键轴应用同一
    空间变换, 每个通道逐行归一化后按冻结分量权重组合。
    """

    torch = _torch()
    resolved_component_weights = validate_attention_relation_component_weights(
        component_weights
    )
    probability = attention_probability(attention)
    matrix = probability.mean(dim=0) if probability.ndim == 3 else probability
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("attention 必须是方形矩阵或带 batch 维的方形矩阵")
    token_count = int(matrix.shape[0])
    if len(token_indices) != token_count:
        raise ValueError("token_indices 数量必须与 attention 宽度一致")
    if len(stable_pair_weights.token_weights) != token_count:
        raise ValueError("稳定 token pair 权重宽度必须与 attention 宽度一致")
    if not stable_pair_weights.pair_weight_identity_digest:
        raise ValueError("稳定 token pair 权重必须具有可复现身份")
    alignment_gate = attention_alignment_gate_record(
        anchor_count,
        residual_threshold,
        minimum_inlier_ratio,
    )
    anchor_count = int(alignment_gate["attention_anchor_count"])
    residual_threshold = float(
        alignment_gate["attention_residual_threshold"]
    )
    minimum_inlier_ratio = float(
        alignment_gate["attention_minimum_inlier_ratio"]
    )
    if anchor_count > token_count:
        raise ValueError("attention_anchor_count 不得超过实际 token 数量")

    descriptor = build_attention_relation_descriptor(
        attention,
        token_indices,
        resolved_component_weights,
    )
    relation_values = descriptor.values.mean(dim=0)
    relation_projection = keyed_attention_relation_projection(
        descriptor,
        key_material,
        layer_name,
        prg_version,
        component_weights=resolved_component_weights,
    )
    relation_graph_identity = build_attention_relation_graph_identity(
        ((layer_name, attention, token_indices),),
        key_material,
        prg_version=prg_version,
        component_weights=resolved_component_weights,
    )
    coordinates = _grid_coordinates(token_indices, matrix.device)
    coarse = _evaluate_candidates(
        relation_values,
        relation_projection.values,
        coordinates,
        _coarse_affine_candidates(matrix.device),
        stable_pair_weights,
        resolved_component_weights,
    )
    coarse_best_index = int(torch.argmax(coarse.objectives).item())
    evaluations = [coarse]
    current_best_transform = coarse.transforms[coarse_best_index]
    local_search_schedule = _local_affine_refinement_schedule()
    for rotation_delta, log_scale_delta, translation_delta in local_search_schedule:
        local = _evaluate_candidates(
            relation_values,
            relation_projection.values,
            coordinates,
            _local_affine_candidates(
                current_best_transform,
                rotation_delta_degrees=rotation_delta,
                log_scale_delta=log_scale_delta,
                translation_delta=translation_delta,
            ),
            stable_pair_weights,
            resolved_component_weights,
        )
        evaluations.append(local)
        current_best_transform = local.transforms[
            int(torch.argmax(local.objectives).item())
        ]
    evaluated = _combine_evaluations(evaluations)
    best_index = int(torch.argmax(evaluated.objectives).item())
    best_transform = evaluated.transforms[best_index]
    best_weights = evaluated.weights[best_index]
    best_valid = evaluated.valid[best_index]
    best_residuals = evaluated.nearest_residuals[best_index]
    canonical_pair_weights = transport_stable_attention_pair_weights(
        stable_pair_weights,
        best_weights,
        best_valid,
        coordinate_space="registered_canonical_qk_grid",
    )
    best_score = float(evaluated.scores[best_index].item())
    best_component_scores = {
        component_name: float(value)
        for component_name, value in zip(
            ATTENTION_RELATION_COMPONENT_NAMES,
            evaluated.relation_component_scores[best_index].detach().cpu().tolist(),
        )
    }
    best_observation_score = float(
        evaluated.observation_relation_scores[best_index].item()
    )
    best_observation_component_scores = {
        component_name: float(value)
        for component_name, value in zip(
            ATTENTION_RELATION_COMPONENT_NAMES,
            evaluated.observation_relation_component_scores[best_index]
            .detach()
            .cpu()
            .tolist(),
        )
    }
    best_bidirectional_score = float(
        evaluated.bidirectional_relation_scores[best_index].item()
    )
    best_objective = float(evaluated.objectives[best_index].item())
    best_coverage_penalty = float(
        evaluated.coverage_penalties[best_index].item()
    )
    canonical_coverage_ratio = float(
        evaluated.canonical_coverage_ratios[best_index].item()
    )
    observation_coverage_ratio = float(
        evaluated.observation_coverage_ratios[best_index].item()
    )
    canonical_unique_ratio = float(
        evaluated.canonical_unique_ratios[best_index].item()
    )
    observation_unique_ratio = float(
        evaluated.observation_unique_ratios[best_index].item()
    )
    identity_transform = torch.tensor(
        ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        device=matrix.device,
        dtype=evaluated.transforms.dtype,
    )
    identity_distance_sq = (
        evaluated.transforms - identity_transform.unsqueeze(0)
    ).square().sum(dim=(1, 2))
    identity_index = int(identity_distance_sq.argmin().item())
    if not torch.equal(
        evaluated.transforms[identity_index],
        identity_transform,
    ):
        raise RuntimeError("配准候选集合缺少精确 identity 候选")
    identity_registration_objective_score = float(
        evaluated.objectives[identity_index].item()
    )
    identity_observation_score = float(
        evaluated.observation_relation_scores[identity_index].item()
    )
    identity_observation_component_scores = {
        component_name: float(value)
        for component_name, value in zip(
            ATTENTION_RELATION_COMPONENT_NAMES,
            evaluated.observation_relation_component_scores[identity_index]
            .detach()
            .cpu()
            .tolist(),
        )
    }
    registration_alignment_gain = (
        best_observation_score - identity_observation_score
    )
    expected_indices = _anchor_indices(token_count, anchor_count)
    expected_tensor = torch.tensor(expected_indices, device=matrix.device, dtype=torch.long)
    observed_indices_tensor = best_weights.argmax(dim=-1).index_select(0, expected_tensor)
    anchor_valid = best_valid.index_select(0, expected_tensor)
    anchor_residuals = best_residuals.index_select(0, expected_tensor)
    observed_counts = torch.bincount(observed_indices_tensor, minlength=token_count)
    unique_observed = observed_counts.index_select(0, observed_indices_tensor) == 1
    inliers = anchor_valid & unique_observed & (anchor_residuals <= residual_threshold)
    valid_anchor_count = int(anchor_valid.sum().item())
    inlier_ratio = (
        float(inliers.float().sum().item()) / float(valid_anchor_count)
        if valid_anchor_count > 0
        else 0.0
    )
    mean_residual = (
        float(anchor_residuals[inliers].mean().item())
        if bool(inliers.any())
        else float("inf")
    )
    registration_objective_margin = (
        best_objective - identity_registration_objective_score
    )
    confidence = (
        max(0.0, best_bidirectional_score)
        * inlier_ratio
        * math.exp(-mean_residual)
        * min(canonical_coverage_ratio, observation_coverage_ratio)
        if math.isfinite(mean_residual)
        else 0.0
    )
    geometry_reliable = (
        math.isfinite(best_score)
        and math.isfinite(best_observation_score)
        and math.isfinite(best_bidirectional_score)
        and best_observation_score > 0.0
        and best_bidirectional_score > 0.0
        and inlier_ratio >= minimum_inlier_ratio
        and mean_residual <= residual_threshold
        and canonical_coverage_ratio >= _MINIMUM_REGISTRATION_COVERAGE
        and observation_coverage_ratio >= _MINIMUM_REGISTRATION_COVERAGE
        and registration_objective_margin > 0.0
        and descriptor.relation_source == DIRECT_QK_RELATION_SOURCE
        and relation_graph_identity.qk_operator_metadata_ready
        and relation_graph_identity.qk_atomic_content_ready
    )
    transform_tuple = tuple(
        tuple(float(value) for value in row)
        for row in best_transform.detach().cpu().tolist()
    )
    observed_indices = tuple(int(value) for value in observed_indices_tensor.detach().cpu().tolist())
    inlier_values = tuple(bool(value) for value in inliers.detach().cpu().tolist())
    payload = {
        "layer_name": layer_name,
        "token_indices": token_indices,
        **alignment_gate,
        "expected_anchor_indices": expected_indices,
        "observed_anchor_indices": observed_indices,
        "inlier_mask": list(inlier_values),
        "affine_transform": transform_tuple,
        "inlier_ratio": round(inlier_ratio, 12),
        "mean_inlier_residual": round(mean_residual, 12),
        "relation_sync_score": round(best_score, 12),
        "relation_component_scores": {
            key: round(value, 12)
            for key, value in best_component_scores.items()
        },
        "observation_relation_score": round(best_observation_score, 12),
        "observation_relation_component_scores": {
            key: round(value, 12)
            for key, value in best_observation_component_scores.items()
        },
        "identity_observation_relation_score": round(
            identity_observation_score,
            12,
        ),
        "identity_observation_relation_component_scores": {
            key: round(value, 12)
            for key, value in identity_observation_component_scores.items()
        },
        "registration_alignment_gain": round(registration_alignment_gain, 12),
        "bidirectional_relation_score": round(best_bidirectional_score, 12),
        "registration_objective_score": round(best_objective, 12),
        "identity_registration_objective_score": round(
            identity_registration_objective_score,
            12,
        ),
        "registration_objective_margin": round(registration_objective_margin, 12),
        "registration_coverage_penalty": round(best_coverage_penalty, 12),
        "canonical_coverage_ratio": round(canonical_coverage_ratio, 12),
        "observation_coverage_ratio": round(observation_coverage_ratio, 12),
        "canonical_unique_ratio": round(canonical_unique_ratio, 12),
        "observation_unique_ratio": round(observation_unique_ratio, 12),
        "canonical_token_weights": [
            round(float(value), 12)
            for value in canonical_pair_weights.token_weights
        ],
        "stable_pair_weight_identity_digest": (
            stable_pair_weights.pair_weight_identity_digest
        ),
        "observed_pair_weight_realization_digest": (
            stable_pair_weights.pair_weight_realization_digest
        ),
        "canonical_pair_weight_realization_digest": (
            canonical_pair_weights.pair_weight_realization_digest
        ),
        "attention_relation_source": descriptor.relation_source,
        "attention_relation_active_component_names": (
            relation_graph_identity.active_component_names
        ),
        "attention_relation_component_weights": resolved_component_weights,
        "attention_relation_component_protocol_digest": (
            relation_graph_identity.component_protocol_digest
        ),
        "attention_relation_component_identity_digest": (
            descriptor.component_identity_digest
        ),
        "attention_relation_keyed_projection_digest": (
            relation_projection.projection_digest
        ),
        "attention_relation_qk_operator_metadata_digest": (
            relation_graph_identity.qk_operator_metadata_digest
        ),
        "attention_relation_qk_operator_metadata_ready": (
            relation_graph_identity.qk_operator_metadata_ready
        ),
        "attention_relation_qk_atomic_content_digest": (
            relation_graph_identity.qk_atomic_content_digest
        ),
        "attention_relation_qk_atomic_content_ready": (
            relation_graph_identity.qk_atomic_content_ready
        ),
        "coordinate_convention": ATTENTION_COORDINATE_CONVENTION,
        "grid_align_corners": ATTENTION_GRID_ALIGN_CORNERS,
        "registration_confidence": round(confidence, 12),
        "geometry_reliable": geometry_reliable,
    }
    return AttentionAlignmentResult(
        layer_name=layer_name,
        token_indices=token_indices,
        affine_transform=transform_tuple,  # type: ignore[arg-type]
        expected_anchor_indices=expected_indices,
        observed_anchor_indices=observed_indices,
        inlier_mask=inlier_values,
        attention_anchor_count=anchor_count,
        attention_residual_threshold=residual_threshold,
        attention_minimum_inlier_ratio=minimum_inlier_ratio,
        inlier_ratio=inlier_ratio,
        mean_inlier_residual=mean_residual,
        relation_sync_score=best_score,
        relation_component_scores=best_component_scores,
        observation_relation_score=best_observation_score,
        observation_relation_component_scores=(
            best_observation_component_scores
        ),
        identity_observation_relation_score=identity_observation_score,
        identity_observation_relation_component_scores=(
            identity_observation_component_scores
        ),
        registration_alignment_gain=registration_alignment_gain,
        bidirectional_relation_score=best_bidirectional_score,
        registration_objective_score=best_objective,
        identity_registration_objective_score=(
            identity_registration_objective_score
        ),
        registration_objective_margin=registration_objective_margin,
        registration_coverage_penalty=best_coverage_penalty,
        canonical_coverage_ratio=canonical_coverage_ratio,
        observation_coverage_ratio=observation_coverage_ratio,
        canonical_unique_ratio=canonical_unique_ratio,
        observation_unique_ratio=observation_unique_ratio,
        canonical_token_weights=canonical_pair_weights.token_weights,
        stable_pair_weight_identity_digest=(
            stable_pair_weights.pair_weight_identity_digest
        ),
        observed_pair_weight_realization_digest=(
            stable_pair_weights.pair_weight_realization_digest
        ),
        canonical_pair_weight_realization_digest=(
            canonical_pair_weights.pair_weight_realization_digest
        ),
        attention_relation_source=descriptor.relation_source,
        attention_relation_active_component_names=(
            relation_graph_identity.active_component_names
        ),
        attention_relation_component_weights=resolved_component_weights,
        attention_relation_component_protocol_digest=(
            relation_graph_identity.component_protocol_digest
        ),
        attention_relation_component_identity_digest=(
            descriptor.component_identity_digest
        ),
        attention_relation_keyed_projection_digest=(
            relation_projection.projection_digest
        ),
        attention_relation_qk_operator_metadata_digest=(
            relation_graph_identity.qk_operator_metadata_digest
        ),
        attention_relation_qk_operator_metadata_ready=(
            relation_graph_identity.qk_operator_metadata_ready
        ),
        attention_relation_qk_atomic_content_digest=(
            relation_graph_identity.qk_atomic_content_digest
        ),
        attention_relation_qk_atomic_content_ready=(
            relation_graph_identity.qk_atomic_content_ready
        ),
        registration_confidence=confidence,
        geometry_reliable=geometry_reliable,
        alignment_digest=build_stable_digest(payload),
        metadata={
            "matcher": "double_sided_keyed_relation_graph_registration",
            "relation_transform": "canonical_relation_components_equal_w_observed_relation_components_w_transpose",
            "observation_relation_transform": "observed_key_components_equal_v_key_components_v_transpose",
            "registration_objective": "weighted_bidirectional_relation_minus_coverage_penalty",
            "transform_family": "bounded_affine_similarity_with_dihedral_support",
            "robust_estimator": "attack_independent_hierarchical_affine_relation_search",
            "local_search_schedule": [
                {
                    "rotation_delta_degrees": rotation_delta,
                    "log_scale_delta": log_scale_delta,
                    "translation_delta": translation_delta,
                }
                for rotation_delta, log_scale_delta, translation_delta in local_search_schedule
            ],
            "coarse_similarity_domain": {
                "rotation_bound_degrees": _SIMILARITY_ROTATION_BOUND_DEGREES,
                "rotation_interval_count": _SIMILARITY_ROTATION_INTERVAL_COUNT,
                "log_scale_anchor_bound": _SIMILARITY_LOG_SCALE_ANCHOR_BOUND,
                "translation_anchor_bound": _SIMILARITY_TRANSLATION_ANCHOR_BOUND,
            },
            "search_parameter_source": "public_continuous_similarity_domain",
            "similarity_domain_strictly_bounded": True,
            "similarity_domain_reference": "square_dihedral_residual",
            "similarity_residual_rotation_bound_degrees": (
                _SIMILARITY_ROTATION_BOUND_DEGREES
            ),
            "similarity_residual_scale_lower_bound": math.exp(
                -_SIMILARITY_LOG_SCALE_ANCHOR_BOUND
            ),
            "similarity_residual_scale_upper_bound": math.exp(
                _SIMILARITY_LOG_SCALE_ANCHOR_BOUND
            ),
            "similarity_translation_bound": (
                _SIMILARITY_TRANSLATION_ANCHOR_BOUND
            ),
            "local_candidate_boundary_filter": (
                "dihedral_residual_similarity_domain"
            ),
            "stable_pair_weight_identity_digest": (
                stable_pair_weights.pair_weight_identity_digest
            ),
            "stable_pair_weight_identity_ready": (
                stable_pair_weights.pair_weight_identity_digest
                == canonical_pair_weights.pair_weight_identity_digest
            ),
            "pair_weight_transport": (
                "token_weight_field_bilinear_transport_then_outer_product"
            ),
            "attention_relation_component_names": list(
                ATTENTION_RELATION_COMPONENT_NAMES
            ),
            "attention_relation_active_component_names": list(
                relation_graph_identity.active_component_names
            ),
            "attention_relation_component_weights": list(
                resolved_component_weights
            ),
            "attention_relation_component_protocol_digest": (
                relation_graph_identity.component_protocol_digest
            ),
            "attention_relation_source": descriptor.relation_source,
            "attention_relation_direct_qk_source_ready": (
                descriptor.relation_source == DIRECT_QK_RELATION_SOURCE
            ),
            "attention_relation_probability_scope": (
                "sampled_image_token_qk_relation_probability"
            ),
            "attention_relation_component_identity_digest": (
                descriptor.component_identity_digest
            ),
            "attention_relation_keyed_projection_digest": (
                relation_projection.projection_digest
            ),
            "attention_relation_qk_operator_metadata_records": list(
                relation_graph_identity.qk_operator_metadata_records
            ),
            "attention_relation_qk_operator_metadata_digest": (
                relation_graph_identity.qk_operator_metadata_digest
            ),
            "attention_relation_qk_operator_metadata_ready": (
                relation_graph_identity.qk_operator_metadata_ready
            ),
            "attention_relation_qk_atomic_content_records": list(
                relation_graph_identity.qk_atomic_content_records
            ),
            "attention_relation_qk_atomic_content_digest": (
                relation_graph_identity.qk_atomic_content_digest
            ),
            "attention_relation_qk_atomic_content_ready": (
                relation_graph_identity.qk_atomic_content_ready
            ),
            "attention_relation_soft_rank_temperature": (
                descriptor.soft_rank_temperature
            ),
            "attention_relation_soft_rank_scale": descriptor.soft_rank_scale,
            "attention_relation_relative_distance_scale": (
                descriptor.relative_distance_scale
            ),
            "coordinate_source": "original_image_token_grid",
            "coordinate_convention": ATTENTION_COORDINATE_CONVENTION,
            "grid_align_corners": ATTENTION_GRID_ALIGN_CORNERS,
            "registration_candidate_count": int(evaluated.transforms.shape[0]),
            "canonical_relation_weight": _CANONICAL_RELATION_WEIGHT,
            "observation_relation_weight": _OBSERVATION_RELATION_WEIGHT,
            "registration_coverage_penalty_weight": _COVERAGE_PENALTY_WEIGHT,
            "minimum_registration_coverage": _MINIMUM_REGISTRATION_COVERAGE,
            "attention_alignment_gate": dict(alignment_gate),
            **alignment_gate,
            "inlier_ratio_denominator": "valid_covered_anchor_count",
        },
    )
