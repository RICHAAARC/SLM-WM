"""根据密钥注意力关系图恢复图像 token 网格的仿射参考系。

几何变换会同时重排注意力矩阵的行和列. 若规范关系图为 ``A``, 观测关系图
满足 ``A_obs = P A P^T``. 注册目标同时计算 ``W A_obs W^T`` 的规范拉回
一致性和 ``V S_key V^T`` 的观测前推一致性, 并显式记录双向覆盖惩罚.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable

from main.core.digest import build_stable_digest
from main.methods.geometry.differentiable_attention import keyed_relation_signs


_CANONICAL_RELATION_WEIGHT = 0.10
_OBSERVATION_RELATION_WEIGHT = 0.90
_COVERAGE_PENALTY_WEIGHT = 0.01
_MINIMUM_REGISTRATION_COVERAGE = 0.45


def _torch() -> Any:
    """延迟导入 PyTorch。"""

    import torch

    return torch


def _grid_coordinates(token_indices: tuple[int, ...], device: Any) -> Any:
    """把原始图像 token 索引转换成归一化二维网格坐标。"""

    torch = _torch()
    if len(token_indices) < 4 or len(set(token_indices)) != len(token_indices):
        raise ValueError("几何恢复要求至少4个不重复的原始 token 索引")
    source_token_count = max(token_indices) + 1
    source_side = int(round(math.sqrt(source_token_count)))
    if source_side * source_side != source_token_count:
        raise ValueError("原始 token 索引无法还原为方形图像网格")
    coordinates = []
    for token_index in token_indices:
        row, column = divmod(token_index, source_side)
        x = -1.0 + 2.0 * column / (source_side - 1)
        y = -1.0 + 2.0 * row / (source_side - 1)
        coordinates.append((x, y))
    return torch.tensor(coordinates, device=device, dtype=torch.float32)


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

    bounded = max(3, min(anchor_count, token_count))
    if bounded == token_count:
        return tuple(range(token_count))
    return tuple(round(index * (token_count - 1) / (bounded - 1)) for index in range(bounded))


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


def _coarse_affine_candidates(device: Any) -> Any:
    """构造冻结的低成本仿射搜索集合。

    搜索集合覆盖常见旋转、裁剪重缩放和位移, 并显式加入完整方形二面体变换。
    该集合属于公开检测协议的一部分, 不读取生成侧状态。
    """

    torch = _torch()
    candidates = [
        _affine_matrix(rotation, scale, tx, ty, device=device)
        for rotation in (-30.0, -15.0, 0.0, 15.0, 30.0)
        for scale in (0.80, 1.00, 1.20)
        for tx in (-0.20, 0.0, 0.20)
        for ty in (-0.20, 0.0, 0.20)
    ]
    # 正式几何攻击采用中心裁剪和5°/7°旋转, 因此显式包含对应逆尺度。
    candidates.extend(
        _affine_matrix(rotation, scale, 0.0, 0.0, device=device)
        for rotation in (-7.0, -5.0, 0.0, 5.0, 7.0)
        for scale in (1.0, 1.0 / 0.82, 1.0 / 0.80, 1.0 / 0.78)
    )
    dihedral_linear_parts = (
        ((1.0, 0.0), (0.0, 1.0)),
        ((-1.0, 0.0), (0.0, 1.0)),
        ((1.0, 0.0), (0.0, -1.0)),
        ((-1.0, 0.0), (0.0, -1.0)),
        ((0.0, -1.0), (1.0, 0.0)),
        ((0.0, 1.0), (-1.0, 0.0)),
        ((0.0, 1.0), (1.0, 0.0)),
        ((0.0, -1.0), (-1.0, 0.0)),
    )
    candidates.extend(
        torch.tensor(
            ((left[0], left[1], 0.0), (right[0], right[1], 0.0)),
            device=device,
            dtype=torch.float32,
        )
        for left, right in dihedral_linear_parts
    )
    return torch.stack(candidates)


def _local_affine_candidates(best_transform: Any) -> Any:
    """围绕粗搜索最优解构造一次确定性局部细化集合。"""

    torch = _torch()
    device = best_transform.device
    candidates = []
    for rotation in (-5.0, 0.0, 5.0):
        for scale in (0.95, 1.0, 1.05):
            delta = _affine_matrix(rotation, scale, 0.0, 0.0, device=device)
            linear = delta[:, :2] @ best_transform[:, :2]
            base_translation = best_transform[:, 2]
            for tx in (-0.05, 0.0, 0.05):
                for ty in (-0.05, 0.0, 0.05):
                    transform = torch.cat(
                        (
                            linear,
                            (base_translation + torch.tensor((tx, ty), device=device)).reshape(2, 1),
                        ),
                        dim=1,
                    )
                    candidates.append(transform)
    return torch.stack(candidates)


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


def _relation_scores(aligned: Any, relation_signs: Any, valid: Any) -> Any:
    """计算一批双边对齐关系图与密钥图的归一化相关分数。"""

    torch = _torch()
    token_count = int(aligned.shape[-1])
    pair_mask = valid.unsqueeze(-1) & valid.unsqueeze(-2)
    off_diagonal = ~torch.eye(token_count, device=aligned.device, dtype=torch.bool).unsqueeze(0)
    pair_mask = pair_mask & off_diagonal
    float_mask = pair_mask.to(dtype=aligned.dtype)
    row_count = float_mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
    row_mean = (aligned * float_mask).sum(dim=-1, keepdim=True) / row_count
    centered = (aligned - row_mean) * float_mask
    reference = relation_signs.to(dtype=aligned.dtype)
    if reference.ndim == 2:
        reference = reference.unsqueeze(0).expand(aligned.shape[0], -1, -1)
    if reference.ndim != 3 or reference.shape != aligned.shape:
        raise ValueError("relation_signs 必须与 aligned 具有相同关系图宽度")
    reference_mean = (reference * float_mask).sum(dim=-1, keepdim=True) / row_count
    reference = (reference - reference_mean) * float_mask
    numerator = (centered * reference).sum(dim=(-1, -2))
    denominator = (
        centered.square().sum(dim=(-1, -2)).sqrt()
        * reference.square().sum(dim=(-1, -2)).sqrt()
    ).clamp_min(1e-12)
    scores = numerator / denominator
    return torch.where(torch.isfinite(scores), scores, torch.full_like(scores, -1.0))


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
    observation_relation_scores: Any
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
    matrix: Any,
    relation_signs: Any,
    coordinates: Any,
    transforms: Any,
) -> _CandidateEvaluation:
    """计算规范拉回和观测前推一致性共同约束的注册目标."""

    torch = _torch()
    target_coordinates = _apply_affine(coordinates, transforms)
    weights, valid, nearest_residuals = _sampling_weights(target_coordinates, coordinates)
    expanded_matrix = matrix.float().unsqueeze(0).expand(transforms.shape[0], -1, -1)
    aligned = torch.bmm(torch.bmm(weights, expanded_matrix), weights.transpose(1, 2))
    scores = _relation_scores(aligned, relation_signs, valid)
    inverse_transforms = _invert_affine(transforms)
    canonical_coordinates_at_observation = _apply_affine(
        coordinates,
        inverse_transforms,
    )
    observation_weights, observation_valid, _ = _sampling_weights(
        canonical_coordinates_at_observation,
        coordinates,
    )
    expanded_relation_signs = relation_signs.float().unsqueeze(0).expand(
        transforms.shape[0],
        -1,
        -1,
    )
    expected_observation = torch.bmm(
        torch.bmm(observation_weights, expanded_relation_signs),
        observation_weights.transpose(1, 2),
    )
    observation_relation_scores = _relation_scores(
        expanded_matrix,
        expected_observation,
        observation_valid,
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
        observation_relation_scores=observation_relation_scores,
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
        observation_relation_scores=torch.cat(
            tuple(item.observation_relation_scores for item in resolved)
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

    affine_transform: tuple[tuple[float, float, float], tuple[float, float, float]]
    expected_anchor_indices: tuple[int, ...]
    observed_anchor_indices: tuple[int, ...]
    inlier_mask: tuple[bool, ...]
    inlier_ratio: float
    mean_inlier_residual: float
    relation_sync_score: float
    observation_relation_score: float
    identity_observation_relation_score: float
    registration_alignment_gain: float
    bidirectional_relation_score: float
    registration_objective_score: float
    registration_objective_margin: float
    registration_coverage_penalty: float
    canonical_coverage_ratio: float
    observation_coverage_ratio: float
    canonical_unique_ratio: float
    observation_unique_ratio: float
    registration_confidence: float
    geometry_reliable: bool
    alignment_digest: str
    metadata: dict[str, Any]


def recover_attention_affine_alignment(
    attention: Any,
    key_material: str,
    layer_name: str,
    token_indices: tuple[int, ...],
    anchor_count: int = 12,
    residual_threshold: float = 0.20,
    minimum_inlier_ratio: float = 0.50,
) -> AttentionAlignmentResult:
    """通过双边重采样密钥关系图恢复仿射参考系。

    对每个公开候选仿射变换构造规范拉回矩阵 ``W`` 和观测前推矩阵 ``V``.
    注册目标联合比较 ``W A_observed W^T`` 与密钥关系图, 以及真实观测关系图
    与 ``V S_key V^T``. 两个方向都对查询轴和键轴应用同一空间变换.
    """

    torch = _torch()
    matrix = attention.mean(dim=0) if attention.ndim == 3 else attention
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("attention 必须是方形矩阵或带 batch 维的方形矩阵")
    token_count = int(matrix.shape[0])
    if len(token_indices) != token_count:
        raise ValueError("token_indices 数量必须与 attention 宽度一致")
    if not 0.0 < minimum_inlier_ratio <= 1.0:
        raise ValueError("minimum_inlier_ratio 必须位于 (0, 1]")
    if residual_threshold <= 0.0:
        raise ValueError("residual_threshold 必须为正数")

    coordinates = _grid_coordinates(token_indices, matrix.device)
    relation_signs = keyed_relation_signs(matrix, key_material, layer_name)
    coarse = _evaluate_candidates(
        matrix,
        relation_signs,
        coordinates,
        _coarse_affine_candidates(matrix.device),
    )
    coarse_best_index = int(torch.argmax(coarse.objectives).item())
    local = _evaluate_candidates(
        matrix,
        relation_signs,
        coordinates,
        _local_affine_candidates(coarse.transforms[coarse_best_index]),
    )
    evaluated = _combine_evaluations((coarse, local))
    best_index = int(torch.argmax(evaluated.objectives).item())
    best_transform = evaluated.transforms[best_index]
    transform_distances = (
        evaluated.transforms - best_transform.unsqueeze(0)
    ).square().sum(dim=(1, 2)).sqrt()
    distinct_competitors = evaluated.objectives[transform_distances > 1e-4]
    second_objective = (
        float(distinct_competitors.max().item())
        if distinct_competitors.numel()
        else float(evaluated.objectives[best_index].item())
    )
    best_weights = evaluated.weights[best_index]
    best_valid = evaluated.valid[best_index]
    best_residuals = evaluated.nearest_residuals[best_index]
    best_score = float(evaluated.scores[best_index].item())
    best_observation_score = float(
        evaluated.observation_relation_scores[best_index].item()
    )
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
    identity_index = int(
        (
            evaluated.transforms - identity_transform.unsqueeze(0)
        ).square().sum(dim=(1, 2)).argmin().item()
    )
    identity_observation_score = float(
        evaluated.observation_relation_scores[identity_index].item()
    )
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
    inlier_ratio = float(inliers.float().mean().item())
    mean_residual = (
        float(anchor_residuals[inliers].mean().item())
        if bool(inliers.any())
        else float("inf")
    )
    registration_objective_margin = max(0.0, best_objective - second_objective)
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
        "expected_anchor_indices": expected_indices,
        "observed_anchor_indices": observed_indices,
        "inlier_mask": list(inlier_values),
        "affine_transform": transform_tuple,
        "inlier_ratio": round(inlier_ratio, 12),
        "mean_inlier_residual": round(mean_residual, 12),
        "relation_sync_score": round(best_score, 12),
        "observation_relation_score": round(best_observation_score, 12),
        "identity_observation_relation_score": round(
            identity_observation_score,
            12,
        ),
        "registration_alignment_gain": round(registration_alignment_gain, 12),
        "bidirectional_relation_score": round(best_bidirectional_score, 12),
        "registration_objective_score": round(best_objective, 12),
        "registration_objective_margin": round(registration_objective_margin, 12),
        "registration_coverage_penalty": round(best_coverage_penalty, 12),
        "canonical_coverage_ratio": round(canonical_coverage_ratio, 12),
        "observation_coverage_ratio": round(observation_coverage_ratio, 12),
        "canonical_unique_ratio": round(canonical_unique_ratio, 12),
        "observation_unique_ratio": round(observation_unique_ratio, 12),
    }
    return AttentionAlignmentResult(
        affine_transform=transform_tuple,  # type: ignore[arg-type]
        expected_anchor_indices=expected_indices,
        observed_anchor_indices=observed_indices,
        inlier_mask=inlier_values,
        inlier_ratio=inlier_ratio,
        mean_inlier_residual=mean_residual,
        relation_sync_score=best_score,
        observation_relation_score=best_observation_score,
        identity_observation_relation_score=identity_observation_score,
        registration_alignment_gain=registration_alignment_gain,
        bidirectional_relation_score=best_bidirectional_score,
        registration_objective_score=best_objective,
        registration_objective_margin=registration_objective_margin,
        registration_coverage_penalty=best_coverage_penalty,
        canonical_coverage_ratio=canonical_coverage_ratio,
        observation_coverage_ratio=observation_coverage_ratio,
        canonical_unique_ratio=canonical_unique_ratio,
        observation_unique_ratio=observation_unique_ratio,
        registration_confidence=confidence,
        geometry_reliable=geometry_reliable,
        alignment_digest=build_stable_digest(payload),
        metadata={
            "matcher": "double_sided_keyed_relation_graph_registration",
            "relation_transform": "canonical_attention_equals_w_observed_attention_w_transpose",
            "observation_relation_transform": "observed_key_relation_equals_v_key_relation_v_transpose",
            "registration_objective": "weighted_bidirectional_relation_minus_coverage_penalty",
            "transform_family": "bounded_affine_similarity_with_dihedral_support",
            "robust_estimator": "deterministic_coarse_to_local_relation_search",
            "coordinate_source": "original_image_token_grid",
            "registration_candidate_count": int(evaluated.transforms.shape[0]),
            "sync_margin_duplicate_transform_tolerance": 1e-4,
            "canonical_relation_weight": _CANONICAL_RELATION_WEIGHT,
            "observation_relation_weight": _OBSERVATION_RELATION_WEIGHT,
            "registration_coverage_penalty_weight": _COVERAGE_PENALTY_WEIGHT,
            "minimum_registration_coverage": _MINIMUM_REGISTRATION_COVERAGE,
        },
    )
