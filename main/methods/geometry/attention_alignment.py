"""根据密钥注意力关系恢复图像 token 网格的仿射参考系。"""

from __future__ import annotations

from dataclasses import dataclass
import itertools
import math
from typing import Any

from main.core.digest import build_stable_digest
from main.methods.geometry.differentiable_attention import keyed_relation_signs


def _torch() -> Any:
    """延迟导入 PyTorch。"""

    import torch

    return torch


def _grid_coordinates(token_count: int, device: Any) -> Any:
    """把 token 索引转换成归一化二维网格坐标。"""

    torch = _torch()
    side = int(round(math.sqrt(token_count)))
    if side * side != token_count:
        raise ValueError("几何恢复要求抽样 token 数量构成方形网格")
    axis = torch.linspace(-1.0, 1.0, side, device=device)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    return torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=1)


def _fit_affine(source: Any, target: Any) -> Any:
    """用最小二乘拟合从 source 到 target 的二维仿射矩阵。"""

    torch = _torch()
    ones = torch.ones(source.shape[0], 1, device=source.device, dtype=source.dtype)
    design = torch.cat((source, ones), dim=1)
    solution = torch.linalg.lstsq(design, target).solution
    return solution.transpose(0, 1)


def _apply_affine(points: Any, transform: Any) -> Any:
    """将 2×3 仿射矩阵应用到二维点。"""

    torch = _torch()
    ones = torch.ones(points.shape[0], 1, device=points.device, dtype=points.dtype)
    return torch.cat((points, ones), dim=1) @ transform.transpose(0, 1)


def _anchor_indices(token_count: int, anchor_count: int) -> tuple[int, ...]:
    """在完整 token 网格上稳定选择锚点。"""

    bounded = max(3, min(anchor_count, token_count))
    if bounded == token_count:
        return tuple(range(token_count))
    return tuple(round(index * (token_count - 1) / (bounded - 1)) for index in range(bounded))


def _match_anchor_rows(attention: Any, reference_rows: Any) -> tuple[int, ...]:
    """以余弦相似度执行一对一贪心锚点匹配。"""

    import torch.nn.functional as functional

    observed = attention.mean(dim=0) if attention.ndim == 3 else attention
    observed = functional.normalize(observed - observed.mean(dim=-1, keepdim=True), dim=-1)
    reference = functional.normalize(reference_rows, dim=-1)
    similarities = reference @ observed.transpose(0, 1)
    candidates = []
    for reference_index in range(similarities.shape[0]):
        for observed_index in range(similarities.shape[1]):
            candidates.append(
                (float(similarities[reference_index, observed_index].item()), reference_index, observed_index)
            )
    matched_reference: set[int] = set()
    matched_observed: set[int] = set()
    assignments = [-1] * int(similarities.shape[0])
    for _, reference_index, observed_index in sorted(candidates, reverse=True):
        if reference_index in matched_reference or observed_index in matched_observed:
            continue
        assignments[reference_index] = observed_index
        matched_reference.add(reference_index)
        matched_observed.add(observed_index)
        if len(matched_reference) == len(assignments):
            break
    if any(index < 0 for index in assignments):
        raise RuntimeError("注意力锚点无法形成完整一对一匹配")
    return tuple(assignments)


@dataclass(frozen=True)
class AttentionAlignmentResult:
    """保存注意力锚点匹配和仿射恢复结果。"""

    affine_transform: tuple[tuple[float, float, float], tuple[float, float, float]]
    expected_anchor_indices: tuple[int, ...]
    observed_anchor_indices: tuple[int, ...]
    inlier_mask: tuple[bool, ...]
    inlier_ratio: float
    mean_inlier_residual: float
    registration_confidence: float
    geometry_reliable: bool
    alignment_digest: str
    metadata: dict[str, Any]


def recover_attention_affine_alignment(
    attention: Any,
    key_material: str,
    layer_name: str,
    anchor_count: int = 12,
    residual_threshold: float = 0.20,
    minimum_inlier_ratio: float = 0.50,
) -> AttentionAlignmentResult:
    """使用密钥关系行匹配和确定性 RANSAC 恢复仿射参考系。"""

    torch = _torch()
    matrix = attention.mean(dim=0) if attention.ndim == 3 else attention
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("attention 必须是方形矩阵或带 batch 维的方形矩阵")
    token_count = int(matrix.shape[0])
    coordinates = _grid_coordinates(token_count, matrix.device)
    expected_indices = _anchor_indices(token_count, anchor_count)
    relation_signs = keyed_relation_signs(matrix, key_material, layer_name)
    reference_rows = relation_signs[list(expected_indices)]
    observed_indices = _match_anchor_rows(matrix, reference_rows)
    source = coordinates[list(expected_indices)]
    target = coordinates[list(observed_indices)]

    best_transform = _fit_affine(source, target)
    best_inliers = torch.zeros(source.shape[0], dtype=torch.bool, device=source.device)
    best_residual = float("inf")
    combination_limit = 2048
    for combination_index, subset in enumerate(itertools.combinations(range(source.shape[0]), 3)):
        if combination_index >= combination_limit:
            break
        subset_tensor = torch.tensor(subset, device=source.device)
        transform = _fit_affine(source.index_select(0, subset_tensor), target.index_select(0, subset_tensor))
        residuals = (_apply_affine(source, transform) - target).norm(dim=1)
        inliers = residuals <= residual_threshold
        inlier_count = int(inliers.sum().item())
        if inlier_count < 3:
            continue
        mean_residual = float(residuals[inliers].mean().item())
        if inlier_count > int(best_inliers.sum().item()) or (
            inlier_count == int(best_inliers.sum().item()) and mean_residual < best_residual
        ):
            best_inliers = inliers
            best_residual = mean_residual
            best_transform = _fit_affine(source[inliers], target[inliers])
    residuals = (_apply_affine(source, best_transform) - target).norm(dim=1)
    best_inliers = residuals <= residual_threshold
    inlier_ratio = float(best_inliers.float().mean().item())
    mean_residual = (
        float(residuals[best_inliers].mean().item()) if bool(best_inliers.any()) else float("inf")
    )
    confidence = inlier_ratio * math.exp(-mean_residual) if math.isfinite(mean_residual) else 0.0
    geometry_reliable = inlier_ratio >= minimum_inlier_ratio and mean_residual <= residual_threshold
    transform_tuple = tuple(
        tuple(float(value) for value in row) for row in best_transform.detach().cpu().tolist()
    )
    payload = {
        "layer_name": layer_name,
        "expected_anchor_indices": expected_indices,
        "observed_anchor_indices": observed_indices,
        "inlier_mask": [bool(value) for value in best_inliers.detach().cpu().tolist()],
        "affine_transform": transform_tuple,
        "inlier_ratio": round(inlier_ratio, 12),
        "mean_inlier_residual": round(mean_residual, 12),
    }
    return AttentionAlignmentResult(
        affine_transform=transform_tuple,  # type: ignore[arg-type]
        expected_anchor_indices=expected_indices,
        observed_anchor_indices=observed_indices,
        inlier_mask=tuple(bool(value) for value in best_inliers.detach().cpu().tolist()),
        inlier_ratio=inlier_ratio,
        mean_inlier_residual=mean_residual,
        registration_confidence=confidence,
        geometry_reliable=geometry_reliable,
        alignment_digest=build_stable_digest(payload),
        metadata={
            "matcher": "keyed_relation_row_cosine_assignment",
            "transform_family": "affine",
            "robust_estimator": "deterministic_three_point_ransac",
        },
    )
