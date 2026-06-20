"""从注意力图构造相对几何恢复统计。"""

from __future__ import annotations

import math
from typing import Sequence

from main.core.digest import build_stable_digest
from main.methods.geometry.attention_graph_types import AttentionGraphRecord, GeometryEvidenceRecord


Matrix = tuple[tuple[float, ...], ...]


def row_softmax(logits: Sequence[float]) -> tuple[float, ...]:
    """对单行 logits 执行稳定 softmax。"""
    max_value = max(logits)
    exponents = tuple(math.exp(value - max_value) for value in logits)
    denominator = sum(exponents)
    return tuple(value / denominator for value in exponents)


def normalize_attention_rows(matrix: Sequence[Sequence[float]]) -> Matrix:
    """把非负注意力权重归一化为逐行概率分布。"""
    normalized_rows: list[tuple[float, ...]] = []
    for row in matrix:
        total = sum(max(float(value), 0.0) for value in row)
        if total <= 0.0:
            width = len(row)
            normalized_rows.append(tuple(1.0 / width for _ in row))
        else:
            normalized_rows.append(tuple(max(float(value), 0.0) / total for value in row))
    return tuple(normalized_rows)


def attention_from_query_key(query_vectors: Sequence[Sequence[float]], key_vectors: Sequence[Sequence[float]]) -> Matrix:
    """按 softmax(Q K^T / sqrt(d)) 构造注意力图。"""
    if not query_vectors or not key_vectors:
        raise ValueError("query_vectors 与 key_vectors 不得为空")
    feature_width = len(query_vectors[0])
    if feature_width <= 0:
        raise ValueError("query_vectors 的特征维度必须为正数")
    if any(len(row) != feature_width for row in query_vectors):
        raise ValueError("query_vectors 的每一行长度必须一致")
    if any(len(row) != feature_width for row in key_vectors):
        raise ValueError("key_vectors 的每一行长度必须等于 query_vectors 特征维度")
    scale = math.sqrt(feature_width)
    rows: list[tuple[float, ...]] = []
    for query in query_vectors:
        logits = []
        for key in key_vectors:
            logits.append(sum(float(left) * float(right) for left, right in zip(query, key)) / scale)
        rows.append(row_softmax(logits))
    return tuple(rows)


def stable_token_set(attention_matrix: Matrix, token_count: int = 3) -> tuple[int, ...]:
    """根据入度和出度质量选择稳定 token 集。"""
    row_count = len(attention_matrix)
    column_count = len(attention_matrix[0]) if attention_matrix else 0
    if row_count <= 0 or column_count <= 0:
        raise ValueError("attention_matrix 不得为空")
    bounded_count = max(1, min(token_count, row_count, column_count))
    scores = []
    for index in range(min(row_count, column_count)):
        outgoing = sum(attention_matrix[index]) / column_count if column_count else 0.0
        incoming = sum(row[index] for row in attention_matrix) / row_count if row_count else 0.0
        self_weight = attention_matrix[index][index] if index < row_count and index < column_count else 0.0
        scores.append((outgoing + incoming + self_weight, index))
    ordered = sorted(scores, key=lambda item: (-item[0], item[1]))
    return tuple(index for _, index in ordered[:bounded_count])


def relative_relation_values(attention_matrix: Matrix, token_indices: Sequence[int]) -> tuple[tuple[int, int, float], ...]:
    """提取稳定 token 集内部的相对关系权重。"""
    relations: list[tuple[int, int, float]] = []
    for source in token_indices:
        for target in token_indices:
            if source == target:
                continue
            if source < len(attention_matrix) and target < len(attention_matrix[source]):
                relations.append((int(source), int(target), float(attention_matrix[source][target])))
    return tuple(relations)


def relation_consistency(attention_matrix: Matrix, token_indices: Sequence[int]) -> float:
    """计算稳定 token 内双向注意力关系的一致性。"""
    pair_scores: list[float] = []
    for position, source in enumerate(token_indices):
        for target in token_indices[position + 1 :]:
            forward = attention_matrix[source][target]
            backward = attention_matrix[target][source]
            denominator = max(forward + backward, 1e-12)
            pair_scores.append(1.0 - abs(forward - backward) / denominator)
    if not pair_scores:
        return 1.0
    return max(0.0, min(1.0, sum(pair_scores) / len(pair_scores)))


def _row_correlation(left: Sequence[float], right: Sequence[float]) -> float:
    """计算两行注意力分布的非负相关一致性。"""
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    left_centered = tuple(value - left_mean for value in left)
    right_centered = tuple(value - right_mean for value in right)
    numerator = sum(left_value * right_value for left_value, right_value in zip(left_centered, right_centered))
    left_norm = math.sqrt(sum(value * value for value in left_centered))
    right_norm = math.sqrt(sum(value * value for value in right_centered))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.5
    return (numerator / (left_norm * right_norm) + 1.0) / 2.0


def recovered_sync_consistency(attention_matrix: Matrix, token_indices: Sequence[int]) -> float:
    """用稳定 token 行分布的一致性估计恢复同步质量。"""
    pair_scores: list[float] = []
    for position, source in enumerate(token_indices):
        for target in token_indices[position + 1 :]:
            pair_scores.append(_row_correlation(attention_matrix[source], attention_matrix[target]))
    if not pair_scores:
        return 1.0
    return max(0.0, min(1.0, sum(pair_scores) / len(pair_scores)))


def build_attention_graph_record(
    capture_id: str,
    attention_layer: str,
    attention_map_digest: str,
    attention_matrix: Sequence[Sequence[float]],
    unsupported_reason: str = "",
    token_count: int = 3,
) -> AttentionGraphRecord:
    """从注意力矩阵派生稳定锚点图记录。"""
    normalized_matrix = normalize_attention_rows(attention_matrix)
    token_indices = stable_token_set(normalized_matrix, token_count=token_count)
    relations = relative_relation_values(normalized_matrix, token_indices)
    consistency = relation_consistency(normalized_matrix, token_indices)
    payload = {
        "capture_id": capture_id,
        "attention_layer": attention_layer,
        "attention_map_digest": attention_map_digest,
        "stable_token_indices": token_indices,
        "relative_relation_values": [(source, target, round(weight, 12)) for source, target, weight in relations],
        "attention_relation_consistency": round(consistency, 12),
    }
    anchor_digest = build_stable_digest(payload)
    return AttentionGraphRecord(
        attention_graph_id=f"attention_graph_{anchor_digest[:16]}",
        capture_id=capture_id,
        attention_layer=attention_layer,
        attention_map_digest=attention_map_digest,
        attention_shape=(len(normalized_matrix), len(normalized_matrix[0]) if normalized_matrix else 0),
        stable_token_indices=token_indices,
        relative_relation_values=relations,
        attention_relation_consistency=consistency,
        anchor_graph_digest=anchor_digest,
        unsupported_reason=unsupported_reason,
        supports_paper_claim=False,
        metadata={"graph_source": "attention_matrix", "direct_positive_decision": False},
    )


def build_geometry_evidence_record(attention_graph: AttentionGraphRecord) -> GeometryEvidenceRecord:
    """从注意力图构造几何恢复统计证据。"""
    relation_weights = tuple(weight for _, _, weight in attention_graph.relative_relation_values)
    mean_relation = sum(relation_weights) / len(relation_weights) if relation_weights else 0.0
    inlier_ratio = (
        sum(1 for value in relation_weights if value >= mean_relation) / len(relation_weights)
        if relation_weights
        else 1.0
    )
    sync_consistency = max(0.0, min(1.0, 0.5 + 0.5 * attention_graph.attention_relation_consistency - 0.25 * abs(0.5 - inlier_ratio)))
    registration_confidence = max(
        0.0,
        min(1.0, 0.55 * attention_graph.attention_relation_consistency + 0.30 * inlier_ratio + 0.15 * sync_consistency),
    )
    alignment_residual = max(0.0, min(1.0, 1.0 - registration_confidence))
    geometry_reliable = registration_confidence >= 0.50 and inlier_ratio >= 0.50 and alignment_residual <= 0.50
    payload = {
        "attention_graph_id": attention_graph.attention_graph_id,
        "anchor_graph_digest": attention_graph.anchor_graph_digest,
        "registration_confidence": round(registration_confidence, 12),
        "anchor_inlier_ratio": round(inlier_ratio, 12),
        "recovered_sync_consistency": round(sync_consistency, 12),
        "alignment_residual": round(alignment_residual, 12),
    }
    evidence_digest = build_stable_digest(payload)
    return GeometryEvidenceRecord(
        geometry_evidence_record_id=f"geometry_evidence_{evidence_digest[:16]}",
        attention_graph_id=attention_graph.attention_graph_id,
        capture_id=attention_graph.capture_id,
        attention_relation_consistency=attention_graph.attention_relation_consistency,
        anchor_inlier_ratio=inlier_ratio,
        registration_confidence=registration_confidence,
        recovered_sync_consistency=sync_consistency,
        alignment_residual=alignment_residual,
        geometry_reliable=geometry_reliable,
        direct_positive_decision=False,
        unsupported_reason=attention_graph.unsupported_reason,
        supports_paper_claim=False,
        metadata={"anchor_graph_digest": attention_graph.anchor_graph_digest},
    )
