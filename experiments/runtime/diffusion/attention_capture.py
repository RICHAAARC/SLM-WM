"""构造可审计 attention capture 记录。

当前实现只在 synthetic latent adapter 上生成摘要型 attention map。真实 Q/K hook 应在后续
运行单元接入, 这里必须通过 unsupported_reason 明确边界。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

from main.core.digest import build_stable_digest
from main.methods.geometry import attention_from_query_key


@dataclass(frozen=True)
class AttentionCaptureRecord:
    """描述一次 attention map 捕获或降级摘要。"""

    run_id: str
    model_family: str
    model_id: str
    capture_id: str
    attention_layer: str
    attention_map_digest: str
    attention_shape: tuple[int, int]
    attention_mean: float
    attention_entropy: float
    capture_backend: str
    unsupported_reason: str
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """转为 JSON 兼容字典。"""
        return asdict(self)


def _attention_matrix(values: tuple[float, ...], width: int = 4) -> tuple[tuple[float, ...], ...]:
    """从 latent 摘要中构造小型 row-normalized attention matrix。"""
    trimmed = values[:width] if len(values) >= width else values
    logits = tuple(tuple(abs(left * right) + 1e-6 for right in trimmed) for left in trimmed)
    rows: list[tuple[float, ...]] = []
    for row in logits:
        row_sum = sum(row)
        rows.append(tuple(value / row_sum for value in row))
    return tuple(rows)


def _matrix_entropy(matrix: tuple[tuple[float, ...], ...]) -> float:
    """计算 attention matrix 的平均熵。"""
    entropies = []
    for row in matrix:
        entropies.append(-sum(value * math.log(value) for value in row if value > 0.0))
    return sum(entropies) / len(entropies) if entropies else 0.0


def build_qk_attention_capture_record(
    run_id: str,
    model_family: str,
    model_id: str,
    attention_layer: str,
    query_vectors: tuple[tuple[float, ...], ...],
    key_vectors: tuple[tuple[float, ...], ...],
    capture_backend: str,
    unsupported_reason: str = "",
) -> AttentionCaptureRecord:
    """从 Q/K 向量构造可审计 attention capture 记录。"""
    matrix = attention_from_query_key(query_vectors, key_vectors)
    flattened = [value for row in matrix for value in row]
    digest = build_stable_digest([[round(value, 12) for value in row] for row in matrix])
    return AttentionCaptureRecord(
        run_id=run_id,
        model_family=model_family,
        model_id=model_id,
        capture_id=f"{attention_layer}_{digest[:12]}",
        attention_layer=attention_layer,
        attention_map_digest=digest,
        attention_shape=(len(matrix), len(matrix[0]) if matrix else 0),
        attention_mean=sum(flattened) / len(flattened),
        attention_entropy=_matrix_entropy(matrix),
        capture_backend=capture_backend,
        unsupported_reason=unsupported_reason,
        metadata={
            "capture_is_synthetic": False,
            "supports_paper_claim": False,
        },
    )


def build_attention_capture_records(
    run_id: str,
    model_family: str,
    model_id: str,
    backend_name: str,
    trajectory_vectors: tuple[tuple[float, ...], ...],
    unsupported_reason: str,
) -> tuple[AttentionCaptureRecord, ...]:
    """从首尾 latent 构造 attention capture records。"""
    if not trajectory_vectors:
        return ()
    selected = (("synthetic_attention_early", trajectory_vectors[0]), ("synthetic_attention_late", trajectory_vectors[-1]))
    records: list[AttentionCaptureRecord] = []
    for attention_layer, values in selected:
        matrix = _attention_matrix(values)
        flattened = [value for row in matrix for value in row]
        digest = build_stable_digest([[round(value, 12) for value in row] for row in matrix])
        records.append(
            AttentionCaptureRecord(
                run_id=run_id,
                model_family=model_family,
                model_id=model_id,
                capture_id=f"{attention_layer}_{digest[:12]}",
                attention_layer=attention_layer,
                attention_map_digest=digest,
                attention_shape=(len(matrix), len(matrix[0]) if matrix else 0),
                attention_mean=sum(flattened) / len(flattened),
                attention_entropy=_matrix_entropy(matrix),
                capture_backend=backend_name,
                unsupported_reason=unsupported_reason,
                metadata={
                    "capture_is_synthetic": True,
                    "supports_paper_claim": False,
                },
            )
        )
    return tuple(records)
