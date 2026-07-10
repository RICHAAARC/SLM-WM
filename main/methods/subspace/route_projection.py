"""将安全基底投影到 LF、尾部截断和 attention 路由。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from main.core.digest import build_stable_digest
from main.methods.semantic.routing import SemanticRoute
from main.methods.subspace.safe_basis import SafeBasisPlan


@dataclass(frozen=True)
class RouteBasisProjection:
    """安全基底的路由投影结果。"""

    lf_basis: tuple[tuple[float, ...], ...]
    tail_basis: tuple[tuple[float, ...], ...]
    attention_basis: tuple[tuple[float, ...], ...]
    route_projection_digest: str
    supports_paper_claim: bool
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """转为可写入 JSON 的字典。"""
        return asdict(self)


def _row_axis(row: tuple[float, ...]) -> int:
    """返回 one-hot 行中幅值最大的轴。"""
    return max(range(len(row)), key=lambda index: abs(row[index]))


def _filter_rows(rows: tuple[tuple[float, ...], ...], allowed_indices: tuple[int, ...]) -> tuple[tuple[float, ...], ...]:
    """按路由索引筛选基底行, 无命中时保留原基底以保证可运行。"""
    allowed = set(allowed_indices)
    filtered = tuple(row for row in rows if _row_axis(row) in allowed)
    return filtered or rows


def project_basis_by_route(safe_basis: SafeBasisPlan, semantic_route: SemanticRoute) -> RouteBasisProjection:
    """将安全基底投影到各路由分量。"""
    lf_basis = _filter_rows(safe_basis.safe_basis, semantic_route.lf_indices)
    tail_basis = _filter_rows(safe_basis.safe_basis, semantic_route.tail_indices)
    attention_basis = _filter_rows(safe_basis.safe_basis, semantic_route.attention_indices)
    payload = {
        "basis_digest": safe_basis.basis_digest,
        "route_digest": semantic_route.route_digest,
        "lf_basis": lf_basis,
        "tail_basis": tail_basis,
        "attention_basis": attention_basis,
    }
    return RouteBasisProjection(
        lf_basis=lf_basis,
        tail_basis=tail_basis,
        attention_basis=attention_basis,
        route_projection_digest=build_stable_digest(payload),
        supports_paper_claim=False,
        metadata={
            "lf_basis_count": len(lf_basis),
            "tail_basis_count": len(tail_basis),
            "attention_basis_count": len(attention_basis),
        },
    )
