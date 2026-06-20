"""内容检测分数。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

from main.core.digest import build_stable_digest
from main.methods.carrier.compose import ContentUpdate


@dataclass(frozen=True)
class ContentScore:
    """LF/HF 统一内容分数。"""

    lf_score: float
    hf_score: float
    content_score: float
    lambda_lf: float
    lambda_hf: float
    used_independent_branch_vote: bool
    fixed_fpr_ready: bool
    score_digest: str
    supports_paper_claim: bool
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        """校验融合权重边界。"""
        if not math.isclose(self.lambda_lf + self.lambda_hf, 1.0, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError("lambda_lf 与 lambda_hf 之和必须为 1")
        if self.lambda_lf <= self.lambda_hf:
            raise ValueError("lambda_lf 必须大于 lambda_hf")

    def to_dict(self) -> dict[str, Any]:
        """转为可写入 JSON 的字典。"""
        return asdict(self)


def vector_norm(values: tuple[float, ...]) -> float:
    """计算向量二范数。"""
    return math.sqrt(sum(value * value for value in values))


def correlation(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    """计算归一化相关分数。"""
    if len(left) != len(right):
        raise ValueError("相关分数输入长度必须一致")
    denominator = vector_norm(left) * vector_norm(right)
    if denominator == 0.0:
        return 0.0
    return sum(left_value * right_value for left_value, right_value in zip(left, right)) / denominator


def compute_unified_content_score(
    observed_values: tuple[float, ...],
    content_update: ContentUpdate,
    lambda_lf: float = 0.70,
    lambda_hf: float = 0.30,
) -> ContentScore:
    """计算统一内容分数, 不使用 LF/HF 独立阈值投票。"""
    lf_score = correlation(observed_values, content_update.lf_update_values)
    hf_score = correlation(observed_values, content_update.hf_update_values)
    content_score = lambda_lf * lf_score + lambda_hf * hf_score
    payload = {
        "content_update_digest": content_update.content_update_digest,
        "lf_score": round(lf_score, 12),
        "hf_score": round(hf_score, 12),
        "content_score": round(content_score, 12),
        "lambda_lf": lambda_lf,
        "lambda_hf": lambda_hf,
    }
    return ContentScore(
        lf_score=lf_score,
        hf_score=hf_score,
        content_score=content_score,
        lambda_lf=lambda_lf,
        lambda_hf=lambda_hf,
        used_independent_branch_vote=False,
        fixed_fpr_ready=True,
        score_digest=build_stable_digest(payload),
        supports_paper_claim=False,
        metadata={"score_name": "unified_content_score"},
    )
