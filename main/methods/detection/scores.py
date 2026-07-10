"""内容检测分数。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

from main.core.digest import build_stable_digest
from main.methods.carrier.compose import ContentUpdate


@dataclass(frozen=True)
class ContentScore:
    """内容载体检测分数。

    content_score 是正式 fixed-FPR 检测使用的分数。真实 latent 写入仍使用
    LF/高斯幅值尾部截断合成后的 combined_update_values, 但正式检测必须同时约束 LF 与尾部截断
    两条证据链的一致性, 以降低 wrong-key 或 wrong-message carrier 在单一
    combined 方向上偶然高相关造成的 clean negative 高尾。
    """

    lf_score: float
    tail_score: float
    combined_score: float
    lf_tail_fusion_score: float
    content_score: float
    lambda_lf: float
    lambda_tail: float
    used_independent_branch_vote: bool
    fixed_fpr_ready: bool
    score_digest: str
    supports_paper_claim: bool
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        """校验融合权重边界。"""
        if not math.isclose(self.lambda_lf + self.lambda_tail, 1.0, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError("lambda_lf 与 lambda_tail 之和必须为 1")
        if self.lambda_lf <= self.lambda_tail:
            raise ValueError("lambda_lf 必须大于 lambda_tail")

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
    lambda_tail: float = 0.30,
) -> ContentScore:
    """计算带 LF/高斯幅值尾部截断一致性约束的统一内容分数。

    该函数属于项目特定方法逻辑: runtime 写入的水印方向是
    `combined_update_values`, 但论文级 fixed-FPR 检测不能只看单一 combined
    相关性。正式分数取 combined 相关性与 LF/尾部截断加权一致性分数的较小值,
    这样只有 combined 方向和 LF/尾部截断分支同时支持同一个 content carrier 时,
    样本才会获得高分。该设计属于项目特定写法, 主要用于压低 clean negative
    的 wrong-key 高分尾部, 为 full_paper 的 FPR=0.001 边界保留统计余量。
    """
    lf_score = correlation(observed_values, content_update.lf_update_values)
    tail_score = correlation(observed_values, content_update.tail_update_values)
    combined_score = correlation(observed_values, content_update.combined_update_values)
    lf_tail_fusion_score = lambda_lf * lf_score + lambda_tail * tail_score
    content_score = min(combined_score, lf_tail_fusion_score)
    payload = {
        "content_update_digest": content_update.content_update_digest,
        "lf_score": round(lf_score, 12),
        "tail_score": round(tail_score, 12),
        "combined_score": round(combined_score, 12),
        "lf_tail_fusion_score": round(lf_tail_fusion_score, 12),
        "content_score": round(content_score, 12),
        "lambda_lf": lambda_lf,
        "lambda_tail": lambda_tail,
        "formal_score_source": "lf_tail_consistency_guarded_combined_correlation",
    }
    return ContentScore(
        lf_score=lf_score,
        tail_score=tail_score,
        combined_score=combined_score,
        lf_tail_fusion_score=lf_tail_fusion_score,
        content_score=content_score,
        lambda_lf=lambda_lf,
        lambda_tail=lambda_tail,
        used_independent_branch_vote=False,
        fixed_fpr_ready=True,
        score_digest=build_stable_digest(payload),
        supports_paper_claim=False,
        metadata={
            "score_name": "lf_tail_consistency_guarded_content_score",
            "combined_score_name": "diagnostic_combined_update_correlation",
            "lf_tail_fusion_score_name": "lf_tail_weighted_consistency_score",
            "formal_score_source": "lf_tail_consistency_guarded_combined_correlation",
        },
    )
