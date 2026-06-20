"""定义 SLM-WM 核心方法对象的最小 typed object。

本模块只描述对象边界和跨模块数据形状, 不执行模型推理、不访问 Notebook、
不读取外部实验目录, 也不写出论文产物。该设计属于通用工程写法与项目特定
方法对象的结合: 通用部分是使用不可变 dataclass 传递结构化配置, 项目特定
部分是将语义条件、潜空间子空间、水印载体和检测证据拆成可独立审计的对象。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class SemanticConditionSpec:
    """描述语义条件化水印路由的最小输入对象。

    该对象保存语义摘要和语义标签, 用于后续方法模块选择风险策略和潜空间路由。
    此处只保留可复现的结构化信息, 避免把外部执行环境或临时输入通道写入核心包。
    """

    condition_id: str
    semantic_digest: str
    semantic_tags: tuple[str, ...]
    risk_policy: str
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """转为普通字典, 便于脚本、测试或后续 manifest 记录复用。"""
        return asdict(self)


@dataclass(frozen=True)
class LatentSubspaceSpec:
    """描述潜空间流形中可承载水印的安全子空间。

    该对象不保存真实 latent 张量, 只保存子空间标识、基向量摘要和维度信息。
    这种设计便于在不绑定具体模型运行环境的前提下冻结方法边界。
    """

    subspace_id: str
    basis_digest: str
    manifold_dimension: int
    safe_axes: tuple[str, ...]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """转为普通字典, 便于后续协议层或测试层读取。"""
        return asdict(self)


@dataclass(frozen=True)
class WatermarkCarrierSpec:
    """描述潜空间水印载体的最小机制对象。

    该对象把载体族、频带和嵌入强度分开保存, 使后续 LF/HF 载体、注意力载体或
    其他机制变体可以复用同一对象边界。
    """

    carrier_id: str
    carrier_family: str
    frequency_band: str
    embedding_strength: float
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """转为普通字典, 便于方法组合与轻量测试复用。"""
        return asdict(self)


@dataclass(frozen=True)
class AttentionAnchorSpec:
    """描述注意力相对几何约束中的锚点对象。

    该对象只记录锚点摘要和注意力层名称, 不保存模型内部状态。后续实现可通过
    该对象连接自注意力图、几何 rescue 和潜空间相对更新机制。
    """

    anchor_id: str
    attention_layer: str
    anchor_digest: str
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """转为普通字典, 便于后续证据对象引用。"""
        return asdict(self)


@dataclass(frozen=True)
class DetectionEvidenceSpec:
    """描述单个检测证据通道的最小对象。

    该对象用于承载内容证据、几何证据或注意力证据的分数。它不负责 fixed-FPR
    阈值校准, 只提供后续协议层可消费的证据形状。
    """

    evidence_id: str
    evidence_type: str
    score_name: str
    score_value: float
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """转为普通字典, 便于后续融合决策或报告重建流程引用。"""
        return asdict(self)


@dataclass(frozen=True)
class FusionDecisionSpec:
    """描述多证据融合后的最小检测决策对象。

    该对象仅保存决策标签、阈值名称、阈值数值和被引用的证据标识。完整统计边界
    仍由后续协议单元定义, 当前单元只冻结对象形状和核心包依赖边界。
    """

    decision_id: str
    decision_label: str
    threshold_name: str
    threshold_value: float
    evidence_ids: tuple[str, ...]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """转为普通字典, 便于记录、测试和后续 artifact builder 复用。"""
        return asdict(self)


CORE_METHOD_OBJECT_NAMES = (
    "semantic_condition_spec",
    "latent_subspace_spec",
    "watermark_carrier_spec",
    "attention_anchor_spec",
    "detection_evidence_spec",
    "fusion_decision_spec",
)
