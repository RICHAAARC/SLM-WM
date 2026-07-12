"""构造可由仅图像检测器重建的密钥化 latent 载体。

正式检测不能读取生成轨迹或样本级安全基底。因此检测模板在密钥、模型标识和
latent 形状确定后必须固定。嵌入端只把固定模板投影到安全子空间, 检测端仍与
原始固定模板计算统计量。投影会降低载体能量, 但不会引入检测端私有状态依赖。
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from main.core.digest import build_stable_digest
from main.core.keyed_prg import (
    KEYED_PRG_VERSION,
    build_keyed_gaussian_tensor,
    keyed_prg_protocol_record,
)
from main.methods.subspace.jacobian_nullspace import JacobianNullSpaceResult


def _torch() -> Any:
    """延迟导入 PyTorch。"""

    import torch

    return torch


def _normalize(template: Any) -> Any:
    """将模板转换为零均值单位二范数向量。"""

    centered = template.float() - template.float().mean()
    norm = centered.norm().clamp_min(1e-12)
    return centered / norm


def build_low_frequency_template(
    reference: Any,
    key_material: str,
    model_id: str,
    kernel_size: int = 5,
    *,
    prg_version: str = KEYED_PRG_VERSION,
) -> Any:
    """在真实 latent 空间轴上构造低通密钥模板。

    该算子只在 latent 的二维空间轴执行平均低通, 不混淆 batch 维和通道维。
    该定义使“低频”具有明确的空间含义。
    """

    import torch.nn.functional as functional

    if reference.ndim != 4:
        raise ValueError("低频模板要求 latent 具有 [batch, channel, height, width] 形状")
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError("kernel_size 必须为正奇数")
    shape = tuple(int(value) for value in reference.shape)
    raw = build_keyed_gaussian_tensor(
        shape,
        key_material,
        {
            "operator": "latent_carrier_template",
            "model_id": model_id,
            "branch_name": "lf_content",
        },
        prg_version=prg_version,
    )
    low_pass = functional.avg_pool2d(raw, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
    return _normalize(low_pass).to(
        device=reference.device,
        dtype=reference.dtype,
    )


def build_tail_robust_template(
    reference: Any,
    key_material: str,
    model_id: str,
    tail_fraction: float = 0.20,
    *,
    prg_version: str = KEYED_PRG_VERSION,
) -> tuple[Any, float, float]:
    """构造高斯幅值尾部截断的鲁棒载体模板。

    该分支只定义分布尾部筛选, 不定义空间频带。
    """

    torch = _torch()
    if not 0.0 < tail_fraction <= 1.0:
        raise ValueError("tail_fraction 必须位于 (0, 1]")
    shape = tuple(int(value) for value in reference.shape)
    raw = build_keyed_gaussian_tensor(
        shape,
        key_material,
        {
            "operator": "latent_carrier_template",
            "model_id": model_id,
            "branch_name": "tail_robust",
        },
        prg_version=prg_version,
    )
    flat = raw.reshape(-1)
    flat_values = flat.tolist()
    retained_count = max(1, math.ceil(len(flat_values) * tail_fraction))
    ranked_indices = sorted(
        range(len(flat_values)),
        key=lambda index: (-abs(flat_values[index]), index),
    )
    selected_indices = ranked_indices[:retained_count]
    selected_index_tensor = torch.tensor(
        selected_indices,
        dtype=torch.long,
        device="cpu",
    )
    truncated_flat = torch.zeros_like(flat)
    truncated_flat[selected_index_tensor] = flat[selected_index_tensor]
    truncated = truncated_flat.reshape(shape)
    threshold = abs(flat_values[selected_indices[-1]])
    retained_fraction = retained_count / len(flat_values)
    return (
        _normalize(truncated).to(
            device=reference.device,
            dtype=reference.dtype,
        ),
        float(threshold),
        float(retained_fraction),
    )


def normalized_correlation(observed: Any, template: Any) -> float:
    """计算去均值后的归一化相关统计量。"""

    left = observed.detach().float().reshape(-1)
    right = template.detach().float().reshape(-1)
    if left.numel() != right.numel():
        raise ValueError("observed 与 template 元素数必须一致")
    left = left - left.mean()
    right = right - right.mean()
    denominator = left.norm() * right.norm()
    if float(denominator.item()) <= 1e-12:
        return 0.0
    return float(((left * right).sum() / denominator).item())


@dataclass
class KeyedTensorCarrier:
    """保存固定检测模板和安全子空间投影后的嵌入方向。"""

    branch_name: str
    canonical_template: Any
    embedded_direction: Any
    projection_energy_retention: float
    template_digest: str
    metadata: dict[str, Any]

    def scaled_update(self, strength: float) -> Any:
        """生成指定二范数强度的 latent 更新。"""

        direction = self.embedded_direction.float()
        return (direction / direction.norm().clamp_min(1e-12) * strength).to(
            dtype=self.embedded_direction.dtype
        )

    def score(self, observed: Any) -> float:
        """使用检测端可重建的固定模板计算分支分数。"""

        return normalized_correlation(observed, self.canonical_template)


def project_canonical_template(
    branch_name: str,
    canonical_template: Any,
    null_space: JacobianNullSpaceResult,
    minimum_energy_retention: float = 0.01,
    *,
    prg_version: str = KEYED_PRG_VERSION,
) -> KeyedTensorCarrier:
    """将固定模板投影到真实 Jacobian Null Space。"""

    if not 0.0 < minimum_energy_retention <= 1.0:
        raise ValueError("minimum_energy_retention 必须位于 (0, 1]")
    prg_record = keyed_prg_protocol_record(prg_version)

    projected = null_space.project(canonical_template)
    canonical_energy = float(canonical_template.detach().float().square().sum().item())
    projected_energy = float(projected.detach().float().square().sum().item())
    retention = projected_energy / max(canonical_energy, 1e-12)
    if retention < minimum_energy_retention:
        raise RuntimeError("固定检测模板在安全子空间中的投影能量低于正式门禁")
    digest_payload = {
        "branch_name": branch_name,
        "shape": tuple(int(value) for value in canonical_template.shape),
        "projection_energy_retention": round(retention, 12),
        "null_space_digest": null_space.solver_digest,
        "keyed_prg_version": prg_version,
        "keyed_prg_protocol_digest": prg_record["keyed_prg_protocol_digest"],
    }
    return KeyedTensorCarrier(
        branch_name=branch_name,
        canonical_template=canonical_template,
        embedded_direction=projected,
        projection_energy_retention=retention,
        template_digest=build_stable_digest(digest_payload),
        metadata={
            "detector_reference": "key_model_and_latent_shape_only",
            "subspace_projection": "orthogonal_projection",
            "minimum_projection_energy_retention": minimum_energy_retention,
            "blind_detection_requires_generation_trace": False,
            "keyed_prg_version": prg_version,
            "keyed_prg_protocol_digest": prg_record[
                "keyed_prg_protocol_digest"
            ],
        },
    )


@dataclass(frozen=True)
class BlindContentScore:
    """保存仅图像检测使用的 LF 与尾部鲁棒分支统计量。"""

    lf_score: float
    tail_robust_score: float
    content_score: float
    lf_weight: float
    tail_robust_weight: float
    score_digest: str
    metadata: dict[str, Any]


def compute_blind_content_score(
    observed_latent: Any,
    lf_template: Any,
    tail_robust_template: Any,
    lf_weight: float = 0.70,
    tail_robust_weight: float = 0.30,
) -> BlindContentScore:
    """从图像编码 latent 与固定密钥模板计算统一内容分数。"""

    if not math.isclose(lf_weight + tail_robust_weight, 1.0, abs_tol=1e-9):
        raise ValueError("两个分支权重之和必须为 1")
    lf_score = normalized_correlation(observed_latent, lf_template)
    tail_score = normalized_correlation(observed_latent, tail_robust_template)
    content_score = lf_weight * lf_score + tail_robust_weight * tail_score
    payload = {
        "lf_score": round(lf_score, 12),
        "tail_robust_score": round(tail_score, 12),
        "content_score": round(content_score, 12),
        "lf_weight": lf_weight,
        "tail_robust_weight": tail_robust_weight,
    }
    return BlindContentScore(
        lf_score=lf_score,
        tail_robust_score=tail_score,
        content_score=content_score,
        lf_weight=lf_weight,
        tail_robust_weight=tail_robust_weight,
        score_digest=build_stable_digest(payload),
        metadata={
            "detector_access": "image_key_and_public_model_only",
            "tail_branch_semantics": "gaussian_amplitude_tail_truncation",
        },
    )
