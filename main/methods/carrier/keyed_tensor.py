"""构造可由仅图像检测器重建的密钥化 latent 载体。

正式检测不能读取生成轨迹或样本级安全基底。因此检测模板在密钥、模型标识和
latent 形状确定后必须固定。嵌入端只把固定模板投影到安全子空间, 检测端仍与
原始固定模板计算统计量。投影会降低载体能量, 但不会引入检测端私有状态依赖。
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING, Any, Mapping

from main.core.digest import (
    TENSOR_CONTENT_DIGEST_VERSION,
    build_stable_digest,
    tensor_content_sha256,
)
from main.core.keyed_prg import (
    KEYED_PRG_VERSION,
    build_keyed_gaussian_tensor,
    keyed_prg_protocol_record,
    require_supported_keyed_prg_version,
)
if TYPE_CHECKING:
    from main.methods.subspace.jacobian_nullspace import JacobianNullSpaceResult


LOW_FREQUENCY_CARRIER_PROTOCOL_SCHEMA = (
    "slm_wm_low_frequency_carrier_protocol"
)
LOW_FREQUENCY_KERNEL_SIZE = 5
LOW_FREQUENCY_STRIDE = 1
LOW_FREQUENCY_PADDING = 2
LOW_FREQUENCY_BOUNDARY_MODE = "zero_padding"
LOW_FREQUENCY_CEIL_MODE = False
LOW_FREQUENCY_COUNT_INCLUDE_PAD = True
LOW_FREQUENCY_DIVISOR_OVERRIDE = None
TAIL_ROBUST_CARRIER_PROTOCOL_SCHEMA = (
    "slm_wm_tail_robust_carrier_protocol"
)


def _torch() -> Any:
    """延迟导入 PyTorch。"""

    import torch

    return torch


def _center_and_l2_normalize(template: Any) -> Any:
    """在整个 Tensor 上执行一次去均值和 L2 归一化。"""

    values = template.float()
    if not bool(_torch().isfinite(values).all()):
        raise RuntimeError("载体模板必须全部有限")
    centered = values - values.mean()
    norm = centered.norm()
    if not bool(_torch().isfinite(norm)):
        raise RuntimeError("去均值载体模板能量必须有限")
    if float(norm.item()) == 0.0:
        raise RuntimeError("去均值载体模板没有可归一化的非零能量")
    normalized = centered / norm
    if not bool(_torch().isfinite(normalized).all()):
        raise RuntimeError("载体模板归一化产生非有限值")
    return normalized


def _l2_normalize_without_centering(template: Any) -> Any:
    """保留模板的精确零支持, 仅执行全 Tensor L2 归一化。"""

    values = template.float()
    if not bool(_torch().isfinite(values).all()):
        raise RuntimeError("载体模板必须全部有限")
    norm = values.norm()
    if not bool(_torch().isfinite(norm)):
        raise RuntimeError("载体模板能量必须有限")
    if float(norm.item()) == 0.0:
        raise RuntimeError("载体模板没有可归一化的非零能量")
    normalized = values / norm
    if not bool(_torch().isfinite(normalized).all()):
        raise RuntimeError("载体模板归一化产生非有限值")
    return normalized


@dataclass(frozen=True)
class LowFrequencyCarrierConfig:
    """冻结 LF 二维平均低通算子的完整离散协议.

    全部离散字段必须由调用方显式提供. 该对象同时服务嵌入端和仅图像检测端,
    从类型边界阻止任一端重新依赖运行库默认值或另一套硬编码参数.
    """

    kernel_size: int
    stride: int
    padding: int
    boundary_mode: str
    ceil_mode: bool
    count_include_pad: bool
    divisor_override: int | None

    def __post_init__(self) -> None:
        """严格拒绝类型伪装和正式 LF 协议漂移."""

        for field_name, value in (
            ("kernel_size", self.kernel_size),
            ("stride", self.stride),
            ("padding", self.padding),
        ):
            if type(value) is not int:
                raise TypeError(f"{field_name} 必须为精确 int")
        if type(self.boundary_mode) is not str:
            raise TypeError("boundary_mode 必须为精确 str")
        if type(self.ceil_mode) is not bool:
            raise TypeError("ceil_mode 必须为精确 bool")
        if type(self.count_include_pad) is not bool:
            raise TypeError("count_include_pad 必须为精确 bool")
        if self.divisor_override is not None:
            raise TypeError("divisor_override 必须为 None")
        if (
            self.kernel_size != LOW_FREQUENCY_KERNEL_SIZE
            or self.stride != LOW_FREQUENCY_STRIDE
            or self.padding != LOW_FREQUENCY_PADDING
            or self.boundary_mode != LOW_FREQUENCY_BOUNDARY_MODE
            or self.ceil_mode is not LOW_FREQUENCY_CEIL_MODE
            or self.count_include_pad is not LOW_FREQUENCY_COUNT_INCLUDE_PAD
            or self.divisor_override is not LOW_FREQUENCY_DIVISOR_OVERRIDE
        ):
            raise ValueError("LF 载体必须使用冻结的二维平均低通协议")

    def protocol_payload(self) -> dict[str, Any]:
        """返回不包含自摘要字段的规范协议正文."""

        return {
            "lf_carrier_protocol_schema": (
                LOW_FREQUENCY_CARRIER_PROTOCOL_SCHEMA
            ),
            "lf_kernel_size": self.kernel_size,
            "lf_stride": self.stride,
            "lf_padding": self.padding,
            "lf_boundary_mode": self.boundary_mode,
            "lf_ceil_mode": self.ceil_mode,
            "lf_count_include_pad": self.count_include_pad,
            "lf_divisor_override": self.divisor_override,
            "lf_pooling_axes": "height_width_only",
            "lf_batch_channel_isolation": True,
            "lf_normalization_scope": "global_tensor_center_then_l2",
        }

    @property
    def protocol_digest(self) -> str:
        """返回 LF 协议正文的稳定 SHA-256 摘要."""

        return build_stable_digest(self.protocol_payload())

    def to_record(self) -> dict[str, Any]:
        """返回可独立重算摘要的完整 LF 协议记录."""

        return {
            **self.protocol_payload(),
            "lf_carrier_protocol_digest": self.protocol_digest,
        }


def validate_low_frequency_carrier_protocol_record(
    record: Mapping[str, Any],
) -> LowFrequencyCarrierConfig:
    """从持久化正文重建 LF 配置并独立复验协议摘要."""

    expected_fields = {
        "lf_carrier_protocol_schema",
        "lf_kernel_size",
        "lf_stride",
        "lf_padding",
        "lf_boundary_mode",
        "lf_ceil_mode",
        "lf_count_include_pad",
        "lf_divisor_override",
        "lf_pooling_axes",
        "lf_batch_channel_isolation",
        "lf_normalization_scope",
        "lf_carrier_protocol_digest",
    }
    if not isinstance(record, Mapping) or set(record) != expected_fields:
        raise ValueError("LF 载体协议记录字段集合不完整")
    config = LowFrequencyCarrierConfig(
        kernel_size=record["lf_kernel_size"],
        stride=record["lf_stride"],
        padding=record["lf_padding"],
        boundary_mode=record["lf_boundary_mode"],
        ceil_mode=record["lf_ceil_mode"],
        count_include_pad=record["lf_count_include_pad"],
        divisor_override=record["lf_divisor_override"],
    )
    expected_record = config.to_record()
    if dict(record) != expected_record:
        raise ValueError("LF 载体协议正文或摘要发生漂移")
    return config


def tail_robust_carrier_protocol_record(
    tail_fraction: float,
    *,
    prg_version: str,
) -> dict[str, Any]:
    """返回高斯幅值尾部载体的单一版本化协议记录."""

    if type(tail_fraction) is not float or not 0.0 < tail_fraction <= 1.0:
        raise ValueError("tail_fraction 必须为 (0, 1] 内的精确 float")
    require_supported_keyed_prg_version(prg_version)
    payload = {
        "tail_carrier_protocol_schema": TAIL_ROBUST_CARRIER_PROTOCOL_SCHEMA,
        "tail_fraction": tail_fraction,
        "tail_selection_rule": (
            "all_elements_without_amplitude_ranking"
            if tail_fraction == 1.0
            else "descending_absolute_value_then_ascending_flat_index"
        ),
        "keyed_prg_version": prg_version,
    }
    return {
        **payload,
        "tail_carrier_protocol_digest": build_stable_digest(payload),
    }


def validate_tail_robust_carrier_protocol_record(
    record: Mapping[str, Any],
) -> dict[str, Any]:
    """独立重建尾部载体协议并复验自摘要."""

    expected_fields = {
        "tail_carrier_protocol_schema",
        "tail_fraction",
        "tail_selection_rule",
        "keyed_prg_version",
        "tail_carrier_protocol_digest",
    }
    if not isinstance(record, Mapping) or set(record) != expected_fields:
        raise ValueError("尾部载体协议记录字段集合不完整")
    expected = tail_robust_carrier_protocol_record(
        record["tail_fraction"],
        prg_version=record["keyed_prg_version"],
    )
    if dict(record) != expected:
        raise ValueError("尾部载体协议正文或摘要发生漂移")
    return expected


def build_low_frequency_template(
    reference: Any,
    key_material: str,
    model_id: str,
    low_frequency_config: LowFrequencyCarrierConfig,
    *,
    prg_version: str,
) -> Any:
    """在真实 latent 空间轴上构造低通密钥模板。

    该算子只在 latent 的二维空间轴执行平均低通, 不混淆 batch 维和通道维。
    该定义使“低频”具有明确的空间含义。
    """

    import torch.nn.functional as functional

    if not isinstance(low_frequency_config, LowFrequencyCarrierConfig):
        raise TypeError("low_frequency_config 必须为 LowFrequencyCarrierConfig")
    if reference.ndim != 4:
        raise ValueError("低频模板要求 latent 具有 [batch, channel, height, width] 形状")
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
    low_pass = functional.avg_pool2d(
        raw,
        kernel_size=low_frequency_config.kernel_size,
        stride=low_frequency_config.stride,
        padding=low_frequency_config.padding,
        ceil_mode=low_frequency_config.ceil_mode,
        count_include_pad=low_frequency_config.count_include_pad,
        divisor_override=low_frequency_config.divisor_override,
    )
    if tuple(low_pass.shape) != shape:
        raise ValueError("低频池化参数必须保持载体 Tensor 形状不变")
    return _center_and_l2_normalize(low_pass).to(
        device=reference.device,
        dtype=_torch().float32,
    )


def build_tail_robust_template(
    reference: Any,
    key_material: str,
    model_id: str,
    tail_fraction: float,
    *,
    prg_version: str,
) -> tuple[Any, float, float]:
    """构造高斯幅值尾部截断的鲁棒载体模板。

    该分支只定义分布尾部筛选, 不定义空间频带。
    """

    torch = _torch()
    if getattr(reference, "ndim", None) != 4:
        raise ValueError("尾部模板要求 latent 具有 [batch, channel, height, width] 形状")
    if type(tail_fraction) is not float or not 0.0 < tail_fraction <= 1.0:
        raise ValueError("tail_fraction 必须为 (0, 1] 内的精确 float")
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
    if tail_fraction == 1.0:
        return (
            _l2_normalize_without_centering(raw).to(
                device=reference.device,
                dtype=torch.float32,
            ),
            0.0,
            1.0,
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
        _l2_normalize_without_centering(truncated).to(
            device=reference.device,
            dtype=torch.float32,
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
    if not bool(_torch().isfinite(left).all()) or not bool(
        _torch().isfinite(right).all()
    ):
        raise RuntimeError("相关统计量输入必须全部有限")
    left = left - left.mean()
    right = right - right.mean()
    left_norm = left.norm()
    right_norm = right.norm()
    if not bool(_torch().isfinite(left_norm)) or not bool(
        _torch().isfinite(right_norm)
    ):
        raise RuntimeError("相关统计量的中心化能量必须有限")
    if float(left_norm.item()) == 0.0 or float(right_norm.item()) == 0.0:
        raise RuntimeError("相关统计量要求观测量和模板都具有非零中心化能量")
    score = (left / left_norm * (right / right_norm)).sum()
    if not bool(_torch().isfinite(score)):
        raise RuntimeError("相关统计量产生非有限值")
    return float(score.item())


@dataclass
class KeyedTensorCarrier:
    """保存固定检测模板和安全子空间投影后的嵌入方向."""

    branch_name: str
    canonical_template: Any
    embedded_direction: Any
    projection_energy_retention: float
    template_digest: str
    template_shape: tuple[int, ...]
    carrier_protocol_digest: str
    metadata: dict[str, Any]

    def score(self, observed: Any) -> float:
        """使用检测端可重建的固定模板计算分支分数."""

        return normalized_correlation(observed, self.canonical_template)


def project_canonical_template(
    branch_name: str,
    canonical_template: Any,
    null_space: JacobianNullSpaceResult,
    minimum_energy_retention: float,
    *,
    carrier_protocol_digest: str,
    prg_version: str,
) -> KeyedTensorCarrier:
    """将固定模板投影到真实 Jacobian Null Space。"""

    if not 0.0 < minimum_energy_retention <= 1.0:
        raise ValueError("minimum_energy_retention 必须位于 (0, 1]")
    if not isinstance(carrier_protocol_digest, str) or len(
        carrier_protocol_digest
    ) != 64 or any(
        character not in "0123456789abcdef"
        for character in carrier_protocol_digest
    ):
        raise ValueError("carrier_protocol_digest 必须为规范 SHA-256")
    prg_record = keyed_prg_protocol_record(prg_version)

    projected = null_space.project(canonical_template)
    canonical_energy = float(canonical_template.detach().float().square().sum().item())
    projected_energy = float(projected.detach().float().square().sum().item())
    retention = projected_energy / max(canonical_energy, 1e-12)
    if retention < minimum_energy_retention:
        raise RuntimeError("固定检测模板在安全子空间中的投影能量低于正式门禁")
    digest_payload = {
        "branch_name": branch_name,
        "template_shape": [
            int(value) for value in canonical_template.shape
        ],
        "projection_energy_retention": round(retention, 12),
        "minimum_projection_energy_retention": (
            minimum_energy_retention
        ),
        "null_space_digest": null_space.solver_digest,
        "canonical_template_content_sha256": tensor_content_sha256(
            canonical_template
        ),
        "embedded_direction_content_sha256": tensor_content_sha256(projected),
        "tensor_content_digest_version": TENSOR_CONTENT_DIGEST_VERSION,
        "keyed_prg_version": prg_version,
        "keyed_prg_protocol_digest": prg_record["keyed_prg_protocol_digest"],
        "carrier_protocol_digest": carrier_protocol_digest,
    }
    template_digest = build_stable_digest(digest_payload)
    return KeyedTensorCarrier(
        branch_name=branch_name,
        canonical_template=canonical_template,
        embedded_direction=projected,
        projection_energy_retention=retention,
        template_digest=template_digest,
        template_shape=tuple(int(value) for value in canonical_template.shape),
        carrier_protocol_digest=carrier_protocol_digest,
        metadata={
            "detector_reference": "key_model_and_latent_shape_only",
            "subspace_projection": "orthogonal_projection",
            "minimum_projection_energy_retention": minimum_energy_retention,
            "blind_detection_requires_generation_trace": False,
            "keyed_prg_version": prg_version,
            "keyed_prg_protocol_digest": prg_record[
                "keyed_prg_protocol_digest"
            ],
            "carrier_protocol_digest": carrier_protocol_digest,
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
    lf_template: Any | None,
    tail_robust_template: Any | None,
    lf_weight: float,
    tail_robust_weight: float,
) -> BlindContentScore:
    """从图像编码 latent 与固定密钥模板计算统一内容分数。"""

    if (
        type(lf_weight) is not float
        or type(tail_robust_weight) is not float
    ):
        raise TypeError("两个内容分支权重必须为精确 float")
    if (
        not 0.0 <= lf_weight <= 1.0
        or not 0.0 <= tail_robust_weight <= 1.0
    ):
        raise ValueError("两个内容分支权重必须位于 [0, 1]")
    if not math.isclose(lf_weight + tail_robust_weight, 1.0, abs_tol=1e-9):
        raise ValueError("两个分支权重之和必须为 1")
    if (lf_weight > 0.0) != (lf_template is not None):
        raise ValueError("LF 模板必须且只能在 LF 权重大于0时提供")
    if (tail_robust_weight > 0.0) != (tail_robust_template is not None):
        raise ValueError("尾部模板必须且只能在尾部分支权重大于0时提供")
    lf_score = (
        normalized_correlation(observed_latent, lf_template)
        if lf_template is not None
        else 0.0
    )
    tail_score = (
        normalized_correlation(observed_latent, tail_robust_template)
        if tail_robust_template is not None
        else 0.0
    )
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
