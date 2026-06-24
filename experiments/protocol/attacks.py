"""攻击矩阵协议与受治理记录构造。"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Iterable

from main.core.digest import build_stable_digest


@dataclass(frozen=True)
class AttackConfig:
    """描述单个攻击配置。

    该配置属于通用工程写法: 将攻击名称、强度、资源档位和参数集中到
    dataclass 构造时校验, 使后续记录构建函数只表达攻击矩阵的数据流。
    """

    attack_id: str
    attack_family: str
    attack_name: str
    attack_strength: float
    resource_profile: str
    requires_gpu: bool
    enabled: bool
    attack_parameters: dict[str, Any]

    def __post_init__(self) -> None:
        """集中校验攻击配置边界。"""
        if self.attack_strength < 0.0:
            raise ValueError("attack_strength 不得小于 0")
        if self.resource_profile not in {"probe", "full_main", "full_extra"}:
            raise ValueError("resource_profile 必须属于受治理资源档位")
        if not self.attack_id or not self.attack_name or not self.attack_family:
            raise ValueError("攻击配置标识、名称和族名称不得为空")

    def to_dict(self) -> dict[str, Any]:
        """转为 JSON 兼容字典。"""
        return asdict(self)


@dataclass(frozen=True)
class AttackEvaluationBoundary:
    """描述攻击后检测复用的 fixed-FPR 与 rescue 统计边界。"""

    calibrated_content_threshold: float
    target_fpr: float
    rescue_margin_low: float
    allowed_fail_reasons: tuple[str, ...]
    fixed_fpr_control_scope: str = "calibration_clean_negative"
    fixed_fpr_denominator_role: str = "clean_negative_only"
    rescue_control_scope: str = "evidence_clean_negative"
    rescue_changes_fpr_denominator: bool = False
    attacked_negative_boundary_role: str = "attack_robustness_diagnostic_not_fpr_denominator"
    attacked_negative_governs_fixed_fpr: bool = False

    def __post_init__(self) -> None:
        """集中校验检测边界。"""
        if not 0.0 < self.target_fpr < 1.0:
            raise ValueError("target_fpr 必须位于 (0, 1)")
        if self.rescue_margin_low >= 0.0:
            raise ValueError("rescue_margin_low 必须小于 0")
        if not self.allowed_fail_reasons:
            raise ValueError("allowed_fail_reasons 不得为空")

    def to_dict(self) -> dict[str, Any]:
        """转为 JSON 兼容字典。"""
        payload = asdict(self)
        payload["allowed_fail_reasons"] = list(self.allowed_fail_reasons)
        return payload


@dataclass(frozen=True)
class AttackDetectionRecord:
    """记录攻击配置作用后的检测与恢复统计。

    当前本地实现只生成 record-level proxy, 不生成真实 attacked image 文件。
    因此所有记录都保持 supports_paper_claim=false, 真实图像攻击需要后续 GPU
    工作流补齐后再进入论文主张。
    """

    attack_record_id: str
    attack_record_digest: str
    source_record_id: str
    source_image_digest: str
    source_image_digest_source: str
    attack_id: str
    attack_family: str
    attack_name: str
    attack_strength: float
    resource_profile: str
    requires_gpu: bool
    attack_parameters: dict[str, Any]
    attack_config_digest: str
    attacked_image_digest: str
    attacked_image_digest_source: str
    attacked_image_available: bool
    attack_performed: bool
    split: str
    sample_role: str
    raw_content_score_before: float
    raw_content_score_after: float
    aligned_content_score_before: float
    aligned_content_score_after: float
    lf_score_retention: float
    hf_score_retention: float
    score_retention: float
    quality_score_proxy: float
    attention_consistency_proxy: float
    geometry_reliable: bool
    rescue_eligible: bool
    rescue_applied: bool
    evidence_decision: bool
    metric_status: str
    unsupported_reason: str
    supports_paper_claim: bool
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """转为可写入 JSON 的字典。"""
        return asdict(self)


def attack_config_digest(config: AttackConfig) -> str:
    """生成攻击配置的稳定摘要。"""
    return build_stable_digest(config.to_dict())


def default_attack_configs() -> tuple[AttackConfig, ...]:
    """返回默认攻击矩阵配置。"""
    return (
        AttackConfig(
            attack_id="jpeg_compression_probe",
            attack_family="standard_distortion",
            attack_name="jpeg_compression",
            attack_strength=0.10,
            resource_profile="probe",
            requires_gpu=False,
            enabled=True,
            attack_parameters={"quality": 90},
        ),
        AttackConfig(
            attack_id="jpeg_compression_main",
            attack_family="standard_distortion",
            attack_name="jpeg_compression",
            attack_strength=0.35,
            resource_profile="full_main",
            requires_gpu=False,
            enabled=True,
            attack_parameters={"quality": 60},
        ),
        AttackConfig(
            attack_id="gaussian_noise_main",
            attack_family="standard_distortion",
            attack_name="gaussian_noise",
            attack_strength=0.30,
            resource_profile="full_main",
            requires_gpu=False,
            enabled=True,
            attack_parameters={"sigma": 0.03},
        ),
        AttackConfig(
            attack_id="gaussian_blur_main",
            attack_family="standard_distortion",
            attack_name="gaussian_blur",
            attack_strength=0.28,
            resource_profile="full_main",
            requires_gpu=False,
            enabled=True,
            attack_parameters={"radius": 1.2},
        ),
        AttackConfig(
            attack_id="resize_main",
            attack_family="geometric_transform",
            attack_name="resize",
            attack_strength=0.25,
            resource_profile="full_main",
            requires_gpu=False,
            enabled=True,
            attack_parameters={"scale": 0.75},
        ),
        AttackConfig(
            attack_id="crop_main",
            attack_family="geometric_transform",
            attack_name="crop",
            attack_strength=0.30,
            resource_profile="full_main",
            requires_gpu=False,
            enabled=True,
            attack_parameters={"crop_ratio": 0.82},
        ),
        AttackConfig(
            attack_id="rotation_main",
            attack_family="geometric_transform",
            attack_name="rotation",
            attack_strength=0.24,
            resource_profile="full_main",
            requires_gpu=False,
            enabled=True,
            attack_parameters={"degrees": 5.0},
        ),
        AttackConfig(
            attack_id="crop_resize_main",
            attack_family="geometric_transform",
            attack_name="crop_resize",
            attack_strength=0.34,
            resource_profile="full_main",
            requires_gpu=False,
            enabled=True,
            attack_parameters={"crop_ratio": 0.80, "resize_scale": 1.0},
        ),
        AttackConfig(
            attack_id="composite_geometric_main",
            attack_family="geometric_transform",
            attack_name="composite_geometric_attacks",
            attack_strength=0.42,
            resource_profile="full_main",
            requires_gpu=False,
            enabled=True,
            attack_parameters={"crop_ratio": 0.78, "degrees": 7.0, "resize_scale": 0.85},
        ),
        AttackConfig(
            attack_id="img2img_regeneration_extra",
            attack_family="regeneration_attack",
            attack_name="img2img_regeneration",
            attack_strength=0.35,
            resource_profile="full_extra",
            requires_gpu=True,
            enabled=True,
            attack_parameters={"denoise_strength": 0.35},
        ),
        AttackConfig(
            attack_id="ddim_inversion_regeneration_extra",
            attack_family="regeneration_attack",
            attack_name="ddim_inversion_regeneration",
            attack_strength=0.40,
            resource_profile="full_extra",
            requires_gpu=True,
            enabled=True,
            attack_parameters={"inversion_steps": 30, "denoise_strength": 0.40},
        ),
        AttackConfig(
            attack_id="sdedit_regeneration_extra",
            attack_family="regeneration_attack",
            attack_name="sdedit_regeneration",
            attack_strength=0.45,
            resource_profile="full_extra",
            requires_gpu=True,
            enabled=True,
            attack_parameters={"noise_level": 0.45},
        ),
        AttackConfig(
            attack_id="diffusion_purification_extra",
            attack_family="regeneration_attack",
            attack_name="diffusion_purification",
            attack_strength=0.32,
            resource_profile="full_extra",
            requires_gpu=True,
            enabled=True,
            attack_parameters={"purification_steps": 20, "noise_level": 0.32},
        ),
    )


_ATTACK_EFFECTS: dict[str, dict[str, float]] = {
    "jpeg_compression": {"lf": 0.30, "hf": 0.58, "quality": 0.42, "attention": 0.18},
    "gaussian_noise": {"lf": 0.20, "hf": 0.72, "quality": 0.55, "attention": 0.24},
    "gaussian_blur": {"lf": 0.38, "hf": 0.62, "quality": 0.46, "attention": 0.26},
    "resize": {"lf": 0.28, "hf": 0.36, "quality": 0.32, "attention": 0.34},
    "crop": {"lf": 0.42, "hf": 0.40, "quality": 0.40, "attention": 0.48},
    "rotation": {"lf": 0.34, "hf": 0.35, "quality": 0.36, "attention": 0.52},
    "crop_resize": {"lf": 0.48, "hf": 0.46, "quality": 0.44, "attention": 0.56},
    "composite_geometric_attacks": {"lf": 0.55, "hf": 0.50, "quality": 0.52, "attention": 0.70},
    "img2img_regeneration": {"lf": 0.62, "hf": 0.70, "quality": 0.64, "attention": 0.78},
    "ddim_inversion_regeneration": {"lf": 0.66, "hf": 0.74, "quality": 0.68, "attention": 0.80},
    "sdedit_regeneration": {"lf": 0.72, "hf": 0.78, "quality": 0.72, "attention": 0.84},
    "diffusion_purification": {"lf": 0.58, "hf": 0.68, "quality": 0.60, "attention": 0.76},
}


def _field_value(record: dict[str, Any], field_name: str, default: Any = None) -> Any:
    """从字典记录中读取字段值。"""
    return record.get(field_name, default)


def _clamp01(value: float) -> float:
    """将数值限制在 [0, 1] 区间。"""
    return max(0.0, min(1.0, value))


def _deterministic_unit(*parts: Any) -> float:
    """根据输入片段生成 [0, 1] 内的确定性扰动。"""
    digest = build_stable_digest({"parts": parts})
    return int(digest[:12], 16) / float(0xFFFFFFFFFFFF)


def source_image_digest_from_record(source_record: dict[str, Any]) -> str:
    """由受治理源记录生成本地 source image 摘要代理。"""
    payload = {
        "aligned_detection_record_id": _field_value(source_record, "aligned_detection_record_id"),
        "content_detection_record_id": _field_value(source_record, "content_detection_record_id"),
        "prompt_id": _field_value(source_record, "prompt_id"),
        "split": _field_value(source_record, "split"),
        "sample_role": _field_value(source_record, "sample_role"),
    }
    return build_stable_digest(payload)


def deterministic_attack_effect(source_record: dict[str, Any], config: AttackConfig) -> dict[str, Any]:
    """计算 record-level 攻击代理带来的分数保持率和质量退化。

    该函数属于项目特定的轻量代理实现: 它不伪造真实攻击图片, 只为攻击
    矩阵协议提供可重建的本地审计记录。
    """
    effects = _ATTACK_EFFECTS[config.attack_name]
    source_id = str(_field_value(source_record, "aligned_detection_record_id", ""))
    variation = 0.92 + 0.16 * _deterministic_unit(source_id, config.attack_id, "variation")
    lf_loss = _clamp01(config.attack_strength * effects["lf"] * variation)
    hf_loss = _clamp01(config.attack_strength * effects["hf"] * (1.02 - 0.04 * variation))
    score_loss = _clamp01(0.62 * lf_loss + 0.38 * hf_loss)
    quality_loss = _clamp01(config.attack_strength * effects["quality"] * (0.95 + 0.10 * variation))
    attention_loss = _clamp01(config.attack_strength * effects["attention"] * (0.90 + 0.15 * variation))
    return {
        "lf_score_retention": _clamp01(1.0 - lf_loss),
        "hf_score_retention": _clamp01(1.0 - hf_loss),
        "score_retention": _clamp01(1.0 - score_loss),
        "quality_score_proxy": _clamp01(1.0 - quality_loss),
        "attention_consistency_proxy": _clamp01(1.0 - attention_loss),
        "score_loss": score_loss,
    }


def build_attack_detection_record(
    source_record: dict[str, Any],
    config: AttackConfig,
    boundary: AttackEvaluationBoundary,
) -> AttackDetectionRecord:
    """由源检测记录和攻击配置构造攻击后检测记录。"""
    config_digest = attack_config_digest(config)
    source_record_id = str(_field_value(source_record, "aligned_detection_record_id"))
    source_image_digest = source_image_digest_from_record(source_record)
    raw_before = float(_field_value(source_record, "raw_content_score", 0.0))
    aligned_before = float(_field_value(source_record, "aligned_content_score", raw_before))
    unsupported_reason = "real_gpu_attack_required" if config.requires_gpu else ""
    attack_performed = config.enabled and not config.requires_gpu
    metric_status = "measured_from_local_proxy" if attack_performed else "unsupported"

    if attack_performed:
        effect = deterministic_attack_effect(source_record, config)
        raw_after = _clamp01(raw_before * effect["score_retention"])
        retained_gain = max(0.0, aligned_before - raw_before) * effect["attention_consistency_proxy"]
        geometry_reliable = bool(_field_value(source_record, "geometry_reliable", False)) and effect[
            "attention_consistency_proxy"
        ] >= 0.55
        aligned_after = _clamp01(raw_after + retained_gain if geometry_reliable else raw_after)
        lf_retention = effect["lf_score_retention"]
        hf_retention = effect["hf_score_retention"]
        score_retention = effect["score_retention"]
        quality_score_proxy = effect["quality_score_proxy"]
        attention_consistency_proxy = effect["attention_consistency_proxy"]
    else:
        raw_after = raw_before
        aligned_after = aligned_before
        geometry_reliable = bool(_field_value(source_record, "geometry_reliable", False))
        lf_retention = 1.0
        hf_retention = 1.0
        score_retention = 1.0
        quality_score_proxy = 1.0
        attention_consistency_proxy = 1.0

    margin_after = raw_after - boundary.calibrated_content_threshold
    aligned_margin_after = aligned_after - boundary.calibrated_content_threshold
    positive_by_content = margin_after >= 0.0
    fail_reason = str(_field_value(source_record, "fail_reason", "geometry_suspected"))
    rescue_eligible = (
        attack_performed
        and boundary.rescue_margin_low <= margin_after < 0.0
        and geometry_reliable
        and fail_reason in boundary.allowed_fail_reasons
    )
    rescue_applied = rescue_eligible and aligned_margin_after >= 0.0
    evidence_decision = positive_by_content or rescue_applied
    attacked_image_digest = build_stable_digest(
        {
            "source_image_digest": source_image_digest,
            "attack_config_digest": config_digest,
            "raw_content_score_after": round(raw_after, 12),
            "aligned_content_score_after": round(aligned_after, 12),
            "metric_status": metric_status,
        }
    )
    record_digest = build_stable_digest(
        {
            "source_record_id": source_record_id,
            "attack_config_digest": config_digest,
            "attacked_image_digest": attacked_image_digest,
            "boundary": boundary.to_dict(),
        }
    )
    return AttackDetectionRecord(
        attack_record_id=f"attack_record_{record_digest[:16]}",
        attack_record_digest=record_digest,
        source_record_id=source_record_id,
        source_image_digest=source_image_digest,
        source_image_digest_source="source_record_local_proxy",
        attack_id=config.attack_id,
        attack_family=config.attack_family,
        attack_name=config.attack_name,
        attack_strength=config.attack_strength,
        resource_profile=config.resource_profile,
        requires_gpu=config.requires_gpu,
        attack_parameters=dict(config.attack_parameters),
        attack_config_digest=config_digest,
        attacked_image_digest=attacked_image_digest,
        attacked_image_digest_source="record_level_proxy_digest" if attack_performed else "not_generated",
        attacked_image_available=False,
        attack_performed=attack_performed,
        split=str(_field_value(source_record, "split", "unknown")),
        sample_role=str(_field_value(source_record, "sample_role", "unknown")),
        raw_content_score_before=raw_before,
        raw_content_score_after=raw_after,
        aligned_content_score_before=aligned_before,
        aligned_content_score_after=aligned_after,
        lf_score_retention=lf_retention,
        hf_score_retention=hf_retention,
        score_retention=score_retention,
        quality_score_proxy=quality_score_proxy,
        attention_consistency_proxy=attention_consistency_proxy,
        geometry_reliable=geometry_reliable,
        rescue_eligible=rescue_eligible,
        rescue_applied=rescue_applied,
        evidence_decision=evidence_decision,
        metric_status=metric_status,
        unsupported_reason=unsupported_reason,
        supports_paper_claim=False,
        metadata={
            "calibrated_content_threshold": boundary.calibrated_content_threshold,
            "target_fpr": boundary.target_fpr,
            "rescue_margin_low": boundary.rescue_margin_low,
            "source_supports_paper_claim": bool(_field_value(source_record, "supports_paper_claim", False)),
        },
    )


def build_attack_detection_records(
    source_records: Iterable[dict[str, Any]],
    attack_configs: Iterable[AttackConfig],
    boundary: AttackEvaluationBoundary,
) -> tuple[dict[str, Any], ...]:
    """批量构造攻击检测记录。"""
    rows: list[dict[str, Any]] = []
    for source_record in source_records:
        for config in attack_configs:
            if config.enabled:
                rows.append(build_attack_detection_record(source_record, config, boundary).to_dict())
    return tuple(rows)


def _mean(records: Iterable[dict[str, Any]], field_name: str) -> float:
    """计算记录字段均值。"""
    values = [float(record[field_name]) for record in records]
    return sum(values) / len(values) if values else 0.0


def _rate(records: Iterable[dict[str, Any]], field_name: str) -> float:
    """计算布尔字段触发率。"""
    rows = list(records)
    return sum(1 for record in rows if bool(record[field_name])) / len(rows) if rows else 0.0


def _group_records(records: Iterable[dict[str, Any]], keys: tuple[str, ...]) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    """按多个字段聚合记录。"""
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[tuple(record[key] for key in keys)].append(record)
    return dict(grouped)


def _supported(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """筛选已经执行本地代理攻击的记录。"""
    return [record for record in records if record.get("metric_status") != "unsupported"]


def _metric_status_for_group(records: Iterable[dict[str, Any]]) -> str:
    """根据受支持记录的真实来源, 归纳分组级 metric_status。"""
    supported = list(records)
    if not supported:
        return "unsupported"
    statuses = {str(record.get("metric_status", "")) for record in supported}
    real_status = "measured_from_real_attacked_image_formal_protocol"
    if statuses == {real_status}:
        return real_status
    if real_status in statuses:
        return "measured_from_mixed_real_and_local_proxy"
    return "measured_from_local_proxy"


def _rates_for_group(records: list[dict[str, Any]]) -> dict[str, Any]:
    """计算单个攻击分组的检测率与恢复率。"""
    supported = _supported(records)
    positives = [record for record in supported if record["sample_role"] == "positive_source"]
    clean_negatives = [record for record in supported if record["sample_role"] == "clean_negative"]
    attacked_negatives = [record for record in supported if record["sample_role"] == "attacked_negative"]
    negatives = [record for record in supported if record["sample_role"] != "positive_source"]
    metric_status = _metric_status_for_group(supported)
    return {
        "metric_status": metric_status,
        "attack_record_count": len(records),
        "supported_record_count": len(supported),
        "unsupported_record_count": len(records) - len(supported),
        "positive_count": len(positives),
        "negative_count": len(negatives),
        "true_positive_rate": _rate(positives, "evidence_decision"),
        "false_positive_rate": _rate(negatives, "evidence_decision"),
        "clean_false_positive_rate": _rate(clean_negatives, "evidence_decision"),
        "attacked_false_positive_rate": _rate(attacked_negatives, "evidence_decision"),
        "quality_score_proxy_mean": _mean(supported, "quality_score_proxy"),
        "score_retention_mean": _mean(supported, "score_retention"),
        "lf_score_retention_mean": _mean(supported, "lf_score_retention"),
        "hf_score_retention_mean": _mean(supported, "hf_score_retention"),
        "attention_consistency_proxy_mean": _mean(supported, "attention_consistency_proxy"),
        "geometry_reliable_rate": _rate(supported, "geometry_reliable"),
        "rescue_rate": _rate(supported, "rescue_applied"),
        "supports_paper_claim": False,
    }


def family_metrics(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """构造 attack family metrics 表。"""
    rows: list[dict[str, Any]] = []
    for key, group in sorted(_group_records(records, ("attack_family", "attack_name", "resource_profile")).items()):
        attack_family, attack_name, resource_profile = key
        rows.append(
            {
                "attack_family": attack_family,
                "attack_name": attack_name,
                "resource_profile": resource_profile,
                **_rates_for_group(group),
            }
        )
    return rows


def strength_curve(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """构造攻击强度曲线表。"""
    rows: list[dict[str, Any]] = []
    keys = ("attack_family", "attack_name", "attack_strength", "resource_profile")
    for key, group in sorted(_group_records(records, keys).items()):
        attack_family, attack_name, attack_strength, resource_profile = key
        rates = _rates_for_group(group)
        rows.append(
            {
                "attack_family": attack_family,
                "attack_name": attack_name,
                "attack_strength": attack_strength,
                "resource_profile": resource_profile,
                "metric_status": rates["metric_status"],
                "attack_record_count": rates["attack_record_count"],
                "supported_record_count": rates["supported_record_count"],
                "true_positive_rate": rates["true_positive_rate"],
                "false_positive_rate": rates["false_positive_rate"],
                "score_retention_mean": rates["score_retention_mean"],
                "quality_score_proxy_mean": rates["quality_score_proxy_mean"],
                "supports_paper_claim": False,
            }
        )
    return rows


def score_retention_rows(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """构造按攻击聚合的 score retention 表。"""
    rows: list[dict[str, Any]] = []
    keys = ("attack_family", "attack_name", "attack_strength", "resource_profile")
    for key, group in sorted(_group_records(records, keys).items()):
        attack_family, attack_name, attack_strength, resource_profile = key
        supported = _supported(group)
        score_values = [float(record["score_retention"]) for record in supported]
        rows.append(
            {
                "attack_family": attack_family,
                "attack_name": attack_name,
                "attack_strength": attack_strength,
                "resource_profile": resource_profile,
                "metric_status": _metric_status_for_group(supported),
                "attack_record_count": len(group),
                "supported_record_count": len(supported),
                "score_retention_mean": sum(score_values) / len(score_values) if score_values else 0.0,
                "score_retention_min": min(score_values) if score_values else 0.0,
                "score_retention_max": max(score_values) if score_values else 0.0,
                "lf_score_retention_mean": _mean(supported, "lf_score_retention"),
                "hf_score_retention_mean": _mean(supported, "hf_score_retention"),
                "supports_paper_claim": False,
            }
        )
    return rows


def rescue_by_attack_rows(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """构造按攻击聚合的 rescue 统计表。"""
    rows: list[dict[str, Any]] = []
    keys = ("attack_family", "attack_name", "attack_strength", "resource_profile")
    for key, group in sorted(_group_records(records, keys).items()):
        attack_family, attack_name, attack_strength, resource_profile = key
        supported = _supported(group)
        rows.append(
            {
                "attack_family": attack_family,
                "attack_name": attack_name,
                "attack_strength": attack_strength,
                "resource_profile": resource_profile,
                "metric_status": _metric_status_for_group(supported),
                "attack_record_count": len(group),
                "supported_record_count": len(supported),
                "rescue_eligible_count": sum(1 for record in supported if bool(record["rescue_eligible"])),
                "rescue_applied_count": sum(1 for record in supported if bool(record["rescue_applied"])),
                "rescue_rate": _rate(supported, "rescue_applied"),
                "geometry_reliable_rate": _rate(supported, "geometry_reliable"),
                "attention_consistency_proxy_mean": _mean(supported, "attention_consistency_proxy"),
                "supports_paper_claim": False,
            }
        )
    return rows
