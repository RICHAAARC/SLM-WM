"""外部 baseline 的公平对比协议与受治理观测记录。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

from main.core.digest import build_stable_digest


@dataclass(frozen=True)
class BaselineSpec:
    """描述一个外部 baseline 的对比配置。

    该对象属于通用工程写法: 把 baseline 身份、资源需求、实现状态和结果来源集中到
    dataclass 构造时校验, 让后续观测记录构建函数只表达协议数据流。
    """

    baseline_id: str
    baseline_family: str
    baseline_name: str
    comparison_group: str
    expected_input_mode: str
    requires_gpu: bool
    requires_training: bool
    baseline_adapter_ready: bool
    baseline_official_code_ready: bool
    baseline_reproduced_result_ready: bool
    baseline_imported_result_ready: bool
    baseline_result_source: str
    unsupported_reason: str

    def __post_init__(self) -> None:
        """集中校验 baseline 配置的不可恢复边界。"""
        if self.comparison_group not in {"primary", "supplemental"}:
            raise ValueError("comparison_group 必须是 primary 或 supplemental")
        if not self.baseline_id or not self.baseline_family or not self.baseline_name:
            raise ValueError("baseline 身份字段不得为空")
        if (self.baseline_reproduced_result_ready or self.baseline_imported_result_ready) and self.unsupported_reason:
            raise ValueError("已就绪 baseline 结果不得同时携带 unsupported_reason")

    def to_dict(self) -> dict[str, Any]:
        """转为 JSON 兼容字典。"""
        return asdict(self)


@dataclass(frozen=True)
class BaselineObservation:
    """记录外部 baseline 在共同协议边界下的一条可审计观测。

    当前仓库尚未运行外部方法官方代码, 因此该记录只冻结公平对比边界和缺失原因,
    不手工填写外部方法指标, 也不支持论文主张。
    """

    baseline_observation_id: str
    baseline_observation_digest: str
    baseline_id: str
    baseline_family: str
    baseline_name: str
    comparison_group: str
    expected_input_mode: str
    attack_family: str
    attack_name: str
    resource_profile: str
    comparable_operating_point: str
    common_prompt_protocol_ready: bool
    common_attack_protocol_ready: bool
    common_threshold_protocol_ready: bool
    baseline_protocol_compatible: bool
    baseline_requires_gpu: bool
    baseline_requires_training: bool
    baseline_adapter_ready: bool
    baseline_official_code_ready: bool
    baseline_reproduced_result_ready: bool
    baseline_imported_result_ready: bool
    baseline_result_source: str
    metric_status: str
    unsupported_reason: str
    positive_count: int
    negative_count: int
    attack_record_count: int
    supported_record_count: int
    true_positive_rate: str
    false_positive_rate: str
    clean_false_positive_rate: str
    attacked_false_positive_rate: str
    quality_score_proxy_mean: str
    score_retention_mean: str
    supports_paper_claim: bool

    def to_dict(self) -> dict[str, Any]:
        """转为 JSON 兼容字典。"""
        return asdict(self)


def default_baseline_specs() -> tuple[BaselineSpec, ...]:
    """返回默认外部 baseline 对比清单。

    这里实现的是协议 adapter, 不是外部方法本体。外部官方代码、复现实验结果或导入结果
    未完成前, 每个 baseline 只能写出 unsupported 观测记录。
    """
    common = {
        "requires_gpu": True,
        "requires_training": False,
        "baseline_adapter_ready": True,
        "baseline_official_code_ready": False,
        "baseline_reproduced_result_ready": False,
        "baseline_imported_result_ready": False,
        "baseline_result_source": "not_available",
        "unsupported_reason": "external_baseline_result_missing",
    }
    return (
        BaselineSpec(
            baseline_id="tree_ring",
            baseline_family="diffusion_latent_watermark",
            baseline_name="Tree-Ring",
            comparison_group="primary",
            expected_input_mode="diffusion_latent",
            **common,
        ),
        BaselineSpec(
            baseline_id="gaussian_shading",
            baseline_family="diffusion_noise_watermark",
            baseline_name="Gaussian Shading",
            comparison_group="primary",
            expected_input_mode="diffusion_noise",
            **common,
        ),
        BaselineSpec(
            baseline_id="shallow_diffuse",
            baseline_family="diffusion_image_watermark",
            baseline_name="Shallow Diffuse",
            comparison_group="primary",
            expected_input_mode="diffusion_image",
            **common,
        ),
        BaselineSpec(
            baseline_id="t2smark",
            baseline_family="text_to_image_watermark",
            baseline_name="T2SMark",
            comparison_group="primary",
            expected_input_mode="text_to_image",
            **common,
        ),
        BaselineSpec(
            baseline_id="stable_signature",
            baseline_family="decoder_signature_watermark",
            baseline_name="Stable Signature",
            comparison_group="supplemental",
            expected_input_mode="decoder_signature",
            **common,
        ),
        BaselineSpec(
            baseline_id="rivagan",
            baseline_family="image_space_watermark",
            baseline_name="RivaGAN",
            comparison_group="supplemental",
            expected_input_mode="image_space",
            **common,
        ),
        BaselineSpec(
            baseline_id="trustmark",
            baseline_family="image_space_watermark",
            baseline_name="TrustMark",
            comparison_group="supplemental",
            expected_input_mode="image_space",
            **common,
        ),
        BaselineSpec(
            baseline_id="watermark_anything",
            baseline_family="image_space_watermark",
            baseline_name="Watermark Anything",
            comparison_group="supplemental",
            expected_input_mode="image_space",
            **common,
        ),
    )


def _int_field(row: dict[str, Any], field_name: str) -> int:
    """读取 CSV 行中的整数指标。"""
    return int(float(row.get(field_name, 0) or 0))


def _float_field(row: dict[str, Any], field_name: str) -> float:
    """读取 CSV 行中的浮点指标。"""
    return float(row.get(field_name, 0.0) or 0.0)


def _weighted_mean(rows: Iterable[dict[str, Any]], value_field: str, weight_field: str) -> float:
    """按给定计数字段计算加权均值。"""
    total_weight = 0
    weighted_sum = 0.0
    for row in rows:
        weight = _int_field(row, weight_field)
        total_weight += weight
        weighted_sum += _float_field(row, value_field) * weight
    return weighted_sum / total_weight if total_weight else 0.0


def _operating_point(boundary: dict[str, Any]) -> str:
    """根据 fixed-FPR 边界生成可读 operating point 名称。"""
    target_fpr = float(boundary.get("target_fpr", 0.05))
    return f"fixed_fpr_{target_fpr:g}"


def _baseline_protocol_compatible(spec: BaselineSpec, attack_row: dict[str, Any]) -> bool:
    """判断 baseline 是否可登记到共同 prompt、攻击和阈值协议下。"""
    return bool(spec.baseline_adapter_ready and attack_row.get("attack_name") and attack_row.get("resource_profile"))


def build_baseline_observations(
    baseline_specs: Iterable[BaselineSpec],
    attack_metric_rows: Iterable[dict[str, Any]],
    boundary: dict[str, Any],
) -> tuple[dict[str, Any], ...]:
    """基于攻击矩阵行构造外部 baseline 观测记录。"""
    rows: list[dict[str, Any]] = []
    comparable_operating_point = _operating_point(boundary)
    for spec in baseline_specs:
        for attack_row in attack_metric_rows:
            protocol_compatible = _baseline_protocol_compatible(spec, attack_row)
            ready = spec.baseline_reproduced_result_ready or spec.baseline_imported_result_ready
            metric_status = "measured" if ready else "unsupported"
            payload = {
                "baseline_id": spec.baseline_id,
                "attack_family": attack_row.get("attack_family", ""),
                "attack_name": attack_row.get("attack_name", ""),
                "resource_profile": attack_row.get("resource_profile", ""),
                "comparable_operating_point": comparable_operating_point,
            }
            digest = build_stable_digest({**spec.to_dict(), **payload})
            rows.append(
                BaselineObservation(
                    baseline_observation_id=f"baseline_observation_{digest[:16]}",
                    baseline_observation_digest=digest,
                    baseline_id=spec.baseline_id,
                    baseline_family=spec.baseline_family,
                    baseline_name=spec.baseline_name,
                    comparison_group=spec.comparison_group,
                    expected_input_mode=spec.expected_input_mode,
                    attack_family=str(attack_row.get("attack_family", "")),
                    attack_name=str(attack_row.get("attack_name", "")),
                    resource_profile=str(attack_row.get("resource_profile", "")),
                    comparable_operating_point=comparable_operating_point,
                    common_prompt_protocol_ready=True,
                    common_attack_protocol_ready=True,
                    common_threshold_protocol_ready=True,
                    baseline_protocol_compatible=protocol_compatible,
                    baseline_requires_gpu=spec.requires_gpu,
                    baseline_requires_training=spec.requires_training,
                    baseline_adapter_ready=spec.baseline_adapter_ready,
                    baseline_official_code_ready=spec.baseline_official_code_ready,
                    baseline_reproduced_result_ready=spec.baseline_reproduced_result_ready,
                    baseline_imported_result_ready=spec.baseline_imported_result_ready,
                    baseline_result_source=spec.baseline_result_source,
                    metric_status=metric_status,
                    unsupported_reason="" if ready else spec.unsupported_reason,
                    positive_count=_int_field(attack_row, "positive_count"),
                    negative_count=_int_field(attack_row, "negative_count"),
                    attack_record_count=_int_field(attack_row, "attack_record_count"),
                    supported_record_count=0,
                    true_positive_rate="unsupported",
                    false_positive_rate="unsupported",
                    clean_false_positive_rate="unsupported",
                    attacked_false_positive_rate="unsupported",
                    quality_score_proxy_mean="unsupported",
                    score_retention_mean="unsupported",
                    supports_paper_claim=False,
                ).to_dict()
            )
    return tuple(rows)


def aggregate_baseline_metrics(observations: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """把 baseline 观测记录聚合为每个 baseline 一行的指标摘要。"""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for observation in observations:
        grouped.setdefault(str(observation["baseline_id"]), []).append(observation)
    rows: list[dict[str, Any]] = []
    for baseline_id, group in sorted(grouped.items()):
        first = group[0]
        ready_count = sum(1 for item in group if item["metric_status"] != "unsupported")
        compatible_count = sum(1 for item in group if item["baseline_protocol_compatible"])
        rows.append(
            {
                "baseline_id": baseline_id,
                "baseline_family": first["baseline_family"],
                "baseline_name": first["baseline_name"],
                "comparison_group": first["comparison_group"],
                "baseline_adapter_ready": first["baseline_adapter_ready"],
                "baseline_official_code_ready": first["baseline_official_code_ready"],
                "baseline_reproduced_result_ready": first["baseline_reproduced_result_ready"],
                "baseline_imported_result_ready": first["baseline_imported_result_ready"],
                "baseline_result_source": first["baseline_result_source"],
                "baseline_protocol_compatible": compatible_count == len(group),
                "baseline_requires_gpu": first["baseline_requires_gpu"],
                "baseline_requires_training": first["baseline_requires_training"],
                "baseline_observation_count": len(group),
                "baseline_result_ready_count": ready_count,
                "unsupported_record_count": len(group) - ready_count,
                "metric_status": "measured" if ready_count else "unsupported",
                "unsupported_reason": "" if ready_count else first["unsupported_reason"],
                "supports_paper_claim": False,
            }
        )
    return rows


def aggregate_slm_proxy_metrics(attack_metric_rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """聚合当前方法在攻击矩阵中的本地代理统计。"""
    rows = tuple(attack_metric_rows)
    supported_rows = tuple(row for row in rows if row.get("metric_status") != "unsupported")
    return {
        "method_id": "slm_wm_current",
        "method_role": "proposed_method_local_proxy",
        "comparison_scope": "attack_matrix_local_proxy",
        "common_prompt_protocol_ready": True,
        "common_attack_protocol_ready": True,
        "common_threshold_protocol_ready": True,
        "metric_status": "measured_from_local_proxy" if supported_rows else "unsupported",
        "true_positive_rate": _weighted_mean(supported_rows, "true_positive_rate", "positive_count"),
        "false_positive_rate": _weighted_mean(supported_rows, "false_positive_rate", "negative_count"),
        "clean_false_positive_rate": _weighted_mean(supported_rows, "clean_false_positive_rate", "negative_count"),
        "attacked_false_positive_rate": _weighted_mean(supported_rows, "attacked_false_positive_rate", "negative_count"),
        "quality_score_proxy_mean": _weighted_mean(supported_rows, "quality_score_proxy_mean", "supported_record_count"),
        "score_retention_mean": _weighted_mean(supported_rows, "score_retention_mean", "supported_record_count"),
        "supports_paper_claim": False,
    }


def build_comparison_rows(
    slm_proxy_metrics: dict[str, Any],
    baseline_metric_rows: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """构造 SLM-WM 与外部 baseline 的同协议对比表。"""
    rows = [slm_proxy_metrics]
    for baseline_row in baseline_metric_rows:
        rows.append(
            {
                "method_id": baseline_row["baseline_id"],
                "method_role": f"external_baseline_{baseline_row['comparison_group']}",
                "comparison_scope": "common_protocol_without_external_result",
                "common_prompt_protocol_ready": True,
                "common_attack_protocol_ready": True,
                "common_threshold_protocol_ready": True,
                "metric_status": baseline_row["metric_status"],
                "true_positive_rate": "unsupported",
                "false_positive_rate": "unsupported",
                "clean_false_positive_rate": "unsupported",
                "attacked_false_positive_rate": "unsupported",
                "quality_score_proxy_mean": "unsupported",
                "score_retention_mean": "unsupported",
                "supports_paper_claim": False,
            }
        )
    return rows
