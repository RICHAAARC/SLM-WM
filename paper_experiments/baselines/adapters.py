"""外部 baseline 的共同协议接入与受治理结果导入。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
import json
from typing import Any, Iterable, Mapping

from experiments.protocol.pilot_paper_fixed_fpr import PILOT_PAPER_FIXED_FPR
from main.core.digest import build_stable_digest


@dataclass(frozen=True)
class BaselineSpec:
    """描述一个外部 baseline 的协议身份和接入状态。

    该对象属于通用工程写法: 把外部方法的身份, 源码缓存状态, 复现结果状态和导入结果状态集中到
    dataclass 构造边界中, 让后续观测记录构建函数只关注共同协议下的数据流。
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
    official_repository_url: str = ""
    official_repository_commit: str = ""
    official_repository_branch: str = ""
    source_dir: str = ""
    source_status: str = "not_registered"

    def __post_init__(self) -> None:
        """集中校验 baseline 配置的不可恢复边界。"""
        if self.comparison_group not in {"primary", "supplemental"}:
            raise ValueError("comparison_group 必须是 primary 或 supplemental")
        if not self.baseline_id or not self.baseline_family or not self.baseline_name:
            raise ValueError("baseline 身份字段不得为空")
        result_ready = self.baseline_reproduced_result_ready or self.baseline_imported_result_ready
        if result_ready and self.unsupported_reason:
            raise ValueError("已有 baseline 结果时不得同时携带 unsupported_reason")

    def to_dict(self) -> dict[str, Any]:
        """转换为 JSON 兼容字典。"""
        return asdict(self)


@dataclass(frozen=True)
class BaselineResultRecord:
    """记录外部 baseline 在共同协议边界下的一条可导入指标结果。

    该结构只描述已经复现或从受治理来源导入的指标, 不直接调用第三方源码, 也不把手工表格伪装成论文证据。
    """

    baseline_result_record_id: str
    baseline_result_digest: str
    baseline_id: str
    attack_family: str
    attack_name: str
    resource_profile: str
    comparable_operating_point: str
    result_protocol_name: str
    result_source_type: str
    baseline_result_source: str
    baseline_result_source_digest: str
    metric_status: str
    positive_count: int
    negative_count: int
    attack_record_count: int
    supported_record_count: int
    true_positive_rate: float
    false_positive_rate: float
    clean_false_positive_rate: float
    attacked_false_positive_rate: float
    quality_score_mean: float
    score_retention_mean: float
    supports_paper_claim: bool = False

    def to_dict(self) -> dict[str, Any]:
        """转换为 JSON 兼容字典。"""
        return asdict(self)


@dataclass(frozen=True)
class BaselineObservation:
    """记录外部 baseline 在共同协议边界下的一条可审计观测。

    当外部方法尚无复现结果或导入结果时, 该记录保留 unsupported 状态; 当结果已导入时, 该记录沿用共同
    prompt, attack 和 fixed-FPR 边界下的指标字段, 供下游表格重建使用。
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
    baseline_result_source_digest: str
    metric_status: str
    unsupported_reason: str
    positive_count: int
    negative_count: int
    attack_record_count: int
    supported_record_count: int
    true_positive_rate: str | float
    false_positive_rate: str | float
    clean_false_positive_rate: str | float
    attacked_false_positive_rate: str | float
    quality_score_mean: str | float
    score_retention_mean: str | float
    supports_paper_claim: bool

    def to_dict(self) -> dict[str, Any]:
        """转换为 JSON 兼容字典。"""
        return asdict(self)


def default_baseline_specs() -> tuple[BaselineSpec, ...]:
    """返回默认外部 baseline 对比清单。

    这里实现的是共同协议 adapter, 不是外部方法本体。外部官方源码, 复现实验结果或导入结果未完成前,
    每个 baseline 只能写出 claim-safe 的 unsupported 观测记录。
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


def load_baseline_source_registry(path: str | Path | None) -> dict[str, Any]:
    """读取外部 baseline 源码登记文件; 文件缺失时返回空登记。"""
    if path is None:
        return {}
    registry_path = Path(path)
    if not registry_path.exists():
        return {}
    return json.loads(registry_path.read_text(encoding="utf-8"))


def overlay_specs_with_source_registry(
    baseline_specs: Iterable[BaselineSpec],
    source_registry: Mapping[str, Any],
    root: str | Path = ".",
) -> tuple[BaselineSpec, ...]:
    """把源码登记状态合并到 baseline spec 中。

    此处属于项目特定写法: 第三方源码保存在被忽略的缓存目录中, 因此正式仓库只记录来源摘要和本地检查状态,
    不把第三方源码纳入本项目实现。
    """
    root_path = Path(root).resolve()
    source_items = {item["baseline_id"]: item for item in source_registry.get("baseline_sources", ())}
    merged: list[BaselineSpec] = []
    for spec in baseline_specs:
        item = source_items.get(spec.baseline_id)
        if not item:
            merged.append(spec)
            continue
        source_dir = str(item.get("source_dir", ""))
        source_path = root_path / source_dir if source_dir else None
        source_status = str(item.get("source_status", "not_registered"))
        source_cache_present = bool(source_path and source_path.exists())
        official_code_ready = source_status == "downloaded" and source_cache_present
        merged.append(
            replace(
                spec,
                baseline_official_code_ready=official_code_ready,
                official_repository_url=str(item.get("official_repository_url", "")),
                official_repository_commit=str(item.get("official_repository_commit", "")),
                official_repository_branch=str(item.get("official_repository_branch", "")),
                source_dir=source_dir,
                source_status=source_status,
            )
        )
    return tuple(merged)


def _int_field(row: Mapping[str, Any], field_name: str) -> int:
    """读取行中的整数指标。"""
    return int(float(row.get(field_name, 0) or 0))


def _float_field(row: Mapping[str, Any], field_name: str) -> float:
    """读取行中的浮点指标。"""
    return float(row.get(field_name, 0.0) or 0.0)


def _str_field(row: Mapping[str, Any], field_name: str) -> str:
    """读取行中的字符串字段。"""
    return str(row.get(field_name, "") or "")


def _weighted_mean(rows: Iterable[Mapping[str, Any]], value_field: str, weight_field: str) -> float:
    """按给定计数字段计算加权均值。"""
    total_weight = 0
    weighted_sum = 0.0
    for row in rows:
        weight = _int_field(row, weight_field)
        total_weight += weight
        weighted_sum += _float_field(row, value_field) * weight
    return weighted_sum / total_weight if total_weight else 0.0


def _operating_point(boundary: Mapping[str, Any]) -> str:
    """根据 fixed-FPR 边界生成可读 operating point 名称。"""
    target_fpr = float(boundary.get("target_fpr", PILOT_PAPER_FIXED_FPR))
    return f"fixed_fpr_{target_fpr:g}"


def _baseline_protocol_compatible(spec: BaselineSpec, attack_row: Mapping[str, Any]) -> bool:
    """判断 baseline 是否可登记到共同 prompt, 攻击和阈值协议下。"""
    return bool(spec.baseline_adapter_ready and attack_row.get("attack_name") and attack_row.get("resource_profile"))


def normalize_baseline_result_record(row: Mapping[str, Any]) -> BaselineResultRecord:
    """把受治理导入行规范化为稳定的 baseline 结果记录。"""
    payload = {
        "baseline_id": _str_field(row, "baseline_id"),
        "attack_family": _str_field(row, "attack_family"),
        "attack_name": _str_field(row, "attack_name"),
        "resource_profile": _str_field(row, "resource_profile"),
        "comparable_operating_point": _str_field(row, "comparable_operating_point"),
        "result_protocol_name": _str_field(row, "result_protocol_name") or "common_watermark_protocol",
        "result_source_type": _str_field(row, "result_source_type") or "governed_import",
        "baseline_result_source": _str_field(row, "baseline_result_source"),
        "baseline_result_source_digest": _str_field(row, "baseline_result_source_digest"),
        "metric_status": _str_field(row, "metric_status") or "measured",
        "positive_count": _int_field(row, "positive_count"),
        "negative_count": _int_field(row, "negative_count"),
        "attack_record_count": _int_field(row, "attack_record_count"),
        "supported_record_count": _int_field(row, "supported_record_count"),
        "true_positive_rate": _float_field(row, "true_positive_rate"),
        "false_positive_rate": _float_field(row, "false_positive_rate"),
        "clean_false_positive_rate": _float_field(row, "clean_false_positive_rate"),
        "attacked_false_positive_rate": _float_field(row, "attacked_false_positive_rate"),
        "quality_score_mean": _float_field(row, "quality_score_mean"),
        "score_retention_mean": _float_field(row, "score_retention_mean"),
    }
    digest = build_stable_digest(payload)
    return BaselineResultRecord(
        baseline_result_record_id=_str_field(row, "baseline_result_record_id") or f"baseline_result_{digest[:16]}",
        baseline_result_digest=_str_field(row, "baseline_result_digest") or digest,
        supports_paper_claim=bool(row.get("supports_paper_claim", False)),
        **payload,
    )


def build_baseline_result_index(
    records: Iterable[Mapping[str, Any] | BaselineResultRecord],
) -> dict[tuple[str, str, str, str, str], BaselineResultRecord]:
    """把 baseline 结果记录索引到共同协议键上。"""
    index: dict[tuple[str, str, str, str, str], BaselineResultRecord] = {}
    for item in records:
        record = item if isinstance(item, BaselineResultRecord) else normalize_baseline_result_record(item)
        key = (
            record.baseline_id,
            record.attack_family,
            record.attack_name,
            record.resource_profile,
            record.comparable_operating_point,
        )
        index[key] = record
    return index


def build_baseline_observations(
    baseline_specs: Iterable[BaselineSpec],
    attack_metric_rows: Iterable[Mapping[str, Any]],
    boundary: Mapping[str, Any],
    baseline_result_records: Iterable[Mapping[str, Any] | BaselineResultRecord] = (),
) -> tuple[dict[str, Any], ...]:
    """基于攻击矩阵行构造外部 baseline 观测记录。"""
    rows: list[dict[str, Any]] = []
    comparable_operating_point = _operating_point(boundary)
    result_index = build_baseline_result_index(baseline_result_records)
    for spec in baseline_specs:
        for attack_row in attack_metric_rows:
            payload = {
                "baseline_id": spec.baseline_id,
                "attack_family": _str_field(attack_row, "attack_family"),
                "attack_name": _str_field(attack_row, "attack_name"),
                "resource_profile": _str_field(attack_row, "resource_profile"),
                "comparable_operating_point": comparable_operating_point,
            }
            result_key = (
                payload["baseline_id"],
                payload["attack_family"],
                payload["attack_name"],
                payload["resource_profile"],
                payload["comparable_operating_point"],
            )
            result_record = result_index.get(result_key)
            protocol_compatible = _baseline_protocol_compatible(spec, attack_row)
            result_ready = result_record is not None and result_record.metric_status != "unsupported"
            digest = build_stable_digest({**spec.to_dict(), **payload, "result_ready": result_ready})
            rows.append(
                BaselineObservation(
                    baseline_observation_id=f"baseline_observation_{digest[:16]}",
                    baseline_observation_digest=digest,
                    baseline_id=spec.baseline_id,
                    baseline_family=spec.baseline_family,
                    baseline_name=spec.baseline_name,
                    comparison_group=spec.comparison_group,
                    expected_input_mode=spec.expected_input_mode,
                    attack_family=payload["attack_family"],
                    attack_name=payload["attack_name"],
                    resource_profile=payload["resource_profile"],
                    comparable_operating_point=comparable_operating_point,
                    common_prompt_protocol_ready=True,
                    common_attack_protocol_ready=True,
                    common_threshold_protocol_ready=True,
                    baseline_protocol_compatible=protocol_compatible,
                    baseline_requires_gpu=spec.requires_gpu,
                    baseline_requires_training=spec.requires_training,
                    baseline_adapter_ready=spec.baseline_adapter_ready,
                    baseline_official_code_ready=spec.baseline_official_code_ready,
                    baseline_reproduced_result_ready=result_ready
                    and result_record.result_source_type == "official_reproduction",
                    baseline_imported_result_ready=result_ready and result_record.result_source_type != "official_reproduction",
                    baseline_result_source=result_record.baseline_result_source if result_ready else spec.baseline_result_source,
                    baseline_result_source_digest=result_record.baseline_result_source_digest if result_ready else "",
                    metric_status=result_record.metric_status if result_ready else "unsupported",
                    unsupported_reason="" if result_ready else spec.unsupported_reason,
                    positive_count=result_record.positive_count if result_ready else _int_field(attack_row, "positive_count"),
                    negative_count=result_record.negative_count if result_ready else _int_field(attack_row, "negative_count"),
                    attack_record_count=result_record.attack_record_count
                    if result_ready
                    else _int_field(attack_row, "attack_record_count"),
                    supported_record_count=result_record.supported_record_count if result_ready else 0,
                    true_positive_rate=result_record.true_positive_rate if result_ready else "unsupported",
                    false_positive_rate=result_record.false_positive_rate if result_ready else "unsupported",
                    clean_false_positive_rate=result_record.clean_false_positive_rate if result_ready else "unsupported",
                    attacked_false_positive_rate=result_record.attacked_false_positive_rate if result_ready else "unsupported",
                    quality_score_mean=result_record.quality_score_mean if result_ready else "unsupported",
                    score_retention_mean=result_record.score_retention_mean if result_ready else "unsupported",
                    supports_paper_claim=bool(result_record.supports_paper_claim) if result_ready else False,
                ).to_dict()
            )
    return tuple(rows)


def aggregate_baseline_metrics(observations: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """把 baseline 观测记录聚合为每个 baseline 一行的指标摘要。"""
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for observation in observations:
        grouped.setdefault(str(observation["baseline_id"]), []).append(observation)
    rows: list[dict[str, Any]] = []
    for baseline_id, group in sorted(grouped.items()):
        first = group[0]
        ready_rows = [item for item in group if item["metric_status"] != "unsupported"]
        ready_count = len(ready_rows)
        compatible_count = sum(1 for item in group if item["baseline_protocol_compatible"])
        rows.append(
            {
                "baseline_id": baseline_id,
                "baseline_family": first["baseline_family"],
                "baseline_name": first["baseline_name"],
                "comparison_group": first["comparison_group"],
                "baseline_adapter_ready": first["baseline_adapter_ready"],
                "baseline_official_code_ready": first["baseline_official_code_ready"],
                "baseline_reproduced_result_ready": any(item["baseline_reproduced_result_ready"] for item in group),
                "baseline_imported_result_ready": any(item["baseline_imported_result_ready"] for item in group),
                "baseline_result_source": ";".join(
                    sorted({str(item["baseline_result_source"]) for item in ready_rows if item["baseline_result_source"]})
                )
                if ready_rows
                else first["baseline_result_source"],
                "baseline_protocol_compatible": compatible_count == len(group),
                "baseline_requires_gpu": first["baseline_requires_gpu"],
                "baseline_requires_training": first["baseline_requires_training"],
                "baseline_observation_count": len(group),
                "baseline_result_ready_count": ready_count,
                "unsupported_record_count": len(group) - ready_count,
                "metric_status": "measured" if ready_count else "unsupported",
                "true_positive_rate": _weighted_mean(ready_rows, "true_positive_rate", "positive_count")
                if ready_rows
                else "unsupported",
                "false_positive_rate": _weighted_mean(ready_rows, "false_positive_rate", "negative_count")
                if ready_rows
                else "unsupported",
                "clean_false_positive_rate": _weighted_mean(ready_rows, "clean_false_positive_rate", "negative_count")
                if ready_rows
                else "unsupported",
                "attacked_false_positive_rate": _weighted_mean(
                    ready_rows,
                    "attacked_false_positive_rate",
                    "negative_count",
                )
                if ready_rows
                else "unsupported",
                "quality_score_mean": _weighted_mean(
                    ready_rows,
                    "quality_score_mean",
                    "supported_record_count",
                )
                if ready_rows
                else "unsupported",
                "score_retention_mean": _weighted_mean(ready_rows, "score_retention_mean", "supported_record_count")
                if ready_rows
                else "unsupported",
                "unsupported_reason": "" if ready_count else first["unsupported_reason"],
                "supports_paper_claim": bool(ready_rows) and all(bool(item["supports_paper_claim"]) for item in ready_rows),
            }
        )
    return rows


def aggregate_slm_metrics(attack_metric_rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """聚合当前方法在共同真实图像攻击矩阵中的正式统计。"""
    rows = tuple(attack_metric_rows)
    supported_rows = tuple(row for row in rows if row.get("metric_status") != "unsupported")
    return {
        "method_id": "slm_wm_current",
        "method_role": "proposed_method",
        "comparison_scope": "common_protocol_real_image_detection",
        "common_prompt_protocol_ready": True,
        "common_attack_protocol_ready": True,
        "common_threshold_protocol_ready": True,
        "metric_status": "measured_real_attacked_image_image_only_detection" if supported_rows else "unsupported",
        "true_positive_rate": _weighted_mean(supported_rows, "true_positive_rate", "positive_count"),
        "false_positive_rate": _weighted_mean(supported_rows, "false_positive_rate", "negative_count"),
        "clean_false_positive_rate": _weighted_mean(supported_rows, "clean_false_positive_rate", "negative_count"),
        "attacked_false_positive_rate": _weighted_mean(
            supported_rows,
            "attacked_false_positive_rate",
            "negative_count",
        ),
        "quality_score_mean": _weighted_mean(supported_rows, "quality_score_mean", "supported_record_count"),
        "score_retention_mean": _weighted_mean(supported_rows, "score_retention_mean", "supported_record_count"),
        "supports_paper_claim": bool(supported_rows) and all(bool(row.get("supports_paper_claim")) for row in supported_rows),
    }


def build_comparison_rows(
    slm_metrics: Mapping[str, Any],
    baseline_metric_rows: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """构造 SLM-WM 与外部 baseline 的同协议对比表。"""
    rows = [dict(slm_metrics)]
    for baseline_row in baseline_metric_rows:
        measured = baseline_row["metric_status"] != "unsupported"
        rows.append(
            {
                "method_id": baseline_row["baseline_id"],
                "method_role": f"external_baseline_{baseline_row['comparison_group']}",
                "comparison_scope": "common_protocol_governed_result" if measured else "common_protocol_result_missing",
                "common_prompt_protocol_ready": True,
                "common_attack_protocol_ready": True,
                "common_threshold_protocol_ready": True,
                "metric_status": baseline_row["metric_status"],
                "true_positive_rate": baseline_row["true_positive_rate"] if measured else "unsupported",
                "false_positive_rate": baseline_row["false_positive_rate"] if measured else "unsupported",
                "clean_false_positive_rate": baseline_row["clean_false_positive_rate"] if measured else "unsupported",
                "attacked_false_positive_rate": baseline_row["attacked_false_positive_rate"] if measured else "unsupported",
                "quality_score_mean": baseline_row["quality_score_mean"] if measured else "unsupported",
                "score_retention_mean": baseline_row["score_retention_mean"] if measured else "unsupported",
                "supports_paper_claim": bool(baseline_row.get("supports_paper_claim", False)) if measured else False,
            }
        )
    return rows
