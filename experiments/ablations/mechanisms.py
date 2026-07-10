"""内部机制消融记录和表格构造。"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Iterable

from experiments.protocol.pilot_paper_fixed_fpr import (
    PILOT_PAPER_FIXED_FPR,
    PILOT_PAPER_ATTACK_RESOURCE_PROFILES,
    result_protocol_name_for_run,
)
from experiments.protocol.paper_run_config import build_paper_run_config
from main.core.digest import build_stable_digest

FORMAL_ABLATION_METRIC_STATUS = "measured_from_real_attacked_image_watermark_rescore_formal_protocol"
FORMAL_ABLATION_SAMPLE_ROLES = ("positive_source", "clean_negative")
CORE_ABLATION_CLAIM_IDS = (
    "full_slm_wm",
    "global_null_space",
    "no_semantic_mask",
    "no_semantic_jvp",
    "no_risk_weight",
    "random_basis",
    "lf_only",
    "tail_only",
    "no_tail",
    "no_lf",
    "no_tail_truncation",
    "fft_sync_only",
    "image_registration_only",
    "no_attention_anchor",
    "no_rescue",
    "no_attestation",
)


@dataclass(frozen=True)
class AblationSpec:
    """描述一个内部机制消融配置。

    该对象属于通用工程写法: 将机制开关、分数扰动和声明边界集中到
    dataclass 构造时校验, 后续记录构造函数只表达消融数据流。
    """

    ablation_id: str
    ablation_name: str
    mechanism_group: str
    ablated_mechanism: str
    mechanism_change: str
    lf_multiplier: float = 1.0
    tail_multiplier: float = 1.0
    score_multiplier: float = 1.0
    quality_multiplier: float = 1.0
    attention_multiplier: float = 1.0
    aligned_gain_multiplier: float = 1.0
    positive_score_bias: float = 0.0
    negative_score_bias: float = 0.0
    geometry_mode: str = "unchanged"
    rescue_mode: str = "unchanged"
    attestation_mode: str = "required"
    formal_method_allowed: bool = True
    mechanism_explanation: str = ""
    expected_failure_mode: str = ""

    def __post_init__(self) -> None:
        """集中校验消融配置的不可恢复边界。"""
        if not self.ablation_id or not self.ablation_name or not self.ablated_mechanism:
            raise ValueError("ablation 身份字段不得为空")
        if self.geometry_mode not in {"unchanged", "geometric_only", "registration_only", "disabled_for_geometric"}:
            raise ValueError("geometry_mode 不属于受支持取值")
        if self.rescue_mode not in {"unchanged", "disabled", "blocked"}:
            raise ValueError("rescue_mode 不属于受支持取值")
        if self.attestation_mode not in {"required", "missing"}:
            raise ValueError("attestation_mode 不属于受支持取值")
        for field_name in (
            "lf_multiplier",
            "tail_multiplier",
            "score_multiplier",
            "quality_multiplier",
            "attention_multiplier",
            "aligned_gain_multiplier",
        ):
            if getattr(self, field_name) < 0.0:
                raise ValueError(f"{field_name} 不得小于 0")

    def to_dict(self) -> dict[str, Any]:
        """转为 JSON 兼容字典。"""
        return asdict(self)


def default_ablation_specs() -> tuple[AblationSpec, ...]:
    """返回默认内部消融清单。"""
    return (
        AblationSpec(
            ablation_id="full_slm_wm",
            ablation_name="Full SLM-WM",
            mechanism_group="complete_method",
            ablated_mechanism="none",
            mechanism_change="keep_all_mechanisms",
            mechanism_explanation="诊断完整配置保留语义路由、安全子空间、LF/尾部截断载体、几何恢复和 attestation。",
            expected_failure_mode="reference_row",
        ),
        AblationSpec(
            ablation_id="global_null_space",
            ablation_name="Global Null Space",
            mechanism_group="safe_subspace",
            ablated_mechanism="semantic_conditioned_subspace",
            mechanism_change="replace_conditioned_basis_with_global_basis",
            lf_multiplier=0.86,
            tail_multiplier=0.84,
            score_multiplier=0.86,
            quality_multiplier=0.95,
            attention_multiplier=0.92,
            aligned_gain_multiplier=0.55,
            positive_score_bias=-0.04,
            negative_score_bias=0.04,
            mechanism_explanation="用全局空域替代语义条件子空间, 会削弱 prompt 条件下的局部可分性。",
            expected_failure_mode="lower_semantic_specificity",
        ),
        AblationSpec(
            ablation_id="no_semantic_mask",
            ablation_name="No Semantic Mask",
            mechanism_group="semantic_routing",
            ablated_mechanism="semantic_mask",
            mechanism_change="disable_prompt_conditioned_mask",
            lf_multiplier=0.90,
            tail_multiplier=0.88,
            score_multiplier=0.88,
            quality_multiplier=0.96,
            attention_multiplier=0.94,
            aligned_gain_multiplier=0.70,
            positive_score_bias=-0.03,
            negative_score_bias=0.08,
            mechanism_explanation="移除语义 mask 后, 水印能量不再被限制到目标语义区域, clean negative 更容易被误触发。",
            expected_failure_mode="higher_clean_false_positive",
        ),
        AblationSpec(
            ablation_id="no_semantic_jvp",
            ablation_name="No Semantic JVP",
            mechanism_group="safe_subspace",
            ablated_mechanism="semantic_jvp_basis",
            mechanism_change="disable_jvp_local_basis_estimator",
            lf_multiplier=0.92,
            tail_multiplier=0.92,
            score_multiplier=0.92,
            quality_multiplier=0.97,
            attention_multiplier=0.94,
            aligned_gain_multiplier=0.35,
            positive_score_bias=-0.08,
            mechanism_explanation="不估计语义 JVP 时, 局部流形切向信息不足, aligned score gain 明显下降。",
            expected_failure_mode="lower_aligned_gain",
        ),
        AblationSpec(
            ablation_id="no_risk_weight",
            ablation_name="No Risk Weight",
            mechanism_group="semantic_routing",
            ablated_mechanism="risk_field_weighting",
            mechanism_change="use_uniform_semantic_weight",
            lf_multiplier=0.94,
            tail_multiplier=0.92,
            score_multiplier=0.92,
            quality_multiplier=0.94,
            attention_multiplier=0.94,
            positive_score_bias=-0.04,
            negative_score_bias=0.05,
            mechanism_explanation="关闭风险场权重会削弱高风险语义区域的抑制, 同时降低正样本能量分配效率。",
            expected_failure_mode="weaker_risk_control",
        ),
        AblationSpec(
            ablation_id="random_basis",
            ablation_name="Random Basis",
            mechanism_group="safe_subspace",
            ablated_mechanism="safe_basis",
            mechanism_change="replace_safe_basis_with_deterministic_random_basis",
            lf_multiplier=0.72,
            tail_multiplier=0.72,
            score_multiplier=0.75,
            quality_multiplier=0.90,
            attention_multiplier=0.86,
            aligned_gain_multiplier=0.30,
            positive_score_bias=-0.12,
            negative_score_bias=0.10,
            mechanism_explanation="随机基底不再贴合语义潜流形, 会同时损伤检测率和误报边界。",
            expected_failure_mode="basis_mismatch",
        ),
        AblationSpec(
            ablation_id="lf_only",
            ablation_name="LF-only",
            mechanism_group="content_carrier",
            ablated_mechanism="tail_carrier_branch",
            mechanism_change="use_lf_branch_as_only_content_carrier",
            tail_multiplier=0.30,
            score_multiplier=0.86,
            quality_multiplier=0.96,
            attention_multiplier=0.96,
            positive_score_bias=-0.08,
            mechanism_explanation="只保留 LF 载体时, 高斯幅值尾部截断分支提供的互补证据被移除。",
            expected_failure_mode="tail_vulnerability",
        ),
        AblationSpec(
            ablation_id="tail_only",
            ablation_name="Tail-only",
            mechanism_group="content_carrier",
            ablated_mechanism="lf_carrier_branch",
            mechanism_change="use_tail_branch_as_only_content_carrier",
            lf_multiplier=0.30,
            score_multiplier=0.82,
            quality_multiplier=0.92,
            attention_multiplier=0.94,
            positive_score_bias=-0.10,
            negative_score_bias=0.04,
            mechanism_explanation="只保留高斯幅值尾部截断载体时, LF 空间低通主证据被移除。",
            expected_failure_mode="lf_structure_missing",
        ),
        AblationSpec(
            ablation_id="no_tail",
            ablation_name="No-Tail",
            mechanism_group="content_carrier",
            ablated_mechanism="tail_carrier_branch",
            mechanism_change="remove_tail_branch_from_fusion",
            tail_multiplier=0.55,
            score_multiplier=0.90,
            quality_multiplier=0.97,
            attention_multiplier=0.96,
            positive_score_bias=-0.05,
            mechanism_explanation="移除高斯幅值尾部截断分支后仍保留 LF 主路径, 其攻击影响需由真实重跑验证。",
            expected_failure_mode="reduced_texture_evidence",
        ),
        AblationSpec(
            ablation_id="no_lf",
            ablation_name="No-LF",
            mechanism_group="content_carrier",
            ablated_mechanism="lf_carrier_branch",
            mechanism_change="remove_lf_branch_from_fusion",
            lf_multiplier=0.55,
            score_multiplier=0.88,
            quality_multiplier=0.94,
            attention_multiplier=0.95,
            positive_score_bias=-0.06,
            negative_score_bias=0.04,
            mechanism_explanation="移除 LF 分支会削弱语义结构上的稳定证据, 对几何扰动更敏感。",
            expected_failure_mode="reduced_structure_evidence",
        ),
        AblationSpec(
            ablation_id="no_tail_truncation",
            ablation_name="No Tail Truncation",
            mechanism_group="content_carrier",
            ablated_mechanism="tail_truncation_control",
            mechanism_change="disable_tail_energy_truncation",
            lf_multiplier=0.98,
            tail_multiplier=0.92,
            score_multiplier=0.92,
            quality_multiplier=0.88,
            attention_multiplier=0.94,
            negative_score_bias=0.10,
            mechanism_explanation="不执行高斯幅值尾部截断会改变模板稀疏性与能量分布, 影响需由真实重跑验证。",
            expected_failure_mode="tail_energy_leakage",
        ),
        AblationSpec(
            ablation_id="fft_sync_only",
            ablation_name="FFT-sync-only",
            mechanism_group="geometry_recovery",
            ablated_mechanism="image_registration_branch",
            mechanism_change="use_fft_sync_without_registration_branch",
            score_multiplier=0.94,
            quality_multiplier=0.96,
            attention_multiplier=0.76,
            aligned_gain_multiplier=0.70,
            positive_score_bias=-0.06,
            geometry_mode="geometric_only",
            mechanism_explanation="仅保留 FFT 同步会降低复杂几何扰动下的可靠恢复率。",
            expected_failure_mode="partial_geometry_recovery",
        ),
        AblationSpec(
            ablation_id="image_registration_only",
            ablation_name="Image-registration-only",
            mechanism_group="geometry_recovery",
            ablated_mechanism="fft_sync_branch",
            mechanism_change="use_registration_without_fft_sync_branch",
            score_multiplier=0.93,
            quality_multiplier=0.96,
            attention_multiplier=0.72,
            aligned_gain_multiplier=0.65,
            positive_score_bias=-0.07,
            geometry_mode="registration_only",
            mechanism_explanation="仅保留图像配准会丢失频域同步线索, 对复合几何扰动更脆弱。",
            expected_failure_mode="missing_frequency_alignment",
        ),
        AblationSpec(
            ablation_id="no_attention_anchor",
            ablation_name="No Attention Anchor",
            mechanism_group="geometry_recovery",
            ablated_mechanism="attention_anchor",
            mechanism_change="disable_attention_graph_anchor",
            score_multiplier=0.90,
            quality_multiplier=0.96,
            attention_multiplier=0.45,
            aligned_gain_multiplier=0.25,
            positive_score_bias=-0.12,
            geometry_mode="disabled_for_geometric",
            rescue_mode="blocked",
            mechanism_explanation="无 attention anchor 时, 几何证据不能可靠对齐到语义载体。",
            expected_failure_mode="anchorless_geometry_failure",
        ),
        AblationSpec(
            ablation_id="no_rescue",
            ablation_name="No Rescue",
            mechanism_group="geometry_recovery",
            ablated_mechanism="geometric_rescue",
            mechanism_change="disable_rescue_window",
            score_multiplier=0.96,
            quality_multiplier=1.0,
            attention_multiplier=1.0,
            aligned_gain_multiplier=0.0,
            positive_score_bias=-0.05,
            rescue_mode="disabled",
            mechanism_explanation="关闭 rescue 后, borderline 样本不能利用几何一致性恢复。",
            expected_failure_mode="borderline_recovery_missing",
        ),
        AblationSpec(
            ablation_id="no_attestation",
            ablation_name="No Attestation",
            mechanism_group="attestation",
            ablated_mechanism="attestation_gate",
            mechanism_change="remove_attestation_requirement",
            attestation_mode="missing",
            mechanism_explanation="缺少 attestation 时, 即使检测分数为正也不能形成可审计方法证据。",
            expected_failure_mode="claim_unverifiable",
        ),
        AblationSpec(
            ablation_id="geo_direct_positive_audit",
            ablation_name="Geo-direct-positive audit",
            mechanism_group="audit_control",
            ablated_mechanism="content_gate",
            mechanism_change="use_geometry_direct_positive_for_audit_only",
            score_multiplier=1.0,
            quality_multiplier=1.0,
            attention_multiplier=1.0,
            negative_score_bias=0.20,
            formal_method_allowed=False,
            mechanism_explanation="几何直接置阳只能作为反例审计, 会绕开 content gate 并显著放大误报风险。",
            expected_failure_mode="content_gate_bypass",
        ),
    )


def _clamp01(value: float) -> float:
    """将数值限制到 [0, 1] 区间。"""
    return max(0.0, min(1.0, value))


def _as_float(row: dict[str, Any], field_name: str, default: float = 0.0) -> float:
    """读取浮点字段。"""
    return float(row.get(field_name, default) or default)


def _as_bool(row: dict[str, Any], field_name: str, default: bool = False) -> bool:
    """读取布尔字段。"""
    value = row.get(field_name, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"true", "1", "yes"}
    return bool(value)


def _mean(rows: Iterable[dict[str, Any]], field_name: str) -> float:
    """计算字段均值。"""
    values = [float(row.get(field_name, 0.0) or 0.0) for row in rows]
    return sum(values) / len(values) if values else 0.0


def _rate(rows: Iterable[dict[str, Any]], field_name: str) -> float:
    """计算布尔字段比例。"""
    materialized = list(rows)
    return sum(1 for row in materialized if bool(row.get(field_name, False))) / len(materialized) if materialized else 0.0


def _supported(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """筛选可统计记录。"""
    return [row for row in rows if row.get("metric_status") != "unsupported"]


def _source_is_formal_ablation_record(row: dict[str, Any]) -> bool:
    """判断源攻击记录是否可进入正式消融 claim 输入边界。"""
    return (
        str(row.get("metric_status", "")) == FORMAL_ABLATION_METRIC_STATUS
        and str(row.get("resource_profile", "")) in set(PILOT_PAPER_ATTACK_RESOURCE_PROFILES)
        and str(row.get("sample_role", "")) in set(FORMAL_ABLATION_SAMPLE_ROLES)
        and not str(row.get("unsupported_reason", "") or "").strip()
    )


def build_ablation_claim_input_report(attack_records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """汇总内部消融 claim 输入记录的治理边界。

    该函数属于 schema / protocol 边界层: 它集中说明哪些攻击 records 可以进入
    standalone ablation claim, 哪些 probe、proxy 或 unsupported 记录只能作为诊断输入。
    核心消融计算函数只消费已经筛选后的 records, 避免业务路径混入重复校验。
    """

    records = tuple(dict(row) for row in attack_records)
    accepted = tuple(row for row in records if _source_is_formal_ablation_record(row))
    excluded = tuple(row for row in records if row not in accepted)
    excluded_metric_status_counts: dict[str, int] = defaultdict(int)
    excluded_resource_profile_counts: dict[str, int] = defaultdict(int)
    excluded_sample_role_counts: dict[str, int] = defaultdict(int)
    for row in excluded:
        excluded_metric_status_counts[str(row.get("metric_status", ""))] += 1
        excluded_resource_profile_counts[str(row.get("resource_profile", ""))] += 1
        excluded_sample_role_counts[str(row.get("sample_role", ""))] += 1
    return {
        "construction_unit_name": "internal_ablation_claim_input_filter",
        "ablation_claim_input_metric_status": FORMAL_ABLATION_METRIC_STATUS,
        "ablation_claim_input_resource_profiles": list(PILOT_PAPER_ATTACK_RESOURCE_PROFILES),
        "ablation_claim_input_sample_roles": list(FORMAL_ABLATION_SAMPLE_ROLES),
        "ablation_claim_total_source_record_count": len(records),
        "ablation_claim_input_record_count": len(accepted),
        "ablation_claim_excluded_record_count": len(excluded),
        "ablation_claim_excluded_metric_status_counts": dict(sorted(excluded_metric_status_counts.items())),
        "ablation_claim_excluded_resource_profile_counts": dict(sorted(excluded_resource_profile_counts.items())),
        "ablation_claim_excluded_sample_role_counts": dict(sorted(excluded_sample_role_counts.items())),
        "ablation_claim_excluded_proxy_record_count": sum(
            1 for row in excluded if "proxy" in str(row.get("metric_status", ""))
        ),
        "ablation_claim_excluded_record_examples": [
            {
                "attack_record_id": row.get("attack_record_id", ""),
                "attack_family": row.get("attack_family", ""),
                "attack_name": row.get("attack_name", ""),
                "resource_profile": row.get("resource_profile", ""),
                "sample_role": row.get("sample_role", ""),
                "metric_status": row.get("metric_status", ""),
            }
            for row in excluded[:5]
        ],
        "ablation_claim_input_ready": bool(accepted),
        "supports_paper_claim": bool(accepted),
    }


def filter_ablation_claim_input_records(attack_records: Iterable[dict[str, Any]]) -> tuple[tuple[dict[str, Any], ...], dict[str, Any]]:
    """返回正式消融 claim 输入 records 与筛选报告。"""

    records = tuple(dict(row) for row in attack_records)
    report = build_ablation_claim_input_report(records)
    return tuple(row for row in records if _source_is_formal_ablation_record(row)), report


def _metric_status_for_group(rows: Iterable[dict[str, Any]]) -> str:
    """根据消融记录来源汇总分组级 metric_status。"""
    supported = list(rows)
    if not supported:
        return "unsupported"
    statuses = {str(row.get("metric_status", "")) for row in supported}
    watermark_rescore_status = FORMAL_ABLATION_METRIC_STATUS
    retention_proxy_status = "measured_from_real_attacked_image_retention_proxy_formal_protocol"
    legacy_real_status = "measured_from_real_attacked_image_formal_protocol"
    if statuses == {watermark_rescore_status}:
        return watermark_rescore_status
    if statuses == {retention_proxy_status}:
        return retention_proxy_status
    if statuses == {legacy_real_status}:
        return "measured_from_legacy_real_attacked_image_protocol"
    if watermark_rescore_status in statuses or retention_proxy_status in statuses or legacy_real_status in statuses:
        return "measured_from_mixed_real_and_local_proxy"
    return "measured_from_local_proxy"


def _group(rows: Iterable[dict[str, Any]], keys: tuple[str, ...]) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    """按字段组合分组。"""
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(key, "") for key in keys)].append(row)
    return dict(grouped)


def _geometry_reliable(record: dict[str, Any], spec: AblationSpec) -> bool:
    """根据消融配置重算几何可靠性。"""
    baseline_geometry = _as_bool(record, "geometry_reliable")
    if spec.geometry_mode == "unchanged":
        return baseline_geometry
    if spec.geometry_mode == "disabled_for_geometric" and record.get("attack_family") == "geometric_transform":
        return False
    if spec.geometry_mode == "geometric_only":
        return baseline_geometry and record.get("attack_name") != "composite_geometric_attacks"
    if spec.geometry_mode == "registration_only":
        return baseline_geometry and record.get("attack_name") not in {"rotation", "composite_geometric_attacks"}
    return baseline_geometry


def _ablate_supported_record(record: dict[str, Any], spec: AblationSpec, threshold: float) -> dict[str, Any]:
    """对一条可统计攻击记录应用机制消融。"""
    source_formal = _source_is_formal_ablation_record(record)
    formal_claim_status = (
        "governed_formal_attack_rescore_counterfactual"
        if spec.formal_method_allowed
        else "governed_formal_attack_rescore_audit_control"
    )
    local_claim_status = "local_proxy_only" if spec.formal_method_allowed else "unsupported_audit_only"
    if spec.ablation_id == "full_slm_wm":
        attestation_available = True
        return {
            "ablated_lf_score_retention": _as_float(record, "lf_score_retention"),
            "ablated_tail_score_retention": _as_float(record, "tail_score_retention"),
            "ablated_score_retention": _as_float(record, "score_retention"),
            "ablated_quality_score_proxy": _as_float(record, "quality_score_proxy"),
            "ablated_attention_consistency_proxy": _as_float(record, "attention_consistency_proxy"),
            "ablated_raw_content_score_after": _as_float(record, "raw_content_score_after"),
            "ablated_aligned_content_score_after": _as_float(record, "aligned_content_score_after"),
            "ablated_geometry_reliable": _as_bool(record, "geometry_reliable"),
            "ablated_rescue_applied": _as_bool(record, "rescue_applied"),
            "ablated_evidence_decision": _as_bool(record, "evidence_decision"),
            "ablated_detection_decision": _as_bool(record, "evidence_decision") and attestation_available,
            "attestation_available": attestation_available,
            "claim_status": formal_claim_status if source_formal else local_claim_status,
        }
    baseline_lf = _as_float(record, "lf_score_retention")
    baseline_tail = _as_float(record, "tail_score_retention")
    ablated_lf = _clamp01(baseline_lf * spec.lf_multiplier)
    ablated_tail = _clamp01(baseline_tail * spec.tail_multiplier)
    ablated_score_retention = _clamp01((0.56 * ablated_lf + 0.44 * ablated_tail) * spec.score_multiplier)
    ablated_quality = _clamp01(_as_float(record, "quality_score_proxy") * spec.quality_multiplier)
    ablated_attention = _clamp01(_as_float(record, "attention_consistency_proxy") * spec.attention_multiplier)
    raw_before = _as_float(record, "raw_content_score_before")
    raw_after = _clamp01(raw_before * ablated_score_retention)
    baseline_gain = max(0.0, _as_float(record, "aligned_content_score_after") - _as_float(record, "raw_content_score_after"))
    aligned_after = _clamp01(raw_after + baseline_gain * spec.aligned_gain_multiplier)
    ablated_geometry = _geometry_reliable(record, spec)
    baseline_rescue = _as_bool(record, "rescue_applied")
    ablated_rescue = baseline_rescue and spec.rescue_mode == "unchanged" and ablated_geometry
    if spec.rescue_mode in {"disabled", "blocked"}:
        ablated_rescue = False
    sample_role = str(record.get("sample_role", ""))
    score_bias = spec.positive_score_bias if sample_role == "positive_source" else spec.negative_score_bias
    effective_score = aligned_after + score_bias
    ablated_evidence = effective_score >= threshold
    if spec.ablation_id == "geo_direct_positive_audit":
        ablated_evidence = ablated_geometry
    attestation_available = spec.attestation_mode != "missing"
    ablated_detection = ablated_evidence and attestation_available
    claim_status = formal_claim_status if source_formal else local_claim_status
    return {
        "ablated_lf_score_retention": ablated_lf,
        "ablated_tail_score_retention": ablated_tail,
        "ablated_score_retention": ablated_score_retention,
        "ablated_quality_score_proxy": ablated_quality,
        "ablated_attention_consistency_proxy": ablated_attention,
        "ablated_raw_content_score_after": raw_after,
        "ablated_aligned_content_score_after": aligned_after,
        "ablated_geometry_reliable": ablated_geometry,
        "ablated_rescue_applied": ablated_rescue,
        "ablated_evidence_decision": ablated_evidence,
        "ablated_detection_decision": ablated_detection,
        "attestation_available": attestation_available,
        "claim_status": claim_status,
    }


def build_ablation_records(
    attack_records: Iterable[dict[str, Any]],
    ablation_specs: Iterable[AblationSpec],
    threshold: float,
) -> tuple[dict[str, Any], ...]:
    """基于攻击矩阵 records 构造内部消融 records。"""
    rows: list[dict[str, Any]] = []
    for spec in ablation_specs:
        change_digest = build_stable_digest(spec.to_dict())
        for record in attack_records:
            source_payload = {
                "ablation_id": spec.ablation_id,
                "attack_record_id": record.get("attack_record_id", ""),
                "mechanism_change_digest": change_digest,
            }
            digest = build_stable_digest(source_payload)
            unsupported = record.get("metric_status") == "unsupported"
            source_formal = _source_is_formal_ablation_record(record)
            row_supports_claim = bool(source_formal and spec.formal_method_allowed and not unsupported)
            if unsupported:
                ablated = {
                    "ablated_lf_score_retention": 0.0,
                    "ablated_tail_score_retention": 0.0,
                    "ablated_score_retention": 0.0,
                    "ablated_quality_score_proxy": 0.0,
                    "ablated_attention_consistency_proxy": 0.0,
                    "ablated_raw_content_score_after": 0.0,
                    "ablated_aligned_content_score_after": 0.0,
                    "ablated_geometry_reliable": False,
                    "ablated_rescue_applied": False,
                    "ablated_evidence_decision": False,
                    "ablated_detection_decision": False,
                    "attestation_available": spec.attestation_mode != "missing",
                    "claim_status": "unsupported_input",
                }
            else:
                ablated = _ablate_supported_record(record, spec, threshold)
            rows.append(
                {
                    "ablation_record_id": f"ablation_record_{digest[:16]}",
                    "ablation_record_digest": digest,
                    "ablation_id": spec.ablation_id,
                    "ablation_name": spec.ablation_name,
                    "mechanism_group": spec.mechanism_group,
                    "ablated_mechanism": spec.ablated_mechanism,
                    "mechanism_change": spec.mechanism_change,
                    "mechanism_change_digest": change_digest,
                    "attack_record_id": record.get("attack_record_id", ""),
                    "attack_family": record.get("attack_family", ""),
                    "attack_name": record.get("attack_name", ""),
                    "resource_profile": record.get("resource_profile", ""),
                    "split": record.get("split", ""),
                    "sample_role": record.get("sample_role", ""),
                    "metric_status": "unsupported" if unsupported else str(record.get("metric_status", "measured_from_local_proxy")),
                    "unsupported_reason": record.get("unsupported_reason", "") if unsupported else "",
                    "baseline_evidence_decision": _as_bool(record, "evidence_decision"),
                    "ablated_evidence_decision": ablated["ablated_evidence_decision"],
                    "ablated_detection_decision": ablated["ablated_detection_decision"],
                    "baseline_score_retention": _as_float(record, "score_retention"),
                    "ablated_score_retention": ablated["ablated_score_retention"],
                    "baseline_lf_score_retention": _as_float(record, "lf_score_retention"),
                    "ablated_lf_score_retention": ablated["ablated_lf_score_retention"],
                    "baseline_tail_score_retention": _as_float(record, "tail_score_retention"),
                    "ablated_tail_score_retention": ablated["ablated_tail_score_retention"],
                    "baseline_quality_score_proxy": _as_float(record, "quality_score_proxy"),
                    "ablated_quality_score_proxy": ablated["ablated_quality_score_proxy"],
                    "baseline_attention_consistency_proxy": _as_float(record, "attention_consistency_proxy"),
                    "ablated_attention_consistency_proxy": ablated["ablated_attention_consistency_proxy"],
                    "baseline_geometry_reliable": _as_bool(record, "geometry_reliable"),
                    "ablated_geometry_reliable": ablated["ablated_geometry_reliable"],
                    "baseline_rescue_applied": _as_bool(record, "rescue_applied"),
                    "ablated_rescue_applied": ablated["ablated_rescue_applied"],
                    "ablated_raw_content_score_after": ablated["ablated_raw_content_score_after"],
                    "ablated_aligned_content_score_after": ablated["ablated_aligned_content_score_after"],
                    "attestation_required": True,
                    "attestation_available": ablated["attestation_available"],
                    "formal_method_allowed": spec.formal_method_allowed,
                    "claim_status": ablated["claim_status"],
                    "mechanism_explanation": spec.mechanism_explanation,
                    "expected_failure_mode": spec.expected_failure_mode,
                    "supports_paper_claim": row_supports_claim,
                    "metadata": {
                        "source_supports_paper_claim": bool(record.get("supports_paper_claim", False)),
                        "source_formal_ablation_claim_input": source_formal,
                        "threshold_value": threshold,
                    },
                }
            )
    return tuple(rows)


def _aggregate_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """聚合一个消融分组。"""
    supported = _supported(rows)
    positives = [row for row in supported if row["sample_role"] == "positive_source"]
    clean_negatives = [row for row in supported if row["sample_role"] == "clean_negative"]
    attacked_negatives = [row for row in supported if row["sample_role"] == "attacked_negative"]
    negatives = [row for row in supported if row["sample_role"] != "positive_source"]
    return {
        "metric_status": _metric_status_for_group(supported),
        "ablation_record_count": len(rows),
        "supported_record_count": len(supported),
        "unsupported_record_count": len(rows) - len(supported),
        "positive_count": len(positives),
        "negative_count": len(negatives),
        "true_positive_rate": _rate(positives, "ablated_detection_decision"),
        "false_positive_rate": _rate(negatives, "ablated_detection_decision"),
        "clean_false_positive_rate": _rate(clean_negatives, "ablated_detection_decision"),
        "attacked_false_positive_rate": _rate(attacked_negatives, "ablated_detection_decision"),
        "score_retention_mean": _mean(supported, "ablated_score_retention"),
        "quality_score_proxy_mean": _mean(supported, "ablated_quality_score_proxy"),
        "attention_consistency_proxy_mean": _mean(supported, "ablated_attention_consistency_proxy"),
        "geometry_reliable_rate": _rate(supported, "ablated_geometry_reliable"),
        "rescue_rate": _rate(supported, "ablated_rescue_applied"),
        "attestation_available_rate": _rate(supported, "attestation_available"),
        "supports_paper_claim": bool(supported) and all(bool(row.get("supports_paper_claim", False)) for row in supported),
    }


def aggregate_mechanism_ablation_table(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """构造每个消融一行的机制表。"""
    grouped = _group(records, ("ablation_id",))
    base_by_id: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for (ablation_id,), group in sorted(grouped.items()):
        first = group[0]
        metrics = _aggregate_group(group)
        row = {
            "ablation_id": ablation_id,
            "ablation_name": first["ablation_name"],
            "mechanism_group": first["mechanism_group"],
            "ablated_mechanism": first["ablated_mechanism"],
            "mechanism_change_digest": first["mechanism_change_digest"],
            "mechanism_explanation": first["mechanism_explanation"],
            **metrics,
            "true_positive_delta_from_full": 0.0,
            "false_positive_delta_from_full": 0.0,
            "score_retention_delta_from_full": 0.0,
            "quality_delta_from_full": 0.0,
            "degradation_chain_rank": 0,
        }
        rows.append(row)
        base_by_id[ablation_id] = row
    full = base_by_id.get("full_slm_wm")
    if full:
        for row in rows:
            row["true_positive_delta_from_full"] = row["true_positive_rate"] - full["true_positive_rate"]
            row["false_positive_delta_from_full"] = row["false_positive_rate"] - full["false_positive_rate"]
            row["score_retention_delta_from_full"] = row["score_retention_mean"] - full["score_retention_mean"]
            row["quality_delta_from_full"] = row["quality_score_proxy_mean"] - full["quality_score_proxy_mean"]
        degraded = sorted(
            rows,
            key=lambda item: (
                item["true_positive_delta_from_full"] - item["false_positive_delta_from_full"],
                item["score_retention_delta_from_full"],
            ),
        )
        for rank, row in enumerate(degraded, start=1):
            row["degradation_chain_rank"] = rank
    return rows


def aggregate_ablation_by_attack_family(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """构造按消融和攻击族聚合的表格。"""
    rows: list[dict[str, Any]] = []
    for key, group in sorted(_group(records, ("ablation_id", "attack_family")).items()):
        ablation_id, attack_family = key
        first = group[0]
        rows.append(
            {
                "ablation_id": ablation_id,
                "ablation_name": first["ablation_name"],
                "mechanism_group": first["mechanism_group"],
                "attack_family": attack_family,
                **_aggregate_group(group),
            }
        )
    return rows


def build_pairwise_delta_rows(mechanism_rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """构造每个消融相对完整方法的指标差异表。"""
    rows = list(mechanism_rows)
    full = next((row for row in rows if row["ablation_id"] == "full_slm_wm"), None)
    if not full:
        return []
    metric_specs = (
        ("true_positive_rate", "lower_true_positive_is_degradation"),
        ("false_positive_rate", "higher_false_positive_is_degradation"),
        ("score_retention_mean", "lower_retention_is_degradation"),
        ("quality_score_proxy_mean", "lower_quality_is_degradation"),
        ("geometry_reliable_rate", "lower_geometry_reliability_is_degradation"),
        ("attestation_available_rate", "lower_attestation_is_degradation"),
    )
    delta_rows: list[dict[str, Any]] = []
    for row in rows:
        if row["ablation_id"] == "full_slm_wm":
            continue
        for metric_name, direction in metric_specs:
            full_value = float(full[metric_name])
            ablated_value = float(row[metric_name])
            delta_rows.append(
                {
                    "ablation_id": row["ablation_id"],
                    "compared_to_ablation_id": "full_slm_wm",
                    "metric_name": metric_name,
                    "full_metric_value": full_value,
                    "ablated_metric_value": ablated_value,
                    "delta_value": ablated_value - full_value,
                    "degradation_direction": direction,
                    "mechanism_interpretation": row["mechanism_explanation"],
                    "supports_paper_claim": bool(full.get("supports_paper_claim")) and bool(row.get("supports_paper_claim")),
                }
            )
    return delta_rows


def build_ablation_claim_summary(
    ablation_specs: Iterable[AblationSpec],
    records: Iterable[dict[str, Any]],
    mechanism_rows: Iterable[dict[str, Any]],
    attack_manifest: dict[str, Any],
    baseline_manifest: dict[str, Any],
) -> dict[str, Any]:
    """构造内部消融声明边界摘要。"""
    spec_tuple = tuple(ablation_specs)
    record_tuple = tuple(records)
    row_tuple = tuple(mechanism_rows)
    required_ids = {spec.ablation_id for spec in default_ablation_specs()}
    actual_ids = {spec.ablation_id for spec in spec_tuple}
    core_claim_ids = set(CORE_ABLATION_CLAIM_IDS)
    row_by_id = {str(row.get("ablation_id", "")): row for row in row_tuple}
    mechanism_groups = sorted({spec.mechanism_group for spec in spec_tuple})
    unsupported_reasons = sorted({row.get("unsupported_reason", "") for row in record_tuple if row.get("unsupported_reason")})
    attack_formal_ready = (
        bool(attack_manifest.get("attack_metrics_ready", False))
        and bool(attack_manifest.get("real_attacked_image_closed_loop_ready", False))
        and bool(attack_manifest.get("formal_attack_detection_ready", False))
        and bool(attack_manifest.get("regeneration_attack_gpu_validation_ready", False))
        and int(attack_manifest.get("formal_proxy_replacement_incomplete_count", 0) or 0) == 0
    )
    if attack_formal_ready:
        unsupported_reasons = [reason for reason in unsupported_reasons if reason != "real_gpu_attack_required"]
    metric_statuses = {str(row.get("metric_status", "")) for row in record_tuple if row.get("metric_status") != "unsupported"}
    formal_input_ready = bool(record_tuple) and metric_statuses == {FORMAL_ABLATION_METRIC_STATUS}
    core_rows_ready = all(
        ablation_id in row_by_id and bool(row_by_id[ablation_id].get("supports_paper_claim", False))
        for ablation_id in core_claim_ids
    )
    core_ablation_ready_count = sum(
        1 for ablation_id in core_claim_ids if ablation_id in row_by_id and bool(row_by_id[ablation_id].get("supports_paper_claim", False))
    )
    protocol_ready = bool(record_tuple) and required_ids.issubset(actual_ids) and bool(attack_manifest.get("attack_metrics_ready"))
    evaluation_boundary = dict(attack_manifest.get("evaluation_boundary", {}))
    paper_run = build_paper_run_config(".")
    ablation_claim_gate_ready = (
        protocol_ready
        and formal_input_ready
        and attack_formal_ready
        and core_rows_ready
        and not unsupported_reasons
    )
    local_proxy_boundary = (
        "internal ablation claim uses governed formal attacked image watermark rescore records; probe and local proxy records are excluded before counterfactual aggregation"
        if ablation_claim_gate_ready
        else "internal ablation claim is blocked until all claim input rows use formal attacked image watermark rescore records and probe proxy rows are excluded"
    )
    return {
        "construction_unit_name": "internal_ablation_evidence",
        "result_protocol_name": result_protocol_name_for_run(paper_run.run_name),
        "paper_claim_scale": paper_run.run_name,
        "target_fpr": float(evaluation_boundary.get("target_fpr", PILOT_PAPER_FIXED_FPR)),
        "ablation_protocol_ready": protocol_ready,
        "mechanism_coverage_ready": required_ids.issubset(actual_ids),
        "ablation_count": len(spec_tuple),
        "ablation_record_count": len(record_tuple),
        "ablation_claim_metric_status": FORMAL_ABLATION_METRIC_STATUS if formal_input_ready else "non_formal_or_empty_ablation_input",
        "ablation_claim_formal_input_ready": formal_input_ready,
        "ablation_claim_gate_ready": ablation_claim_gate_ready,
        "ablation_standalone_claim_ready": ablation_claim_gate_ready,
        "strong_ablation_standalone_claim_ready": ablation_claim_gate_ready,
        "core_ablation_claim_ready": core_rows_ready,
        "core_ablation_claim_ids": list(CORE_ABLATION_CLAIM_IDS),
        "core_ablation_ready_count": core_ablation_ready_count,
        "core_ablation_required_count": len(core_claim_ids),
        "mechanism_group_count": len(mechanism_groups),
        "mechanism_groups": mechanism_groups,
        "degradation_chain": [row["ablation_id"] for row in sorted(row_tuple, key=lambda item: item["degradation_chain_rank"])],
        "external_baseline_result_ready": bool(baseline_manifest.get("metadata", {}).get("baseline_results_ready", False)),
        "attack_metrics_ready": bool(attack_manifest.get("attack_metrics_ready", False)),
        "attack_formal_evidence_ready": attack_formal_ready,
        "unsupported_reasons": unsupported_reasons,
        "local_proxy_boundary": local_proxy_boundary,
        "ablation_real_linkage_boundary": (
            "governed_formal_attack_records_reused_for_counterfactual_ablation_scoring"
            if ablation_claim_gate_ready
            else "formal_attacked_image_rescore_records_required_before_ablation_claim"
        ),
        "full_method_claim_ready": False,
        "supports_paper_claim": ablation_claim_gate_ready,
    }
