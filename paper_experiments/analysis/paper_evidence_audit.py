"""论文图表与声明证据审计构造器。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping

from main.core.digest import build_stable_digest


@dataclass(frozen=True)
class AuditInputBundle:
    """封装论文证据审计需要的上游摘要。

    该对象属于通用工程写法: 将跨产物的 manifest 和 report 汇总成统一输入,
    让后续审计函数只关注 claim、artifact 和 gap 的判定逻辑。
    """

    threshold_report: dict[str, Any]
    threshold_manifest: dict[str, Any]
    threshold_audit_report: dict[str, Any]
    threshold_audit_manifest: dict[str, Any]
    attack_manifest: dict[str, Any]
    attack_matrix_manifest: dict[str, Any]
    baseline_manifest: dict[str, Any]
    baseline_runtime_report: dict[str, Any]
    dataset_quality_manifest: dict[str, Any]
    dataset_quality_summary: dict[str, Any]
    ablation_manifest: dict[str, Any]
    ablation_claim_summary: dict[str, Any]
    source_path_map: dict[str, str]
    artifact_data_validation: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """转为 JSON 兼容字典。"""
        return asdict(self)


def build_evidence_audit_materialization(
    bundle: AuditInputBundle,
) -> dict[str, Any]:
    """从审计输入唯一重建正式行、readiness 报告和阻断报告.

    该函数属于通用工程写法: writer 与最终结果闭合门禁共享同一纯函数,
    从而避免持久化报告自行声明 ready 后与真实输入脱离.返回值中的所有内容
    均由 ``AuditInputBundle`` 决定, 可在不访问文件系统的环境中重复核验.
    """

    claim_rows = build_claim_audit_rows(bundle)
    table_rows = build_table_readiness_rows(bundle)
    figure_rows = build_figure_readiness_rows(bundle)
    gap_rows = build_evidence_gap_rows(bundle)
    builder_report = build_builder_readiness_report(
        claim_rows,
        table_rows,
        figure_rows,
    )
    artifact_data_validation = bundle.artifact_data_validation
    builder_report = {
        **builder_report,
        "artifact_data_validation_ready": artifact_data_validation.get(
            "artifact_data_validation_ready",
            False,
        ),
        "blocked_artifact_data_count": artifact_data_validation.get(
            "blocked_artifact_data_count",
            0,
        ),
        "blocked_artifact_data_ids": artifact_data_validation.get(
            "blocked_artifact_data_ids",
            [],
        ),
        "raw_image_only_detection_records_ready": artifact_data_validation.get(
            "raw_image_only_detection_records_ready",
            False,
        ),
        "raw_image_only_detection_records_sha256": artifact_data_validation.get(
            "raw_image_only_detection_records_sha256",
            "",
        ),
    }
    blocker_report = build_submission_blocker_report(
        claim_rows,
        gap_rows,
        builder_report,
    )
    return {
        "claim_rows": claim_rows,
        "table_rows": table_rows,
        "figure_rows": figure_rows,
        "gap_rows": gap_rows,
        "builder_report": builder_report,
        "blocker_report": blocker_report,
    }


def build_evidence_audit_manifest_config(
    bundle: AuditInputBundle,
    materialization: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    """精确重建证据审计 manifest 的配置摘要输入.

    此处设计的主要考虑在于: ``config_digest`` 必须绑定完整审计输入、全部
    派生行和实际数据验证报告, 而不能仅检查其是否具有 SHA-256 的外形.
    """

    rebuilt = dict(materialization or build_evidence_audit_materialization(bundle))
    summary = {
        key: rebuilt[key]
        for key in (
            "claim_rows",
            "table_rows",
            "figure_rows",
            "gap_rows",
            "builder_report",
            "blocker_report",
        )
    }
    summary["artifact_data_validation"] = bundle.artifact_data_validation
    return {
        "summary_digest": build_stable_digest(summary),
        "input_bundle_digest": build_stable_digest(bundle.to_dict()),
        "artifact_data_validation_digest": build_stable_digest(
            bundle.artifact_data_validation
        ),
    }


def _yes(value: Any) -> bool:
    """把 manifest 中常见的布尔或字符串状态转为布尔值。"""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"true", "pass", "ready", "measured"}
    return bool(value)


def _source(bundle: AuditInputBundle, key: str, default_value: str) -> str:
    """读取脚本层注入的受治理路径, 让核心分析逻辑不绑定外层目录命名。"""
    return bundle.source_path_map.get(key, default_value)


def _artifact_data_ready(bundle: AuditInputBundle, check_id: str) -> bool:
    """读取 writer 的实际文件验证结果.

    直接构造 bundle 的纯函数测试可省略该映射; 正式 writer 始终注入完整验证结果.
    """

    if not bundle.artifact_data_validation:
        return True
    return bundle.artifact_data_validation.get(check_id) is True


def _method_curve_data_ready(bundle: AuditInputBundle) -> bool:
    """判断原始记录,冻结协议和重建后的连续检测表是否均通过审计."""

    return all(
        _artifact_data_ready(bundle, check_id)
        for check_id in (
            "frozen_evidence_protocol_ready",
            "raw_image_only_detection_records_ready",
            "test_detection_metrics_ready",
            "score_distribution_table_ready",
            "roc_curve_points_ready",
            "det_curve_points_ready",
        )
    )


def _real_attack_closed_loop_ready(attack_manifest: dict[str, Any]) -> bool:
    """判断真实 attacked image 文件与摘要闭环是否已经进入正式 manifest。"""
    return (
        _yes(attack_manifest.get("real_attacked_image_closed_loop_ready"))
        and _yes(attack_manifest.get("formal_attack_detection_ready"))
        and int(attack_manifest.get("real_attacked_image_count", 0)) > 0
    )


def _regeneration_attack_gpu_ready(attack_manifest: dict[str, Any]) -> bool:
    """判断再扩散类攻击是否已经由真实 GPU formal records 覆盖。"""
    required_count = int(
        attack_manifest.get("required_real_gpu_attack_count", attack_manifest.get("required_regeneration_attack_count", 0))
    )
    measured_count = int(
        attack_manifest.get("measured_real_gpu_attack_count", attack_manifest.get("measured_regeneration_attack_count", 0))
    )
    ready_flag = attack_manifest.get(
        "real_gpu_attack_validation_ready",
        attack_manifest.get("regeneration_attack_gpu_validation_ready"),
    )
    return _yes(ready_flag) and required_count > 0 and measured_count >= required_count


def _image_only_detector_ready(report: dict[str, Any]) -> bool:
    """判断正式检测是否严格限制为图像、密钥和公开模型。"""

    return (
        str(report.get("detector_input_access_mode", "")) == "image_key_public_model_only"
        and not _yes(report.get("generation_latent_trace_required", True))
    )


def _runtime_sample_scale_ready(report: dict[str, Any]) -> bool:
    """判断当前运行是否完整覆盖所声明的 70/700/7000 Prompt 协议。"""

    run_name = str(report.get("paper_run_name", ""))
    expected_prompt_counts = {"probe_paper": 70, "pilot_paper": 700, "full_paper": 7000}
    expected_test_counts = {"probe_paper": 34, "pilot_paper": 340, "full_paper": 3400}
    expected_prompt_count = expected_prompt_counts.get(run_name)
    expected_test_count = expected_test_counts.get(run_name)
    split_counts = report.get("split_counts", {})
    return (
        expected_prompt_count is not None
        and int(report.get("prompt_count", 0)) == expected_prompt_count
        and int(report.get("runtime_result_count", 0)) == expected_prompt_count
        and int(split_counts.get("test", 0)) == expected_test_count
        and str(report.get("protocol_decision", "")).lower() == "pass"
    )


def _scientific_operator_ready(report: dict[str, Any]) -> bool:
    """判断真实关键科学算子记录是否完整通过运行门禁。"""

    return (
        _yes(report.get("scientific_operator_gate_ready"))
        and int(report.get("scientific_operator_failure_count", 1)) == 0
        and int(report.get("scientific_update_record_count", 0))
        == int(report.get("expected_scientific_update_record_count", -1))
        and int(report.get("scientific_update_record_count", 0)) > 0
    )


def _attack_robustness_blockers(
    attack_manifest: dict[str, Any],
    *,
    attack_metrics_data_ready: bool = True,
) -> list[str]:
    """生成攻击鲁棒性声明的当前阻断项。"""
    blockers = []
    if not _real_attack_closed_loop_ready(attack_manifest):
        blockers.append("attacked_image_files_missing")
    if not _regeneration_attack_gpu_ready(attack_manifest):
        blockers.append("regeneration_attack_real_gpu_missing")
    if not _image_only_detector_ready(attack_manifest):
        blockers.append("image_only_detector_boundary_not_ready")
    if not attack_metrics_data_ready:
        blockers.append("attack_family_metrics_data_invalid")
    return blockers


def _fixed_fpr_and_rescue_boundary_ready(threshold_report: dict[str, Any], attack_manifest: dict[str, Any]) -> bool:
    """判断 fixed-FPR 与 rescue 是否在同一正式检测协议内闭合。"""
    return (
        _yes(threshold_report.get("fixed_fpr_and_rescue_boundary_ready"))
        and _yes(threshold_report.get("fixed_fpr_boundary_ready"))
        and _yes(threshold_report.get("rescue_boundary_ready"))
        and _real_attack_closed_loop_ready(attack_manifest)
        and _regeneration_attack_gpu_ready(attack_manifest)
    )


def _threshold_audit_ready(bundle: AuditInputBundle) -> bool:
    """判断主方法与四个 baseline 是否全部通过独立 fixed-FPR 重算。"""

    report = bundle.threshold_audit_report
    return (
        report.get("method_identity_ready") is True
        and report.get("all_method_thresholds_ready") is True
        and report.get("fixed_fpr_threshold_audit_ready") is True
        and report.get("supports_paper_claim") is True
    )


def _dataset_level_quality_blockers(
    dataset_quality_summary: dict[str, Any],
    *,
    metrics_data_ready: bool = True,
) -> list[str]:
    """生成数据集级质量指标的当前阻断项。"""

    blockers = []
    if not _yes(dataset_quality_summary.get("formal_fid_kid_ready")):
        blockers.append("fid_kid_dataset_metrics_missing")
    if not _yes(dataset_quality_summary.get("canonical_formal_feature_extractor_ready")):
        blockers.append("canonical_inception_feature_extractor_missing")
    if not _yes(dataset_quality_summary.get("formal_fid_kid_claim_gate_ready")):
        blockers.append("formal_fid_kid_claim_gate_not_ready")
    if not metrics_data_ready:
        blockers.append("fid_kid_table_data_invalid")
    return blockers


def _baseline_comparison_ready(
    report: dict[str, Any],
    *,
    table_data_ready: bool = True,
) -> bool:
    """判断四个主表 baseline 是否在完整攻击模板上形成可支撑论文的比较。"""

    required_flags = (
        "comparison_table_supports_paper_claim",
        "supports_paper_claim",
        "primary_baseline_formal_ready",
        "primary_baseline_results_ready",
        "primary_baseline_formal_template_coverage_ready",
        "primary_baseline_formal_evidence_collection_ready",
        "formal_import_validation_ready",
        "formal_evidence_path_resolution_ready",
    )
    return table_data_ready and all(_yes(report.get(field_name)) for field_name in required_flags)


def _row(
    claim_id: str,
    claim_scope: str,
    claim_text: str,
    claim_decision: str,
    evidence_path: str,
    blockers: Iterable[str],
    paper_claim_supported: bool = False,
) -> dict[str, Any]:
    """构造 claim audit 行。"""
    blocker_list = [blocker for blocker in blockers if blocker]
    return {
        "claim_id": claim_id,
        "claim_scope": claim_scope,
        "claim_text": claim_text,
        "claim_decision": claim_decision,
        "evidence_path": evidence_path,
        "blocker_count": len(blocker_list),
        "primary_blocker": blocker_list[0] if blocker_list else "",
        "paper_claim_supported": paper_claim_supported,
        "supports_paper_claim": paper_claim_supported,
    }


def build_claim_audit_rows(bundle: AuditInputBundle) -> list[dict[str, Any]]:
    """构造论文 claim 到证据路径的审计表。"""
    threshold = bundle.threshold_report
    attack = bundle.attack_manifest
    baseline = bundle.baseline_runtime_report
    dataset_quality = bundle.dataset_quality_summary
    ablation = bundle.ablation_claim_summary
    full_ready = (
        _yes(threshold.get("full_method_claim_ready"))
        and _threshold_audit_ready(bundle)
        and _yes(attack.get("full_method_claim_ready"))
        and _image_only_detector_ready(threshold)
        and _scientific_operator_ready(threshold)
        and _method_curve_data_ready(bundle)
    )
    attack_metrics_data_ready = _artifact_data_ready(bundle, "attack_family_metrics_ready")
    attack_blockers = _attack_robustness_blockers(
        attack,
        attack_metrics_data_ready=attack_metrics_data_ready,
    )
    attack_ready = not attack_blockers
    baseline_ready = _baseline_comparison_ready(
        baseline,
        table_data_ready=_artifact_data_ready(bundle, "baseline_comparison_table_ready"),
    )
    dataset_metrics_data_ready = _artifact_data_ready(bundle, "dataset_quality_metrics_ready")
    dataset_quality_ready = (
        _yes(dataset_quality.get("formal_fid_kid_ready"))
        and _yes(dataset_quality.get("canonical_formal_feature_extractor_ready"))
        and _yes(dataset_quality.get("formal_fid_kid_claim_gate_ready"))
        and dataset_metrics_data_ready
    )
    ablation_claim_ready = (
        _yes(ablation.get("ablation_claim_gate_ready"))
        and _yes(ablation.get("supports_paper_claim"))
        and _artifact_data_ready(bundle, "mechanism_ablation_metrics_ready")
        and _artifact_data_ready(bundle, "mechanism_pairwise_delta_ready")
    )
    sample_scale_ready = _runtime_sample_scale_ready(threshold)
    submission_core_ready = (
        full_ready
        and attack_ready
        and baseline_ready
        and dataset_quality_ready
        and ablation_claim_ready
        and sample_scale_ready
    )
    return [
        _row(
            "claim_raw_content_fixed_fpr_boundary",
            "method_metric",
            "raw content 分支具备 fixed-FPR 校准边界。",
            "engineering_supported_not_paper_final",
            _source(bundle, "threshold_report", "threshold_report"),
            [] if _yes(threshold.get("raw_content_claim_ready")) else ["raw_content_claim_not_ready"],
        ),
        _row(
            "claim_full_method_fixed_fpr_boundary",
            "method_metric",
            "完整 SLM-WM 方法满足 fixed-FPR 统计边界。",
            "paper_supported" if full_ready else "unsupported",
            _source(
                bundle,
                "threshold_audit_report",
                "threshold_audit_report",
            ),
            []
            if full_ready
            else [
                "full_method_claim_ready_false",
                "image_only_detector_boundary_not_ready",
                "continuous_detection_curve_data_invalid"
                if not _method_curve_data_ready(bundle)
                else "",
            ],
            paper_claim_supported=full_ready,
        ),
        _row(
            "claim_attack_robustness_under_common_matrix",
            "robustness",
            "SLM-WM 在共同攻击矩阵下具有稳健检测表现。",
            "paper_supported" if attack_ready else "preview_only",
            _source(bundle, "attack_manifest", "attack_manifest"),
            attack_blockers,
            paper_claim_supported=attack_ready,
        ),
        _row(
            "claim_baseline_superiority",
            "baseline_comparison",
            "SLM-WM 优于外部 watermark baseline。",
            "paper_supported" if baseline_ready else "unsupported",
            _source(bundle, "baseline_runtime_report", "baseline_runtime_report.json"),
            [] if baseline_ready else ["baseline_result_missing"],
            paper_claim_supported=baseline_ready,
        ),
        _row(
            "claim_internal_mechanism_necessity",
            "ablation",
            "语义路由、Jacobian Null Space、空间 LF、幅值尾部稳健载体和注意力几何均为必要机制。",
            "paper_supported" if ablation_claim_ready else "preview_only",
            _source(bundle, "ablation_claim_summary", "ablation_claim_summary"),
            [] if ablation_claim_ready else ["ablation_claim_gate_not_ready"],
            paper_claim_supported=ablation_claim_ready,
        ),
        _row(
            "claim_quality_preservation_pair_metrics",
            "quality",
            "真实 clean/watermarked 图像对的 pair-level 质量指标可被下游审计。",
            "engineering_supported_not_paper_final",
            _source(bundle, "quality_metrics_summary", "quality_metrics_summary"),
            [] if _yes(threshold.get("perceptual_metrics_ready")) else ["perceptual_metrics_missing"],
        ),
        _row(
            "claim_dataset_level_quality_boundary",
            "quality",
            "数据集级 FID / KID 由正式特征后端和完整成组图像集合计算。",
            "paper_supported" if dataset_quality_ready else "engineering_supported_not_paper_final",
            _source(bundle, "dataset_quality_summary", "dataset_quality_summary"),
            _dataset_level_quality_blockers(
                dataset_quality,
                metrics_data_ready=dataset_metrics_data_ready,
            ),
            paper_claim_supported=dataset_quality_ready,
        ),
        _row(
            "claim_submission_ready_package",
            "submission_readiness",
            "当前仓库已具备投稿冻结所需的完整证据。",
            "paper_supported" if submission_core_ready else "unsupported",
            "submission_blocker_report",
            []
            if submission_core_ready
            else [
                "" if full_ready else "full_method_claim_ready_false",
                "" if baseline_ready else "baseline_result_missing",
                "" if attack_ready else "real_attack_evidence_missing",
                "" if dataset_quality_ready else "formal_dataset_quality_missing",
                "" if ablation_claim_ready else "formal_ablation_missing",
                "" if sample_scale_ready else "declared_sample_scale_incomplete",
            ],
            paper_claim_supported=submission_core_ready,
        ),
    ]


def _artifact_row(
    audit_item_id: str,
    artifact_kind: str,
    artifact_name: str,
    source_paths: Iterable[str],
    builder_status: str,
    paper_ready: bool,
    blockers: Iterable[str],
) -> dict[str, Any]:
    """构造表格或图数据 readiness 行。"""
    blocker_list = [blocker for blocker in blockers if blocker]
    return {
        "audit_item_id": audit_item_id,
        "artifact_kind": artifact_kind,
        "artifact_name": artifact_name,
        "source_paths": ";".join(source_paths),
        "builder_status": builder_status,
        "paper_ready": paper_ready,
        "blocker_count": len(blocker_list),
        "primary_blocker": blocker_list[0] if blocker_list else "",
        "supports_paper_claim": paper_ready,
    }


def build_table_readiness_rows(bundle: AuditInputBundle) -> list[dict[str, Any]]:
    """构造论文表格 readiness 审计行。"""
    threshold = bundle.threshold_report
    attack = bundle.attack_manifest
    baseline = bundle.baseline_runtime_report
    dataset_quality = bundle.dataset_quality_summary
    ablation = bundle.ablation_claim_summary
    full_ready = (
        _yes(threshold.get("full_method_claim_ready"))
        and _threshold_audit_ready(bundle)
        and _image_only_detector_ready(threshold)
        and _scientific_operator_ready(threshold)
        and _runtime_sample_scale_ready(threshold)
        and _method_curve_data_ready(bundle)
    )
    frozen_data_ready = _artifact_data_ready(bundle, "frozen_evidence_protocol_ready")
    test_metrics_data_ready = _artifact_data_ready(bundle, "test_detection_metrics_ready")
    attack_metrics_data_ready = _artifact_data_ready(bundle, "attack_family_metrics_ready")
    baseline_table_data_ready = _artifact_data_ready(bundle, "baseline_comparison_table_ready")
    ablation_metrics_data_ready = _artifact_data_ready(bundle, "mechanism_ablation_metrics_ready")
    dataset_metrics_data_ready = _artifact_data_ready(bundle, "dataset_quality_metrics_ready")
    attack_blockers = _attack_robustness_blockers(
        attack,
        attack_metrics_data_ready=attack_metrics_data_ready,
    )
    attack_ready = not attack_blockers
    dataset_quality_ready = (
        _yes(dataset_quality.get("formal_fid_kid_ready"))
        and _yes(dataset_quality.get("canonical_formal_feature_extractor_ready"))
        and _yes(dataset_quality.get("formal_fid_kid_claim_gate_ready"))
        and dataset_metrics_data_ready
    )
    ablation_claim_ready = (
        _yes(ablation.get("ablation_claim_gate_ready"))
        and _yes(ablation.get("supports_paper_claim"))
        and ablation_metrics_data_ready
        and _artifact_data_ready(bundle, "mechanism_pairwise_delta_ready")
    )
    baseline_ready = _baseline_comparison_ready(
        baseline,
        table_data_ready=baseline_table_data_ready,
    )
    return [
        _artifact_row(
            "table_fixed_fpr_operating_points",
            "table",
            "fixed-FPR operating point 表",
            [
                _source(bundle, "fixed_fpr_operating_points", "fixed_fpr_operating_points"),
                _source(
                    bundle,
                    "threshold_audit_report",
                    "threshold_audit_report",
                ),
            ],
            (
                "blocked"
                if not frozen_data_ready
                else ("rebuildable_paper_claim" if full_ready else "rebuildable_preview")
            ),
            full_ready,
            []
            if full_ready
            else (
                ["frozen_evidence_protocol_data_invalid"]
                if not frozen_data_ready
                else (["threshold_degenerate"] if _yes(threshold.get("threshold_degenerate")) else ["formal_image_only_runtime_not_ready"])
            ),
        ),
        _artifact_row(
            "table_main_method_metrics",
            "table",
            "主方法检测指标表",
            [_source(bundle, "standard_watermark_metrics", "standard_watermark_metrics")],
            (
                "blocked"
                if not test_metrics_data_ready
                else ("rebuildable_paper_claim" if full_ready else "rebuildable_preview")
            ),
            full_ready,
            []
            if full_ready
            else (["test_detection_metrics_data_invalid"] if not test_metrics_data_ready else ["full_method_claim_ready_false"]),
        ),
        _artifact_row(
            "table_attack_robustness",
            "table",
            "攻击鲁棒性表",
            [_source(bundle, "attack_family_metrics", "attack_family_metrics")],
            (
                "blocked"
                if not attack_metrics_data_ready
                else ("rebuildable_paper_claim" if attack_ready else "rebuildable_preview")
            ),
            attack_ready,
            attack_blockers,
        ),
        _artifact_row(
            "table_baseline_comparison",
            "table",
            "外部 baseline 对比表",
            [_source(bundle, "baseline_comparison_table", "baseline_comparison_table.csv")],
            (
                "blocked"
                if not baseline_table_data_ready
                else ("rebuildable_paper_claim" if baseline_ready else "protocol_ready_result_missing")
            ),
            baseline_ready,
            []
            if baseline_ready
            else (["baseline_comparison_table_data_invalid"] if not baseline_table_data_ready else ["baseline_result_missing"]),
        ),
        _artifact_row(
            "table_internal_ablation",
            "table",
            "内部机制消融表",
            [_source(bundle, "mechanism_ablation_table", "mechanism_ablation_table")],
            (
                "blocked"
                if not ablation_metrics_data_ready
                else ("rebuildable_paper_claim" if ablation_claim_ready else "rebuildable_preview")
            ),
            ablation_claim_ready,
            []
            if ablation_claim_ready
            else (["ablation_metrics_data_invalid"] if not ablation_metrics_data_ready else ["ablation_claim_gate_not_ready"]),
        ),
        _artifact_row(
            "table_quality_metrics",
            "table",
            "图像质量与感知指标表",
            [
                _source(bundle, "quality_metrics_summary", "quality_metrics_summary"),
                _source(bundle, "dataset_quality_metrics", "dataset_quality_metrics"),
            ],
            (
                "blocked"
                if not dataset_metrics_data_ready
                else ("rebuildable_paper_claim" if dataset_quality_ready else "rebuildable_preview")
            ),
            dataset_quality_ready,
            _dataset_level_quality_blockers(
                dataset_quality,
                metrics_data_ready=dataset_metrics_data_ready,
            ),
        ),
    ]


def build_figure_readiness_rows(bundle: AuditInputBundle) -> list[dict[str, Any]]:
    """构造论文图数据 readiness 审计行。"""
    baseline = bundle.baseline_runtime_report
    attack = bundle.attack_manifest
    threshold = bundle.threshold_report
    ablation = bundle.ablation_claim_summary
    full_ready = (
        _yes(threshold.get("full_method_claim_ready"))
        and _threshold_audit_ready(bundle)
        and _image_only_detector_ready(threshold)
        and _scientific_operator_ready(threshold)
        and _runtime_sample_scale_ready(threshold)
        and _method_curve_data_ready(bundle)
    )
    score_distribution_data_ready = _artifact_data_ready(bundle, "score_distribution_table_ready")
    roc_data_ready = _artifact_data_ready(bundle, "roc_curve_points_ready")
    det_data_ready = _artifact_data_ready(bundle, "det_curve_points_ready")
    attack_metrics_data_ready = _artifact_data_ready(bundle, "attack_family_metrics_ready")
    ablation_delta_data_ready = _artifact_data_ready(bundle, "mechanism_pairwise_delta_ready")
    baseline_table_data_ready = _artifact_data_ready(bundle, "baseline_comparison_table_ready")
    attack_blockers = _attack_robustness_blockers(
        attack,
        attack_metrics_data_ready=attack_metrics_data_ready,
    )
    attack_ready = not attack_blockers
    ablation_claim_ready = (
        _yes(ablation.get("ablation_claim_gate_ready"))
        and _yes(ablation.get("supports_paper_claim"))
        and _artifact_data_ready(bundle, "mechanism_ablation_metrics_ready")
        and ablation_delta_data_ready
    )
    baseline_ready = _baseline_comparison_ready(
        baseline,
        table_data_ready=baseline_table_data_ready,
    )
    return [
        _artifact_row(
            "figure_score_distribution",
            "figure_data",
            "score distribution 图数据",
            [
                _source(
                    bundle,
                    "raw_image_only_detection_records",
                    "image_only_detection_records.jsonl",
                ),
                _source(bundle, "score_distribution_table", "score_distribution_table"),
            ],
            (
                "blocked"
                if not score_distribution_data_ready
                else ("rebuildable_paper_claim" if full_ready else "rebuildable_preview")
            ),
            full_ready and score_distribution_data_ready,
            []
            if full_ready and score_distribution_data_ready
            else (["score_distribution_table_data_invalid"] if not score_distribution_data_ready else ["formal_image_only_runtime_not_ready"]),
        ),
        _artifact_row(
            "figure_roc_det",
            "figure_data",
            "ROC / DET 图数据",
            [
                _source(
                    bundle,
                    "raw_image_only_detection_records",
                    "image_only_detection_records.jsonl",
                ),
                _source(bundle, "roc_curve_points", "roc_curve_points"),
                _source(bundle, "det_curve_points", "det_curve_points"),
            ],
            (
                "blocked"
                if not (roc_data_ready and det_data_ready)
                else ("rebuildable_paper_claim" if full_ready else "rebuildable_preview")
            ),
            full_ready and roc_data_ready and det_data_ready,
            []
            if full_ready and roc_data_ready and det_data_ready
            else (["roc_det_complete_threshold_sweep_invalid"] if not (roc_data_ready and det_data_ready) else ["full_method_claim_ready_false"]),
        ),
        _artifact_row(
            "figure_attack_robustness",
            "figure_data",
            "攻击鲁棒性图数据",
            [
                _source(bundle, "attack_strength_curve", "attack_strength_curve"),
                _source(bundle, "attack_family_metrics", "attack_family_metrics"),
            ],
            (
                "blocked"
                if not attack_metrics_data_ready
                else ("rebuildable_paper_claim" if attack_ready else "rebuildable_preview")
            ),
            attack_ready,
            attack_blockers,
        ),
        _artifact_row(
            "figure_ablation_delta",
            "figure_data",
            "内部消融 delta 图数据",
            [_source(bundle, "method_pairwise_delta_table", "method_pairwise_delta_table")],
            (
                "blocked"
                if not ablation_delta_data_ready
                else ("rebuildable_paper_claim" if ablation_claim_ready else "rebuildable_preview")
            ),
            ablation_claim_ready,
            []
            if ablation_claim_ready
            else (["ablation_delta_data_invalid"] if not ablation_delta_data_ready else ["ablation_claim_gate_not_ready"]),
        ),
        _artifact_row(
            "figure_baseline_comparison",
            "figure_data",
            "外部 baseline 对比图数据",
            [_source(bundle, "baseline_comparison_table", "baseline_comparison_table.csv")],
            "rebuildable_paper_claim" if baseline_ready else "blocked",
            baseline_ready,
            []
            if baseline_ready
            else (["baseline_comparison_table_data_invalid"] if not baseline_table_data_ready else ["baseline_result_missing"]),
        ),
    ]


def build_evidence_gap_rows(bundle: AuditInputBundle) -> list[dict[str, Any]]:
    """构造投稿前证据缺口清单。"""
    attack = bundle.attack_manifest
    threshold = bundle.threshold_report
    baseline = bundle.baseline_runtime_report
    dataset_quality = bundle.dataset_quality_summary
    ablation = bundle.ablation_claim_summary
    rows: list[dict[str, Any]] = []
    if not _real_attack_closed_loop_ready(attack):
        rows.append(
            {
                "gap_id": "gap_real_attacked_image_closed_loop",
                "gap_area": "attack_matrix",
                "blocker_severity": "critical",
                "required_action": "生成真实 attacked image 文件, 记录 source / attacked image digest, 并重跑攻击后检测。",
                "related_artifacts": _source(
                    bundle,
                    "attacked_image_registry",
                    "attacked_image_registry",
                ),
                "closes_claim_ids": "claim_attack_robustness_under_common_matrix",
                "recommended_order": 1,
                "supports_paper_claim": False,
            }
        )
    if not _regeneration_attack_gpu_ready(attack):
        rows.append(
            {
                "gap_id": "gap_regeneration_attack_gpu_validation",
                "gap_area": "attack_matrix",
                "blocker_severity": "critical",
            "required_action": "在真实 GPU 环境补齐 img2img、flow-matching inversion、SDEdit 和 diffusion purification 攻击。",
                "related_artifacts": _source(
                    bundle,
                    "attack_family_metrics",
                    "attack_family_metrics",
                ),
                "closes_claim_ids": "claim_attack_robustness_under_common_matrix",
                "recommended_order": 2,
                "supports_paper_claim": False,
            }
        )
    if not _baseline_comparison_ready(
        baseline,
        table_data_ready=_artifact_data_ready(bundle, "baseline_comparison_table_ready"),
    ):
        rows.append(
            {
                "gap_id": "gap_baseline_results",
                "gap_area": "baseline_comparison",
                "blocker_severity": "critical",
                "required_action": "接入外部 baseline 官方代码复现结果或受治理导入结果, 并在共同协议下重建对比表。",
                "related_artifacts": _source(bundle, "baseline_comparison_table", "baseline_comparison_table.csv"),
                "closes_claim_ids": "claim_baseline_superiority",
                "recommended_order": 3,
                "supports_paper_claim": False,
            }
        )
    if not _runtime_sample_scale_ready(threshold):
        rows.append(
            {
                "gap_id": "gap_paper_run_sample_scale",
                "gap_area": "statistical_power",
                "blocker_severity": "critical",
                "required_action": "按当前运行层级完整执行 70/700/7000 Prompt, 并保持 34/340/3400 个 test Prompt 的冻结划分。",
                "related_artifacts": "image_only_dataset_runtime;formal_mechanism_ablation",
                "closes_claim_ids": "claim_full_method_fixed_fpr_boundary;claim_submission_ready_package",
                "recommended_order": 4,
                "supports_paper_claim": False,
            }
        )
    if not (
        _fixed_fpr_and_rescue_boundary_ready(threshold, attack)
        and _threshold_audit_ready(bundle)
    ):
        rows.append(
            {
                "gap_id": "gap_full_method_fixed_fpr_recalibration",
                "gap_area": "threshold_calibration",
                "blocker_severity": "major",
                "required_action": "在仅图像检测和真实攻击闭环上重新冻结包含几何救回的完整 fixed-FPR 判定。",
                "related_artifacts": _source(
                    bundle,
                    "threshold_audit_report",
                    "threshold_audit_report",
                ),
                "closes_claim_ids": "claim_full_method_fixed_fpr_boundary",
                "recommended_order": 5,
                "supports_paper_claim": False,
            },
        )
    if not (
        _yes(ablation.get("ablation_claim_gate_ready"))
        and _yes(ablation.get("supports_paper_claim"))
        and _artifact_data_ready(bundle, "mechanism_ablation_metrics_ready")
        and _artifact_data_ready(bundle, "mechanism_pairwise_delta_ready")
    ):
        rows.append(
            {
                "gap_id": "gap_formal_mechanism_ablation",
                "gap_area": "ablation",
                "blocker_severity": "major",
                "required_action": "对每个机制开关重新执行生成、攻击和仅图像检测, 禁止分数倍率或反事实变换。",
                "related_artifacts": _source(
                    bundle,
                    "ablation_claim_summary",
                    "ablation_claim_summary",
                ),
                "closes_claim_ids": "claim_internal_mechanism_necessity;claim_submission_ready_package",
                "recommended_order": 6,
                "supports_paper_claim": False,
            }
        )
    if not (
        _yes(dataset_quality.get("formal_fid_kid_ready"))
        and _artifact_data_ready(bundle, "dataset_quality_metrics_ready")
    ):
        rows.append(
            {
                "gap_id": "gap_dataset_level_fid_kid",
                "gap_area": "quality_metrics",
                "blocker_severity": "major",
                "required_action": "在完整 clean/watermarked 图像集合上使用正式 Inception 特征后端计算 FID / KID。",
                "related_artifacts": _source(bundle, "dataset_quality_metrics", "dataset_quality_metrics"),
                "closes_claim_ids": "claim_dataset_level_quality_boundary;claim_quality_preservation_pair_metrics",
                "recommended_order": 7,
                "supports_paper_claim": False,
            }
        )
    if bundle.artifact_data_validation and not _yes(
        bundle.artifact_data_validation.get("artifact_data_validation_ready")
    ):
        rows.append(
            {
                "gap_id": "gap_paper_artifact_source_data",
                "gap_area": "artifact_data_validation",
                "blocker_severity": "critical",
                "required_action": "补齐并修复实际论文表图数据文件, 重新验证列集合, 数值范围, 曲线端点与单调性.",
                "related_artifacts": ";".join(
                    str(value)
                    for value in bundle.artifact_data_validation.get("source_paths", {}).values()
                ),
                "closes_claim_ids": "claim_submission_ready_package",
                "recommended_order": 0,
                "supports_paper_claim": False,
            }
        )
    return rows


def _recommended_next_action(gap_rows: Iterable[dict[str, Any]]) -> str:
    """根据剩余缺口生成投稿前推进建议, 避免已经闭合的工程边界继续出现在建议中。"""
    gap_ids = {str(row["gap_id"]) for row in gap_rows}
    if {"gap_real_attacked_image_closed_loop", "gap_regeneration_attack_gpu_validation"} & gap_ids:
        return "先按 evidence_gap_list.csv 补齐真实攻击闭环、外部 baseline 结果和当前运行层级的完整统计, 再进入投稿冻结。"
    actions = []
    if "gap_paper_artifact_source_data" in gap_ids:
        actions.append("实际论文表图数据验证")
    if "gap_baseline_results" in gap_ids:
        actions.append("外部 baseline 结果")
    if "gap_paper_run_sample_scale" in gap_ids:
        actions.append("当前运行层级的完整统计")
    if "gap_full_method_fixed_fpr_recalibration" in gap_ids:
        actions.append("完整方法 fixed-FPR 重校准")
    if "gap_formal_mechanism_ablation" in gap_ids:
        actions.append("真实重运行机制消融")
    if "gap_dataset_level_fid_kid" in gap_ids:
        actions.append("dataset-level FID / KID")
    if not actions:
        return "重新运行投稿就绪门禁并审计 release dry-run 产物。"
    return f"先按 evidence_gap_list.csv 补齐{'、'.join(actions)}, 再进入投稿冻结。"


def build_builder_readiness_report(
    claim_rows: Iterable[dict[str, Any]],
    table_rows: Iterable[dict[str, Any]],
    figure_rows: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """构造 artifact builder readiness 摘要。"""
    claims = list(claim_rows)
    tables = list(table_rows)
    figures = list(figure_rows)
    artifacts = tables + figures
    rebuildable_count = sum(1 for row in artifacts if row["builder_status"] != "blocked")
    blocked_count = sum(1 for row in artifacts if row["builder_status"] == "blocked")
    paper_ready_count = sum(1 for row in artifacts if _yes(row["paper_ready"]))
    artifact_builder_ready = bool(artifacts) and blocked_count == 0
    paper_artifact_claim_ready = bool(artifacts) and paper_ready_count == len(artifacts)
    return {
        "construction_unit_name": "paper_artifact_evidence_audit",
        "artifact_builder_ready": artifact_builder_ready,
        "paper_artifact_claim_ready": paper_artifact_claim_ready,
        "paper_artifact_audit_ready": True,
        "claim_audit_row_count": len(claims),
        "table_readiness_row_count": len(tables),
        "figure_readiness_row_count": len(figures),
        "rebuildable_artifact_count": rebuildable_count,
        "blocked_artifact_count": blocked_count,
        "paper_ready_artifact_count": paper_ready_count,
        "supports_paper_claim": False,
    }


def build_submission_blocker_report(
    claim_rows: Iterable[dict[str, Any]],
    gap_rows: Iterable[dict[str, Any]],
    builder_report: dict[str, Any],
) -> dict[str, Any]:
    """构造投稿冻结阻断摘要。"""
    claims = list(claim_rows)
    gaps = list(gap_rows)
    critical_gaps = [row for row in gaps if row["blocker_severity"] == "critical"]
    blocking_claims = [row for row in claims if row["claim_decision"] in {"unsupported", "preview_only"}]
    recommended_next_action = _recommended_next_action(gaps)
    submission_ready = (
        not gaps
        and not blocking_claims
        and bool(builder_report.get("artifact_builder_ready"))
        and bool(builder_report.get("paper_artifact_claim_ready"))
    )
    full_method_claim_ready = any(
        row.get("claim_id") == "claim_full_method_fixed_fpr_boundary"
        and _yes(row.get("paper_claim_supported"))
        for row in claims
    )
    return {
        "construction_unit_name": "paper_artifact_evidence_audit",
        "submission_ready": submission_ready,
        "artifact_builder_ready": bool(builder_report.get("artifact_builder_ready")),
        "paper_artifact_claim_ready": bool(builder_report.get("paper_artifact_claim_ready")),
        "paper_artifact_audit_ready": True,
        "blocking_claim_count": len(blocking_claims),
        "critical_gap_count": len(critical_gaps),
        "gap_count": len(gaps),
        "primary_blockers": [row["gap_id"] for row in sorted(gaps, key=lambda item: int(item["recommended_order"]))[:4]],
        "recommended_next_action": recommended_next_action,
        "full_method_claim_ready": full_method_claim_ready,
        "supports_paper_claim": submission_ready,
    }
