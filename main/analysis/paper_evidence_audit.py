"""论文图表与声明证据审计构造器。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class AuditInputBundle:
    """封装论文证据审计需要的上游摘要。

    该对象属于通用工程写法: 将跨产物的 manifest 和 report 汇总成统一输入,
    让后续审计函数只关注 claim、artifact 和 gap 的判定逻辑。
    """

    threshold_report: dict[str, Any]
    threshold_manifest: dict[str, Any]
    attack_manifest: dict[str, Any]
    attack_matrix_manifest: dict[str, Any]
    baseline_manifest: dict[str, Any]
    baseline_runtime_report: dict[str, Any]
    ablation_manifest: dict[str, Any]
    ablation_claim_summary: dict[str, Any]
    source_path_map: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        """转为 JSON 兼容字典。"""
        return asdict(self)


def _yes(value: Any) -> bool:
    """把 manifest 中常见的布尔或字符串状态转为布尔值。"""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"true", "pass", "ready", "measured"}
    return bool(value)


def _source(bundle: AuditInputBundle, key: str, fallback: str) -> str:
    """读取脚本层注入的受治理路径, 让核心分析逻辑不绑定外层目录命名。"""
    return bundle.source_path_map.get(key, fallback)


def _real_attack_closed_loop_ready(attack_manifest: dict[str, Any]) -> bool:
    """判断真实 attacked image 文件与摘要闭环是否已经进入正式 manifest。"""
    return (
        _yes(attack_manifest.get("real_attacked_image_closed_loop_ready"))
        and _yes(attack_manifest.get("formal_attack_detection_ready"))
        and int(attack_manifest.get("real_attacked_image_count", 0)) > 0
    )


def _regeneration_attack_gpu_ready(attack_manifest: dict[str, Any]) -> bool:
    """判断再扩散类攻击是否已经由真实 GPU formal records 覆盖。"""
    required_count = int(attack_manifest.get("required_regeneration_attack_count", 0))
    measured_count = int(attack_manifest.get("measured_regeneration_attack_count", 0))
    return _yes(attack_manifest.get("regeneration_attack_gpu_validation_ready")) and required_count > 0 and measured_count >= required_count


def _attack_robustness_blockers(attack_manifest: dict[str, Any]) -> list[str]:
    """生成攻击鲁棒性声明的当前阻断项。"""
    blockers = []
    if not _real_attack_closed_loop_ready(attack_manifest):
        blockers.append("attacked_image_files_missing")
    if not _regeneration_attack_gpu_ready(attack_manifest):
        blockers.append("regeneration_attack_real_gpu_missing")
    blockers.append("record_level_proxy_boundary")
    return blockers


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
        "supports_paper_claim": False,
    }


def build_claim_audit_rows(bundle: AuditInputBundle) -> list[dict[str, Any]]:
    """构造论文 claim 到证据路径的审计表。"""
    threshold = bundle.threshold_report
    attack = bundle.attack_manifest
    baseline = bundle.baseline_runtime_report
    ablation = bundle.ablation_claim_summary
    full_ready = _yes(threshold.get("full_method_claim_ready")) and _yes(attack.get("full_method_claim_ready"))
    return [
        _row(
            "claim_raw_content_fixed_fpr_boundary",
            "method_metric",
            "raw content 分支具备 fixed-FPR 校准边界。",
            "engineering_supported_not_paper_final",
            _source(bundle, "threshold_report", "outputs/threshold_calibration/threshold_degeneracy_report.json"),
            [] if _yes(threshold.get("raw_content_claim_ready")) else ["raw_content_claim_not_ready"],
        ),
        _row(
            "claim_full_method_fixed_fpr_boundary",
            "method_metric",
            "完整 SLM-WM 方法满足 fixed-FPR 统计边界。",
            "unsupported",
            _source(bundle, "threshold_report", "outputs/threshold_calibration/threshold_degeneracy_report.json"),
            [] if full_ready else ["full_method_claim_ready_false", "aligned_content_score_local_proxy"],
        ),
        _row(
            "claim_attack_robustness_under_common_matrix",
            "robustness",
            "SLM-WM 在共同攻击矩阵下具有稳健检测表现。",
            "preview_only",
            _source(bundle, "attack_manifest", "outputs/attack_matrix/attack_manifest.json"),
            _attack_robustness_blockers(attack),
        ),
        _row(
            "claim_baseline_superiority",
            "baseline_comparison",
            "SLM-WM 优于外部 watermark baseline。",
            "unsupported",
            _source(bundle, "baseline_runtime_report", "baseline_runtime_report.json"),
            [] if _yes(baseline.get("baseline_results_ready")) else ["baseline_result_missing"],
        ),
        _row(
            "claim_internal_mechanism_necessity",
            "ablation",
            "语义路由、安全子空间、LF/HF 载体、几何恢复和 attestation 均为必要机制。",
            "preview_only",
            _source(bundle, "ablation_claim_summary", "outputs/internal_ablation_evidence/ablation_claim_summary.json"),
            [] if _yes(ablation.get("mechanism_coverage_ready")) else ["mechanism_coverage_missing"],
        ),
        _row(
            "claim_quality_preservation_pair_metrics",
            "quality",
            "真实 aligned rescoring pair-level 质量指标可被下游审计。",
            "engineering_supported_not_paper_final",
            _source(bundle, "quality_metrics_summary", "outputs/threshold_calibration/quality_metrics_summary.csv"),
            [] if _yes(threshold.get("perceptual_metrics_ready")) else ["perceptual_metrics_missing"],
        ),
        _row(
            "claim_submission_ready_package",
            "submission_readiness",
            "当前仓库已具备投稿冻结所需的完整证据。",
            "unsupported",
            "outputs/paper_artifact_evidence_audit/submission_blocker_report.json",
            [
                "full_method_claim_ready_false",
                "baseline_result_missing",
                "" if _real_attack_closed_loop_ready(attack) and _regeneration_attack_gpu_ready(attack) else "real_attack_evidence_missing",
            ],
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
        "supports_paper_claim": False,
    }


def build_table_readiness_rows(bundle: AuditInputBundle) -> list[dict[str, Any]]:
    """构造论文表格 readiness 审计行。"""
    threshold = bundle.threshold_report
    attack = bundle.attack_manifest
    baseline = bundle.baseline_runtime_report
    ablation = bundle.ablation_claim_summary
    return [
        _artifact_row(
            "table_fixed_fpr_operating_points",
            "table",
            "fixed-FPR operating point 表",
            [
                _source(bundle, "fixed_fpr_operating_points", "outputs/threshold_calibration/fixed_fpr_operating_points.csv"),
                _source(bundle, "threshold_report", "outputs/threshold_calibration/threshold_degeneracy_report.json"),
            ],
            "rebuildable_preview",
            False,
            [] if not _yes(threshold.get("threshold_degenerate")) else ["threshold_degenerate"],
        ),
        _artifact_row(
            "table_main_method_metrics",
            "table",
            "主方法检测指标表",
            [_source(bundle, "standard_watermark_metrics", "outputs/threshold_calibration/standard_watermark_metrics.csv")],
            "rebuildable_preview",
            False,
            ["full_method_claim_ready_false"],
        ),
        _artifact_row(
            "table_attack_robustness",
            "table",
            "攻击鲁棒性表",
            [_source(bundle, "attack_family_metrics", "outputs/attack_matrix/attack_family_metrics.csv")],
            "rebuildable_preview",
            False,
            _attack_robustness_blockers(attack),
        ),
        _artifact_row(
            "table_baseline_comparison",
            "table",
            "外部 baseline 对比表",
            [_source(bundle, "baseline_comparison_table", "baseline_comparison_table.csv")],
            "protocol_ready_result_missing",
            False,
            [] if _yes(baseline.get("baseline_results_ready")) else ["baseline_result_missing"],
        ),
        _artifact_row(
            "table_internal_ablation",
            "table",
            "内部机制消融表",
            [_source(bundle, "mechanism_ablation_table", "outputs/internal_ablation_evidence/mechanism_ablation_table.csv")],
            "rebuildable_preview",
            False,
            [] if _yes(ablation.get("mechanism_coverage_ready")) else ["mechanism_coverage_missing"],
        ),
        _artifact_row(
            "table_quality_metrics",
            "table",
            "图像质量与感知指标表",
            [_source(bundle, "quality_metrics_summary", "outputs/threshold_calibration/quality_metrics_summary.csv")],
            "rebuildable_preview",
            False,
            ["fid_kid_dataset_metrics_missing"],
        ),
    ]


def build_figure_readiness_rows(bundle: AuditInputBundle) -> list[dict[str, Any]]:
    """构造论文图数据 readiness 审计行。"""
    baseline = bundle.baseline_runtime_report
    attack = bundle.attack_manifest
    return [
        _artifact_row(
            "figure_score_distribution",
            "figure_data",
            "score distribution 图数据",
            [_source(bundle, "score_distribution_table", "outputs/threshold_calibration/score_distribution_table.csv")],
            "rebuildable_preview",
            False,
            ["full_main_final_records_missing"],
        ),
        _artifact_row(
            "figure_roc_det",
            "figure_data",
            "ROC / DET 图数据",
            [
                _source(bundle, "roc_curve_points", "outputs/threshold_calibration/roc_curve_points.csv"),
                _source(bundle, "det_curve_points", "outputs/threshold_calibration/det_curve_points.csv"),
            ],
            "rebuildable_preview",
            False,
            ["full_method_claim_ready_false"],
        ),
        _artifact_row(
            "figure_attack_robustness",
            "figure_data",
            "攻击鲁棒性图数据",
            [
                _source(bundle, "attack_strength_curve", "outputs/attack_matrix/attack_strength_curve.csv"),
                _source(bundle, "score_retention_by_attack", "outputs/attack_matrix/score_retention_by_attack.csv"),
            ],
            "rebuildable_preview",
            False,
            _attack_robustness_blockers(attack),
        ),
        _artifact_row(
            "figure_ablation_delta",
            "figure_data",
            "内部消融 delta 图数据",
            [_source(bundle, "method_pairwise_delta_table", "outputs/internal_ablation_evidence/method_pairwise_delta_table.csv")],
            "rebuildable_preview",
            False,
            ["record_level_proxy_boundary"],
        ),
        _artifact_row(
            "figure_baseline_comparison",
            "figure_data",
            "外部 baseline 对比图数据",
            [_source(bundle, "baseline_comparison_table", "baseline_comparison_table.csv")],
            "blocked",
            False,
            [] if _yes(baseline.get("baseline_results_ready")) else ["baseline_result_missing"],
        ),
    ]


def build_evidence_gap_rows(bundle: AuditInputBundle) -> list[dict[str, Any]]:
    """构造投稿前证据缺口清单。"""
    attack = bundle.attack_manifest
    rows: list[dict[str, Any]] = []
    if not _real_attack_closed_loop_ready(attack):
        rows.append(
            {
                "gap_id": "gap_real_attacked_image_closed_loop",
                "gap_area": "attack_matrix",
                "blocker_severity": "critical",
                "required_action": "生成真实 attacked image 文件, 记录 source / attacked image digest, 并重跑攻击后检测。",
                "related_artifacts": "outputs/attack_matrix/attacked_images;outputs/attack_matrix/attacked_image_registry.jsonl",
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
                "required_action": "在真实 GPU 环境补齐 img2img、DDIM inversion、SDEdit 和 diffusion purification 攻击。",
                "related_artifacts": "outputs/attack_matrix/attack_family_metrics.csv",
                "closes_claim_ids": "claim_attack_robustness_under_common_matrix",
                "recommended_order": 2,
                "supports_paper_claim": False,
            }
        )
    rows.extend(
        [
        {
            "gap_id": "gap_baseline_results",
            "gap_area": "baseline_comparison",
            "blocker_severity": "critical",
            "required_action": "接入外部 baseline 官方代码复现结果或受治理导入结果, 并在共同协议下重建对比表。",
            "related_artifacts": _source(bundle, "baseline_comparison_table", "baseline_comparison_table.csv"),
            "closes_claim_ids": "claim_baseline_superiority",
            "recommended_order": 3,
            "supports_paper_claim": False,
        },
        {
            "gap_id": "gap_full_main_sample_scale",
            "gap_area": "statistical_power",
            "blocker_severity": "critical",
            "required_action": "冻结 full-main prompt split、样本量和随机种子, 重建 threshold、attack、baseline 与 ablation 统计。",
            "related_artifacts": "outputs/threshold_calibration;outputs/attack_matrix;outputs/internal_ablation_evidence",
            "closes_claim_ids": "claim_full_method_fixed_fpr_boundary;claim_submission_ready_package",
            "recommended_order": 4,
            "supports_paper_claim": False,
        },
        {
            "gap_id": "gap_full_method_fixed_fpr_recalibration",
            "gap_area": "threshold_calibration",
            "blocker_severity": "major",
            "required_action": "在真实 aligned content score 与真实攻击闭环完成后重新校准 fixed-FPR 和 rescue 边界。",
            "related_artifacts": "outputs/threshold_calibration/threshold_degeneracy_report.json",
            "closes_claim_ids": "claim_full_method_fixed_fpr_boundary",
            "recommended_order": 5,
            "supports_paper_claim": False,
        },
        {
            "gap_id": "gap_dataset_level_fid_kid",
            "gap_area": "quality_metrics",
            "blocker_severity": "major",
            "required_action": "在成组图像集合上计算 FID / KID, pair-level LPIPS / CLIP 不能替代数据集级指标。",
            "related_artifacts": "outputs/threshold_calibration/quality_metrics_summary.csv",
            "closes_claim_ids": "claim_quality_preservation_pair_metrics",
            "recommended_order": 6,
            "supports_paper_claim": False,
        },
        ]
    )
    return rows


def build_builder_readiness_report(
    claim_rows: Iterable[dict[str, Any]],
    table_rows: Iterable[dict[str, Any]],
    figure_rows: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """构造 artifact builder readiness 摘要。"""
    claims = list(claim_rows)
    tables = list(table_rows)
    figures = list(figure_rows)
    rebuildable_count = sum(1 for row in tables + figures if row["builder_status"] != "blocked")
    blocked_count = sum(1 for row in tables + figures if row["builder_status"] == "blocked")
    return {
        "construction_unit_name": "paper_artifact_evidence_audit",
        "artifact_builder_ready": rebuildable_count > 0,
        "paper_artifact_audit_ready": True,
        "claim_audit_row_count": len(claims),
        "table_readiness_row_count": len(tables),
        "figure_readiness_row_count": len(figures),
        "rebuildable_artifact_count": rebuildable_count,
        "blocked_artifact_count": blocked_count,
        "paper_ready_artifact_count": sum(1 for row in tables + figures if _yes(row["paper_ready"])),
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
    real_attack_gap_ids = {"gap_real_attacked_image_closed_loop", "gap_regeneration_attack_gpu_validation"}
    real_attack_gap_present = any(row["gap_id"] in real_attack_gap_ids for row in gaps)
    recommended_next_action = (
        "先按 evidence_gap_list.csv 补齐真实攻击闭环、外部 baseline 结果和 full-main 统计, 再进入投稿冻结。"
        if real_attack_gap_present
        else "先按 evidence_gap_list.csv 补齐外部 baseline 结果、full-main 统计、完整方法 fixed-FPR 重校准和 dataset-level FID / KID, 再进入投稿冻结。"
    )
    return {
        "construction_unit_name": "paper_artifact_evidence_audit",
        "submission_ready": False,
        "artifact_builder_ready": bool(builder_report.get("artifact_builder_ready")),
        "paper_artifact_audit_ready": True,
        "blocking_claim_count": len(blocking_claims),
        "critical_gap_count": len(critical_gaps),
        "gap_count": len(gaps),
        "primary_blockers": [row["gap_id"] for row in sorted(gaps, key=lambda item: int(item["recommended_order"]))[:4]],
        "recommended_next_action": recommended_next_action,
        "full_method_claim_ready": False,
        "supports_paper_claim": False,
    }
