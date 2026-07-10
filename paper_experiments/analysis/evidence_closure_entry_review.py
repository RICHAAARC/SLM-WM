"""论文投稿级证据闭合入口审计的判定逻辑。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class EvidenceClosureEntryInput:
    """封装证据闭合入口审计所需的受治理输入。

    该对象属于通用工程写法: 它把多个上游审计报告集中到只读数据结构中, 让判定函数只表达规则,
    避免在业务路径中重复读取文件和构造分散错误信息。
    """

    submission_readiness_report: dict[str, Any]
    required_evidence_rows: tuple[dict[str, Any], ...]
    paper_blocker_report: dict[str, Any]
    baseline_runtime_report: dict[str, Any]
    dataset_quality_summary: dict[str, Any]
    baseline_small_sample_summary: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """转换为 JSON 兼容字典, 便于 manifest 记录稳定摘要。"""

        return asdict(self)


def _bool_value(value: Any) -> bool:
    """把常见状态值转换为布尔值。"""

    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "pass", "ready", "measured"}
    return bool(value)


def _required_input_ids(rows: Iterable[Mapping[str, Any]]) -> set[str]:
    """提取当前仍需补齐的证据输入 id 集合。"""

    return {str(row.get("required_input_id", "")) for row in rows if str(row.get("required_input_id", ""))}


def _check_row(
    review_item_id: str,
    review_area: str,
    ready: bool,
    source_artifact: str,
    blocker_reason: str,
    user_audit_note: str,
) -> dict[str, Any]:
    """构造入口审计清单行。

    该函数属于通用可复用结构: 后续新增检查项时只需提供语义化 id、来源和阻断说明。
    """

    return {
        "review_item_id": review_item_id,
        "review_area": review_area,
        "review_status": "ready" if ready else "blocked",
        "source_artifact": source_artifact,
        "blocker_reason": "" if ready else blocker_reason,
        "user_audit_note": user_audit_note,
        "supports_paper_claim": False,
    }


def build_evidence_closure_entry_checklist(bundle: EvidenceClosureEntryInput) -> list[dict[str, Any]]:
    """构造进入论文投稿级证据闭合前的审计清单。

    此处设计的主要考虑在于: 当前工作只产生“是否允许进入证据闭合”的审计依据,
    不生成论文级主表、主图或 supported claim。
    """

    readiness = bundle.submission_readiness_report
    required_ids = _required_input_ids(bundle.required_evidence_rows)
    baseline = bundle.baseline_runtime_report
    dataset_quality = bundle.dataset_quality_summary
    small_sample = bundle.baseline_small_sample_summary
    return [
        _check_row(
            "upstream_audit_rebuildable",
            "artifact_rebuild",
            _bool_value(readiness.get("artifact_builder_ready")) and _bool_value(readiness.get("release_dry_run_ready")),
            "outputs/submission_readiness/readiness_blocker_report.json",
            "upstream_audit_or_release_dry_run_not_ready",
            "确认审计报告与 release dry-run 已可重建。",
        ),
        _check_row(
            "small_sample_boundary_preserved",
            "baseline_boundary",
            _bool_value(readiness.get("small_sample_baseline_boundary_ready"))
            and not _bool_value(small_sample.get("paper_claim_ready")),
            "outputs/primary_baseline_small_sample_evidence/primary_baseline_small_sample_evidence_summary.json",
            "small_sample_boundary_not_preserved",
            "确认当前小样本证据只作为工程链路审计, 不替代正式统计结论。",
        ),
        _check_row(
            "formal_comparison_reference_ready",
            "baseline_comparison",
            _bool_value(baseline.get("baseline_results_ready"))
            and _bool_value(baseline.get("formal_import_validation_ready"))
            and int(baseline.get("accepted_formal_import_count", 0)) > 0,
            "baseline_runtime_report",
            "formal_comparison_reference_results_missing",
            "确认主表对照方法已通过共同协议正式导入, 而不是仅有小样本或 smoke 记录。",
        ),
        _check_row(
            "full_main_sample_scale_ready",
            "statistical_power",
            "gap_full_main_sample_scale" not in required_ids,
            "outputs/submission_readiness/required_evidence_inputs.csv",
            "full_main_sample_scale_missing",
            "确认 full-main prompt split、样本量和随机种子已冻结并重建统计。",
        ),
        _check_row(
            "fixed_fpr_recalibration_ready",
            "threshold_calibration",
            "gap_full_method_fixed_fpr_recalibration" not in required_ids,
            "outputs/submission_readiness/required_evidence_inputs.csv",
            "fixed_fpr_recalibration_missing",
            "确认 fixed-FPR 与 rescue 边界已在正式证据上重新校准。",
        ),
        _check_row(
            "dataset_level_quality_ready",
            "quality_metrics",
            _bool_value(dataset_quality.get("formal_fid_kid_ready"))
            and _bool_value(dataset_quality.get("formal_sample_scale_ready"))
            and _bool_value(dataset_quality.get("formal_feature_backend_ready")),
            "outputs/dataset_level_quality/dataset_quality_summary.json",
            "dataset_level_fid_kid_missing",
            "确认 FID / KID 使用正式特征后端和足够样本量计算。",
        ),
        _check_row(
            "submission_readiness_ready",
            "submission_readiness",
            _bool_value(readiness.get("submission_ready")) and str(readiness.get("readiness_decision", "")) == "ready",
            "outputs/submission_readiness/readiness_blocker_report.json",
            "submission_readiness_blocked",
            "确认投稿就绪门禁已经给出 ready 判定。",
        ),
    ]


def build_evidence_closure_entry_review_report(
    bundle: EvidenceClosureEntryInput,
    checklist_rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """构造证据闭合入口审计报告。

    这一实现属于项目特定写法: 它显式把“可供用户审计”和“允许进入论文投稿级证据闭合”分开。
    当前如果仍存在关键缺口, 报告会到达可审计状态, 但不会允许进入证据闭合。
    """

    checklist = [dict(row) for row in checklist_rows]
    blocked_rows = [row for row in checklist if row.get("review_status") != "ready"]
    readiness = bundle.submission_readiness_report
    paper_blocker = bundle.paper_blocker_report
    baseline = bundle.baseline_runtime_report
    dataset_quality = bundle.dataset_quality_summary
    small_sample = bundle.baseline_small_sample_summary
    evidence_closure_allowed = not blocked_rows
    return {
        "construction_unit_name": "evidence_closure_entry_review",
        "entry_review_ready": True,
        "user_audit_required": True,
        "evidence_closure_allowed": evidence_closure_allowed,
        "entry_review_decision": "ready_for_user_audit" if evidence_closure_allowed else "blocked_before_evidence_closure",
        "review_item_count": len(checklist),
        "blocked_review_item_count": len(blocked_rows),
        "blocked_review_item_ids": [str(row.get("review_item_id", "")) for row in blocked_rows],
        "required_input_count": int(readiness.get("required_input_count", 0)),
        "critical_required_input_count": int(readiness.get("critical_required_input_count", 0)),
        "blocking_claim_count": int(paper_blocker.get("blocking_claim_count", 0)),
        "primary_blockers": list(readiness.get("primary_blockers", ())),
        "baseline_results_ready": _bool_value(baseline.get("baseline_results_ready")),
        "formal_import_validation_ready": _bool_value(baseline.get("formal_import_validation_ready")),
        "accepted_formal_import_count": int(baseline.get("accepted_formal_import_count", 0)),
        "formal_evidence_path_resolution_ready": _bool_value(
            baseline.get("formal_evidence_path_resolution_ready")
        ),
        "dataset_level_quality_proxy_ready": _bool_value(dataset_quality.get("dataset_level_quality_proxy_ready")),
        "formal_fid_kid_ready": _bool_value(dataset_quality.get("formal_fid_kid_ready")),
        "formal_sample_scale_ready": _bool_value(dataset_quality.get("formal_sample_scale_ready")),
        "formal_feature_backend_ready": _bool_value(dataset_quality.get("formal_feature_backend_ready")),
        "small_sample_baseline_boundary_ready": _bool_value(
            readiness.get("small_sample_baseline_boundary_ready")
        ),
        "small_sample_baseline_covered_count": int(
            readiness.get("small_sample_baseline_covered_count", 0)
        ),
        "formal_full_paper_run_requested": _bool_value(small_sample.get("formal_full_paper_run_requested")),
        "formal_full_paper_run_permitted": _bool_value(small_sample.get("formal_full_paper_run_permitted")),
        "excluded_operating_points": list(small_sample.get("excluded_operating_points", ())),
        "recommended_next_action": readiness.get("recommended_next_action", ""),
        "user_audit_question": "是否允许进入论文投稿级证据闭合, 需要由用户在审计本报告后决定。",
        "supports_paper_claim": False,
    }
