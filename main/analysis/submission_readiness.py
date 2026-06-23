"""投稿就绪门禁的证据汇总逻辑。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class SubmissionReadinessInput:
    """封装投稿就绪判定需要的受治理输入。

    该对象属于通用工程写法: 将证据审计产物、阻断报告、证据缺口和 release dry-run 摘要集中成一个只读输入,
    让后续函数只表达判定规则, 不直接读取文件系统。
    """

    evidence_manifest: dict[str, Any]
    builder_report: dict[str, Any]
    blocker_report: dict[str, Any]
    evidence_gaps: tuple[dict[str, Any], ...]
    release_profiles: tuple[dict[str, Any], ...]
    baseline_small_sample_summary: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """转换为 JSON 兼容字典, 便于生成 manifest 摘要。"""
        return asdict(self)


def _bool_value(value: Any) -> bool:
    """把常见字符串状态转换为布尔值。"""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "pass", "ready", "measured"}
    return bool(value)


def _small_sample_baseline_boundary_ready(summary: dict[str, Any]) -> bool:
    """判定小样本 baseline 证据是否只在受限边界内可审计。

    该函数属于项目特定写法: 它不是判断论文级 full paper 结论是否成立, 而是检查当前小样本 baseline
    是否已经覆盖主表候选方法、共同协议边界和 fixed-FPR 边界, 同时显式保持 full paper 统计声明关闭。
    """
    return (
        _bool_value(summary.get("small_sample_evidence_ready"))
        and _bool_value(summary.get("small_sample_common_protocol_ready"))
        and int(summary.get("covered_primary_baseline_count", 0)) >= 4
        and not _bool_value(summary.get("paper_claim_ready"))
        and not _bool_value(summary.get("supports_paper_claim"))
        and not _bool_value(summary.get("formal_full_paper_run_requested"))
        and not _bool_value(summary.get("formal_full_paper_run_permitted"))
    )


def _small_sample_baseline_limitations(summary: dict[str, Any]) -> list[str]:
    """生成小样本 baseline 证据边界说明, 避免将链路测试误读为正式论文统计。"""
    excluded_points = list(summary.get("excluded_operating_points", ()))
    limitations = [
        "当前 baseline 证据仅覆盖小样本共同协议边界, 不支持正式 full paper 统计声明。",
    ]
    if excluded_points:
        limitations.append(f"已显式排除的操作点包括: {', '.join(str(point) for point in excluded_points)}。")
    if not _small_sample_baseline_boundary_ready(summary):
        limitations.append("小样本 baseline 共同协议边界尚未完全就绪, 不能关闭正式 baseline 结果缺口。")
    return limitations


def build_required_evidence_rows(bundle: SubmissionReadinessInput) -> list[dict[str, Any]]:
    """把证据缺口转换为投稿就绪门禁所需的输入清单。"""
    rows: list[dict[str, Any]] = []
    for gap in sorted(bundle.evidence_gaps, key=lambda item: int(item.get("recommended_order", 0))):
        rows.append(
            {
                "required_input_id": gap.get("gap_id", ""),
                "required_input_area": gap.get("gap_area", ""),
                "required_input_severity": gap.get("blocker_severity", ""),
                "required_action": gap.get("required_action", ""),
                "related_artifacts": gap.get("related_artifacts", ""),
                "closes_claim_ids": gap.get("closes_claim_ids", ""),
                "recommended_order": int(gap.get("recommended_order", 0)),
                "input_ready": False,
                "supports_paper_claim": False,
            }
        )
    return rows


def build_release_profile_rows(bundle: SubmissionReadinessInput) -> list[dict[str, Any]]:
    """根据 release dry-run 摘要构造发布范围审计行。"""
    rows: list[dict[str, Any]] = []
    submission_ready = _bool_value(bundle.blocker_report.get("submission_ready"))
    for profile in bundle.release_profiles:
        copied_files = tuple(profile.get("copied_files", ()))
        missing_paths = tuple(profile.get("missing_paths", ()))
        dry_run_ready = bool(copied_files) and not missing_paths and _bool_value(profile.get("dry_run"))
        rows.append(
            {
                "release_profile_name": profile.get("profile_name", ""),
                "release_profile_file_count": len(copied_files),
                "release_profile_missing_count": len(missing_paths),
                "release_dry_run_ready": dry_run_ready,
                "release_package_allowed": submission_ready and dry_run_ready,
                "package_freeze_allowed": submission_ready and dry_run_ready,
                "release_scope": "dry_run_only" if not submission_ready else "candidate_package",
                "supports_paper_claim": False,
            }
        )
    return rows


def build_submission_readiness_report(
    bundle: SubmissionReadinessInput,
    required_rows: Iterable[dict[str, Any]],
    release_rows: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """构造投稿就绪门禁报告。

    此处设计的主要考虑在于: 只要上游 evidence audit 未允许投稿冻结, 本报告就只能给出 blocked 判定,
    不能因为 release dry-run 可运行而升级为投稿就绪。
    """
    required = list(required_rows)
    releases = list(release_rows)
    critical_rows = [row for row in required if row["required_input_severity"] == "critical"]
    blocker_submission_ready = _bool_value(bundle.blocker_report.get("submission_ready"))
    artifact_builder_ready = _bool_value(bundle.builder_report.get("artifact_builder_ready"))
    release_dry_run_ready = bool(releases) and all(_bool_value(row["release_dry_run_ready"]) for row in releases)
    small_sample_summary = bundle.baseline_small_sample_summary
    small_sample_boundary_ready = _small_sample_baseline_boundary_ready(small_sample_summary)
    package_freeze_allowed = blocker_submission_ready and release_dry_run_ready and not required
    readiness_decision = "ready" if package_freeze_allowed else "blocked"
    return {
        "construction_unit_name": "submission_readiness_gate",
        "readiness_decision": readiness_decision,
        "submission_ready": package_freeze_allowed,
        "package_freeze_allowed": package_freeze_allowed,
        "artifact_builder_ready": artifact_builder_ready,
        "release_dry_run_ready": release_dry_run_ready,
        "required_input_count": len(required),
        "critical_required_input_count": len(critical_rows),
        "release_profile_count": len(releases),
        "blocking_claim_count": int(bundle.blocker_report.get("blocking_claim_count", 0)),
        "paper_ready_artifact_count": int(bundle.builder_report.get("paper_ready_artifact_count", 0)),
        "small_sample_baseline_evidence_ready": _bool_value(small_sample_summary.get("small_sample_evidence_ready")),
        "small_sample_baseline_common_protocol_ready": _bool_value(
            small_sample_summary.get("small_sample_common_protocol_ready")
        ),
        "small_sample_baseline_boundary_ready": small_sample_boundary_ready,
        "small_sample_baseline_covered_count": int(small_sample_summary.get("covered_primary_baseline_count", 0)),
        "small_sample_baseline_formal_import_ready_count": int(small_sample_summary.get("formal_import_ready_count", 0)),
        "formal_full_paper_run_requested": _bool_value(small_sample_summary.get("formal_full_paper_run_requested")),
        "formal_full_paper_run_permitted": _bool_value(small_sample_summary.get("formal_full_paper_run_permitted")),
        "excluded_operating_points": list(small_sample_summary.get("excluded_operating_points", ())),
        "primary_blockers": [row["required_input_id"] for row in required[:4]],
        "recommended_next_action": bundle.blocker_report.get(
            "recommended_next_action",
            "先补齐 evidence audit 列出的关键证据缺口, 再重新运行投稿就绪门禁。",
        ),
        "limitations": [
            "当前报告只审计投稿冻结边界, 不生成论文级主表或主图。",
            "release dry-run 可运行不等价于投稿就绪。",
        ]
        + _small_sample_baseline_limitations(small_sample_summary),
        "supports_paper_claim": False,
    }
