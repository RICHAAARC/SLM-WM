"""主表外部 baseline 的官方复现计划与受治理导入模板。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from experiments.protocol.pilot_paper_fixed_fpr import PILOT_PAPER_FIXED_FPR
from main.core.digest import build_stable_digest

PRIMARY_BASELINE_IDS = ("tree_ring", "gaussian_shading", "shallow_diffuse", "t2smark")


@dataclass(frozen=True)
class PrimaryBaselineCommandProfile:
    """描述主表 baseline 的官方入口命令和依赖边界。

    该对象属于通用工程写法: 把第三方源码的运行入口, 依赖约束和结果适配类型集中登记, 避免在后续
    notebook 或脚本中散落手写命令。
    """

    baseline_id: str
    command_profile_name: str
    dependency_profile: str
    recommended_python: str
    official_entrypoint: str
    reproduction_command: str
    expected_result_adapter: str
    model_alignment_status: str
    notes: str

    def to_dict(self) -> dict[str, Any]:
        """转换为 JSON 兼容字典。"""
        return asdict(self)


@dataclass(frozen=True)
class PrimaryBaselineExecutionPlan:
    """记录一个主表 baseline 的可审计复现计划。"""

    baseline_execution_id: str
    baseline_execution_digest: str
    baseline_id: str
    baseline_name: str
    baseline_family: str
    comparison_group: str
    source_dir: str
    source_status: str
    source_entry_ready: bool
    official_repository_url: str
    official_repository_commit: str
    official_repository_branch: str
    dependency_profile: str
    recommended_python: str
    official_entrypoint: str
    reproduction_command: str
    expected_result_adapter: str
    model_alignment_status: str
    result_import_required: bool
    supports_paper_claim: bool
    unsupported_reason: str

    def to_dict(self) -> dict[str, Any]:
        """转换为 JSON 兼容字典。"""
        return asdict(self)


@dataclass(frozen=True)
class PrimaryBaselineResultTemplate:
    """描述主表 baseline 需要补齐的一条共同协议结果模板。

    模板不是结果, 只声明官方复现完成后必须填入的共同协议键和指标字段。
    """

    result_record_template_id: str
    result_record_template_digest: str
    baseline_id: str
    attack_family: str
    attack_name: str
    resource_profile: str
    comparable_operating_point: str
    required_metric_fields: tuple[str, ...]
    required_source_fields: tuple[str, ...]
    metric_status: str
    supports_paper_claim: bool

    def to_dict(self) -> dict[str, Any]:
        """转换为 JSON 兼容字典。"""
        data = asdict(self)
        data["required_metric_fields"] = list(self.required_metric_fields)
        data["required_source_fields"] = list(self.required_source_fields)
        return data


def default_primary_command_profiles() -> dict[str, PrimaryBaselineCommandProfile]:
    """返回主表 baseline 的官方入口命令画像。"""
    return {
        "tree_ring": PrimaryBaselineCommandProfile(
            baseline_id="tree_ring",
            command_profile_name="tree_ring_official_text_to_image",
            dependency_profile="legacy_diffusers_ddim_inversion",
            recommended_python="3.8",
            official_entrypoint="run_tree_ring_watermark.py",
            reproduction_command=(
                "python run_tree_ring_watermark.py --run_name slm_common_protocol "
                "--w_channel 3 --w_pattern ring --start 0 --end {sample_count} --with_tracking"
            ),
            expected_result_adapter="tree_ring_detection_metrics_adapter",
            model_alignment_status="sd35_method_faithful_adapter_plus_legacy_official_reference",
            notes=(
                "主表使用 method_faithful_sd35 adapter 在 SD3.5 Medium 上重建 ring key 注入与检测; "
                "补充表使用官方 legacy 入口隔离复现, 再以 governed import 记录忠实度参考。"
            ),
        ),
        "gaussian_shading": PrimaryBaselineCommandProfile(
            baseline_id="gaussian_shading",
            command_profile_name="gaussian_shading_official_text_to_image",
            dependency_profile="legacy_diffusers_ddim_inversion",
            recommended_python="3.8",
            official_entrypoint="run_gaussian_shading.py",
            reproduction_command=(
                "python run_gaussian_shading.py --fpr {target_fpr} --channel_copy 1 --hw_copy 8 "
                "--chacha --num {sample_count}"
            ),
            expected_result_adapter="gaussian_shading_detection_metrics_adapter",
            model_alignment_status="legacy_stable_diffusion_requires_protocol_adapter",
            notes="官方入口使用 Stable Diffusion 2.x latent 尺寸, 需记录与 SD3.5 主线的模型边界。",
        ),
        "shallow_diffuse": PrimaryBaselineCommandProfile(
            baseline_id="shallow_diffuse",
            command_profile_name="shallow_diffuse_official_text_to_image",
            dependency_profile="legacy_diffusers_ddim_inversion",
            recommended_python="3.8",
            official_entrypoint="run_shallow_diffuse_t2i.py",
            reproduction_command=(
                "python run_shallow_diffuse_t2i.py --run_name slm_common_protocol --w_pattern complex2_ring "
                "--start 0 --end {sample_count} --w_channel 3 --w_mask_shape circle "
                "--w_radius 10 --w_measurement l1_complex2 --w_injection complex2 --edit_time_list 0.3"
            ),
            expected_result_adapter="shallow_diffuse_detection_metrics_adapter",
            model_alignment_status="legacy_stable_diffusion_requires_protocol_adapter",
            notes="官方入口与 Tree-Ring 共享 Stable Diffusion 2.x 依赖边界, 需用共同 Prompt 和攻击矩阵重跑。",
        ),
        "t2smark": PrimaryBaselineCommandProfile(
            baseline_id="t2smark",
            command_profile_name="t2smark_official_sd35_medium",
            dependency_profile="sd35_diffusers_runtime",
            recommended_python="3.10",
            official_entrypoint="run_sd35.py",
            reproduction_command="python run_sd35.py --name slm_common_protocol",
            expected_result_adapter="t2smark_detection_metrics_adapter",
            model_alignment_status="sd35_medium_native_entrypoint",
            notes="官方源码提供 SD3.5 Medium 入口, 可优先作为主表 baseline 的真实 GPU 复现对象。",
        ),
    }


def _source_items(source_registry: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    """把来源登记列表转换为 baseline_id 索引。"""
    return {str(item.get("baseline_id", "")): item for item in source_registry.get("baseline_sources", ())}


def _relative_source_present(root: Path, source_dir: str) -> bool:
    """检查源码目录是否存在且包含文件。"""
    if not source_dir:
        return False
    path = root / source_dir
    return path.exists() and any(path.iterdir())


def _operating_point(boundary: Mapping[str, Any]) -> str:
    """根据 fixed-FPR 边界生成共同 operating point 名称。"""
    target_fpr = float(boundary.get("target_fpr", PILOT_PAPER_FIXED_FPR))
    return f"fixed_fpr_{target_fpr:g}"


def build_primary_baseline_execution_plans(
    source_registry: Mapping[str, Any],
    root: str | Path = ".",
) -> tuple[dict[str, Any], ...]:
    """根据源码登记文件构造主表 baseline 复现计划。"""
    root_path = Path(root).resolve()
    source_by_id = _source_items(source_registry)
    command_profiles = default_primary_command_profiles()
    rows: list[dict[str, Any]] = []
    for baseline_id in PRIMARY_BASELINE_IDS:
        source = source_by_id.get(baseline_id, {})
        profile = command_profiles[baseline_id]
        source_dir = str(source.get("source_dir", ""))
        source_entry_ready = _relative_source_present(root_path, source_dir) and bool(profile.official_entrypoint)
        payload = {
            "baseline_id": baseline_id,
            "source_dir": source_dir,
            "source_status": str(source.get("source_status", "not_registered")),
            "official_repository_commit": str(source.get("official_repository_commit", "")),
            "command_profile": profile.to_dict(),
        }
        digest = build_stable_digest(payload)
        plan = PrimaryBaselineExecutionPlan(
            baseline_execution_id=f"primary_baseline_execution_{digest[:16]}",
            baseline_execution_digest=digest,
            baseline_id=baseline_id,
            baseline_name=str(source.get("baseline_name", "")),
            baseline_family=str(source.get("baseline_family", "")),
            comparison_group=str(source.get("comparison_group", "primary")),
            source_dir=source_dir,
            source_status=str(source.get("source_status", "not_registered")),
            source_entry_ready=source_entry_ready,
            official_repository_url=str(source.get("official_repository_url", "")),
            official_repository_commit=str(source.get("official_repository_commit", "")),
            official_repository_branch=str(source.get("official_repository_branch", "")),
            dependency_profile=profile.dependency_profile,
            recommended_python=profile.recommended_python,
            official_entrypoint=profile.official_entrypoint,
            reproduction_command=profile.reproduction_command,
            expected_result_adapter=profile.expected_result_adapter,
            model_alignment_status=profile.model_alignment_status,
            result_import_required=True,
            supports_paper_claim=False,
            unsupported_reason="" if source_entry_ready else "official_source_entry_missing",
        )
        rows.append(plan.to_dict())
    return tuple(rows)


def build_primary_result_templates(
    execution_plans: Iterable[Mapping[str, Any]],
    attack_metric_rows: Iterable[Mapping[str, Any]],
    boundary: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    """构造主表 baseline 需要补齐的共同协议结果模板。"""
    operating_point = _operating_point(boundary)
    metric_fields = (
        "positive_count",
        "negative_count",
        "attack_record_count",
        "supported_record_count",
        "true_positive_rate",
        "false_positive_rate",
        "clean_false_positive_rate",
        "attacked_false_positive_rate",
        "quality_score_mean",
    )
    source_fields = (
        "baseline_result_source",
        "baseline_result_source_digest",
        "result_protocol_name",
        "result_source_type",
    )
    rows: list[dict[str, Any]] = []
    for plan in execution_plans:
        if str(plan.get("comparison_group")) != "primary":
            continue
        for attack_row in attack_metric_rows:
            payload = {
                "baseline_id": plan["baseline_id"],
                "attack_family": str(attack_row.get("attack_family", "")),
                "attack_name": str(attack_row.get("attack_name", "")),
                "resource_profile": str(attack_row.get("resource_profile", "")),
                "comparable_operating_point": operating_point,
            }
            digest = build_stable_digest(payload)
            rows.append(
                PrimaryBaselineResultTemplate(
                    result_record_template_id=f"primary_baseline_result_template_{digest[:16]}",
                    result_record_template_digest=digest,
                    baseline_id=str(plan["baseline_id"]),
                    attack_family=payload["attack_family"],
                    attack_name=payload["attack_name"],
                    resource_profile=payload["resource_profile"],
                    comparable_operating_point=operating_point,
                    required_metric_fields=metric_fields,
                    required_source_fields=source_fields,
                    metric_status="result_required",
                    supports_paper_claim=False,
                ).to_dict()
            )
    return tuple(rows)


def build_primary_baseline_report(
    execution_plans: Iterable[Mapping[str, Any]],
    result_templates: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """聚合主表 baseline 复现计划与导入模板状态。"""
    plans = list(execution_plans)
    templates = list(result_templates)
    source_ready_count = sum(1 for row in plans if row.get("source_entry_ready"))
    return {
        "construction_unit_name": "primary_baseline_reproduction",
        "primary_baseline_count": len(plans),
        "primary_source_ready_count": source_ready_count,
        "result_record_template_count": len(templates),
        "primary_baseline_plan_ready": bool(plans) and source_ready_count == len(plans),
        "result_import_template_ready": bool(templates),
        "baseline_results_ready": False,
        "full_method_claim_ready": False,
        "supports_paper_claim": False,
    }
