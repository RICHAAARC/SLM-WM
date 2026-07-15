"""从精确随机化聚合包重建统计并发布自包含论文结果包.

该入口只消费通过生产 validator 复验的聚合 ZIP。五类统计 Writer 均从包内
原始记录重算结果, 随后本模块独立核对输出 manifest、文件摘要和共同来源
身份. 只有全部证据通过门禁后才会构造最终归档.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePath
import shutil
import subprocess
import sys
from typing import Any, Callable
from zipfile import ZIP_STORED, ZipFile

from experiments.artifacts.manifest_schema import manifest_config_digest_ready
from experiments.protocol.paper_run_config import (
    normalize_paper_run_name,
    validate_frozen_paper_run_target_fpr,
)
from experiments.runtime.repository_environment import resolve_code_version
from main.core.digest import build_stable_digest
from paper_experiments.runners.closure_package_selection import (
    normalize_clean_code_version,
)
from paper_experiments.runners.randomization_aggregate_provenance import (
    RandomizationAggregateProvenance,
    RandomizationAggregateProvenanceError,
    validate_randomization_aggregate_provenance,
)


CommandHook = Callable[[list[str]], None]
ProgressHook = Callable[[int, int, str], None]


@dataclass(frozen=True)
class _ClosureArtifactSpec:
    """登记一个必须由聚合原始记录重建的正式统计产物."""

    module_name: str
    output_root: str
    artifact_id: str
    ready_field: str
    summary_file_name: str
    file_names: tuple[str, ...]


_CLOSURE_ARTIFACT_SPECS = (
    _ClosureArtifactSpec(
        module_name=(
            "paper_experiments.runners.randomization_detection_statistics"
        ),
        output_root="outputs/randomization_detection_statistics",
        artifact_id="randomization_detection_statistics_manifest",
        ready_field="randomization_detection_statistics_ready",
        summary_file_name="randomization_detection_statistics_summary.json",
        file_names=(
            "prompt_cluster_detection_records.jsonl",
            "method_detection_operating_points.csv",
            "method_attack_detection_metrics.csv",
            "slm_wrong_key_detection_metric.csv",
            "per_attack_superiority_table.csv",
            "randomization_detection_statistics_summary.json",
            "randomization_detection_statistics_report.json",
            "manifest.local.json",
        ),
    ),
    _ClosureArtifactSpec(
        module_name=(
            "paper_experiments.runners.randomization_paired_superiority"
        ),
        output_root="outputs/randomization_paired_superiority",
        artifact_id="randomization_paired_superiority_manifest",
        ready_field="randomization_paired_statistics_ready",
        summary_file_name="paired_superiority_summary.json",
        file_names=(
            "method_repeat_threshold_records.jsonl",
            "paired_outcomes.jsonl",
            "quality_matching_records.jsonl",
            "paired_superiority_table.csv",
            "paired_superiority_summary.json",
            "randomization_paired_superiority_report.json",
            "manifest.local.json",
        ),
    ),
    _ClosureArtifactSpec(
        module_name="paper_experiments.runners.randomization_dataset_quality",
        output_root="outputs/randomization_dataset_quality",
        artifact_id="randomization_dataset_quality_manifest",
        ready_field="randomization_dataset_quality_statistics_ready",
        summary_file_name="randomization_dataset_quality_summary.json",
        file_names=(
            "fid_kid_metrics.csv",
            "quality_feature_membership.jsonl",
            "randomization_dataset_quality_summary.json",
            "randomization_dataset_quality_report.json",
            "manifest.local.json",
        ),
    ),
    _ClosureArtifactSpec(
        module_name=(
            "paper_experiments.runners.randomization_ablation_necessity"
        ),
        output_root="outputs/randomization_ablation_necessity",
        artifact_id="randomization_ablation_necessity_manifest",
        ready_field="randomization_aggregate_statistics_ready",
        summary_file_name="mechanism_necessity_summary.json",
        file_names=(
            "mechanism_necessity_statistics.csv",
            "mechanism_necessity_summary.json",
            "randomization_ablation_necessity_report.json",
            "manifest.local.json",
        ),
    ),
    _ClosureArtifactSpec(
        module_name=(
            "paper_experiments.runners.randomization_parameter_sensitivity"
        ),
        output_root=(
            "outputs/randomization_branch_risk_parameter_sensitivity"
        ),
        artifact_id=(
            "randomization_branch_risk_parameter_sensitivity_manifest"
        ),
        ready_field="parameter_sensitivity_aggregate_ready",
        summary_file_name="parameter_sensitivity_aggregate_summary.json",
        file_names=(
            "parameter_sensitivity_repeat_metrics.csv",
            "parameter_sensitivity_aggregate_metrics.csv",
            "parameter_sensitivity_aggregate_summary.json",
            "parameter_sensitivity_source_report.json",
            "manifest.local.json",
        ),
    ),
)

PAPER_RESULT_CLOSURE_COMMAND_COUNT = len(_CLOSURE_ARTIFACT_SPECS)
CLOSURE_RAW_OUTPUT_DIR_TEMPLATES: tuple[str, ...] = ()
CLOSURE_DERIVED_OUTPUT_DIR_TEMPLATES: tuple[str, ...] = tuple(
    f"{spec.output_root}/{{paper_run_name}}"
    for spec in _CLOSURE_ARTIFACT_SPECS
) + ("outputs/paper_result_closure/{paper_run_name}",)


def _file_sha256(path: Path) -> str:
    """流式计算文件 SHA-256."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bytes_sha256(payload: bytes) -> str:
    """计算内存字节 SHA-256."""

    return hashlib.sha256(payload).hexdigest()


def _read_json_object(path: Path) -> dict[str, Any]:
    """读取门禁所需 JSON 对象."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"结果闭合 JSON 不可读取: {path}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"结果闭合 JSON 根节点必须是对象: {path}")
    return payload


def _resolve_output_directory(
    root_path: Path,
    output_dir: str | Path,
) -> Path:
    """把最终归档目录限制在仓库 ``outputs/`` 内."""

    value = Path(output_dir).expanduser()
    resolved = (
        (root_path / value).resolve()
        if not value.is_absolute()
        else value.resolve()
    )
    try:
        resolved.relative_to((root_path / "outputs").resolve())
    except ValueError as exc:
        raise ValueError("论文结果闭合归档目录必须位于 outputs/ 下") from exc
    return resolved


def clean_paper_result_closure_outputs(
    *,
    root: str | Path,
    paper_run_name: str,
    selected_package_paths: Sequence[str | Path],
) -> tuple[str, ...]:
    """清理当前运行的派生统计, 且绝不删除锁定聚合来源."""

    root_path = Path(root).resolve()
    outputs_root = (root_path / "outputs").resolve()
    run_name = normalize_paper_run_name(paper_run_name)
    selected_paths = tuple(
        Path(path).expanduser().resolve() for path in selected_package_paths
    )
    managed_paths: list[Path] = []
    for template in CLOSURE_DERIVED_OUTPUT_DIR_TEMPLATES:
        managed_path = (
            root_path / template.format(paper_run_name=run_name)
        ).resolve()
        try:
            managed_path.relative_to(outputs_root)
        except ValueError as exc:
            raise ValueError("结果闭合清理路径必须位于 outputs/ 下") from exc
        if any(
            package_path == managed_path or managed_path in package_path.parents
            for package_path in selected_paths
        ):
            raise ValueError(
                f"锁定输入包位于受管清理目录内, 拒绝删除: {managed_path.as_posix()}"
            )
        managed_paths.append(managed_path)

    removed_paths: list[str] = []
    for managed_path in managed_paths:
        if managed_path.is_dir():
            shutil.rmtree(managed_path)
            removed_paths.append(managed_path.relative_to(root_path).as_posix())
        elif managed_path.exists():
            managed_path.unlink()
            removed_paths.append(managed_path.relative_to(root_path).as_posix())
    return tuple(removed_paths)


def _select_randomization_aggregate(
    package_search_root: str | Path,
    *,
    paper_run_name: str,
    target_fpr: float,
) -> RandomizationAggregateProvenance:
    """从显式文件或目录中选择唯一可复验的聚合 ZIP."""

    search_path = Path(package_search_root).expanduser().resolve()
    if search_path.is_file():
        candidates = (search_path,)
    elif search_path.is_dir():
        candidates = tuple(sorted(search_path.rglob("*.zip")))
    else:
        raise FileNotFoundError(f"聚合包搜索路径不存在: {search_path}")
    valid_sources: list[RandomizationAggregateProvenance] = []
    for candidate in candidates:
        try:
            valid_sources.append(
                validate_randomization_aggregate_provenance(
                    candidate,
                    paper_run_name=paper_run_name,
                    target_fpr=target_fpr,
                )
            )
        except (OSError, RandomizationAggregateProvenanceError, ValueError):
            continue
    if len(valid_sources) != 1:
        raise RuntimeError(
            "论文结果闭合要求唯一且完整的精确随机化聚合 ZIP;"
            f"实际通过验证数量={len(valid_sources)}"
        )
    return valid_sources[0]


def _commands_for_source(
    source: RandomizationAggregateProvenance,
    *,
    root_path: Path,
) -> list[list[str]]:
    """为同一不可变聚合来源构造五类统计重建命令。"""

    paper_run_name = str(source.payload["paper_run_name"])
    target_fpr = float(source.payload["target_fpr"])
    return [
        [
            sys.executable,
            "-m",
            spec.module_name,
            "--root",
            str(root_path),
            "--paper-run-name",
            paper_run_name,
            "--target-fpr",
            str(target_fpr),
            "--aggregate-package-path",
            str(source.package_path),
        ]
        for spec in _CLOSURE_ARTIFACT_SPECS
    ]


def build_paper_result_closure_commands(
    *,
    randomization_aggregate_package_path: str | Path,
    paper_run_name: str,
    target_fpr: float,
    root: str | Path = ".",
) -> list[list[str]]:
    """验证聚合来源并返回可脱离 Notebook 执行的统计命令."""

    run_name = normalize_paper_run_name(paper_run_name)
    frozen_target_fpr = validate_frozen_paper_run_target_fpr(
        run_name,
        target_fpr,
    )
    source = validate_randomization_aggregate_provenance(
        randomization_aggregate_package_path,
        paper_run_name=run_name,
        target_fpr=frozen_target_fpr,
    )
    return _commands_for_source(source, root_path=Path(root).resolve())


def _expected_artifact_paths(
    root_path: Path,
    spec: _ClosureArtifactSpec,
    paper_run_name: str,
) -> tuple[Path, ...]:
    """返回一个统计 Writer 必须精确发布的文件集合."""

    output_dir = root_path / spec.output_root / paper_run_name
    return tuple(output_dir / file_name for file_name in spec.file_names)


def _require_matching_summary_fields(
    summary: Mapping[str, Any],
    manifest_section: Mapping[str, Any],
    field_names: Sequence[str],
    *,
    artifact_id: str,
) -> None:
    """要求摘要中的正式判定字段与 manifest 声明逐项相同."""

    mismatched = [
        field_name
        for field_name in field_names
        if field_name not in summary
        or field_name not in manifest_section
        or summary[field_name] != manifest_section[field_name]
    ]
    if mismatched:
        raise RuntimeError(
            f"{artifact_id} 摘要与 manifest 判定字段不一致: {mismatched}"
        )


def _validate_embedded_summary_digest(
    summary: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    digest_field: str,
    artifact_id: str,
) -> None:
    """重算包含自摘要字段的正式摘要, 避免直接信任 Writer 声明."""

    summary_payload = dict(summary)
    stored_digest = summary_payload.pop(digest_field, None)
    if (
        not isinstance(stored_digest, str)
        or config.get(digest_field) != stored_digest
        or build_stable_digest(summary_payload) != stored_digest
    ):
        raise RuntimeError(f"{artifact_id} 正式摘要的摘要值无法独立重算")


def _require_summary_run_identity(
    summary: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    artifact_id: str,
) -> None:
    """要求统计摘要与 manifest 使用同一运行层级和冻结 FPR."""

    try:
        target_fpr_matches = float(summary.get("target_fpr", -1.0)) == float(
            config["target_fpr"]
        )
    except (KeyError, TypeError, ValueError):
        target_fpr_matches = False
    if (
        summary.get("paper_claim_scale") != config.get("paper_run_name")
        or not target_fpr_matches
    ):
        raise RuntimeError(f"{artifact_id} 摘要运行层级或冻结 FPR 不一致")


def _derive_detection_gate(
    summary: Mapping[str, Any],
    config: Mapping[str, Any],
    metadata: Mapping[str, Any],
    *,
    artifact_id: str,
) -> dict[str, Any]:
    """从统一阴性总体的 fixed-FPR 字段独立形成检测门禁."""

    _require_summary_run_identity(
        summary,
        config,
        artifact_id=artifact_id,
    )
    _validate_embedded_summary_digest(
        summary,
        config,
        digest_field="randomization_detection_statistics_summary_digest",
        artifact_id=artifact_id,
    )
    _require_matching_summary_fields(
        summary,
        metadata,
        (
            "randomization_detection_statistics_ready",
            "main_method_clean_fixed_fpr_ready",
            "main_method_wrong_key_fixed_fpr_ready",
            "all_per_attack_fixed_fpr_ready",
            "all_test_negative_populations_fixed_fpr_ready",
            "universal_per_attack_superiority_claim_ready",
            "supports_paper_claim",
        ),
        artifact_id=artifact_id,
    )
    _require_matching_summary_fields(
        summary,
        config,
        (
            "all_test_negative_populations_fixed_fpr_ready",
            "universal_per_attack_superiority_claim_ready",
        ),
        artifact_id=artifact_id,
    )
    individual_gate_fields = {
        "all_methods_clean_fixed_fpr_ready": summary.get(
            "all_methods_clean_fixed_fpr_ready"
        )
        is True,
        "main_method_wrong_key_fixed_fpr_ready": summary.get(
            "main_method_wrong_key_fixed_fpr_ready"
        )
        is True,
        "all_per_attack_fixed_fpr_ready": summary.get(
            "all_per_attack_fixed_fpr_ready"
        )
        is True,
    }
    independently_rebuilt_gate = all(individual_gate_fields.values())
    if (
        summary.get("randomization_detection_statistics_ready") is not True
        or summary.get("supports_paper_claim") is not False
        or summary.get("all_test_negative_populations_fixed_fpr_ready")
        is not independently_rebuilt_gate
    ):
        raise RuntimeError(f"{artifact_id} 检测摘要未保持统一阴性门禁语义")
    return {
        "component_role": "fixed_fpr_negative_population_gate",
        "component_evidence_ready": True,
        "central_claim_gate_ready": independently_rebuilt_gate,
        "component_decision": (
            "fixed_fpr_negative_populations_ready"
            if independently_rebuilt_gate
            else "fixed_fpr_negative_populations_not_ready"
        ),
        "gate_fields": individual_gate_fields,
    }


def _derive_paired_superiority_gate(
    summary: Mapping[str, Any],
    config: Mapping[str, Any],
    metadata: Mapping[str, Any],
    *,
    artifact_id: str,
) -> dict[str, Any]:
    """从全样本与质量匹配结果独立形成总体优势门禁."""

    _require_summary_run_identity(
        summary,
        config,
        artifact_id=artifact_id,
    )
    _validate_embedded_summary_digest(
        summary,
        config,
        digest_field="randomization_paired_superiority_summary_digest",
        artifact_id=artifact_id,
    )
    _require_matching_summary_fields(
        summary,
        metadata,
        (
            "randomization_paired_statistics_ready",
            "conclusion_decision",
            "supports_paper_claim",
        ),
        artifact_id=artifact_id,
    )
    if (
        summary.get("randomization_paired_statistics_ready") is not True
        or summary.get("quality_matched_statistics_ready") is not True
    ):
        raise RuntimeError(f"{artifact_id} 总体优势统计未完整重建")
    gate_fields = {
        "overall_paired_superiority_ready": summary.get(
            "overall_paired_superiority_ready"
        )
        is True,
        "overall_quality_matched_superiority_ready": summary.get(
            "overall_quality_matched_superiority_ready"
        )
        is True,
    }
    independently_rebuilt_gate = all(gate_fields.values())
    expected_decision = (
        "supported" if independently_rebuilt_gate else "measured_not_supported"
    )
    if (
        summary.get("conclusion_decision") != expected_decision
        or summary.get("supports_paper_claim") is not independently_rebuilt_gate
    ):
        raise RuntimeError(f"{artifact_id} 总体优势结论与显式门禁不一致")
    return {
        "component_role": "paired_quality_matched_superiority_gate",
        "component_evidence_ready": True,
        "central_claim_gate_ready": independently_rebuilt_gate,
        "component_decision": expected_decision,
        "gate_fields": gate_fields,
    }


def _derive_dataset_quality_component(
    summary: Mapping[str, Any],
    config: Mapping[str, Any],
    metadata: Mapping[str, Any],
    *,
    artifact_id: str,
) -> dict[str, Any]:
    """确认 FID/KID 是完整测量分量, 但不把它误作优势判定."""

    _require_summary_run_identity(
        summary,
        config,
        artifact_id=artifact_id,
    )
    _validate_embedded_summary_digest(
        summary,
        config,
        digest_field="randomization_dataset_quality_summary_digest",
        artifact_id=artifact_id,
    )
    _require_matching_summary_fields(
        summary,
        metadata,
        (
            "randomization_dataset_quality_statistics_ready",
            "conclusion_decision",
            "supports_paper_claim",
        ),
        artifact_id=artifact_id,
    )
    if not all(
        (
            summary.get("randomization_dataset_quality_statistics_ready") is True,
            summary.get("quality_metric_status") == "measured",
            summary.get("conclusion_decision") == "measured_evidence_component",
            summary.get("supports_paper_claim") is False,
        )
    ):
        raise RuntimeError(f"{artifact_id} FID/KID 测量分量未完整重建")
    return {
        "component_role": "dataset_quality_measurement",
        "component_evidence_ready": True,
        "contributes_to_central_claim_gate": False,
        "component_decision": "measured_evidence_component",
    }


def _derive_ablation_necessity_gate(
    summary: Mapping[str, Any],
    config: Mapping[str, Any],
    metadata: Mapping[str, Any],
    *,
    artifact_id: str,
) -> dict[str, Any]:
    """从真实重运行的机制必要性摘要独立形成消融门禁."""

    stored_digest = config.get("necessity_summary_digest")
    if (
        not isinstance(stored_digest, str)
        or build_stable_digest(dict(summary)) != stored_digest
    ):
        raise RuntimeError(f"{artifact_id} 机制必要性摘要无法独立重算")
    _require_matching_summary_fields(
        summary,
        metadata,
        (
            "randomization_aggregate_statistics_ready",
            "necessity_component_decision",
            "supports_paper_claim",
        ),
        artifact_id=artifact_id,
    )
    if (
        summary.get("randomization_aggregate_statistics_ready") is not True
        or summary.get("ablation_necessity_statistics_ready") is not True
    ):
        raise RuntimeError(f"{artifact_id} 机制必要性统计未完整重建")
    independently_rebuilt_gate = (
        summary.get("all_mechanism_necessity_components_supported") is True
    )
    expected_decision = (
        "measured_supported"
        if independently_rebuilt_gate
        else "measured_not_supported"
    )
    if (
        summary.get("necessity_component_decision") != expected_decision
        or summary.get("supports_paper_claim") is not independently_rebuilt_gate
    ):
        raise RuntimeError(f"{artifact_id} 机制必要性结论与显式门禁不一致")
    return {
        "component_role": "mechanism_necessity_gate",
        "component_evidence_ready": True,
        "central_claim_gate_ready": independently_rebuilt_gate,
        "component_decision": expected_decision,
        "gate_fields": {
            "all_mechanism_necessity_components_supported": (
                independently_rebuilt_gate
            ),
        },
    }


def _derive_parameter_sensitivity_component(
    summary: Mapping[str, Any],
    config: Mapping[str, Any],
    metadata: Mapping[str, Any],
    *,
    artifact_id: str,
) -> dict[str, Any]:
    """确认18项单模型内部敏感性已经由精确9重复完整测量。"""

    _require_summary_run_identity(
        summary,
        config,
        artifact_id=artifact_id,
    )
    _validate_embedded_summary_digest(
        summary,
        config,
        digest_field="parameter_sensitivity_summary_digest",
        artifact_id=artifact_id,
    )
    _require_matching_summary_fields(
        summary,
        metadata,
        (
            "parameter_sensitivity_aggregate_ready",
            "claim_boundary",
            "supports_paper_claim",
        ),
        artifact_id=artifact_id,
    )
    if not all(
        (
            summary.get("parameter_sensitivity_aggregate_ready") is True,
            summary.get("sensitivity_setting_count") == 18,
            summary.get("randomization_repeat_count") == 9,
            summary.get("sensitivity_model_scope")
            == "registered_primary_diffusion_model_only",
            summary.get("cross_model_evidence_provided") is False,
            summary.get("claim_boundary")
            == "single_model_internal_parameter_sensitivity_only",
            summary.get("supports_paper_claim") is True,
        )
    ):
        raise RuntimeError(f"{artifact_id} 单模型参数敏感性未完整重建")
    return {
        "component_role": "single_model_parameter_sensitivity",
        "component_evidence_ready": True,
        "contributes_to_central_claim_gate": False,
        "component_decision": "measured_single_model_sensitivity",
        "claim_boundary": summary["claim_boundary"],
    }


def _derive_artifact_component(
    spec: _ClosureArtifactSpec,
    summary: Mapping[str, Any],
    config: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """把五个冻结 Writer 映射到各自的显式论文判定规则。"""

    derivations = {
        "randomization_detection_statistics_manifest": _derive_detection_gate,
        "randomization_paired_superiority_manifest": (
            _derive_paired_superiority_gate
        ),
        "randomization_dataset_quality_manifest": (
            _derive_dataset_quality_component
        ),
        "randomization_ablation_necessity_manifest": (
            _derive_ablation_necessity_gate
        ),
        "randomization_branch_risk_parameter_sensitivity_manifest": (
            _derive_parameter_sensitivity_component
        ),
    }
    try:
        derivation = derivations[spec.artifact_id]
    except KeyError as exc:
        raise RuntimeError(f"未登记的论文统计组件: {spec.artifact_id}") from exc
    return derivation(
        summary,
        config,
        metadata,
        artifact_id=spec.artifact_id,
    )


def _validate_artifact_manifest(
    source: RandomizationAggregateProvenance,
    *,
    root_path: Path,
    spec: _ClosureArtifactSpec,
) -> dict[str, Any]:
    """独立核对一个统计产物的来源身份、文件集合与逐文件摘要."""

    paper_run_name = str(source.payload["paper_run_name"])
    expected_paths = _expected_artifact_paths(root_path, spec, paper_run_name)
    manifest_path = expected_paths[-1]
    manifest = _read_json_object(manifest_path)
    config = manifest.get("config")
    metadata = manifest.get("metadata")
    if not isinstance(config, Mapping) or not isinstance(metadata, Mapping):
        raise RuntimeError(f"{spec.artifact_id} 缺少结构化 config 或 metadata")
    expected_relative_paths = {
        path.relative_to(root_path).as_posix() for path in expected_paths
    }
    output_paths = manifest.get("output_paths")
    if (
        not isinstance(output_paths, list)
        or len(output_paths) != len(expected_relative_paths)
        or set(output_paths) != expected_relative_paths
    ):
        raise RuntimeError(f"{spec.artifact_id} 输出文件集合不完整")
    output_sha256 = metadata.get("output_sha256")
    expected_data_paths = expected_paths[:-1]
    if not isinstance(output_sha256, Mapping) or set(output_sha256) != {
        path.relative_to(root_path).as_posix() for path in expected_data_paths
    }:
        raise RuntimeError(f"{spec.artifact_id} 输出摘要登记不完整")
    identity_ready = all(
        (
            manifest.get("artifact_id") == spec.artifact_id,
            manifest.get("code_version") == source.common_code_version,
            manifest_config_digest_ready(manifest),
            config.get("paper_run_name") == paper_run_name,
            float(config.get("target_fpr", -1.0))
            == float(source.payload["target_fpr"]),
            config.get("randomization_aggregate_package_sha256")
            == source.package_sha256,
            config.get("randomization_aggregate_digest")
            == source.randomization_aggregate_digest,
            config.get("common_code_version") == source.common_code_version,
            metadata.get(spec.ready_field) is True,
            isinstance(metadata.get("supports_paper_claim"), bool),
            manifest.get("input_paths") == [source.package_path.as_posix()],
        )
    )
    if not identity_ready:
        raise RuntimeError(f"{spec.artifact_id} 未绑定同一聚合来源或正式门禁")
    for path in expected_data_paths:
        relative_path = path.relative_to(root_path).as_posix()
        if not path.is_file() or _file_sha256(path) != output_sha256[relative_path]:
            raise RuntimeError(f"{spec.artifact_id} 文件摘要不匹配: {relative_path}")
    summary_path = manifest_path.parent / spec.summary_file_name
    summary = _read_json_object(summary_path)
    component_record = _derive_artifact_component(
        spec,
        summary,
        config,
        metadata,
    )
    return {
        "artifact_id": spec.artifact_id,
        "manifest_path": manifest_path.relative_to(root_path).as_posix(),
        "manifest_sha256": _file_sha256(manifest_path),
        "summary_path": summary_path.relative_to(root_path).as_posix(),
        "summary_sha256": _file_sha256(summary_path),
        "output_sha256": dict(output_sha256),
        "supports_paper_claim": bool(metadata["supports_paper_claim"]),
        "conclusion_decision": str(metadata.get("conclusion_decision", "")),
        **component_record,
    }


def _write_closure_gate(
    source: RandomizationAggregateProvenance,
    *,
    root_path: Path,
) -> tuple[Path, Path, dict[str, Any]]:
    """全部统计产物通过独立核对后发布闭合门禁报告."""

    artifact_records = [
        _validate_artifact_manifest(source, root_path=root_path, spec=spec)
        for spec in _CLOSURE_ARTIFACT_SPECS
    ]
    claim_gate_records = {
        str(record["component_role"]): bool(record["central_claim_gate_ready"])
        for record in artifact_records
        if "central_claim_gate_ready" in record
    }
    required_claim_gate_roles = {
        "fixed_fpr_negative_population_gate",
        "paired_quality_matched_superiority_gate",
        "mechanism_necessity_gate",
    }
    if set(claim_gate_records) != required_claim_gate_roles:
        raise RuntimeError("论文中心结论门禁组件集合不完整")
    measurement_component_roles = {
        str(record["component_role"])
        for record in artifact_records
        if record.get("component_evidence_ready") is True
        and record.get("contributes_to_central_claim_gate") is False
    }
    required_measurement_component_roles = {
        "dataset_quality_measurement",
        "single_model_parameter_sensitivity",
    }
    if measurement_component_roles != required_measurement_component_roles:
        raise RuntimeError("论文必要测量证据组件集合不完整")
    supports_paper_claim = all(claim_gate_records.values())
    conclusion_decision = (
        "supported" if supports_paper_claim else "measured_not_supported"
    )
    paper_run_name = str(source.payload["paper_run_name"])
    output_dir = root_path / "outputs" / "paper_result_closure" / paper_run_name
    if output_dir.exists():
        raise RuntimeError("论文结果闭合门禁目录已存在, 不得混选旧运行")
    output_dir.mkdir(parents=True)
    report_core = {
        "report_schema": "paper_result_closure_gate_report",
        "paper_run_name": paper_run_name,
        "target_fpr": float(source.payload["target_fpr"]),
        "randomization_aggregate_package_sha256": source.package_sha256,
        "randomization_aggregate_digest": source.randomization_aggregate_digest,
        "common_code_version": source.common_code_version,
        "artifact_records": artifact_records,
        "claim_gate_records": claim_gate_records,
        "measurement_component_roles": sorted(
            measurement_component_roles
        ),
        "unsupported_claim_gate_roles": sorted(
            role for role, ready in claim_gate_records.items() if not ready
        ),
        "statistics_rebuilt_from_aggregate": True,
        "paper_result_evidence_ready": True,
        "conclusion_decision": conclusion_decision,
        "supports_paper_claim": supports_paper_claim,
    }
    report = {
        **report_core,
        "paper_result_closure_gate_digest": build_stable_digest(report_core),
    }
    report_path = output_dir / "paper_result_closure_gate_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest_core = {
        "artifact_id": "paper_result_closure_manifest",
        "artifact_type": "local_manifest",
        "input_paths": [source.package_path.as_posix()],
        "output_paths": [
            report_path.relative_to(root_path).as_posix(),
            (output_dir / "manifest.local.json").relative_to(root_path).as_posix(),
        ],
        "code_version": source.common_code_version,
        "config": {
            "paper_run_name": paper_run_name,
            "target_fpr": float(source.payload["target_fpr"]),
            "randomization_aggregate_package_sha256": source.package_sha256,
            "randomization_aggregate_digest": source.randomization_aggregate_digest,
            "artifact_manifest_sha256": {
                record["artifact_id"]: record["manifest_sha256"]
                for record in artifact_records
            },
        },
        "metadata": {
            "paper_result_evidence_ready": True,
            "report_sha256": _file_sha256(report_path),
            "conclusion_decision": conclusion_decision,
            "supports_paper_claim": supports_paper_claim,
        },
    }
    manifest_without_digest = {
        **manifest_core,
        "config_digest": build_stable_digest(manifest_core["config"]),
    }
    manifest = {
        **manifest_without_digest,
        "manifest_digest": build_stable_digest(manifest_without_digest),
    }
    manifest_path = output_dir / "manifest.local.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report_path, manifest_path, report


def _archive_name(value: str) -> str:
    """要求最终归档名称是单个语义化 ZIP 文件名."""

    if PurePath(value).name != value or not value.endswith(".zip"):
        raise ValueError("论文结果归档名称必须是单个 .zip 文件名")
    return value


def _write_complete_archive(
    source: RandomizationAggregateProvenance,
    *,
    root_path: Path,
    complete_output_dir: Path,
    archive_name: str,
    gate_report_path: Path,
) -> Path:
    """把聚合来源、重建统计与门禁报告封装为自包含结果包."""

    paper_run_name = str(source.payload["paper_run_name"])
    gate_report = _read_json_object(gate_report_path)
    if (
        gate_report.get("paper_result_evidence_ready") is not True
        or not isinstance(gate_report.get("supports_paper_claim"), bool)
        or gate_report.get("paper_run_name") != paper_run_name
        or float(gate_report.get("target_fpr", -1.0))
        != float(source.payload["target_fpr"])
    ):
        raise RuntimeError("论文结果闭合报告尚未形成可归档判定")
    supports_paper_claim = bool(gate_report["supports_paper_claim"])
    conclusion_decision = str(gate_report.get("conclusion_decision", ""))
    source_paths: list[Path] = []
    for spec in _CLOSURE_ARTIFACT_SPECS:
        source_paths.extend(_expected_artifact_paths(root_path, spec, paper_run_name))
    gate_dir = gate_report_path.parent
    source_paths.extend(sorted(path for path in gate_dir.iterdir() if path.is_file()))
    member_payloads: dict[str, tuple[Path, str]] = {
        path.relative_to(root_path).as_posix(): (path, _file_sha256(path))
        for path in source_paths
    }
    aggregate_member = "inputs/randomization_aggregate_provenance.zip"
    member_payloads[aggregate_member] = (
        source.package_path,
        source.package_sha256,
    )
    archive_manifest_core = {
        "report_schema": "paper_complete_result_archive_manifest",
        "paper_run_name": paper_run_name,
        "target_fpr": float(source.payload["target_fpr"]),
        "randomization_aggregate_package_sha256": source.package_sha256,
        "randomization_aggregate_digest": source.randomization_aggregate_digest,
        "common_code_version": source.common_code_version,
        "entry_sha256": {
            member_name: digest
            for member_name, (_path, digest) in sorted(member_payloads.items())
        },
        "paper_result_evidence_ready": True,
        "conclusion_decision": conclusion_decision,
        "supports_paper_claim": supports_paper_claim,
    }
    archive_manifest = {
        **archive_manifest_core,
        "archive_manifest_digest": build_stable_digest(archive_manifest_core),
    }
    archive_manifest_bytes = (
        json.dumps(
            archive_manifest,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    manifest_member = "paper_complete_result_archive_manifest.json"
    complete_output_dir.mkdir(parents=True, exist_ok=True)
    destination = complete_output_dir / _archive_name(archive_name)
    if destination.exists():
        raise FileExistsError(f"论文结果归档已存在: {destination}")
    temporary_path = destination.with_name(destination.name + ".partial")
    try:
        with ZipFile(temporary_path, "x", compression=ZIP_STORED) as archive:
            for member_name, (path, _digest) in sorted(member_payloads.items()):
                archive.write(path, member_name)
            archive.writestr(manifest_member, archive_manifest_bytes)
        with ZipFile(temporary_path) as archive:
            expected_names = {*member_payloads, manifest_member}
            if set(archive.namelist()) != expected_names:
                raise RuntimeError("论文结果归档成员集合不完整")
            for member_name, (_path, digest) in member_payloads.items():
                if _bytes_sha256(archive.read(member_name)) != digest:
                    raise RuntimeError(f"论文结果归档成员摘要不匹配: {member_name}")
            if archive.read(manifest_member) != archive_manifest_bytes:
                raise RuntimeError("论文结果归档 manifest 字节不匹配")
        os.replace(temporary_path, destination)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    return destination


def run_paper_result_closure_commands(
    *,
    package_search_root: str | Path,
    complete_drive_output_dir: str | Path,
    paper_run_name: str,
    target_fpr: float,
    expected_repository_commit: str,
    root: str | Path = ".",
    archive_name: str | None = None,
    before_command: CommandHook | None = None,
    progress_hook: ProgressHook | None = None,
) -> dict[str, Any]:
    """执行聚合选择、统计重建、证据门禁和最终归档."""

    root_path = Path(root).resolve()
    run_name = normalize_paper_run_name(paper_run_name)
    frozen_target_fpr = validate_frozen_paper_run_target_fpr(
        run_name,
        target_fpr,
    )
    source = _select_randomization_aggregate(
        package_search_root,
        paper_run_name=run_name,
        target_fpr=frozen_target_fpr,
    )
    expected_commit = normalize_clean_code_version(expected_repository_commit)
    if resolve_code_version(root_path) != expected_commit:
        raise RuntimeError("汇总执行仓库必须位于聚合包对应的 clean Git 提交")
    if source.common_code_version != expected_commit:
        raise RuntimeError("聚合包代码提交与汇总服务器提交不一致")
    complete_output_dir = _resolve_output_directory(
        root_path,
        complete_drive_output_dir,
    )
    resolved_archive_name = _archive_name(
        archive_name or f"{run_name}_complete_result_package.zip"
    )
    if (complete_output_dir / resolved_archive_name).exists():
        raise FileExistsError("论文结果归档已存在, 不得覆盖既有运行")
    removed_paths = clean_paper_result_closure_outputs(
        root=root_path,
        paper_run_name=run_name,
        selected_package_paths=(source.package_path,),
    )
    commands = _commands_for_source(source, root_path=root_path)
    for index, (spec, command) in enumerate(
        zip(_CLOSURE_ARTIFACT_SPECS, commands, strict=True),
        start=1,
    ):
        if before_command is not None:
            before_command(command)
        subprocess.run(command, cwd=root_path, check=True, shell=False)
        if progress_hook is not None:
            progress_hook(index, len(commands), spec.artifact_id)
    gate_report_path, gate_manifest_path, gate_report = _write_closure_gate(
        source,
        root_path=root_path,
    )
    archive_path = _write_complete_archive(
        source,
        root_path=root_path,
        complete_output_dir=complete_output_dir,
        archive_name=resolved_archive_name,
        gate_report_path=gate_report_path,
    )
    return {
        "paper_run_name": run_name,
        "target_fpr": frozen_target_fpr,
        "randomization_aggregate_package_path": source.package_path.as_posix(),
        "randomization_aggregate_package_sha256": source.package_sha256,
        "randomization_aggregate_digest": source.randomization_aggregate_digest,
        "common_code_version": source.common_code_version,
        "removed_output_paths": list(removed_paths),
        "statistics_command_count": len(commands),
        "gate_report_path": gate_report_path.relative_to(root_path).as_posix(),
        "gate_manifest_path": gate_manifest_path.relative_to(root_path).as_posix(),
        "paper_result_evidence_ready": gate_report["paper_result_evidence_ready"],
        "archive_path": archive_path.as_posix(),
        "archive_sha256": _file_sha256(archive_path),
        "conclusion_decision": gate_report["conclusion_decision"],
        "supports_paper_claim": gate_report["supports_paper_claim"],
    }


__all__ = [
    "PAPER_RESULT_CLOSURE_COMMAND_COUNT",
    "build_paper_result_closure_commands",
    "clean_paper_result_closure_outputs",
    "run_paper_result_closure_commands",
]
