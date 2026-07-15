"""从精确9重复聚合来源重建单模型风险参数敏感性证据。

该模块不读取单重复派生 CSV。它从每个 leaf 包的逐 prompt 运行记录、检测原子、
冻结阈值协议和 manifest 重新核验18项设置, 再以注册 repeat 均值作为独立统计
单位构造分布无关置信区间。该证据只支持单模型内部参数敏感性结论。
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping

from experiments.ablations.branch_risk_sensitivity import (
    FORMAL_BRANCH_RISK_SENSITIVITY_IDS,
    FORMAL_BRANCH_RISK_SENSITIVITY_SPEC_DIGEST,
    default_branch_risk_sensitivity_specs,
)
from experiments.ablations.branch_risk_sensitivity_runtime import (
    _sensitivity_formal_record,
)
from experiments.ablations.runtime_rerun import _formal_attack_coverage_ready
from experiments.artifacts.artifact_manifest import build_artifact_manifest
from experiments.protocol.formal_randomization import formal_randomization_repeat_ids
from experiments.protocol.image_only_evidence import (
    FrozenEvidenceProtocol,
    apply_frozen_evidence_protocol,
    validate_frozen_evidence_protocol_integrity,
)
from experiments.protocol.paper_fixed_fpr import bounded_hoeffding_confidence_interval
from experiments.protocol.paper_run_config import (
    RUN_EXPECTED_PROMPT_COUNTS,
    normalize_paper_run_name,
)
from experiments.runtime.repository_environment import resolve_code_version
from experiments.runners.semantic_watermark_runtime import (
    SemanticWatermarkRuntimeConfig,
    validate_semantic_watermark_runtime_result_provenance,
)
from main.core.digest import build_stable_digest
from paper_experiments.runners.randomization_aggregate_provenance import (
    RandomizationAggregateProvenance,
    validate_randomization_aggregate_provenance,
)
from paper_experiments.runners.randomization_aggregate_record_workspace import (
    RandomizationAggregateRecordSource,
    open_randomization_aggregate_record_workspace,
)
from paper_experiments.runners.randomization_prompt_source_contract import (
    rebuild_randomization_prompt_source_contract,
)


RANDOMIZATION_PARAMETER_SENSITIVITY_REPORT_SCHEMA = (
    "randomization_branch_risk_parameter_sensitivity_report"
)
RANDOMIZATION_PARAMETER_SENSITIVITY_OUTPUT_ROOT = (
    "outputs/randomization_branch_risk_parameter_sensitivity"
)
_CONFIDENCE_LEVEL = 0.95
_METRIC_SPECS = (
    ("clean_false_positive_rate", "clean_negative_positive", 0.0, 1.0),
    (
        "wrong_key_false_positive_rate",
        "wrong_key_negative_positive",
        0.0,
        1.0,
    ),
    ("clean_true_positive_rate", "positive_source_positive", 0.0, 1.0),
    ("attacked_true_positive_rate", "attacked_positive_rate", 0.0, 1.0),
    ("attacked_false_positive_rate", "attacked_negative_rate", 0.0, 1.0),
    ("paired_ssim_mean", "paired_ssim", -1.0, 1.0),
)


class RandomizationParameterSensitivityError(ValueError):
    """表示跨重复参数敏感性证据不能形成唯一正式统计。"""


@dataclass(frozen=True)
class RandomizationParameterSensitivityResult:
    """保存逐重复指标、聚合指标、摘要和来源报告。"""

    repeat_rows: tuple[Mapping[str, Any], ...]
    aggregate_rows: tuple[Mapping[str, Any], ...]
    summary: Mapping[str, Any]
    report: Mapping[str, Any]


def _materialize(value: Any) -> Any:
    """把工作区只读视图转换为普通 JSON 值。"""

    if isinstance(value, Mapping):
        return {str(key): _materialize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_materialize(item) for item in value]
    return value


def _source_record(source: RandomizationAggregateRecordSource) -> dict[str, str]:
    """记录统计重建使用的 leaf 成员与逐层摘要。"""

    return {
        "record_role": source.record_role,
        "record_member": source.record_member,
        "record_sha256": source.record_sha256,
        "leaf_package_sha256": source.leaf_package_sha256,
        "randomization_repeat_component_sha256": (
            source.randomization_repeat_component_sha256
        ),
    }


def _require_provenance(source: RandomizationAggregateProvenance) -> None:
    """只接受生产 validator 返回的精确9重复来源对象。"""

    if not isinstance(source, RandomizationAggregateProvenance):
        raise TypeError("参数敏感性重建只接受 RandomizationAggregateProvenance")
    payload = source.payload
    if not all(
        (
            payload.get("randomization_aggregate_ready") is True,
            payload.get("supports_paper_claim") is False,
            payload.get("randomization_aggregate_digest")
            == source.randomization_aggregate_digest,
            payload.get("common_code_version") == source.common_code_version,
            tuple(payload.get("randomization_repeat_ids", ()))
            == formal_randomization_repeat_ids(),
        )
    ):
        raise RandomizationParameterSensitivityError(
            "聚合来源对象未保持 validator 冻结身份"
        )


def _mean(values: list[float]) -> float:
    """计算经过非空检查的数值均值。"""

    if not values:
        raise RandomizationParameterSensitivityError("统计集合不得为空")
    return sum(values) / len(values)


def _rebuild_repeat_records(
    runtime_records: tuple[dict[str, Any], ...],
    detection_records: tuple[dict[str, Any], ...],
    frozen_protocols: dict[str, Any],
    *,
    repeat_id: str,
    prompt_split_by_id: Mapping[str, str],
    prompt_digest_by_id: Mapping[str, str],
    prompt_index_by_id: Mapping[str, int],
    expected_sensitivity_config_by_id: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    """从检测原子重新冻结阈值并复核逐 prompt 敏感性记录。"""

    expected_ids = set(FORMAL_BRANCH_RISK_SENSITIVITY_IDS)
    if set(frozen_protocols) != expected_ids:
        raise RandomizationParameterSensitivityError(
            f"敏感性冻结协议未覆盖18项设置: {repeat_id}"
        )
    protocols: dict[str, FrozenEvidenceProtocol] = {}
    for sensitivity_id, payload in frozen_protocols.items():
        if not isinstance(payload, dict):
            raise RandomizationParameterSensitivityError("冻结协议必须是完整对象")
        protocol = FrozenEvidenceProtocol(**payload)
        validate_frozen_evidence_protocol_integrity(protocol)
        protocols[sensitivity_id] = protocol

    detections_by_key: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for record in detection_records:
        key = (
            str(record.get("sensitivity_id", "")),
            str(record.get("sensitivity_prompt_id", "")),
        )
        if key[0] not in expected_ids or key[1] not in prompt_split_by_id:
            raise RandomizationParameterSensitivityError(
                "敏感性检测原子包含未知设置或 prompt"
            )
        detections_by_key.setdefault(key, []).append(record)

    rebuilt_records: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, str]] = set()
    for record in runtime_records:
        sensitivity_id = str(record.get("sensitivity_id", ""))
        prompt_id = str(record.get("prompt_id", ""))
        key = (sensitivity_id, prompt_id)
        if key in seen_keys:
            raise RandomizationParameterSensitivityError(
                "敏感性运行记录包含重复 setting × prompt"
            )
        seen_keys.add(key)
        if (
            sensitivity_id not in expected_ids
            or prompt_id not in prompt_split_by_id
            or record.get("split") != prompt_split_by_id[prompt_id]
            or record.get("prompt_digest") != prompt_digest_by_id[prompt_id]
            or int(record.get("prompt_index", -1)) != prompt_index_by_id[prompt_id]
            or record.get("generation_rerun") is not True
            or record.get("runtime_result", {}).get("run_decision") != "pass"
            or record.get("sensitivity_config")
            != expected_sensitivity_config_by_id.get(sensitivity_id)
        ):
            raise RandomizationParameterSensitivityError(
                "敏感性运行记录未绑定 prompt 或真实重运行身份"
            )
        raw_detections = detections_by_key.get(key)
        if not raw_detections:
            raise RandomizationParameterSensitivityError(
                "敏感性运行记录缺少检测原子"
            )
        protocol = protocols[sensitivity_id]
        detections = apply_frozen_evidence_protocol(raw_detections, protocol)
        rebuilt = _sensitivity_formal_record(
            record,
            detections,
            protocol.threshold_digest,
            protocol.content_threshold,
        )
        expected_attack_ready = _formal_attack_coverage_ready(
            detections,
            split=str(record["split"]),
            expected_generation_seed_random=int(
                record["runtime_result"]["metadata"][
                    "formal_randomization_reference"
                ]["generation_seed_random"]
            ),
            expected_threshold_digest=protocol.threshold_digest,
        )
        if record.get("formal_attack_coverage_ready") is not expected_attack_ready:
            raise RandomizationParameterSensitivityError(
                "敏感性攻击覆盖门禁不能从检测原子重建"
            )
        comparable = dict(record)
        comparable.pop("formal_attack_coverage_ready", None)
        if comparable != rebuilt:
            raise RandomizationParameterSensitivityError(
                "敏感性逐 prompt 记录不能从检测原子与冻结协议重建"
            )
        rebuilt_records.append(
            {
                **rebuilt,
                "formal_attack_coverage_ready": expected_attack_ready,
                "randomization_repeat_id": repeat_id,
            }
        )
    expected_keys = {
        (sensitivity_id, prompt_id)
        for sensitivity_id in FORMAL_BRANCH_RISK_SENSITIVITY_IDS
        for prompt_id in prompt_split_by_id
    }
    if seen_keys != expected_keys or set(detections_by_key) != expected_keys:
        raise RandomizationParameterSensitivityError(
            "敏感性记录未精确覆盖18项设置与全部 prompt"
        )
    return tuple(rebuilt_records)


def _repeat_metric_rows(
    records: tuple[dict[str, Any], ...],
    *,
    expected_test_count: int,
) -> tuple[dict[str, Any], ...]:
    """按 repeat 与设置汇总 prompt 级原始指标。"""

    rows: list[dict[str, Any]] = []
    for repeat_id in formal_randomization_repeat_ids():
        for sensitivity_id in FORMAL_BRANCH_RISK_SENSITIVITY_IDS:
            group = [
                record
                for record in records
                if record["randomization_repeat_id"] == repeat_id
                and record["sensitivity_id"] == sensitivity_id
                and record["split"] == "test"
            ]
            if len(group) != expected_test_count:
                raise RandomizationParameterSensitivityError(
                    "每个 repeat × setting 必须覆盖完整 test prompt"
                )
            row: dict[str, Any] = {
                "randomization_repeat_id": repeat_id,
                "sensitivity_id": sensitivity_id,
                "test_prompt_count": len(group),
            }
            for metric_name, record_field, _lower, _upper in _METRIC_SPECS:
                row[metric_name] = _mean(
                    [float(record[record_field]) for record in group]
                )
            rows.append(row)
    return tuple(rows)


def _validate_manifest_unit_configs(
    runtime_records: tuple[dict[str, Any], ...],
    unit_identity_records: Any,
    *,
    repeat_id: str,
    paper_run_name: str,
    expected_sensitivity_config_by_id: Mapping[str, Mapping[str, Any]],
) -> None:
    """复验每条运行记录的完整科学配置与真实参数设置。"""

    if not isinstance(unit_identity_records, list):
        raise RandomizationParameterSensitivityError(
            "敏感性 manifest 缺少逐运行科学身份"
        )
    identities = {
        str(record.get("run_id", "")): record
        for record in unit_identity_records
        if isinstance(record, dict)
    }
    if len(identities) != len(runtime_records):
        raise RandomizationParameterSensitivityError(
            "敏感性逐运行科学身份数量不一致"
        )
    risk_fields = (
        "lf_content_risk_config",
        "tail_robust_risk_config",
        "attention_geometry_risk_config",
    )
    for record in runtime_records:
        runtime_result = record.get("runtime_result", {})
        identity = identities.get(str(runtime_result.get("run_id", "")))
        unit_config = (
            identity.get("scientific_unit_config")
            if isinstance(identity, dict)
            else None
        )
        sensitivity_id = str(record.get("sensitivity_id", ""))
        expected_config = expected_sensitivity_config_by_id.get(
            sensitivity_id
        )
        expected_risk_configs = (
            expected_config.get("resolved_branch_risk_configs")
            if isinstance(expected_config, Mapping)
            else None
        )
        if not isinstance(unit_config, dict):
            raise RandomizationParameterSensitivityError(
                "敏感性逐运行科学配置缺失"
            )
        validate_semantic_watermark_runtime_result_provenance(
            runtime_result,
            unit_config=unit_config,
        )
        if not all(
            (
                unit_config.get("randomization_repeat_id") == repeat_id,
                unit_config.get("risk_parameter_protocol")
                == "single_model_internal_sensitivity",
                unit_config.get("output_dir")
                == (
                    "outputs/formal_branch_risk_sensitivity/"
                    f"{paper_run_name}/runs/{sensitivity_id}"
                ),
                isinstance(expected_risk_configs, Mapping),
                all(
                    unit_config.get(field_name)
                    == expected_risk_configs.get(field_name)
                    for field_name in risk_fields
                )
                if isinstance(expected_risk_configs, Mapping)
                else False,
            )
        ):
            raise RandomizationParameterSensitivityError(
                "敏感性逐运行配置未绑定真实参数或 repeat 身份"
            )


def _aggregate_metric_rows(
    repeat_rows: tuple[dict[str, Any], ...],
) -> tuple[dict[str, Any], ...]:
    """以9个注册 repeat 均值构造设置指标与对参考设置的配对差值。"""

    rows_by_key = {
        (row["randomization_repeat_id"], row["sensitivity_id"]): row
        for row in repeat_rows
    }
    aggregate_rows: list[dict[str, Any]] = []
    repeat_count = len(formal_randomization_repeat_ids())
    for sensitivity_id in FORMAL_BRANCH_RISK_SENSITIVITY_IDS:
        row: dict[str, Any] = {
            "sensitivity_id": sensitivity_id,
            "randomization_repeat_count": repeat_count,
            "confidence_interval_method": "bounded_hoeffding_repeat_mean",
            "confidence_level": _CONFIDENCE_LEVEL,
            "single_model_internal_sensitivity": True,
        }
        for metric_name, _record_field, lower, upper in _METRIC_SPECS:
            values = [
                float(rows_by_key[(repeat_id, sensitivity_id)][metric_name])
                for repeat_id in formal_randomization_repeat_ids()
            ]
            mean_value = _mean(values)
            ci_low, ci_high = bounded_hoeffding_confidence_interval(
                mean_value,
                repeat_count,
                _CONFIDENCE_LEVEL,
                lower_bound=lower,
                upper_bound=upper,
            )
            row[metric_name] = mean_value
            row[f"{metric_name}_ci_low"] = ci_low
            row[f"{metric_name}_ci_high"] = ci_high
            if sensitivity_id != "formal_reference":
                delta_values = [
                    float(rows_by_key[(repeat_id, sensitivity_id)][metric_name])
                    - float(
                        rows_by_key[(repeat_id, "formal_reference")][metric_name]
                    )
                    for repeat_id in formal_randomization_repeat_ids()
                ]
                delta_mean = _mean(delta_values)
                delta_low, delta_high = bounded_hoeffding_confidence_interval(
                    delta_mean,
                    repeat_count,
                    _CONFIDENCE_LEVEL,
                    lower_bound=lower - upper,
                    upper_bound=upper - lower,
                )
                row[f"{metric_name}_delta"] = delta_mean
                row[f"{metric_name}_delta_ci_low"] = delta_low
                row[f"{metric_name}_delta_ci_high"] = delta_high
        aggregate_rows.append(row)
    return tuple(aggregate_rows)


def rebuild_randomization_parameter_sensitivity(
    source: RandomizationAggregateProvenance,
    *,
    root: str | Path = ".",
) -> RandomizationParameterSensitivityResult:
    """重建精确9重复的单模型内部参数敏感性统计。"""

    _require_provenance(source)
    repository_root = Path(root).resolve()
    if resolve_code_version(repository_root) != source.common_code_version:
        raise RandomizationParameterSensitivityError(
            "参数敏感性分析必须使用与聚合来源相同的 clean Git 提交"
        )
    paper_run_name = normalize_paper_run_name(
        str(source.payload.get("paper_run_name", ""))
    )
    target_fpr = float(source.payload.get("target_fpr", float("nan")))
    expected_prompt_count = RUN_EXPECTED_PROMPT_COUNTS[paper_run_name]
    reference_config = SemanticWatermarkRuntimeConfig()
    expected_specs = [
        spec.to_dict(reference_config)
        for spec in default_branch_risk_sensitivity_specs()
    ]
    expected_sensitivity_config_by_id = {
        str(spec["sensitivity_id"]): spec for spec in expected_specs
    }
    all_records: list[dict[str, Any]] = []
    repeat_rebuild_records: list[dict[str, Any]] = []
    with open_randomization_aggregate_record_workspace(source) as workspace:
        prompt_contract = rebuild_randomization_prompt_source_contract(
            workspace,
            source,
            paper_run_name=paper_run_name,
        )
        prompt_rows = tuple(_materialize(row) for row in prompt_contract["prompt_rows"])
        if len(prompt_rows) != expected_prompt_count:
            raise RandomizationParameterSensitivityError(
                "参数敏感性 prompt 来源未匹配论文层级"
            )
        prompt_split_by_id = {
            str(row["prompt_id"]): str(row["split"]) for row in prompt_rows
        }
        prompt_digest_by_id = {
            str(row["prompt_id"]): str(row["prompt_digest"]) for row in prompt_rows
        }
        prompt_index_by_id = {
            str(row["prompt_id"]): int(row["prompt_index"]) for row in prompt_rows
        }
        expected_test_count = sum(
            split == "test" for split in prompt_split_by_id.values()
        )
        for repeat_id in formal_randomization_repeat_ids():
            runtime_source = workspace.find_source(
                randomization_repeat_id=repeat_id,
                package_family="branch_risk_parameter_sensitivity",
                record_role="parameter_sensitivity_runtime_record",
            )
            detection_source = workspace.find_source(
                randomization_repeat_id=repeat_id,
                package_family="branch_risk_parameter_sensitivity",
                record_role="parameter_sensitivity_detection_record",
            )
            protocol_source = workspace.find_source(
                randomization_repeat_id=repeat_id,
                package_family="branch_risk_parameter_sensitivity",
                record_role="parameter_sensitivity_frozen_protocol",
            )
            manifest_source = workspace.find_source(
                randomization_repeat_id=repeat_id,
                package_family="branch_risk_parameter_sensitivity",
                record_role="parameter_sensitivity_run_manifest",
            )
            runtime_records = tuple(
                _materialize(record)
                for record in workspace.iter_records(runtime_source)
            )
            detection_records = tuple(
                _materialize(record)
                for record in workspace.iter_records(detection_source)
            )
            protocols = _materialize(workspace.read_object(protocol_source))
            manifest = _materialize(workspace.read_object(manifest_source))
            config = manifest.get("config", {})
            repeat_identity = config.get("randomization_repeat_identity", {})
            if not all(
                (
                    manifest.get("code_version") == source.common_code_version,
                    config.get("sensitivity_setting_ids")
                    == list(FORMAL_BRANCH_RISK_SENSITIVITY_IDS),
                    config.get("sensitivity_spec_digest")
                    == FORMAL_BRANCH_RISK_SENSITIVITY_SPEC_DIGEST,
                    config.get("specs") == expected_specs,
                    config.get("target_fpr") == target_fpr,
                    repeat_identity.get("randomization_repeat_id") == repeat_id,
                )
            ):
                raise RandomizationParameterSensitivityError(
                    f"敏感性 manifest 未匹配设置、FPR 或 repeat: {repeat_id}"
                )
            _validate_manifest_unit_configs(
                runtime_records,
                config.get("scientific_unit_identity_records"),
                repeat_id=repeat_id,
                paper_run_name=paper_run_name,
                expected_sensitivity_config_by_id=(
                    expected_sensitivity_config_by_id
                ),
            )
            rebuilt = _rebuild_repeat_records(
                runtime_records,
                detection_records,
                protocols,
                repeat_id=repeat_id,
                prompt_split_by_id=prompt_split_by_id,
                prompt_digest_by_id=prompt_digest_by_id,
                prompt_index_by_id=prompt_index_by_id,
                expected_sensitivity_config_by_id=(
                    expected_sensitivity_config_by_id
                ),
            )
            all_records.extend(rebuilt)
            repeat_rebuild_records.append(
                {
                    "randomization_repeat_id": repeat_id,
                    "runtime_source": _source_record(runtime_source),
                    "detection_source": _source_record(detection_source),
                    "protocol_source": _source_record(protocol_source),
                    "manifest_source": _source_record(manifest_source),
                    "rebuilt_record_count": len(rebuilt),
                    "rebuilt_records_digest": build_stable_digest(rebuilt),
                }
            )
    repeat_rows = _repeat_metric_rows(
        tuple(all_records),
        expected_test_count=expected_test_count,
    )
    aggregate_rows = _aggregate_metric_rows(repeat_rows)
    prompt_report = _materialize(prompt_contract["report"])
    summary = {
        "paper_run_name": paper_run_name,
        "paper_claim_scale": paper_run_name,
        "target_fpr": target_fpr,
        "sensitivity_protocol": "single_model_one_parameter_at_a_time",
        "sensitivity_model_scope": "registered_primary_diffusion_model_only",
        "sensitivity_setting_ids": list(FORMAL_BRANCH_RISK_SENSITIVITY_IDS),
        "sensitivity_setting_count": len(FORMAL_BRANCH_RISK_SENSITIVITY_IDS),
        "sensitivity_spec_digest": FORMAL_BRANCH_RISK_SENSITIVITY_SPEC_DIGEST,
        "randomization_repeat_ids": list(formal_randomization_repeat_ids()),
        "randomization_repeat_count": len(formal_randomization_repeat_ids()),
        "confidence_interval_method": "bounded_hoeffding_repeat_mean",
        "confidence_level": _CONFIDENCE_LEVEL,
        "repeat_metric_rows_digest": build_stable_digest(repeat_rows),
        "aggregate_metric_rows_digest": build_stable_digest(aggregate_rows),
        "prompt_source_contract_digest": prompt_report[
            "prompt_source_contract_digest"
        ],
        "randomization_aggregate_digest": source.randomization_aggregate_digest,
        "common_code_version": source.common_code_version,
        "cross_model_evidence_provided": False,
        "parameter_sensitivity_aggregate_ready": True,
        "claim_boundary": "single_model_internal_parameter_sensitivity_only",
        "protocol_decision": "pass",
        "supports_paper_claim": True,
    }
    summary["parameter_sensitivity_summary_digest"] = build_stable_digest(
        summary
    )
    report = {
        "report_schema": RANDOMIZATION_PARAMETER_SENSITIVITY_REPORT_SCHEMA,
        "randomization_aggregate_package_sha256": source.package_sha256,
        "randomization_aggregate_digest": source.randomization_aggregate_digest,
        "common_code_version": source.common_code_version,
        "repeat_rebuild_records": repeat_rebuild_records,
        "repeat_rebuild_records_digest": build_stable_digest(
            repeat_rebuild_records
        ),
        "parameter_sensitivity_aggregate_ready": True,
        "supports_paper_claim": True,
    }
    report["report_digest"] = build_stable_digest(report)
    return RandomizationParameterSensitivityResult(
        repeat_rows=repeat_rows,
        aggregate_rows=aggregate_rows,
        summary=summary,
        report=report,
    )


def _write_csv(path: Path, rows: tuple[Mapping[str, Any], ...]) -> None:
    """按稳定并集列写出非空 CSV。"""

    fieldnames = sorted({field for row in rows for field in row})
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _file_sha256(path: Path) -> str:
    """计算持久输出的 SHA-256。"""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_randomization_parameter_sensitivity_outputs(
    source: RandomizationAggregateProvenance,
    *,
    root: str | Path = ".",
    output_dir: str | Path | None = None,
) -> Path:
    """写出跨重复敏感性表、摘要、来源报告和 manifest。"""

    result = rebuild_randomization_parameter_sensitivity(source, root=root)
    repository_root = Path(root).resolve()
    paper_run_name = str(result.summary["paper_run_name"])
    requested = (
        repository_root
        / RANDOMIZATION_PARAMETER_SENSITIVITY_OUTPUT_ROOT
        / paper_run_name
        if output_dir is None
        else Path(output_dir).expanduser()
    )
    if not requested.is_absolute():
        requested = repository_root / requested
    destination = requested.resolve()
    try:
        destination.relative_to((repository_root / "outputs").resolve())
    except ValueError as exc:
        raise RandomizationParameterSensitivityError(
            "参数敏感性输出目录必须位于 outputs 下"
        ) from exc
    if destination.exists():
        raise RandomizationParameterSensitivityError(
            "参数敏感性正式输出目录已存在, 不得覆盖"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}_publish_", dir=destination.parent)
    )
    try:
        repeat_path = temporary / "parameter_sensitivity_repeat_metrics.csv"
        aggregate_path = temporary / "parameter_sensitivity_aggregate_metrics.csv"
        summary_path = temporary / "parameter_sensitivity_aggregate_summary.json"
        report_path = temporary / "parameter_sensitivity_source_report.json"
        manifest_path = temporary / "manifest.local.json"
        _write_csv(repeat_path, result.repeat_rows)
        _write_csv(aggregate_path, result.aggregate_rows)
        summary_path.write_text(
            json.dumps(result.summary, ensure_ascii=False, sort_keys=True, indent=2)
            + "\n",
            encoding="utf-8",
        )
        report_path.write_text(
            json.dumps(result.report, ensure_ascii=False, sort_keys=True, indent=2)
            + "\n",
            encoding="utf-8",
        )
        data_paths = (repeat_path, aggregate_path, summary_path, report_path)
        published_paths = tuple(destination / path.name for path in data_paths)
        output_sha256 = {
            path.relative_to(repository_root).as_posix(): _file_sha256(temp)
            for temp, path in zip(data_paths, published_paths, strict=True)
        }
        published_manifest = destination / manifest_path.name
        manifest = build_artifact_manifest(
            artifact_id="randomization_branch_risk_parameter_sensitivity_manifest",
            artifact_type="local_manifest",
            input_paths=(source.package_path.as_posix(),),
            output_paths=(
                *output_sha256,
                published_manifest.relative_to(repository_root).as_posix(),
            ),
            config={
                "paper_run_name": paper_run_name,
                "target_fpr": result.summary["target_fpr"],
                "randomization_aggregate_package_sha256": (
                    source.package_sha256
                ),
                "randomization_aggregate_digest": source.randomization_aggregate_digest,
                "common_code_version": source.common_code_version,
                "sensitivity_setting_ids": list(
                    FORMAL_BRANCH_RISK_SENSITIVITY_IDS
                ),
                "sensitivity_spec_digest": (
                    FORMAL_BRANCH_RISK_SENSITIVITY_SPEC_DIGEST
                ),
                "repeat_metric_rows_digest": result.summary[
                    "repeat_metric_rows_digest"
                ],
                "aggregate_metric_rows_digest": result.summary[
                    "aggregate_metric_rows_digest"
                ],
                "parameter_sensitivity_summary_digest": result.summary[
                    "parameter_sensitivity_summary_digest"
                ],
                "report_digest": result.report["report_digest"],
            },
            code_version=source.common_code_version,
            rebuild_command=(
                "python -m paper_experiments.runners."
                "randomization_parameter_sensitivity"
            ),
            metadata={
                "output_sha256": output_sha256,
                "parameter_sensitivity_aggregate_ready": True,
                "claim_boundary": result.summary["claim_boundary"],
                "supports_paper_claim": True,
            },
        ).to_dict()
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2)
            + "\n",
            encoding="utf-8",
        )
        temporary.rename(destination)
        return destination / manifest_path.name
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    """构造可脱离 Notebook 使用的跨重复敏感性入口。"""

    parser = argparse.ArgumentParser(
        description="从精确9重复聚合包重建单模型参数敏感性统计。"
    )
    parser.add_argument("--root", default=".", help="仓库根目录。")
    parser.add_argument(
        "--paper-run-name",
        required=True,
        choices=tuple(RUN_EXPECTED_PROMPT_COUNTS),
        help="论文运行层级。",
    )
    parser.add_argument(
        "--target-fpr",
        required=True,
        type=float,
        help="聚合包冻结的目标 FPR。",
    )
    parser.add_argument(
        "--aggregate-package-path",
        required=True,
        help="精确9重复聚合来源 ZIP。",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="可选输出目录, 必须位于 outputs/ 下。",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """验证聚合来源后重建并写出正式敏感性证据。"""

    arguments = build_parser().parse_args(argv)
    source = validate_randomization_aggregate_provenance(
        arguments.aggregate_package_path,
        paper_run_name=arguments.paper_run_name,
        target_fpr=arguments.target_fpr,
    )
    manifest_path = write_randomization_parameter_sensitivity_outputs(
        source,
        root=arguments.root,
        output_dir=arguments.output_dir,
    )
    print(manifest_path.as_posix())


if __name__ == "__main__":
    main()


__all__ = [
    "RANDOMIZATION_PARAMETER_SENSITIVITY_OUTPUT_ROOT",
    "RANDOMIZATION_PARAMETER_SENSITIVITY_REPORT_SCHEMA",
    "RandomizationParameterSensitivityError",
    "RandomizationParameterSensitivityResult",
    "build_parser",
    "rebuild_randomization_parameter_sensitivity",
    "write_randomization_parameter_sensitivity_outputs",
]
