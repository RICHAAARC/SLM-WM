"""执行单模型内部的分支风险参数敏感性实验。

每个参数设置均重新执行生成、嵌入、既有攻击矩阵与图像检测, 并使用自己的
calibration split 冻结阈值。该实现不会把完整方法的阈值借给其他参数设置,
因此测得的是方法对参数变化的真实敏感性, 而不是阈值失配。
"""

from __future__ import annotations

from dataclasses import replace
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Iterable
from zipfile import ZIP_STORED, ZipFile

from experiments.ablations.branch_risk_sensitivity import (
    BranchRiskSensitivitySpec,
    branch_risk_sensitivity_contract,
    default_branch_risk_sensitivity_specs,
)
from experiments.ablations.runtime_rerun import (
    _canonical_prompt_contract,
    _formal_attack_coverage_ready,
    _mean,
    _read_jsonl,
    _shared_runtime_context_config,
    _write_csv,
    runtime_rerun_randomization_plan,
)
from experiments.artifacts.artifact_manifest import build_artifact_manifest
from experiments.artifacts.manifest_schema import manifest_config_digest_ready
from experiments.protocol.paper_fixed_fpr import bounded_hoeffding_confidence_interval
from experiments.protocol.paper_run_config import (
    build_paper_run_config,
    normalize_paper_run_name,
)
from experiments.protocol.image_only_evidence import (
    apply_frozen_evidence_protocol,
    calibrate_complete_evidence_protocol,
)
from experiments.protocol.content_routing_reference_quantile import (
    ContentRoutingReferenceScalars,
)
from experiments.runners.semantic_watermark_runtime import (
    SemanticWatermarkRuntimeConfig,
    load_completed_semantic_watermark_runtime_result,
    load_semantic_watermark_runtime_context,
    semantic_watermark_runtime_config_payload,
    validate_semantic_watermark_runtime_result_provenance,
    write_semantic_watermark_runtime_outputs,
)
from experiments.runtime import repository_environment
from experiments.runtime.resume_checkpoint import (
    clear_progress_checkpoints,
    persist_progress_checkpoint,
    restore_role_checkpoints,
)
from experiments.runtime.archive_naming import utc_archive_token
from experiments.runtime.package_input_manifest import (
    collect_exact_package_entries,
    validate_exact_package_archive,
    write_exact_package_input_manifest,
)
from experiments.runtime.scientific_execution_binding import (
    validate_scientific_execution_binding,
)
from experiments.runtime.scientific_unit_provenance import (
    aggregate_scientific_unit_provenance,
)
from main.core.digest import build_stable_digest


_ARTIFACT_ROLE = "branch_risk_parameter_sensitivity"
_CONFIDENCE_LEVEL = 0.95
_RISK_CONFIG_FIELD_NAMES = (
    "lf_content_risk_config",
    "tail_robust_risk_config",
    "attention_geometry_risk_config",
)
PACKAGE_INPUT_MANIFEST_FILE_NAME = (
    "branch_risk_parameter_sensitivity_package_input_manifest.json"
)


def _sensitivity_run_entry(
    prompt_index: int,
    base_config: SemanticWatermarkRuntimeConfig,
    spec: BranchRiskSensitivitySpec,
    result: Any,
    detections: tuple[dict[str, Any], ...],
) -> dict[str, Any]:
    """构造一个 prompt 与一个参数设置的完整运行记录。"""

    return {
        "prompt_index": prompt_index,
        "prompt_id": base_config.prompt_id,
        "prompt_digest": build_stable_digest(
            {"prompt_text": base_config.prompt}
        ),
        "split": base_config.split,
        "sensitivity_id": spec.sensitivity_id,
        "sensitivity_config": spec.to_dict(base_config),
        "runtime_result": result.to_dict(),
        "detections": detections,
    }


def _sensitivity_formal_record(
    entry: dict[str, Any],
    detections: tuple[dict[str, Any], ...],
    threshold_digest: str,
    threshold: float,
) -> dict[str, Any]:
    """用参数设置自己的冻结阈值构造逐 prompt 记录。"""

    clean_negative = next(
        record
        for record in detections
        if record.get("sample_role") == "clean_negative"
        and not record.get("attack_id")
    )
    positive_source = next(
        record
        for record in detections
        if record.get("sample_role") == "positive_source"
        and not record.get("attack_id")
    )
    wrong_key_negative = next(
        record
        for record in detections
        if record.get("sample_role") == "wrong_key_negative"
        and not record.get("attack_id")
    )
    attacked_positive = tuple(
        record
        for record in detections
        if record.get("sample_role") == "positive_source"
        and record.get("attack_id")
    )
    attacked_negative = tuple(
        record
        for record in detections
        if record.get("sample_role") == "clean_negative"
        and record.get("attack_id")
    )
    runtime_result = entry["runtime_result"]
    return {
        "prompt_index": entry["prompt_index"],
        "prompt_id": entry["prompt_id"],
        "prompt_digest": entry["prompt_digest"],
        "split": entry["split"],
        "sensitivity_id": entry["sensitivity_id"],
        "sensitivity_config": entry["sensitivity_config"],
        "runtime_result": runtime_result,
        "generation_rerun": True,
        "attack_and_detection_rerun": any(
            record.get("attack_id") for record in detections
        ),
        "threshold_calibration_scope": (
            "per_sensitivity_setting_calibration_split"
        ),
        "frozen_content_threshold": threshold,
        "frozen_threshold_digest": threshold_digest,
        "clean_negative_positive": bool(
            clean_negative["formal_evidence_positive"]
        ),
        "positive_source_positive": bool(
            positive_source["formal_evidence_positive"]
        ),
        "wrong_key_negative_positive": bool(
            wrong_key_negative["formal_evidence_positive"]
        ),
        "clean_negative_content_score": float(clean_negative["content_score"]),
        "positive_source_content_score": float(
            positive_source["content_score"]
        ),
        "attacked_positive_count": len(attacked_positive),
        "attacked_positive_rate": _mean(
            float(bool(record["formal_evidence_positive"]))
            for record in attacked_positive
        ),
        "attacked_negative_count": len(attacked_negative),
        "attacked_negative_rate": _mean(
            float(bool(record["formal_evidence_positive"]))
            for record in attacked_negative
        ),
        "paired_ssim": float(runtime_result["metadata"]["paired_quality"]["ssim"]),
    }


def _metric_summary(
    records: list[dict[str, Any]],
    *,
    field_name: str,
    value_function: Any,
    lower_bound: float,
    upper_bound: float,
) -> dict[str, Any]:
    """计算有界 prompt 级均值及分布无关置信区间。"""

    values = [float(value_function(record)) for record in records]
    mean_value = _mean(values)
    ci_low, ci_high = bounded_hoeffding_confidence_interval(
        mean_value,
        len(values),
        confidence_level=_CONFIDENCE_LEVEL,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
    )
    return {
        field_name: mean_value,
        f"{field_name}_ci_low": ci_low,
        f"{field_name}_ci_high": ci_high,
        f"{field_name}_sample_count": len(values),
    }


def _sensitivity_metric_row(
    sensitivity_id: str,
    records: list[dict[str, Any]],
    threshold_digest: str,
) -> dict[str, Any]:
    """汇总一个参数设置的检测与质量指标。"""

    summaries = (
        _metric_summary(
            records,
            field_name="clean_false_positive_rate",
            value_function=lambda record: record["clean_negative_positive"],
            lower_bound=0.0,
            upper_bound=1.0,
        ),
        _metric_summary(
            records,
            field_name="wrong_key_false_positive_rate",
            value_function=lambda record: record["wrong_key_negative_positive"],
            lower_bound=0.0,
            upper_bound=1.0,
        ),
        _metric_summary(
            records,
            field_name="clean_true_positive_rate",
            value_function=lambda record: record["positive_source_positive"],
            lower_bound=0.0,
            upper_bound=1.0,
        ),
        _metric_summary(
            records,
            field_name="attacked_true_positive_rate",
            value_function=lambda record: record["attacked_positive_rate"],
            lower_bound=0.0,
            upper_bound=1.0,
        ),
        _metric_summary(
            records,
            field_name="attacked_false_positive_rate",
            value_function=lambda record: record["attacked_negative_rate"],
            lower_bound=0.0,
            upper_bound=1.0,
        ),
        _metric_summary(
            records,
            field_name="paired_ssim_mean",
            value_function=lambda record: record["paired_ssim"],
            lower_bound=-1.0,
            upper_bound=1.0,
        ),
    )
    row: dict[str, Any] = {
        "sensitivity_id": sensitivity_id,
        "test_prompt_count": len(records),
        "frozen_threshold_digest": threshold_digest,
        "confidence_interval_method": "bounded_hoeffding",
        "confidence_level": _CONFIDENCE_LEVEL,
        "metric_status": "measured_full_runtime_rerun",
    }
    for summary in summaries:
        row.update(summary)
    return row


def run_branch_risk_parameter_sensitivity(
    base_configs: Iterable[SemanticWatermarkRuntimeConfig],
    target_fpr: float,
    paper_run_name: str,
    root: str | Path = ".",
    specs: tuple[BranchRiskSensitivitySpec, ...] | None = None,
    max_new_runs_per_session: int = 0,
    *,
    content_routing_references: ContentRoutingReferenceScalars | None = None,
) -> dict[str, Any]:
    """在完整 prompt 集上执行受治理的18项单参数敏感性实验。"""

    if type(content_routing_references) is not ContentRoutingReferenceScalars:
        raise RuntimeError(
            "旧branch-risk实验入口缺少已资格化content routing references，禁止回退旧链"
        )
    root_path = Path(root).resolve()
    formal_execution_run_lock = (
        repository_environment.require_published_formal_execution_lock(root_path)
    )
    resolved_run_name = normalize_paper_run_name(paper_run_name)
    paper_run = build_paper_run_config(root_path)
    if paper_run.run_name != resolved_run_name or abs(
        float(target_fpr) - paper_run.target_fpr
    ) > 1e-12:
        raise ValueError("敏感性实验身份必须与当前论文运行配置一致")
    if not 0.0 < target_fpr < 1.0:
        raise ValueError("target_fpr 必须位于 (0, 1)")
    if max_new_runs_per_session < 0:
        raise ValueError("max_new_runs_per_session 不得为负")

    output_dir = f"outputs/formal_branch_risk_sensitivity/{resolved_run_name}"
    resolved_output = (root_path / output_dir).resolve()
    resolved_output.mkdir(parents=True, exist_ok=True)
    progress_path = resolved_output / "parameter_sensitivity_progress.json"
    restore_role_checkpoints(
        repository_root=root_path,
        artifact_role=_ARTIFACT_ROLE,
        paper_run_name=resolved_run_name,
        allowed_output_prefix=output_dir,
    )

    resolved_specs = specs or default_branch_risk_sensitivity_specs()
    sensitivity_contract = branch_risk_sensitivity_contract(resolved_specs)
    if not sensitivity_contract["sensitivity_exact_set_ready"]:
        raise ValueError("正式敏感性实验必须精确使用受治理的18项设置")
    resolved_base_configs = tuple(base_configs)
    if not resolved_base_configs:
        raise ValueError("敏感性实验至少需要一个 prompt 配置")
    shared_context_config = _shared_runtime_context_config(
        resolved_base_configs,
        resolved_specs,
        output_dir,
    )
    prompt_contract, canonical_prompt_index_by_id = _canonical_prompt_contract(
        root_path,
        paper_run,
        resolved_base_configs,
    )
    split_counts = {
        split: sum(config.split == split for config in resolved_base_configs)
        for split in ("dev", "calibration", "test")
    }
    if split_counts["calibration"] == 0 or split_counts["test"] == 0:
        raise ValueError("敏感性实验必须同时包含 calibration 和 test split")

    shared_context = None
    run_entries: list[dict[str, Any]] = []
    unit_identity_records: list[dict[str, Any]] = []
    resumed_run_count = 0
    new_run_count = 0
    expected_run_count = len(resolved_base_configs) * len(resolved_specs)

    def write_progress() -> dict[str, Any]:
        """原子保存中间进度, 中间状态不得支持论文主张。"""

        progress = {
            "paper_run_name": resolved_run_name,
            **sensitivity_contract,
            "expected_run_count": expected_run_count,
            "completed_run_count": len(run_entries),
            "remaining_run_count": expected_run_count - len(run_entries),
            "resumed_run_count": resumed_run_count,
            "new_run_count": new_run_count,
            "max_new_runs_per_session": max_new_runs_per_session,
            "prompt_count": len(resolved_base_configs),
            "target_fpr": target_fpr,
            "protocol_decision": "resume_required",
            "evidence_eligibility": "intermediate_state_only",
            "supports_paper_claim": False,
        }
        temporary_path = progress_path.with_name(progress_path.name + ".partial")
        temporary_path.write_text(
            json.dumps(progress, ensure_ascii=False, sort_keys=True, indent=2)
            + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(progress_path)
        persist_progress_checkpoint(
            progress_path,
            repository_root=root_path,
            artifact_role=_ARTIFACT_ROLE,
            paper_run_name=resolved_run_name,
        )
        return progress

    calibration_configs = tuple(
        config for config in resolved_base_configs if config.split == "calibration"
    )
    remaining_configs = tuple(
        config for config in resolved_base_configs if config.split != "calibration"
    )
    protocols: dict[str, Any] = {}
    for base_config in calibration_configs + remaining_configs:
        if base_config.split != "calibration" and not protocols:
            expected_calibration_count = len(calibration_configs) * len(
                resolved_specs
            )
            calibration_entries = tuple(
                entry for entry in run_entries if entry["split"] == "calibration"
            )
            if len(calibration_entries) != expected_calibration_count:
                break
            for spec in resolved_specs:
                negatives = tuple(
                    detection
                    for entry in calibration_entries
                    if entry["sensitivity_id"] == spec.sensitivity_id
                    for detection in entry["detections"]
                    if detection.get("sample_role") == "clean_negative"
                    and not detection.get("attack_id")
                )
                protocols[spec.sensitivity_id] = (
                    calibrate_complete_evidence_protocol(negatives, target_fpr)
                )
        prompt_index = canonical_prompt_index_by_id[base_config.prompt_id]
        for spec in resolved_specs:
            run_config = spec.apply(base_config, output_dir)
            run_config = replace(
                run_config,
                detector_guided_attack_threshold_protocol=(
                    protocols[spec.sensitivity_id].to_dict()
                    if base_config.split == "test" and protocols
                    else None
                ),
            )
            result = load_completed_semantic_watermark_runtime_result(
                run_config,
                root=root_path,
            )
            generated_now = False
            if result is not None:
                resumed_run_count += 1
            elif max_new_runs_per_session and new_run_count >= max_new_runs_per_session:
                continue
            else:
                if shared_context is None:
                    shared_context = load_semantic_watermark_runtime_context(
                        shared_context_config,
                        verified_formal_execution_lock=formal_execution_run_lock,
                        repository_root=root_path,
                    )
                result = write_semantic_watermark_runtime_outputs(
                    run_config,
                    root=root_path,
                    references=content_routing_references,
                    verified_formal_execution_lock=formal_execution_run_lock,
                    runtime_context=shared_context,
                )
                new_run_count += 1
                generated_now = True
            result_payload = result.to_dict()
            validate_semantic_watermark_runtime_result_provenance(
                result_payload,
                expected_config=run_config,
            )
            unit_identity_records.append(
                {
                    "run_id": result_payload["run_id"],
                    "scientific_unit_config": (
                        semantic_watermark_runtime_config_payload(run_config)
                    ),
                    "formal_randomization_reference": result_payload["metadata"][
                        "formal_randomization_reference"
                    ],
                }
            )
            detections = _read_jsonl(root_path / result.detection_record_path)
            run_entries.append(
                _sensitivity_run_entry(
                    prompt_index,
                    base_config,
                    spec,
                    result,
                    detections,
                )
            )
            if generated_now and len(run_entries) < expected_run_count:
                write_progress()

    if len(run_entries) != expected_run_count:
        return write_progress()
    progress_path.unlink(missing_ok=True)
    clear_progress_checkpoints(
        artifact_role=_ARTIFACT_ROLE,
        paper_run_name=resolved_run_name,
    )

    formal_records: list[dict[str, Any]] = []
    formal_detection_records: list[dict[str, Any]] = []
    for spec in resolved_specs:
        spec_entries = tuple(
            entry
            for entry in run_entries
            if entry["sensitivity_id"] == spec.sensitivity_id
        )
        calibration_negatives = tuple(
            detection
            for entry in spec_entries
            if entry["split"] == "calibration"
            for detection in entry["detections"]
            if detection.get("sample_role") == "clean_negative"
            and not detection.get("attack_id")
        )
        rebuilt_protocol = calibrate_complete_evidence_protocol(
            calibration_negatives,
            target_fpr=target_fpr,
        )
        if (
            spec.sensitivity_id in protocols
            and rebuilt_protocol != protocols[spec.sensitivity_id]
        ):
            raise RuntimeError("敏感性 test 运行绑定阈值不能由 calibration 重建")
        protocols[spec.sensitivity_id] = rebuilt_protocol
        for entry in spec_entries:
            detections = apply_frozen_evidence_protocol(
                entry["detections"],
                rebuilt_protocol,
            )
            attack_ready = _formal_attack_coverage_ready(
                detections,
                split=str(entry["split"]),
                expected_generation_seed_random=int(
                    entry["runtime_result"]["metadata"][
                        "formal_randomization_reference"
                    ]["generation_seed_random"]
                ),
                expected_threshold_digest=rebuilt_protocol.threshold_digest,
            )
            formal_record = {
                **_sensitivity_formal_record(
                    entry,
                    detections,
                    rebuilt_protocol.threshold_digest,
                    rebuilt_protocol.content_threshold,
                ),
                "formal_attack_coverage_ready": attack_ready,
            }
            formal_records.append(formal_record)
            formal_detection_records.extend(
                {
                    **record,
                    "sensitivity_id": spec.sensitivity_id,
                    "sensitivity_prompt_id": entry["prompt_id"],
                }
                for record in detections
            )

    grouped_test_records = {
        spec.sensitivity_id: [
            record
            for record in formal_records
            if record["sensitivity_id"] == spec.sensitivity_id
            and record["split"] == "test"
        ]
        for spec in resolved_specs
    }
    metric_rows = [
        _sensitivity_metric_row(
            spec.sensitivity_id,
            grouped_test_records[spec.sensitivity_id],
            protocols[spec.sensitivity_id].threshold_digest,
        )
        for spec in resolved_specs
    ]
    reference_row = next(
        row for row in metric_rows if row["sensitivity_id"] == "formal_reference"
    )
    delta_fields = (
        "clean_false_positive_rate",
        "wrong_key_false_positive_rate",
        "clean_true_positive_rate",
        "attacked_true_positive_rate",
        "attacked_false_positive_rate",
        "paired_ssim_mean",
    )
    delta_rows = [
        {
            "sensitivity_id": row["sensitivity_id"],
            **{
                f"{field_name}_delta": (
                    float(row[field_name]) - float(reference_row[field_name])
                )
                for field_name in delta_fields
            },
            "metric_status": "measured_full_runtime_rerun",
        }
        for row in metric_rows
        if row["sensitivity_id"] != "formal_reference"
    ]

    records_path = resolved_output / "parameter_sensitivity_records.jsonl"
    detections_path = resolved_output / "formal_detection_records.jsonl"
    protocols_path = resolved_output / "per_setting_frozen_protocols.json"
    metrics_path = resolved_output / "parameter_sensitivity_metrics.csv"
    delta_path = resolved_output / "parameter_sensitivity_delta.csv"
    summary_path = resolved_output / "parameter_sensitivity_summary.json"
    manifest_path = resolved_output / "manifest.local.json"
    records_path.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
            for record in formal_records
        ),
        encoding="utf-8",
    )
    detections_path.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
            for record in formal_detection_records
        ),
        encoding="utf-8",
    )
    protocol_payload = {
        name: protocol.to_dict() for name, protocol in protocols.items()
    }
    protocols_path.write_text(
        json.dumps(
            protocol_payload,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_csv(metrics_path, metric_rows)
    _write_csv(delta_path, delta_rows)

    scientific_unit_provenance = aggregate_scientific_unit_provenance(
        (
            record["runtime_result"]["metadata"]["scientific_unit_provenance"]
            for record in formal_records
        ),
        expected_reference_count=expected_run_count,
    )
    expected_attack_count = split_counts["test"] * len(resolved_specs)
    actual_attack_count = sum(
        bool(record["attack_and_detection_rerun"])
        for record in formal_records
    )
    component_ready = (
        sensitivity_contract["sensitivity_exact_set_ready"]
        and prompt_contract["prompt_protocol_exact_set_ready"]
        and len(formal_records) == expected_run_count
        and all(
            record["runtime_result"]["run_decision"] == "pass"
            for record in formal_records
        )
        and all(
            record["formal_attack_coverage_ready"] for record in formal_records
        )
        and actual_attack_count == expected_attack_count
        and len(protocols) == len(resolved_specs)
        and scientific_unit_provenance["scientific_unit_provenance_ready"]
        and all(
            protocol.calibration_source_negative_count
            == split_counts["calibration"]
            and protocol.rescue_window_fit_negative_count
            == split_counts["calibration"] // 3
            and protocol.threshold_freeze_negative_count
            == (
                split_counts["calibration"]
                - split_counts["calibration"] // 3
            )
            for protocol in protocols.values()
        )
        and all(
            len(grouped_test_records[spec.sensitivity_id])
            == split_counts["test"]
            for spec in resolved_specs
        )
    )
    attacked_tpr_values = [
        float(row["attacked_true_positive_rate"]) for row in metric_rows
    ]
    paired_ssim_values = [float(row["paired_ssim_mean"]) for row in metric_rows]
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "paper_run_name": resolved_run_name,
        "paper_claim_scale": resolved_run_name,
        "randomization_repeat_identity": {
            "randomization_repeat_id": paper_run.randomization_repeat_id,
            "generation_seed_index": paper_run.generation_seed_index,
            "generation_seed_offset": paper_run.generation_seed_offset,
            "watermark_key_index": paper_run.watermark_key_index,
            "formal_randomization_protocol_digest": (
                paper_run.formal_randomization_protocol_digest
            ),
        },
        **sensitivity_contract,
        **prompt_contract,
        "record_count": len(formal_records),
        "resumed_run_count": resumed_run_count,
        "new_run_count": new_run_count,
        "prompt_count": len(resolved_base_configs),
        "split_counts": split_counts,
        "target_fpr": target_fpr,
        "generation_rerun_count": len(formal_records),
        "expected_attack_and_detection_rerun_count": expected_attack_count,
        "attack_and_detection_rerun_count": actual_attack_count,
        "formal_attack_coverage_ready": all(
            record["formal_attack_coverage_ready"] for record in formal_records
        ),
        "confidence_interval_method": "bounded_hoeffding",
        "confidence_level": _CONFIDENCE_LEVEL,
        "minimum_attacked_true_positive_rate_across_settings": min(
            attacked_tpr_values
        ),
        "maximum_attacked_true_positive_rate_across_settings": max(
            attacked_tpr_values
        ),
        "attacked_true_positive_rate_range": (
            max(attacked_tpr_values) - min(attacked_tpr_values)
        ),
        "minimum_paired_ssim_across_settings": min(paired_ssim_values),
        "maximum_paired_ssim_across_settings": max(paired_ssim_values),
        "parameter_sensitivity_records_digest": build_stable_digest(
            formal_records
        ),
        "formal_detection_records_sha256": (
            repository_environment.file_digest(detections_path)
        ),
        "per_setting_frozen_protocols_digest": build_stable_digest(
            protocol_payload
        ),
        **scientific_unit_provenance,
        "parameter_sensitivity_component_ready": component_ready,
        "protocol_decision": "pass" if component_ready else "fail",
        "repeat_component_ready": component_ready,
        "randomization_aggregate_ready": False,
        "supports_paper_claim": False,
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )

    formal_execution_run_lock = (
        repository_environment.validate_formal_execution_lock_pair(
            formal_execution_run_lock,
            repository_environment.require_published_formal_execution_lock(
                root_path
            ),
            formal_execution_run_lock["formal_execution_commit"],
        )
    )
    manifest = build_artifact_manifest(
        artifact_id="formal_branch_risk_parameter_sensitivity_manifest",
        artifact_type="local_manifest",
        input_paths=(),
        output_paths=(
            records_path.relative_to(root_path).as_posix(),
            detections_path.relative_to(root_path).as_posix(),
            protocols_path.relative_to(root_path).as_posix(),
            metrics_path.relative_to(root_path).as_posix(),
            delta_path.relative_to(root_path).as_posix(),
            summary_path.relative_to(root_path).as_posix(),
            manifest_path.relative_to(root_path).as_posix(),
        ),
        config={
            "formal_randomization_plan": runtime_rerun_randomization_plan(
                resolved_base_configs[0]
            ),
            "scientific_unit_identity_records": unit_identity_records,
            "specs": [
                spec.to_dict(resolved_base_configs[0])
                for spec in resolved_specs
            ],
            **sensitivity_contract,
            **prompt_contract,
            "prompt_count": len(resolved_base_configs),
            "split_counts": split_counts,
            "target_fpr": target_fpr,
            "randomization_repeat_identity": {
                "randomization_repeat_id": paper_run.randomization_repeat_id,
                "generation_seed_index": paper_run.generation_seed_index,
                "generation_seed_offset": paper_run.generation_seed_offset,
                "watermark_key_index": paper_run.watermark_key_index,
                "formal_randomization_protocol_digest": (
                    paper_run.formal_randomization_protocol_digest
                ),
            },
            "record_digest": build_stable_digest(formal_records),
            "formal_detection_records_sha256": summary[
                "formal_detection_records_sha256"
            ],
            "per_setting_frozen_protocols_digest": summary[
                "per_setting_frozen_protocols_digest"
            ],
        },
        code_version=formal_execution_run_lock["formal_execution_commit"],
        rebuild_command=(
            "调用 experiments.ablations.branch_risk_sensitivity_runtime."
            "run_branch_risk_parameter_sensitivity"
        ),
        metadata={
            "protocol_decision": summary["protocol_decision"],
            **sensitivity_contract,
            **prompt_contract,
            "generation_rerun_required": True,
            "per_setting_calibration_required": True,
            "formal_attack_coverage_ready": summary[
                "formal_attack_coverage_ready"
            ],
            "scientific_unit_provenance_ready": summary[
                "scientific_unit_provenance_ready"
            ],
            "parameter_sensitivity_component_ready": summary[
                "parameter_sensitivity_component_ready"
            ],
            "repeat_component_ready": summary["repeat_component_ready"],
            "randomization_aggregate_ready": False,
            "supports_paper_claim": False,
        },
    ).to_dict()
    manifest["formal_execution_run_lock"] = formal_execution_run_lock
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def package_branch_risk_parameter_sensitivity(
    paper_run_name: str,
    root: str | Path = ".",
) -> Path:
    """核验并打包当前重复的单模型参数敏感性证据。"""

    root_path = Path(root).resolve()
    package_lock = repository_environment.require_published_formal_execution_lock(
        root_path
    )
    resolved_run_name = normalize_paper_run_name(paper_run_name)
    paper_run = build_paper_run_config(root_path)
    if paper_run.run_name != resolved_run_name:
        raise ValueError("敏感性打包层级必须与当前论文配置一致")
    source_dir = (
        root_path
        / "outputs"
        / "formal_branch_risk_sensitivity"
        / resolved_run_name
    ).resolve()
    required_names = (
        "parameter_sensitivity_records.jsonl",
        "formal_detection_records.jsonl",
        "per_setting_frozen_protocols.json",
        "parameter_sensitivity_metrics.csv",
        "parameter_sensitivity_delta.csv",
        "parameter_sensitivity_summary.json",
        "manifest.local.json",
    )
    required_paths = tuple(source_dir / name for name in required_names)
    if any(not path.is_file() for path in required_paths):
        raise FileNotFoundError("单模型参数敏感性输出不完整, 不得打包")

    summary = json.loads(
        (source_dir / "parameter_sensitivity_summary.json").read_text(
            encoding="utf-8-sig"
        )
    )
    manifest_path = source_dir / "manifest.local.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    if not isinstance(summary, dict) or not isinstance(manifest, dict):
        raise TypeError("敏感性 summary 与 manifest 必须是 JSON object")
    manifest_config = manifest.get("config", {})
    manifest_metadata = manifest.get("metadata", {})
    expected_contract = branch_risk_sensitivity_contract(
        default_branch_risk_sensitivity_specs()
    )
    expected_ids = expected_contract["sensitivity_setting_ids"]
    reference_config = SemanticWatermarkRuntimeConfig()
    expected_specs = [
        spec.to_dict(reference_config)
        for spec in default_branch_risk_sensitivity_specs()
    ]
    expected_spec_by_id = {
        str(spec["sensitivity_id"]): spec for spec in expected_specs
    }
    with (source_dir / "parameter_sensitivity_metrics.csv").open(
        encoding="utf-8-sig",
        newline="",
    ) as stream:
        metric_ids = [
            str(row.get("sensitivity_id", ""))
            for row in csv.DictReader(stream)
        ]
    protocols = json.loads(
        (source_dir / "per_setting_frozen_protocols.json").read_text(
            encoding="utf-8-sig"
        )
    )
    records = _read_jsonl(source_dir / "parameter_sensitivity_records.jsonl")
    unit_identity_records = manifest_config.get(
        "scientific_unit_identity_records"
    )
    if not isinstance(unit_identity_records, list):
        raise RuntimeError("敏感性 manifest 缺少逐运行科学身份")
    unit_identity_by_run = {
        str(record.get("run_id", "")): record
        for record in unit_identity_records
        if isinstance(record, dict)
    }
    if len(unit_identity_by_run) != len(records):
        raise RuntimeError("敏感性逐运行科学身份集合与记录不一致")
    for record in records:
        runtime_result = record.get("runtime_result", {})
        identity = unit_identity_by_run.get(str(runtime_result.get("run_id", "")))
        if not isinstance(identity, dict):
            raise RuntimeError("敏感性记录缺少对应的逐运行科学身份")
        unit_config = identity.get("scientific_unit_config")
        if not isinstance(unit_config, dict):
            raise RuntimeError("敏感性逐运行科学配置不是完整对象")
        validate_semantic_watermark_runtime_result_provenance(
            runtime_result,
            unit_config=unit_config,
        )
        sensitivity_config = record.get("sensitivity_config", {})
        sensitivity_id = str(record.get("sensitivity_id", ""))
        expected_sensitivity_config = expected_spec_by_id.get(sensitivity_id)
        expected_risk_configs = (
            expected_sensitivity_config.get("resolved_branch_risk_configs")
            if isinstance(expected_sensitivity_config, dict)
            else None
        )
        if (
            unit_config.get("risk_parameter_protocol")
            != "single_model_internal_sensitivity"
            or unit_config.get("prompt_id") != record.get("prompt_id")
            or unit_config.get("split") != record.get("split")
            or not isinstance(sensitivity_config, dict)
            or sensitivity_config != expected_sensitivity_config
            or unit_config.get("output_dir")
            != (
                "outputs/formal_branch_risk_sensitivity/"
                f"{resolved_run_name}/runs/{sensitivity_id}"
            )
            or not isinstance(expected_risk_configs, dict)
            or any(
                unit_config.get(field_name)
                != expected_risk_configs.get(field_name)
                for field_name in _RISK_CONFIG_FIELD_NAMES
            )
        ):
            raise RuntimeError("敏感性记录未精确绑定参数设置与 prompt 身份")

    formal_execution_run_lock = (
        repository_environment.validate_formal_execution_lock_pair(
            manifest.get("formal_execution_run_lock"),
            package_lock,
            manifest.get("code_version"),
        )
    )
    exact_ready = all(
        (
            summary.get("paper_run_name") == resolved_run_name,
            summary.get("paper_claim_scale") == resolved_run_name,
            abs(float(summary.get("target_fpr", -1.0)) - paper_run.target_fpr)
            <= 1e-12,
            summary.get("sensitivity_setting_ids") == expected_ids,
            summary.get("sensitivity_spec_digest")
            == expected_contract["sensitivity_spec_digest"],
            summary.get("sensitivity_exact_set_ready") is True,
            summary.get("parameter_sensitivity_component_ready") is True,
            summary.get("repeat_component_ready") is True,
            summary.get("randomization_aggregate_ready") is False,
            summary.get("supports_paper_claim") is False,
            metric_ids == expected_ids,
            isinstance(protocols, dict),
            len(protocols) == len(expected_ids),
            set(protocols) == set(expected_ids),
            manifest.get("artifact_id")
            == "formal_branch_risk_parameter_sensitivity_manifest",
            manifest_config_digest_ready(manifest),
            manifest_config.get("sensitivity_setting_ids") == expected_ids,
            manifest_config.get("specs") == expected_specs,
            manifest_config.get("sensitivity_spec_digest")
            == expected_contract["sensitivity_spec_digest"],
            manifest_metadata.get("sensitivity_setting_ids") == expected_ids,
            manifest_metadata.get("sensitivity_spec_digest")
            == expected_contract["sensitivity_spec_digest"],
            manifest_metadata.get("parameter_sensitivity_component_ready")
            is True,
        )
    )
    if not exact_ready:
        raise RuntimeError("单模型参数敏感性身份、18项设置或完成门禁未通过")

    manifest["formal_execution_package_lock"] = package_lock
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    binding_path = source_dir / "scientific_execution_binding.json"
    validate_scientific_execution_binding(
        binding_path,
        expected_artifact_role="branch_risk_parameter_sensitivity",
        expected_paper_run_name=resolved_run_name,
        repository_root=root_path,
    )
    code_version = package_lock["formal_execution_commit"]
    archive_path = source_dir / (
        "branch_risk_parameter_sensitivity_package_"
        f"{utc_archive_token()}_{code_version[:7]}.zip"
    )
    package_input_manifest_path = source_dir / PACKAGE_INPUT_MANIFEST_FILE_NAME
    package_input_manifest_path.unlink(missing_ok=True)
    entries = collect_exact_package_entries(
        repository_root=root_path,
        source_dir=source_dir,
        artifact_manifest=manifest,
        scientific_binding_path=binding_path,
    )
    if not set(required_paths).issubset(entries):
        raise RuntimeError("artifact manifest 未精确声明全部敏感性必要产物")
    write_exact_package_input_manifest(
        package_input_manifest_path,
        repository_root=root_path,
        package_family="branch_risk_parameter_sensitivity",
        paper_run_name=resolved_run_name,
        target_fpr=paper_run.target_fpr,
        randomization_repeat_identity={
            "randomization_repeat_id": paper_run.randomization_repeat_id,
            "generation_seed_index": paper_run.generation_seed_index,
            "generation_seed_offset": paper_run.generation_seed_offset,
            "watermark_key_index": paper_run.watermark_key_index,
        },
        formal_randomization_protocol_digest=(
            paper_run.formal_randomization_protocol_digest
        ),
        entries=entries,
        formal_execution_run_lock=formal_execution_run_lock,
        formal_execution_package_lock=package_lock,
    )
    archive_entries = (*entries, package_input_manifest_path)
    with ZipFile(
        archive_path,
        "w",
        compression=ZIP_STORED,
        allowZip64=True,
    ) as archive:
        for path in archive_entries:
            archive.write(path, path.relative_to(root_path).as_posix())
    try:
        validate_exact_package_archive(
            archive_path,
            repository_root=root_path,
            package_input_manifest_path=package_input_manifest_path,
        )
        final_lock = repository_environment.require_published_formal_execution_lock(
            root_path
        )
        repository_environment.validate_formal_execution_lock_pair(
            package_lock,
            final_lock,
            code_version,
        )
    except Exception:
        archive_path.unlink(missing_ok=True)
        raise
    return archive_path
