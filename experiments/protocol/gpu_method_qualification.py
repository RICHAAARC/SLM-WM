"""从真实单 Prompt 运行证据构造 GPU 方法资格化报告."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.runners.image_only_dataset_runtime import (
    _carrier_only_final_image_preservation_ready,
    _detection_qk_atomic_content_ready,
    _final_image_attention_observability_ready,
    _final_image_preservation_ready,
    _scientific_content_binding_record_ready,
    _scientific_update_record_ready,
)
from experiments.runners.semantic_watermark_runtime import (
    SemanticWatermarkRuntimeConfig,
)
from experiments.runtime.scientific_unit_provenance import (
    validate_scientific_unit_provenance,
)
from main.core.digest import build_stable_digest, tensor_content_identity
from main.core.keyed_prg import (
    KEYED_PRG_VERSION,
    build_keyed_gaussian_tensor,
    build_keyed_uniform_tensor,
    keyed_prg_protocol_record,
)


GPU_METHOD_QUALIFICATION_SCHEMA = "gpu_method_qualification_report_v1"
GPU_OPERATOR_FACT_SCHEMA = "gpu_operator_preflight_fact_v1"
GPU_RESOURCE_BUDGET_SCHEMA = "gpu_resource_budget_decision_v1"
COMPLETE_FEATURE_WIDTH = 716


def _read_json_mapping(path: Path) -> dict[str, Any]:
    """读取 JSON 映射, 并在边界拒绝其他顶层类型."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON 顶层必须是映射: {path}")
    return payload


def _finite_nonnegative(value: Any) -> bool:
    """判断资源观测是否为有限非负实数, 并排除 bool."""

    return bool(
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and float(value) >= 0.0
    )


def _operator_fact(
    fact_id: str,
    ready: bool,
    **evidence: Any,
) -> dict[str, Any]:
    """构造单项方法事实, 便于其他 GPU 算子报告复用统一结构."""

    return {
        "operator_fact_schema": GPU_OPERATOR_FACT_SCHEMA,
        "operator_fact_id": fact_id,
        "operator_fact_ready": bool(ready),
        "evidence": evidence,
    }


def rebuild_keyed_prg_known_answer_report(
    known_answer_path: str | Path,
) -> dict[str, Any]:
    """在当前平台重建公开 PRG 固定向量并逐字节复验."""

    path = Path(known_answer_path)
    protocol = _read_json_mapping(path)
    expected_protocol = keyed_prg_protocol_record(KEYED_PRG_VERSION)
    protocol_identity_ready = bool(
        protocol.get("protocol_schema")
        == "keyed_prg_cross_platform_known_answer_v1"
        and protocol.get("keyed_prg_version") == KEYED_PRG_VERSION
        and protocol.get("keyed_prg_protocol_digest")
        == expected_protocol["keyed_prg_protocol_digest"]
    )
    vector_reports: list[dict[str, Any]] = []
    vectors = protocol.get("vectors")
    if not isinstance(vectors, list):
        raise ValueError("PRG 固定向量协议缺少 vectors 列表")
    for row in vectors:
        if not isinstance(row, dict):
            raise ValueError("PRG 固定向量必须是映射")
        distribution = str(row.get("distribution", ""))
        builder = {
            "uniform": build_keyed_uniform_tensor,
            "gaussian": build_keyed_gaussian_tensor,
        }.get(distribution)
        if builder is None:
            raise ValueError(f"未知 PRG 固定向量分布: {distribution}")
        shape = tuple(int(value) for value in row.get("shape", ()))
        tensor = builder(
            shape,
            str(row.get("public_key_material", "")),
            dict(row.get("domain_fields", {})),
            KEYED_PRG_VERSION,
        )
        actual_identity = tensor_content_identity(tensor)
        expected_identity = {
            field_name: row.get(field_name)
            for field_name in (
                "tensor_content_digest_version",
                "tensor_dtype",
                "tensor_shape",
                "tensor_content_sha256",
            )
        }
        vector_reports.append(
            {
                "vector_id": row.get("vector_id"),
                "distribution": distribution,
                "known_answer_ready": actual_identity == expected_identity,
                "expected_tensor_identity": expected_identity,
                "actual_tensor_identity": actual_identity,
            }
        )
    ready = bool(
        protocol_identity_ready
        and vector_reports
        and all(row["known_answer_ready"] for row in vector_reports)
    )
    report = {
        "known_answer_protocol_path": path.as_posix(),
        "known_answer_protocol_digest": build_stable_digest(protocol),
        "known_answer_protocol_identity_ready": protocol_identity_ready,
        "known_answer_vector_reports": vector_reports,
        "keyed_prg_cross_platform_known_answer_ready": ready,
    }
    report["known_answer_report_digest"] = build_stable_digest(report)
    return report


def _null_space_operator_evidence(
    update_records: Sequence[Mapping[str, Any]],
    config: SemanticWatermarkRuntimeConfig,
) -> dict[str, Any]:
    """从每次真实注入记录提取完整 JVP/VJP 与 PSD-CG 事实."""

    expected_branches = {
        branch_name
        for branch_name, enabled in (
            ("lf_content", config.lf_enabled),
            ("tail_robust", config.tail_robust_enabled),
            ("attention_geometry", config.attention_geometry_enabled),
        )
        if enabled
    }
    rows: list[dict[str, Any]] = []
    all_ready = bool(update_records and expected_branches)
    for record in update_records:
        branch_records = record.get("null_space_records")
        branch_records = branch_records if isinstance(branch_records, dict) else {}
        branch_rows: list[dict[str, Any]] = []
        branch_set_ready = set(branch_records) == expected_branches
        for branch_name in sorted(expected_branches):
            branch = branch_records.get(branch_name)
            branch = branch if isinstance(branch, dict) else {}
            metadata = branch.get("metadata")
            metadata = metadata if isinstance(metadata, dict) else {}
            response_shape = branch.get("response_shape")
            iteration_counts = branch.get("cg_iteration_counts")
            residuals = branch.get("cg_relative_residuals")
            row_ready = bool(
                metadata.get("full_feature_jvp") is True
                and metadata.get("full_feature_vjp") is True
                and metadata.get("cg_damping") == 0.0
                and metadata.get("cg_maximum_iterations")
                == config.null_space_cg_max_iterations
                and metadata.get("cg_relative_tolerance")
                == config.null_space_cg_relative_tolerance
                and response_shape
                == [COMPLETE_FEATURE_WIDTH, config.null_rank]
                and branch.get("cg_converged") is True
                and isinstance(iteration_counts, list)
                and len(iteration_counts) == config.null_rank
                and all(
                    type(value) is int
                    and 0 <= value <= config.null_space_cg_max_iterations
                    for value in iteration_counts
                )
                and isinstance(residuals, list)
                and len(residuals) == config.null_rank
                and all(
                    _finite_nonnegative(value)
                    and float(value) <= config.null_space_cg_relative_tolerance
                    for value in residuals
                )
            )
            branch_rows.append(
                {
                    "branch_name": branch_name,
                    "complete_feature_width": COMPLETE_FEATURE_WIDTH,
                    "response_shape": response_shape,
                    "cg_iteration_counts": iteration_counts,
                    "cg_relative_residuals": residuals,
                    "branch_operator_ready": row_ready,
                }
            )
            all_ready = all_ready and row_ready
        all_ready = all_ready and branch_set_ready
        rows.append(
            {
                "step_index": record.get("step_index"),
                "branch_set_ready": branch_set_ready,
                "branches": branch_rows,
            }
        )
    return {
        "expected_branch_names": sorted(expected_branches),
        "update_operator_records": rows,
        "complete_jvp_vjp_psd_cg_ready": bool(all_ready),
    }


def build_gpu_operator_preflight_report(
    runtime_result: Mapping[str, Any],
    update_records: Sequence[Mapping[str, Any]],
    detection_records: Sequence[Mapping[str, Any]],
    config: SemanticWatermarkRuntimeConfig,
    known_answer_path: str | Path,
) -> dict[str, Any]:
    """验证单 Prompt 方法机制真实性, 不消费资源预算阈值."""

    result = dict(runtime_result)
    updates = tuple(dict(row) for row in update_records)
    detections = tuple(dict(row) for row in detection_records)
    expected_steps = tuple(int(value) for value in config.injection_step_indices)
    actual_steps = tuple(row.get("step_index") for row in updates)
    scientific_update_ready = tuple(
        _scientific_update_record_ready(row, config) for row in updates
    )
    null_space_evidence = _null_space_operator_evidence(updates, config)
    write_rows = [
        {
            "step_index": row.get("step_index"),
            "quantized_write_update_norm": row.get(
                "quantized_write_update_norm"
            ),
            "quantized_write_backtracking_step_count": row.get(
                "quantized_write_backtracking_step_count"
            ),
            "quantized_write_common_scale": row.get(
                "quantized_write_common_scale"
            ),
        }
        for row in updates
    ]
    quantized_writes_ready = bool(
        actual_steps == expected_steps
        and len(updates) == len(expected_steps)
        and all(
            _finite_nonnegative(row["quantized_write_update_norm"])
            and float(row["quantized_write_update_norm"]) > 0.0
            and type(row["quantized_write_backtracking_step_count"]) is int
            and 0
            <= row["quantized_write_backtracking_step_count"]
            <= config.quantized_budget_envelope_backtracking_maximum_steps
            and _finite_nonnegative(row["quantized_write_common_scale"])
            and float(row["quantized_write_common_scale"]) > 0.0
            for row in write_rows
        )
    )
    update_qk_ready = bool(
        updates
        and all(
            row.get("attention_module_names")
            == list(config.attention_module_names)
            and isinstance(row.get("attention_qk_atomic_content_records"), list)
            and bool(row["attention_qk_atomic_content_records"])
            for row in updates
        )
    )
    detection_qk_ready = bool(
        detections
        and all(_detection_qk_atomic_content_ready(row, config) for row in detections)
    )
    provenance = result.get("metadata", {}).get(
        "scientific_unit_provenance",
        {},
    )
    try:
        validated_provenance = validate_scientific_unit_provenance(provenance)
    except (KeyError, TypeError, ValueError):
        validated_provenance = {}
    execution_environment = validated_provenance.get(
        "scientific_execution_environment",
        {},
    )
    real_cuda_ready = bool(
        isinstance(execution_environment, dict)
        and str(execution_environment.get("execution_device_name", ""))
        .startswith("cuda")
        and execution_environment.get("cuda_available") is True
        and int(execution_environment.get("visible_cuda_device_count", 0)) > 0
    )
    known_answer = rebuild_keyed_prg_known_answer_report(known_answer_path)
    facts = (
        _operator_fact(
            "registered_sd35_qk_layers_exist_on_real_cuda",
            real_cuda_ready and update_qk_ready,
            execution_device_name=execution_environment.get(
                "execution_device_name"
            ),
            cuda_device_name=execution_environment.get("cuda_device_name"),
            attention_module_names=list(config.attention_module_names),
        ),
        _operator_fact(
            "complete_716_feature_jvp_vjp_psd_cg",
            null_space_evidence["complete_jvp_vjp_psd_cg_ready"],
            **null_space_evidence,
        ),
        _operator_fact(
            "three_nonzero_quantized_latent_injections",
            quantized_writes_ready,
            expected_injection_step_indices=list(expected_steps),
            actual_injection_step_indices=list(actual_steps),
            quantized_write_records=write_rows,
        ),
        _operator_fact(
            "scientific_update_records_are_content_bound",
            bool(scientific_update_ready and all(scientific_update_ready)),
            scientific_update_record_count=len(updates),
            scientific_update_record_ready=list(scientific_update_ready),
        ),
        _operator_fact(
            "real_qk_tensor_records_are_present",
            update_qk_ready and detection_qk_ready,
            generation_qk_ready=update_qk_ready,
            detection_qk_ready=detection_qk_ready,
        ),
        _operator_fact(
            "final_three_image_feature_preservation",
            bool(
                _final_image_preservation_ready(result, config)
                and _carrier_only_final_image_preservation_ready(result, config)
            ),
            clean_watermarked_preservation_ready=(
                _final_image_preservation_ready(result, config)
            ),
            carrier_only_preservation_ready=(
                _carrier_only_final_image_preservation_ready(result, config)
            ),
        ),
        _operator_fact(
            "final_image_qk_dual_attribution_gain",
            _final_image_attention_observability_ready(result, config),
        ),
        _operator_fact(
            "scientific_content_binding",
            _scientific_content_binding_record_ready(result),
        ),
        _operator_fact(
            "keyed_prg_cross_platform_known_answer",
            known_answer["keyed_prg_cross_platform_known_answer_ready"],
            known_answer_report_digest=known_answer[
                "known_answer_report_digest"
            ],
        ),
        _operator_fact(
            "runtime_run_decision",
            result.get("run_decision") == "pass",
            run_decision=result.get("run_decision"),
            run_id=result.get("run_id"),
        ),
    )
    ready = bool(facts and all(row["operator_fact_ready"] for row in facts))
    report = {
        "gpu_operator_preflight_schema": "gpu_operator_preflight_report_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_id": result.get("run_id"),
        "operator_facts": list(facts),
        "known_answer_report": known_answer,
        "gpu_operator_preflight_ready": ready,
        "supports_paper_claim": False,
        "paper_claim_boundary": "gpu_method_operator_preflight_only",
    }
    report["gpu_operator_preflight_report_digest"] = build_stable_digest(
        report
    )
    return report


def build_gpu_resource_budget_report(
    resource_observation: Mapping[str, Any] | None,
    registered_budget: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """独立判断资源可执行性, 不改变方法算子真实性结论."""

    observation = dict(resource_observation or {})
    budget = dict(registered_budget or {})
    comparisons = (
        (
            "peak_gpu_memory_bytes",
            "maximum_peak_gpu_memory_bytes",
        ),
        (
            "single_prompt_wall_time_seconds",
            "maximum_single_prompt_wall_time_seconds",
        ),
        (
            "estimated_probe_total_gpu_hours",
            "maximum_estimated_probe_total_gpu_hours",
        ),
    )
    rows: list[dict[str, Any]] = []
    inputs_ready = True
    for observed_field, limit_field in comparisons:
        observed_value = observation.get(observed_field)
        limit_value = budget.get(limit_field)
        row_inputs_ready = bool(
            _finite_nonnegative(observed_value)
            and _finite_nonnegative(limit_value)
        )
        inputs_ready = inputs_ready and row_inputs_ready
        rows.append(
            {
                "observed_field": observed_field,
                "observed_value": observed_value,
                "registered_limit_field": limit_field,
                "registered_limit_value": limit_value,
                "resource_budget_item_ready": bool(
                    row_inputs_ready
                    and float(observed_value) <= float(limit_value)
                ),
            }
        )
    ready = bool(
        inputs_ready and rows and all(row["resource_budget_item_ready"] for row in rows)
    )
    status = (
        "pass"
        if ready
        else "fail"
        if inputs_ready
        else "not_evaluated_missing_observation_or_registered_limit"
    )
    report = {
        "gpu_resource_budget_schema": GPU_RESOURCE_BUDGET_SCHEMA,
        "resource_budget_evaluation_status": status,
        "resource_budget_items": rows,
        "gpu_resource_budget_ready": ready,
        "affects_gpu_operator_preflight_ready": False,
        "supports_paper_claim": False,
    }
    report["gpu_resource_budget_report_digest"] = build_stable_digest(report)
    return report


def build_gpu_method_qualification_report(
    runtime_result: Mapping[str, Any],
    update_records: Sequence[Mapping[str, Any]],
    detection_records: Sequence[Mapping[str, Any]],
    config: SemanticWatermarkRuntimeConfig,
    known_answer_path: str | Path,
    resource_observation: Mapping[str, Any] | None = None,
    registered_budget: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """组合方法和资源报告, 同时保持两个布尔门禁相互独立."""

    operator = build_gpu_operator_preflight_report(
        runtime_result,
        update_records,
        detection_records,
        config,
        known_answer_path,
    )
    resource = build_gpu_resource_budget_report(
        resource_observation,
        registered_budget,
    )
    report = {
        "qualification_report_schema": GPU_METHOD_QUALIFICATION_SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_id": runtime_result.get("run_id"),
        "gpu_operator_preflight": operator,
        "gpu_resource_budget": resource,
        "gpu_operator_preflight_ready": operator[
            "gpu_operator_preflight_ready"
        ],
        "gpu_resource_budget_ready": resource[
            "gpu_resource_budget_ready"
        ],
        "supports_paper_claim": False,
    }
    report["qualification_report_digest"] = build_stable_digest(report)
    return report
