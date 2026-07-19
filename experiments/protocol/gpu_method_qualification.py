"""从真实单 Prompt 运行证据构造 GPU 方法资格化报告."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.runners.image_only_dataset_runtime import (
    _detection_qk_atomic_content_ready,
)
from experiments.runners.semantic_watermark_runtime import (
    SemanticWatermarkRuntimeConfig,
    semantic_watermark_runtime_config_digest,
)
from experiments.protocol.detection_key_identity import (
    REGISTERED_WATERMARK_KEY_ROLE,
    REGISTERED_WRONG_KEY_ROLE,
    build_detection_key_plan_record,
    validate_detection_key_identity_record,
)
from experiments.protocol.image_only_evidence import (
    FrozenEvidenceProtocol,
    complete_evidence_decision,
    validate_frozen_evidence_protocol_integrity,
)
from experiments.protocol.gpu_method_qualification_schema import (
    GPU_METHOD_QUALIFICATION_SCHEMA,
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


GPU_OPERATOR_FACT_SCHEMA = "gpu_operator_preflight_fact_v1"
GPU_RESOURCE_BUDGET_SCHEMA = "gpu_resource_budget_decision_v1"


def _registered_and_wrong_key_attribution_ready(
    detection_records: Sequence[Mapping[str, Any]],
    config: SemanticWatermarkRuntimeConfig,
) -> tuple[bool, dict[str, Any]]:
    """复验同一水印图像上的注册密钥与 wrong-key 归因身份."""

    plan = build_detection_key_plan_record(config.key_material)
    positive = [
        row
        for row in detection_records
        if row.get("sample_role") == "positive_source"
        and not row.get("attack_id")
    ]
    wrong_key = [
        row
        for row in detection_records
        if row.get("sample_role") == "wrong_key_negative"
        and not row.get("attack_id")
    ]
    try:
        positive_identity = validate_detection_key_identity_record(
            positive[0], plan
        )
        wrong_key_identity = validate_detection_key_identity_record(
            wrong_key[0], plan
        )
    except (IndexError, KeyError, TypeError, ValueError):
        positive_identity = {}
        wrong_key_identity = {}
    positive_score = positive[0].get("content_score") if positive else None
    wrong_key_score = wrong_key[0].get("content_score") if wrong_key else None
    score_gain = (
        float(positive_score) - float(wrong_key_score)
        if isinstance(positive_score, (int, float))
        and not isinstance(positive_score, bool)
        and math.isfinite(float(positive_score))
        and isinstance(wrong_key_score, (int, float))
        and not isinstance(wrong_key_score, bool)
        and math.isfinite(float(wrong_key_score))
        else math.nan
    )
    ready = bool(
        len(positive) == len(wrong_key) == 1
        and positive_identity.get("detection_key_role")
        == REGISTERED_WATERMARK_KEY_ROLE
        and wrong_key_identity.get("detection_key_role")
        == REGISTERED_WRONG_KEY_ROLE
        and positive_identity.get("detection_key_material_digest_random")
        != wrong_key_identity.get("detection_key_material_digest_random")
        and positive[0].get("source_image_digest")
        == wrong_key[0].get("source_image_digest")
        and positive[0].get("evaluated_image_digest")
        == wrong_key[0].get("evaluated_image_digest")
        and math.isfinite(score_gain)
        and score_gain > 0.0
    )
    return ready, {
        "detection_key_plan_digest_random": plan[
            "detection_key_plan_digest_random"
        ],
        "registered_key_identity": positive_identity,
        "wrong_key_identity": wrong_key_identity,
        "shared_watermarked_image_ready": bool(
            positive and wrong_key
            and positive[0].get("evaluated_image_digest")
            == wrong_key[0].get("evaluated_image_digest")
        ),
        "registered_key_content_score": positive_score,
        "wrong_key_content_score": wrong_key_score,
        "registered_over_wrong_key_content_score_gain": score_gain,
    }


def _qualification_binding_ready(
    binding: Mapping[str, Any] | None,
    runtime_result: Mapping[str, Any],
    config: SemanticWatermarkRuntimeConfig,
    execution_environment: Mapping[str, Any],
) -> tuple[bool, dict[str, Any]]:
    """复验提交、依赖、模型 revision、Prompt 与运行输入摘要绑定."""

    resolved = dict(binding or {})
    metadata = runtime_result.get("metadata")
    metadata = metadata if isinstance(metadata, Mapping) else {}
    diffusion_source = metadata.get("diffusion_model_source")
    diffusion_source = (
        diffusion_source if isinstance(diffusion_source, Mapping) else {}
    )
    input_summary = resolved.get("input_summary")
    input_summary = input_summary if isinstance(input_summary, Mapping) else {}
    model_revisions = resolved.get("model_revisions")
    model_revisions = (
        model_revisions if isinstance(model_revisions, Mapping) else {}
    )
    reference_identity = resolved.get("content_routing_reference_identity")
    reference_identity = (
        reference_identity if isinstance(reference_identity, Mapping) else {}
    )
    reference_values = reference_identity.get("reference_values")
    reference_values = (
        reference_values if isinstance(reference_values, Mapping) else {}
    )
    digest_payload = {
        field_name: value
        for field_name, value in resolved.items()
        if field_name != "qualification_binding_digest"
    }
    ready = bool(
        len(str(resolved.get("code_version", ""))) == 40
        and resolved.get("code_version")
        == execution_environment.get("formal_execution_commit")
        and resolved.get("dependency_profile_id")
        == execution_environment.get("dependency_profile_id")
        == "sd35_method_runtime_gpu"
        and resolved.get("dependency_profile_digest")
        == execution_environment.get("dependency_profile_digest")
        and resolved.get("complete_hash_lock_digest")
        == execution_environment.get("complete_hash_lock_digest")
        and model_revisions.get("sd35_model_id") == config.model_id
        and model_revisions.get("sd35_model_revision") == config.model_revision
        and model_revisions.get("vae_model_id") == config.model_id
        and model_revisions.get("vae_model_revision") == config.model_revision
        and model_revisions.get("vae_class_name") == config.vae_class_name
        and model_revisions.get("vision_model_id") == config.vision_model_id
        and model_revisions.get("vision_model_revision")
        == config.vision_model_revision
        and diffusion_source.get("repository_id") == config.model_id
        and diffusion_source.get("revision") == config.model_revision
        and _finite_nonnegative(
            reference_values.get("reference_gradient")
        )
        and float(reference_values.get("reference_gradient")) > 0.0
        and _finite_nonnegative(
            reference_values.get("reference_response")
        )
        and float(reference_values.get("reference_response")) > 0.0
        and _finite_nonnegative(
            reference_values.get("reference_sensitivity")
        )
        and float(reference_values.get("reference_sensitivity")) > 0.0
        and reference_identity.get("reference_input_role")
        == "explicit_smoke_only_unqualified"
        and reference_identity.get("supports_paper_claim") is False
        and input_summary.get("prompt_id") == config.prompt_id
        and input_summary.get("prompt_digest")
        == build_stable_digest({"prompt": config.prompt})
        and input_summary.get("method_runtime_config_digest")
        == semantic_watermark_runtime_config_digest(config)
        and resolved.get("qualification_binding_digest")
        == build_stable_digest(digest_payload)
    )
    return ready, resolved


def _formal_same_threshold_decision_ready(
    record: Mapping[str, Any],
    protocol: FrozenEvidenceProtocol,
) -> bool:
    """只接受由冻结 evidence 协议物化的同阈值判定记录。"""

    try:
        validate_frozen_evidence_protocol_integrity(protocol)
    except (TypeError, ValueError):
        return False
    structural_ready = bool(
        record.get("frozen_threshold_digest") == protocol.threshold_digest
        and record.get("frozen_content_threshold")
        == protocol.content_threshold
        and record.get("frozen_rescue_margin_low")
        == protocol.rescue_margin_low
        and record.get("frozen_geometry_score_threshold")
        == protocol.geometry_score_threshold
        and record.get("frozen_registration_confidence_threshold")
        == protocol.registration_confidence_threshold
        and record.get("frozen_attention_sync_score_threshold")
        == protocol.attention_sync_score_threshold
        and record.get("frozen_image_only_measurement_config_digest")
        == protocol.image_only_measurement_config_digest
        and record.get("frozen_attention_geometry_enabled")
        is protocol.attention_geometry_enabled
        and record.get("frozen_image_alignment_enabled")
        is protocol.image_alignment_enabled
        and record.get("frozen_geometry_rescue_enabled")
        is protocol.geometry_rescue_enabled
        and record.get("lf_carrier_protocol_digest")
        == protocol.lf_carrier_protocol_digest
        and record.get("tail_carrier_protocol_digest")
        == protocol.tail_carrier_protocol_digest
        and record.get("lf_weight") == protocol.lf_weight
        and record.get("tail_robust_weight")
        == protocol.tail_robust_weight
        and record.get("tail_fraction") == protocol.tail_fraction
        and type(record.get("formal_positive_by_content")) is bool
        and type(record.get("formal_geometry_reliable")) is bool
        and type(record.get("formal_rescue_eligible")) is bool
        and type(record.get("formal_rescue_applied")) is bool
        and type(record.get("formal_evidence_positive")) is bool
        and record.get("formal_metric_status")
        == "measured_image_only_detection"
        and record.get("frozen_image_only_measurement_config_digest")
        == record.get("image_only_measurement_config_digest")
    )
    if not structural_ready:
        return False
    try:
        decision = complete_evidence_decision(
            dict(record),
            content_threshold=record["frozen_content_threshold"],
            geometry_rescue_enabled=record["frozen_geometry_rescue_enabled"],
            rescue_margin_low=record["frozen_rescue_margin_low"],
            geometry_score_threshold=record[
                "frozen_geometry_score_threshold"
            ],
            registration_confidence_threshold=record[
                "frozen_registration_confidence_threshold"
            ],
            attention_sync_score_threshold=record[
                "frozen_attention_sync_score_threshold"
            ],
        )
    except (KeyError, TypeError, ValueError):
        return False
    return bool(
        record["formal_positive_by_content"] is decision.positive_by_content
        and record["formal_geometry_reliable"]
        is decision.calibrated_geometry_reliable
        and record["formal_rescue_eligible"] is decision.rescue_eligible
        and record["formal_rescue_applied"] is decision.rescue_applied
        and record["formal_evidence_positive"] is decision.evidence_positive
    )


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


def _finite_number(value: Any) -> bool:
    """判断值为有限实数并排除 bool。"""

    return bool(
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
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


def build_gpu_operator_preflight_report(
    runtime_result: Mapping[str, Any],
    update_records: Sequence[Mapping[str, Any]],
    detection_records: Sequence[Mapping[str, Any]],
    config: SemanticWatermarkRuntimeConfig,
    known_answer_path: str | Path,
    qualification_binding: Mapping[str, Any] | None = None,
    frozen_evidence_protocol: FrozenEvidenceProtocol | None = None,
) -> dict[str, Any]:
    """验证单 Prompt 方法机制真实性, 不消费资源预算阈值."""

    result = dict(runtime_result)
    updates = tuple(dict(row) for row in update_records)
    detections = tuple(dict(row) for row in detection_records)
    update = updates[0] if len(updates) == 1 else {}
    branch_fields = (
        "lf_effective_l2",
        "hf_tail_effective_l2",
        "geometry_effective_l2",
    )
    single_write_ready = bool(
        config.injection_step_indices == (10,)
        and len(updates) == 1
        and update.get("step_index") == 10
        and update.get("captured_previous_index") == 9
        and update.get("captured_previous_count") == 1
        and update.get("callback_write_index") == 10
        and update.get("callback_write_count") == 1
        and update.get("actual_dtype_single_write_count") == 1
        and update.get("current_image_decode_count") == 1
        and update.get("public_probe_additional_decode_count") == 1
        and update.get("method_role") == "full_dual_chain"
        and all(
            _finite_nonnegative(update.get(field_name))
            and float(update[field_name]) > 0.0
            for field_name in branch_fields
        )
        and _finite_nonnegative(update.get("combined_effective_l2"))
        and float(update.get("combined_effective_l2", 0.0)) > 0.0
        and _finite_nonnegative(update.get("combined_effective_l2_limit"))
        and float(update.get("combined_effective_l2", math.inf))
        <= float(update.get("combined_effective_l2_limit", -math.inf))
        and update.get("combined_effective_l2_ready") is True
        and update.get("post_write_qk_strict_ready") is True
        and _finite_number(update.get("content_only_postwrite_qk_score"))
        and _finite_number(update.get("final_postwrite_qk_score"))
        and float(update.get("final_postwrite_qk_score", -math.inf))
        > float(update.get("content_only_postwrite_qk_score", math.inf))
        and update.get("attention_module_names")
        == list(config.attention_module_names)
    )
    update_qk_ready = bool(
        single_write_ready
        and all(
            type(update.get(field_name)) is str
            and len(update[field_name]) == 64
            and all(
                character in "0123456789abcdef"
                for character in update[field_name]
            )
            for field_name in (
                "geometry_qk_atomic_records_digest",
                "content_only_postwrite_qk_digest",
                "final_postwrite_qk_digest",
            )
        )
    )
    detection_qk_ready = bool(
        detections
        and all(_detection_qk_atomic_content_ready(row, config) for row in detections)
    )
    threshold_free_detection_ready = bool(
        len(detections) == 3
        and {row.get("sample_role") for row in detections}
        == {"clean_negative", "positive_source", "wrong_key_negative"}
        and all(
            _finite_number(row.get("lf_score"))
            and _finite_number(row.get("tail_robust_score"))
            and _finite_number(row.get("content_score"))
            and row.get("metadata", {}).get("measurement_status")
            == "threshold_independent_image_only_evidence"
            for row in detections
        )
    )
    try:
        if frozen_evidence_protocol is None:
            raise ValueError("正式资格化缺少冻结 evidence protocol")
        validate_frozen_evidence_protocol_integrity(frozen_evidence_protocol)
        frozen_evidence_protocol_ready = True
    except (TypeError, ValueError):
        frozen_evidence_protocol_ready = False
    blind_detection_ready = bool(
        threshold_free_detection_ready
        and frozen_evidence_protocol_ready
        and all(
            _formal_same_threshold_decision_ready(
                row,
                frozen_evidence_protocol,
            )
            for row in detections
        )
        and len({row["frozen_threshold_digest"] for row in detections}) == 1
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
    binding_ready, resolved_binding = _qualification_binding_ready(
        qualification_binding,
        result,
        config,
        execution_environment,
    )
    key_attribution_ready, key_attribution_evidence = (
        _registered_and_wrong_key_attribution_ready(detections, config)
    )
    known_answer = rebuild_keyed_prg_known_answer_report(known_answer_path)
    metadata = result.get("metadata")
    metadata = metadata if isinstance(metadata, Mapping) else {}
    runtime_identity_text = json.dumps(metadata, ensure_ascii=False, sort_keys=True)
    legacy_runtime_absent = all(
        token not in runtime_identity_text
        for token in (
            "semantic_feature_operator_contract",
            "complete_716",
            "exact_jvp",
            "exact_vjp",
            "psd_cg",
        )
    )
    legacy_runtime_absent = bool(
        legacy_runtime_absent
        and metadata.get("legacy_runtime_dependency_absence_ready") is True
        and metadata.get("forbidden_runtime_modules")
        == [
            "main.methods.subspace.jacobian_nullspace",
            "main.methods.semantic.runtime",
        ]
    )
    facts = (
        _operator_fact(
            "exact_commit_dependency_model_and_input_binding",
            binding_ready,
            qualification_binding=resolved_binding,
        ),
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
            "formal_s_t_r_q_lf_hf_qk_common_gamma_single_write",
            single_write_ready,
            update_record=update,
        ),
        _operator_fact(
            "formal_image_only_lf_hf_tail_blind_detection",
            blind_detection_ready and detection_qk_ready,
            detection_record_count=len(detections),
            detection_qk_ready=detection_qk_ready,
            threshold_free_measurement_ready=threshold_free_detection_ready,
            frozen_evidence_protocol_ready=frozen_evidence_protocol_ready,
            frozen_threshold_digest=(
                frozen_evidence_protocol.threshold_digest
                if frozen_evidence_protocol_ready
                else None
            ),
        ),
        _operator_fact(
            "legacy_716_jvp_vjp_psd_cg_multi_injection_absent",
            legacy_runtime_absent,
        ),
        _operator_fact(
            "registered_key_and_wrong_key_attribution",
            key_attribution_ready,
            **key_attribution_evidence,
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
        "qualification_binding": resolved_binding,
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
    qualification_binding: Mapping[str, Any] | None = None,
    frozen_evidence_protocol: FrozenEvidenceProtocol | None = None,
) -> dict[str, Any]:
    """组合方法和资源报告, 同时保持两个布尔门禁相互独立."""

    operator = build_gpu_operator_preflight_report(
        runtime_result,
        update_records,
        detection_records,
        config,
        known_answer_path,
        qualification_binding,
        frozen_evidence_protocol,
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
