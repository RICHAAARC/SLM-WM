"""在 70/700/7000 Prompt 协议上运行真实方法并冻结完整检测判定。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Iterable
from zipfile import ZIP_STORED, ZipFile

from experiments.protocol.paper_run_config import (
    PaperRunConfig,
    build_paper_run_config,
    normalize_paper_run_name,
)
from experiments.protocol.prompts import build_prompt_records, read_prompt_file
from experiments.protocol.splits import apply_split_assignments, build_group_split_counts
from experiments.protocol.attacks import default_attack_configs
from experiments.runtime import repository_environment
from experiments.runtime.scientific_execution_binding import (
    validate_scientific_execution_binding,
)
from experiments.runtime.package_input_manifest import (
    collect_exact_package_entries,
    validate_exact_package_archive,
    write_exact_package_input_manifest,
)
from experiments.runtime.resume_checkpoint import (
    clear_progress_checkpoints,
    persist_progress_checkpoint,
    restore_role_checkpoints,
)
from experiments.runtime.scientific_unit_provenance import (
    SCIENTIFIC_UNIT_PROVENANCE_AGGREGATE_FIELDS,
    aggregate_scientific_unit_provenance,
)
from experiments.runtime.diffusion.regeneration_attacks import default_diffusion_attack_specs
from experiments.runtime.diffusion.semantic_features import (
    JOINT_FEATURE_WIDTH,
    SEMANTIC_FEATURE_SCHEMA,
    SEMANTIC_FEATURE_WIDTH,
    VISUAL_FEATURE_SCHEMA,
    VISUAL_FEATURE_WIDTH,
)
from experiments.runners.semantic_watermark_runtime import (
    SemanticWatermarkRuntimeConfig,
    load_completed_semantic_watermark_runtime_result,
    load_semantic_watermark_runtime_context,
    semantic_watermark_runtime_config_payload,
    validate_semantic_watermark_runtime_result_provenance,
    write_semantic_watermark_runtime_outputs,
)
from experiments.runtime.repository_environment import file_digest
from experiments.runtime.archive_naming import utc_archive_token
from experiments.artifacts.artifact_manifest import build_artifact_manifest
from experiments.artifacts.detection_score_curves import (
    build_detection_score_tables,
    write_detection_score_tables,
)
from experiments.artifacts.image_only_detection_metrics import (
    build_image_only_test_metric_rows,
)
from main.methods.carrier import keyed_prg_protocol_record
from main.core.digest import build_stable_digest
from main.methods.geometry import (
    ATTENTION_RELATION_COMPONENT_NAMES,
    DIRECT_QK_RELATION_SOURCE,
)


PACKAGE_INPUT_MANIFEST_FILE_NAME = "image_only_dataset_package_input_manifest.json"


@dataclass(frozen=True)
class FrozenEvidenceProtocol:
    """保存 calibration split 冻结的完整 evidence 判定参数。"""

    content_threshold: float
    rescue_margin_low: float
    geometry_score_threshold: float
    registration_confidence_threshold: float
    attention_sync_score_threshold: float
    geometry_calibration_negative_count: int
    geometry_calibration_exceedance_count: int
    registration_calibration_negative_count: int
    registration_calibration_exceedance_count: int
    sync_calibration_negative_count: int
    sync_calibration_exceedance_count: int
    geometry_protocol_calibration_ready: bool
    calibration_negative_count: int
    calibration_false_positive_count: int
    calibration_false_positive_rate: float
    target_fpr: float
    threshold_digest: str

    def to_dict(self) -> dict[str, Any]:
        """转为可写入 JSON 的字典。"""

        return asdict(self)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """读取 JSONL 记录。"""

    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _decision(
    record: dict[str, Any],
    threshold: float,
    rescue_margin_low: float,
    geometry_score_threshold: float = 0.0,
    registration_confidence_threshold: float = 0.0,
    attention_sync_score_threshold: float = 0.0,
) -> tuple[bool, bool, bool, str]:
    """用冻结阈值重算内容主判和同阈值几何救回。"""

    raw_score = float(record["content_score"])
    raw_margin = raw_score - threshold
    positive_by_content = raw_margin >= 0.0
    aligned_score = record.get("aligned_content_score")
    alignment = record.get("alignment")
    if isinstance(alignment, dict):
        alignment_reliable = bool(
            alignment.get(
                "registration_geometry_reliable",
                alignment.get("geometry_reliable", False),
            )
        )
    else:
        alignment_reliable = False
    geometry_score = record.get("attention_geometry_score")
    registration_confidence = record.get("registration_confidence")
    attention_sync_score = record.get("attention_sync_score")
    geometry_reliable = (
        alignment_reliable
        and isinstance(geometry_score, (int, float))
        and math.isfinite(float(geometry_score))
        and float(geometry_score) >= geometry_score_threshold
        and isinstance(registration_confidence, (int, float))
        and math.isfinite(float(registration_confidence))
        and float(registration_confidence) >= registration_confidence_threshold
        and isinstance(attention_sync_score, (int, float))
        and math.isfinite(float(attention_sync_score))
        and float(attention_sync_score) >= attention_sync_score_threshold
    )
    if positive_by_content:
        failure_reason = "content_positive"
    elif rescue_margin_low <= raw_margin < 0.0 and geometry_reliable:
        failure_reason = "geometry_suspected"
    elif rescue_margin_low <= raw_margin < 0.0:
        failure_reason = "low_confidence"
    else:
        failure_reason = "content_evidence_absent"
    rescue_eligible = (
        rescue_margin_low <= raw_margin < 0.0
        and geometry_reliable
        and aligned_score is not None
        and failure_reason in {"geometry_suspected", "low_confidence"}
    )
    rescue_applied = rescue_eligible and float(aligned_score) - threshold >= 0.0
    return positive_by_content, rescue_applied, positive_by_content or rescue_applied, failure_reason


def calibrate_complete_evidence_protocol(
    calibration_records: Iterable[dict[str, Any]],
    target_fpr: float,
    rescue_margin_low: float,
) -> FrozenEvidenceProtocol:
    """在 clean negative 上冻结包含 rescue 的完整判定协议。

    阈值搜索直接调用最终布尔决策, 因而不会把“内容阈值达到目标 FPR”错误地
    等同于“加入几何救回后仍达到目标 FPR”。
    """

    records = tuple(calibration_records)
    if not records:
        raise ValueError("calibration clean negative 记录不得为空")
    if not 0.0 < target_fpr < 1.0:
        raise ValueError("target_fpr 必须位于 (0, 1)")
    allowed_false_positives = max(0, math.floor(target_fpr * (len(records) + 1)) - 1)
    def freeze_geometry_gate(field_name: str) -> tuple[float, int, int]:
        """从全部未删失 clean negatives 冻结单个几何门禁。"""

        values = tuple(
            float(record[field_name])
            for record in records
            if isinstance(record.get(field_name), (int, float))
            and math.isfinite(float(record[field_name]))
        )
        if not values:
            return 0.0, 0, 0
        candidates = sorted({math.nextafter(value, math.inf) for value in values})
        selected = candidates[-1]
        selected_count = sum(value >= selected for value in values)
        for candidate in candidates:
            exceedance_count = sum(value >= candidate for value in values)
            if exceedance_count <= allowed_false_positives:
                selected = candidate
                selected_count = exceedance_count
                break
        return selected, len(values), selected_count

    (
        geometry_score_threshold,
        geometry_negative_count,
        geometry_exceedance_count,
    ) = freeze_geometry_gate("attention_geometry_score")
    (
        registration_confidence_threshold,
        registration_negative_count,
        registration_exceedance_count,
    ) = freeze_geometry_gate("registration_confidence")
    (
        attention_sync_score_threshold,
        sync_negative_count,
        sync_exceedance_count,
    ) = freeze_geometry_gate("attention_sync_score")
    geometry_protocol_calibration_ready = (
        geometry_negative_count == len(records)
        and registration_negative_count == len(records)
        and sync_negative_count == len(records)
        and all(
            isinstance(record.get("alignment"), dict)
            and isinstance(
                record["alignment"].get(
                    "registration_geometry_reliable",
                    record["alignment"].get("geometry_reliable"),
                ),
                bool,
            )
            for record in records
        )
    )
    score_candidates = []
    for record in records:
        score_candidates.append(float(record["content_score"]))
        if record.get("aligned_content_score") is not None:
            score_candidates.append(float(record["aligned_content_score"]))
    thresholds = sorted({math.nextafter(value, math.inf) for value in score_candidates})
    selected_threshold = thresholds[-1]
    selected_false_positives = 0
    for threshold in thresholds:
        false_positives = sum(
            _decision(
                record,
                threshold,
                rescue_margin_low,
                geometry_score_threshold,
                registration_confidence_threshold,
                attention_sync_score_threshold,
            )[2]
            for record in records
        )
        if false_positives <= allowed_false_positives:
            selected_threshold = threshold
            selected_false_positives = false_positives
            break
    payload = {
        "content_threshold": selected_threshold,
        "rescue_margin_low": rescue_margin_low,
        "geometry_score_threshold": geometry_score_threshold,
        "registration_confidence_threshold": registration_confidence_threshold,
        "attention_sync_score_threshold": attention_sync_score_threshold,
        "geometry_calibration_negative_count": geometry_negative_count,
        "geometry_calibration_exceedance_count": geometry_exceedance_count,
        "registration_calibration_negative_count": registration_negative_count,
        "registration_calibration_exceedance_count": registration_exceedance_count,
        "sync_calibration_negative_count": sync_negative_count,
        "sync_calibration_exceedance_count": sync_exceedance_count,
        "geometry_protocol_calibration_ready": geometry_protocol_calibration_ready,
        "calibration_negative_count": len(records),
        "calibration_false_positive_count": selected_false_positives,
        "target_fpr": target_fpr,
        "decision_scope": "content_or_same_threshold_aligned_content_rescue",
    }
    return FrozenEvidenceProtocol(
        content_threshold=selected_threshold,
        rescue_margin_low=rescue_margin_low,
        geometry_score_threshold=geometry_score_threshold,
        registration_confidence_threshold=registration_confidence_threshold,
        attention_sync_score_threshold=attention_sync_score_threshold,
        geometry_calibration_negative_count=geometry_negative_count,
        geometry_calibration_exceedance_count=geometry_exceedance_count,
        registration_calibration_negative_count=registration_negative_count,
        registration_calibration_exceedance_count=registration_exceedance_count,
        sync_calibration_negative_count=sync_negative_count,
        sync_calibration_exceedance_count=sync_exceedance_count,
        geometry_protocol_calibration_ready=geometry_protocol_calibration_ready,
        calibration_negative_count=len(records),
        calibration_false_positive_count=selected_false_positives,
        calibration_false_positive_rate=selected_false_positives / len(records),
        target_fpr=target_fpr,
        threshold_digest=build_stable_digest(payload),
    )


def apply_frozen_evidence_protocol(
    records: Iterable[dict[str, Any]],
    protocol: FrozenEvidenceProtocol,
) -> tuple[dict[str, Any], ...]:
    """对全部 split 和攻击记录应用同一冻结协议。"""

    resolved = []
    for record in records:
        positive_by_content, rescue_applied, evidence_positive, failure_reason = _decision(
            record,
            protocol.content_threshold,
            protocol.rescue_margin_low,
            protocol.geometry_score_threshold,
            protocol.registration_confidence_threshold,
            protocol.attention_sync_score_threshold,
        )
        raw_margin = float(record["content_score"]) - protocol.content_threshold
        aligned_score = record.get("aligned_content_score")
        resolved.append(
            {
                **record,
                "frozen_content_threshold": protocol.content_threshold,
                "frozen_geometry_score_threshold": protocol.geometry_score_threshold,
                "frozen_registration_confidence_threshold": (
                    protocol.registration_confidence_threshold
                ),
                "frozen_attention_sync_score_threshold": (
                    protocol.attention_sync_score_threshold
                ),
                "frozen_threshold_digest": protocol.threshold_digest,
                "formal_raw_content_margin": raw_margin,
                "formal_aligned_content_margin": (
                    None if aligned_score is None else float(aligned_score) - protocol.content_threshold
                ),
                "formal_positive_by_content": positive_by_content,
                "formal_content_failure_reason": failure_reason,
                "formal_rescue_applied": rescue_applied,
                "formal_evidence_positive": evidence_positive,
                "formal_metric_status": "measured_image_only_detection",
                "supports_paper_claim": False,
            }
        )
    return tuple(resolved)


def _write_csv(path: Path, rows: tuple[dict[str, Any], ...]) -> None:
    """写出列集合稳定的 CSV。"""

    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _scientific_update_record_ready(
    record: dict[str, Any],
    config: SemanticWatermarkRuntimeConfig,
) -> bool:
    """验证单个注入记录确实执行全部关键科学算子。"""

    def finite_at_least(value: Any, minimum: float, *, strict: bool = False) -> bool:
        """集中校验一个受治理数值是否有限并满足下界。"""

        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            return False
        return float(value) > minimum if strict else float(value) >= minimum

    null_space_records = record.get("null_space_records")
    if not isinstance(null_space_records, dict) or set(null_space_records) != {
        "lf_content",
        "tail_robust",
        "attention_geometry",
    }:
        return False
    allowed_jvp_modes = {
        "torch_func_exact_jvp_vjp",
        "torch_autograd_exact_jvp_vjp_compatibility",
    }
    expected_prg_digest = keyed_prg_protocol_record(
        config.keyed_prg_version
    )["keyed_prg_protocol_digest"]
    for subspace_record in null_space_records.values():
        metadata = subspace_record.get("metadata", {})
        numeric_values = (
            subspace_record.get("response_residual"),
            subspace_record.get("relative_response_residual"),
            subspace_record.get("orthogonality_error"),
        )
        if not all(isinstance(value, (int, float)) and math.isfinite(float(value)) for value in numeric_values):
            return False
        if float(subspace_record["relative_response_residual"]) > config.maximum_relative_response_residual:
            return False
        if float(subspace_record["orthogonality_error"]) > 1e-4:
            return False
        if metadata.get("jvp_mode") not in allowed_jvp_modes:
            return False
        if metadata.get("solver") != "matrix_free_full_jacobian_psd_cg":
            return False
        if subspace_record.get("cg_converged") is not True:
            return False
        if int(metadata.get("preferred_direction_count", 0)) < 1:
            return False
        if metadata.get("semantic_feature_schema") != SEMANTIC_FEATURE_SCHEMA:
            return False
        if metadata.get("visual_feature_schema") != VISUAL_FEATURE_SCHEMA:
            return False
        if int(metadata.get("semantic_feature_width", 0)) != SEMANTIC_FEATURE_WIDTH:
            return False
        if int(metadata.get("visual_feature_width", 0)) != VISUAL_FEATURE_WIDTH:
            return False
        if int(metadata.get("joint_feature_width", 0)) != JOINT_FEATURE_WIDTH:
            return False
        if metadata.get("feature_compression_applied") is not False:
            return False
        if (
            metadata.get("keyed_prg_version") != config.keyed_prg_version
            or metadata.get("keyed_prg_protocol_digest")
            != expected_prg_digest
        ):
            return False
        column_residuals = subspace_record.get(
            "column_relative_response_residuals"
        )
        energy_retentions = subspace_record.get("projection_energy_retentions")
        cg_residuals = subspace_record.get("cg_relative_residuals")
        if not all(
            isinstance(values, list)
            and len(values) == config.null_rank
            and all(isinstance(value, (int, float)) and math.isfinite(float(value)) for value in values)
            for values in (column_residuals, energy_retentions, cg_residuals)
        ):
            return False
        if any(
            float(value) > config.maximum_relative_response_residual
            for value in column_residuals
        ):
            return False
        if any(
            float(value) < config.minimum_projection_energy_retention
            for value in energy_retentions
        ):
            return False
        if any(
            float(value) > config.null_space_cg_relative_tolerance
            for value in cg_residuals
        ):
            return False
    quantized_update_sha256 = str(
        record.get("quantized_write_update_content_sha256", "")
    )
    quantized_relative_response = record.get(
        "quantized_write_relative_jacobian_response"
    )
    if (
        len(quantized_update_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in quantized_update_sha256
        )
        or record.get("quantized_write_jacobian_gate_applicable") is not True
        or record.get("quantized_write_jacobian_gate_ready") is not True
        or record.get("quantized_write_jacobian_status")
        != "measured_from_actual_quantized_latent_delta"
        or not isinstance(quantized_relative_response, (int, float))
        or not math.isfinite(float(quantized_relative_response))
        or float(quantized_relative_response)
        > config.maximum_quantized_write_relative_jacobian_response
        or record.get(
            "maximum_quantized_write_relative_jacobian_response"
        )
        != config.maximum_quantized_write_relative_jacobian_response
        or record.get("keyed_prg_version") != config.keyed_prg_version
        or record.get("keyed_prg_protocol_digest") != expected_prg_digest
    ):
        return False
    if not finite_at_least(
        record.get("lf_projection_energy_retention"),
        config.minimum_projection_energy_retention,
    ):
        return False
    if not finite_at_least(
        record.get("tail_projection_energy_retention"),
        config.minimum_projection_energy_retention,
    ):
        return False
    if not finite_at_least(record.get("attention_score_gain"), 0.0, strict=True):
        return False
    if not finite_at_least(record.get("attention_applied_update_strength"), 0.0, strict=True):
        return False
    stable_token_indices = record.get("stable_token_indices")
    if (
        not isinstance(stable_token_indices, list)
        or len(stable_token_indices) < 4
        or len(set(stable_token_indices)) != len(stable_token_indices)
    ):
        return False
    stable_token_selection_digest = str(
        record.get("stable_token_selection_digest", "")
    )
    if len(stable_token_selection_digest) != 64 or any(
        character not in "0123456789abcdef"
        for character in stable_token_selection_digest
    ):
        return False
    for digest_field in (
        "stable_pair_weight_identity_digest",
        "stable_pair_weight_realization_digest",
        "attention_relation_component_identity_digest",
        "attention_relation_keyed_projection_digest",
    ):
        digest = str(record.get(digest_field, ""))
        if len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            return False
    if (
        record.get("attention_relation_component_names")
        != list(ATTENTION_RELATION_COMPONENT_NAMES)
        or record.get("attention_relation_source")
        != DIRECT_QK_RELATION_SOURCE
        or record.get("attention_relation_direct_qk_source_ready") is not True
        or record.get("attention_relation_probability_scope")
        != "sampled_image_token_qk_relation_probability"
    ):
        return False
    semantic_cosine = record.get("full_semantic_cosine_similarity")
    visual_relative_drift = record.get("full_visual_feature_relative_drift")
    if not finite_at_least(
        semantic_cosine,
        config.minimum_semantic_preservation_cosine,
    ):
        return False
    if (
        not isinstance(visual_relative_drift, (int, float))
        or not math.isfinite(float(visual_relative_drift))
        or float(visual_relative_drift)
        > config.maximum_visual_feature_relative_drift
    ):
        return False
    if record.get("semantic_preservation_gate_ready") is not True:
        return False
    branch_risk_records = record.get("branch_risk_records")
    return (
        bool(record.get("branch_risk_bundle_digest"))
        and isinstance(branch_risk_records, dict)
        and set(branch_risk_records) == {"lf_content", "tail_robust", "attention_geometry"}
        and all(int(value.get("eligible_position_count", 0)) > 0 for value in branch_risk_records.values())
    )


def _final_image_preservation_ready(
    result: dict[str, Any],
    config: SemanticWatermarkRuntimeConfig,
) -> bool:
    """验证一次运行的最终成图累计完整特征门禁。"""

    record = result.get("metadata", {}).get("final_image_preservation", {})
    semantic_cosine = record.get("final_image_semantic_cosine_similarity")
    visual_drift = record.get("final_image_visual_feature_relative_drift")
    return bool(
        record.get("final_image_preservation_gate_ready") is True
        and isinstance(semantic_cosine, (int, float))
        and math.isfinite(float(semantic_cosine))
        and float(semantic_cosine) >= config.minimum_semantic_preservation_cosine
        and isinstance(visual_drift, (int, float))
        and math.isfinite(float(visual_drift))
        and float(visual_drift) <= config.maximum_visual_feature_relative_drift
    )


def _carrier_only_final_image_preservation_ready(
    result: dict[str, Any],
    config: SemanticWatermarkRuntimeConfig,
) -> bool:
    """验证 clean 到 carrier-only 的最终内容保持与产物身份绑定。"""

    if not config.attention_geometry_enabled:
        return True
    metadata = result.get("metadata", {})
    record = metadata.get("carrier_only_final_image_preservation") or {}
    observability = metadata.get("final_image_attention_observability") or {}
    semantic_cosine = record.get(
        "carrier_only_final_image_semantic_cosine_similarity"
    )
    visual_drift = record.get(
        "carrier_only_final_image_visual_feature_relative_drift"
    )
    identity_digest = str(
        record.get("carrier_only_counterfactual_identity_digest", "")
    )
    observability_identity_digest = str(
        observability.get("carrier_only_counterfactual_identity_digest", "")
    )
    image_path = str(record.get("carrier_only_counterfactual_image_path", ""))
    observability_image_path = str(
        observability.get("carrier_only_counterfactual_image_path", "")
    )
    image_digest = str(
        record.get("carrier_only_counterfactual_image_digest", "")
    )
    observability_image_digest = str(
        observability.get("carrier_only_counterfactual_image_digest", "")
    )
    return bool(
        record.get("carrier_only_final_image_preservation_applicable") is True
        and record.get("carrier_only_final_image_preservation_gate_ready") is True
        and isinstance(semantic_cosine, (int, float))
        and math.isfinite(float(semantic_cosine))
        and float(semantic_cosine) >= config.minimum_semantic_preservation_cosine
        and isinstance(visual_drift, (int, float))
        and math.isfinite(float(visual_drift))
        and float(visual_drift) <= config.maximum_visual_feature_relative_drift
        and len(identity_digest) == 64
        and identity_digest == observability_identity_digest
        and all(character in "0123456789abcdef" for character in identity_digest)
        and bool(image_path)
        and image_path == observability_image_path
        and len(image_digest) == 64
        and image_digest == observability_image_digest
        and all(character in "0123456789abcdef" for character in image_digest)
    )


def _final_image_attention_observability_ready(
    result: dict[str, Any],
    config: SemanticWatermarkRuntimeConfig,
) -> bool:
    """验证最终成图重编码的真实 Q/K 水印增益证据。"""

    if not config.attention_geometry_enabled:
        return True
    record = result.get("metadata", {}).get(
        "final_image_attention_observability",
        {},
    )
    blind_gain = record.get("final_image_attention_blind_attribution_gain")
    paired_gain = record.get(
        "final_image_attention_carrier_paired_attribution_gain"
    )
    paired_digest = str(
        record.get("final_carrier_only_pair_weight_identity_digest", "")
    )
    record_schema_digest = str(
        record.get("final_image_attention_record_schema_digest", "")
    )
    component_identity_digest = str(
        record.get("attention_relation_component_identity_digest", "")
    )
    keyed_projection_digest = str(
        record.get("attention_relation_keyed_projection_digest", "")
    )
    component_names = record.get("attention_relation_component_names")
    paired_component_gains = record.get(
        "final_image_attention_carrier_paired_component_gains"
    )
    counterfactual_digests = tuple(
        str(record.get(field_name, ""))
        for field_name in (
            "carrier_only_counterfactual_identity_digest",
            "carrier_only_counterfactual_config_digest",
            "carrier_only_counterfactual_update_records_digest",
            "carrier_only_counterfactual_scheduler_trace_digest",
        )
    )
    return bool(
        record.get("final_image_attention_observability_applicable") is True
        and record.get("carrier_only_counterfactual_ready") is True
        and record.get("carrier_only_counterfactual_changed_fields")
        == ["attention_geometry_enabled"]
        and record.get(
            "carrier_only_counterfactual_scheduler_identity_ready"
        )
        is True
        and record.get(
            "carrier_only_counterfactual_attention_geometry_enabled"
        )
        is False
        and record.get("final_image_attention_observability_gate_ready") is True
        and record.get("final_image_attention_observability_requires_gpu") is True
        and record.get(
            "final_image_attention_observability_gpu_execution_verified"
        )
        is True
        and record.get("final_image_attention_observability_source")
        == "image_reencoded_public_noise_real_qk"
        and record.get("attention_relation_source")
        == DIRECT_QK_RELATION_SOURCE
        and record.get("attention_relation_direct_qk_source_ready") is True
        and record.get("attention_relation_probability_scope")
        == "sampled_image_token_qk_relation_probability"
        and component_names == list(ATTENTION_RELATION_COMPONENT_NAMES)
        and len(component_identity_digest) == 64
        and all(
            character in "0123456789abcdef"
            for character in component_identity_digest
        )
        and len(keyed_projection_digest) == 64
        and all(
            character in "0123456789abcdef"
            for character in keyed_projection_digest
        )
        and isinstance(paired_component_gains, dict)
        and set(paired_component_gains) == set(ATTENTION_RELATION_COMPONENT_NAMES)
        and all(
            isinstance(value, (int, float)) and math.isfinite(float(value))
            for value in paired_component_gains.values()
        )
        and isinstance(blind_gain, (int, float))
        and math.isfinite(float(blind_gain))
        and float(blind_gain) > config.minimum_final_image_attention_score_gain
        and isinstance(paired_gain, (int, float))
        and math.isfinite(float(paired_gain))
        and float(paired_gain) > config.minimum_final_image_attention_score_gain
        and len(paired_digest) == 64
        and all(character in "0123456789abcdef" for character in paired_digest)
        and len(record_schema_digest) == 64
        and all(
            character in "0123456789abcdef"
            for character in record_schema_digest
        )
        and all(
            len(digest) == 64
            and all(character in "0123456789abcdef" for character in digest)
            for digest in counterfactual_digests
        )
        and _carrier_only_final_image_preservation_ready(result, config)
    )


def run_image_only_dataset_runtime(
    base_method_config: SemanticWatermarkRuntimeConfig,
    root: str | Path = ".",
    paper_run: PaperRunConfig | None = None,
    max_new_prompts_per_session: int = 0,
) -> dict[str, Any]:
    """运行当前论文规模的全部 Prompt 并生成可校准记录。"""

    root_path = Path(root).resolve()
    formal_execution_run_lock = (
        repository_environment.require_published_formal_execution_lock(root_path)
    )
    resolved_paper_run = paper_run or build_paper_run_config(root_path)
    prompt_path = (root_path / resolved_paper_run.prompt_file).resolve()
    prompt_records = apply_split_assignments(
        build_prompt_records(
            resolved_paper_run.prompt_set,
            read_prompt_file(prompt_path),
        )
    )[: resolved_paper_run.sample_count]
    output_dir = root_path / "outputs" / "image_only_dataset_runtime" / resolved_paper_run.run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "dataset_runtime_progress.json"
    restore_role_checkpoints(
        repository_root=root_path,
        artifact_role="image_only_dataset_runtime",
        paper_run_name=resolved_paper_run.run_name,
        allowed_output_prefix=(
            f"outputs/image_only_dataset_runtime/{resolved_paper_run.run_name}"
        ),
    )
    if max_new_prompts_per_session < 0:
        raise ValueError("max_new_prompts_per_session 不得为负")
    shared_context = None
    attack_prompt_ids = {
        record.prompt_id
        for record in tuple(record for record in prompt_records if record.split == "test")[
            : resolved_paper_run.minimum_clean_negative_count
        ]
    }
    runtime_results = []
    detection_records: list[dict[str, Any]] = []
    scientific_update_records: list[dict[str, Any]] = []
    completed_prompt_ids: list[str] = []
    resumed_prompt_count = 0
    new_prompt_count = 0

    def write_resume_progress() -> dict[str, Any]:
        """原子保存当前 Prompt 进度并同步到外部检查点目录."""

        progress = {
            "paper_run_name": resolved_paper_run.run_name,
            "prompt_count": len(prompt_records),
            "completed_prompt_count": len(runtime_results),
            "remaining_prompt_count": len(prompt_records) - len(runtime_results),
            "resumed_prompt_count": resumed_prompt_count,
            "new_prompt_count": new_prompt_count,
            "max_new_prompts_per_session": max_new_prompts_per_session,
            "completed_prompt_digest": build_stable_digest(
                sorted(completed_prompt_ids)
            ),
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
            artifact_role="image_only_dataset_runtime",
            paper_run_name=resolved_paper_run.run_name,
        )
        return progress

    for prompt_record in prompt_records:
        run_attacks = prompt_record.prompt_id in attack_prompt_ids
        run_config = replace(
            base_method_config,
            prompt=prompt_record.prompt_text,
            prompt_id=prompt_record.prompt_id,
            split=prompt_record.split,
            seed=base_method_config.seed + prompt_record.prompt_index,
            inference_steps=resolved_paper_run.inference_steps,
            guidance_scale=resolved_paper_run.guidance_scale,
            injection_step_indices=resolved_paper_run.attention_injection_steps,
            standard_attack_profiles=(base_method_config.standard_attack_profiles if run_attacks else ()),
            diffusion_attacks_enabled=base_method_config.diffusion_attacks_enabled and run_attacks,
            output_dir=(
                f"outputs/image_only_dataset_runtime/{resolved_paper_run.run_name}/runs"
            ),
        )
        result = load_completed_semantic_watermark_runtime_result(run_config, root=root_path)
        generated_now = False
        if result is not None:
            resumed_prompt_count += 1
        elif max_new_prompts_per_session and new_prompt_count >= max_new_prompts_per_session:
            continue
        else:
            if shared_context is None:
                shared_context = load_semantic_watermark_runtime_context(base_method_config)
            result = write_semantic_watermark_runtime_outputs(
                run_config,
                root=root_path,
                runtime_context=shared_context,
            )
            new_prompt_count += 1
            generated_now = True
        result_payload = result.to_dict()
        validate_semantic_watermark_runtime_result_provenance(
            result_payload,
            expected_config=run_config,
        )
        runtime_results.append(result_payload)
        detection_records.extend(_read_jsonl(root_path / result.detection_record_path))
        scientific_update_records.extend(_read_jsonl(root_path / result.update_record_path))
        completed_prompt_ids.append(prompt_record.prompt_id)
        if generated_now and len(runtime_results) < len(prompt_records):
            write_resume_progress()

    if len(runtime_results) != len(prompt_records):
        return write_resume_progress()

    # 完整运行到达后删除续跑状态, 避免 Colab 入口把上一次中断记录误判为当前状态。
    progress_path.unlink(missing_ok=True)
    clear_progress_checkpoints(
        artifact_role="image_only_dataset_runtime",
        paper_run_name=resolved_paper_run.run_name,
    )

    calibration_negatives = tuple(
        record
        for record in detection_records
        if record.get("split") == "calibration"
        and record.get("sample_role") == "clean_negative"
        and not record.get("attack_id")
    )
    protocol = calibrate_complete_evidence_protocol(
        calibration_negatives,
        resolved_paper_run.target_fpr,
        base_method_config.rescue_margin_low,
    )
    formal_records = apply_frozen_evidence_protocol(detection_records, protocol)
    metric_rows = build_image_only_test_metric_rows(
        formal_records,
        resolved_paper_run.target_fpr,
    )
    detection_score_tables = build_detection_score_tables(formal_records, protocol.to_dict())

    runtime_results_path = output_dir / "runtime_results.jsonl"
    detection_records_path = output_dir / "image_only_detection_records.jsonl"
    quality_registry_path = output_dir / "watermark_quality_image_registry.jsonl"
    protocol_path = output_dir / "frozen_evidence_protocol.json"
    metrics_path = output_dir / "test_detection_metrics.csv"
    summary_path = output_dir / "dataset_runtime_summary.json"
    manifest_path = output_dir / "manifest.local.json"
    runtime_results_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in runtime_results),
        encoding="utf-8",
    )
    detection_records_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in formal_records),
        encoding="utf-8",
    )
    quality_registry_rows = []
    for result, prompt_record in zip(runtime_results, prompt_records):
        clean_path = root_path / str(result["clean_image_path"])
        watermarked_path = root_path / str(result["watermarked_image_path"])
        quality_registry_rows.append(
            {
                "run_id": result["run_id"],
                "prompt_id": prompt_record.prompt_id,
                "source_image_path": clean_path.relative_to(root_path).as_posix(),
                "source_image_digest": file_digest(clean_path),
                "attacked_image_path": watermarked_path.relative_to(root_path).as_posix(),
                "attacked_image_digest": file_digest(watermarked_path),
                "attack_name": "watermark_embedding",
                "image_pair_role": "clean_to_watermarked",
                "supports_paper_claim": False,
            }
        )
    quality_registry_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in quality_registry_rows),
        encoding="utf-8",
    )
    protocol_path.write_text(json.dumps(protocol.to_dict(), ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    _write_csv(metrics_path, metric_rows)
    detection_score_table_paths = write_detection_score_tables(
        output_dir,
        detection_score_tables,
    )
    split_counts = {
        split: sum(record.split == split for record in prompt_records)
        for split in ("dev", "calibration", "test")
    }
    clean_test_row = next(
        (
            row
            for row in metric_rows
            if row["attack_name"] == "none" and row["sample_role"] == "clean_negative"
        ),
        None,
    )
    wrong_key_test_row = next(
        (
            row
            for row in metric_rows
            if row["attack_name"] == "none" and row["sample_role"] == "wrong_key_negative"
        ),
        None,
    )
    paired_ssim_values = [
        float(result["metadata"]["paired_quality"]["ssim"])
        for result in runtime_results
        if result.get("metadata", {}).get("paired_quality", {}).get("ssim") is not None
    ]
    paired_psnr_values = [
        float(result["metadata"]["paired_quality"]["psnr"])
        for result in runtime_results
        if isinstance(result.get("metadata", {}).get("paired_quality", {}).get("psnr"), (int, float))
    ]
    expected_split_counts = build_group_split_counts(resolved_paper_run.prompt_count)
    expected_scientific_update_count = len(prompt_records) * len(resolved_paper_run.attention_injection_steps)
    scientific_operator_failure_count = sum(
        not _scientific_update_record_ready(record, base_method_config)
        for record in scientific_update_records
    )
    final_image_preservation_failure_count = sum(
        not _final_image_preservation_ready(result, base_method_config)
        for result in runtime_results
    )
    final_image_attention_observability_failure_count = sum(
        not _final_image_attention_observability_ready(
            result,
            base_method_config,
        )
        for result in runtime_results
    )
    scientific_operator_gate_ready = (
        len(scientific_update_records) == expected_scientific_update_count
        and scientific_operator_failure_count == 0
        and final_image_preservation_failure_count == 0
        and final_image_attention_observability_failure_count == 0
    )
    scientific_unit_provenance = aggregate_scientific_unit_provenance(
        (
            result["metadata"]["scientific_unit_provenance"]
            for result in runtime_results
        ),
        expected_reference_count=len(prompt_records),
    )
    protocol_decision = (
        "pass"
        if len(prompt_records) == resolved_paper_run.prompt_count
        and resolved_paper_run.sample_count == resolved_paper_run.prompt_count
        and split_counts == expected_split_counts
        and all(result.get("run_decision") == "pass" for result in runtime_results)
        and scientific_operator_gate_ready
        and scientific_unit_provenance["scientific_unit_provenance_ready"]
        and scientific_unit_provenance["scientific_unit_provenance_record_count"]
        == len(prompt_records)
        else "fail"
    )
    attacked_records = tuple(record for record in formal_records if record.get("attack_id"))
    standard_attack_ids = {
        attack.attack_id
        for attack in default_attack_configs()
        if attack.enabled
        and not attack.requires_gpu
        and attack.resource_profile in set(base_method_config.standard_attack_profiles)
    }
    diffusion_attack_ids = {
        attack.attack_id for attack in default_diffusion_attack_specs()
    } if base_method_config.diffusion_attacks_enabled else set()
    expected_attack_ids = standard_attack_ids | diffusion_attack_ids
    actual_attack_ids = {str(record.get("attack_id")) for record in attacked_records}
    attack_role_counts = {
        (attack_id, sample_role): sum(
            str(record.get("attack_id")) == attack_id and record.get("sample_role") == sample_role
            for record in attacked_records
        )
        for attack_id in expected_attack_ids
        for sample_role in ("clean_negative", "positive_source")
    }
    attack_record_coverage_ready = (
        bool(expected_attack_ids)
        and actual_attack_ids == expected_attack_ids
        and all(count == len(attack_prompt_ids) for count in attack_role_counts.values())
    )
    attacked_image_evidence_chain_ready = bool(attacked_records) and all(
        record.get("attacked_image_path")
        and record.get("attacked_image_digest")
        and (root_path / str(record["attacked_image_path"])).is_file()
        for record in attacked_records
    )
    clean_fixed_fpr_ready = bool(clean_test_row and clean_test_row["fixed_fpr_upper_bound_ready"])
    wrong_key_fixed_fpr_ready = bool(wrong_key_test_row and wrong_key_test_row["fixed_fpr_upper_bound_ready"])
    image_only_protocol_ready = all(
        str(record.get("metadata", {}).get("detector_input_access_mode", ""))
        == "image_key_public_model_only"
        and not bool(record.get("metadata", {}).get("generation_latent_trace_required", True))
        and bool(record.get("metadata", {}).get("blind_image_detector", False))
        for record in formal_records
    )
    full_method_claim_ready = (
        protocol_decision == "pass"
        and clean_fixed_fpr_ready
        and wrong_key_fixed_fpr_ready
        and image_only_protocol_ready
        and protocol.geometry_protocol_calibration_ready
        and scientific_operator_gate_ready
        and scientific_unit_provenance["scientific_unit_provenance_ready"]
    )
    required_real_gpu_attack_count = len(diffusion_attack_ids)
    measured_real_gpu_attack_count = len(
        {
            str(record.get("attack_id"))
            for record in attacked_records
            if record.get("resource_profile") == "full_extra"
        }
    )
    real_gpu_attack_validation_ready = (
        required_real_gpu_attack_count == 0
        or measured_real_gpu_attack_count >= required_real_gpu_attack_count
    )
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "paper_run_name": resolved_paper_run.run_name,
        "prompt_count": len(prompt_records),
        "split_counts": split_counts,
        "runtime_result_count": len(runtime_results),
        "resumed_prompt_count": resumed_prompt_count,
        "new_prompt_count": new_prompt_count,
        "attack_prompt_count": len(attack_prompt_ids),
        "detection_record_count": len(formal_records),
        "score_distribution_row_count": len(detection_score_tables["score_distribution_table"]),
        "roc_curve_point_count": len(detection_score_tables["roc_curve_points"]),
        "det_curve_point_count": len(detection_score_tables["det_curve_points"]),
        "detection_curve_data_ready": all(detection_score_tables.values()),
        "watermark_quality_pair_count": len(quality_registry_rows),
        "scientific_update_record_count": len(scientific_update_records),
        "expected_scientific_update_record_count": expected_scientific_update_count,
        "scientific_operator_failure_count": scientific_operator_failure_count,
        "final_image_preservation_failure_count": (
            final_image_preservation_failure_count
        ),
        "final_image_attention_observability_failure_count": (
            final_image_attention_observability_failure_count
        ),
        "final_image_attention_observability_ready": (
            final_image_attention_observability_failure_count == 0
        ),
        "scientific_operator_gate_ready": scientific_operator_gate_ready,
        **scientific_unit_provenance,
        "frozen_threshold_digest": protocol.threshold_digest,
        "geometry_protocol_calibration_ready": (
            protocol.geometry_protocol_calibration_ready
        ),
        "target_fpr": resolved_paper_run.target_fpr,
        "clean_test_fixed_fpr_upper_bound_ready": clean_fixed_fpr_ready,
        "wrong_key_test_fixed_fpr_upper_bound_ready": wrong_key_fixed_fpr_ready,
        "paired_ssim_mean": sum(paired_ssim_values) / len(paired_ssim_values) if paired_ssim_values else None,
        "paired_psnr_mean": sum(paired_psnr_values) / len(paired_psnr_values) if paired_psnr_values else None,
        "fixed_fpr_and_rescue_boundary_ready": (
            protocol.geometry_protocol_calibration_ready
        ),
        "fixed_fpr_boundary_ready": True,
        "rescue_boundary_ready": protocol.geometry_protocol_calibration_ready,
        "raw_content_claim_ready": True,
        "perceptual_metrics_ready": bool(paired_ssim_values),
        "real_attacked_image_count": len(attacked_records),
        "real_attacked_image_closed_loop_ready": attacked_image_evidence_chain_ready,
        "attacked_image_evidence_chain_ready": attacked_image_evidence_chain_ready,
        "formal_attack_detection_ready": attack_record_coverage_ready,
        "attack_record_coverage_ready": attack_record_coverage_ready,
        "required_attack_id_count": len(expected_attack_ids),
        "measured_attack_id_count": len(actual_attack_ids),
        "required_real_gpu_attack_count": required_real_gpu_attack_count,
        "measured_real_gpu_attack_count": measured_real_gpu_attack_count,
        "real_gpu_attack_validation_ready": real_gpu_attack_validation_ready,
        "full_method_claim_ready": full_method_claim_ready,
        "detector_input_access_mode": "image_key_public_model_only",
        "generation_latent_trace_required": False,
        "protocol_decision": protocol_decision,
        "supports_paper_claim": (
            full_method_claim_ready
            and attack_record_coverage_ready
            and attacked_image_evidence_chain_ready
            and real_gpu_attack_validation_ready
        ),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    formal_execution_run_lock = repository_environment.validate_formal_execution_lock_pair(
        formal_execution_run_lock,
        repository_environment.require_published_formal_execution_lock(root_path),
        formal_execution_run_lock["formal_execution_commit"],
    )
    manifest = build_artifact_manifest(
        artifact_id=f"{resolved_paper_run.run_name}_image_only_dataset_runtime_manifest",
        artifact_type="local_manifest",
        input_paths=(prompt_path.relative_to(root_path).as_posix(), "configs/prompt_source_registry.json"),
        output_paths=(
            runtime_results_path.relative_to(root_path).as_posix(),
            detection_records_path.relative_to(root_path).as_posix(),
            quality_registry_path.relative_to(root_path).as_posix(),
            protocol_path.relative_to(root_path).as_posix(),
            metrics_path.relative_to(root_path).as_posix(),
            detection_score_table_paths["score_distribution_table"].relative_to(root_path).as_posix(),
            detection_score_table_paths["roc_curve_points"].relative_to(root_path).as_posix(),
            detection_score_table_paths["det_curve_points"].relative_to(root_path).as_posix(),
            summary_path.relative_to(root_path).as_posix(),
            manifest_path.relative_to(root_path).as_posix(),
        ),
        config={
            "paper_run": resolved_paper_run.to_dict(),
            # manifest 现在保存完整配置, 因此必须复用运行时的密钥脱敏配置。
            # 该结构保留全部可复现实验参数, 但只记录 key material 的稳定摘要。
            "method_config": semantic_watermark_runtime_config_payload(
                base_method_config
            ),
            "method_key_digest": build_stable_digest({"key_material": base_method_config.key_material}),
        },
        code_version=formal_execution_run_lock["formal_execution_commit"],
        rebuild_command="调用 experiments.runners.image_only_dataset_runtime.run_image_only_dataset_runtime",
        metadata={
            "protocol_decision": summary["protocol_decision"],
            "detector_input_access_mode": "image_key_public_model_only",
            "full_method_claim_ready": summary["full_method_claim_ready"],
            "geometry_protocol_calibration_ready": summary[
                "geometry_protocol_calibration_ready"
            ],
            "attack_record_coverage_ready": summary["attack_record_coverage_ready"],
            "attacked_image_evidence_chain_ready": summary["attacked_image_evidence_chain_ready"],
            "scientific_operator_gate_ready": summary["scientific_operator_gate_ready"],
            "final_image_attention_observability_failure_count": summary[
                "final_image_attention_observability_failure_count"
            ],
            "final_image_attention_observability_ready": summary[
                "final_image_attention_observability_ready"
            ],
            "scientific_unit_provenance_ready": summary[
                "scientific_unit_provenance_ready"
            ],
            "scientific_unit_provenance_records_digest": summary[
                "scientific_unit_provenance_records_digest"
            ],
            "supports_paper_claim": summary["supports_paper_claim"],
        },
    ).to_dict()
    manifest["formal_execution_run_lock"] = formal_execution_run_lock
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return summary


def package_image_only_dataset_runtime(
    paper_run_name: str,
    root: str | Path = ".",
) -> Path:
    """把真实运行 records、图像、阈值和 manifest 打包为受治理输入包。"""

    root_path = Path(root).resolve()
    formal_execution_package_lock = (
        repository_environment.require_published_formal_execution_lock(root_path)
    )
    resolved_paper_run_name = normalize_paper_run_name(paper_run_name)
    paper_run = build_paper_run_config(root_path)
    if paper_run.run_name != resolved_paper_run_name:
        raise ValueError("仅图像运行打包层级必须与当前论文配置一致")
    source_dir = (
        root_path / "outputs" / "image_only_dataset_runtime" / resolved_paper_run_name
    )
    required_paths = tuple(
        source_dir / filename
        for filename in (
            "runtime_results.jsonl",
            "image_only_detection_records.jsonl",
            "watermark_quality_image_registry.jsonl",
            "frozen_evidence_protocol.json",
            "test_detection_metrics.csv",
            "score_distribution_table.csv",
            "roc_curve_points.csv",
            "det_curve_points.csv",
            "dataset_runtime_summary.json",
            "manifest.local.json",
        )
    )
    if any(not path.is_file() for path in required_paths):
        raise FileNotFoundError("仅图像数据集运行输出不完整, 不得打包")
    summary = json.loads((source_dir / "dataset_runtime_summary.json").read_text(encoding="utf-8-sig"))
    manifest = json.loads((source_dir / "manifest.local.json").read_text(encoding="utf-8-sig"))
    packaged_runtime_results = _read_jsonl(source_dir / "runtime_results.jsonl")
    for result in packaged_runtime_results:
        validate_semantic_watermark_runtime_result_provenance(result)
    packaged_prompt_records = apply_split_assignments(
        build_prompt_records(
            paper_run.prompt_set,
            read_prompt_file(root_path / paper_run.prompt_file),
        )
    )[: paper_run.sample_count]
    packaged_unit_configs = [
        result["metadata"]["scientific_unit_config"]
        for result in packaged_runtime_results
    ]
    packaged_unit_config_contract_ready = (
        len(packaged_unit_configs) == len(packaged_prompt_records)
        and len(
            {
                int(config["seed"]) - prompt.prompt_index
                for config, prompt in zip(
                    packaged_unit_configs,
                    packaged_prompt_records,
                )
            }
        )
        == 1
        and all(
            config.get("prompt_id") == prompt.prompt_id
            and config.get("prompt") == prompt.prompt_text
            and config.get("split") == prompt.split
            and int(config.get("inference_steps", -1))
            == paper_run.inference_steps
            and math.isclose(
                float(config.get("guidance_scale", -1.0)),
                paper_run.guidance_scale,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            and tuple(config.get("injection_step_indices", ()))
            == paper_run.attention_injection_steps
            and config.get("output_dir")
            == (
                f"outputs/image_only_dataset_runtime/"
                f"{resolved_paper_run_name}/runs"
            )
            and all(
                config.get(field_name) is True
                for field_name in (
                    "semantic_routing_enabled",
                    "null_space_enabled",
                    "lf_enabled",
                    "tail_robust_enabled",
                    "tail_truncation_enabled",
                    "attention_geometry_enabled",
                    "image_alignment_enabled",
                )
            )
            for config, prompt in zip(
                packaged_unit_configs,
                packaged_prompt_records,
            )
        )
    )
    packaged_scientific_unit_provenance = aggregate_scientific_unit_provenance(
        (
            result["metadata"]["scientific_unit_provenance"]
            for result in packaged_runtime_results
        ),
        expected_reference_count=paper_run.prompt_count,
    )
    scientific_unit_provenance_summary_bound = all(
        summary.get(field_name)
        == packaged_scientific_unit_provenance[field_name]
        for field_name in SCIENTIFIC_UNIT_PROVENANCE_AGGREGATE_FIELDS
    )
    formal_execution_run_lock = repository_environment.validate_formal_execution_lock_pair(
        manifest.get("formal_execution_run_lock"),
        formal_execution_package_lock,
        manifest.get("code_version"),
    )
    if not all(
        (
            summary.get("paper_run_name") == resolved_paper_run_name,
            math.isclose(
                float(summary.get("target_fpr", -1.0)),
                paper_run.target_fpr,
                rel_tol=0.0,
                abs_tol=1e-12,
            ),
            bool(summary.get("generated_at")),
            summary.get("protocol_decision") == "pass",
            summary.get("full_method_claim_ready") is True,
            summary.get("geometry_protocol_calibration_ready") is True,
            summary.get("scientific_unit_provenance_ready") is True,
            summary.get("scientific_unit_provenance_record_count")
            == paper_run.prompt_count,
            bool(summary.get("scientific_unit_provenance_records_digest")),
            scientific_unit_provenance_summary_bound,
            packaged_unit_config_contract_ready,
            summary.get("detection_curve_data_ready") is True,
            summary.get("supports_paper_claim") is True,
            manifest.get("artifact_id")
            == f"{resolved_paper_run_name}_image_only_dataset_runtime_manifest",
            manifest.get("metadata", {}).get(
                "geometry_protocol_calibration_ready"
            )
            is True,
        )
    ):
        raise RuntimeError("仅图像数据集运行身份或 ready 门禁未通过")
    manifest["formal_execution_package_lock"] = formal_execution_package_lock
    (source_dir / "manifest.local.json").write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    validate_scientific_execution_binding(
        source_dir / "scientific_execution_binding.json",
        expected_artifact_role="image_only_dataset_runtime",
        expected_paper_run_name=resolved_paper_run_name,
        repository_root=root_path,
    )
    code_version = formal_execution_package_lock["formal_execution_commit"]
    archive_path = source_dir / (
        f"image_only_dataset_runtime_package_{utc_archive_token()}_{code_version[:7]}.zip"
    )
    package_input_manifest_path = source_dir / PACKAGE_INPUT_MANIFEST_FILE_NAME
    package_input_manifest_path.unlink(missing_ok=True)
    entries = collect_exact_package_entries(
        repository_root=root_path,
        source_dir=source_dir,
        artifact_manifest=manifest,
        scientific_binding_path=source_dir / "scientific_execution_binding.json",
    )
    if not set(required_paths).issubset(entries):
        raise RuntimeError("artifact manifest 未精确声明全部仅图像运行必要产物")
    write_exact_package_input_manifest(
        package_input_manifest_path,
        repository_root=root_path,
        package_family="image_only_dataset_runtime",
        paper_run_name=resolved_paper_run_name,
        target_fpr=paper_run.target_fpr,
        entries=entries,
        formal_execution_run_lock=formal_execution_run_lock,
        formal_execution_package_lock=formal_execution_package_lock,
    )
    entries = (*entries, package_input_manifest_path)
    with ZipFile(archive_path, "w", compression=ZIP_STORED, allowZip64=True) as archive:
        for path in entries:
            archive.write(path, path.relative_to(root_path).as_posix())
    try:
        validate_exact_package_archive(
            archive_path,
            repository_root=root_path,
            package_input_manifest_path=package_input_manifest_path,
        )
        final_package_lock = (
            repository_environment.require_published_formal_execution_lock(root_path)
        )
        repository_environment.validate_formal_execution_lock_pair(
            formal_execution_package_lock,
            final_package_lock,
            code_version,
        )
    except Exception:
        archive_path.unlink(missing_ok=True)
        raise
    return archive_path
