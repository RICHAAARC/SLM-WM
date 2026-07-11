"""在 70/700/7000 Prompt 协议上运行真实方法并冻结完整检测判定。"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass, replace
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable
from zipfile import ZIP_STORED, ZipFile

from experiments.protocol.calibration import binomial_rate_upper_confidence_bound
from experiments.protocol.paper_run_config import PaperRunConfig, build_paper_run_config
from experiments.protocol.prompts import build_prompt_records, read_prompt_file
from experiments.protocol.splits import apply_split_assignments, build_group_split_counts
from experiments.protocol.attacks import default_attack_configs
from experiments.runtime.diffusion.regeneration_attacks import default_diffusion_attack_specs
from experiments.runners.semantic_watermark_runtime import (
    SemanticWatermarkRuntimeConfig,
    load_completed_semantic_watermark_runtime_result,
    load_semantic_watermark_runtime_context,
    write_semantic_watermark_runtime_outputs,
)
from experiments.runtime.repository_environment import file_digest, resolve_code_version
from experiments.runtime.archive_naming import utc_archive_token
from experiments.artifacts.artifact_manifest import build_artifact_manifest
from main.core.digest import build_stable_digest


@dataclass(frozen=True)
class FrozenEvidenceProtocol:
    """保存 calibration split 冻结的完整 evidence 判定参数。"""

    content_threshold: float
    rescue_margin_low: float
    geometry_score_threshold: float
    geometry_calibration_negative_count: int
    geometry_calibration_exceedance_count: int
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
) -> tuple[bool, bool, bool, str]:
    """用冻结阈值重算内容主判和同阈值几何救回。"""

    raw_score = float(record["content_score"])
    raw_margin = raw_score - threshold
    positive_by_content = raw_margin >= 0.0
    aligned_score = record.get("aligned_content_score")
    alignment = record.get("alignment")
    if isinstance(alignment, dict):
        alignment_reliable = bool(alignment.get("geometry_reliable", False))
    else:
        alignment_reliable = bool(record.get("geometry_reliable", False))
    geometry_score = record.get("attention_geometry_score")
    geometry_reliable = alignment_reliable and (
        bool(record.get("geometry_reliable", False))
        if geometry_score is None
        else float(geometry_score) >= geometry_score_threshold
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
    reliable_geometry_scores = tuple(
        float(record["attention_geometry_score"])
        for record in records
        if bool(record.get("geometry_reliable", False))
        and isinstance(record.get("attention_geometry_score"), (int, float))
        and math.isfinite(float(record["attention_geometry_score"]))
    )
    geometry_score_threshold = 0.0
    geometry_exceedance_count = 0
    if reliable_geometry_scores:
        geometry_candidates = sorted(
            {math.nextafter(value, math.inf) for value in reliable_geometry_scores}
        )
        geometry_score_threshold = geometry_candidates[-1]
        for candidate in geometry_candidates:
            exceedance_count = sum(value >= candidate for value in reliable_geometry_scores)
            if exceedance_count <= allowed_false_positives:
                geometry_score_threshold = candidate
                geometry_exceedance_count = exceedance_count
                break
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
            _decision(record, threshold, rescue_margin_low, geometry_score_threshold)[2]
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
        "geometry_calibration_negative_count": len(reliable_geometry_scores),
        "geometry_calibration_exceedance_count": geometry_exceedance_count,
        "calibration_negative_count": len(records),
        "calibration_false_positive_count": selected_false_positives,
        "target_fpr": target_fpr,
        "decision_scope": "content_or_same_threshold_aligned_content_rescue",
    }
    return FrozenEvidenceProtocol(
        content_threshold=selected_threshold,
        rescue_margin_low=rescue_margin_low,
        geometry_score_threshold=geometry_score_threshold,
        geometry_calibration_negative_count=len(reliable_geometry_scores),
        geometry_calibration_exceedance_count=geometry_exceedance_count,
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
        )
        raw_margin = float(record["content_score"]) - protocol.content_threshold
        aligned_score = record.get("aligned_content_score")
        resolved.append(
            {
                **record,
                "frozen_content_threshold": protocol.content_threshold,
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


def _aggregate_test_metrics(
    records: Iterable[dict[str, Any]],
    target_fpr: float,
) -> tuple[dict[str, Any], ...]:
    """按攻击和 sample role 聚合 test split 的真实检测率。"""

    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record.get("split") != "test":
            continue
        attack_family = str(record.get("attack_family", "clean"))
        attack_name = str(record.get("attack_name", "none"))
        resource_profile = str(record.get("resource_profile", "clean"))
        grouped[(attack_family, attack_name, resource_profile, str(record["sample_role"]))].append(record)
    rows = []
    for (attack_family, attack_name, resource_profile, sample_role), group_records in sorted(grouped.items()):
        positive_count = sum(bool(record["formal_evidence_positive"]) for record in group_records)
        rate = positive_count / len(group_records)
        upper = binomial_rate_upper_confidence_bound(positive_count, len(group_records), 0.95)
        quality_ssim_values = [
            float(record["source_to_evaluated_ssim"])
            for record in group_records
            if isinstance(record.get("source_to_evaluated_ssim"), (int, float))
        ]
        quality_psnr_values = [
            float(record["source_to_evaluated_psnr"])
            for record in group_records
            if isinstance(record.get("source_to_evaluated_psnr"), (int, float))
        ]
        rows.append(
            {
                "attack_family": attack_family,
                "attack_name": attack_name,
                "resource_profile": resource_profile,
                "sample_role": sample_role,
                "record_count": len(group_records),
                "positive_count": positive_count,
                "positive_rate": rate,
                "content_score_mean": sum(float(record["content_score"]) for record in group_records) / len(group_records),
                "source_to_evaluated_ssim_mean": (
                    sum(quality_ssim_values) / len(quality_ssim_values) if quality_ssim_values else None
                ),
                "source_to_evaluated_psnr_mean": (
                    sum(quality_psnr_values) / len(quality_psnr_values) if quality_psnr_values else None
                ),
                "positive_rate_upper_95": upper,
                "target_fpr": target_fpr,
                "fixed_fpr_upper_bound_ready": (
                    sample_role in {"clean_negative", "wrong_key_negative"} and upper <= target_fpr
                ),
                "metric_status": "measured_image_only_detection",
                "supports_paper_claim": False,
            }
        )
    return tuple(rows)


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
        "torch_func_linearize_exact_jvp",
        "torch_autograd_exact_jvp",
        "torch_autograd_exact_jvp_compatibility",
    }
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
        if int(metadata.get("preferred_direction_count", 0)) < 1:
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
    branch_risk_records = record.get("branch_risk_records")
    return (
        bool(record.get("branch_risk_bundle_digest"))
        and isinstance(branch_risk_records, dict)
        and set(branch_risk_records) == {"lf_content", "tail_robust", "attention_geometry"}
        and all(int(value.get("eligible_position_count", 0)) > 0 for value in branch_risk_records.values())
    )


def run_image_only_dataset_runtime(
    base_method_config: SemanticWatermarkRuntimeConfig,
    root: str | Path = ".",
    paper_run: PaperRunConfig | None = None,
    max_new_prompts_per_session: int = 0,
) -> dict[str, Any]:
    """运行当前论文规模的全部 Prompt 并生成可校准记录。"""

    root_path = Path(root).resolve()
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
        runtime_results.append(result.to_dict())
        detection_records.extend(_read_jsonl(root_path / result.detection_record_path))
        scientific_update_records.extend(_read_jsonl(root_path / result.update_record_path))
        completed_prompt_ids.append(prompt_record.prompt_id)

    if len(runtime_results) != len(prompt_records):
        progress = {
            "paper_run_name": resolved_paper_run.run_name,
            "prompt_count": len(prompt_records),
            "completed_prompt_count": len(runtime_results),
            "remaining_prompt_count": len(prompt_records) - len(runtime_results),
            "resumed_prompt_count": resumed_prompt_count,
            "new_prompt_count": new_prompt_count,
            "max_new_prompts_per_session": max_new_prompts_per_session,
            "completed_prompt_digest": build_stable_digest(sorted(completed_prompt_ids)),
            "protocol_decision": "resume_required",
            "supports_paper_claim": False,
        }
        progress_path.write_text(
            json.dumps(progress, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        return progress

    # 完整运行到达后删除续跑状态, 避免 Colab 入口把上一次中断记录误判为当前状态。
    progress_path.unlink(missing_ok=True)

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
    metric_rows = _aggregate_test_metrics(formal_records, resolved_paper_run.target_fpr)

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
    scientific_operator_gate_ready = (
        len(scientific_update_records) == expected_scientific_update_count
        and scientific_operator_failure_count == 0
    )
    protocol_decision = (
        "pass"
        if len(prompt_records) == resolved_paper_run.prompt_count
        and resolved_paper_run.sample_count == resolved_paper_run.prompt_count
        and split_counts == expected_split_counts
        and all(result.get("run_decision") == "pass" for result in runtime_results)
        and scientific_operator_gate_ready
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
        and scientific_operator_gate_ready
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
        "paper_run_name": resolved_paper_run.run_name,
        "prompt_count": len(prompt_records),
        "split_counts": split_counts,
        "runtime_result_count": len(runtime_results),
        "resumed_prompt_count": resumed_prompt_count,
        "new_prompt_count": new_prompt_count,
        "attack_prompt_count": len(attack_prompt_ids),
        "detection_record_count": len(formal_records),
        "watermark_quality_pair_count": len(quality_registry_rows),
        "scientific_update_record_count": len(scientific_update_records),
        "expected_scientific_update_record_count": expected_scientific_update_count,
        "scientific_operator_failure_count": scientific_operator_failure_count,
        "scientific_operator_gate_ready": scientific_operator_gate_ready,
        "frozen_threshold_digest": protocol.threshold_digest,
        "target_fpr": resolved_paper_run.target_fpr,
        "clean_test_fixed_fpr_upper_bound_ready": clean_fixed_fpr_ready,
        "wrong_key_test_fixed_fpr_upper_bound_ready": wrong_key_fixed_fpr_ready,
        "paired_ssim_mean": sum(paired_ssim_values) / len(paired_ssim_values) if paired_ssim_values else None,
        "paired_psnr_mean": sum(paired_psnr_values) / len(paired_psnr_values) if paired_psnr_values else None,
        "fixed_fpr_and_rescue_boundary_ready": True,
        "fixed_fpr_boundary_ready": True,
        "rescue_boundary_ready": True,
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
            summary_path.relative_to(root_path).as_posix(),
            manifest_path.relative_to(root_path).as_posix(),
        ),
        config={
            "paper_run": resolved_paper_run.to_dict(),
            "method_config": asdict(base_method_config),
            "method_key_digest": build_stable_digest({"key_material": base_method_config.key_material}),
        },
        code_version=resolve_code_version(root_path),
        rebuild_command="调用 experiments.runners.image_only_dataset_runtime.run_image_only_dataset_runtime",
        metadata={
            "protocol_decision": summary["protocol_decision"],
            "detector_input_access_mode": "image_key_public_model_only",
            "full_method_claim_ready": summary["full_method_claim_ready"],
            "attack_record_coverage_ready": summary["attack_record_coverage_ready"],
            "attacked_image_evidence_chain_ready": summary["attacked_image_evidence_chain_ready"],
            "scientific_operator_gate_ready": summary["scientific_operator_gate_ready"],
            "supports_paper_claim": summary["supports_paper_claim"],
        },
    ).to_dict()
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return summary


def package_image_only_dataset_runtime(
    paper_run_name: str,
    root: str | Path = ".",
) -> Path:
    """把真实运行 records、图像、阈值和 manifest 打包为受治理输入包。"""

    root_path = Path(root).resolve()
    source_dir = root_path / "outputs" / "image_only_dataset_runtime" / paper_run_name
    if not source_dir.is_dir():
        raise FileNotFoundError("缺少仅图像数据集运行输出目录")
    code_version = resolve_code_version(root_path).replace("-dirty", "")
    archive_path = source_dir / f"image_only_dataset_runtime_package_{utc_archive_token()}_{code_version}.zip"
    entries = tuple(
        path
        for path in sorted(source_dir.rglob("*"))
        if path.is_file() and path.suffix.lower() != ".zip"
    )
    with ZipFile(archive_path, "w", compression=ZIP_STORED, allowZip64=True) as archive:
        for path in entries:
            archive.write(path, path.relative_to(root_path).as_posix())
    return archive_path
