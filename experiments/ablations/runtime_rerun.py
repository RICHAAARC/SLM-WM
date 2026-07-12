"""通过完整重新生成、独立校准和图像盲检执行正式机制消融。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Iterable
from zipfile import ZIP_STORED, ZipFile

from experiments.artifacts.artifact_manifest import build_artifact_manifest
from experiments.artifacts.manifest_schema import manifest_config_digest_ready
from experiments.ablations.necessity_statistics import (
    ABLATION_NECESSITY_FIELDNAMES,
    build_ablation_necessity_statistics,
)
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
from experiments.protocol.paper_run_config import (
    build_paper_run_config,
    normalize_paper_run_name,
)
from experiments.protocol.attacks import attack_config_digest, default_attack_configs
from experiments.protocol.prompts import build_prompt_records, read_prompt_file
from experiments.protocol.splits import apply_split_assignments, group_prompt_ids_by_split
from experiments.runners.image_only_dataset_runtime import (
    apply_frozen_evidence_protocol,
    calibrate_complete_evidence_protocol,
)
from experiments.runners.semantic_watermark_runtime import (
    SemanticWatermarkRuntimeConfig,
    load_completed_semantic_watermark_runtime_result,
    load_semantic_watermark_runtime_context,
    validate_semantic_watermark_runtime_result_provenance,
    write_semantic_watermark_runtime_outputs,
)
from experiments.runtime.archive_naming import utc_archive_token
from main.core.digest import build_stable_digest
from main.methods.geometry import ATTENTION_RELATION_COMPONENT_WEIGHTS


PACKAGE_INPUT_MANIFEST_FILE_NAME = "mechanism_ablation_package_input_manifest.json"


def _read_jsonl(path: Path) -> tuple[dict[str, Any], ...]:
    """读取真实运行检测记录。"""

    return tuple(
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def _mean(values: Iterable[float]) -> float:
    """计算非空数值集合均值。"""

    resolved = tuple(values)
    return sum(resolved) / len(resolved) if resolved else 0.0


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """用所有结果行的稳定列集合写出 CSV。"""

    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({name for row in rows for name in row})
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


@dataclass(frozen=True)
class RuntimeRerunAblationSpec:
    """描述需要重新执行生成、嵌入、攻击和检测的单个机制配置。"""

    ablation_id: str
    semantic_routing_enabled: bool = True
    branch_risk_mode: str = "branch_specific"
    null_space_enabled: bool = True
    lf_enabled: bool = True
    tail_robust_enabled: bool = True
    tail_truncation_enabled: bool = True
    attention_geometry_enabled: bool = True
    image_alignment_enabled: bool = True
    attention_relation_component_weights: tuple[
        float,
        ...,
    ] = ATTENTION_RELATION_COMPONENT_WEIGHTS

    def to_dict(self) -> dict[str, Any]:
        """返回跨 JSON 边界保持同一类型的正式机制配置."""

        payload = asdict(self)
        payload["attention_relation_component_weights"] = list(
            self.attention_relation_component_weights
        )
        return payload

    def apply(
        self,
        config: SemanticWatermarkRuntimeConfig,
        output_root: str,
    ) -> SemanticWatermarkRuntimeConfig:
        """把唯一机制差异写入完整方法运行配置。"""

        return replace(
            config,
            semantic_routing_enabled=self.semantic_routing_enabled,
            branch_risk_mode=self.branch_risk_mode,
            null_space_enabled=self.null_space_enabled,
            lf_enabled=self.lf_enabled,
            tail_robust_enabled=self.tail_robust_enabled,
            tail_truncation_enabled=self.tail_truncation_enabled,
            attention_geometry_enabled=self.attention_geometry_enabled,
            image_alignment_enabled=self.image_alignment_enabled,
            attention_relation_component_weights=(
                self.attention_relation_component_weights
            ),
            output_dir=f"{output_root}/runs/{self.ablation_id}",
        )


FORMAL_RUNTIME_RERUN_ABLATION_SPECS = (
    RuntimeRerunAblationSpec("complete_method"),
    RuntimeRerunAblationSpec(
        "shared_global_risk_routing",
        branch_risk_mode="shared_global",
    ),
    RuntimeRerunAblationSpec(
        "without_branch_risk_routing",
        semantic_routing_enabled=False,
    ),
    RuntimeRerunAblationSpec(
        "without_jacobian_null_space",
        null_space_enabled=False,
    ),
    RuntimeRerunAblationSpec(
        "lf_content_only",
        tail_robust_enabled=False,
        attention_geometry_enabled=False,
        image_alignment_enabled=False,
    ),
    RuntimeRerunAblationSpec(
        "tail_robust_only",
        lf_enabled=False,
        attention_geometry_enabled=False,
        image_alignment_enabled=False,
    ),
    RuntimeRerunAblationSpec(
        "without_lf_content_carrier",
        lf_enabled=False,
    ),
    RuntimeRerunAblationSpec(
        "without_tail_robust_carrier",
        tail_robust_enabled=False,
    ),
    RuntimeRerunAblationSpec(
        "without_tail_amplitude_truncation",
        tail_truncation_enabled=False,
    ),
    RuntimeRerunAblationSpec(
        "without_centered_qk_logit",
        attention_relation_component_weights=(
            0.0,
            1.0 / 3.0,
            1.0 / 3.0,
            1.0 / 3.0,
        ),
    ),
    RuntimeRerunAblationSpec(
        "without_differentiable_row_rank",
        attention_relation_component_weights=(
            1.0 / 3.0,
            0.0,
            1.0 / 3.0,
            1.0 / 3.0,
        ),
    ),
    RuntimeRerunAblationSpec(
        "without_attention_probability",
        attention_relation_component_weights=(
            1.0 / 3.0,
            1.0 / 3.0,
            0.0,
            1.0 / 3.0,
        ),
    ),
    RuntimeRerunAblationSpec(
        "without_distance_modulated_probability",
        attention_relation_component_weights=(
            1.0 / 3.0,
            1.0 / 3.0,
            1.0 / 3.0,
            0.0,
        ),
    ),
    RuntimeRerunAblationSpec(
        "without_attention_geometry",
        attention_geometry_enabled=False,
    ),
    RuntimeRerunAblationSpec(
        "without_image_alignment",
        image_alignment_enabled=False,
    ),
)
FORMAL_RUNTIME_RERUN_ABLATION_IDS = tuple(
    spec.ablation_id for spec in FORMAL_RUNTIME_RERUN_ABLATION_SPECS
)
FORMAL_RUNTIME_RERUN_ABLATION_SPEC_DIGEST = build_stable_digest(
    [spec.to_dict() for spec in FORMAL_RUNTIME_RERUN_ABLATION_SPECS]
)


def default_runtime_rerun_ablation_specs() -> tuple[RuntimeRerunAblationSpec, ...]:
    """返回论文协议唯一允许的15项正式重运行消融规范."""

    return FORMAL_RUNTIME_RERUN_ABLATION_SPECS


def runtime_rerun_ablation_contract(
    specs: Iterable[RuntimeRerunAblationSpec],
) -> dict[str, Any]:
    """把实际消融规范与唯一正式规范进行精确比较。

    该函数属于可复用的协议校验写法: 运行器、打包器和最外层论文门禁
    共用同一组标识和摘要, 避免各层分别维护容易漂移的消融清单。
    """

    actual_specs = tuple(specs)
    actual_ids = tuple(spec.ablation_id for spec in actual_specs)
    actual_digest = build_stable_digest(
        [spec.to_dict() for spec in actual_specs]
    )
    return {
        "expected_ablation_ids": list(FORMAL_RUNTIME_RERUN_ABLATION_IDS),
        "actual_ablation_ids": list(actual_ids),
        "ablation_spec_digest": actual_digest,
        "ablation_exact_set_ready": (
            actual_ids == FORMAL_RUNTIME_RERUN_ABLATION_IDS
            and actual_digest == FORMAL_RUNTIME_RERUN_ABLATION_SPEC_DIGEST
        ),
    }


def _canonical_prompt_contract(
    root_path: Path,
    paper_run: Any,
    base_configs: tuple[SemanticWatermarkRuntimeConfig, ...],
) -> dict[str, Any]:
    """核验消融配置精确覆盖当前论文运行的规范 Prompt 与 split.

    此处设计的主要考虑在于: 仅比较 Prompt 数量无法发现重复 Prompt、跨层级
    Prompt 或 split 漂移. 该校验把正式消融直接绑定到受治理 Prompt 文件, 并把
    三个 split 的身份摘要写入最终摘要, 供最外层论文门禁复验.
    """

    canonical_records = apply_split_assignments(
        build_prompt_records(
            paper_run.prompt_set,
            read_prompt_file(root_path / paper_run.prompt_file),
        )
    )
    expected_by_id = {
        record.prompt_id: (record.prompt_text, record.split)
        for record in canonical_records
    }
    actual_ids = [config.prompt_id for config in base_configs]
    actual_by_id = {
        config.prompt_id: (config.prompt, config.split)
        for config in base_configs
    }
    exact_set_ready = bool(
        len(canonical_records) == paper_run.prompt_count
        and len(base_configs) == paper_run.prompt_count
        and len(set(actual_ids)) == len(actual_ids)
        and actual_by_id == expected_by_id
    )
    if not exact_set_ready:
        raise ValueError("正式消融必须精确覆盖当前论文运行的规范 Prompt 与 split")
    split_prompt_ids = group_prompt_ids_by_split(canonical_records)
    return {
        "prompt_id_digest": build_stable_digest(sorted(expected_by_id)),
        "calibration_prompt_id_digest": build_stable_digest(
            sorted(split_prompt_ids["calibration"])
        ),
        "test_prompt_id_digest": build_stable_digest(
            sorted(split_prompt_ids["test"])
        ),
        "prompt_protocol_exact_set_ready": True,
    }


def _formal_attack_coverage_ready(
    detections: tuple[dict[str, Any], ...],
    *,
    split: str,
) -> bool:
    """核验单个消融运行只在 test split 精确执行完整正式攻击集."""

    attacked_records = tuple(record for record in detections if record.get("attack_id"))
    if split != "test":
        return not attacked_records
    formal_configs = tuple(
        config
        for config in default_attack_configs()
        if config.enabled and config.resource_profile in {"full_main", "full_extra"}
    )
    expected_by_key = {
        (config.attack_id, sample_role): config
        for config in formal_configs
        for sample_role in ("clean_negative", "positive_source")
    }
    actual_by_key: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for record in attacked_records:
        key = (str(record.get("attack_id", "")), str(record.get("sample_role", "")))
        actual_by_key.setdefault(key, []).append(record)
    if set(actual_by_key) != set(expected_by_key) or any(
        len(rows) != 1 for rows in actual_by_key.values()
    ):
        return False
    for key, config in expected_by_key.items():
        record = actual_by_key[key][0]
        if not all(
            (
                record.get("attack_family") == config.attack_family,
                record.get("attack_name") == config.attack_name,
                record.get("resource_profile") == config.resource_profile,
                record.get("attack_config_digest") == attack_config_digest(config),
                record.get("attack_parameters") == config.attack_parameters,
                record.get("attack_performed") is True,
            )
        ):
            return False
    return True


def _run_entry(
    prompt_index: int,
    base_config: SemanticWatermarkRuntimeConfig,
    spec: RuntimeRerunAblationSpec,
    result: Any,
    detections: tuple[dict[str, Any], ...],
) -> dict[str, Any]:
    """构造一个 Prompt 与一个消融机制的完整运行记录。"""

    return {
        "prompt_index": prompt_index,
        "prompt_id": base_config.prompt_id,
        "prompt_digest": build_stable_digest(
            {"prompt_text": base_config.prompt}
        ),
        "split": base_config.split,
        "ablation_id": spec.ablation_id,
        "runtime_config": spec.to_dict(),
        "runtime_result": result.to_dict(),
        "detections": detections,
    }


def _formal_record(
    entry: dict[str, Any],
    detections: tuple[dict[str, Any], ...],
    threshold_digest: str,
    threshold: float,
) -> dict[str, Any]:
    """从独立冻结阈值后的检测结果构造逐 Prompt 论文记录。"""

    clean_negative = next(
        record
        for record in detections
        if record.get("sample_role") == "clean_negative" and not record.get("attack_id")
    )
    positive_source = next(
        record
        for record in detections
        if record.get("sample_role") == "positive_source" and not record.get("attack_id")
    )
    wrong_key_negative = next(
        record
        for record in detections
        if record.get("sample_role") == "wrong_key_negative" and not record.get("attack_id")
    )
    attacked_positive = tuple(
        record
        for record in detections
        if record.get("sample_role") == "positive_source" and record.get("attack_id")
    )
    attacked_negative = tuple(
        record
        for record in detections
        if record.get("sample_role") == "clean_negative" and record.get("attack_id")
    )
    runtime_result = entry["runtime_result"]
    return {
        "prompt_index": entry["prompt_index"],
        "prompt_id": entry["prompt_id"],
        "prompt_digest": entry["prompt_digest"],
        "split": entry["split"],
        "ablation_id": entry["ablation_id"],
        "runtime_config": entry["runtime_config"],
        "runtime_result": runtime_result,
        "generation_rerun": True,
        "attack_and_detection_rerun": any(record.get("attack_id") for record in detections),
        "threshold_calibration_scope": "per_ablation_calibration_split",
        "frozen_content_threshold": threshold,
        "frozen_threshold_digest": threshold_digest,
        "clean_negative_positive": bool(clean_negative["formal_evidence_positive"]),
        "positive_source_positive": bool(positive_source["formal_evidence_positive"]),
        "wrong_key_negative_positive": bool(wrong_key_negative["formal_evidence_positive"]),
        "clean_negative_content_score": float(clean_negative["content_score"]),
        "positive_source_content_score": float(positive_source["content_score"]),
        "attacked_positive_count": len(attacked_positive),
        "attacked_positive_rate": _mean(
            float(bool(record["formal_evidence_positive"])) for record in attacked_positive
        ),
        "attacked_negative_count": len(attacked_negative),
        "attacked_negative_rate": _mean(
            float(bool(record["formal_evidence_positive"])) for record in attacked_negative
        ),
        "paired_ssim": float(runtime_result["metadata"]["paired_quality"]["ssim"]),
    }


def run_runtime_rerun_ablations(
    base_configs: Iterable[SemanticWatermarkRuntimeConfig],
    target_fpr: float,
    paper_run_name: str,
    root: str | Path = ".",
    specs: tuple[RuntimeRerunAblationSpec, ...] | None = None,
    max_new_runs_per_session: int = 0,
) -> dict[str, Any]:
    """在完整 Prompt 集上重运行每个机制配置并分别冻结检测阈值。

    每个消融配置使用自己的 calibration split 负样本冻结内容阈值和注意力几何
    阈值, 再仅在 test split 汇总结果。不同配置之间不会共享完整方法阈值, 因而
    测量的是机制移除后的检测能力, 不是阈值失配造成的差异。
    """

    root_path = Path(root).resolve()
    formal_execution_run_lock = (
        repository_environment.require_published_formal_execution_lock(root_path)
    )
    resolved_paper_run_name = normalize_paper_run_name(paper_run_name)
    paper_run = build_paper_run_config(root_path)
    if paper_run.run_name != resolved_paper_run_name or abs(
        float(target_fpr) - paper_run.target_fpr
    ) > 1e-12:
        raise ValueError("正式消融身份必须与当前论文运行配置一致")
    output_dir = f"outputs/formal_mechanism_ablation/{resolved_paper_run_name}"
    resolved_output = (root_path / output_dir).resolve()
    if not 0.0 < target_fpr < 1.0:
        raise ValueError("target_fpr 必须位于 (0, 1)")
    if max_new_runs_per_session < 0:
        raise ValueError("max_new_runs_per_session 不得为负")
    resolved_output.mkdir(parents=True, exist_ok=True)
    progress_path = resolved_output / "runtime_rerun_progress.json"
    restore_role_checkpoints(
        repository_root=root_path,
        artifact_role="runtime_rerun_ablation",
        paper_run_name=resolved_paper_run_name,
        allowed_output_prefix=(
            f"outputs/formal_mechanism_ablation/{resolved_paper_run_name}"
        ),
    )
    resolved_specs = specs or default_runtime_rerun_ablation_specs()
    ablation_contract = runtime_rerun_ablation_contract(resolved_specs)
    if not ablation_contract["ablation_exact_set_ready"]:
        raise ValueError("正式消融必须精确使用受治理的15项机制规范")
    resolved_base_configs = tuple(base_configs)
    if not resolved_base_configs:
        raise ValueError("真实重运行消融至少需要一个 Prompt 配置")
    prompt_contract = _canonical_prompt_contract(
        root_path,
        paper_run,
        resolved_base_configs,
    )
    split_counts = {
        split: sum(config.split == split for config in resolved_base_configs)
        for split in ("dev", "calibration", "test")
    }
    if split_counts["calibration"] == 0 or split_counts["test"] == 0:
        raise ValueError("正式消融必须同时包含 calibration 和 test split")

    shared_context = None
    run_entries: list[dict[str, Any]] = []
    resumed_run_count = 0
    new_run_count = 0
    expected_run_count = len(resolved_base_configs) * len(resolved_specs)

    def write_resume_progress() -> dict[str, Any]:
        """原子保存正式消融进度并同步到外部检查点目录."""

        progress = {
            "paper_run_name": resolved_paper_run_name,
            **ablation_contract,
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
            artifact_role="runtime_rerun_ablation",
            paper_run_name=resolved_paper_run_name,
        )
        return progress

    for prompt_index, base_config in enumerate(resolved_base_configs):
        for spec in resolved_specs:
            run_config = spec.apply(base_config, output_dir)
            result = load_completed_semantic_watermark_runtime_result(run_config, root=root_path)
            generated_now = False
            if result is not None:
                resumed_run_count += 1
            elif max_new_runs_per_session and new_run_count >= max_new_runs_per_session:
                continue
            else:
                if shared_context is None:
                    shared_context = load_semantic_watermark_runtime_context(resolved_base_configs[0])
                result = write_semantic_watermark_runtime_outputs(
                    run_config,
                    root=root_path,
                    runtime_context=shared_context,
                )
                new_run_count += 1
                generated_now = True
            result_payload = result.to_dict()
            validate_semantic_watermark_runtime_result_provenance(
                result_payload,
                expected_config=run_config,
            )
            detections = _read_jsonl(root_path / result.detection_record_path)
            run_entries.append(_run_entry(prompt_index, base_config, spec, result, detections))
            if generated_now and len(run_entries) < expected_run_count:
                write_resume_progress()

    if len(run_entries) != expected_run_count:
        return write_resume_progress()
    progress_path.unlink(missing_ok=True)
    clear_progress_checkpoints(
        artifact_role="runtime_rerun_ablation",
        paper_run_name=resolved_paper_run_name,
    )

    protocols: dict[str, Any] = {}
    formal_records: list[dict[str, Any]] = []
    formal_detection_records: list[dict[str, Any]] = []
    for spec in resolved_specs:
        spec_entries = tuple(entry for entry in run_entries if entry["ablation_id"] == spec.ablation_id)
        calibration_negatives = tuple(
            detection
            for entry in spec_entries
            if entry["split"] == "calibration"
            for detection in entry["detections"]
            if detection.get("sample_role") == "clean_negative" and not detection.get("attack_id")
        )
        reference_config = next(
            config for config in resolved_base_configs if config.split == "calibration"
        )
        protocol = calibrate_complete_evidence_protocol(
            calibration_negatives,
            target_fpr=target_fpr,
            rescue_margin_low=reference_config.rescue_margin_low,
        )
        protocols[spec.ablation_id] = protocol
        for entry in spec_entries:
            detections = apply_frozen_evidence_protocol(entry["detections"], protocol)
            formal_attack_coverage_ready = _formal_attack_coverage_ready(
                detections,
                split=str(entry["split"]),
            )
            formal_records.append(
                {
                    **_formal_record(
                        entry,
                        detections,
                        protocol.threshold_digest,
                        protocol.content_threshold,
                    ),
                    "formal_attack_coverage_ready": formal_attack_coverage_ready,
                }
            )
            formal_detection_records.extend(
                {
                    **record,
                    "ablation_id": spec.ablation_id,
                    "ablation_prompt_id": entry["prompt_id"],
                }
                for record in detections
            )

    grouped_test_records = {
        spec.ablation_id: [
            record
            for record in formal_records
            if record["ablation_id"] == spec.ablation_id and record["split"] == "test"
        ]
        for spec in resolved_specs
    }
    metric_rows: list[dict[str, Any]] = []
    for spec in resolved_specs:
        group = grouped_test_records[spec.ablation_id]
        metric_rows.append(
            {
                "ablation_id": spec.ablation_id,
                "test_prompt_count": len(group),
                "clean_false_positive_rate": _mean(
                    float(record["clean_negative_positive"]) for record in group
                ),
                "wrong_key_false_positive_rate": _mean(
                    float(record["wrong_key_negative_positive"]) for record in group
                ),
                "clean_true_positive_rate": _mean(
                    float(record["positive_source_positive"]) for record in group
                ),
                "attacked_true_positive_rate": _mean(
                    float(record["attacked_positive_rate"]) for record in group
                ),
                "attacked_false_positive_rate": _mean(
                    float(record["attacked_negative_rate"]) for record in group
                ),
                "positive_content_score_mean": _mean(
                    float(record["positive_source_content_score"]) for record in group
                ),
                "paired_ssim_mean": _mean(float(record["paired_ssim"]) for record in group),
                "frozen_threshold_digest": protocols[spec.ablation_id].threshold_digest,
                "metric_status": "measured_full_runtime_rerun",
            }
        )
    complete_row = next(row for row in metric_rows if row["ablation_id"] == "complete_method")
    delta_rows = [
        {
            "ablation_id": row["ablation_id"],
            "clean_true_positive_rate_delta": (
                row["clean_true_positive_rate"] - complete_row["clean_true_positive_rate"]
            ),
            "attacked_true_positive_rate_delta": (
                row["attacked_true_positive_rate"] - complete_row["attacked_true_positive_rate"]
            ),
            "paired_ssim_delta": row["paired_ssim_mean"] - complete_row["paired_ssim_mean"],
            "metric_status": "measured_full_runtime_rerun",
        }
        for row in metric_rows
        if row["ablation_id"] != "complete_method"
    ]

    records_path = resolved_output / "runtime_rerun_records.jsonl"
    detections_path = resolved_output / "formal_detection_records.jsonl"
    thresholds_path = resolved_output / "per_ablation_frozen_protocols.json"
    metrics_path = resolved_output / "mechanism_ablation_metrics.csv"
    delta_path = resolved_output / "mechanism_pairwise_delta.csv"
    necessity_statistics_path = (
        resolved_output / "mechanism_necessity_statistics.csv"
    )
    necessity_summary_path = (
        resolved_output / "mechanism_necessity_summary.json"
    )
    summary_path = resolved_output / "ablation_claim_summary.json"
    manifest_path = resolved_output / "manifest.local.json"
    records_path.write_text(
        "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in formal_records),
        encoding="utf-8",
    )
    detections_path.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
            for record in formal_detection_records
        ),
        encoding="utf-8",
    )
    thresholds_path.write_text(
        json.dumps(
            {name: protocol.to_dict() for name, protocol in protocols.items()},
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    formal_detection_records_sha256 = repository_environment.file_digest(
        detections_path
    )
    formal_detection_records_digest = build_stable_digest(
        formal_detection_records
    )
    frozen_protocol_payload = {
        name: protocol.to_dict() for name, protocol in protocols.items()
    }
    per_ablation_frozen_protocols_sha256 = repository_environment.file_digest(
        thresholds_path
    )
    per_ablation_frozen_protocols_digest = build_stable_digest(
        frozen_protocol_payload
    )
    _write_csv(metrics_path, metric_rows)
    _write_csv(delta_path, delta_rows)
    necessity_rows, necessity_summary = build_ablation_necessity_statistics(
        formal_records,
        expected_ablation_ids=tuple(
            ablation_id
            for ablation_id in FORMAL_RUNTIME_RERUN_ABLATION_IDS
            if ablation_id != "complete_method"
        ),
        expected_paired_prompt_count=split_counts["test"],
    )
    with necessity_statistics_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=ABLATION_NECESSITY_FIELDNAMES,
        )
        writer.writeheader()
        writer.writerows(necessity_rows)
    necessity_summary_path.write_text(
        json.dumps(
            necessity_summary,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    scientific_unit_provenance = aggregate_scientific_unit_provenance(
        (
            record["runtime_result"]["metadata"]["scientific_unit_provenance"]
            for record in formal_records
        ),
        expected_reference_count=expected_run_count,
    )
    expected_attack_and_detection_rerun_count = (
        split_counts["test"] * len(resolved_specs)
    )
    attack_and_detection_rerun_count = sum(
        bool(record["attack_and_detection_rerun"]) for record in formal_records
    )
    ablation_claim_gate_ready = (
        ablation_contract["ablation_exact_set_ready"]
        and prompt_contract["prompt_protocol_exact_set_ready"]
        and len(formal_records) == expected_run_count
        and all(record["runtime_result"]["run_decision"] == "pass" for record in formal_records)
        and all(record["formal_attack_coverage_ready"] for record in formal_records)
        and attack_and_detection_rerun_count
        == expected_attack_and_detection_rerun_count
        and len(protocols) == len(resolved_specs)
        and scientific_unit_provenance["scientific_unit_provenance_ready"]
        and scientific_unit_provenance["scientific_unit_provenance_record_count"]
        == expected_run_count
        and all(
            protocol.calibration_negative_count == split_counts["calibration"]
            for protocol in protocols.values()
        )
        and all(
            len(grouped_test_records[spec.ablation_id]) == split_counts["test"]
            for spec in resolved_specs
        )
    )
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "paper_run_name": resolved_paper_run_name,
        **ablation_contract,
        **prompt_contract,
        "record_count": len(formal_records),
        "resumed_run_count": resumed_run_count,
        "new_run_count": new_run_count,
        "prompt_count": len(resolved_base_configs),
        "split_counts": split_counts,
        "ablation_count": len(resolved_specs),
        "per_ablation_calibration_count": len(protocols),
        "target_fpr": target_fpr,
        "generation_rerun_count": len(formal_records),
        "expected_attack_and_detection_rerun_count": (
            expected_attack_and_detection_rerun_count
        ),
        "attack_and_detection_rerun_count": attack_and_detection_rerun_count,
        "formal_attack_coverage_ready_count": sum(
            bool(record["formal_attack_coverage_ready"])
            for record in formal_records
        ),
        "formal_attack_coverage_ready": all(
            record["formal_attack_coverage_ready"] for record in formal_records
        ),
        "formal_detection_records_sha256": formal_detection_records_sha256,
        "formal_detection_records_digest": formal_detection_records_digest,
        "per_ablation_frozen_protocols_sha256": (
            per_ablation_frozen_protocols_sha256
        ),
        "per_ablation_frozen_protocols_digest": (
            per_ablation_frozen_protocols_digest
        ),
        **scientific_unit_provenance,
        **necessity_summary,
        "ablation_claim_gate_ready": ablation_claim_gate_ready,
        "protocol_decision": (
            "pass"
            if ablation_claim_gate_ready
            and necessity_summary["ablation_necessity_statistics_ready"]
            else "fail"
        ),
        "supports_paper_claim": (
            ablation_claim_gate_ready
            and necessity_summary["ablation_necessity_statistics_ready"]
        ),
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    formal_execution_run_lock = repository_environment.validate_formal_execution_lock_pair(
        formal_execution_run_lock,
        repository_environment.require_published_formal_execution_lock(root_path),
        formal_execution_run_lock["formal_execution_commit"],
    )
    manifest = build_artifact_manifest(
        artifact_id="formal_mechanism_ablation_manifest",
        artifact_type="local_manifest",
        input_paths=(),
        output_paths=(
            records_path.relative_to(root_path).as_posix(),
            detections_path.relative_to(root_path).as_posix(),
            thresholds_path.relative_to(root_path).as_posix(),
            metrics_path.relative_to(root_path).as_posix(),
            delta_path.relative_to(root_path).as_posix(),
            necessity_statistics_path.relative_to(root_path).as_posix(),
            necessity_summary_path.relative_to(root_path).as_posix(),
            summary_path.relative_to(root_path).as_posix(),
            manifest_path.relative_to(root_path).as_posix(),
        ),
        config={
            "specs": [spec.to_dict() for spec in resolved_specs],
            **ablation_contract,
            **prompt_contract,
            "prompt_count": len(resolved_base_configs),
            "split_counts": split_counts,
            "target_fpr": target_fpr,
            "record_digest": build_stable_digest(formal_records),
            "formal_detection_records_sha256": formal_detection_records_sha256,
            "formal_detection_records_digest": formal_detection_records_digest,
            "per_ablation_frozen_protocols_sha256": (
                per_ablation_frozen_protocols_sha256
            ),
            "per_ablation_frozen_protocols_digest": (
                per_ablation_frozen_protocols_digest
            ),
            "necessity_statistic_rows_digest": necessity_summary[
                "necessity_statistic_rows_digest"
            ],
            "necessity_summary_digest": build_stable_digest(necessity_summary),
        },
        code_version=formal_execution_run_lock["formal_execution_commit"],
        rebuild_command="调用 experiments.ablations.runtime_rerun.run_runtime_rerun_ablations",
        metadata={
            "protocol_decision": summary["protocol_decision"],
            **ablation_contract,
            **prompt_contract,
            "generation_rerun_required": True,
            "per_ablation_calibration_required": True,
            "formal_attack_coverage_ready": summary[
                "formal_attack_coverage_ready"
            ],
            "formal_detection_records_sha256": formal_detection_records_sha256,
            "formal_detection_records_digest": formal_detection_records_digest,
            "per_ablation_frozen_protocols_sha256": (
                per_ablation_frozen_protocols_sha256
            ),
            "per_ablation_frozen_protocols_digest": (
                per_ablation_frozen_protocols_digest
            ),
            "scientific_unit_provenance_ready": summary[
                "scientific_unit_provenance_ready"
            ],
            "scientific_unit_provenance_records_digest": summary[
                "scientific_unit_provenance_records_digest"
            ],
            "ablation_necessity_statistics_ready": summary[
                "ablation_necessity_statistics_ready"
            ],
            "necessity_statistic_rows_digest": summary[
                "necessity_statistic_rows_digest"
            ],
            "necessity_supported_ablation_ids": summary[
                "necessity_supported_ablation_ids"
            ],
            "necessity_not_supported_ablation_ids": summary[
                "necessity_not_supported_ablation_ids"
            ],
            "all_mechanism_necessity_claims_supported": summary[
                "all_mechanism_necessity_claims_supported"
            ],
            "supports_paper_claim": summary["supports_paper_claim"],
        },
    ).to_dict()
    manifest["formal_execution_run_lock"] = formal_execution_run_lock
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def package_runtime_rerun_ablations(
    paper_run_name: str,
    root: str | Path = ".",
) -> Path:
    """打包真实重运行消融记录和逐配置运行证据。"""

    root_path = Path(root).resolve()
    formal_execution_package_lock = (
        repository_environment.require_published_formal_execution_lock(root_path)
    )
    resolved_paper_run_name = normalize_paper_run_name(paper_run_name)
    paper_run = build_paper_run_config(root_path)
    if paper_run.run_name != resolved_paper_run_name:
        raise ValueError("正式消融打包层级必须与当前论文配置一致")
    source_dir = (
        root_path / "outputs" / "formal_mechanism_ablation" / resolved_paper_run_name
    ).resolve()
    required_paths = tuple(
        source_dir / filename
        for filename in (
            "runtime_rerun_records.jsonl",
            "formal_detection_records.jsonl",
            "per_ablation_frozen_protocols.json",
            "mechanism_ablation_metrics.csv",
            "mechanism_pairwise_delta.csv",
            "mechanism_necessity_statistics.csv",
            "mechanism_necessity_summary.json",
            "ablation_claim_summary.json",
            "manifest.local.json",
        )
    )
    if any(not path.is_file() for path in required_paths):
        raise FileNotFoundError("真实重运行消融输出不完整, 不得打包")
    summary = json.loads((source_dir / "ablation_claim_summary.json").read_text(encoding="utf-8-sig"))
    manifest = json.loads((source_dir / "manifest.local.json").read_text(encoding="utf-8-sig"))
    packaged_records = _read_jsonl(source_dir / "runtime_rerun_records.jsonl")
    for record in packaged_records:
        validate_semantic_watermark_runtime_result_provenance(
            record["runtime_result"]
        )
    packaged_unit_config_contract_ready = all(
        record["runtime_result"]["metadata"]["scientific_unit_config"].get(
            field_name
        )
        == record["runtime_config"][field_name]
        for record in packaged_records
        for field_name in (
            "semantic_routing_enabled",
            "branch_risk_mode",
            "null_space_enabled",
            "lf_enabled",
            "tail_robust_enabled",
            "tail_truncation_enabled",
            "attention_geometry_enabled",
            "image_alignment_enabled",
            "attention_relation_component_weights",
        )
    ) and all(
        record["runtime_result"]["metadata"]["scientific_unit_config"].get(
            field_name
        )
        == record[field_name]
        for record in packaged_records
        for field_name in ("prompt_id", "split")
    ) and all(
        build_stable_digest(
            {
                "prompt_text": record["runtime_result"]["metadata"][
                    "scientific_unit_config"
                ]["prompt"]
            }
        )
        == record["prompt_digest"]
        and record["runtime_result"]["metadata"]["scientific_unit_config"].get(
            "output_dir"
        )
        == (
            f"outputs/formal_mechanism_ablation/"
            f"{resolved_paper_run_name}/runs/{record['ablation_id']}"
        )
        for record in packaged_records
    )
    packaged_scientific_unit_provenance = aggregate_scientific_unit_provenance(
        (
            record["runtime_result"]["metadata"]["scientific_unit_provenance"]
            for record in packaged_records
        ),
        expected_reference_count=len(packaged_records),
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
    expected_ids = list(FORMAL_RUNTIME_RERUN_ABLATION_IDS)
    manifest_config = manifest.get("config", {})
    manifest_metadata = manifest.get("metadata", {})
    with (source_dir / "mechanism_ablation_metrics.csv").open(
        encoding="utf-8-sig",
        newline="",
    ) as stream:
        metric_ids = [str(row.get("ablation_id", "")) for row in csv.DictReader(stream)]
    with (source_dir / "mechanism_necessity_statistics.csv").open(
        encoding="utf-8-sig",
        newline="",
    ) as stream:
        necessity_statistic_rows = [dict(row) for row in csv.DictReader(stream)]
    necessity_summary = json.loads(
        (source_dir / "mechanism_necessity_summary.json").read_text(
            encoding="utf-8-sig"
        )
    )
    frozen_protocols_path = source_dir / "per_ablation_frozen_protocols.json"
    frozen_protocol_payload = json.loads(
        frozen_protocols_path.read_text(encoding="utf-8-sig")
    )
    protocol_ids = list(frozen_protocol_payload)
    formal_detection_records_path = source_dir / "formal_detection_records.jsonl"
    packaged_detection_records = _read_jsonl(formal_detection_records_path)
    atom_identity = {
        "formal_detection_records_sha256": repository_environment.file_digest(
            formal_detection_records_path
        ),
        "formal_detection_records_digest": build_stable_digest(
            packaged_detection_records
        ),
        "per_ablation_frozen_protocols_sha256": (
            repository_environment.file_digest(frozen_protocols_path)
        ),
        "per_ablation_frozen_protocols_digest": build_stable_digest(
            frozen_protocol_payload
        ),
    }
    exact_set_ready = all(
        (
            summary.get("expected_ablation_ids") == expected_ids,
            summary.get("actual_ablation_ids") == expected_ids,
            summary.get("ablation_spec_digest")
            == FORMAL_RUNTIME_RERUN_ABLATION_SPEC_DIGEST,
            summary.get("ablation_exact_set_ready") is True,
            manifest_config.get("expected_ablation_ids") == expected_ids,
            manifest_config.get("actual_ablation_ids") == expected_ids,
            manifest_config.get("ablation_spec_digest")
            == FORMAL_RUNTIME_RERUN_ABLATION_SPEC_DIGEST,
            manifest_config.get("ablation_exact_set_ready") is True,
            manifest_metadata.get("expected_ablation_ids") == expected_ids,
            manifest_metadata.get("actual_ablation_ids") == expected_ids,
            manifest_metadata.get("ablation_spec_digest")
            == FORMAL_RUNTIME_RERUN_ABLATION_SPEC_DIGEST,
            manifest_metadata.get("ablation_exact_set_ready") is True,
            metric_ids == expected_ids,
            len(protocol_ids) == len(expected_ids),
            set(protocol_ids) == set(expected_ids),
            all(
                summary.get(field_name) == value
                and manifest_config.get(field_name) == value
                and manifest_metadata.get(field_name) == value
                for field_name, value in atom_identity.items()
            ),
        )
    )
    expected_variant_ids = [
        ablation_id
        for ablation_id in expected_ids
        if ablation_id != "complete_method"
    ]
    necessity_ids = [
        str(row.get("ablation_id", "")) for row in necessity_statistic_rows
    ]
    necessity_partition = [
        *necessity_summary.get("necessity_supported_ablation_ids", []),
        *necessity_summary.get("necessity_not_supported_ablation_ids", []),
    ]
    necessity_statistics_ready = all(
        (
            necessity_ids == expected_variant_ids,
            sorted(necessity_partition) == sorted(expected_variant_ids),
            len(necessity_partition) == len(set(necessity_partition)),
            necessity_summary.get("expected_variant_ablation_ids")
            == expected_variant_ids,
            necessity_summary.get("ablation_necessity_statistics_ready") is True,
            necessity_summary.get("necessity_statistic_row_count")
            == len(expected_variant_ids),
            necessity_summary.get("paired_prompt_count")
            == paper_run.minimum_clean_negative_count,
            necessity_summary.get("expected_paired_prompt_count")
            == paper_run.minimum_clean_negative_count,
            necessity_summary.get("necessity_statistic_rows_digest")
            == build_stable_digest(necessity_statistic_rows),
            summary.get("necessity_statistic_rows_digest")
            == necessity_summary.get("necessity_statistic_rows_digest"),
            summary.get("ablation_necessity_statistics_ready") is True,
            summary.get("necessity_supported_ablation_ids")
            == necessity_summary.get("necessity_supported_ablation_ids"),
            summary.get("necessity_not_supported_ablation_ids")
            == necessity_summary.get("necessity_not_supported_ablation_ids"),
            manifest_config.get("necessity_statistic_rows_digest")
            == necessity_summary.get("necessity_statistic_rows_digest"),
            manifest_config.get("necessity_summary_digest")
            == build_stable_digest(necessity_summary),
            manifest_metadata.get("ablation_necessity_statistics_ready") is True,
            manifest_metadata.get("necessity_statistic_rows_digest")
            == necessity_summary.get("necessity_statistic_rows_digest"),
        )
    )
    if not all(
        (
            summary.get("paper_run_name") == resolved_paper_run_name,
            abs(float(summary.get("target_fpr", -1.0)) - paper_run.target_fpr) <= 1e-12,
            bool(summary.get("generated_at")),
            summary.get("protocol_decision") == "pass",
            summary.get("ablation_claim_gate_ready") is True,
            summary.get("scientific_unit_provenance_ready") is True,
            summary.get("scientific_unit_provenance_record_count")
            == int(summary.get("record_count", -1)),
            bool(summary.get("scientific_unit_provenance_records_digest")),
            scientific_unit_provenance_summary_bound,
            packaged_unit_config_contract_ready,
            summary.get("supports_paper_claim") is True,
            manifest.get("artifact_id") == "formal_mechanism_ablation_manifest",
            manifest_config_digest_ready(manifest),
            exact_set_ready,
            necessity_statistics_ready,
        )
    ):
        raise RuntimeError("真实重运行消融身份、精确15项规范或 ready 门禁未通过")
    manifest["formal_execution_package_lock"] = formal_execution_package_lock
    (source_dir / "manifest.local.json").write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    validate_scientific_execution_binding(
        source_dir / "scientific_execution_binding.json",
        expected_artifact_role="runtime_rerun_ablation",
        expected_paper_run_name=resolved_paper_run_name,
        repository_root=root_path,
    )
    code_version = formal_execution_package_lock["formal_execution_commit"]
    archive_path = source_dir / (
        f"runtime_rerun_ablation_package_{utc_archive_token()}_{code_version[:7]}.zip"
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
        raise RuntimeError("artifact manifest 未精确声明全部正式消融必要产物")
    write_exact_package_input_manifest(
        package_input_manifest_path,
        repository_root=root_path,
        package_family="runtime_rerun_ablation",
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
