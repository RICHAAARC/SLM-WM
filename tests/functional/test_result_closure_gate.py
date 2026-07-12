"""论文结果闭合语义门禁的轻量功能测试。"""

from __future__ import annotations

import csv
from copy import deepcopy
from dataclasses import replace
from functools import lru_cache
import hashlib
import io
import json
from pathlib import Path
import sys

import pytest

from experiments.ablations.runtime_rerun import (
    FORMAL_RUNTIME_RERUN_ABLATION_IDS,
    FORMAL_RUNTIME_RERUN_ABLATION_SPECS,
    FORMAL_RUNTIME_RERUN_ABLATION_SPEC_DIGEST,
)
from experiments.protocol.attacks import (
    attack_config_digest,
    build_attack_record_digest,
    build_attack_matrix_manifest_config,
    default_attack_configs,
)
from experiments.ablations.necessity_statistics import (
    ABLATION_NECESSITY_FIELDNAMES,
    build_ablation_necessity_statistics,
)
from experiments.protocol.paper_run_config import (
    RUN_DEFAULTS,
    PaperRunPromptContract,
)
from experiments.protocol.prompts import build_prompt_records
from experiments.protocol.splits import group_prompt_ids_by_split
from experiments.protocol.pilot_paper_fixed_fpr import (
    PILOT_PAPER_METRIC_BOUNDS,
    PilotPaperFixedFprConfig,
    bounded_hoeffding_confidence_interval,
    build_attack_matrix_digest,
    build_fixed_fpr_protocol_digest,
    build_pilot_paper_attack_matrix_rows,
    build_pilot_paper_prompt_split_summary,
    build_pilot_paper_result_import_schema,
    build_pilot_paper_result_record_set_digest,
    build_pilot_paper_result_records_manifest_config,
    clamp_unit_interval,
    validate_pilot_paper_result_import_rows,
)
from experiments.runtime.image_metrics import measured_score_retention
from experiments.artifacts.detection_score_curves import (
    CURVE_POINT_FIELDNAMES,
    SCORE_DISTRIBUTION_FIELDNAMES,
    build_detection_score_tables,
)
from experiments.artifacts.image_only_detection_metrics import (
    build_image_only_test_metric_rows,
)
from experiments.protocol.formal_randomization import (
    build_formal_randomization_identity,
    resolve_formal_randomization_repeat,
)
from experiments.artifacts.dataset_level_quality_outputs import (
    _inception_batch_config_digest,
    validate_inception_feature_provenance_groups,
)
from experiments.runners.image_only_dataset_runtime import (
    apply_frozen_evidence_protocol,
    calibrate_complete_evidence_protocol,
)
from experiments.runtime.scientific_unit_provenance import (
    aggregate_scientific_unit_provenance,
)
from main.methods.method_definition import (
    semantic_conditioned_latent_method_definition,
    semantic_conditioned_latent_method_definition_digest,
)
from main.core.digest import build_stable_digest
from paper_experiments.analysis.result_closure_gate import (
    ResultClosureGateInput,
    build_result_closure_gate_checks,
    build_result_closure_gate_report,
)
from paper_experiments.analysis.paired_superiority import (
    BOOTSTRAP_ANALYSIS_SCHEMA,
    BOOTSTRAP_BIT_GENERATOR,
    BOOTSTRAP_QUANTILE_METHOD,
    CLAIM_P_VALUE_METHOD,
    DEFAULT_BOOTSTRAP_RESAMPLE_COUNT,
    DEFAULT_CONFIDENCE_LEVEL,
    SHARP_NULL_DIAGNOSTIC_METHOD,
    build_paired_superiority_protocol_digest,
    build_paired_outcomes,
    build_paired_superiority_manifest_config,
    build_paired_superiority_rows,
    build_paired_superiority_summary,
    build_quality_matched_superiority_rows,
    build_quality_matched_superiority_summary,
    canonical_attack_registry_rows,
    canonical_threshold_audit_rows,
    merge_paired_and_quality_matched_rows,
)
from paper_experiments.analysis.fixed_fpr_threshold_audit import (
    build_fixed_fpr_threshold_manifest_config,
)
from paper_experiments.analysis.paper_artifact_data_validation import (
    ABLATION_DELTA_FIELDS,
    ABLATION_METRIC_FIELDS,
    BASELINE_COMPARISON_FIELDS,
    DATASET_QUALITY_FIELDS,
    TEST_METRIC_FIELDS,
)
from paper_experiments.analysis.paper_evidence_audit import (
    AuditInputBundle,
    build_evidence_audit_manifest_config,
    build_evidence_audit_materialization,
)
from experiments.protocol.dataset_quality import (
    FORMAL_FEATURE_BACKEND,
    FORMAL_FEATURE_EXTRACTOR_ID,
    formal_dataset_quality_metric_protocol,
    rebuild_formal_fid_kid_metric_rows,
)
from paper_experiments.analysis.result_analysis_payload import (
    build_confidence_interval_rows,
    build_failure_case_records,
    build_failure_case_svg_text,
    build_governed_paper_payload_path_map,
    build_main_comparison_rows_from_result_records,
    build_per_attack_superiority_rows,
    rebuild_and_validate_result_analysis_derived_payload,
    build_result_analysis_manifest_config,
)
from scripts.write_result_closure_gate_outputs import main, write_result_closure_gate_outputs
from scripts.write_paper_artifact_evidence_audit_outputs import (
    write_paper_artifact_evidence_audit_outputs,
)
from tests.helpers.closure_input_lock import (
    build_test_closure_input_lock_payloads,
)
from tests.helpers.formal_execution_lock import (
    build_test_formal_execution_lock,
)
from tests.helpers.scientific_unit_provenance import (
    build_test_scientific_unit_provenance,
)


SCALE = "probe_paper"
TARGET_FPR = 0.1
PROMPT_COUNT = 70
TEST_COUNT = 34
MAIN_THRESHOLD_DIGEST = "1" * 64
PROMPT_RECORDS = build_prompt_records(
    SCALE,
    tuple(f"a governed prompt {index}" for index in range(PROMPT_COUNT)),
)
PAPER_CONFIG = PilotPaperFixedFprConfig(
    paper_run_name=SCALE,
    prompt_set=SCALE,
    prompt_file=str(RUN_DEFAULTS[SCALE]["prompt_file"]),
    prompt_protocol_name=f"paper_main_{SCALE}_prompt_protocol",
    result_protocol_name=f"{SCALE}_fixed_fpr_common_protocol",
    result_scope=f"{SCALE}_common_protocol",
    result_claim_scope="probe_claim",
    target_fpr=TARGET_FPR,
    minimum_clean_negative_count=TEST_COUNT,
)
PROMPT_SPLIT_SUMMARY = build_pilot_paper_prompt_split_summary(
    PROMPT_RECORDS,
    PAPER_CONFIG,
)
PROMPT_SPLIT_DIGEST = str(PROMPT_SPLIT_SUMMARY["prompt_split_digest"])
PROMPT_ID_DIGEST = build_stable_digest(sorted(record.prompt_id for record in PROMPT_RECORDS))
TEST_PROMPT_IDS = tuple(group_prompt_ids_by_split(PROMPT_RECORDS)["test"])
CALIBRATION_PROMPT_IDS = tuple(
    group_prompt_ids_by_split(PROMPT_RECORDS)["calibration"]
)
PROMPT_SPLIT_BY_ID = {
    prompt_id: split
    for split, prompt_ids in group_prompt_ids_by_split(PROMPT_RECORDS).items()
    for prompt_id in prompt_ids
}
PROMPT_DIGEST_BY_ID = {
    record.prompt_id: record.prompt_digest
    for record in PROMPT_RECORDS
}
PROMPT_INDEX_BY_ID = {
    record.prompt_id: record.prompt_index
    for record in PROMPT_RECORDS
}


def paired_randomization_identity(prompt_id: str) -> dict[str, object]:
    """构造闭合夹具中所有方法共享的正式随机化身份."""

    prompt_index = PROMPT_INDEX_BY_ID[prompt_id]
    identity = build_formal_randomization_identity(
        base_seed=1703,
        prompt_index=prompt_index,
        root_key_material="slm_wm_paper_key",
        repeat=resolve_formal_randomization_repeat("seed_00_key_00"),
    )
    base_content_digest = build_stable_digest(
        {"prompt_id": prompt_id, "tensor_role": "base_latent"}
    )
    return {
        **identity,
        "base_latent_content_digest_random": base_content_digest,
        "base_latent_identity_digest_random": build_stable_digest(
            {
                "prompt_id": prompt_id,
                "base_latent_content_digest_random": base_content_digest,
            }
        ),
    }
from experiments.artifacts.attack_family_metrics import (
    build_attack_family_metrics,
)
CALIBRATION_PROMPT_ID_DIGEST = build_stable_digest(
    sorted(CALIBRATION_PROMPT_IDS)
)
TEST_PROMPT_ID_DIGEST = build_stable_digest(
    TEST_PROMPT_IDS
)
QUALITY_FORMAL_EXECUTION_LOCK = build_test_formal_execution_lock()


def dataset_quality_atomic_records() -> tuple[
    tuple[dict[str, object], ...],
    tuple[dict[str, object], ...],
    tuple[dict[str, object], ...],
    dict[str, object],
]:
    """构造图像、2048维 Inception 特征和完整科学来源绑定。"""

    image_records: list[dict[str, object]] = []
    resolution_records: list[dict[str, object]] = []
    feature_records: list[dict[str, object]] = []
    item_identity: list[dict[str, object]] = []
    for pair_index, prompt_record in enumerate(PROMPT_RECORDS):
        source_path = (
            f"outputs/dataset_level_quality/{SCALE}/images/"
            f"source_{pair_index:04d}.png"
        )
        comparison_path = (
            f"outputs/dataset_level_quality/{SCALE}/images/"
            f"comparison_{pair_index:04d}.png"
        )
        source_bytes = (
            f"quality-image:{prompt_record.prompt_id}:source"
        ).encode("utf-8")
        comparison_bytes = (
            f"quality-image:{prompt_record.prompt_id}:comparison"
        ).encode("utf-8")
        source_digest = hashlib.sha256(source_bytes).hexdigest()
        comparison_digest = hashlib.sha256(comparison_bytes).hexdigest()
        image_payload: dict[str, object] = {
            "run_id": f"quality_runtime_{pair_index:05d}",
            "prompt_id": prompt_record.prompt_id,
            "attack_name": "watermark_embedding",
            "image_pair_index": pair_index,
            "image_pair_role": "clean_to_watermarked",
            "source_image_path": source_path,
            "source_image_digest": source_digest,
            "comparison_image_path": comparison_path,
            "comparison_image_digest": comparison_digest,
            "feature_backend": FORMAL_FEATURE_BACKEND,
            "supports_paper_claim": False,
        }
        image_digest = build_stable_digest(image_payload)
        record_id = f"dataset_quality_record_{image_digest[:16]}"
        image_records.append(
            {
                "dataset_quality_record_id": record_id,
                "dataset_quality_record_digest": image_digest,
                **image_payload,
            }
        )
        for role, path, digest, first_value in (
            (
                "source",
                source_path,
                source_digest,
                pair_index / PROMPT_COUNT,
            ),
            (
                "comparison",
                comparison_path,
                comparison_digest,
                pair_index / PROMPT_COUNT + 0.1,
            ),
        ):
            resolution_payload: dict[str, object] = {
                "requested_image_path": path,
                "resolved_image_path": path,
                "resolved_from_package_path": "",
                "resolution_status": "resolved_existing_image_file",
                "resolved_image_digest": digest,
                "materialized_image_input": False,
                "supports_paper_claim": False,
            }
            resolution_digest = build_stable_digest(resolution_payload)
            resolution_records.append(
                {
                    **resolution_payload,
                    "image_resolution_record_digest": resolution_digest,
                    "image_resolution_record_id": (
                        "dataset_quality_image_resolution_"
                        f"{resolution_digest[:16]}"
                    ),
                }
            )
            identity: dict[str, object] = {
                "dataset_quality_record_id": record_id,
                "dataset_quality_image_role": role,
                "image_path": path,
                "image_digest": digest,
            }
            item_identity.append(identity)
            feature_records.append(
                {
                    **identity,
                    "feature_backend": FORMAL_FEATURE_BACKEND,
                    "feature_extractor_id": FORMAL_FEATURE_EXTRACTOR_ID,
                    "feature_dimension": 2048,
                    "feature_vector": [float(first_value), *([0.0] * 2047)],
                    "supports_paper_claim": False,
                }
            )

    batch_identity_digest = build_stable_digest(
        [
            (
                identity["dataset_quality_record_id"],
                identity["dataset_quality_image_role"],
            )
            for identity in item_identity
        ]
    )
    provenance = build_test_scientific_unit_provenance(
        f"feature_batch_{batch_identity_digest[:16]}",
        _inception_batch_config_digest(item_identity),
        formal_execution_lock=QUALITY_FORMAL_EXECUTION_LOCK,
    )
    for feature_record in feature_records:
        feature_record["scientific_unit_provenance"] = provenance
    provenance_summary = aggregate_scientific_unit_provenance(
        validate_inception_feature_provenance_groups(feature_records),
        expected_reference_count=len(feature_records),
    )
    return (
        tuple(image_records),
        tuple(resolution_records),
        tuple(feature_records),
        provenance_summary,
    )


(
    QUALITY_IMAGE_RECORDS,
    QUALITY_IMAGE_RESOLUTION_RECORDS,
    QUALITY_FEATURE_RECORDS,
    QUALITY_PROVENANCE_SUMMARY,
) = dataset_quality_atomic_records()
FEATURE_RECORDS_TEXT = "".join(
    json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
    for record in QUALITY_FEATURE_RECORDS
)
FEATURE_RECORDS_SHA256 = hashlib.sha256(
    FEATURE_RECORDS_TEXT.encode("utf-8")
).hexdigest()
COMMON_CODE_VERSION = "a" * 40
METHOD_THRESHOLD_DIGEST_MAP = {
    "slm_wm_current": MAIN_THRESHOLD_DIGEST,
    "tree_ring": "3" * 64,
    "gaussian_shading": "4" * 64,
    "shallow_diffuse": "5" * 64,
    "t2smark": "6" * 64,
}
PRIMARY_BASELINE_IDS = (
    "tree_ring",
    "gaussian_shading",
    "shallow_diffuse",
    "t2smark",
)
OFFICIAL_REFERENCE_BASELINE_IDS = PRIMARY_BASELINE_IDS[:3]
FORMAL_ATTACK_CONFIGS = tuple(
    config
    for config in default_attack_configs()
    if config.enabled and config.resource_profile in {"full_main", "full_extra"}
)
ATTACK_REGISTRY = canonical_attack_registry_rows(
    {
        "attack_id": config.attack_id,
        "attack_family": config.attack_family,
        "attack_name": config.attack_name,
        "resource_profile": config.resource_profile,
        "attack_config_digest": attack_config_digest(config),
    }
    for config in FORMAL_ATTACK_CONFIGS
)
FORMAL_ATTACK_CONFIG_BY_ID = {
    config.attack_id: config for config in FORMAL_ATTACK_CONFIGS
}
ATTACK_MATRIX_ROWS = build_pilot_paper_attack_matrix_rows(
    default_attack_configs(),
    PAPER_CONFIG,
)
ATTACK_MATRIX_DIGEST = build_attack_matrix_digest(ATTACK_MATRIX_ROWS)
FIXED_FPR_PROTOCOL_DIGEST = build_fixed_fpr_protocol_digest(PAPER_CONFIG)
ATTACK_SPECS = tuple(
    (row["attack_family"], row["attack_name"]) for row in ATTACK_REGISTRY
)
METHOD_OBSERVATION_SOURCE_PATH_MAP = {
    "slm_wm": (
        f"outputs/image_only_dataset_runtime/{SCALE}/image_only_detection_records.jsonl"
    ),
    "tree_ring": (
        f"outputs/external_baseline_method_faithful/{SCALE}/split_observations/"
        "tree_ring_baseline_observations.json"
    ),
    "gaussian_shading": (
        f"outputs/external_baseline_method_faithful/{SCALE}/split_observations/"
        "gaussian_shading_baseline_observations.json"
    ),
    "shallow_diffuse": (
        f"outputs/external_baseline_method_faithful/{SCALE}/split_observations/"
        "shallow_diffuse_baseline_observations.json"
    ),
    "t2smark": (
        f"outputs/t2smark_formal_reproduction/{SCALE}/t2smark_adapter/"
        "baseline_observations.json"
    ),
}
PROPOSED_OBSERVATION_RECORDS = tuple(
    [
        *(
            {
                "prompt_id": prompt_id,
                **paired_randomization_identity(prompt_id),
                "split": "calibration",
                "sample_role": "clean_negative",
                "attack_family": "clean",
                "attack_name": "none",
                "resource_profile": "clean",
                "content_score": 0.1,
                "frozen_threshold_digest": MAIN_THRESHOLD_DIGEST,
                "formal_evidence_positive": False,
            }
            for prompt_id in CALIBRATION_PROMPT_IDS
        ),
        *(
            {
                "prompt_id": prompt_id,
                **paired_randomization_identity(prompt_id),
                "split": "test",
                "sample_role": sample_role,
                "attack_family": "clean",
                "attack_name": "none",
                "resource_profile": "clean",
                "content_score": (
                    1.0 if sample_role == "positive_source" else 0.1
                ),
                "source_to_evaluated_ssim": 1.0,
                "source_to_evaluated_psnr": 100.0,
                "embedding_pair_ssim": 0.9,
                "frozen_threshold_digest": MAIN_THRESHOLD_DIGEST,
                "formal_evidence_positive": sample_role == "positive_source",
            }
            for prompt_id in TEST_PROMPT_IDS
            for sample_role in (
                "clean_negative",
                "wrong_key_negative",
                "positive_source",
            )
        ),
        *(
            {
                "prompt_id": prompt_id,
                **paired_randomization_identity(prompt_id),
                "split": "test",
                "sample_role": sample_role,
                **attack,
                "content_score": (
                    0.1 if sample_role == "clean_negative" else 0.8
                ),
                "source_to_evaluated_ssim": 0.9,
                "source_to_evaluated_psnr": 35.0,
                "frozen_threshold_digest": MAIN_THRESHOLD_DIGEST,
                "formal_evidence_positive": sample_role == "positive_source",
                "final_decision": sample_role == "positive_source",
            }
            for prompt_id in TEST_PROMPT_IDS
            for attack in ATTACK_REGISTRY
            for sample_role in ("clean_negative", "positive_source")
        ),
    ]
)
METHOD_OBSERVATION_RECORDS_BY_METHOD = {
    "slm_wm": PROPOSED_OBSERVATION_RECORDS,
    **{
        baseline_id: tuple(
            [
                *(
                    {
                        "baseline_id": baseline_id,
                        "prompt_id": prompt_id,
                        **paired_randomization_identity(prompt_id),
                        "split": "calibration",
                        "sample_role": "clean_negative",
                        "attack_family": "clean",
                        "attack_name": "clean_none",
                        "quality_score": 1.0,
                        "threshold_digest": METHOD_THRESHOLD_DIGEST_MAP[
                            baseline_id
                        ],
                        "detection_decision": False,
                    }
                    for prompt_id in CALIBRATION_PROMPT_IDS
                ),
                *(
                    {
                        "baseline_id": baseline_id,
                        "prompt_id": prompt_id,
                        **paired_randomization_identity(prompt_id),
                        "split": "test",
                        "sample_role": "clean_negative",
                        "attack_family": "clean",
                        "attack_name": "clean_none",
                        "quality_score": 1.0,
                        "threshold_digest": METHOD_THRESHOLD_DIGEST_MAP[
                            baseline_id
                        ],
                        "detection_decision": False,
                    }
                    for prompt_id in TEST_PROMPT_IDS
                ),
                *(
                    {
                        "baseline_id": baseline_id,
                        "prompt_id": prompt_id,
                        **paired_randomization_identity(prompt_id),
                        "split": "test",
                        "sample_role": "positive_source",
                        "attack_family": "clean",
                        "attack_name": "clean_none",
                        "quality_score": 0.9,
                        "threshold_digest": METHOD_THRESHOLD_DIGEST_MAP[
                            baseline_id
                        ],
                        "detection_decision": True,
                        "final_decision": True,
                    }
                    for prompt_id in TEST_PROMPT_IDS
                ),
                *(
                    {
                        "baseline_id": baseline_id,
                        "prompt_id": prompt_id,
                        **paired_randomization_identity(prompt_id),
                        "split": "test",
                        "sample_role": sample_role,
                        **attack,
                        "quality_score": 0.9,
                        "threshold_digest": METHOD_THRESHOLD_DIGEST_MAP[
                            baseline_id
                        ],
                        "detection_decision": (
                            prompt_index < 5
                            if sample_role == "attacked_positive"
                            else False
                        ),
                        "final_decision": (
                            prompt_index < 5
                            if sample_role == "attacked_positive"
                            else False
                        ),
                    }
                    for prompt_index, prompt_id in enumerate(TEST_PROMPT_IDS)
                    for attack in ATTACK_REGISTRY
                    for sample_role in (
                        "attacked_positive",
                        "attacked_negative",
                    )
                ),
            ]
        )
        for baseline_id in PRIMARY_BASELINE_IDS
    },
}
METHOD_OBSERVATION_SOURCE_PAYLOADS = {
    METHOD_OBSERVATION_SOURCE_PATH_MAP[method_id]: (
        "".join(
            json.dumps(row, sort_keys=True) + "\n"
            for row in METHOD_OBSERVATION_RECORDS_BY_METHOD[method_id]
        ).encode("utf-8")
        if method_id == "slm_wm"
        else json.dumps(
            list(METHOD_OBSERVATION_RECORDS_BY_METHOD[method_id]),
            sort_keys=True,
        ).encode("utf-8")
    )
    for method_id in ("slm_wm", *PRIMARY_BASELINE_IDS)
}
METHOD_OBSERVATION_SOURCE_SHA256_MAP = {
    method_id: hashlib.sha256(
        METHOD_OBSERVATION_SOURCE_PAYLOADS[path]
    ).hexdigest()
    for method_id, path in METHOD_OBSERVATION_SOURCE_PATH_MAP.items()
}
ARTIFACT_SOURCE_PATHS = {
    "frozen_evidence_protocol_ready": (
        f"outputs/image_only_dataset_runtime/{SCALE}/frozen_evidence_protocol.json"
    ),
    "raw_image_only_detection_records_ready": (
        f"outputs/image_only_dataset_runtime/{SCALE}/image_only_detection_records.jsonl"
    ),
    "test_detection_metrics_ready": (
        f"outputs/image_only_dataset_runtime/{SCALE}/test_detection_metrics.csv"
    ),
    "score_distribution_table_ready": (
        f"outputs/image_only_dataset_runtime/{SCALE}/score_distribution_table.csv"
    ),
    "roc_curve_points_ready": (
        f"outputs/image_only_dataset_runtime/{SCALE}/roc_curve_points.csv"
    ),
    "det_curve_points_ready": (
        f"outputs/image_only_dataset_runtime/{SCALE}/det_curve_points.csv"
    ),
    "attack_family_metrics_ready": (
        f"outputs/attack_matrix/{SCALE}/attack_family_metrics.csv"
    ),
    "baseline_comparison_table_ready": (
        f"outputs/external_baseline_comparison/{SCALE}/baseline_comparison_table.csv"
    ),
    "mechanism_ablation_metrics_ready": (
        f"outputs/formal_mechanism_ablation/{SCALE}/mechanism_ablation_metrics.csv"
    ),
    "mechanism_pairwise_delta_ready": (
        f"outputs/formal_mechanism_ablation/{SCALE}/mechanism_pairwise_delta.csv"
    ),
    "mechanism_necessity_statistics_ready": (
        f"outputs/formal_mechanism_ablation/{SCALE}/mechanism_necessity_statistics.csv"
    ),
    "dataset_quality_metrics_ready": (
        f"outputs/dataset_level_quality/{SCALE}/dataset_quality_metrics.csv"
    ),
}
RESULT_ANALYSIS_PAYLOAD_PATH_MAP = {
    "main_confidence_interval_table": (
        f"outputs/pilot_paper_result_analysis/{SCALE}/confidence_interval_table.csv"
    ),
    "per_attack_superiority_table": (
        f"outputs/pilot_paper_result_analysis/{SCALE}/per_attack_superiority_table.csv"
    ),
    "failure_case_records": (
        f"outputs/pilot_paper_result_analysis/{SCALE}/failure_case_records.jsonl"
    ),
    "failure_case_figure": (
        f"outputs/pilot_paper_result_analysis/{SCALE}/failure_case_figure.svg"
    ),
}
def evidence_audit_source_path_map() -> dict[str, str]:
    """构造证据审计纯函数使用的当前 run 受治理路径映射."""

    runtime_root = f"outputs/image_only_dataset_runtime/{SCALE}"
    threshold_root = f"outputs/fixed_fpr_threshold_audit/{SCALE}"
    attack_root = f"outputs/attack_matrix/{SCALE}"
    baseline_root = f"outputs/external_baseline_comparison/{SCALE}"
    quality_root = f"outputs/dataset_level_quality/{SCALE}"
    ablation_root = f"outputs/formal_mechanism_ablation/{SCALE}"
    return {
        "threshold_report": f"{runtime_root}/dataset_runtime_summary.json",
        "threshold_audit_report": f"{threshold_root}/threshold_audit_report.json",
        "threshold_audit_rows": f"{threshold_root}/threshold_audit_rows.csv",
        "fixed_fpr_operating_points": f"{runtime_root}/frozen_evidence_protocol.json",
        "standard_watermark_metrics": f"{runtime_root}/test_detection_metrics.csv",
        "quality_metrics_summary": f"{runtime_root}/runtime_results.jsonl",
        "raw_image_only_detection_records": f"{runtime_root}/image_only_detection_records.jsonl",
        "dataset_quality_summary": f"{quality_root}/dataset_quality_summary.json",
        "dataset_quality_metrics": f"{quality_root}/dataset_quality_metrics.csv",
        "score_distribution_table": f"{runtime_root}/score_distribution_table.csv",
        "roc_curve_points": f"{runtime_root}/roc_curve_points.csv",
        "det_curve_points": f"{runtime_root}/det_curve_points.csv",
        "attack_manifest": f"{attack_root}/attack_manifest.json",
        "attack_family_metrics": f"{attack_root}/attack_family_metrics.csv",
        "attack_strength_curve": f"{attack_root}/attack_strength_curve.csv",
        "score_retention_by_attack": f"{attack_root}/score_retention_by_attack.csv",
        "attacked_image_root": f"{runtime_root}/runs",
        "attacked_image_registry": f"{attack_root}/attacked_image_registry.jsonl",
        "baseline_runtime_report": f"{baseline_root}/baseline_runtime_report.json",
        "baseline_comparison_table": f"{baseline_root}/baseline_comparison_table.csv",
        "ablation_claim_summary": f"{ablation_root}/ablation_claim_summary.json",
        "mechanism_ablation_table": f"{ablation_root}/mechanism_ablation_metrics.csv",
        "method_pairwise_delta_table": f"{ablation_root}/mechanism_pairwise_delta.csv",
        "mechanism_necessity_statistics": (
            f"{ablation_root}/mechanism_necessity_statistics.csv"
        ),
        "mechanism_necessity_summary": (
            f"{ablation_root}/mechanism_necessity_summary.json"
        ),
    }


def manifest(
    artifact_id: str,
    output_paths: tuple[str, ...],
    metadata: dict[str, object],
    *,
    input_paths: tuple[str, ...] = (),
    config: dict[str, object] | None = None,
    code_version: str = "a" * 40,
) -> dict[str, object]:
    """构造满足通用 provenance schema 的测试 manifest。"""

    return {
        "artifact_id": artifact_id,
        "artifact_type": "local_manifest",
        "input_paths": list(input_paths),
        "output_paths": list(output_paths),
        "config_digest": build_stable_digest(config or {}),
        "code_version": code_version,
        "rebuild_command": "python scripts/test_builder.py",
        "config": config or {},
        "metadata": metadata,
    }


def csv_bytes(rows: tuple[dict[str, object], ...]) -> bytes:
    """按正式 writer 的 CSV 规则构造可复算字节."""

    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def csv_bytes_with_fields(
    rows: tuple[dict[str, object], ...],
    fieldnames: tuple[str, ...],
) -> bytes:
    """按正式 schema 列顺序构造稳定 CSV 字节."""

    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def schema_row(fields: set[str], **values: object) -> dict[str, object]:
    """用空字符串补齐固定 schema, 让测试数据只突出关键事实."""

    return {field_name: values.get(field_name, "") for field_name in fields}


def artifact_source_payloads(
    quality_metrics: tuple[dict[str, object], ...],
    attack_family_metrics: tuple[dict[str, object], ...],
    necessity_rows: tuple[dict[str, object], ...],
    baseline_rows: tuple[dict[str, object], ...],
) -> dict[str, bytes]:
    """构造可由真实 validator 独立重建的12类源文件字节."""

    protocol = {
        "content_threshold": 0.5,
        "rescue_margin_low": -0.2,
        "geometry_score_threshold": 0.5,
        "geometry_calibration_negative_count": 2,
        "geometry_calibration_exceedance_count": 0,
        "calibration_negative_count": 10,
        "calibration_false_positive_count": 1,
        "calibration_false_positive_rate": TARGET_FPR,
        "target_fpr": TARGET_FPR,
        "threshold_digest": MAIN_THRESHOLD_DIGEST,
    }
    detection_records = PROPOSED_OBSERVATION_RECORDS
    detection_tables = build_detection_score_tables(detection_records, protocol)
    test_metrics = build_image_only_test_metric_rows(
        detection_records,
        TARGET_FPR,
    )
    ablation_rows = tuple(
        schema_row(
            ABLATION_METRIC_FIELDS,
            ablation_id=ablation_id,
            test_prompt_count=TEST_COUNT,
            clean_false_positive_rate=0.05,
            wrong_key_false_positive_rate=0.05,
            clean_true_positive_rate=0.9,
            attacked_true_positive_rate=0.8,
            attacked_false_positive_rate=0.05,
            positive_content_score_mean=0.7,
            paired_ssim_mean=0.95,
            frozen_threshold_digest=f"digest_{ablation_id}",
            metric_status="measured_full_runtime_rerun",
        )
        for ablation_id in FORMAL_RUNTIME_RERUN_ABLATION_IDS
    )
    delta_rows = tuple(
        schema_row(
            ABLATION_DELTA_FIELDS,
            ablation_id=ablation_id,
            clean_true_positive_rate_delta=-0.1,
            attacked_true_positive_rate_delta=-0.1,
            paired_ssim_delta=0.0,
            metric_status="measured_full_runtime_rerun",
        )
        for ablation_id in FORMAL_RUNTIME_RERUN_ABLATION_IDS
        if ablation_id != "complete_method"
    )
    return {
        ARTIFACT_SOURCE_PATHS["frozen_evidence_protocol_ready"]: (
            json.dumps(protocol, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ),
        ARTIFACT_SOURCE_PATHS["raw_image_only_detection_records_ready"]: (
            "".join(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                for row in detection_records
            ).encode("utf-8")
        ),
        ARTIFACT_SOURCE_PATHS["test_detection_metrics_ready"]: (
            csv_bytes_with_fields(test_metrics, tuple(sorted(TEST_METRIC_FIELDS)))
        ),
        ARTIFACT_SOURCE_PATHS["score_distribution_table_ready"]: (
            csv_bytes_with_fields(
                detection_tables["score_distribution_table"],
                SCORE_DISTRIBUTION_FIELDNAMES,
            )
        ),
        ARTIFACT_SOURCE_PATHS["roc_curve_points_ready"]: csv_bytes_with_fields(
            detection_tables["roc_curve_points"],
            CURVE_POINT_FIELDNAMES,
        ),
        ARTIFACT_SOURCE_PATHS["det_curve_points_ready"]: csv_bytes_with_fields(
            detection_tables["det_curve_points"],
            CURVE_POINT_FIELDNAMES,
        ),
        ARTIFACT_SOURCE_PATHS["attack_family_metrics_ready"]: csv_bytes(
            attack_family_metrics
        ),
        ARTIFACT_SOURCE_PATHS["baseline_comparison_table_ready"]: (
            csv_bytes_with_fields(
                baseline_rows,
                tuple(BASELINE_COMPARISON_FIELDS),
            )
        ),
        ARTIFACT_SOURCE_PATHS["mechanism_ablation_metrics_ready"]: (
            csv_bytes_with_fields(
                ablation_rows,
                tuple(ABLATION_METRIC_FIELDS),
            )
        ),
        ARTIFACT_SOURCE_PATHS["mechanism_pairwise_delta_ready"]: (
            csv_bytes_with_fields(
                delta_rows,
                tuple(ABLATION_DELTA_FIELDS),
            )
        ),
        ARTIFACT_SOURCE_PATHS["mechanism_necessity_statistics_ready"]: (
            csv_bytes_with_fields(
                necessity_rows,
                ABLATION_NECESSITY_FIELDNAMES,
            )
        ),
        ARTIFACT_SOURCE_PATHS["dataset_quality_metrics_ready"]: (
            csv_bytes_with_fields(
                quality_metrics,
                tuple(DATASET_QUALITY_FIELDS),
            )
        ),
    }


def official_reference_source_payloads() -> dict[str, bytes]:
    """构造三个官方参考方法的七类治理源文件字节."""

    payloads: dict[str, bytes] = {}
    for baseline_id in OFFICIAL_REFERENCE_BASELINE_IDS:
        source_root = f"outputs/{baseline_id}_official_reference/{SCALE}"
        role_to_name = {
            "summary": f"{baseline_id}_official_reference_summary.json",
            "run_manifest": "manifest.local.json",
            "records": f"{baseline_id}_official_reference_records.jsonl",
            "validation_report": (
                f"{baseline_id}_official_reference_validation_report.json"
            ),
            "package_input_manifest": (
                f"{baseline_id}_official_reference_package_input_manifest.json"
            ),
            "archive_summary": (
                f"{baseline_id}_official_reference_archive_summary.json"
            ),
            "archive_manifest": (
                f"{baseline_id}_official_reference_archive_manifest.local.json"
            ),
        }
        for source_role, file_name in role_to_name.items():
            payload = {
                "baseline_id": baseline_id,
                "paper_claim_scale": SCALE,
                "source_role": source_role,
            }
            payloads[f"{source_root}/{file_name}"] = (
                json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n"
            ).encode("utf-8")
    return payloads


def official_reference_record(
    baseline_id: str,
    source_payloads: dict[str, bytes],
) -> dict[str, object]:
    """构造来源路径与字节摘要均可独立核验的官方参考记录."""

    prefix = f"outputs/{baseline_id}_official_reference/{SCALE}/"
    source_paths = {
        "summary": prefix + f"{baseline_id}_official_reference_summary.json",
        "run_manifest": prefix + "manifest.local.json",
        "records": prefix + f"{baseline_id}_official_reference_records.jsonl",
        "validation_report": (
            prefix + f"{baseline_id}_official_reference_validation_report.json"
        ),
        "package_input_manifest": (
            prefix + f"{baseline_id}_official_reference_package_input_manifest.json"
        ),
        "archive_summary": (
            prefix + f"{baseline_id}_official_reference_archive_summary.json"
        ),
        "archive_manifest": (
            prefix
            + f"{baseline_id}_official_reference_archive_manifest.local.json"
        ),
    }
    source_digests = {
        role: hashlib.sha256(source_payloads[path]).hexdigest()
        for role, path in source_paths.items()
    }
    payload: dict[str, object] = {
        "baseline_id": baseline_id,
        "paper_claim_scale": SCALE,
        "target_fpr": TARGET_FPR,
        "supplemental_table_role": "supplemental_method_fidelity_reference",
        "run_decision": "pass",
        "official_reference_ready": True,
        "reference_import_ready": True,
        "governed_reference_record_count": TEST_COUNT,
        "records_nonempty_ready": True,
        "records_baseline_identity_ready": True,
        "validation_zero_rejection_ready": True,
        "run_manifest_ready": True,
        "package_input_exact_set_ready": True,
        "package_input_digests_ready": True,
        "package_governance_semantics_ready": True,
        "source_code_version_consistent_ready": True,
        "code_version": COMMON_CODE_VERSION,
        "declared_package_entry_count": len(source_paths),
        "official_reference_package_entry_digest": build_stable_digest(source_paths),
        "official_reference_source_paths": source_paths,
        "official_reference_source_artifact_digests": source_digests,
        "main_table_eligible": False,
        "supports_main_table_superiority_claim": False,
        "supplemental_method_fidelity_evidence_ready": True,
        "official_reference_fidelity_evidence_ready": True,
        "supports_paper_claim": False,
    }
    digest = build_stable_digest(payload)
    return {
        "official_reference_fidelity_record_id": (
            f"{baseline_id}_official_reference_fidelity_{digest[:16]}"
        ),
        "official_reference_fidelity_record_digest": digest,
        **payload,
    }


def paired_superiority_evidence(
    threshold_report: dict[str, object],
    threshold_rows: tuple[dict[str, object], ...],
) -> tuple[
    tuple[dict[str, object], ...],
    tuple[dict[str, object], ...],
    dict[str, object],
]:
    """构造4方法 x 规范 test Prompt x 完整正式攻击的配对证据."""

    canonical_threshold_rows = canonical_threshold_audit_rows(threshold_rows)
    protocol_digest = build_paired_superiority_protocol_digest(
        threshold_report,
        canonical_threshold_rows,
        build_stable_digest(
            build_fixed_fpr_threshold_manifest_config(threshold_report)
        ),
    )
    materialized_outcomes = tuple(
        outcome
        for baseline_id in PRIMARY_BASELINE_IDS
        for outcome in build_paired_outcomes(
            METHOD_OBSERVATION_RECORDS_BY_METHOD["slm_wm"],
            METHOD_OBSERVATION_RECORDS_BY_METHOD[baseline_id],
            baseline_id=baseline_id,
            proposed_method_threshold_digest=MAIN_THRESHOLD_DIGEST,
            baseline_method_threshold_digest=METHOD_THRESHOLD_DIGEST_MAP[
                baseline_id
            ],
            attack_registry_rows=ATTACK_REGISTRY,
            include_quality_matching=True,
        )
    )
    all_sample_rows = tuple(
        build_paired_superiority_rows(
            materialized_outcomes,
            protocol_digest=protocol_digest,
        )
    )
    quality_rows = tuple(
        build_quality_matched_superiority_rows(
            materialized_outcomes,
            protocol_digest=protocol_digest,
        )
    )
    rows = tuple(
        merge_paired_and_quality_matched_rows(
            all_sample_rows,
            quality_rows,
        )
    )
    statistical_summary = build_paired_superiority_summary(
        rows,
        paired_outcomes=materialized_outcomes,
    )
    quality_summary = build_quality_matched_superiority_summary(quality_rows)
    summary = {
        "paper_claim_scale": SCALE,
        "target_fpr": TARGET_FPR,
        "expected_test_count": TEST_COUNT,
        "expected_attack_count": len(ATTACK_REGISTRY),
        "paired_prompt_counts": [TEST_COUNT],
        "paired_attack_counts": [len(ATTACK_REGISTRY)],
        "paired_outcome_count": len(materialized_outcomes),
        "paired_outcome_set_digest": build_stable_digest(materialized_outcomes),
        "paired_superiority_protocol_digest": protocol_digest,
        "claim_p_value_method": CLAIM_P_VALUE_METHOD,
        "sharp_null_diagnostic_method": SHARP_NULL_DIAGNOSTIC_METHOD,
        "bootstrap_analysis_schema": BOOTSTRAP_ANALYSIS_SCHEMA,
        "bootstrap_bit_generator": BOOTSTRAP_BIT_GENERATOR,
        "bootstrap_quantile_method": BOOTSTRAP_QUANTILE_METHOD,
        "bootstrap_resample_count": DEFAULT_BOOTSTRAP_RESAMPLE_COUNT,
        "confidence_level": DEFAULT_CONFIDENCE_LEVEL,
        "method_threshold_digest_map": {
            "slm_wm": MAIN_THRESHOLD_DIGEST,
            **{
                baseline_id: METHOD_THRESHOLD_DIGEST_MAP[baseline_id]
                for baseline_id in PRIMARY_BASELINE_IDS
            },
        },
        "method_observation_source_sha256_map": (
            METHOD_OBSERVATION_SOURCE_SHA256_MAP
        ),
        "method_observation_source_path_map": METHOD_OBSERVATION_SOURCE_PATH_MAP,
        "threshold_audit_rows_digest": build_stable_digest(
            list(canonical_threshold_rows)
        ),
        "paired_attack_registry_digest": build_stable_digest(
            list(ATTACK_REGISTRY)
        ),
        "paired_superiority_scale_ready": True,
        **statistical_summary,
        **quality_summary,
        "overall_paired_superiority_ready": True,
        "supports_paper_claim": True,
    }
    return materialized_outcomes, rows, summary

def formal_result_record(
    method_id: str,
    attack: dict[str, str],
) -> dict[str, object]:
    """构造可由同一批原始 observation 完整重算的正式结果记录."""

    is_main = method_id == "slm_wm_current"
    metric_values = {
        "positive_count": TEST_COUNT,
        "negative_count": TEST_COUNT,
        "attacked_negative_count": TEST_COUNT,
        "attack_record_count": TEST_COUNT * 2,
        "supported_record_count": TEST_COUNT,
        "true_positive_rate": 1.0 if is_main else 5 / TEST_COUNT,
        "false_positive_rate": 0.0,
        "clean_false_positive_rate": 0.0,
        "attacked_false_positive_rate": 0.0,
        "quality_score_mean": 0.9,
        "score_retention_mean": (
            measured_score_retention(1.0, 0.8) if is_main else 0.0
        ),
    }
    ci_fields = {
        "true_positive_rate": (
            TEST_COUNT,
            "true_positive_rate_ci_low",
            "true_positive_rate_ci_high",
        ),
        "false_positive_rate": (
            TEST_COUNT,
            "false_positive_rate_ci_low",
            "false_positive_rate_ci_high",
        ),
        "clean_false_positive_rate": (
            TEST_COUNT,
            "clean_false_positive_rate_ci_low",
            "clean_false_positive_rate_ci_high",
        ),
        "attacked_false_positive_rate": (
            TEST_COUNT,
            "attacked_false_positive_rate_ci_low",
            "attacked_false_positive_rate_ci_high",
        ),
        "quality_score_mean": (
            TEST_COUNT,
            "quality_score_ci_low",
            "quality_score_ci_high",
        ),
        "score_retention_mean": (
            TEST_COUNT,
            "score_retention_ci_low",
            "score_retention_ci_high",
        ),
    }
    for metric_name, (count, low_name, high_name) in ci_fields.items():
        lower_bound, upper_bound = (
            PILOT_PAPER_METRIC_BOUNDS["quality_score_mean"]
            if metric_name == "quality_score_mean"
            else (0.0, 1.0)
        )
        if metric_name != "quality_score_mean":
            metric_values[metric_name] = clamp_unit_interval(
                float(metric_values[metric_name])
            )
        low, high = bounded_hoeffding_confidence_interval(
            float(metric_values[metric_name]),
            count,
            0.95,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        )
        metric_values[low_name] = low
        metric_values[high_name] = high

    source_method_id = "slm_wm" if is_main else method_id
    source_path = METHOD_OBSERVATION_SOURCE_PATH_MAP[source_method_id]
    source_digest = METHOD_OBSERVATION_SOURCE_SHA256_MAP[source_method_id]

    payload: dict[str, object] = {
        "paper_claim_scale": SCALE,
        "target_fpr": TARGET_FPR,
        "result_protocol_name": PAPER_CONFIG.result_protocol_name,
        "result_scope": PAPER_CONFIG.result_scope,
        "result_claim_scope": PAPER_CONFIG.result_claim_scope,
        "method_id": method_id,
        "attack_id": attack["attack_id"],
        "attack_family": attack["attack_family"],
        "attack_name": attack["attack_name"],
        "resource_profile": attack["resource_profile"],
        "attack_config_digest": attack["attack_config_digest"],
        "prompt_protocol_name": PAPER_CONFIG.prompt_protocol_name,
        "prompt_split_digest": PROMPT_SPLIT_DIGEST,
        "attack_matrix_digest": ATTACK_MATRIX_DIGEST,
        "fixed_fpr_protocol_digest": FIXED_FPR_PROTOCOL_DIGEST,
        "method_threshold_digest": METHOD_THRESHOLD_DIGEST_MAP[method_id],
        "confidence_interval_method": "bounded_hoeffding",
        "confidence_level": 0.95,
        "baseline_result_source": source_path,
        "baseline_result_source_digest": source_digest,
        "evidence_paths": [source_path],
        "metric_status": (
            "measured_image_only_detection_formal_protocol"
            if is_main
            else "measured"
        ),
        "result_source_kind": (
            "slm_wm_image_only_dataset_runtime"
            if is_main
            else "external_baseline_result"
        ),
        **metric_values,
        "strict_formal_result_ready": True,
        "supports_paper_claim": True,
    }
    digest = build_stable_digest(payload)
    payload["pilot_paper_result_record_digest"] = digest
    payload["pilot_paper_result_record_id"] = f"pilot_paper_result_record_{digest[:16]}"
    return payload


def formal_attack_detection_records() -> tuple[dict[str, object], ...]:
    """构造直接携带正式攻击身份和记录摘要的真实攻击记录."""

    rows = []
    for prompt_id in TEST_PROMPT_IDS:
        for attack in ATTACK_REGISTRY:
            config = FORMAL_ATTACK_CONFIG_BY_ID[attack["attack_id"]]
            for sample_role in ("positive_source", "clean_negative"):
                source_digest = build_stable_digest(
                    ["source", prompt_id, sample_role]
                )
                attacked_digest = build_stable_digest(
                    ["attacked", prompt_id, sample_role, attack["attack_id"]]
                )
                detector_digest = build_stable_digest(
                    ["detector", prompt_id, sample_role, attack["attack_id"]]
                )
                record = {
                    "run_id": f"run_{prompt_id}",
                    "prompt_id": prompt_id,
                    "split": "test",
                    "sample_role": sample_role,
                    **attack,
                    "attack_parameters": config.attack_parameters,
                    "attack_strength": config.attack_strength,
                    "requires_gpu": config.requires_gpu,
                    "attack_performed": True,
                    "source_image_path": (
                        f"outputs/test_images/{prompt_id}_{sample_role}.png"
                    ),
                    "source_image_digest": source_digest,
                    "attacked_image_path": (
                        "outputs/test_images/"
                        f"{prompt_id}_{sample_role}_{attack['attack_id']}.png"
                    ),
                    "attacked_image_digest": attacked_digest,
                    "detector_digest": detector_digest,
                    "frozen_threshold_digest": MAIN_THRESHOLD_DIGEST,
                    "formal_evidence_positive": (
                        sample_role == "positive_source"
                    ),
                    "quality_score": 0.9,
                    "quality_ssim": 0.9,
                    "quality_psnr": 35.0,
                    "score_retention": 0.8,
                    "lf_score_retention": 0.8,
                    "tail_score_retention": 0.8,
                    "geometry_reliable": False,
                    "formal_rescue_applied": False,
                    "evidence_decision": sample_role == "positive_source",
                    "formal_metric_status": "measured_image_only_detection",
                    "metric_status": (
                        "measured_real_attacked_image_image_only_detection"
                    ),
                    "attacked_image_available": True,
                    "supports_paper_claim": True,
                }
                digest = build_attack_record_digest(record)
                rows.append(
                    {
                        **record,
                        "attack_record_digest": digest,
                        "attack_record_id": f"attack_{digest[:24]}",
                    }
                )
    return tuple(rows)


def formal_attacked_image_registry(
    records: tuple[dict[str, object], ...],
) -> tuple[dict[str, object], ...]:
    """按正式 producer 字段投影攻击图像 registry."""

    fields = (
        "attack_record_id",
        "run_id",
        "prompt_id",
        "split",
        "sample_role",
        "attack_id",
        "attack_family",
        "attack_name",
        "resource_profile",
        "source_image_path",
        "source_image_digest",
        "attacked_image_path",
        "attacked_image_digest",
        "attack_config_digest",
        "metric_status",
        "supports_paper_claim",
    )
    return tuple(
        {field_name: record[field_name] for field_name in fields}
        for record in records
    )


def primary_evidence_record(baseline_id: str) -> dict[str, object]:
    """构造可独立复算身份摘要的 primary baseline 证据记录."""

    numerical_fidelity_mode = (
        "native_official_result_exact_rebuild"
        if baseline_id == "t2smark"
        else (
            "official_source_bound_rfc8439_and_operator_equivalence"
            if baseline_id == "gaussian_shading"
            else "executed_official_commit_operator_equivalence"
        )
    )
    numerical_fidelity_report_digest = build_stable_digest(
        {"baseline_id": baseline_id, "mode": numerical_fidelity_mode}
    )
    payload: dict[str, object] = {
        "baseline_id": baseline_id,
        "source_status": "downloaded",
        "official_repository_commit": "a" * 40,
        "adapter_status": "sd35_native_result_adapter_ready"
        if baseline_id == "t2smark"
        else "method_faithful_sd35_adapter_available",
        "adapter_run_ready": True,
        "adapter_run_observation_count": PROMPT_COUNT,
        "method_faithful_adapter_ready": True,
        "numerical_fidelity_mode": numerical_fidelity_mode,
        "numerical_fidelity_report_digest": numerical_fidelity_report_digest,
        "baseline_numerical_fidelity_ready": True,
        "blocking_reasons": (),
    }
    digest = build_stable_digest(payload)
    return {
        "primary_baseline_evidence_id": f"primary_baseline_evidence_{digest[:16]}",
        "primary_baseline_evidence_digest": digest,
        "baseline_id": baseline_id,
        "comparison_group": "primary",
        "source_status": "downloaded",
        "source_dir": f"external_baseline/primary/{baseline_id}/source",
        "official_repository_commit": "a" * 40,
        "adapter_status": payload["adapter_status"],
        "model_alignment_status": "sd35_medium_formal",
        "adapter_run_ready": True,
        "adapter_run_observation_count": PROMPT_COUNT,
        "adapter_run_execution_devices": ["cuda"],
        "adapter_run_sample_roles": ["clean_negative", "positive_source"],
        "adapter_run_latent_shapes": [[1, 16, 64, 64]],
        "method_faithful_adapter_ready": True,
        "numerical_fidelity_mode": numerical_fidelity_mode,
        "numerical_fidelity_report_path": (
            f"outputs/formal/{baseline_id}/numerical_fidelity_report.json"
        ),
        "numerical_fidelity_report_digest": numerical_fidelity_report_digest,
        "baseline_numerical_fidelity_ready": True,
        "paper_run_prompt_protocol_ready": True,
        "fixed_fpr_baseline_calibration_ready": True,
        "attack_matrix_baseline_detection_ready": True,
        "formal_evidence_paths": [f"outputs/formal/{baseline_id}/observations.jsonl"],
        "formal_evidence_paths_ready": True,
        "formal_result_ready": True,
        "blocking_reasons": [],
        "supports_paper_claim": False,
    }


def closure_input_lock() -> tuple[dict[str, object], dict[str, object]]:
    """构造精确10类输入包的当前 run 锁及独立 manifest."""

    return build_test_closure_input_lock_payloads(
        paper_run_name=SCALE,
        target_fpr=TARGET_FPR,
        common_code_version=COMMON_CODE_VERSION,
    )


def _raw_ablation_detection(
    *,
    run_id: str,
    prompt_id: str,
    split: str,
    sample_role: str,
    content_score: float,
    attack: object | None = None,
) -> dict[str, object]:
    """构造应用消融冻结协议前的最小图像盲检原子。"""

    record: dict[str, object] = {
        "run_id": run_id,
        "prompt_id": prompt_id,
        "split": split,
        "sample_role": sample_role,
        "content_score": content_score,
        "aligned_content_score": None,
        "attention_geometry_score": 0.0,
        "registration_confidence": 0.0,
        "attention_sync_score": 0.0,
        "alignment": {"registration_geometry_reliable": False},
    }
    if attack is not None:
        record.update(
            {
                "attack_id": attack.attack_id,
                "attack_family": attack.attack_family,
                "attack_name": attack.attack_name,
                "resource_profile": attack.resource_profile,
                "attack_config_digest": attack_config_digest(attack),
                "attack_parameters": attack.attack_parameters,
                "attack_performed": True,
            }
        )
    return record


def ablation_atomic_records() -> tuple[
    tuple[dict[str, object], ...],
    tuple[dict[str, object], ...],
    dict[str, dict[str, object]],
]:
    """构造15项消融、70个 Prompt 和正式攻击集合的完整原子链."""

    runtime_records: list[dict[str, object]] = []
    detection_records: list[dict[str, object]] = []
    protocols: dict[str, dict[str, object]] = {}
    for spec in FORMAL_RUNTIME_RERUN_ABLATION_SPECS:
        calibration_negatives = tuple(
            _raw_ablation_detection(
                run_id=f"ablation_{spec.ablation_id}_{prompt_id}",
                prompt_id=prompt_id,
                split="calibration",
                sample_role="clean_negative",
                content_score=0.0,
            )
            for prompt_id in CALIBRATION_PROMPT_IDS
        )
        protocol = calibrate_complete_evidence_protocol(
            calibration_negatives,
            target_fpr=TARGET_FPR,
            rescue_margin_low=-0.05,
        )
        protocols[spec.ablation_id] = protocol.to_dict()
        runtime_config = spec.to_dict()
        for prompt_record in PROMPT_RECORDS:
            prompt_id = prompt_record.prompt_id
            split = PROMPT_SPLIT_BY_ID[prompt_id]
            scientific_config = {
                "prompt": prompt_record.prompt_text,
                "prompt_id": prompt_id,
                "split": split,
                "method_definition": (
                    semantic_conditioned_latent_method_definition()
                ),
                "method_definition_digest": (
                    semantic_conditioned_latent_method_definition_digest()
                ),
                "output_dir": (
                    f"outputs/formal_mechanism_ablation/{SCALE}/"
                    f"runs/{spec.ablation_id}"
                ),
                **{
                    field_name: field_value
                    for field_name, field_value in runtime_config.items()
                    if field_name != "ablation_id"
                },
            }
            scientific_config_digest = build_stable_digest(scientific_config)
            run_id = f"semantic_watermark_{scientific_config_digest[:16]}"
            positive_score = 1.0 if spec.ablation_id == "complete_method" else 0.0
            raw_detections = [
                _raw_ablation_detection(
                    run_id=run_id,
                    prompt_id=prompt_id,
                    split=split,
                    sample_role="clean_negative",
                    content_score=0.0,
                ),
                _raw_ablation_detection(
                    run_id=run_id,
                    prompt_id=prompt_id,
                    split=split,
                    sample_role="positive_source",
                    content_score=positive_score,
                ),
                _raw_ablation_detection(
                    run_id=run_id,
                    prompt_id=prompt_id,
                    split=split,
                    sample_role="wrong_key_negative",
                    content_score=0.0,
                ),
            ]
            if split == "test":
                raw_detections.extend(
                    _raw_ablation_detection(
                        run_id=run_id,
                        prompt_id=prompt_id,
                        split=split,
                        sample_role=sample_role,
                        content_score=(
                            positive_score
                            if sample_role == "positive_source"
                            else 0.0
                        ),
                        attack=attack,
                    )
                    for attack in FORMAL_ATTACK_CONFIGS
                    for sample_role in ("clean_negative", "positive_source")
                )
            detections = apply_frozen_evidence_protocol(
                raw_detections,
                protocol,
            )
            detection_records.extend(
                {
                    **record,
                    "ablation_id": spec.ablation_id,
                    "ablation_prompt_id": prompt_id,
                }
                for record in detections
            )
            un_attacked = {
                str(record["sample_role"]): record
                for record in detections
                if not record.get("attack_id")
            }
            attacked_positive = tuple(
                record
                for record in detections
                if record.get("attack_id")
                and record.get("sample_role") == "positive_source"
            )
            attacked_negative = tuple(
                record
                for record in detections
                if record.get("attack_id")
                and record.get("sample_role") == "clean_negative"
            )
            runtime_records.append(
                {
                    "prompt_index": prompt_record.prompt_index,
                    "prompt_id": prompt_id,
                    "prompt_digest": PROMPT_DIGEST_BY_ID[prompt_id],
                    "split": split,
                    "ablation_id": spec.ablation_id,
                    "runtime_config": runtime_config,
                    "runtime_result": {
                        "run_id": run_id,
                        "run_decision": "pass",
                        "metadata": {
                            "scientific_unit_config": scientific_config,
                            "scientific_unit_provenance": (
                                build_test_scientific_unit_provenance(
                                    run_id,
                                    scientific_config_digest,
                                )
                            ),
                            "paired_quality": {"ssim": 0.95},
                        },
                    },
                    "generation_rerun": True,
                    "attack_and_detection_rerun": split == "test",
                    "threshold_calibration_scope": (
                        "per_ablation_calibration_split"
                    ),
                    "frozen_content_threshold": protocol.content_threshold,
                    "frozen_threshold_digest": protocol.threshold_digest,
                    "clean_negative_positive": bool(
                        un_attacked["clean_negative"][
                            "formal_evidence_positive"
                        ]
                    ),
                    "positive_source_positive": bool(
                        un_attacked["positive_source"][
                            "formal_evidence_positive"
                        ]
                    ),
                    "wrong_key_negative_positive": bool(
                        un_attacked["wrong_key_negative"][
                            "formal_evidence_positive"
                        ]
                    ),
                    "clean_negative_content_score": float(
                        un_attacked["clean_negative"]["content_score"]
                    ),
                    "positive_source_content_score": float(
                        un_attacked["positive_source"]["content_score"]
                    ),
                    "attacked_positive_count": len(attacked_positive),
                    "attacked_positive_rate": (
                        sum(
                            bool(record["formal_evidence_positive"])
                            for record in attacked_positive
                        )
                        / len(attacked_positive)
                        if attacked_positive
                        else 0.0
                    ),
                    "attacked_negative_count": len(attacked_negative),
                    "attacked_negative_rate": (
                        sum(
                            bool(record["formal_evidence_positive"])
                            for record in attacked_negative
                        )
                        / len(attacked_negative)
                        if attacked_negative
                        else 0.0
                    ),
                    "formal_attack_coverage_ready": True,
                    "paired_ssim": 0.95,
                }
            )
    return tuple(runtime_records), tuple(detection_records), protocols


@lru_cache(maxsize=1)
def _ready_bundle_template() -> ResultClosureGateInput:
    """构造证据完整且允许至少一个逐攻击比较未显著胜出的输入包。"""

    necessity_variant_ids = tuple(
        ablation_id
        for ablation_id in FORMAL_RUNTIME_RERUN_ABLATION_IDS
        if ablation_id != "complete_method"
    )
    (
        necessity_records,
        ablation_detection_records,
        ablation_frozen_protocols,
    ) = ablation_atomic_records()
    ablation_detection_records_bytes = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
        for record in ablation_detection_records
    ).encode("utf-8")
    ablation_frozen_protocols_bytes = json.dumps(
        ablation_frozen_protocols,
        ensure_ascii=False,
        sort_keys=True,
    ).encode("utf-8")
    ablation_atom_identity = {
        "formal_detection_records_sha256": hashlib.sha256(
            ablation_detection_records_bytes
        ).hexdigest(),
        "formal_detection_records_digest": build_stable_digest(
            ablation_detection_records
        ),
        "per_ablation_frozen_protocols_sha256": hashlib.sha256(
            ablation_frozen_protocols_bytes
        ).hexdigest(),
        "per_ablation_frozen_protocols_digest": build_stable_digest(
            ablation_frozen_protocols
        ),
    }
    necessity_rows, necessity_summary = build_ablation_necessity_statistics(
        necessity_records,
        expected_ablation_ids=necessity_variant_ids,
        expected_paired_prompt_count=TEST_COUNT,
    )

    result_records = tuple(
        formal_result_record(method_id, attack)
        for method_id in METHOD_THRESHOLD_DIGEST_MAP
        for attack in ATTACK_REGISTRY
    )
    result_analysis_governed_path_map = (
        build_governed_paper_payload_path_map(SCALE)
    )
    result_analysis_baseline_rows = tuple(
        build_main_comparison_rows_from_result_records(result_records)
    )
    result_analysis_confidence_rows = tuple(
        build_confidence_interval_rows(result_records)
    )
    result_analysis_per_attack_rows = tuple(
        build_per_attack_superiority_rows(result_records)
    )
    result_record_set_digest = build_pilot_paper_result_record_set_digest(result_records)
    primary_evidence_records = tuple(
        primary_evidence_record(baseline_id)
        for baseline_id in ("tree_ring", "gaussian_shading", "shallow_diffuse", "t2smark")
    )
    primary_evidence_records_digest = build_stable_digest(
        sorted(primary_evidence_records, key=lambda row: str(row["baseline_id"]))
    )
    lock_payload, lock_manifest = closure_input_lock()
    lock_digest = str(lock_payload["closure_input_lock_digest"])
    official_source_payload_map = official_reference_source_payloads()
    official_records = tuple(
        official_reference_record(baseline_id, official_source_payload_map)
        for baseline_id in OFFICIAL_REFERENCE_BASELINE_IDS
    )
    official_records_digest = build_stable_digest(list(official_records))
    official_summary = {
        "paper_claim_scale": SCALE,
        "target_fpr": TARGET_FPR,
        "expected_official_reference_baseline_ids": list(
            OFFICIAL_REFERENCE_BASELINE_IDS
        ),
        "actual_official_reference_baseline_ids": list(
            OFFICIAL_REFERENCE_BASELINE_IDS
        ),
        "missing_official_reference_baseline_ids": [],
        "unexpected_official_reference_baseline_ids": [],
        "duplicate_official_reference_baseline_ids": [],
        "official_reference_exact_set_ready": True,
        "official_reference_fidelity_record_count": len(official_records),
        "official_reference_fidelity_ready_count": len(official_records),
        "common_code_version": COMMON_CODE_VERSION,
        "common_code_version_ready": True,
        "official_reference_fidelity_evidence_digest": official_records_digest,
        "main_table_eligible": False,
        "supports_main_table_superiority_claim": False,
        "supplemental_method_fidelity_evidence_ready": True,
        "official_reference_fidelity_evidence_ready": True,
        "supports_paper_claim": False,
    }
    attack_detection_records = formal_attack_detection_records()
    result_analysis_failure_rows = tuple(
        build_failure_case_records(
            attack_detection_records,
            limit=12,
        )
    )
    result_analysis_failure_figure_path = (
        RESULT_ANALYSIS_PAYLOAD_PATH_MAP["failure_case_figure"]
    )
    result_analysis_failure_svg_text = build_failure_case_svg_text(
        result_analysis_failure_rows,
        failure_figure_path=result_analysis_failure_figure_path,
    )
    result_analysis_payload_bytes = {
        RESULT_ANALYSIS_PAYLOAD_PATH_MAP["main_confidence_interval_table"]: (
            csv_bytes(result_analysis_confidence_rows)
        ),
        RESULT_ANALYSIS_PAYLOAD_PATH_MAP["per_attack_superiority_table"]: (
            csv_bytes(result_analysis_per_attack_rows)
        ),
        RESULT_ANALYSIS_PAYLOAD_PATH_MAP["failure_case_records"]: (
            "".join(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                for row in result_analysis_failure_rows
            ).encode("utf-8")
        ),
        result_analysis_failure_figure_path: (
            result_analysis_failure_svg_text.encode("utf-8")
        ),
    }
    result_analysis_payload_sha256_map = {
        role: hashlib.sha256(
            result_analysis_payload_bytes[RESULT_ANALYSIS_PAYLOAD_PATH_MAP[role]]
        ).hexdigest()
        for role in RESULT_ANALYSIS_PAYLOAD_PATH_MAP
    }
    result_analysis_payload_digest = build_stable_digest(
        {
            "result_analysis_payload_path_map": (
                RESULT_ANALYSIS_PAYLOAD_PATH_MAP
            ),
            "result_analysis_payload_sha256_map": (
                result_analysis_payload_sha256_map
            ),
        }
    )
    result_analysis_semantic_evidence = (
        rebuild_and_validate_result_analysis_derived_payload(
            result_records=result_records,
            attack_detection_records=attack_detection_records,
            confidence_interval_rows=result_analysis_confidence_rows,
            per_attack_superiority_rows=result_analysis_per_attack_rows,
            failure_case_rows=result_analysis_failure_rows,
            failure_case_svg_text=result_analysis_failure_svg_text,
            failure_figure_path=result_analysis_failure_figure_path,
            failure_case_limit=12,
        )
    )
    attack_family_metrics = build_attack_family_metrics(
        attack_detection_records,
        TARGET_FPR,
        True,
    )
    attacked_image_registry = formal_attacked_image_registry(
        attack_detection_records
    )
    attack_record_count = len(ATTACK_REGISTRY) * 2 * TEST_COUNT
    required_gpu_attack_count = sum(
        config.requires_gpu for config in FORMAL_ATTACK_CONFIGS
    )
    attack_report = {
        "paper_run_name": SCALE,
        "input_records_path": METHOD_OBSERVATION_SOURCE_PATH_MAP["slm_wm"],
        "image_attack_evidence_records_path": (
            f"outputs/image_attack_evidence/{SCALE}/"
            "formal_attack_detection_records.jsonl"
        ),
        "evaluation_boundary": {"target_fpr": TARGET_FPR, "threshold_digest": MAIN_THRESHOLD_DIGEST},
        "attack_record_count": attack_record_count,
        "performed_attack_record_count": attack_record_count,
        "formal_real_attack_record_count": attack_record_count,
        "formal_image_attack_record_count": attack_record_count,
        "real_attacked_image_count": attack_record_count,
        "attack_config_count": len(FORMAL_ATTACK_CONFIGS),
        "attack_family_count": len(
            {config.attack_family for config in FORMAL_ATTACK_CONFIGS}
        ),
        "resource_profiles": sorted(
            {config.resource_profile for config in FORMAL_ATTACK_CONFIGS}
        ),
        "expected_attack_ids": sorted(row["attack_id"] for row in ATTACK_REGISTRY),
        "actual_attack_ids": sorted(row["attack_id"] for row in ATTACK_REGISTRY),
        "missing_attack_ids": [],
        "unexpected_attack_ids": [],
        "expected_attack_split_role_count": TEST_COUNT,
        "attack_split_role_counts": {
            f"{attack['attack_id']}|{sample_role}": TEST_COUNT
            for attack in ATTACK_REGISTRY
            for sample_role in ("positive_source", "clean_negative")
        },
        "required_real_gpu_attack_count": required_gpu_attack_count,
        "measured_real_gpu_attack_count": required_gpu_attack_count,
        "gpu_attack_real_measurement_missing_count": 0,
        "real_attacked_image_closed_loop_ready": True,
        "formal_attack_detection_ready": True,
        "detector_input_access_mode": "image_key_public_model_only",
        "generation_latent_trace_required": False,
        "attack_metrics_ready": True,
        "attack_record_coverage_ready": True,
        "real_gpu_attack_validation_ready": True,
        "full_method_claim_ready": True,
        "supports_paper_claim": True,
    }
    attack_metadata = {
        "protocol_decision": "pass",
        "full_method_claim_ready": True,
        "supports_paper_claim": True,
    }
    threshold_rows = tuple(
        {
            "method_id": method_id,
            "threshold_source": "calibration_clean_negative",
            "target_fpr": TARGET_FPR,
            "calibration_clean_negative_count": len(CALIBRATION_PROMPT_IDS),
            "test_clean_negative_count": TEST_COUNT,
            "calibrated_detection_threshold": 0.5,
            "threshold_digest": METHOD_THRESHOLD_DIGEST_MAP[
                "slm_wm_current" if method_id == "slm_wm" else method_id
            ],
            "observation_source_sha256": (
                METHOD_OBSERVATION_SOURCE_SHA256_MAP[method_id]
            ),
            "protocol_target_ready": True,
            "protocol_value_ready": True,
            "detection_decision_ready": True,
            "split_count_ready": True,
            "fixed_fpr_threshold_ready": True,
            "supports_paper_claim": False,
        }
        for index, method_id in enumerate(
            ("slm_wm", "tree_ring", "gaussian_shading", "shallow_diffuse", "t2smark")
        )
    )
    canonical_threshold_rows = canonical_threshold_audit_rows(threshold_rows)
    threshold_method_digest_map = {
        "slm_wm": MAIN_THRESHOLD_DIGEST,
        **{
            baseline_id: METHOD_THRESHOLD_DIGEST_MAP[baseline_id]
            for baseline_id in PRIMARY_BASELINE_IDS
        },
    }
    threshold_report = {
        "paper_claim_scale": SCALE,
        "target_fpr": TARGET_FPR,
        "expected_method_ids": [
            "slm_wm",
            "tree_ring",
            "gaussian_shading",
            "shallow_diffuse",
            "t2smark",
        ],
        "audited_method_ids": [
            "slm_wm",
            "tree_ring",
            "gaussian_shading",
            "shallow_diffuse",
            "t2smark",
        ],
        "audited_method_count": 5,
        "method_identity_ready": True,
        "all_method_thresholds_ready": True,
        "method_threshold_digest_map": threshold_method_digest_map,
        "method_observation_source_sha256_map": (
            METHOD_OBSERVATION_SOURCE_SHA256_MAP
        ),
        "threshold_audit_rows_digest": build_stable_digest(
            list(canonical_threshold_rows)
        ),
        "threshold_observation_binding_ready": True,
        "fixed_fpr_threshold_audit_ready": True,
        "supports_paper_claim": True,
    }
    paired_outcomes, paired_rows, paired_summary = paired_superiority_evidence(
        threshold_report,
        threshold_rows,
    )
    primary_evidence_summary = {
        "paper_claim_scale": SCALE,
        "target_fpr": TARGET_FPR,
        "primary_baseline_count": 4,
        "adapter_run_ready_count": 4,
        "adapter_run_ready_ids": ["tree_ring", "gaussian_shading", "shallow_diffuse", "t2smark"],
        "formal_result_ready_count": 4,
        "formal_result_ready_ids": ["tree_ring", "gaussian_shading", "shallow_diffuse", "t2smark"],
        "numerical_fidelity_ready_count": 4,
        "numerical_fidelity_ready_ids": [
            "tree_ring",
            "gaussian_shading",
            "shallow_diffuse",
            "t2smark",
        ],
        "primary_baseline_numerical_fidelity_ready": True,
        "input_baseline_ids": ["tree_ring", "gaussian_shading", "shallow_diffuse", "t2smark"],
        "primary_baseline_formal_ready": True,
        "blocking_reasons": [],
        "input_observation_count": PROMPT_COUNT * 4,
        "input_command_result_count": 4,
        "t2smark_formal_evidence_digest": "9" * 64,
        "primary_baseline_evidence_records_digest": primary_evidence_records_digest,
        "supports_paper_claim": False,
    }
    baseline_report = {
        "paper_claim_scale": SCALE,
        "target_fpr": TARGET_FPR,
        "accepted_formal_import_count": 5,
        "rejected_formal_import_count": 0,
        "formal_import_issue_count": 0,
        "missing_candidate_template_count": 0,
        "missing_formal_template_count": 0,
        "unexpected_candidate_record_count": 0,
        "unexpected_accepted_record_count": 0,
        "duplicate_candidate_template_count": 0,
        "duplicate_accepted_template_count": 0,
        "missing_formal_evidence_collection_task_count": 0,
        "missing_formal_evidence_path_count": 0,
        "baseline_results_ready": False,
        "comparison_protocol_ready": True,
        "comparison_table_supports_paper_claim": True,
        "primary_baseline_formal_ready": True,
        "primary_baseline_results_ready": True,
        "primary_baseline_formal_template_coverage_ready": True,
        "primary_baseline_formal_evidence_collection_ready": True,
        "formal_import_validation_ready": True,
        "formal_evidence_path_resolution_ready": True,
        "baseline_source_registry_ready": True,
        "supports_paper_claim": True,
    }
    record_summary = {
        "paper_claim_scale": SCALE,
        "pilot_paper_result_record_count": len(result_records),
        "pilot_paper_template_record_count": len(result_records),
        "pilot_paper_template_covered_count": len(result_records),
        "pilot_paper_template_missing_count": 0,
        "accepted_pilot_paper_import_count": len(result_records),
        "accepted_pilot_paper_claim_record_count": len(result_records),
        "pilot_paper_template_coverage_ready": True,
        "pilot_paper_result_import_ready": True,
        "pilot_paper_claim_record_ready": True,
        "result_record_set_digest": result_record_set_digest,
        "method_threshold_digest_map": METHOD_THRESHOLD_DIGEST_MAP,
        "closure_input_lock_digest": lock_digest,
        "common_code_version": COMMON_CODE_VERSION,
        "require_existing_evidence": False,
        "supports_paper_claim": True,
    }
    common_summary = {
        "paper_claim_scale": SCALE,
        "paper_target_fpr": TARGET_FPR,
        "expected_target_fpr": TARGET_FPR,
        "paper_prompt_count": PROMPT_COUNT,
        "pilot_paper_import_template_count": len(result_records),
        "accepted_pilot_paper_import_count": len(result_records),
        "accepted_pilot_paper_claim_record_count": len(result_records),
        "pilot_paper_negative_count_minimum_required": TEST_COUNT,
        "minimum_result_positive_count": TEST_COUNT,
        "minimum_result_negative_count": TEST_COUNT,
        "minimum_result_attacked_negative_count": TEST_COUNT,
        "paper_run_result_missing_template_count": 0,
        "paper_run_result_unexpected_template_count": 0,
        "paper_run_result_duplicate_template_count": 0,
        "paper_run_allows_paper_claim": True,
        "strict_formal_evidence_required": True,
        "pilot_paper_common_protocol_ready": True,
        "paper_run_workflow_validation_ready": True,
        "pilot_paper_prompt_split_ready": True,
        "paper_prompt_split_ready": True,
        "calibration_prompt_id_digest": CALIBRATION_PROMPT_ID_DIGEST,
        "test_prompt_id_digest": TEST_PROMPT_ID_DIGEST,
        "pilot_paper_result_import_ready": True,
        "pilot_paper_claim_record_ready": True,
        "paper_run_result_import_coverage_ready": True,
        "paper_run_template_registry_unique": True,
        "pilot_paper_evidence_coverage_ready": True,
        "point_estimate_effect_direction_ready": True,
        "paired_superiority_ready": True,
        "paired_superiority_exact_set_ready": True,
        "overall_paired_superiority_ready": True,
        "overall_quality_matched_superiority_ready": True,
        "quality_matched_exact_set_ready": True,
        "quality_matching_uses_detection_labels": False,
        "pilot_paper_effectiveness_gate_ready": True,
        "slm_wm_fixed_fpr_boundary_ready": True,
        "paper_run_claim_ready": True,
        "paper_run_supports_superiority_claim": True,
        "paper_claim_ready": True,
        "result_record_set_digest": result_record_set_digest,
        "method_threshold_digest_map": METHOD_THRESHOLD_DIGEST_MAP,
        "closure_input_lock_digest": lock_digest,
        "common_code_version": COMMON_CODE_VERSION,
        "paired_outcome_set_digest": paired_summary["paired_outcome_set_digest"],
        "paired_superiority_rows_digest": paired_summary[
            "paired_superiority_rows_digest"
        ],
        "paired_superiority_protocol_digest": paired_summary[
            "paired_superiority_protocol_digest"
        ],
        "quality_matching_protocol_schema": paired_summary[
            "quality_matching_protocol_schema"
        ],
        "quality_matching_protocol_digest": paired_summary[
            "quality_matching_protocol_digest"
        ],
        "quality_metric_name": paired_summary["quality_metric_name"],
        "quality_match_caliper": paired_summary["quality_match_caliper"],
        "minimum_matched_prompt_fraction": paired_summary[
            "minimum_matched_prompt_fraction"
        ],
        "quality_matched_rows_digest": paired_summary[
            "quality_matched_rows_digest"
        ],
        "paired_test_prompt_count": paired_summary["paired_test_prompt_count"],
        "paired_test_prompt_id_digest": paired_summary[
            "paired_test_prompt_id_digest"
        ],
        "paired_attack_registry_digest": paired_summary[
            "paired_attack_registry_digest"
        ],
        "method_observation_source_sha256_map": paired_summary[
            "method_observation_source_sha256_map"
        ],
        "threshold_audit_rows_digest": paired_summary[
            "threshold_audit_rows_digest"
        ],
        "claim_p_value_method": paired_summary["claim_p_value_method"],
        "sharp_null_diagnostic_method": paired_summary[
            "sharp_null_diagnostic_method"
        ],
        "bootstrap_analysis_schema": paired_summary[
            "bootstrap_analysis_schema"
        ],
        "bootstrap_bit_generator": paired_summary["bootstrap_bit_generator"],
        "bootstrap_quantile_method": paired_summary[
            "bootstrap_quantile_method"
        ],
        "bootstrap_resample_count": paired_summary["bootstrap_resample_count"],
        "confidence_level": paired_summary["confidence_level"],
        "slm_wm_mean_true_positive_rate": 1.0,
        "slm_wm_mean_false_positive_rate": 0.0,
        "best_baseline_method_id": "tree_ring",
        "best_baseline_mean_true_positive_rate": 5 / TEST_COUNT,
    }
    common_schema = build_pilot_paper_result_import_schema(
        prompt_split_digest=PROMPT_SPLIT_DIGEST,
        attack_matrix_digest=ATTACK_MATRIX_DIGEST,
        fixed_fpr_protocol_digest=FIXED_FPR_PROTOCOL_DIGEST,
        config=PAPER_CONFIG,
    )
    common_schema.update(
        {
            "result_record_set_digest": result_record_set_digest,
            "calibration_prompt_id_digest": CALIBRATION_PROMPT_ID_DIGEST,
            "test_prompt_id_digest": TEST_PROMPT_ID_DIGEST,
            "method_threshold_digest_map": METHOD_THRESHOLD_DIGEST_MAP,
            "closure_input_lock_digest": lock_digest,
            "common_code_version": COMMON_CODE_VERSION,
            "paired_superiority_ready": True,
            "overall_paired_superiority_ready": True,
            "overall_quality_matched_superiority_ready": True,
            "quality_matched_exact_set_ready": True,
            "quality_matching_uses_detection_labels": False,
            "paired_outcome_set_digest": paired_summary[
                "paired_outcome_set_digest"
            ],
            "paired_superiority_rows_digest": paired_summary[
                "paired_superiority_rows_digest"
            ],
            "paired_superiority_protocol_digest": paired_summary[
                "paired_superiority_protocol_digest"
            ],
            "quality_matching_protocol_schema": paired_summary[
                "quality_matching_protocol_schema"
            ],
            "quality_matching_protocol_digest": paired_summary[
                "quality_matching_protocol_digest"
            ],
            "quality_metric_name": paired_summary["quality_metric_name"],
            "quality_match_caliper": paired_summary["quality_match_caliper"],
            "minimum_matched_prompt_fraction": paired_summary[
                "minimum_matched_prompt_fraction"
            ],
            "quality_matched_rows_digest": paired_summary[
                "quality_matched_rows_digest"
            ],
            "paired_test_prompt_count": paired_summary[
                "paired_test_prompt_count"
            ],
            "paired_test_prompt_id_digest": paired_summary[
                "paired_test_prompt_id_digest"
            ],
            "paired_attack_registry_digest": paired_summary[
                "paired_attack_registry_digest"
            ],
            "method_observation_source_sha256_map": paired_summary[
                "method_observation_source_sha256_map"
            ],
            "threshold_audit_rows_digest": paired_summary[
                "threshold_audit_rows_digest"
            ],
            "claim_p_value_method": paired_summary["claim_p_value_method"],
            "sharp_null_diagnostic_method": paired_summary[
                "sharp_null_diagnostic_method"
            ],
            "bootstrap_analysis_schema": paired_summary[
                "bootstrap_analysis_schema"
            ],
            "bootstrap_bit_generator": paired_summary[
                "bootstrap_bit_generator"
            ],
            "bootstrap_quantile_method": paired_summary[
                "bootstrap_quantile_method"
            ],
            "bootstrap_resample_count": paired_summary[
                "bootstrap_resample_count"
            ],
            "confidence_level": paired_summary["confidence_level"],
        }
    )
    result_record_validation_report = validate_pilot_paper_result_import_rows(
        result_records,
        common_schema,
        require_existing_evidence=False,
    )
    result_record_template_coverage = tuple(
        {
            "method_id": method_id,
            "attack_id": str(attack["attack_id"]),
            "attack_family": str(attack["attack_family"]),
            "attack_name": str(attack["attack_name"]),
            "resource_profile": str(attack["resource_profile"]),
            "attack_config_digest": str(attack["attack_config_digest"]),
            "template_covered": True,
            "supports_paper_claim": False,
        }
        for method_id in ("slm_wm_current", *PRIMARY_BASELINE_IDS)
        for attack in ATTACK_MATRIX_ROWS
    )
    analysis_summary = {
        "paper_claim_scale": SCALE,
        "target_fpr": TARGET_FPR,
        "result_record_count": len(result_records),
        "expected_result_record_count": len(result_records),
        "actual_result_record_count": len(result_records),
        "unique_result_record_key_count": len(result_records),
        "confidence_interval_row_count": len(result_records),
        "expected_superiority_row_count": len(ATTACK_REGISTRY),
        "per_attack_superiority_row_count": len(ATTACK_REGISTRY),
        "superiority_claim_ready_count": 0,
        "duplicate_result_record_count": 0,
        "missing_result_record_count": 0,
        "unexpected_result_record_count": 0,
        "failure_case_limit": 12,
        "failure_case_figure_ready": True,
        "result_analysis_payload_path_map": RESULT_ANALYSIS_PAYLOAD_PATH_MAP,
        "result_analysis_payload_sha256_map": (
            result_analysis_payload_sha256_map
        ),
        "result_analysis_payload_digest": result_analysis_payload_digest,
        **result_analysis_semantic_evidence,
        "result_template_coverage_ready": True,
        "per_attack_ci_coverage_ready": True,
        "per_attack_superiority_evaluation_ready": True,
        "universal_per_attack_superiority_claim_ready": False,
        "paired_superiority_ready": True,
        "overall_paired_superiority_ready": True,
        "overall_quality_matched_superiority_ready": True,
        "quality_matched_exact_set_ready": True,
        "quality_matching_uses_detection_labels": False,
        "paired_superiority_row_count": len(paired_rows),
        "paired_outcome_set_digest": paired_summary["paired_outcome_set_digest"],
        "paired_superiority_rows_digest": paired_summary[
            "paired_superiority_rows_digest"
        ],
        "paired_superiority_protocol_digest": paired_summary[
            "paired_superiority_protocol_digest"
        ],
        "quality_matching_protocol_schema": paired_summary[
            "quality_matching_protocol_schema"
        ],
        "quality_matching_protocol_digest": paired_summary[
            "quality_matching_protocol_digest"
        ],
        "quality_metric_name": paired_summary["quality_metric_name"],
        "quality_match_caliper": paired_summary["quality_match_caliper"],
        "minimum_matched_prompt_fraction": paired_summary[
            "minimum_matched_prompt_fraction"
        ],
        "quality_matched_rows_digest": paired_summary[
            "quality_matched_rows_digest"
        ],
        "paired_test_prompt_count": paired_summary["paired_test_prompt_count"],
        "paired_test_prompt_id_digest": paired_summary[
            "paired_test_prompt_id_digest"
        ],
        "paired_attack_registry_digest": paired_summary[
            "paired_attack_registry_digest"
        ],
        "method_observation_source_sha256_map": paired_summary[
            "method_observation_source_sha256_map"
        ],
        "threshold_audit_rows_digest": paired_summary[
            "threshold_audit_rows_digest"
        ],
        "claim_p_value_method": paired_summary["claim_p_value_method"],
        "sharp_null_diagnostic_method": paired_summary[
            "sharp_null_diagnostic_method"
        ],
        "bootstrap_analysis_schema": paired_summary[
            "bootstrap_analysis_schema"
        ],
        "bootstrap_bit_generator": paired_summary["bootstrap_bit_generator"],
        "bootstrap_quantile_method": paired_summary[
            "bootstrap_quantile_method"
        ],
        "bootstrap_resample_count": paired_summary["bootstrap_resample_count"],
        "confidence_level": paired_summary["confidence_level"],
        "result_record_set_digest": result_record_set_digest,
        "supports_paper_claim": True,
    }
    ablation_summary = {
        "paper_claim_scale": SCALE,
        "target_fpr": TARGET_FPR,
        "split_counts": {"test": TEST_COUNT},
        "record_count": PROMPT_COUNT * len(FORMAL_RUNTIME_RERUN_ABLATION_IDS),
        "prompt_count": PROMPT_COUNT,
        "ablation_count": len(FORMAL_RUNTIME_RERUN_ABLATION_IDS),
        "per_ablation_calibration_count": len(FORMAL_RUNTIME_RERUN_ABLATION_IDS),
        "generation_rerun_count": PROMPT_COUNT * len(FORMAL_RUNTIME_RERUN_ABLATION_IDS),
        "prompt_protocol_exact_set_ready": True,
        "prompt_id_digest": PROMPT_ID_DIGEST,
        "calibration_prompt_id_digest": CALIBRATION_PROMPT_ID_DIGEST,
        "test_prompt_id_digest": TEST_PROMPT_ID_DIGEST,
        "expected_attack_and_detection_rerun_count": TEST_COUNT
        * len(FORMAL_RUNTIME_RERUN_ABLATION_IDS),
        "attack_and_detection_rerun_count": TEST_COUNT
        * len(FORMAL_RUNTIME_RERUN_ABLATION_IDS),
        "formal_attack_coverage_ready_count": PROMPT_COUNT
        * len(FORMAL_RUNTIME_RERUN_ABLATION_IDS),
        "formal_attack_coverage_ready": True,
        **ablation_atom_identity,
        "expected_ablation_ids": list(FORMAL_RUNTIME_RERUN_ABLATION_IDS),
        "actual_ablation_ids": list(FORMAL_RUNTIME_RERUN_ABLATION_IDS),
        "ablation_spec_digest": FORMAL_RUNTIME_RERUN_ABLATION_SPEC_DIGEST,
        "ablation_exact_set_ready": True,
        "ablation_claim_gate_ready": True,
        **necessity_summary,
        "ablation_necessity_statistics_ready": True,
        "necessity_statistic_row_count": len(
            FORMAL_RUNTIME_RERUN_ABLATION_IDS
        )
        - 1,
        "paired_prompt_count": TEST_COUNT,
        "expected_paired_prompt_count": TEST_COUNT,
        "necessity_statistic_rows_digest": necessity_summary[
            "necessity_statistic_rows_digest"
        ],
        "necessity_supported_ablation_ids": [
            ablation_id
            for ablation_id in FORMAL_RUNTIME_RERUN_ABLATION_IDS
            if ablation_id != "complete_method"
        ],
        "necessity_not_supported_ablation_ids": [],
        "all_mechanism_necessity_claims_supported": True,
        "protocol_decision": "pass",
        "supports_paper_claim": True,
    }
    quality_summary = {
        "paper_claim_scale": SCALE,
        "target_fpr": TARGET_FPR,
        "expected_prompt_count": PROMPT_COUNT,
        "registry_prompt_count": PROMPT_COUNT,
        "duplicate_registry_prompt_id_count": 0,
        "missing_registry_prompt_id_count": 0,
        "unexpected_registry_prompt_id_count": 0,
        "canonical_prompt_id_digest": PROMPT_ID_DIGEST,
        "registry_prompt_id_digest": PROMPT_ID_DIGEST,
        "prompt_registry_exact_set_ready": True,
        "sample_pair_count": PROMPT_COUNT,
        "source_image_count": PROMPT_COUNT,
        "comparison_image_count": PROMPT_COUNT,
        "formal_fid_kid_ready": True,
        "formal_fid_kid_metric_names_ready": True,
        "formal_feature_backend_ready": True,
        "formal_sample_scale_ready": True,
        "canonical_formal_feature_extractor_ready": True,
        "formal_feature_record_count": PROMPT_COUNT * 2,
        "expected_feature_pair_count": PROMPT_COUNT,
        "accepted_feature_pair_count": PROMPT_COUNT,
        "missing_feature_pair_count": 0,
        "feature_issue_count": 0,
        "formal_feature_records_sha256": FEATURE_RECORDS_SHA256,
        "formal_fid_kid_claim_gate_ready": True,
        "image_resolution_record_count": len(
            QUALITY_IMAGE_RESOLUTION_RECORDS
        ),
        "resolved_image_file_count": len(QUALITY_IMAGE_RESOLUTION_RECORDS),
        "missing_image_file_count": 0,
        "image_resolution_identity_ready": True,
        "kid_effective_subset_size": PROMPT_COUNT,
        "formal_metric_protocol": formal_dataset_quality_metric_protocol(),
        "formal_metric_protocol_digest": formal_dataset_quality_metric_protocol()[
            "formal_metric_protocol_digest"
        ],
        **QUALITY_PROVENANCE_SUMMARY,
    }
    quality_feature_report = {
        "paper_run_name": SCALE,
        "target_fpr": TARGET_FPR,
        "canonical_prompt_id_digest": PROMPT_ID_DIGEST,
        "registry_prompt_id_digest": PROMPT_ID_DIGEST,
        "prompt_registry_exact_set_ready": True,
        "formal_feature_record_count": PROMPT_COUNT * 2,
        "expected_feature_pair_count": PROMPT_COUNT,
        "accepted_feature_pair_count": PROMPT_COUNT,
        "missing_feature_pair_count": 0,
        "feature_issue_count": 0,
        "formal_feature_records_sha256": FEATURE_RECORDS_SHA256,
        **QUALITY_PROVENANCE_SUMMARY,
    }
    source_features = [
        record["feature_vector"]
        for record in QUALITY_FEATURE_RECORDS
        if record["dataset_quality_image_role"] == "source"
    ]
    comparison_features = [
        record["feature_vector"]
        for record in QUALITY_FEATURE_RECORDS
        if record["dataset_quality_image_role"] == "comparison"
    ]
    quality_metrics = tuple(
        {
            **row,
            "quality_metric_value": str(row["quality_metric_value"]),
            "source_image_count": str(row["source_image_count"]),
            "comparison_image_count": str(row["comparison_image_count"]),
            "sample_pair_count": str(row["sample_pair_count"]),
        }
        for row in rebuild_formal_fid_kid_metric_rows(
            source_features,
            comparison_features,
            sample_pair_count=PROMPT_COUNT,
        )
    )
    artifact_payload_map = artifact_source_payloads(
        quality_metrics,
        attack_family_metrics,
        tuple(necessity_rows),
        result_analysis_baseline_rows,
    )
    artifact_source_sha256 = {
        path: hashlib.sha256(payload).hexdigest()
        for path, payload in artifact_payload_map.items()
    }
    artifact_check_ids = (*ARTIFACT_SOURCE_PATHS, "ready_flag_consistency_ready")
    artifact_checks = {
        check_id: {
            "check_id": check_id,
            "data_ready": True,
            "row_count": 1,
            "issues": [],
        }
        for check_id in artifact_check_ids
    }
    artifact_data_validation_report = {
        **{check_id: True for check_id in artifact_check_ids},
        "artifact_data_validation_ready": True,
        "artifact_data_check_count": len(artifact_checks),
        "blocked_artifact_data_count": 0,
        "blocked_artifact_data_ids": [],
        "source_paths": ARTIFACT_SOURCE_PATHS,
        "evidence_source_file_sha256": artifact_source_sha256,
        "raw_image_only_detection_records_sha256": artifact_source_sha256[
            ARTIFACT_SOURCE_PATHS["raw_image_only_detection_records_ready"]
        ],
        "checks": artifact_checks,
        "supports_paper_claim": False,
    }
    quality_resolution_records_bytes = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
        for record in QUALITY_IMAGE_RESOLUTION_RECORDS
    ).encode("utf-8")
    quality_actual_image_sha256 = {
        str(record["resolved_image_path"]): str(record["resolved_image_digest"])
        for record in QUALITY_IMAGE_RESOLUTION_RECORDS
    }
    source_file_sha256 = {
        **{
            path: hashlib.sha256(payload).hexdigest()
            for path, payload in official_source_payload_map.items()
        },
        **artifact_source_sha256,
        f"outputs/dataset_level_quality/{SCALE}/dataset_quality_image_resolution_records.jsonl": (
            hashlib.sha256(quality_resolution_records_bytes).hexdigest()
        ),
        **quality_actual_image_sha256,
        f"outputs/formal_mechanism_ablation/{SCALE}/runtime_rerun_records.jsonl": (
            hashlib.sha256(
                "".join(
                    json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
                    for record in necessity_records
                ).encode("utf-8")
            ).hexdigest()
        ),
        f"outputs/formal_mechanism_ablation/{SCALE}/formal_detection_records.jsonl": (
            ablation_atom_identity["formal_detection_records_sha256"]
        ),
        f"outputs/formal_mechanism_ablation/{SCALE}/per_ablation_frozen_protocols.json": (
            ablation_atom_identity["per_ablation_frozen_protocols_sha256"]
        ),
        **{
            RESULT_ANALYSIS_PAYLOAD_PATH_MAP[role]: digest
            for role, digest in result_analysis_payload_sha256_map.items()
        },
        **{
            METHOD_OBSERVATION_SOURCE_PATH_MAP[method_id]: digest
            for method_id, digest in METHOD_OBSERVATION_SOURCE_SHA256_MAP.items()
        },
    }
    evidence_runtime_report = {
        "paper_run_name": SCALE,
        "prompt_count": PROMPT_COUNT,
        "runtime_result_count": PROMPT_COUNT,
        "split_counts": {"test": TEST_COUNT},
        "protocol_decision": "pass",
        "target_fpr": TARGET_FPR,
        "frozen_threshold_digest": MAIN_THRESHOLD_DIGEST,
        "raw_content_claim_ready": True,
        "full_method_claim_ready": True,
        "perceptual_metrics_ready": True,
        "scientific_operator_gate_ready": True,
        "scientific_operator_failure_count": 0,
        "scientific_update_record_count": PROMPT_COUNT,
        "expected_scientific_update_record_count": PROMPT_COUNT,
        "detector_input_access_mode": "image_key_public_model_only",
        "generation_latent_trace_required": False,
        "fixed_fpr_and_rescue_boundary_ready": True,
        "fixed_fpr_boundary_ready": True,
        "rescue_boundary_ready": True,
        "supports_paper_claim": True,
    }
    evidence_runtime_manifest = manifest(
        f"{SCALE}_image_only_dataset_runtime_manifest",
        (
            f"outputs/image_only_dataset_runtime/{SCALE}/dataset_runtime_summary.json",
            f"outputs/image_only_dataset_runtime/{SCALE}/manifest.local.json",
        ),
        evidence_runtime_report,
    )
    evidence_source_path_map = evidence_audit_source_path_map()
    builder_report = {
        "artifact_builder_ready": True,
        "paper_artifact_claim_ready": True,
        "paper_artifact_audit_ready": True,
        "blocked_artifact_count": 0,
    }
    blocker_report = {
        "submission_ready": True,
        "artifact_builder_ready": True,
        "paper_artifact_claim_ready": True,
        "paper_artifact_audit_ready": True,
        "full_method_claim_ready": True,
        "supports_paper_claim": True,
        "blocking_claim_count": 0,
        "critical_gap_count": 0,
        "gap_count": 0,
    }
    submission_report = {
        "readiness_decision": "ready",
        "submission_ready": True,
        "package_freeze_allowed": True,
        "artifact_builder_ready": True,
        "paper_artifact_claim_ready": True,
        "release_dry_run_ready": True,
        "required_input_count": 0,
        "critical_required_input_count": 0,
        "blocking_claim_count": 0,
    }
    entry_report = {
        "entry_review_decision": "ready_for_evidence_closure",
        "entry_review_ready": True,
        "evidence_closure_allowed": True,
        "blocked_review_item_count": 0,
        "required_input_count": 0,
        "critical_required_input_count": 0,
        "blocking_claim_count": 0,
        "primary_baseline_results_ready": True,
        "formal_import_validation_ready": True,
        "accepted_formal_import_count": 5,
        "formal_evidence_path_resolution_ready": True,
        "formal_fid_kid_ready": True,
        "formal_sample_scale_ready": True,
        "formal_feature_backend_ready": True,
    }
    bundle = ResultClosureGateInput(
        expected_paper_claim_scale=SCALE,
        expected_target_fpr=TARGET_FPR,
        expected_prompt_count=PROMPT_COUNT,
        expected_test_count=TEST_COUNT,
        expected_prompt_split_digest=PROMPT_SPLIT_DIGEST,
        expected_prompt_id_digest=PROMPT_ID_DIGEST,
        expected_calibration_prompt_id_digest=(
            CALIBRATION_PROMPT_ID_DIGEST
        ),
        expected_test_prompt_id_digest=TEST_PROMPT_ID_DIGEST,
        expected_prompt_split_by_id=PROMPT_SPLIT_BY_ID,
        expected_prompt_digest_by_id=PROMPT_DIGEST_BY_ID,
            source_file_sha256=source_file_sha256,
            attack_report=attack_report,
            attack_detection_records=attack_detection_records,
            attack_family_metrics=attack_family_metrics,
            attacked_image_registry=attacked_image_registry,
            attack_manifest=manifest(
                f"{SCALE}_attack_matrix_manifest",
                (
                    f"outputs/attack_matrix/{SCALE}/attack_manifest.json",
                    f"outputs/attack_matrix/{SCALE}/attack_detection_records.jsonl",
                    f"outputs/attack_matrix/{SCALE}/attacked_image_registry.jsonl",
                    f"outputs/attack_matrix/{SCALE}/attack_family_metrics.csv",
                    f"outputs/attack_matrix/{SCALE}/manifest.local.json",
                ),
                attack_metadata,
                input_paths=(
                    METHOD_OBSERVATION_SOURCE_PATH_MAP["slm_wm"],
                ),
                config=build_attack_matrix_manifest_config(
                    paper_run_name=SCALE,
                    evaluation_boundary=attack_report[
                        "evaluation_boundary"
                    ],
                    attack_configs=FORMAL_ATTACK_CONFIGS,
                    attack_records=attack_detection_records,
                ),
                code_version=COMMON_CODE_VERSION,
            ),
        threshold_audit_report=threshold_report,
        threshold_audit_rows=threshold_rows,
        threshold_audit_manifest=manifest(
            "fixed_fpr_threshold_audit_manifest",
            (
                f"outputs/fixed_fpr_threshold_audit/{SCALE}/threshold_audit_rows.csv",
                f"outputs/fixed_fpr_threshold_audit/{SCALE}/threshold_audit_report.json",
                f"outputs/fixed_fpr_threshold_audit/{SCALE}/manifest.local.json",
            ),
            threshold_report,
            config=build_fixed_fpr_threshold_manifest_config(threshold_report),
        ),
        closure_input_lock=lock_payload,
        closure_input_lock_manifest=lock_manifest,
        official_reference_fidelity_records=official_records,
        official_reference_fidelity_summary=official_summary,
        official_reference_fidelity_manifest=manifest(
            "official_reference_fidelity_evidence_manifest",
            (
                f"outputs/official_reference_fidelity_evidence/{SCALE}/official_reference_fidelity_evidence_records.jsonl",
                f"outputs/official_reference_fidelity_evidence/{SCALE}/official_reference_fidelity_evidence_summary.json",
                f"outputs/official_reference_fidelity_evidence/{SCALE}/manifest.local.json",
            ),
            official_summary,
            input_paths=tuple(official_source_payload_map),
            config={
                "official_reference_fidelity_evidence_digest": (
                    official_records_digest
                )
            },
            code_version=COMMON_CODE_VERSION,
        ),
        primary_baseline_evidence_records=primary_evidence_records,
        primary_baseline_evidence_summary=primary_evidence_summary,
        primary_baseline_evidence_manifest=manifest(
            "primary_baseline_evidence_manifest",
            (
                f"outputs/primary_baseline_evidence/{SCALE}/primary_baseline_evidence_records.jsonl",
                f"outputs/primary_baseline_evidence/{SCALE}/primary_baseline_evidence_summary.json",
                f"outputs/primary_baseline_evidence/{SCALE}/manifest.local.json",
            ),
            primary_evidence_summary,
            config={
                "records_digest": primary_evidence_records_digest,
                "primary_baseline_evidence_records_digest": primary_evidence_records_digest,
            },
        ),
        baseline_report=baseline_report,
        baseline_manifest=manifest(
            "external_baseline_comparison_manifest",
            (
                f"outputs/external_baseline_comparison/{SCALE}/baseline_runtime_report.json",
                f"outputs/external_baseline_comparison/{SCALE}/baseline_comparison_table.csv",
                f"outputs/external_baseline_comparison/{SCALE}/manifest.local.json",
            ),
            baseline_report,
        ),
        result_records=result_records,
        result_record_validation_report=result_record_validation_report,
        result_record_template_coverage=result_record_template_coverage,
        result_record_summary=record_summary,
        result_record_manifest=manifest(
            "pilot_paper_fixed_fpr_result_records_manifest",
            (
                f"outputs/pilot_paper_fixed_fpr_results/{SCALE}/pilot_paper_result_records.jsonl",
                f"outputs/pilot_paper_fixed_fpr_results/{SCALE}/pilot_paper_result_import_validation_report.json",
                f"outputs/pilot_paper_fixed_fpr_results/{SCALE}/pilot_paper_result_template_coverage.csv",
                f"outputs/pilot_paper_fixed_fpr_results/{SCALE}/pilot_paper_result_record_summary.json",
                f"outputs/pilot_paper_fixed_fpr_results/{SCALE}/manifest.local.json",
            ),
            record_summary,
            input_paths=tuple(METHOD_OBSERVATION_SOURCE_PATH_MAP.values()),
            config=build_pilot_paper_result_records_manifest_config(
                result_records=result_records,
                method_threshold_digest_map=METHOD_THRESHOLD_DIGEST_MAP,
                closure_input_lock_digest=lock_digest,
                common_code_version=COMMON_CODE_VERSION,
                validation_report=result_record_validation_report,
                template_coverage_rows=result_record_template_coverage,
                summary=record_summary,
                require_existing_evidence=False,
            ),
            code_version=COMMON_CODE_VERSION,
        ),
        common_protocol_summary=common_summary,
        common_protocol_schema=common_schema,
        common_protocol_manifest=manifest(
            "pilot_paper_fixed_fpr_common_protocol_manifest",
            (
                f"outputs/pilot_paper_fixed_fpr_common_protocol/{SCALE}/pilot_paper_result_import_schema.json",
                f"outputs/pilot_paper_fixed_fpr_common_protocol/{SCALE}/pilot_paper_common_protocol_summary.json",
                f"outputs/pilot_paper_fixed_fpr_common_protocol/{SCALE}/manifest.local.json",
            ),
            common_summary,
            input_paths=(
                f"outputs/paired_superiority_analysis/{SCALE}/paired_superiority_summary.json",
                f"outputs/paired_superiority_analysis/{SCALE}/manifest.local.json",
            ),
            config={
                "result_record_set_digest": result_record_set_digest,
                "method_threshold_digest_map": METHOD_THRESHOLD_DIGEST_MAP,
                "closure_input_lock_digest": lock_digest,
                "common_code_version": COMMON_CODE_VERSION,
            },
        ),
        result_analysis_summary=analysis_summary,
        result_analysis_manifest=manifest(
            "pilot_paper_result_analysis_manifest",
            (
                f"outputs/pilot_paper_result_analysis/{SCALE}/confidence_interval_table.csv",
                f"outputs/pilot_paper_result_analysis/{SCALE}/per_attack_superiority_table.csv",
                f"outputs/pilot_paper_result_analysis/{SCALE}/failure_case_records.jsonl",
                f"outputs/pilot_paper_result_analysis/{SCALE}/failure_case_figure.svg",
                f"outputs/pilot_paper_result_analysis/{SCALE}/result_analysis_summary.json",
                f"outputs/pilot_paper_result_analysis/{SCALE}/manifest.local.json",
            ),
            analysis_summary,
            input_paths=(
                f"outputs/paired_superiority_analysis/{SCALE}/paired_superiority_summary.json",
                f"outputs/paired_superiority_analysis/{SCALE}/paired_superiority_table.csv",
                f"outputs/paired_superiority_analysis/{SCALE}/manifest.local.json",
            ),
            config=build_result_analysis_manifest_config(analysis_summary),
        ),
        result_analysis_governed_payload_path_map=(
            result_analysis_governed_path_map
        ),
        result_analysis_baseline_comparison_rows=(
            result_analysis_baseline_rows
        ),
        result_analysis_confidence_interval_rows=(
            result_analysis_confidence_rows
        ),
        result_analysis_per_attack_superiority_rows=(
            result_analysis_per_attack_rows
        ),
        result_analysis_failure_case_rows=result_analysis_failure_rows,
        result_analysis_failure_case_svg_text=(
            result_analysis_failure_svg_text
        ),
        result_analysis_failure_figure_path=(
            result_analysis_failure_figure_path
        ),
        paired_observation_records_by_method=(
            METHOD_OBSERVATION_RECORDS_BY_METHOD
        ),
        paired_outcomes=paired_outcomes,
        paired_superiority_rows=paired_rows,
        paired_superiority_summary=paired_summary,
        paired_superiority_manifest=manifest(
            "paired_superiority_analysis_manifest",
            (
                f"outputs/paired_superiority_analysis/{SCALE}/paired_outcomes.jsonl",
                f"outputs/paired_superiority_analysis/{SCALE}/paired_superiority_table.csv",
                f"outputs/paired_superiority_analysis/{SCALE}/paired_superiority_summary.json",
                f"outputs/paired_superiority_analysis/{SCALE}/manifest.local.json",
            ),
            paired_summary,
            input_paths=(
                *tuple(METHOD_OBSERVATION_SOURCE_PATH_MAP.values()),
                f"outputs/fixed_fpr_threshold_audit/{SCALE}/threshold_audit_rows.csv",
                f"outputs/fixed_fpr_threshold_audit/{SCALE}/threshold_audit_report.json",
                f"outputs/fixed_fpr_threshold_audit/{SCALE}/manifest.local.json",
            ),
            config=build_paired_superiority_manifest_config(paired_summary),
            code_version=COMMON_CODE_VERSION,
        ),
        ablation_summary=ablation_summary,
        ablation_manifest=manifest(
            "formal_mechanism_ablation_manifest",
            (
                f"outputs/formal_mechanism_ablation/{SCALE}/runtime_rerun_records.jsonl",
                f"outputs/formal_mechanism_ablation/{SCALE}/formal_detection_records.jsonl",
                f"outputs/formal_mechanism_ablation/{SCALE}/per_ablation_frozen_protocols.json",
                f"outputs/formal_mechanism_ablation/{SCALE}/mechanism_ablation_metrics.csv",
                f"outputs/formal_mechanism_ablation/{SCALE}/mechanism_necessity_statistics.csv",
                f"outputs/formal_mechanism_ablation/{SCALE}/mechanism_necessity_summary.json",
                f"outputs/formal_mechanism_ablation/{SCALE}/ablation_claim_summary.json",
                f"outputs/formal_mechanism_ablation/{SCALE}/manifest.local.json",
            ),
            {
                "protocol_decision": "pass",
                "expected_ablation_ids": list(FORMAL_RUNTIME_RERUN_ABLATION_IDS),
                "actual_ablation_ids": list(FORMAL_RUNTIME_RERUN_ABLATION_IDS),
                "ablation_spec_digest": FORMAL_RUNTIME_RERUN_ABLATION_SPEC_DIGEST,
                "ablation_exact_set_ready": True,
                "prompt_protocol_exact_set_ready": True,
                "prompt_id_digest": PROMPT_ID_DIGEST,
                "calibration_prompt_id_digest": CALIBRATION_PROMPT_ID_DIGEST,
                "test_prompt_id_digest": TEST_PROMPT_ID_DIGEST,
                "formal_attack_coverage_ready": True,
                **ablation_atom_identity,
                "ablation_necessity_statistics_ready": True,
                "necessity_statistic_rows_digest": necessity_summary[
                    "necessity_statistic_rows_digest"
                ],
                "necessity_supported_ablation_ids": [
                    ablation_id
                    for ablation_id in FORMAL_RUNTIME_RERUN_ABLATION_IDS
                    if ablation_id != "complete_method"
                ],
                "necessity_not_supported_ablation_ids": [],
                "all_mechanism_necessity_claims_supported": True,
                "generation_rerun_required": True,
                "per_ablation_calibration_required": True,
                "supports_paper_claim": True,
            },
            config={
                "expected_ablation_ids": list(FORMAL_RUNTIME_RERUN_ABLATION_IDS),
                "actual_ablation_ids": list(FORMAL_RUNTIME_RERUN_ABLATION_IDS),
                "ablation_spec_digest": FORMAL_RUNTIME_RERUN_ABLATION_SPEC_DIGEST,
                "ablation_exact_set_ready": True,
                "record_digest": build_stable_digest(necessity_records),
                **ablation_atom_identity,
            },
        ),
        ablation_runtime_records=tuple(necessity_records),
        ablation_detection_records=ablation_detection_records,
        ablation_frozen_protocols=ablation_frozen_protocols,
        ablation_necessity_rows=tuple(necessity_rows),
        ablation_necessity_summary=necessity_summary,
        dataset_quality_summary=quality_summary,
        dataset_quality_image_records=QUALITY_IMAGE_RECORDS,
        dataset_quality_image_resolution_records=(
            QUALITY_IMAGE_RESOLUTION_RECORDS
        ),
        dataset_quality_feature_report=quality_feature_report,
        dataset_quality_metrics=quality_metrics,
        dataset_quality_feature_records=QUALITY_FEATURE_RECORDS,
        dataset_quality_feature_records_sha256=FEATURE_RECORDS_SHA256,
        dataset_quality_manifest={
            **manifest(
                "dataset_level_quality_manifest",
                (
                    f"outputs/dataset_level_quality/{SCALE}/dataset_quality_image_records.jsonl",
                    f"outputs/dataset_level_quality/{SCALE}/dataset_quality_image_resolution_records.jsonl",
                    f"outputs/dataset_level_quality/{SCALE}/dataset_quality_formal_feature_records.jsonl",
                    f"outputs/dataset_level_quality/{SCALE}/dataset_quality_formal_feature_import_report.json",
                    f"outputs/dataset_level_quality/{SCALE}/dataset_quality_metrics.csv",
                    f"outputs/dataset_level_quality/{SCALE}/dataset_quality_summary.json",
                    f"outputs/dataset_level_quality/{SCALE}/manifest.local.json",
                ),
                quality_summary,
                config={
                    "canonical_prompt_id_digest": PROMPT_ID_DIGEST,
                    "registry_prompt_id_digest": PROMPT_ID_DIGEST,
                    "prompt_registry_exact_set_ready": True,
                    "accepted_feature_pair_count": PROMPT_COUNT,
                    "missing_feature_pair_count": 0,
                    "feature_issue_count": 0,
                    "formal_feature_record_count": PROMPT_COUNT * 2,
                    "formal_feature_records_sha256": FEATURE_RECORDS_SHA256,
                },
            ),
            "formal_execution_run_lock": QUALITY_FORMAL_EXECUTION_LOCK,
        },
        evidence_builder_report=builder_report,
        evidence_blocker_report=blocker_report,
        evidence_audit_runtime_report=evidence_runtime_report,
        evidence_audit_runtime_manifest=evidence_runtime_manifest,
        evidence_audit_source_path_map=evidence_source_path_map,
        artifact_data_validation_report=artifact_data_validation_report,
        recomputed_artifact_data_validation_report=json.loads(
            json.dumps(artifact_data_validation_report)
        ),
        evidence_audit_manifest=manifest(
            "paper_artifact_evidence_audit_manifest",
            (
                f"outputs/paper_artifact_evidence_audit/{SCALE}/artifact_builder_readiness_report.json",
                f"outputs/paper_artifact_evidence_audit/{SCALE}/submission_blocker_report.json",
                f"outputs/paper_artifact_evidence_audit/{SCALE}/artifact_data_validation_report.json",
                f"outputs/paper_artifact_evidence_audit/{SCALE}/manifest.local.json",
            ),
            {
                **blocker_report,
                "artifact_data_validation_ready": True,
                "blocked_artifact_data_ids": [],
                "evidence_source_file_sha256": artifact_source_sha256,
            },
            input_paths=(
                f"outputs/image_only_dataset_runtime/{SCALE}/dataset_runtime_summary.json",
                f"outputs/formal_mechanism_ablation/{SCALE}/ablation_claim_summary.json",
                f"outputs/dataset_level_quality/{SCALE}/dataset_quality_summary.json",
                *tuple(ARTIFACT_SOURCE_PATHS.values()),
            ),
            config={
                "artifact_data_validation_digest": build_stable_digest(
                    artifact_data_validation_report
                )
            },
        ),
        submission_readiness_report=submission_report,
        submission_readiness_manifest=manifest(
            "submission_readiness_manifest",
            (
                f"outputs/submission_readiness/{SCALE}/readiness_blocker_report.json",
                f"outputs/submission_readiness/{SCALE}/submission_readiness_manifest.local.json",
            ),
            submission_report,
        ),
        entry_review_report=entry_report,
        entry_review_manifest=manifest(
            "evidence_closure_entry_review_manifest",
            (
                f"outputs/evidence_closure_entry_review/{SCALE}/entry_review_report.json",
                f"outputs/evidence_closure_entry_review/{SCALE}/manifest.local.json",
            ),
            entry_report,
            input_paths=(f"outputs/dataset_level_quality/{SCALE}/dataset_quality_summary.json",),
        ),
    )
    audit_bundle = AuditInputBundle(
        threshold_report=bundle.evidence_audit_runtime_report,
        threshold_manifest=bundle.evidence_audit_runtime_manifest,
        threshold_audit_report=bundle.threshold_audit_report,
        threshold_audit_manifest=bundle.threshold_audit_manifest,
        attack_manifest=bundle.attack_report,
        attack_matrix_manifest=bundle.attack_manifest,
        baseline_manifest=bundle.baseline_manifest,
        baseline_runtime_report=bundle.baseline_report,
        dataset_quality_manifest=bundle.dataset_quality_manifest,
        dataset_quality_summary=bundle.dataset_quality_summary,
        ablation_manifest=bundle.ablation_manifest,
        ablation_claim_summary=bundle.ablation_summary,
        source_path_map=bundle.evidence_audit_source_path_map,
        artifact_data_validation=(
            {
                **bundle.recomputed_artifact_data_validation_report,
                "mechanism_necessity_statistics_ready": True,
            }
        ),
        ablation_necessity_rows=tuple(necessity_rows),
        ablation_necessity_summary=necessity_summary,
    )
    materialization = build_evidence_audit_materialization(audit_bundle)
    audit_manifest_config = build_evidence_audit_manifest_config(
        audit_bundle,
        materialization,
    )
    return replace(
        bundle,
        evidence_builder_report=materialization["builder_report"],
        evidence_blocker_report=materialization["blocker_report"],
        evidence_audit_manifest={
            **bundle.evidence_audit_manifest,
            "config": audit_manifest_config,
            "config_digest": build_stable_digest(audit_manifest_config),
            "metadata": {
                **materialization["blocker_report"],
                "artifact_data_validation_ready": True,
                "blocked_artifact_data_ids": [],
                "evidence_source_file_sha256": artifact_source_sha256,
                "raw_image_only_detection_records_ready": True,
                "raw_image_only_detection_records_sha256": (
                    artifact_data_validation_report[
                        "raw_image_only_detection_records_sha256"
                    ]
                ),
            },
        },
    )


def synchronize_paired_evidence(
    bundle: ResultClosureGateInput,
    *,
    outcomes: tuple[dict[str, object], ...],
    rows: tuple[dict[str, object], ...],
) -> ResultClosureGateInput:
    """同步伪造配对链的全部自声明摘要, 用于验证门禁会独立复算."""

    quality_rows = tuple(
        build_quality_matched_superiority_rows(
            outcomes,
            protocol_digest=str(
                bundle.paired_superiority_summary[
                    "paired_superiority_protocol_digest"
                ]
            ),
        )
    )
    merged_rows = tuple(
        merge_paired_and_quality_matched_rows(rows, quality_rows)
    )
    rebuilt_summary = build_paired_superiority_summary(
        merged_rows,
        paired_outcomes=outcomes,
    )
    quality_summary = build_quality_matched_superiority_summary(quality_rows)
    summary = {
        **bundle.paired_superiority_summary,
        **rebuilt_summary,
        **quality_summary,
        "paired_outcome_count": len(outcomes),
        "paired_outcome_set_digest": build_stable_digest(outcomes),
    }
    propagated_fields = (
        "paired_outcome_set_digest",
        "paired_superiority_rows_digest",
        "paired_superiority_protocol_digest",
        "quality_matching_protocol_schema",
        "quality_matching_protocol_digest",
        "quality_metric_name",
        "quality_match_caliper",
        "minimum_matched_prompt_fraction",
        "quality_matched_rows_digest",
        "overall_quality_matched_superiority_ready",
        "quality_matched_exact_set_ready",
        "quality_matching_uses_detection_labels",
        "paired_test_prompt_count",
        "paired_test_prompt_id_digest",
        "paired_attack_registry_digest",
        "method_observation_source_sha256_map",
        "threshold_audit_rows_digest",
        "claim_p_value_method",
        "sharp_null_diagnostic_method",
        "bootstrap_analysis_schema",
        "bootstrap_bit_generator",
        "bootstrap_quantile_method",
        "bootstrap_resample_count",
        "confidence_level",
        "overall_paired_superiority_ready",
    )
    common_summary = {
        **bundle.common_protocol_summary,
        **{field_name: summary[field_name] for field_name in propagated_fields},
    }
    common_schema = {
        **bundle.common_protocol_schema,
        **{field_name: summary[field_name] for field_name in propagated_fields},
    }
    analysis_summary = {
        **bundle.result_analysis_summary,
        **{field_name: summary[field_name] for field_name in propagated_fields},
    }
    paired_manifest_config = build_paired_superiority_manifest_config(summary)
    paired_manifest = {
        **bundle.paired_superiority_manifest,
        "metadata": summary,
        "config": paired_manifest_config,
        "config_digest": build_stable_digest(paired_manifest_config),
    }
    return replace(
        bundle,
        paired_outcomes=outcomes,
        paired_superiority_rows=merged_rows,
        paired_superiority_summary=summary,
        paired_superiority_manifest=paired_manifest,
        common_protocol_summary=common_summary,
        common_protocol_schema=common_schema,
        common_protocol_manifest={
            **bundle.common_protocol_manifest,
            "metadata": common_summary,
        },
        result_analysis_summary=analysis_summary,
        result_analysis_manifest={
            **bundle.result_analysis_manifest,
            "metadata": analysis_summary,
        },
    )


def synchronize_result_record_evidence(
    bundle: ResultClosureGateInput,
    records: tuple[dict[str, object], ...],
) -> ResultClosureGateInput:
    """同步 result record 自声明链, 用于验证原始 observation 独立复算."""

    result_record_set_digest = build_pilot_paper_result_record_set_digest(
        records
    )
    record_summary = {
        **bundle.result_record_summary,
        "result_record_set_digest": result_record_set_digest,
    }
    common_schema = {
        **bundle.common_protocol_schema,
        "result_record_set_digest": result_record_set_digest,
    }
    validation_report = validate_pilot_paper_result_import_rows(
        records,
        common_schema,
        require_existing_evidence=False,
    )
    slm_rows = [
        row for row in records if row.get("method_id") == "slm_wm_current"
    ]
    common_summary = {
        **bundle.common_protocol_summary,
        "result_record_set_digest": result_record_set_digest,
        "slm_wm_mean_false_positive_rate": (
            sum(float(row["false_positive_rate"]) for row in slm_rows)
            / len(slm_rows)
        ),
    }
    analysis_summary = {
        **bundle.result_analysis_summary,
        "result_record_set_digest": result_record_set_digest,
    }
    result_manifest_config = build_pilot_paper_result_records_manifest_config(
        result_records=records,
        method_threshold_digest_map=METHOD_THRESHOLD_DIGEST_MAP,
        closure_input_lock_digest=str(
            bundle.closure_input_lock["closure_input_lock_digest"]
        ),
        common_code_version=COMMON_CODE_VERSION,
        validation_report=validation_report,
        template_coverage_rows=bundle.result_record_template_coverage,
        summary=record_summary,
        require_existing_evidence=False,
    )
    return replace(
        bundle,
        result_records=records,
        result_record_validation_report=validation_report,
        result_record_summary=record_summary,
        result_record_manifest={
            **bundle.result_record_manifest,
            "metadata": record_summary,
            "config": result_manifest_config,
            "config_digest": build_stable_digest(result_manifest_config),
        },
        common_protocol_schema=common_schema,
        common_protocol_summary=common_summary,
        common_protocol_manifest={
            **bundle.common_protocol_manifest,
            "metadata": common_summary,
        },
        result_analysis_summary=analysis_summary,
        result_analysis_manifest={
            **bundle.result_analysis_manifest,
            "metadata": analysis_summary,
        },
    )


def ready_bundle() -> ResultClosureGateInput:
    """复制一次构造的只读模板,避免轻量测试重复执行统计计算."""

    return deepcopy(_ready_bundle_template())


def write_json(path: Path, payload: object) -> None:
    """写出测试 JSON 文件."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def write_jsonl(path: Path, rows: tuple[dict[str, object], ...]) -> None:
    """写出测试 JSONL 文件."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ).encode("utf-8")
    )


def write_csv(path: Path, rows: tuple[dict[str, object], ...]) -> None:
    """写出测试 CSV 文件."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_bundle_inputs(root: Path, bundle: ResultClosureGateInput) -> None:
    """按正式默认路径写出脚本测试需要的输入证据."""

    repository_root = Path(__file__).resolve().parents[2]
    for relative_path in (
        Path("configs/prompt_source_registry.json"),
        Path("configs/prompt_selection_manifest.jsonl"),
        Path(RUN_DEFAULTS[SCALE]["prompt_file"]),
    ):
        target_path = root / relative_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_bytes((repository_root / relative_path).read_bytes())
    json_paths = {
        f"outputs/image_only_dataset_runtime/{SCALE}/dataset_runtime_summary.json": bundle.evidence_audit_runtime_report,
        f"outputs/image_only_dataset_runtime/{SCALE}/manifest.local.json": bundle.evidence_audit_runtime_manifest,
        f"outputs/attack_matrix/{SCALE}/attack_manifest.json": bundle.attack_report,
        f"outputs/attack_matrix/{SCALE}/manifest.local.json": bundle.attack_manifest,
        f"outputs/fixed_fpr_threshold_audit/{SCALE}/threshold_audit_report.json": bundle.threshold_audit_report,
        f"outputs/fixed_fpr_threshold_audit/{SCALE}/manifest.local.json": bundle.threshold_audit_manifest,
        f"outputs/paper_result_closure/{SCALE}/closure_input_lock.json": bundle.closure_input_lock,
        f"outputs/paper_result_closure/{SCALE}/input_lock_manifest.local.json": bundle.closure_input_lock_manifest,
        f"outputs/official_reference_fidelity_evidence/{SCALE}/official_reference_fidelity_evidence_summary.json": bundle.official_reference_fidelity_summary,
        f"outputs/official_reference_fidelity_evidence/{SCALE}/manifest.local.json": bundle.official_reference_fidelity_manifest,
        f"outputs/primary_baseline_evidence/{SCALE}/primary_baseline_evidence_summary.json": bundle.primary_baseline_evidence_summary,
        f"outputs/primary_baseline_evidence/{SCALE}/manifest.local.json": bundle.primary_baseline_evidence_manifest,
        f"outputs/external_baseline_comparison/{SCALE}/baseline_runtime_report.json": bundle.baseline_report,
        f"outputs/external_baseline_comparison/{SCALE}/manifest.local.json": bundle.baseline_manifest,
        f"outputs/pilot_paper_fixed_fpr_results/{SCALE}/pilot_paper_result_record_summary.json": bundle.result_record_summary,
        f"outputs/pilot_paper_fixed_fpr_results/{SCALE}/pilot_paper_result_import_validation_report.json": bundle.result_record_validation_report,
        f"outputs/pilot_paper_fixed_fpr_results/{SCALE}/manifest.local.json": bundle.result_record_manifest,
        f"outputs/pilot_paper_fixed_fpr_common_protocol/{SCALE}/pilot_paper_common_protocol_summary.json": bundle.common_protocol_summary,
        f"outputs/pilot_paper_fixed_fpr_common_protocol/{SCALE}/pilot_paper_result_import_schema.json": bundle.common_protocol_schema,
        f"outputs/pilot_paper_fixed_fpr_common_protocol/{SCALE}/manifest.local.json": bundle.common_protocol_manifest,
        f"outputs/pilot_paper_result_analysis/{SCALE}/result_analysis_summary.json": bundle.result_analysis_summary,
        f"outputs/pilot_paper_result_analysis/{SCALE}/manifest.local.json": bundle.result_analysis_manifest,
        f"outputs/paired_superiority_analysis/{SCALE}/paired_superiority_summary.json": bundle.paired_superiority_summary,
        f"outputs/paired_superiority_analysis/{SCALE}/manifest.local.json": bundle.paired_superiority_manifest,
        f"outputs/formal_mechanism_ablation/{SCALE}/ablation_claim_summary.json": bundle.ablation_summary,
        f"outputs/formal_mechanism_ablation/{SCALE}/manifest.local.json": bundle.ablation_manifest,
        f"outputs/formal_mechanism_ablation/{SCALE}/per_ablation_frozen_protocols.json": bundle.ablation_frozen_protocols,
        f"outputs/formal_mechanism_ablation/{SCALE}/mechanism_necessity_summary.json": bundle.ablation_necessity_summary,
        f"outputs/dataset_level_quality/{SCALE}/dataset_quality_summary.json": bundle.dataset_quality_summary,
        f"outputs/dataset_level_quality/{SCALE}/dataset_quality_formal_feature_import_report.json": bundle.dataset_quality_feature_report,
        f"outputs/dataset_level_quality/{SCALE}/manifest.local.json": bundle.dataset_quality_manifest,
        f"outputs/paper_artifact_evidence_audit/{SCALE}/artifact_builder_readiness_report.json": bundle.evidence_builder_report,
        f"outputs/paper_artifact_evidence_audit/{SCALE}/submission_blocker_report.json": bundle.evidence_blocker_report,
        f"outputs/paper_artifact_evidence_audit/{SCALE}/artifact_data_validation_report.json": bundle.artifact_data_validation_report,
        f"outputs/paper_artifact_evidence_audit/{SCALE}/manifest.local.json": bundle.evidence_audit_manifest,
        f"outputs/submission_readiness/{SCALE}/readiness_blocker_report.json": bundle.submission_readiness_report,
        f"outputs/submission_readiness/{SCALE}/submission_readiness_manifest.local.json": bundle.submission_readiness_manifest,
        f"outputs/evidence_closure_entry_review/{SCALE}/entry_review_report.json": bundle.entry_review_report,
        f"outputs/evidence_closure_entry_review/{SCALE}/manifest.local.json": bundle.entry_review_manifest,
    }
    for relative_path, payload in json_paths.items():
        write_json(root / relative_path, payload)
    write_csv(
        root / f"outputs/fixed_fpr_threshold_audit/{SCALE}/threshold_audit_rows.csv",
        bundle.threshold_audit_rows,
    )
    write_jsonl(
        root
        / f"outputs/attack_matrix/{SCALE}/attack_detection_records.jsonl",
        bundle.attack_detection_records,
    )
    write_jsonl(
        root
        / f"outputs/attack_matrix/{SCALE}/attacked_image_registry.jsonl",
        bundle.attacked_image_registry,
    )
    write_csv(
        root / f"outputs/attack_matrix/{SCALE}/attack_family_metrics.csv",
        bundle.attack_family_metrics,
    )
    write_jsonl(
        root
        / f"outputs/official_reference_fidelity_evidence/{SCALE}/official_reference_fidelity_evidence_records.jsonl",
        bundle.official_reference_fidelity_records,
    )
    write_jsonl(
        root / f"outputs/primary_baseline_evidence/{SCALE}/primary_baseline_evidence_records.jsonl",
        bundle.primary_baseline_evidence_records,
    )
    write_jsonl(
        root / f"outputs/pilot_paper_fixed_fpr_results/{SCALE}/pilot_paper_result_records.jsonl",
        bundle.result_records,
    )
    write_csv(
        root
        / f"outputs/pilot_paper_fixed_fpr_results/{SCALE}/pilot_paper_result_template_coverage.csv",
        bundle.result_record_template_coverage,
    )
    write_jsonl(
        root / f"outputs/paired_superiority_analysis/{SCALE}/paired_outcomes.jsonl",
        bundle.paired_outcomes,
    )
    write_csv(
        root
        / f"outputs/paired_superiority_analysis/{SCALE}/paired_superiority_table.csv",
        bundle.paired_superiority_rows,
    )
    write_jsonl(
        root
        / f"outputs/formal_mechanism_ablation/{SCALE}/runtime_rerun_records.jsonl",
        bundle.ablation_runtime_records,
    )
    write_jsonl(
        root
        / f"outputs/formal_mechanism_ablation/{SCALE}/formal_detection_records.jsonl",
        bundle.ablation_detection_records,
    )
    write_csv(
        root
        / f"outputs/formal_mechanism_ablation/{SCALE}/mechanism_necessity_statistics.csv",
        bundle.ablation_necessity_rows,
    )
    write_csv(
        root / f"outputs/dataset_level_quality/{SCALE}/dataset_quality_metrics.csv",
        bundle.dataset_quality_metrics,
    )
    write_jsonl(
        root
        / f"outputs/dataset_level_quality/{SCALE}/dataset_quality_image_records.jsonl",
        bundle.dataset_quality_image_records,
    )
    write_jsonl(
        root
        / f"outputs/dataset_level_quality/{SCALE}/dataset_quality_image_resolution_records.jsonl",
        bundle.dataset_quality_image_resolution_records,
    )
    image_record_by_path = {
        str(record[field_name]): (
            str(record["prompt_id"]),
            "source" if field_name == "source_image_path" else "comparison",
        )
        for record in bundle.dataset_quality_image_records
        for field_name in ("source_image_path", "comparison_image_path")
    }
    for resolution in bundle.dataset_quality_image_resolution_records:
        resolved_path = str(resolution["resolved_image_path"])
        prompt_id, role = image_record_by_path[
            str(resolution["requested_image_path"])
        ]
        image_path = root / resolved_path
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.write_bytes(
            f"quality-image:{prompt_id}:{role}".encode("utf-8")
        )
    feature_records_path = (
        root
        / f"outputs/dataset_level_quality/{SCALE}/dataset_quality_formal_feature_records.jsonl"
    )
    feature_records_path.parent.mkdir(parents=True, exist_ok=True)
    feature_records_path.write_bytes(
        "".join(
            json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
            for record in bundle.dataset_quality_feature_records
        ).encode("utf-8")
    )
    result_analysis_payload_bytes = {
        RESULT_ANALYSIS_PAYLOAD_PATH_MAP["main_confidence_interval_table"]: (
            csv_bytes(bundle.result_analysis_confidence_interval_rows)
        ),
        RESULT_ANALYSIS_PAYLOAD_PATH_MAP["per_attack_superiority_table"]: (
            csv_bytes(bundle.result_analysis_per_attack_superiority_rows)
        ),
        RESULT_ANALYSIS_PAYLOAD_PATH_MAP["failure_case_records"]: (
            "".join(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                for row in bundle.result_analysis_failure_case_rows
            ).encode("utf-8")
        ),
        bundle.result_analysis_failure_figure_path: (
            bundle.result_analysis_failure_case_svg_text.encode("utf-8")
        ),
    }
    for relative_path, payload in {
        **official_reference_source_payloads(),
        **artifact_source_payloads(
            bundle.dataset_quality_metrics,
            bundle.attack_family_metrics,
            bundle.ablation_necessity_rows,
            bundle.result_analysis_baseline_comparison_rows,
        ),
        **result_analysis_payload_bytes,
        **METHOD_OBSERVATION_SOURCE_PAYLOADS,
    }.items():
        source_path = root / relative_path
        source_path.parent.mkdir(parents=True, exist_ok=True)
        source_path.write_bytes(payload)
    prompt_path = root / "configs/paper_main_probe_paper_prompts.txt"
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_path.write_text(
        "\n".join(f"a governed prompt {index}" for index in range(PROMPT_COUNT)) + "\n",
        encoding="utf-8",
    )
    write_paper_artifact_evidence_audit_outputs(
        root=root,
        prompt_contract=test_prompt_contract(root),
    )


def test_prompt_contract(root: Path) -> PaperRunPromptContract:
    """显式声明 result closure 临时 Prompt 的测试依赖。"""

    relative_path = Path("configs/paper_main_probe_paper_prompts.txt")
    path = root / relative_path
    return PaperRunPromptContract(
        run_name=SCALE,
        prompt_file=relative_path.as_posix(),
        expected_prompt_count=PROMPT_COUNT,
        prompt_file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )


@pytest.mark.quick
def test_result_closure_gate_passes_only_when_all_semantic_evidence_is_ready() -> None:
    """完整逐攻击披露可闭合, 不要求每个攻击均形成显著优势."""

    bundle = ready_bundle()
    checks = build_result_closure_gate_checks(bundle)
    report = build_result_closure_gate_report(bundle, checks)

    assert report["result_closure_ready"] is True
    assert report["evidence_closure_allowed"] is True
    assert report["closure_decision"] == "pass"
    assert report["blocked_check_count"] == 0
    assert all(row["check_status"] == "pass" for row in checks)
    assert bundle.result_analysis_summary["superiority_claim_ready_count"] == 0
    assert bundle.result_analysis_summary["universal_per_attack_superiority_claim_ready"] is False
    assert len(bundle.official_reference_fidelity_records) == 3
    assert len(bundle.paired_outcomes) == 4 * TEST_COUNT * len(ATTACK_SPECS)
    assert len(bundle.paired_superiority_rows) == 4
    assert len(bundle.artifact_data_validation_report["source_paths"]) == 12
    assert bundle.entry_review_report["entry_review_decision"] == "ready_for_evidence_closure"


@pytest.mark.quick
def test_result_closure_gate_rejects_ablation_detection_atom_drift() -> None:
    """消融检测原子的冻结判定被篡改时必须阻断逐 Prompt 重建。"""

    bundle = ready_bundle()
    records = list(bundle.ablation_detection_records)
    forged_record = dict(records[0])
    forged_record["formal_evidence_positive"] = not bool(
        forged_record["formal_evidence_positive"]
    )
    records[0] = forged_record
    forged = replace(bundle, ablation_detection_records=tuple(records))

    checks = build_result_closure_gate_checks(forged)

    rebuild_check = next(
        row
        for row in checks
        if row["check_id"] == "ablation_atomic_record_rebuild_ready"
    )
    assert rebuild_check["check_status"] == "blocked"


@pytest.mark.quick
def test_result_closure_gate_rejects_ablation_atom_file_sha_drift() -> None:
    """解析内容未变时，正式消融 atom 文件字节 SHA 漂移仍必须阻断。"""

    bundle = ready_bundle()
    path = (
        f"outputs/formal_mechanism_ablation/{SCALE}/"
        "formal_detection_records.jsonl"
    )
    bundle.source_file_sha256[path] = "f" * 64

    checks = build_result_closure_gate_checks(bundle)

    rebuild_check = next(
        row
        for row in checks
        if row["check_id"] == "ablation_atomic_record_rebuild_ready"
    )
    assert rebuild_check["check_status"] == "blocked"


@pytest.mark.quick
def test_result_closure_gate_rejects_feature_image_identity_drift() -> None:
    """正式 feature 脱离图像记录路径时必须阻断来源身份重建。"""

    bundle = ready_bundle()
    records = list(bundle.dataset_quality_feature_records)
    forged_record = dict(records[0])
    forged_record["image_path"] = (
        "outputs/dataset_level_quality/probe_paper/images/forged.png"
    )
    records[0] = forged_record
    forged = replace(bundle, dataset_quality_feature_records=tuple(records))

    checks = build_result_closure_gate_checks(forged)

    rebuild_check = next(
        row
        for row in checks
        if row["check_id"]
        == "dataset_quality_feature_identity_rebuild_ready"
    )
    assert rebuild_check["check_status"] == "blocked"


@pytest.mark.quick
@pytest.mark.parametrize("mutation", ("pair_role", "duplicate_path"))
def test_result_closure_gate_rejects_quality_pair_identity_forgery(
    mutation: str,
) -> None:
    """同步重签记录也不得伪造 clean/watermarked 语义或复用样本路径."""

    bundle = ready_bundle()
    records = [dict(record) for record in bundle.dataset_quality_image_records]
    forged_record = records[1]
    if mutation == "pair_role":
        forged_record["image_pair_role"] = "clean_to_attacked"
    else:
        forged_record["source_image_path"] = records[0]["source_image_path"]
        forged_record["source_image_digest"] = records[0][
            "source_image_digest"
        ]
    payload = {
        key: value
        for key, value in forged_record.items()
        if key
        not in {
            "dataset_quality_record_id",
            "dataset_quality_record_digest",
        }
    }
    digest = build_stable_digest(payload)
    forged_record["dataset_quality_record_digest"] = digest
    forged_record["dataset_quality_record_id"] = (
        f"dataset_quality_record_{digest[:16]}"
    )
    forged = replace(bundle, dataset_quality_image_records=tuple(records))

    checks = build_result_closure_gate_checks(forged)

    rebuild_check = next(
        row
        for row in checks
        if row["check_id"]
        == "dataset_quality_feature_identity_rebuild_ready"
    )
    assert rebuild_check["check_status"] == "blocked"


@pytest.mark.quick
def test_result_closure_gate_rejects_actual_quality_image_sha_drift() -> None:
    """图像解析记录的声明 SHA 不得覆盖闭合侧即时文件摘要。"""

    bundle = ready_bundle()
    image_path = str(
        bundle.dataset_quality_image_resolution_records[0][
            "resolved_image_path"
        ]
    )
    bundle.source_file_sha256[image_path] = "f" * 64

    checks = build_result_closure_gate_checks(bundle)

    rebuild_check = next(
        row
        for row in checks
        if row["check_id"]
        == "dataset_quality_feature_identity_rebuild_ready"
    )
    assert rebuild_check["check_status"] == "blocked"


@pytest.mark.quick
def test_result_closure_gate_rejects_result_analysis_cell_drift() -> None:
    """结果分析 CI 表任一单元偏离正式记录时必须阻断语义闭合。"""

    bundle = ready_bundle()
    rows = list(bundle.result_analysis_confidence_interval_rows)
    forged_row = dict(rows[0])
    forged_row["true_positive_rate"] = 0.123
    rows[0] = forged_row
    forged = replace(
        bundle,
        result_analysis_confidence_interval_rows=tuple(rows),
    )

    checks = build_result_closure_gate_checks(forged)

    rebuild_check = next(
        row
        for row in checks
        if row["check_id"] == "result_analysis_semantic_rebuild_ready"
    )
    assert rebuild_check["check_status"] == "blocked"


@pytest.mark.quick
def test_result_closure_gate_rejects_resigned_semantic_digest_forgery() -> None:
    """summary 与 manifest 同步重签也不得替代闭合侧独立语义摘要。"""

    bundle = ready_bundle()
    forged_digest = "f" * 64
    bundle.result_analysis_summary[
        "result_analysis_semantic_rebuild_digest"
    ] = forged_digest
    bundle.result_analysis_manifest["metadata"][
        "result_analysis_semantic_rebuild_digest"
    ] = forged_digest
    manifest_config = build_result_analysis_manifest_config(
        bundle.result_analysis_summary
    )
    bundle.result_analysis_manifest["config"] = manifest_config
    bundle.result_analysis_manifest["config_digest"] = build_stable_digest(
        manifest_config
    )

    checks = build_result_closure_gate_checks(bundle)

    rebuild_check = next(
        row
        for row in checks
        if row["check_id"] == "result_analysis_semantic_rebuild_ready"
    )
    assert rebuild_check["check_status"] == "blocked"


@pytest.mark.quick
def test_result_closure_gate_rejects_ablation_raw_record_statistic_drift() -> None:
    """消融 CSV/summary 未随逐 Prompt 原始记录变化时必须 fail-closed."""

    bundle = ready_bundle()
    records = [dict(record) for record in bundle.ablation_runtime_records]
    target_index = next(
        index
        for index, record in enumerate(records)
        if record["ablation_id"] == "complete_method" and record["split"] == "test"
    )
    records[target_index]["attacked_positive_rate"] = 0.5
    forged = replace(bundle, ablation_runtime_records=tuple(records))

    checks = build_result_closure_gate_checks(forged)

    rebuild_check = next(
        row
        for row in checks
        if row["check_id"] == "ablation_raw_record_rebuild_ready"
    )
    assert rebuild_check["check_status"] == "blocked"


@pytest.mark.quick
def test_result_closure_gate_rejects_self_declared_ablation_split_relabeling() -> None:
    """重算全部派生统计也不得把非 test Prompt 伪装成正式 test 样本。"""

    bundle = ready_bundle()
    records = [dict(record) for record in bundle.ablation_runtime_records]
    calibration_prompt_id = next(
        prompt_id
        for prompt_id, split in PROMPT_SPLIT_BY_ID.items()
        if split == "calibration"
    )
    test_prompt_id = next(
        prompt_id
        for prompt_id, split in PROMPT_SPLIT_BY_ID.items()
        if split == "test"
    )
    for record in records:
        if record["prompt_id"] == calibration_prompt_id:
            record["split"] = "test"
        elif record["prompt_id"] == test_prompt_id:
            record["split"] = "calibration"
    variant_ids = tuple(
        ablation_id
        for ablation_id in FORMAL_RUNTIME_RERUN_ABLATION_IDS
        if ablation_id != "complete_method"
    )
    forged_rows, forged_necessity_summary = build_ablation_necessity_statistics(
        records,
        expected_ablation_ids=variant_ids,
        expected_paired_prompt_count=TEST_COUNT,
    )
    forged_claim_summary = {
        **bundle.ablation_summary,
        **{
            field_name: field_value
            for field_name, field_value in forged_necessity_summary.items()
            if field_name != "supports_paper_claim"
        },
    }
    manifest_config = {
        **bundle.ablation_manifest["config"],
        "record_digest": build_stable_digest(records),
        "necessity_statistic_rows_digest": forged_necessity_summary[
            "necessity_statistic_rows_digest"
        ],
        "necessity_summary_digest": build_stable_digest(
            forged_necessity_summary
        ),
    }
    forged_manifest = {
        **bundle.ablation_manifest,
        "config": manifest_config,
        "config_digest": build_stable_digest(manifest_config),
    }
    forged = replace(
        bundle,
        ablation_summary=forged_claim_summary,
        ablation_manifest=forged_manifest,
        ablation_runtime_records=tuple(records),
        ablation_necessity_rows=tuple(forged_rows),
        ablation_necessity_summary=forged_necessity_summary,
    )

    checks = build_result_closure_gate_checks(forged)

    rebuild_check = next(
        row
        for row in checks
        if row["check_id"] == "ablation_raw_record_rebuild_ready"
    )
    assert rebuild_check["check_status"] == "blocked"


@pytest.mark.quick
def test_result_closure_gate_rejects_dataset_quality_metric_value_drift() -> None:
    """FID/KID 表值脱离正式 feature records 时必须 fail-closed."""

    bundle = ready_bundle()
    rows = [dict(row) for row in bundle.dataset_quality_metrics]
    rows[0]["quality_metric_value"] = str(
        float(rows[0]["quality_metric_value"]) + 0.1
    )
    forged = replace(bundle, dataset_quality_metrics=tuple(rows))

    checks = build_result_closure_gate_checks(forged)

    rebuild_check = next(
        row
        for row in checks
        if row["check_id"] == "dataset_quality_feature_record_rebuild_ready"
    )
    assert rebuild_check["check_status"] == "blocked"


@pytest.mark.quick
def test_result_closure_gate_rejects_full_sample_kid_std_drift() -> None:
    """全样本 KID std 必须精确为0, 不能被一般浮点容差掩盖。"""

    bundle = ready_bundle()
    rows = [dict(row) for row in bundle.dataset_quality_metrics]
    kid_std_row = next(
        row for row in rows if row["quality_metric_name"] == "kid_std"
    )
    kid_std_row["quality_metric_value"] = "1e-12"
    forged = replace(bundle, dataset_quality_metrics=tuple(rows))

    checks = build_result_closure_gate_checks(forged)

    rebuild_check = next(
        row
        for row in checks
        if row["check_id"] == "dataset_quality_feature_record_rebuild_ready"
    )
    assert rebuild_check["check_status"] == "blocked"


@pytest.mark.quick
def test_result_closure_gate_rejects_negative_or_legacy_kid_std() -> None:
    """负标准差和旧 `kid` 行身份都不得通过正式质量闭合。"""

    bundle = ready_bundle()
    for mutation in ("negative_std", "legacy_kid"):
        rows = [dict(row) for row in bundle.dataset_quality_metrics]
        kid_std_row = next(
            row for row in rows if row["quality_metric_name"] == "kid_std"
        )
        if mutation == "negative_std":
            kid_std_row["quality_metric_value"] = "-0.001"
        else:
            kid_std_row["quality_metric_name"] = "kid"
            kid_std_row["paper_metric_name"] = "kid"
        forged = replace(bundle, dataset_quality_metrics=tuple(rows))

        checks = build_result_closure_gate_checks(forged)

        quality_check = next(
            row
            for row in checks
            if row["check_id"] == "formal_fid_kid_ready"
        )
        assert quality_check["check_status"] == "blocked"


@pytest.mark.quick
def test_result_closure_gate_rejects_dataset_quality_feature_record_drift() -> None:
    """feature vector 被替换但正式指标表未重算时必须 fail-closed."""

    bundle = ready_bundle()
    records = [dict(record) for record in bundle.dataset_quality_feature_records]
    records[0] = {
        **records[0],
        "feature_vector": [
            float(records[0]["feature_vector"][0]) + 0.25,
            float(records[0]["feature_vector"][1]),
        ],
    }
    forged = replace(bundle, dataset_quality_feature_records=tuple(records))

    checks = build_result_closure_gate_checks(forged)

    rebuild_check = next(
        row
        for row in checks
        if row["check_id"] == "dataset_quality_feature_record_rebuild_ready"
    )
    assert rebuild_check["check_status"] == "blocked"


@pytest.mark.quick
def test_result_closure_gate_rejects_result_analysis_payload_byte_drift() -> None:
    """结果分析图表字节摘要变化时不得继续信任 ready 布尔值."""

    bundle = ready_bundle()
    source_file_sha256 = dict(bundle.source_file_sha256)
    source_file_sha256[
        RESULT_ANALYSIS_PAYLOAD_PATH_MAP["failure_case_figure"]
    ] = "f" * 64
    forged = replace(bundle, source_file_sha256=source_file_sha256)

    checks = build_result_closure_gate_checks(forged)

    result_analysis_check = next(
        row for row in checks if row["check_id"] == "result_analysis_ready"
    )
    assert result_analysis_check["check_status"] == "blocked"


@pytest.mark.quick
def test_result_closure_gate_rejects_incomplete_result_analysis_payload_roles() -> None:
    """summary 或 manifest 缺少任一固定 payload 角色时必须 fail-closed."""

    bundle = ready_bundle()
    summary = dict(bundle.result_analysis_summary)
    path_map = dict(summary["result_analysis_payload_path_map"])
    path_map.pop("failure_case_figure")
    summary["result_analysis_payload_path_map"] = path_map
    forged = replace(bundle, result_analysis_summary=summary)

    checks = build_result_closure_gate_checks(forged)

    result_analysis_check = next(
        row for row in checks if row["check_id"] == "result_analysis_ready"
    )
    assert result_analysis_check["check_status"] == "blocked"


@pytest.mark.quick
def test_result_closure_gate_rejects_synchronized_fpr_forgery() -> None:
    """同步重签 record 和下游摘要也不得覆盖原始 clean negative FPR."""

    bundle = ready_bundle()
    records = [dict(row) for row in bundle.result_records]
    record = next(
        row for row in records if row["method_id"] == "slm_wm_current"
    )
    record["false_positive_rate"] = 0.05
    low, high = bounded_hoeffding_confidence_interval(
        0.05,
        TEST_COUNT,
        0.95,
    )
    record["false_positive_rate_ci_low"] = low
    record["false_positive_rate_ci_high"] = high
    payload = {
        key: value
        for key, value in record.items()
        if key
        not in {
            "pilot_paper_result_record_digest",
            "pilot_paper_result_record_id",
        }
    }
    digest = build_stable_digest(payload)
    record["pilot_paper_result_record_digest"] = digest
    record["pilot_paper_result_record_id"] = (
        f"pilot_paper_result_record_{digest[:16]}"
    )
    blocked_bundle = synchronize_result_record_evidence(
        bundle,
        tuple(records),
    )

    report = build_result_closure_gate_report(
        blocked_bundle,
        build_result_closure_gate_checks(blocked_bundle),
    )

    assert "paired_superiority_ready" in report["blocked_check_ids"]


@pytest.mark.quick
def test_result_closure_gate_rejects_synchronized_quality_forgery() -> None:
    """质量均值和区间即使同步重签也必须等于原始攻击 observation."""

    bundle = ready_bundle()
    records = [dict(row) for row in bundle.result_records]
    record = next(
        row for row in records if row["method_id"] == "tree_ring"
    )
    record["quality_score_mean"] = 0.99
    low, high = bounded_hoeffding_confidence_interval(
        0.99,
        TEST_COUNT,
        0.95,
        lower_bound=-1.0,
        upper_bound=1.0,
    )
    record["quality_score_ci_low"] = low
    record["quality_score_ci_high"] = high
    payload = {
        key: value
        for key, value in record.items()
        if key
        not in {
            "pilot_paper_result_record_digest",
            "pilot_paper_result_record_id",
        }
    }
    digest = build_stable_digest(payload)
    record["pilot_paper_result_record_digest"] = digest
    record["pilot_paper_result_record_id"] = (
        f"pilot_paper_result_record_{digest[:16]}"
    )
    blocked_bundle = synchronize_result_record_evidence(
        bundle,
        tuple(records),
    )

    report = build_result_closure_gate_report(
        blocked_bundle,
        build_result_closure_gate_checks(blocked_bundle),
    )

    assert "paired_superiority_ready" in report["blocked_check_ids"]


@pytest.mark.quick
def test_result_closure_gate_rejects_quality_ci_with_three_group_denominator() -> None:
    """质量均值不变时, 把 CI 分母伪造为三组记录总数也必须被独立复算阻断."""

    bundle = ready_bundle()
    records = [dict(row) for row in bundle.result_records]
    record = next(
        row for row in records if row["method_id"] == "tree_ring"
    )
    low, high = bounded_hoeffding_confidence_interval(
        float(record["quality_score_mean"]),
        3 * TEST_COUNT,
        0.95,
        lower_bound=-1.0,
        upper_bound=1.0,
    )
    record["quality_score_ci_low"] = low
    record["quality_score_ci_high"] = high
    payload = {
        key: value
        for key, value in record.items()
        if key
        not in {
            "pilot_paper_result_record_digest",
            "pilot_paper_result_record_id",
        }
    }
    digest = build_stable_digest(payload)
    record["pilot_paper_result_record_digest"] = digest
    record["pilot_paper_result_record_id"] = (
        f"pilot_paper_result_record_{digest[:16]}"
    )
    blocked_bundle = synchronize_result_record_evidence(
        bundle,
        tuple(records),
    )

    report = build_result_closure_gate_report(
        blocked_bundle,
        build_result_closure_gate_checks(blocked_bundle),
    )

    assert "paired_superiority_ready" in report["blocked_check_ids"]


@pytest.mark.quick
def test_result_closure_gate_rejects_synchronized_unit_range_ssim_ci() -> None:
    """同步重签旧 [0,1] SSIM 区间也必须被独立 signed-range 复算阻断."""

    bundle = ready_bundle()
    records = [dict(row) for row in bundle.result_records]
    record = next(
        row for row in records if row["method_id"] == "tree_ring"
    )
    low, high = bounded_hoeffding_confidence_interval(
        float(record["quality_score_mean"]),
        TEST_COUNT,
        0.95,
    )
    record["quality_score_ci_low"] = low
    record["quality_score_ci_high"] = high
    payload = {
        key: value
        for key, value in record.items()
        if key
        not in {
            "pilot_paper_result_record_digest",
            "pilot_paper_result_record_id",
        }
    }
    digest = build_stable_digest(payload)
    record["pilot_paper_result_record_digest"] = digest
    record["pilot_paper_result_record_id"] = (
        f"pilot_paper_result_record_{digest[:16]}"
    )
    blocked_bundle = synchronize_result_record_evidence(
        bundle,
        tuple(records),
    )

    report = build_result_closure_gate_report(
        blocked_bundle,
        build_result_closure_gate_checks(blocked_bundle),
    )

    assert "paired_superiority_ready" in report["blocked_check_ids"]


@pytest.mark.quick
def test_result_closure_gate_rejects_synchronized_confidence_level_forgery() -> None:
    """协议置信度失配即使同步重签自声明链也必须由独立导入校验阻断."""

    bundle = ready_bundle()
    records = [dict(row) for row in bundle.result_records]
    record = records[0]
    record["confidence_level"] = 0.90
    payload = {
        key: value
        for key, value in record.items()
        if key
        not in {
            "pilot_paper_result_record_digest",
            "pilot_paper_result_record_id",
        }
    }
    digest = build_stable_digest(payload)
    record["pilot_paper_result_record_digest"] = digest
    record["pilot_paper_result_record_id"] = (
        f"pilot_paper_result_record_{digest[:16]}"
    )
    blocked_bundle = synchronize_result_record_evidence(
        bundle,
        tuple(records),
    )

    assert (
        blocked_bundle.result_record_validation_report[
            "pilot_paper_result_import_ready"
        ]
        is False
    )
    report = build_result_closure_gate_report(
        blocked_bundle,
        build_result_closure_gate_checks(blocked_bundle),
    )

    assert "result_records_ready" in report["blocked_check_ids"]


@pytest.mark.quick
def test_result_closure_gate_rejects_incomplete_main_quality_records() -> None:
    """任一正式 attacked positive 缺少 SSIM 时不得按剩余子集计算质量."""

    bundle = ready_bundle()
    observations = {
        method_id: tuple(dict(row) for row in rows)
        for method_id, rows in bundle.paired_observation_records_by_method.items()
    }
    attacked_positive = next(
        row
        for row in observations["slm_wm"]
        if row.get("attack_id") and row.get("sample_role") == "positive_source"
    )
    attacked_positive.pop("source_to_evaluated_ssim")
    blocked_bundle = replace(
        bundle,
        paired_observation_records_by_method=observations,
    )

    report = build_result_closure_gate_report(
        blocked_bundle,
        build_result_closure_gate_checks(blocked_bundle),
    )

    assert "paired_superiority_ready" in report["blocked_check_ids"]


@pytest.mark.quick
def test_result_closure_gate_rejects_incomplete_baseline_quality_records() -> None:
    """任一 baseline attacked positive 缺少质量值时不得按0补齐."""

    bundle = ready_bundle()
    observations = {
        method_id: tuple(dict(row) for row in rows)
        for method_id, rows in bundle.paired_observation_records_by_method.items()
    }
    attacked_positive = next(
        row
        for row in observations["tree_ring"]
        if row.get("attack_id")
        and row.get("sample_role") == "attacked_positive"
    )
    attacked_positive.pop("quality_score")
    blocked_bundle = replace(
        bundle,
        paired_observation_records_by_method=observations,
    )

    report = build_result_closure_gate_report(
        blocked_bundle,
        build_result_closure_gate_checks(blocked_bundle),
    )

    assert "paired_superiority_ready" in report["blocked_check_ids"]


@pytest.mark.quick
def test_result_closure_gate_rejects_test_negative_prompt_replacement() -> None:
    """test negative 必须无重复地覆盖规范 test Prompt 集合."""

    bundle = ready_bundle()
    observations = {
        method_id: tuple(dict(row) for row in rows)
        for method_id, rows in bundle.paired_observation_records_by_method.items()
    }
    clean_negative = next(
        row
        for row in observations["slm_wm"]
        if row.get("split") == "test"
        and row.get("sample_role") == "clean_negative"
        and not row.get("attack_id")
    )
    clean_negative["prompt_id"] = "unexpected_test_prompt"
    blocked_bundle = replace(
        bundle,
        paired_observation_records_by_method=observations,
    )

    report = build_result_closure_gate_report(
        blocked_bundle,
        build_result_closure_gate_checks(blocked_bundle),
    )

    assert "paired_superiority_ready" in report["blocked_check_ids"]


@pytest.mark.quick
def test_result_closure_gate_rejects_calibration_prompt_replacement() -> None:
    """calibration negative 必须绑定规范 calibration Prompt 集合."""

    bundle = ready_bundle()
    observations = {
        method_id: tuple(dict(row) for row in rows)
        for method_id, rows in bundle.paired_observation_records_by_method.items()
    }
    calibration_negative = next(
        row
        for row in observations["tree_ring"]
        if row.get("split") == "calibration"
        and row.get("sample_role") == "clean_negative"
    )
    calibration_negative["prompt_id"] = "unexpected_calibration_prompt"
    blocked_bundle = replace(
        bundle,
        paired_observation_records_by_method=observations,
    )

    report = build_result_closure_gate_report(
        blocked_bundle,
        build_result_closure_gate_checks(blocked_bundle),
    )

    assert "paired_superiority_ready" in report["blocked_check_ids"]


@pytest.mark.quick
def test_result_closure_gate_rejects_result_source_digest_forgery() -> None:
    """来源摘要同步写入 record 与 manifest 后仍须匹配即时读取字节."""

    bundle = ready_bundle()
    records = [dict(row) for row in bundle.result_records]
    records[0]["baseline_result_source_digest"] = "f" * 64
    payload = {
        key: value
        for key, value in records[0].items()
        if key
        not in {
            "pilot_paper_result_record_digest",
            "pilot_paper_result_record_id",
        }
    }
    digest = build_stable_digest(payload)
    records[0]["pilot_paper_result_record_digest"] = digest
    records[0]["pilot_paper_result_record_id"] = (
        f"pilot_paper_result_record_{digest[:16]}"
    )
    blocked_bundle = synchronize_result_record_evidence(
        bundle,
        tuple(records),
    )

    report = build_result_closure_gate_report(
        blocked_bundle,
        build_result_closure_gate_checks(blocked_bundle),
    )

    assert "result_records_ready" in report["blocked_check_ids"]


@pytest.mark.quick
def test_result_closure_gate_rejects_attack_identity_relabeling() -> None:
    """攻击记录、registry 和 manifest 同步重签也不得后贴配置身份."""

    bundle = ready_bundle()
    records = [dict(row) for row in bundle.attack_detection_records]
    records[0]["attack_config_digest"] = "f" * 64
    digest = build_attack_record_digest(records[0])
    records[0]["attack_record_digest"] = digest
    records[0]["attack_record_id"] = f"attack_{digest[:24]}"
    registry = formal_attacked_image_registry(tuple(records))
    attack_manifest_config = build_attack_matrix_manifest_config(
        paper_run_name=SCALE,
        evaluation_boundary=bundle.attack_report["evaluation_boundary"],
        attack_configs=FORMAL_ATTACK_CONFIGS,
        attack_records=records,
    )
    blocked_bundle = replace(
        bundle,
        attack_detection_records=tuple(records),
        attacked_image_registry=registry,
        attack_manifest={
            **bundle.attack_manifest,
            "config": attack_manifest_config,
            "config_digest": build_stable_digest(attack_manifest_config),
        },
    )

    report = build_result_closure_gate_report(
        blocked_bundle,
        build_result_closure_gate_checks(blocked_bundle),
    )

    assert "attack_matrix_ready" in report["blocked_check_ids"]


@pytest.mark.quick
def test_result_closure_gate_rejects_synchronized_attack_prompt_duplication() -> None:
    """攻击记录同步重签后仍须无重复地覆盖规范 test Prompt 全集."""

    bundle = ready_bundle()
    records = [dict(row) for row in bundle.attack_detection_records]
    record = records[0]
    duplicate_source = next(
        candidate
        for candidate in records[1:]
        if candidate["attack_id"] == record["attack_id"]
        and candidate["sample_role"] == record["sample_role"]
    )
    record["prompt_id"] = duplicate_source["prompt_id"]
    record["run_id"] = duplicate_source["run_id"]
    digest = build_attack_record_digest(record)
    record["attack_record_digest"] = digest
    record["attack_record_id"] = f"attack_{digest[:24]}"
    registry = formal_attacked_image_registry(tuple(records))
    attack_manifest_config = build_attack_matrix_manifest_config(
        paper_run_name=SCALE,
        evaluation_boundary=bundle.attack_report["evaluation_boundary"],
        attack_configs=FORMAL_ATTACK_CONFIGS,
        attack_records=records,
    )
    blocked_bundle = replace(
        bundle,
        attack_detection_records=tuple(records),
        attacked_image_registry=registry,
        attack_manifest={
            **bundle.attack_manifest,
            "config": attack_manifest_config,
            "config_digest": build_stable_digest(attack_manifest_config),
        },
    )

    report = build_result_closure_gate_report(
        blocked_bundle,
        build_result_closure_gate_checks(blocked_bundle),
    )

    assert "attack_matrix_ready" in report["blocked_check_ids"]


@pytest.mark.quick
def test_result_closure_gate_rejects_attack_family_metric_tampering() -> None:
    """持久化攻击指标被修改后必须与原始 detection records 复算值冲突."""

    bundle = ready_bundle()
    metric_rows = [dict(row) for row in bundle.attack_family_metrics]
    metric_rows[0]["true_positive_rate"] = 0.5
    blocked_bundle = replace(
        bundle,
        attack_family_metrics=tuple(metric_rows),
    )

    report = build_result_closure_gate_report(
        blocked_bundle,
        build_result_closure_gate_checks(blocked_bundle),
    )

    assert "attack_matrix_ready" in report["blocked_check_ids"]


@pytest.mark.quick
def test_result_closure_gate_rejects_attack_family_metric_row_deletion() -> None:
    """攻击指标缺失任一正式攻击 ID 时必须阻断完整攻击矩阵."""

    bundle = ready_bundle()
    blocked_bundle = replace(
        bundle,
        attack_family_metrics=bundle.attack_family_metrics[1:],
    )

    report = build_result_closure_gate_report(
        blocked_bundle,
        build_result_closure_gate_checks(blocked_bundle),
    )

    assert "attack_matrix_ready" in report["blocked_check_ids"]


@pytest.mark.quick
def test_result_closure_gate_rejects_official_reference_source_digest_drift() -> None:
    """官方参考记录即使身份摘要自洽, 源文件摘要漂移仍必须阻断."""

    bundle = ready_bundle()
    records = [dict(row) for row in bundle.official_reference_fidelity_records]
    record = records[0]
    record["official_reference_source_artifact_digests"] = {
        **dict(record["official_reference_source_artifact_digests"]),
        "summary": "f" * 64,
    }
    payload = {
        key: value
        for key, value in record.items()
        if key
        not in {
            "official_reference_fidelity_record_id",
            "official_reference_fidelity_record_digest",
        }
    }
    digest = build_stable_digest(payload)
    record["official_reference_fidelity_record_digest"] = digest
    record["official_reference_fidelity_record_id"] = (
        f"{record['baseline_id']}_official_reference_fidelity_{digest[:16]}"
    )
    blocked_bundle = replace(
        bundle,
        official_reference_fidelity_records=tuple(records),
    )

    report = build_result_closure_gate_report(
        blocked_bundle,
        build_result_closure_gate_checks(blocked_bundle),
    )

    assert "official_reference_fidelity_evidence_ready" in report["blocked_check_ids"]


@pytest.mark.quick
def test_result_closure_gate_rejects_paired_prompt_key_drift() -> None:
    """配对结果重复一个 Prompt x attack 键时必须阻断总体优势证据."""

    bundle = ready_bundle()
    outcomes = [dict(row) for row in bundle.paired_outcomes]
    drift_index = len(ATTACK_REGISTRY)
    outcomes[drift_index]["prompt_id"] = outcomes[0]["prompt_id"]
    payload = {
        key: value
        for key, value in outcomes[drift_index].items()
        if key != "paired_outcome_digest"
    }
    outcomes[drift_index]["paired_outcome_digest"] = build_stable_digest(payload)
    blocked_bundle = replace(bundle, paired_outcomes=tuple(outcomes))

    report = build_result_closure_gate_report(
        blocked_bundle,
        build_result_closure_gate_checks(blocked_bundle),
    )

    assert "paired_superiority_ready" in report["blocked_check_ids"]


@pytest.mark.quick
def test_result_closure_gate_recomputes_forged_paired_statistics() -> None:
    """伪造统计行并同步全部自声明摘要后仍必须被独立复算阻断."""

    bundle = ready_bundle()
    rows = [dict(row) for row in bundle.paired_superiority_rows]
    rows[0]["mean_paired_true_positive_rate_difference"] = 0.75
    blocked_bundle = synchronize_paired_evidence(
        bundle,
        outcomes=bundle.paired_outcomes,
        rows=tuple(rows),
    )

    report = build_result_closure_gate_report(
        blocked_bundle,
        build_result_closure_gate_checks(blocked_bundle),
    )

    assert "paired_superiority_ready" in report["blocked_check_ids"]


@pytest.mark.quick
def test_result_closure_gate_rejects_noncanonical_paired_prompt_set() -> None:
    """Prompt 数量不变但集合身份漂移时不得闭合总体优势结论."""

    bundle = ready_bundle()
    outcomes = []
    replaced_prompt_id = TEST_PROMPT_IDS[0]
    for source in bundle.paired_outcomes:
        row = dict(source)
        if row["prompt_id"] == replaced_prompt_id:
            row["prompt_id"] = "foreign_test_prompt_id"
            row["paired_outcome_digest"] = build_stable_digest(
                {
                    key: value
                    for key, value in row.items()
                    if key != "paired_outcome_digest"
                }
            )
        outcomes.append(row)
    materialized_outcomes = tuple(outcomes)
    rows = tuple(
        build_paired_superiority_rows(
            materialized_outcomes,
            protocol_digest=bundle.paired_superiority_summary[
                "paired_superiority_protocol_digest"
            ],
        )
    )
    blocked_bundle = synchronize_paired_evidence(
        bundle,
        outcomes=materialized_outcomes,
        rows=rows,
    )

    report = build_result_closure_gate_report(
        blocked_bundle,
        build_result_closure_gate_checks(blocked_bundle),
    )

    assert "paired_superiority_ready" in report["blocked_check_ids"]


@pytest.mark.quick
def test_result_closure_gate_rejects_paired_result_rate_mismatch() -> None:
    """正式结果 TPR 与逐 Prompt 二元配对计数不一致时必须阻断."""

    bundle = ready_bundle()
    records = [dict(row) for row in bundle.result_records]
    baseline_record = next(
        row for row in records if row["method_id"] == "tree_ring"
    )
    baseline_record["true_positive_rate"] = 0.5
    digest = build_stable_digest(
        {
            key: value
            for key, value in baseline_record.items()
            if key
            not in {
                "pilot_paper_result_record_digest",
                "pilot_paper_result_record_id",
            }
        }
    )
    baseline_record["pilot_paper_result_record_digest"] = digest
    baseline_record["pilot_paper_result_record_id"] = (
        f"pilot_paper_result_record_{digest[:16]}"
    )
    blocked_bundle = replace(bundle, result_records=tuple(records))

    report = build_result_closure_gate_report(
        blocked_bundle,
        build_result_closure_gate_checks(blocked_bundle),
    )

    assert "paired_superiority_ready" in report["blocked_check_ids"]


@pytest.mark.quick
def test_result_closure_gate_rejects_prompt_pairing_reassignment() -> None:
    """逐攻击 TP 不变但 Prompt 对应关系被置换时必须由原始 observation 阻断."""

    bundle = ready_bundle()
    outcomes = [dict(row) for row in bundle.paired_outcomes]
    attack_id = ATTACK_REGISTRY[0]["attack_id"]
    targets = {
        TEST_PROMPT_IDS[0]: False,
        TEST_PROMPT_IDS[5]: True,
    }
    for row in outcomes:
        if (
            row["baseline_id"] == "tree_ring"
            and row["attack_id"] == attack_id
            and row["prompt_id"] in targets
        ):
            row["baseline_decision"] = targets[str(row["prompt_id"])]
            row["paired_difference"] = int(row["proposed_decision"]) - int(
                row["baseline_decision"]
            )
            row["paired_outcome_digest"] = build_stable_digest(
                {
                    key: value
                    for key, value in row.items()
                    if key != "paired_outcome_digest"
                }
            )
    materialized_outcomes = tuple(outcomes)
    rows = tuple(
        build_paired_superiority_rows(
            materialized_outcomes,
            protocol_digest=bundle.paired_superiority_summary[
                "paired_superiority_protocol_digest"
            ],
        )
    )
    blocked_bundle = synchronize_paired_evidence(
        bundle,
        outcomes=materialized_outcomes,
        rows=rows,
    )

    report = build_result_closure_gate_report(
        blocked_bundle,
        build_result_closure_gate_checks(blocked_bundle),
    )

    assert "paired_superiority_ready" in report["blocked_check_ids"]


@pytest.mark.quick
def test_result_closure_gate_rejects_observation_file_sha256_drift() -> None:
    """配对 manifest 声明的 observation 文件字节发生漂移时必须阻断."""

    bundle = ready_bundle()
    tree_ring_path = METHOD_OBSERVATION_SOURCE_PATH_MAP["tree_ring"]
    blocked_bundle = replace(
        bundle,
        source_file_sha256={
            **bundle.source_file_sha256,
            tree_ring_path: "f" * 64,
        },
    )

    report = build_result_closure_gate_report(
        blocked_bundle,
        build_result_closure_gate_checks(blocked_bundle),
    )

    assert "paired_superiority_ready" in report["blocked_check_ids"]


@pytest.mark.quick
def test_result_closure_gate_rejects_artifact_source_file_sha256_drift() -> None:
    """任一论文表图源文件即时 SHA-256 漂移时必须阻断证据审计."""

    bundle = ready_bundle()
    source_path = ARTIFACT_SOURCE_PATHS["roc_curve_points_ready"]
    blocked_bundle = replace(
        bundle,
        source_file_sha256={**bundle.source_file_sha256, source_path: "f" * 64},
    )

    report = build_result_closure_gate_report(
        blocked_bundle,
        build_result_closure_gate_checks(blocked_bundle),
    )

    assert "paper_evidence_audit_ready" in report["blocked_check_ids"]


@pytest.mark.quick
def test_result_closure_gate_rejects_blocked_entry_decision() -> None:
    """入口决定明确阻断时不得冒充自动证据闭合许可."""

    bundle = ready_bundle()
    blocked_bundle = replace(
        bundle,
        entry_review_report={
            **bundle.entry_review_report,
            "entry_review_decision": "blocked_before_evidence_closure",
        },
    )

    report = build_result_closure_gate_report(
        blocked_bundle,
        build_result_closure_gate_checks(blocked_bundle),
    )

    assert "evidence_closure_entry_ready" in report["blocked_check_ids"]


@pytest.mark.quick
def test_result_closure_gate_blocks_unscoped_fpr_and_inexact_attack_roles() -> None:
    """baseline、质量或攻击角色边界不精确时必须阻断证据闭合。"""

    bundle = ready_bundle()
    polluted_attack_report = {
        **bundle.attack_report,
        "performed_attack_record_count": int(bundle.attack_report["attack_record_count"]) - 1,
        "attack_split_role_counts": {
            **dict(bundle.attack_report["attack_split_role_counts"]),
            "jpeg_70|unexpected_role": TEST_COUNT,
        },
    }
    blocked_bundle = replace(
        bundle,
        attack_report=polluted_attack_report,
        baseline_report={**bundle.baseline_report, "target_fpr": 0.01},
        dataset_quality_summary={**bundle.dataset_quality_summary, "target_fpr": 0.01},
    )

    report = build_result_closure_gate_report(
        blocked_bundle,
        build_result_closure_gate_checks(blocked_bundle),
    )

    assert report["result_closure_ready"] is False
    assert "target_fpr_consistent" in report["blocked_check_ids"]
    assert "test_split_count_consistent" in report["blocked_check_ids"]
    assert "attack_matrix_ready" in report["blocked_check_ids"]


@pytest.mark.quick
def test_result_closure_gate_blocks_mixed_scope_and_entry_review_denial() -> None:
    """混入其他层级结果或入口拒绝时, 门禁必须 fail-closed。"""

    bundle = ready_bundle()
    mixed_record = dict(bundle.result_records[0])
    mixed_record["paper_claim_scale"] = "pilot_paper"
    mixed_digest_payload = {
        key: value
        for key, value in mixed_record.items()
        if key not in {"pilot_paper_result_record_digest", "pilot_paper_result_record_id"}
    }
    mixed_digest = build_stable_digest(mixed_digest_payload)
    mixed_record["pilot_paper_result_record_digest"] = mixed_digest
    mixed_record["pilot_paper_result_record_id"] = f"pilot_paper_result_record_{mixed_digest[:16]}"
    denied_entry = {**bundle.entry_review_report, "evidence_closure_allowed": False}
    incomplete_primary_evidence = {
        **bundle.primary_baseline_evidence_summary,
        "formal_result_ready_count": 3,
        "formal_result_ready_ids": ["tree_ring", "gaussian_shading", "shallow_diffuse"],
        "primary_baseline_formal_ready": False,
    }
    blocked_bundle = replace(
        bundle,
        result_records=(mixed_record,),
        primary_baseline_evidence_summary=incomplete_primary_evidence,
        entry_review_report=denied_entry,
    )

    checks = build_result_closure_gate_checks(blocked_bundle)
    report = build_result_closure_gate_report(blocked_bundle, checks)

    assert report["result_closure_ready"] is False
    assert report["closure_decision"] == "blocked"
    assert "current_run_scope_consistent" in report["blocked_check_ids"]
    assert "primary_baseline_evidence_ready" in report["blocked_check_ids"]
    assert "evidence_closure_entry_ready" in report["blocked_check_ids"]


@pytest.mark.quick
def test_result_closure_gate_rejects_per_method_threshold_digest_mismatch() -> None:
    """任一 baseline 正式记录脱离统一阈值审计时必须阻断."""

    bundle = ready_bundle()
    records = [dict(row) for row in bundle.result_records]
    tree_ring = next(row for row in records if row["method_id"] == "tree_ring")
    tree_ring["method_threshold_digest"] = "f" * 64
    payload = {
        key: value
        for key, value in tree_ring.items()
        if key not in {"pilot_paper_result_record_digest", "pilot_paper_result_record_id"}
    }
    digest = build_stable_digest(payload)
    tree_ring["pilot_paper_result_record_digest"] = digest
    tree_ring["pilot_paper_result_record_id"] = f"pilot_paper_result_record_{digest[:16]}"
    blocked_bundle = replace(bundle, result_records=tuple(records))

    report = build_result_closure_gate_report(
        blocked_bundle,
        build_result_closure_gate_checks(blocked_bundle),
    )

    assert "threshold_digest_consistent" in report["blocked_check_ids"]


@pytest.mark.quick
def test_result_closure_gate_rejects_record_set_digest_drift() -> None:
    """结果分析若登记了不同记录集合摘要, 不得进入闭合."""

    bundle = ready_bundle()
    blocked_bundle = replace(
        bundle,
        result_analysis_summary={
            **bundle.result_analysis_summary,
            "result_record_set_digest": "f" * 64,
        },
    )

    report = build_result_closure_gate_report(
        blocked_bundle,
        build_result_closure_gate_checks(blocked_bundle),
    )

    assert "result_record_set_provenance_consistent" in report["blocked_check_ids"]


@pytest.mark.quick
def test_result_closure_gate_rejects_primary_evidence_record_identity_drift() -> None:
    """primary baseline records 的来源分组或正文摘要漂移时必须阻断."""

    bundle = ready_bundle()
    records = [dict(row) for row in bundle.primary_baseline_evidence_records]
    records[0]["comparison_group"] = "supplemental"
    blocked_bundle = replace(bundle, primary_baseline_evidence_records=tuple(records))

    report = build_result_closure_gate_report(
        blocked_bundle,
        build_result_closure_gate_checks(blocked_bundle),
    )

    assert "primary_baseline_evidence_ready" in report["blocked_check_ids"]


@pytest.mark.quick
def test_result_closure_gate_rejects_closure_lock_provenance_drift() -> None:
    """共同协议传播了不同输入锁摘要时必须阻断."""

    bundle = ready_bundle()
    blocked_bundle = replace(
        bundle,
        common_protocol_schema={
            **bundle.common_protocol_schema,
            "closure_input_lock_digest": "f" * 64,
        },
    )

    report = build_result_closure_gate_report(
        blocked_bundle,
        build_result_closure_gate_checks(blocked_bundle),
    )

    assert "closure_input_provenance_consistent" in report["blocked_check_ids"]


@pytest.mark.quick
def test_result_closure_gate_rejects_six_item_ablation_summary() -> None:
    """旧的6项消融即使自行声明 ready 也不得通过完整正式集合门禁."""

    bundle = ready_bundle()
    six_ids = list(FORMAL_RUNTIME_RERUN_ABLATION_IDS[:6])
    six_item_summary = {
        **bundle.ablation_summary,
        "record_count": PROMPT_COUNT * 6,
        "ablation_count": 6,
        "per_ablation_calibration_count": 6,
        "generation_rerun_count": PROMPT_COUNT * 6,
        "expected_attack_and_detection_rerun_count": TEST_COUNT * 6,
        "attack_and_detection_rerun_count": TEST_COUNT * 6,
        "formal_attack_coverage_ready_count": PROMPT_COUNT * 6,
        "actual_ablation_ids": six_ids,
        "ablation_spec_digest": build_stable_digest(six_ids),
        "ablation_exact_set_ready": True,
    }
    blocked_bundle = replace(bundle, ablation_summary=six_item_summary)

    report = build_result_closure_gate_report(
        blocked_bundle,
        build_result_closure_gate_checks(blocked_bundle),
    )

    assert report["result_closure_ready"] is False
    assert "formal_ablation_ready" in report["blocked_check_ids"]


@pytest.mark.quick
def test_result_closure_gate_rejects_ablation_attack_count_from_all_splits() -> None:
    """正式消融只允许 test split 执行攻击, 不得伪报为全部 Prompt 已攻击."""

    bundle = ready_bundle()
    invalid_summary = {
        **bundle.ablation_summary,
        "attack_and_detection_rerun_count": PROMPT_COUNT
        * len(FORMAL_RUNTIME_RERUN_ABLATION_IDS),
    }
    blocked_bundle = replace(bundle, ablation_summary=invalid_summary)

    report = build_result_closure_gate_report(
        blocked_bundle,
        build_result_closure_gate_checks(blocked_bundle),
    )

    assert report["result_closure_ready"] is False
    assert "formal_ablation_ready" in report["blocked_check_ids"]


@pytest.mark.quick
def test_result_closure_gate_rejects_ablation_prompt_identity_drift() -> None:
    """消融数量正确但 Prompt 集摘要漂移时仍必须阻断论文闭合."""

    bundle = ready_bundle()
    blocked_bundle = replace(
        bundle,
        ablation_summary={
            **bundle.ablation_summary,
            "test_prompt_id_digest": "f" * 64,
        },
    )

    report = build_result_closure_gate_report(
        blocked_bundle,
        build_result_closure_gate_checks(blocked_bundle),
    )

    assert report["result_closure_ready"] is False
    assert "formal_ablation_ready" in report["blocked_check_ids"]


@pytest.mark.quick
def test_result_closure_gate_rejects_incomplete_feature_coverage_and_metric_rows() -> None:
    """特征缺配对或 FID/KID 行数不精确时必须阻断质量证据。"""

    bundle = ready_bundle()
    incomplete_report = {
        **bundle.dataset_quality_feature_report,
        "accepted_feature_pair_count": PROMPT_COUNT - 1,
        "missing_feature_pair_count": 1,
    }
    blocked_bundle = replace(
        bundle,
        dataset_quality_feature_report=incomplete_report,
        dataset_quality_metrics=bundle.dataset_quality_metrics[:1],
    )

    report = build_result_closure_gate_report(
        blocked_bundle,
        build_result_closure_gate_checks(blocked_bundle),
    )

    assert report["result_closure_ready"] is False
    assert "formal_fid_kid_ready" in report["blocked_check_ids"]


@pytest.mark.quick
def test_result_closure_writer_is_run_scoped_and_require_pass_returns_nonzero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """脚本应按 current run 写出 manifest, 阻断时 `--require-pass` 返回非零。"""

    monkeypatch.setenv("SLM_WM_PAPER_RUN_NAME", SCALE)
    bundle = ready_bundle()
    write_bundle_inputs(tmp_path, bundle)

    report = write_result_closure_gate_outputs(
        root=tmp_path,
        prompt_contract=test_prompt_contract(tmp_path),
    )
    output_dir = tmp_path / "outputs/result_closure_gate" / SCALE
    written_report = json.loads((output_dir / "result_closure_gate_report.json").read_text(encoding="utf-8"))
    written_manifest = json.loads((output_dir / "manifest.local.json").read_text(encoding="utf-8"))

    assert report["result_closure_ready"] is True
    assert written_report["result_closure_ready"] is True
    assert written_manifest["artifact_id"] == f"{SCALE}_result_closure_gate_manifest"
    assert written_manifest["metadata"]["result_closure_ready"] is True
    source_map = written_report["closure_source_file_sha256"]
    assert set(source_map) == set(written_manifest["input_paths"])
    assert all(
        hashlib.sha256((tmp_path / path).read_bytes()).hexdigest() == digest
        for path, digest in source_map.items()
    )
    assert written_report["closure_source_file_digest"] == build_stable_digest(source_map)
    assert written_manifest["metadata"]["closure_source_file_sha256"] == source_map
    assert written_manifest["metadata"]["closure_source_file_digest"] == written_report[
        "closure_source_file_digest"
    ]
    assert written_manifest["metadata"]["report_digest"] == hashlib.sha256(
        (output_dir / "result_closure_gate_report.json").read_bytes()
    ).hexdigest()
    assert written_manifest["metadata"]["expected_prompt_id_digest"] == PROMPT_ID_DIGEST

    denied_entry_path = (
        tmp_path / f"outputs/evidence_closure_entry_review/{SCALE}/entry_review_report.json"
    )
    denied_entry = {**bundle.entry_review_report, "evidence_closure_allowed": False}
    write_json(denied_entry_path, denied_entry)
    repository_root = Path(__file__).resolve().parents[2]
    canonical_prompt_path = repository_root / "configs/paper_main_probe_paper_prompts.txt"
    (tmp_path / "configs/paper_main_probe_paper_prompts.txt").write_bytes(
        canonical_prompt_path.read_bytes()
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["write_result_closure_gate_outputs.py", "--root", str(tmp_path), "--require-pass"],
    )
    with pytest.raises(SystemExit) as exit_info:
        main()
    assert exit_info.value.code == 1


@pytest.mark.quick
@pytest.mark.parametrize(
    "source_id",
    ("dataset_quality_metrics_ready", "roc_curve_points_ready"),
)
def test_result_closure_writer_rejects_synchronized_artifact_self_declaration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source_id: str,
) -> None:
    """FID 或 ROC 被改写后, 同步 SHA 与持久化 ready 自声明仍必须阻断."""

    monkeypatch.setenv("SLM_WM_PAPER_RUN_NAME", SCALE)
    write_bundle_inputs(tmp_path, ready_bundle())
    relative_source_path = ARTIFACT_SOURCE_PATHS[source_id]
    source_path = tmp_path / relative_source_path
    with source_path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        fieldnames = tuple(reader.fieldnames or ())
        rows = list(reader)
    if source_id == "dataset_quality_metrics_ready":
        fid_row = next(
            row for row in rows if row["quality_metric_name"] == "fid"
        )
        fid_row["quality_metric_value"] = "9.875"
    else:
        finite_row = next(
            row
            for row in rows
            if row["threshold_kind"] == "observed_score"
        )
        finite_row["threshold"] = str(float(finite_row["threshold"]) - 1e-4)
    with source_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    synchronized_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
    audit_dir = (
        tmp_path / "outputs" / "paper_artifact_evidence_audit" / SCALE
    )
    persisted_report_path = audit_dir / "artifact_data_validation_report.json"
    persisted_report = json.loads(
        persisted_report_path.read_text(encoding="utf-8")
    )
    persisted_report["evidence_source_file_sha256"][
        relative_source_path
    ] = synchronized_sha256
    write_json(persisted_report_path, persisted_report)

    evidence_manifest_path = audit_dir / "manifest.local.json"
    evidence_manifest = json.loads(
        evidence_manifest_path.read_text(encoding="utf-8")
    )
    evidence_manifest["metadata"]["evidence_source_file_sha256"] = dict(
        persisted_report["evidence_source_file_sha256"]
    )
    write_json(evidence_manifest_path, evidence_manifest)

    report = write_result_closure_gate_outputs(
        root=tmp_path,
        prompt_contract=test_prompt_contract(tmp_path),
    )

    assert report["result_closure_ready"] is False
    assert "paper_evidence_audit_ready" in report["blocked_check_ids"]


@pytest.mark.quick
def test_result_closure_gate_rejects_persisted_and_recomputed_report_divergence() -> None:
    """持久化数据报告即使仍自称 ready, 与即时重算报告不同也必须阻断."""

    bundle = ready_bundle()
    persisted_report = json.loads(
        json.dumps(bundle.artifact_data_validation_report)
    )
    persisted_report["checks"]["dataset_quality_metrics_ready"][
        "row_count"
    ] += 1
    blocked_bundle = replace(
        bundle,
        artifact_data_validation_report=persisted_report,
    )

    report = build_result_closure_gate_report(
        blocked_bundle,
        build_result_closure_gate_checks(blocked_bundle),
    )

    assert "paper_evidence_audit_ready" in report["blocked_check_ids"]


@pytest.mark.quick
def test_result_closure_gate_rejects_shape_valid_evidence_config_digest() -> None:
    """合法 SHA-256 外形不能替代 evidence manifest 配置的精确重建."""

    bundle = ready_bundle()
    blocked_bundle = replace(
        bundle,
        evidence_audit_manifest={
            **bundle.evidence_audit_manifest,
            "config_digest": "f" * 64,
        },
    )

    report = build_result_closure_gate_report(
        blocked_bundle,
        build_result_closure_gate_checks(blocked_bundle),
    )

    assert "paper_evidence_audit_ready" in report["blocked_check_ids"]
