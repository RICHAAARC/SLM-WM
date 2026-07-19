"""验证消融检测原子和数据集质量特征来源能够独立重建。"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest
import torch

from experiments.artifacts.dataset_level_quality_outputs import (
    FORMAL_FEATURE_BACKEND,
    FORMAL_FEATURE_EXTRACTOR_ID,
    _inception_batch_config_digest,
    validate_inception_feature_provenance_groups,
)
from experiments.protocol.attacks import (
    attack_config_digest,
    default_attack_configs,
    formal_attack_seed_protocol_record,
    formal_attack_seed_random,
)
from experiments.protocol.detection_key_identity import (
    REGISTERED_WATERMARK_KEY_ROLE,
    REGISTERED_WRONG_KEY_ROLE,
    build_detection_key_plan_record,
    resolve_detection_key_material_and_identity,
)
from experiments.protocol.formal_randomization import (
    build_canonical_sd35_base_latent,
    build_formal_randomization_identity,
    formal_randomization_sample_reference,
    formal_runtime_randomization_plan_record,
    formal_watermark_key_material_from_seed,
    resolve_formal_randomization_repeat,
)
from experiments.protocol.method_runtime_config import (
    FORMAL_METHOD_PACKAGE_ROOT,
    load_formal_method_runtime_config,
)
from experiments.runners.image_only_dataset_runtime import (
    FrozenEvidenceProtocol,
    apply_frozen_evidence_protocol,
    calibrate_complete_evidence_protocol,
)
from experiments.runtime.scientific_unit_provenance import (
    aggregate_scientific_unit_provenance,
)
from experiments.runtime.scientific_content_binding import (
    SCIENTIFIC_CONTENT_BINDING_SCHEMA,
)
from main.core.digest import build_stable_digest
from main.methods.method_definition import (
    semantic_conditioned_latent_method_definition,
    semantic_conditioned_latent_method_definition_digest,
)
from paper_experiments.analysis.formal_record_statistics import (
    FormalRecordStatisticsError,
    _formal_attack_coverage_ready as _analysis_formal_attack_coverage_ready,
    _validate_formal_detections_against_scientific_binding,
    rebuild_and_validate_ablation_runtime_aggregates,
    rebuild_and_validate_dataset_quality_feature_identity,
)
from tests.helpers.scientific_unit_provenance import (
    build_test_scientific_unit_provenance,
)
from tests.helpers.formal_detection_record import bind_formal_detection_record


TARGET_FPR = 0.1
ABLATION_IDS = ("complete_method", "without_test_mechanism")
ABLATION_RUNTIME_CONFIGS = {
    "complete_method": {
        "ablation_id": "complete_method",
        "semantic_routing_enabled": True,
        "null_space_enabled": True,
    },
    "without_test_mechanism": {
        "ablation_id": "without_test_mechanism",
        "semantic_routing_enabled": False,
        "null_space_enabled": True,
    },
}
ABLATION_RUNTIME_OUTPUT_ROOT = "outputs/formal/runs"
PROMPT_SPLITS = {
    "prompt_calibration": "calibration",
    "prompt_calibration_01": "calibration",
    "prompt_calibration_02": "calibration",
    "prompt_test": "test",
}
PROMPT_DIGESTS = {
    prompt_id: build_stable_digest(
        {"prompt_text": f"正式消融 {prompt_id}"}
    )
    for prompt_id in PROMPT_SPLITS
}
PROMPT_INDEXES = {
    prompt_id: prompt_index
    for prompt_index, prompt_id in enumerate(PROMPT_SPLITS)
}
_FORMAL_METHOD_CONFIG = load_formal_method_runtime_config(
    FORMAL_METHOD_PACKAGE_ROOT
)
_FORMAL_REPEAT = resolve_formal_randomization_repeat("seed_00_key_00")
FORMAL_RANDOMIZATION_PLAN = formal_runtime_randomization_plan_record(
    _FORMAL_METHOD_CONFIG.seed,
    base_latent_dtype=f"torch.{_FORMAL_METHOD_CONFIG.latent_torch_dtype}",
    base_latent_shape=(1, 16, 64, 64),
)
RANDOMIZATION_REPEAT_IDENTITY = {
    **_FORMAL_REPEAT.to_dict(),
    "formal_randomization_protocol_digest": FORMAL_RANDOMIZATION_PLAN[
        "formal_randomization_protocol_digest"
    ],
}
_FORMAL_DERIVED_DETECTION_FIELDS = {
    "frozen_content_threshold",
    "frozen_rescue_margin_low",
    "frozen_geometry_score_threshold",
    "frozen_registration_confidence_threshold",
    "frozen_attention_sync_score_threshold",
    "frozen_threshold_digest",
    "frozen_image_only_measurement_config_digest",
    "frozen_attention_geometry_enabled",
    "frozen_image_alignment_enabled",
    "frozen_geometry_rescue_enabled",
    "formal_raw_content_margin",
    "formal_aligned_content_margin",
    "formal_positive_by_content",
    "formal_geometry_reliable",
    "formal_content_failure_reason",
    "formal_rescue_eligible",
    "formal_rescue_applied",
    "formal_evidence_positive",
    "formal_metric_status",
    "supports_paper_claim",
}


def _canonical_randomization_reference(prompt_id: str) -> dict[str, Any]:
    """按冻结正式计划构造功能测试使用的真实样本随机引用."""

    prompt_index = PROMPT_INDEXES[prompt_id]
    identity = build_formal_randomization_identity(
        base_seed=_FORMAL_METHOD_CONFIG.seed,
        prompt_index=prompt_index,
        root_key_material="slm_wm_paper_key",
        repeat=_FORMAL_REPEAT,
    )
    _, base_latent_identity = build_canonical_sd35_base_latent(
        shape=(1, 16, 64, 64),
        generation_seed_random=identity["generation_seed_random"],
        model_id=_FORMAL_METHOD_CONFIG.model_id,
        model_revision=_FORMAL_METHOD_CONFIG.model_revision,
        device="cpu",
        dtype=torch.float16,
    )
    return formal_randomization_sample_reference(
        identity,
        base_latent_identity=base_latent_identity,
    )


CANONICAL_RANDOMIZATION_REFERENCES = {
    prompt_id: _canonical_randomization_reference(prompt_id)
    for prompt_id in PROMPT_SPLITS
}


def _scientific_detection_binding_record(
    *,
    run_id: str,
    detections: list[dict[str, Any]],
    protocol: FrozenEvidenceProtocol,
    randomization_reference: dict[str, Any],
) -> dict[str, Any]:
    """为功能测试绑定真实运行先于论文判定产生的检测身份."""

    registered_key_material = formal_watermark_key_material_from_seed(
        int(randomization_reference["watermark_key_seed_random"]),
        _FORMAL_REPEAT,
    )
    key_plan = build_detection_key_plan_record(registered_key_material)
    identities = []
    for detection_index, detection in enumerate(detections):
        raw_record = {
            key: value
            for key, value in detection.items()
            if key
            not in {
                "ablation_id",
                "ablation_prompt_id",
                *_FORMAL_DERIVED_DETECTION_FIELDS,
            }
        }
        attack_id = str(raw_record.get("attack_id", "none"))
        sample_role = str(raw_record["sample_role"])
        key_role = (
            REGISTERED_WRONG_KEY_ROLE
            if sample_role == "wrong_key_negative"
            and attack_id in {"", "none"}
            else REGISTERED_WATERMARK_KEY_ROLE
        )
        _material, key_identity = resolve_detection_key_material_and_identity(
            registered_key_material,
            key_role,
        )
        identities.append(
            {
                "detection_index": detection_index,
                "sample_role": sample_role,
                "attack_id": attack_id,
                "detection_key_identity": key_identity,
                "detection_record_content_digest": build_stable_digest(
                    raw_record
                ),
            }
        )
    payload = {
        "scientific_content_binding_schema": SCIENTIFIC_CONTENT_BINDING_SCHEMA,
        "run_id": run_id,
        "image_only_measurement_config_digest": (
            protocol.image_only_measurement_config_digest
        ),
        "detection_key_plan_digest_random": key_plan[
            "detection_key_plan_digest_random"
        ],
        "detection_content_identities": identities,
    }
    return {
        **payload,
        "scientific_content_binding_digest": build_stable_digest(payload),
    }


def _bind_attention_alignment_gate(
    record: dict[str, Any],
) -> dict[str, Any]:
    """为检测夹具绑定预注册注意力配准门禁."""

    return bind_formal_detection_record(record)


def _raw_detection(
    *,
    run_id: str,
    prompt_id: str,
    split: str,
    sample_role: str,
    content_score: float,
    randomization_reference: dict[str, Any],
    attack: Any | None = None,
    attention_geometry_enabled: bool = True,
    image_alignment_enabled: bool = True,
    detector_guided_attack_threshold_digest: str = "",
) -> dict[str, Any]:
    """构造冻结协议尚未应用的图像盲检原子。"""

    record: dict[str, Any] = {
        "run_id": run_id,
        "prompt_id": prompt_id,
        "split": split,
        "sample_role": sample_role,
        "detection_key_role": (
            REGISTERED_WRONG_KEY_ROLE
            if sample_role == "wrong_key_negative"
            else REGISTERED_WATERMARK_KEY_ROLE
        ),
        "content_score": content_score,
        "aligned_content_score": (
            content_score if image_alignment_enabled else None
        ),
        "raw_attention_geometry_score": (
            0.0 if attention_geometry_enabled else None
        ),
        "attention_geometry_score": (
            0.0 if image_alignment_enabled else None
        ),
        "registration_confidence": (
            0.0 if image_alignment_enabled else None
        ),
        "attention_sync_score": (
            0.0 if image_alignment_enabled else None
        ),
        "alignment": (
            {"registration_geometry_reliable": False}
            if image_alignment_enabled
            else None
        ),
        "metadata": {
            "attention_geometry_enabled": attention_geometry_enabled,
            "image_alignment_enabled": image_alignment_enabled,
        },
        **randomization_reference,
    }
    if attack is not None:
        generation_seed_random = int(
            randomization_reference["generation_seed_random"]
        )
        record.update(
            {
                "attack_id": attack.attack_id,
                "attack_family": attack.attack_family,
                "attack_name": attack.attack_name,
                "resource_profile": attack.resource_profile,
                "attack_config_digest": attack_config_digest(attack),
                "attack_parameters": attack.attack_parameters,
                "attack_performed": True,
                "generation_seed_random": generation_seed_random,
                "attack_seed_random": formal_attack_seed_random(
                    generation_seed_random,
                    attack.attack_id,
                ),
                "formal_attack_seed_protocol_digest": (
                    formal_attack_seed_protocol_record()[
                        "formal_attack_seed_protocol_digest"
                    ]
                ),
            }
        )
        if attack.attack_name == "adversarial_removal_attack":
            record["detector_guided_attack_threshold_digest"] = (
                detector_guided_attack_threshold_digest
            )
    return _bind_attention_alignment_gate(record)


def _formal_detection_group(
    *,
    ablation_id: str,
    prompt_id: str,
    split: str,
    protocol: FrozenEvidenceProtocol,
    randomization_reference: dict[str, Any],
) -> list[dict[str, Any]]:
    """生成一个变体与一个 Prompt 的完整冻结检测记录组。"""

    run_id = f"run_{ablation_id}_{prompt_id}"
    positive_score = 1.0 if ablation_id == "complete_method" else 0.75
    runtime_config = ABLATION_RUNTIME_CONFIGS[ablation_id]
    attention_geometry_enabled = bool(
        runtime_config.get("attention_geometry_enabled", True)
    )
    image_alignment_enabled = bool(
        runtime_config.get("image_alignment_enabled", True)
    )
    records = [
        _raw_detection(
            run_id=run_id,
            prompt_id=prompt_id,
            split=split,
            sample_role="clean_negative",
            content_score=0.0,
            randomization_reference=randomization_reference,
            attention_geometry_enabled=attention_geometry_enabled,
            image_alignment_enabled=image_alignment_enabled,
        ),
        _raw_detection(
            run_id=run_id,
            prompt_id=prompt_id,
            split=split,
            sample_role="positive_source",
            content_score=positive_score,
            randomization_reference=randomization_reference,
            attention_geometry_enabled=attention_geometry_enabled,
            image_alignment_enabled=image_alignment_enabled,
        ),
        _raw_detection(
            run_id=run_id,
            prompt_id=prompt_id,
            split=split,
            sample_role="wrong_key_negative",
            content_score=0.0,
            randomization_reference=randomization_reference,
            attention_geometry_enabled=attention_geometry_enabled,
            image_alignment_enabled=image_alignment_enabled,
        ),
    ]
    if split == "test":
        records.extend(
            _raw_detection(
                run_id=run_id,
                prompt_id=prompt_id,
                split=split,
                sample_role=sample_role,
                content_score=(
                    positive_score if sample_role == "positive_source" else 0.0
                ),
                randomization_reference=randomization_reference,
                attack=attack,
                attention_geometry_enabled=attention_geometry_enabled,
                image_alignment_enabled=image_alignment_enabled,
                detector_guided_attack_threshold_digest=(
                    protocol.threshold_digest
                ),
            )
            for attack in default_attack_configs()
            if attack.enabled
            and attack.resource_profile in {"full_main", "full_extra"}
            for sample_role in ("clean_negative", "positive_source")
        )
    return [
        {
            **record,
            "ablation_id": ablation_id,
            "ablation_prompt_id": prompt_id,
        }
        for record in apply_frozen_evidence_protocol(records, protocol)
    ]


def _ablation_atomic_fixture() -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, dict[str, Any]],
    list[dict[str, Any]],
]:
    """构造能够独立重建的逐 Prompt 消融记录、检测原子和协议。"""

    runtime_records: list[dict[str, Any]] = []
    detection_records: list[dict[str, Any]] = []
    protocols: dict[str, dict[str, Any]] = {}
    unit_identity_records: list[dict[str, Any]] = []
    for ablation_id in ABLATION_IDS:
        runtime_config = ABLATION_RUNTIME_CONFIGS[ablation_id]
        calibration_negatives = tuple(
            _raw_detection(
                run_id=f"run_{ablation_id}_{prompt_id}",
                prompt_id=prompt_id,
                split="calibration",
                sample_role="clean_negative",
                content_score=0.0,
                randomization_reference=(
                    CANONICAL_RANDOMIZATION_REFERENCES[prompt_id]
                ),
                attention_geometry_enabled=bool(
                    runtime_config.get("attention_geometry_enabled", True)
                ),
                image_alignment_enabled=bool(
                    runtime_config.get("image_alignment_enabled", True)
                ),
            )
            for prompt_id, split in PROMPT_SPLITS.items()
            if split == "calibration"
        )
        protocol = calibrate_complete_evidence_protocol(
            calibration_negatives,
            target_fpr=TARGET_FPR,
        )
        protocols[ablation_id] = protocol.to_dict()
        runtime_config = dict(runtime_config)
        for prompt_index, (prompt_id, split) in enumerate(PROMPT_SPLITS.items()):
            prompt_text = f"正式消融 {prompt_id}"
            randomization_reference = CANONICAL_RANDOMIZATION_REFERENCES[
                prompt_id
            ]
            scientific_config = {
                "prompt": prompt_text,
                "prompt_id": prompt_id,
                "split": split,
                "output_dir": f"{ABLATION_RUNTIME_OUTPUT_ROOT}/{ablation_id}",
                "method_definition": (
                    semantic_conditioned_latent_method_definition()
                ),
                "method_definition_digest": (
                    semantic_conditioned_latent_method_definition_digest()
                ),
                "model_id": _FORMAL_METHOD_CONFIG.model_id,
                "model_revision": _FORMAL_METHOD_CONFIG.model_revision,
                "seed": randomization_reference["generation_seed_random"],
                "randomization_repeat_id": _FORMAL_REPEAT.randomization_repeat_id,
                "generation_seed_index": _FORMAL_REPEAT.generation_seed_index,
                "generation_seed_offset": _FORMAL_REPEAT.generation_seed_offset,
                "watermark_key_index": _FORMAL_REPEAT.watermark_key_index,
                "watermark_key_seed_random": randomization_reference[
                    "watermark_key_seed_random"
                ],
                "formal_randomization_protocol_digest": (
                    randomization_reference[
                        "formal_randomization_protocol_digest"
                    ]
                ),
                "key_material_digest_random": randomization_reference[
                    "watermark_key_material_digest_random"
                ],
                "torch_dtype": _FORMAL_METHOD_CONFIG.latent_torch_dtype,
                "latent_torch_dtype": _FORMAL_METHOD_CONFIG.latent_torch_dtype,
                "width": _FORMAL_METHOD_CONFIG.width,
                "height": _FORMAL_METHOD_CONFIG.height,
                "detector_guided_attack_threshold_protocol": (
                    protocol.to_dict() if split == "test" else None
                ),
                **{
                    field_name: field_value
                    for field_name, field_value in runtime_config.items()
                    if field_name != "ablation_id"
                },
            }
            config_digest = build_stable_digest(scientific_config)
            run_id = f"semantic_watermark_{config_digest[:16]}"
            unit_identity_records.append(
                {
                    "run_id": run_id,
                    "scientific_unit_config": scientific_config,
                    "formal_randomization_reference": (
                        randomization_reference
                    ),
                }
            )
            detections = _formal_detection_group(
                ablation_id=ablation_id,
                prompt_id=prompt_id,
                split=split,
                protocol=protocol,
                randomization_reference=randomization_reference,
            )
            for detection in detections:
                detection["run_id"] = run_id
            scientific_binding = _scientific_detection_binding_record(
                run_id=run_id,
                detections=detections,
                protocol=protocol,
                randomization_reference=randomization_reference,
            )
            detection_records.extend(detections)
            un_attacked = {
                record["sample_role"]: record
                for record in detections
                if not record.get("attack_id")
            }
            attacked_positive = [
                record
                for record in detections
                if record.get("attack_id")
                and record["sample_role"] == "positive_source"
            ]
            attacked_negative = [
                record
                for record in detections
                if record.get("attack_id")
                and record["sample_role"] == "clean_negative"
            ]
            runtime_records.append(
                {
                    "prompt_index": prompt_index,
                    "prompt_id": prompt_id,
                    "prompt_digest": build_stable_digest(
                        {"prompt_text": prompt_text}
                    ),
                    "split": split,
                    "ablation_id": ablation_id,
                    "runtime_config": runtime_config,
                    "runtime_result": {
                        "run_id": run_id,
                        "run_decision": "pass",
                        "metadata": {
                            "scientific_unit_config_digest": config_digest,
                            "formal_randomization_reference": (
                                randomization_reference
                            ),
                            "scientific_unit_provenance": (
                                build_test_scientific_unit_provenance(
                                    run_id,
                                    config_digest,
                                )
                            ),
                            "scientific_content_binding_schema": (
                                SCIENTIFIC_CONTENT_BINDING_SCHEMA
                            ),
                            "scientific_content_binding_record": (
                                scientific_binding
                            ),
                            "scientific_content_binding_digest": (
                                scientific_binding[
                                    "scientific_content_binding_digest"
                                ]
                            ),
                            "paired_quality": {"ssim": 0.95},
                        },
                    },
                    "generation_rerun": True,
                    "attack_and_detection_rerun": bool(attacked_positive),
                    "threshold_calibration_scope": (
                        "per_ablation_calibration_split"
                    ),
                    "frozen_content_threshold": protocol.content_threshold,
                    "frozen_threshold_digest": protocol.threshold_digest,
                    "clean_negative_positive": bool(
                        un_attacked["clean_negative"]["formal_evidence_positive"]
                    ),
                    "positive_source_positive": bool(
                        un_attacked["positive_source"]["formal_evidence_positive"]
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
    return (
        runtime_records,
        detection_records,
        protocols,
        unit_identity_records,
    )


@pytest.mark.quick
def test_ablation_runtime_aggregates_rebuild_from_detection_atoms() -> None:
    """逐 Prompt 聚合值与检测原子、冻结阈值一致时才允许闭合。"""

    runtime_records, detections, protocols, unit_identities = (
        _ablation_atomic_fixture()
    )

    result = rebuild_and_validate_ablation_runtime_aggregates(
        runtime_records,
        detections,
        protocols,
        scientific_unit_identity_records=unit_identities,
        expected_ablation_ids=ABLATION_IDS,
        expected_prompt_split_by_id=PROMPT_SPLITS,
        expected_prompt_digest_by_id=PROMPT_DIGESTS,
        expected_prompt_index_by_id=PROMPT_INDEXES,
        expected_runtime_config_by_ablation_id=ABLATION_RUNTIME_CONFIGS,
        expected_runtime_output_root=ABLATION_RUNTIME_OUTPUT_ROOT,
        expected_target_fpr=TARGET_FPR,
        formal_randomization_plan=FORMAL_RANDOMIZATION_PLAN,
        randomization_repeat_identity=RANDOMIZATION_REPEAT_IDENTITY,
    )

    assert result["ablation_runtime_aggregate_rebuild_ready"] is True
    assert result["ablation_runtime_record_count"] == 8
    assert result["ablation_detection_record_count"] == len(detections)


@pytest.mark.quick
@pytest.mark.parametrize(
    ("prompt_id", "first_role", "second_role", "attack_required"),
    (
        ("prompt_test", "positive_source", "wrong_key_negative", False),
        ("prompt_test", "clean_negative", "positive_source", True),
        ("prompt_calibration", "clean_negative", "positive_source", False),
    ),
)
def test_scientific_binding_rejects_detection_role_relabeling(
    prompt_id: str,
    first_role: str,
    second_role: str,
    attack_required: bool,
) -> None:
    """角色重标即使保持记录数量不变也不得改变固定 FPR 或攻击统计."""

    runtime_records, detections, protocols, _unit_identities = (
        _ablation_atomic_fixture()
    )
    runtime_record = next(
        record
        for record in runtime_records
        if record["ablation_id"] == "complete_method"
        and record["prompt_id"] == prompt_id
    )
    run_id = runtime_record["runtime_result"]["run_id"]
    group = [record for record in detections if record["run_id"] == run_id]
    candidates = [
        record
        for record in group
        if bool(record.get("attack_id")) is attack_required
    ]
    first = next(
        record for record in candidates if record["sample_role"] == first_role
    )
    attack_id = first.get("attack_id")
    second = next(
        record
        for record in candidates
        if record["sample_role"] == second_role
        and (
            not attack_required or record.get("attack_id") == attack_id
        )
    )
    first["sample_role"], second["sample_role"] = (
        second["sample_role"],
        first["sample_role"],
    )
    key_material = formal_watermark_key_material_from_seed(
        int(
            CANONICAL_RANDOMIZATION_REFERENCES[prompt_id][
                "watermark_key_seed_random"
            ]
        ),
        _FORMAL_REPEAT,
    )
    key_plan_digest = build_detection_key_plan_record(key_material)[
        "detection_key_plan_digest_random"
    ]

    with pytest.raises(FormalRecordStatisticsError, match="科学运行|样本角色"):
        _validate_formal_detections_against_scientific_binding(
            tuple(group),
            runtime_record["runtime_result"],
            FrozenEvidenceProtocol(**protocols["complete_method"]),
            expected_detection_key_plan_digest_random=key_plan_digest,
        )


@pytest.mark.quick
@pytest.mark.parametrize(
    "mutation",
    (
        "aggregate_rate",
        "formal_decision",
        "detection_split",
        "prompt_digest",
        "threshold_digest",
        "lf_carrier_identity",
    ),
)
def test_ablation_runtime_aggregate_rebuild_fails_closed_on_drift(
    mutation: str,
) -> None:
    """任一聚合、原子、split 或阈值漂移都必须阻断。"""

    runtime_records, detections, protocols, unit_identities = (
        _ablation_atomic_fixture()
    )
    if mutation == "aggregate_rate":
        target = next(record for record in runtime_records if record["split"] == "test")
        target["attacked_positive_rate"] = 0.25
    elif mutation == "formal_decision":
        target = next(
            record
            for record in detections
            if record["split"] == "test"
            and record["sample_role"] == "positive_source"
            and record.get("attack_id")
        )
        target["formal_evidence_positive"] = not target[
            "formal_evidence_positive"
        ]
    elif mutation == "detection_split":
        detections[0]["split"] = "test"
    elif mutation == "prompt_digest":
        runtime_records[0]["prompt_digest"] = "f" * 64
    elif mutation == "threshold_digest":
        protocols["complete_method"]["threshold_digest"] = "f" * 64
    else:
        target = next(
            record
            for record in detections
            if record["split"] == "test"
            and record["sample_role"] == "positive_source"
            and not record.get("attack_id")
        )
        target.update(
            bind_formal_detection_record(
                target,
                method_role="lf_only_content",
                lf_weight=1.0,
                tail_robust_weight=0.0,
                tail_fraction=0.20,
            )
        )

    with pytest.raises(FormalRecordStatisticsError):
        rebuild_and_validate_ablation_runtime_aggregates(
            runtime_records,
            detections,
            protocols,
            scientific_unit_identity_records=unit_identities,
            expected_ablation_ids=ABLATION_IDS,
            expected_prompt_split_by_id=PROMPT_SPLITS,
            expected_prompt_digest_by_id=PROMPT_DIGESTS,
            expected_prompt_index_by_id=PROMPT_INDEXES,
            expected_runtime_config_by_ablation_id=ABLATION_RUNTIME_CONFIGS,
            expected_runtime_output_root=ABLATION_RUNTIME_OUTPUT_ROOT,
            expected_target_fpr=TARGET_FPR,
            formal_randomization_plan=FORMAL_RANDOMIZATION_PLAN,
            randomization_repeat_identity=RANDOMIZATION_REPEAT_IDENTITY,
        )


@pytest.mark.quick
@pytest.mark.parametrize("mutation", ("mechanism_config", "paired_ssim"))
def test_ablation_runtime_aggregate_rejects_semantic_identity_drift(
    mutation: str,
) -> None:
    """同步自洽的错误机制开关或图像质量字段也不得冒充正式消融。"""

    runtime_records, detections, protocols, unit_identities = (
        _ablation_atomic_fixture()
    )
    target = runtime_records[0]
    if mutation == "mechanism_config":
        target["runtime_config"] = dict(target["runtime_config"])
        target["runtime_config"]["semantic_routing_enabled"] = False
        target_run_id = target["runtime_result"]["run_id"]
        identity = next(
            record
            for record in unit_identities
            if record["run_id"] == target_run_id
        )
        identity["scientific_unit_config"] = dict(
            identity["scientific_unit_config"]
        )
        identity["scientific_unit_config"][
            "semantic_routing_enabled"
        ] = False
    else:
        target["paired_ssim"] = 0.1

    with pytest.raises(FormalRecordStatisticsError):
        rebuild_and_validate_ablation_runtime_aggregates(
            runtime_records,
            detections,
            protocols,
            scientific_unit_identity_records=unit_identities,
            expected_ablation_ids=ABLATION_IDS,
            expected_prompt_split_by_id=PROMPT_SPLITS,
            expected_prompt_digest_by_id=PROMPT_DIGESTS,
            expected_prompt_index_by_id=PROMPT_INDEXES,
            expected_runtime_config_by_ablation_id=ABLATION_RUNTIME_CONFIGS,
            expected_runtime_output_root=ABLATION_RUNTIME_OUTPUT_ROOT,
            expected_target_fpr=TARGET_FPR,
            formal_randomization_plan=FORMAL_RANDOMIZATION_PLAN,
            randomization_repeat_identity=RANDOMIZATION_REPEAT_IDENTITY,
        )


@pytest.mark.quick
def test_ablation_runtime_rejects_synchronized_random_reference_forgery() -> None:
    """顶层、运行和检测同步写成同一错误引用时仍必须独立拒绝."""

    runtime_records, detections, protocols, unit_identities = (
        _ablation_atomic_fixture()
    )
    forged_reference = deepcopy(
        CANONICAL_RANDOMIZATION_REFERENCES["prompt_test"]
    )
    target_run_ids = {
        record["runtime_result"]["run_id"]
        for record in runtime_records
        if record["prompt_id"] == "prompt_calibration"
    }
    for record in runtime_records:
        if record["prompt_id"] == "prompt_calibration":
            record["runtime_result"]["metadata"][
                "formal_randomization_reference"
            ] = deepcopy(forged_reference)
    for identity in unit_identities:
        if identity["run_id"] in target_run_ids:
            identity["formal_randomization_reference"] = deepcopy(
                forged_reference
            )
    for detection in detections:
        if detection["prompt_id"] == "prompt_calibration":
            detection.update(forged_reference)

    with pytest.raises(FormalRecordStatisticsError):
        rebuild_and_validate_ablation_runtime_aggregates(
            runtime_records,
            detections,
            protocols,
            scientific_unit_identity_records=unit_identities,
            expected_ablation_ids=ABLATION_IDS,
            expected_prompt_split_by_id=PROMPT_SPLITS,
            expected_prompt_digest_by_id=PROMPT_DIGESTS,
            expected_prompt_index_by_id=PROMPT_INDEXES,
            expected_runtime_config_by_ablation_id=ABLATION_RUNTIME_CONFIGS,
            expected_runtime_output_root=ABLATION_RUNTIME_OUTPUT_ROOT,
            expected_target_fpr=TARGET_FPR,
            formal_randomization_plan=FORMAL_RANDOMIZATION_PLAN,
            randomization_repeat_identity=RANDOMIZATION_REPEAT_IDENTITY,
        )


@pytest.mark.quick
def test_ablation_runtime_rejects_canonical_prompt_index_swap() -> None:
    """跨消融一致但与规范 Prompt 文件相反的索引交换必须被拒绝."""

    runtime_records, detections, protocols, unit_identities = (
        _ablation_atomic_fixture()
    )
    for record in runtime_records:
        record["prompt_index"] = 1 - int(record["prompt_index"])

    with pytest.raises(FormalRecordStatisticsError):
        rebuild_and_validate_ablation_runtime_aggregates(
            runtime_records,
            detections,
            protocols,
            scientific_unit_identity_records=unit_identities,
            expected_ablation_ids=ABLATION_IDS,
            expected_prompt_split_by_id=PROMPT_SPLITS,
            expected_prompt_digest_by_id=PROMPT_DIGESTS,
            expected_prompt_index_by_id=PROMPT_INDEXES,
            expected_runtime_config_by_ablation_id=ABLATION_RUNTIME_CONFIGS,
            expected_runtime_output_root=ABLATION_RUNTIME_OUTPUT_ROOT,
            expected_target_fpr=TARGET_FPR,
            formal_randomization_plan=FORMAL_RANDOMIZATION_PLAN,
            randomization_repeat_identity=RANDOMIZATION_REPEAT_IDENTITY,
        )


def _dataset_quality_identity_fixture() -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, str],
]:
    """构造图像记录、特征记录和来源 summary 的完整绑定。"""

    image_records: list[dict[str, Any]] = []
    resolution_records: list[dict[str, Any]] = []
    feature_records: list[dict[str, Any]] = []
    actual_source_sha256: dict[str, str] = {}
    item_identity: list[dict[str, Any]] = []
    for index in range(2):
        prompt_id = f"quality_prompt_{index}"
        source_path = f"outputs/images/source_{index}.png"
        comparison_path = f"outputs/images/comparison_{index}.png"
        source_digest = build_stable_digest({"path": source_path})
        comparison_digest = build_stable_digest({"path": comparison_path})
        payload = {
            "run_id": f"quality_runtime_{index}",
            "prompt_id": prompt_id,
            "attack_name": "watermark_embedding",
            "image_pair_index": index,
            "image_pair_role": "clean_to_watermarked",
            "source_image_path": source_path,
            "source_image_digest": source_digest,
            "comparison_image_path": comparison_path,
            "comparison_image_digest": comparison_digest,
            "feature_backend": FORMAL_FEATURE_BACKEND,
            "supports_paper_claim": False,
        }
        record_digest = build_stable_digest(payload)
        record_id = f"dataset_quality_record_{record_digest[:16]}"
        image_records.append(
            {
                "dataset_quality_record_id": record_id,
                "dataset_quality_record_digest": record_digest,
                **payload,
            }
        )
        for role, image_path, image_digest, value in (
            ("source", source_path, source_digest, float(index)),
            (
                "comparison",
                comparison_path,
                comparison_digest,
                float(index) + 0.25,
            ),
        ):
            resolved_image_path = (
                f"outputs/materialized/{role}_{index}.png"
            )
            resolution_payload = {
                "requested_image_path": image_path,
                "resolved_image_path": resolved_image_path,
                "resolved_from_package_path": "outputs/input_package.zip",
                "resolution_status": "materialized_from_input_package",
                "resolved_image_digest": image_digest,
                "materialized_image_input": True,
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
            actual_source_sha256[resolved_image_path] = image_digest
            identity = {
                "dataset_quality_record_id": record_id,
                "dataset_quality_image_role": role,
                "image_path": resolved_image_path,
                "image_digest": image_digest,
            }
            item_identity.append(identity)
            feature_records.append(
                {
                    **identity,
                    "feature_backend": FORMAL_FEATURE_BACKEND,
                    "feature_extractor_id": FORMAL_FEATURE_EXTRACTOR_ID,
                    "feature_dimension": 2048,
                    "feature_vector": [value, value + 1.0] * 1024,
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
    )
    for record in feature_records:
        record["scientific_unit_provenance"] = provenance
    references = validate_inception_feature_provenance_groups(feature_records)
    provenance_summary = aggregate_scientific_unit_provenance(
        references,
        expected_reference_count=len(feature_records),
    )
    return (
        image_records,
        resolution_records,
        feature_records,
        provenance_summary,
        actual_source_sha256,
    )


def _quality_prompt_digest(image_records: list[dict[str, Any]]) -> str:
    """计算质量图像记录的规范 Prompt 集合摘要。"""

    return build_stable_digest(
        sorted(str(record["prompt_id"]) for record in image_records)
    )


@pytest.mark.quick
def test_dataset_quality_feature_identity_binds_images_and_provenance() -> None:
    """feature 路径、SHA、角色和科学来源全部一致时才允许闭合。"""

    (
        image_records,
        resolution_records,
        feature_records,
        provenance_summary,
        actual_source_sha256,
    ) = (
        _dataset_quality_identity_fixture()
    )

    result = rebuild_and_validate_dataset_quality_feature_identity(
        image_records,
        resolution_records,
        feature_records,
        provenance_summary,
        actual_source_sha256,
        expected_pair_count=2,
        expected_prompt_id_digest=_quality_prompt_digest(image_records),
    )

    assert result["dataset_quality_feature_identity_rebuild_ready"] is True
    assert result["dataset_quality_image_record_count"] == 2
    assert result["dataset_quality_feature_record_count"] == 4


@pytest.mark.quick
@pytest.mark.parametrize(
    "mutation",
    (
        "image_path",
        "image_digest",
        "provenance",
        "image_record_digest",
        "reported_provenance",
        "feature_dimension",
        "resolution_digest",
        "actual_image_sha",
        "duplicate_image_path",
        "pair_role",
        "duplicate_run_id",
    ),
)
def test_dataset_quality_feature_identity_rebuild_fails_closed_on_drift(
    mutation: str,
) -> None:
    """图像身份、特征角色或来源任一脱离都必须阻断。"""

    (
        image_records,
        resolution_records,
        feature_records,
        provenance_summary,
        actual_source_sha256,
    ) = (
        _dataset_quality_identity_fixture()
    )
    expected_prompt_digest = _quality_prompt_digest(image_records)
    if mutation == "image_path":
        feature_records[0]["image_path"] = "outputs/images/forged.png"
    elif mutation == "image_digest":
        feature_records[0]["image_digest"] = "f" * 64
    elif mutation == "provenance":
        original = feature_records[0]["scientific_unit_provenance"]
        forged = build_test_scientific_unit_provenance(
            str(original["scientific_unit_id"]),
            "f" * 64,
        )
        for record in feature_records:
            record["scientific_unit_provenance"] = forged
    elif mutation == "image_record_digest":
        image_records[0]["dataset_quality_record_digest"] = "f" * 64
    elif mutation == "feature_dimension":
        feature_records[0]["feature_dimension"] = 2
        feature_records[0]["feature_vector"] = [0.0, 1.0]
    elif mutation == "resolution_digest":
        resolution = resolution_records[0]
        resolution["resolved_image_digest"] = "f" * 64
        payload = {
            key: value
            for key, value in resolution.items()
            if key not in {
                "image_resolution_record_digest",
                "image_resolution_record_id",
            }
        }
        digest = build_stable_digest(payload)
        resolution["image_resolution_record_digest"] = digest
        resolution["image_resolution_record_id"] = (
            f"dataset_quality_image_resolution_{digest[:16]}"
        )
    elif mutation == "actual_image_sha":
        actual_source_sha256[
            str(resolution_records[0]["resolved_image_path"])
        ] = "f" * 64
    elif mutation in {
        "duplicate_image_path",
        "pair_role",
        "duplicate_run_id",
    }:
        forged_record = image_records[1]
        if mutation == "duplicate_image_path":
            forged_record["source_image_path"] = image_records[0][
                "source_image_path"
            ]
            forged_record["source_image_digest"] = image_records[0][
                "source_image_digest"
            ]
        elif mutation == "pair_role":
            forged_record["image_pair_role"] = "clean_to_attacked"
        else:
            forged_record["run_id"] = image_records[0]["run_id"]
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
    else:
        provenance_summary["scientific_unit_provenance_records_digest"] = (
            "f" * 64
        )

    with pytest.raises(FormalRecordStatisticsError):
        rebuild_and_validate_dataset_quality_feature_identity(
            image_records,
            resolution_records,
            feature_records,
            provenance_summary,
            actual_source_sha256,
            expected_pair_count=2,
            expected_prompt_id_digest=expected_prompt_digest,
        )


def test_ablation_statistics_rejects_attack_seed_drift() -> None:
    """独立统计重建必须拒绝与统一公式不一致的消融攻击 seed."""

    generation_seed_random = 1703
    protocol_digest = formal_attack_seed_protocol_record()[
        "formal_attack_seed_protocol_digest"
    ]
    records = tuple(
        {
            "attack_id": config.attack_id,
            "attack_family": config.attack_family,
            "attack_name": config.attack_name,
            "resource_profile": config.resource_profile,
            "attack_config_digest": attack_config_digest(config),
            "attack_parameters": config.attack_parameters,
            "attack_performed": True,
            "sample_role": sample_role,
            "generation_seed_random": generation_seed_random,
            "attack_seed_random": formal_attack_seed_random(
                generation_seed_random,
                config.attack_id,
            ),
            "formal_attack_seed_protocol_digest": protocol_digest,
        }
        for config in default_attack_configs()
        if config.enabled
        and config.resource_profile in {"full_main", "full_extra"}
        for sample_role in ("clean_negative", "positive_source")
    )
    assert _analysis_formal_attack_coverage_ready(
        records,
        split="test",
        expected_generation_seed_random=generation_seed_random,
    )

    drifted = [dict(record) for record in records]
    drifted[0]["attack_seed_random"] = int(
        drifted[0]["attack_seed_random"]
    ) + 1
    assert not _analysis_formal_attack_coverage_ready(
        tuple(drifted),
        split="test",
        expected_generation_seed_random=generation_seed_random,
    )
