from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from experiments.protocol.fixed_fpr_observation_audit import (
    FORMAL_THRESHOLD_SOURCE,
    audit_fixed_fpr_observation_threshold,
    conformal_threshold_from_clean_negative_scores,
)
from experiments.protocol.formal_randomization import (
    formal_randomization_protocol_record,
    formal_randomization_repeats,
    formal_watermark_key_plan_record,
)
from experiments.protocol.prompts import (
    PROMPT_FILES,
    build_prompt_records,
    read_prompt_file,
)
from experiments.protocol.splits import apply_split_assignments
from experiments.runners.image_only_dataset_runtime import (
    apply_frozen_evidence_protocol,
    calibrate_complete_evidence_protocol,
)
from main.core.digest import build_stable_digest
from paper_experiments.analysis.fixed_fpr_threshold_audit import (
    FIXED_FPR_THRESHOLD_METHOD_IDS,
)
from paper_experiments.analysis.method_repeat_fixed_fpr import (
    FORMAL_BASE_LATENT_DTYPE,
    FORMAL_BASE_LATENT_GENERATION_PROTOCOL,
    FORMAL_BASE_LATENT_SHAPE,
    METHOD_LEAF_PACKAGE_FAMILY,
    METHOD_REPEAT_THRESHOLD_COUNT,
    MethodRepeatFixedFprError,
    MethodRepeatObservationSource,
    recompute_exact_method_repeat_fixed_fpr,
)


pytestmark = pytest.mark.quick

PAPER_RUN_NAME = "probe_paper"
TARGET_FPR = 0.1
MODEL_ID = "stabilityai/stable-diffusion-3.5-medium"
MODEL_REVISION = "b940f670f0eda2d07fbb75229e779da1ad11eb80"
RESCUE_MARGIN_LOW = -0.05
EXPECTED_BASE_SEED = 1703
ATTENTION_ALIGNMENT_GATE = {
    "attention_anchor_count": 12,
    "attention_residual_threshold": 0.20,
    "attention_minimum_inlier_ratio": 0.50,
}


def _bind_attention_alignment_gate(
    record: dict[str, object],
) -> dict[str, object]:
    """为检测夹具绑定预注册注意力配准门禁."""

    gate = dict(ATTENTION_ALIGNMENT_GATE)
    metadata = dict(record.get("metadata", {}))
    metadata.update(gate)
    metadata["attention_alignment_gate"] = dict(gate)
    resolved = {**record, "metadata": metadata}
    alignment = resolved.get("alignment")
    if isinstance(alignment, dict):
        alignment_metadata = dict(alignment.get("metadata", {}))
        alignment_metadata["attention_alignment_gate"] = dict(gate)
        resolved["alignment"] = {
            **alignment,
            **gate,
            "metadata": alignment_metadata,
        }
    return resolved


def _prompt_rows() -> tuple[dict[str, object], ...]:
    """构造 probe 层级精确3/33/34的规范 Prompt 契约."""

    repository_root = Path(__file__).resolve().parents[2]
    records = apply_split_assignments(
        build_prompt_records(
            PAPER_RUN_NAME,
            read_prompt_file(repository_root / PROMPT_FILES[PAPER_RUN_NAME]),
        )
    )
    return tuple(
        {
            "prompt_id": record.prompt_id,
            "prompt_index": record.prompt_index,
            "prompt_text": record.prompt_text,
            "split": record.split,
            "prompt_digest": record.prompt_digest,
        }
        for record in records
    )


def _identity(
    repeat_index: int,
    prompt_index: int,
    *,
    include_base_protocol: bool,
) -> dict[str, object]:
    """构造五方法共享且满足3种 seed 与3种 key 交叉的随机身份."""

    repeat = formal_randomization_repeats()[repeat_index]
    protocol_digest = formal_randomization_protocol_record()[
        "formal_randomization_protocol_digest"
    ]
    generation_seed_random = (
        EXPECTED_BASE_SEED
        + repeat.generation_seed_offset
        + prompt_index
    )
    key_plan_record = formal_watermark_key_plan_record()[
        "watermark_key_records"
    ][repeat.watermark_key_index]
    watermark_key_seed_random = int(
        key_plan_record["watermark_key_seed_random"]
    )
    key_material_digest = str(
        key_plan_record["watermark_key_material_digest_random"]
    )
    formal_payload = {
        **repeat.to_dict(),
        "generation_seed_random": generation_seed_random,
        "watermark_key_seed_random": watermark_key_seed_random,
        "formal_randomization_protocol_digest": protocol_digest,
        "watermark_key_material_digest_random": key_material_digest,
    }
    base_latent_content_digest = build_stable_digest(
        {"canonical_latent_seed": generation_seed_random}
    )
    base_protocol = {
        "base_latent_generation_protocol": (
            FORMAL_BASE_LATENT_GENERATION_PROTOCOL
        ),
        "base_latent_keyed_prg_version": (
            formal_randomization_protocol_record()[
                "base_latent_keyed_prg_version"
            ]
        ),
        "base_latent_keyed_prg_protocol_digest": (
            formal_randomization_protocol_record()[
                "base_latent_keyed_prg_protocol_digest"
            ]
        ),
        "base_latent_dtype": FORMAL_BASE_LATENT_DTYPE,
        "base_latent_shape": list(FORMAL_BASE_LATENT_SHAPE),
    }
    base_identity_payload = {
        "generation_seed_random": generation_seed_random,
        **base_protocol,
        "formal_randomization_protocol_digest": protocol_digest,
        "base_latent_content_digest_random": base_latent_content_digest,
    }
    return {
        **formal_payload,
        "formal_randomization_identity_digest_random": build_stable_digest(
            formal_payload
        ),
        "base_latent_content_digest_random": base_latent_content_digest,
        "base_latent_identity_digest_random": build_stable_digest(
            base_identity_payload
        ),
        **(base_protocol if include_base_protocol else {}),
    }


def _member_paths(method_id: str, repeat_id: str) -> dict[str, str]:
    """构造与嵌套证据包一致的成员路径."""

    family = METHOD_LEAF_PACKAGE_FAMILY[method_id]
    if method_id == "slm_wm":
        observation_member = (
            f"outputs/image_only_dataset_runtime/{PAPER_RUN_NAME}/"
            "image_only_detection_records.jsonl"
        )
        declaration_member = (
            f"outputs/image_only_dataset_runtime/{PAPER_RUN_NAME}/"
            "frozen_evidence_protocol.json"
        )
    elif method_id == "t2smark":
        observation_member = (
            f"outputs/t2smark_formal_reproduction/{PAPER_RUN_NAME}/"
            "t2smark_adapter/baseline_observations.json"
        )
        declaration_member = (
            f"outputs/t2smark_formal_reproduction/{PAPER_RUN_NAME}/"
            "t2smark_formal_import_candidate_records.jsonl"
        )
    else:
        observation_member = (
            f"outputs/external_baseline_method_faithful/{PAPER_RUN_NAME}/"
            f"split_observations/{method_id}_baseline_observations.json"
        )
        declaration_member = (
            f"outputs/external_baseline_method_faithful/{PAPER_RUN_NAME}/"
            f"split_observations/{method_id}_baseline_transfer_manifest.json"
        )
    return {
        "repeat_component_archive_member": f"repeat_components/{repeat_id}.zip",
        "leaf_package_archive_member": (
            f"randomization_repeat_evidence/{repeat_id}/"
            f"leaf_packages/{family}.zip"
        ),
        "observation_archive_member": observation_member,
        "threshold_declaration_archive_member": declaration_member,
    }


def _source(
    *,
    repeat_index: int,
    method_index: int,
) -> MethodRepeatObservationSource:
    """构造一个独立校准的 method-repeat 来源."""

    repeat = formal_randomization_repeats()[repeat_index]
    method_id = FIXED_FPR_THRESHOLD_METHOD_IDS[method_index]
    score_offset = repeat_index * 0.1 + method_index * 0.01
    raw_rows = tuple(
        {
            "prompt_id": str(prompt["prompt_id"]),
            "split": str(prompt["split"]),
            "sample_role": "clean_negative",
            "attack_id": "",
            "attack_family": "" if method_id == "slm_wm" else "clean",
            "attack_name": "" if method_id == "slm_wm" else "clean_none",
            **_identity(
                repeat_index,
                int(prompt["prompt_index"]),
                include_base_protocol=method_id == "slm_wm",
            ),
            **(
                {}
                if method_id == "slm_wm"
                else {
                    "baseline_id": method_id,
                    "prompt_text": str(prompt["prompt_text"]),
                    "generation_model_id": MODEL_ID,
                    "generation_model_revision": MODEL_REVISION,
                }
            ),
            **(
                {
                    "content_score": score_offset
                    + int(prompt["prompt_index"]) / 1000.0,
                    "aligned_content_score": score_offset
                    + int(prompt["prompt_index"]) / 1000.0,
                    "attention_geometry_score": 0.0,
                    "registration_confidence": 0.0,
                    "attention_sync_score": 0.0,
                    "alignment": {
                        "registration_geometry_reliable": False,
                    },
                }
                if method_id == "slm_wm"
                else {
                    "score": score_offset
                    + int(prompt["prompt_index"]) / 1000.0,
                }
            ),
        }
        for prompt in _prompt_rows()
    )
    if method_id == "slm_wm":
        raw_rows = tuple(
            _bind_attention_alignment_gate(row) for row in raw_rows
        )
    calibration_rows = tuple(
        row for row in raw_rows if row["split"] == "calibration"
    )
    if method_id == "slm_wm":
        protocol = calibrate_complete_evidence_protocol(
            calibration_rows,
            TARGET_FPR,
            RESCUE_MARGIN_LOW,
        )
        observation_rows = apply_frozen_evidence_protocol(raw_rows, protocol)
        declaration = protocol.to_dict()
    else:
        threshold = conformal_threshold_from_clean_negative_scores(
            (float(row["score"]) for row in calibration_rows),
            TARGET_FPR,
        )
        observation_rows = tuple(
            {
                **row,
                "threshold": threshold,
                "threshold_source": FORMAL_THRESHOLD_SOURCE,
                "detection_decision": float(row["score"]) >= threshold,
            }
            for row in raw_rows
        )
        audit = audit_fixed_fpr_observation_threshold(
            observation_rows,
            target_fpr=TARGET_FPR,
            expected_calibration_negative_count=33,
        )
        declaration = {
            "calibrated_detection_threshold": threshold,
            "threshold_digest": audit.threshold_digest,
        }
    identity = f"{repeat.randomization_repeat_id}:{method_id}"
    return MethodRepeatObservationSource(
        paper_run_name=PAPER_RUN_NAME,
        method_id=method_id,
        randomization_repeat_id=repeat.randomization_repeat_id,
        generation_model_id=MODEL_ID,
        generation_model_revision=MODEL_REVISION,
        randomization_aggregate_package_sha256=build_stable_digest(
            {"aggregate_package": "shared"}
        ),
        randomization_aggregate_digest=build_stable_digest(
            {"aggregate": "shared"}
        ),
        common_code_version="a" * 40,
        randomization_repeat_component_sha256=build_stable_digest(
            {"repeat_component": repeat.randomization_repeat_id}
        ),
        randomization_repeat_evidence_manifest_digest=build_stable_digest(
            {"repeat_manifest": repeat.randomization_repeat_id}
        ),
        component_content_digest=build_stable_digest(
            {"component_content": repeat.randomization_repeat_id}
        ),
        leaf_package_family=METHOD_LEAF_PACKAGE_FAMILY[method_id],
        leaf_package_sha256=build_stable_digest({"leaf": identity}),
        observation_source_sha256=build_stable_digest(
            {"observation": identity}
        ),
        threshold_declaration_source_sha256=build_stable_digest(
            {"threshold_declaration": identity}
        ),
        declared_threshold_protocol=declaration,
        observation_rows=observation_rows,
        **_member_paths(method_id, repeat.randomization_repeat_id),
    )


@pytest.fixture(scope="module")
def exact_sources() -> tuple[MethodRepeatObservationSource, ...]:
    """构造精确45个来源, 供正反例复用."""

    return tuple(
        _source(repeat_index=repeat_index, method_index=method_index)
        for repeat_index in range(9)
        for method_index in range(5)
    )


def _run(
    sources: tuple[MethodRepeatObservationSource, ...],
) -> dict[str, object]:
    """使用冻结 probe 协议执行纯阈值重算."""

    return recompute_exact_method_repeat_fixed_fpr(
        sources,
        prompt_rows=_prompt_rows(),
        paper_run_name=PAPER_RUN_NAME,
        target_fpr=TARGET_FPR,
        expected_model_id=MODEL_ID,
        expected_model_revision=MODEL_REVISION,
        expected_base_seed=EXPECTED_BASE_SEED,
        main_rescue_margin_low=RESCUE_MARGIN_LOW,
    )


def _shift_generation_seed(
    source: MethodRepeatObservationSource,
    shift: int,
) -> MethodRepeatObservationSource:
    """构造五方法仍彼此一致、但整体偏离冻结 base seed 的反例."""

    rows = []
    for source_row in source.observation_rows:
        row = dict(source_row)
        row["generation_seed_random"] = (
            int(row["generation_seed_random"]) + shift
        )
        formal_payload = {
            field_name: row[field_name]
            for field_name in (
                "randomization_repeat_id",
                "generation_seed_index",
                "generation_seed_offset",
                "watermark_key_index",
                "generation_seed_random",
                "watermark_key_seed_random",
                "formal_randomization_protocol_digest",
                "watermark_key_material_digest_random",
            )
        }
        row["formal_randomization_identity_digest_random"] = (
            build_stable_digest(formal_payload)
        )
        base_identity_payload = {
            "generation_seed_random": row["generation_seed_random"],
            "base_latent_generation_protocol": (
                FORMAL_BASE_LATENT_GENERATION_PROTOCOL
            ),
            "base_latent_keyed_prg_version": (
                formal_randomization_protocol_record()[
                    "base_latent_keyed_prg_version"
                ]
            ),
            "base_latent_keyed_prg_protocol_digest": (
                formal_randomization_protocol_record()[
                    "base_latent_keyed_prg_protocol_digest"
                ]
            ),
            "base_latent_dtype": FORMAL_BASE_LATENT_DTYPE,
            "base_latent_shape": list(FORMAL_BASE_LATENT_SHAPE),
            "formal_randomization_protocol_digest": row[
                "formal_randomization_protocol_digest"
            ],
            "base_latent_content_digest_random": row[
                "base_latent_content_digest_random"
            ],
        }
        row["base_latent_identity_digest_random"] = build_stable_digest(
            base_identity_payload
        )
        rows.append(row)
    return replace(source, observation_rows=tuple(rows))


def test_exact_method_repeat_recomputation_produces_45_independent_thresholds(
    exact_sources: tuple[MethodRepeatObservationSource, ...],
) -> None:
    result = _run(exact_sources)
    records = result["threshold_records"]
    report = result["report"]

    assert len(records) == METHOD_REPEAT_THRESHOLD_COUNT == 45
    assert report["threshold_calculation_unit"] == "method_repeat"
    assert report["repeat_threshold_counts"] == {
        repeat.randomization_repeat_id: 5
        for repeat in formal_randomization_repeats()
    }
    assert report["method_repeat_fixed_fpr_recomputation_ready"] is True
    assert report["expected_base_seed"] == EXPECTED_BASE_SEED
    assert report["supports_paper_claim"] is False
    tree_ring_thresholds = {
        row["calibrated_detection_threshold"]
        for row in records
        if row["method_id"] == "tree_ring"
    }
    assert len(tree_ring_thresholds) == 9
    assert all(row["supports_paper_claim"] is False for row in records)
    assert all(
        row["randomization_aggregate_package_sha256"] for row in records
    )
    assert all(
        row["randomization_repeat_component_sha256"] for row in records
    )
    assert all(row["leaf_package_sha256"] for row in records)
    assert all(row["observation_source_sha256"] for row in records)
    assert all(
        row["threshold_declaration_source_sha256"] for row in records
    )
    assert all(
        row["base_latent_generation_protocol"]
        == FORMAL_BASE_LATENT_GENERATION_PROTOCOL
        for row in result["fairness_records"]
    )


def test_recomputation_rejects_grid_wide_consistent_wrong_generation_seed(
    exact_sources: tuple[MethodRepeatObservationSource, ...],
) -> None:
    shifted_sources = tuple(
        _shift_generation_seed(source, 1) for source in exact_sources
    )

    with pytest.raises(MethodRepeatFixedFprError, match="base seed 公式"):
        _run(shifted_sources)


def test_recomputation_rejects_forged_main_base_latent_identity(
    exact_sources: tuple[MethodRepeatObservationSource, ...],
) -> None:
    source = exact_sources[0]
    rows = [dict(row) for row in source.observation_rows]
    rows[0]["base_latent_identity_digest_random"] = "f" * 64
    forged = replace(source, observation_rows=tuple(rows))

    with pytest.raises(
        MethodRepeatFixedFprError,
        match="精确 payload 重建",
    ):
        _run((forged, *exact_sources[1:]))


def test_recomputation_rejects_baseline_prompt_text_drift(
    exact_sources: tuple[MethodRepeatObservationSource, ...],
) -> None:
    source_index = 1
    source = exact_sources[source_index]
    rows = [dict(row) for row in source.observation_rows]
    rows[0]["prompt_text"] = "A drifted prompt"
    drifted = replace(source, observation_rows=tuple(rows))

    with pytest.raises(MethodRepeatFixedFprError, match="Prompt 文本"):
        _run(
            (
                *exact_sources[:source_index],
                drifted,
                *exact_sources[source_index + 1 :],
            )
        )


def test_recomputation_rejects_prompt_id_not_derived_from_text(
    exact_sources: tuple[MethodRepeatObservationSource, ...],
) -> None:
    prompt_rows = [dict(row) for row in _prompt_rows()]
    prompt_rows[0]["prompt_id"] = "prompt_not_derived_from_text"

    with pytest.raises(MethodRepeatFixedFprError, match="Prompt ID"):
        recompute_exact_method_repeat_fixed_fpr(
            exact_sources,
            prompt_rows=tuple(prompt_rows),
            paper_run_name=PAPER_RUN_NAME,
            target_fpr=TARGET_FPR,
            expected_model_id=MODEL_ID,
            expected_model_revision=MODEL_REVISION,
            expected_base_seed=EXPECTED_BASE_SEED,
            main_rescue_margin_low=RESCUE_MARGIN_LOW,
        )


def test_exact_method_repeat_recomputation_rejects_missing_repeat_method_key(
    exact_sources: tuple[MethodRepeatObservationSource, ...],
) -> None:
    with pytest.raises(
        MethodRepeatFixedFprError,
        match="9个重复与5个方法",
    ):
        _run(exact_sources[:-1])


def test_exact_method_repeat_recomputation_rejects_calibration_duplicate(
    exact_sources: tuple[MethodRepeatObservationSource, ...],
) -> None:
    first = exact_sources[0]
    duplicated = replace(
        first,
        observation_rows=(*first.observation_rows, dict(first.observation_rows[3])),
    )

    with pytest.raises(MethodRepeatFixedFprError, match="精确提供一条"):
        _run((duplicated, *exact_sources[1:]))


def test_exact_method_repeat_recomputation_rejects_cross_method_latent_drift(
    exact_sources: tuple[MethodRepeatObservationSource, ...],
) -> None:
    source_index = 1
    source = exact_sources[source_index]
    rows = [dict(row) for row in source.observation_rows]
    rows[0]["base_latent_content_digest_random"] = "e" * 64
    rows[0]["base_latent_identity_digest_random"] = "d" * 64
    drifted = replace(source, observation_rows=tuple(rows))

    with pytest.raises(MethodRepeatFixedFprError, match="五方法"):
        _run((*exact_sources[:source_index], drifted, *exact_sources[source_index + 1 :]))


def test_exact_method_repeat_recomputation_rejects_declared_threshold_drift(
    exact_sources: tuple[MethodRepeatObservationSource, ...],
) -> None:
    source_index = 1
    source = exact_sources[source_index]
    drifted = replace(
        source,
        declared_threshold_protocol={
            **source.declared_threshold_protocol,
            "threshold_digest": "0" * 64,
        },
    )

    with pytest.raises(MethodRepeatFixedFprError, match="重算或逐条判定"):
        _run((*exact_sources[:source_index], drifted, *exact_sources[source_index + 1 :]))


def test_exact_method_repeat_rejects_float_anchor_in_main_protocol(
    exact_sources: tuple[MethodRepeatObservationSource, ...],
) -> None:
    """主方法重复声明不得把整数锚点改为宽松相等的浮点数."""

    source = exact_sources[0]
    drifted = replace(
        source,
        declared_threshold_protocol={
            **source.declared_threshold_protocol,
            "attention_anchor_count": 12.0,
        },
    )

    with pytest.raises(MethodRepeatFixedFprError, match="正文或摘要无效"):
        _run((drifted, *exact_sources[1:]))


def test_exact_method_repeat_recomputation_rejects_mixed_aggregate_identity(
    exact_sources: tuple[MethodRepeatObservationSource, ...],
) -> None:
    source_index = 1
    source = exact_sources[source_index]
    drifted = replace(
        source,
        randomization_aggregate_package_sha256="f" * 64,
    )

    with pytest.raises(MethodRepeatFixedFprError, match="同一个精确聚合包"):
        _run(
            (
                *exact_sources[:source_index],
                drifted,
                *exact_sources[source_index + 1 :],
            )
        )


def test_source_requires_nested_member_byte_digests() -> None:
    source = _source(repeat_index=0, method_index=0)

    with pytest.raises(MethodRepeatFixedFprError, match="leaf_package_sha256"):
        replace(source, leaf_package_sha256="not-a-sha256")
