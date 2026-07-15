"""验证 Prompt-clustered 配对优势统计."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.protocol.attacks import (
    default_attack_configs,
    formal_attack_seed_protocol_record,
    formal_attack_seed_random,
)
from experiments.protocol.fixed_fpr_observation_audit import FORMAL_THRESHOLD_SOURCE
from experiments.protocol.paper_fixed_fpr import (
    build_paper_attack_matrix_rows,
)
from main.core.digest import build_stable_digest
from paper_experiments.analysis import paired_superiority as paired_module
from paper_experiments.analysis.fixed_fpr_threshold_audit import (
    build_fixed_fpr_threshold_audit_report,
)
from paper_experiments.analysis.paired_superiority import (
    BOOTSTRAP_ANALYSIS_SCHEMA,
    BOOTSTRAP_BIT_GENERATOR,
    BOOTSTRAP_QUANTILE_METHOD,
    CLAIM_P_VALUE_METHOD,
    DEFAULT_BOOTSTRAP_RESAMPLE_COUNT,
    PRIMARY_BASELINE_IDS,
    SHARP_NULL_DIAGNOSTIC_METHOD,
    THRESHOLD_AUDIT_FIELDS,
    THRESHOLD_AUDIT_METHOD_IDS,
    PairedSuperiorityError,
    build_quality_matching_records,
    build_quality_matched_superiority_rows,
    build_paired_outcome_set_digest,
    build_paired_outcomes,
    build_paired_superiority_protocol_digest,
    build_paired_superiority_rows,
    build_paired_superiority_summary,
    canonical_attack_registry_rows,
    canonical_threshold_audit_rows,
)


pytestmark = pytest.mark.quick

PROPOSED_THRESHOLD_DIGEST = "a" * 64
BASELINE_THRESHOLD_VALUE = 0.5
BASELINE_THRESHOLD_DIGESTS = {
    baseline_id: f"{index + 1:x}" * 64
    for index, baseline_id in enumerate(PRIMARY_BASELINE_IDS)
}
UNIT_ATTACK_REGISTRY = (
    {
        "attack_id": "jpeg_compression_main",
        "attack_family": "standard_distortion",
        "attack_name": "jpeg_compression",
        "resource_profile": "full_main",
        "attack_config_digest": "b" * 64,
    },
    {
        "attack_id": "gaussian_noise_main",
        "attack_family": "standard_distortion",
        "attack_name": "gaussian_noise",
        "resource_profile": "full_main",
        "attack_config_digest": "c" * 64,
    },
)


def paired_randomization_identity(prompt_index: int) -> dict[str, object]:
    """构造主方法与 baseline 必须逐字段相同的随机身份."""

    return {
        "randomization_repeat_id": "seed_00_key_00",
        "generation_seed_index": 0,
        "generation_seed_offset": 0,
        "generation_seed_random": 1703 + prompt_index,
        "watermark_key_index": 0,
        "watermark_key_seed_random": 1729,
        "watermark_key_material_digest_random": "d" * 64,
        "formal_randomization_protocol_digest": "e" * 64,
        "formal_randomization_identity_digest_random": "f" * 64,
        "base_latent_content_digest_random": "1" * 64,
        "base_latent_identity_digest_random": "2" * 64,
    }


PAIRED_RANDOMIZATION_FIELDS = tuple(paired_randomization_identity(0))


def quality_source_rows(
    *,
    baseline_id: str = "",
    proposed_quality: float = -0.8,
    baseline_quality: float = 0.4,
) -> list[dict[str, object]]:
    """构造同一 repeat 的真实未攻击 clean-watermarked 质量记录."""

    rows: list[dict[str, object]] = []
    for prompt_index in range(2):
        prompt_id = f"quality_prompt_{prompt_index}"
        identity = paired_randomization_identity(prompt_index)
        for sample_role in ("clean_negative", "positive_source"):
            payload: dict[str, object] = {
                "prompt_id": prompt_id,
                **identity,
                "split": "test",
                "sample_role": sample_role,
                "attack_id": "",
                "attack_family": "clean",
            }
            if baseline_id:
                payload.update(
                    {
                        "baseline_id": baseline_id,
                        "image_path": f"{baseline_id}/{prompt_id}/{sample_role}.png",
                        "image_digest": build_stable_digest(
                            {
                                "baseline_id": baseline_id,
                                "prompt_id": prompt_id,
                                "sample_role": sample_role,
                            }
                        ),
                        "quality_score": baseline_quality,
                    }
                )
            else:
                payload.update(
                    {
                        "source_image_path": f"slm_wm/{prompt_id}/{sample_role}.png",
                        "source_image_digest": build_stable_digest(
                            {
                                "method_id": "slm_wm",
                                "prompt_id": prompt_id,
                                "sample_role": sample_role,
                            }
                        ),
                        "embedding_pair_ssim": proposed_quality,
                    }
                )
            rows.append(payload)
    return rows


def observation_rows(
    *,
    baseline_id: str = "",
    baseline_positive_prompt_ids: set[str] | None = None,
    prompt_count: int = 3,
    attack_registry: tuple[dict[str, str], ...] = UNIT_ATTACK_REGISTRY,
) -> list[dict[str, object]]:
    """构造可配对的 Prompt x 正式攻击观测."""

    positive_ids = baseline_positive_prompt_ids or set()
    rows = []
    for prompt_index in range(prompt_count):
        prompt_id = f"prompt_{prompt_index}"
        for attack in attack_registry:
            decision = prompt_id in positive_ids if baseline_id else True
            row: dict[str, object] = {
                "prompt_id": prompt_id,
                **paired_randomization_identity(prompt_index),
                "split": "test",
                "sample_role": (
                    "attacked_positive" if baseline_id else "positive_source"
                ),
                "attack_id": attack["attack_id"],
                "attack_family": attack["attack_family"],
                "attack_name": attack["attack_name"],
                "resource_profile": attack["resource_profile"],
                "attack_config_digest": attack["attack_config_digest"],
                "attack_seed_random": formal_attack_seed_random(
                    1703 + prompt_index,
                    attack["attack_id"],
                ),
                "formal_attack_seed_protocol_digest": (
                    formal_attack_seed_protocol_record()[
                        "formal_attack_seed_protocol_digest"
                    ]
                ),
            }
            if baseline_id:
                row.update(
                    {
                        "baseline_id": baseline_id,
                        "score": 0.9 if decision else 0.1,
                        "threshold": BASELINE_THRESHOLD_VALUE,
                        "threshold_source": FORMAL_THRESHOLD_SOURCE,
                        "detection_decision": decision,
                    }
                )
            else:
                row.update(
                    {
                        "frozen_threshold_digest": PROPOSED_THRESHOLD_DIGEST,
                        "formal_evidence_positive": decision,
                    }
                )
            rows.append(row)
    return rows


def paired_outcomes(
    *,
    prompt_count: int = 3,
    baseline_positive_prompt_ids: set[str] | None = None,
) -> tuple[dict[str, object], ...]:
    """构造精确覆盖4个 baseline 的配对 outcome."""

    proposed = observation_rows(prompt_count=prompt_count)
    return tuple(
        outcome
        for baseline_id in PRIMARY_BASELINE_IDS
        for outcome in build_paired_outcomes(
            proposed,
            observation_rows(
                baseline_id=baseline_id,
                baseline_positive_prompt_ids=baseline_positive_prompt_ids,
                prompt_count=prompt_count,
            ),
            baseline_id=baseline_id,
            proposed_method_threshold_digest=PROPOSED_THRESHOLD_DIGEST,
            baseline_method_threshold_digest=BASELINE_THRESHOLD_DIGESTS[baseline_id],
            baseline_calibrated_detection_threshold=BASELINE_THRESHOLD_VALUE,
            attack_registry_rows=UNIT_ATTACK_REGISTRY,
        )
    )


def quality_matched_paired_outcomes(
    *,
    baseline_positive_prompt_ids: set[str] | None = None,
) -> tuple[dict[str, object], ...]:
    """构造各 baseline 排除不同 Prompt 的质量匹配 outcome."""

    prompt_count = 5
    proposed_rows = observation_rows(prompt_count=prompt_count)
    proposed_rows.extend(
        {
            "prompt_id": f"prompt_{prompt_index}",
            **paired_randomization_identity(prompt_index),
            "split": "test",
            "sample_role": "positive_source",
            "attack_family": "clean",
            "embedding_pair_ssim": 0.95,
        }
        for prompt_index in range(prompt_count)
    )
    outcomes: list[dict[str, object]] = []
    for baseline_index, baseline_id in enumerate(PRIMARY_BASELINE_IDS):
        baseline_rows = observation_rows(
            baseline_id=baseline_id,
            baseline_positive_prompt_ids=baseline_positive_prompt_ids,
            prompt_count=prompt_count,
        )
        baseline_rows.extend(
            {
                "prompt_id": f"prompt_{prompt_index}",
                **paired_randomization_identity(prompt_index),
                "split": "test",
                "sample_role": "positive_source",
                "attack_family": "clean",
                "baseline_id": baseline_id,
                "quality_score": (0.80 if prompt_index == baseline_index else 0.95),
            }
            for prompt_index in range(prompt_count)
        )
        outcomes.extend(
            build_paired_outcomes(
                proposed_rows,
                baseline_rows,
                baseline_id=baseline_id,
                proposed_method_threshold_digest=PROPOSED_THRESHOLD_DIGEST,
                baseline_method_threshold_digest=BASELINE_THRESHOLD_DIGESTS[
                    baseline_id
                ],
                baseline_calibrated_detection_threshold=BASELINE_THRESHOLD_VALUE,
                attack_registry_rows=UNIT_ATTACK_REGISTRY,
                include_quality_matching=True,
            )
        )
    return tuple(outcomes)


def file_sha256(path: Path) -> str:
    """计算测试输入文件的原始字节 SHA-256."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_paired_outcomes_require_exact_prompt_attack_set() -> None:
    """baseline 缺少任一 Prompt x attack 键时必须阻断."""

    proposed = observation_rows()
    baseline = observation_rows(
        baseline_id="tree_ring",
        baseline_positive_prompt_ids={"prompt_0"},
    )
    baseline.pop()
    with pytest.raises(PairedSuperiorityError, match="配对集合不一致"):
        build_paired_outcomes(
            proposed,
            baseline,
            baseline_id="tree_ring",
            proposed_method_threshold_digest=PROPOSED_THRESHOLD_DIGEST,
            baseline_method_threshold_digest=BASELINE_THRESHOLD_DIGESTS["tree_ring"],
            baseline_calibrated_detection_threshold=BASELINE_THRESHOLD_VALUE,
            attack_registry_rows=UNIT_ATTACK_REGISTRY,
        )


def test_paired_outcomes_only_accept_audited_decision_fields() -> None:
    """汇总判定不得替代主方法或 baseline 的正式审计判定字段."""

    proposed = observation_rows()
    baseline = observation_rows(baseline_id="tree_ring")
    proposed[0]["final_decision"] = proposed[0].pop("formal_evidence_positive")
    with pytest.raises(PairedSuperiorityError, match="formal_evidence_positive"):
        build_paired_outcomes(
            proposed,
            baseline,
            baseline_id="tree_ring",
            proposed_method_threshold_digest=PROPOSED_THRESHOLD_DIGEST,
            baseline_method_threshold_digest=BASELINE_THRESHOLD_DIGESTS["tree_ring"],
            baseline_calibrated_detection_threshold=BASELINE_THRESHOLD_VALUE,
            attack_registry_rows=UNIT_ATTACK_REGISTRY,
        )

    proposed = observation_rows()
    baseline = observation_rows(baseline_id="tree_ring")
    baseline[0]["final_decision"] = baseline[0].pop("detection_decision")
    with pytest.raises(PairedSuperiorityError, match="detection_decision"):
        build_paired_outcomes(
            proposed,
            baseline,
            baseline_id="tree_ring",
            proposed_method_threshold_digest=PROPOSED_THRESHOLD_DIGEST,
            baseline_method_threshold_digest=BASELINE_THRESHOLD_DIGESTS["tree_ring"],
            baseline_calibrated_detection_threshold=BASELINE_THRESHOLD_VALUE,
            attack_registry_rows=UNIT_ATTACK_REGISTRY,
        )

    baseline = observation_rows(baseline_id="tree_ring")
    baseline[0]["final_decision"] = not baseline[0]["detection_decision"]
    with pytest.raises(PairedSuperiorityError, match="不一致"):
        build_paired_outcomes(
            observation_rows(),
            baseline,
            baseline_id="tree_ring",
            proposed_method_threshold_digest=PROPOSED_THRESHOLD_DIGEST,
            baseline_method_threshold_digest=BASELINE_THRESHOLD_DIGESTS["tree_ring"],
            baseline_calibrated_detection_threshold=BASELINE_THRESHOLD_VALUE,
            attack_registry_rows=UNIT_ATTACK_REGISTRY,
        )


def test_paired_outcomes_bind_thresholds_and_formal_attack_registry() -> None:
    """每条 outcome 必须由顶层阈值记录和样本分数重建正式判定."""

    baseline_rows = observation_rows(baseline_id="tree_ring")
    assert all("threshold_digest" not in row for row in baseline_rows)
    outcomes = build_paired_outcomes(
        observation_rows(),
        baseline_rows,
        baseline_id="tree_ring",
        proposed_method_threshold_digest=PROPOSED_THRESHOLD_DIGEST,
        baseline_method_threshold_digest=BASELINE_THRESHOLD_DIGESTS["tree_ring"],
        baseline_calibrated_detection_threshold=BASELINE_THRESHOLD_VALUE,
        attack_registry_rows=UNIT_ATTACK_REGISTRY,
    )
    first = outcomes[0]
    assert first["proposed_method_threshold_digest"] == PROPOSED_THRESHOLD_DIGEST
    assert (
        first["baseline_method_threshold_digest"]
        == BASELINE_THRESHOLD_DIGESTS["tree_ring"]
    )
    registry_by_id = {row["attack_id"]: row for row in UNIT_ATTACK_REGISTRY}
    assert (
        first["resource_profile"]
        == registry_by_id[first["attack_id"]]["resource_profile"]
    )
    assert (
        first["attack_config_digest"]
        == registry_by_id[first["attack_id"]]["attack_config_digest"]
    )


@pytest.mark.parametrize(
    ("field_name", "field_value", "error_pattern"),
    (
        ("threshold", 0.6, "未使用审计冻结阈值"),
        ("threshold_source", "manual", "正式 calibration 阈值来源"),
        ("detection_decision", True, "无法由正式分数和阈值重算"),
    ),
)
def test_paired_outcomes_reject_unreconstructable_baseline_decision(
    field_name: str,
    field_value: object,
    error_pattern: str,
) -> None:
    """baseline 阈值来源、数值或判定漂移时必须阻断论文配对统计."""

    baseline_rows = observation_rows(baseline_id="tree_ring")
    baseline_rows[0][field_name] = field_value
    with pytest.raises(PairedSuperiorityError, match=error_pattern):
        build_paired_outcomes(
            observation_rows(),
            baseline_rows,
            baseline_id="tree_ring",
            proposed_method_threshold_digest=PROPOSED_THRESHOLD_DIGEST,
            baseline_method_threshold_digest=BASELINE_THRESHOLD_DIGESTS[
                "tree_ring"
            ],
            baseline_calibrated_detection_threshold=BASELINE_THRESHOLD_VALUE,
            attack_registry_rows=UNIT_ATTACK_REGISTRY,
        )


def test_paired_outcomes_require_real_attacked_image_only_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """正式跨重复路径必须绑定真实攻击图像和仅图像盲检原子."""

    proposed = observation_rows()
    baseline = observation_rows(baseline_id="tree_ring")
    for row_index, row in enumerate(proposed):
        image_digest = build_stable_digest({"proposed_image": row_index})
        row.update(
            {
                "attack_performed": True,
                "attacked_image_path": f"outputs/attacked/proposed_{row_index}.png",
                "attacked_image_digest": image_digest,
                "evaluated_image_digest": image_digest,
                "measurement_digest": build_stable_digest({"detector": row_index}),
                "metadata": {
                    "detector_input_access_mode": "image_key_public_model_only",
                    "blind_image_detector": True,
                    "generation_latent_trace_required": False,
                },
            }
        )
    for row_index, row in enumerate(baseline):
        row.update(
            {
                "image_path": f"outputs/attacked/baseline_{row_index}.png",
                "image_digest": build_stable_digest({"baseline_image": row_index}),
            }
        )
    monkeypatch.setattr(
        paired_module,
        "validate_image_only_measurement_projection_record",
        lambda record: dict(record),
    )

    outcomes = build_paired_outcomes(
        proposed,
        baseline,
        baseline_id="tree_ring",
        proposed_method_threshold_digest=PROPOSED_THRESHOLD_DIGEST,
        baseline_method_threshold_digest=BASELINE_THRESHOLD_DIGESTS["tree_ring"],
        baseline_calibrated_detection_threshold=BASELINE_THRESHOLD_VALUE,
        attack_registry_rows=UNIT_ATTACK_REGISTRY,
        require_image_only_evidence=True,
    )

    assert len(outcomes) == len(proposed)
    assert len(outcomes[0]["proposed_attacked_image_digest"]) == 64
    assert len(outcomes[0]["baseline_attacked_image_digest"]) == 64
    assert len(outcomes[0]["proposed_measurement_digest"]) == 64
    proposed[0]["metadata"]["blind_image_detector"] = False
    with pytest.raises(PairedSuperiorityError, match="仅图像盲检协议"):
        build_paired_outcomes(
            proposed,
            baseline,
            baseline_id="tree_ring",
            proposed_method_threshold_digest=PROPOSED_THRESHOLD_DIGEST,
            baseline_method_threshold_digest=BASELINE_THRESHOLD_DIGESTS["tree_ring"],
            baseline_calibrated_detection_threshold=BASELINE_THRESHOLD_VALUE,
            attack_registry_rows=UNIT_ATTACK_REGISTRY,
            require_image_only_evidence=True,
        )


def test_paired_outcomes_reject_different_base_latent_identity() -> None:
    """基础 latent 内容不同的样本不得进入方法级配对优势统计."""

    proposed = observation_rows()
    baseline = observation_rows(baseline_id="tree_ring")
    baseline[0]["base_latent_content_digest_random"] = "3" * 64

    with pytest.raises(
        PairedSuperiorityError,
        match="相同的种子、密钥重复和基础 latent",
    ):
        build_paired_outcomes(
            proposed,
            baseline,
            baseline_id="tree_ring",
            proposed_method_threshold_digest=PROPOSED_THRESHOLD_DIGEST,
            baseline_method_threshold_digest=BASELINE_THRESHOLD_DIGESTS["tree_ring"],
            baseline_calibrated_detection_threshold=BASELINE_THRESHOLD_VALUE,
            attack_registry_rows=UNIT_ATTACK_REGISTRY,
        )


def test_paired_outcomes_reject_attack_seed_drift() -> None:
    """主方法与 baseline 任一方攻击 seed 漂移时不得进入配对统计."""

    proposed = observation_rows()
    baseline = observation_rows(baseline_id="tree_ring")
    baseline[0]["attack_seed_random"] = int(baseline[0]["attack_seed_random"]) + 1

    with pytest.raises(PairedSuperiorityError, match="attack_seed_random"):
        build_paired_outcomes(
            proposed,
            baseline,
            baseline_id="tree_ring",
            proposed_method_threshold_digest=PROPOSED_THRESHOLD_DIGEST,
            baseline_method_threshold_digest=BASELINE_THRESHOLD_DIGESTS["tree_ring"],
            baseline_calibrated_detection_threshold=BASELINE_THRESHOLD_VALUE,
            attack_registry_rows=UNIT_ATTACK_REGISTRY,
        )


def test_clustered_superiority_discloses_non_superior_result() -> None:
    """完整统计可披露不优于 baseline 的结果, 但不得通过优势门禁."""

    outcomes = paired_outcomes(
        baseline_positive_prompt_ids={"prompt_0", "prompt_1", "prompt_2"}
    )
    rows = build_paired_superiority_rows(
        outcomes,
        protocol_digest="d" * 64,
    )
    assert all(row["one_sided_bounded_hoeffding_mean_p_value"] == 1.0 for row in rows)
    assert all(row["paired_superiority_ready"] is False for row in rows)
    summary = build_paired_superiority_summary(rows, paired_outcomes=outcomes)
    assert summary["overall_paired_superiority_ready"] is False


def test_bounded_mean_claim_and_exact_sharp_null_diagnostic() -> None:
    """正式 claim 使用 Hoeffding, exact sign-flip 只提供 sharp-null 诊断."""

    outcomes = paired_outcomes()
    first = build_paired_superiority_rows(
        outcomes,
        protocol_digest="e" * 64,
    )
    second = build_paired_superiority_rows(
        outcomes,
        protocol_digest="e" * 64,
    )
    assert first == second
    assert all(row["mean_paired_difference_ci_low"] == 1.0 for row in first)
    assert all(row["claim_p_value_method"] == CLAIM_P_VALUE_METHOD for row in first)
    assert all(
        row["sharp_null_diagnostic_method"] == SHARP_NULL_DIAGNOSTIC_METHOD
        for row in first
    )
    assert all(
        row["exact_prompt_cluster_sign_flip_p_value_is_diagnostic"] is True
        for row in first
    )
    assert all(
        row["one_sided_exact_prompt_cluster_sign_flip_p_value"] == 0.125
        for row in first
    )
    assert all(
        row["one_sided_bounded_hoeffding_mean_p_value"]
        == pytest.approx(0.22313016014842982)
        for row in first
    )
    assert all(
        row["bootstrap_resample_count"] == DEFAULT_BOOTSTRAP_RESAMPLE_COUNT
        for row in first
    )
    assert all(
        row["bootstrap_analysis_schema"] == BOOTSTRAP_ANALYSIS_SCHEMA for row in first
    )
    assert all(
        row["bootstrap_bit_generator"] == BOOTSTRAP_BIT_GENERATOR for row in first
    )
    assert all(
        row["bootstrap_quantile_method"] == BOOTSTRAP_QUANTILE_METHOD for row in first
    )
    assert all("permutation_resample_count" not in row for row in first)
    assert all("permutation_seed_digest_random" not in row for row in first)
    assert all(
        row["holm_adjusted_p_value"] >= row["one_sided_bounded_hoeffding_mean_p_value"]
        for row in first
    )
    assert all(row["paired_superiority_ready"] is False for row in first)

    outcome_set_digest = build_paired_outcome_set_digest(outcomes)
    prompt_id_digest = build_stable_digest(["prompt_0", "prompt_1", "prompt_2"])
    attack_registry_digest = build_stable_digest(
        list(canonical_attack_registry_rows(UNIT_ATTACK_REGISTRY))
    )
    expected_tree_seed = build_stable_digest(
        {
            "analysis_schema": BOOTSTRAP_ANALYSIS_SCHEMA,
            "baseline_id": "tree_ring",
            "paired_test_prompt_id_digest": prompt_id_digest,
            "paired_attack_registry_digest": attack_registry_digest,
            "paired_outcome_set_digest": outcome_set_digest,
            "confidence_level": 0.95,
            "bootstrap_resample_count": DEFAULT_BOOTSTRAP_RESAMPLE_COUNT,
            "bootstrap_bit_generator": BOOTSTRAP_BIT_GENERATOR,
            "bootstrap_quantile_method": BOOTSTRAP_QUANTILE_METHOD,
        }
    )
    tree_row = next(row for row in first if row["baseline_id"] == "tree_ring")
    assert tree_row["bootstrap_seed_digest_random"] == expected_tree_seed
    assert tree_row["paired_test_prompt_id_digest"] == prompt_id_digest
    assert tree_row["paired_attack_registry_digest"] == attack_registry_digest
    assert tree_row["paired_outcome_set_digest"] == outcome_set_digest
    assert tree_row["holm_adjusted_p_value"] == pytest.approx(
        min(1.0, 4.0 * tree_row["one_sided_bounded_hoeffding_mean_p_value"])
    )

    changed_protocol = build_paired_superiority_rows(
        outcomes,
        protocol_digest="f" * 64,
    )
    assert [row["bootstrap_seed_digest_random"] for row in changed_protocol] == [
        row["bootstrap_seed_digest_random"] for row in first
    ]
    assert [row["protocol_digest"] for row in changed_protocol] != [
        row["protocol_digest"] for row in first
    ]

    changed_outcomes = paired_outcomes(baseline_positive_prompt_ids={"prompt_0"})
    changed_data = build_paired_superiority_rows(
        changed_outcomes,
        protocol_digest="e" * 64,
    )
    assert [row["bootstrap_seed_digest_random"] for row in changed_data] != [
        row["bootstrap_seed_digest_random"] for row in first
    ]

    with pytest.raises(PairedSuperiorityError, match="不得小于100000"):
        build_paired_superiority_rows(
            outcomes,
            protocol_digest="e" * 64,
            bootstrap_resample_count=99_999,
        )


def test_summary_requires_exact_four_baselines_and_binds_prompt_ids() -> None:
    """总体摘要同时约束 baseline exact set 与规范 test Prompt 摘要."""

    outcomes = paired_outcomes()
    rows = [
        {
            "baseline_id": baseline_id,
            "paired_superiority_ready": True,
        }
        for baseline_id in PRIMARY_BASELINE_IDS[:-1]
    ]
    summary = build_paired_superiority_summary(rows, paired_outcomes=outcomes)
    assert summary["paired_superiority_exact_set_ready"] is False
    assert summary["overall_paired_superiority_ready"] is False
    assert summary["paired_test_prompt_count"] == 3
    assert summary["paired_test_prompt_id_digest"] == build_stable_digest(
        ["prompt_0", "prompt_1", "prompt_2"]
    )


def test_protocol_digest_covers_canonical_threshold_rows_and_report() -> None:
    """threshold 行或报告任一事实变化都必须改变配对统计协议摘要."""

    rows = []
    for index, method_id in enumerate(reversed(THRESHOLD_AUDIT_METHOD_IDS)):
        rows.append(
            {
                "method_id": method_id,
                "threshold_source": "nested_calibration_threshold_freeze_conformal",
                "target_fpr": "0.1",
                "calibration_clean_negative_count": "36",
                "threshold_freeze_negative_count": "24",
                "calibration_partition_digest": "b" * 64,
                "threshold_freeze_prompt_id_digest": "c" * 64,
                "test_clean_negative_count": "34",
                "calibrated_detection_threshold": "0.5",
                "threshold_digest": f"{index + 1:x}" * 64,
                "observation_source_sha256": f"{index + 6:x}" * 64,
                "protocol_target_ready": "True",
                "protocol_value_ready": "True",
                "detection_decision_ready": "True",
                "split_count_ready": "True",
                "fixed_fpr_threshold_ready": "True",
                "supports_paper_claim": "False",
            }
        )
    normalized_rows = canonical_threshold_audit_rows(rows)
    report = build_fixed_fpr_threshold_audit_report(
        normalized_rows,
        paper_run_name="probe_paper",
        target_fpr=0.1,
    )
    assert report["shared_calibration_partition_ready"] is True
    first = build_paired_superiority_protocol_digest(report, rows, "f" * 64)
    second = build_paired_superiority_protocol_digest(
        {**report, "target_fpr": 0.2},
        rows,
        "f" * 64,
    )
    assert first != second
    assert canonical_threshold_audit_rows(rows) == canonical_threshold_audit_rows(
        reversed(rows)
    )


def test_quality_matching_supports_baseline_specific_prompt_subsets() -> None:
    """质量匹配应允许各 baseline 命中不同 Prompt 且不读取检测标签."""

    negative_outcomes = quality_matched_paired_outcomes()
    positive_outcomes = quality_matched_paired_outcomes(
        baseline_positive_prompt_ids={f"prompt_{index}" for index in range(5)}
    )
    negative_selection = {
        (row["baseline_id"], row["prompt_id"]): row["quality_matched"]
        for row in negative_outcomes
    }
    positive_selection = {
        (row["baseline_id"], row["prompt_id"]): row["quality_matched"]
        for row in positive_outcomes
    }
    assert negative_selection == positive_selection

    quality_rows = build_quality_matched_superiority_rows(
        negative_outcomes,
        protocol_digest="3" * 64,
    )
    assert {
        row["baseline_id"]: (
            row["matched_prompt_count"],
            row["unmatched_prompt_count"],
            row["quality_match_coverage_ready"],
        )
        for row in quality_rows
    } == {baseline_id: (4, 1, True) for baseline_id in PRIMARY_BASELINE_IDS}
    unmatched_by_baseline = {
        baseline_id: {
            prompt_id
            for (current_baseline_id, prompt_id), matched in (
                negative_selection.items()
            )
            if current_baseline_id == baseline_id and matched is False
        }
        for baseline_id in PRIMARY_BASELINE_IDS
    }
    assert len({next(iter(value)) for value in unmatched_by_baseline.values()}) == 4


def test_quality_record_builder_accepts_negative_ssim_and_gap_above_one() -> None:
    """标准 SSIM 域为 [-1, 1], 两方法差值绝对值可以超过1."""

    records = build_quality_matching_records(
        quality_source_rows(proposed_quality=-0.8),
        quality_source_rows(
            baseline_id="tree_ring",
            baseline_quality=0.4,
        ),
        baseline_id="tree_ring",
    )

    assert len(records) == 2
    assert all(
        record["embedding_pair_ssim_gap"] == pytest.approx(-1.2)
        and record["absolute_embedding_pair_ssim_gap"] == pytest.approx(1.2)
        and record["quality_matched"] is False
        for record in records
    )


def test_quality_record_builder_selection_does_not_read_detector_labels() -> None:
    """翻转检测判定不得改变只由未攻击 SSIM 构造的质量记录."""

    proposed = quality_source_rows(proposed_quality=0.95)
    baseline = quality_source_rows(
        baseline_id="tree_ring",
        baseline_quality=0.94,
    )
    first = build_quality_matching_records(
        proposed,
        baseline,
        baseline_id="tree_ring",
    )
    for row in proposed:
        row["formal_evidence_positive"] = False
    for row in baseline:
        row["detection_decision"] = True
    second = build_quality_matching_records(
        proposed,
        baseline,
        baseline_id="tree_ring",
    )

    assert second == first


def test_quality_record_builder_rejects_missing_image_or_random_identity_drift() -> (
    None
):
    """质量值必须同时绑定真实图像摘要和相同 seed、key、基础 latent."""

    proposed = quality_source_rows(proposed_quality=0.95)
    baseline = quality_source_rows(
        baseline_id="tree_ring",
        baseline_quality=0.94,
    )
    proposed[0]["source_image_digest"] = ""
    with pytest.raises(PairedSuperiorityError, match="小写 SHA-256"):
        build_quality_matching_records(
            proposed,
            baseline,
            baseline_id="tree_ring",
        )

    proposed = quality_source_rows(proposed_quality=0.95)
    baseline = quality_source_rows(
        baseline_id="tree_ring",
        baseline_quality=0.94,
    )
    baseline[1]["generation_seed_random"] = 9999
    with pytest.raises(PairedSuperiorityError, match="随机身份不一致"):
        build_quality_matching_records(
            proposed,
            baseline,
            baseline_id="tree_ring",
        )
