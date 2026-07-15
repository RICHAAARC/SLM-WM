"""验证精确9重复数据集质量的公式重建与集合反例."""

from __future__ import annotations

from copy import deepcopy
import random

import pytest

from experiments.protocol.dataset_quality import (
    FORMAL_FEATURE_BACKEND,
    FORMAL_FEATURE_EXTRACTOR_ID,
    rebuild_formal_fid_kid_metric_rows,
)
from experiments.protocol.formal_randomization import (
    formal_randomization_repeat_ids,
)
from experiments.protocol.attacks import (
    attack_config_digest,
    default_attack_configs,
)
from experiments.artifacts.paired_quality_outputs import (
    FORMAL_CLIP_FEATURE_BACKEND,
    PAIRED_QUALITY_METRIC_RECORD_SCHEMA,
)
from main.core.digest import build_stable_digest
from paper_experiments.analysis import randomization_dataset_quality as analysis


pytestmark = pytest.mark.quick


def _quality_evidence() -> tuple[
    tuple[str, ...],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    """构造70个 Prompt 与9重复的小维正式特征."""

    prompt_ids = tuple(f"probe_prompt_{index:03d}" for index in range(70))
    memberships: list[dict[str, object]] = []
    features: list[dict[str, object]] = []
    for repeat_index, repeat_id in enumerate(
        formal_randomization_repeat_ids()
    ):
        for prompt_index, prompt_id in enumerate(prompt_ids):
            record_digest = build_stable_digest(
                {"repeat_id": repeat_id, "prompt_id": prompt_id}
            )
            record_id = f"dataset_quality_record_{record_digest[:16]}"
            source_digest = build_stable_digest(
                {"role": "source", "record_id": record_id}
            )
            comparison_digest = build_stable_digest(
                {"role": "comparison", "record_id": record_id}
            )
            memberships.append(
                {
                    "randomization_repeat_id": repeat_id,
                    "prompt_id": prompt_id,
                    "dataset_quality_record_id": record_id,
                    "dataset_quality_record_digest": record_digest,
                    "source_image_digest": source_digest,
                    "comparison_image_digest": comparison_digest,
                }
            )
            base_vector = [
                prompt_index / 70.0,
                repeat_index / 9.0,
                ((prompt_index + repeat_index) % 11) / 11.0,
            ]
            for role, offset, image_digest in (
                ("source", 0.0, source_digest),
                ("comparison", 0.01, comparison_digest),
            ):
                features.append(
                    {
                        "dataset_quality_record_id": record_id,
                        "dataset_quality_image_role": role,
                        "feature_backend": FORMAL_FEATURE_BACKEND,
                        "feature_extractor_id": FORMAL_FEATURE_EXTRACTOR_ID,
                        "feature_dimension": 3,
                        "feature_vector": [
                            value + offset for value in base_vector
                        ],
                        "image_digest": image_digest,
                        "supports_paper_claim": False,
                    }
                )
    return prompt_ids, memberships, features


def _rebuild(
    prompt_ids: tuple[str, ...],
    memberships: list[dict[str, object]],
    features: list[dict[str, object]],
    monkeypatch: pytest.MonkeyPatch,
) -> analysis.RandomizationDatasetQualityStatistics:
    """把测试特征维度替换为3并调用正式集合与公式路径."""

    monkeypatch.setattr(analysis, "FORMAL_FEATURE_DIMENSION", 3)
    return analysis.rebuild_randomization_dataset_quality_statistics(
        features,
        memberships,
        paper_run_name="probe_paper",
        target_fpr=0.1,
        expected_prompt_ids=prompt_ids,
    )


def test_exact9_fid_kid_equals_independent_formula_rebuild(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """跨重复指标必须等于同一原始矩阵的独立正式公式计算."""

    prompt_ids, memberships, features = _quality_evidence()
    result = _rebuild(prompt_ids, memberships, features, monkeypatch)
    direct_rows = rebuild_formal_fid_kid_metric_rows(
        [
            record["feature_vector"]
            for record in features
            if record["dataset_quality_image_role"] == "source"
        ],
        [
            record["feature_vector"]
            for record in features
            if record["dataset_quality_image_role"] == "comparison"
        ],
        sample_pair_count=9 * 70,
    )

    assert result.summary["aggregate_quality_pair_count"] == 9 * 70
    assert result.summary["aggregate_feature_record_count"] == 2 * 9 * 70
    assert result.summary["randomization_dataset_quality_statistics_ready"] is True
    assert result.summary["supports_paper_claim"] is False
    metric_protocol = analysis.randomization_dataset_quality_metric_protocol()
    assert metric_protocol["aggregate_sample_pair_count_by_paper_run"] == {
        "probe_paper": 630,
        "pilot_paper": 6300,
        "full_paper": 63000,
    }
    assert metric_protocol[
        "randomization_kid_effective_subset_size_by_paper_run"
    ] == {
        "probe_paper": 630,
        "pilot_paper": 1000,
        "full_paper": 1000,
    }
    assert result.summary[
        "randomization_dataset_quality_metric_protocol_digest"
    ] == metric_protocol[
        "randomization_dataset_quality_metric_protocol_digest"
    ]
    assert metric_protocol[
        "randomization_dataset_quality_metric_protocol_digest"
    ] == build_stable_digest(
        {
            field_name: field_value
            for field_name, field_value in metric_protocol.items()
            if field_name
            != "randomization_dataset_quality_metric_protocol_digest"
        }
    )
    assert tuple(row["quality_metric_name"] for row in result.metric_rows) == (
        "fid",
        "kid_mean",
        "kid_std",
    )
    for rebuilt, direct in zip(result.metric_rows, direct_rows, strict=True):
        assert rebuilt["quality_metric_name"] == direct["quality_metric_name"]
        assert rebuilt["quality_metric_value"] == pytest.approx(
            direct["quality_metric_value"],
            rel=1e-12,
            abs=1e-12,
        )


def test_exact9_fid_kid_is_input_order_invariant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """原始 JSONL 物化顺序不得改变成员摘要或 FID/KID 数值."""

    prompt_ids, memberships, features = _quality_evidence()
    first = _rebuild(prompt_ids, memberships, features, monkeypatch)
    shuffled_memberships = list(memberships)
    shuffled_features = list(features)
    random.Random(17).shuffle(shuffled_memberships)
    random.Random(29).shuffle(shuffled_features)
    second = _rebuild(
        prompt_ids,
        shuffled_memberships,
        shuffled_features,
        monkeypatch,
    )

    assert second.membership_records == first.membership_records
    assert second.metric_rows == first.metric_rows
    assert second.summary == first.summary


def test_exact9_quality_preserves_measured_zero_difference_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """clean 与 watermarked 内容相同时必须完成真实测量, 不能拒绝负结果."""

    prompt_ids, memberships, features = _quality_evidence()
    source_feature_by_record_id = {
        str(record["dataset_quality_record_id"]): record
        for record in features
        if record["dataset_quality_image_role"] == "source"
    }
    for membership in memberships:
        membership["comparison_image_digest"] = membership[
            "source_image_digest"
        ]
    for feature in features:
        if feature["dataset_quality_image_role"] != "comparison":
            continue
        source_feature = source_feature_by_record_id[
            str(feature["dataset_quality_record_id"])
        ]
        feature["image_digest"] = source_feature["image_digest"]
        feature["feature_vector"] = list(source_feature["feature_vector"])

    result = _rebuild(prompt_ids, memberships, features, monkeypatch)

    assert result.summary["randomization_dataset_quality_statistics_ready"] is True
    assert result.summary["conclusion_decision"] == "evidence_incomplete"
    assert result.summary["quality_subclaim_decisions"][
        "distributional_preservation_noninferiority"
    ]["decision"] == "supported"
    assert result.summary["quality_subclaim_decisions"][
        "paired_perceptual_quality_noninferiority"
    ]["decision"] == "evidence_incomplete"
    assert result.summary["quality_subclaim_decisions"][
        "semantic_alignment_noninferiority"
    ]["decision"] == "evidence_incomplete"
    assert result.summary["supports_paper_claim"] is False
    metric_by_name = {
        row["quality_metric_name"]: row["quality_metric_value"]
        for row in result.metric_rows
    }
    assert metric_by_name["fid"] == pytest.approx(0.0, abs=1e-12)
    assert metric_by_name["kid_mean"] <= 0.0
    assert metric_by_name["kid_std"] == 0.0


def test_exact9_raw_quality_producer_records_close_all_quality_decisions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """真实格式原始记录必须直接形成逐攻击及跨攻击支持结论."""

    prompt_ids, memberships, features = _quality_evidence()
    source_features = {
        str(row["dataset_quality_record_id"]): list(row["feature_vector"])
        for row in features
        if row["dataset_quality_image_role"] == "source"
    }
    for row in features:
        if row["dataset_quality_image_role"] == "comparison":
            row["feature_vector"] = list(
                source_features[str(row["dataset_quality_record_id"])]
            )

    test_prompt_ids = prompt_ids[-34:]
    attack_configs = tuple(
        config for config in default_attack_configs() if config.enabled
    )
    attack_memberships: list[dict[str, object]] = []
    attack_features: list[dict[str, object]] = []
    paired_metrics: list[dict[str, object]] = []
    clip_features: list[dict[str, object]] = []
    clip_vector = [1.0, 0.0, 0.0]

    def add_clip_pair(
        record_id: str,
        source_digest: str,
        comparison_digest: str,
    ) -> None:
        """为一个图像对补齐可复算的归一化 CLIP 原始向量."""

        for role, image_digest in (
            ("source", source_digest),
            ("comparison", comparison_digest),
        ):
            clip_features.append(
                {
                    "dataset_quality_record_id": record_id,
                    "dataset_quality_image_role": role,
                    "image_digest": image_digest,
                    "feature_backend": FORMAL_CLIP_FEATURE_BACKEND,
                    "feature_dimension": 3,
                    "feature_vector": list(clip_vector),
                    "supports_paper_claim": False,
                }
            )

    def add_metric(
        *,
        repeat_id: str,
        prompt_id: str,
        attack_id: str,
        record_id: str,
    ) -> None:
        """补齐一个 base 或注册攻击配对的 SSIM/CLIP 原始观测."""

        scope = "base" if attack_id == "none" else "registered_attack"
        core = {
            "record_schema": PAIRED_QUALITY_METRIC_RECORD_SCHEMA,
            "dataset_quality_record_id": record_id,
            "randomization_repeat_id": repeat_id,
            "prompt_id": prompt_id,
            "estimand_scope": scope,
            "sample_role": (
                "base_quality_pair"
                if scope == "base"
                else "matched_attack_quality_pair"
            ),
            "attack_id": attack_id,
            "paired_ssim": 1.0,
            "clip_cosine": 1.0,
            "clip_source_feature_digest": build_stable_digest(clip_vector),
            "clip_comparison_feature_digest": build_stable_digest(
                clip_vector
            ),
            "supports_paper_claim": False,
        }
        digest = build_stable_digest(core)
        paired_metrics.append(
            {
                "paired_quality_metric_record_id": (
                    f"paired_quality_metric_{digest[:16]}"
                ),
                "paired_quality_metric_record_digest": digest,
                **core,
            }
        )

    membership_by_key = {
        (
            str(row["randomization_repeat_id"]),
            str(row["prompt_id"]),
        ): row
        for row in memberships
    }
    for repeat_id in formal_randomization_repeat_ids():
        for prompt_id in prompt_ids:
            base = membership_by_key[(repeat_id, prompt_id)]
            base_record_id = str(base["dataset_quality_record_id"])
            add_clip_pair(
                base_record_id,
                str(base["source_image_digest"]),
                str(base["comparison_image_digest"]),
            )
            add_metric(
                repeat_id=repeat_id,
                prompt_id=prompt_id,
                attack_id="none",
                record_id=base_record_id,
            )

        for prompt_index, prompt_id in enumerate(test_prompt_ids):
            for attack_index, attack in enumerate(attack_configs):
                source_digest = build_stable_digest(
                    {
                        "repeat_id": repeat_id,
                        "prompt_id": prompt_id,
                        "attack_id": attack.attack_id,
                        "role": "source",
                    }
                )
                comparison_digest = build_stable_digest(
                    {
                        "repeat_id": repeat_id,
                        "prompt_id": prompt_id,
                        "attack_id": attack.attack_id,
                        "role": "comparison",
                    }
                )
                dataset_core = {
                    "run_id": f"{repeat_id}:{prompt_id}",
                    "prompt_id": prompt_id,
                    "attack_name": attack.attack_id,
                    "image_pair_index": (
                        prompt_index * len(attack_configs) + attack_index
                    ),
                    "image_pair_role": (
                        "matched_attack_clean_to_watermarked"
                    ),
                    "source_image_path": (
                        f"images/{repeat_id}/{prompt_id}/"
                        f"{attack.attack_id}_source.png"
                    ),
                    "source_image_digest": source_digest,
                    "comparison_image_path": (
                        f"images/{repeat_id}/{prompt_id}/"
                        f"{attack.attack_id}_comparison.png"
                    ),
                    "comparison_image_digest": comparison_digest,
                    "feature_backend": FORMAL_FEATURE_BACKEND,
                    "supports_paper_claim": False,
                }
                dataset_digest = build_stable_digest(dataset_core)
                record_id = f"dataset_quality_record_{dataset_digest[:16]}"
                membership = {
                    "dataset_quality_record_id": record_id,
                    "dataset_quality_record_digest": dataset_digest,
                    "randomization_repeat_id": repeat_id,
                    "attack_id": attack.attack_id,
                    "attack_config_digest": attack_config_digest(attack),
                    "attack_seed_random": 3000 + attack_index,
                    **dataset_core,
                }
                attack_memberships.append(membership)
                vector = [
                    prompt_index / 34.0,
                    formal_randomization_repeat_ids().index(repeat_id) / 9.0,
                    (attack_index + 1) / len(attack_configs),
                ]
                for role, image_digest in (
                    ("source", source_digest),
                    ("comparison", comparison_digest),
                ):
                    attack_features.append(
                        {
                            "dataset_quality_record_id": record_id,
                            "dataset_quality_image_role": role,
                            "image_digest": image_digest,
                            "feature_backend": FORMAL_FEATURE_BACKEND,
                            "feature_dimension": 3,
                            "feature_vector": list(vector),
                            "supports_paper_claim": False,
                        }
                    )
                add_clip_pair(record_id, source_digest, comparison_digest)
                add_metric(
                    repeat_id=repeat_id,
                    prompt_id=prompt_id,
                    attack_id=attack.attack_id,
                    record_id=record_id,
                )

    monkeypatch.setattr(analysis, "FORMAL_FEATURE_DIMENSION", 3)
    monkeypatch.setattr(analysis, "FORMAL_CLIP_FEATURE_DIMENSION", 3)
    result = analysis.rebuild_randomization_dataset_quality_statistics(
        features,
        memberships,
        paper_run_name="probe_paper",
        target_fpr=0.1,
        expected_prompt_ids=prompt_ids,
        paired_quality_metric_records=paired_metrics,
        attack_membership_records=attack_memberships,
        attack_feature_records=attack_features,
        clip_feature_records=clip_features,
        expected_attack_prompt_ids=test_prompt_ids,
    )

    assert result.summary["attack_conditioned_quality_statistics_ready"] is True
    assert result.summary["quality_preservation_claim_decision"][
        "decision"
    ] == "supported"
    assert result.summary["cross_attack_quality_decision"]["decision"] == (
        "supported"
    )
    assert all(
        record["decision"] == "supported"
        for record in result.summary[
            "per_attack_quality_decisions"
        ].values()
    )
    assert result.summary["supports_paper_claim"] is True


@pytest.mark.parametrize(
    "mutation_id",
    (
        "missing_repeat",
        "same_count_prompt_duplicate",
        "cross_repeat_record_duplicate",
        "missing_comparison_feature",
        "wrong_feature_image_digest",
        "nonfinite_feature",
    ),
)
def test_exact9_quality_rejects_set_and_feature_counterexamples(
    monkeypatch: pytest.MonkeyPatch,
    mutation_id: str,
) -> None:
    """总行数不变也不能掩盖 repeat, Prompt, 记录或特征角色错误."""

    prompt_ids, memberships, features = _quality_evidence()
    mutated_memberships = deepcopy(memberships)
    mutated_features = deepcopy(features)
    if mutation_id == "missing_repeat":
        removed_repeat = formal_randomization_repeat_ids()[-1]
        removed_record_ids = {
            str(record["dataset_quality_record_id"])
            for record in mutated_memberships
            if record["randomization_repeat_id"] == removed_repeat
        }
        mutated_memberships = [
            record
            for record in mutated_memberships
            if record["randomization_repeat_id"] != removed_repeat
        ]
        mutated_features = [
            record
            for record in mutated_features
            if record["dataset_quality_record_id"] not in removed_record_ids
        ]
    elif mutation_id == "same_count_prompt_duplicate":
        mutated_memberships[1]["prompt_id"] = mutated_memberships[0][
            "prompt_id"
        ]
    elif mutation_id == "cross_repeat_record_duplicate":
        target_index = 70
        mutated_memberships[target_index]["dataset_quality_record_id"] = (
            mutated_memberships[0]["dataset_quality_record_id"]
        )
        mutated_memberships[target_index]["dataset_quality_record_digest"] = (
            mutated_memberships[0]["dataset_quality_record_digest"]
        )
    elif mutation_id == "missing_comparison_feature":
        mutated_features.pop(1)
    elif mutation_id == "wrong_feature_image_digest":
        mutated_features[0]["image_digest"] = build_stable_digest(
            {"wrong": "image"}
        )
    elif mutation_id == "nonfinite_feature":
        mutated_features[0]["feature_vector"][0] = float("nan")
    else:
        raise AssertionError(mutation_id)

    with pytest.raises(analysis.RandomizationDatasetQualityError):
        _rebuild(
            prompt_ids,
            mutated_memberships,
            mutated_features,
            monkeypatch,
        )
