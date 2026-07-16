"""从注册9重复的原始 Inception 特征重建正式 FID/KID.

该模块只计算跨重复数据集质量统计. 每个注册 repeat 必须精确覆盖同一受治理
Prompt 集合, 每个质量记录必须具有唯一的 source/comparison 特征对. FID 和
KID 直接从9份原始特征联合重算, 不读取或平均任何单重复派生指标表.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict
import math
from typing import Any, Iterable, Mapping

import numpy as np

from experiments.protocol.dataset_quality import (
    FORMAL_DATASET_QUALITY_METRIC_NAMES,
    FORMAL_FEATURE_BACKEND,
    formal_dataset_quality_metric_protocol,
    unbiased_polynomial_mmd_exact,
)
from experiments.protocol.formal_randomization import (
    formal_randomization_repeat_ids,
)
from experiments.protocol.attacks import attack_config_digest, default_attack_configs
from experiments.protocol.attack_conditioned_quality import (
    load_attack_conditioned_quality_estimand,
)
from experiments.protocol.independent_semantic_quality import (
    INDEPENDENT_SEMANTIC_FEATURE_BACKEND,
    INDEPENDENT_SEMANTIC_FEATURE_DIMENSION,
    load_independent_semantic_quality_evaluator,
)
from experiments.protocol.paper_run_config import (
    RUN_EXPECTED_PROMPT_COUNTS,
    normalize_paper_run_name,
    validate_frozen_paper_run_target_fpr,
)
from experiments.protocol.splits import build_group_split_counts
from main.core.digest import build_stable_digest
from paper_experiments.analysis.formal_record_statistics import (
    FORMAL_FEATURE_DIMENSION,
    FormalRecordStatisticsError,
    rebuild_formal_fid_kid_metric_rows_from_feature_records,
)
from paper_experiments.analysis.paper_quality_decisions import (
    build_prompt_cluster_mean_inference,
    build_quality_preservation_decisions,
    load_paper_quality_claim_protocol,
)


RANDOMIZATION_DATASET_QUALITY_SUMMARY_SCHEMA = (
    "randomization_dataset_quality_summary"
)
RANDOMIZATION_DATASET_QUALITY_METRIC_PROTOCOL_SCHEMA = (
    "prompt_conditioned_distributional_quality"
)
RANDOMIZATION_DATASET_QUALITY_MEMBERSHIP_FIELDNAMES = (
    "randomization_repeat_id",
    "prompt_id",
    "dataset_quality_record_id",
    "dataset_quality_record_digest",
    "source_image_digest",
    "comparison_image_digest",
)
FORMAL_CLIP_FEATURE_BACKEND = "clip_projected_image_embedding"
FORMAL_CLIP_FEATURE_DIMENSION = 512


class RandomizationDatasetQualityError(ValueError):
    """表示跨重复质量特征不能形成精确且可重建的正式统计."""


@dataclass(frozen=True)
class RandomizationDatasetQualityStatistics:
    """保存规范成员关系、分布统计、FID/KID 和质量决策摘要。"""

    membership_records: tuple[Mapping[str, Any], ...]
    prompt_distribution_records: tuple[Mapping[str, Any], ...]
    attack_prompt_distribution_records: tuple[Mapping[str, Any], ...]
    metric_rows: tuple[Mapping[str, Any], ...]
    summary: Mapping[str, Any]


def randomization_dataset_quality_metric_protocol() -> dict[str, Any]:
    """冻结描述性总体指标与 Prompt 条件 KID 主推断的不同职责。"""

    base_protocol = formal_dataset_quality_metric_protocol()
    repeat_count = len(formal_randomization_repeat_ids())
    sample_pair_count_by_run = {
        run_name: repeat_count * prompt_count
        for run_name, prompt_count in RUN_EXPECTED_PROMPT_COUNTS.items()
    }
    kid_subset_size = int(base_protocol["kid_subset_size"])
    quality_claim_protocol = load_paper_quality_claim_protocol()
    attack_quality_estimand = load_attack_conditioned_quality_estimand()
    payload = {
        "protocol_schema": (
            RANDOMIZATION_DATASET_QUALITY_METRIC_PROTOCOL_SCHEMA
        ),
        "base_formal_metric_protocol_digest": base_protocol[
            "formal_metric_protocol_digest"
        ],
        "registered_repeat_count": repeat_count,
        "feature_population_rule": (
            "joint_raw_feature_rows_across_registered_repeats"
        ),
        "prompt_weighting_rule": (
            "each_prompt_is_one_primary_unit_with_registered_repeats_nested"
        ),
        "primary_sampling_unit": "prompt",
        "nested_sampling_unit": (
            "registered_randomization_repeat_within_prompt"
        ),
        "primary_distributional_preservation_metric": (
            "prompt_conditional_kid_mean_with_prompt_cluster_bootstrap_ci"
        ),
        "joint_fid_kid_evidence_role": "descriptive_distribution_shift",
        "clean_watermarked_interpretation": (
            "distributional_preservation_only_not_real_reference_quality"
        ),
        "paper_quality_claim_protocol_digest": quality_claim_protocol[
            "paper_quality_claim_protocol_digest"
        ],
        "attack_conditioned_quality_estimand_protocol_digest": (
            attack_quality_estimand["quality_estimand_protocol_digest"]
        ),
        "attack_quality_pair_role": "matched_attack_clean_to_watermarked",
        "attack_quality_randomness_rule": (
            "identical_formal_attack_seed_random"
        ),
        "aggregate_sample_pair_count_by_paper_run": (
            sample_pair_count_by_run
        ),
        "randomization_kid_effective_subset_size_by_paper_run": {
            run_name: min(kid_subset_size, pair_count)
            for run_name, pair_count in sample_pair_count_by_run.items()
        },
    }
    payload["randomization_dataset_quality_metric_protocol_digest"] = (
        build_stable_digest(payload)
    )
    return payload


def _prompt_conditional_kid_records(
    feature_records: tuple[Mapping[str, Any], ...],
    membership_records: tuple[Mapping[str, Any], ...],
    *,
    expected_prompt_ids: tuple[str, ...],
) -> tuple[dict[str, Any], ...]:
    """对每个 Prompt 内的9个注册 repeat 独立计算无偏 KID。"""

    feature_by_key = {
        (
            str(record["dataset_quality_record_id"]),
            str(record["dataset_quality_image_role"]),
        ): record
        for record in feature_records
    }
    memberships_by_prompt: dict[str, list[Mapping[str, Any]]] = {
        prompt_id: [] for prompt_id in expected_prompt_ids
    }
    for membership in membership_records:
        memberships_by_prompt[str(membership["prompt_id"])].append(membership)
    repeat_count = len(formal_randomization_repeat_ids())
    rows: list[dict[str, Any]] = []
    for prompt_id in expected_prompt_ids:
        memberships = memberships_by_prompt[prompt_id]
        if len(memberships) != repeat_count:
            raise RandomizationDatasetQualityError(
                "Prompt 条件 KID 未精确覆盖全部注册 repeat"
            )
        source_features = np.asarray(
            [
                feature_by_key[
                    (str(record["dataset_quality_record_id"]), "source")
                ]["feature_vector"]
                for record in memberships
            ],
            dtype=np.float64,
        )
        comparison_features = np.asarray(
            [
                feature_by_key[
                    (str(record["dataset_quality_record_id"]), "comparison")
                ]["feature_vector"]
                for record in memberships
            ],
            dtype=np.float64,
        )
        try:
            value = unbiased_polynomial_mmd_exact(
                source_features,
                comparison_features,
            )
        except (TypeError, ValueError) as exc:
            raise RandomizationDatasetQualityError(
                "Prompt 条件 KID 特征不是同维有限数值"
            ) from exc
        core = {
            "prompt_id": prompt_id,
            "randomization_repeat_count": repeat_count,
            "primary_sampling_unit": "prompt",
            "nested_sampling_unit": (
                "registered_randomization_repeat_within_prompt"
            ),
            "quality_metric_name": "prompt_conditional_kid",
            "quality_metric_value": value,
            "metric_status": "measured",
            "supports_paper_claim": False,
        }
        rows.append(
            {
                **core,
                "prompt_distribution_record_digest": build_stable_digest(core),
            }
        )
    return tuple(rows)


def _is_sha256(value: Any) -> bool:
    """判断值是否为规范小写 SHA-256."""

    text = str(value)
    return len(text) == 64 and all(
        character in "0123456789abcdef" for character in text
    )


def _canonical_membership_records(
    membership_records: Iterable[Mapping[str, Any]],
    *,
    expected_prompt_ids: tuple[str, ...],
) -> tuple[dict[str, Any], ...]:
    """验证9重复与 Prompt 的笛卡尔积并返回规范顺序."""

    repeat_ids = formal_randomization_repeat_ids()
    repeat_order = {
        repeat_id: index for index, repeat_id in enumerate(repeat_ids)
    }
    prompt_order = {
        prompt_id: index for index, prompt_id in enumerate(expected_prompt_ids)
    }
    expected_keys = {
        (repeat_id, prompt_id)
        for repeat_id in repeat_ids
        for prompt_id in expected_prompt_ids
    }
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    record_ids: set[str] = set()
    for row_index, raw_record in enumerate(membership_records):
        record = dict(raw_record)
        if set(record) != set(
            RANDOMIZATION_DATASET_QUALITY_MEMBERSHIP_FIELDNAMES
        ):
            raise RandomizationDatasetQualityError(
                f"质量特征成员字段集合无效: row={row_index}"
            )
        repeat_id = str(record["randomization_repeat_id"])
        prompt_id = str(record["prompt_id"])
        record_id = str(record["dataset_quality_record_id"])
        record_digest = str(record["dataset_quality_record_digest"])
        source_image_digest = str(record["source_image_digest"])
        comparison_image_digest = str(record["comparison_image_digest"])
        key = (repeat_id, prompt_id)
        if (
            key not in expected_keys
            or key in by_key
            or not record_id
            or record_id in record_ids
            or not _is_sha256(record_digest)
            or record_id != f"dataset_quality_record_{record_digest[:16]}"
            or not _is_sha256(source_image_digest)
            or not _is_sha256(comparison_image_digest)
        ):
            raise RandomizationDatasetQualityError(
                f"质量特征成员身份重复, 缺失或无效: row={row_index}"
            )
        by_key[key] = {
            "randomization_repeat_id": repeat_id,
            "prompt_id": prompt_id,
            "dataset_quality_record_id": record_id,
            "dataset_quality_record_digest": record_digest,
            "source_image_digest": source_image_digest,
            "comparison_image_digest": comparison_image_digest,
        }
        record_ids.add(record_id)
    if set(by_key) != expected_keys:
        raise RandomizationDatasetQualityError(
            "质量特征成员未精确覆盖9重复与完整 Prompt 集合"
        )
    return tuple(
        by_key[key]
        for key in sorted(
            by_key,
            key=lambda value: (
                repeat_order[value[0]],
                prompt_order[value[1]],
            ),
        )
    )


def _canonical_feature_records(
    feature_records: Iterable[Mapping[str, Any]],
    *,
    membership_records: tuple[Mapping[str, Any], ...],
) -> tuple[dict[str, Any], ...]:
    """把原始特征绑定到成员图像摘要并返回规范角色顺序."""

    membership_by_record_id = {
        str(record["dataset_quality_record_id"]): record
        for record in membership_records
    }
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for row_index, raw_record in enumerate(feature_records):
        record = dict(raw_record)
        record_id = str(record.get("dataset_quality_record_id", ""))
        role = str(record.get("dataset_quality_image_role", ""))
        membership = membership_by_record_id.get(record_id)
        key = (record_id, role)
        expected_image_digest = (
            str(membership[f"{role}_image_digest"])
            if membership is not None and role in {"source", "comparison"}
            else ""
        )
        declared_repeat_id = record.get("randomization_repeat_id")
        if (
            membership is None
            or role not in {"source", "comparison"}
            or key in by_key
            or str(record.get("image_digest", ""))
            != expected_image_digest
            or str(record.get("feature_backend", ""))
            != FORMAL_FEATURE_BACKEND
            or type(record.get("feature_dimension")) is not int
            or int(record["feature_dimension"]) != FORMAL_FEATURE_DIMENSION
            or (
                declared_repeat_id not in (None, "")
                and str(declared_repeat_id)
                != str(membership["randomization_repeat_id"])
            )
        ):
            raise RandomizationDatasetQualityError(
                f"质量特征角色, 图像身份或正式维度无效: row={row_index}"
            )
        by_key[key] = record
    expected_keys = {
        (record_id, role)
        for record_id in membership_by_record_id
        for role in ("source", "comparison")
    }
    if set(by_key) != expected_keys:
        raise RandomizationDatasetQualityError(
            "质量 feature records 未精确覆盖全部成员与两个图像角色"
        )
    return tuple(
        by_key[(str(membership["dataset_quality_record_id"]), role)]
        for membership in membership_records
        for role in ("source", "comparison")
    )


def _registered_attack_digests() -> dict[str, str]:
    """返回质量结论必须覆盖的启用攻击及冻结配置摘要。"""

    return {
        config.attack_id: attack_config_digest(config)
        for config in default_attack_configs()
        if config.enabled
    }


def _canonical_attack_membership_records(
    records: Iterable[Mapping[str, Any]],
    *,
    expected_attack_prompt_ids: tuple[str, ...],
) -> tuple[dict[str, Any], ...]:
    """验证9重复、test Prompt 和注册攻击的完整笛卡尔积。"""

    repeat_ids = formal_randomization_repeat_ids()
    attack_digests = _registered_attack_digests()
    expected_keys = {
        (repeat_id, prompt_id, attack_id)
        for repeat_id in repeat_ids
        for prompt_id in expected_attack_prompt_ids
        for attack_id in attack_digests
    }
    by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    record_ids: set[str] = set()
    for row_index, raw_record in enumerate(records):
        record = dict(raw_record)
        repeat_id = str(record.get("randomization_repeat_id", ""))
        prompt_id = str(record.get("prompt_id", ""))
        attack_id = str(record.get("attack_id", ""))
        key = (repeat_id, prompt_id, attack_id)
        record_id = str(record.get("dataset_quality_record_id", ""))
        record_digest = str(record.get("dataset_quality_record_digest", ""))
        source_digest = str(record.get("source_image_digest", ""))
        comparison_digest = str(record.get("comparison_image_digest", ""))
        dataset_core = {
            field_name: record.get(field_name)
            for field_name in (
                "run_id",
                "prompt_id",
                "attack_name",
                "image_pair_index",
                "image_pair_role",
                "source_image_path",
                "source_image_digest",
                "comparison_image_path",
                "comparison_image_digest",
                "feature_backend",
                "supports_paper_claim",
            )
        }
        if (
            key not in expected_keys
            or key in by_key
            or record_id in record_ids
            or not _is_sha256(record_digest)
            or record_id != f"dataset_quality_record_{record_digest[:16]}"
            or build_stable_digest(dataset_core) != record_digest
            or not _is_sha256(source_digest)
            or not _is_sha256(comparison_digest)
            or record.get("image_pair_role")
            != "matched_attack_clean_to_watermarked"
            or record.get("attack_config_digest")
            != attack_digests.get(attack_id)
            or type(record.get("attack_seed_random")) is not int
            or record.get("supports_paper_claim") is not False
        ):
            raise RandomizationDatasetQualityError(
                f"逐攻击质量成员身份或冻结攻击绑定无效: row={row_index}"
            )
        by_key[key] = record
        record_ids.add(record_id)
    if set(by_key) != expected_keys:
        raise RandomizationDatasetQualityError(
            "逐攻击质量成员未精确覆盖9重复、test Prompt 与攻击集合"
        )
    repeat_order = {value: index for index, value in enumerate(repeat_ids)}
    prompt_order = {
        value: index for index, value in enumerate(expected_attack_prompt_ids)
    }
    attack_order = {
        value: index for index, value in enumerate(sorted(attack_digests))
    }
    return tuple(
        by_key[key]
        for key in sorted(
            by_key,
            key=lambda value: (
                repeat_order[value[0]],
                prompt_order[value[1]],
                attack_order[value[2]],
            ),
        )
    )


def _canonical_attack_feature_records(
    records: Iterable[Mapping[str, Any]],
    *,
    membership_records: tuple[Mapping[str, Any], ...],
) -> tuple[dict[str, Any], ...]:
    """把逐攻击 Inception 特征绑定到匹配攻击后的两类图像摘要。"""

    membership_by_id = {
        str(record["dataset_quality_record_id"]): record
        for record in membership_records
    }
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for row_index, raw_record in enumerate(records):
        record = dict(raw_record)
        record_id = str(record.get("dataset_quality_record_id", ""))
        role = str(record.get("dataset_quality_image_role", ""))
        membership = membership_by_id.get(record_id)
        key = (record_id, role)
        expected_digest = (
            str(membership[f"{role}_image_digest"])
            if membership is not None and role in {"source", "comparison"}
            else ""
        )
        if (
            membership is None
            or key in by_key
            or role not in {"source", "comparison"}
            or record.get("image_digest") != expected_digest
            or record.get("feature_backend") != FORMAL_FEATURE_BACKEND
            or record.get("feature_dimension") != FORMAL_FEATURE_DIMENSION
            or not isinstance(record.get("feature_vector"), list)
            or len(record["feature_vector"]) != FORMAL_FEATURE_DIMENSION
        ):
            raise RandomizationDatasetQualityError(
                f"逐攻击 Inception 特征身份或维度无效: row={row_index}"
            )
        vector = np.asarray(record["feature_vector"], dtype=np.float64)
        if vector.ndim != 1 or not np.isfinite(vector).all():
            raise RandomizationDatasetQualityError("逐攻击 Inception 特征包含非有限值")
        by_key[key] = record
    expected_keys = {
        (record_id, role)
        for record_id in membership_by_id
        for role in ("source", "comparison")
    }
    if set(by_key) != expected_keys:
        raise RandomizationDatasetQualityError(
            "逐攻击 Inception 特征未精确覆盖全部成员和两个图像角色"
        )
    return tuple(
        by_key[(str(membership["dataset_quality_record_id"]), role)]
        for membership in membership_records
        for role in ("source", "comparison")
    )


def _canonical_paired_quality_metric_records(
    records: Iterable[Mapping[str, Any]],
    *,
    expected_prompt_ids: tuple[str, ...],
    expected_attack_prompt_ids: tuple[str, ...],
) -> tuple[dict[str, Any], ...]:
    """验证 base 与逐攻击 SSIM、诊断 CLIP 及独立语义指标集合。"""

    repeat_ids = formal_randomization_repeat_ids()
    attack_ids = tuple(sorted(_registered_attack_digests()))
    expected_keys = {
        (repeat_id, prompt_id, "none")
        for repeat_id in repeat_ids
        for prompt_id in expected_prompt_ids
    } | {
        (repeat_id, prompt_id, attack_id)
        for repeat_id in repeat_ids
        for prompt_id in expected_attack_prompt_ids
        for attack_id in attack_ids
    }
    by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row_index, raw_record in enumerate(records):
        record = dict(raw_record)
        key = (
            str(record.get("randomization_repeat_id", "")),
            str(record.get("prompt_id", "")),
            str(record.get("attack_id", "")),
        )
        digest = str(record.get("paired_quality_metric_record_digest", ""))
        digest_payload = {
            field_name: value
            for field_name, value in record.items()
            if field_name
            not in {
                "paired_quality_metric_record_id",
                "paired_quality_metric_record_digest",
            }
        }
        ssim = record.get("paired_ssim")
        clip_cosine = record.get("clip_cosine")
        independent_semantic_cosine = record.get(
            "independent_semantic_cosine"
        )
        if (
            key not in expected_keys
            or key in by_key
            or not _is_sha256(digest)
            or record.get("paired_quality_metric_record_id")
            != f"paired_quality_metric_{digest[:16]}"
            or build_stable_digest(digest_payload) != digest
            or not isinstance(ssim, (int, float))
            or not math.isfinite(float(ssim))
            or not -1.0 <= float(ssim) <= 1.0
            or not isinstance(clip_cosine, (int, float))
            or not math.isfinite(float(clip_cosine))
            or not -1.0 <= float(clip_cosine) <= 1.0
            or record.get("clip_evidence_role")
            != "mechanism_consistency_diagnostic"
            or not isinstance(independent_semantic_cosine, (int, float))
            or not math.isfinite(float(independent_semantic_cosine))
            or not -1.0 <= float(independent_semantic_cosine) <= 1.0
            or record.get("independent_semantic_evidence_role")
            != "independent_semantic_preservation_primary"
            or record.get("supports_paper_claim") is not False
        ):
            raise RandomizationDatasetQualityError(
                f"配对 SSIM/CLIP 原始记录身份或数值无效: row={row_index}"
            )
        expected_scope = "base" if key[2] == "none" else "registered_attack"
        if record.get("estimand_scope") != expected_scope:
            raise RandomizationDatasetQualityError("配对指标 estimand scope 与攻击身份不一致")
        by_key[key] = record
    if set(by_key) != expected_keys:
        raise RandomizationDatasetQualityError(
            "配对 SSIM/CLIP 未精确覆盖9重复、Prompt 与逐攻击集合"
        )
    return tuple(by_key[key] for key in sorted(by_key))


def _canonical_clip_feature_records(
    records: Iterable[Mapping[str, Any]],
    *,
    base_membership_records: tuple[Mapping[str, Any], ...],
    attack_membership_records: tuple[Mapping[str, Any], ...],
    paired_metric_records: tuple[Mapping[str, Any], ...],
) -> tuple[dict[str, Any], ...]:
    """复核 CLIP 向量与 metric 记录中的 cosine 和特征摘要。"""

    memberships = (*base_membership_records, *attack_membership_records)
    membership_by_id = {
        str(record["dataset_quality_record_id"]): record
        for record in memberships
    }
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for row_index, raw_record in enumerate(records):
        record = dict(raw_record)
        record_id = str(record.get("dataset_quality_record_id", ""))
        role = str(record.get("dataset_quality_image_role", ""))
        membership = membership_by_id.get(record_id)
        key = (record_id, role)
        vector = record.get("feature_vector")
        expected_digest = (
            str(membership[f"{role}_image_digest"])
            if membership is not None and role in {"source", "comparison"}
            else ""
        )
        if (
            membership is None
            or key in by_key
            or role not in {"source", "comparison"}
            or record.get("image_digest") != expected_digest
            or record.get("feature_backend") != FORMAL_CLIP_FEATURE_BACKEND
            or record.get("feature_dimension") != FORMAL_CLIP_FEATURE_DIMENSION
            or not isinstance(vector, list)
            or len(vector) != FORMAL_CLIP_FEATURE_DIMENSION
        ):
            raise RandomizationDatasetQualityError(
                f"CLIP 特征身份、角色或维度无效: row={row_index}"
            )
        array = np.asarray(vector, dtype=np.float64)
        if (
            array.ndim != 1
            or not np.isfinite(array).all()
            or not math.isclose(
                float(np.linalg.norm(array)),
                1.0,
                rel_tol=1e-5,
                abs_tol=1e-5,
            )
        ):
            raise RandomizationDatasetQualityError("CLIP 特征必须是有限 L2 归一化向量")
        by_key[key] = record
    expected_keys = {
        (record_id, role)
        for record_id in membership_by_id
        for role in ("source", "comparison")
    }
    if set(by_key) != expected_keys:
        raise RandomizationDatasetQualityError("CLIP 特征未精确覆盖全部 base 和攻击图像对")
    for metric in paired_metric_records:
        record_id = str(metric["dataset_quality_record_id"])
        source_vector = by_key[(record_id, "source")]["feature_vector"]
        comparison_vector = by_key[(record_id, "comparison")]["feature_vector"]
        cosine = float(
            np.dot(
                np.asarray(source_vector, dtype=np.float64),
                np.asarray(comparison_vector, dtype=np.float64),
            )
        )
        if (
            not math.isclose(
                cosine,
                float(metric["clip_cosine"]),
                rel_tol=1e-7,
                abs_tol=1e-7,
            )
            or metric.get("clip_source_feature_digest")
            != build_stable_digest(source_vector)
            or metric.get("clip_comparison_feature_digest")
            != build_stable_digest(comparison_vector)
        ):
            raise RandomizationDatasetQualityError("CLIP cosine 与原始向量复算结果不一致")
    return tuple(
        by_key[(str(membership["dataset_quality_record_id"]), role)]
        for membership in memberships
        for role in ("source", "comparison")
    )


def _canonical_independent_semantic_feature_records(
    records: Iterable[Mapping[str, Any]],
    *,
    base_membership_records: tuple[Mapping[str, Any], ...],
    attack_membership_records: tuple[Mapping[str, Any], ...],
    paired_metric_records: tuple[Mapping[str, Any], ...],
) -> tuple[dict[str, Any], ...]:
    """复算独立 DINOv2 cosine, 并拒绝未绑定冻结协议的特征记录."""

    protocol = load_independent_semantic_quality_evaluator()
    protocol_digest = protocol["independent_semantic_quality_protocol_digest"]
    memberships = (*base_membership_records, *attack_membership_records)
    membership_by_id = {
        str(record["dataset_quality_record_id"]): record
        for record in memberships
    }
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for row_index, raw_record in enumerate(records):
        record = dict(raw_record)
        record_id = str(record.get("dataset_quality_record_id", ""))
        role = str(record.get("dataset_quality_image_role", ""))
        membership = membership_by_id.get(record_id)
        key = (record_id, role)
        vector = record.get("feature_vector")
        expected_digest = (
            str(membership[f"{role}_image_digest"])
            if membership is not None and role in {"source", "comparison"}
            else ""
        )
        if (
            membership is None
            or key in by_key
            or role not in {"source", "comparison"}
            or record.get("image_digest") != expected_digest
            or record.get("feature_backend")
            != INDEPENDENT_SEMANTIC_FEATURE_BACKEND
            or record.get("feature_extractor_id")
            != (
                f"{protocol['model_contract']['model_id']}@"
                f"{protocol['model_contract']['model_revision']}"
            )
            or record.get("feature_dimension")
            != INDEPENDENT_SEMANTIC_FEATURE_DIMENSION
            or record.get("feature_layer") != "last_hidden_state_cls_token"
            or record.get("feature_normalization") != "l2"
            or record.get("independent_semantic_quality_protocol_digest")
            != protocol_digest
            or not isinstance(vector, list)
            or len(vector) != INDEPENDENT_SEMANTIC_FEATURE_DIMENSION
            or record.get("feature_vector_digest")
            != build_stable_digest(vector)
        ):
            raise RandomizationDatasetQualityError(
                f"独立语义特征身份、角色或协议无效: row={row_index}"
            )
        array = np.asarray(vector, dtype=np.float64)
        if (
            array.ndim != 1
            or not np.isfinite(array).all()
            or not math.isclose(
                float(np.linalg.norm(array)),
                1.0,
                rel_tol=1e-5,
                abs_tol=1e-5,
            )
        ):
            raise RandomizationDatasetQualityError(
                "独立语义特征必须是有限 L2 归一化向量"
            )
        by_key[key] = record
    expected_keys = {
        (record_id, role)
        for record_id in membership_by_id
        for role in ("source", "comparison")
    }
    if set(by_key) != expected_keys:
        raise RandomizationDatasetQualityError(
            "独立语义特征未精确覆盖全部 base 和攻击图像对"
        )
    for metric in paired_metric_records:
        record_id = str(metric["dataset_quality_record_id"])
        source_vector = by_key[(record_id, "source")]["feature_vector"]
        comparison_vector = by_key[(record_id, "comparison")]["feature_vector"]
        cosine = float(
            np.dot(
                np.asarray(source_vector, dtype=np.float64),
                np.asarray(comparison_vector, dtype=np.float64),
            )
        )
        if not all(
            (
                math.isclose(
                    cosine,
                    float(metric["independent_semantic_cosine"]),
                    rel_tol=1e-7,
                    abs_tol=1e-7,
                ),
                metric.get("independent_semantic_source_feature_digest")
                == build_stable_digest(source_vector),
                metric.get("independent_semantic_comparison_feature_digest")
                == build_stable_digest(comparison_vector),
                metric.get("independent_semantic_quality_protocol_digest")
                == protocol_digest,
            )
        ):
            raise RandomizationDatasetQualityError(
                "独立语义 cosine 与原始 DINOv2 向量复算结果不一致"
            )
    return tuple(
        by_key[(str(membership["dataset_quality_record_id"]), role)]
        for membership in memberships
        for role in ("source", "comparison")
    )


def _prompt_mean_values(
    records: Iterable[Mapping[str, Any]],
    *,
    value_field: str,
    attack_id: str,
    expected_prompt_ids: tuple[str, ...],
) -> dict[str, float]:
    """先在 Prompt 内平均9个 repeat, 再交给 Prompt 聚类 bootstrap。"""

    values: dict[str, list[float]] = defaultdict(list)
    for record in records:
        if str(record["attack_id"]) == attack_id:
            values[str(record["prompt_id"])].append(float(record[value_field]))
    repeat_count = len(formal_randomization_repeat_ids())
    if set(values) != set(expected_prompt_ids) or any(
        len(prompt_values) != repeat_count for prompt_values in values.values()
    ):
        raise RandomizationDatasetQualityError("Prompt 内配对质量值未精确覆盖9个 repeat")
    return {
        prompt_id: float(np.mean(values[prompt_id], dtype=np.float64))
        for prompt_id in expected_prompt_ids
    }


def _attack_prompt_conditional_kid_records(
    feature_records: tuple[Mapping[str, Any], ...],
    membership_records: tuple[Mapping[str, Any], ...],
    *,
    expected_attack_prompt_ids: tuple[str, ...],
) -> tuple[dict[str, Any], ...]:
    """对每个注册攻击和 Prompt 的9重复匹配图像计算无偏 KID。"""

    feature_by_key = {
        (
            str(record["dataset_quality_record_id"]),
            str(record["dataset_quality_image_role"]),
        ): record
        for record in feature_records
    }
    memberships: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for record in membership_records:
        memberships[(str(record["attack_id"]), str(record["prompt_id"]))].append(record)
    repeat_count = len(formal_randomization_repeat_ids())
    rows: list[dict[str, Any]] = []
    for attack_id in sorted(_registered_attack_digests()):
        for prompt_id in expected_attack_prompt_ids:
            group = memberships[(attack_id, prompt_id)]
            if len(group) != repeat_count:
                raise RandomizationDatasetQualityError("逐攻击 Prompt KID 缺少注册 repeat")
            source = np.asarray(
                [
                    feature_by_key[
                        (str(record["dataset_quality_record_id"]), "source")
                    ]["feature_vector"]
                    for record in group
                ],
                dtype=np.float64,
            )
            comparison = np.asarray(
                [
                    feature_by_key[
                        (str(record["dataset_quality_record_id"]), "comparison")
                    ]["feature_vector"]
                    for record in group
                ],
                dtype=np.float64,
            )
            value = unbiased_polynomial_mmd_exact(source, comparison)
            core = {
                "attack_id": attack_id,
                "prompt_id": prompt_id,
                "randomization_repeat_count": repeat_count,
                "quality_metric_name": "attack_prompt_conditional_kid",
                "quality_metric_value": value,
                "primary_sampling_unit": "prompt",
                "nested_sampling_unit": (
                    "registered_randomization_repeat_within_prompt"
                ),
                "metric_status": "measured",
                "supports_paper_claim": False,
            }
            rows.append(
                {
                    **core,
                    "attack_prompt_distribution_record_digest": (
                        build_stable_digest(core)
                    ),
                }
            )
    return tuple(rows)


def rebuild_randomization_dataset_quality_statistics(
    feature_records: Iterable[Mapping[str, Any]],
    membership_records: Iterable[Mapping[str, Any]],
    *,
    paper_run_name: str,
    target_fpr: float,
    expected_prompt_ids: Iterable[str],
    paired_quality_metric_records: Iterable[Mapping[str, Any]] = (),
    attack_membership_records: Iterable[Mapping[str, Any]] = (),
    attack_feature_records: Iterable[Mapping[str, Any]] = (),
    clip_feature_records: Iterable[Mapping[str, Any]] = (),
    independent_semantic_feature_records: Iterable[Mapping[str, Any]] = (),
    expected_attack_prompt_ids: Iterable[str] = (),
) -> RandomizationDatasetQualityStatistics:
    """从9重复原始特征联合重建一次正式 FID/KID 结果."""

    run_name = normalize_paper_run_name(paper_run_name)
    resolved_target_fpr = validate_frozen_paper_run_target_fpr(
        run_name,
        target_fpr,
    )
    prompt_ids = tuple(str(prompt_id) for prompt_id in expected_prompt_ids)
    expected_prompt_count = RUN_EXPECTED_PROMPT_COUNTS[run_name]
    if (
        len(prompt_ids) != expected_prompt_count
        or len(set(prompt_ids)) != expected_prompt_count
        or any(not prompt_id for prompt_id in prompt_ids)
    ):
        raise RandomizationDatasetQualityError(
            "受治理 Prompt 身份未匹配论文运行层级的精确数量"
        )
    canonical_membership = _canonical_membership_records(
        membership_records,
        expected_prompt_ids=prompt_ids,
    )
    canonical_features = _canonical_feature_records(
        feature_records,
        membership_records=canonical_membership,
    )
    prompt_distribution_records = _prompt_conditional_kid_records(
        canonical_features,
        canonical_membership,
        expected_prompt_ids=prompt_ids,
    )
    distributional_inference = build_prompt_cluster_mean_inference(
        {
            str(record["prompt_id"]): float(record["quality_metric_value"])
            for record in prompt_distribution_records
        },
        analysis_id="distributional_preservation_prompt_conditional_kid",
    )
    raw_paired_metrics = tuple(paired_quality_metric_records)
    raw_attack_membership = tuple(attack_membership_records)
    raw_attack_features = tuple(attack_feature_records)
    raw_clip_features = tuple(clip_feature_records)
    raw_independent_semantic_features = tuple(
        independent_semantic_feature_records
    )
    attack_prompt_ids = tuple(
        str(prompt_id) for prompt_id in expected_attack_prompt_ids
    )
    complete_attack_inputs = all(
        (
            bool(raw_paired_metrics),
            bool(raw_attack_membership),
            bool(raw_attack_features),
            bool(raw_clip_features),
            bool(raw_independent_semantic_features),
            bool(attack_prompt_ids),
        )
    )
    any_attack_inputs = any(
        (
            bool(raw_paired_metrics),
            bool(raw_attack_membership),
            bool(raw_attack_features),
            bool(raw_clip_features),
            bool(raw_independent_semantic_features),
            bool(attack_prompt_ids),
        )
    )
    if any_attack_inputs and not complete_attack_inputs:
        raise RandomizationDatasetQualityError(
            "逐攻击质量证据必须同时提供指标、成员、特征和 test Prompt 身份"
        )
    canonical_paired_metrics: tuple[dict[str, Any], ...] = ()
    canonical_attack_membership: tuple[dict[str, Any], ...] = ()
    canonical_attack_features: tuple[dict[str, Any], ...] = ()
    canonical_clip_features: tuple[dict[str, Any], ...] = ()
    canonical_independent_semantic_features: tuple[dict[str, Any], ...] = ()
    attack_prompt_distribution_records: tuple[dict[str, Any], ...] = ()
    paired_perceptual_inference = None
    independent_visual_content_inference = None
    mechanism_consistency_clip_inference = None
    per_attack_inference = None
    if complete_attack_inputs:
        expected_attack_prompt_count = int(
            build_group_split_counts(expected_prompt_count)["test"]
        )
        if (
            len(attack_prompt_ids) != expected_attack_prompt_count
            or len(set(attack_prompt_ids)) != expected_attack_prompt_count
            or not set(attack_prompt_ids).issubset(prompt_ids)
        ):
            raise RandomizationDatasetQualityError(
                "逐攻击质量 test Prompt 身份未匹配论文运行层级"
            )
        canonical_paired_metrics = _canonical_paired_quality_metric_records(
            raw_paired_metrics,
            expected_prompt_ids=prompt_ids,
            expected_attack_prompt_ids=attack_prompt_ids,
        )
        canonical_attack_membership = _canonical_attack_membership_records(
            raw_attack_membership,
            expected_attack_prompt_ids=attack_prompt_ids,
        )
        canonical_attack_features = _canonical_attack_feature_records(
            raw_attack_features,
            membership_records=canonical_attack_membership,
        )
        canonical_clip_features = _canonical_clip_feature_records(
            raw_clip_features,
            base_membership_records=canonical_membership,
            attack_membership_records=canonical_attack_membership,
            paired_metric_records=canonical_paired_metrics,
        )
        canonical_independent_semantic_features = (
            _canonical_independent_semantic_feature_records(
                raw_independent_semantic_features,
                base_membership_records=canonical_membership,
                attack_membership_records=canonical_attack_membership,
                paired_metric_records=canonical_paired_metrics,
            )
        )
        attack_prompt_distribution_records = (
            _attack_prompt_conditional_kid_records(
                canonical_attack_features,
                canonical_attack_membership,
                expected_attack_prompt_ids=attack_prompt_ids,
            )
        )
        paired_perceptual_inference = build_prompt_cluster_mean_inference(
            _prompt_mean_values(
                canonical_paired_metrics,
                value_field="paired_ssim",
                attack_id="none",
                expected_prompt_ids=prompt_ids,
            ),
            analysis_id="paired_perceptual_quality_base_ssim",
        )
        independent_visual_content_inference = build_prompt_cluster_mean_inference(
            _prompt_mean_values(
                canonical_paired_metrics,
                value_field="independent_semantic_cosine",
                attack_id="none",
                expected_prompt_ids=prompt_ids,
            ),
            analysis_id="independent_visual_content_base_cosine",
        )
        mechanism_consistency_clip_inference = build_prompt_cluster_mean_inference(
            _prompt_mean_values(
                canonical_paired_metrics,
                value_field="clip_cosine",
                attack_id="none",
                expected_prompt_ids=prompt_ids,
            ),
            analysis_id="mechanism_consistency_base_clip_cosine",
        )
        per_attack_inference = {}
        for attack_id in sorted(_registered_attack_digests()):
            kid_values = {
                str(record["prompt_id"]): float(
                    record["quality_metric_value"]
                )
                for record in attack_prompt_distribution_records
                if record["attack_id"] == attack_id
            }
            per_attack_inference[attack_id] = {
                "paired_perceptual_quality_noninferiority": (
                    build_prompt_cluster_mean_inference(
                        _prompt_mean_values(
                            canonical_paired_metrics,
                            value_field="paired_ssim",
                            attack_id=attack_id,
                            expected_prompt_ids=attack_prompt_ids,
                        ),
                        analysis_id=f"attack_paired_ssim:{attack_id}",
                    )
                ),
                "independent_visual_content_preservation_noninferiority": (
                    build_prompt_cluster_mean_inference(
                        _prompt_mean_values(
                            canonical_paired_metrics,
                            value_field="independent_semantic_cosine",
                            attack_id=attack_id,
                            expected_prompt_ids=attack_prompt_ids,
                        ),
                        analysis_id=(
                            "attack_independent_semantic_cosine:"
                            f"{attack_id}"
                        ),
                    )
                ),
                "distributional_preservation_noninferiority": (
                    build_prompt_cluster_mean_inference(
                        kid_values,
                        analysis_id=f"attack_prompt_conditional_kid:{attack_id}",
                    )
                ),
            }
    quality_decisions = build_quality_preservation_decisions(
        distributional_inference=distributional_inference,
        paired_perceptual_inference=paired_perceptual_inference,
        independent_visual_content_inference=independent_visual_content_inference,
        per_attack_inference=per_attack_inference,
        evidence_artifact_id="randomization_dataset_quality_manifest",
    )
    quality_preservation_decision = quality_decisions[
        "quality_preservation_claim_decision"
    ]
    aggregate_pair_count = (
        len(formal_randomization_repeat_ids()) * expected_prompt_count
    )
    try:
        metric_rows = (
            rebuild_formal_fid_kid_metric_rows_from_feature_records(
                canonical_features,
                expected_pair_count=aggregate_pair_count,
            )
        )
    except (FormalRecordStatisticsError, TypeError, ValueError) as exc:
        raise RandomizationDatasetQualityError(
            "9重复原始特征不能完成正式 FID/KID 数值重建"
        ) from exc
    if (
        tuple(row["quality_metric_name"] for row in metric_rows)
        != tuple(FORMAL_DATASET_QUALITY_METRIC_NAMES)
        or any(
            row["metric_status"] != "measured"
            or row["feature_backend"] != FORMAL_FEATURE_BACKEND
            or row["source_image_count"] != aggregate_pair_count
            or row["comparison_image_count"] != aggregate_pair_count
            or row["sample_pair_count"] != aggregate_pair_count
            or row["supports_paper_claim"] is not False
            for row in metric_rows
        )
    ):
        raise RandomizationDatasetQualityError(
            "9重复 FID/KID 三行指标未形成完整 measured 结果"
        )

    metric_protocol = randomization_dataset_quality_metric_protocol()
    summary: dict[str, Any] = {
        "summary_schema": RANDOMIZATION_DATASET_QUALITY_SUMMARY_SCHEMA,
        "paper_claim_scale": run_name,
        "target_fpr": resolved_target_fpr,
        "randomization_repeat_ids": list(
            formal_randomization_repeat_ids()
        ),
        "randomization_repeat_count": len(
            formal_randomization_repeat_ids()
        ),
        "prompt_count_per_repeat": expected_prompt_count,
        "aggregate_quality_pair_count": aggregate_pair_count,
        "aggregate_feature_record_count": len(canonical_features),
        "prompt_id_set_digest": build_stable_digest(sorted(prompt_ids)),
        "quality_feature_membership_digest": build_stable_digest(
            canonical_membership
        ),
        "quality_feature_records_digest": build_stable_digest(
            canonical_features
        ),
        "prompt_distribution_records_digest": build_stable_digest(
            prompt_distribution_records
        ),
        "paired_quality_metric_records_digest": build_stable_digest(
            canonical_paired_metrics
        ),
        "attack_quality_membership_records_digest": build_stable_digest(
            canonical_attack_membership
        ),
        "attack_quality_feature_records_digest": build_stable_digest(
            canonical_attack_features
        ),
        "paired_quality_clip_feature_records_digest": build_stable_digest(
            canonical_clip_features
        ),
        "paired_quality_independent_semantic_feature_records_digest": (
            build_stable_digest(canonical_independent_semantic_features)
        ),
        "attack_prompt_distribution_records_digest": build_stable_digest(
            attack_prompt_distribution_records
        ),
        "paired_quality_metric_record_count": len(canonical_paired_metrics),
        "attack_quality_membership_record_count": len(
            canonical_attack_membership
        ),
        "attack_quality_feature_record_count": len(
            canonical_attack_features
        ),
        "paired_quality_clip_feature_record_count": len(
            canonical_clip_features
        ),
        "paired_quality_independent_semantic_feature_record_count": len(
            canonical_independent_semantic_features
        ),
        "attack_prompt_distribution_record_count": len(
            attack_prompt_distribution_records
        ),
        "attack_conditioned_quality_statistics_ready": complete_attack_inputs,
        "distributional_preservation_inference": distributional_inference,
        "paired_perceptual_quality_inference": paired_perceptual_inference,
        "independent_visual_content_inference": independent_visual_content_inference,
        "mechanism_consistency_clip_inference": (
            mechanism_consistency_clip_inference
        ),
        "per_attack_quality_inference": per_attack_inference,
        "randomization_dataset_quality_metric_protocol_digest": (
            metric_protocol[
                "randomization_dataset_quality_metric_protocol_digest"
            ]
        ),
        "fid_kid_metric_rows_digest": build_stable_digest(metric_rows),
        "quality_metric_names": list(FORMAL_DATASET_QUALITY_METRIC_NAMES),
        "quality_metric_status": "measured",
        "randomization_dataset_quality_statistics_ready": True,
        **quality_decisions,
        "conclusion_decision": quality_preservation_decision["decision"],
        "supports_paper_claim": (
            quality_preservation_decision["scientific_support"] is True
        ),
    }
    summary["randomization_dataset_quality_summary_digest"] = (
        build_stable_digest(summary)
    )
    return RandomizationDatasetQualityStatistics(
        membership_records=canonical_membership,
        prompt_distribution_records=prompt_distribution_records,
        attack_prompt_distribution_records=(
            attack_prompt_distribution_records
        ),
        metric_rows=tuple(dict(row) for row in metric_rows),
        summary=summary,
    )


__all__ = [
    "RANDOMIZATION_DATASET_QUALITY_MEMBERSHIP_FIELDNAMES",
    "RANDOMIZATION_DATASET_QUALITY_METRIC_PROTOCOL_SCHEMA",
    "RANDOMIZATION_DATASET_QUALITY_SUMMARY_SCHEMA",
    "RandomizationDatasetQualityError",
    "RandomizationDatasetQualityStatistics",
    "randomization_dataset_quality_metric_protocol",
    "rebuild_randomization_dataset_quality_statistics",
]
