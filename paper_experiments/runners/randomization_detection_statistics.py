"""从精确随机化聚合包重建绝对检测与逐攻击统计.

该入口只接受聚合 provenance validator 返回的对象. 它先重算45个独立
method-repeat 阈值, 再从同一批原始 observation 归一化主方法与4个 baseline
的异构角色. 所有率值只以 Prompt 为独立统计单位, 真实负结果不会阻止结果
包发布.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Mapping

from experiments.artifacts.artifact_manifest import build_artifact_manifest
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
    validate_detection_key_identity_record,
)
from experiments.protocol.fixed_fpr_observation_audit import (
    FORMAL_THRESHOLD_SOURCE,
)
from experiments.protocol.formal_randomization import (
    formal_randomization_repeat_ids,
    formal_watermark_key_material_from_seed,
    resolve_formal_randomization_repeat,
)
from experiments.protocol.paper_run_config import (
    RUN_EXPECTED_PROMPT_COUNTS,
    normalize_paper_run_name,
)
from experiments.protocol.splits import build_group_split_counts
from experiments.runtime.repository_environment import resolve_code_version
from main.core.digest import build_stable_digest
from main.methods.detection.image_only import (
    validate_image_only_detection_digest_record,
)
from paper_experiments.analysis.paired_superiority import (
    PRIMARY_BASELINE_IDS,
    canonical_attack_registry_rows,
)
from paper_experiments.analysis.randomization_detection_statistics import (
    ATTACKED_FALSE_POSITIVE_SCOPE,
    ATTACKED_TRUE_POSITIVE_SCOPE,
    CLEAN_FALSE_POSITIVE_SCOPE,
    CLEAN_TRUE_POSITIVE_SCOPE,
    DETECTION_METHOD_IDS,
    PROPOSED_METHOD_ID,
    RANDOMIZATION_DETECTION_CLUSTER_FIELDNAMES,
    RANDOMIZATION_DETECTION_OPERATING_POINT_FIELDNAMES,
    RANDOMIZATION_PER_ATTACK_COMPARISON_FIELDNAMES,
    RANDOMIZATION_PER_ATTACK_DETECTION_FIELDNAMES,
    RANDOMIZATION_WRONG_KEY_FIELDNAMES,
    WRONG_KEY_FALSE_POSITIVE_SCOPE,
    build_randomization_detection_statistics,
)
from paper_experiments.runners.randomization_aggregate_provenance import (
    RandomizationAggregateProvenance,
    validate_randomization_aggregate_provenance,
)
from paper_experiments.runners.randomization_method_repeat_thresholds import (
    RandomizationMethodRepeatReconstruction,
    rebuild_randomization_method_repeat_observation_sources,
)


RANDOMIZATION_DETECTION_STATISTICS_OUTPUT_ROOT = (
    "outputs/randomization_detection_statistics"
)
RANDOMIZATION_DETECTION_STATISTICS_REPORT_SCHEMA = (
    "randomization_detection_statistics_reconstruction_report"
)
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_PROMPT_RANDOMIZATION_FIELDS = (
    "randomization_repeat_id",
    "generation_seed_index",
    "generation_seed_offset",
    "generation_seed_random",
    "watermark_key_index",
    "watermark_key_seed_random",
    "watermark_key_material_digest_random",
    "formal_randomization_protocol_digest",
    "formal_randomization_identity_digest_random",
    "base_latent_content_digest_random",
    "base_latent_identity_digest_random",
)


class RandomizationDetectionStatisticsRunnerError(ValueError):
    """表示聚合来源不能形成精确9重复检测统计."""


@dataclass(frozen=True)
class RandomizationDetectionStatisticsResult:
    """保存 Prompt 聚类输入、四类表格、摘要与来源报告."""

    cluster_records: tuple[Mapping[str, Any], ...]
    operating_point_rows: tuple[Mapping[str, Any], ...]
    per_attack_rows: tuple[Mapping[str, Any], ...]
    wrong_key_rows: tuple[Mapping[str, Any], ...]
    per_attack_comparison_rows: tuple[Mapping[str, Any], ...]
    summary: Mapping[str, Any]
    report: Mapping[str, Any]


def _require_provenance(source: RandomizationAggregateProvenance) -> None:
    """拒绝路径、字典或失去 validator 冻结身份的聚合来源."""

    if not isinstance(source, RandomizationAggregateProvenance):
        raise TypeError("跨重复检测统计只接受 RandomizationAggregateProvenance")
    payload = source.payload
    if not all(
        (
            payload.get("randomization_aggregate_ready") is True,
            payload.get("supports_paper_claim") is False,
            str(payload.get("randomization_aggregate_digest", ""))
            == source.randomization_aggregate_digest,
            str(payload.get("common_code_version", "")) == source.common_code_version,
            tuple(payload.get("randomization_repeat_ids", ()))
            == formal_randomization_repeat_ids(),
        )
    ):
        raise RandomizationDetectionStatisticsRunnerError(
            "聚合来源对象未保持 validator 冻结身份"
        )


def _formal_attack_registry() -> tuple[dict[str, str], ...]:
    """从唯一攻击配置来源构造17项正式攻击 registry."""

    return canonical_attack_registry_rows(
        {
            "attack_id": config.attack_id,
            "attack_family": config.attack_family,
            "attack_name": config.attack_name,
            "resource_profile": config.resource_profile,
            "attack_config_digest": attack_config_digest(config),
        }
        for config in default_attack_configs()
        if config.enabled and config.resource_profile in {"full_main", "full_extra"}
    )


def _require_sha256(value: Any, field_name: str) -> str:
    """读取正式来源和图像身份所需的规范 SHA-256."""

    resolved = str(value)
    if SHA256_PATTERN.fullmatch(resolved) is None:
        raise RandomizationDetectionStatisticsRunnerError(
            f"{field_name} 必须是小写 SHA-256"
        )
    return resolved


def _strict_bool(value: Any, field_name: str) -> bool:
    """读取二元检测判定, 禁止整数或文本替代 bool."""

    if not isinstance(value, bool):
        raise RandomizationDetectionStatisticsRunnerError(
            f"{field_name} 必须是布尔值"
        )
    return value


def _finite_float(value: Any, field_name: str) -> float:
    """读取非 bool 的有限检测分数."""

    if isinstance(value, bool):
        raise RandomizationDetectionStatisticsRunnerError(
            f"{field_name} 必须是有限数值"
        )
    try:
        resolved = float(value)
    except (TypeError, ValueError) as exc:
        raise RandomizationDetectionStatisticsRunnerError(
            f"{field_name} 必须是有限数值"
        ) from exc
    if not math.isfinite(resolved):
        raise RandomizationDetectionStatisticsRunnerError(
            f"{field_name} 必须是有限数值"
        )
    return resolved


def _validate_threshold_source_binding(
    rebuilt: RandomizationMethodRepeatReconstruction,
) -> tuple[
    dict[tuple[str, str], Any],
    dict[tuple[str, str], dict[str, Any]],
    dict[str, dict[str, str]],
    str,
]:
    """连接45个原始来源与对应阈值并构造唯一阈值映射摘要."""

    expected_count = len(formal_randomization_repeat_ids()) * len(
        DETECTION_METHOD_IDS
    )
    if (
        len(rebuilt.method_sources) != expected_count
        or len(rebuilt.threshold_records) != expected_count
    ):
        raise RandomizationDetectionStatisticsRunnerError(
            "原始来源与阈值数量必须精确等于45"
        )
    source_by_key = {
        (source.randomization_repeat_id, source.method_id): source
        for source in rebuilt.method_sources
    }
    threshold_by_key = {
        (
            str(record.get("randomization_repeat_id", "")),
            str(record.get("method_id", "")),
        ): dict(record)
        for record in rebuilt.threshold_records
    }
    expected_keys = {
        (repeat_id, method_id)
        for repeat_id in formal_randomization_repeat_ids()
        for method_id in DETECTION_METHOD_IDS
    }
    if (
        len(source_by_key) != expected_count
        or len(threshold_by_key) != expected_count
        or set(source_by_key) != expected_keys
        or set(threshold_by_key) != expected_keys
    ):
        raise RandomizationDetectionStatisticsRunnerError(
            "原始来源或阈值未精确覆盖9重复与5方法"
        )
    threshold_map: dict[str, dict[str, str]] = {}
    for repeat_id, method_id in sorted(expected_keys):
        source = source_by_key[(repeat_id, method_id)]
        threshold = threshold_by_key[(repeat_id, method_id)]
        declared_record_digest = _require_sha256(
            threshold.get("method_repeat_threshold_record_digest", ""),
            "method_repeat_threshold_record_digest",
        )
        threshold_payload = {
            field_name: value
            for field_name, value in threshold.items()
            if field_name != "method_repeat_threshold_record_digest"
        }
        if (
            threshold.get("fixed_fpr_threshold_ready") is not True
            or build_stable_digest(threshold_payload) != declared_record_digest
            or str(threshold.get("observation_source_sha256", ""))
            != str(source.observation_source_sha256)
            or str(threshold.get("randomization_aggregate_digest", ""))
            != str(source.randomization_aggregate_digest)
            or str(threshold.get("common_code_version", ""))
            != str(source.common_code_version)
        ):
            raise RandomizationDetectionStatisticsRunnerError(
                "method-repeat 阈值未绑定同一原始 observation 来源"
            )
        threshold_digest = _require_sha256(
            threshold.get("threshold_digest", ""), "threshold_digest"
        )
        threshold_map.setdefault(repeat_id, {})[method_id] = threshold_digest
    threshold_map_digest = build_stable_digest(threshold_map)
    return source_by_key, threshold_by_key, threshold_map, threshold_map_digest


def _main_decision_and_source_atom(
    row: Mapping[str, Any],
    *,
    repeat_id: str,
    expected_threshold_digest: str,
    expected_wrong_key: bool,
    attacked: bool,
) -> tuple[bool, dict[str, Any]]:
    """核验主方法盲检、密钥角色、阈值和真实图像后读取正式判定."""

    if _require_sha256(
        row.get("frozen_threshold_digest", ""), "frozen_threshold_digest"
    ) != expected_threshold_digest:
        raise RandomizationDetectionStatisticsRunnerError(
            "主方法 observation 未使用当前 repeat 的冻结阈值"
        )
    try:
        validate_image_only_detection_digest_record(row)
    except (TypeError, ValueError) as exc:
        raise RandomizationDetectionStatisticsRunnerError(
            "主方法仅图像检测原子无法独立重建"
        ) from exc
    metadata = row.get("metadata")
    evaluated_path = str(row.get("evaluated_image_path", "")).strip()
    evaluated_digest = _require_sha256(
        row.get("evaluated_image_digest", ""), "evaluated_image_digest"
    )
    if (
        not isinstance(metadata, Mapping)
        or metadata.get("detector_input_access_mode")
        != "image_key_public_model_only"
        or metadata.get("blind_image_detector") is not True
        or metadata.get("generation_latent_trace_required") is not False
        or not evaluated_path
    ):
        raise RandomizationDetectionStatisticsRunnerError(
            "主方法 observation 未保持仅图像盲检输入边界"
        )
    if attacked:
        attacked_digest = _require_sha256(
            row.get("attacked_image_digest", ""), "attacked_image_digest"
        )
        if (
            row.get("attack_performed") is not True
            or not str(row.get("attacked_image_path", "")).strip()
            or attacked_digest != evaluated_digest
        ):
            raise RandomizationDetectionStatisticsRunnerError(
                "主方法攻击记录未重新执行真实图像盲检"
            )
    repeat = resolve_formal_randomization_repeat(repeat_id)
    watermark_key_seed = row.get("watermark_key_seed_random")
    if type(watermark_key_seed) is not int:
        raise RandomizationDetectionStatisticsRunnerError(
            "主方法 observation 缺少正式水印密钥 seed"
        )
    registered_key = formal_watermark_key_material_from_seed(
        watermark_key_seed,
        repeat,
    )
    key_plan = build_detection_key_plan_record(registered_key)
    try:
        key_identity = validate_detection_key_identity_record(row, key_plan)
    except (TypeError, ValueError) as exc:
        raise RandomizationDetectionStatisticsRunnerError(
            "主方法检测密钥身份无法由注册计划重建"
        ) from exc
    expected_role = (
        REGISTERED_WRONG_KEY_ROLE
        if expected_wrong_key
        else REGISTERED_WATERMARK_KEY_ROLE
    )
    if key_identity["detection_key_role"] != expected_role:
        raise RandomizationDetectionStatisticsRunnerError(
            "registered-key 与 wrong-key 样本角色混用"
        )
    decision = _strict_bool(
        row.get("formal_evidence_positive"), "formal_evidence_positive"
    )
    raw_decision = _strict_bool(
        row.get("formal_positive_by_content"), "formal_positive_by_content"
    )
    if raw_decision and not decision:
        raise RandomizationDetectionStatisticsRunnerError(
            "完整 evidence 判定不得撤销 raw-content 阳性"
        )
    return decision, {
        "randomization_repeat_id": repeat_id,
        "threshold_digest": expected_threshold_digest,
        "decision": decision,
        "raw_decision": raw_decision,
        "raw_detection_score": _finite_float(row.get("content_score"), "content_score"),
        "aligned_detection_score": (
            None
            if row.get("aligned_content_score") is None
            else _finite_float(
                row.get("aligned_content_score"), "aligned_content_score"
            )
        ),
        "detector_digest": _require_sha256(
            row.get("detector_digest", ""), "detector_digest"
        ),
        "evaluated_image_digest": evaluated_digest,
        "evaluated_image_path": evaluated_path,
        "detection_key_role": expected_role,
        "detection_key_plan_digest_random": key_identity[
            "detection_key_plan_digest_random"
        ],
    }


def _baseline_decision_and_source_atom(
    row: Mapping[str, Any],
    *,
    method_id: str,
    repeat_id: str,
    calibrated_threshold: float,
    expected_threshold_digest: str,
) -> tuple[bool, dict[str, Any]]:
    """由 baseline 原始分数和当前 repeat 阈值重算判定."""

    if str(row.get("baseline_id", "")) != method_id:
        raise RandomizationDetectionStatisticsRunnerError(
            "baseline observation 方法身份不一致"
        )
    observed_threshold = _finite_float(row.get("threshold"), "threshold")
    score = _finite_float(row.get("score"), "score")
    if (
        not math.isclose(
            observed_threshold,
            calibrated_threshold,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or str(row.get("threshold_source", "")) != FORMAL_THRESHOLD_SOURCE
    ):
        raise RandomizationDetectionStatisticsRunnerError(
            "baseline observation 未使用当前 repeat 的 calibration 阈值"
        )
    decision = score >= calibrated_threshold
    if _strict_bool(row.get("detection_decision"), "detection_decision") != decision:
        raise RandomizationDetectionStatisticsRunnerError(
            "baseline 判定无法由真实分数和冻结阈值重建"
        )
    image_path = str(row.get("image_path", "")).strip()
    image_digest = _require_sha256(row.get("image_digest", ""), "image_digest")
    if not image_path or "detection_key_role" in row:
        raise RandomizationDetectionStatisticsRunnerError(
            "baseline 图像证据无效或混入伪造 wrong-key 语义"
        )
    return decision, {
        "randomization_repeat_id": repeat_id,
        "threshold_digest": expected_threshold_digest,
        "decision": decision,
        "raw_decision": decision,
        "raw_detection_score": score,
        "aligned_detection_score": None,
        "image_digest": image_digest,
    }


def _metric_scope(
    *,
    method_id: str,
    row: Mapping[str, Any],
    attacked: bool,
) -> tuple[str, bool]:
    """把主方法与 baseline 的异构角色归一化到统一统计范围."""

    role = str(row.get("sample_role", ""))
    if method_id == PROPOSED_METHOD_ID:
        mapping = (
            {
                "positive_source": ATTACKED_TRUE_POSITIVE_SCOPE,
                "clean_negative": ATTACKED_FALSE_POSITIVE_SCOPE,
            }
            if attacked
            else {
                "positive_source": CLEAN_TRUE_POSITIVE_SCOPE,
                "clean_negative": CLEAN_FALSE_POSITIVE_SCOPE,
                "wrong_key_negative": WRONG_KEY_FALSE_POSITIVE_SCOPE,
            }
        )
    else:
        mapping = (
            {
                "attacked_positive": ATTACKED_TRUE_POSITIVE_SCOPE,
                "attacked_negative": ATTACKED_FALSE_POSITIVE_SCOPE,
            }
            if attacked
            else {
                "positive_source": CLEAN_TRUE_POSITIVE_SCOPE,
                "clean_negative": CLEAN_FALSE_POSITIVE_SCOPE,
            }
        )
    scope = mapping.get(role)
    if scope is None:
        raise RandomizationDetectionStatisticsRunnerError(
            "observation 样本角色未匹配方法与攻击状态"
        )
    return scope, scope == WRONG_KEY_FALSE_POSITIVE_SCOPE


def _build_cluster_records(
    rebuilt: RandomizationMethodRepeatReconstruction,
    *,
    paper_run_name: str,
    attack_registry: tuple[dict[str, str], ...],
) -> tuple[
    tuple[dict[str, Any], ...],
    dict[str, dict[str, str]],
    str,
]:
    """从45组异构 observation 构造 Prompt 聚类统计输入."""

    (
        source_by_key,
        threshold_by_key,
        threshold_map,
        threshold_map_digest,
    ) = _validate_threshold_source_binding(rebuilt)
    attack_by_id = {str(row["attack_id"]): dict(row) for row in attack_registry}
    attack_ids = set(attack_by_id)
    expected_test_count = build_group_split_counts(
        RUN_EXPECTED_PROMPT_COUNTS[paper_run_name]
    )["test"]
    cluster_atoms: dict[
        tuple[str, str, str, str], list[dict[str, Any]]
    ] = {}
    common_test_prompt_ids: set[str] | None = None
    prompt_identity_by_cell: dict[tuple[str, str], dict[str, Any]] = {}

    for repeat_id in formal_randomization_repeat_ids():
        for method_id in DETECTION_METHOD_IDS:
            source = source_by_key[(repeat_id, method_id)]
            threshold = threshold_by_key[(repeat_id, method_id)]
            threshold_digest = threshold_map[repeat_id][method_id]
            calibrated_threshold = _finite_float(
                threshold.get("calibrated_detection_threshold"),
                "calibrated_detection_threshold",
            )
            test_rows = tuple(
                dict(row)
                for row in source.observation_rows
                if str(row.get("split", "")) == "test"
            )
            local_rows: dict[tuple[str, str, str], dict[str, Any]] = {}
            local_atoms: dict[tuple[str, str, str], dict[str, Any]] = {}
            local_prompt_ids: set[str] = set()
            local_prompt_identity: dict[str, dict[str, Any]] = {}
            for row in test_rows:
                prompt_id = str(row.get("prompt_id", ""))
                attack_id = str(row.get("attack_id", "")).strip()
                attacked = bool(attack_id)
                scope, wrong_key = _metric_scope(
                    method_id=method_id,
                    row=row,
                    attacked=attacked,
                )
                if attacked:
                    attack = attack_by_id.get(attack_id)
                    if attack is None or any(
                        str(row.get(field_name, "")) != str(attack[field_name])
                        for field_name in (
                            "attack_family",
                            "attack_name",
                            "resource_profile",
                            "attack_config_digest",
                        )
                    ):
                        raise RandomizationDetectionStatisticsRunnerError(
                            "observation 攻击身份未匹配17项正式 registry"
                        )
                    generation_seed = row.get("generation_seed_random")
                    if type(generation_seed) is not int or (
                        row.get("attack_seed_random")
                        != formal_attack_seed_random(generation_seed, attack_id)
                    ) or (
                        row.get("formal_attack_seed_protocol_digest")
                        != formal_attack_seed_protocol_record()[
                            "formal_attack_seed_protocol_digest"
                        ]
                    ):
                        raise RandomizationDetectionStatisticsRunnerError(
                            "observation 攻击 seed 未匹配共同公式"
                        )
                elif str(row.get("attack_family", "")).strip() not in {"", "clean"}:
                    raise RandomizationDetectionStatisticsRunnerError(
                        "未攻击 observation 携带了攻击身份"
                    )
                if not prompt_id:
                    raise RandomizationDetectionStatisticsRunnerError(
                        "test observation 缺少 prompt_id"
                    )
                identity = {
                    field_name: row.get(field_name)
                    for field_name in _PROMPT_RANDOMIZATION_FIELDS
                }
                previous_local_identity = local_prompt_identity.setdefault(
                    prompt_id,
                    identity,
                )
                if previous_local_identity != identity:
                    raise RandomizationDetectionStatisticsRunnerError(
                        "同一 method-repeat-Prompt 的随机身份发生漂移"
                    )
                common_identity_key = (repeat_id, prompt_id)
                previous_common_identity = prompt_identity_by_cell.setdefault(
                    common_identity_key,
                    identity,
                )
                if previous_common_identity != identity:
                    raise RandomizationDetectionStatisticsRunnerError(
                        "五方法未共享同一 seed、key 与基础 latent 身份"
                    )

                if method_id == PROPOSED_METHOD_ID:
                    decision, atom = _main_decision_and_source_atom(
                        row,
                        repeat_id=repeat_id,
                        expected_threshold_digest=threshold_digest,
                        expected_wrong_key=wrong_key,
                        attacked=attacked,
                    )
                else:
                    decision, atom = _baseline_decision_and_source_atom(
                        row,
                        method_id=method_id,
                        repeat_id=repeat_id,
                        calibrated_threshold=calibrated_threshold,
                        expected_threshold_digest=threshold_digest,
                    )
                local_key = (prompt_id, scope, attack_id)
                if local_key in local_rows:
                    raise RandomizationDetectionStatisticsRunnerError(
                        "method-repeat observation 角色键重复"
                    )
                local_rows[local_key] = row
                local_atoms[local_key] = atom
                local_prompt_ids.add(prompt_id)
                cluster_atoms.setdefault(
                    (method_id, prompt_id, scope, attack_id),
                    [],
                ).append(atom)

            if len(local_prompt_ids) != expected_test_count:
                raise RandomizationDetectionStatisticsRunnerError(
                    "method-repeat 未覆盖论文层级要求的 test Prompt"
                )
            if common_test_prompt_ids is None:
                common_test_prompt_ids = set(local_prompt_ids)
            elif common_test_prompt_ids != local_prompt_ids:
                raise RandomizationDetectionStatisticsRunnerError(
                    "9重复与5方法未共享同一 test Prompt 集合"
                )
            expected_local_keys = {
                (prompt_id, scope, "")
                for prompt_id in local_prompt_ids
                for scope in (
                    CLEAN_TRUE_POSITIVE_SCOPE,
                    CLEAN_FALSE_POSITIVE_SCOPE,
                    *(
                        (WRONG_KEY_FALSE_POSITIVE_SCOPE,)
                        if method_id == PROPOSED_METHOD_ID
                        else ()
                    ),
                )
            }
            expected_local_keys.update(
                (prompt_id, scope, attack_id)
                for prompt_id in local_prompt_ids
                for scope in (
                    ATTACKED_TRUE_POSITIVE_SCOPE,
                    ATTACKED_FALSE_POSITIVE_SCOPE,
                )
                for attack_id in attack_ids
            )
            if set(local_rows) != expected_local_keys:
                raise RandomizationDetectionStatisticsRunnerError(
                    "method-repeat 未精确覆盖 clean、wrong-key 与17项攻击角色"
                )
            if method_id == PROPOSED_METHOD_ID:
                for prompt_id in local_prompt_ids:
                    registered_atom = local_atoms[
                        (prompt_id, CLEAN_TRUE_POSITIVE_SCOPE, "")
                    ]
                    wrong_key_atom = local_atoms[
                        (prompt_id, WRONG_KEY_FALSE_POSITIVE_SCOPE, "")
                    ]
                    if any(
                        registered_atom[field_name]
                        != wrong_key_atom[field_name]
                        for field_name in (
                            "evaluated_image_path",
                            "evaluated_image_digest",
                        )
                    ):
                        raise RandomizationDetectionStatisticsRunnerError(
                            "wrong-key 未检测与 registered-key 相同的水印图像"
                        )

    repeat_ids = set(formal_randomization_repeat_ids())
    cluster_records: list[dict[str, Any]] = []
    for (method_id, prompt_id, scope, attack_id), atoms in sorted(
        cluster_atoms.items()
    ):
        if (
            len(atoms) != len(repeat_ids)
            or {str(atom["randomization_repeat_id"]) for atom in atoms}
            != repeat_ids
        ):
            raise RandomizationDetectionStatisticsRunnerError(
                "Prompt 聚类未精确覆盖9个注册重复"
            )
        ordered_atoms = tuple(
            sorted(atoms, key=lambda atom: str(atom["randomization_repeat_id"]))
        )
        positive_count = sum(bool(atom["decision"]) for atom in ordered_atoms)
        raw_positive_count = sum(
            bool(atom["raw_decision"]) for atom in ordered_atoms
        )
        decision_map = {
            str(atom["randomization_repeat_id"]): bool(atom["decision"])
            for atom in ordered_atoms
        }
        raw_decision_map = {
            str(atom["randomization_repeat_id"]): bool(atom["raw_decision"])
            for atom in ordered_atoms
        }
        attack = attack_by_id.get(attack_id)
        payload = {
            "method_id": method_id,
            "prompt_id": prompt_id,
            "metric_scope": scope,
            "normalized_sample_role": (
                "positive"
                if scope
                in {CLEAN_TRUE_POSITIVE_SCOPE, ATTACKED_TRUE_POSITIVE_SCOPE}
                else "negative"
            ),
            "attack_id": attack_id,
            "attack_family": "" if attack is None else attack["attack_family"],
            "attack_name": "" if attack is None else attack["attack_name"],
            "resource_profile": "" if attack is None else attack["resource_profile"],
            "attack_config_digest": (
                "" if attack is None else attack["attack_config_digest"]
            ),
            "randomization_repeat_count": len(repeat_ids),
            "registered_repeat_decision_map": decision_map,
            "registered_repeat_raw_decision_map": raw_decision_map,
            "positive_repeat_count": positive_count,
            "prompt_cluster_positive_rate": positive_count / len(repeat_ids),
            "raw_positive_repeat_count": raw_positive_count,
            "prompt_cluster_raw_positive_rate": (
                raw_positive_count / len(repeat_ids)
            ),
            "method_repeat_threshold_map_digest": threshold_map_digest,
            "source_outcome_set_digest": build_stable_digest(ordered_atoms),
        }
        payload["cluster_record_digest"] = build_stable_digest(payload)
        if set(payload) != set(RANDOMIZATION_DETECTION_CLUSTER_FIELDNAMES):
            raise RandomizationDetectionStatisticsRunnerError(
                "Prompt 聚类记录字段集合发生漂移"
            )
        cluster_records.append(payload)
    return tuple(cluster_records), threshold_map, threshold_map_digest


def _rebuild_randomization_detection_statistics(
    source: RandomizationAggregateProvenance,
) -> RandomizationDetectionStatisticsResult:
    """从同一聚合 provenance 重建45阈值、检测聚类与四类结果表."""

    _require_provenance(source)
    paper_run_name = normalize_paper_run_name(
        str(source.payload.get("paper_run_name", ""))
    )
    target_fpr = float(source.payload.get("target_fpr", float("nan")))
    rebuilt = rebuild_randomization_method_repeat_observation_sources(source)
    attack_registry = _formal_attack_registry()
    cluster_records, threshold_map, threshold_map_digest = _build_cluster_records(
        rebuilt,
        paper_run_name=paper_run_name,
        attack_registry=attack_registry,
    )
    (
        operating_point_rows,
        per_attack_rows,
        wrong_key_rows,
        per_attack_comparison_rows,
        summary,
    ) = build_randomization_detection_statistics(
        cluster_records,
        paper_run_name=paper_run_name,
        target_fpr=target_fpr,
        attack_registry_rows=attack_registry,
    )
    report: dict[str, Any] = {
        "report_schema": RANDOMIZATION_DETECTION_STATISTICS_REPORT_SCHEMA,
        "paper_run_name": paper_run_name,
        "target_fpr": target_fpr,
        "randomization_aggregate_package_sha256": source.package_sha256,
        "randomization_aggregate_digest": source.randomization_aggregate_digest,
        "common_code_version": source.common_code_version,
        "randomization_repeat_ids": list(formal_randomization_repeat_ids()),
        "method_ids": list(DETECTION_METHOD_IDS),
        "method_repeat_threshold_map": threshold_map,
        "method_repeat_threshold_map_digest": threshold_map_digest,
        "method_repeat_fixed_fpr_report_digest": rebuilt.report[
            "method_repeat_fixed_fpr_report_digest"
        ],
        "method_repeat_reconstruction_report_digest": rebuilt.reconstruction_report[
            "reconstruction_report_digest"
        ],
        "attack_registry": list(attack_registry),
        "attack_registry_digest": build_stable_digest(list(attack_registry)),
        "cluster_record_set_digest": summary["cluster_record_set_digest"],
        "randomization_detection_statistics_summary_digest": summary[
            "randomization_detection_statistics_summary_digest"
        ],
        "randomization_detection_statistics_ready": True,
        "supports_paper_claim": False,
    }
    report["randomization_detection_statistics_report_digest"] = (
        build_stable_digest(report)
    )
    return RandomizationDetectionStatisticsResult(
        cluster_records=cluster_records,
        operating_point_rows=operating_point_rows,
        per_attack_rows=per_attack_rows,
        wrong_key_rows=wrong_key_rows,
        per_attack_comparison_rows=per_attack_comparison_rows,
        summary=summary,
        report=report,
    )


def rebuild_randomization_detection_statistics(
    source: RandomizationAggregateProvenance,
    *,
    root: str | Path = ".",
) -> RandomizationDetectionStatisticsResult:
    """在与聚合来源相同的 clean Git 提交上执行检测统计重建."""

    _require_provenance(source)
    repository_root = Path(root).resolve()
    if resolve_code_version(repository_root) != source.common_code_version:
        raise RandomizationDetectionStatisticsRunnerError(
            "跨重复检测统计必须使用与聚合来源相同的 clean Git 提交"
        )
    return _rebuild_randomization_detection_statistics(source)


def _resolve_output_directory(
    root: Path,
    output_dir: str | Path | None,
    *,
    paper_run_name: str,
) -> Path:
    """把持久结果限制在仓库 outputs 目录内."""

    requested = (
        root / RANDOMIZATION_DETECTION_STATISTICS_OUTPUT_ROOT / paper_run_name
        if output_dir is None
        else Path(output_dir).expanduser()
    )
    if not requested.is_absolute():
        requested = root / requested
    resolved = requested.resolve()
    try:
        resolved.relative_to((root / "outputs").resolve())
    except ValueError as exc:
        raise RandomizationDetectionStatisticsRunnerError(
            "跨重复检测统计输出目录必须位于 outputs 下"
        ) from exc
    return resolved


def _file_sha256(path: Path) -> str:
    """计算持久结果文件的字节摘要."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv(
    path: Path,
    rows: tuple[Mapping[str, Any], ...],
    fieldnames: tuple[str, ...],
) -> None:
    """使用冻结列顺序写出可独立重建的统计表."""

    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_randomization_detection_statistics_outputs(
    source: RandomizationAggregateProvenance,
    *,
    root: str | Path = ".",
    output_dir: str | Path | None = None,
) -> Path:
    """全部重建成功后事务写出最小检测统计证据."""

    result = rebuild_randomization_detection_statistics(source, root=root)
    repository_root = Path(root).resolve()
    paper_run_name = str(result.report["paper_run_name"])
    destination = _resolve_output_directory(
        repository_root,
        output_dir,
        paper_run_name=paper_run_name,
    )
    if destination.exists():
        raise RandomizationDetectionStatisticsRunnerError(
            "跨重复检测统计输出目录已存在, 不得覆盖或混选运行"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_directory = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}_publish_",
            dir=destination.parent,
        )
    )
    try:
        cluster_path = temporary_directory / "prompt_cluster_detection_records.jsonl"
        operating_path = temporary_directory / "method_detection_operating_points.csv"
        per_attack_path = temporary_directory / "method_attack_detection_metrics.csv"
        wrong_key_path = temporary_directory / "slm_wrong_key_detection_metric.csv"
        comparison_path = temporary_directory / "per_attack_superiority_table.csv"
        summary_path = temporary_directory / "randomization_detection_statistics_summary.json"
        report_path = temporary_directory / "randomization_detection_statistics_report.json"
        manifest_path = temporary_directory / "manifest.local.json"

        cluster_path.write_text(
            "".join(
                json.dumps(dict(record), ensure_ascii=False, sort_keys=True) + "\n"
                for record in result.cluster_records
            ),
            encoding="utf-8",
        )
        _write_csv(
            operating_path,
            result.operating_point_rows,
            RANDOMIZATION_DETECTION_OPERATING_POINT_FIELDNAMES,
        )
        _write_csv(
            per_attack_path,
            result.per_attack_rows,
            RANDOMIZATION_PER_ATTACK_DETECTION_FIELDNAMES,
        )
        _write_csv(
            wrong_key_path,
            result.wrong_key_rows,
            RANDOMIZATION_WRONG_KEY_FIELDNAMES,
        )
        _write_csv(
            comparison_path,
            result.per_attack_comparison_rows,
            RANDOMIZATION_PER_ATTACK_COMPARISON_FIELDNAMES,
        )
        summary_path.write_text(
            json.dumps(dict(result.summary), ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        report_path.write_text(
            json.dumps(dict(result.report), ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        data_paths = (
            cluster_path,
            operating_path,
            per_attack_path,
            wrong_key_path,
            comparison_path,
            summary_path,
            report_path,
        )
        published_paths = tuple(destination / path.name for path in data_paths)
        output_sha256 = {
            published.relative_to(repository_root).as_posix(): _file_sha256(path)
            for path, published in zip(data_paths, published_paths, strict=True)
        }
        published_manifest_path = destination / manifest_path.name
        manifest = build_artifact_manifest(
            artifact_id="randomization_detection_statistics_manifest",
            artifact_type="local_manifest",
            input_paths=(source.package_path.as_posix(),),
            output_paths=tuple(output_sha256)
            + (published_manifest_path.relative_to(repository_root).as_posix(),),
            config={
                "paper_run_name": paper_run_name,
                "target_fpr": result.report["target_fpr"],
                "randomization_aggregate_package_sha256": source.package_sha256,
                "randomization_aggregate_digest": source.randomization_aggregate_digest,
                "common_code_version": source.common_code_version,
                "randomization_repeat_ids": list(formal_randomization_repeat_ids()),
                "method_ids": list(DETECTION_METHOD_IDS),
                "method_repeat_threshold_map_digest": result.report[
                    "method_repeat_threshold_map_digest"
                ],
                "attack_registry_digest": result.report[
                    "attack_registry_digest"
                ],
                "cluster_record_set_digest": result.summary[
                    "cluster_record_set_digest"
                ],
                "randomization_detection_statistics_summary_digest": result.summary[
                    "randomization_detection_statistics_summary_digest"
                ],
                "randomization_detection_statistics_report_digest": result.report[
                    "randomization_detection_statistics_report_digest"
                ],
            },
            code_version=source.common_code_version,
            rebuild_command=(
                "python -m paper_experiments.runners."
                "randomization_detection_statistics "
                f"--paper-run-name {paper_run_name} "
                f"--target-fpr {result.report['target_fpr']} "
                "--aggregate-package-path {aggregate_package_path}"
            ),
            metadata={
                "output_sha256": output_sha256,
                "randomization_detection_statistics_ready": True,
                "main_method_clean_fixed_fpr_ready": result.summary[
                    "main_method_clean_fixed_fpr_ready"
                ],
                "main_method_wrong_key_fixed_fpr_ready": result.summary[
                    "main_method_wrong_key_fixed_fpr_ready"
                ],
                "supports_paper_claim": False,
            },
        ).to_dict()
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary_directory.rename(destination)
        return destination / manifest_path.name
    except Exception:
        shutil.rmtree(temporary_directory, ignore_errors=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    """构造可脱离 Notebook 运行的跨重复检测统计入口."""

    parser = argparse.ArgumentParser(
        description="从已验证的精确随机化聚合包重建绝对检测与逐攻击统计。"
    )
    parser.add_argument("--root", default=".", help="仓库根目录。")
    parser.add_argument(
        "--paper-run-name",
        required=True,
        choices=tuple(RUN_EXPECTED_PROMPT_COUNTS),
        help="论文运行层级。",
    )
    parser.add_argument(
        "--target-fpr",
        required=True,
        type=float,
        help="与聚合包冻结协议一致的目标 FPR。",
    )
    parser.add_argument(
        "--aggregate-package-path",
        required=True,
        help="精确随机化聚合来源 ZIP。",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="可选输出目录, 必须位于仓库 outputs/ 下。",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """验证聚合 provenance 后重建并写出正式检测统计."""

    arguments = build_parser().parse_args(argv)
    source = validate_randomization_aggregate_provenance(
        arguments.aggregate_package_path,
        paper_run_name=arguments.paper_run_name,
        target_fpr=arguments.target_fpr,
    )
    manifest_path = write_randomization_detection_statistics_outputs(
        source,
        root=arguments.root,
        output_dir=arguments.output_dir,
    )
    print(manifest_path.as_posix())


if __name__ == "__main__":
    main()


__all__ = [
    "RANDOMIZATION_DETECTION_STATISTICS_OUTPUT_ROOT",
    "RANDOMIZATION_DETECTION_STATISTICS_REPORT_SCHEMA",
    "RandomizationDetectionStatisticsResult",
    "RandomizationDetectionStatisticsRunnerError",
    "rebuild_randomization_detection_statistics",
    "write_randomization_detection_statistics_outputs",
]
