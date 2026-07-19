"""从正式逐样本记录独立重建论文统计并核对派生表.

该模块位于完整论文实验层,而不是核心方法层.它不信任上游 summary 或 CSV
中的 ready 标记,而是重新消费逐 Prompt 消融记录和逐图像 Inception 特征
记录,调用冻结的实验统计算子重算结果,再逐字段比较持久化派生产物.
"""

from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
import math
import os
from typing import Any, Iterable, Mapping

import numpy as np
import torch

from experiments.ablations.necessity_statistics import (
    ABLATION_NECESSITY_BOOTSTRAP_RESAMPLE_COUNT,
    ABLATION_NECESSITY_FIELDNAMES,
    build_ablation_necessity_statistics,
    canonicalize_ablation_necessity_rows,
)
from experiments.artifacts.dataset_level_quality_outputs import (
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
)
from experiments.protocol.dataset_quality import (
    FORMAL_DATASET_QUALITY_ATTACK_NAME,
    FORMAL_DATASET_QUALITY_IMAGE_PAIR_ROLE,
    FORMAL_DATASET_QUALITY_METRIC_NAMES,
    FORMAL_FEATURE_BACKEND,
    FORMAL_FEATURE_EXTRACTOR_ID,
    formal_dataset_quality_metric_protocol,
    rebuild_formal_fid_kid_metric_rows,
)
from experiments.protocol.method_runtime_config import (
    FORMAL_METHOD_PACKAGE_ROOT,
    load_formal_method_runtime_config,
)
from experiments.protocol.formal_randomization import (
    build_canonical_sd35_base_latent,
    formal_generation_seed,
    formal_randomization_sample_reference,
    formal_runtime_randomization_plan_record,
    formal_watermark_key_material_from_seed,
    resolve_formal_randomization_repeat,
    validate_formal_prompt_randomization_identity,
)
from experiments.protocol.image_only_evidence import (
    FrozenEvidenceProtocol,
    apply_frozen_evidence_protocol,
    calibrate_complete_evidence_protocol,
    frozen_evidence_protocol_digest_payload,
    validate_frozen_evidence_protocol_integrity,
)
from experiments.runners.image_only_dataset_runtime import (
    formal_low_frequency_carrier_protocol_record,
)
from experiments.runners.semantic_watermark_runtime import (
    validate_semantic_watermark_runtime_result_provenance,
)
from experiments.runtime.scientific_unit_provenance import (
    SCIENTIFIC_UNIT_PROVENANCE_AGGREGATE_FIELDS,
    aggregate_scientific_unit_provenance,
)
from experiments.runtime.scientific_content_binding import (
    SCIENTIFIC_CONTENT_BINDING_SCHEMA,
    recompute_scientific_content_binding_digest,
)
from main.core.digest import build_stable_digest
from main.methods.carrier.high_frequency_tail import (
    HIGH_FREQUENCY_TAIL_PROTOCOL_DIGEST,
)
from main.methods.detection import project_image_only_measurement_record
from main.methods.geometry import (
    ATTENTION_ALIGNMENT_ANCHOR_COUNT,
    ATTENTION_ALIGNMENT_MINIMUM_INLIER_RATIO,
    ATTENTION_ALIGNMENT_RESIDUAL_THRESHOLD,
    attention_alignment_gate_record,
)


_FORMAL_METHOD_CONFIG = load_formal_method_runtime_config(
    FORMAL_METHOD_PACKAGE_ROOT
)
_FORMAL_CONTENT_WEIGHT_PROTOCOLS = {
    (
        _FORMAL_METHOD_CONFIG.lf_detection_score_weight,
        _FORMAL_METHOD_CONFIG.tail_robust_detection_score_weight,
    ),
    (1.0, 0.0),
    (0.0, 1.0),
}
_FORMAL_TAIL_FRACTIONS = {_FORMAL_METHOD_CONFIG.tail_fraction}


DATASET_QUALITY_METRIC_FIELDNAMES = (
    "quality_metric_name",
    "quality_metric_value",
    "metric_status",
    "paper_metric_name",
    "feature_backend",
    "source_image_count",
    "comparison_image_count",
    "sample_pair_count",
    "supports_paper_claim",
)
FORMAL_METRIC_RELATIVE_TOLERANCE = 1e-8
FORMAL_METRIC_ABSOLUTE_TOLERANCE = 1e-10
FORMAL_FEATURE_DEPENDENCY_PROFILE_ID = "sd35_method_runtime_gpu"
FORMAL_FEATURE_DIMENSION = int(
    formal_dataset_quality_metric_protocol()["feature_dimension"]
)
_SHA256_CHARACTERS = frozenset("0123456789abcdef")
_FORMAL_DETECTION_DERIVED_FIELDS = (
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
)
_FORMAL_RANDOMIZATION_REFERENCE_FIELDS = (
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


class FormalRecordStatisticsError(ValueError):
    """表示原始正式记录无法唯一重建或与派生证据不一致."""


@lru_cache(maxsize=None)
def _canonical_base_latent_identity(
    shape: tuple[int, ...],
    generation_seed_random: int,
    model_id: str,
    model_revision: str,
    dtype_name: str,
) -> dict[str, Any]:
    """在 CPU 上重建正式基础 latent 身份并按公开随机身份缓存."""

    torch_dtype = getattr(torch, dtype_name.removeprefix("torch."), None)
    if not isinstance(torch_dtype, torch.dtype):
        raise FormalRecordStatisticsError("正式 base latent dtype 无法由 PyTorch 解析")
    _, identity = build_canonical_sd35_base_latent(
        shape=shape,
        generation_seed_random=generation_seed_random,
        model_id=model_id,
        model_revision=model_revision,
        device="cpu",
        dtype=torch_dtype,
    )
    return dict(identity)


def _validated_ablation_randomization_context(
    formal_randomization_plan: Mapping[str, Any],
    randomization_repeat_identity: Mapping[str, Any],
) -> tuple[dict[str, Any], Any]:
    """从冻结方法配置独立验证顶层随机化计划和当前 repeat."""

    expected_plan = formal_runtime_randomization_plan_record(
        _FORMAL_METHOD_CONFIG.seed,
        base_latent_dtype=f"torch.{_FORMAL_METHOD_CONFIG.latent_torch_dtype}",
        base_latent_shape=(
            1,
            16,
            _FORMAL_METHOD_CONFIG.height // 8,
            _FORMAL_METHOD_CONFIG.width // 8,
        ),
    )
    if dict(formal_randomization_plan) != expected_plan:
        raise FormalRecordStatisticsError("消融顶层随机化计划未匹配冻结方法配置")
    repeat_id = str(randomization_repeat_identity.get("randomization_repeat_id", ""))
    try:
        repeat = resolve_formal_randomization_repeat(repeat_id)
    except (TypeError, ValueError) as exc:
        raise FormalRecordStatisticsError("消融活动 repeat 未登记") from exc
    expected_repeat_identity = {
        **repeat.to_dict(),
        "formal_randomization_protocol_digest": expected_plan[
            "formal_randomization_protocol_digest"
        ],
    }
    if dict(randomization_repeat_identity) != expected_repeat_identity:
        raise FormalRecordStatisticsError("消融活动 repeat 身份未匹配冻结注册表")
    return expected_plan, repeat


def _canonical_ablation_randomization_reference(
    *,
    prompt_index: int,
    formal_randomization_plan: Mapping[str, Any],
    repeat: Any,
) -> dict[str, Any]:
    """由顶层计划、规范 Prompt 索引和冻结模型重建样本随机引用."""

    generation_seed_random = formal_generation_seed(
        int(formal_randomization_plan["base_generation_seed_random"]),
        prompt_index,
        repeat,
    )
    key_record = next(
        (
            dict(record)
            for record in formal_randomization_plan["watermark_key_records"]
            if record.get("watermark_key_index") == repeat.watermark_key_index
        ),
        None,
    )
    if key_record is None:
        raise FormalRecordStatisticsError("顶层随机化计划缺少当前 repeat 的密钥记录")
    key_material = formal_watermark_key_material_from_seed(
        int(key_record["watermark_key_seed_random"]),
        repeat,
    )
    identity = validate_formal_prompt_randomization_identity(
        base_generation_seed_random=int(
            formal_randomization_plan["base_generation_seed_random"]
        ),
        prompt_index=prompt_index,
        randomization_repeat_id=repeat.randomization_repeat_id,
        generation_seed_index=repeat.generation_seed_index,
        generation_seed_offset=repeat.generation_seed_offset,
        watermark_key_index=repeat.watermark_key_index,
        generation_seed_random=generation_seed_random,
        watermark_key_seed_random=int(key_record["watermark_key_seed_random"]),
        key_material=key_material,
        formal_randomization_protocol_digest=str(
            formal_randomization_plan["formal_randomization_protocol_digest"]
        ),
    )
    base_latent_identity = _canonical_base_latent_identity(
        tuple(int(value) for value in formal_randomization_plan["base_latent_shape"]),
        generation_seed_random,
        _FORMAL_METHOD_CONFIG.model_id,
        _FORMAL_METHOD_CONFIG.model_revision,
        str(formal_randomization_plan["base_latent_dtype"]),
    )
    return formal_randomization_sample_reference(
        identity,
        base_latent_identity=base_latent_identity,
    )


def _strict_bool(value: Any) -> bool:
    """读取 JSON 或 CSV 中无歧义的布尔表示."""

    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1"}:
            return True
        if normalized in {"false", "0"}:
            return False
    raise FormalRecordStatisticsError(f"字段不是严格布尔值: {value!r}")


def _positive_int(value: Any, field_name: str) -> int:
    """读取必须为正整数的计数字段."""

    if isinstance(value, bool):
        raise FormalRecordStatisticsError(f"{field_name} 不能使用布尔值")
    try:
        resolved = int(value)
    except (TypeError, ValueError) as exc:
        raise FormalRecordStatisticsError(f"{field_name} 不是整数") from exc
    if str(value).strip() not in {str(resolved), f"{resolved}.0"} or resolved <= 0:
        raise FormalRecordStatisticsError(f"{field_name} 必须为正整数")
    return resolved


def _nonnegative_int(value: Any, field_name: str) -> int:
    """读取必须为非负整数的计数字段."""

    if isinstance(value, bool):
        raise FormalRecordStatisticsError(f"{field_name} 不能使用布尔值")
    try:
        resolved = int(value)
    except (TypeError, ValueError) as exc:
        raise FormalRecordStatisticsError(f"{field_name} 不是整数") from exc
    if str(value).strip() not in {str(resolved), f"{resolved}.0"} or resolved < 0:
        raise FormalRecordStatisticsError(f"{field_name} 必须为非负整数")
    return resolved


def _finite_float(value: Any, field_name: str) -> float:
    """读取必须为有限数值的字段."""

    if isinstance(value, bool):
        raise FormalRecordStatisticsError(f"{field_name} 不能使用布尔值")
    try:
        resolved = float(value)
    except (TypeError, ValueError) as exc:
        raise FormalRecordStatisticsError(f"{field_name} 不是数值") from exc
    if not math.isfinite(resolved):
        raise FormalRecordStatisticsError(f"{field_name} 必须为有限数值")
    return resolved


def _is_sha256(value: Any) -> bool:
    """判断字段是否为规范小写 SHA-256."""

    text = str(value)
    return len(text) == 64 and set(text) <= _SHA256_CHARACTERS


def _same_value(reported: Any, rebuilt: Any) -> bool:
    """以冻结数值容差比较 JSON 往返后的标量, 其他类型要求精确相等."""

    if isinstance(reported, bool) or isinstance(rebuilt, bool):
        return type(reported) is type(rebuilt) and reported == rebuilt
    if isinstance(reported, (int, float)) and isinstance(rebuilt, (int, float)):
        return math.isclose(
            float(reported),
            float(rebuilt),
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
    return reported == rebuilt


def validate_frozen_evidence_protocol_record(
    raw_protocol: Mapping[str, Any],
    *,
    expected_target_fpr: float,
) -> FrozenEvidenceProtocol:
    """验证冻结阈值正文、自摘要和目标 FPR, 再构造判定协议."""

    resolved_expected_target_fpr = _finite_float(
        expected_target_fpr,
        "expected_target_fpr",
    )
    if not 0.0 < resolved_expected_target_fpr < 1.0:
        raise FormalRecordStatisticsError("正式消融 target_fpr 必须位于 (0, 1)")
    try:
        protocol = FrozenEvidenceProtocol(**dict(raw_protocol))
        validate_frozen_evidence_protocol_integrity(protocol)
    except (TypeError, ValueError) as exc:
        raise FormalRecordStatisticsError("消融冻结检测协议字段集合无效") from exc
    if not math.isclose(
        _finite_float(protocol.target_fpr, "target_fpr"),
        resolved_expected_target_fpr,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise FormalRecordStatisticsError("消融冻结检测协议的 target_fpr 漂移")
    if (
        protocol.geometry_rescue_enabled
        and _finite_float(protocol.rescue_margin_low, "rescue_margin_low")
        >= 0.0
    ):
        raise FormalRecordStatisticsError(
            "消融冻结检测协议的 rescue_margin_low 必须小于0"
        )
    calibration_negative_count = _positive_int(
        protocol.calibration_negative_count,
        "calibration_negative_count",
    )
    calibration_false_positive_count = _nonnegative_int(
        protocol.calibration_false_positive_count,
        "calibration_false_positive_count",
    )
    if calibration_false_positive_count > calibration_negative_count:
        raise FormalRecordStatisticsError("冻结协议校准假阳性数量超过负样本数量")
    expected_false_positive_rate = (
        calibration_false_positive_count / calibration_negative_count
    )
    if not math.isclose(
        _finite_float(
            protocol.calibration_false_positive_rate,
            "calibration_false_positive_rate",
        ),
        expected_false_positive_rate,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise FormalRecordStatisticsError("冻结协议校准假阳性率与计数不一致")
    try:
        alignment_gate = attention_alignment_gate_record(
            protocol.attention_anchor_count,
            protocol.attention_residual_threshold,
            protocol.attention_minimum_inlier_ratio,
        )
    except ValueError as exc:
        raise FormalRecordStatisticsError(
            "消融冻结检测协议的注意力结构门禁无效"
        ) from exc
    formal_alignment_gate = attention_alignment_gate_record(
        ATTENTION_ALIGNMENT_ANCHOR_COUNT,
        ATTENTION_ALIGNMENT_RESIDUAL_THRESHOLD,
        ATTENTION_ALIGNMENT_MINIMUM_INLIER_RATIO,
    )
    if alignment_gate != formal_alignment_gate:
        raise FormalRecordStatisticsError(
            "消融冻结检测协议的注意力结构门禁发生漂移"
        )
    formal_lf_protocol = formal_low_frequency_carrier_protocol_record()
    if protocol.lf_carrier_protocol_digest != formal_lf_protocol[
        "lf_carrier_protocol_digest"
    ]:
        raise FormalRecordStatisticsError(
            "消融冻结检测协议的 LF 载体协议发生漂移"
        )
    if (
        type(protocol.lf_weight) is not float
        or type(protocol.tail_robust_weight) is not float
        or not math.isclose(
            protocol.lf_weight + protocol.tail_robust_weight,
            1.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or (protocol.lf_weight, protocol.tail_robust_weight)
        not in _FORMAL_CONTENT_WEIGHT_PROTOCOLS
        or type(protocol.tail_fraction) is not float
        or protocol.tail_fraction not in _FORMAL_TAIL_FRACTIONS
    ):
        raise FormalRecordStatisticsError(
            "消融冻结检测协议的内容分支权重无效"
        )
    if (
        protocol.tail_fraction != _FORMAL_METHOD_CONFIG.tail_fraction
        or protocol.tail_carrier_protocol_digest
        != HIGH_FREQUENCY_TAIL_PROTOCOL_DIGEST
    ):
        raise FormalRecordStatisticsError(
            "消融冻结检测协议的尾部载体协议发生漂移"
        )
    return protocol


def _formal_attack_coverage_ready(
    detections: tuple[dict[str, Any], ...],
    *,
    split: str,
    expected_generation_seed_random: int | None = None,
    expected_threshold_digest: str,
) -> bool:
    """独立核验 test split 的完整攻击笛卡尔积及其冻结配置身份."""

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
    actual_by_key: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in attacked_records:
        actual_by_key[
            (
                str(record.get("attack_id", "")),
                str(record.get("sample_role", "")),
            )
        ].append(record)
    if set(actual_by_key) != set(expected_by_key) or any(
        len(rows) != 1 for rows in actual_by_key.values()
    ):
        return False
    attack_seed_protocol_digest = formal_attack_seed_protocol_record()[
        "formal_attack_seed_protocol_digest"
    ]
    for key, config in expected_by_key.items():
        for record in actual_by_key[key]:
            try:
                expected_attack_seed = formal_attack_seed_random(
                    expected_generation_seed_random,
                    config.attack_id,
                )
            except (TypeError, ValueError):
                return False
            if not (
                record.get("attack_family") == config.attack_family
                and record.get("attack_name") == config.attack_name
                and record.get("resource_profile") == config.resource_profile
                and record.get("attack_config_digest")
                == attack_config_digest(config)
                and record.get("attack_parameters")
                == config.attack_parameters
                and record.get("attack_performed") is True
                and record.get("generation_seed_random")
                == expected_generation_seed_random
                and record.get("attack_seed_random")
                == expected_attack_seed
                and record.get("formal_attack_seed_protocol_digest")
                == attack_seed_protocol_digest
                and (
                    config.attack_name != "adversarial_removal_attack"
                    or record.get(
                        "detector_guided_attack_threshold_digest"
                    )
                    == expected_threshold_digest
                )
            ):
                return False
    return True


def _revalidated_detection_records(
    detections: tuple[dict[str, Any], ...],
    protocol: FrozenEvidenceProtocol,
) -> tuple[dict[str, Any], ...]:
    """从原始分数重新应用冻结协议并拒绝持久化判定字段漂移."""

    rebuilt_records: list[dict[str, Any]] = []
    for detection in detections:
        source_with_identity = {
            key: value
            for key, value in detection.items()
            if key not in {"ablation_id", "ablation_prompt_id"}
        }
        try:
            source = project_image_only_measurement_record(
                source_with_identity
            )
            rebuilt = apply_frozen_evidence_protocol((source,), protocol)[0]
        except (KeyError, TypeError, ValueError) as exc:
            raise FormalRecordStatisticsError(
                "消融检测原子缺少冻结判定所需的原始分数"
            ) from exc
        for field_name in _FORMAL_DETECTION_DERIVED_FIELDS:
            if field_name not in detection or not _same_value(
                detection[field_name],
                rebuilt[field_name],
            ):
                raise FormalRecordStatisticsError(
                    "消融检测原子的冻结判定字段无法独立重建: "
                    f"{field_name}"
                )
        rebuilt_records.append(dict(detection))
    return tuple(rebuilt_records)


def _raw_detection_record_content_digest(
    detection: Mapping[str, Any],
) -> str:
    """移除论文派生字段后重建科学运行产生的原始检测记录摘要."""

    raw_record = {
        key: value
        for key, value in detection.items()
        if key
        not in {
            "ablation_id",
            "ablation_prompt_id",
            *_FORMAL_DETECTION_DERIVED_FIELDS,
        }
    }
    return build_stable_digest(raw_record)


def _validate_formal_detections_against_scientific_binding(
    detections: tuple[dict[str, Any], ...],
    runtime_result: Mapping[str, Any],
    protocol: FrozenEvidenceProtocol,
    *,
    expected_detection_key_plan_digest_random: str,
) -> None:
    """把论文检测原子绑定到同一次科学运行的角色、密钥和图像身份."""

    metadata = runtime_result.get("metadata")
    if not isinstance(metadata, Mapping):
        raise FormalRecordStatisticsError("科学运行缺少检测内容绑定 metadata")
    binding_value = metadata.get("scientific_content_binding_record")
    if not isinstance(binding_value, Mapping):
        raise FormalRecordStatisticsError("科学运行缺少检测内容绑定记录")
    binding = dict(binding_value)
    supplied_digest = metadata.get("scientific_content_binding_digest")
    identities = binding.get("detection_content_identities")
    if (
        metadata.get("scientific_content_binding_schema")
        != SCIENTIFIC_CONTENT_BINDING_SCHEMA
        or binding.get("scientific_content_binding_schema")
        != SCIENTIFIC_CONTENT_BINDING_SCHEMA
        or binding.get("scientific_content_binding_digest") != supplied_digest
        or recompute_scientific_content_binding_digest(binding)
        != supplied_digest
        or binding.get("run_id") != runtime_result.get("run_id")
        or binding.get("image_only_measurement_config_digest")
        != protocol.image_only_measurement_config_digest
        or binding.get("detection_key_plan_digest_random")
        != expected_detection_key_plan_digest_random
        or not isinstance(identities, list)
        or len(identities) != len(detections)
    ):
        raise FormalRecordStatisticsError("科学运行的检测内容绑定身份无效")

    actual_by_digest: dict[str, dict[str, Any]] = {}
    for detection in detections:
        digest = _raw_detection_record_content_digest(detection)
        if digest in actual_by_digest:
            raise FormalRecordStatisticsError("论文检测原子包含重复的科学内容身份")
        actual_by_digest[digest] = detection

    expected_digests: set[str] = set()
    for expected_index, identity_value in enumerate(identities):
        if not isinstance(identity_value, Mapping):
            raise FormalRecordStatisticsError("科学检测内容身份不是对象")
        identity = dict(identity_value)
        digest = identity.get("detection_record_content_digest")
        key_identity = identity.get("detection_key_identity")
        if (
            not _is_sha256(digest)
            or digest in expected_digests
            or identity.get("detection_index") != expected_index
            or not isinstance(key_identity, Mapping)
            or key_identity.get("detection_key_plan_digest_random")
            != expected_detection_key_plan_digest_random
        ):
            raise FormalRecordStatisticsError("科学检测内容身份的索引、摘要或密钥计划无效")
        actual = actual_by_digest.get(str(digest))
        if actual is None:
            raise FormalRecordStatisticsError("论文检测原子未匹配科学运行原始记录")
        sample_role = str(actual.get("sample_role", ""))
        attack_id = str(actual.get("attack_id", "none"))
        attack_present = attack_id not in {"", "none"}
        expected_key_role = (
            REGISTERED_WRONG_KEY_ROLE
            if sample_role == "wrong_key_negative" and not attack_present
            else REGISTERED_WATERMARK_KEY_ROLE
        )
        if (
            identity.get("sample_role") != sample_role
            or str(identity.get("attack_id", "none")) != attack_id
            or key_identity.get("detection_key_role") != expected_key_role
        ):
            raise FormalRecordStatisticsError("检测样本角色、攻击或密钥角色未匹配科学运行")
        expected_digests.add(str(digest))
    if expected_digests != set(actual_by_digest):
        raise FormalRecordStatisticsError("论文检测原子与科学运行内容集合不一致")


def rebuild_and_validate_ablation_runtime_aggregates(
    runtime_records: Iterable[Mapping[str, Any]],
    formal_detection_records: Iterable[Mapping[str, Any]],
    frozen_protocols: Mapping[str, Mapping[str, Any]],
    *,
    scientific_unit_identity_records: Iterable[Mapping[str, Any]],
    expected_ablation_ids: Iterable[str],
    expected_prompt_split_by_id: Mapping[str, str],
    expected_prompt_digest_by_id: Mapping[str, str],
    expected_prompt_index_by_id: Mapping[str, int],
    expected_runtime_config_by_ablation_id: Mapping[str, Mapping[str, Any]],
    expected_runtime_output_root: str,
    expected_target_fpr: float,
    formal_randomization_plan: Mapping[str, Any],
    randomization_repeat_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """从检测原子和冻结协议独立重建逐 Prompt 消融聚合字段.

    该函数不信任 ``runtime_rerun_records`` 中的检测率、布尔判定、攻击覆盖或
    阈值身份。它先重新应用冻结判定协议, 再按 ``ablation_id/prompt_id`` 聚合
    原子检测记录, 并把每个 Prompt 的 split、索引和随机身份绑定到规范来源.
    """

    declared_ablation_ids = tuple(str(value) for value in expected_ablation_ids)
    expected_prompt_map = {
        str(prompt_id): str(split)
        for prompt_id, split in expected_prompt_split_by_id.items()
    }
    expected_prompt_digests = {
        str(prompt_id): str(digest)
        for prompt_id, digest in expected_prompt_digest_by_id.items()
    }
    expected_prompt_indices = {
        str(prompt_id): _nonnegative_int(prompt_index, "expected_prompt_index")
        for prompt_id, prompt_index in expected_prompt_index_by_id.items()
    }
    expected_runtime_configs = {
        str(ablation_id): dict(config)
        for ablation_id, config in expected_runtime_config_by_ablation_id.items()
    }
    runtime_output_root = str(expected_runtime_output_root).replace("\\", "/").rstrip("/")
    if (
        not declared_ablation_ids
        or len(set(declared_ablation_ids)) != len(declared_ablation_ids)
        or any(not ablation_id for ablation_id in declared_ablation_ids)
        or not expected_prompt_map
        or len(expected_prompt_map) != len(expected_prompt_split_by_id)
        or any(not prompt_id for prompt_id in expected_prompt_map)
        or set(expected_prompt_digests) != set(expected_prompt_map)
        or set(expected_prompt_indices) != set(expected_prompt_map)
        or set(expected_prompt_indices.values())
        != set(range(len(expected_prompt_indices)))
        or any(not _is_sha256(digest) for digest in expected_prompt_digests.values())
        or any(
            split not in {"dev", "calibration", "test"}
            for split in expected_prompt_map.values()
        )
        or set(expected_runtime_configs) != set(declared_ablation_ids)
        or any(
            config.get("ablation_id") != ablation_id
            for ablation_id, config in expected_runtime_configs.items()
        )
        or not runtime_output_root
    ):
        raise FormalRecordStatisticsError("正式消融身份或规范 Prompt split 映射无效")
    try:
        validated_randomization_plan, active_repeat = (
            _validated_ablation_randomization_context(
                formal_randomization_plan,
                randomization_repeat_identity,
            )
        )
        canonical_randomization_reference_by_prompt_id = {
            prompt_id: _canonical_ablation_randomization_reference(
                prompt_index=expected_prompt_indices[prompt_id],
                formal_randomization_plan=validated_randomization_plan,
                repeat=active_repeat,
            )
            for prompt_id in expected_prompt_map
        }
    except (KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, FormalRecordStatisticsError):
            raise
        raise FormalRecordStatisticsError(
            "正式消融随机身份无法由顶层计划独立重建"
        ) from exc
    if set(frozen_protocols) != set(declared_ablation_ids):
        raise FormalRecordStatisticsError("冻结检测协议未精确覆盖正式消融身份")
    protocols = {
        ablation_id: validate_frozen_evidence_protocol_record(
            frozen_protocols[ablation_id],
            expected_target_fpr=expected_target_fpr,
        )
        for ablation_id in declared_ablation_ids
    }

    materialized_runtime = tuple(dict(record) for record in runtime_records)
    materialized_detections = tuple(
        dict(record) for record in formal_detection_records
    )
    runtime_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for record in materialized_runtime:
        ablation_id = str(record.get("ablation_id", ""))
        prompt_id = str(record.get("prompt_id", ""))
        key = (ablation_id, prompt_id)
        if (
            ablation_id not in protocols
            or prompt_id not in expected_prompt_map
            or key in runtime_by_key
            or str(record.get("split", "")) != expected_prompt_map[prompt_id]
        ):
            raise FormalRecordStatisticsError("逐 Prompt 消融记录身份、split 或唯一性无效")
        prompt_index = _nonnegative_int(record.get("prompt_index"), "prompt_index")
        if prompt_index != expected_prompt_indices[prompt_id]:
            raise FormalRecordStatisticsError("逐 Prompt 消融记录索引未匹配规范 Prompt 文件")
        runtime_by_key[key] = record
    expected_keys = {
        (ablation_id, prompt_id)
        for ablation_id in declared_ablation_ids
        for prompt_id in expected_prompt_map
    }
    if set(runtime_by_key) != expected_keys:
        raise FormalRecordStatisticsError("逐 Prompt 消融记录未精确覆盖规范笛卡尔积")
    materialized_unit_identities = tuple(
        dict(record) for record in scientific_unit_identity_records
    )
    unit_identity_by_run_id = {
        str(record.get("run_id", "")): record
        for record in materialized_unit_identities
    }
    if (
        len(unit_identity_by_run_id) != len(materialized_unit_identities)
        or len(unit_identity_by_run_id) != len(expected_keys)
    ):
        raise FormalRecordStatisticsError(
            "消融顶层 manifest 逐单元身份未精确覆盖运行笛卡尔积"
        )

    detections_by_key: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for detection in materialized_detections:
        key = (
            str(detection.get("ablation_id", "")),
            str(detection.get("ablation_prompt_id", "")),
        )
        if key not in expected_keys:
            raise FormalRecordStatisticsError("消融检测原子包含未声明身份")
        detections_by_key[key].append(detection)
    if set(detections_by_key) != expected_keys:
        raise FormalRecordStatisticsError("消融检测原子未覆盖全部 Prompt 与变体")

    for ablation_id, protocol in protocols.items():
        calibration_negatives = tuple(
            project_image_only_measurement_record(detection)
            for (record_ablation_id, prompt_id), detections in detections_by_key.items()
            if record_ablation_id == ablation_id
            and expected_prompt_map[prompt_id] == "calibration"
            for detection in detections
            if detection.get("sample_role") == "clean_negative"
            and not detection.get("attack_id")
        )
        try:
            rebuilt_protocol = calibrate_complete_evidence_protocol(
                calibration_negatives,
                target_fpr=expected_target_fpr,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise FormalRecordStatisticsError(
                "消融 calibration 检测原子无法重建冻结协议"
            ) from exc
        if any(
            field_name not in frozen_protocols[ablation_id]
            or not _same_value(
                frozen_protocols[ablation_id][field_name],
                rebuilt_value,
            )
            for field_name, rebuilt_value in rebuilt_protocol.to_dict().items()
        ):
            raise FormalRecordStatisticsError(
                "消融冻结检测协议与 calibration 原子独立重建值不一致"
            )

    rebuilt_aggregate_rows: list[dict[str, Any]] = []
    for key in sorted(expected_keys):
        ablation_id, prompt_id = key
        runtime_record = runtime_by_key[key]
        split = expected_prompt_map[prompt_id]
        protocol = protocols[ablation_id]
        detections = _revalidated_detection_records(
            tuple(detections_by_key[key]),
            protocol,
        )
        runtime_result = runtime_record.get("runtime_result")
        runtime_config = runtime_record.get("runtime_config")
        runtime_metadata = (
            runtime_result.get("metadata")
            if isinstance(runtime_result, Mapping)
            else None
        )
        expected_randomization_reference = (
            canonical_randomization_reference_by_prompt_id[prompt_id]
        )
        run_id = str(runtime_result.get("run_id", "")) if isinstance(
            runtime_result,
            Mapping,
        ) else ""
        unit_identity = unit_identity_by_run_id.get(run_id)
        scientific_config = (
            unit_identity.get("scientific_unit_config")
            if isinstance(unit_identity, Mapping)
            else None
        )
        if (
            not isinstance(runtime_result, Mapping)
            or not isinstance(runtime_config, Mapping)
            or not isinstance(runtime_metadata, Mapping)
            or not isinstance(
                scientific_config,
                Mapping,
            )
        ):
            raise FormalRecordStatisticsError("逐 Prompt 消融记录缺少科学运行配置")
        if dict(runtime_config) != expected_runtime_configs[ablation_id]:
            raise FormalRecordStatisticsError("逐 Prompt 消融记录未执行声明消融的精确机制配置")
        try:
            if (
                unit_identity.get("formal_randomization_reference")
                != expected_randomization_reference
                or runtime_metadata.get("formal_randomization_reference")
                != expected_randomization_reference
            ):
                raise ValueError("运行结果未引用独立重建的正式随机身份")
            validate_semantic_watermark_runtime_result_provenance(
                runtime_result,
                unit_config=scientific_config,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise FormalRecordStatisticsError(
                "逐 Prompt 消融结果缺少可重建的科学运行来源"
            ) from exc
        paired_quality = runtime_metadata.get("paired_quality")
        if (
            runtime_result.get("run_decision") != "pass"
            or not isinstance(paired_quality, Mapping)
        ):
            raise FormalRecordStatisticsError("逐 Prompt 消融科学运行未完成或缺少图像质量")
        paired_ssim = _finite_float(paired_quality.get("ssim"), "paired_quality.ssim")
        expected_scientific_randomization_config = {
            "model_id": _FORMAL_METHOD_CONFIG.model_id,
            "model_revision": _FORMAL_METHOD_CONFIG.model_revision,
            "seed": expected_randomization_reference["generation_seed_random"],
            "randomization_repeat_id": active_repeat.randomization_repeat_id,
            "generation_seed_index": active_repeat.generation_seed_index,
            "generation_seed_offset": active_repeat.generation_seed_offset,
            "watermark_key_index": active_repeat.watermark_key_index,
            "watermark_key_seed_random": expected_randomization_reference[
                "watermark_key_seed_random"
            ],
            "formal_randomization_protocol_digest": (
                expected_randomization_reference[
                    "formal_randomization_protocol_digest"
                ]
            ),
            "key_material_digest_random": expected_randomization_reference[
                "watermark_key_material_digest_random"
            ],
            "torch_dtype": str(
                validated_randomization_plan["base_latent_dtype"]
            ).removeprefix("torch."),
            "latent_torch_dtype": _FORMAL_METHOD_CONFIG.latent_torch_dtype,
            "width": _FORMAL_METHOD_CONFIG.width,
            "height": _FORMAL_METHOD_CONFIG.height,
        }
        scientific_attack_protocol = scientific_config.get(
            "detector_guided_attack_threshold_protocol"
        )
        attack_protocol_ready = (
            scientific_attack_protocol == protocol.to_dict()
            if split == "test"
            else scientific_attack_protocol is None
        )
        if (
            str(runtime_config.get("ablation_id", "")) != ablation_id
            or not attack_protocol_ready
            or scientific_config.get("prompt_id") != prompt_id
            or scientific_config.get("split") != split
            or str(scientific_config.get("output_dir", "")).replace("\\", "/")
            != f"{runtime_output_root}/{ablation_id}"
            or build_stable_digest(
                {"prompt_text": scientific_config.get("prompt")}
            )
            != str(runtime_record.get("prompt_digest", ""))
            or str(runtime_record.get("prompt_digest", ""))
            != expected_prompt_digests[prompt_id]
            or any(
                scientific_config.get(field_name) != field_value
                for field_name, field_value in runtime_config.items()
                if field_name != "ablation_id"
            )
            or any(
                scientific_config.get(field_name) != field_value
                for field_name, field_value in (
                    expected_scientific_randomization_config.items()
                )
            )
        ):
            raise FormalRecordStatisticsError("逐 Prompt 消融记录与科学运行配置身份不一致")
        if not run_id or any(
            detection.get("prompt_id") != prompt_id
            or detection.get("split") != split
            or detection.get("run_id") != run_id
            or {
                field_name: detection.get(field_name)
                for field_name in _FORMAL_RANDOMIZATION_REFERENCE_FIELDS
            }
            != expected_randomization_reference
            for detection in detections
        ):
            raise FormalRecordStatisticsError(
                "消融检测原子的 Prompt、运行或随机身份发生漂移"
            )
        try:
            registered_key_material = formal_watermark_key_material_from_seed(
                int(
                    expected_randomization_reference[
                        "watermark_key_seed_random"
                    ]
                ),
                active_repeat,
            )
            expected_detection_key_plan = build_detection_key_plan_record(
                registered_key_material
            )
            _validate_formal_detections_against_scientific_binding(
                detections,
                runtime_result,
                protocol,
                expected_detection_key_plan_digest_random=(
                    expected_detection_key_plan[
                        "detection_key_plan_digest_random"
                    ]
                ),
            )
        except (KeyError, TypeError, ValueError) as exc:
            if isinstance(exc, FormalRecordStatisticsError):
                raise
            raise FormalRecordStatisticsError(
                "消融检测原子无法绑定正式密钥计划和科学运行内容"
            ) from exc

        un_attacked = tuple(
            detection for detection in detections if not detection.get("attack_id")
        )
        un_attacked_by_role: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for detection in un_attacked:
            un_attacked_by_role[str(detection.get("sample_role", ""))].append(
                detection
            )
        expected_un_attacked_roles = {
            "clean_negative",
            "positive_source",
            "wrong_key_negative",
        }
        if set(un_attacked_by_role) != expected_un_attacked_roles or any(
            len(rows) != 1 for rows in un_attacked_by_role.values()
        ):
            raise FormalRecordStatisticsError("消融检测原子缺少唯一 clean/positive/wrong-key 记录")
        attacked_positive = tuple(
            detection
            for detection in detections
            if detection.get("attack_id")
            and detection.get("sample_role") == "positive_source"
        )
        attacked_negative = tuple(
            detection
            for detection in detections
            if detection.get("attack_id")
            and detection.get("sample_role") == "clean_negative"
        )
        coverage_ready = _formal_attack_coverage_ready(
            detections,
            split=split,
            expected_generation_seed_random=int(
                expected_randomization_reference["generation_seed_random"]
            ),
            expected_threshold_digest=protocol.threshold_digest,
        )
        rebuilt = {
            "generation_rerun": True,
            "attack_and_detection_rerun": bool(attacked_positive or attacked_negative),
            "threshold_calibration_scope": "per_ablation_calibration_split",
            "frozen_content_threshold": float(protocol.content_threshold),
            "frozen_threshold_digest": protocol.threshold_digest,
            "clean_negative_positive": bool(
                un_attacked_by_role["clean_negative"][0]["formal_evidence_positive"]
            ),
            "positive_source_positive": bool(
                un_attacked_by_role["positive_source"][0]["formal_evidence_positive"]
            ),
            "wrong_key_negative_positive": bool(
                un_attacked_by_role["wrong_key_negative"][0]["formal_evidence_positive"]
            ),
            "clean_negative_content_score": float(
                un_attacked_by_role["clean_negative"][0]["content_score"]
            ),
            "positive_source_content_score": float(
                un_attacked_by_role["positive_source"][0]["content_score"]
            ),
            "attacked_positive_count": len(attacked_positive),
            "attacked_positive_rate": (
                sum(bool(row["formal_evidence_positive"]) for row in attacked_positive)
                / len(attacked_positive)
                if attacked_positive
                else 0.0
            ),
            "attacked_negative_count": len(attacked_negative),
            "attacked_negative_rate": (
                sum(bool(row["formal_evidence_positive"]) for row in attacked_negative)
                / len(attacked_negative)
                if attacked_negative
                else 0.0
            ),
            "formal_attack_coverage_ready": coverage_ready,
            "paired_ssim": paired_ssim,
        }
        for field_name, rebuilt_value in rebuilt.items():
            if field_name not in runtime_record or not _same_value(
                runtime_record[field_name],
                rebuilt_value,
            ):
                raise FormalRecordStatisticsError(
                    "逐 Prompt 消融聚合字段与检测原子重建值不一致: "
                    f"{ablation_id}/{prompt_id}/{field_name}"
                )
        if not coverage_ready:
            raise FormalRecordStatisticsError("逐 Prompt 消融攻击覆盖不符合冻结协议")
        rebuilt_aggregate_rows.append(
            {
                "ablation_id": ablation_id,
                "prompt_id": prompt_id,
                "split": split,
                "image_only_measurement_config_digest": (
                    protocol.image_only_measurement_config_digest
                ),
                **rebuilt,
            }
        )
    return {
        "ablation_runtime_record_count": len(materialized_runtime),
        "ablation_detection_record_count": len(materialized_detections),
        "ablation_frozen_protocol_count": len(protocols),
        "ablation_runtime_records_digest": build_stable_digest(
            materialized_runtime
        ),
        "ablation_detection_records_digest": build_stable_digest(
            materialized_detections
        ),
        "ablation_frozen_protocols_digest": build_stable_digest(
            dict(frozen_protocols)
        ),
        "ablation_rebuilt_runtime_aggregates_digest": build_stable_digest(
            rebuilt_aggregate_rows
        ),
        "ablation_expected_runtime_configs_digest": build_stable_digest(
            expected_runtime_configs
        ),
        "ablation_image_only_measurement_config_digests": {
            ablation_id: protocols[
                ablation_id
            ].image_only_measurement_config_digest
            for ablation_id in declared_ablation_ids
        },
        "ablation_runtime_aggregate_rebuild_ready": True,
    }


def rebuild_and_validate_ablation_necessity_statistics(
    raw_records: Iterable[Mapping[str, Any]],
    reported_rows: Iterable[Mapping[str, Any]],
    reported_summary: Mapping[str, Any],
    claim_summary: Mapping[str, Any],
    *,
    expected_ablation_ids: Iterable[str],
    expected_paired_prompt_count: int,
) -> dict[str, Any]:
    """从逐 Prompt rerun records 重建必要性统计并逐字段比对.

    ``expected_ablation_ids`` 来自当前正式 manifest/summary 契约,因此该函数
    不绑定当前变体数量.列表必须包含一次 ``complete_method``,其余身份按
    声明顺序生成统计行;未来扩展消融集合时可直接复用.
    """

    declared_ids = tuple(str(value) for value in expected_ablation_ids)
    if (
        not declared_ids
        or declared_ids.count("complete_method") != 1
        or len(set(declared_ids)) != len(declared_ids)
    ):
        raise FormalRecordStatisticsError("正式消融身份必须唯一包含 complete_method")
    variant_ids = tuple(
        ablation_id
        for ablation_id in declared_ids
        if ablation_id != "complete_method"
    )
    if not variant_ids:
        raise FormalRecordStatisticsError("正式消融至少需要一个机制变体")

    materialized_records = tuple(dict(record) for record in raw_records)
    rebuilt_rows, rebuilt_summary = build_ablation_necessity_statistics(
        materialized_records,
        expected_ablation_ids=variant_ids,
        expected_paired_prompt_count=expected_paired_prompt_count,
        bootstrap_resample_count=ABLATION_NECESSITY_BOOTSTRAP_RESAMPLE_COUNT,
    )
    materialized_reported_rows = tuple(dict(row) for row in reported_rows)
    if len(materialized_reported_rows) != len(rebuilt_rows):
        raise FormalRecordStatisticsError("机制必要性统计行数与原始记录重建结果不一致")
    if any(set(row) != set(ABLATION_NECESSITY_FIELDNAMES) for row in materialized_reported_rows):
        raise FormalRecordStatisticsError("机制必要性统计字段集合不符合冻结 schema")
    rebuilt_canonical_rows = canonicalize_ablation_necessity_rows(rebuilt_rows)
    reported_canonical_rows = canonicalize_ablation_necessity_rows(
        materialized_reported_rows
    )
    for row_index, (reported, rebuilt) in enumerate(
        zip(reported_canonical_rows, rebuilt_canonical_rows, strict=True)
    ):
        for field_name in ABLATION_NECESSITY_FIELDNAMES:
            if reported[field_name] != rebuilt[field_name]:
                raise FormalRecordStatisticsError(
                    "机制必要性统计与 raw rerun records 重建值不一致: "
                    f"row={row_index}, field={field_name}"
                )

    for field_name, rebuilt_value in rebuilt_summary.items():
        if reported_summary.get(field_name) != rebuilt_value:
            raise FormalRecordStatisticsError(
                f"机制必要性 summary 与 raw rerun records 重建值不一致: {field_name}"
            )
        # ablation claim summary 的 supports_paper_claim 表示协议闭合,而
        # necessity summary 的同名字段表示全部单机制必要性主张均获支持.
        # 两者语义不同,不得强制相等;其余必要性字段必须逐项绑定.
        if (
            field_name != "supports_paper_claim"
            and claim_summary.get(field_name) != rebuilt_value
        ):
            raise FormalRecordStatisticsError(
                f"消融 claim summary 与 raw rerun records 重建值不一致: {field_name}"
            )
    return {
        "ablation_raw_record_count": len(materialized_records),
        "ablation_raw_records_digest": build_stable_digest(materialized_records),
        "ablation_statistics_rebuilt_rows_digest": build_stable_digest(
            rebuilt_canonical_rows
        ),
        "ablation_statistics_rebuild_ready": True,
    }


def _validated_dataset_quality_image_records(
    image_records: Iterable[Mapping[str, Any]],
    *,
    expected_pair_count: int,
    expected_prompt_id_digest: str,
) -> tuple[dict[str, Any], ...]:
    """验证正式图像对记录的自摘要、Prompt 集合和唯一配对身份."""

    materialized = tuple(dict(record) for record in image_records)
    if len(materialized) != expected_pair_count:
        raise FormalRecordStatisticsError("数据集质量图像记录数量与正式样本规模不一致")
    record_ids: set[str] = set()
    run_ids: set[str] = set()
    prompt_ids: list[str] = []
    pair_indices: list[int] = []
    source_path_identities: set[str] = set()
    comparison_path_identities: set[str] = set()
    expected_fields = {
        "dataset_quality_record_id",
        "dataset_quality_record_digest",
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
    }
    for row_index, record in enumerate(materialized):
        record_id = str(record.get("dataset_quality_record_id", ""))
        run_id = str(record.get("run_id", ""))
        prompt_id = str(record.get("prompt_id", ""))
        if (
            set(record) != expected_fields
            or not record_id
            or record_id in record_ids
            or not run_id
            or run_id in run_ids
            or not prompt_id
        ):
            raise FormalRecordStatisticsError(
                f"数据集质量图像记录身份重复或缺失: row={row_index}"
            )
        pair_index = _nonnegative_int(
            record.get("image_pair_index"),
            "image_pair_index",
        )
        payload = {
            "run_id": run_id,
            "prompt_id": prompt_id,
            "attack_name": str(record.get("attack_name", "")),
            "image_pair_index": pair_index,
            "image_pair_role": str(record.get("image_pair_role", "")),
            "source_image_path": str(record.get("source_image_path", "")),
            "source_image_digest": str(record.get("source_image_digest", "")),
            "comparison_image_path": str(
                record.get("comparison_image_path", "")
            ),
            "comparison_image_digest": str(
                record.get("comparison_image_digest", "")
            ),
            "feature_backend": str(record.get("feature_backend", "")),
            "supports_paper_claim": record.get("supports_paper_claim"),
        }
        digest = str(record.get("dataset_quality_record_digest", ""))
        if (
            not payload["image_pair_role"]
            or payload["attack_name"] != FORMAL_DATASET_QUALITY_ATTACK_NAME
            or payload["image_pair_role"]
            != FORMAL_DATASET_QUALITY_IMAGE_PAIR_ROLE
            or not payload["source_image_path"]
            or not payload["comparison_image_path"]
            or not _is_sha256(payload["source_image_digest"])
            or not _is_sha256(payload["comparison_image_digest"])
            or payload["feature_backend"] != FORMAL_FEATURE_BACKEND
            or payload["supports_paper_claim"] is not False
            or not _is_sha256(digest)
            or build_stable_digest(payload) != digest
            or record_id != f"dataset_quality_record_{digest[:16]}"
        ):
            raise FormalRecordStatisticsError(
                f"数据集质量图像记录正文、自摘要或后端无效: row={row_index}"
            )
        record_ids.add(record_id)
        run_ids.add(run_id)
        prompt_ids.append(prompt_id)
        pair_indices.append(pair_index)
        source_path_identity = os.path.normcase(
            os.path.normpath(payload["source_image_path"])
        )
        comparison_path_identity = os.path.normcase(
            os.path.normpath(payload["comparison_image_path"])
        )
        if (
            source_path_identity == comparison_path_identity
            or source_path_identity in source_path_identities
            or comparison_path_identity in comparison_path_identities
            or source_path_identity in comparison_path_identities
            or comparison_path_identity in source_path_identities
        ):
            raise FormalRecordStatisticsError(
                "数据集质量图像对未使用角色内唯一且跨角色不相交的独立路径"
            )
        source_path_identities.add(source_path_identity)
        comparison_path_identities.add(comparison_path_identity)
    if len(set(prompt_ids)) != expected_pair_count:
        raise FormalRecordStatisticsError("数据集质量图像记录未一对一覆盖 Prompt")
    if sorted(pair_indices) != list(range(expected_pair_count)):
        raise FormalRecordStatisticsError("数据集质量图像对索引不是完整连续集合")
    if build_stable_digest(sorted(prompt_ids)) != expected_prompt_id_digest:
        raise FormalRecordStatisticsError("数据集质量图像记录 Prompt 集合摘要漂移")
    return materialized


def validate_dataset_quality_image_records(
    image_records: Iterable[Mapping[str, Any]],
    *,
    expected_pair_count: int,
    expected_prompt_id_digest: str,
) -> tuple[dict[str, Any], ...]:
    """公开复用正式质量图像记录的自摘要与 Prompt 精确集合校验."""

    return _validated_dataset_quality_image_records(
        image_records,
        expected_pair_count=expected_pair_count,
        expected_prompt_id_digest=expected_prompt_id_digest,
    )


def rebuild_and_validate_dataset_quality_feature_identity(
    image_records: Iterable[Mapping[str, Any]],
    image_resolution_records: Iterable[Mapping[str, Any]],
    feature_records: Iterable[Mapping[str, Any]],
    reported_provenance_summary: Mapping[str, Any],
    actual_source_sha256: Mapping[str, str],
    *,
    expected_pair_count: int,
    expected_prompt_id_digest: str,
    expected_dependency_profile_id: str = FORMAL_FEATURE_DEPENDENCY_PROFILE_ID,
    expected_formal_execution_commit: str | None = None,
    expected_formal_execution_lock_digest: str | None = None,
) -> dict[str, Any]:
    """把正式 Inception 特征绑定到图像记录与科学完成单元来源.

    通用工程写法是先以 ``dataset_quality_record_id/image role`` 建立精确键,
    再逐键比较图像路径和 SHA。项目特定约束是每个 feature batch 的
    ``scientific_unit_provenance`` 还必须绑定同一组图像身份、固定 GPU profile
    和最终 summary 中的完整来源聚合字段。
    """

    resolved_expected_pair_count = _positive_int(
        expected_pair_count,
        "expected_pair_count",
    )
    if not _is_sha256(expected_prompt_id_digest):
        raise FormalRecordStatisticsError("正式 feature 身份重建参数无效")
    materialized_images = _validated_dataset_quality_image_records(
        image_records,
        expected_pair_count=resolved_expected_pair_count,
        expected_prompt_id_digest=expected_prompt_id_digest,
    )
    materialized_resolutions = tuple(
        dict(record) for record in image_resolution_records
    )
    expected_requested_paths = {
        str(image[field_name])
        for image in materialized_images
        for field_name in ("source_image_path", "comparison_image_path")
    }
    resolution_by_requested_path: dict[str, dict[str, Any]] = {}
    resolved_path_identities: set[str] = set()
    expected_resolution_fields = {
        "requested_image_path",
        "resolved_image_path",
        "resolved_from_package_path",
        "resolution_status",
        "resolved_image_digest",
        "materialized_image_input",
        "supports_paper_claim",
        "image_resolution_record_digest",
        "image_resolution_record_id",
    }
    for row_index, resolution in enumerate(materialized_resolutions):
        requested_path = str(resolution.get("requested_image_path", ""))
        resolved_path = str(resolution.get("resolved_image_path", ""))
        digest_payload = {
            key: value
            for key, value in resolution.items()
            if key not in {
                "image_resolution_record_digest",
                "image_resolution_record_id",
            }
        }
        digest = str(resolution.get("image_resolution_record_digest", ""))
        resolved_path_identity = os.path.normcase(os.path.normpath(resolved_path))
        if (
            set(resolution) != expected_resolution_fields
            or not requested_path
            or requested_path in resolution_by_requested_path
            or not resolved_path
            or resolution.get("resolution_status")
            not in {
                "resolved_existing_image_file",
                "materialized_from_input_package",
            }
            or resolution.get("supports_paper_claim") is not False
            or not isinstance(resolution.get("materialized_image_input"), bool)
            or not _is_sha256(resolution.get("resolved_image_digest", ""))
            or not _is_sha256(digest)
            or build_stable_digest(digest_payload) != digest
            or resolution.get("image_resolution_record_id")
            != f"dataset_quality_image_resolution_{digest[:16]}"
            or actual_source_sha256.get(resolved_path)
            != resolution.get("resolved_image_digest")
            or resolved_path_identity in resolved_path_identities
        ):
            raise FormalRecordStatisticsError(
                f"数据集质量图像解析记录或实际文件 SHA 无效: row={row_index}"
            )
        resolution_by_requested_path[requested_path] = resolution
        resolved_path_identities.add(resolved_path_identity)
    if (
        len(materialized_resolutions) != resolved_expected_pair_count * 2
        or len(resolution_by_requested_path) != resolved_expected_pair_count * 2
        or set(resolution_by_requested_path) != expected_requested_paths
    ):
        raise FormalRecordStatisticsError("图像解析记录未精确覆盖全部 source/comparison 路径")

    materialized_features = tuple(dict(record) for record in feature_records)
    image_by_id = {
        str(record["dataset_quality_record_id"]): record
        for record in materialized_images
    }
    feature_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for row_index, feature in enumerate(materialized_features):
        record_id = str(feature.get("dataset_quality_record_id", ""))
        role = str(feature.get("dataset_quality_image_role", ""))
        key = (record_id, role)
        image_record = image_by_id.get(record_id)
        if (
            image_record is None
            or role not in {"source", "comparison"}
            or key in feature_by_key
            or feature.get("supports_paper_claim") is not False
            or feature.get("feature_dimension") != FORMAL_FEATURE_DIMENSION
        ):
            raise FormalRecordStatisticsError(
                f"正式 feature record 的图像对身份或角色无效: row={row_index}"
            )
        path_field = f"{role}_image_path"
        digest_field = f"{role}_image_digest"
        resolution = resolution_by_requested_path[str(image_record[path_field])]
        if (
            str(feature.get("image_path", ""))
            != str(resolution["resolved_image_path"])
            or str(feature.get("image_digest", ""))
            != str(image_record[digest_field])
            or str(resolution["resolved_image_digest"])
            != str(image_record[digest_field])
        ):
            raise FormalRecordStatisticsError(
                f"正式 feature record 与图像路径/SHA 脱离: row={row_index}"
            )
        feature_by_key[key] = feature
    expected_feature_keys = {
        (record_id, role)
        for record_id in image_by_id
        for role in ("source", "comparison")
    }
    if set(feature_by_key) != expected_feature_keys:
        raise FormalRecordStatisticsError("正式 feature records 未精确覆盖全部图像角色")

    # 复用 FID/KID 重建的向量 schema 校验, 避免身份验证与数值验证维护两套
    # feature backend、提取器、维度和有限值规则。
    _formal_feature_arrays(
        materialized_features,
        expected_pair_count=resolved_expected_pair_count,
    )
    try:
        provenance_references = validate_inception_feature_provenance_groups(
            list(materialized_features)
        )
        provenance_summary = aggregate_scientific_unit_provenance(
            provenance_references,
            expected_reference_count=len(materialized_features),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise FormalRecordStatisticsError(
            "正式 feature records 的科学完成单元来源无效"
        ) from exc
    if provenance_summary.get("scientific_unit_provenance_ready") is not True:
        raise FormalRecordStatisticsError("正式 feature 科学来源聚合未就绪")
    if provenance_summary.get("scientific_dependency_profile_ids") != [
        expected_dependency_profile_id
    ]:
        raise FormalRecordStatisticsError("正式 feature 未使用冻结 GPU 依赖 profile")
    if (
        expected_formal_execution_commit is not None
        and provenance_summary.get("scientific_formal_execution_commits")
        != [expected_formal_execution_commit]
    ):
        raise FormalRecordStatisticsError("正式 feature 科学来源的代码提交漂移")
    if (
        expected_formal_execution_lock_digest is not None
        and provenance_summary.get("scientific_formal_execution_lock_digests")
        != [expected_formal_execution_lock_digest]
    ):
        raise FormalRecordStatisticsError("正式 feature 科学来源的执行锁漂移")
    for field_name in SCIENTIFIC_UNIT_PROVENANCE_AGGREGATE_FIELDS:
        if reported_provenance_summary.get(field_name) != provenance_summary.get(
            field_name
        ):
            raise FormalRecordStatisticsError(
                "正式 feature 科学来源与报告聚合字段不一致: "
                f"{field_name}"
            )
    return {
        "dataset_quality_image_record_count": len(materialized_images),
        "dataset_quality_feature_record_count": len(materialized_features),
        "dataset_quality_image_resolution_record_count": len(
            materialized_resolutions
        ),
        "dataset_quality_image_records_digest": build_stable_digest(
            materialized_images
        ),
        "dataset_quality_feature_records_digest": build_stable_digest(
            materialized_features
        ),
        "dataset_quality_image_resolution_records_digest": build_stable_digest(
            materialized_resolutions
        ),
        "dataset_quality_actual_image_sha256_map_digest": build_stable_digest(
            {
                str(resolution["resolved_image_path"]): str(
                    resolution["resolved_image_digest"]
                )
                for resolution in materialized_resolutions
            }
        ),
        "dataset_quality_feature_image_identity_digest": build_stable_digest(
            [
                {
                    "dataset_quality_record_id": record_id,
                    "dataset_quality_image_role": role,
                    "image_path": feature_by_key[(record_id, role)]["image_path"],
                    "image_digest": feature_by_key[(record_id, role)][
                        "image_digest"
                    ],
                }
                for record_id, role in sorted(expected_feature_keys)
            ]
        ),
        "dataset_quality_scientific_unit_provenance_records_digest": (
            provenance_summary["scientific_unit_provenance_records_digest"]
        ),
        "dataset_quality_feature_identity_rebuild_ready": True,
    }


def _formal_feature_arrays(
    feature_records: Iterable[Mapping[str, Any]],
    *,
    expected_pair_count: int,
) -> tuple[np.ndarray, np.ndarray, tuple[dict[str, Any], ...]]:
    """验证 feature records 的身份与角色并构造确定顺序的两组矩阵."""

    materialized = tuple(dict(record) for record in feature_records)
    if len(materialized) != expected_pair_count * 2:
        raise FormalRecordStatisticsError("正式 feature record 数量不是样本对数量的2倍")
    grouped: dict[str, dict[str, np.ndarray]] = defaultdict(dict)
    feature_dimension: int | None = None
    for row_index, record in enumerate(materialized):
        record_id = str(record.get("dataset_quality_record_id", "")).strip()
        role = str(record.get("dataset_quality_image_role", "")).strip()
        if not record_id or role not in {"source", "comparison"}:
            raise FormalRecordStatisticsError(
                f"正式 feature record 身份或角色无效: row={row_index}"
            )
        if role in grouped[record_id]:
            raise FormalRecordStatisticsError(
                f"正式 feature record 出现重复身份角色: {record_id}/{role}"
            )
        if str(record.get("feature_backend", "")) != FORMAL_FEATURE_BACKEND:
            raise FormalRecordStatisticsError("正式 feature record 后端不符合冻结协议")
        if str(record.get("feature_extractor_id", "")) != FORMAL_FEATURE_EXTRACTOR_ID:
            raise FormalRecordStatisticsError("正式 feature record 提取器身份不符合冻结协议")
        if _strict_bool(record.get("supports_paper_claim")):
            raise FormalRecordStatisticsError("原始 feature record 不得直接声明论文结论")
        vector = np.asarray(record.get("feature_vector"), dtype=np.float64)
        if vector.ndim != 1 or vector.size <= 0 or not np.isfinite(vector).all():
            raise FormalRecordStatisticsError("正式 feature vector 必须是一维非空有限向量")
        declared_dimension = _positive_int(
            record.get("feature_dimension"),
            "feature_dimension",
        )
        if declared_dimension != int(vector.size):
            raise FormalRecordStatisticsError("feature_dimension 与向量长度不一致")
        if feature_dimension is None:
            feature_dimension = declared_dimension
        elif feature_dimension != declared_dimension:
            raise FormalRecordStatisticsError("正式 feature records 的向量维度不一致")
        grouped[record_id][role] = vector

    if len(grouped) != expected_pair_count or any(
        set(role_map) != {"source", "comparison"}
        for role_map in grouped.values()
    ):
        raise FormalRecordStatisticsError("正式 feature records 未形成精确 source/comparison 配对")
    ordered_ids = tuple(sorted(grouped))
    source_features = np.stack(
        [grouped[record_id]["source"] for record_id in ordered_ids]
    )
    comparison_features = np.stack(
        [grouped[record_id]["comparison"] for record_id in ordered_ids]
    )
    return source_features, comparison_features, materialized


def _normalized_dataset_metric_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """把 CSV/JSON 指标行规整为可逐字段核对的统一类型."""

    if set(row) != set(DATASET_QUALITY_METRIC_FIELDNAMES):
        raise FormalRecordStatisticsError("FID/KID 指标字段集合不符合冻结 schema")
    try:
        value = float(row["quality_metric_value"])
    except (TypeError, ValueError) as exc:
        raise FormalRecordStatisticsError("FID/KID 指标值不是有限数值") from exc
    if not math.isfinite(value):
        raise FormalRecordStatisticsError("FID/KID 指标值不是有限数值")
    metric_name = str(row["quality_metric_name"])
    paper_metric_name = str(row["paper_metric_name"])
    if (
        metric_name not in FORMAL_DATASET_QUALITY_METRIC_NAMES
        or paper_metric_name != metric_name
    ):
        raise FormalRecordStatisticsError("FID/KID 指标名称不符合冻结三行 schema")
    if metric_name in {"fid", "kid_std"} and value < 0.0:
        raise FormalRecordStatisticsError("FID 与 KID 子集标准差不得为负数")
    return {
        "quality_metric_name": metric_name,
        "quality_metric_value": value,
        "metric_status": str(row["metric_status"]),
        "paper_metric_name": paper_metric_name,
        "feature_backend": str(row["feature_backend"]),
        "source_image_count": _positive_int(
            row["source_image_count"], "source_image_count"
        ),
        "comparison_image_count": _positive_int(
            row["comparison_image_count"], "comparison_image_count"
        ),
        "sample_pair_count": _positive_int(
            row["sample_pair_count"], "sample_pair_count"
        ),
        "supports_paper_claim": _strict_bool(row["supports_paper_claim"]),
    }


def rebuild_formal_fid_kid_metric_rows_from_feature_records(
    feature_records: Iterable[Mapping[str, Any]],
    *,
    expected_pair_count: int,
) -> tuple[dict[str, Any], ...]:
    """从原始正式 feature records 重建规范 FID/KID 三行指标.

    该公共入口集中复用 feature 身份, 角色, 后端, 维度和有限值校验. 调用方
    只需要提供原始记录与精确样本对数量, 不需要先构造或信任任何派生指标表.
    """

    source_features, comparison_features, _ = _formal_feature_arrays(
        feature_records,
        expected_pair_count=expected_pair_count,
    )
    rebuilt_rows = rebuild_formal_fid_kid_metric_rows(
        source_features,
        comparison_features,
        sample_pair_count=expected_pair_count,
    )
    return tuple(
        _normalized_dataset_metric_row(row) for row in rebuilt_rows
    )


def rebuild_and_validate_formal_fid_kid_metrics(
    feature_records: Iterable[Mapping[str, Any]],
    reported_rows: Iterable[Mapping[str, Any]],
    *,
    expected_pair_count: int,
) -> dict[str, Any]:
    """从正式 feature records 重算 FID/KID 并逐字段核对指标表."""

    materialized_records = tuple(dict(record) for record in feature_records)
    normalized_rebuilt = (
        rebuild_formal_fid_kid_metric_rows_from_feature_records(
            materialized_records,
            expected_pair_count=expected_pair_count,
        )
    )
    normalized_reported = tuple(
        _normalized_dataset_metric_row(row) for row in reported_rows
    )
    if len(normalized_reported) != len(normalized_rebuilt):
        raise FormalRecordStatisticsError("FID/KID 指标行数与 feature records 重建结果不一致")
    for row_index, (reported, rebuilt) in enumerate(
        zip(normalized_reported, normalized_rebuilt, strict=True)
    ):
        for field_name in DATASET_QUALITY_METRIC_FIELDNAMES:
            if field_name == "quality_metric_value":
                reported_value = float(reported[field_name])
                rebuilt_value = float(rebuilt[field_name])
                if (
                    rebuilt["quality_metric_name"] == "kid_std"
                    and rebuilt_value == 0.0
                ):
                    metric_value_matches = reported_value == 0.0
                else:
                    metric_value_matches = math.isclose(
                        reported_value,
                        rebuilt_value,
                        rel_tol=FORMAL_METRIC_RELATIVE_TOLERANCE,
                        abs_tol=FORMAL_METRIC_ABSOLUTE_TOLERANCE,
                    )
                if not metric_value_matches:
                    raise FormalRecordStatisticsError(
                        "FID/KID 指标值与 feature records 独立重算结果不一致: "
                        f"row={row_index}"
                    )
            elif reported[field_name] != rebuilt[field_name]:
                raise FormalRecordStatisticsError(
                    "FID/KID 指标字段与 feature records 重建结果不一致: "
                    f"row={row_index}, field={field_name}"
                )
    return {
        "dataset_quality_feature_record_count": len(materialized_records),
        "dataset_quality_rebuilt_metric_rows_digest": build_stable_digest(
            normalized_rebuilt
        ),
        "dataset_quality_metric_relative_tolerance": (
            FORMAL_METRIC_RELATIVE_TOLERANCE
        ),
        "dataset_quality_metric_absolute_tolerance": (
            FORMAL_METRIC_ABSOLUTE_TOLERANCE
        ),
        "dataset_quality_metric_rebuild_ready": True,
    }
