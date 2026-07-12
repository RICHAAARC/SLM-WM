"""基于 Prompt 聚类配对的主方法总体优势统计."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math
import re
from typing import Any, Iterable, Mapping

import numpy as np

from main.core.digest import build_stable_digest


PRIMARY_BASELINE_IDS = (
    "tree_ring",
    "gaussian_shading",
    "shallow_diffuse",
    "t2smark",
)
THRESHOLD_AUDIT_METHOD_IDS = ("slm_wm", *PRIMARY_BASELINE_IDS)
DEFAULT_CONFIDENCE_LEVEL = 0.95
DEFAULT_BOOTSTRAP_RESAMPLE_COUNT = 100_000
MINIMUM_BOOTSTRAP_RESAMPLE_COUNT = 100_000
RESAMPLE_BATCH_SIZE = 128
BOOTSTRAP_ANALYSIS_SCHEMA = "paired_prompt_cluster_bootstrap_v1"
BOOTSTRAP_BIT_GENERATOR = "PCG64"
BOOTSTRAP_QUANTILE_METHOD = "linear"
CLAIM_P_VALUE_METHOD = "bounded_hoeffding_prompt_cluster_mean"
SHARP_NULL_DIAGNOSTIC_METHOD = "exact_prompt_cluster_sign_flip_dp"
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")

THRESHOLD_AUDIT_TEXT_FIELDS = (
    "method_id",
    "threshold_source",
    "threshold_digest",
    "observation_source_sha256",
)
THRESHOLD_AUDIT_FLOAT_FIELDS = (
    "target_fpr",
    "calibrated_detection_threshold",
)
THRESHOLD_AUDIT_INTEGER_FIELDS = (
    "calibration_clean_negative_count",
    "test_clean_negative_count",
)
THRESHOLD_AUDIT_BOOLEAN_FIELDS = (
    "protocol_target_ready",
    "protocol_value_ready",
    "detection_decision_ready",
    "split_count_ready",
    "fixed_fpr_threshold_ready",
    "supports_paper_claim",
)
THRESHOLD_AUDIT_FIELDS = (
    *THRESHOLD_AUDIT_TEXT_FIELDS[:2],
    "target_fpr",
    *THRESHOLD_AUDIT_INTEGER_FIELDS,
    "calibrated_detection_threshold",
    *THRESHOLD_AUDIT_TEXT_FIELDS[2:],
    *THRESHOLD_AUDIT_BOOLEAN_FIELDS,
)
THRESHOLD_AUDIT_REPORT_FIELDS = (
    "paper_claim_scale",
    "target_fpr",
    "expected_method_ids",
    "audited_method_ids",
    "audited_method_count",
    "method_observation_source_sha256_map",
    "method_threshold_digest_map",
    "threshold_audit_rows_digest",
    "method_identity_ready",
    "all_method_thresholds_ready",
    "threshold_observation_binding_ready",
    "fixed_fpr_threshold_audit_ready",
    "supports_paper_claim",
)


class PairedSuperiorityError(ValueError):
    """表示配对观测未满足正式统计的精确集合契约."""


@dataclass(frozen=True)
class PairedOutcome:
    """记录一个 Prompt 在一个正式攻击下的主方法与 baseline 配对判定."""

    baseline_id: str
    prompt_id: str
    randomization_repeat_id: str
    generation_seed_index: int
    generation_seed_offset: int
    generation_seed_random: int
    watermark_key_index: int
    watermark_key_seed_random: int
    watermark_key_material_digest_random: str
    formal_randomization_protocol_digest: str
    formal_randomization_identity_digest_random: str
    base_latent_content_digest_random: str
    base_latent_identity_digest_random: str
    attack_id: str
    attack_family: str
    attack_name: str
    resource_profile: str
    attack_config_digest: str
    proposed_method_threshold_digest: str
    baseline_method_threshold_digest: str
    proposed_decision: bool
    baseline_decision: bool

    @property
    def paired_difference(self) -> int:
        """返回主方法减 baseline 的二元检测差值."""

        return int(self.proposed_decision) - int(self.baseline_decision)

    def to_dict(self) -> dict[str, Any]:
        """转换为可持久化的稳定记录."""

        payload = {
            "baseline_id": self.baseline_id,
            "prompt_id": self.prompt_id,
            "randomization_repeat_id": self.randomization_repeat_id,
            "generation_seed_index": self.generation_seed_index,
            "generation_seed_offset": self.generation_seed_offset,
            "generation_seed_random": self.generation_seed_random,
            "watermark_key_index": self.watermark_key_index,
            "watermark_key_seed_random": self.watermark_key_seed_random,
            "watermark_key_material_digest_random": (
                self.watermark_key_material_digest_random
            ),
            "formal_randomization_protocol_digest": (
                self.formal_randomization_protocol_digest
            ),
            "formal_randomization_identity_digest_random": (
                self.formal_randomization_identity_digest_random
            ),
            "base_latent_content_digest_random": (
                self.base_latent_content_digest_random
            ),
            "base_latent_identity_digest_random": (
                self.base_latent_identity_digest_random
            ),
            "attack_id": self.attack_id,
            "attack_family": self.attack_family,
            "attack_name": self.attack_name,
            "resource_profile": self.resource_profile,
            "attack_config_digest": self.attack_config_digest,
            "proposed_method_threshold_digest": self.proposed_method_threshold_digest,
            "baseline_method_threshold_digest": self.baseline_method_threshold_digest,
            "proposed_decision": self.proposed_decision,
            "baseline_decision": self.baseline_decision,
            "paired_difference": self.paired_difference,
        }
        payload["paired_outcome_digest"] = build_stable_digest(payload)
        return payload


def _text(row: Mapping[str, Any], field_name: str) -> str:
    """读取必须存在的非空文本字段."""

    value = str(row.get(field_name, "")).strip()
    if not value:
        raise PairedSuperiorityError(f"配对观测缺少 {field_name}")
    return value


def _sha256_text(value: Any, field_name: str) -> str:
    """读取规范小写 SHA-256 文本."""

    digest = str(value).strip()
    if SHA256_PATTERN.fullmatch(digest) is None:
        raise PairedSuperiorityError(f"{field_name} 必须是小写 SHA-256")
    return digest


def _strict_boolean(value: Any, field_name: str) -> bool:
    """读取 JSON 或 CSV 中的严格布尔值."""

    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip() in {"True", "False"}:
        return value.strip() == "True"
    raise PairedSuperiorityError(f"{field_name} 必须是布尔值")


def _verified_decision(
    row: Mapping[str, Any],
    *,
    required_field: str,
) -> bool:
    """只读取审计正式校验的判定字段, 并核验可选汇总判定一致."""

    if required_field not in row:
        raise PairedSuperiorityError(f"配对观测缺少判定字段: {required_field}")
    decision = _strict_boolean(row[required_field], required_field)
    if "final_decision" in row:
        final_decision = _strict_boolean(row["final_decision"], "final_decision")
        if final_decision != decision:
            raise PairedSuperiorityError(
                f"final_decision 与 {required_field} 不一致"
            )
    return decision


def _attack_name(row: Mapping[str, Any]) -> str:
    """读取攻击名称, 并拒绝 attack_name 与 attack_condition 分叉."""

    attack_name = str(row.get("attack_name", "")).strip()
    attack_condition = str(row.get("attack_condition", "")).strip()
    if attack_name and attack_condition and attack_name != attack_condition:
        raise PairedSuperiorityError("attack_name 与 attack_condition 不一致")
    value = attack_name or attack_condition
    if not value:
        raise PairedSuperiorityError("配对观测缺少 attack_name")
    return value


def _observation_key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    """构造 Prompt 与攻击条件的唯一配对键."""

    return (
        _text(row, "prompt_id"),
        _text(row, "attack_family"),
        _attack_name(row),
    )


def _formal_attacked_positive_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    accepted_sample_roles: tuple[str, ...],
) -> tuple[dict[str, Any], ...]:
    """仅保留 test split 中实际攻击后的正样本观测."""

    materialized = tuple(dict(row) for row in rows)
    accepted_roles = set(accepted_sample_roles)
    selected = []
    for row in materialized:
        if str(row.get("split", "")) != "test":
            continue
        if str(row.get("sample_role", "")) not in accepted_roles:
            continue
        attack_name = _attack_name(row)
        attack_family = _text(row, "attack_family")
        if attack_name in {"none", "clean", "clean_none"}:
            continue
        if attack_family in {"none", "clean"}:
            continue
        selected.append(row)
    if not selected:
        raise PairedSuperiorityError("配对优势统计缺少 test attacked-positive 观测")
    return tuple(selected)


def _normalize_integer(value: Any, field_name: str) -> int:
    """把 CSV 整数字段恢复为稳定整数类型."""

    if isinstance(value, bool):
        raise PairedSuperiorityError(f"{field_name} 必须是整数")
    try:
        resolved = int(value)
    except (TypeError, ValueError) as error:
        raise PairedSuperiorityError(f"{field_name} 必须是整数") from error
    if isinstance(value, float) and not value.is_integer():
        raise PairedSuperiorityError(f"{field_name} 必须是整数")
    if resolved < 0:
        raise PairedSuperiorityError(f"{field_name} 不得为负数")
    return resolved


def _normalize_float(
    value: Any,
    field_name: str,
    *,
    allow_none: bool = False,
) -> float | None:
    """把 CSV 浮点字段恢复为稳定数值类型."""

    if allow_none and value in {None, ""}:
        return None
    if isinstance(value, bool):
        raise PairedSuperiorityError(f"{field_name} 必须是有限浮点数")
    try:
        resolved = float(value)
    except (TypeError, ValueError) as error:
        raise PairedSuperiorityError(f"{field_name} 必须是有限浮点数") from error
    if not math.isfinite(resolved):
        raise PairedSuperiorityError(f"{field_name} 必须是有限浮点数")
    return resolved


def normalize_threshold_audit_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """把 threshold CSV 行恢复为审计报告使用的规范字段与类型."""

    payload = dict(row)
    if set(payload) != set(THRESHOLD_AUDIT_FIELDS):
        missing = sorted(set(THRESHOLD_AUDIT_FIELDS) - set(payload))
        unexpected = sorted(set(payload) - set(THRESHOLD_AUDIT_FIELDS))
        raise PairedSuperiorityError(
            "threshold audit 字段集合不一致:"
            f"missing={','.join(missing)};unexpected={','.join(unexpected)}"
        )
    normalized = {
        "method_id": _text(payload, "method_id"),
        "threshold_source": _text(payload, "threshold_source"),
        "target_fpr": _normalize_float(payload["target_fpr"], "target_fpr"),
        "calibration_clean_negative_count": _normalize_integer(
            payload["calibration_clean_negative_count"],
            "calibration_clean_negative_count",
        ),
        "test_clean_negative_count": _normalize_integer(
            payload["test_clean_negative_count"],
            "test_clean_negative_count",
        ),
        "calibrated_detection_threshold": _normalize_float(
            payload["calibrated_detection_threshold"],
            "calibrated_detection_threshold",
            allow_none=True,
        ),
        "threshold_digest": _sha256_text(
            payload["threshold_digest"], "threshold_digest"
        ),
        "observation_source_sha256": _sha256_text(
            payload["observation_source_sha256"], "observation_source_sha256"
        ),
    }
    normalized.update(
        {
            field_name: _strict_boolean(payload[field_name], field_name)
            for field_name in THRESHOLD_AUDIT_BOOLEAN_FIELDS
        }
    )
    return normalized


def canonical_threshold_audit_rows(
    rows: Iterable[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    """规范化并按 method_id 排序五方法 threshold 审计行."""

    normalized = tuple(normalize_threshold_audit_row(row) for row in rows)
    method_ids = tuple(str(row["method_id"]) for row in normalized)
    if (
        len(method_ids) != len(THRESHOLD_AUDIT_METHOD_IDS)
        or len(set(method_ids)) != len(method_ids)
        or set(method_ids) != set(THRESHOLD_AUDIT_METHOD_IDS)
    ):
        raise PairedSuperiorityError("threshold audit 必须精确覆盖五个正式方法")
    return tuple(sorted(normalized, key=lambda row: str(row["method_id"])))


def build_threshold_audit_binding_maps(
    rows: Iterable[Mapping[str, Any]],
) -> tuple[dict[str, str], dict[str, str]]:
    """从规范五方法审计行构造阈值与 observation 字节摘要映射."""

    canonical = canonical_threshold_audit_rows(rows)
    threshold_map = {
        str(row["method_id"]): str(row["threshold_digest"])
        for row in canonical
    }
    observation_map = {
        str(row["method_id"]): str(row["observation_source_sha256"])
        for row in canonical
    }
    return threshold_map, observation_map


def build_paired_superiority_protocol_digest(
    threshold_report: Mapping[str, Any],
    threshold_rows: Iterable[Mapping[str, Any]],
    threshold_manifest_config_digest: str,
) -> str:
    """绑定规范 threshold 行, 审计报告和审计 manifest 配置摘要."""

    canonical_rows = canonical_threshold_audit_rows(threshold_rows)
    if set(threshold_report) != set(THRESHOLD_AUDIT_REPORT_FIELDS):
        raise PairedSuperiorityError("threshold audit report 字段集合不一致")
    canonical_report = {
        field_name: threshold_report[field_name]
        for field_name in THRESHOLD_AUDIT_REPORT_FIELDS
    }
    config_digest = _sha256_text(
        threshold_manifest_config_digest,
        "threshold_manifest_config_digest",
    )
    return build_stable_digest(
        {
            "threshold_audit_rows": list(canonical_rows),
            "threshold_audit_report": canonical_report,
            "threshold_audit_manifest_config_digest": config_digest,
        }
    )


def canonical_attack_registry_rows(
    rows: Iterable[Mapping[str, Any]],
) -> tuple[dict[str, str], ...]:
    """提取配对统计所需的正式攻击身份并拒绝重复配置."""

    canonical: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    seen_keys: set[tuple[str, str]] = set()
    for row in rows:
        payload = {
            "attack_id": _text(row, "attack_id"),
            "attack_family": _text(row, "attack_family"),
            "attack_name": _text(row, "attack_name"),
            "resource_profile": _text(row, "resource_profile"),
            "attack_config_digest": _sha256_text(
                row.get("attack_config_digest", ""),
                "attack_config_digest",
            ),
        }
        attack_id = payload["attack_id"]
        attack_key = (payload["attack_family"], payload["attack_name"])
        if attack_id in seen_ids or attack_key in seen_keys:
            raise PairedSuperiorityError("正式攻击 registry 包含重复身份")
        seen_ids.add(attack_id)
        seen_keys.add(attack_key)
        canonical.append(payload)
    if not canonical:
        raise PairedSuperiorityError("正式攻击 registry 不得为空")
    return tuple(sorted(canonical, key=lambda row: row["attack_id"]))


def _validate_observation_attack_identity(
    row: Mapping[str, Any],
    registry_row: Mapping[str, str],
    *,
    require_declared_identity: bool,
) -> None:
    """核验 observation 声明的攻击身份与正式 registry 一致."""

    for field_name in ("attack_id", "resource_profile", "attack_config_digest"):
        declared = str(row.get(field_name, "")).strip()
        if require_declared_identity and not declared:
            raise PairedSuperiorityError(f"observation 缺少 {field_name}")
        if declared and declared != str(registry_row[field_name]):
            raise PairedSuperiorityError(
                f"observation 的 {field_name} 与正式攻击 registry 不一致"
            )


def _paired_randomization_identity(
    row: Mapping[str, Any],
) -> dict[str, Any]:
    """读取配对观测必须共享的种子、密钥和基础 latent 身份."""

    return {
        "randomization_repeat_id": _text(row, "randomization_repeat_id"),
        "generation_seed_index": _normalize_integer(
            row.get("generation_seed_index"),
            "generation_seed_index",
        ),
        "generation_seed_offset": _normalize_integer(
            row.get("generation_seed_offset"),
            "generation_seed_offset",
        ),
        "generation_seed_random": _normalize_integer(
            row.get("generation_seed_random"),
            "generation_seed_random",
        ),
        "watermark_key_index": _normalize_integer(
            row.get("watermark_key_index"),
            "watermark_key_index",
        ),
        "watermark_key_seed_random": _normalize_integer(
            row.get("watermark_key_seed_random"),
            "watermark_key_seed_random",
        ),
        "watermark_key_material_digest_random": _sha256_text(
            row.get("watermark_key_material_digest_random", ""),
            "watermark_key_material_digest_random",
        ),
        "formal_randomization_protocol_digest": _sha256_text(
            row.get("formal_randomization_protocol_digest", ""),
            "formal_randomization_protocol_digest",
        ),
        "formal_randomization_identity_digest_random": _sha256_text(
            row.get("formal_randomization_identity_digest_random", ""),
            "formal_randomization_identity_digest_random",
        ),
        "base_latent_content_digest_random": _sha256_text(
            row.get("base_latent_content_digest_random", ""),
            "base_latent_content_digest_random",
        ),
        "base_latent_identity_digest_random": _sha256_text(
            row.get("base_latent_identity_digest_random", ""),
            "base_latent_identity_digest_random",
        ),
    }


def build_paired_outcomes(
    proposed_rows: Iterable[Mapping[str, Any]],
    baseline_rows: Iterable[Mapping[str, Any]],
    *,
    baseline_id: str,
    proposed_method_threshold_digest: str,
    baseline_method_threshold_digest: str,
    attack_registry_rows: Iterable[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    """对齐一个 baseline 与主方法的完整 Prompt x 正式攻击观测."""

    if baseline_id not in PRIMARY_BASELINE_IDS:
        raise PairedSuperiorityError(f"未登记的主表 baseline: {baseline_id}")
    proposed_threshold_digest = _sha256_text(
        proposed_method_threshold_digest,
        "proposed_method_threshold_digest",
    )
    baseline_threshold_digest = _sha256_text(
        baseline_method_threshold_digest,
        "baseline_method_threshold_digest",
    )
    attack_registry = canonical_attack_registry_rows(attack_registry_rows)
    registry_by_key = {
        (row["attack_family"], row["attack_name"]): row
        for row in attack_registry
    }
    proposed = _formal_attacked_positive_rows(
        proposed_rows,
        accepted_sample_roles=("positive_source",),
    )
    baseline = tuple(
        row
        for row in _formal_attacked_positive_rows(
            baseline_rows,
            accepted_sample_roles=("attacked_positive",),
        )
        if str(row.get("baseline_id", "")) == baseline_id
    )
    if not baseline:
        raise PairedSuperiorityError(f"{baseline_id} 缺少正式配对观测")

    def index_rows(
        rows: tuple[dict[str, Any], ...],
        role: str,
    ) -> dict[tuple[str, str, str], dict[str, Any]]:
        """建立唯一索引并拒绝重复键."""

        indexed: dict[tuple[str, str, str], dict[str, Any]] = {}
        for row in rows:
            key = _observation_key(row)
            if key in indexed:
                raise PairedSuperiorityError(f"{role} 存在重复配对键: {key}")
            indexed[key] = row
        return indexed

    proposed_by_key = index_rows(proposed, "proposed")
    baseline_by_key = index_rows(baseline, baseline_id)
    if set(proposed_by_key) != set(baseline_by_key):
        missing = sorted(set(proposed_by_key) - set(baseline_by_key))
        unexpected = sorted(set(baseline_by_key) - set(proposed_by_key))
        raise PairedSuperiorityError(
            f"{baseline_id} 配对集合不一致: missing={len(missing)};unexpected={len(unexpected)}"
        )
    observed_attack_keys = {
        (attack_family, attack_name)
        for _prompt_id, attack_family, attack_name in proposed_by_key
    }
    if observed_attack_keys != set(registry_by_key):
        raise PairedSuperiorityError(
            f"{baseline_id} 未精确覆盖正式攻击 registry"
        )

    outcomes = []
    for prompt_id, attack_family, attack_name in sorted(proposed_by_key):
        proposed_row = proposed_by_key[(prompt_id, attack_family, attack_name)]
        baseline_row = baseline_by_key[(prompt_id, attack_family, attack_name)]
        registry_row = registry_by_key[(attack_family, attack_name)]
        _validate_observation_attack_identity(
            proposed_row,
            registry_row,
            require_declared_identity=True,
        )
        _validate_observation_attack_identity(
            baseline_row,
            registry_row,
            require_declared_identity=True,
        )
        proposed_randomization = _paired_randomization_identity(proposed_row)
        baseline_randomization = _paired_randomization_identity(baseline_row)
        if proposed_randomization != baseline_randomization:
            raise PairedSuperiorityError(
                f"{baseline_id} 未使用与主方法相同的种子、密钥重复和基础 latent"
            )
        if (
            _sha256_text(
                proposed_row.get("frozen_threshold_digest", ""),
                "frozen_threshold_digest",
            )
            != proposed_threshold_digest
        ):
            raise PairedSuperiorityError("主方法 observation 未使用审计冻结阈值")
        declared_baseline_digest = _sha256_text(
            baseline_row.get("threshold_digest", ""),
            "threshold_digest",
        )
        if declared_baseline_digest != baseline_threshold_digest:
            raise PairedSuperiorityError("baseline observation 未使用审计冻结阈值")
        outcomes.append(
            PairedOutcome(
                baseline_id=baseline_id,
                prompt_id=prompt_id,
                **proposed_randomization,
                attack_id=registry_row["attack_id"],
                attack_family=attack_family,
                attack_name=attack_name,
                resource_profile=registry_row["resource_profile"],
                attack_config_digest=registry_row["attack_config_digest"],
                proposed_method_threshold_digest=proposed_threshold_digest,
                baseline_method_threshold_digest=baseline_threshold_digest,
                proposed_decision=_verified_decision(
                    proposed_row,
                    required_field="formal_evidence_positive",
                ),
                baseline_decision=_verified_decision(
                    baseline_row,
                    required_field="detection_decision",
                ),
            ).to_dict()
        )
    return tuple(outcomes)


def _bounded_hoeffding_mean_p_value(prompt_values: np.ndarray) -> float:
    """计算有界 Prompt 聚类均值的单侧 Hoeffding p 值.

    每个 Prompt 聚类差值都位于 [-1, 1]. 该上界直接检验总体均值不大于0,
    不依赖差值分布的符号对称性, 因而可以作为正式 claim p 值.
    """

    if prompt_values.ndim != 1 or prompt_values.size <= 0:
        raise PairedSuperiorityError("Hoeffding 检验需要非空 Prompt 聚类差值")
    observed_mean = float(prompt_values.mean())
    if observed_mean <= 0.0:
        return 1.0
    exponent = -float(prompt_values.size) * observed_mean * observed_mean / 2.0
    return max(math.exp(exponent), float(np.finfo(np.float64).tiny))


def _exact_cluster_sign_flip_p_value(
    prompt_integer_sums: np.ndarray,
) -> float:
    """用动态规划枚举全部 Prompt sign-flip 组合的单侧概率.

    该检验只对应方法标签可交换的 sharp null. 动态规划对所有符号组合求和,
    不执行 Monte Carlo 抽样, 因而不需要 permutation seed 或重采样次数.
    """

    if prompt_integer_sums.ndim != 1 or prompt_integer_sums.size <= 0:
        raise PairedSuperiorityError("exact sign-flip 需要非空 Prompt 聚类整数和")
    if not np.issubdtype(prompt_integer_sums.dtype, np.integer):
        raise PairedSuperiorityError("exact sign-flip 只接受 Prompt 聚类整数和")
    observed_sum = int(prompt_integer_sums.sum())
    magnitudes = sorted(
        abs(int(value)) for value in prompt_integer_sums.tolist() if int(value) != 0
    )
    distribution = np.asarray([1.0], dtype=np.float64)
    maximum_sum = 0
    for magnitude in magnitudes:
        updated = np.zeros(distribution.size + 2 * magnitude, dtype=np.float64)
        updated[: distribution.size] += 0.5 * distribution
        updated[2 * magnitude :] += 0.5 * distribution
        distribution = updated
        maximum_sum += magnitude
    threshold_index = max(0, observed_sum + maximum_sum)
    if threshold_index >= distribution.size:
        return float(np.finfo(np.float64).tiny)
    probability = float(distribution[threshold_index:].sum(dtype=np.float64))
    return min(
        max(probability, float(np.finfo(np.float64).tiny)),
        1.0,
    )


def build_paired_outcome_set_digest(
    paired_outcomes: Iterable[Mapping[str, Any]],
) -> str:
    """按正式 baseline, Prompt 与攻击身份构造规范 outcome 集合摘要."""

    baseline_order = {
        baseline_id: index for index, baseline_id in enumerate(PRIMARY_BASELINE_IDS)
    }
    materialized = tuple(dict(row) for row in paired_outcomes)
    canonical = sorted(
        materialized,
        key=lambda row: (
            baseline_order.get(_text(row, "baseline_id"), len(baseline_order)),
            _text(row, "prompt_id"),
            _text(row, "attack_id"),
        ),
    )
    return build_stable_digest(canonical)


def _bootstrap_seed_digest_random(
    *,
    baseline_id: str,
    paired_test_prompt_id_digest: str,
    paired_attack_registry_digest: str,
    paired_outcome_set_digest: str,
    confidence_level: float,
    resample_count: int,
) -> tuple[str, int]:
    """用精确分析规范与规范数据摘要构造无自由 nonce 的 bootstrap seed."""

    seed_payload = {
        "analysis_schema": BOOTSTRAP_ANALYSIS_SCHEMA,
        "baseline_id": baseline_id,
        "paired_test_prompt_id_digest": _sha256_text(
            paired_test_prompt_id_digest,
            "paired_test_prompt_id_digest",
        ),
        "paired_attack_registry_digest": _sha256_text(
            paired_attack_registry_digest,
            "paired_attack_registry_digest",
        ),
        "paired_outcome_set_digest": _sha256_text(
            paired_outcome_set_digest,
            "paired_outcome_set_digest",
        ),
        "confidence_level": float(confidence_level),
        "bootstrap_resample_count": int(resample_count),
        "bootstrap_bit_generator": BOOTSTRAP_BIT_GENERATOR,
        "bootstrap_quantile_method": BOOTSTRAP_QUANTILE_METHOD,
    }
    digest = build_stable_digest(seed_payload)
    return digest, int(digest, 16)


def _cluster_bootstrap_interval(
    prompt_values: np.ndarray,
    *,
    confidence_level: float,
    resample_count: int,
    seed: int,
) -> tuple[float, float]:
    """以 Prompt 为聚类单位计算配对均值差的 percentile CI."""

    if prompt_values.ndim != 1 or prompt_values.size <= 0:
        raise PairedSuperiorityError("bootstrap 需要非空 Prompt 聚类差值")
    if not 0.0 < confidence_level < 1.0:
        raise PairedSuperiorityError("confidence_level 必须位于 (0, 1)")
    if resample_count < MINIMUM_BOOTSTRAP_RESAMPLE_COUNT:
        raise PairedSuperiorityError(
            "bootstrap_resample_count 不得小于"
            f"{MINIMUM_BOOTSTRAP_RESAMPLE_COUNT}"
        )
    generator = np.random.Generator(np.random.PCG64(seed))
    estimates = np.empty(resample_count, dtype=np.float64)
    prompt_count = int(prompt_values.size)
    for start in range(0, resample_count, RESAMPLE_BATCH_SIZE):
        stop = min(start + RESAMPLE_BATCH_SIZE, resample_count)
        indices = generator.integers(
            0,
            prompt_count,
            size=(stop - start, prompt_count),
            endpoint=False,
        )
        estimates[start:stop] = prompt_values[indices].mean(axis=1)
    alpha = 1.0 - confidence_level
    low, high = np.quantile(
        estimates,
        [alpha / 2.0, 1.0 - alpha / 2.0],
        method=BOOTSTRAP_QUANTILE_METHOD,
    )
    return float(low), float(high)


def _outcome_attack_descriptor(row: Mapping[str, Any]) -> dict[str, str]:
    """提取配对 outcome 已绑定的正式攻击身份."""

    return {
        "attack_id": _text(row, "attack_id"),
        "attack_family": _text(row, "attack_family"),
        "attack_name": _text(row, "attack_name"),
        "resource_profile": _text(row, "resource_profile"),
        "attack_config_digest": _sha256_text(
            row.get("attack_config_digest", ""),
            "attack_config_digest",
        ),
    }


def build_paired_test_prompt_identity(
    paired_outcomes: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """从四 baseline 的 exact outcome 集合计算规范 test Prompt 摘要."""

    grouped: dict[str, set[str]] = defaultdict(set)
    for row in paired_outcomes:
        baseline_id = _text(row, "baseline_id")
        if baseline_id not in PRIMARY_BASELINE_IDS:
            raise PairedSuperiorityError(f"未登记的主表 baseline: {baseline_id}")
        grouped[baseline_id].add(_text(row, "prompt_id"))
    if set(grouped) != set(PRIMARY_BASELINE_IDS):
        raise PairedSuperiorityError("Prompt identity 必须精确覆盖4个主表 baseline")
    prompt_sets = {frozenset(values) for values in grouped.values()}
    if len(prompt_sets) != 1:
        raise PairedSuperiorityError("4个 baseline 的 test Prompt 集合不一致")
    prompt_ids = tuple(sorted(next(iter(prompt_sets))))
    if not prompt_ids:
        raise PairedSuperiorityError("test Prompt 集合不得为空")
    return {
        "paired_test_prompt_count": len(prompt_ids),
        "paired_test_prompt_id_digest": build_stable_digest(list(prompt_ids)),
    }


def build_paired_superiority_rows(
    paired_outcomes: Iterable[Mapping[str, Any]],
    *,
    protocol_digest: str,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    bootstrap_resample_count: int = DEFAULT_BOOTSTRAP_RESAMPLE_COUNT,
) -> list[dict[str, Any]]:
    """对每个主表 baseline 计算 Prompt-clustered 总体优势证据."""

    resolved_protocol_digest = _sha256_text(protocol_digest, "protocol_digest")
    rows = [dict(row) for row in paired_outcomes]
    paired_outcome_set_digest = build_paired_outcome_set_digest(rows)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        baseline_id = _text(row, "baseline_id")
        if baseline_id not in PRIMARY_BASELINE_IDS:
            raise PairedSuperiorityError(f"未登记的主表 baseline: {baseline_id}")
        grouped[baseline_id].append(row)
    if set(grouped) != set(PRIMARY_BASELINE_IDS):
        raise PairedSuperiorityError("配对优势统计必须精确覆盖4个主表 baseline")

    summaries: list[dict[str, Any]] = []
    common_prompt_ids: tuple[str, ...] | None = None
    common_attack_descriptors: tuple[dict[str, str], ...] | None = None
    common_proposed_threshold_digest: str | None = None
    for baseline_id in PRIMARY_BASELINE_IDS:
        baseline_rows = grouped[baseline_id]
        by_prompt: dict[str, list[int]] = defaultdict(list)
        attack_ids_by_prompt: dict[str, set[str]] = defaultdict(set)
        attack_descriptor_by_id: dict[str, dict[str, str]] = {}
        proposed_threshold_digests: set[str] = set()
        baseline_threshold_digests: set[str] = set()
        for row in baseline_rows:
            prompt_id = _text(row, "prompt_id")
            descriptor = _outcome_attack_descriptor(row)
            attack_id = descriptor["attack_id"]
            if attack_id in attack_ids_by_prompt[prompt_id]:
                raise PairedSuperiorityError(
                    f"{baseline_id} 的 Prompt {prompt_id} 存在重复攻击键"
                )
            previous_descriptor = attack_descriptor_by_id.setdefault(
                attack_id,
                descriptor,
            )
            if previous_descriptor != descriptor:
                raise PairedSuperiorityError("同一 attack_id 绑定了不同正式攻击配置")
            attack_ids_by_prompt[prompt_id].add(attack_id)
            proposed = row.get("proposed_decision")
            baseline = row.get("baseline_decision")
            if not isinstance(proposed, bool) or not isinstance(baseline, bool):
                raise PairedSuperiorityError("配对 outcome 判定字段必须是布尔值")
            difference = int(row.get("paired_difference", 99))
            expected_difference = int(proposed) - int(baseline)
            if difference != expected_difference or difference not in {-1, 0, 1}:
                raise PairedSuperiorityError("paired_difference 与二元判定不一致")
            by_prompt[prompt_id].append(difference)
            proposed_threshold_digests.add(
                _sha256_text(
                    row.get("proposed_method_threshold_digest", ""),
                    "proposed_method_threshold_digest",
                )
            )
            baseline_threshold_digests.add(
                _sha256_text(
                    row.get("baseline_method_threshold_digest", ""),
                    "baseline_method_threshold_digest",
                )
            )
        prompt_ids = tuple(sorted(by_prompt))
        attack_sets = {frozenset(value) for value in attack_ids_by_prompt.values()}
        if len(attack_sets) != 1:
            raise PairedSuperiorityError(f"{baseline_id} 的 Prompt 未覆盖同一组攻击")
        attack_ids = next(iter(attack_sets))
        if attack_ids != set(attack_descriptor_by_id):
            raise PairedSuperiorityError(f"{baseline_id} 的攻击 registry 覆盖不完整")
        attack_descriptors = tuple(
            attack_descriptor_by_id[attack_id] for attack_id in sorted(attack_ids)
        )
        if common_prompt_ids is None:
            common_prompt_ids = prompt_ids
            common_attack_descriptors = attack_descriptors
        elif (
            prompt_ids != common_prompt_ids
            or attack_descriptors != common_attack_descriptors
        ):
            raise PairedSuperiorityError("4个 baseline 未共享同一 Prompt x attack 集合")
        if len(proposed_threshold_digests) != 1 or len(baseline_threshold_digests) != 1:
            raise PairedSuperiorityError("配对 outcome 未共享唯一方法阈值摘要")
        proposed_threshold_digest = next(iter(proposed_threshold_digests))
        baseline_threshold_digest = next(iter(baseline_threshold_digests))
        if common_proposed_threshold_digest is None:
            common_proposed_threshold_digest = proposed_threshold_digest
        elif proposed_threshold_digest != common_proposed_threshold_digest:
            raise PairedSuperiorityError("4个比较未共享主方法阈值摘要")
        prompt_values = np.asarray(
            [float(np.mean(by_prompt[prompt_id])) for prompt_id in prompt_ids],
            dtype=np.float64,
        )
        prompt_integer_sums = np.asarray(
            [sum(by_prompt[prompt_id]) for prompt_id in prompt_ids],
            dtype=np.int64,
        )
        positive_prompt_count = int(np.count_nonzero(prompt_values > 0.0))
        negative_prompt_count = int(np.count_nonzero(prompt_values < 0.0))
        tied_prompt_count = (
            int(prompt_values.size) - positive_prompt_count - negative_prompt_count
        )
        paired_test_prompt_id_digest = build_stable_digest(list(prompt_ids))
        paired_attack_registry_digest = build_stable_digest(
            list(attack_descriptors)
        )
        bootstrap_seed_digest, bootstrap_seed = _bootstrap_seed_digest_random(
            baseline_id=baseline_id,
            paired_test_prompt_id_digest=paired_test_prompt_id_digest,
            paired_attack_registry_digest=paired_attack_registry_digest,
            paired_outcome_set_digest=paired_outcome_set_digest,
            confidence_level=confidence_level,
            resample_count=int(bootstrap_resample_count),
        )
        ci_low, ci_high = _cluster_bootstrap_interval(
            prompt_values,
            confidence_level=confidence_level,
            resample_count=int(bootstrap_resample_count),
            seed=bootstrap_seed,
        )
        claim_p_value = _bounded_hoeffding_mean_p_value(prompt_values)
        sharp_null_p_value = _exact_cluster_sign_flip_p_value(prompt_integer_sums)
        mean_difference = float(prompt_values.mean())
        summaries.append(
            {
                "baseline_id": baseline_id,
                "paired_prompt_count": len(prompt_ids),
                "paired_attack_count": len(attack_ids),
                "paired_observation_count": len(baseline_rows),
                "mean_paired_true_positive_rate_difference": mean_difference,
                "mean_paired_difference_ci_low": ci_low,
                "mean_paired_difference_ci_high": ci_high,
                "positive_prompt_cluster_count": positive_prompt_count,
                "negative_prompt_cluster_count": negative_prompt_count,
                "tied_prompt_cluster_count": tied_prompt_count,
                "one_sided_bounded_hoeffding_mean_p_value": claim_p_value,
                "one_sided_exact_prompt_cluster_sign_flip_p_value": (
                    sharp_null_p_value
                ),
                "exact_prompt_cluster_sign_flip_p_value_is_diagnostic": True,
                "sharp_null_diagnostic_method": SHARP_NULL_DIAGNOSTIC_METHOD,
                "claim_p_value_method": CLAIM_P_VALUE_METHOD,
                "confidence_level": confidence_level,
                "bootstrap_resample_count": int(bootstrap_resample_count),
                "bootstrap_seed_digest_random": bootstrap_seed_digest,
                "bootstrap_analysis_schema": BOOTSTRAP_ANALYSIS_SCHEMA,
                "bootstrap_bit_generator": BOOTSTRAP_BIT_GENERATOR,
                "bootstrap_quantile_method": BOOTSTRAP_QUANTILE_METHOD,
                "proposed_method_threshold_digest": proposed_threshold_digest,
                "baseline_method_threshold_digest": baseline_threshold_digest,
                "paired_test_prompt_id_digest": paired_test_prompt_id_digest,
                "paired_attack_registry_digest": paired_attack_registry_digest,
                "paired_outcome_set_digest": paired_outcome_set_digest,
                "protocol_digest": resolved_protocol_digest,
            }
        )

    ordered = sorted(
        enumerate(summaries),
        key=lambda item: (
            item[1]["one_sided_bounded_hoeffding_mean_p_value"],
            item[1]["baseline_id"],
        ),
    )
    running_adjusted = 0.0
    comparison_count = len(summaries)
    for rank, (original_index, row) in enumerate(ordered, start=1):
        adjusted = min(
            1.0,
            max(
                running_adjusted,
                float(row["one_sided_bounded_hoeffding_mean_p_value"])
                * (comparison_count - rank + 1),
            ),
        )
        running_adjusted = adjusted
        summaries[original_index]["holm_adjusted_p_value"] = adjusted
    for row in summaries:
        ready = bool(
            float(row["mean_paired_true_positive_rate_difference"]) > 0.0
            and float(row["mean_paired_difference_ci_low"]) > 0.0
            and float(row["holm_adjusted_p_value"]) < 0.05
        )
        row["paired_superiority_ready"] = ready
        row["supports_paper_claim"] = ready
    return summaries


def build_paired_superiority_summary(
    rows: Iterable[Mapping[str, Any]],
    *,
    paired_outcomes: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """汇总4个主表 baseline 的总体配对优势门禁."""

    materialized = sorted(
        (dict(row) for row in rows),
        key=lambda row: str(row.get("baseline_id", "")),
    )
    baseline_ids = [str(row.get("baseline_id", "")) for row in materialized]
    exact_set_ready = (
        len(baseline_ids) == len(PRIMARY_BASELINE_IDS)
        and len(set(baseline_ids)) == len(baseline_ids)
        and set(baseline_ids) == set(PRIMARY_BASELINE_IDS)
    )
    ready_ids = [
        baseline_id
        for baseline_id in PRIMARY_BASELINE_IDS
        if any(
            row.get("baseline_id") == baseline_id
            and row.get("paired_superiority_ready") is True
            for row in materialized
        )
    ]
    overall_ready = exact_set_ready and len(ready_ids) == len(PRIMARY_BASELINE_IDS)
    return {
        "primary_baseline_ids": list(PRIMARY_BASELINE_IDS),
        "paired_superiority_row_count": len(materialized),
        "paired_superiority_ready_ids": ready_ids,
        "paired_superiority_exact_set_ready": exact_set_ready,
        "overall_paired_superiority_ready": overall_ready,
        "paired_superiority_rows_digest": build_stable_digest(materialized),
        **build_paired_test_prompt_identity(paired_outcomes),
        "supports_paper_claim": overall_ready,
    }


def build_paired_superiority_manifest_config(
    summary: Mapping[str, Any],
) -> dict[str, Any]:
    """从配对 summary 重建无自由字段的正式 manifest 配置."""

    return {
        "paper_claim_scale": str(summary.get("paper_claim_scale", "")),
        "target_fpr": float(summary.get("target_fpr", math.nan)),
        "bootstrap_resample_count": int(
            summary.get("bootstrap_resample_count", 0)
        ),
        "confidence_level": float(summary.get("confidence_level", math.nan)),
        "bootstrap_analysis_schema": str(
            summary.get("bootstrap_analysis_schema", "")
        ),
        "bootstrap_bit_generator": str(
            summary.get("bootstrap_bit_generator", "")
        ),
        "bootstrap_quantile_method": str(
            summary.get("bootstrap_quantile_method", "")
        ),
        "claim_p_value_method": str(summary.get("claim_p_value_method", "")),
        "sharp_null_diagnostic_method": str(
            summary.get("sharp_null_diagnostic_method", "")
        ),
        "paired_outcome_set_digest": str(
            summary.get("paired_outcome_set_digest", "")
        ),
        "paired_superiority_rows_digest": str(
            summary.get("paired_superiority_rows_digest", "")
        ),
        "paired_superiority_protocol_digest": str(
            summary.get("paired_superiority_protocol_digest", "")
        ),
        "paired_test_prompt_count": int(
            summary.get("paired_test_prompt_count", 0)
        ),
        "paired_test_prompt_id_digest": str(
            summary.get("paired_test_prompt_id_digest", "")
        ),
        "paired_attack_registry_digest": str(
            summary.get("paired_attack_registry_digest", "")
        ),
        "method_threshold_digest_map": dict(
            summary.get("method_threshold_digest_map", {})
        ),
        "method_observation_source_sha256_map": dict(
            summary.get("method_observation_source_sha256_map", {})
        ),
        "method_observation_source_path_map": dict(
            summary.get("method_observation_source_path_map", {})
        ),
        "threshold_audit_rows_digest": str(
            summary.get("threshold_audit_rows_digest", "")
        ),
    }
