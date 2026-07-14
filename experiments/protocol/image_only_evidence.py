"""冻结仅图像盲检的嵌套 calibration 判定协议。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import hashlib
import math
from typing import Any, Iterable, Mapping

from experiments.protocol.detection_key_identity import (
    REGISTERED_WATERMARK_KEY_ROLE,
)
from experiments.protocol.calibration import is_clean_unattacked_negative
from main.core.digest import build_stable_digest
from main.methods.detection import (
    validate_image_only_measurement_digest_record,
)


FROZEN_EVIDENCE_PROTOCOL_SCHEMA = "slm_wm_frozen_evidence_protocol_v3"
CALIBRATION_PARTITION_PROTOCOL = (
    "prompt_id_sha256_nested_rescue_calibration_v1"
)
RESCUE_WINDOW_SELECTION_PROTOCOL = (
    "widest_empirical_fixed_fpr_negative_margin_window_v1"
)


@dataclass(frozen=True)
class FrozenEvidenceProtocol:
    """保存由 clean-negative calibration 独立冻结的唯一判定协议。"""

    frozen_evidence_protocol_schema: str
    calibration_partition_protocol: str
    calibration_partition_digest: str
    calibration_source_negative_count: int
    rescue_window_fit_negative_count: int
    rescue_window_fit_prompt_id_digest: str
    threshold_freeze_negative_count: int
    threshold_freeze_prompt_id_digest: str
    window_fit_allowed_false_positive_count: int
    threshold_freeze_allowed_false_positive_count: int
    rescue_window_fit_content_threshold: float | None
    rescue_window_selection_protocol: str
    rescue_margin_low: float | None
    rescue_window_candidate_count: int
    rescue_window_fit_false_positive_count: int
    content_threshold: float
    geometry_score_threshold: float | None
    registration_confidence_threshold: float | None
    attention_sync_score_threshold: float | None
    attention_anchor_count: int
    attention_residual_threshold: float
    attention_minimum_inlier_ratio: float
    lf_carrier_protocol_digest: str
    tail_carrier_protocol_digest: str
    lf_weight: float
    tail_robust_weight: float
    tail_fraction: float
    image_only_measurement_config_digest: str
    attention_geometry_enabled: bool
    image_alignment_enabled: bool
    geometry_rescue_enabled: bool
    geometry_calibration_negative_count: int
    geometry_calibration_exceedance_count: int
    registration_calibration_negative_count: int
    registration_calibration_exceedance_count: int
    sync_calibration_negative_count: int
    sync_calibration_exceedance_count: int
    geometry_protocol_calibration_ready: bool
    calibration_negative_count: int
    calibration_false_positive_count: int
    calibration_false_positive_rate: float
    target_fpr: float
    threshold_digest: str

    def to_dict(self) -> dict[str, Any]:
        """转为可写入 JSON 的字典。"""

        return asdict(self)


@dataclass(frozen=True)
class EvidenceDecision:
    """保存同一阈值下内容主判和几何救回的可重建原子。"""

    positive_by_content: bool
    calibrated_geometry_reliable: bool
    content_failure_reason: str
    rescue_eligible: bool
    rescue_applied: bool
    evidence_positive: bool


_DIGEST_EXCLUDED_FIELDS = frozenset(
    {"calibration_false_positive_rate", "threshold_digest"}
)


def _sha256_text(value: Any) -> bool:
    """判断值是否为规范小写 SHA-256。"""

    return bool(
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _finite_number(value: Any) -> bool:
    """判断值是否为非 bool 的有限实数。"""

    return bool(
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def allowed_false_positive_count(negative_count: int, target_fpr: float) -> int:
    """按有限样本 fixed-FPR 规则计算允许的假阳性数量。"""

    if type(negative_count) is not int or negative_count <= 0:
        raise ValueError("negative_count 必须为正整数")
    if type(target_fpr) is not float or not 0.0 < target_fpr < 1.0:
        raise ValueError("target_fpr 必须为 (0, 1) 内的精确 float")
    return max(0, math.floor(target_fpr * (negative_count + 1)) - 1)


def _prompt_identity_digest(prompt_ids: Iterable[str]) -> str:
    """对有序 Prompt 身份集合构造稳定摘要。"""

    return build_stable_digest(list(prompt_ids))


def _partition_digest(
    *,
    source_count: int,
    window_fit_count: int,
    window_fit_prompt_id_digest: str,
    threshold_freeze_count: int,
    threshold_freeze_prompt_id_digest: str,
) -> str:
    """构造嵌套 calibration 分区摘要。"""

    return build_stable_digest(
        {
            "calibration_partition_protocol": CALIBRATION_PARTITION_PROTOCOL,
            "calibration_source_negative_count": source_count,
            "rescue_window_fit_negative_count": window_fit_count,
            "rescue_window_fit_prompt_id_digest": (
                window_fit_prompt_id_digest
            ),
            "threshold_freeze_negative_count": threshold_freeze_count,
            "threshold_freeze_prompt_id_digest": (
                threshold_freeze_prompt_id_digest
            ),
        }
    )


def partition_calibration_prompt_ids(
    prompt_ids: Iterable[str],
) -> tuple[tuple[str, ...], tuple[str, ...], str]:
    """按版本化 Prompt 散列构造所有方法共享的1/3与2/3分区。"""

    resolved_prompt_ids = tuple(prompt_ids)
    if len(resolved_prompt_ids) < 3:
        raise ValueError("嵌套 calibration 至少需要3个 clean-negative Prompt")
    if any(
        type(prompt_id) is not str or not prompt_id
        for prompt_id in resolved_prompt_ids
    ):
        raise ValueError("calibration clean negative 缺少非空 prompt_id")
    if len(set(resolved_prompt_ids)) != len(resolved_prompt_ids):
        raise ValueError("calibration clean negative 的 prompt_id 必须唯一")

    def sort_key(prompt_id: str) -> tuple[str, str]:
        payload = (
            CALIBRATION_PARTITION_PROTOCOL + "\0" + prompt_id
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest(), prompt_id

    ordered = tuple(sorted(resolved_prompt_ids, key=sort_key))
    window_fit_count = len(ordered) // 3
    window_fit_ids = ordered[:window_fit_count]
    threshold_freeze_ids = ordered[window_fit_count:]
    if set(window_fit_ids).intersection(threshold_freeze_ids):
        raise RuntimeError("嵌套 calibration 子集发生重叠")
    partition_digest = _partition_digest(
        source_count=len(ordered),
        window_fit_count=len(window_fit_ids),
        window_fit_prompt_id_digest=_prompt_identity_digest(window_fit_ids),
        threshold_freeze_count=len(threshold_freeze_ids),
        threshold_freeze_prompt_id_digest=(
            _prompt_identity_digest(threshold_freeze_ids)
        ),
    )
    return window_fit_ids, threshold_freeze_ids, partition_digest


def partition_calibration_clean_negatives(
    calibration_records: Iterable[dict[str, Any]],
) -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...], str]:
    """按共享 Prompt 分区将 calibration negatives 确定性拆为1/3与2/3。

    排序键只读取 `prompt_id`, 因而输入容器顺序不会改变分区。两个子集互斥,
    前者只拟合 rescue 相关参数, 后者只冻结所有方法的最终 fixed-FPR 阈值。
    """

    records = tuple(calibration_records)
    records_by_prompt_id: dict[str, dict[str, Any]] = {}
    for record in records:
        prompt_id = record.get("prompt_id")
        if type(prompt_id) is not str or not prompt_id:
            raise ValueError("calibration clean negative 缺少非空 prompt_id")
        if prompt_id in records_by_prompt_id:
            raise ValueError("calibration clean negative 的 prompt_id 必须唯一")
        records_by_prompt_id[prompt_id] = record
    window_fit_ids, threshold_freeze_ids, partition_digest = (
        partition_calibration_prompt_ids(records_by_prompt_id)
    )
    window_fit = tuple(
        records_by_prompt_id[prompt_id] for prompt_id in window_fit_ids
    )
    threshold_freeze = tuple(
        records_by_prompt_id[prompt_id] for prompt_id in threshold_freeze_ids
    )
    return window_fit, threshold_freeze, partition_digest


def calibrated_geometry_ready(
    record: Mapping[str, Any],
    *,
    geometry_rescue_enabled: bool,
    geometry_score_threshold: float | None,
    registration_confidence_threshold: float | None,
    attention_sync_score_threshold: float | None,
) -> bool:
    """用结构门和 calibration 阈值判断注意力几何证据是否可用于 rescue。"""

    if type(geometry_rescue_enabled) is not bool:
        raise TypeError("geometry_rescue_enabled 必须为精确 bool")
    if not geometry_rescue_enabled:
        return False
    if not all(
        _finite_number(value)
        for value in (
            geometry_score_threshold,
            registration_confidence_threshold,
            attention_sync_score_threshold,
        )
    ):
        raise ValueError("启用几何 rescue 时必须提供有限 calibration 门")

    alignment = record.get("alignment")
    metadata = record.get("metadata")
    return bool(
        isinstance(alignment, Mapping)
        and alignment.get("geometry_reliable") is True
        and isinstance(metadata, Mapping)
        and metadata.get("stable_pair_weight_identity_ready") is True
        and _finite_number(record.get("attention_geometry_score"))
        and float(record["attention_geometry_score"])
        >= geometry_score_threshold
        and _finite_number(record.get("registration_confidence"))
        and float(record["registration_confidence"])
        >= registration_confidence_threshold
        and _finite_number(record.get("attention_sync_score"))
        and float(record["attention_sync_score"])
        >= attention_sync_score_threshold
    )


def complete_evidence_decision(
    record: Mapping[str, Any],
    *,
    content_threshold: float,
    geometry_rescue_enabled: bool,
    rescue_margin_low: float | None,
    geometry_score_threshold: float | None,
    registration_confidence_threshold: float | None,
    attention_sync_score_threshold: float | None,
) -> EvidenceDecision:
    """用唯一冻结参数重建内容主判和同阈值几何救回。"""

    raw_score = float(record["content_score"])
    raw_margin = raw_score - content_threshold
    positive_by_content = raw_margin >= 0.0
    geometry_reliable = calibrated_geometry_ready(
        record,
        geometry_rescue_enabled=geometry_rescue_enabled,
        geometry_score_threshold=geometry_score_threshold,
        registration_confidence_threshold=registration_confidence_threshold,
        attention_sync_score_threshold=attention_sync_score_threshold,
    )
    aligned_score = record.get("aligned_content_score")
    aligned_score_ready = _finite_number(aligned_score)
    # 使用与等价分数相同的运算顺序, 避免浮点减法在窗口下界产生一位 ULP
    # 的布尔分叉。该比较与实数公式 `delta <= raw-threshold < 0` 等价。
    within_rescue_window = bool(
        geometry_rescue_enabled
        and _finite_number(rescue_margin_low)
        and raw_score < content_threshold
        and raw_score - float(rescue_margin_low) >= content_threshold
    )
    if positive_by_content:
        failure_reason = "content_positive"
    elif within_rescue_window and geometry_reliable:
        failure_reason = "geometry_suspected"
    elif within_rescue_window:
        failure_reason = "low_confidence"
    else:
        failure_reason = "content_evidence_absent"
    rescue_eligible = bool(
        within_rescue_window and geometry_reliable and aligned_score_ready
    )
    rescue_applied = bool(
        rescue_eligible and float(aligned_score) >= content_threshold
    )
    return EvidenceDecision(
        positive_by_content=positive_by_content,
        calibrated_geometry_reliable=geometry_reliable,
        content_failure_reason=failure_reason,
        rescue_eligible=rescue_eligible,
        rescue_applied=rescue_applied,
        evidence_positive=positive_by_content or rescue_applied,
    )


def decision_equivalent_score(
    record: Mapping[str, Any],
    *,
    geometry_rescue_enabled: bool,
    rescue_margin_low: float | None,
    geometry_score_threshold: float | None,
    registration_confidence_threshold: float | None,
    attention_sync_score_threshold: float | None,
) -> float:
    """计算与任意同阈值完整布尔判定严格等价的连续分数。"""

    raw_score = float(record["content_score"])
    aligned_score = record.get("aligned_content_score")
    if not (
        calibrated_geometry_ready(
            record,
            geometry_rescue_enabled=geometry_rescue_enabled,
            geometry_score_threshold=geometry_score_threshold,
            registration_confidence_threshold=(
                registration_confidence_threshold
            ),
            attention_sync_score_threshold=attention_sync_score_threshold,
        )
        and _finite_number(aligned_score)
    ):
        return raw_score
    if not _finite_number(rescue_margin_low):
        raise ValueError("启用几何 rescue 时必须提供有限窗口下界")
    return max(
        raw_score,
        min(float(aligned_score), raw_score - float(rescue_margin_low)),
    )


def frozen_evidence_protocol_digest_payload(
    protocol_record: Any,
) -> dict[str, Any]:
    """从完整冻结协议正文构造唯一阈值摘要 payload。"""

    if isinstance(protocol_record, FrozenEvidenceProtocol):
        resolved = protocol_record.to_dict()
    elif isinstance(protocol_record, Mapping):
        resolved = dict(protocol_record)
    else:
        raise TypeError("冻结 evidence protocol 必须为 dataclass 或 mapping")
    digest_field_names = tuple(
        field.name
        for field in fields(FrozenEvidenceProtocol)
        if field.name not in _DIGEST_EXCLUDED_FIELDS
    )
    missing_fields = tuple(
        field_name
        for field_name in digest_field_names
        if field_name not in resolved
    )
    if missing_fields:
        raise ValueError(
            "冻结 evidence protocol 缺少阈值摘要字段: "
            + ",".join(missing_fields)
        )
    return {
        **{
            field_name: resolved[field_name]
            for field_name in digest_field_names
        },
        "decision_scope": (
            "content_or_same_threshold_aligned_content_rescue"
        ),
    }


def validate_frozen_evidence_protocol_integrity(
    protocol: FrozenEvidenceProtocol,
) -> None:
    """在应用判定前复验嵌套分区、数值、计数和自摘要。"""

    if not isinstance(protocol, FrozenEvidenceProtocol):
        raise TypeError("protocol 必须为 FrozenEvidenceProtocol")
    if (
        protocol.frozen_evidence_protocol_schema
        != FROZEN_EVIDENCE_PROTOCOL_SCHEMA
        or protocol.calibration_partition_protocol
        != CALIBRATION_PARTITION_PROTOCOL
        or protocol.rescue_window_selection_protocol
        != RESCUE_WINDOW_SELECTION_PROTOCOL
    ):
        raise ValueError("冻结 evidence protocol 的版本化算法身份无效")
    if (
        type(protocol.attention_geometry_enabled) is not bool
        or type(protocol.image_alignment_enabled) is not bool
        or type(protocol.geometry_rescue_enabled) is not bool
        or protocol.geometry_rescue_enabled
        is not (
            protocol.attention_geometry_enabled
            and protocol.image_alignment_enabled
        )
        or (
            protocol.image_alignment_enabled
            and not protocol.attention_geometry_enabled
        )
    ):
        raise ValueError("冻结 evidence protocol 的机制开关身份无效")
    for digest in (
        protocol.calibration_partition_digest,
        protocol.rescue_window_fit_prompt_id_digest,
        protocol.threshold_freeze_prompt_id_digest,
        protocol.image_only_measurement_config_digest,
        protocol.lf_carrier_protocol_digest,
        protocol.tail_carrier_protocol_digest,
        protocol.threshold_digest,
    ):
        if not _sha256_text(digest):
            raise ValueError("冻结 evidence protocol 包含无效摘要")
    if (
        protocol.calibration_source_negative_count < 3
        or protocol.rescue_window_fit_negative_count
        != protocol.calibration_source_negative_count // 3
        or protocol.threshold_freeze_negative_count
        != protocol.calibration_source_negative_count
        - protocol.rescue_window_fit_negative_count
        or protocol.calibration_negative_count
        != protocol.threshold_freeze_negative_count
    ):
        raise ValueError("冻结 evidence protocol 的嵌套分区计数不一致")
    expected_partition_digest = _partition_digest(
        source_count=protocol.calibration_source_negative_count,
        window_fit_count=protocol.rescue_window_fit_negative_count,
        window_fit_prompt_id_digest=(
            protocol.rescue_window_fit_prompt_id_digest
        ),
        threshold_freeze_count=protocol.threshold_freeze_negative_count,
        threshold_freeze_prompt_id_digest=(
            protocol.threshold_freeze_prompt_id_digest
        ),
    )
    if protocol.calibration_partition_digest != expected_partition_digest:
        raise ValueError("冻结 evidence protocol 的分区摘要不能重建")
    expected_window_budget = allowed_false_positive_count(
        protocol.rescue_window_fit_negative_count,
        protocol.target_fpr,
    )
    expected_threshold_budget = allowed_false_positive_count(
        protocol.threshold_freeze_negative_count,
        protocol.target_fpr,
    )
    if (
        protocol.window_fit_allowed_false_positive_count
        != expected_window_budget
        or protocol.threshold_freeze_allowed_false_positive_count
        != expected_threshold_budget
        or protocol.rescue_window_fit_false_positive_count
        > expected_window_budget
        or protocol.calibration_false_positive_count
        > expected_threshold_budget
    ):
        raise ValueError("冻结 evidence protocol 的假阳性预算无效")
    common_finite_fields = (
        protocol.content_threshold,
        protocol.attention_residual_threshold,
        protocol.attention_minimum_inlier_ratio,
        protocol.lf_weight,
        protocol.tail_robust_weight,
        protocol.tail_fraction,
        protocol.target_fpr,
    )
    if (
        any(
            type(value) is not float or not math.isfinite(value)
            for value in common_finite_fields
        )
        or not 0.0 < protocol.target_fpr < 1.0
        or not math.isclose(
            protocol.lf_weight + protocol.tail_robust_weight,
            1.0,
            abs_tol=1e-12,
        )
        or not 0.0 < protocol.tail_fraction <= 1.0
    ):
        raise ValueError("冻结 evidence protocol 的连续参数无效")
    rescue_continuous_fields = (
        protocol.rescue_window_fit_content_threshold,
        protocol.rescue_margin_low,
        protocol.geometry_score_threshold,
        protocol.registration_confidence_threshold,
        protocol.attention_sync_score_threshold,
    )
    if protocol.geometry_rescue_enabled:
        if (
            protocol.geometry_protocol_calibration_ready is not True
            or protocol.rescue_window_candidate_count <= 0
            or any(
                type(value) is not float or not math.isfinite(value)
                for value in rescue_continuous_fields
            )
            or float(protocol.rescue_margin_low) >= 0.0
        ):
            raise ValueError("启用的几何 rescue 冻结参数无效")
    elif (
        protocol.geometry_protocol_calibration_ready is not False
        or protocol.rescue_window_candidate_count != 0
        or protocol.rescue_window_fit_false_positive_count != 0
        or any(value is not None for value in rescue_continuous_fields)
    ):
        raise ValueError("禁用的几何 rescue 不得保留窗口或几何门")
    count_fields = (
        protocol.geometry_calibration_negative_count,
        protocol.geometry_calibration_exceedance_count,
        protocol.registration_calibration_negative_count,
        protocol.registration_calibration_exceedance_count,
        protocol.sync_calibration_negative_count,
        protocol.sync_calibration_exceedance_count,
        protocol.calibration_false_positive_count,
    )
    if any(type(value) is not int or value < 0 for value in count_fields):
        raise ValueError("冻结 evidence protocol 的统计计数无效")
    geometry_count_pairs = (
        (
            protocol.geometry_calibration_negative_count,
            protocol.geometry_calibration_exceedance_count,
        ),
        (
            protocol.registration_calibration_negative_count,
            protocol.registration_calibration_exceedance_count,
        ),
        (
            protocol.sync_calibration_negative_count,
            protocol.sync_calibration_exceedance_count,
        ),
    )
    for negative_count, exceedance_count in geometry_count_pairs:
        expected_negative_count = (
            protocol.rescue_window_fit_negative_count
            if protocol.geometry_rescue_enabled
            else 0
        )
        if (
            negative_count != expected_negative_count
            or exceedance_count > negative_count
        ):
            raise ValueError("冻结 evidence protocol 的几何计数不一致")
    expected_rate = (
        protocol.calibration_false_positive_count
        / protocol.calibration_negative_count
    )
    if (
        type(protocol.calibration_false_positive_rate) is not float
        or not math.isclose(
            protocol.calibration_false_positive_rate,
            expected_rate,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise ValueError("冻结 evidence protocol 的假阳性率不能由计数重建")
    expected_digest = build_stable_digest(
        frozen_evidence_protocol_digest_payload(protocol)
    )
    if protocol.threshold_digest != expected_digest:
        raise ValueError("冻结 evidence protocol 的阈值摘要不能由正文重建")


def _validate_calibration_measurements(
    records: tuple[dict[str, Any], ...],
) -> tuple[dict[str, Any], dict[str, Any], bool]:
    """复验 calibration 输入角色并返回共同测量与结构身份。"""

    identities: list[dict[str, Any]] = []
    gates: list[dict[str, Any]] = []
    rescue_flags: list[bool] = []
    for record in records:
        if not is_clean_unattacked_negative(
            record,
            split="calibration",
            expected_detection_key_role=REGISTERED_WATERMARK_KEY_ROLE,
        ):
            raise ValueError(
                "calibrator 只接受 calibration registered-key clean negatives"
            )
        validate_image_only_measurement_digest_record(record)
        metadata = record.get("metadata")
        if not isinstance(metadata, Mapping):
            raise ValueError("calibration 测量记录缺少 metadata")
        attention_enabled = metadata.get("attention_geometry_enabled")
        alignment_enabled = metadata.get("image_alignment_enabled")
        if (
            type(attention_enabled) is not bool
            or type(alignment_enabled) is not bool
            or (alignment_enabled and not attention_enabled)
        ):
            raise ValueError("calibration 测量记录的机制开关身份无效")
        geometry_rescue_enabled = bool(
            attention_enabled and alignment_enabled
        )
        alignment = record.get("alignment")
        if geometry_rescue_enabled and (
            not isinstance(alignment, Mapping)
            or not all(
                _finite_number(record.get(field_name))
                for field_name in (
                    "raw_attention_geometry_score",
                    "attention_geometry_score",
                    "registration_confidence",
                    "attention_sync_score",
                    "aligned_content_score",
                )
            )
        ):
            raise ValueError(
                "calibration source 必须完整测量注意力、配准、同步和 aligned 内容原子"
            )
        if not alignment_enabled and (
            alignment is not None
            or record.get("attention_geometry_score") is not None
            or record.get("registration_confidence") is not None
            or record.get("attention_sync_score") is not None
            or record.get("aligned_content_score") is not None
            or record.get("aligned_lf_score") is not None
            or record.get("aligned_tail_robust_score") is not None
        ):
            raise ValueError("禁用 image alignment 的消融不得保留 rescue 原子")
        digest = record.get("image_only_measurement_config_digest")
        if metadata.get("image_only_measurement_config_digest") != digest:
            raise ValueError("测量记录的配置摘要引用发生分叉")
        identities.append(
            {
                "image_only_measurement_config_digest": digest,
                "lf_carrier_protocol_digest": record.get(
                    "lf_carrier_protocol_digest"
                ),
                "tail_carrier_protocol_digest": record.get(
                    "tail_carrier_protocol_digest"
                ),
                "lf_weight": record.get("lf_weight"),
                "tail_robust_weight": record.get("tail_robust_weight"),
                "tail_fraction": record.get("tail_fraction"),
                "attention_geometry_enabled": attention_enabled,
                "image_alignment_enabled": alignment_enabled,
                "geometry_rescue_enabled": geometry_rescue_enabled,
            }
        )
        rescue_flags.append(geometry_rescue_enabled)
        gate = metadata.get("attention_alignment_gate")
        if not isinstance(gate, Mapping):
            raise ValueError("calibration 测量记录缺少注意力结构门")
        gates.append(dict(gate))
    first_identity = identities[0]
    first_gate = gates[0]
    if any(identity != first_identity for identity in identities[1:]):
        raise ValueError("calibration clean negatives 混用了测量配置或载体身份")
    if any(gate != first_gate for gate in gates[1:]):
        raise ValueError("calibration clean negatives 混用了注意力结构门")
    required_gate_fields = {
        "attention_anchor_count",
        "attention_residual_threshold",
        "attention_minimum_inlier_ratio",
    }
    if set(first_gate) != required_gate_fields:
        raise ValueError("calibration 注意力结构门字段不完整")
    if any(flag is not rescue_flags[0] for flag in rescue_flags[1:]):
        raise ValueError("calibration clean negatives 混用了几何 rescue 开关")
    return first_identity, first_gate, rescue_flags[0]


def _freeze_upper_tail_gate(
    records: tuple[dict[str, Any], ...],
    field_name: str,
    allowed_count: int,
) -> tuple[float, int, int]:
    """从 window-fit negatives 冻结单个上尾几何门。"""

    values = tuple(
        float(record[field_name])
        for record in records
        if _finite_number(record.get(field_name))
    )
    if not values:
        return 0.0, 0, 0
    candidates = tuple(
        sorted({math.nextafter(value, math.inf) for value in values})
    )
    for candidate in candidates:
        exceedance_count = sum(value >= candidate for value in values)
        if exceedance_count <= allowed_count:
            return candidate, len(values), exceedance_count
    raise RuntimeError("几何门禁候选未能满足假阳性预算")


def _freeze_score_threshold(
    scores: Iterable[float],
    allowed_count: int,
) -> tuple[float, int]:
    """选择满足预算的最低 `nextafter(score,+inf)` 阈值。"""

    values = tuple(float(score) for score in scores)
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("阈值冻结要求非空有限分数")
    candidates = tuple(
        sorted({math.nextafter(value, math.inf) for value in values})
    )
    for candidate in candidates:
        exceedance_count = sum(value >= candidate for value in values)
        if exceedance_count <= allowed_count:
            return candidate, exceedance_count
    raise RuntimeError("分数阈值候选未能满足假阳性预算")


def calibrate_complete_evidence_protocol(
    calibration_records: Iterable[dict[str, Any]],
    target_fpr: float,
) -> FrozenEvidenceProtocol:
    """由 clean negatives 独立派生几何门、rescue 窗口和最终阈值。"""

    records = tuple(calibration_records)
    if type(target_fpr) is not float or not 0.0 < target_fpr < 1.0:
        raise ValueError("target_fpr 必须为 (0, 1) 内的精确 float")
    if len(records) < 3:
        raise ValueError("calibration clean negative 记录至少需要3条")
    (
        common_identity,
        attention_gate,
        geometry_rescue_enabled,
    ) = _validate_calibration_measurements(records)
    window_fit, threshold_freeze, partition_digest = (
        partition_calibration_clean_negatives(records)
    )
    window_budget = allowed_false_positive_count(
        len(window_fit), target_fpr
    )
    threshold_budget = allowed_false_positive_count(
        len(threshold_freeze), target_fpr
    )
    geometry_threshold: float | None = None
    registration_threshold: float | None = None
    sync_threshold: float | None = None
    provisional_threshold: float | None = None
    selected_rescue_margin: float | None = None
    ordered_rescue_candidates: tuple[float, ...] = ()
    geometry_count = 0
    geometry_exceedance = 0
    registration_count = 0
    registration_exceedance = 0
    sync_count = 0
    sync_exceedance = 0
    selected_window_false_positives = 0
    if geometry_rescue_enabled:
        (
            geometry_threshold,
            geometry_count,
            geometry_exceedance,
        ) = _freeze_upper_tail_gate(
            window_fit, "attention_geometry_score", window_budget
        )
        (
            registration_threshold,
            registration_count,
            registration_exceedance,
        ) = _freeze_upper_tail_gate(
            window_fit, "registration_confidence", window_budget
        )
        (
            sync_threshold,
            sync_count,
            sync_exceedance,
        ) = _freeze_upper_tail_gate(
            window_fit, "attention_sync_score", window_budget
        )
        provisional_threshold, _ = _freeze_score_threshold(
            (float(record["content_score"]) for record in window_fit),
            window_budget,
        )
        rescue_candidates = {math.nextafter(0.0, -math.inf)}
        for record in window_fit:
            margin = float(record["content_score"]) - provisional_threshold
            if math.isfinite(margin) and margin < 0.0:
                rescue_candidates.add(margin)
                rescue_candidates.add(math.nextafter(margin, 0.0))
        ordered_rescue_candidates = tuple(
            sorted(
                candidate
                for candidate in rescue_candidates
                if math.isfinite(candidate) and candidate < 0.0
            )
        )
        for candidate in ordered_rescue_candidates:
            false_positive_count = sum(
                complete_evidence_decision(
                    record,
                    content_threshold=provisional_threshold,
                    geometry_rescue_enabled=True,
                    rescue_margin_low=candidate,
                    geometry_score_threshold=geometry_threshold,
                    registration_confidence_threshold=registration_threshold,
                    attention_sync_score_threshold=sync_threshold,
                ).evidence_positive
                for record in window_fit
            )
            if false_positive_count <= window_budget:
                selected_rescue_margin = candidate
                selected_window_false_positives = false_positive_count
                break
        if selected_rescue_margin is None:
            raise RuntimeError(
                "rescue 窗口候选未能满足 window-fit 假阳性预算"
            )

    effective_scores = tuple(
        decision_equivalent_score(
            record,
            geometry_rescue_enabled=geometry_rescue_enabled,
            rescue_margin_low=selected_rescue_margin,
            geometry_score_threshold=geometry_threshold,
            registration_confidence_threshold=registration_threshold,
            attention_sync_score_threshold=sync_threshold,
        )
        for record in threshold_freeze
    )
    content_threshold, final_false_positive_count = _freeze_score_threshold(
        effective_scores,
        threshold_budget,
    )
    for record, effective_score in zip(
        threshold_freeze, effective_scores, strict=True
    ):
        decision = complete_evidence_decision(
            record,
            content_threshold=content_threshold,
            geometry_rescue_enabled=geometry_rescue_enabled,
            rescue_margin_low=selected_rescue_margin,
            geometry_score_threshold=geometry_threshold,
            registration_confidence_threshold=registration_threshold,
            attention_sync_score_threshold=sync_threshold,
        )
        if decision.evidence_positive != (
            effective_score >= content_threshold
        ):
            raise RuntimeError("完整 evidence 判定与等价连续分数不一致")

    window_fit_ids = tuple(str(record["prompt_id"]) for record in window_fit)
    threshold_freeze_ids = tuple(
        str(record["prompt_id"]) for record in threshold_freeze
    )
    payload = {
        "frozen_evidence_protocol_schema": FROZEN_EVIDENCE_PROTOCOL_SCHEMA,
        "calibration_partition_protocol": CALIBRATION_PARTITION_PROTOCOL,
        "calibration_partition_digest": partition_digest,
        "calibration_source_negative_count": len(records),
        "rescue_window_fit_negative_count": len(window_fit),
        "rescue_window_fit_prompt_id_digest": (
            _prompt_identity_digest(window_fit_ids)
        ),
        "threshold_freeze_negative_count": len(threshold_freeze),
        "threshold_freeze_prompt_id_digest": (
            _prompt_identity_digest(threshold_freeze_ids)
        ),
        "window_fit_allowed_false_positive_count": window_budget,
        "threshold_freeze_allowed_false_positive_count": threshold_budget,
        "rescue_window_fit_content_threshold": provisional_threshold,
        "rescue_window_selection_protocol": RESCUE_WINDOW_SELECTION_PROTOCOL,
        "rescue_margin_low": selected_rescue_margin,
        "rescue_window_candidate_count": len(ordered_rescue_candidates),
        "rescue_window_fit_false_positive_count": (
            selected_window_false_positives
        ),
        "content_threshold": content_threshold,
        "geometry_score_threshold": geometry_threshold,
        "registration_confidence_threshold": registration_threshold,
        "attention_sync_score_threshold": sync_threshold,
        "attention_anchor_count": int(attention_gate["attention_anchor_count"]),
        "attention_residual_threshold": float(
            attention_gate["attention_residual_threshold"]
        ),
        "attention_minimum_inlier_ratio": float(
            attention_gate["attention_minimum_inlier_ratio"]
        ),
        **common_identity,
        "geometry_calibration_negative_count": geometry_count,
        "geometry_calibration_exceedance_count": geometry_exceedance,
        "registration_calibration_negative_count": registration_count,
        "registration_calibration_exceedance_count": registration_exceedance,
        "sync_calibration_negative_count": sync_count,
        "sync_calibration_exceedance_count": sync_exceedance,
        "geometry_protocol_calibration_ready": geometry_rescue_enabled,
        "calibration_negative_count": len(threshold_freeze),
        "calibration_false_positive_count": final_false_positive_count,
        "target_fpr": target_fpr,
    }
    protocol = FrozenEvidenceProtocol(
        **payload,
        calibration_false_positive_rate=(
            final_false_positive_count / len(threshold_freeze)
        ),
        threshold_digest=build_stable_digest(
            frozen_evidence_protocol_digest_payload(payload)
        ),
    )
    validate_frozen_evidence_protocol_integrity(protocol)
    return protocol


def apply_frozen_evidence_protocol(
    records: Iterable[dict[str, Any]],
    protocol: FrozenEvidenceProtocol,
) -> tuple[dict[str, Any], ...]:
    """首次且唯一地对原始测量记录物化 calibration 判定字段。"""

    validate_frozen_evidence_protocol_integrity(protocol)
    resolved: list[dict[str, Any]] = []
    for record in records:
        validate_image_only_measurement_digest_record(record)
        metadata = record.get("metadata")
        if (
            record.get("image_only_measurement_config_digest")
            != protocol.image_only_measurement_config_digest
            or not isinstance(metadata, Mapping)
            or metadata.get("image_only_measurement_config_digest")
            != protocol.image_only_measurement_config_digest
            or record.get("lf_carrier_protocol_digest")
            != protocol.lf_carrier_protocol_digest
            or record.get("tail_carrier_protocol_digest")
            != protocol.tail_carrier_protocol_digest
            or record.get("lf_weight") != protocol.lf_weight
            or record.get("tail_robust_weight")
            != protocol.tail_robust_weight
            or record.get("tail_fraction") != protocol.tail_fraction
            or metadata.get("attention_geometry_enabled")
            is not protocol.attention_geometry_enabled
            or metadata.get("image_alignment_enabled")
            is not protocol.image_alignment_enabled
        ):
            raise ValueError("测量记录与冻结协议的配置或载体身份不一致")
        gate = metadata.get("attention_alignment_gate")
        expected_gate = {
            "attention_anchor_count": protocol.attention_anchor_count,
            "attention_residual_threshold": (
                protocol.attention_residual_threshold
            ),
            "attention_minimum_inlier_ratio": (
                protocol.attention_minimum_inlier_ratio
            ),
        }
        if gate != expected_gate:
            raise ValueError("测量记录与冻结协议的注意力结构门不一致")
        decision = complete_evidence_decision(
            record,
            content_threshold=protocol.content_threshold,
            geometry_rescue_enabled=protocol.geometry_rescue_enabled,
            rescue_margin_low=protocol.rescue_margin_low,
            geometry_score_threshold=protocol.geometry_score_threshold,
            registration_confidence_threshold=(
                protocol.registration_confidence_threshold
            ),
            attention_sync_score_threshold=(
                protocol.attention_sync_score_threshold
            ),
        )
        raw_margin = float(record["content_score"]) - protocol.content_threshold
        aligned_score = record.get("aligned_content_score")
        resolved.append(
            {
                **record,
                "frozen_content_threshold": protocol.content_threshold,
                "frozen_rescue_margin_low": protocol.rescue_margin_low,
                "frozen_geometry_score_threshold": (
                    protocol.geometry_score_threshold
                ),
                "frozen_registration_confidence_threshold": (
                    protocol.registration_confidence_threshold
                ),
                "frozen_attention_sync_score_threshold": (
                    protocol.attention_sync_score_threshold
                ),
                "frozen_threshold_digest": protocol.threshold_digest,
                "frozen_image_only_measurement_config_digest": (
                    protocol.image_only_measurement_config_digest
                ),
                "frozen_attention_geometry_enabled": (
                    protocol.attention_geometry_enabled
                ),
                "frozen_image_alignment_enabled": (
                    protocol.image_alignment_enabled
                ),
                "frozen_geometry_rescue_enabled": (
                    protocol.geometry_rescue_enabled
                ),
                "formal_raw_content_margin": raw_margin,
                "formal_aligned_content_margin": (
                    None
                    if aligned_score is None
                    else float(aligned_score) - protocol.content_threshold
                ),
                "formal_positive_by_content": (
                    decision.positive_by_content
                ),
                "formal_geometry_reliable": (
                    decision.calibrated_geometry_reliable
                ),
                "formal_content_failure_reason": (
                    decision.content_failure_reason
                ),
                "formal_rescue_eligible": decision.rescue_eligible,
                "formal_rescue_applied": decision.rescue_applied,
                "formal_evidence_positive": decision.evidence_positive,
                "formal_metric_status": "measured_image_only_detection",
                "supports_paper_claim": False,
            }
        )
    return tuple(resolved)


__all__ = [
    "CALIBRATION_PARTITION_PROTOCOL",
    "EvidenceDecision",
    "FROZEN_EVIDENCE_PROTOCOL_SCHEMA",
    "FrozenEvidenceProtocol",
    "RESCUE_WINDOW_SELECTION_PROTOCOL",
    "allowed_false_positive_count",
    "apply_frozen_evidence_protocol",
    "calibrate_complete_evidence_protocol",
    "calibrated_geometry_ready",
    "complete_evidence_decision",
    "decision_equivalent_score",
    "frozen_evidence_protocol_digest_payload",
    "partition_calibration_clean_negatives",
    "partition_calibration_prompt_ids",
    "validate_frozen_evidence_protocol_integrity",
]
