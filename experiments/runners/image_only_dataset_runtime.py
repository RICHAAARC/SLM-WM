"""在 70/700/7000 Prompt 协议上运行真实方法并冻结完整检测判定。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Iterable
from zipfile import ZIP_STORED, ZipFile

from experiments.protocol.paper_run_config import (
    PaperRunConfig,
    build_paper_run_config,
    normalize_paper_run_name,
)
from experiments.protocol.prompts import build_prompt_records, read_prompt_file
from experiments.protocol.prompt_sources import (
    PROMPT_CONFIG_NAMES,
    PROMPT_SELECTION_MANIFEST_PATH,
    PROMPT_SOURCE_REGISTRY_PATH,
    audit_packaged_prompt_set_bytes,
)
from experiments.protocol.method_runtime_config import (
    FORMAL_METHOD_PACKAGE_ROOT,
    load_formal_method_runtime_config,
)
from experiments.protocol.formal_randomization import (
    formal_runtime_randomization_plan_record,
    validate_formal_prompt_randomization_identity,
)
from experiments.protocol.splits import apply_split_assignments, build_group_split_counts
from experiments.protocol.attacks import default_attack_configs
from experiments.runtime import repository_environment
from experiments.runtime.scientific_execution_binding import (
    validate_scientific_execution_binding,
)
from experiments.runtime.scientific_content_binding import (
    SCIENTIFIC_CONTENT_BINDING_SCHEMA,
    recompute_scientific_content_binding_digest,
)
from experiments.runtime.package_input_manifest import (
    collect_exact_package_entries,
    validate_exact_package_archive,
    write_exact_package_input_manifest,
)
from experiments.runtime.resume_checkpoint import (
    clear_progress_checkpoints,
    persist_progress_checkpoint,
    restore_role_checkpoints,
)
from experiments.runtime.scientific_unit_provenance import (
    SCIENTIFIC_UNIT_PROVENANCE_AGGREGATE_FIELDS,
    aggregate_scientific_unit_provenance,
)
from experiments.runtime.diffusion.regeneration_attacks import default_diffusion_attack_specs
from experiments.runtime.diffusion.semantic_features import (
    JOINT_FEATURE_WIDTH,
    SEMANTIC_FEATURE_SCHEMA,
    SEMANTIC_FEATURE_WIDTH,
    HANDCRAFTED_STRUCTURE_FEATURE_SCHEMA,
    HANDCRAFTED_STRUCTURE_FEATURE_WIDTH,
)
from experiments.runners.semantic_watermark_runtime import (
    SemanticWatermarkRuntimeConfig,
    _carrier_only_counterfactual_artifact_binding_ready,
    _scientific_content_binding_artifact_ready,
    load_completed_semantic_watermark_runtime_result,
    load_semantic_watermark_runtime_context,
    semantic_watermark_runtime_config_payload,
    validate_semantic_watermark_runtime_result_provenance,
    write_semantic_watermark_runtime_outputs,
)
from experiments.runtime.repository_environment import file_digest
from experiments.runtime.archive_naming import utc_archive_token
from experiments.artifacts.artifact_manifest import build_artifact_manifest
from experiments.artifacts.detection_score_curves import (
    build_detection_score_tables,
    write_detection_score_tables,
)
from experiments.artifacts.image_only_detection_metrics import (
    build_image_only_test_metric_rows,
)
from main.methods.carrier import (
    keyed_prg_protocol_record,
    tail_robust_carrier_protocol_record,
    validate_low_frequency_carrier_protocol_record,
)
from main.core.digest import (
    TENSOR_CONTENT_DIGEST_VERSION,
    build_stable_digest,
)
from main.methods.geometry import (
    ATTENTION_ALIGNMENT_ANCHOR_COUNT,
    ATTENTION_ALIGNMENT_MINIMUM_INLIER_RATIO,
    ATTENTION_ALIGNMENT_RESIDUAL_THRESHOLD,
    ATTENTION_COORDINATE_CONVENTION,
    ATTENTION_GRID_ALIGN_CORNERS,
    ATTENTION_RELATION_COMPONENT_NAMES,
    DIRECT_QK_RELATION_SOURCE,
    attention_relation_component_protocol,
    attention_alignment_gate_record,
    qk_atomic_evaluation_records_ready,
    qk_operator_metadata_records_digest,
    qk_operator_metadata_records_ready,
)
from main.methods.detection import validate_image_only_detection_digest_record
from main.methods.update_composition import (
    QUANTIZED_COMPOSITION_EVIDENCE_VERSION,
    recompute_quantized_composition_evidence_digest,
)
from main.methods.subspace import (
    JACOBIAN_NULL_SPACE_EVIDENCE_VERSION,
    recompute_jacobian_null_space_result_digest,
)


PACKAGE_INPUT_MANIFEST_FILE_NAME = "image_only_dataset_package_input_manifest.json"
PROMPT_SOURCE_SNAPSHOT_DIRECTORY_NAME = "prompt_source_snapshot"
_FORMAL_ATTENTION_ALIGNMENT_GATE = attention_alignment_gate_record(
    ATTENTION_ALIGNMENT_ANCHOR_COUNT,
    ATTENTION_ALIGNMENT_RESIDUAL_THRESHOLD,
    ATTENTION_ALIGNMENT_MINIMUM_INLIER_RATIO,
)
_FORMAL_METHOD_CONFIG = load_formal_method_runtime_config(
    FORMAL_METHOD_PACKAGE_ROOT
)


def validate_formal_dataset_randomization_identity(
    config: SemanticWatermarkRuntimeConfig,
    paper_run: PaperRunConfig,
    *,
    prompt_index: int,
) -> dict[str, Any]:
    """校验正式主方法或消融样本是否使用当前 repeat 的冻结身份公式."""

    declared_repeat = {
        "randomization_repeat_id": paper_run.randomization_repeat_id,
        "generation_seed_index": paper_run.generation_seed_index,
        "generation_seed_offset": paper_run.generation_seed_offset,
        "watermark_key_index": paper_run.watermark_key_index,
        "formal_randomization_protocol_digest": (
            paper_run.formal_randomization_protocol_digest
        ),
    }
    actual_repeat = {
        field_name: getattr(config, field_name)
        for field_name in declared_repeat
    }
    if actual_repeat != declared_repeat:
        raise ValueError("正式运行配置未匹配当前 PaperRun repeat 身份")
    return validate_formal_prompt_randomization_identity(
        base_generation_seed_random=_FORMAL_METHOD_CONFIG.seed,
        prompt_index=prompt_index,
        randomization_repeat_id=config.randomization_repeat_id,
        generation_seed_index=config.generation_seed_index,
        generation_seed_offset=config.generation_seed_offset,
        watermark_key_index=config.watermark_key_index,
        generation_seed_random=config.seed,
        watermark_key_seed_random=config.watermark_key_seed_random,
        key_material=config.key_material,
        formal_randomization_protocol_digest=(
            config.formal_randomization_protocol_digest
        ),
    )


_FORMAL_LF_CARRIER_PROTOCOL = (
    _FORMAL_METHOD_CONFIG.low_frequency_carrier_config.to_record()
)
_FORMAL_CONTENT_WEIGHT_PROTOCOLS = {
    (
        _FORMAL_METHOD_CONFIG.lf_detection_score_weight,
        _FORMAL_METHOD_CONFIG.tail_robust_detection_score_weight,
    ),
    (1.0, 0.0),
    (0.0, 1.0),
}
_FORMAL_TAIL_FRACTIONS = {_FORMAL_METHOD_CONFIG.tail_fraction, 1.0}
_FORMAL_TAIL_CARRIER_PROTOCOLS = {
    tail_fraction: tail_robust_carrier_protocol_record(
        tail_fraction,
        prg_version=_FORMAL_METHOD_CONFIG.keyed_prg_version,
    )
    for tail_fraction in _FORMAL_TAIL_FRACTIONS
}


def _formal_attention_alignment_gate_fields_ready(
    record: Any,
) -> bool:
    """判断记录是否逐字段绑定唯一正式注意力结构门禁."""

    return bool(isinstance(record, dict) and all(
        type(record.get(field_name)) is type(value)
        and record.get(field_name) == value
        for field_name, value in _FORMAL_ATTENTION_ALIGNMENT_GATE.items()
    ))


def _formal_attention_alignment_gate_record_ready(
    record: Any,
) -> bool:
    """判断记录是否同时保存规范门禁对象和三个平坦字段."""

    if not isinstance(record, dict):
        return False
    nested_gate = record.get("attention_alignment_gate")
    return bool(
        isinstance(nested_gate, dict)
        and set(nested_gate) == set(_FORMAL_ATTENTION_ALIGNMENT_GATE)
        and _formal_attention_alignment_gate_fields_ready(nested_gate)
        and _formal_attention_alignment_gate_fields_ready(record)
    )


def formal_low_frequency_carrier_protocol_record() -> dict[str, Any]:
    """返回由唯一方法 YAML 构造的正式 LF 载体协议记录."""

    return dict(_FORMAL_LF_CARRIER_PROTOCOL)


def _formal_content_carrier_fields_ready(record: Any) -> bool:
    """判断完整方法记录是否绑定正式内容载体协议和检测权重."""

    expected_tail_protocol = _FORMAL_TAIL_CARRIER_PROTOCOLS[
        _FORMAL_METHOD_CONFIG.tail_fraction
    ]
    return bool(
        isinstance(record, dict)
        and record.get("lf_carrier_protocol_digest")
        == _FORMAL_LF_CARRIER_PROTOCOL["lf_carrier_protocol_digest"]
        and type(record.get("lf_weight")) is float
        and record.get("lf_weight")
        == _FORMAL_METHOD_CONFIG.lf_detection_score_weight
        and type(record.get("tail_robust_weight")) is float
        and record.get("tail_robust_weight")
        == _FORMAL_METHOD_CONFIG.tail_robust_detection_score_weight
        and type(record.get("tail_fraction")) is float
        and record.get("tail_fraction") == _FORMAL_METHOD_CONFIG.tail_fraction
        and record.get("tail_carrier_protocol_digest")
        == expected_tail_protocol["tail_carrier_protocol_digest"]
    )


def validate_detection_content_carrier_protocol(
    record: dict[str, Any],
) -> dict[str, Any]:
    """复验样本级内容分数及其正式协议摘要引用."""

    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("检测记录缺少 metadata")
    validate_image_only_detection_digest_record(record)
    detector_config_digest = record.get(
        "image_only_detector_config_digest"
    )
    if (
        not isinstance(detector_config_digest, str)
        or metadata.get("image_only_detector_config_digest")
        != detector_config_digest
    ):
        raise ValueError("检测记录的检测器配置摘要引用发生分叉")
    lf_weight = record.get("lf_weight")
    tail_weight = record.get("tail_robust_weight")
    tail_fraction = record.get("tail_fraction")
    expected_tail_protocol = _FORMAL_TAIL_CARRIER_PROTOCOLS.get(
        tail_fraction
    )
    if (
        record.get("lf_carrier_protocol_digest")
        != _FORMAL_LF_CARRIER_PROTOCOL["lf_carrier_protocol_digest"]
        or type(lf_weight) is not float
        or type(tail_weight) is not float
        or type(tail_fraction) is not float
        or not math.isclose(lf_weight + tail_weight, 1.0, abs_tol=1e-12)
        or (lf_weight, tail_weight) not in _FORMAL_CONTENT_WEIGHT_PROTOCOLS
        or tail_fraction not in _FORMAL_TAIL_FRACTIONS
        or expected_tail_protocol is None
        or record.get("tail_carrier_protocol_digest")
        != expected_tail_protocol["tail_carrier_protocol_digest"]
    ):
        raise ValueError("检测记录与正式内容载体摘要或权重不一致")
    return {
        "lf_carrier_protocol_digest": record[
            "lf_carrier_protocol_digest"
        ],
        "lf_weight": lf_weight,
        "tail_robust_weight": tail_weight,
        "tail_fraction": tail_fraction,
        "tail_carrier_protocol_digest": record[
            "tail_carrier_protocol_digest"
        ],
        "image_only_detector_config_digest": detector_config_digest,
    }


def _prompt_source_snapshot_paths(
    output_dir: Path,
    paper_run_name: str,
) -> tuple[Path, Path, Path]:
    """返回当前运行层级三份自包含 Prompt 来源快照路径."""

    snapshot_dir = output_dir / PROMPT_SOURCE_SNAPSHOT_DIRECTORY_NAME
    return (
        snapshot_dir / PROMPT_CONFIG_NAMES[paper_run_name],
        snapshot_dir / PROMPT_SELECTION_MANIFEST_PATH.name,
        snapshot_dir / PROMPT_SOURCE_REGISTRY_PATH.name,
    )


def _write_prompt_source_snapshot(
    *,
    root_path: Path,
    output_dir: Path,
    paper_run_name: str,
    prompt_path: Path,
) -> tuple[tuple[Path, Path, Path], dict[str, Any]]:
    """逐字节复制并复验本次 GPU 运行实际消费的 Prompt 来源."""

    snapshot_paths = _prompt_source_snapshot_paths(
        output_dir,
        paper_run_name,
    )
    source_paths = (
        prompt_path,
        root_path / PROMPT_SELECTION_MANIFEST_PATH,
        root_path / PROMPT_SOURCE_REGISTRY_PATH,
    )
    snapshot_paths[0].parent.mkdir(parents=True, exist_ok=True)
    for source_path, snapshot_path in zip(
        source_paths,
        snapshot_paths,
        strict=True,
    ):
        if not source_path.is_file() or source_path.is_symlink():
            raise FileNotFoundError(
                f"Prompt 来源快照缺少普通源文件: {source_path}"
            )
        payload = source_path.read_bytes()
        temporary_path = snapshot_path.with_name(
            snapshot_path.name + ".partial"
        )
        temporary_path.write_bytes(payload)
        temporary_path.replace(snapshot_path)
        if snapshot_path.read_bytes() != payload:
            raise RuntimeError("Prompt 来源快照写后字节复验失败")
    report = audit_packaged_prompt_set_bytes(
        prompt_set=paper_run_name,
        prompt_file_payload=snapshot_paths[0].read_bytes(),
        selection_manifest_payload=snapshot_paths[1].read_bytes(),
        source_registry_payload=snapshot_paths[2].read_bytes(),
    )
    return snapshot_paths, report


def _audit_prompt_source_snapshot(
    *,
    output_dir: Path,
    paper_run_name: str,
) -> tuple[tuple[Path, Path, Path], dict[str, Any]]:
    """在打包前重新读取并审计三份自包含 Prompt 来源快照."""

    snapshot_paths = _prompt_source_snapshot_paths(
        output_dir,
        paper_run_name,
    )
    if any(
        not path.is_file() or path.is_symlink()
        for path in snapshot_paths
    ):
        raise FileNotFoundError("主方法结果缺少自包含 Prompt 来源快照")
    report = audit_packaged_prompt_set_bytes(
        prompt_set=paper_run_name,
        prompt_file_payload=snapshot_paths[0].read_bytes(),
        selection_manifest_payload=snapshot_paths[1].read_bytes(),
        source_registry_payload=snapshot_paths[2].read_bytes(),
    )
    return snapshot_paths, report


@dataclass(frozen=True)
class FrozenEvidenceProtocol:
    """保存 calibration split 冻结的完整 evidence 判定参数。"""

    content_threshold: float
    rescue_margin_low: float
    geometry_score_threshold: float
    registration_confidence_threshold: float
    attention_sync_score_threshold: float
    attention_anchor_count: int
    attention_residual_threshold: float
    attention_minimum_inlier_ratio: float
    lf_carrier_protocol_digest: str
    tail_carrier_protocol_digest: str
    lf_weight: float
    tail_robust_weight: float
    tail_fraction: float
    image_only_detector_config_digest: str
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


_FROZEN_EVIDENCE_DIGEST_EXCLUDED_FIELDS = frozenset(
    {"calibration_false_positive_rate", "threshold_digest"}
)


def frozen_evidence_protocol_digest_payload(
    protocol_record: Any,
) -> dict[str, Any]:
    """从完整冻结协议正文构造唯一阈值摘要 payload."""

    if isinstance(protocol_record, FrozenEvidenceProtocol):
        resolved = protocol_record.to_dict()
    elif isinstance(protocol_record, dict):
        resolved = dict(protocol_record)
    else:
        raise TypeError("冻结 evidence protocol 必须为 dataclass 或 dict")
    digest_field_names = tuple(
        field.name
        for field in fields(FrozenEvidenceProtocol)
        if field.name not in _FROZEN_EVIDENCE_DIGEST_EXCLUDED_FIELDS
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
    """在应用阈值前复验冻结协议的数值、计数和自摘要完整性."""

    if not isinstance(protocol, FrozenEvidenceProtocol):
        raise TypeError("protocol 必须为 FrozenEvidenceProtocol")
    if (
        type(protocol.image_only_detector_config_digest) is not str
        or len(protocol.image_only_detector_config_digest) != 64
        or any(
            character not in "0123456789abcdef"
            for character in protocol.image_only_detector_config_digest
        )
    ):
        raise ValueError("冻结 evidence protocol 的检测器配置摘要无效")
    finite_fields = (
        protocol.content_threshold,
        protocol.rescue_margin_low,
        protocol.geometry_score_threshold,
        protocol.registration_confidence_threshold,
        protocol.attention_sync_score_threshold,
        protocol.target_fpr,
    )
    count_fields = (
        protocol.geometry_calibration_negative_count,
        protocol.geometry_calibration_exceedance_count,
        protocol.registration_calibration_negative_count,
        protocol.registration_calibration_exceedance_count,
        protocol.sync_calibration_negative_count,
        protocol.sync_calibration_exceedance_count,
        protocol.calibration_negative_count,
        protocol.calibration_false_positive_count,
    )
    if (
        any(
            type(value) is not float or not math.isfinite(value)
            for value in finite_fields
        )
        or protocol.rescue_margin_low >= 0.0
        or not 0.0 < protocol.target_fpr < 1.0
        or any(type(value) is not int or value < 0 for value in count_fields)
        or protocol.calibration_negative_count <= 0
        or protocol.calibration_false_positive_count
        > protocol.calibration_negative_count
        or type(protocol.geometry_protocol_calibration_ready) is not bool
    ):
        raise ValueError("冻结 evidence protocol 的数值或计数语义无效")
    for negative_count, exceedance_count in (
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
    ):
        if (
            negative_count > protocol.calibration_negative_count
            or exceedance_count > negative_count
        ):
            raise ValueError("冻结 evidence protocol 的几何计数不一致")
    expected_false_positive_rate = (
        protocol.calibration_false_positive_count
        / protocol.calibration_negative_count
    )
    if (
        type(protocol.calibration_false_positive_rate) is not float
        or not math.isfinite(protocol.calibration_false_positive_rate)
        or not math.isclose(
            protocol.calibration_false_positive_rate,
            expected_false_positive_rate,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise ValueError("冻结 evidence protocol 的校准假阳性率与计数不一致")
    expected_threshold_digest = build_stable_digest(
        frozen_evidence_protocol_digest_payload(protocol)
    )
    if protocol.threshold_digest != expected_threshold_digest:
        raise ValueError("冻结 evidence protocol 的阈值摘要不能由正文重建")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """读取 JSONL 记录。"""

    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def validate_detection_attention_alignment_gate(
    record: dict[str, Any],
) -> dict[str, int | float]:
    """从检测及对齐记录重建唯一结构门禁, 拒绝缺失或身份分叉."""

    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("检测记录缺少 attention alignment metadata")
    raw_gate = metadata.get("attention_alignment_gate")
    if not isinstance(raw_gate, dict) or set(raw_gate) != set(
        _FORMAL_ATTENTION_ALIGNMENT_GATE
    ):
        raise ValueError("检测记录缺少完整 attention_alignment_gate")
    try:
        gate = attention_alignment_gate_record(
            raw_gate["attention_anchor_count"],
            raw_gate["attention_residual_threshold"],
            raw_gate["attention_minimum_inlier_ratio"],
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("检测记录 attention_alignment_gate 无效") from exc
    if any(
        type(metadata.get(field_name)) is not type(value)
        or metadata.get(field_name) != value
        for field_name, value in gate.items()
    ):
        raise ValueError("检测 metadata 与 attention_alignment_gate 不一致")
    alignment = record.get("alignment")
    if isinstance(alignment, dict):
        if any(
            type(alignment.get(field_name)) is not type(value)
            or alignment.get(field_name) != value
            for field_name, value in gate.items()
        ):
            raise ValueError("alignment 与检测结构门禁不一致")
        alignment_metadata = alignment.get("metadata")
        raw_alignment_gate = (
            alignment_metadata.get("attention_alignment_gate")
            if isinstance(alignment_metadata, dict)
            else None
        )
        if (
            not isinstance(raw_alignment_gate, dict)
            or set(raw_alignment_gate) != set(gate)
            or any(
                type(raw_alignment_gate.get(field_name)) is not type(value)
                or raw_alignment_gate.get(field_name) != value
                for field_name, value in gate.items()
            )
        ):
            raise ValueError("alignment metadata 未绑定结构门禁")
    if gate != _FORMAL_ATTENTION_ALIGNMENT_GATE:
        raise ValueError("检测记录未使用预注册注意力结构门禁")
    return gate


def _decision(
    record: dict[str, Any],
    threshold: float,
    rescue_margin_low: float,
    geometry_score_threshold: float,
    registration_confidence_threshold: float,
    attention_sync_score_threshold: float,
) -> tuple[bool, bool, bool, str]:
    """用冻结阈值重算内容主判和同阈值几何救回。"""

    raw_score = float(record["content_score"])
    raw_margin = raw_score - threshold
    positive_by_content = raw_margin >= 0.0
    aligned_score = record.get("aligned_content_score")
    alignment = record.get("alignment")
    if isinstance(alignment, dict):
        alignment_reliable = alignment.get("geometry_reliable") is True
    else:
        alignment_reliable = False
    metadata = record.get("metadata")
    stable_pair_identity_ready = bool(
        isinstance(metadata, dict)
        and metadata.get("stable_pair_weight_identity_ready") is True
    )
    geometry_score = record.get("attention_geometry_score")
    registration_confidence = record.get("registration_confidence")
    attention_sync_score = record.get("attention_sync_score")
    geometry_reliable = (
        alignment_reliable
        and stable_pair_identity_ready
        and isinstance(geometry_score, (int, float))
        and math.isfinite(float(geometry_score))
        and float(geometry_score) >= geometry_score_threshold
        and isinstance(registration_confidence, (int, float))
        and math.isfinite(float(registration_confidence))
        and float(registration_confidence) >= registration_confidence_threshold
        and isinstance(attention_sync_score, (int, float))
        and math.isfinite(float(attention_sync_score))
        and float(attention_sync_score) >= attention_sync_score_threshold
    )
    if positive_by_content:
        failure_reason = "content_positive"
    elif rescue_margin_low <= raw_margin < 0.0 and geometry_reliable:
        failure_reason = "geometry_suspected"
    elif rescue_margin_low <= raw_margin < 0.0:
        failure_reason = "low_confidence"
    else:
        failure_reason = "content_evidence_absent"
    rescue_eligible = (
        rescue_margin_low <= raw_margin < 0.0
        and geometry_reliable
        and aligned_score is not None
        and failure_reason in {"geometry_suspected", "low_confidence"}
    )
    rescue_applied = rescue_eligible and float(aligned_score) - threshold >= 0.0
    return positive_by_content, rescue_applied, positive_by_content or rescue_applied, failure_reason


def calibrate_complete_evidence_protocol(
    calibration_records: Iterable[dict[str, Any]],
    target_fpr: float,
    rescue_margin_low: float,
) -> FrozenEvidenceProtocol:
    """在 clean negative 上冻结包含 rescue 的完整判定协议。

    阈值搜索直接调用最终布尔决策, 因而不会把“内容阈值达到目标 FPR”错误地
    等同于“加入几何救回后仍达到目标 FPR”。
    """

    records = tuple(calibration_records)
    if not records:
        raise ValueError("calibration clean negative 记录不得为空")
    if not 0.0 < target_fpr < 1.0:
        raise ValueError("target_fpr 必须位于 (0, 1)")
    if (
        isinstance(rescue_margin_low, bool)
        or not isinstance(rescue_margin_low, (int, float))
        or not math.isfinite(rescue_margin_low)
        or rescue_margin_low >= 0.0
    ):
        raise ValueError("rescue_margin_low 必须为负有限数")
    content_carrier_identities = []
    for record in records:
        validate_detection_attention_alignment_gate(record)
        content_carrier_identities.append(
            validate_detection_content_carrier_protocol(record)
        )
    first_content_carrier_identity = content_carrier_identities[0]
    first_detector_config_digest = first_content_carrier_identity[
        "image_only_detector_config_digest"
    ]
    if any(
        identity["image_only_detector_config_digest"]
        != first_detector_config_digest
        for identity in content_carrier_identities[1:]
    ):
        raise ValueError(
            "calibration clean negatives 混用了检测器配置身份"
        )
    if any(
        any(
            identity[field_name]
            != first_content_carrier_identity[field_name]
            for field_name in (
                "lf_carrier_protocol_digest",
                "lf_weight",
                "tail_robust_weight",
                "tail_fraction",
                "tail_carrier_protocol_digest",
            )
        )
        for identity in content_carrier_identities[1:]
    ):
        raise ValueError("calibration clean negatives 混用了内容载体协议")
    if any(
        record.get("metadata", {}).get("rescue_margin_low")
        != float(rescue_margin_low)
        for record in records
    ):
        raise ValueError("calibration rescue margin 与检测器配置身份不一致")
    allowed_false_positives = max(0, math.floor(target_fpr * (len(records) + 1)) - 1)
    def freeze_geometry_gate(field_name: str) -> tuple[float, int, int]:
        """从全部未删失 clean negatives 冻结单个几何门禁。"""

        values = tuple(
            float(record[field_name])
            for record in records
            if isinstance(record.get(field_name), (int, float))
            and math.isfinite(float(record[field_name]))
        )
        if not values:
            return 0.0, 0, 0
        candidates = sorted({math.nextafter(value, math.inf) for value in values})
        selected = candidates[-1]
        selected_count = sum(value >= selected for value in values)
        for candidate in candidates:
            exceedance_count = sum(value >= candidate for value in values)
            if exceedance_count <= allowed_false_positives:
                selected = candidate
                selected_count = exceedance_count
                break
        return selected, len(values), selected_count

    (
        geometry_score_threshold,
        geometry_negative_count,
        geometry_exceedance_count,
    ) = freeze_geometry_gate("attention_geometry_score")
    (
        registration_confidence_threshold,
        registration_negative_count,
        registration_exceedance_count,
    ) = freeze_geometry_gate("registration_confidence")
    (
        attention_sync_score_threshold,
        sync_negative_count,
        sync_exceedance_count,
    ) = freeze_geometry_gate("attention_sync_score")
    geometry_protocol_calibration_ready = (
        geometry_negative_count == len(records)
        and registration_negative_count == len(records)
        and sync_negative_count == len(records)
        and all(
            isinstance(record.get("alignment"), dict)
            and type(record["alignment"].get("geometry_reliable")) is bool
            for record in records
        )
    )
    score_candidates = []
    for record in records:
        score_candidates.append(float(record["content_score"]))
        if record.get("aligned_content_score") is not None:
            score_candidates.append(float(record["aligned_content_score"]))
    thresholds = sorted({math.nextafter(value, math.inf) for value in score_candidates})
    selected_threshold = thresholds[-1]
    selected_false_positives = 0
    for threshold in thresholds:
        false_positives = sum(
            _decision(
                record,
                threshold,
                rescue_margin_low,
                geometry_score_threshold,
                registration_confidence_threshold,
                attention_sync_score_threshold,
            )[2]
            for record in records
        )
        if false_positives <= allowed_false_positives:
            selected_threshold = threshold
            selected_false_positives = false_positives
            break
    payload = {
        "content_threshold": selected_threshold,
        "rescue_margin_low": rescue_margin_low,
        "geometry_score_threshold": geometry_score_threshold,
        "registration_confidence_threshold": registration_confidence_threshold,
        "attention_sync_score_threshold": attention_sync_score_threshold,
        **_FORMAL_ATTENTION_ALIGNMENT_GATE,
        "lf_carrier_protocol_digest": _FORMAL_LF_CARRIER_PROTOCOL[
            "lf_carrier_protocol_digest"
        ],
        "tail_carrier_protocol_digest": (
            first_content_carrier_identity[
                "tail_carrier_protocol_digest"
            ]
        ),
        "lf_weight": first_content_carrier_identity["lf_weight"],
        "tail_robust_weight": first_content_carrier_identity[
            "tail_robust_weight"
        ],
        "tail_fraction": first_content_carrier_identity["tail_fraction"],
        "image_only_detector_config_digest": (
            first_detector_config_digest
        ),
        "geometry_calibration_negative_count": geometry_negative_count,
        "geometry_calibration_exceedance_count": geometry_exceedance_count,
        "registration_calibration_negative_count": registration_negative_count,
        "registration_calibration_exceedance_count": registration_exceedance_count,
        "sync_calibration_negative_count": sync_negative_count,
        "sync_calibration_exceedance_count": sync_exceedance_count,
        "geometry_protocol_calibration_ready": geometry_protocol_calibration_ready,
        "calibration_negative_count": len(records),
        "calibration_false_positive_count": selected_false_positives,
        "target_fpr": target_fpr,
        "decision_scope": "content_or_same_threshold_aligned_content_rescue",
    }
    return FrozenEvidenceProtocol(
        content_threshold=selected_threshold,
        rescue_margin_low=rescue_margin_low,
        geometry_score_threshold=geometry_score_threshold,
        registration_confidence_threshold=registration_confidence_threshold,
        attention_sync_score_threshold=attention_sync_score_threshold,
        attention_anchor_count=int(
            _FORMAL_ATTENTION_ALIGNMENT_GATE["attention_anchor_count"]
        ),
        attention_residual_threshold=float(
            _FORMAL_ATTENTION_ALIGNMENT_GATE[
                "attention_residual_threshold"
            ]
        ),
        attention_minimum_inlier_ratio=float(
            _FORMAL_ATTENTION_ALIGNMENT_GATE[
                "attention_minimum_inlier_ratio"
            ]
        ),
        lf_carrier_protocol_digest=_FORMAL_LF_CARRIER_PROTOCOL[
            "lf_carrier_protocol_digest"
        ],
        tail_carrier_protocol_digest=first_content_carrier_identity[
            "tail_carrier_protocol_digest"
        ],
        lf_weight=first_content_carrier_identity["lf_weight"],
        tail_robust_weight=first_content_carrier_identity[
            "tail_robust_weight"
        ],
        tail_fraction=first_content_carrier_identity["tail_fraction"],
        image_only_detector_config_digest=first_detector_config_digest,
        geometry_calibration_negative_count=geometry_negative_count,
        geometry_calibration_exceedance_count=geometry_exceedance_count,
        registration_calibration_negative_count=registration_negative_count,
        registration_calibration_exceedance_count=registration_exceedance_count,
        sync_calibration_negative_count=sync_negative_count,
        sync_calibration_exceedance_count=sync_exceedance_count,
        geometry_protocol_calibration_ready=geometry_protocol_calibration_ready,
        calibration_negative_count=len(records),
        calibration_false_positive_count=selected_false_positives,
        calibration_false_positive_rate=selected_false_positives / len(records),
        target_fpr=target_fpr,
        threshold_digest=build_stable_digest(
            frozen_evidence_protocol_digest_payload(payload)
        ),
    )


def apply_frozen_evidence_protocol(
    records: Iterable[dict[str, Any]],
    protocol: FrozenEvidenceProtocol,
) -> tuple[dict[str, Any], ...]:
    """对全部 split 和攻击记录应用同一冻结协议。"""

    validate_frozen_evidence_protocol_integrity(protocol)
    protocol_alignment_gate = attention_alignment_gate_record(
        protocol.attention_anchor_count,
        protocol.attention_residual_threshold,
        protocol.attention_minimum_inlier_ratio,
    )
    if protocol_alignment_gate != _FORMAL_ATTENTION_ALIGNMENT_GATE:
        raise ValueError("冻结 evidence protocol 的注意力结构门禁发生漂移")
    if protocol.lf_carrier_protocol_digest != _FORMAL_LF_CARRIER_PROTOCOL[
        "lf_carrier_protocol_digest"
    ]:
        raise ValueError("冻结 evidence protocol 的 LF 载体协议发生漂移")
    expected_tail_protocol = _FORMAL_TAIL_CARRIER_PROTOCOLS.get(
        protocol.tail_fraction
    )
    if (
        type(protocol.lf_weight) is not float
        or type(protocol.tail_robust_weight) is not float
        or not 0.0 <= protocol.lf_weight <= 1.0
        or not 0.0 <= protocol.tail_robust_weight <= 1.0
        or not math.isclose(
            protocol.lf_weight + protocol.tail_robust_weight,
            1.0,
            abs_tol=1e-12,
        )
        or (protocol.lf_weight, protocol.tail_robust_weight)
        not in _FORMAL_CONTENT_WEIGHT_PROTOCOLS
        or type(protocol.tail_fraction) is not float
        or protocol.tail_fraction not in _FORMAL_TAIL_FRACTIONS
        or expected_tail_protocol is None
        or protocol.tail_carrier_protocol_digest
        != expected_tail_protocol["tail_carrier_protocol_digest"]
    ):
        raise ValueError("冻结 evidence protocol 的内容检测权重发生漂移")
    resolved = []
    for record in records:
        if validate_detection_attention_alignment_gate(record) != (
            protocol_alignment_gate
        ):
            raise ValueError("检测记录与冻结注意力结构门禁不一致")
        content_carrier_identity = (
            validate_detection_content_carrier_protocol(record)
        )
        if (
            content_carrier_identity["lf_carrier_protocol_digest"]
            != protocol.lf_carrier_protocol_digest
            or content_carrier_identity["lf_weight"] != protocol.lf_weight
            or content_carrier_identity["tail_robust_weight"]
            != protocol.tail_robust_weight
            or content_carrier_identity["tail_fraction"]
            != protocol.tail_fraction
            or content_carrier_identity["tail_carrier_protocol_digest"]
            != protocol.tail_carrier_protocol_digest
            or content_carrier_identity[
                "image_only_detector_config_digest"
            ]
            != protocol.image_only_detector_config_digest
        ):
            raise ValueError("检测记录与冻结载体或检测器配置身份不一致")
        positive_by_content, rescue_applied, evidence_positive, failure_reason = _decision(
            record,
            protocol.content_threshold,
            protocol.rescue_margin_low,
            protocol.geometry_score_threshold,
            protocol.registration_confidence_threshold,
            protocol.attention_sync_score_threshold,
        )
        raw_margin = float(record["content_score"]) - protocol.content_threshold
        aligned_score = record.get("aligned_content_score")
        resolved.append(
            {
                **record,
                "frozen_content_threshold": protocol.content_threshold,
                "frozen_geometry_score_threshold": protocol.geometry_score_threshold,
                "frozen_registration_confidence_threshold": (
                    protocol.registration_confidence_threshold
                ),
                "frozen_attention_sync_score_threshold": (
                    protocol.attention_sync_score_threshold
                ),
                "frozen_threshold_digest": protocol.threshold_digest,
                "frozen_image_only_detector_config_digest": (
                    protocol.image_only_detector_config_digest
                ),
                "formal_raw_content_margin": raw_margin,
                "formal_aligned_content_margin": (
                    None if aligned_score is None else float(aligned_score) - protocol.content_threshold
                ),
                "formal_positive_by_content": positive_by_content,
                "formal_content_failure_reason": failure_reason,
                "formal_rescue_applied": rescue_applied,
                "formal_evidence_positive": evidence_positive,
                "formal_metric_status": "measured_image_only_detection",
                "supports_paper_claim": False,
            }
        )
    return tuple(resolved)


def _write_csv(path: Path, rows: tuple[dict[str, Any], ...]) -> None:
    """写出列集合稳定的 CSV。"""

    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _sha256_ready(value: Any) -> bool:
    """判断字段是否为规范小写 SHA-256, 不接受对象字符串化。"""

    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(
            character in "0123456789abcdef" for character in value
        )
    )


def _scientific_content_binding_record_ready(
    result: dict[str, Any],
) -> bool:
    """复验单 Prompt 内嵌总记录的 schema、run id 与自摘要。"""

    metadata = result.get("metadata")
    if not isinstance(metadata, dict):
        return False
    record = metadata.get("scientific_content_binding_record")
    supplied_digest = metadata.get("scientific_content_binding_digest")
    if (
        not isinstance(record, dict)
        or metadata.get("scientific_content_binding_schema")
        != SCIENTIFIC_CONTENT_BINDING_SCHEMA
        or record.get("run_id") != result.get("run_id")
        or record.get("scientific_content_binding_digest")
        != supplied_digest
    ):
        return False
    try:
        return bool(
            recompute_scientific_content_binding_digest(record)
            == supplied_digest
        )
    except (TypeError, ValueError):
        return False


def _scientific_update_record_ready(
    record: dict[str, Any],
    config: SemanticWatermarkRuntimeConfig,
) -> bool:
    """验证单个注入记录确实执行全部关键科学算子。"""

    def finite_at_least(value: Any, minimum: float, *, strict: bool = False) -> bool:
        """集中校验一个受治理数值是否有限并满足下界。"""

        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            return False
        return float(value) > minimum if strict else float(value) >= minimum

    def sha256_ready(value: Any) -> bool:
        """判断一个科学内容字段是否为规范 SHA-256."""

        return _sha256_ready(value)

    expected_lf_protocol_digest = (
        config.low_frequency_carrier_config.protocol_digest
    )
    tail_carrier_protocol = tail_robust_carrier_protocol_record(
        config.tail_fraction if config.tail_truncation_enabled else 1.0,
        prg_version=config.keyed_prg_version,
    )
    lf_template_shape = record.get("lf_template_shape")
    tail_template_shape = record.get("tail_template_shape")
    tail_element_count = record.get("tail_template_element_count")
    tail_selected_count = record.get("tail_selected_element_count")
    expected_tail_fraction = (
        config.tail_fraction if config.tail_truncation_enabled else 1.0
    )
    if (
        record.get("lf_carrier_protocol_digest")
        != expected_lf_protocol_digest
        or record.get("tail_carrier_protocol_digest")
        != tail_carrier_protocol["tail_carrier_protocol_digest"]
        or not sha256_ready(record.get("lf_template_content_sha256"))
        or not sha256_ready(record.get("lf_template_digest"))
        or not sha256_ready(record.get("tail_template_content_sha256"))
        or not sha256_ready(record.get("tail_template_digest"))
        or not isinstance(lf_template_shape, list)
        or not isinstance(tail_template_shape, list)
        or lf_template_shape != tail_template_shape
        or len(tail_template_shape) != 4
        or any(
            type(value) is not int or value <= 0
            for value in tail_template_shape
        )
        or type(tail_element_count) is not int
        or tail_element_count != math.prod(tail_template_shape)
        or type(tail_selected_count) is not int
        or tail_selected_count
        != math.ceil(tail_element_count * expected_tail_fraction)
        or not finite_at_least(
            record.get("lf_projection_energy_retention"),
            config.minimum_projection_energy_retention,
        )
        or not finite_at_least(
            record.get("tail_projection_energy_retention"),
            config.minimum_projection_energy_retention,
        )
    ):
        return False

    adjacent_reference_sha256 = str(
        record.get("adjacent_step_reference_latent_content_sha256", "")
    )
    step_index = record.get("step_index")
    if (
        record.get("tensor_content_digest_version")
        != TENSOR_CONTENT_DIGEST_VERSION
        or len(adjacent_reference_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in adjacent_reference_sha256
        )
        or not isinstance(step_index, int)
        or record.get("adjacent_step_reference_index") != step_index - 1
        or record.get("adjacent_step_stability_status")
        != "measured_from_immediately_previous_scheduler_step"
    ):
        return False
    if not all(
        sha256_ready(record.get(field_name))
        for field_name in (
            "current_decoded_rgb_content_sha256",
            "previous_step_decoded_rgb_content_sha256",
            "clip_patch_tokens_content_sha256",
            "clip_cls_token_content_sha256",
            "semantic_risk_signal_content_sha256",
            "texture_risk_signal_content_sha256",
            "local_contrast_risk_signal_content_sha256",
            "adjacent_step_stability_signal_content_sha256",
            "attention_stability_signal_content_sha256",
        )
    ):
        return False
    risk_signal_content_records = {
        field_name: record.get(field_name)
        for field_name in (
            "current_decoded_rgb_content_sha256",
            "previous_step_decoded_rgb_content_sha256",
            "clip_patch_tokens_content_sha256",
            "clip_cls_token_content_sha256",
            "semantic_risk_signal_content_sha256",
            "texture_risk_signal_content_sha256",
            "local_contrast_risk_signal_content_sha256",
            "adjacent_step_stability_signal_content_sha256",
            "attention_stability_signal_content_sha256",
        )
    }

    null_space_records = record.get("null_space_records")
    if not isinstance(null_space_records, dict) or set(null_space_records) != {
        "lf_content",
        "tail_robust",
        "attention_geometry",
    }:
        return False
    allowed_jvp_modes = {
        "torch_func_exact_jvp_vjp",
        "torch_autograd_exact_jvp_vjp_reexecution",
    }
    expected_prg_digest = keyed_prg_protocol_record(
        config.keyed_prg_version
    )["keyed_prg_protocol_digest"]
    quantized_write_shape_for_subspace = record.get(
        "quantized_write_update_shape"
    )
    if (
        not isinstance(quantized_write_shape_for_subspace, list)
        or not quantized_write_shape_for_subspace
        or not all(
            isinstance(value, int)
            and not isinstance(value, bool)
            and value > 0
            for value in quantized_write_shape_for_subspace
        )
    ):
        return False
    latent_element_count = math.prod(
        quantized_write_shape_for_subspace
    )
    for expected_branch_name, subspace_record in null_space_records.items():
        metadata = subspace_record.get("metadata", {})
        numeric_values = (
            subspace_record.get("response_residual"),
            subspace_record.get("relative_response_residual"),
            subspace_record.get("orthogonality_error"),
        )
        if not all(
            finite_at_least(value, 0.0)
            for value in numeric_values
        ):
            return False
        if (
            subspace_record.get("branch_name") != expected_branch_name
            or subspace_record.get("null_space_evidence_version")
            != JACOBIAN_NULL_SPACE_EVIDENCE_VERSION
            or not sha256_ready(subspace_record.get("solver_digest"))
        ):
            return False
        try:
            recomputed_solver_digest = (
                recompute_jacobian_null_space_result_digest(
                    subspace_record
                )
            )
        except (TypeError, ValueError, OverflowError):
            return False
        if recomputed_solver_digest != subspace_record.get("solver_digest"):
            return False
        if float(subspace_record["relative_response_residual"]) > config.maximum_relative_response_residual:
            return False
        if (
            float(subspace_record["orthogonality_error"])
            > config.maximum_orthogonality_error
        ):
            return False
        if metadata.get("jvp_mode") not in allowed_jvp_modes:
            return False
        if metadata.get("solver") != "matrix_free_full_jacobian_psd_cg":
            return False
        if subspace_record.get("cg_converged") is not True:
            return False
        if int(metadata.get("preferred_direction_count", 0)) < 1:
            return False
        if metadata.get("semantic_feature_schema") != SEMANTIC_FEATURE_SCHEMA:
            return False
        if (
            metadata.get("handcrafted_structure_feature_schema")
            != HANDCRAFTED_STRUCTURE_FEATURE_SCHEMA
        ):
            return False
        if int(metadata.get("semantic_feature_width", 0)) != SEMANTIC_FEATURE_WIDTH:
            return False
        if (
            int(metadata.get("handcrafted_structure_feature_width", 0))
            != HANDCRAFTED_STRUCTURE_FEATURE_WIDTH
        ):
            return False
        if int(metadata.get("joint_feature_width", 0)) != JOINT_FEATURE_WIDTH:
            return False
        if metadata.get("feature_compression_applied") is not False:
            return False
        if metadata.get("tensor_content_digest_version") != (
            TENSOR_CONTENT_DIGEST_VERSION
        ):
            return False
        qr_condition_number = metadata.get("qr_condition_number")
        if (
            metadata.get("null_space_numerical_epsilon")
            != config.null_space_numerical_epsilon
            or metadata.get("maximum_qr_condition_number")
            != config.maximum_qr_condition_number
            or metadata.get("maximum_orthogonality_error")
            != config.maximum_orthogonality_error
            or metadata.get("qr_reference_solve_protocol")
            != config.qr_reference_solve_protocol
            or metadata.get("risk_budget_operator")
            != "explicit_diagonal_B"
            or metadata.get("full_feature_jvp") is not True
            or metadata.get("full_feature_vjp") is not True
            or metadata.get("cg_damping") != 0.0
            or metadata.get("cg_maximum_iterations")
            != config.null_space_cg_max_iterations
            or metadata.get("cg_relative_tolerance")
            != config.null_space_cg_relative_tolerance
            or metadata.get("maximum_relative_response_residual")
            != config.maximum_relative_response_residual
            or metadata.get("minimum_projection_energy_retention")
            != config.minimum_projection_energy_retention
            or metadata.get("latent_basis_formula")
            != "qr(Bd - B^2 J^T solve_psd_cg(J B^2 J^T, J B d))"
            or not finite_at_least(qr_condition_number, 0.0, strict=True)
            or float(qr_condition_number)
            > config.maximum_qr_condition_number
        ):
            return False
        if not all(
            sha256_ready(subspace_record.get(field_name))
            for field_name in (
                "candidate_matrix_content_sha256",
                "risk_budget_content_sha256",
                "routed_candidate_response_matrix_content_sha256",
                "projected_direction_matrix_content_sha256",
                "projected_direction_response_matrix_content_sha256",
                "latent_basis_content_sha256",
                "basis_response_matrix_content_sha256",
                "basis_reference_response_matrix_content_sha256",
            )
        ):
            return False
        if (
            metadata.get("keyed_prg_version") != config.keyed_prg_version
            or metadata.get("keyed_prg_protocol_digest")
            != expected_prg_digest
        ):
            return False
        column_residuals = subspace_record.get(
            "column_relative_response_residuals"
        )
        column_response_norms = subspace_record.get(
            "column_response_norms"
        )
        column_reference_norms = subspace_record.get(
            "column_reference_response_norms"
        )
        energy_retentions = subspace_record.get("projection_energy_retentions")
        cg_residuals = subspace_record.get("cg_relative_residuals")
        cg_iteration_counts = subspace_record.get("cg_iteration_counts")
        evaluated_indices = subspace_record.get(
            "evaluated_direction_indices"
        )
        candidate_shape = subspace_record.get("candidate_shape")
        response_shape = subspace_record.get("response_shape")
        if (
            subspace_record.get("basis_rank") != config.null_rank
            or subspace_record.get("null_rank") != config.null_rank
            or not isinstance(candidate_shape, list)
            or len(candidate_shape) != 2
            or not all(
                isinstance(value, int) and not isinstance(value, bool) and value > 0
                for value in candidate_shape
            )
            or subspace_record.get("candidate_count") != candidate_shape[1]
            or candidate_shape[1] != config.candidate_count
            or candidate_shape[0] != latent_element_count
            or not isinstance(response_shape, list)
            or len(response_shape) != 2
            or not all(
                isinstance(value, int) and not isinstance(value, bool) and value > 0
                for value in response_shape
            )
            or response_shape[1] != config.null_rank
            or response_shape[0] != JOINT_FEATURE_WIDTH
            or subspace_record.get("response_width") != response_shape[0]
            or not isinstance(evaluated_indices, list)
            or len(evaluated_indices) != config.null_rank
            or not all(
                isinstance(value, int)
                and not isinstance(value, bool)
                and 0 <= value < candidate_shape[1]
                for value in evaluated_indices
            )
            or evaluated_indices != sorted(set(evaluated_indices))
            or subspace_record.get("evaluated_direction_count")
            != max(evaluated_indices) + 1
        ):
            return False
        if not all(
            isinstance(values, list)
            and len(values) == config.null_rank
            and all(finite_at_least(value, 0.0) for value in values)
            for values in (
                column_response_norms,
                column_residuals,
                column_reference_norms,
                energy_retentions,
                cg_residuals,
            )
        ):
            return False
        if (
            not isinstance(cg_iteration_counts, list)
            or len(cg_iteration_counts) != config.null_rank
            or not all(
                isinstance(value, int)
                and not isinstance(value, bool)
                and 0 <= value <= config.null_space_cg_max_iterations
                for value in cg_iteration_counts
            )
        ):
            return False
        if any(
            float(value) > config.maximum_relative_response_residual
            for value in column_residuals
        ):
            return False
        if any(
            float(value) < config.minimum_projection_energy_retention
            or float(value) > 1.0 + config.maximum_orthogonality_error
            for value in energy_retentions
        ):
            return False
        if any(
            float(value) > config.null_space_cg_relative_tolerance
            for value in cg_residuals
        ):
            return False
    quantized_update_sha256 = str(
        record.get("quantized_write_update_content_sha256", "")
    )
    quantized_relative_response = record.get(
        "quantized_write_relative_jacobian_response"
    )
    quantized_update_norm = record.get("quantized_write_update_norm")
    quantized_response_norm = record.get(
        "quantized_write_jacobian_response_norm"
    )
    quantized_reference_feature_norm = record.get(
        "quantized_write_reference_feature_norm"
    )
    if (
        len(quantized_update_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in quantized_update_sha256
        )
        or record.get("quantized_write_jacobian_gate_applicable") is not True
        or record.get("quantized_write_jacobian_gate_ready") is not True
        or record.get("quantized_write_jacobian_status")
        != "measured_from_actual_quantized_latent_delta"
        or not isinstance(quantized_relative_response, (int, float))
        or not math.isfinite(float(quantized_relative_response))
        or float(quantized_relative_response)
        > config.maximum_quantized_write_relative_jacobian_response
        or not isinstance(quantized_update_norm, (int, float))
        or not math.isfinite(float(quantized_update_norm))
        or float(quantized_update_norm) <= 0.0
        or not isinstance(quantized_response_norm, (int, float))
        or not math.isfinite(float(quantized_response_norm))
        or float(quantized_response_norm) < 0.0
        or not isinstance(
            quantized_reference_feature_norm,
            (int, float),
        )
        or not math.isfinite(float(quantized_reference_feature_norm))
        or float(quantized_reference_feature_norm)
        <= config.null_space_numerical_epsilon
        or not sha256_ready(
            record.get(
                "quantized_write_reference_feature_content_sha256"
            )
        )
        or not sha256_ready(
            record.get(
                "quantized_write_jacobian_response_content_sha256"
            )
        )
        or not math.isclose(
            float(quantized_relative_response),
            float(quantized_response_norm)
            / float(quantized_reference_feature_norm),
            rel_tol=1e-9,
            abs_tol=1e-12,
        )
        or record.get(
            "maximum_quantized_write_relative_jacobian_response"
        )
        != config.maximum_quantized_write_relative_jacobian_response
        or record.get("keyed_prg_version") != config.keyed_prg_version
        or record.get("keyed_prg_protocol_digest") != expected_prg_digest
    ):
        return False
    quantized_envelope_ratio = record.get(
        "quantized_write_maximum_envelope_ratio"
    )
    quantized_common_scale = record.get("quantized_write_common_scale")
    quantized_backtracking_factor = record.get(
        "quantized_write_backtracking_factor"
    )
    quantized_backtracking_steps = record.get(
        "quantized_write_backtracking_step_count"
    )
    quantized_write_shape = record.get("quantized_write_update_shape")
    expected_active_branches = [
        branch_name
        for branch_name, enabled in (
            ("lf_content", config.lf_enabled),
            ("tail_robust", config.tail_robust_enabled),
            ("attention_geometry", config.attention_geometry_enabled),
        )
        if enabled
    ]
    try:
        recomputed_composition_digest = (
            recompute_quantized_composition_evidence_digest(record)
        )
    except (TypeError, ValueError):
        return False
    if (
        not sha256_ready(record.get("combined_update_content_sha256"))
        or not sha256_ready(
            record.get("combined_budget_envelope_content_sha256")
        )
        or record.get("quantized_write_composition_order")
        != list(config.quantized_branch_composition_order)
        or record.get("quantized_write_budget_envelope_ready") is not True
        or not isinstance(quantized_envelope_ratio, (int, float))
        or not math.isfinite(float(quantized_envelope_ratio))
        or float(quantized_envelope_ratio) < 0.0
        or float(quantized_envelope_ratio) > 1.0
        or not isinstance(quantized_common_scale, (int, float))
        or not math.isfinite(float(quantized_common_scale))
        or not 0.0 < float(quantized_common_scale) <= 1.0
        or quantized_backtracking_factor
        != config.quantized_budget_envelope_backtracking_factor
        or not isinstance(quantized_backtracking_steps, int)
        or not 0
        <= quantized_backtracking_steps
        <= config.quantized_budget_envelope_backtracking_maximum_steps
        or float(quantized_common_scale)
        != float(quantized_backtracking_factor)
        ** quantized_backtracking_steps
        or record.get("quantized_composition_evidence_version")
        != QUANTIZED_COMPOSITION_EVIDENCE_VERSION
        or record.get("quantized_composition_evidence_digest")
        != recomputed_composition_digest
        or record.get("quantized_write_original_latent_content_sha256")
        != record.get("latent_content_sha256_before")
        or record.get("quantized_write_candidate_latent_content_sha256")
        != record.get("latent_content_sha256_after")
        or record.get("quantized_write_update_dtype")
        != f"torch.{config.latent_torch_dtype}"
        or not isinstance(quantized_write_shape, list)
        or len(quantized_write_shape) != 4
        or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 0
            for value in quantized_write_shape
        )
        or record.get("quantized_write_active_branch_order")
        != expected_active_branches
    ):
        return False
    branch_update_content_records = {
        "lf_content": record.get("lf_update_content_sha256"),
        "tail_robust": record.get("tail_robust_update_content_sha256"),
        "attention_geometry": record.get(
            "attention_geometry_update_content_sha256"
        ),
    }
    if (
        not all(
            sha256_ready(value)
            for value in branch_update_content_records.values()
        )
        or record.get("branch_updates_content_digest")
        != build_stable_digest(branch_update_content_records)
    ):
        return False
    if not finite_at_least(
        record.get("lf_projection_energy_retention"),
        config.minimum_projection_energy_retention,
    ):
        return False
    if not finite_at_least(
        record.get("tail_projection_energy_retention"),
        config.minimum_projection_energy_retention,
    ):
        return False
    attention_score_before = record.get("attention_score_before")
    attention_content_base_score = record.get("attention_content_base_score")
    attention_score_after = record.get("attention_score_after")
    attention_actual_content_base_score = record.get(
        "attention_actual_written_content_base_score"
    )
    attention_final_score = record.get("attention_final_combined_score")
    attention_score_gain = record.get("attention_score_gain")
    attention_scores = (
        attention_score_before,
        attention_content_base_score,
        attention_score_after,
        attention_actual_content_base_score,
        attention_final_score,
        attention_score_gain,
    )
    if not all(
        isinstance(value, (int, float)) and math.isfinite(float(value))
        for value in attention_scores
    ):
        return False
    if (
        float(attention_score_after)
        <= max(
            float(attention_score_before),
            float(attention_content_base_score),
        )
        or float(attention_final_score)
        <= max(
            float(attention_score_before),
            float(attention_actual_content_base_score),
        )
        or not math.isclose(
            float(attention_score_gain),
            float(attention_final_score) - float(attention_score_before),
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
    ):
        return False
    if not finite_at_least(record.get("attention_applied_update_strength"), 0.0, strict=True):
        return False
    stable_token_indices = record.get("stable_token_indices")
    if (
        not isinstance(stable_token_indices, list)
        or len(stable_token_indices) < 4
        or len(set(stable_token_indices)) != len(stable_token_indices)
    ):
        return False
    if (
        record.get("attention_qk_atomic_content_ready") is not True
        or not qk_atomic_evaluation_records_ready(
            record.get("attention_qk_atomic_content_records"),
            record.get("attention_qk_atomic_content_digest"),
            aggregate_field_name="attention_qk_atomic_content_records",
            expected_roles=(
                "latent_before",
                "optimization_content_base_latent",
                "accepted_attention_candidate",
                "actual_written_content_base_latent",
                "actual_written_combined_latent",
            ),
            expected_layer_names=config.attention_module_names,
            require_evaluation_identity=True,
        )
    ):
        return False
    qk_evaluation_by_role = {
        str(item["qk_evaluation_role"]): item
        for item in record["attention_qk_atomic_content_records"]
    }
    qk_role_score_fields = {
        "latent_before": "attention_score_before",
        "optimization_content_base_latent": (
            "attention_content_base_score"
        ),
        "accepted_attention_candidate": "attention_score_after",
        "actual_written_content_base_latent": (
            "attention_actual_written_content_base_score"
        ),
        "actual_written_combined_latent": (
            "attention_final_combined_score"
        ),
    }
    if any(
        not math.isclose(
            float(qk_evaluation_by_role[role]["evaluation_score"]),
            float(record[field_name]),
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
        for role, field_name in qk_role_score_fields.items()
    ):
        return False
    stable_token_selection_digest = str(
        record.get("stable_token_selection_digest", "")
    )
    if len(stable_token_selection_digest) != 64 or any(
        character not in "0123456789abcdef"
        for character in stable_token_selection_digest
    ):
        return False
    for digest_field in (
        "stable_pair_weight_identity_digest",
        "stable_pair_weight_realization_digest",
        "attention_relation_component_identity_digest",
        "attention_relation_keyed_projection_digest",
        "attention_relation_component_protocol_digest",
    ):
        digest = str(record.get(digest_field, ""))
        if len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            return False
    component_protocol = attention_relation_component_protocol(
        config.attention_relation_component_weights
    )
    if (
        record.get("attention_relation_component_names")
        != list(ATTENTION_RELATION_COMPONENT_NAMES)
        or record.get("attention_relation_active_component_names")
        != list(
            component_protocol[
                "attention_relation_active_component_names"
            ]
        )
        or record.get("attention_relation_component_weights")
        != list(config.attention_relation_component_weights)
        or record.get("attention_relation_component_protocol_digest")
        != component_protocol[
            "attention_relation_component_protocol_digest"
        ]
        or record.get("attention_relation_source")
        != DIRECT_QK_RELATION_SOURCE
        or record.get("attention_relation_direct_qk_source_ready") is not True
        or record.get("attention_relation_probability_scope")
        != "sampled_image_token_qk_relation_probability"
    ):
        return False
    qk_operator_records = record.get(
        "attention_relation_qk_operator_metadata_records"
    )
    if (
        record.get("attention_module_names")
        != list(config.attention_module_names)
        or record.get("attention_coordinate_convention")
        != ATTENTION_COORDINATE_CONVENTION
        or record.get("attention_grid_align_corners")
        is not ATTENTION_GRID_ALIGN_CORNERS
        or record.get("attention_relation_qk_operator_metadata_ready")
        is not True
        or not isinstance(qk_operator_records, list)
        or not qk_operator_metadata_records_ready(
            qk_operator_records,
            config.attention_module_names,
        )
        or record.get("attention_relation_qk_operator_metadata_digest")
        != qk_operator_metadata_records_digest(
            qk_operator_records
        )
    ):
        return False
    semantic_cosine = record.get("full_semantic_cosine_similarity")
    structure_relative_drift = record.get(
        "full_handcrafted_structure_feature_relative_drift"
    )
    if not finite_at_least(
        semantic_cosine,
        config.minimum_semantic_preservation_cosine,
    ):
        return False
    if (
        not isinstance(structure_relative_drift, (int, float))
        or not math.isfinite(float(structure_relative_drift))
        or float(structure_relative_drift)
        > config.maximum_handcrafted_structure_feature_relative_drift
    ):
        return False
    if record.get("semantic_preservation_gate_ready") is not True:
        return False
    branch_risk_records = record.get("branch_risk_records")
    if (
        record.get("active_carrier_branches") != expected_active_branches
        or not isinstance(branch_risk_records, dict)
        or set(branch_risk_records)
        != {"lf_content", "tail_robust", "attention_geometry"}
    ):
        return False
    risk_configs = {
        "lf_content": config.lf_content_risk_config,
        "tail_robust": config.tail_robust_risk_config,
        "attention_geometry": config.attention_geometry_risk_config,
    }
    update_digest_fields = {
        "lf_content": "lf_update_content_sha256",
        "tail_robust": "tail_robust_update_content_sha256",
        "attention_geometry": "attention_geometry_update_content_sha256",
    }
    composition_branch_identities = record.get(
        "quantized_write_branch_content_identities"
    )
    if (
        not isinstance(composition_branch_identities, dict)
        or set(composition_branch_identities) != set(expected_active_branches)
    ):
        return False
    branch_risk_content_records: dict[str, dict[str, Any]] = {}
    for branch_name, branch_record_value in branch_risk_records.items():
        if not isinstance(branch_record_value, dict):
            return False
        branch_record = branch_record_value
        if (
            branch_record.get("branch_name") != branch_name
            or not sha256_ready(branch_record.get("risk_field_digest"))
        ):
            return False
        base_content = {
            field_name: branch_record.get(field_name)
            for field_name in (
                "risk_values_content_sha256",
                "budget_values_content_sha256",
                "eligible_mask_content_sha256",
            )
        }
        if not all(sha256_ready(value) for value in base_content.values()):
            return False
        if branch_name in expected_active_branches:
            bounded_content = {
                field_name: branch_record.get(field_name)
                for field_name in (
                    "effective_budget_values_content_sha256",
                    "branch_unit_direction_content_sha256",
                    "branch_budget_envelope_content_sha256",
                    "branch_written_update_content_sha256",
                    "branch_post_risk_direction_content_sha256",
                    "branch_post_risk_reference_direction_content_sha256",
                    "branch_post_risk_response_content_sha256",
                    "branch_post_risk_reference_response_content_sha256",
                )
            }
            nominal_strength = branch_record.get("branch_nominal_strength")
            applied_strength = branch_record.get("branch_applied_strength")
            scale_factor = branch_record.get("branch_risk_scale_factor")
            maximum_ratio = branch_record.get(
                "branch_written_update_maximum_envelope_ratio"
            )
            post_risk_response_norm = branch_record.get(
                "branch_post_risk_response_norm"
            )
            post_risk_reference_response_norm = branch_record.get(
                "branch_post_risk_reference_response_norm"
            )
            post_risk_relative_response = branch_record.get(
                "branch_post_risk_relative_response_residual"
            )
            if (
                (
                    config.semantic_routing_enabled
                    and int(branch_record.get("eligible_position_count", 0))
                    <= 0
                )
                or not all(
                    sha256_ready(value) for value in bounded_content.values()
                )
                or bounded_content[
                    "effective_budget_values_content_sha256"
                ]
                != null_space_records[branch_name].get(
                    "risk_budget_content_sha256"
                )
                or bounded_content["branch_written_update_content_sha256"]
                != record.get(update_digest_fields[branch_name])
                or bounded_content["branch_unit_direction_content_sha256"]
                != bounded_content[
                    "branch_post_risk_direction_content_sha256"
                ]
                or branch_record.get("branch_budget_ceiling")
                != risk_configs[branch_name].budget_ceiling
                or not isinstance(nominal_strength, (int, float))
                or not math.isfinite(float(nominal_strength))
                or float(nominal_strength) <= 0.0
                or not isinstance(applied_strength, (int, float))
                or not math.isfinite(float(applied_strength))
                or not 0.0 < float(applied_strength) <= float(nominal_strength)
                or not isinstance(scale_factor, (int, float))
                or not math.isfinite(float(scale_factor))
                or abs(
                    float(scale_factor)
                    - float(applied_strength) / float(nominal_strength)
                )
                > 1e-6
                or not isinstance(maximum_ratio, (int, float))
                or not math.isfinite(float(maximum_ratio))
                or float(maximum_ratio) < 0.0
                or float(maximum_ratio) > 1.0
                or branch_record.get("branch_direction_epsilon")
                != config.risk_bounded_scale_direction_epsilon
                or branch_record.get("branch_numerical_epsilon")
                != config.null_space_numerical_epsilon
                or branch_record.get("branch_post_risk_jacobian_gate_ready")
                is not True
                or branch_record.get("branch_post_risk_jvp_mode")
                != "torch_autograd_exact_jvp_vjp_reexecution"
                or not isinstance(post_risk_response_norm, (int, float))
                or not math.isfinite(float(post_risk_response_norm))
                or float(post_risk_response_norm) < 0.0
                or not isinstance(
                    post_risk_reference_response_norm,
                    (int, float),
                )
                or not math.isfinite(
                    float(post_risk_reference_response_norm)
                )
                or float(post_risk_reference_response_norm)
                <= config.null_space_numerical_epsilon
                or not isinstance(post_risk_relative_response, (int, float))
                or not math.isfinite(float(post_risk_relative_response))
                or float(post_risk_relative_response)
                > config.maximum_relative_response_residual
                or not math.isclose(
                    float(post_risk_relative_response),
                    float(post_risk_response_norm)
                    / float(post_risk_reference_response_norm),
                    rel_tol=1e-9,
                    abs_tol=1e-12,
                )
                or composition_branch_identities.get(branch_name)
                != {
                    "branch_written_update_content_sha256": (
                        bounded_content[
                            "branch_written_update_content_sha256"
                        ]
                    ),
                    "branch_budget_envelope_content_sha256": (
                        bounded_content[
                            "branch_budget_envelope_content_sha256"
                        ]
                    ),
                }
                or (
                    branch_name == "attention_geometry"
                    and not math.isclose(
                        float(record["attention_applied_update_strength"]),
                        float(applied_strength),
                        rel_tol=0.0,
                        abs_tol=0.0,
                    )
                )
            ):
                return False
            base_content.update(bounded_content)
        branch_risk_content_records[branch_name] = base_content
    if (
        not sha256_ready(
            record.get("attention_update_unit_direction_content_sha256")
        )
        or record.get("attention_update_unit_direction_content_sha256")
        != branch_risk_records["attention_geometry"].get(
            "branch_post_risk_direction_content_sha256"
        )
    ):
        return False
    if (
        not sha256_ready(record.get("attention_update_content_sha256"))
        or record.get("attention_update_content_sha256")
        != branch_risk_records["attention_geometry"].get(
            "branch_written_update_content_sha256"
        )
    ):
        return False
    expected_risk_bundle_digest = build_stable_digest(
        {
            branch_name: branch_risk_records[branch_name][
                "risk_field_digest"
            ]
            for branch_name in (
                "lf_content",
                "tail_robust",
                "attention_geometry",
            )
        }
    )
    expected_risk_content_digest = build_stable_digest(
        {
            "risk_signal_content_records": risk_signal_content_records,
            "branch_risk_content_records": branch_risk_content_records,
        }
    )
    return bool(
        record.get("branch_risk_bundle_digest")
        == expected_risk_bundle_digest
        and record.get("branch_risk_content_digest")
        == expected_risk_content_digest
    )



def _final_image_preservation_ready(
    result: dict[str, Any],
    config: SemanticWatermarkRuntimeConfig,
) -> bool:
    """验证一次运行的最终成图累计完整特征门禁。"""

    record = result.get("metadata", {}).get("final_image_preservation", {})
    semantic_cosine = record.get("final_image_semantic_cosine_similarity")
    structure_drift = record.get(
        "final_image_handcrafted_structure_feature_relative_drift"
    )
    return bool(
        record.get("final_image_preservation_gate_ready") is True
        and isinstance(semantic_cosine, (int, float))
        and math.isfinite(float(semantic_cosine))
        and float(semantic_cosine) >= config.minimum_semantic_preservation_cosine
        and isinstance(structure_drift, (int, float))
        and math.isfinite(float(structure_drift))
        and float(structure_drift) <= config.maximum_handcrafted_structure_feature_relative_drift
    )


def _detection_qk_atomic_content_ready(
    record: dict[str, Any],
    config: SemanticWatermarkRuntimeConfig,
) -> bool:
    """验证一次仅图像检测的 raw 与 aligned Q/K 原子摘要."""

    if not config.attention_geometry_enabled:
        return True
    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        return False
    component_protocol = attention_relation_component_protocol(
        config.attention_relation_component_weights
    )
    qk_operator_records = metadata.get(
        "attention_relation_qk_operator_metadata_records"
    )
    return bool(
        metadata.get("detection_qk_atomic_content_ready") is True
        and metadata.get("attention_relation_active_component_names")
        == list(
            component_protocol[
                "attention_relation_active_component_names"
            ]
        )
        and metadata.get("attention_relation_component_weights")
        == list(config.attention_relation_component_weights)
        and metadata.get("attention_relation_component_protocol_digest")
        == component_protocol[
            "attention_relation_component_protocol_digest"
        ]
        and metadata.get("attention_relation_qk_operator_metadata_ready")
        is True
        and isinstance(qk_operator_records, list)
        and qk_operator_metadata_records_ready(
            qk_operator_records,
            config.attention_module_names,
        )
        and metadata.get("attention_relation_qk_operator_metadata_digest")
        == qk_operator_metadata_records_digest(qk_operator_records)
        and qk_atomic_evaluation_records_ready(
            metadata.get("detection_qk_atomic_content_records"),
            metadata.get("detection_qk_atomic_content_digest"),
            aggregate_field_name="detection_qk_atomic_content_records",
            expected_roles=(
                "raw_detection_image",
                "aligned_detection_image",
            ),
            expected_layer_names=config.attention_module_names,
        )
    )


def _carrier_only_final_image_preservation_ready(
    result: dict[str, Any],
    config: SemanticWatermarkRuntimeConfig,
) -> bool:
    """验证 clean 到 carrier-only 的最终内容保持与产物身份绑定。"""

    if not config.attention_geometry_enabled:
        return True
    metadata = result.get("metadata", {})
    record = metadata.get("carrier_only_final_image_preservation") or {}
    observability = metadata.get("final_image_attention_observability") or {}
    semantic_cosine = record.get(
        "carrier_only_final_image_semantic_cosine_similarity"
    )
    structure_drift = record.get(
        "carrier_only_final_image_handcrafted_structure_feature_relative_drift"
    )
    identity_digest = str(
        record.get("carrier_only_counterfactual_identity_digest", "")
    )
    observability_identity_digest = str(
        observability.get("carrier_only_counterfactual_identity_digest", "")
    )
    image_path = str(record.get("carrier_only_counterfactual_image_path", ""))
    observability_image_path = str(
        observability.get("carrier_only_counterfactual_image_path", "")
    )
    image_digest = str(
        record.get("carrier_only_counterfactual_image_digest", "")
    )
    observability_image_digest = str(
        observability.get("carrier_only_counterfactual_image_digest", "")
    )
    return bool(
        record.get("carrier_only_final_image_preservation_applicable") is True
        and record.get("carrier_only_final_image_preservation_gate_ready") is True
        and isinstance(semantic_cosine, (int, float))
        and math.isfinite(float(semantic_cosine))
        and float(semantic_cosine) >= config.minimum_semantic_preservation_cosine
        and isinstance(structure_drift, (int, float))
        and math.isfinite(float(structure_drift))
        and float(structure_drift) <= config.maximum_handcrafted_structure_feature_relative_drift
        and len(identity_digest) == 64
        and identity_digest == observability_identity_digest
        and all(character in "0123456789abcdef" for character in identity_digest)
        and bool(image_path)
        and image_path == observability_image_path
        and len(image_digest) == 64
        and image_digest == observability_image_digest
        and all(character in "0123456789abcdef" for character in image_digest)
    )


def _final_image_attention_observability_ready(
    result: dict[str, Any],
    config: SemanticWatermarkRuntimeConfig,
) -> bool:
    """验证最终成图重编码的真实 Q/K 水印增益证据。"""

    if not config.attention_geometry_enabled:
        return True
    record = result.get("metadata", {}).get(
        "final_image_attention_observability",
        {},
    )
    blind_gain = record.get("final_image_attention_blind_attribution_gain")
    paired_gain = record.get(
        "final_image_attention_carrier_paired_attribution_gain"
    )
    paired_digest = str(
        record.get("final_carrier_only_pair_weight_identity_digest", "")
    )
    record_schema_digest = str(
        record.get("final_image_attention_record_schema_digest", "")
    )
    component_identity_digest = str(
        record.get("attention_relation_component_identity_digest", "")
    )
    keyed_projection_digest = str(
        record.get("attention_relation_keyed_projection_digest", "")
    )
    component_names = record.get("attention_relation_component_names")
    paired_component_gains = record.get(
        "final_image_attention_carrier_paired_component_gains"
    )
    qk_operator_records = record.get(
        "attention_relation_qk_operator_metadata_records"
    )
    counterfactual_digests = tuple(
        str(record.get(field_name, ""))
        for field_name in (
            "carrier_only_counterfactual_identity_digest",
            "carrier_only_counterfactual_config_digest",
            "carrier_only_counterfactual_update_records_digest",
            "carrier_only_counterfactual_scheduler_trace_digest",
        )
    )
    component_protocol = attention_relation_component_protocol(
        config.attention_relation_component_weights
    )
    return bool(
        record.get("final_image_attention_observability_applicable") is True
        and record.get("carrier_only_counterfactual_ready") is True
        and record.get("carrier_only_counterfactual_changed_fields")
        == ["attention_geometry_enabled"]
        and record.get(
            "carrier_only_counterfactual_scheduler_identity_ready"
        )
        is True
        and record.get(
            "carrier_only_counterfactual_attention_geometry_enabled"
        )
        is False
        and record.get("final_image_attention_observability_gate_ready") is True
        and record.get("final_image_attention_observability_requires_gpu") is True
        and record.get(
            "final_image_attention_observability_gpu_execution_verified"
        )
        is True
        and record.get("final_image_attention_observability_source")
        == "image_reencoded_public_noise_real_qk"
        and record.get("attention_relation_source")
        == DIRECT_QK_RELATION_SOURCE
        and record.get("attention_relation_direct_qk_source_ready") is True
        and record.get("attention_relation_probability_scope")
        == "sampled_image_token_qk_relation_probability"
        and record.get("attention_module_names")
        == list(config.attention_module_names)
        and record.get("attention_coordinate_convention")
        == ATTENTION_COORDINATE_CONVENTION
        and record.get("attention_grid_align_corners")
        is ATTENTION_GRID_ALIGN_CORNERS
        and record.get("attention_relation_qk_operator_metadata_ready")
        is True
        and isinstance(qk_operator_records, list)
        and qk_operator_metadata_records_ready(
            qk_operator_records,
            config.attention_module_names,
        )
        and record.get("attention_relation_qk_operator_metadata_digest")
        == qk_operator_metadata_records_digest(qk_operator_records)
        and record.get("final_image_qk_atomic_content_ready") is True
        and qk_atomic_evaluation_records_ready(
            record.get("final_image_qk_atomic_content_records"),
            record.get("final_image_qk_atomic_content_digest"),
            aggregate_field_name="final_image_qk_atomic_content_records",
            expected_roles=(
                "final_clean_image",
                "final_carrier_only_image",
                "final_watermarked_image",
            ),
            expected_layer_names=config.attention_module_names,
        )
        and component_names == list(ATTENTION_RELATION_COMPONENT_NAMES)
        and record.get("attention_relation_active_component_names")
        == list(
            component_protocol[
                "attention_relation_active_component_names"
            ]
        )
        and record.get("attention_relation_component_weights")
        == list(config.attention_relation_component_weights)
        and record.get("attention_relation_component_protocol_digest")
        == component_protocol[
            "attention_relation_component_protocol_digest"
        ]
        and len(component_identity_digest) == 64
        and all(
            character in "0123456789abcdef"
            for character in component_identity_digest
        )
        and len(keyed_projection_digest) == 64
        and all(
            character in "0123456789abcdef"
            for character in keyed_projection_digest
        )
        and isinstance(paired_component_gains, dict)
        and set(paired_component_gains) == set(ATTENTION_RELATION_COMPONENT_NAMES)
        and all(
            isinstance(value, (int, float)) and math.isfinite(float(value))
            for value in paired_component_gains.values()
        )
        and isinstance(blind_gain, (int, float))
        and math.isfinite(float(blind_gain))
        and float(blind_gain) > config.minimum_final_image_attention_score_gain
        and isinstance(paired_gain, (int, float))
        and math.isfinite(float(paired_gain))
        and float(paired_gain) > config.minimum_final_image_attention_score_gain
        and len(paired_digest) == 64
        and all(character in "0123456789abcdef" for character in paired_digest)
        and len(record_schema_digest) == 64
        and all(
            character in "0123456789abcdef"
            for character in record_schema_digest
        )
        and all(
            len(digest) == 64
            and all(character in "0123456789abcdef" for character in digest)
            for digest in counterfactual_digests
        )
        and _carrier_only_final_image_preservation_ready(result, config)
    )


def run_image_only_dataset_runtime(
    base_method_config: SemanticWatermarkRuntimeConfig,
    root: str | Path = ".",
    paper_run: PaperRunConfig | None = None,
    max_new_prompts_per_session: int = 0,
) -> dict[str, Any]:
    """运行当前论文规模的全部 Prompt 并生成可校准记录。"""

    root_path = Path(root).resolve()
    formal_execution_run_lock = (
        repository_environment.require_published_formal_execution_lock(root_path)
    )
    resolved_paper_run = paper_run or build_paper_run_config(root_path)
    validate_formal_dataset_randomization_identity(
        base_method_config,
        resolved_paper_run,
        prompt_index=0,
    )
    prompt_path = (root_path / resolved_paper_run.prompt_file).resolve()
    prompt_records = apply_split_assignments(
        build_prompt_records(
            resolved_paper_run.prompt_set,
            read_prompt_file(prompt_path),
        )
    )[: resolved_paper_run.sample_count]
    output_dir = root_path / "outputs" / "image_only_dataset_runtime" / resolved_paper_run.run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "dataset_runtime_progress.json"
    restore_role_checkpoints(
        repository_root=root_path,
        artifact_role="image_only_dataset_runtime",
        paper_run_name=resolved_paper_run.run_name,
        allowed_output_prefix=(
            f"outputs/image_only_dataset_runtime/{resolved_paper_run.run_name}"
        ),
    )
    if max_new_prompts_per_session < 0:
        raise ValueError("max_new_prompts_per_session 不得为负")
    shared_context = None
    attack_prompt_ids = {
        record.prompt_id
        for record in tuple(record for record in prompt_records if record.split == "test")[
            : resolved_paper_run.minimum_clean_negative_count
        ]
    }
    runtime_results = []
    scientific_unit_configs: list[dict[str, Any]] = []
    detection_records: list[dict[str, Any]] = []
    scientific_update_records: list[dict[str, Any]] = []
    completed_prompt_ids: list[str] = []
    resumed_prompt_count = 0
    new_prompt_count = 0

    def write_resume_progress() -> dict[str, Any]:
        """原子保存当前 Prompt 进度并同步到外部检查点目录."""

        progress = {
            "paper_run_name": resolved_paper_run.run_name,
            "prompt_count": len(prompt_records),
            "completed_prompt_count": len(runtime_results),
            "remaining_prompt_count": len(prompt_records) - len(runtime_results),
            "resumed_prompt_count": resumed_prompt_count,
            "new_prompt_count": new_prompt_count,
            "max_new_prompts_per_session": max_new_prompts_per_session,
            "completed_prompt_digest": build_stable_digest(
                sorted(completed_prompt_ids)
            ),
            "protocol_decision": "resume_required",
            "evidence_eligibility": "intermediate_state_only",
            "supports_paper_claim": False,
        }
        temporary_path = progress_path.with_name(progress_path.name + ".partial")
        temporary_path.write_text(
            json.dumps(progress, ensure_ascii=False, sort_keys=True, indent=2)
            + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(progress_path)
        persist_progress_checkpoint(
            progress_path,
            repository_root=root_path,
            artifact_role="image_only_dataset_runtime",
            paper_run_name=resolved_paper_run.run_name,
        )
        return progress

    for prompt_record in prompt_records:
        run_attacks = prompt_record.prompt_id in attack_prompt_ids
        run_config = replace(
            base_method_config,
            prompt=prompt_record.prompt_text,
            prompt_id=prompt_record.prompt_id,
            split=prompt_record.split,
            seed=base_method_config.seed + prompt_record.prompt_index,
            inference_steps=resolved_paper_run.inference_steps,
            guidance_scale=resolved_paper_run.guidance_scale,
            injection_step_indices=resolved_paper_run.attention_injection_steps,
            standard_attack_profiles=(base_method_config.standard_attack_profiles if run_attacks else ()),
            diffusion_attacks_enabled=base_method_config.diffusion_attacks_enabled and run_attacks,
            output_dir=(
                f"outputs/image_only_dataset_runtime/{resolved_paper_run.run_name}/runs"
            ),
        )
        validate_formal_dataset_randomization_identity(
            run_config,
            resolved_paper_run,
            prompt_index=prompt_record.prompt_index,
        )
        result = load_completed_semantic_watermark_runtime_result(run_config, root=root_path)
        generated_now = False
        if result is not None:
            resumed_prompt_count += 1
        elif max_new_prompts_per_session and new_prompt_count >= max_new_prompts_per_session:
            continue
        else:
            if shared_context is None:
                shared_context = load_semantic_watermark_runtime_context(base_method_config)
            result = write_semantic_watermark_runtime_outputs(
                run_config,
                root=root_path,
                runtime_context=shared_context,
            )
            new_prompt_count += 1
            generated_now = True
        result_payload = result.to_dict()
        validate_semantic_watermark_runtime_result_provenance(
            result_payload,
            expected_config=run_config,
        )
        runtime_results.append(result_payload)
        scientific_unit_configs.append(
            semantic_watermark_runtime_config_payload(run_config)
        )
        detection_records.extend(_read_jsonl(root_path / result.detection_record_path))
        scientific_update_records.extend(_read_jsonl(root_path / result.update_record_path))
        completed_prompt_ids.append(prompt_record.prompt_id)
        if generated_now and len(runtime_results) < len(prompt_records):
            write_resume_progress()

    if len(runtime_results) != len(prompt_records):
        return write_resume_progress()

    # 完整运行到达后删除续跑状态, 避免 Colab 入口把上一次中断记录误判为当前状态。
    progress_path.unlink(missing_ok=True)
    clear_progress_checkpoints(
        artifact_role="image_only_dataset_runtime",
        paper_run_name=resolved_paper_run.run_name,
    )

    calibration_negatives = tuple(
        record
        for record in detection_records
        if record.get("split") == "calibration"
        and record.get("sample_role") == "clean_negative"
        and not record.get("attack_id")
    )
    protocol = calibrate_complete_evidence_protocol(
        calibration_negatives,
        resolved_paper_run.target_fpr,
        base_method_config.rescue_margin_low,
    )
    formal_records = apply_frozen_evidence_protocol(detection_records, protocol)
    metric_rows = build_image_only_test_metric_rows(
        formal_records,
        resolved_paper_run.target_fpr,
    )
    detection_score_tables = build_detection_score_tables(formal_records, protocol.to_dict())

    runtime_results_path = output_dir / "runtime_results.jsonl"
    detection_records_path = output_dir / "image_only_detection_records.jsonl"
    quality_registry_path = output_dir / "watermark_quality_image_registry.jsonl"
    protocol_path = output_dir / "frozen_evidence_protocol.json"
    metrics_path = output_dir / "test_detection_metrics.csv"
    summary_path = output_dir / "dataset_runtime_summary.json"
    manifest_path = output_dir / "manifest.local.json"
    runtime_results_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in runtime_results),
        encoding="utf-8",
    )
    detection_records_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in formal_records),
        encoding="utf-8",
    )
    quality_registry_rows = []
    for result, prompt_record in zip(runtime_results, prompt_records):
        clean_path = root_path / str(result["clean_image_path"])
        watermarked_path = root_path / str(result["watermarked_image_path"])
        quality_registry_rows.append(
            {
                "run_id": result["run_id"],
                "prompt_id": prompt_record.prompt_id,
                "source_image_path": clean_path.relative_to(root_path).as_posix(),
                "source_image_digest": file_digest(clean_path),
                "attacked_image_path": watermarked_path.relative_to(root_path).as_posix(),
                "attacked_image_digest": file_digest(watermarked_path),
                "attack_name": "watermark_embedding",
                "image_pair_role": "clean_to_watermarked",
                "supports_paper_claim": False,
            }
        )
    quality_registry_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in quality_registry_rows),
        encoding="utf-8",
    )
    protocol_path.write_text(json.dumps(protocol.to_dict(), ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    _write_csv(metrics_path, metric_rows)
    detection_score_table_paths = write_detection_score_tables(
        output_dir,
        detection_score_tables,
    )
    split_counts = {
        split: sum(record.split == split for record in prompt_records)
        for split in ("dev", "calibration", "test")
    }
    clean_test_row = next(
        (
            row
            for row in metric_rows
            if row["attack_name"] == "none" and row["sample_role"] == "clean_negative"
        ),
        None,
    )
    wrong_key_test_row = next(
        (
            row
            for row in metric_rows
            if row["attack_name"] == "none" and row["sample_role"] == "wrong_key_negative"
        ),
        None,
    )
    paired_ssim_values = [
        float(result["metadata"]["paired_quality"]["ssim"])
        for result in runtime_results
        if result.get("metadata", {}).get("paired_quality", {}).get("ssim") is not None
    ]
    paired_psnr_values = [
        float(result["metadata"]["paired_quality"]["psnr"])
        for result in runtime_results
        if isinstance(result.get("metadata", {}).get("paired_quality", {}).get("psnr"), (int, float))
    ]
    expected_split_counts = build_group_split_counts(resolved_paper_run.prompt_count)
    expected_scientific_update_count = len(prompt_records) * len(resolved_paper_run.attention_injection_steps)
    scientific_operator_failure_count = sum(
        not _scientific_update_record_ready(record, base_method_config)
        for record in scientific_update_records
    )
    final_image_preservation_failure_count = sum(
        not _final_image_preservation_ready(result, base_method_config)
        for result in runtime_results
    )
    final_image_attention_observability_failure_count = sum(
        not _final_image_attention_observability_ready(
            result,
            base_method_config,
        )
        for result in runtime_results
    )
    detection_qk_atomic_content_failure_count = sum(
        not _detection_qk_atomic_content_ready(
            record,
            base_method_config,
        )
        for record in detection_records
    )
    scientific_content_binding_digests = [
        str(
            result.get("metadata", {}).get(
                "scientific_content_binding_digest",
                "",
            )
        )
        for result in runtime_results
    ]
    scientific_content_binding_failure_count = sum(
        not _scientific_content_binding_record_ready(result)
        for result in runtime_results
    )
    scientific_content_binding_gate_ready = bool(
        len(scientific_content_binding_digests) == len(prompt_records)
        and scientific_content_binding_failure_count == 0
    )
    scientific_content_binding_digest = (
        build_stable_digest(
            {
                "scientific_content_binding_digests": (
                    scientific_content_binding_digests
                )
            }
        )
        if scientific_content_binding_gate_ready
        else ""
    )
    scientific_operator_gate_ready = (
        len(scientific_update_records) == expected_scientific_update_count
        and scientific_operator_failure_count == 0
        and final_image_preservation_failure_count == 0
        and final_image_attention_observability_failure_count == 0
        and detection_qk_atomic_content_failure_count == 0
        and scientific_content_binding_gate_ready
    )
    scientific_unit_provenance = aggregate_scientific_unit_provenance(
        (
            result["metadata"]["scientific_unit_provenance"]
            for result in runtime_results
        ),
        expected_reference_count=len(prompt_records),
    )
    protocol_decision = (
        "pass"
        if len(prompt_records) == resolved_paper_run.prompt_count
        and resolved_paper_run.sample_count == resolved_paper_run.prompt_count
        and split_counts == expected_split_counts
        and all(result.get("run_decision") == "pass" for result in runtime_results)
        and scientific_operator_gate_ready
        and scientific_unit_provenance["scientific_unit_provenance_ready"]
        and scientific_unit_provenance["scientific_unit_provenance_record_count"]
        == len(prompt_records)
        else "fail"
    )
    attacked_records = tuple(record for record in formal_records if record.get("attack_id"))
    standard_attack_ids = {
        attack.attack_id
        for attack in default_attack_configs()
        if attack.enabled
        and not attack.requires_gpu
        and attack.resource_profile in set(base_method_config.standard_attack_profiles)
    }
    diffusion_attack_ids = {
        attack.attack_id for attack in default_diffusion_attack_specs()
    } if base_method_config.diffusion_attacks_enabled else set()
    expected_attack_ids = standard_attack_ids | diffusion_attack_ids
    actual_attack_ids = {str(record.get("attack_id")) for record in attacked_records}
    attack_role_counts = {
        (attack_id, sample_role): sum(
            str(record.get("attack_id")) == attack_id and record.get("sample_role") == sample_role
            for record in attacked_records
        )
        for attack_id in expected_attack_ids
        for sample_role in ("clean_negative", "positive_source")
    }
    attack_record_coverage_ready = (
        bool(expected_attack_ids)
        and actual_attack_ids == expected_attack_ids
        and all(count == len(attack_prompt_ids) for count in attack_role_counts.values())
    )
    attacked_image_evidence_chain_ready = bool(attacked_records) and all(
        record.get("attacked_image_path")
        and record.get("attacked_image_digest")
        and (root_path / str(record["attacked_image_path"])).is_file()
        for record in attacked_records
    )
    clean_fixed_fpr_ready = bool(clean_test_row and clean_test_row["fixed_fpr_upper_bound_ready"])
    wrong_key_fixed_fpr_ready = bool(wrong_key_test_row and wrong_key_test_row["fixed_fpr_upper_bound_ready"])
    image_only_protocol_ready = all(
        str(record.get("metadata", {}).get("detector_input_access_mode", ""))
        == "image_key_public_model_only"
        and not bool(record.get("metadata", {}).get("generation_latent_trace_required", True))
        and bool(record.get("metadata", {}).get("blind_image_detector", False))
        for record in formal_records
    )
    full_method_component_ready = (
        protocol_decision == "pass"
        and clean_fixed_fpr_ready
        and wrong_key_fixed_fpr_ready
        and image_only_protocol_ready
        and protocol.geometry_protocol_calibration_ready
        and scientific_operator_gate_ready
        and scientific_unit_provenance["scientific_unit_provenance_ready"]
    )
    required_real_gpu_attack_count = len(diffusion_attack_ids)
    measured_real_gpu_attack_count = len(
        {
            str(record.get("attack_id"))
            for record in attacked_records
            if record.get("resource_profile") == "full_extra"
        }
    )
    real_gpu_attack_validation_ready = (
        required_real_gpu_attack_count == 0
        or measured_real_gpu_attack_count >= required_real_gpu_attack_count
    )
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "paper_run_name": resolved_paper_run.run_name,
        "randomization_repeat_identity": {
            "randomization_repeat_id": (
                resolved_paper_run.randomization_repeat_id
            ),
            "generation_seed_index": (
                resolved_paper_run.generation_seed_index
            ),
            "generation_seed_offset": (
                resolved_paper_run.generation_seed_offset
            ),
            "watermark_key_index": resolved_paper_run.watermark_key_index,
            "formal_randomization_protocol_digest": (
                resolved_paper_run.formal_randomization_protocol_digest
            ),
        },
        "prompt_count": len(prompt_records),
        "split_counts": split_counts,
        "runtime_result_count": len(runtime_results),
        "resumed_prompt_count": resumed_prompt_count,
        "new_prompt_count": new_prompt_count,
        "attack_prompt_count": len(attack_prompt_ids),
        "detection_record_count": len(formal_records),
        "score_distribution_row_count": len(detection_score_tables["score_distribution_table"]),
        "roc_curve_point_count": len(detection_score_tables["roc_curve_points"]),
        "det_curve_point_count": len(detection_score_tables["det_curve_points"]),
        "detection_curve_data_ready": all(detection_score_tables.values()),
        "watermark_quality_pair_count": len(quality_registry_rows),
        "scientific_update_record_count": len(scientific_update_records),
        "expected_scientific_update_record_count": expected_scientific_update_count,
        "scientific_operator_failure_count": scientific_operator_failure_count,
        "final_image_preservation_failure_count": (
            final_image_preservation_failure_count
        ),
        "final_image_attention_observability_failure_count": (
            final_image_attention_observability_failure_count
        ),
        "final_image_attention_observability_ready": (
            final_image_attention_observability_failure_count == 0
        ),
        "detection_qk_atomic_content_failure_count": (
            detection_qk_atomic_content_failure_count
        ),
        "detection_qk_atomic_content_ready": (
            detection_qk_atomic_content_failure_count == 0
        ),
        "scientific_content_binding_digests": (
            scientific_content_binding_digests
        ),
        "scientific_content_binding_digest": (
            scientific_content_binding_digest
        ),
        "scientific_content_binding_failure_count": (
            scientific_content_binding_failure_count
        ),
        "scientific_content_binding_gate_ready": (
            scientific_content_binding_gate_ready
        ),
        "scientific_operator_gate_ready": scientific_operator_gate_ready,
        **scientific_unit_provenance,
        "attention_alignment_gate": dict(
            _FORMAL_ATTENTION_ALIGNMENT_GATE
        ),
        **_FORMAL_ATTENTION_ALIGNMENT_GATE,
        "lf_carrier_protocol_digest": protocol.lf_carrier_protocol_digest,
        "tail_carrier_protocol_digest": (
            protocol.tail_carrier_protocol_digest
        ),
        "lf_weight": protocol.lf_weight,
        "tail_robust_weight": protocol.tail_robust_weight,
        "tail_fraction": protocol.tail_fraction,
        "image_only_detector_config_digest": (
            protocol.image_only_detector_config_digest
        ),
        "frozen_threshold_digest": protocol.threshold_digest,
        "geometry_protocol_calibration_ready": (
            protocol.geometry_protocol_calibration_ready
        ),
        "target_fpr": resolved_paper_run.target_fpr,
        "clean_test_fixed_fpr_upper_bound_ready": clean_fixed_fpr_ready,
        "wrong_key_test_fixed_fpr_upper_bound_ready": wrong_key_fixed_fpr_ready,
        "paired_ssim_mean": sum(paired_ssim_values) / len(paired_ssim_values) if paired_ssim_values else None,
        "paired_psnr_mean": sum(paired_psnr_values) / len(paired_psnr_values) if paired_psnr_values else None,
        "fixed_fpr_and_rescue_boundary_ready": (
            protocol.geometry_protocol_calibration_ready
        ),
        "fixed_fpr_boundary_ready": True,
        "rescue_boundary_ready": protocol.geometry_protocol_calibration_ready,
        "raw_content_measurement_ready": True,
        "perceptual_metrics_ready": bool(paired_ssim_values),
        "real_attacked_image_count": len(attacked_records),
        "real_attacked_image_closed_loop_ready": attacked_image_evidence_chain_ready,
        "attacked_image_evidence_chain_ready": attacked_image_evidence_chain_ready,
        "formal_attack_detection_ready": attack_record_coverage_ready,
        "attack_record_coverage_ready": attack_record_coverage_ready,
        "required_attack_id_count": len(expected_attack_ids),
        "measured_attack_id_count": len(actual_attack_ids),
        "required_real_gpu_attack_count": required_real_gpu_attack_count,
        "measured_real_gpu_attack_count": measured_real_gpu_attack_count,
        "real_gpu_attack_validation_ready": real_gpu_attack_validation_ready,
        "full_method_component_ready": full_method_component_ready,
        "detector_input_access_mode": "image_key_public_model_only",
        "generation_latent_trace_required": False,
        "protocol_decision": protocol_decision,
        "repeat_component_ready": (
            full_method_component_ready
            and attack_record_coverage_ready
            and attacked_image_evidence_chain_ready
            and real_gpu_attack_validation_ready
        ),
        "randomization_aggregate_ready": False,
        "supports_paper_claim": False,
    }
    prompt_source_snapshot_paths, prompt_source_report = (
        _write_prompt_source_snapshot(
            root_path=root_path,
            output_dir=output_dir,
            paper_run_name=resolved_paper_run.run_name,
            prompt_path=prompt_path,
        )
    )
    summary.update(
        {
            "prompt_file_sha256": prompt_source_report[
                "prompt_file_sha256"
            ],
            "prompt_source_registry_digest": prompt_source_report[
                "prompt_source_registry_digest"
            ],
            "selection_manifest_sha256": prompt_source_report[
                "selection_manifest_sha256"
            ],
            "selection_manifest_digest": prompt_source_report[
                "selection_manifest_digest"
            ],
            "packaged_prompt_source_audit_digest": prompt_source_report[
                "packaged_prompt_source_audit_digest"
            ],
            "prompt_source_contract_ready": True,
        }
    )
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    scientific_unit_output_paths: list[str] = []
    scientific_unit_identity_records: list[dict[str, Any]] = []
    for result, full_unit_config in zip(
        runtime_results,
        scientific_unit_configs,
        strict=True,
    ):
        unit_manifest_path = root_path / str(result["manifest_path"])
        unit_manifest = json.loads(
            unit_manifest_path.read_text(encoding="utf-8-sig")
        )
        unit_output_paths = unit_manifest.get("output_paths")
        if not isinstance(unit_output_paths, list) or not unit_output_paths:
            raise RuntimeError("单 Prompt 方法产物清单缺少科学内容叶子")
        scientific_unit_output_paths.extend(
            str(path) for path in unit_output_paths
        )
        unit_manifest_config = unit_manifest.get("config")
        randomization_reference = result.get("metadata", {}).get(
            "formal_randomization_reference"
        )
        if not isinstance(unit_manifest_config, dict) or not isinstance(
            randomization_reference,
            dict,
        ) or unit_manifest_config != {
            "scientific_unit_config_digest": build_stable_digest(
                full_unit_config
            ),
            "formal_randomization_reference": randomization_reference,
        }:
            raise RuntimeError("单 Prompt manifest 缺少正式配置或随机身份引用")
        scientific_unit_identity_records.append(
            {
                "run_id": result["run_id"],
                "scientific_unit_config": full_unit_config,
                "formal_randomization_reference": randomization_reference,
            }
        )
    if len(scientific_unit_output_paths) != len(
        set(scientific_unit_output_paths)
    ):
        raise RuntimeError("单 Prompt 方法产物清单包含跨单元重复路径")
    formal_execution_run_lock = repository_environment.validate_formal_execution_lock_pair(
        formal_execution_run_lock,
        repository_environment.require_published_formal_execution_lock(root_path),
        formal_execution_run_lock["formal_execution_commit"],
    )
    manifest = build_artifact_manifest(
        artifact_id=f"{resolved_paper_run.run_name}_image_only_dataset_runtime_manifest",
        artifact_type="local_manifest",
        input_paths=(
            prompt_path.relative_to(root_path).as_posix(),
            "configs/prompt_source_registry.json",
            "configs/prompt_selection_manifest.jsonl",
        ),
        output_paths=(
            runtime_results_path.relative_to(root_path).as_posix(),
            detection_records_path.relative_to(root_path).as_posix(),
            quality_registry_path.relative_to(root_path).as_posix(),
            protocol_path.relative_to(root_path).as_posix(),
            metrics_path.relative_to(root_path).as_posix(),
            detection_score_table_paths["score_distribution_table"].relative_to(root_path).as_posix(),
            detection_score_table_paths["roc_curve_points"].relative_to(root_path).as_posix(),
            detection_score_table_paths["det_curve_points"].relative_to(root_path).as_posix(),
            summary_path.relative_to(root_path).as_posix(),
            manifest_path.relative_to(root_path).as_posix(),
            *(
                path.relative_to(root_path).as_posix()
                for path in prompt_source_snapshot_paths
            ),
            *scientific_unit_output_paths,
        ),
        config={
            "paper_run": resolved_paper_run.to_dict(),
            "formal_randomization_plan": (
                formal_runtime_randomization_plan_record(
                    _FORMAL_METHOD_CONFIG.seed,
                    base_latent_dtype=(
                        f"torch.{base_method_config.torch_dtype}"
                    ),
                    base_latent_shape=(
                        1,
                        16,
                        base_method_config.height // 8,
                        base_method_config.width // 8,
                    ),
                )
            ),
            "scientific_unit_identity_records": (
                scientific_unit_identity_records
            ),
            # manifest 现在保存完整配置, 因此必须复用运行时的密钥脱敏配置。
            # 该结构保留全部可复现实验参数, 但只记录 key material 的稳定摘要。
            "method_config": semantic_watermark_runtime_config_payload(
                base_method_config
            ),
        },
        code_version=formal_execution_run_lock["formal_execution_commit"],
        rebuild_command="调用 experiments.runners.image_only_dataset_runtime.run_image_only_dataset_runtime",
        metadata={
            "protocol_decision": summary["protocol_decision"],
            "detector_input_access_mode": "image_key_public_model_only",
            "attention_alignment_gate": dict(
                _FORMAL_ATTENTION_ALIGNMENT_GATE
            ),
            **_FORMAL_ATTENTION_ALIGNMENT_GATE,
            "lf_carrier_protocol_digest": (
                protocol.lf_carrier_protocol_digest
            ),
            "tail_carrier_protocol_digest": (
                protocol.tail_carrier_protocol_digest
            ),
            "lf_weight": protocol.lf_weight,
            "tail_robust_weight": protocol.tail_robust_weight,
            "tail_fraction": protocol.tail_fraction,
            "image_only_detector_config_digest": (
                protocol.image_only_detector_config_digest
            ),
            "full_method_component_ready": summary[
                "full_method_component_ready"
            ],
            "repeat_component_ready": summary["repeat_component_ready"],
            "randomization_aggregate_ready": False,
            "geometry_protocol_calibration_ready": summary[
                "geometry_protocol_calibration_ready"
            ],
            "attack_record_coverage_ready": summary["attack_record_coverage_ready"],
            "attacked_image_evidence_chain_ready": summary["attacked_image_evidence_chain_ready"],
            "scientific_operator_gate_ready": summary["scientific_operator_gate_ready"],
            "final_image_attention_observability_failure_count": summary[
                "final_image_attention_observability_failure_count"
            ],
            "final_image_attention_observability_ready": summary[
                "final_image_attention_observability_ready"
            ],
            "detection_qk_atomic_content_failure_count": summary[
                "detection_qk_atomic_content_failure_count"
            ],
            "detection_qk_atomic_content_ready": summary[
                "detection_qk_atomic_content_ready"
            ],
            "scientific_content_binding_digest": summary[
                "scientific_content_binding_digest"
            ],
            "scientific_content_binding_failure_count": summary[
                "scientific_content_binding_failure_count"
            ],
            "scientific_content_binding_gate_ready": summary[
                "scientific_content_binding_gate_ready"
            ],
            "scientific_unit_provenance_ready": summary[
                "scientific_unit_provenance_ready"
            ],
            "scientific_unit_provenance_records_digest": summary[
                "scientific_unit_provenance_records_digest"
            ],
            "supports_paper_claim": summary["supports_paper_claim"],
        },
    ).to_dict()
    manifest["formal_execution_run_lock"] = formal_execution_run_lock
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return summary


def package_image_only_dataset_runtime(
    paper_run_name: str,
    root: str | Path = ".",
) -> Path:
    """把真实运行 records、图像、阈值和 manifest 打包为受治理输入包。"""

    root_path = Path(root).resolve()
    formal_execution_package_lock = (
        repository_environment.require_published_formal_execution_lock(root_path)
    )
    resolved_paper_run_name = normalize_paper_run_name(paper_run_name)
    paper_run = build_paper_run_config(root_path)
    if paper_run.run_name != resolved_paper_run_name:
        raise ValueError("仅图像运行打包层级必须与当前论文配置一致")
    source_dir = (
        root_path / "outputs" / "image_only_dataset_runtime" / resolved_paper_run_name
    )
    required_paths = tuple(
        source_dir / filename
        for filename in (
            "runtime_results.jsonl",
            "image_only_detection_records.jsonl",
            "watermark_quality_image_registry.jsonl",
            "frozen_evidence_protocol.json",
            "test_detection_metrics.csv",
            "score_distribution_table.csv",
            "roc_curve_points.csv",
            "det_curve_points.csv",
            "dataset_runtime_summary.json",
            "manifest.local.json",
        )
    )
    prompt_source_snapshot_paths, prompt_source_report = (
        _audit_prompt_source_snapshot(
            output_dir=source_dir,
            paper_run_name=resolved_paper_run_name,
        )
    )
    required_paths = (*required_paths, *prompt_source_snapshot_paths)
    if any(not path.is_file() for path in required_paths):
        raise FileNotFoundError("仅图像数据集运行输出不完整, 不得打包")
    summary = json.loads((source_dir / "dataset_runtime_summary.json").read_text(encoding="utf-8-sig"))
    manifest = json.loads((source_dir / "manifest.local.json").read_text(encoding="utf-8-sig"))
    frozen_protocol_record = json.loads(
        (source_dir / "frozen_evidence_protocol.json").read_text(
            encoding="utf-8-sig"
        )
    )
    dataset_output_paths = manifest.get("output_paths")
    if not isinstance(dataset_output_paths, list) or not dataset_output_paths:
        raise RuntimeError("数据集 manifest 缺少正式输出路径")
    dataset_output_path_set = set(dataset_output_paths)
    if len(dataset_output_path_set) != len(dataset_output_paths):
        raise RuntimeError("数据集 manifest 不得包含重复输出路径")
    scientific_unit_output_paths: list[str] = []
    packaged_runtime_results = _read_jsonl(source_dir / "runtime_results.jsonl")
    manifest_config = manifest.get("config")
    unit_identity_records = (
        manifest_config.get("scientific_unit_identity_records")
        if isinstance(manifest_config, dict)
        else None
    )
    if not isinstance(unit_identity_records, list):
        raise RuntimeError("数据集顶层 manifest 缺少逐单元完整身份正文")
    unit_identity_by_run_id = {
        str(record.get("run_id", "")): record
        for record in unit_identity_records
        if isinstance(record, dict)
    }
    if (
        len(unit_identity_by_run_id) != len(unit_identity_records)
        or len(unit_identity_by_run_id) != len(packaged_runtime_results)
    ):
        raise RuntimeError("数据集顶层 manifest 的逐单元身份集合无效")
    for result in packaged_runtime_results:
        unit_identity = unit_identity_by_run_id.get(str(result.get("run_id", "")))
        unit_config = (
            unit_identity.get("scientific_unit_config")
            if isinstance(unit_identity, dict)
            else None
        )
        randomization_reference = (
            unit_identity.get("formal_randomization_reference")
            if isinstance(unit_identity, dict)
            else None
        )
        if not isinstance(unit_config, dict) or not isinstance(
            randomization_reference,
            dict,
        ):
            raise RuntimeError("顶层 manifest 逐单元配置或随机身份无效")
        validate_semantic_watermark_runtime_result_provenance(
            result,
            unit_config=unit_config,
        )
        unit_manifest_relative = Path(
            str(result.get("manifest_path", ""))
        )
        unit_manifest_path = (root_path / unit_manifest_relative).resolve()
        try:
            unit_manifest_path.relative_to(source_dir / "runs")
        except ValueError as error:
            raise RuntimeError(
                "单 Prompt manifest 必须位于当前数据集 runs 目录"
            ) from error
        if (
            unit_manifest_relative.is_absolute()
            or not unit_manifest_path.is_file()
            or unit_manifest_path.is_symlink()
        ):
            raise FileNotFoundError("单 Prompt 科学内容 manifest 不存在")
        unit_manifest = json.loads(
            unit_manifest_path.read_text(encoding="utf-8-sig")
        )
        unit_output_paths = unit_manifest.get("output_paths")
        unit_manifest_relative_path = unit_manifest_path.relative_to(
            root_path
        ).as_posix()
        if (
            not isinstance(unit_output_paths, list)
            or not unit_output_paths
            or any(not isinstance(path, str) or not path for path in unit_output_paths)
            or len(unit_output_paths) != len(set(unit_output_paths))
            or unit_manifest_relative_path not in unit_output_paths
        ):
            raise RuntimeError("单 Prompt manifest 的科学叶子路径集合无效")
        scientific_unit_output_paths.extend(unit_output_paths)
        expected_unit_manifest_config = {
            "scientific_unit_config_digest": build_stable_digest(
                unit_config
            ),
            "formal_randomization_reference": randomization_reference,
        }
        if (
            result.get("metadata", {}).get("formal_randomization_reference")
            != randomization_reference
            or unit_manifest.get("config") != expected_unit_manifest_config
            or unit_manifest.get("config_digest")
            != build_stable_digest(expected_unit_manifest_config)
            or not (
            _carrier_only_counterfactual_artifact_binding_ready(
                result,
                unit_manifest,
                root_path,
                unit_config,
            )
            and _scientific_content_binding_artifact_ready(
                result,
                unit_manifest,
                root_path,
                unit_config,
            )
            )
        ):
            raise RuntimeError("单 Prompt 科学内容无法从完成包叶子重建")
    if (
        len(scientific_unit_output_paths)
        != len(set(scientific_unit_output_paths))
        or not set(scientific_unit_output_paths).issubset(
            dataset_output_path_set
        )
    ):
        raise RuntimeError("数据集 manifest 未完整覆盖全部单 Prompt 科学叶子")
    packaged_prompt_records = apply_split_assignments(
        build_prompt_records(
            paper_run.prompt_set,
            read_prompt_file(root_path / paper_run.prompt_file),
        )
    )[: paper_run.sample_count]
    packaged_unit_configs = [
        unit_identity_by_run_id[str(result["run_id"])][
            "scientific_unit_config"
        ]
        for result in packaged_runtime_results
    ]
    packaged_unit_config_contract_ready = (
        len(packaged_unit_configs) == len(packaged_prompt_records)
        and len(
            {
                int(config["seed"]) - prompt.prompt_index
                for config, prompt in zip(
                    packaged_unit_configs,
                    packaged_prompt_records,
                )
            }
        )
        == 1
        and all(
            config.get("prompt_id") == prompt.prompt_id
            and config.get("prompt") == prompt.prompt_text
            and config.get("split") == prompt.split
            and int(config.get("inference_steps", -1))
            == paper_run.inference_steps
            and math.isclose(
                float(config.get("guidance_scale", -1.0)),
                paper_run.guidance_scale,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            and tuple(config.get("injection_step_indices", ()))
            == paper_run.attention_injection_steps
            and config.get("output_dir")
            == (
                f"outputs/image_only_dataset_runtime/"
                f"{resolved_paper_run_name}/runs"
            )
            and all(
                config.get(field_name) is True
                for field_name in (
                    "semantic_routing_enabled",
                    "null_space_enabled",
                    "lf_enabled",
                    "tail_robust_enabled",
                    "tail_truncation_enabled",
                    "attention_geometry_enabled",
                    "image_alignment_enabled",
                )
            )
            for config, prompt in zip(
                packaged_unit_configs,
                packaged_prompt_records,
            )
        )
    )
    packaged_scientific_unit_provenance = aggregate_scientific_unit_provenance(
        (
            result["metadata"]["scientific_unit_provenance"]
            for result in packaged_runtime_results
        ),
        expected_reference_count=paper_run.prompt_count,
    )
    scientific_unit_provenance_summary_bound = all(
        summary.get(field_name)
        == packaged_scientific_unit_provenance[field_name]
        for field_name in SCIENTIFIC_UNIT_PROVENANCE_AGGREGATE_FIELDS
    )
    packaged_scientific_content_binding_digests = [
        str(
            result.get("metadata", {}).get(
                "scientific_content_binding_digest",
                "",
            )
        )
        for result in packaged_runtime_results
    ]
    packaged_scientific_content_binding_digest = build_stable_digest(
        {
            "scientific_content_binding_digests": (
                packaged_scientific_content_binding_digests
            )
        }
    )
    scientific_content_binding_summary_bound = bool(
        len(packaged_scientific_content_binding_digests)
        == paper_run.prompt_count
        and all(
            _sha256_ready(digest)
            for digest in packaged_scientific_content_binding_digests
        )
        and summary.get("scientific_content_binding_digests")
        == packaged_scientific_content_binding_digests
        and summary.get("scientific_content_binding_digest")
        == packaged_scientific_content_binding_digest
    )
    formal_execution_run_lock = repository_environment.validate_formal_execution_lock_pair(
        manifest.get("formal_execution_run_lock"),
        formal_execution_package_lock,
        manifest.get("code_version"),
    )
    if not all(
        (
            summary.get("paper_run_name") == resolved_paper_run_name,
            math.isclose(
                float(summary.get("target_fpr", -1.0)),
                paper_run.target_fpr,
                rel_tol=0.0,
                abs_tol=1e-12,
            ),
            bool(summary.get("generated_at")),
            summary.get("protocol_decision") == "pass",
            _formal_attention_alignment_gate_record_ready(summary),
            _formal_content_carrier_fields_ready(summary),
            _sha256_ready(
                summary.get("image_only_detector_config_digest")
            ),
            _formal_attention_alignment_gate_fields_ready(
                frozen_protocol_record
            ),
            _formal_content_carrier_fields_ready(
                frozen_protocol_record
            ),
            _sha256_ready(
                frozen_protocol_record.get(
                    "image_only_detector_config_digest"
                )
            ),
            frozen_protocol_record.get(
                "image_only_detector_config_digest"
            )
            == summary.get("image_only_detector_config_digest"),
            frozen_protocol_record.get("threshold_digest")
            == summary.get("frozen_threshold_digest"),
            summary.get("full_method_component_ready") is True,
            summary.get("geometry_protocol_calibration_ready") is True,
            summary.get("scientific_unit_provenance_ready") is True,
            summary.get("scientific_unit_provenance_record_count")
            == paper_run.prompt_count,
            bool(summary.get("scientific_unit_provenance_records_digest")),
            summary.get("scientific_content_binding_gate_ready") is True,
            summary.get("scientific_content_binding_failure_count") == 0,
            _sha256_ready(
                summary.get("scientific_content_binding_digest")
            ),
            scientific_content_binding_summary_bound,
            scientific_unit_provenance_summary_bound,
            packaged_unit_config_contract_ready,
            summary.get("detection_curve_data_ready") is True,
            summary.get("repeat_component_ready") is True,
            summary.get("randomization_aggregate_ready") is False,
            summary.get("supports_paper_claim") is False,
            summary.get("prompt_source_contract_ready") is True,
            summary.get("prompt_file_sha256")
            == prompt_source_report["prompt_file_sha256"],
            summary.get("prompt_source_registry_digest")
            == prompt_source_report["prompt_source_registry_digest"],
            summary.get("selection_manifest_sha256")
            == prompt_source_report["selection_manifest_sha256"],
            summary.get("selection_manifest_digest")
            == prompt_source_report["selection_manifest_digest"],
            summary.get("packaged_prompt_source_audit_digest")
            == prompt_source_report["packaged_prompt_source_audit_digest"],
            manifest.get("artifact_id")
            == f"{resolved_paper_run_name}_image_only_dataset_runtime_manifest",
            _formal_attention_alignment_gate_record_ready(
                manifest.get("metadata", {})
            ),
            _formal_content_carrier_fields_ready(
                manifest.get("metadata", {})
            ),
            _sha256_ready(
                manifest.get("metadata", {}).get(
                    "image_only_detector_config_digest"
                )
            ),
            manifest.get("metadata", {}).get(
                "image_only_detector_config_digest"
            )
            == summary.get("image_only_detector_config_digest"),
            _formal_attention_alignment_gate_fields_ready(
                manifest.get("config", {}).get("method_config", {})
            ),
            manifest.get("metadata", {}).get(
                "geometry_protocol_calibration_ready"
            )
            is True,
            manifest.get("metadata", {}).get(
                "scientific_content_binding_gate_ready"
            )
            is True,
            manifest.get("metadata", {}).get(
                "scientific_content_binding_digest"
            )
            == summary.get("scientific_content_binding_digest"),
        )
    ):
        raise RuntimeError("仅图像数据集运行身份或 ready 门禁未通过")
    manifest["formal_execution_package_lock"] = formal_execution_package_lock
    (source_dir / "manifest.local.json").write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    validate_scientific_execution_binding(
        source_dir / "scientific_execution_binding.json",
        expected_artifact_role="image_only_dataset_runtime",
        expected_paper_run_name=resolved_paper_run_name,
        repository_root=root_path,
    )
    code_version = formal_execution_package_lock["formal_execution_commit"]
    archive_path = source_dir / (
        f"image_only_dataset_runtime_package_{utc_archive_token()}_{code_version[:7]}.zip"
    )
    package_input_manifest_path = source_dir / PACKAGE_INPUT_MANIFEST_FILE_NAME
    package_input_manifest_path.unlink(missing_ok=True)
    entries = collect_exact_package_entries(
        repository_root=root_path,
        source_dir=source_dir,
        artifact_manifest=manifest,
        scientific_binding_path=source_dir / "scientific_execution_binding.json",
    )
    if not set(required_paths).issubset(entries):
        raise RuntimeError("artifact manifest 未精确声明全部仅图像运行必要产物")
    write_exact_package_input_manifest(
        package_input_manifest_path,
        repository_root=root_path,
        package_family="image_only_dataset_runtime",
        paper_run_name=resolved_paper_run_name,
        target_fpr=paper_run.target_fpr,
        randomization_repeat_identity={
            "randomization_repeat_id": paper_run.randomization_repeat_id,
            "generation_seed_index": paper_run.generation_seed_index,
            "generation_seed_offset": paper_run.generation_seed_offset,
            "watermark_key_index": paper_run.watermark_key_index,
        },
        formal_randomization_protocol_digest=(
            paper_run.formal_randomization_protocol_digest
        ),
        entries=entries,
        formal_execution_run_lock=formal_execution_run_lock,
        formal_execution_package_lock=formal_execution_package_lock,
    )
    entries = (*entries, package_input_manifest_path)
    with ZipFile(archive_path, "w", compression=ZIP_STORED, allowZip64=True) as archive:
        for path in entries:
            archive.write(path, path.relative_to(root_path).as_posix())
    try:
        validate_exact_package_archive(
            archive_path,
            repository_root=root_path,
            package_input_manifest_path=package_input_manifest_path,
        )
        final_package_lock = (
            repository_environment.require_published_formal_execution_lock(root_path)
        )
        repository_environment.validate_formal_execution_lock_pair(
            formal_execution_package_lock,
            final_package_lock,
            code_version,
        )
    except Exception:
        archive_path.unlink(missing_ok=True)
        raise
    return archive_path
