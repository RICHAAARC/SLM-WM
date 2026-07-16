"""重建三个论文运行规模之间的协议同构与流程迁移结论。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
import json
import math
from pathlib import Path
from typing import Any

from experiments.protocol.attacks import (
    attack_config_digest,
    default_attack_configs,
    formal_attack_seed_protocol_record,
)
from experiments.protocol.formal_randomization import (
    formal_randomization_protocol_record,
    formal_randomization_repeat_ids,
    formal_randomization_repeat_registry_digest,
)
from experiments.protocol.method_runtime_config import (
    formal_method_config_digest,
    formal_method_config_payload,
    load_formal_method_runtime_config,
)
from experiments.protocol.paper_fixed_fpr import PAPER_CONFIDENCE_LEVEL
from experiments.protocol.paper_run_config import (
    FULL_PAPER_RUN_NAME,
    PILOT_PAPER_RUN_NAME,
    PROBE_PAPER_RUN_NAME,
    RUN_DEFAULTS,
    RUN_EXPECTED_PROMPT_COUNTS,
    derive_dataset_level_quality_minimum_count,
    derive_minimum_clean_negative_count,
)
from experiments.protocol.splits import (
    CALIBRATION_RATIO,
    DEV_RATIO,
    SPLIT_ASSIGNMENT_BLOCK_SIZE,
    SPLIT_NAMES,
    build_group_split_counts,
)
from main.core.digest import build_stable_digest
from paper_experiments.analysis.paired_superiority import (
    BOOTSTRAP_ANALYSIS_SCHEMA,
    CLAIM_P_VALUE_METHOD,
    PRIMARY_BASELINE_IDS,
    QUALITY_MATCHING_CALIPER,
    QUALITY_MATCHING_METRIC_NAME,
    QUALITY_MATCHING_MINIMUM_FRACTION,
    QUALITY_MATCHING_PROTOCOL_SCHEMA,
)
from paper_experiments.analysis.paper_claim_decisions import (
    load_paper_claim_registry,
    validate_claim_decision_bundle,
)
from paper_experiments.analysis.paper_quality_decisions import (
    load_paper_quality_claim_protocol,
)
from paper_experiments.analysis.randomization_detection_statistics import (
    RANDOMIZATION_DETECTION_CONFIDENCE_INTERVAL_METHOD,
    RANDOMIZATION_DETECTION_FALSE_POSITIVE_BOUND_METHOD,
    RANDOMIZATION_DETECTION_STATISTICAL_UNIT,
)
from paper_experiments.baselines.adapters import default_baseline_specs


PAPER_PROFILE_IDS = (
    PROBE_PAPER_RUN_NAME,
    PILOT_PAPER_RUN_NAME,
    FULL_PAPER_RUN_NAME,
)
DEFAULT_PAPER_PROFILE_PROTOCOL_REGISTRY_PATH = (
    Path(__file__).resolve().parents[2]
    / "configs"
    / "paper_profile_protocol_registry.json"
)


class PaperProfileProtocolError(ValueError):
    """表示 profile 注册、规范化比较或流程迁移派生不合法。"""


def _nonempty_unique_strings(
    values: Sequence[Any],
    *,
    field_name: str,
) -> list[str]:
    """规范化字符串序列, 并拒绝空值或重复值。"""

    resolved = [str(value).strip() for value in values]
    if any(not value for value in resolved) or len(resolved) != len(set(resolved)):
        raise PaperProfileProtocolError(f"{field_name} 必须是非空且不重复的字符串序列")
    return resolved


def load_paper_profile_protocol_registry(
    path: str | Path = DEFAULT_PAPER_PROFILE_PROTOCOL_REGISTRY_PATH,
) -> dict[str, Any]:
    """读取并验证命令图、产物契约和 gate 角色的冻结登记。"""

    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PaperProfileProtocolError("论文 profile 协议登记表无法读取") from exc
    if not isinstance(payload, dict) or payload.get("registry_schema") != (
        "paper_profile_protocol_registry_v1"
    ):
        raise PaperProfileProtocolError("论文 profile 协议登记表 schema 不受支持")
    if tuple(payload.get("profile_ids", ())) != PAPER_PROFILE_IDS:
        raise PaperProfileProtocolError("论文 profile 集合或顺序发生漂移")
    method_role_contract = payload.get("method_role_contract")
    sample_role_contract = payload.get("sample_role_contract")
    evaluation_schema_contract = payload.get("formal_evaluation_schema_contract")
    if not all(
        isinstance(value, dict)
        for value in (
            method_role_contract,
            sample_role_contract,
            evaluation_schema_contract,
        )
    ):
        raise PaperProfileProtocolError("方法角色、样本角色和正式记录 schema 契约必须是对象")
    if method_role_contract.get("contract_schema") != (
        "dual_chain_method_role_contract_v1"
    ) or _nonempty_unique_strings(
        method_role_contract.get("role_ids", ()),
        field_name="method_role_contract.role_ids",
    ) != [
        "full_dual_chain",
        "uniform_content_routing",
        "lf_only_content",
        "hf_tail_only_content",
        "content_chain_only",
        "geometry_recovery_without_embedded_sync",
    ]:
        raise PaperProfileProtocolError("正式方法角色集合或顺序发生漂移")
    if sample_role_contract.get("contract_schema") != (
        "dual_chain_sample_role_contract_v1"
    ) or _nonempty_unique_strings(
        sample_role_contract.get("role_ids", ()),
        field_name="sample_role_contract.role_ids",
    ) != ["watermarked_positive", "clean_negative", "attacked_negative"]:
        raise PaperProfileProtocolError("正式样本角色集合或顺序发生漂移")
    if evaluation_schema_contract.get("schema_id") != (
        "dual_chain_formal_evaluation_success_failure_v1"
    ):
        raise PaperProfileProtocolError("正式 success/failure 联合 schema 身份发生漂移")
    for field_name in ("identity_fields", "union_fields"):
        _nonempty_unique_strings(
            evaluation_schema_contract.get(field_name, ()),
            field_name=f"formal_evaluation_schema_contract.{field_name}",
        )
    allowed_fields = _nonempty_unique_strings(
        payload.get("allowed_profile_variation_fields", ()),
        field_name="allowed_profile_variation_fields",
    )
    if "profile_id" not in allowed_fields or not all(
        field_name.startswith("scale_contract.")
        for field_name in allowed_fields
        if field_name != "profile_id"
    ):
        raise PaperProfileProtocolError("允许变化字段必须严格位于 profile 或规模契约")

    command_edges = payload.get("command_dependency_edges")
    gate_roles = payload.get("gate_roles")
    artifact_contract = payload.get("artifact_contract")
    diagnostic_artifact_contract = payload.get("diagnostic_artifact_contract")
    if not all(
        isinstance(value, list) and value
        for value in (command_edges, gate_roles, artifact_contract)
    ) or not isinstance(diagnostic_artifact_contract, list):
        raise PaperProfileProtocolError(
            "命令图、正式产物契约和 gate 角色不得为空, 诊断产物契约必须为列表"
        )
    edge_keys: set[tuple[str, str]] = set()
    for edge in command_edges:
        if not isinstance(edge, dict) or set(edge) != {"producer", "consumer"}:
            raise PaperProfileProtocolError("命令依赖边必须精确声明 producer 和 consumer")
        key = (str(edge["producer"]).strip(), str(edge["consumer"]).strip())
        if not all(key) or key in edge_keys:
            raise PaperProfileProtocolError("命令依赖边为空或重复")
        edge_keys.add(key)

    artifact_ids: set[str] = set()
    formal_artifact_ids: set[str] = set()
    for contract_role, records in (
        ("formal", artifact_contract),
        ("diagnostic", diagnostic_artifact_contract),
    ):
        for artifact in records:
            if not isinstance(artifact, dict) or set(artifact) != {
                "artifact_id",
                "writer_module",
                "ready_field",
                "file_names",
            }:
                raise PaperProfileProtocolError("产物契约字段集合发生漂移")
            artifact_id = str(artifact["artifact_id"]).strip()
            if not artifact_id or artifact_id in artifact_ids:
                raise PaperProfileProtocolError("产物契约标识为空或跨职责重复")
            artifact_ids.add(artifact_id)
            if contract_role == "formal":
                formal_artifact_ids.add(artifact_id)
            _nonempty_unique_strings(
                artifact.get("file_names", ()),
                field_name=f"{artifact_id}.file_names",
            )
            if not str(artifact["writer_module"]).strip() or not str(
                artifact["ready_field"]
            ).strip():
                raise PaperProfileProtocolError("产物 writer 或 ready 字段不得为空")

    gate_ids: set[str] = set()
    gate_artifact_ids: set[str] = set()
    for role in gate_roles:
        if not isinstance(role, dict) or set(role) != {
            "gate_role",
            "claim_id",
            "artifact_id",
            "requirement_role",
        }:
            raise PaperProfileProtocolError("gate 角色字段集合发生漂移")
        gate_id = str(role["gate_role"]).strip()
        artifact_id = str(role["artifact_id"]).strip()
        if (
            not gate_id
            or gate_id in gate_ids
            or artifact_id not in formal_artifact_ids
            or artifact_id in gate_artifact_ids
            or role.get("requirement_role") not in {"required_claim", "optional_claim"}
        ):
            raise PaperProfileProtocolError("gate 角色与产物契约未形成一一对应")
        gate_ids.add(gate_id)
        gate_artifact_ids.add(artifact_id)
    if gate_artifact_ids != formal_artifact_ids:
        raise PaperProfileProtocolError("gate 角色未精确覆盖正式产物")
    claim_registry = load_paper_claim_registry()
    role_by_claim = {
        str(role["claim_id"]): str(role["requirement_role"])
        for role in gate_roles
    }
    expected_role_by_claim = {
        claim_id: "required_claim"
        for claim_id in claim_registry["required_claims"]
    } | {
        claim_id: "optional_claim"
        for claim_id in claim_registry["optional_claims"]
    }
    if role_by_claim != expected_role_by_claim:
        raise PaperProfileProtocolError("gate 角色未精确匹配登记主张的必要性")
    return {
        **deepcopy(payload),
        "allowed_profile_variation_fields": allowed_fields,
    }


def registered_artifact_contract(
    artifact_id: str,
    *,
    path: str | Path = DEFAULT_PAPER_PROFILE_PROTOCOL_REGISTRY_PATH,
) -> dict[str, Any]:
    """从唯一登记表返回一个正式产物契约, 避免消费者复制文件名清单."""

    resolved_id = str(artifact_id).strip()
    registry = load_paper_profile_protocol_registry(path)
    matches = [
        dict(record)
        for record in (
            *registry["artifact_contract"],
            *registry["diagnostic_artifact_contract"],
        )
        if record["artifact_id"] == resolved_id
    ]
    if len(matches) != 1:
        raise PaperProfileProtocolError("受治理产物契约标识必须唯一存在")
    return deepcopy(matches[0])


def _baseline_protocol_definitions() -> list[dict[str, Any]]:
    """返回主表 baseline 的机制与共同输入协议身份, 不混入运行结果状态。"""

    primary_ids = set(PRIMARY_BASELINE_IDS)
    rows = [
        {
            "baseline_id": spec.baseline_id,
            "baseline_family": spec.baseline_family,
            "baseline_name": spec.baseline_name,
            "comparison_group": spec.comparison_group,
            "expected_input_mode": spec.expected_input_mode,
            "requires_gpu": spec.requires_gpu,
            "requires_training": spec.requires_training,
        }
        for spec in default_baseline_specs()
        if spec.baseline_id in primary_ids
    ]
    if [row["baseline_id"] for row in rows] != list(PRIMARY_BASELINE_IDS):
        raise PaperProfileProtocolError("主表 baseline 定义未精确匹配统计协议")
    return rows


def _shared_protocol_contract(
    root: str | Path,
    registry: Mapping[str, Any],
) -> dict[str, Any]:
    """从实际方法、攻击、统计和主张注册表重建共享协议正文。"""

    method_config = load_formal_method_runtime_config(root)
    enabled_attacks = [config for config in default_attack_configs() if config.enabled]
    attack_rows = [
        {
            **config.to_dict(),
            "attack_config_digest": attack_config_digest(config),
        }
        for config in enabled_attacks
    ]
    claim_registry = load_paper_claim_registry()
    quality_protocol = load_paper_quality_claim_protocol()
    return {
        "core_method": {
            **formal_method_config_payload(method_config),
            "formal_method_config_digest": formal_method_config_digest(method_config),
        },
        "attack_definitions": {
            "attack_records": attack_rows,
            "attack_registry_digest": build_stable_digest(attack_rows),
            **formal_attack_seed_protocol_record(),
        },
        "baseline_definitions": {
            "primary_baseline_records": _baseline_protocol_definitions(),
            "primary_baseline_ids": list(PRIMARY_BASELINE_IDS),
        },
        "formal_method_roles": deepcopy(registry["method_role_contract"]),
        "formal_sample_roles": deepcopy(registry["sample_role_contract"]),
        "formal_evaluation_schema": {
            **deepcopy(registry["formal_evaluation_schema_contract"]),
            "contract_digest": build_stable_digest(
                registry["formal_evaluation_schema_contract"]
            ),
        },
        "data_split_principle": {
            "split_names": list(SPLIT_NAMES),
            "dev_ratio": DEV_RATIO,
            "calibration_ratio": CALIBRATION_RATIO,
            "test_ratio": 1.0 - DEV_RATIO - CALIBRATION_RATIO,
            "assignment_block_size": SPLIT_ASSIGNMENT_BLOCK_SIZE,
            "assignment_rule": (
                "prompt_prefix_block_risk_stratified_digest_order"
            ),
        },
        "threshold_calibration_principle": {
            "threshold_source": "dual_chain_nested_three_negative_group_budget_v1",
            "calibration_partition_protocol": "dual_chain_nested_calibration_v1",
            "confidence_level": PAPER_CONFIDENCE_LEVEL,
            "calibration_negative_roles": [
                "clean_negative_registered",
                "attacked_negative_registered",
                "watermarked_wrong_key",
            ],
            "negative_role_budget_rule": (
                "max(0,floor(target_fpr*(negative_role_sample_count+1))-1)"
            ),
            "negative_role_budget_scope": "each_role_independently",
            "threshold_reuse_scope": "same_method_same_repeat_calibration_to_test",
        },
        "metric_semantics": {
            "detection_statistical_unit": RANDOMIZATION_DETECTION_STATISTICAL_UNIT,
            "detection_confidence_interval_method": (
                RANDOMIZATION_DETECTION_CONFIDENCE_INTERVAL_METHOD
            ),
            "false_positive_bound_method": (
                RANDOMIZATION_DETECTION_FALSE_POSITIVE_BOUND_METHOD
            ),
            "paired_superiority_analysis_schema": BOOTSTRAP_ANALYSIS_SCHEMA,
            "paired_superiority_claim_p_value_method": CLAIM_P_VALUE_METHOD,
            "quality_matching_protocol_schema": QUALITY_MATCHING_PROTOCOL_SCHEMA,
            "quality_matching_metric_name": QUALITY_MATCHING_METRIC_NAME,
            "quality_matching_caliper": QUALITY_MATCHING_CALIPER,
            "quality_matching_minimum_fraction": QUALITY_MATCHING_MINIMUM_FRACTION,
            "paper_quality_claim_protocol": quality_protocol,
        },
        "randomization_protocol": {
            **formal_randomization_protocol_record(),
            "registered_repeat_ids": list(formal_randomization_repeat_ids()),
            "registered_repeat_registry_digest": (
                formal_randomization_repeat_registry_digest()
            ),
        },
        "command_dependency_edges": deepcopy(
            registry["command_dependency_edges"]
        ),
        "gate_roles": deepcopy(registry["gate_roles"]),
        "claim_decision_structure": claim_registry,
    }


def _scale_contract(profile_id: str) -> dict[str, Any]:
    """返回一个论文运行规模的全部允许变化字段。"""

    defaults = RUN_DEFAULTS[profile_id]
    prompt_count = int(RUN_EXPECTED_PROMPT_COUNTS[profile_id])
    target_fpr = float(defaults["target_fpr"])
    split_counts = build_group_split_counts(prompt_count)
    return {
        "protocol_profile": str(defaults["protocol_profile"]),
        "target_fpr": target_fpr,
        "prompt_set": str(defaults["prompt_set"]),
        "prompt_file": str(defaults["prompt_file"]),
        "prompt_count": prompt_count,
        "sample_count": str(defaults["sample_count"]),
        "split_counts": split_counts,
        "minimum_negative_role_counts": {
            role_id: derive_minimum_clean_negative_count(prompt_count, target_fpr)
            for role_id in (
                "clean_negative_registered",
                "attacked_negative_registered",
                "watermarked_wrong_key",
            )
        },
        "dataset_level_quality_minimum_count": (
            derive_dataset_level_quality_minimum_count(prompt_count)
        ),
        "record_count_derivation": {
            "prompt_primary_unit_count": prompt_count,
            "registered_repeat_count": len(formal_randomization_repeat_ids()),
            "test_prompt_count": int(split_counts["test"]),
        },
    }


def _operational_metadata(profile_id: str) -> dict[str, str]:
    """从 profile 身份确定性派生输出位置, 不把路径计入科学规模契约."""

    return {
        "drive_result_root": str(RUN_DEFAULTS[profile_id]["drive_result_root"]),
        "derivation_rule": "paper_run_defaults_by_profile_id",
    }


def build_paper_profile_protocol_records(
    root: str | Path = ".",
    *,
    registry: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """从唯一注册源构造 probe、pilot 与 full 的独立可比较协议记录。"""

    resolved_registry = dict(registry or load_paper_profile_protocol_registry())
    shared_protocol = _shared_protocol_contract(root, resolved_registry)
    artifact_contract = deepcopy(
        [
            *resolved_registry["artifact_contract"],
            *resolved_registry["diagnostic_artifact_contract"],
        ]
    )
    return {
        profile_id: {
            "profile_id": profile_id,
            "scale_contract": _scale_contract(profile_id),
            "operational_metadata": _operational_metadata(profile_id),
            "protocol_contract": deepcopy(shared_protocol),
            "artifact_contract": deepcopy(artifact_contract),
        }
        for profile_id in PAPER_PROFILE_IDS
    }


def _normalize_profile_records(
    records: Mapping[str, Mapping[str, Any]],
    *,
    registry: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """校验 profile 记录结构, 禁止把未登记变化隐藏在规模字段之外。"""

    if set(records) != set(PAPER_PROFILE_IDS):
        raise PaperProfileProtocolError("profile 记录未精确覆盖三个论文运行规模")
    expected_scale_fields = {
        field_name.removeprefix("scale_contract.")
        for field_name in registry["allowed_profile_variation_fields"]
        if field_name.startswith("scale_contract.")
    }
    normalized: dict[str, dict[str, Any]] = {}
    for profile_id in PAPER_PROFILE_IDS:
        source = records[profile_id]
        if not isinstance(source, Mapping) or set(source) != {
            "profile_id",
            "scale_contract",
            "operational_metadata",
            "protocol_contract",
            "artifact_contract",
        }:
            raise PaperProfileProtocolError("profile 协议记录字段集合发生漂移")
        if source.get("profile_id") != profile_id:
            raise PaperProfileProtocolError("profile 映射键与记录身份不一致")
        if source.get("operational_metadata") != _operational_metadata(profile_id):
            raise PaperProfileProtocolError("操作元数据未由 profile 身份确定性派生")
        scale_contract = source.get("scale_contract")
        if not isinstance(scale_contract, Mapping) or set(scale_contract) != (
            expected_scale_fields
        ):
            raise PaperProfileProtocolError("规模契约未精确匹配允许变化字段")
        if not isinstance(source.get("protocol_contract"), Mapping) or not isinstance(
            source.get("artifact_contract"), Sequence
        ) or isinstance(source.get("artifact_contract"), (str, bytes)):
            raise PaperProfileProtocolError("协议正文或产物契约类型无效")
        normalized[profile_id] = deepcopy(dict(source))
    return normalized


def _difference_paths(left: Any, right: Any, *, path: str = "") -> list[str]:
    """返回两个 JSON 兼容结构的全部差异路径。"""

    if isinstance(left, Mapping) and isinstance(right, Mapping):
        differences: list[str] = []
        for key in sorted(set(left).union(right)):
            child_path = f"{path}.{key}" if path else str(key)
            if key not in left or key not in right:
                differences.append(child_path)
            else:
                differences.extend(
                    _difference_paths(left[key], right[key], path=child_path)
                )
        return differences
    if (
        isinstance(left, Sequence)
        and isinstance(right, Sequence)
        and not isinstance(left, (str, bytes))
        and not isinstance(right, (str, bytes))
    ):
        differences = []
        for index in range(max(len(left), len(right))):
            child_path = f"{path}[{index}]"
            if index >= len(left) or index >= len(right):
                differences.append(child_path)
            else:
                differences.extend(
                    _difference_paths(left[index], right[index], path=child_path)
                )
        return differences
    return [] if left == right else [path or "$"]


def _probe_closure_binding(
    report: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """提取 probe 流程事实与分主张决策, 不用科学结果反推流程状态。"""

    claim_bundle = validate_claim_decision_bundle(report)
    profile_candidates = {
        str(value)
        for value in (report.get("paper_run_name"), report.get("paper_claim_scale"))
        if value is not None
    }
    profile_identity_ready = profile_candidates == {PROBE_PAPER_RUN_NAME}
    try:
        target_fpr = float(report.get("target_fpr"))
    except (TypeError, ValueError):
        target_fpr = math.nan
    target_fpr_ready = math.isclose(
        target_fpr,
        float(RUN_DEFAULTS[PROBE_PAPER_RUN_NAME]["target_fpr"]),
        rel_tol=0.0,
        abs_tol=1e-12,
    )
    closure_check_count = report.get("closure_check_count")
    blocked_check_count = report.get("blocked_check_count")
    check_counts_ready = (
        type(closure_check_count) is int
        and closure_check_count > 0
        and type(blocked_check_count) is int
        and blocked_check_count == 0
    )
    binding = {
        "paper_run_name": PROBE_PAPER_RUN_NAME if profile_identity_ready else None,
        "target_fpr": target_fpr if math.isfinite(target_fpr) else None,
        "common_code_version": str(report.get("common_code_version", "")),
        "profile_identity_ready": profile_identity_ready,
        "target_fpr_ready": target_fpr_ready,
        "closure_check_count": closure_check_count,
        "blocked_check_count": blocked_check_count,
        "evidence_closure_allowed": report.get("evidence_closure_allowed") is True,
        "result_closure_ready": report.get("result_closure_ready") is True,
        **claim_bundle,
    }
    return binding, claim_bundle


def _profile_scientific_scope(
    probe_claim_bundle: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """明确限制 probe 科学结论的 FPR 作用域, 不向更严格工作点外推。"""

    scopes: dict[str, dict[str, Any]] = {}
    for profile_id in PAPER_PROFILE_IDS:
        target_fpr = float(RUN_DEFAULTS[profile_id]["target_fpr"])
        if profile_id == PROBE_PAPER_RUN_NAME:
            decision = probe_claim_bundle["registered_claim_set_decision"]
            support = probe_claim_bundle["registered_claim_set_scientific_support"]
            evidence_source = "probe_result_closure_report"
        else:
            decision = "evidence_incomplete"
            support = None
            evidence_source = "not_inferred_from_probe"
        scopes[profile_id] = {
            "profile_id": profile_id,
            "target_fpr": target_fpr,
            "registered_claim_set_decision": decision,
            "registered_claim_set_scientific_support": support,
            "evidence_source": evidence_source,
            "scientific_support_transferred_from_probe": False,
        }
    return scopes


def build_paper_profile_protocol_isomorphism_report(
    probe_result_closure_report: Mapping[str, Any],
    *,
    root: str | Path = ".",
    profile_records: Mapping[str, Mapping[str, Any]] | None = None,
    registry: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """比较规范化协议并独立派生流程迁移状态与科学结论作用域。"""

    resolved_registry = dict(registry or load_paper_profile_protocol_registry())
    records = _normalize_profile_records(
        profile_records
        or build_paper_profile_protocol_records(root, registry=resolved_registry),
        registry=resolved_registry,
    )
    reference = records[PROBE_PAPER_RUN_NAME]
    protocol_differences: dict[str, list[str]] = {}
    artifact_differences: dict[str, list[str]] = {}
    for profile_id in PAPER_PROFILE_IDS[1:]:
        protocol_differences[profile_id] = _difference_paths(
            reference["protocol_contract"],
            records[profile_id]["protocol_contract"],
        )
        artifact_differences[profile_id] = _difference_paths(
            reference["artifact_contract"],
            records[profile_id]["artifact_contract"],
        )
    protocol_isomorphism_ready = not any(protocol_differences.values())
    artifact_contract_isomorphic = not any(artifact_differences.values())

    expected_scales = {
        profile_id: _scale_contract(profile_id)
        for profile_id in PAPER_PROFILE_IDS
    }
    scale_registration_differences = {
        profile_id: _difference_paths(
            expected_scales[profile_id],
            records[profile_id]["scale_contract"],
        )
        for profile_id in PAPER_PROFILE_IDS
    }
    profile_scale_registration_ready = not any(
        scale_registration_differences.values()
    )
    probe_binding, probe_claim_bundle = _probe_closure_binding(
        probe_result_closure_report
    )
    probe_workflow_closed = all(
        (
            probe_binding["profile_identity_ready"],
            probe_binding["target_fpr_ready"],
            probe_binding["evidence_closure_allowed"],
            probe_binding["result_closure_ready"],
            type(probe_binding["closure_check_count"]) is int
            and probe_binding["closure_check_count"] > 0,
            probe_binding["blocked_check_count"] == 0,
        )
    )
    workflow_transfer_ready = (
        probe_workflow_closed
        and protocol_isomorphism_ready
        and artifact_contract_isomorphic
    )
    core = {
        "report_schema": "paper_profile_protocol_isomorphism_report_v1",
        "profile_ids": list(PAPER_PROFILE_IDS),
        "paper_profile_protocol_registry_digest": build_stable_digest(
            resolved_registry
        ),
        "allowed_profile_variation_fields": list(
            resolved_registry["allowed_profile_variation_fields"]
        ),
        "profile_records": records,
        "normalized_protocol_digests": {
            profile_id: build_stable_digest(records[profile_id]["protocol_contract"])
            for profile_id in PAPER_PROFILE_IDS
        },
        "artifact_contract_digests": {
            profile_id: build_stable_digest(records[profile_id]["artifact_contract"])
            for profile_id in PAPER_PROFILE_IDS
        },
        "profile_record_digests": {
            profile_id: build_stable_digest(records[profile_id])
            for profile_id in PAPER_PROFILE_IDS
        },
        "protocol_difference_paths": protocol_differences,
        "artifact_contract_difference_paths": artifact_differences,
        "scale_registration_difference_paths": scale_registration_differences,
        "profile_scale_registration_ready": profile_scale_registration_ready,
        "probe_result_closure_binding": probe_binding,
        "probe_workflow_closed": probe_workflow_closed,
        "protocol_isomorphism_ready": protocol_isomorphism_ready,
        "artifact_contract_isomorphic": artifact_contract_isomorphic,
        "workflow_transfer_ready": workflow_transfer_ready,
        "workflow_transfer_basis": {
            "probe_workflow_closed": probe_workflow_closed,
            "protocol_isomorphism_ready": protocol_isomorphism_ready,
            "artifact_contract_isomorphic": artifact_contract_isomorphic,
        },
        "scientific_scope_by_profile": _profile_scientific_scope(
            probe_claim_bundle
        ),
        "workflow_transfer_boundary": (
            "same_code_and_protocol_execution_only_not_scientific_effect_transfer"
        ),
    }
    return {
        **core,
        "paper_profile_protocol_isomorphism_report_digest": build_stable_digest(core),
    }


def validate_paper_profile_protocol_isomorphism_report(
    report: Mapping[str, Any],
    *,
    root: str | Path = ".",
    registry: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """从报告内的协议记录和 probe 绑定重算全部派生状态与摘要。"""

    binding = report.get("probe_result_closure_binding")
    records = report.get("profile_records")
    if not isinstance(binding, Mapping) or not isinstance(records, Mapping):
        raise PaperProfileProtocolError("同构报告缺少可重算的 profile 或 probe 绑定")
    rebuilt = build_paper_profile_protocol_isomorphism_report(
        binding,
        root=root,
        profile_records=records,
        registry=registry,
    )
    if dict(report) != rebuilt:
        raise PaperProfileProtocolError("同构报告字段、派生状态或摘要与重算结果不一致")
    return rebuilt


__all__ = [
    "DEFAULT_PAPER_PROFILE_PROTOCOL_REGISTRY_PATH",
    "PAPER_PROFILE_IDS",
    "PaperProfileProtocolError",
    "build_paper_profile_protocol_isomorphism_report",
    "build_paper_profile_protocol_records",
    "load_paper_profile_protocol_registry",
    "validate_paper_profile_protocol_isomorphism_report",
]
