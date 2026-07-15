"""集中构造和复验论文分主张三态决策。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from pathlib import Path
from typing import Any

from main.core.digest import build_stable_digest


CLAIM_DECISION_STATES = (
    "supported",
    "measured_not_supported",
    "evidence_incomplete",
)
DEFAULT_PAPER_CLAIM_REGISTRY_PATH = (
    Path(__file__).resolve().parents[2] / "configs" / "paper_claim_registry.json"
)


class ClaimDecisionGovernanceError(ValueError):
    """表示主张登记、三态派生或兼容字段绑定不合法。"""


def _nonempty_unique_strings(values: Sequence[Any], *, field_name: str) -> list[str]:
    """把协议字符串序列规范化, 同时拒绝空值和重复值。"""

    resolved = [str(value).strip() for value in values]
    if any(not value for value in resolved) or len(resolved) != len(set(resolved)):
        raise ClaimDecisionGovernanceError(f"{field_name} 必须是非空且不重复的字符串序列")
    return resolved


def load_paper_claim_registry(
    path: str | Path = DEFAULT_PAPER_CLAIM_REGISTRY_PATH,
) -> dict[str, Any]:
    """读取并验证冻结的论文主张登记表。"""

    registry_path = Path(path)
    try:
        payload = json.loads(registry_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ClaimDecisionGovernanceError("论文主张登记表无法读取") from exc
    if not isinstance(payload, dict):
        raise ClaimDecisionGovernanceError("论文主张登记表必须是 JSON 对象")
    if payload.get("registry_schema") != "paper_claim_registry_v1":
        raise ClaimDecisionGovernanceError("论文主张登记表 schema 不受支持")
    registered = _nonempty_unique_strings(
        payload.get("registered_claim_ids", ()),
        field_name="registered_claim_ids",
    )
    required = _nonempty_unique_strings(
        payload.get("required_claims", ()),
        field_name="required_claims",
    )
    optional = _nonempty_unique_strings(
        payload.get("optional_claims", ()),
        field_name="optional_claims",
    )
    if set(required).intersection(optional) or set(required).union(optional) != set(registered):
        raise ClaimDecisionGovernanceError("required_claims 与 optional_claims 未精确覆盖登记集合")
    if tuple(payload.get("decision_states", ())) != CLAIM_DECISION_STATES:
        raise ClaimDecisionGovernanceError("论文主张三态集合或顺序发生漂移")
    if (
        payload.get("compatibility_field") != "supports_paper_claim"
        or payload.get("compatibility_derivation")
        != "registered_claim_set_supported"
    ):
        raise ClaimDecisionGovernanceError("兼容结论字段未绑定登记主张集合")
    return {
        **payload,
        "registered_claim_ids": registered,
        "required_claims": required,
        "optional_claims": optional,
    }


def build_claim_decision(
    claim_id: str,
    *,
    evidence_complete: bool,
    scientific_support: bool | None,
    evidence_artifact_ids: Sequence[str],
    evidence_blockers: Sequence[str] = (),
) -> dict[str, Any]:
    """由证据完整性和科学判据结果唯一派生一个主张决策。"""

    resolved_claim_id = str(claim_id).strip()
    if not resolved_claim_id:
        raise ClaimDecisionGovernanceError("claim_id 不得为空")
    if not isinstance(evidence_complete, bool):
        raise ClaimDecisionGovernanceError("evidence_complete 必须是严格布尔值")
    artifact_ids = _nonempty_unique_strings(
        evidence_artifact_ids,
        field_name="evidence_artifact_ids",
    )
    blockers = _nonempty_unique_strings(
        evidence_blockers,
        field_name="evidence_blockers",
    ) if evidence_blockers else []
    if evidence_complete:
        if not isinstance(scientific_support, bool):
            raise ClaimDecisionGovernanceError("完整证据必须给出严格布尔科学判定")
        if blockers:
            raise ClaimDecisionGovernanceError("完整证据不得同时登记证据缺失原因")
        decision = "supported" if scientific_support else "measured_not_supported"
    else:
        if scientific_support is not None:
            raise ClaimDecisionGovernanceError("证据不完整时不得伪造科学支持判定")
        if not blockers:
            raise ClaimDecisionGovernanceError("证据不完整时必须登记缺失原因")
        decision = "evidence_incomplete"
    core = {
        "claim_id": resolved_claim_id,
        "evidence_complete": evidence_complete,
        "scientific_support": scientific_support,
        "decision": decision,
        "evidence_artifact_ids": artifact_ids,
        "evidence_blockers": blockers,
    }
    return {**core, "claim_decision_digest": build_stable_digest(core)}


def _validate_claim_decision(record: Mapping[str, Any]) -> dict[str, Any]:
    """重算单项决策并拒绝字段或摘要伪造。"""

    rebuilt = build_claim_decision(
        str(record.get("claim_id", "")),
        evidence_complete=record.get("evidence_complete"),
        scientific_support=record.get("scientific_support"),
        evidence_artifact_ids=record.get("evidence_artifact_ids", ()),
        evidence_blockers=record.get("evidence_blockers", ()),
    )
    if dict(record) != rebuilt:
        raise ClaimDecisionGovernanceError("主张决策字段或摘要与重算结果不一致")
    return rebuilt


def build_claim_decision_bundle(
    claim_decisions: Mapping[str, Mapping[str, Any]],
    *,
    registry: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """按登记的 required_claims 汇总三态决策, optional 主张不越权投票。"""

    resolved_registry = dict(registry or load_paper_claim_registry())
    registered_claim_ids = list(resolved_registry["registered_claim_ids"])
    required_claims = list(resolved_registry["required_claims"])
    optional_claims = list(resolved_registry["optional_claims"])
    if set(claim_decisions) != set(registered_claim_ids):
        raise ClaimDecisionGovernanceError("claim_decisions 未精确覆盖登记主张集合")
    normalized: dict[str, dict[str, Any]] = {}
    for claim_id in registered_claim_ids:
        record = _validate_claim_decision(claim_decisions[claim_id])
        if record["claim_id"] != claim_id:
            raise ClaimDecisionGovernanceError("主张映射键与 claim_id 不一致")
        normalized[claim_id] = record
    required_records = [normalized[claim_id] for claim_id in required_claims]
    evidence_complete = all(record["evidence_complete"] for record in required_records)
    registered_supported = evidence_complete and all(
        record["scientific_support"] is True for record in required_records
    )
    if not evidence_complete:
        set_decision = "evidence_incomplete"
        set_scientific_support: bool | None = None
    elif registered_supported:
        set_decision = "supported"
        set_scientific_support = True
    else:
        set_decision = "measured_not_supported"
        set_scientific_support = False
    return {
        "paper_claim_registry_digest": build_stable_digest(resolved_registry),
        "registered_claim_ids": registered_claim_ids,
        "required_claims": required_claims,
        "optional_claims": optional_claims,
        "claim_decisions": normalized,
        "registered_claim_set_evidence_complete": evidence_complete,
        "registered_claim_set_scientific_support": set_scientific_support,
        "registered_claim_set_decision": set_decision,
        "registered_claim_set_supported": registered_supported,
        "conclusion_decision": set_decision,
        "supports_paper_claim": registered_supported,
    }


def validate_claim_decision_bundle(
    payload: Mapping[str, Any],
    *,
    registry: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """从单项决策重算集合结论, 防止 report 或 manifest 改写兼容布尔值。"""

    decisions = payload.get("claim_decisions")
    if not isinstance(decisions, Mapping):
        raise ClaimDecisionGovernanceError("缺少结构化 claim_decisions")
    rebuilt = build_claim_decision_bundle(decisions, registry=registry)
    for field_name, expected in rebuilt.items():
        if payload.get(field_name) != expected:
            raise ClaimDecisionGovernanceError(
                f"{field_name} 与登记主张集合重算结果不一致"
            )
    return rebuilt
