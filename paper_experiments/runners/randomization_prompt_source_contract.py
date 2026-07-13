"""从精确聚合包内嵌来源逐字节重建受治理 Prompt 契约.

该模块只消费已经进入 ``RandomizationAggregateRecordWorkspace`` 的固定成员.
它不读取当前仓库的 Prompt 文件, 也不接受调用方提供文本、路径或成员覆盖.
因此后续统计绑定的是 GPU 结果包实际携带的来源注册表、选择清单和 Prompt
文件字节, 而不是9份 runtime records 之间的内部一致性.
"""

from __future__ import annotations

import hashlib
from typing import Any

from experiments.protocol.formal_randomization import (
    formal_randomization_repeat_ids,
)
from experiments.protocol.prompt_sources import (
    PROMPT_CONFIG_NAMES,
    PROMPT_SELECTION_MANIFEST_DIGEST,
    PROMPT_SELECTION_MANIFEST_SHA256,
    audit_packaged_prompt_set_bytes,
)
from experiments.protocol.prompts import build_prompt_records
from experiments.protocol.splits import apply_split_assignments
from main.core.digest import build_stable_digest
from paper_experiments.runners.randomization_aggregate_provenance import (
    RandomizationAggregateProvenance,
)
from paper_experiments.runners.randomization_aggregate_record_workspace import (
    RandomizationAggregateRecordSource,
    RandomizationAggregateRecordWorkspace,
)


PROMPT_SOURCE_CONTRACT_REPORT_SCHEMA = (
    "randomization_prompt_source_contract_report"
)
_PROMPT_FILE_ROLE = "governed_prompt_file_bytes"
_SELECTION_MANIFEST_ROLE = "governed_prompt_selection_manifest_bytes"
_SOURCE_REGISTRY_ROLE = "governed_prompt_source_registry_bytes"


class RandomizationPromptSourceContractError(ValueError):
    """表示 aggregate 内嵌 Prompt 来源不能形成冻结运行契约."""


def _require_source_identity(
    source: RandomizationAggregateRecordSource,
    provenance: RandomizationAggregateProvenance,
    *,
    repeat_id: str,
    expected_member: str,
) -> None:
    """核对一个 Prompt 来源成员的完整 aggregate 摘要链."""

    repeat_record = next(
        (
            record
            for record in provenance.randomization_repeat_components
            if record.get("randomization_repeat_id") == repeat_id
        ),
        None,
    )
    if repeat_record is None or not all(
        (
            source.randomization_scope == "active_repeat_component",
            source.randomization_repeat_id == repeat_id,
            source.package_family == "image_only_dataset_runtime",
            source.record_member == expected_member,
            source.randomization_aggregate_package_sha256
            == provenance.package_sha256,
            source.randomization_aggregate_digest
            == provenance.randomization_aggregate_digest,
            source.common_code_version == provenance.common_code_version,
            source.randomization_repeat_component_sha256
            == repeat_record.get("package_sha256"),
            source.randomization_repeat_evidence_manifest_digest
            == repeat_record.get(
                "randomization_repeat_evidence_manifest_digest"
            ),
            source.component_content_digest
            == repeat_record.get("component_content_digest"),
        )
    ):
        raise RandomizationPromptSourceContractError(
            "Prompt 来源成员没有绑定精确 aggregate 摘要链"
        )


def _prompt_rows_from_bytes(
    prompt_file_payload: bytes,
    *,
    paper_run_name: str,
) -> tuple[dict[str, Any], ...]:
    """从已通过逐字节来源审计的文件构造 Prompt 和 split 记录."""

    try:
        prompt_text = prompt_file_payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RandomizationPromptSourceContractError(
            "受治理 Prompt 文件不是 UTF-8"
        ) from exc
    if not prompt_text.endswith("\n") or "\r" in prompt_text:
        raise RandomizationPromptSourceContractError(
            "受治理 Prompt 文件必须使用末尾换行的 UTF-8 LF 字节"
        )
    prompt_texts = tuple(prompt_text.splitlines())
    records = apply_split_assignments(
        build_prompt_records(paper_run_name, prompt_texts)
    )
    return tuple(
        {
            "prompt_id": record.prompt_id,
            "prompt_index": record.prompt_index,
            "prompt_text": record.prompt_text,
            "prompt_digest": record.prompt_digest,
            "split": record.split,
            "semantic_tags": list(record.semantic_tags),
            "risk_profile": record.risk_profile,
        }
        for record in records
    )


def rebuild_randomization_prompt_source_contract(
    workspace: RandomizationAggregateRecordWorkspace,
    provenance: RandomizationAggregateProvenance,
    *,
    paper_run_name: str,
) -> dict[str, Any]:
    """从9份内嵌 Prompt 来源重建唯一运行文本与来源报告."""

    if not isinstance(provenance, RandomizationAggregateProvenance):
        raise TypeError("Prompt 来源重建只接受 aggregate provenance")
    repeat_ids = formal_randomization_repeat_ids()
    if len(workspace.prompt_source_sources) != len(repeat_ids) * 3:
        raise RandomizationPromptSourceContractError(
            "Prompt 来源必须精确覆盖9个重复的三份内嵌原始事实"
        )

    canonical_prompt_rows: tuple[dict[str, Any], ...] | None = None
    canonical_audit_identity: tuple[Any, ...] | None = None
    source_records: list[dict[str, Any]] = []
    for repeat_id in repeat_ids:
        expected_members = {
            _PROMPT_FILE_ROLE: (
                f"outputs/image_only_dataset_runtime/{paper_run_name}/"
                "prompt_source_snapshot/"
                f"{PROMPT_CONFIG_NAMES[paper_run_name]}"
            ),
            _SELECTION_MANIFEST_ROLE: (
                f"outputs/image_only_dataset_runtime/{paper_run_name}/"
                "prompt_source_snapshot/prompt_selection_manifest.jsonl"
            ),
            _SOURCE_REGISTRY_ROLE: (
                f"outputs/image_only_dataset_runtime/{paper_run_name}/"
                "prompt_source_snapshot/prompt_source_registry.json"
            ),
        }
        sources = {
            role: workspace.find_source(
                randomization_repeat_id=repeat_id,
                package_family="image_only_dataset_runtime",
                record_role=role,
            )
            for role in expected_members
        }
        for role, source in sources.items():
            _require_source_identity(
                source,
                provenance,
                repeat_id=repeat_id,
                expected_member=expected_members[role],
            )
        if len(
            {
                (
                    source.leaf_package_sha256,
                    source.randomization_repeat_component_sha256,
                    source.randomization_repeat_evidence_manifest_digest,
                    source.component_content_digest,
                )
                for source in sources.values()
            }
        ) != 1:
            raise RandomizationPromptSourceContractError(
                "同一 repeat 的三份 Prompt 来源没有绑定同一 runtime leaf"
            )
        prompt_file_payload = workspace.read_bytes(sources[_PROMPT_FILE_ROLE])
        selection_manifest_payload = workspace.read_bytes(
            sources[_SELECTION_MANIFEST_ROLE]
        )
        source_registry_payload = workspace.read_bytes(
            sources[_SOURCE_REGISTRY_ROLE]
        )
        source_payloads = {
            _PROMPT_FILE_ROLE: prompt_file_payload,
            _SELECTION_MANIFEST_ROLE: selection_manifest_payload,
            _SOURCE_REGISTRY_ROLE: source_registry_payload,
        }
        if any(
            hashlib.sha256(source_payloads[role]).hexdigest()
            != sources[role].record_sha256
            for role in source_payloads
        ):
            raise RandomizationPromptSourceContractError(
                "Prompt 来源成员原始字节与描述符摘要不一致"
            )
        audit = audit_packaged_prompt_set_bytes(
            prompt_set=paper_run_name,
            prompt_file_payload=prompt_file_payload,
            selection_manifest_payload=selection_manifest_payload,
            source_registry_payload=source_registry_payload,
        )
        prompt_rows = _prompt_rows_from_bytes(
            prompt_file_payload,
            paper_run_name=paper_run_name,
        )
        audit_identity = (
            audit["prompt_file_sha256"],
            audit["prompt_source_registry_digest"],
            audit["selection_manifest_sha256"],
            audit["selection_manifest_digest"],
            audit["packaged_prompt_source_audit_digest"],
            build_stable_digest(prompt_rows),
        )
        if canonical_prompt_rows is None:
            canonical_prompt_rows = prompt_rows
            canonical_audit_identity = audit_identity
        elif (
            prompt_rows != canonical_prompt_rows
            or audit_identity != canonical_audit_identity
        ):
            raise RandomizationPromptSourceContractError(
                "9个 repeat 的内嵌 Prompt 来源字节或审计身份不一致"
            )
        source_records.append(
            {
                "randomization_repeat_id": repeat_id,
                "prompt_file_member": sources[
                    _PROMPT_FILE_ROLE
                ].record_member,
                "prompt_file_sha256": sources[
                    _PROMPT_FILE_ROLE
                ].record_sha256,
                "selection_manifest_member": sources[
                    _SELECTION_MANIFEST_ROLE
                ].record_member,
                "selection_manifest_sha256": sources[
                    _SELECTION_MANIFEST_ROLE
                ].record_sha256,
                "prompt_source_registry_member": sources[
                    _SOURCE_REGISTRY_ROLE
                ].record_member,
                "prompt_source_registry_sha256": sources[
                    _SOURCE_REGISTRY_ROLE
                ].record_sha256,
                "leaf_package_sha256": sources[
                    _PROMPT_FILE_ROLE
                ].leaf_package_sha256,
                "randomization_repeat_component_sha256": sources[
                    _PROMPT_FILE_ROLE
                ].randomization_repeat_component_sha256,
                "randomization_repeat_evidence_manifest_digest": sources[
                    _PROMPT_FILE_ROLE
                ].randomization_repeat_evidence_manifest_digest,
                "component_content_digest": sources[
                    _PROMPT_FILE_ROLE
                ].component_content_digest,
            }
        )

    if canonical_prompt_rows is None or canonical_audit_identity is None:
        raise RandomizationPromptSourceContractError(
            "Prompt 来源重建没有产生规范记录"
        )
    report = {
        "report_schema": PROMPT_SOURCE_CONTRACT_REPORT_SCHEMA,
        "paper_claim_scale": paper_run_name,
        "prompt_count": len(canonical_prompt_rows),
        "prompt_file_sha256": canonical_audit_identity[0],
        "prompt_source_registry_digest": canonical_audit_identity[1],
        "selection_manifest_sha256": PROMPT_SELECTION_MANIFEST_SHA256,
        "selection_manifest_digest": PROMPT_SELECTION_MANIFEST_DIGEST,
        "packaged_prompt_source_audit_digest": canonical_audit_identity[4],
        "prompt_rows_digest": canonical_audit_identity[5],
        "prompt_source_record_map": {
            str(record["randomization_repeat_id"]): record
            for record in source_records
        },
        "prompt_source_records_digest": build_stable_digest(source_records),
        "prompt_source_contract_ready": True,
        "supports_paper_claim": False,
    }
    report["prompt_source_contract_digest"] = build_stable_digest(report)
    return {
        "prompt_rows": canonical_prompt_rows,
        "report": report,
    }


__all__ = [
    "PROMPT_SOURCE_CONTRACT_REPORT_SCHEMA",
    "RandomizationPromptSourceContractError",
    "rebuild_randomization_prompt_source_contract",
]
