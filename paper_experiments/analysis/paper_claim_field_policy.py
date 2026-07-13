"""定义单重复证据中正式论文结论字段的允许负值协议.

该模块不根据字段名子串、Python truthiness 或自然语言关键词猜测结论语义.
每个受治理字段都必须显式登记, 且单重复组件只允许保存登记的负值. 这一
设计可同时识别布尔结论和枚举结论, 又不会把普通组件状态误判为论文结论.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import io
import json
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Mapping
from zipfile import ZipFile


@dataclass(frozen=True)
class PaperClaimFieldPolicy:
    """保存一个正式论文结论字段在单重复组件中的允许负值."""

    field_name: str
    allowed_scalar_values: tuple[Any, ...] = ()
    allowed_text_values: frozenset[str] = frozenset()

    def allows_component_value(self, value: Any) -> bool:
        """按显式类型和值判断当前字段是否仍保持负结论语义."""

        if isinstance(value, str):
            return value.strip().lower() in self.allowed_text_values
        return any(
            type(value) is type(allowed) and value == allowed
            for allowed in self.allowed_scalar_values
        )


@dataclass(frozen=True)
class PaperClaimFieldViolation:
    """记录递归扫描发现的第一个越权正向结论字段."""

    path: str
    field_name: str
    value: Any


_BOOLEAN_NEGATIVE_SCALARS = (False,)
_BOOLEAN_NEGATIVE_TEXT = frozenset({"false"})
_COUNT_NEGATIVE_SCALARS = (0,)
_COUNT_NEGATIVE_TEXT = frozenset({"0"})


def _boolean_policy(field_name: str) -> PaperClaimFieldPolicy:
    """构造只允许显式布尔负值的字段策略."""

    return PaperClaimFieldPolicy(
        field_name=field_name,
        allowed_scalar_values=_BOOLEAN_NEGATIVE_SCALARS,
        allowed_text_values=_BOOLEAN_NEGATIVE_TEXT,
    )


def _count_policy(field_name: str) -> PaperClaimFieldPolicy:
    """构造只允许零计数的字段策略."""

    return PaperClaimFieldPolicy(
        field_name=field_name,
        allowed_scalar_values=_COUNT_NEGATIVE_SCALARS,
        allowed_text_values=_COUNT_NEGATIVE_TEXT,
    )


_BOOLEAN_PAPER_CLAIM_FIELDS = (
    "ablation_standalone_claim_ready",
    "attack_manifest_supports_paper_claim",
    "baseline_claim_ready",
    "comparison_table_supports_paper_claim",
    "evidence_closure_allowed",
    "formal_result_claim",
    "full_claim_ready",
    "full_paper_claim_ready",
    "package_freeze_allowed",
    "paper_artifact_claim_ready",
    "paper_claim_support",
    "paper_claim_ready",
    "paper_claim_supported",
    "paper_ready",
    "paper_run_claim_ready",
    "paper_run_supports_superiority_claim",
    "pilot_claim_ready",
    "pilot_paper_claim_ready",
    "pilot_paper_claim_record_ready",
    "pilot_paper_supports_superiority_claim",
    "probe_claim_ready",
    "probe_paper_claim_ready",
    "raw_content_claim_ready",
    "release_package_allowed",
    "source_supports_paper_claim",
    "submission_ready",
    "strong_ablation_standalone_claim_ready",
    "superiority_claim_ready",
    "supports_main_table_superiority_claim",
    "supports_paper_claim",
    "supports_quality_matched_paper_claim",
    "universal_per_attack_superiority_claim_ready",
)

_ENUM_PAPER_CLAIM_POLICIES = (
    PaperClaimFieldPolicy(
        field_name="claim_decision",
        allowed_text_values=frozenset(
            {
                "engineering_supported_not_paper_final",
                "preview_only",
                "unsupported",
            }
        ),
    ),
    PaperClaimFieldPolicy(
        field_name="entry_review_decision",
        allowed_text_values=frozenset(
            {"blocked_before_evidence_closure"}
        ),
    ),
    PaperClaimFieldPolicy(
        field_name="readiness_decision",
        allowed_text_values=frozenset({"blocked"}),
    ),
    PaperClaimFieldPolicy(
        field_name="closure_decision",
        allowed_text_values=frozenset({"blocked"}),
    ),
)

_COUNT_PAPER_CLAIM_FIELDS = (
    "accepted_pilot_paper_claim_record_count",
    "paper_ready_artifact_count",
    "superiority_claim_ready_count",
)


def _build_policy_registry() -> Mapping[str, PaperClaimFieldPolicy]:
    """构造字段名唯一且只读的论文结论策略注册表."""

    policies = [
        *(_boolean_policy(field_name) for field_name in _BOOLEAN_PAPER_CLAIM_FIELDS),
        *_ENUM_PAPER_CLAIM_POLICIES,
        *(_count_policy(field_name) for field_name in _COUNT_PAPER_CLAIM_FIELDS),
    ]
    registry = {policy.field_name: policy for policy in policies}
    if len(registry) != len(policies):
        raise RuntimeError("论文结论字段策略不得重复登记")
    return MappingProxyType(registry)


PAPER_CLAIM_COMPONENT_FIELD_POLICIES = _build_policy_registry()


def find_component_paper_claim_violation(
    value: Any,
    *,
    path: str = "$",
) -> PaperClaimFieldViolation | None:
    """递归返回第一个不属于显式允许负值的论文结论字段."""

    if isinstance(value, Mapping):
        for field_name, field_value in value.items():
            normalized_name = str(field_name)
            field_path = f"{path}.{normalized_name}"
            policy = PAPER_CLAIM_COMPONENT_FIELD_POLICIES.get(normalized_name)
            if policy is not None and not policy.allows_component_value(
                field_value
            ):
                return PaperClaimFieldViolation(
                    path=field_path,
                    field_name=normalized_name,
                    value=field_value,
                )
            nested = find_component_paper_claim_violation(
                field_value,
                path=field_path,
            )
            if nested is not None:
                return nested
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            nested = find_component_paper_claim_violation(
                item,
                path=f"{path}[{index}]",
            )
            if nested is not None:
                return nested
    return None


def find_zip_paper_claim_violation(
    archive_path: str | Path,
) -> PaperClaimFieldViolation | None:
    """扫描 ZIP 的 JSON、JSONL 与 CSV 成员并返回首个越权结论字段.

    该函数只解释结构化文本成员, 不根据文件名或自然语言内容推断结论.
    调用方仍需负责 ZIP 身份、成员安全性和业务 schema 校验; 因此该扫描器
    可同时复用于单重复 leaf 包和跨重复不变包, 而不耦合任何包 family.
    """

    with ZipFile(Path(archive_path)) as archive:
        for info in archive.infolist():
            suffix = PurePosixPath(info.filename).suffix.lower()
            if info.is_dir() or suffix not in {".json", ".jsonl", ".csv"}:
                continue
            with archive.open(info, "r") as binary_stream:
                text_stream = io.TextIOWrapper(
                    binary_stream,
                    encoding="utf-8-sig",
                )
                if suffix == ".json":
                    documents = (json.load(text_stream),)
                elif suffix == ".jsonl":
                    documents = (
                        json.loads(line)
                        for line in text_stream
                        if line.strip()
                    )
                else:
                    documents = csv.DictReader(text_stream)
                for row_index, document in enumerate(documents):
                    violation = find_component_paper_claim_violation(
                        document,
                        path=f"{info.filename}[{row_index}]",
                    )
                    if violation is not None:
                        return violation
    return None


__all__ = [
    "PAPER_CLAIM_COMPONENT_FIELD_POLICIES",
    "PaperClaimFieldPolicy",
    "PaperClaimFieldViolation",
    "find_component_paper_claim_violation",
    "find_zip_paper_claim_violation",
]
