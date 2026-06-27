"""主表 external baseline 方法忠实 SD3.5 适配协议。

该模块的作用是把 Tree-Ring 已经跑通的 method-faithful SD3.5 adapter 经验
固化为统一协议, 使 Gaussian Shading 和 Shallow Diffuse 后续适配时使用同一套
observation 字段、图像 provenance、分数方向、阈值语义和正式导入边界。

通用工程写法:
- 将 schema、记录聚合和 readiness 判断集中在协议层, 下游表格与报告只消费结果。
- 不在每个 adapter 业务路径中重复构造大量错误信息。

项目特定写法:
- T2SMark 属于 SD3.5 native official reproduction, 不需要 method-faithful adapter。
- Tree-Ring、Gaussian Shading 和 Shallow Diffuse 属于 legacy diffusion watermark,
  必须先达到 method-faithful SD3.5 adapter 边界, 才能进入正式导入候选。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping

from main.core.digest import build_stable_digest

PRIMARY_BASELINE_METHOD_FAITHFUL_ADAPTER_PROTOCOL_NAME = "primary_baseline_method_faithful_adapter_protocol"
METHOD_FAITHFUL_ADAPTER_REQUIRED_IDS = ("tree_ring", "gaussian_shading", "shallow_diffuse")
NATIVE_OFFICIAL_REPRODUCTION_IDS = ("t2smark",)
METHOD_FAITHFUL_ADAPTER_BOUNDARY = "method_faithful_sd35_adapter_reproduction"
REJECTED_INCOMPLETE_ADAPTER_BOUNDARIES = (
    "method_faithful_not_full_external_baseline_comparison",
    "sd35_method_faithful_adapter_not_formal_external_baseline_evidence",
)
REQUIRED_SAMPLE_ROLES = ("clean_negative", "positive_source")
REQUIRED_OBSERVATION_FIELDS = (
    "event_id",
    "baseline_id",
    "score",
    "threshold",
    "score_name",
    "higher_is_positive",
    "detection_decision",
    "sample_role",
    "attack_family",
    "attack_condition",
    "prompt_id",
    "image_id",
    "adapter_boundary",
    "formal_result_claim",
    "supports_paper_claim",
)
REQUIRED_IMAGE_PROVENANCE_FIELDS = ("image_path", "image_digest")
REQUIRED_SCORE_FIELDS = ("score", "threshold", "score_name", "higher_is_positive")


@dataclass(frozen=True)
class MethodFaithfulAdapterStatusRecord:
    """记录单个主表 baseline 的方法忠实适配协议状态。

    该对象不是论文结果, 而是工程治理记录。它显式区分 method-faithful adapter、
    native official reproduction 和不合格 incomplete adapter boundary, 防止 method-faithful observation
    被误导入为主表正式结果。
    """

    method_faithful_adapter_status_id: str
    method_faithful_adapter_status_digest: str
    baseline_id: str
    protocol_role: str
    expected_adapter_boundary: str
    observed_adapter_boundaries: tuple[str, ...]
    observation_count: int
    clean_negative_count: int
    positive_source_count: int
    attacked_observation_count: int
    score_protocol_ready: bool
    image_provenance_ready: bool
    method_faithful_adapter_ready: bool
    formal_import_candidate_allowed: bool
    blocking_reasons: tuple[str, ...]
    supports_paper_claim: bool

    def to_dict(self) -> dict[str, Any]:
        """转换为 JSON 兼容字典。"""

        data = asdict(self)
        data["observed_adapter_boundaries"] = list(self.observed_adapter_boundaries)
        data["blocking_reasons"] = list(self.blocking_reasons)
        return data


def build_primary_baseline_method_faithful_adapter_schema() -> dict[str, Any]:
    """构造方法忠实 SD3.5 适配协议 schema 的可落盘描述。"""

    return {
        "protocol_name": PRIMARY_BASELINE_METHOD_FAITHFUL_ADAPTER_PROTOCOL_NAME,
        "method_faithful_adapter_required_ids": list(METHOD_FAITHFUL_ADAPTER_REQUIRED_IDS),
        "native_official_reproduction_ids": list(NATIVE_OFFICIAL_REPRODUCTION_IDS),
        "accepted_adapter_boundary": METHOD_FAITHFUL_ADAPTER_BOUNDARY,
        "rejected_incomplete_adapter_boundaries": list(REJECTED_INCOMPLETE_ADAPTER_BOUNDARIES),
        "required_sample_roles": list(REQUIRED_SAMPLE_ROLES),
        "required_observation_fields": list(REQUIRED_OBSERVATION_FIELDS),
        "required_image_provenance_fields": list(REQUIRED_IMAGE_PROVENANCE_FIELDS),
        "required_score_fields": list(REQUIRED_SCORE_FIELDS),
        "formal_import_gate": {
            "full_main_prompt_protocol_ready": "required_after_adapter_protocol",
            "fixed_fpr_baseline_calibration_ready": "required_after_adapter_protocol",
            "attack_matrix_baseline_detection_ready": "required_after_adapter_protocol",
            "formal_evidence_paths_ready": "required_after_adapter_protocol",
        },
        "supports_paper_claim": False,
    }


def _as_text(value: Any) -> str:
    """将可选值规范化为字符串。"""

    return str(value or "").strip()


def _bool_value(value: Any) -> bool:
    """读取 JSON / CSV 常见布尔表示。"""

    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _is_bool_token(value: Any) -> bool:
    """判断输入是否是可审计的布尔表示。"""

    if isinstance(value, bool):
        return True
    return str(value).strip().lower() in {"true", "false", "1", "0", "yes", "no", "y", "n"}


def _rows_by_baseline(rows: Iterable[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """按 baseline_id 聚合 observation。"""

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        baseline_id = _as_text(row.get("baseline_id"))
        if baseline_id:
            grouped.setdefault(baseline_id, []).append(dict(row))
    return grouped


def _unique_texts(values: Iterable[Any]) -> tuple[str, ...]:
    """稳定去重字符串集合。"""

    return tuple(sorted({_as_text(value) for value in values if _as_text(value)}))


def _missing_required_fields(rows: Iterable[Mapping[str, Any]], field_names: Iterable[str]) -> tuple[str, ...]:
    """收集 observation 中缺失的必需字段名。"""

    missing: set[str] = set()
    for row in rows:
        for field_name in field_names:
            if field_name not in row or row.get(field_name) in {None, ""}:
                missing.add(field_name)
    return tuple(sorted(missing))


def _count_role(rows: Iterable[Mapping[str, Any]], sample_role: str) -> int:
    """统计指定样本角色数量。"""

    return sum(1 for row in rows if _as_text(row.get("sample_role")) == sample_role)


def _attacked_count(rows: Iterable[Mapping[str, Any]]) -> int:
    """统计攻击后 observation 数量。"""

    return sum(1 for row in rows if _as_text(row.get("attack_family")) not in {"", "clean"})


def _score_protocol_ready(rows: list[dict[str, Any]]) -> bool:
    """判断分数字段、阈值字段和 score 方向是否齐备。"""

    if not rows:
        return False
    if _missing_required_fields(rows, REQUIRED_SCORE_FIELDS):
        return False
    return all(_is_bool_token(row.get("higher_is_positive")) for row in rows)


def _image_provenance_ready(rows: list[dict[str, Any]]) -> bool:
    """判断图像路径与图像 digest 是否齐备。"""

    return bool(rows) and not _missing_required_fields(rows, REQUIRED_IMAGE_PROVENANCE_FIELDS)


def _method_faithful_reasons(
    *,
    rows: list[dict[str, Any]],
    baseline_id: str,
    observed_boundaries: tuple[str, ...],
    clean_negative_count: int,
    positive_source_count: int,
    score_protocol_ready: bool,
    image_provenance_ready: bool,
) -> tuple[str, ...]:
    """生成方法忠实适配协议阻断原因。"""

    reasons: list[str] = []
    if not rows:
        reasons.append("method_faithful_observations_missing")
    if baseline_id in METHOD_FAITHFUL_ADAPTER_REQUIRED_IDS:
        if METHOD_FAITHFUL_ADAPTER_BOUNDARY not in observed_boundaries:
            reasons.append("method_faithful_adapter_boundary_required")
        if any(boundary in REJECTED_INCOMPLETE_ADAPTER_BOUNDARIES for boundary in observed_boundaries):
            reasons.append("incomplete_adapter_boundary_rejected")
    if clean_negative_count <= 0:
        reasons.append("clean_negative_observations_required")
    if positive_source_count <= 0:
        reasons.append("positive_source_observations_required")
    if not score_protocol_ready:
        reasons.append("score_protocol_fields_required")
    if baseline_id in METHOD_FAITHFUL_ADAPTER_REQUIRED_IDS and not image_provenance_ready:
        reasons.append("image_path_and_digest_required")
    missing_fields = _missing_required_fields(rows, REQUIRED_OBSERVATION_FIELDS)
    if missing_fields:
        reasons.append("required_observation_fields_missing")
    return tuple(dict.fromkeys(reasons))


def build_method_faithful_adapter_status_records(
    observation_rows: Iterable[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    """根据 observation 行生成主表 baseline 方法忠实适配状态记录。"""

    grouped = _rows_by_baseline(observation_rows)
    records: list[dict[str, Any]] = []
    for baseline_id in (*METHOD_FAITHFUL_ADAPTER_REQUIRED_IDS, *NATIVE_OFFICIAL_REPRODUCTION_IDS):
        rows = grouped.get(baseline_id, [])
        observed_boundaries = _unique_texts(row.get("adapter_boundary") for row in rows)
        clean_negative_count = _count_role(rows, "clean_negative")
        positive_source_count = _count_role(rows, "positive_source")
        attacked_observation_count = _attacked_count(rows)
        score_ready = _score_protocol_ready(rows)
        image_ready = _image_provenance_ready(rows)
        protocol_role = (
            "method_faithful_adapter_required"
            if baseline_id in METHOD_FAITHFUL_ADAPTER_REQUIRED_IDS
            else "native_official_reproduction"
        )
        expected_boundary = METHOD_FAITHFUL_ADAPTER_BOUNDARY if baseline_id in METHOD_FAITHFUL_ADAPTER_REQUIRED_IDS else ""
        reasons = (
            ()
            if baseline_id in NATIVE_OFFICIAL_REPRODUCTION_IDS
            else _method_faithful_reasons(
                rows=rows,
                baseline_id=baseline_id,
                observed_boundaries=observed_boundaries,
                clean_negative_count=clean_negative_count,
                positive_source_count=positive_source_count,
                score_protocol_ready=score_ready,
                image_provenance_ready=image_ready,
            )
        )
        method_ready = baseline_id in NATIVE_OFFICIAL_REPRODUCTION_IDS or not reasons
        payload = {
            "baseline_id": baseline_id,
            "protocol_role": protocol_role,
            "observed_adapter_boundaries": observed_boundaries,
            "observation_count": len(rows),
            "clean_negative_count": clean_negative_count,
            "positive_source_count": positive_source_count,
            "attacked_observation_count": attacked_observation_count,
            "method_faithful_adapter_ready": method_ready,
            "blocking_reasons": reasons,
        }
        digest = build_stable_digest(payload)
        record = MethodFaithfulAdapterStatusRecord(
            method_faithful_adapter_status_id=f"method_faithful_adapter_status_{digest[:16]}",
            method_faithful_adapter_status_digest=digest,
            baseline_id=baseline_id,
            protocol_role=protocol_role,
            expected_adapter_boundary=expected_boundary,
            observed_adapter_boundaries=observed_boundaries,
            observation_count=len(rows),
            clean_negative_count=clean_negative_count,
            positive_source_count=positive_source_count,
            attacked_observation_count=attacked_observation_count,
            score_protocol_ready=score_ready,
            image_provenance_ready=image_ready,
            method_faithful_adapter_ready=method_ready,
            formal_import_candidate_allowed=baseline_id in METHOD_FAITHFUL_ADAPTER_REQUIRED_IDS and method_ready,
            blocking_reasons=reasons,
            supports_paper_claim=False,
        )
        records.append(record.to_dict())
    return tuple(records)


def build_method_faithful_adapter_summary(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """聚合方法忠实适配协议状态摘要。"""

    rows = [dict(row) for row in records]
    required_rows = [row for row in rows if row.get("baseline_id") in METHOD_FAITHFUL_ADAPTER_REQUIRED_IDS]
    ready_ids = [str(row["baseline_id"]) for row in required_rows if row.get("method_faithful_adapter_ready")]
    missing_ids = [str(row["baseline_id"]) for row in required_rows if not row.get("method_faithful_adapter_ready")]
    blocking_reasons = sorted({reason for row in required_rows for reason in row.get("blocking_reasons", [])})
    return {
        "protocol_name": PRIMARY_BASELINE_METHOD_FAITHFUL_ADAPTER_PROTOCOL_NAME,
        "method_faithful_adapter_required_count": len(required_rows),
        "method_faithful_adapter_ready_count": len(ready_ids),
        "method_faithful_adapter_ready_ids": ready_ids,
        "missing_method_faithful_adapter_ids": missing_ids,
        "native_official_reproduction_ids": list(NATIVE_OFFICIAL_REPRODUCTION_IDS),
        "method_faithful_adapter_protocol_ready": len(ready_ids) == len(required_rows) and bool(required_rows),
        "blocking_reasons": blocking_reasons,
        "formal_import_candidate_allowed_ids": [
            str(row["baseline_id"]) for row in required_rows if row.get("formal_import_candidate_allowed")
        ],
        "supports_paper_claim": False,
    }
