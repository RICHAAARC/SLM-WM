"""主表 external baseline 的 smoke 证据与正式结果边界审计。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json
from typing import Any, Iterable, Mapping

from main.core.digest import build_stable_digest

PRIMARY_BASELINE_IDS = ("tree_ring", "gaussian_shading", "shallow_diffuse", "t2smark")
METHOD_FAITHFUL_ADAPTER_REQUIRED_IDS = ("tree_ring", "gaussian_shading", "shallow_diffuse")


@dataclass(frozen=True)
class PrimaryBaselineEvidenceRecord:
    """记录一个主表 baseline 当前证据链的工程状态与正式结果缺口。

    该对象属于项目特定写法: 它把 GPU smoke 链路和论文级正式结果边界拆开记录, 防止把 smoke observation
    误当成正式 external baseline 对比指标。
    """

    primary_baseline_evidence_id: str
    primary_baseline_evidence_digest: str
    baseline_id: str
    comparison_group: str
    source_status: str
    source_dir: str
    official_repository_commit: str
    adapter_status: str
    model_alignment_status: str
    adapter_smoke_ready: bool
    adapter_smoke_observation_count: int
    adapter_smoke_execution_devices: tuple[str, ...]
    adapter_smoke_sample_roles: tuple[str, ...]
    adapter_smoke_latent_shapes: tuple[tuple[int, ...], ...]
    method_faithful_adapter_ready: bool
    full_main_prompt_protocol_ready: bool
    fixed_fpr_baseline_calibration_ready: bool
    attack_matrix_baseline_detection_ready: bool
    formal_evidence_paths_ready: bool
    formal_result_ready: bool
    blocking_reasons: tuple[str, ...]
    supports_paper_claim: bool

    def to_dict(self) -> dict[str, Any]:
        """转换为 JSON 兼容字典。"""

        data = asdict(self)
        data["adapter_smoke_execution_devices"] = list(self.adapter_smoke_execution_devices)
        data["adapter_smoke_sample_roles"] = list(self.adapter_smoke_sample_roles)
        data["adapter_smoke_latent_shapes"] = [list(shape) for shape in self.adapter_smoke_latent_shapes]
        data["blocking_reasons"] = list(self.blocking_reasons)
        return data


def load_json(path: str | Path) -> Any:
    """读取 JSON 文件, 兼容 Colab 和 Windows 常见编码。"""

    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def load_optional_json(path: str | Path | None) -> Any:
    """读取可选 JSON 文件, 缺失时返回空列表。"""

    if not path:
        return []
    input_path = Path(path)
    if not input_path.is_file():
        return []
    return load_json(input_path)


def _source_by_baseline(source_registry: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """把 source registry 转换为 baseline_id 索引。"""

    return {
        str(item.get("baseline_id")): dict(item)
        for item in source_registry.get("baseline_sources", [])
        if item.get("baseline_id")
    }


def _observation_rows_by_baseline(observation_rows: Iterable[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """按 baseline_id 聚合 adapter observation。"""

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in observation_rows:
        baseline_id = str(row.get("baseline_id", ""))
        if baseline_id:
            grouped.setdefault(baseline_id, []).append(dict(row))
    return grouped


def _command_result_by_baseline(command_results: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """按 baseline_id 聚合命令执行结果。"""

    return {
        str(row.get("baseline_id")): dict(row)
        for row in command_results
        if row.get("baseline_id")
    }


def _unique_strings(values: Iterable[Any]) -> tuple[str, ...]:
    """把可迭代值规范化为稳定去重字符串元组。"""

    return tuple(sorted({str(value) for value in values if str(value).strip()}))


def _unique_shapes(values: Iterable[Any]) -> tuple[tuple[int, ...], ...]:
    """把 latent shape 值规范化为稳定去重元组。"""

    shapes: set[tuple[int, ...]] = set()
    for value in values:
        if isinstance(value, list) and value:
            shapes.add(tuple(int(item) for item in value))
    return tuple(sorted(shapes))


def _adapter_smoke_ready(command_result: Mapping[str, Any] | None, rows: list[dict[str, Any]]) -> bool:
    """判断 adapter smoke 命令是否成功并产生 observation。"""

    if not command_result:
        return False
    return int(command_result.get("return_code", 1)) == 0 and int(command_result.get("observation_count", 0)) > 0 and bool(rows)


def _method_faithful_adapter_ready(baseline_id: str, adapter_smoke_ready: bool) -> bool:
    """判断当前 adapter 是否已达到方法忠实复现边界。"""

    if baseline_id in METHOD_FAITHFUL_ADAPTER_REQUIRED_IDS:
        return False
    return baseline_id == "t2smark" and adapter_smoke_ready


def _blocking_reasons(
    *,
    baseline_id: str,
    adapter_smoke_ready: bool,
    method_faithful_adapter_ready: bool,
    full_main_prompt_protocol_ready: bool,
    fixed_fpr_baseline_calibration_ready: bool,
    attack_matrix_baseline_detection_ready: bool,
    formal_evidence_paths_ready: bool,
) -> tuple[str, ...]:
    """生成正式主表 baseline 结果仍缺失的原因集合。"""

    reasons: list[str] = []
    if not adapter_smoke_ready:
        reasons.append("adapter_smoke_missing")
    if baseline_id in METHOD_FAITHFUL_ADAPTER_REQUIRED_IDS and not method_faithful_adapter_ready:
        reasons.append("method_faithful_sd35_adapter_required")
    if not full_main_prompt_protocol_ready:
        reasons.append("full_main_prompt_protocol_required")
    if not fixed_fpr_baseline_calibration_ready:
        reasons.append("fixed_fpr_baseline_calibration_required")
    if not attack_matrix_baseline_detection_ready:
        reasons.append("attack_matrix_baseline_detection_required")
    if not formal_evidence_paths_ready:
        reasons.append("formal_evidence_paths_required")
    return tuple(reasons)


def build_primary_baseline_evidence_records(
    *,
    source_registry: Mapping[str, Any],
    command_results: Iterable[Mapping[str, Any]] = (),
    observation_rows: Iterable[Mapping[str, Any]] = (),
    full_main_prompt_protocol_ready: bool = False,
    fixed_fpr_baseline_calibration_ready: bool = False,
    attack_matrix_baseline_detection_ready: bool = False,
    formal_evidence_paths_ready: bool = False,
) -> tuple[dict[str, Any], ...]:
    """构造主表 baseline 证据边界记录。

    通用工程写法是将命令执行结果、observation 和源码登记解耦; 项目特定写法是显式阻断 smoke 结果升级为论文级
    formal result。
    """

    source_items = _source_by_baseline(source_registry)
    results_by_id = _command_result_by_baseline(command_results)
    rows_by_id = _observation_rows_by_baseline(observation_rows)
    records: list[dict[str, Any]] = []
    for baseline_id in PRIMARY_BASELINE_IDS:
        source = source_items.get(baseline_id, {})
        rows = rows_by_id.get(baseline_id, [])
        command_result = results_by_id.get(baseline_id)
        smoke_ready = _adapter_smoke_ready(command_result, rows)
        method_ready = _method_faithful_adapter_ready(baseline_id, smoke_ready)
        reasons = _blocking_reasons(
            baseline_id=baseline_id,
            adapter_smoke_ready=smoke_ready,
            method_faithful_adapter_ready=method_ready,
            full_main_prompt_protocol_ready=full_main_prompt_protocol_ready,
            fixed_fpr_baseline_calibration_ready=fixed_fpr_baseline_calibration_ready,
            attack_matrix_baseline_detection_ready=attack_matrix_baseline_detection_ready,
            formal_evidence_paths_ready=formal_evidence_paths_ready,
        )
        formal_ready = not reasons
        payload = {
            "baseline_id": baseline_id,
            "source_status": str(source.get("source_status", "not_registered")),
            "official_repository_commit": str(source.get("official_repository_commit", "")),
            "adapter_status": str(source.get("adapter_status", "")),
            "adapter_smoke_ready": smoke_ready,
            "adapter_smoke_observation_count": len(rows),
            "method_faithful_adapter_ready": method_ready,
            "blocking_reasons": reasons,
        }
        digest = build_stable_digest(payload)
        record = PrimaryBaselineEvidenceRecord(
            primary_baseline_evidence_id=f"primary_baseline_evidence_{digest[:16]}",
            primary_baseline_evidence_digest=digest,
            baseline_id=baseline_id,
            comparison_group=str(source.get("comparison_group", "primary")),
            source_status=str(source.get("source_status", "not_registered")),
            source_dir=str(source.get("source_dir", "")),
            official_repository_commit=str(source.get("official_repository_commit", "")),
            adapter_status=str(source.get("adapter_status", "")),
            model_alignment_status=str(source.get("model_alignment_status", "")),
            adapter_smoke_ready=smoke_ready,
            adapter_smoke_observation_count=len(rows),
            adapter_smoke_execution_devices=_unique_strings(row.get("execution_device", "") for row in rows),
            adapter_smoke_sample_roles=_unique_strings(row.get("sample_role", "") for row in rows),
            adapter_smoke_latent_shapes=_unique_shapes(row.get("latent_shape", []) for row in rows),
            method_faithful_adapter_ready=method_ready,
            full_main_prompt_protocol_ready=full_main_prompt_protocol_ready,
            fixed_fpr_baseline_calibration_ready=fixed_fpr_baseline_calibration_ready,
            attack_matrix_baseline_detection_ready=attack_matrix_baseline_detection_ready,
            formal_evidence_paths_ready=formal_evidence_paths_ready,
            formal_result_ready=formal_ready,
            blocking_reasons=reasons,
            supports_paper_claim=False,
        )
        records.append(record.to_dict())
    return tuple(records)


def build_primary_baseline_evidence_summary(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """聚合主表 baseline 证据边界摘要。"""

    rows = [dict(row) for row in records]
    smoke_ready_ids = [str(row["baseline_id"]) for row in rows if row.get("adapter_smoke_ready")]
    formal_ready_ids = [str(row["baseline_id"]) for row in rows if row.get("formal_result_ready")]
    blocking_reasons = sorted({reason for row in rows for reason in row.get("blocking_reasons", [])})
    return {
        "primary_baseline_count": len(rows),
        "adapter_smoke_ready_count": len(smoke_ready_ids),
        "adapter_smoke_ready_ids": smoke_ready_ids,
        "formal_result_ready_count": len(formal_ready_ids),
        "formal_result_ready_ids": formal_ready_ids,
        "primary_baseline_formal_ready": len(formal_ready_ids) == len(rows) and bool(rows),
        "blocking_reasons": blocking_reasons,
        "supports_paper_claim": False,
    }
