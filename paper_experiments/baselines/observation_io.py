"""读取外部 baseline 适配器输出并生成受治理执行 manifest。"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Iterable

from main.core.digest import build_stable_digest

REQUIRED_BASELINE_OBSERVATION_FIELDS = ("event_id", "baseline_id", "score", "threshold")
BASELINE_EXECUTION_MANIFEST_NAME = "baseline_execution_manifest.json"


def _load_json_or_jsonl(path: Path) -> list[dict[str, Any]]:
    """读取 JSON 数组或 JSONL 文件。"""

    text = path.read_text(encoding="utf-8-sig")
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    payload = json.loads(text)
    if not isinstance(payload, list):
        raise TypeError("baseline observation JSON 必须是列表")
    return [dict(row) for row in payload]


def _load_csv(path: Path) -> list[dict[str, Any]]:
    """读取 CSV 形式的 observation 文件。"""

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def load_baseline_observation_rows(path: str | Path) -> list[dict[str, Any]]:
    """从 JSON / JSONL / CSV 文件读取外部 baseline observation rows。

    该函数属于通用工程写法: 第三方方法可以在不同环境中运行, 只要把结果导出为统一字段,
    本项目即可用同一入口读取。字段完整性校验集中在此处, 业务路径不重复构造同类错误信息。
    """

    input_path = Path(path)
    if input_path.suffix in {".json", ".jsonl"}:
        rows = _load_json_or_jsonl(input_path)
    elif input_path.suffix == ".csv":
        rows = _load_csv(input_path)
    else:
        raise ValueError(f"不支持的 baseline observation 文件后缀: {input_path.suffix}")
    missing_by_index = {
        index: [field for field in REQUIRED_BASELINE_OBSERVATION_FIELDS if field not in row]
        for index, row in enumerate(rows)
    }
    missing_by_index = {index: missing for index, missing in missing_by_index.items() if missing}
    if missing_by_index:
        raise ValueError(f"baseline observation rows 缺少字段: {missing_by_index}")
    return rows


@dataclass(frozen=True)
class BaselineExecutionManifest:
    """描述外部 baseline 命令执行结果和证据边界。"""

    artifact_name: str
    producer_id: str
    producer_role: str
    formal_result_claim: bool
    execution_boundary: str
    command_count: int
    observation_count: int
    baseline_ids: tuple[str, ...]
    failed_command_count: int
    evidence_paths: tuple[str, ...]
    baseline_observations_path: str
    command_results_path: str
    execution_digest: str

    def to_dict(self) -> dict[str, Any]:
        """转换为 JSON 兼容字典。"""

        data = asdict(self)
        data["baseline_ids"] = list(self.baseline_ids)
        data["evidence_paths"] = list(self.evidence_paths)
        return data


def build_baseline_execution_manifest(
    *,
    command_specs: Iterable[dict[str, Any]],
    command_results: Iterable[dict[str, Any]],
    observation_rows: Iterable[dict[str, Any]],
    baseline_observations_path: str | Path,
    command_results_path: str | Path,
    formal_result_claim: bool = False,
    evidence_paths: Iterable[str | Path] = (),
    producer_id: str = "external_baseline_command_runner",
) -> BaselineExecutionManifest:
    """生成外部 baseline 执行 manifest。

    该 manifest 明确区分工程链路是否跑通与论文结论是否可用。只有在 `formal_result_claim=True`
    且证据路径存在时, 下游才可以把该结果纳入论文级对比结论。
    """

    specs = [dict(row) for row in command_specs]
    results = [dict(row) for row in command_results]
    observations = [dict(row) for row in observation_rows]
    materialized_evidence_paths = tuple(str(Path(path)) for path in evidence_paths)
    baseline_ids = tuple(sorted({str(row.get("baseline_id", "")) for row in specs if row.get("baseline_id")}))
    failed_count = sum(1 for row in results if int(row.get("return_code", 1)) != 0)
    payload = {
        "command_specs": specs,
        "command_results": results,
        "observation_rows": observations,
        "formal_result_claim": bool(formal_result_claim),
        "evidence_paths": materialized_evidence_paths,
    }
    return BaselineExecutionManifest(
        artifact_name=BASELINE_EXECUTION_MANIFEST_NAME,
        producer_id=producer_id,
        producer_role="external_baseline_command_execution",
        formal_result_claim=bool(formal_result_claim),
        execution_boundary=(
            "external_command_results_bound_to_formal_evidence"
            if formal_result_claim
            else "external_command_results_require_separate_formal_evidence"
        ),
        command_count=len(specs),
        observation_count=len(observations),
        baseline_ids=baseline_ids,
        failed_command_count=failed_count,
        evidence_paths=materialized_evidence_paths,
        baseline_observations_path=str(baseline_observations_path),
        command_results_path=str(command_results_path),
        execution_digest=build_stable_digest(payload),
    )
