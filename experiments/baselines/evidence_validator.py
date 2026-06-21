"""校验外部 baseline 执行 manifest 是否具备论文级证据边界。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _load_optional_manifest(path: str | Path | None) -> dict[str, Any] | None:
    """读取可选 manifest 路径。"""

    if not path:
        return None
    manifest_path = Path(path)
    if not manifest_path.is_file():
        return None
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _missing_paths(values: list[str]) -> list[str]:
    """返回不存在的证据文件路径。"""

    return [value for value in values if not Path(value).is_file()]


def validate_external_baseline_evidence(
    *,
    baseline_execution_manifest: str | Path | None = None,
    require_formal_claim: bool = False,
) -> dict[str, Any]:
    """校验外部 baseline 执行证据是否满足正式对比要求。

    该函数只做证据边界判定, 不重新解释指标数值。指标数值应由下游共同协议表格重建逻辑处理。
    """

    manifest = _load_optional_manifest(baseline_execution_manifest)
    violations: list[dict[str, Any]] = []
    if manifest is None:
        violations.append({"reason": "baseline_execution_manifest_missing", "path": str(baseline_execution_manifest or "")})
    else:
        formal_result_claim = bool(manifest.get("formal_result_claim"))
        evidence_paths = [str(path) for path in manifest.get("evidence_paths", [])]
        missing_evidence_paths = _missing_paths(evidence_paths)
        if require_formal_claim and not formal_result_claim:
            violations.append({"reason": "formal_result_claim_required", "path": str(baseline_execution_manifest)})
        if formal_result_claim and not evidence_paths:
            violations.append({"reason": "formal_result_claim_requires_evidence_paths", "path": str(baseline_execution_manifest)})
        for missing_path in missing_evidence_paths:
            violations.append({"reason": "evidence_path_missing", "path": missing_path})
    return {
        "overall_decision": "fail" if violations else "pass",
        "require_formal_claim": bool(require_formal_claim),
        "baseline_execution_manifest": str(baseline_execution_manifest or ""),
        "violations": violations,
    }
