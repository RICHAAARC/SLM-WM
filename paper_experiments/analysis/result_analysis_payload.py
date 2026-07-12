"""绑定论文结果分析表、失败记录与失败案例图的精确字节身份.

该模块只定义结果分析 payload 的角色集合、摘要协议和 manifest 配置. 生成器
与最终闭合门禁共享同一实现, 避免一侧只检查 ready 布尔值而另一侧维护不同
的文件清单.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import re
from typing import Any, Mapping

from main.core.digest import build_stable_digest
from paper_experiments.analysis.paired_superiority import PRIMARY_BASELINE_IDS


RESULT_ANALYSIS_PAYLOAD_FILE_NAMES = {
    "main_confidence_interval_table": "confidence_interval_table.csv",
    "per_attack_superiority_table": "per_attack_superiority_table.csv",
    "failure_case_records": "failure_case_records.jsonl",
    "failure_case_figure": "failure_case_figure.svg",
}
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def build_governed_paper_payload_path_map(paper_claim_scale: str) -> dict[str, str]:
    """返回完整包必须逐字节复验的主表、攻击表、质量表和结果分析表图."""

    scale = str(paper_claim_scale).strip()
    if not scale:
        raise ValueError("paper_claim_scale 不得为空")
    result_analysis_root = f"outputs/pilot_paper_result_analysis/{scale}"
    return {
        "main_comparison_table": (
            f"outputs/external_baseline_comparison/{scale}/baseline_comparison_table.csv"
        ),
        "attack_table": f"outputs/attack_matrix/{scale}/attack_family_metrics.csv",
        "quality_table": (
            f"outputs/dataset_level_quality/{scale}/dataset_quality_metrics.csv"
        ),
        **{
            role: f"{result_analysis_root}/{file_name}"
            for role, file_name in RESULT_ANALYSIS_PAYLOAD_FILE_NAMES.items()
        },
    }


def _recorded_path(path: Path, repository_root: Path) -> str:
    """仓库内文件使用相对路径, 仓库外文件保留绝对路径."""

    try:
        return path.resolve().relative_to(repository_root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def file_sha256(path: Path) -> str:
    """流式计算普通文件 SHA-256."""

    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"结果分析 payload 必须是普通文件: {path.as_posix()}")
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_result_analysis_payload_binding(
    *,
    repository_root: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """从刚写出的四类 payload 构造精确路径与字节摘要绑定."""

    root = Path(repository_root).resolve()
    resolved_output_dir = Path(output_dir).resolve()
    path_map = {
        role: _recorded_path(resolved_output_dir / file_name, root)
        for role, file_name in RESULT_ANALYSIS_PAYLOAD_FILE_NAMES.items()
    }
    sha256_map = {
        role: file_sha256(resolved_output_dir / file_name)
        for role, file_name in RESULT_ANALYSIS_PAYLOAD_FILE_NAMES.items()
    }
    digest_payload = {
        "result_analysis_payload_path_map": path_map,
        "result_analysis_payload_sha256_map": sha256_map,
    }
    return {
        **digest_payload,
        "result_analysis_payload_digest": build_stable_digest(digest_payload),
    }


def build_result_analysis_manifest_config(summary: Mapping[str, Any]) -> dict[str, Any]:
    """从 summary 重建结果分析 manifest 的完整配置身份."""

    return {
        "failure_case_limit": summary.get("failure_case_limit"),
        "primary_baseline_method_ids": list(PRIMARY_BASELINE_IDS),
        "proposed_method_id": "slm_wm_current",
        "result_record_set_digest": summary.get("result_record_set_digest", ""),
        "paired_superiority_rows_digest": summary.get(
            "paired_superiority_rows_digest", ""
        ),
        "paired_superiority_protocol_digest": summary.get(
            "paired_superiority_protocol_digest", ""
        ),
        "paired_test_prompt_count": summary.get("paired_test_prompt_count", 0),
        "paired_test_prompt_id_digest": summary.get(
            "paired_test_prompt_id_digest", ""
        ),
        "paired_attack_registry_digest": summary.get(
            "paired_attack_registry_digest", ""
        ),
        "method_observation_source_sha256_map": summary.get(
            "method_observation_source_sha256_map", {}
        ),
        "threshold_audit_rows_digest": summary.get(
            "threshold_audit_rows_digest", ""
        ),
        "claim_p_value_method": summary.get("claim_p_value_method", ""),
        "sharp_null_diagnostic_method": summary.get(
            "sharp_null_diagnostic_method", ""
        ),
        "bootstrap_analysis_schema": summary.get("bootstrap_analysis_schema", ""),
        "bootstrap_bit_generator": summary.get("bootstrap_bit_generator", ""),
        "bootstrap_quantile_method": summary.get("bootstrap_quantile_method", ""),
        "bootstrap_resample_count": summary.get("bootstrap_resample_count", 0),
        "confidence_level": summary.get("confidence_level", 0.0),
        "result_analysis_payload_digest": summary.get(
            "result_analysis_payload_digest", ""
        ),
    }


def result_analysis_payload_binding_ready(
    *,
    summary: Mapping[str, Any],
    manifest: Mapping[str, Any],
    actual_source_sha256: Mapping[str, str],
) -> bool:
    """核验 summary、manifest、实际源文件摘要和固定角色集合完全一致."""

    path_map = summary.get("result_analysis_payload_path_map")
    sha256_map = summary.get("result_analysis_payload_sha256_map")
    declared_digest = str(summary.get("result_analysis_payload_digest", ""))
    metadata = manifest.get("metadata")
    output_paths = manifest.get("output_paths")
    expected_roles = set(RESULT_ANALYSIS_PAYLOAD_FILE_NAMES)
    if (
        not isinstance(path_map, Mapping)
        or set(path_map) != expected_roles
        or not isinstance(sha256_map, Mapping)
        or set(sha256_map) != expected_roles
        or not isinstance(metadata, Mapping)
        or not isinstance(output_paths, list | tuple)
    ):
        return False
    normalized_paths = {role: str(path_map[role]) for role in expected_roles}
    normalized_sha256 = {role: str(sha256_map[role]) for role in expected_roles}
    paper_claim_scale = str(summary.get("paper_claim_scale", ""))
    expected_root = f"outputs/pilot_paper_result_analysis/{paper_claim_scale}/"
    digest_payload = {
        "result_analysis_payload_path_map": normalized_paths,
        "result_analysis_payload_sha256_map": normalized_sha256,
    }
    output_path_set = {
        str(path).replace("\\", "/").lstrip("./") for path in output_paths
    }
    return bool(
        len(set(normalized_paths.values())) == len(expected_roles)
        and all(_SHA256_PATTERN.fullmatch(value) for value in normalized_sha256.values())
        and all(
            normalized_paths[role]
            .replace("\\", "/")
            .endswith(expected_root + RESULT_ANALYSIS_PAYLOAD_FILE_NAMES[role])
            for role in expected_roles
        )
        and all(
            normalized_paths[role].replace("\\", "/").lstrip("./")
            in output_path_set
            for role in expected_roles
        )
        and all(
            actual_source_sha256.get(normalized_paths[role])
            == normalized_sha256[role]
            for role in expected_roles
        )
        and _SHA256_PATTERN.fullmatch(declared_digest) is not None
        and declared_digest == build_stable_digest(digest_payload)
        and metadata.get("result_analysis_payload_path_map") == dict(path_map)
        and metadata.get("result_analysis_payload_sha256_map") == dict(sha256_map)
        and metadata.get("result_analysis_payload_digest") == declared_digest
        and manifest.get("config_digest")
        == build_stable_digest(build_result_analysis_manifest_config(summary))
    )
