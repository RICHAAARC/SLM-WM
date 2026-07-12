"""验证结果分析表图的精确路径、摘要与 manifest 绑定."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from main.core.digest import build_stable_digest
from paper_experiments.analysis.result_analysis_payload import (
    RESULT_ANALYSIS_PAYLOAD_FILE_NAMES,
    build_governed_paper_payload_path_map,
    build_result_analysis_manifest_config,
    build_result_analysis_payload_binding,
    result_analysis_payload_binding_ready,
)
from scripts.write_pilot_paper_complete_result_package import (
    collect_result_closure_source_entries,
)


pytestmark = pytest.mark.quick


def _ready_payload(tmp_path: Path) -> tuple[dict[str, object], dict[str, object], dict[str, str]]:
    """写出四类最小 payload 并构造相互一致的 summary 与 manifest."""

    output_dir = (
        tmp_path / "outputs/pilot_paper_result_analysis/probe_paper"
    )
    output_dir.mkdir(parents=True)
    contents = {
        "main_confidence_interval_table": b"method_id,tpr\nslm_wm_current,0.9\n",
        "per_attack_superiority_table": b"attack_name,margin\njpeg,0.2\n",
        "failure_case_records": b'{"attack_name":"jpeg"}\n',
        "failure_case_figure": b"<svg><title>failure</title></svg>\n",
    }
    for role, file_name in RESULT_ANALYSIS_PAYLOAD_FILE_NAMES.items():
        (output_dir / file_name).write_bytes(contents[role])
    binding = build_result_analysis_payload_binding(
        repository_root=tmp_path,
        output_dir=output_dir,
    )
    summary: dict[str, object] = {
        "paper_claim_scale": "probe_paper",
        "failure_case_limit": 12,
        **binding,
    }
    config = build_result_analysis_manifest_config(summary)
    manifest: dict[str, object] = {
        "output_paths": [
            *binding["result_analysis_payload_path_map"].values(),
            "outputs/pilot_paper_result_analysis/probe_paper/result_analysis_summary.json",
            "outputs/pilot_paper_result_analysis/probe_paper/manifest.local.json",
        ],
        "config_digest": build_stable_digest(config),
        "metadata": dict(summary),
    }
    actual_source_sha256 = {
        path: binding["result_analysis_payload_sha256_map"][role]
        for role, path in binding["result_analysis_payload_path_map"].items()
    }
    return summary, manifest, actual_source_sha256


def test_result_analysis_payload_binding_requires_exact_bytes_and_roles(
    tmp_path: Path,
) -> None:
    """完整角色集合、实际字节、summary 和 manifest 一致时才通过."""

    summary, manifest, actual_source_sha256 = _ready_payload(tmp_path)

    assert result_analysis_payload_binding_ready(
        summary=summary,
        manifest=manifest,
        actual_source_sha256=actual_source_sha256,
    )

    drifted_source_sha256 = dict(actual_source_sha256)
    failure_path = summary["result_analysis_payload_path_map"][
        "failure_case_figure"
    ]
    drifted_source_sha256[failure_path] = "f" * 64
    assert not result_analysis_payload_binding_ready(
        summary=summary,
        manifest=manifest,
        actual_source_sha256=drifted_source_sha256,
    )


def test_result_analysis_payload_binding_rejects_missing_role_or_manifest_drift(
    tmp_path: Path,
) -> None:
    """删除角色或只改 manifest metadata 都不得保留 ready 状态."""

    summary, manifest, actual_source_sha256 = _ready_payload(tmp_path)
    incomplete_summary = dict(summary)
    incomplete_path_map = dict(summary["result_analysis_payload_path_map"])
    incomplete_path_map.pop("failure_case_records")
    incomplete_summary["result_analysis_payload_path_map"] = incomplete_path_map
    assert not result_analysis_payload_binding_ready(
        summary=incomplete_summary,
        manifest=manifest,
        actual_source_sha256=actual_source_sha256,
    )

    drifted_manifest = dict(manifest)
    drifted_metadata = dict(manifest["metadata"])
    drifted_metadata["result_analysis_payload_digest"] = "0" * 64
    drifted_manifest["metadata"] = drifted_metadata
    assert not result_analysis_payload_binding_ready(
        summary=summary,
        manifest=drifted_manifest,
        actual_source_sha256=actual_source_sha256,
    )


def test_complete_package_revalidates_result_analysis_payload_bytes(
    tmp_path: Path,
) -> None:
    """完整包收集边界必须再次读取 closure source map 中的表图字节."""

    summary, _, actual_source_sha256 = _ready_payload(tmp_path)
    required_payload_paths = build_governed_paper_payload_path_map("probe_paper")
    for role, relative_path in required_payload_paths.items():
        if relative_path in actual_source_sha256:
            continue
        path = tmp_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"role,value\n{role},1\n", encoding="utf-8")
        actual_source_sha256[relative_path] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    gate_dir = tmp_path / "outputs/result_closure_gate/probe_paper"
    gate_dir.mkdir(parents=True)
    (gate_dir / "result_closure_gate_report.json").write_text(
        json.dumps({"closure_source_file_sha256": actual_source_sha256}),
        encoding="utf-8",
    )
    (gate_dir / "manifest.local.json").write_text("{}\n", encoding="utf-8")

    entries, declared_map, ready = collect_result_closure_source_entries(
        tmp_path,
        paper_run_name="probe_paper",
        excluded_paths=(),
    )

    assert ready
    assert declared_map == actual_source_sha256
    assert len(entries) == len(actual_source_sha256) + 2

    incomplete_map = dict(actual_source_sha256)
    incomplete_map.pop(required_payload_paths["quality_table"])
    (gate_dir / "result_closure_gate_report.json").write_text(
        json.dumps({"closure_source_file_sha256": incomplete_map}),
        encoding="utf-8",
    )
    _, _, ready_without_quality = collect_result_closure_source_entries(
        tmp_path,
        paper_run_name="probe_paper",
        excluded_paths=(),
    )
    assert not ready_without_quality
    (gate_dir / "result_closure_gate_report.json").write_text(
        json.dumps({"closure_source_file_sha256": actual_source_sha256}),
        encoding="utf-8",
    )

    failure_figure_path = tmp_path / summary["result_analysis_payload_path_map"][
        "failure_case_figure"
    ]
    failure_figure_path.write_text("<svg>tampered</svg>\n", encoding="utf-8")
    _, _, ready_after_drift = collect_result_closure_source_entries(
        tmp_path,
        paper_run_name="probe_paper",
        excluded_paths=(),
    )
    assert not ready_after_drift
