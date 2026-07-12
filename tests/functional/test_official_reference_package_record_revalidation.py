"""验证 official-reference 打包会拒绝被篡改的 governed record 与报告."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
from typing import Any

import pytest

from paper_experiments.runners import gaussian_shading_official_reference as gaussian_runner
from paper_experiments.runners import shallow_diffuse_official_reference as shallow_runner
from paper_experiments.runners import tree_ring_official_reference as tree_runner


@pytest.mark.quick
@pytest.mark.parametrize(
    ("runner", "helper_name", "builder_name", "validator_name", "prefix", "has_record_gate"),
    (
        (
            tree_runner,
            "_validate_packaged_tree_ring_reference_evidence",
            "build_tree_ring_official_reference_record",
            "validate_tree_ring_official_reference_records",
            "tree_ring",
            True,
        ),
        (
            gaussian_runner,
            "_validate_packaged_gaussian_shading_reference_evidence",
            "build_gaussian_shading_official_reference_record",
            "validate_gaussian_shading_official_reference_records",
            "gaussian_shading",
            False,
        ),
        (
            shallow_runner,
            "_validate_packaged_shallow_diffuse_reference_evidence",
            "build_shallow_diffuse_official_reference_record",
            "validate_shallow_diffuse_official_reference_records",
            "shallow_diffuse",
            True,
        ),
    ),
)
def test_packaged_official_reference_rejects_record_or_validation_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    runner: Any,
    helper_name: str,
    builder_name: str,
    validator_name: str,
    prefix: str,
    has_record_gate: bool,
) -> None:
    """三条打包路径都必须精确比较重建记录与 validator 输出."""

    source_dir = tmp_path / prefix
    source_dir.mkdir(parents=True)
    (source_dir / "scientific_units").mkdir()
    metric_path = source_dir / f"{prefix}_official_metric_summary.json"
    metric_path.write_text("{}\n", encoding="utf-8")
    records_path = source_dir / f"{prefix}_official_reference_records.jsonl"
    validation_path = source_dir / f"{prefix}_official_reference_validation_report.json"
    expected_record = {
        "baseline_id": prefix,
        "reference_record_digest": "a" * 64,
        "unit_rebuilt": True,
        "official_scientific_config": {"sample_count": 20},
        "official_scientific_config_digest": "e" * 64,
    }
    base_validation = {
        "reference_import_ready": True,
        "accepted_reference_record_ids": ["record"],
    }
    metric_validation = {
        "required_metric_fields": [],
        "missing_required_metric_fields": [],
        "invalid_required_metric_fields": [],
        "required_metrics_ready": True,
    }
    expected_validation = dict(base_validation)
    if has_record_gate:
        expected_validation["record_gate"] = {
            "official_execution_ready": True,
            "required_metrics_ready": True,
            "source_revision_ready": True,
            "dependency_environment_ready": True,
            "model_source_ready": True,
            "openclip_source_ready": True,
            "metric_validation": metric_validation,
            "record_ready": True,
        }
    records_path.write_text(
        json.dumps(expected_record, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    validation_path.write_text(
        json.dumps(expected_validation, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    builder_calls: list[dict[str, Any]] = []

    def rebuild_record(**kwargs: Any) -> dict[str, Any]:
        """记录打包重建输入, 证明规范配置来自已复验科学单元."""

        builder_calls.append(kwargs)
        return dict(expected_record)

    monkeypatch.setattr(runner, builder_name, rebuild_record)
    monkeypatch.setattr(
        runner,
        validator_name,
        lambda rows: dict(base_validation)
        if list(rows) == [expected_record]
        else {"reference_import_ready": False},
    )
    if prefix == "tree_ring":
        monkeypatch.setattr(
            runner,
            "validate_tree_ring_metric_summary",
            lambda *_args: dict(metric_validation),
        )
    if prefix == "shallow_diffuse":
        monkeypatch.setattr(
            runner,
            "validate_shallow_diffuse_metric_summary",
            lambda *_args: dict(metric_validation),
        )

    summary: defaultdict[str, Any] = defaultdict(str)
    summary.update(
        {
            "sample_count": 10,
            "official_run_name": runner.DEFAULT_RUN_NAME,
            "primary_edit_timestep": 15,
        }
    )
    unit_validation = {
        "metric_summary": {},
        "official_scientific_config": {"sample_count": 20},
        "official_scientific_config_digest": "e" * 64,
        "stable_unit_identity": {
            "official_repository_commit": "b" * 40,
            "source_worktree_digest": "c" * 64,
            "source_patch_sha256": "d" * 64,
        },
    }
    helper = getattr(runner, helper_name)
    helper(tmp_path, source_dir, summary, unit_validation)
    assert builder_calls[-1]["source_provenance"]["official_scientific_config"] == {
        "sample_count": 20
    }
    assert (
        builder_calls[-1]["source_provenance"][
            "official_scientific_config_digest"
        ]
        == "e" * 64
    )

    records_path.write_text('{"tampered":true}\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match="record"):
        helper(tmp_path, source_dir, summary, unit_validation)

    records_path.write_text(
        json.dumps(expected_record, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    validation_path.write_text('{"tampered":true}\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match="validation"):
        helper(tmp_path, source_dir, summary, unit_validation)
