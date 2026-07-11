from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.protocol.fixed_fpr_observation_audit import (
    FORMAL_THRESHOLD_SOURCE,
    audit_fixed_fpr_observation_threshold,
    conformal_threshold_from_clean_negative_scores,
)
from experiments.runners.image_only_dataset_runtime import (
    apply_frozen_evidence_protocol,
    calibrate_complete_evidence_protocol,
)
from main.core.digest import build_stable_digest
from paper_experiments.analysis.fixed_fpr_threshold_audit import (
    MAIN_THRESHOLD_SOURCE,
    audit_baseline_fixed_fpr,
    audit_main_method_fixed_fpr,
    build_fixed_fpr_threshold_audit_report,
    build_fixed_fpr_threshold_manifest_config,
)
from scripts import write_fixed_fpr_threshold_audit_outputs as threshold_writer


pytestmark = pytest.mark.quick


OBSERVATION_SOURCE_SHA256 = "a" * 64


def _main_method_rows() -> tuple[tuple[dict[str, object], ...], dict[str, object]]:
    """构造包含 calibration 和 test clean negative 的主方法记录。"""

    raw_rows = tuple(
        {
            "prompt_id": f"prompt-{index}",
            "split": split,
            "sample_role": "clean_negative",
            "attack_id": "",
            "content_score": score,
            "aligned_content_score": score,
            "attention_geometry_score": 0.0,
            "geometry_reliable": False,
        }
        for index, (split, score) in enumerate(
            (
                ("calibration", 0.1),
                ("calibration", 0.2),
                ("calibration", 0.3),
                ("test", 0.15),
                ("test", 0.35),
            )
        )
    )
    protocol = calibrate_complete_evidence_protocol(
        raw_rows[:3],
        target_fpr=0.25,
        rescue_margin_low=-0.05,
    )
    return apply_frozen_evidence_protocol(raw_rows, protocol), protocol.to_dict()


def _baseline_rows() -> tuple[dict[str, object], ...]:
    """构造共享 calibration 冻结阈值的 baseline observation。"""

    threshold = conformal_threshold_from_clean_negative_scores((0.1, 0.2, 0.3), 0.25)
    return tuple(
        {
            "prompt_id": f"prompt-{index}",
            "event_id": f"event-{index}",
            "split": split,
            "sample_role": "clean_negative",
            "attack_family": "clean",
            "score": score,
            "threshold": threshold,
            "threshold_source": FORMAL_THRESHOLD_SOURCE,
            "detection_decision": score >= threshold,
        }
        for index, (split, score) in enumerate(
            (
                ("calibration", 0.1),
                ("calibration", 0.2),
                ("calibration", 0.3),
                ("test", 0.15),
                ("test", 0.35),
            )
        )
    )


def test_main_method_threshold_audit_recomputes_complete_rescue_protocol() -> None:
    rows, protocol = _main_method_rows()

    result = audit_main_method_fixed_fpr(
        rows,
        protocol,
        observation_source_sha256=OBSERVATION_SOURCE_SHA256,
        target_fpr=0.25,
        expected_calibration_negative_count=3,
        expected_test_negative_count=2,
    )

    assert result["fixed_fpr_threshold_ready"] is True
    assert result["protocol_value_ready"] is True
    assert result["threshold_source"] == MAIN_THRESHOLD_SOURCE
    assert "conformal" not in result["threshold_source"]
    tampered = [dict(row) for row in rows]
    tampered[0]["formal_evidence_positive"] = not bool(
        tampered[0]["formal_evidence_positive"]
    )
    failed = audit_main_method_fixed_fpr(
        tampered,
        protocol,
        observation_source_sha256=OBSERVATION_SOURCE_SHA256,
        target_fpr=0.25,
        expected_calibration_negative_count=3,
        expected_test_negative_count=2,
    )
    assert failed["detection_decision_ready"] is False
    assert failed["fixed_fpr_threshold_ready"] is False

    tampered_margin = [dict(row) for row in rows]
    tampered_margin[0]["formal_raw_content_margin"] = 999.0
    failed_margin = audit_main_method_fixed_fpr(
        tampered_margin,
        protocol,
        observation_source_sha256=OBSERVATION_SOURCE_SHA256,
        target_fpr=0.25,
        expected_calibration_negative_count=3,
        expected_test_negative_count=2,
    )
    assert failed_margin["detection_decision_ready"] is False


def test_baseline_threshold_audit_binds_recomputed_threshold_and_digest() -> None:
    rows = _baseline_rows()
    primitive_audit = audit_fixed_fpr_observation_threshold(
        rows,
        target_fpr=0.25,
        expected_calibration_negative_count=3,
    )

    result = audit_baseline_fixed_fpr(
        "tree_ring",
        rows,
        observation_source_sha256=OBSERVATION_SOURCE_SHA256,
        target_fpr=0.25,
        expected_calibration_negative_count=3,
        expected_test_negative_count=2,
        declared_threshold=primitive_audit.frozen_threshold,
        declared_threshold_digest=primitive_audit.threshold_digest,
    )

    assert result["fixed_fpr_threshold_ready"] is True
    failed = audit_baseline_fixed_fpr(
        "tree_ring",
        rows,
        observation_source_sha256=OBSERVATION_SOURCE_SHA256,
        target_fpr=0.25,
        expected_calibration_negative_count=3,
        expected_test_negative_count=2,
        declared_threshold=primitive_audit.frozen_threshold,
        declared_threshold_digest="0" * 64,
    )
    assert failed["protocol_value_ready"] is False
    assert failed["fixed_fpr_threshold_ready"] is False


def test_threshold_audit_report_requires_exact_five_method_identity_set() -> None:
    method_ids = (
        "slm_wm",
        "tree_ring",
        "gaussian_shading",
        "shallow_diffuse",
        "t2smark",
    )
    rows = tuple(
        {
            "method_id": method_id,
            "target_fpr": 0.1,
            "threshold_digest": f"{index + 1:x}" * 64,
            "observation_source_sha256": f"{index + 6:x}" * 64,
            "fixed_fpr_threshold_ready": True,
        }
        for index, method_id in enumerate(method_ids)
    )

    report = build_fixed_fpr_threshold_audit_report(
        rows,
        paper_run_name="probe_paper",
        target_fpr=0.1,
    )
    assert report["fixed_fpr_threshold_audit_ready"] is True
    assert report["threshold_observation_binding_ready"] is True
    assert report["method_observation_source_sha256_map"] == {
        method_id: f"{index + 6:x}" * 64
        for index, method_id in enumerate(method_ids)
    }
    assert report["method_threshold_digest_map"] == {
        method_id: f"{index + 1:x}" * 64
        for index, method_id in enumerate(method_ids)
    }
    reordered = build_fixed_fpr_threshold_audit_report(
        reversed(rows),
        paper_run_name="probe_paper",
        target_fpr=0.1,
    )
    assert reordered["threshold_audit_rows_digest"] == report[
        "threshold_audit_rows_digest"
    ]

    duplicate_rows = (*rows[:-1], dict(rows[0]))
    failed = build_fixed_fpr_threshold_audit_report(
        duplicate_rows,
        paper_run_name="probe_paper",
        target_fpr=0.1,
    )
    assert failed["method_identity_ready"] is False
    assert failed["threshold_observation_binding_ready"] is False
    assert failed["fixed_fpr_threshold_audit_ready"] is False

    incomplete = build_fixed_fpr_threshold_audit_report(
        rows[:-1],
        paper_run_name="probe_paper",
        target_fpr=0.1,
    )
    assert incomplete["expected_method_ids"] == list(method_ids)
    assert incomplete["method_identity_ready"] is False
    assert incomplete["fixed_fpr_threshold_audit_ready"] is False

    malformed_digest_rows = tuple(dict(row) for row in rows)
    malformed_digest_rows[0]["observation_source_sha256"] = "A" * 64
    malformed = build_fixed_fpr_threshold_audit_report(
        malformed_digest_rows,
        paper_run_name="probe_paper",
        target_fpr=0.1,
    )
    assert malformed["all_method_thresholds_ready"] is True
    assert malformed["threshold_observation_binding_ready"] is False
    assert malformed["fixed_fpr_threshold_audit_ready"] is False


def test_threshold_audit_writer_binds_actual_observation_file_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """核验 writer 重算五个 observation 文件的字节摘要并同步 manifest."""

    run_name = "probe_paper"
    target_fpr = 0.25
    paper_run = SimpleNamespace(
        run_name=run_name,
        target_fpr=target_fpr,
        to_dict=lambda: {
            "run_name": run_name,
            "target_fpr": target_fpr,
        },
    )
    monkeypatch.setattr(threshold_writer, "build_paper_run_config", lambda root: paper_run)
    monkeypatch.setattr(
        threshold_writer,
        "build_group_split_counts",
        lambda prompt_count: {"calibration": 3, "test": 2},
    )
    paper_run.prompt_count = 5
    monkeypatch.setattr(threshold_writer, "resolve_code_version", lambda root: "test-code")

    main_rows, main_protocol = _main_method_rows()
    main_dir = tmp_path / "outputs" / "image_only_dataset_runtime" / run_name
    main_dir.mkdir(parents=True)
    main_observation_path = main_dir / "image_only_detection_records.jsonl"
    main_observation_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in main_rows),
        encoding="utf-8",
    )
    (main_dir / "frozen_evidence_protocol.json").write_text(
        json.dumps(main_protocol, sort_keys=True),
        encoding="utf-8",
    )

    baseline_rows = _baseline_rows()
    primitive_audit = audit_fixed_fpr_observation_threshold(
        baseline_rows,
        target_fpr=target_fpr,
        expected_calibration_negative_count=3,
    )
    common_sources = []
    observation_paths = {"slm_wm": main_observation_path}
    collection_dir = tmp_path / "outputs" / "external_baseline_method_faithful" / run_name
    for index, baseline_id in enumerate(
        ("tree_ring", "gaussian_shading", "shallow_diffuse")
    ):
        baseline_dir = collection_dir / baseline_id
        baseline_dir.mkdir(parents=True)
        observations_path = baseline_dir / "baseline_observations.json"
        observations_path.write_text(
            json.dumps(list(baseline_rows), sort_keys=True) + "\n" + (" " * index),
            encoding="utf-8",
        )
        supporting_paths = tuple(
            baseline_dir / name
            for name in (
                "transfer_manifest.json",
                "prompt_plan.json",
                "adapter_manifest.json",
                "execution_manifest.json",
            )
        )
        for path in supporting_paths:
            path.write_text("{}\n", encoding="utf-8")
        common_sources.append(
            SimpleNamespace(
                baseline_id=baseline_id,
                rows=baseline_rows,
                observations_path=observations_path,
                transfer_manifest_path=supporting_paths[0],
                prompt_plan_path=supporting_paths[1],
                adapter_manifest_path=supporting_paths[2],
                execution_manifest_path=supporting_paths[3],
                transfer_manifest={
                    "threshold": primitive_audit.frozen_threshold,
                    "threshold_digest": primitive_audit.threshold_digest,
                },
            )
        )
        observation_paths[baseline_id] = observations_path
    monkeypatch.setattr(
        threshold_writer,
        "load_method_faithful_observation_collection",
        lambda collection_root, project_root: tuple(common_sources),
    )

    t2_dir = tmp_path / "outputs" / "t2smark_formal_reproduction" / run_name
    (t2_dir / "t2smark_adapter").mkdir(parents=True)
    t2_observation_path = t2_dir / "t2smark_adapter" / "baseline_observations.json"
    t2_observation_path.write_text(
        json.dumps(list(baseline_rows), sort_keys=True),
        encoding="utf-8",
    )
    observation_paths["t2smark"] = t2_observation_path
    (t2_dir / "t2smark_formal_import_candidate_records.jsonl").write_text(
        json.dumps(
            {
                "calibrated_detection_threshold": primitive_audit.frozen_threshold,
                "threshold_digest": primitive_audit.threshold_digest,
                "target_fpr": target_fpr,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    output_dir = tmp_path / "outputs" / "fixed_fpr_threshold_audit" / run_name
    report = threshold_writer.write_fixed_fpr_threshold_audit_outputs(
        root=tmp_path,
        output_dir=output_dir,
        method_faithful_collection_root=collection_dir,
        t2smark_output_dir=t2_dir,
    )

    expected_observation_map = {
        method_id: hashlib.sha256(path.read_bytes()).hexdigest()
        for method_id, path in observation_paths.items()
    }
    assert report["method_observation_source_sha256_map"] == expected_observation_map
    assert report["threshold_observation_binding_ready"] is True
    with (output_dir / "threshold_audit_rows.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        csv_rows = tuple(csv.DictReader(handle))
    assert {row["method_id"]: row["observation_source_sha256"] for row in csv_rows} == (
        expected_observation_map
    )
    manifest = json.loads((output_dir / "manifest.local.json").read_text(encoding="utf-8"))
    assert manifest["metadata"] == report
    assert manifest["config_digest"] == build_stable_digest(
        build_fixed_fpr_threshold_manifest_config(report)
    )
