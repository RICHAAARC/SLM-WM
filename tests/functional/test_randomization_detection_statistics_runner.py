"""验证检测统计 runner 的来源连接、身份反例与事务发布."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.protocol.attacks import (
    attack_config_digest,
    default_attack_configs,
    formal_attack_seed_protocol_record,
    formal_attack_seed_random,
)
from experiments.protocol.detection_key_identity import (
    REGISTERED_WATERMARK_KEY_ROLE,
    REGISTERED_WRONG_KEY_ROLE,
)
from experiments.protocol.formal_randomization import formal_randomization_repeat_ids
from main.core.digest import build_stable_digest
from paper_experiments.analysis.randomization_detection_statistics import (
    DETECTION_METHOD_IDS,
    RANDOMIZATION_DETECTION_OPERATING_POINT_FIELDNAMES,
    RANDOMIZATION_PER_ATTACK_COMPARISON_FIELDNAMES,
    RANDOMIZATION_PER_ATTACK_DETECTION_FIELDNAMES,
    RANDOMIZATION_WRONG_KEY_FIELDNAMES,
)
from paper_experiments.runners.randomization_aggregate_provenance import (
    RandomizationAggregateProvenance,
)
from paper_experiments.runners.randomization_method_repeat_thresholds import (
    RandomizationMethodRepeatReconstruction,
)
from paper_experiments.runners import randomization_detection_statistics as runner


pytestmark = pytest.mark.quick

CODE_VERSION = "a" * 40
AGGREGATE_SHA = build_stable_digest({"aggregate_package": "detection"})
AGGREGATE_DIGEST = build_stable_digest({"aggregate": "detection"})


def _attack_registry() -> tuple[dict[str, str], ...]:
    """构造 runner 使用的17项正式攻击 registry."""

    return tuple(
        {
            "attack_id": config.attack_id,
            "attack_family": config.attack_family,
            "attack_name": config.attack_name,
            "resource_profile": config.resource_profile,
            "attack_config_digest": attack_config_digest(config),
        }
        for config in default_attack_configs()
        if config.enabled and config.resource_profile in {"full_main", "full_extra"}
    )


ATTACK_REGISTRY = _attack_registry()


def _provenance(tmp_path: Path) -> RandomizationAggregateProvenance:
    """构造保持 validator 冻结关系的内存 provenance."""

    package_path = tmp_path / "aggregate.zip"
    package_path.write_bytes(b"aggregate")
    payload = {
        "paper_run_name": "probe_paper",
        "target_fpr": 0.1,
        "randomization_aggregate_ready": True,
        "supports_paper_claim": False,
        "randomization_aggregate_digest": AGGREGATE_DIGEST,
        "common_code_version": CODE_VERSION,
        "randomization_repeat_ids": list(formal_randomization_repeat_ids()),
    }
    return RandomizationAggregateProvenance(
        package_path=package_path,
        package_sha256=AGGREGATE_SHA,
        payload_path="randomization_aggregate.json",
        payload_sha256=build_stable_digest(payload),
        manifest_path="manifest.local.json",
        manifest_sha256=build_stable_digest({"manifest": "detection"}),
        payload=payload,
        manifest={},
        randomization_repeat_components=(),
        invariant_packages=(),
        common_code_version=CODE_VERSION,
        randomization_aggregate_digest=AGGREGATE_DIGEST,
    )


def _identity(repeat_id: str, prompt_index: int) -> dict[str, object]:
    """构造五方法共享的最小 Prompt 随机身份."""

    repeat_index = formal_randomization_repeat_ids().index(repeat_id)
    return {
        "randomization_repeat_id": repeat_id,
        "generation_seed_index": repeat_index // 3,
        "generation_seed_offset": repeat_index // 3,
        "generation_seed_random": 1703 + prompt_index + repeat_index // 3,
        "watermark_key_index": repeat_index % 3,
        "watermark_key_seed_random": 9000 + repeat_index % 3,
        "watermark_key_material_digest_random": build_stable_digest(
            {"key": repeat_index % 3}
        ),
        "formal_randomization_protocol_digest": build_stable_digest(
            {"protocol": "randomization"}
        ),
        "formal_randomization_identity_digest_random": build_stable_digest(
            {"identity": repeat_id, "prompt": prompt_index}
        ),
        "base_latent_content_digest_random": build_stable_digest(
            {"latent": repeat_id, "prompt": prompt_index}
        ),
        "base_latent_identity_digest_random": build_stable_digest(
            {"latent_identity": repeat_id, "prompt": prompt_index}
        ),
    }


def _observation_rows(repeat_id: str, method_id: str) -> tuple[dict[str, object], ...]:
    """构造单个 method-repeat 的完整 test 角色和攻击集合."""

    rows: list[dict[str, object]] = []
    for prompt_index in range(34):
        prompt_id = f"prompt_{prompt_index:03d}"
        identity = _identity(repeat_id, prompt_index)
        clean_roles = (
            ("clean_negative", "positive_source", "wrong_key_negative")
            if method_id == "slm_wm"
            else ("clean_negative", "positive_source")
        )
        for role in clean_roles:
            rows.append(
                {
                    "method_id": method_id,
                    "baseline_id": "" if method_id == "slm_wm" else method_id,
                    "prompt_id": prompt_id,
                    "split": "test",
                    "sample_role": role,
                    "attack_family": "clean",
                    "attack_name": "clean_none",
                    "declared_decision": role == "positive_source",
                    **identity,
                }
            )
        for attack in ATTACK_REGISTRY:
            roles = (
                ("clean_negative", "positive_source")
                if method_id == "slm_wm"
                else ("attacked_negative", "attacked_positive")
            )
            for role in roles:
                rows.append(
                    {
                        "method_id": method_id,
                        "baseline_id": "" if method_id == "slm_wm" else method_id,
                        "prompt_id": prompt_id,
                        "split": "test",
                        "sample_role": role,
                        **attack,
                        "attack_seed_random": formal_attack_seed_random(
                            int(identity["generation_seed_random"]),
                            attack["attack_id"],
                        ),
                        "formal_attack_seed_protocol_digest": (
                            formal_attack_seed_protocol_record()[
                                "formal_attack_seed_protocol_digest"
                            ]
                        ),
                        "declared_decision": role
                        in {"positive_source", "attacked_positive"},
                        **identity,
                    }
                )
    return tuple(rows)


def _reconstruction() -> RandomizationMethodRepeatReconstruction:
    """构造45个完整来源及其独立阈值记录."""

    sources = []
    thresholds = []
    for repeat_id in formal_randomization_repeat_ids():
        for method_id in DETECTION_METHOD_IDS:
            source_sha = build_stable_digest(
                {"source": repeat_id, "method": method_id}
            )
            source = SimpleNamespace(
                randomization_repeat_id=repeat_id,
                method_id=method_id,
                observation_source_sha256=source_sha,
                randomization_aggregate_digest=AGGREGATE_DIGEST,
                common_code_version=CODE_VERSION,
                observation_rows=_observation_rows(repeat_id, method_id),
            )
            threshold = {
                "randomization_repeat_id": repeat_id,
                "method_id": method_id,
                "fixed_fpr_threshold_ready": True,
                "calibrated_detection_threshold": 0.5,
                "threshold_digest": build_stable_digest(
                    {"threshold": repeat_id, "method": method_id}
                ),
                "observation_source_sha256": source_sha,
                "randomization_aggregate_digest": AGGREGATE_DIGEST,
                "common_code_version": CODE_VERSION,
            }
            threshold["method_repeat_threshold_record_digest"] = build_stable_digest(
                threshold
            )
            sources.append(source)
            thresholds.append(threshold)
    return RandomizationMethodRepeatReconstruction(
        method_sources=tuple(sources),
        threshold_records=tuple(thresholds),
        fairness_records=(),
        report={
            "method_repeat_fixed_fpr_report_digest": build_stable_digest(
                {"threshold_report": "detection"}
            )
        },
        reconstruction_report={
            "reconstruction_report_digest": build_stable_digest(
                {"reconstruction": "detection"}
            )
        },
    )


def _fake_decision_atom(row, *, repeat_id: str, expected_threshold_digest: str, **_kwargs):
    """隔离角色、集合和攻击公式测试, 不绕过生产入口测试."""

    decision = bool(row["declared_decision"])
    image_role = (
        "watermarked"
        if row["sample_role"] in {"positive_source", "wrong_key_negative"}
        else "clean"
    )
    return decision, {
        "randomization_repeat_id": repeat_id,
        "threshold_digest": expected_threshold_digest,
        "decision": decision,
        "raw_decision": decision,
        "evaluated_image_path": (
            f"outputs/{repeat_id}/{row['prompt_id']}_{image_role}.png"
        ),
        "evaluated_image_digest": build_stable_digest(
            {
                "repeat_id": repeat_id,
                "prompt_id": row["prompt_id"],
                "image_role": image_role,
            }
        ),
        "source_digest": build_stable_digest(dict(row)),
    }


def test_cluster_builder_exactly_covers_registered_repeats_methods_and_attacks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """完整来源必须生成34×181个聚类且每个聚类含9重复."""

    monkeypatch.setattr(runner, "_main_decision_and_source_atom", _fake_decision_atom)
    monkeypatch.setattr(
        runner,
        "_baseline_decision_and_source_atom",
        _fake_decision_atom,
    )
    records, threshold_map, threshold_map_digest = runner._build_cluster_records(
        _reconstruction(),
        paper_run_name="probe_paper",
        attack_registry=ATTACK_REGISTRY,
    )

    assert len(records) == 34 * (5 * 2 + 1 + 5 * 17 * 2)
    assert set(threshold_map) == set(formal_randomization_repeat_ids())
    assert all(set(methods) == set(DETECTION_METHOD_IDS) for methods in threshold_map.values())
    assert threshold_map_digest == build_stable_digest(threshold_map)
    assert all(record["randomization_repeat_count"] == 9 for record in records)


def test_cluster_builder_rejects_attack_seed_drift_and_role_exchange(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """攻击 seed 漂移与 baseline 攻击角色互换必须失败."""

    monkeypatch.setattr(runner, "_main_decision_and_source_atom", _fake_decision_atom)
    monkeypatch.setattr(
        runner,
        "_baseline_decision_and_source_atom",
        _fake_decision_atom,
    )
    rebuilt = _reconstruction()
    sources = list(rebuilt.method_sources)
    attacked_index = next(
        index
        for index, row in enumerate(sources[0].observation_rows)
        if row.get("attack_id")
    )
    mutated_rows = [dict(row) for row in sources[0].observation_rows]
    mutated_rows[attacked_index]["attack_seed_random"] += 1
    sources[0] = SimpleNamespace(
        **{
            **sources[0].__dict__,
            "observation_rows": tuple(mutated_rows),
        }
    )
    drifted = RandomizationMethodRepeatReconstruction(
        method_sources=tuple(sources),
        threshold_records=rebuilt.threshold_records,
        fairness_records=(),
        report=rebuilt.report,
        reconstruction_report=rebuilt.reconstruction_report,
    )
    with pytest.raises(
        runner.RandomizationDetectionStatisticsRunnerError,
        match="攻击 seed",
    ):
        runner._build_cluster_records(
            drifted,
            paper_run_name="probe_paper",
            attack_registry=ATTACK_REGISTRY,
        )

    rebuilt = _reconstruction()
    sources = list(rebuilt.method_sources)
    baseline_source_index = next(
        index
        for index, source in enumerate(sources)
        if source.method_id == "tree_ring"
    )
    mutated_rows = [dict(row) for row in sources[baseline_source_index].observation_rows]
    attacked_positive_index = next(
        index
        for index, row in enumerate(mutated_rows)
        if row.get("sample_role") == "attacked_positive"
    )
    mutated_rows[attacked_positive_index]["sample_role"] = "attacked_negative"
    sources[baseline_source_index] = SimpleNamespace(
        **{
            **sources[baseline_source_index].__dict__,
            "observation_rows": tuple(mutated_rows),
        }
    )
    exchanged = RandomizationMethodRepeatReconstruction(
        method_sources=tuple(sources),
        threshold_records=rebuilt.threshold_records,
        fairness_records=(),
        report=rebuilt.report,
        reconstruction_report=rebuilt.reconstruction_report,
    )
    with pytest.raises(
        runner.RandomizationDetectionStatisticsRunnerError,
        match="角色键重复|精确覆盖",
    ):
        runner._build_cluster_records(
            exchanged,
            paper_run_name="probe_paper",
            attack_registry=ATTACK_REGISTRY,
        )


def test_source_binding_rejects_duplicate_keys_and_wrong_key_image_swap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """重复来源不得覆盖, wrong-key 必须检测同一 watermarked 图像."""

    rebuilt = _reconstruction()
    duplicated = RandomizationMethodRepeatReconstruction(
        method_sources=(*rebuilt.method_sources, rebuilt.method_sources[0]),
        threshold_records=rebuilt.threshold_records,
        fairness_records=(),
        report=rebuilt.report,
        reconstruction_report=rebuilt.reconstruction_report,
    )
    with pytest.raises(
        runner.RandomizationDetectionStatisticsRunnerError,
        match="数量必须精确等于45",
    ):
        runner._validate_threshold_source_binding(duplicated)
    duplicated_threshold = RandomizationMethodRepeatReconstruction(
        method_sources=rebuilt.method_sources,
        threshold_records=(
            *rebuilt.threshold_records,
            rebuilt.threshold_records[0],
        ),
        fairness_records=(),
        report=rebuilt.report,
        reconstruction_report=rebuilt.reconstruction_report,
    )
    with pytest.raises(
        runner.RandomizationDetectionStatisticsRunnerError,
        match="数量必须精确等于45",
    ):
        runner._validate_threshold_source_binding(duplicated_threshold)

    def swapped_wrong_key_atom(row, **kwargs):
        decision, atom = _fake_decision_atom(row, **kwargs)
        if (
            row["sample_role"] == "wrong_key_negative"
            and row["prompt_id"] == "prompt_000"
        ):
            atom["evaluated_image_path"] = "outputs/swapped.png"
            atom["evaluated_image_digest"] = build_stable_digest(
                {"swapped_wrong_key_image": True}
            )
        return decision, atom

    monkeypatch.setattr(
        runner,
        "_main_decision_and_source_atom",
        swapped_wrong_key_atom,
    )
    monkeypatch.setattr(
        runner,
        "_baseline_decision_and_source_atom",
        _fake_decision_atom,
    )
    with pytest.raises(
        runner.RandomizationDetectionStatisticsRunnerError,
        match="相同的水印图像",
    ):
        runner._build_cluster_records(
            rebuilt,
            paper_run_name="probe_paper",
            attack_registry=ATTACK_REGISTRY,
        )


def test_main_key_role_and_baseline_score_must_rebuild(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """wrong-key 身份与 baseline 分数判定属于不可绕过的 P1 边界."""

    monkeypatch.setattr(
        runner,
        "validate_image_only_measurement_projection_record",
        lambda row: {},
    )
    monkeypatch.setattr(
        runner,
        "formal_watermark_key_material_from_seed",
        lambda seed, repeat: "registered-key",
    )

    def fake_key_identity(row, key_plan):
        return {
            "detection_key_role": row["detection_key_role"],
            "detection_key_plan_digest_random": build_stable_digest(key_plan),
        }

    monkeypatch.setattr(
        runner,
        "validate_detection_key_identity_record",
        fake_key_identity,
    )
    threshold_digest = build_stable_digest({"threshold": "main"})
    main_row = {
        "frozen_threshold_digest": threshold_digest,
        "metadata": {
            "detector_input_access_mode": "image_key_public_model_only",
            "blind_image_detector": True,
            "generation_latent_trace_required": False,
        },
        "evaluated_image_path": "outputs/image.png",
        "evaluated_image_digest": build_stable_digest({"image": 1}),
        "watermark_key_seed_random": 9000,
        "detection_key_role": REGISTERED_WATERMARK_KEY_ROLE,
        "formal_evidence_positive": False,
        "formal_positive_by_content": False,
        "content_score": 0.1,
        "aligned_content_score": None,
        "measurement_digest": build_stable_digest({"detector": 1}),
    }
    with pytest.raises(
        runner.RandomizationDetectionStatisticsRunnerError,
        match="角色混用",
    ):
        runner._main_decision_and_source_atom(
            main_row,
            repeat_id=formal_randomization_repeat_ids()[0],
            expected_threshold_digest=threshold_digest,
            expected_wrong_key=True,
            attacked=False,
        )
    main_row["detection_key_role"] = REGISTERED_WRONG_KEY_ROLE
    decision, atom = runner._main_decision_and_source_atom(
        main_row,
        repeat_id=formal_randomization_repeat_ids()[0],
        expected_threshold_digest=threshold_digest,
        expected_wrong_key=True,
        attacked=False,
    )
    assert decision is False
    assert atom["detection_key_role"] == REGISTERED_WRONG_KEY_ROLE

    baseline_row = {
        "baseline_id": "tree_ring",
        "threshold": 0.5,
        "threshold_source": "nested_calibration_threshold_freeze_conformal_v1",
        "score": 0.9,
        "detection_decision": False,
        "image_path": "outputs/tree_ring.png",
        "image_digest": build_stable_digest({"image": "tree_ring"}),
    }
    with pytest.raises(
        runner.RandomizationDetectionStatisticsRunnerError,
        match="判定无法",
    ):
        runner._baseline_decision_and_source_atom(
            baseline_row,
            method_id="tree_ring",
            repeat_id=formal_randomization_repeat_ids()[0],
            calibrated_threshold=0.5,
            expected_threshold_digest=build_stable_digest(
                {"threshold": "tree_ring"}
            ),
        )


def _result() -> runner.RandomizationDetectionStatisticsResult:
    """构造事务 writer 所需的最小已验证结果对象."""

    def row(fieldnames: tuple[str, ...]) -> dict[str, object]:
        return {field_name: "" for field_name in fieldnames}

    summary = {
        "cluster_record_set_digest": build_stable_digest({"clusters": 1}),
        "randomization_detection_statistics_summary_digest": build_stable_digest(
            {"summary": 1}
        ),
        "main_method_clean_fixed_fpr_ready": False,
        "main_method_wrong_key_fixed_fpr_ready": False,
        "randomization_detection_statistics_ready": True,
        "supports_paper_claim": False,
    }
    report = {
        "paper_run_name": "probe_paper",
        "target_fpr": 0.1,
        "method_repeat_threshold_map_digest": build_stable_digest(
            {"threshold_map": 1}
        ),
        "attack_registry_digest": build_stable_digest({"attacks": 1}),
        "randomization_detection_statistics_report_digest": build_stable_digest(
            {"report": 1}
        ),
    }
    return runner.RandomizationDetectionStatisticsResult(
        cluster_records=({"cluster": 1},),
        operating_point_rows=(row(RANDOMIZATION_DETECTION_OPERATING_POINT_FIELDNAMES),),
        per_attack_rows=(row(RANDOMIZATION_PER_ATTACK_DETECTION_FIELDNAMES),),
        wrong_key_rows=(row(RANDOMIZATION_WRONG_KEY_FIELDNAMES),),
        per_attack_comparison_rows=(
            row(RANDOMIZATION_PER_ATTACK_COMPARISON_FIELDNAMES),
        ),
        summary=summary,
        report=report,
    )


def test_writer_publishes_negative_result_transactionally(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """坏数值仍必须发布完整包, 且 manifest 不得提升论文主张."""

    provenance = _provenance(tmp_path)
    monkeypatch.setattr(
        runner,
        "rebuild_randomization_detection_statistics",
        lambda source, root=".": _result(),
    )
    manifest_path = runner.write_randomization_detection_statistics_outputs(
        provenance,
        root=tmp_path,
        output_dir=tmp_path / "outputs" / "detection_negative",
    )

    assert manifest_path.is_file()
    output_names = {path.name for path in manifest_path.parent.iterdir()}
    assert output_names == {
        "prompt_cluster_detection_records.jsonl",
        "method_detection_operating_points.csv",
        "method_attack_detection_metrics.csv",
        "slm_wrong_key_detection_metric.csv",
        "per_attack_superiority_table.csv",
        "randomization_detection_statistics_summary.json",
        "randomization_detection_statistics_report.json",
        "manifest.local.json",
    }
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["metadata"]["randomization_detection_statistics_ready"] is True
    assert manifest["metadata"]["main_method_clean_fixed_fpr_ready"] is False
    assert manifest["metadata"]["supports_paper_claim"] is False
