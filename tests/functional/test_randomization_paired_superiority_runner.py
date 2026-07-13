"""验证跨重复配对 runner 的来源连接、Git 边界与事务发布."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.protocol.formal_randomization import formal_randomization_repeat_ids
from main.core.digest import build_stable_digest
from paper_experiments.analysis.paired_superiority import PRIMARY_BASELINE_IDS
from paper_experiments.analysis.randomization_paired_superiority import (
    RANDOMIZATION_PAIRED_SUPERIORITY_FIELDNAMES,
)
from paper_experiments.runners.randomization_aggregate_provenance import (
    RandomizationAggregateProvenance,
)
from paper_experiments.runners.randomization_method_repeat_thresholds import (
    RandomizationMethodRepeatReconstruction,
)
from paper_experiments.runners import randomization_paired_superiority as runner


pytestmark = pytest.mark.quick

CODE_VERSION = "a" * 40
AGGREGATE_SHA = build_stable_digest({"aggregate_package": 1})
AGGREGATE_DIGEST = build_stable_digest({"aggregate": 1})


def _provenance(tmp_path: Path) -> RandomizationAggregateProvenance:
    """构造保持 validator 冻结字段关系的内存 provenance."""

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
        manifest_sha256=build_stable_digest({"manifest": 1}),
        payload=payload,
        manifest={},
        randomization_repeat_components=(),
        invariant_packages=(),
        common_code_version=CODE_VERSION,
        randomization_aggregate_digest=AGGREGATE_DIGEST,
    )


def _reconstruction() -> RandomizationMethodRepeatReconstruction:
    """构造已通过上游阈值边界的45个来源与阈值记录."""

    sources = []
    thresholds = []
    for repeat_id in formal_randomization_repeat_ids():
        for method_id in ("slm_wm", *PRIMARY_BASELINE_IDS):
            source_sha = build_stable_digest(
                {"source": repeat_id, "method": method_id}
            )
            leaf_sha = build_stable_digest(
                {"leaf": repeat_id, "method": method_id}
            )
            component_sha = build_stable_digest({"component": repeat_id})
            member = f"records/{repeat_id}/{method_id}.jsonl"
            source = SimpleNamespace(
                randomization_repeat_id=repeat_id,
                method_id=method_id,
                observation_source_sha256=source_sha,
                observation_archive_member=member,
                leaf_package_sha256=leaf_sha,
                randomization_repeat_component_sha256=component_sha,
                randomization_aggregate_package_sha256=AGGREGATE_SHA,
                randomization_aggregate_digest=AGGREGATE_DIGEST,
                common_code_version=CODE_VERSION,
                observation_rows=({"source": source_sha},),
            )
            threshold = {
                "randomization_repeat_id": repeat_id,
                "method_id": method_id,
                "fixed_fpr_threshold_ready": True,
                "threshold_digest": build_stable_digest(
                    {"threshold": repeat_id, "method": method_id}
                ),
                "observation_source_sha256": source_sha,
                "observation_archive_member": member,
                "leaf_package_sha256": leaf_sha,
                "randomization_repeat_component_sha256": component_sha,
                "randomization_aggregate_package_sha256": AGGREGATE_SHA,
                "randomization_aggregate_digest": AGGREGATE_DIGEST,
                "common_code_version": CODE_VERSION,
            }
            threshold["method_repeat_threshold_record_digest"] = (
                build_stable_digest(threshold)
            )
            sources.append(source)
            thresholds.append(threshold)
    report = {
        "method_repeat_fixed_fpr_report_digest": build_stable_digest(
            {"threshold_report": 1}
        ),
        "fairness_records_digest": build_stable_digest({"fairness": 1}),
        "prompt_protocol_digest": build_stable_digest({"prompts": 1}),
    }
    reconstruction_report = {
        "reconstruction_report_digest": build_stable_digest(
            {"reconstruction": 1}
        )
    }
    return RandomizationMethodRepeatReconstruction(
        method_sources=tuple(sources),
        threshold_records=tuple(thresholds),
        fairness_records=(),
        report=report,
        reconstruction_report=reconstruction_report,
    )


def _result() -> runner.RandomizationPairedSuperiorityResult:
    """构造 writer 事务测试所需的最小完整结果."""

    statistic_row = {
        field_name: "" for field_name in RANDOMIZATION_PAIRED_SUPERIORITY_FIELDNAMES
    }
    summary = {
        "paired_outcome_set_digest": build_stable_digest({"outcomes": 1}),
        "paired_superiority_rows_digest": build_stable_digest({"rows": 1}),
        "randomization_paired_superiority_summary_digest": build_stable_digest(
            {"summary": 1}
        ),
        "conclusion_decision": "measured_not_supported",
        "supports_paper_claim": False,
    }
    report = {
        "paper_run_name": "probe_paper",
        "target_fpr": 0.1,
        "method_repeat_threshold_records_digest": build_stable_digest(
            {"thresholds": 1}
        ),
        "randomization_paired_superiority_report_digest": build_stable_digest(
            {"report": 1}
        ),
    }
    return runner.RandomizationPairedSuperiorityResult(
        threshold_records=({"threshold": 1},),
        paired_outcomes=({"outcome": 1},),
        superiority_rows=(statistic_row,),
        summary=summary,
        report=report,
    )


def test_runner_uses_each_repeat_specific_threshold_and_image_only_pairing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """36次配对调用必须逐 repeat 使用对应主方法与 baseline 阈值."""

    provenance = _provenance(tmp_path)
    rebuilt = _reconstruction()
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        runner,
        "rebuild_randomization_method_repeat_observation_sources",
        lambda source: rebuilt,
    )

    def fake_pair(proposed_rows, baseline_rows, **kwargs):
        calls.append(
            {
                "proposed_rows": proposed_rows,
                "baseline_rows": baseline_rows,
                **kwargs,
            }
        )
        return ({"call_index": len(calls)},)

    monkeypatch.setattr(runner, "build_paired_outcomes", fake_pair)

    def fake_statistics(outcomes, **kwargs):
        assert len(outcomes) == 9 * 4
        return (
            [{"baseline_id": baseline_id} for baseline_id in PRIMARY_BASELINE_IDS],
            {
                "paired_outcome_set_digest": build_stable_digest(outcomes),
                "paired_superiority_rows_digest": build_stable_digest(
                    list(PRIMARY_BASELINE_IDS)
                ),
                "randomization_paired_superiority_summary_digest": (
                    build_stable_digest({"summary": 1})
                ),
                "conclusion_decision": "measured_not_supported",
                "supports_paper_claim": False,
            },
        )

    monkeypatch.setattr(
        runner,
        "build_randomization_aggregate_paired_superiority_statistics",
        fake_statistics,
    )

    result = runner._rebuild_randomization_paired_superiority(provenance)

    assert len(calls) == 9 * 4
    assert len(result.threshold_records) == 45
    threshold_by_key = {
        (row["randomization_repeat_id"], row["method_id"]): row
        for row in rebuilt.threshold_records
    }
    for call in calls:
        baseline_id = str(call["baseline_id"])
        repeat_id = str(call["proposed_rows"][0]["source"])
        source = next(
            source
            for source in rebuilt.method_sources
            if source.observation_source_sha256 == repeat_id
        )
        assert call["require_image_only_evidence"] is True
        assert call["proposed_method_threshold_digest"] == threshold_by_key[
            (source.randomization_repeat_id, "slm_wm")
        ]["threshold_digest"]
        assert call["baseline_method_threshold_digest"] == threshold_by_key[
            (source.randomization_repeat_id, baseline_id)
        ]["threshold_digest"]


def test_public_runner_rejects_wrong_or_dirty_git_before_reconstruction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """正式入口在读取45来源前先锁定同一 clean Git commit."""

    provenance = _provenance(tmp_path)
    reached = False

    def forbidden_rebuild(source):
        nonlocal reached
        reached = True
        return _reconstruction()

    monkeypatch.setattr(runner, "resolve_code_version", lambda root: "b" * 40)
    monkeypatch.setattr(
        runner,
        "rebuild_randomization_method_repeat_observation_sources",
        forbidden_rebuild,
    )

    with pytest.raises(
        runner.RandomizationPairedSuperiorityRunnerError,
        match="相同的 clean Git 提交",
    ):
        runner.rebuild_randomization_paired_superiority(
            provenance,
            root=tmp_path,
        )
    assert reached is False


def test_public_runner_only_accepts_validated_provenance(tmp_path: Path) -> None:
    """路径或字典不得绕过聚合 provenance validator 进入统计入口."""

    with pytest.raises(TypeError, match="只接受 RandomizationAggregateProvenance"):
        runner.rebuild_randomization_paired_superiority(
            {"package_path": str(tmp_path / "aggregate.zip")},
            root=tmp_path,
        )


def test_writer_publishes_complete_directory_transactionally(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """正式目录只能在全部数据、报告和 manifest 写完后一次发布."""

    provenance = _provenance(tmp_path)
    monkeypatch.setattr(
        runner,
        "rebuild_randomization_paired_superiority",
        lambda source, root=".": _result(),
    )

    manifest_path = runner.write_randomization_paired_superiority_outputs(
        provenance,
        root=tmp_path,
    )

    output_dir = manifest_path.parent
    assert manifest_path.is_file()
    assert {path.name for path in output_dir.iterdir()} == {
        "method_repeat_threshold_records.jsonl",
        "paired_outcomes.jsonl",
        "paired_superiority_table.csv",
        "paired_superiority_summary.json",
        "randomization_paired_superiority_report.json",
        "manifest.local.json",
    }
    assert not list(output_dir.parent.glob(f".{output_dir.name}_publish_*"))


def test_writer_rejects_existing_directory_and_cleans_mid_write_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """不得覆盖旧结果, 写中失败也不得留下半成品或临时目录."""

    provenance = _provenance(tmp_path)
    monkeypatch.setattr(
        runner,
        "rebuild_randomization_paired_superiority",
        lambda source, root=".": _result(),
    )
    destination = (
        tmp_path
        / "outputs"
        / "randomization_paired_superiority"
        / "probe_paper"
    )
    destination.mkdir(parents=True)
    with pytest.raises(
        runner.RandomizationPairedSuperiorityRunnerError,
        match="不得覆盖",
    ):
        runner.write_randomization_paired_superiority_outputs(
            provenance,
            root=tmp_path,
        )
    destination.rmdir()

    monkeypatch.setattr(
        runner,
        "_write_jsonl",
        lambda path, rows: (_ for _ in ()).throw(RuntimeError("mid-write")),
    )
    with pytest.raises(RuntimeError, match="mid-write"):
        runner.write_randomization_paired_superiority_outputs(
            provenance,
            root=tmp_path,
        )
    assert not destination.exists()
    assert not list(destination.parent.glob(".probe_paper_publish_*"))
