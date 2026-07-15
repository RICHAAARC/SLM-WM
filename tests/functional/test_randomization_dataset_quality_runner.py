"""验证精确9重复质量 runner 的来源连接与事务发布."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.protocol.dataset_quality import (
    FORMAL_DATASET_QUALITY_ATTACK_NAME,
    FORMAL_DATASET_QUALITY_IMAGE_PAIR_ROLE,
    FORMAL_FEATURE_BACKEND,
)
from experiments.artifacts import dataset_level_quality_outputs as quality_writer
from experiments.protocol.formal_randomization import (
    formal_randomization_repeat_ids,
)
from main.core.digest import build_stable_digest
from paper_experiments.analysis.randomization_dataset_quality import (
    RandomizationDatasetQualityStatistics,
)
from paper_experiments.analysis.paper_quality_decisions import (
    build_quality_preservation_decisions,
)
from paper_experiments.runners.randomization_aggregate_provenance import (
    RandomizationAggregateProvenance,
)
from paper_experiments.runners import randomization_dataset_quality as runner
from tests.helpers.scientific_unit_provenance import (
    build_test_scientific_unit_provenance,
)


pytestmark = pytest.mark.quick

CODE_VERSION = "a" * 40
AGGREGATE_SHA = build_stable_digest({"aggregate_package": 1})
AGGREGATE_DIGEST = build_stable_digest({"aggregate": 1})


def _provenance(tmp_path: Path) -> RandomizationAggregateProvenance:
    """构造保持生产 validator 关键字段关系的 provenance."""

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


def _source(repeat_id: str, role: str) -> SimpleNamespace:
    """构造 writer 报告所需的最小聚合成员来源."""

    return SimpleNamespace(
        randomization_repeat_id=repeat_id,
        record_role=role,
        record_member=f"records/{repeat_id}/{role}.jsonl",
        record_sha256=build_stable_digest(
            {"repeat_id": repeat_id, "role": role}
        ),
        leaf_package_sha256=build_stable_digest(
            {"leaf": repeat_id, "role": role}
        ),
        randomization_repeat_component_sha256=build_stable_digest(
            {"component": repeat_id}
        ),
    )


def _pair_records() -> tuple[dict[str, tuple[SimpleNamespace, ...]], tuple[str, ...]]:
    """构造每个 repeat 精确覆盖70个 Prompt 的图像与特征对."""

    prompt_ids = tuple(f"probe_prompt_{index:03d}" for index in range(70))
    by_repeat: dict[str, tuple[SimpleNamespace, ...]] = {}
    for repeat_index, repeat_id in enumerate(
        formal_randomization_repeat_ids()
    ):
        image_source = _source(repeat_id, "quality_image_record")
        feature_source = _source(repeat_id, "quality_feature_record")
        pairs: list[SimpleNamespace] = []
        for prompt_index, prompt_id in enumerate(prompt_ids):
            source_digest = build_stable_digest(
                {"source": repeat_id, "prompt_id": prompt_id}
            )
            comparison_digest = build_stable_digest(
                {"comparison": repeat_id, "prompt_id": prompt_id}
            )
            image_payload = {
                "run_id": f"{repeat_id}:{prompt_id}",
                "prompt_id": prompt_id,
                "attack_name": FORMAL_DATASET_QUALITY_ATTACK_NAME,
                "image_pair_index": prompt_index,
                "image_pair_role": FORMAL_DATASET_QUALITY_IMAGE_PAIR_ROLE,
                "source_image_path": (
                    f"images/{repeat_id}/{prompt_id}_source.png"
                ),
                "source_image_digest": source_digest,
                "comparison_image_path": (
                    f"images/{repeat_id}/{prompt_id}_comparison.png"
                ),
                "comparison_image_digest": comparison_digest,
                "feature_backend": FORMAL_FEATURE_BACKEND,
                "supports_paper_claim": False,
            }
            record_digest = build_stable_digest(image_payload)
            record_id = f"dataset_quality_record_{record_digest[:16]}"
            image_record = {
                "dataset_quality_record_id": record_id,
                "dataset_quality_record_digest": record_digest,
                **image_payload,
            }
            base_vector = [
                prompt_index / 70.0,
                repeat_index / 9.0,
                0.25,
            ]
            feature_common = {
                "dataset_quality_record_id": record_id,
                "feature_backend": FORMAL_FEATURE_BACKEND,
                "feature_extractor_id": "formal_test_extractor",
                "feature_dimension": 3,
                "supports_paper_claim": False,
            }
            pairs.append(
                SimpleNamespace(
                    randomization_repeat_id=repeat_id,
                    dataset_quality_record_id=record_id,
                    image_record_source=image_source,
                    feature_record_source=feature_source,
                    image_record=image_record,
                    source_feature_record={
                        **feature_common,
                        "dataset_quality_image_role": "source",
                        "image_digest": source_digest,
                        "feature_vector": base_vector,
                    },
                    comparison_feature_record={
                        **feature_common,
                        "dataset_quality_image_role": "comparison",
                        "image_digest": comparison_digest,
                        "feature_vector": [
                            value + 0.01 for value in base_vector
                        ],
                    },
                )
            )
        by_repeat[repeat_id] = tuple(pairs)
    return by_repeat, prompt_ids


class _Workspace:
    """提供 runner 所需的只读质量对迭代接口."""

    def __init__(self, pairs_by_repeat: dict[str, tuple[SimpleNamespace, ...]]):
        self._pairs_by_repeat = pairs_by_repeat

    def __enter__(self) -> "_Workspace":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def iter_quality_feature_pairs(
        self,
        repeat_id: str,
    ) -> tuple[SimpleNamespace, ...]:
        return self._pairs_by_repeat[repeat_id]

    def find_source(
        self,
        *,
        randomization_repeat_id: str,
        package_family: str,
        record_role: str,
    ) -> SimpleNamespace:
        """返回新质量原始成员的最小来源身份."""

        assert package_family == "dataset_level_quality"
        return _source(randomization_repeat_id, record_role)

    def iter_records(
        self,
        source: SimpleNamespace,
    ) -> tuple[dict[str, object], ...]:
        """按 runner 的精确数量契约提供不进入公式层的测试记录."""

        record_counts = {
            "attack_quality_pair_record": 0,
            "attack_quality_inception_feature_record": 0,
            "paired_quality_clip_feature_record": 2 * 70,
            "paired_quality_independent_semantic_feature_record": 2 * 70,
            "paired_quality_metric_record": 70,
        }
        return tuple({} for _ in range(record_counts[source.record_role]))


def _statistics(
    memberships: tuple[dict[str, object], ...],
) -> RandomizationDatasetQualityStatistics:
    """构造 runner 和 writer 测试共用的最小已测量统计."""

    metric_rows = tuple(
        {
            "quality_metric_name": metric_name,
            "quality_metric_value": metric_value,
            "metric_status": "measured",
            "paper_metric_name": metric_name,
            "feature_backend": FORMAL_FEATURE_BACKEND,
            "source_image_count": 9 * 70,
            "comparison_image_count": 9 * 70,
            "sample_pair_count": 9 * 70,
            "supports_paper_claim": False,
        }
        for metric_name, metric_value in (
            ("fid", 1.0),
            ("kid_mean", 0.01),
            ("kid_std", 0.001),
        )
    )
    prompt_distribution_records = (
        {
            "prompt_id": "probe_prompt_000",
            "quality_metric_name": "prompt_conditional_kid",
            "quality_metric_value": 0.0,
        },
    )
    quality_decisions = build_quality_preservation_decisions(
        distributional_inference={
            "confidence_interval_low": 0.0,
            "confidence_interval_high": 0.0,
        },
        evidence_artifact_id="randomization_dataset_quality_manifest",
    )
    quality_decision = quality_decisions[
        "quality_preservation_claim_decision"
    ]
    summary = {
        "aggregate_quality_pair_count": 9 * 70,
        "quality_feature_membership_digest": build_stable_digest(memberships),
        "quality_feature_records_digest": build_stable_digest(
            {"features": 9 * 70 * 2}
        ),
        "randomization_dataset_quality_metric_protocol_digest": (
            build_stable_digest({"metric_protocol": 1})
        ),
        "fid_kid_metric_rows_digest": build_stable_digest(metric_rows),
        "prompt_distribution_records_digest": build_stable_digest(
            prompt_distribution_records
        ),
        "attack_prompt_distribution_records_digest": build_stable_digest(
            ()
        ),
        "paired_quality_metric_records_digest": build_stable_digest(
            {"paired_quality_metrics": 9 * 70}
        ),
        "attack_quality_membership_records_digest": build_stable_digest(()),
        "attack_quality_feature_records_digest": build_stable_digest(()),
        "paired_quality_clip_feature_records_digest": build_stable_digest(
            {"clip_features": 2 * 9 * 70}
        ),
        "paired_quality_independent_semantic_feature_records_digest": (
            build_stable_digest({"independent_semantic_features": 2 * 9 * 70})
        ),
        "randomization_dataset_quality_summary_digest": build_stable_digest(
            {"summary": 1}
        ),
        "randomization_dataset_quality_statistics_ready": True,
        **quality_decisions,
        "conclusion_decision": quality_decision["decision"],
        "supports_paper_claim": quality_decision["scientific_support"] is True,
    }
    return RandomizationDatasetQualityStatistics(
        membership_records=memberships,
        prompt_distribution_records=prompt_distribution_records,
        attack_prompt_distribution_records=(),
        metric_rows=metric_rows,
        summary=summary,
    )


def test_runner_joins_each_repeat_to_same_prompt_contract_and_raw_features(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """runner 必须收集9乘70个成员与两倍原始特征后再调用公式层."""

    provenance = _provenance(tmp_path)
    pairs_by_repeat, prompt_ids = _pair_records()
    workspace = _Workspace(pairs_by_repeat)
    monkeypatch.setattr(
        runner,
        "open_randomization_aggregate_record_workspace",
        lambda source: workspace,
    )
    monkeypatch.setattr(
        runner,
        "rebuild_randomization_prompt_source_contract",
        lambda workspace_value, source, paper_run_name: {
            "prompt_rows": tuple(
                {"prompt_id": prompt_id} for prompt_id in prompt_ids
            ),
            "report": {
                "prompt_source_contract_digest": build_stable_digest(
                    {"prompt_contract": 1}
                ),
                "prompt_rows_digest": build_stable_digest(prompt_ids),
            },
        },
    )
    monkeypatch.setattr(
        runner,
        "_validated_scientific_provenance",
        lambda feature_records, expected_code_version, **kwargs: {
            "scientific_unit_provenance_ready": True,
        },
    )
    captured: dict[str, object] = {}

    def capture_statistics(feature_records, membership_records, **kwargs):
        captured["feature_records"] = tuple(feature_records)
        captured["membership_records"] = tuple(membership_records)
        captured["kwargs"] = kwargs
        return _statistics(tuple(membership_records))

    monkeypatch.setattr(
        runner,
        "rebuild_randomization_dataset_quality_statistics",
        capture_statistics,
    )

    result = runner._rebuild_randomization_dataset_quality(provenance)

    assert len(captured["membership_records"]) == 9 * 70
    assert len(captured["feature_records"]) == 2 * 9 * 70
    assert captured["kwargs"]["expected_prompt_ids"] == prompt_ids
    assert len(result.report["repeat_source_records"]) == 9
    assert result.report["randomization_dataset_quality_statistics_ready"] is True
    assert result.report["supports_paper_claim"] is False
    for repeat_id in formal_randomization_repeat_ids():
        repeat_prompt_ids = {
            record["prompt_id"]
            for record in captured["membership_records"]
            if record["randomization_repeat_id"] == repeat_id
        }
        assert repeat_prompt_ids == set(prompt_ids)


def test_runner_requires_real_inception_cuda_provenance_and_matching_commit() -> None:
    """正式特征来源必须通过 CUDA 完成单元复算并绑定聚合 Git 提交."""

    record_id = "dataset_quality_record_0000000000000000"
    item_identity = [
        {
            "dataset_quality_record_id": record_id,
            "dataset_quality_image_role": role,
            "image_path": f"images/{role}.png",
            "image_digest": build_stable_digest({"role": role}),
        }
        for role in ("source", "comparison")
    ]
    unit_digest = build_stable_digest(
        [
            (
                identity["dataset_quality_record_id"],
                identity["dataset_quality_image_role"],
            )
            for identity in item_identity
        ]
    )
    provenance = build_test_scientific_unit_provenance(
        f"feature_batch_{unit_digest[:16]}",
        quality_writer._inception_batch_config_digest(item_identity),
    )
    feature_records = tuple(
        {
            **identity,
            "scientific_unit_provenance": provenance,
        }
        for identity in item_identity
    )

    summary = runner._validated_scientific_provenance(
        feature_records,
        expected_code_version="b" * 40,
    )

    assert summary["scientific_unit_provenance_ready"] is True
    assert summary["scientific_dependency_profile_ids"] == [
        "sd35_method_runtime_gpu"
    ]
    assert summary["scientific_cuda_device_names"] == ["NVIDIA T4"]
    with pytest.raises(
        runner.RandomizationDatasetQualityRunnerError,
        match="Git 提交或真实 CUDA 身份",
    ):
        runner._validated_scientific_provenance(
            feature_records,
            expected_code_version="c" * 40,
        )


def _result() -> runner.RandomizationDatasetQualityResult:
    """构造事务 writer 测试所需的最小完整结果."""

    statistics = _statistics(
        (
            {
                "randomization_repeat_id": "seed_00_key_00",
                "prompt_id": "probe_prompt_000",
                "dataset_quality_record_id": "dataset_quality_record_0001",
            },
        )
    )
    report = {
        "paper_run_name": "probe_paper",
        "target_fpr": 0.1,
        "prompt_source_contract_digest": build_stable_digest(
            {"prompt_contract": 1}
        ),
        "randomization_dataset_quality_report_digest": build_stable_digest(
            {"report": 1}
        ),
    }
    return runner.RandomizationDatasetQualityResult(
        membership_records=statistics.membership_records,
        prompt_distribution_records=statistics.prompt_distribution_records,
        attack_prompt_distribution_records=(
            statistics.attack_prompt_distribution_records
        ),
        metric_rows=statistics.metric_rows,
        summary=statistics.summary,
        report=report,
    )


def test_writer_publishes_minimal_quality_directory_transactionally(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """正式目录必须在指标, 成员, 报告和 manifest 完整后发布."""

    provenance = _provenance(tmp_path)
    monkeypatch.setattr(
        runner,
        "rebuild_randomization_dataset_quality",
        lambda source, root=".": _result(),
    )

    manifest_path = runner.write_randomization_dataset_quality_outputs(
        provenance,
        root=tmp_path,
    )

    output_dir = manifest_path.parent
    assert manifest_path.is_file()
    assert {path.name for path in output_dir.iterdir()} == {
        "fid_kid_metrics.csv",
        "quality_feature_membership.jsonl",
        "prompt_distributional_quality_records.jsonl",
        "attack_prompt_distributional_quality_records.jsonl",
        "randomization_dataset_quality_summary.json",
        "randomization_dataset_quality_report.json",
        "manifest.local.json",
    }
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["metadata"][
        "randomization_dataset_quality_statistics_ready"
    ] is True
    assert manifest["metadata"]["supports_paper_claim"] is False
    assert not list(output_dir.parent.glob(f".{output_dir.name}_publish_*"))


def test_public_runner_rejects_code_mismatch_before_opening_workspace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """聚合 Git 提交不一致时不得读取任何原始成员."""

    provenance = _provenance(tmp_path)
    reached = False

    def forbidden_open(source):
        nonlocal reached
        reached = True
        raise AssertionError("不应打开工作区")

    monkeypatch.setattr(runner, "resolve_code_version", lambda root: "b" * 40)
    monkeypatch.setattr(
        runner,
        "open_randomization_aggregate_record_workspace",
        forbidden_open,
    )

    with pytest.raises(
        runner.RandomizationDatasetQualityRunnerError,
        match="相同的 clean Git 提交",
    ):
        runner.rebuild_randomization_dataset_quality(
            provenance,
            root=tmp_path,
        )
    assert reached is False


def test_writer_rejects_existing_output_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """现有正式目录不得被覆盖或与新聚合来源混选."""

    provenance = _provenance(tmp_path)
    monkeypatch.setattr(
        runner,
        "rebuild_randomization_dataset_quality",
        lambda source, root=".": _result(),
    )
    destination = (
        tmp_path
        / "outputs"
        / "randomization_dataset_quality"
        / "probe_paper"
    )
    destination.mkdir(parents=True)

    with pytest.raises(
        runner.RandomizationDatasetQualityRunnerError,
        match="不得覆盖",
    ):
        runner.write_randomization_dataset_quality_outputs(
            provenance,
            root=tmp_path,
        )
