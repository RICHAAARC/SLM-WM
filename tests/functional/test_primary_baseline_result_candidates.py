"""主表 baseline 正式候选记录来源边界测试。"""

from __future__ import annotations

import json
from pathlib import Path
from zipfile import ZipFile

import pytest

from experiments.protocol.attacks import attack_config_digest, resolve_formal_attack_config
from paper_experiments.baselines import build_t2smark_formal_candidate_records
from paper_experiments.baselines.method_faithful_observation_collection import (
    FORMAL_MODEL_ID,
    FORMAL_MODEL_REVISION,
    MethodFaithfulObservationSource,
    file_sha256,
)
from scripts.write_primary_baseline_result_candidates import (
    _measured_baseline_readiness,
    build_method_candidate_rows,
    evidence_path_for_source,
    load_t2smark_candidate_rows,
    load_canonical_prompt_protocol,
    normalize_t2smark_candidate_rows,
    t2smark_candidate_records_entry,
)
from experiments.protocol.fixed_fpr_observation_audit import (
    conformal_threshold_from_clean_negative_scores,
)


pytestmark = pytest.mark.quick


def test_t2smark_package_entry_is_scoped_by_paper_run() -> None:
    """T2SMark 包直读路径必须包含当前论文运行层级, 不得读取全局旧布局。"""

    assert t2smark_candidate_records_entry("full_paper") == (
        "outputs/t2smark_formal_reproduction/full_paper/"
        "t2smark_formal_import_candidate_records.jsonl"
    )


def probe_observations(*, baseline_id: str = "tree_ring") -> list[dict[str, object]]:
    """构造具有完整 probe split 和真实 conformal 阈值的轻量 observation。"""

    root_path = Path(__file__).resolve().parents[2]
    canonical_prompts, _ = load_canonical_prompt_protocol(root_path)
    calibration_scores = tuple(index / 100.0 for index in range(33))
    threshold = conformal_threshold_from_clean_negative_scores(calibration_scores, 0.1)
    rows: list[dict[str, object]] = []
    calibration_index = 0
    for prompt_id, identity in canonical_prompts.items():
        split = identity["split"]
        clean_score = calibration_scores[calibration_index] if split == "calibration" else 0.01
        if split == "calibration":
            calibration_index += 1
        common = {
            "baseline_id": baseline_id,
            "split": split,
            "prompt_id": prompt_id,
            "prompt_text": identity["prompt_text"],
            "prompt_digest": identity["prompt_digest"],
            "threshold": threshold,
            "threshold_source": "calibration_clean_negative_conformal",
            "attack_family": "clean",
            "attack_name": "clean_none",
        }
        rows.append(
            {
                **common,
                "event_id": f"{prompt_id}_negative",
                "sample_role": "clean_negative",
                "score": clean_score,
                "detection_decision": clean_score >= threshold,
            }
        )
        rows.append(
            {
                **common,
                "event_id": f"{prompt_id}_positive",
                "sample_role": "positive_source",
                "score": 0.9,
                "detection_decision": 0.9 >= threshold,
            }
        )
    return rows


def test_t2smark_package_must_contain_canonical_entry(tmp_path: Path) -> None:
    """显式 T2SMark 结果包缺少规范条目时必须停止导入。"""

    local_path = tmp_path / "outputs" / "t2smark_records.jsonl"
    local_path.parent.mkdir(parents=True)
    local_path.write_text('{"baseline_id":"t2smark"}\n', encoding="utf-8")
    package_path = tmp_path / "t2smark_results.zip"
    with ZipFile(package_path, "w") as archive:
        archive.writestr("outputs/unrelated.jsonl", "{}\n")

    with pytest.raises(ValueError, match="T2SMark 正式候选记录为空"):
        load_t2smark_candidate_rows(
            candidate_records_path=local_path,
            package_path=package_path,
            package_entry_name=t2smark_candidate_records_entry("probe_paper"),
        )


def test_t2smark_package_supplies_canonical_candidate_records(tmp_path: Path) -> None:
    """T2SMark 专用结果包的规范条目应成为候选记录来源。"""

    t2smark_package = tmp_path / "t2smark_results.zip"
    entry_name = t2smark_candidate_records_entry("probe_paper")
    with ZipFile(t2smark_package, "w") as archive:
        archive.writestr(
            entry_name,
            json.dumps({"baseline_id": "t2smark", "attack_name": "jpeg_compression"}) + "\n",
        )

    t2smark_rows = load_t2smark_candidate_rows(
        candidate_records_path=tmp_path / "missing_t2smark.jsonl",
        package_path=t2smark_package,
        package_entry_name=entry_name,
    )

    assert t2smark_rows == [{"baseline_id": "t2smark", "attack_name": "jpeg_compression"}]


def test_evidence_source_must_exist(tmp_path: Path) -> None:
    """不存在的证据文件不得形成 baseline 候选记录来源。"""

    with pytest.raises(FileNotFoundError, match="baseline 证据来源不存在"):
        evidence_path_for_source(tmp_path / "missing.zip", tmp_path)


def test_t2smark_normalization_does_not_promote_protocol_readiness(tmp_path: Path) -> None:
    """来源归一化不得根据计数自行提升 Prompt 协议门禁。"""

    source_path = tmp_path / "t2smark_results.jsonl"
    source_path.write_text('{"baseline_id":"t2smark"}\n', encoding="utf-8")
    _, prompt_protocol_digest = load_canonical_prompt_protocol(Path(__file__).resolve().parents[2])
    rows = normalize_t2smark_candidate_rows(
        rows=[
            {
                "baseline_id": "t2smark",
                "positive_count": 340,
                "negative_count": 340,
                "prompt_protocol_digest": prompt_protocol_digest,
                "paper_run_prompt_protocol_ready": False,
            }
        ],
        source_path=source_path,
        root_path=tmp_path,
    )

    assert rows[0]["paper_run_prompt_protocol_ready"] is False
    assert rows[0]["formal_evidence_paths_ready"] is True


def test_t2smark_normalization_rejects_noncanonical_prompt_digest(tmp_path: Path) -> None:
    """T2SMark 候选记录不得仅凭非空摘要冒充当前受治理 Prompt 协议。"""

    source_path = tmp_path / "t2smark_results.jsonl"
    source_path.write_text('{"baseline_id":"t2smark"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="未绑定当前受治理 Prompt 协议摘要"):
        normalize_t2smark_candidate_rows(
            rows=[{"baseline_id": "t2smark", "prompt_protocol_digest": "wrong"}],
            source_path=source_path,
            root_path=tmp_path,
        )


def test_prompt_readiness_rejects_same_size_noncanonical_prompt_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """数量与 split 完整但文本不属于受治理 Prompt 集时必须阻断。"""

    monkeypatch.setenv("SLM_WM_PAPER_RUN_NAME", "probe_paper")
    rows = probe_observations()
    rows[0]["prompt_text"] = "an ungoverned replacement prompt"

    readiness = _measured_baseline_readiness(
        rows,
        Path(__file__).resolve().parents[2],
        0.1,
    )

    assert readiness["prompt_protocol_ready"] is False


def test_fixed_fpr_readiness_rejects_mislabeled_arbitrary_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """来源标签正确但数值并非 calibration 重算结果的阈值不得通过。"""

    monkeypatch.setenv("SLM_WM_PAPER_RUN_NAME", "probe_paper")
    rows = probe_observations()
    for row in rows:
        row["threshold"] = 100.0
        row["detection_decision"] = False

    readiness = _measured_baseline_readiness(rows, Path(__file__).resolve().parents[2], 0.1)

    assert readiness["prompt_protocol_ready"] is True
    assert readiness["fixed_fpr_ready"] is False


def test_fixed_fpr_readiness_rejects_decision_inconsistent_with_score(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """任一 observation 的 decision 与冻结阈值不一致时必须阻断。"""

    monkeypatch.setenv("SLM_WM_PAPER_RUN_NAME", "probe_paper")
    rows = probe_observations()
    test_positive = next(
        row
        for row in rows
        if row["split"] == "test" and row["sample_role"] == "positive_source"
    )
    test_positive["detection_decision"] = not bool(test_positive["detection_decision"])

    readiness = _measured_baseline_readiness(rows, Path(__file__).resolve().parents[2], 0.1)

    assert readiness["prompt_protocol_ready"] is True
    assert readiness["fixed_fpr_ready"] is False


@pytest.mark.parametrize("baseline_id", ("tree_ring", "gaussian_shading", "shallow_diffuse"))
def test_method_candidate_statistics_use_test_split_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    baseline_id: str,
) -> None:
    """三类 method-faithful baseline 的正式指标只能统计 test observation。"""

    monkeypatch.setenv("SLM_WM_PAPER_RUN_NAME", "probe_paper")
    attack_config = resolve_formal_attack_config(
        attack_family="standard_distortion",
        attack_name="jpeg_compression",
    )
    attack_identity = {
        "attack_id": attack_config.attack_id,
        "resource_profile": attack_config.resource_profile,
        "attack_config_digest": attack_config_digest(attack_config),
    }
    observations: list[dict[str, object]] = []
    for split, quality_score in (
        ("dev", 0.1),
        ("calibration", 0.3),
        ("test", 0.8),
    ):
        common = {
            "baseline_id": baseline_id,
            "split": split,
            "prompt_id": f"{split}_prompt",
            "prompt_text": "a ceramic fox",
            "threshold": 0.5,
            "threshold_source": "calibration_clean_negative_conformal",
            "quality_score": quality_score,
        }
        observations.extend(
            (
                {
                    **common,
                    "attack_family": "clean",
                    "attack_name": "clean_none",
                    "sample_role": "clean_negative",
                    "score": 0.1,
                    "detection_decision": split != "test",
                },
                {
                    **common,
                    "attack_family": "clean",
                    "attack_name": "clean_none",
                    "sample_role": "positive_source",
                    "score": 0.9,
                    "detection_decision": split == "test",
                },
                {
                    **common,
                    **attack_identity,
                    "attack_family": "standard_distortion",
                    "attack_name": "jpeg_compression",
                    "sample_role": "attacked_negative",
                    "score": 0.1,
                    "detection_decision": split != "test",
                },
                {
                    **common,
                    **attack_identity,
                    "attack_family": "standard_distortion",
                    "attack_name": "jpeg_compression",
                    "sample_role": "attacked_positive",
                    "score": 0.9,
                    "detection_decision": split == "test",
                },
            )
        )
    source_path = tmp_path / "baseline_observations.json"
    source_path.write_text(json.dumps(observations, ensure_ascii=False), encoding="utf-8")
    transfer_manifest_path = tmp_path / "baseline_transfer_manifest.json"
    transfer_manifest_path.write_text(
        json.dumps({"baseline_id": baseline_id}, ensure_ascii=False),
        encoding="utf-8",
    )
    source = MethodFaithfulObservationSource(
        baseline_id=baseline_id,
        observations_path=source_path,
        transfer_manifest_path=transfer_manifest_path,
        observations_sha256=file_sha256(source_path),
        prompt_plan_path=source_path,
        adapter_manifest_path=transfer_manifest_path,
        execution_manifest_path=transfer_manifest_path,
        model_id=FORMAL_MODEL_ID,
        model_revision=FORMAL_MODEL_REVISION,
        rows=tuple(dict(row) for row in observations),
        transfer_manifest={"baseline_id": baseline_id},
    )

    records = build_method_candidate_rows(
        sources=(source,),
        root_path=Path(__file__).resolve().parents[2],
        target_fpr=0.1,
    )

    assert len(records) == 1
    record = records[0]
    assert record["positive_count"] == 1
    assert record["negative_count"] == 1
    assert record["attacked_negative_count"] == 1
    assert record["true_positive_rate"] == 1.0
    assert record["clean_false_positive_rate"] == 0.0
    assert record["attacked_false_positive_rate"] == 0.0
    assert record["quality_score_mean"] == pytest.approx(0.8)
    assert "score_retention_mean" not in record


def test_t2smark_candidate_statistics_use_test_split_only() -> None:
    """T2SMark 正式候选必须与三个 method-faithful baseline 共享 test-only 统计边界。"""

    attack_config = resolve_formal_attack_config(
        attack_family="standard_distortion",
        attack_name="jpeg_compression",
    )
    attack_identity = {
        "attack_id": attack_config.attack_id,
        "resource_profile": attack_config.resource_profile,
        "attack_config_digest": attack_config_digest(attack_config),
    }
    observations: list[dict[str, object]] = []
    for split, quality_score in (
        ("dev", 0.1),
        ("calibration", 0.3),
        ("test", 0.8),
    ):
        common = {
            "baseline_id": "t2smark",
            "split": split,
            "prompt_id": f"{split}_prompt",
            "quality_score": quality_score,
        }
        observations.extend(
            (
                {
                    **common,
                    "attack_family": "clean",
                    "attack_condition": "clean_none",
                    "sample_role": "clean_negative",
                    "detection_decision": split != "test",
                },
                {
                    **common,
                    **attack_identity,
                    "attack_family": "standard_distortion",
                    "attack_condition": "jpeg_compression",
                    "sample_role": "attacked_negative",
                    "detection_decision": split != "test",
                },
                {
                    **common,
                    **attack_identity,
                    "attack_family": "standard_distortion",
                    "attack_condition": "jpeg_compression",
                    "sample_role": "attacked_positive",
                    "detection_decision": split == "test",
                },
            )
        )

    records = build_t2smark_formal_candidate_records(
        observation_rows=observations,
        target_fpr=0.1,
        baseline_result_source="outputs/t2smark/results.json",
        baseline_result_source_digest="digest",
        evidence_paths=["outputs/t2smark/results.json"],
        prompt_protocol_digest="prompt_digest",
        paper_run_prompt_protocol_ready=False,
        fixed_fpr_baseline_calibration_ready=False,
        attack_matrix_baseline_detection_ready=False,
    )

    assert len(records) == 1
    record = records[0]
    assert record["positive_count"] == 1
    assert record["negative_count"] == 1
    assert record["attacked_negative_count"] == 1
    assert record["true_positive_rate"] == 1.0
    assert record["clean_false_positive_rate"] == 0.0
    assert record["attacked_false_positive_rate"] == 0.0
    assert record["quality_score_mean"] == pytest.approx(0.8)
    assert "score_retention_mean" not in record
