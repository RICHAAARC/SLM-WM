"""主表 external baseline 正式结果导入协议的轻量功能测试。"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from experiments.protocol.fixed_fpr_observation_audit import (
    FORMAL_THRESHOLD_SOURCE,
    conformal_threshold_from_clean_negative_scores,
)
from experiments.protocol.attacks import attack_config_digest, resolve_formal_attack_config
from experiments.protocol.image_only_evidence import (
    partition_calibration_prompt_ids,
)
from experiments.protocol.paper_fixed_fpr import (
    FULL_PAPER_FIXED_FPR,
    PILOT_PAPER_FIXED_FPR,
)
from main.core.digest import build_stable_digest
from paper_experiments.baselines import (
    build_primary_baseline_formal_evidence_collection_rows,
    build_primary_baseline_formal_evidence_collection_summary,
    build_primary_baseline_formal_import_schema,
    build_primary_baseline_method_threshold_digest_map,
    build_primary_baseline_formal_template_coverage_rows,
    build_primary_baseline_formal_template_coverage_summary,
    build_t2smark_formal_candidate_records,
    build_tree_ring_method_faithful_candidate_records,
    validate_primary_baseline_formal_import_rows,
)
from scripts.write_primary_baseline_formal_import_protocol import write_primary_baseline_formal_import_protocol_outputs
from tests.helpers.formal_prompt_source import copy_governed_prompt_file


PAPER_RUN_PARAMETERS = {
    "probe_paper": {"calibration": 33, "test": 34, "target_fpr": 0.1},
    "pilot_paper": {"calibration": 330, "test": 340, "target_fpr": 0.01},
    "full_paper": {"calibration": 3300, "test": 3400, "target_fpr": 0.001},
}


@pytest.fixture(autouse=True)
def _select_pilot_paper(monkeypatch: pytest.MonkeyPatch) -> None:
    """本模块未显式切换层级的导入夹具固定使用 pilot_paper."""

    monkeypatch.setenv("SLM_WM_PAPER_RUN_NAME", "pilot_paper")


def formal_attack_descriptor(
    attack_family: str,
    attack_name: str,
    resource_profile: str | None = None,
) -> dict[str, str]:
    """从唯一攻击注册表构造测试模板使用的完整正式身份."""

    config = resolve_formal_attack_config(
        attack_family=attack_family,
        attack_name=attack_name,
        resource_profile=resource_profile,
    )
    return {
        "attack_id": config.attack_id,
        "attack_family": config.attack_family,
        "attack_name": config.attack_name,
        "resource_profile": config.resource_profile,
        "attack_config_digest": attack_config_digest(config),
    }


def build_formal_tree_ring_observations(
    *,
    paper_run_name: str = "pilot_paper",
) -> list[dict[str, object]]:
    """构造共享冻结阈值且逐 Prompt 一一覆盖的 Tree-Ring observations。"""

    parameters = PAPER_RUN_PARAMETERS[paper_run_name]
    calibration_count = int(parameters["calibration"])
    test_count = int(parameters["test"])
    target_fpr = float(parameters["target_fpr"])
    calibration_scores = [
        index / calibration_count for index in range(calibration_count)
    ]
    calibration_prompt_ids = tuple(
        f"calibration_{index:05d}" for index in range(calibration_count)
    )
    _, threshold_freeze_prompt_ids, _ = partition_calibration_prompt_ids(
        calibration_prompt_ids
    )
    score_by_prompt_id = dict(
        zip(calibration_prompt_ids, calibration_scores, strict=True)
    )
    threshold = conformal_threshold_from_clean_negative_scores(
        (
            score_by_prompt_id[prompt_id]
            for prompt_id in threshold_freeze_prompt_ids
        ),
        target_fpr=target_fpr,
    )

    def observation(
        *,
        split: str,
        prompt_id: str,
        event_id: str,
        attack_family: str,
        attack_condition: str,
        sample_role: str,
        score: float,
        quality_score: float | None = None,
    ) -> dict[str, object]:
        row: dict[str, object] = {
            "baseline_id": "tree_ring",
            "split": split,
            "prompt_id": prompt_id,
            "event_id": event_id,
            "attack_family": attack_family,
            "attack_condition": attack_condition,
            "sample_role": sample_role,
            "score": score,
            "threshold": threshold,
            "threshold_source": FORMAL_THRESHOLD_SOURCE,
            "detection_decision": score >= threshold,
        }
        if quality_score is not None:
            row["quality_score"] = quality_score
            row["score_retention"] = quality_score
        if sample_role in {"attacked_negative", "attacked_positive"}:
            attack_config = resolve_formal_attack_config(
                attack_family=attack_family,
                attack_name=attack_condition,
            )
            row.update(
                {
                    "attack_id": attack_config.attack_id,
                    "resource_profile": attack_config.resource_profile,
                    "attack_config_digest": attack_config_digest(attack_config),
                }
            )
        return row

    rows = [
        *[
            observation(
                split="calibration",
                prompt_id=f"calibration_{index:05d}",
                event_id=f"calibration_clean_negative_{index:05d}",
                attack_family="clean",
                attack_condition="clean_none",
                sample_role="clean_negative",
                score=score,
            )
            for index, score in enumerate(calibration_scores)
        ],
        *[
            observation(
                split="test",
                prompt_id=f"test_{index:05d}",
                event_id=f"test_clean_negative_{index:05d}",
                attack_family="clean",
                attack_condition="clean_none",
                sample_role="clean_negative",
                score=threshold - 1.0,
            )
            for index in range(test_count)
        ],
        *[
            observation(
                split="test",
                prompt_id=f"test_{index:05d}",
                event_id=f"test_attacked_positive_{index:05d}",
                attack_family="standard_distortion",
                attack_condition="jpeg_compression",
                sample_role="attacked_positive",
                score=threshold + 1.0,
                quality_score=0.88,
            )
            for index in range(test_count)
        ],
        *[
            observation(
                split="test",
                prompt_id=f"test_{index:05d}",
                event_id=f"test_attacked_negative_{index:05d}",
                attack_family="standard_distortion",
                attack_condition="jpeg_compression",
                sample_role="attacked_negative",
                score=threshold - 1.0,
                quality_score=0.88,
            )
            for index in range(test_count)
        ],
    ]
    return rows


def write_formal_tree_ring_row(
    tmp_path: Path,
    *,
    paper_run_name: str = "pilot_paper",
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """写出受治理 observation evidence，并构造一条可验证的正式候选。"""

    observations = build_formal_tree_ring_observations(paper_run_name=paper_run_name)
    evidence_relative_path = "outputs/external_baseline_results/tree_ring_observations.json"
    evidence_path = tmp_path / evidence_relative_path
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.write_text(json.dumps(observations, ensure_ascii=False), encoding="utf-8")
    parameters = PAPER_RUN_PARAMETERS[paper_run_name]
    records = build_tree_ring_method_faithful_candidate_records(
        observation_rows=observations,
        target_fpr=float(parameters["target_fpr"]),
        baseline_result_source=evidence_relative_path,
        baseline_result_source_digest=hashlib.sha256(evidence_path.read_bytes()).hexdigest(),
        evidence_paths=[evidence_relative_path],
        prompt_protocol_digest="prompt_digest",
        paper_run_prompt_protocol_ready=True,
        fixed_fpr_baseline_calibration_ready=True,
        attack_matrix_baseline_detection_ready=True,
    )
    assert len(records) == 1
    return records[0], observations


@pytest.mark.quick
def test_formal_import_validator_accepts_governed_full_main_record(tmp_path: Path) -> None:
    """完整边界均满足时, validator 应接受主表正式导入记录。"""

    row, _ = write_formal_tree_ring_row(tmp_path)

    report = validate_primary_baseline_formal_import_rows([row], evidence_root=tmp_path, target_fpr=PILOT_PAPER_FIXED_FPR)

    assert report["overall_decision"] == "pass"
    assert report["accepted_formal_import_count"] == 1
    assert report["formal_import_validation_ready"] is True
    assert report["accepted_records"][0]["baseline_id"] == "tree_ring"


@pytest.mark.quick
def test_formal_import_validator_recomputes_reported_metrics_from_observations(
    tmp_path: Path,
) -> None:
    """候选记录不得用真实 observation 路径配合伪造的 TPR 或质量均值。"""

    row, _ = write_formal_tree_ring_row(tmp_path)
    row["true_positive_rate"] = 0.5
    row["quality_score_mean"] = 0.5

    report = validate_primary_baseline_formal_import_rows(
        [row],
        evidence_root=tmp_path,
        target_fpr=PILOT_PAPER_FIXED_FPR,
    )

    issue_fields = {
        issue["field_name"]
        for issue in report["issues"]
        if issue["reason"] == "formal_metric_must_match_governed_observations"
    }
    assert issue_fields == {"true_positive_rate", "quality_score_mean"}
    assert report["accepted_formal_import_count"] == 0


@pytest.mark.quick
def test_formal_import_validator_rejects_forged_result_source_digest(
    tmp_path: Path,
) -> None:
    """正式结果来源摘要必须等于 evidence 文件的真实 SHA-256。"""

    row, _ = write_formal_tree_ring_row(tmp_path)
    row["baseline_result_source_digest"] = "f" * 64

    report = validate_primary_baseline_formal_import_rows(
        [row],
        evidence_root=tmp_path,
        target_fpr=PILOT_PAPER_FIXED_FPR,
    )

    assert "baseline_result_source_sha256_mismatch" in {
        issue["reason"] for issue in report["issues"]
    }
    assert report["accepted_formal_import_count"] == 0


@pytest.mark.quick
def test_formal_import_validator_rejects_duplicate_formal_template_key(tmp_path: Path) -> None:
    """baseline 正式导入不得包含重复的 baseline × attack 模板键。"""

    row, _ = write_formal_tree_ring_row(tmp_path)

    report = validate_primary_baseline_formal_import_rows(
        [row, dict(row)],
        evidence_root=tmp_path,
        target_fpr=PILOT_PAPER_FIXED_FPR,
    )

    assert report["formal_import_validation_ready"] is False
    assert report["accepted_formal_import_count"] == 1
    assert {issue["reason"] for issue in report["issues"]} == {"duplicate_formal_template_key"}


@pytest.mark.quick
def test_formal_import_validator_requires_complete_scale_and_fixed_fpr_confidence(tmp_path: Path) -> None:
    """baseline 必须覆盖完整 test split 且 clean FPR 置信上界不超过目标。"""

    complete_row, _ = write_formal_tree_ring_row(tmp_path)
    incomplete_row = dict(complete_row)
    incomplete_row["positive_count"] = 341
    incomplete_row["attack_record_count"] = 1021
    high_upper_bound_row = dict(complete_row)
    high_upper_bound_row["false_positive_rate"] = 25 / 340
    high_upper_bound_row["clean_false_positive_rate"] = 25 / 340

    incomplete_report = validate_primary_baseline_formal_import_rows(
        [incomplete_row], evidence_root=tmp_path, target_fpr=PILOT_PAPER_FIXED_FPR
    )
    upper_bound_report = validate_primary_baseline_formal_import_rows(
        [high_upper_bound_row], evidence_root=tmp_path, target_fpr=PILOT_PAPER_FIXED_FPR
    )

    assert "complete_test_positive_count_required" in {
        issue["reason"] for issue in incomplete_report["issues"]
    }
    assert "complete_test_attack_record_count_required" in {
        issue["reason"] for issue in incomplete_report["issues"]
    }
    assert "clean_fpr_confidence_upper_bound_exceeds_target" in {
        issue["reason"] for issue in upper_bound_report["issues"]
    }


@pytest.mark.quick
def test_formal_import_validator_recomputes_threshold_from_bound_observations(tmp_path: Path) -> None:
    """长度合法但与 observation 不一致的阈值摘要和数值不得通过正式门禁。"""

    row, _ = write_formal_tree_ring_row(tmp_path)
    row["calibrated_detection_threshold"] = float(row["calibrated_detection_threshold"]) + 0.1
    row["threshold_digest"] = "f" * 64
    row["fixed_fpr_observation_evidence_digest"] = "e" * 64

    report = validate_primary_baseline_formal_import_rows([row], evidence_root=tmp_path, target_fpr=PILOT_PAPER_FIXED_FPR)
    reasons = {issue["reason"] for issue in report["issues"]}

    assert report["accepted_formal_import_count"] == 0
    assert "calibrated_threshold_must_match_governed_observations" in reasons
    assert "threshold_digest_must_match_governed_observations" in reasons
    assert "threshold_observation_evidence_digest_mismatch" in reasons


@pytest.mark.quick
def test_formal_import_validator_rejects_explicit_target_fpr_drift(
    tmp_path: Path,
) -> None:
    """candidate 明示的 target FPR 必须与验证入口冻结值一致."""

    row, _ = write_formal_tree_ring_row(tmp_path)
    row["target_fpr"] = 0.05

    report = validate_primary_baseline_formal_import_rows(
        [row],
        evidence_root=tmp_path,
        target_fpr=PILOT_PAPER_FIXED_FPR,
    )

    assert "frozen_target_fpr_required" in {
        issue["reason"] for issue in report["issues"]
    }


@pytest.mark.quick
def test_formal_import_validator_requires_exact_attack_prompt_coverage(tmp_path: Path) -> None:
    """非 clean 攻击不得替换样本角色，也不得重复或遗漏 test Prompt。"""

    row, observations = write_formal_tree_ring_row(tmp_path)
    attacked_positive_rows = [
        observation for observation in observations if observation["sample_role"] == "attacked_positive"
    ]
    attacked_negative_rows = [
        observation for observation in observations if observation["sample_role"] == "attacked_negative"
    ]
    attacked_positive_rows[0]["sample_role"] = "positive_source"
    attacked_positive_rows[1]["prompt_id"] = attacked_positive_rows[2]["prompt_id"]
    attacked_negative_rows[0]["prompt_id"] = attacked_negative_rows[1]["prompt_id"]
    evidence_path = tmp_path / str(row["fixed_fpr_observation_evidence_path"])
    evidence_path.write_text(json.dumps(observations, ensure_ascii=False), encoding="utf-8")
    row["fixed_fpr_observation_evidence_digest"] = build_stable_digest(observations)

    report = validate_primary_baseline_formal_import_rows([row], evidence_root=tmp_path, target_fpr=PILOT_PAPER_FIXED_FPR)
    reasons = {issue["reason"] for issue in report["issues"]}

    assert report["accepted_formal_import_count"] == 0
    assert "non_clean_attack_requires_attacked_positive_role" in reasons
    assert "complete_test_attacked_positive_observations_required" in reasons
    assert "exact_test_prompt_coverage_required_for_attacked_positive" in reasons
    assert "exact_test_prompt_coverage_required_for_attacked_negative" in reasons


@pytest.mark.quick
def test_formal_import_protocol_switches_prompt_schema_to_full_paper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """主表 baseline 正式导入 schema 和 validator 应跟随 full_paper 运行层级。"""

    copy_governed_prompt_file(tmp_path, "full_paper")
    monkeypatch.setenv("SLM_WM_PAPER_RUN_NAME", "full_paper")
    row, _ = write_formal_tree_ring_row(tmp_path, paper_run_name="full_paper")

    schema = build_primary_baseline_formal_import_schema(target_fpr=FULL_PAPER_FIXED_FPR, root=tmp_path)
    report = validate_primary_baseline_formal_import_rows([row], evidence_root=tmp_path, target_fpr=FULL_PAPER_FIXED_FPR)

    assert schema["paper_claim_scale"] == "full_paper"
    assert schema["prompt_protocol_name"] == "paper_main_full_paper_prompt_protocol"
    assert schema["allowed_resource_profiles"] == ["full_main", "full_extra"]
    assert report["overall_decision"] == "pass"
    assert report["accepted_formal_import_count"] == 1


@pytest.mark.quick
def test_formal_import_validator_rejects_incomplete_adapter_boundary_and_missing_readiness(tmp_path: Path) -> None:
    """method-faithful adapter observation 不得被升级为主表正式结果。"""

    row, _ = write_formal_tree_ring_row(tmp_path)
    row["adapter_boundary"] = "sd35_method_faithful_adapter_not_formal_external_baseline_evidence"
    row["fixed_fpr_baseline_calibration_ready"] = False

    report = validate_primary_baseline_formal_import_rows([row], evidence_root=tmp_path, target_fpr=PILOT_PAPER_FIXED_FPR)
    reasons = {issue["reason"] for issue in report["issues"]}

    assert report["overall_decision"] == "fail"
    assert report["accepted_formal_import_count"] == 0
    assert "adapter_boundary_not_formal" in reasons
    assert "fixed_fpr_baseline_calibration_ready_required" in reasons


@pytest.mark.quick
def test_t2smark_candidate_records_remain_rejected_until_attack_and_threshold_ready(tmp_path: Path) -> None:
    """T2SMark formal 候选记录在攻击矩阵和 fixed-FPR 未闭合前应保持未通过正式导入。"""

    evidence_path = tmp_path / "outputs" / "t2smark_formal_reproduction" / "results.json"
    evidence_path.parent.mkdir(parents=True)
    evidence_path.write_text('{"0":{"robustness":{"norm1_no_w":0.1,"norm1_w":0.9}}}\n', encoding="utf-8")
    jpeg_config = resolve_formal_attack_config(
        attack_family="standard_distortion",
        attack_name="jpeg_compression",
    )
    regeneration_config = resolve_formal_attack_config(
        attack_family="regeneration_attack",
        attack_name="img2img_regeneration",
    )
    jpeg_identity = {
        "attack_id": jpeg_config.attack_id,
        "resource_profile": jpeg_config.resource_profile,
        "attack_config_digest": attack_config_digest(jpeg_config),
    }
    regeneration_identity = {
        "attack_id": regeneration_config.attack_id,
        "resource_profile": regeneration_config.resource_profile,
        "attack_config_digest": attack_config_digest(regeneration_config),
    }
    observations = [
        {"baseline_id": "t2smark", "split": "test", "attack_family": "clean", "attack_condition": "clean_none", "sample_role": "clean_negative", "detection_decision": False},
        {"baseline_id": "t2smark", "split": "test", "attack_family": "standard_distortion", "attack_condition": "jpeg_compression", "sample_role": "attacked_positive", "detection_decision": True, "quality_score": 1.0, "score_retention": 1.0, **jpeg_identity},
        {"baseline_id": "t2smark", "split": "test", "attack_family": "standard_distortion", "attack_condition": "jpeg_compression", "sample_role": "attacked_negative", "detection_decision": False, "quality_score": 1.0, "score_retention": 1.0, **jpeg_identity},
        {"baseline_id": "t2smark", "split": "test", "attack_family": "regeneration_attack", "attack_condition": "img2img_regeneration", "sample_role": "attacked_positive", "detection_decision": True, "quality_score": 0.9, "score_retention": 0.8, **regeneration_identity},
        {"baseline_id": "t2smark", "split": "test", "attack_family": "regeneration_attack", "attack_condition": "img2img_regeneration", "sample_role": "attacked_negative", "detection_decision": False, "quality_score": 0.9, "score_retention": 0.8, **regeneration_identity},
    ]

    records = build_t2smark_formal_candidate_records(
        observation_rows=observations,
        target_fpr=PILOT_PAPER_FIXED_FPR,
        baseline_result_source="outputs/t2smark_formal_reproduction/results.json",
        baseline_result_source_digest=hashlib.sha256(evidence_path.read_bytes()).hexdigest(),
        evidence_paths=["outputs/t2smark_formal_reproduction/results.json"],
        prompt_protocol_digest="prompt_digest",
        paper_run_prompt_protocol_ready=True,
        fixed_fpr_baseline_calibration_ready=False,
        attack_matrix_baseline_detection_ready=False,
    )
    report = validate_primary_baseline_formal_import_rows(records, evidence_root=tmp_path, target_fpr=PILOT_PAPER_FIXED_FPR)
    reasons = {issue["reason"] for issue in report["issues"]}

    assert len(records) == 2
    assert {record["resource_profile"] for record in records} == {"full_main", "full_extra"}
    assert all(record["adapter_boundary"] == "sd35_medium_native_official_reproduction" for record in records)
    assert report["accepted_formal_import_count"] == 0
    assert "fixed_fpr_baseline_calibration_ready_required" in reasons
    assert "attack_matrix_baseline_detection_ready_required" in reasons


@pytest.mark.quick
def test_tree_ring_method_faithful_candidate_records_are_schema_compatible(tmp_path: Path) -> None:
    """Tree-Ring 方法忠实 observations 应能聚合为 formal import 候选记录。"""

    evidence_path = tmp_path / "outputs" / "tree_ring_method_faithful" / "baseline_observations.json"
    evidence_path.parent.mkdir(parents=True)
    observations = build_formal_tree_ring_observations()
    evidence_path.write_text(json.dumps(observations, ensure_ascii=False), encoding="utf-8")

    records = build_tree_ring_method_faithful_candidate_records(
        observation_rows=observations,
        target_fpr=PILOT_PAPER_FIXED_FPR,
        baseline_result_source="outputs/tree_ring_method_faithful/baseline_observations.json",
        baseline_result_source_digest=hashlib.sha256(evidence_path.read_bytes()).hexdigest(),
        evidence_paths=["outputs/tree_ring_method_faithful/baseline_observations.json"],
        prompt_protocol_digest="prompt_digest",
        paper_run_prompt_protocol_ready=True,
        fixed_fpr_baseline_calibration_ready=True,
        attack_matrix_baseline_detection_ready=True,
    )
    report = validate_primary_baseline_formal_import_rows(records, evidence_root=tmp_path, target_fpr=PILOT_PAPER_FIXED_FPR)

    assert len(records) == 1
    assert records[0]["baseline_id"] == "tree_ring"
    assert records[0]["adapter_boundary"] == "method_faithful_sd35_adapter_reproduction"
    assert records[0]["result_source_type"] == "governed_import"
    assert records[0]["attack_record_count"] == 2 * PAPER_RUN_PARAMETERS["pilot_paper"]["test"]
    assert records[0]["supported_record_count"] == PAPER_RUN_PARAMETERS["pilot_paper"]["test"]
    assert report["accepted_formal_import_count"] == 1


@pytest.mark.quick
def test_candidate_builder_rejects_post_labeled_attack_identity(
    tmp_path: Path,
) -> None:
    """候选聚合器不得按攻击名称为缺失身份的 observation 后贴标签."""

    observations = build_formal_tree_ring_observations()
    attacked_positive = next(
        row for row in observations if row["sample_role"] == "attacked_positive"
    )
    del attacked_positive["attack_id"]
    evidence_path = tmp_path / "outputs" / "tree_ring" / "observations.json"
    evidence_path.parent.mkdir(parents=True)
    evidence_path.write_text(json.dumps(observations), encoding="utf-8")

    with pytest.raises(ValueError, match="AttackConfig"):
        build_tree_ring_method_faithful_candidate_records(
            observation_rows=observations,
            target_fpr=PILOT_PAPER_FIXED_FPR,
            baseline_result_source="outputs/tree_ring/observations.json",
            baseline_result_source_digest=hashlib.sha256(
                evidence_path.read_bytes()
            ).hexdigest(),
            evidence_paths=["outputs/tree_ring/observations.json"],
            prompt_protocol_digest="prompt_digest",
            paper_run_prompt_protocol_ready=True,
            fixed_fpr_baseline_calibration_ready=True,
            attack_matrix_baseline_detection_ready=True,
        )


@pytest.mark.quick
def test_formal_template_coverage_requires_matching_formal_attack_records(tmp_path: Path) -> None:
    """正式模板覆盖应检查候选记录是否覆盖共同协议要求的攻击模板。"""

    accepted_row, _ = write_formal_tree_ring_row(tmp_path)
    missing_template = {
        "baseline_id": "tree_ring",
        **formal_attack_descriptor(
            "standard_distortion",
            "gaussian_noise",
            "full_main",
        ),
        "comparable_operating_point": "fixed_fpr_0.01",
    }
    template_rows = [
        {
            "baseline_id": "tree_ring",
            **formal_attack_descriptor(
                "standard_distortion",
                "jpeg_compression",
                "full_main",
            ),
            "comparable_operating_point": "fixed_fpr_0.01",
        },
        missing_template,
    ]
    report = validate_primary_baseline_formal_import_rows([accepted_row], evidence_root=tmp_path, target_fpr=PILOT_PAPER_FIXED_FPR)

    coverage_rows = build_primary_baseline_formal_template_coverage_rows(template_rows, [accepted_row], report)
    coverage_summary = build_primary_baseline_formal_template_coverage_summary(coverage_rows)
    tree_row = next(row for row in coverage_rows if row["baseline_id"] == "tree_ring")

    assert tree_row["expected_formal_template_count"] == 2
    assert tree_row["accepted_template_match_count"] == 1
    assert tree_row["missing_formal_template_count"] == 1
    assert tree_row["formal_template_coverage_ready"] is False
    assert coverage_summary["primary_baseline_formal_template_coverage_ready"] is False


@pytest.mark.quick
def test_formal_template_coverage_rejects_forged_attack_identity() -> None:
    """同名攻击的 attack_id 或配置摘要漂移时不得视为模板已覆盖."""

    template = {
        "baseline_id": "tree_ring",
        **formal_attack_descriptor(
            "standard_distortion",
            "jpeg_compression",
            "full_main",
        ),
        "comparable_operating_point": "fixed_fpr_0.01",
    }
    forged_record = {
        **template,
        "attack_config_digest": "f" * 64,
    }

    coverage_rows = build_primary_baseline_formal_template_coverage_rows(
        [template],
        [forged_record],
        {"accepted_records": [forged_record]},
    )
    tree_row = next(
        row for row in coverage_rows if row["baseline_id"] == "tree_ring"
    )

    assert tree_row["accepted_template_match_count"] == 0
    assert tree_row["unexpected_accepted_record_count"] == 1
    assert tree_row["formal_template_coverage_ready"] is False


@pytest.mark.quick
def test_formal_template_coverage_rejects_unexpected_and_duplicate_accepted_records() -> None:
    """正式模板覆盖必须与当前攻击模板严格相等, 不得接受额外或重复记录。"""

    template = {
        "baseline_id": "tree_ring",
        **formal_attack_descriptor(
            "standard_distortion",
            "jpeg_compression",
            "full_main",
        ),
        "comparable_operating_point": "fixed_fpr_0.01",
    }
    unexpected = {
        "baseline_id": "tree_ring",
        **formal_attack_descriptor(
            "standard_distortion",
            "gaussian_noise",
            "full_main",
        ),
        "comparable_operating_point": "fixed_fpr_0.01",
    }
    accepted_records = [template, dict(template), unexpected]
    report = {"accepted_records": accepted_records}

    coverage_rows = build_primary_baseline_formal_template_coverage_rows(
        [template],
        accepted_records,
        report,
    )
    coverage_summary = build_primary_baseline_formal_template_coverage_summary(coverage_rows)
    tree_row = next(row for row in coverage_rows if row["baseline_id"] == "tree_ring")

    assert tree_row["missing_formal_template_count"] == 0
    assert tree_row["unexpected_accepted_record_count"] == 1
    assert tree_row["duplicate_accepted_template_count"] == 1
    assert tree_row["formal_template_coverage_ready"] is False
    assert coverage_summary["unexpected_accepted_record_count"] == 1
    assert coverage_summary["duplicate_accepted_template_count"] == 1


@pytest.mark.quick
def test_formal_template_coverage_separates_candidate_and_accepted_matches(tmp_path: Path) -> None:
    """已有候选但未通过 validator 时, 摘要应保留候选覆盖进度并继续阻断正式结论。"""

    candidate_row, _ = write_formal_tree_ring_row(tmp_path)
    candidate_row["fixed_fpr_baseline_calibration_ready"] = False
    template_rows = [
        {
            "baseline_id": "tree_ring",
            **formal_attack_descriptor(
                "standard_distortion",
                "jpeg_compression",
                "full_main",
            ),
            "comparable_operating_point": "fixed_fpr_0.01",
        }
    ]
    report = validate_primary_baseline_formal_import_rows([candidate_row], evidence_root=tmp_path, target_fpr=PILOT_PAPER_FIXED_FPR)

    coverage_rows = build_primary_baseline_formal_template_coverage_rows(template_rows, [candidate_row], report)
    coverage_summary = build_primary_baseline_formal_template_coverage_summary(coverage_rows)

    assert report["accepted_formal_import_count"] == 0
    assert coverage_summary["candidate_template_match_count"] == 1
    assert coverage_summary["accepted_template_match_count"] == 0
    assert coverage_summary["missing_candidate_template_count"] == 0
    assert coverage_summary["missing_formal_template_count"] == 1
    assert coverage_summary["primary_baseline_formal_template_coverage_ready"] is False


@pytest.mark.quick
def test_formal_evidence_collection_plan_marks_missing_templates(tmp_path: Path) -> None:
    """正式证据收集计划应把未通过正式导入的模板转换为可执行补证任务。"""

    accepted_row, _ = write_formal_tree_ring_row(tmp_path)
    missing_template = {
        "baseline_id": "tree_ring",
        **formal_attack_descriptor(
            "standard_distortion",
            "gaussian_noise",
            "full_main",
        ),
        "comparable_operating_point": "fixed_fpr_0.01",
        "required_metric_fields": ["true_positive_rate"],
        "required_source_fields": ["baseline_result_source"],
    }
    template_rows = [
        {
            "baseline_id": "tree_ring",
            **formal_attack_descriptor(
                "standard_distortion",
                "jpeg_compression",
                "full_main",
            ),
            "comparable_operating_point": "fixed_fpr_0.01",
            "required_metric_fields": ["true_positive_rate"],
            "required_source_fields": ["baseline_result_source"],
        },
        missing_template,
    ]
    report = validate_primary_baseline_formal_import_rows([accepted_row], evidence_root=tmp_path, target_fpr=PILOT_PAPER_FIXED_FPR)

    collection_rows = build_primary_baseline_formal_evidence_collection_rows(
        template_rows,
        [accepted_row],
        report,
        paper_run_name="pilot_paper",
    )
    collection_summary = build_primary_baseline_formal_evidence_collection_summary(collection_rows)
    missing_row = next(row for row in collection_rows if row["attack_name"] == "gaussian_noise")

    assert len(collection_rows) == 2
    assert missing_row["formal_evidence_collection_ready"] is False
    assert missing_row["required_result_record_path"] == (
        "outputs/external_baseline_results/pilot_paper/baseline_result_records.jsonl"
    )
    assert "generate_paper_baseline_result_record" in missing_row["required_collection_actions"]
    assert collection_summary["formal_evidence_collection_task_count"] == 2
    assert collection_summary["missing_formal_evidence_collection_task_count"] == 1
    assert collection_summary["primary_baseline_formal_evidence_collection_ready"] is False


@pytest.mark.quick
def test_formal_import_protocol_writer_outputs_schema_template_and_validation(tmp_path: Path) -> None:
    """协议写出脚本应生成 schema、模板、候选校验报告和 manifest。"""

    source_root = tmp_path / "external_baseline" / "primary" / "tree_ring" / "source"
    source_root.mkdir(parents=True)
    registry_path = tmp_path / "external_baseline" / "source_registry.json"
    registry_path.write_text(
        json.dumps(
            {
                "baseline_sources": [
                    {
                        "baseline_id": "tree_ring",
                        "baseline_name": "Tree-Ring",
                        "baseline_family": "diffusion_latent_watermark",
                        "comparison_group": "primary",
                        "source_status": "downloaded",
                        "source_dir": "external_baseline/primary/tree_ring/source",
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    attack_dir = tmp_path / "outputs" / "attack_matrix"
    attack_dir.mkdir(parents=True)
    attack_manifest_path = attack_dir / "attack_manifest.json"
    attack_metrics_path = attack_dir / "attack_family_metrics.csv"
    attack_manifest_path.write_text(
        json.dumps({"evaluation_boundary": {"target_fpr": PILOT_PAPER_FIXED_FPR}}, ensure_ascii=False),
        encoding="utf-8",
    )
    with attack_metrics_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "attack_id",
                "attack_family",
                "attack_name",
                "resource_profile",
                "attack_config_digest",
            ],
        )
        writer.writeheader()
        writer.writerow(
            formal_attack_descriptor(
                "standard_distortion",
                "jpeg_compression",
                "full_main",
            )
        )
        writer.writerow(
            formal_attack_descriptor(
                "regeneration_attack",
                "img2img_regeneration",
                "full_extra",
            )
        )
    candidate_records_path = tmp_path / "outputs" / "external_baseline_results" / "baseline_result_records.jsonl"
    candidate_records_path.parent.mkdir(parents=True)
    candidate_records_path.write_text("", encoding="utf-8")

    manifest = write_primary_baseline_formal_import_protocol_outputs(
        root=tmp_path,
        source_registry_path=registry_path,
        attack_manifest_path=attack_manifest_path,
        attack_family_metrics_path=attack_metrics_path,
        candidate_records_path=candidate_records_path,
    )
    output_dir = tmp_path / "outputs" / "primary_baseline_formal_import" / "pilot_paper"
    schema = json.loads((output_dir / "primary_baseline_formal_import_schema.json").read_text(encoding="utf-8"))
    validation = json.loads((output_dir / "primary_baseline_formal_import_validation_report.json").read_text(encoding="utf-8"))
    template_rows = [
        json.loads(line)
        for line in (output_dir / "primary_baseline_formal_result_template.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    readiness_rows = list(
        csv.DictReader((output_dir / "primary_baseline_formal_import_readiness.csv").open(encoding="utf-8"))
    )
    coverage_rows = list(
        csv.DictReader((output_dir / "primary_baseline_formal_template_coverage.csv").open(encoding="utf-8"))
    )
    coverage_summary = json.loads(
        (output_dir / "primary_baseline_formal_template_coverage_summary.json").read_text(encoding="utf-8")
    )
    collection_rows = [
        json.loads(line)
        for line in (output_dir / "primary_baseline_formal_evidence_collection_plan.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    collection_summary = json.loads(
        (output_dir / "primary_baseline_formal_evidence_collection_summary.json").read_text(encoding="utf-8")
    )
    summary = json.loads((output_dir / "primary_baseline_formal_import_summary.json").read_text(encoding="utf-8"))

    assert manifest["artifact_id"] == "primary_baseline_formal_import_protocol_manifest"
    assert schema == build_primary_baseline_formal_import_schema(target_fpr=PILOT_PAPER_FIXED_FPR)
    assert schema["required_threshold_fields"] == [
        "evaluation_split",
        "target_fpr",
        "calibrated_detection_threshold",
        "threshold_source",
        "calibration_clean_negative_count",
        "test_clean_negative_count",
        "threshold_digest",
        "fixed_fpr_observation_evidence_path",
        "fixed_fpr_observation_evidence_digest",
    ]
    assert len(template_rows) == 8
    assert {row["resource_profile"] for row in template_rows} == {"full_main", "full_extra"}
    assert validation["input_record_count"] == 0
    assert len(readiness_rows) == 4
    assert len(coverage_rows) == 4
    assert coverage_summary["formal_template_record_count"] == 8
    assert coverage_summary["missing_formal_template_count"] == 8
    assert len(collection_rows) == 8
    assert collection_summary["formal_evidence_collection_task_count"] == 8
    assert collection_summary["missing_formal_evidence_collection_task_count"] == 8
    assert summary["primary_baseline_formal_ready"] is False
    assert summary["method_threshold_digest_map"] == {}
    assert summary["method_threshold_digest_map_ready"] is False
    assert all(str(path).startswith("outputs/") for path in manifest["output_paths"])


@pytest.mark.quick
def test_formal_import_threshold_digest_map_requires_exact_unique_primary_set() -> None:
    """四个主表 baseline 必须逐方法绑定唯一阈值摘要."""

    rows = [
        {"baseline_id": baseline_id, "threshold_digest": f"{index + 1:x}" * 64}
        for index, baseline_id in enumerate(
            ("tree_ring", "gaussian_shading", "shallow_diffuse", "t2smark")
        )
    ]

    digest_map = build_primary_baseline_method_threshold_digest_map(rows)

    assert set(digest_map) == {
        "tree_ring",
        "gaussian_shading",
        "shallow_diffuse",
        "t2smark",
    }
    with pytest.raises(ValueError, match="多个阈值摘要"):
        build_primary_baseline_method_threshold_digest_map(
            [*rows, {"baseline_id": "tree_ring", "threshold_digest": "f" * 64}]
        )
