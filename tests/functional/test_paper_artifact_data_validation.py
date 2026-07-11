"""论文表图实际数据验证器的轻量功能测试."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from experiments.ablations.runtime_rerun import FORMAL_RUNTIME_RERUN_ABLATION_IDS
from experiments.artifacts.detection_score_curves import (
    build_detection_score_tables,
    write_detection_score_tables,
)
from experiments.artifacts.image_only_detection_metrics import (
    build_image_only_test_metric_rows,
)
from experiments.protocol.attacks import attack_config_digest, default_attack_configs
from paper_experiments.analysis.paper_artifact_data_validation import (
    ABLATION_DELTA_FIELDS,
    ABLATION_METRIC_FIELDS,
    ATTACK_METRIC_FIELDS,
    BASELINE_COMPARISON_FIELDS,
    DATASET_QUALITY_FIELDS,
    TEST_METRIC_FIELDS,
    validate_paper_artifact_source_data,
)


TARGET_FPR = 0.1
THRESHOLD_DIGEST = "threshold_digest_validation"
FORMAL_ATTACK_CONFIGS = tuple(
    config
    for config in default_attack_configs()
    if config.enabled
    and config.resource_profile in {"full_main", "full_extra"}
)


def _write_csv(path: Path, fieldnames: tuple[str, ...] | set[str], rows: list[dict]) -> None:
    """按调用者给出的受治理列顺序写出测试 CSV."""

    resolved_fields = tuple(fieldnames)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=resolved_fields)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> tuple[tuple[str, ...], list[dict[str, str]]]:
    """读取测试 CSV 的列名与行, 供负向测试精确改写单一事实."""

    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        return tuple(reader.fieldnames or ()), list(reader)


def _detection_record(prompt_id: str, sample_role: str, score: float) -> dict:
    """构造与冻结协议一致的 test 连续检测记录."""

    return {
        "run_id": f"run_{prompt_id}",
        "prompt_id": prompt_id,
        "split": "test",
        "sample_role": sample_role,
        "attack_family": "none",
        "attack_name": "none",
        "resource_profile": "clean",
        "content_score": score,
        "aligned_content_score": None,
        "source_to_evaluated_ssim": 1.0,
        "source_to_evaluated_psnr": 40.0,
        "geometry_reliable": False,
        "attention_geometry_score": 0.0,
        "formal_evidence_positive": score >= 0.5,
        "detector_digest": f"detector_{prompt_id}_{sample_role}",
    }


def _row(fields: set[str], **values: object) -> dict:
    """以空字符串补齐固定 schema, 便于突出当前测试关注字段."""

    return {field_name: values.get(field_name, "") for field_name in fields}


def _prepare_valid_sources(root: Path) -> dict[str, Path]:
    """写出能够独立通过全部11类数据检查的最小正式形状."""

    runtime_dir = root / "runtime"
    attack_dir = root / "attack"
    baseline_dir = root / "baseline"
    ablation_dir = root / "ablation"
    quality_dir = root / "quality"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    protocol = {
        "content_threshold": 0.5,
        "rescue_margin_low": -0.2,
        "geometry_score_threshold": 0.5,
        "geometry_calibration_negative_count": 2,
        "geometry_calibration_exceedance_count": 0,
        "calibration_negative_count": 10,
        "calibration_false_positive_count": 1,
        "calibration_false_positive_rate": 0.1,
        "target_fpr": TARGET_FPR,
        "threshold_digest": THRESHOLD_DIGEST,
    }
    (runtime_dir / "frozen_evidence_protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False),
        encoding="utf-8",
    )
    detection_records = (
        _detection_record("p0", "positive_source", 0.9),
        _detection_record("p1", "positive_source", 0.7),
        _detection_record("p2", "clean_negative", 0.3),
        _detection_record("p3", "wrong_key_negative", 0.1),
    )
    test_rows = build_image_only_test_metric_rows(
        detection_records,
        TARGET_FPR,
    )
    _write_csv(
        runtime_dir / "test_detection_metrics.csv",
        tuple(sorted(TEST_METRIC_FIELDS)),
        list(test_rows),
    )
    (runtime_dir / "image_only_detection_records.jsonl").write_text(
        "\n".join(
            json.dumps(record, ensure_ascii=False, sort_keys=True)
            for record in detection_records
        )
        + "\n",
        encoding="utf-8",
    )
    write_detection_score_tables(
        runtime_dir,
        build_detection_score_tables(detection_records, protocol),
    )

    attack_rows = [
        _row(
            ATTACK_METRIC_FIELDS,
            attack_id=config.attack_id,
            attack_family=config.attack_family,
            attack_name=config.attack_name,
            resource_profile=config.resource_profile,
            attack_config_digest=attack_config_digest(config),
            metric_status=(
                "measured_real_attacked_image_image_only_detection"
            ),
            attack_record_count=2,
            supported_record_count=2,
            unsupported_record_count=0,
            positive_count=1,
            negative_count=1,
            true_positive_rate=1.0,
            false_positive_rate=0.0,
            clean_false_positive_rate=0.0,
            attacked_false_positive_rate=0.0,
            false_positive_rate_upper_95=0.95,
            target_fpr=TARGET_FPR,
            fixed_fpr_upper_bound_ready=False,
            quality_score_mean=0.9,
            quality_ssim_mean=0.9,
            quality_psnr_mean=30.0,
            attacked_positive_source_to_attacked_ssim_mean=0.9,
            score_retention_mean=0.8,
            lf_score_retention_mean=0.8,
            tail_score_retention_mean=0.8,
            geometry_reliable_rate=1.0,
            rescue_rate=0.0,
            supports_paper_claim=True,
        )
        for config in FORMAL_ATTACK_CONFIGS
    ]
    _write_csv(
        attack_dir / "attack_family_metrics.csv",
        tuple(sorted(ATTACK_METRIC_FIELDS)),
        attack_rows,
    )

    baseline_rows = []
    for index, (method_id, method_role) in enumerate(
        (
            ("slm_wm_current", "proposed_method_governed_result"),
            ("tree_ring", "external_baseline_primary"),
            ("gaussian_shading", "external_baseline_primary"),
            ("shallow_diffuse", "external_baseline_primary"),
            ("t2smark", "external_baseline_primary"),
        )
    ):
        baseline_rows.append(
            _row(
                BASELINE_COMPARISON_FIELDS,
                method_id=method_id,
                method_role=method_role,
                comparison_scope="common_protocol_governed_result",
                common_prompt_protocol_ready=True,
                common_attack_protocol_ready=True,
                common_threshold_protocol_ready=True,
                metric_status="measured_governed_result",
                true_positive_rate=0.9 - index * 0.05,
                false_positive_rate=0.05,
                clean_false_positive_rate=0.05,
                attacked_false_positive_rate=0.05,
                quality_score_mean=0.9,
                supports_paper_claim=True,
            )
        )
    _write_csv(
        baseline_dir / "baseline_comparison_table.csv",
        tuple(BASELINE_COMPARISON_FIELDS),
        baseline_rows,
    )

    ablation_rows = [
        _row(
            ABLATION_METRIC_FIELDS,
            ablation_id=ablation_id,
            test_prompt_count=34,
            clean_false_positive_rate=0.05,
            wrong_key_false_positive_rate=0.05,
            clean_true_positive_rate=0.9,
            attacked_true_positive_rate=0.8,
            attacked_false_positive_rate=0.05,
            positive_content_score_mean=0.7,
            paired_ssim_mean=0.95,
            frozen_threshold_digest=f"digest_{ablation_id}",
            metric_status="measured_full_runtime_rerun",
        )
        for ablation_id in FORMAL_RUNTIME_RERUN_ABLATION_IDS
    ]
    _write_csv(
        ablation_dir / "mechanism_ablation_metrics.csv",
        tuple(ABLATION_METRIC_FIELDS),
        ablation_rows,
    )
    delta_rows = [
        _row(
            ABLATION_DELTA_FIELDS,
            ablation_id=ablation_id,
            clean_true_positive_rate_delta=-0.1,
            attacked_true_positive_rate_delta=-0.1,
            paired_ssim_delta=0.0,
            metric_status="measured_full_runtime_rerun",
        )
        for ablation_id in FORMAL_RUNTIME_RERUN_ABLATION_IDS
        if ablation_id != "complete_method"
    ]
    _write_csv(
        ablation_dir / "mechanism_pairwise_delta.csv",
        tuple(ABLATION_DELTA_FIELDS),
        delta_rows,
    )

    quality_rows = [
        _row(
            DATASET_QUALITY_FIELDS,
            quality_metric_name=metric_name,
            quality_metric_value=value,
            metric_status="measured",
            paper_metric_name=metric_name.upper(),
            feature_backend="torch_fidelity_inception_v3_compat",
            source_image_count=70,
            comparison_image_count=70,
            sample_pair_count=70,
            supports_paper_claim=False,
        )
        for metric_name, value in (("fid", 8.0), ("kid", 0.01))
    ]
    _write_csv(
        quality_dir / "dataset_quality_metrics.csv",
        tuple(DATASET_QUALITY_FIELDS),
        quality_rows,
    )
    return {
        "frozen_evidence_protocol_ready": runtime_dir / "frozen_evidence_protocol.json",
        "raw_image_only_detection_records_ready": (
            runtime_dir / "image_only_detection_records.jsonl"
        ),
        "test_detection_metrics_ready": runtime_dir / "test_detection_metrics.csv",
        "score_distribution_table_ready": runtime_dir / "score_distribution_table.csv",
        "roc_curve_points_ready": runtime_dir / "roc_curve_points.csv",
        "det_curve_points_ready": runtime_dir / "det_curve_points.csv",
        "attack_family_metrics_ready": attack_dir / "attack_family_metrics.csv",
        "baseline_comparison_table_ready": baseline_dir / "baseline_comparison_table.csv",
        "mechanism_ablation_metrics_ready": ablation_dir / "mechanism_ablation_metrics.csv",
        "mechanism_pairwise_delta_ready": ablation_dir / "mechanism_pairwise_delta.csv",
        "dataset_quality_metrics_ready": quality_dir / "dataset_quality_metrics.csv",
    }


def _validate(root: Path, source_paths: dict[str, Path]) -> dict:
    """以全部 ready 的摘要调用实际数据验证器."""

    return validate_paper_artifact_source_data(
        root_path=root,
        source_paths=source_paths,
        threshold_report={
            "target_fpr": TARGET_FPR,
            "frozen_threshold_digest": THRESHOLD_DIGEST,
            "full_method_claim_ready": True,
        },
        attack_manifest={"attack_metrics_ready": True},
        baseline_runtime_report={"comparison_table_supports_paper_claim": True},
        dataset_quality_summary={"formal_fid_kid_claim_gate_ready": True},
        ablation_claim_summary={"ablation_claim_gate_ready": True},
    )


@pytest.mark.quick
def test_actual_paper_artifact_data_passes_exact_schema_and_numeric_validation(tmp_path: Path) -> None:
    """全部实际数据有效时应产生11文件摘要与一致的 ready 结论."""

    source_paths = _prepare_valid_sources(tmp_path)

    report = _validate(tmp_path, source_paths)

    assert report["artifact_data_validation_ready"] is True
    assert report["raw_image_only_detection_records_ready"] is True
    assert report["blocked_artifact_data_ids"] == []
    assert report["artifact_data_check_count"] == 12
    assert len(report["evidence_source_file_sha256"]) == 11
    assert all(len(digest) == 64 for digest in report["evidence_source_file_sha256"].values())
    raw_path = report["source_paths"]["raw_image_only_detection_records_ready"]
    assert report["raw_image_only_detection_records_sha256"] == report[
        "evidence_source_file_sha256"
    ][raw_path]


@pytest.mark.quick
def test_invalid_roc_endpoint_blocks_curve_and_ready_flag_consistency(tmp_path: Path) -> None:
    """单点或缺端点的伪曲线不得被 readiness flag 掩盖."""

    source_paths = _prepare_valid_sources(tmp_path)
    roc_path = source_paths["roc_curve_points_ready"]
    with roc_path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        fieldnames = tuple(reader.fieldnames or ())
    rows[0]["threshold"] = "1.0"
    _write_csv(roc_path, fieldnames, rows)

    report = _validate(tmp_path, source_paths)

    assert report["roc_curve_points_ready"] is False
    assert report["ready_flag_consistency_ready"] is False
    assert report["artifact_data_validation_ready"] is False
    assert "roc_curve_points_ready" in report["blocked_artifact_data_ids"]


@pytest.mark.quick
def test_monotonic_but_forged_roc_is_rejected_by_raw_record_rebuild(tmp_path: Path) -> None:
    """结构单调但不等于原始记录重建值的 ROC 不得进入正式证据."""

    source_paths = _prepare_valid_sources(tmp_path)
    roc_path = source_paths["roc_curve_points_ready"]
    with roc_path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        fieldnames = tuple(reader.fieldnames or ())
    rows[1]["threshold"] = "0.89"
    _write_csv(roc_path, fieldnames, rows)

    report = _validate(tmp_path, source_paths)

    assert report["raw_image_only_detection_records_ready"] is True
    assert report["roc_curve_points_ready"] is False
    assert any(
        "原始检测记录重建值不一致" in issue
        for issue in report["checks"]["roc_curve_points_ready"]["issues"]
    )


@pytest.mark.quick
def test_reordered_score_distribution_is_rejected_by_exact_rebuild(tmp_path: Path) -> None:
    """字段和值未变但行序被改写的分数分布表不得通过正式审计."""

    source_paths = _prepare_valid_sources(tmp_path)
    distribution_path = source_paths["score_distribution_table_ready"]
    with distribution_path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        fieldnames = tuple(reader.fieldnames or ())
    rows[0], rows[1] = rows[1], rows[0]
    _write_csv(distribution_path, fieldnames, rows)

    report = _validate(tmp_path, source_paths)

    assert report["score_distribution_table_ready"] is False
    assert any(
        "原始检测记录重建值不一致" in issue
        for issue in report["checks"]["score_distribution_table_ready"]["issues"]
    )


@pytest.mark.quick
def test_synchronized_test_metric_forgery_is_rejected_by_raw_record_rebuild(
    tmp_path: Path,
) -> None:
    """计数、比率、分数与 SSIM 同步改写后仍须匹配记录级事实来源."""

    source_paths = _prepare_valid_sources(tmp_path)
    metrics_path = source_paths["test_detection_metrics_ready"]
    fieldnames, rows = _read_csv(metrics_path)
    positive_row = next(
        row for row in rows if row["sample_role"] == "positive_source"
    )
    positive_row["positive_count"] = "1"
    positive_row["positive_rate"] = "0.5"
    positive_row["content_score_mean"] = "0.75"
    positive_row["source_to_evaluated_ssim_mean"] = "0.9"
    _write_csv(metrics_path, fieldnames, rows)

    report = _validate(tmp_path, source_paths)

    assert report["raw_image_only_detection_records_ready"] is True
    assert report["test_detection_metrics_ready"] is False
    assert any(
        "原始检测记录重建值不一致" in issue
        for issue in report["checks"]["test_detection_metrics_ready"]["issues"]
    )


@pytest.mark.quick
def test_missing_raw_detection_records_blocks_all_derived_curve_tables(tmp_path: Path) -> None:
    """缺少记录级事实来源时, 四张派生检测表必须共同 fail closed."""

    source_paths = _prepare_valid_sources(tmp_path)
    source_paths["raw_image_only_detection_records_ready"].unlink()

    report = _validate(tmp_path, source_paths)

    assert report["raw_image_only_detection_records_ready"] is False
    assert report["raw_image_only_detection_records_sha256"] == ""
    assert all(
        report[check_id] is False
        for check_id in (
            "test_detection_metrics_ready",
            "score_distribution_table_ready",
            "roc_curve_points_ready",
            "det_curve_points_ready",
        )
    )


@pytest.mark.quick
def test_attack_metrics_reject_missing_formal_attack_identity(tmp_path: Path) -> None:
    """任一正式攻击行缺少 attack_id 时不得通过表图证据审计."""

    source_paths = _prepare_valid_sources(tmp_path)
    metrics_path = source_paths["attack_family_metrics_ready"]
    fieldnames, rows = _read_csv(metrics_path)
    rows[0]["attack_id"] = ""
    _write_csv(metrics_path, fieldnames, rows)

    report = _validate(tmp_path, source_paths)

    assert report["attack_family_metrics_ready"] is False
    assert any(
        "攻击指标身份或配置摘要不一致" in issue
        for issue in report["checks"]["attack_family_metrics_ready"]["issues"]
    )


@pytest.mark.quick
def test_attack_metrics_reject_forged_attack_config_digest(tmp_path: Path) -> None:
    """攻击摘要即使形状合法, 与正式配置不一致时也必须阻断."""

    source_paths = _prepare_valid_sources(tmp_path)
    metrics_path = source_paths["attack_family_metrics_ready"]
    fieldnames, rows = _read_csv(metrics_path)
    rows[0]["attack_config_digest"] = "f" * 64
    _write_csv(metrics_path, fieldnames, rows)

    report = _validate(tmp_path, source_paths)

    assert report["attack_family_metrics_ready"] is False
    assert any(
        "攻击指标身份或配置摘要不一致" in issue
        for issue in report["checks"]["attack_family_metrics_ready"]["issues"]
    )


@pytest.mark.quick
def test_attack_metrics_require_exact_formal_attack_set(tmp_path: Path) -> None:
    """指标表缺少任一 full_main 或 full_extra 攻击时必须阻断."""

    source_paths = _prepare_valid_sources(tmp_path)
    metrics_path = source_paths["attack_family_metrics_ready"]
    fieldnames, rows = _read_csv(metrics_path)
    _write_csv(metrics_path, fieldnames, rows[:-1])

    report = _validate(tmp_path, source_paths)

    assert report["attack_family_metrics_ready"] is False
    assert any(
        "攻击指标未精确覆盖全部正式攻击" in issue
        for issue in report["checks"]["attack_family_metrics_ready"]["issues"]
    )
