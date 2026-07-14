"""真实连续检测分数曲线构造器的轻量功能测试."""

from __future__ import annotations

import csv
import math

import pytest

import experiments.artifacts.detection_score_curves as curve_module
from experiments.artifacts.detection_score_curves import (
    CURVE_POINT_FIELDNAMES,
    SCORE_DISTRIBUTION_FIELDNAMES,
    build_detection_score_tables,
    decision_equivalent_score,
    write_detection_score_tables,
)
from tests.helpers.formal_detection_record import bind_formal_detection_record


def _record(
    prompt_id: str,
    sample_role: str,
    score: float,
    *,
    attack_name: str = "none",
    aligned_score: float | None = None,
    geometry_reliable: bool = False,
) -> dict:
    """构造一条带真实标签和连续分数的 test 记录."""

    record = bind_formal_detection_record({
        "run_id": f"run_{prompt_id}",
        "prompt_id": prompt_id,
        "split": "test",
        "sample_role": sample_role,
        "attack_family": "none" if attack_name == "none" else "compression",
        "attack_name": attack_name,
        "resource_profile": "clean" if attack_name == "none" else "full_main",
        "content_score": score,
        "aligned_content_score": aligned_score,
        "geometry_reliable": geometry_reliable,
        "attention_geometry_score": 0.8 if geometry_reliable else 0.0,
        "registration_confidence": 0.8 if geometry_reliable else 0.0,
        "attention_sync_score": 0.8 if geometry_reliable else 0.0,
        "alignment": {
            "registration_geometry_reliable": geometry_reliable,
        },
    })
    equivalent_score = decision_equivalent_score(
        record,
        geometry_rescue_enabled=True,
        rescue_margin_low=-0.2,
        geometry_score_threshold=0.5,
        registration_confidence_threshold=0.5,
        attention_sync_score_threshold=0.5,
    )
    record["formal_evidence_positive"] = equivalent_score >= 0.5
    return record


def _protocol() -> dict:
    """返回测试使用的冻结完整判定协议."""

    return {
        "content_threshold": 0.5,
        "attention_geometry_enabled": True,
        "image_alignment_enabled": True,
        "geometry_rescue_enabled": True,
        "rescue_margin_low": -0.2,
        "geometry_score_threshold": 0.5,
        "registration_confidence_threshold": 0.5,
        "attention_sync_score_threshold": 0.5,
        "threshold_digest": "threshold_digest_test",
    }


@pytest.mark.quick
def test_decision_equivalent_score_preserves_geometry_rescue_boundary() -> None:
    """连续等价分数应覆盖 aligned score, 但不得突破冻结救回带宽."""

    record = _record(
        "p0",
        "positive_source",
        0.4,
        aligned_score=0.9,
        geometry_reliable=True,
    )

    score = decision_equivalent_score(
        record,
        geometry_rescue_enabled=True,
        rescue_margin_low=-0.2,
        geometry_score_threshold=0.5,
        registration_confidence_threshold=0.5,
        attention_sync_score_threshold=0.5,
    )

    assert score == pytest.approx(0.6)
    assert score >= 0.5
    assert score < 0.61


@pytest.mark.quick
def test_decision_equivalent_score_rejects_registration_below_threshold() -> None:
    """注册置信度未通过冻结阈值时 aligned score 不得形成救回分数."""

    record = _record(
        "registration_gate",
        "positive_source",
        0.4,
        aligned_score=0.9,
        geometry_reliable=True,
    )
    record["registration_confidence"] = 0.49

    score = decision_equivalent_score(
        record,
        geometry_rescue_enabled=True,
        rescue_margin_low=-0.2,
        geometry_score_threshold=0.5,
        registration_confidence_threshold=0.5,
        attention_sync_score_threshold=0.5,
    )

    assert score == pytest.approx(0.4)
    assert score < 0.5


@pytest.mark.quick
def test_decision_equivalent_score_rejects_sync_below_threshold() -> None:
    """恢复后同步分未通过冻结阈值时 aligned score 不得形成救回分数."""

    record = _record(
        "sync_gate",
        "positive_source",
        0.4,
        aligned_score=0.9,
        geometry_reliable=True,
    )
    record["attention_sync_score"] = 0.49

    score = decision_equivalent_score(
        record,
        geometry_rescue_enabled=True,
        rescue_margin_low=-0.2,
        geometry_score_threshold=0.5,
        registration_confidence_threshold=0.5,
        attention_sync_score_threshold=0.5,
    )

    assert score == pytest.approx(0.4)
    assert score < 0.5


def _raw_only_record(
    prompt_id: str,
    sample_role: str,
    score: float,
) -> dict:
    """构造不含注意力与配准原子的 raw-only test 记录。"""

    record = bind_formal_detection_record(
        {
            "run_id": f"run_{prompt_id}",
            "prompt_id": prompt_id,
            "split": "test",
            "sample_role": sample_role,
            "attack_family": "none",
            "attack_name": "none",
            "resource_profile": "clean",
            "content_score": score,
            "aligned_content_score": None,
            "attention_geometry_score": None,
            "registration_confidence": None,
            "attention_sync_score": None,
            "alignment": None,
            "metadata": {
                "attention_geometry_enabled": False,
                "image_alignment_enabled": False,
            },
        }
    )
    record["formal_evidence_positive"] = score >= 0.5
    return record


def _raw_only_protocol() -> dict:
    """返回禁用几何 rescue 的冻结 raw-only 协议。"""

    return {
        "content_threshold": 0.5,
        "attention_geometry_enabled": False,
        "image_alignment_enabled": False,
        "geometry_rescue_enabled": False,
        "rescue_margin_low": None,
        "geometry_score_threshold": None,
        "registration_confidence_threshold": None,
        "attention_sync_score_threshold": None,
        "threshold_digest": "raw_only_threshold_digest_test",
    }


@pytest.mark.quick
def test_curve_builder_uses_raw_score_when_geometry_rescue_is_disabled() -> None:
    """正式消融关闭 rescue 时必须以 raw 分数构造曲线。"""

    records = (
        _raw_only_record("raw_positive", "positive_source", 0.8),
        _raw_only_record("raw_negative", "clean_negative", 0.2),
    )

    tables = build_detection_score_tables(records, _raw_only_protocol())

    distribution = tables["score_distribution_table"]
    assert len(distribution) == 4
    expected_score_by_role = {
        "positive_source": 0.8,
        "clean_negative": 0.2,
    }
    assert all(
        row["score"]
        == pytest.approx(expected_score_by_role[row["sample_role"]])
        for row in distribution
    )


@pytest.mark.quick
@pytest.mark.parametrize(
    "updates",
    (
        {"geometry_rescue_enabled": True},
        {"rescue_margin_low": -0.2},
    ),
)
def test_curve_builder_rejects_disabled_rescue_identity_drift(
    updates: dict,
) -> None:
    """机制开关与 rescue 参数不一致时不得生成曲线。"""

    protocol = {**_raw_only_protocol(), **updates}
    records = (
        _raw_only_record("raw_positive", "positive_source", 0.8),
        _raw_only_record("raw_negative", "clean_negative", 0.2),
    )

    with pytest.raises(ValueError, match="rescue"):
        build_detection_score_tables(records, protocol)


@pytest.mark.quick
def test_complete_threshold_sweep_has_real_endpoints_and_monotone_counts() -> None:
    """ROC / DET 必须枚举唯一连续分数, 并从全阴性端点走到全阳性端点."""

    records = (
        _record("p0", "positive_source", 0.9),
        _record("p1", "positive_source", 0.7),
        _record("p2", "clean_negative", 0.7),
        _record("p3", "wrong_key_negative", 0.2),
        _record("p4", "positive_source", 0.8, attack_name="jpeg"),
        _record("p5", "clean_negative", 0.3, attack_name="jpeg"),
        {**_record("ignored", "positive_source", 1.0), "split": "calibration"},
    )

    tables = build_detection_score_tables(records, _protocol())
    overall = [
        row
        for row in tables["roc_curve_points"]
        if row["sample_scope"] == "test_overall"
    ]

    assert len(overall) == len({0.9, 0.8, 0.7, 0.3, 0.2}) + 2
    assert overall[0]["threshold_kind"] == "positive_infinity_endpoint"
    assert math.isinf(overall[0]["threshold"]) and overall[0]["threshold"] > 0
    assert (overall[0]["tp"], overall[0]["fp"]) == (0, 0)
    assert overall[-1]["threshold_kind"] == "negative_infinity_endpoint"
    assert math.isinf(overall[-1]["threshold"]) and overall[-1]["threshold"] < 0
    assert overall[-1]["tp"] == overall[-1]["positive_count"]
    assert overall[-1]["fp"] == overall[-1]["negative_count"]
    assert [row["tpr"] for row in overall] == sorted(row["tpr"] for row in overall)
    assert [row["fpr"] for row in overall] == sorted(row["fpr"] for row in overall)
    assert [row["fnr"] for row in overall] == sorted(
        (row["fnr"] for row in overall),
        reverse=True,
    )
    assert tables["roc_curve_points"] == tables["det_curve_points"]
    assert any(
        row["sample_scope"] == "test_attack_condition" and row["attack_name"] == "jpeg"
        for row in tables["roc_curve_points"]
    )


@pytest.mark.quick
def test_curve_writer_uses_exact_governed_columns(tmp_path) -> None:
    """三张图数据表应采用固定列集合, 避免下游依赖隐式列推断."""

    records = (
        _record("p0", "positive_source", 0.8),
        _record("p1", "clean_negative", 0.1),
    )
    tables = build_detection_score_tables(records, _protocol())

    paths = write_detection_score_tables(tmp_path, tables)

    with paths["score_distribution_table"].open(encoding="utf-8", newline="") as stream:
        distribution_reader = csv.DictReader(stream)
        distribution_rows = list(distribution_reader)
    with paths["roc_curve_points"].open(encoding="utf-8", newline="") as stream:
        curve_reader = csv.DictReader(stream)
        curve_rows = list(curve_reader)
    assert tuple(distribution_reader.fieldnames or ()) == SCORE_DISTRIBUTION_FIELDNAMES
    assert tuple(curve_reader.fieldnames or ()) == CURVE_POINT_FIELDNAMES
    assert distribution_rows
    assert curve_rows


@pytest.mark.quick
def test_curve_builder_rejects_single_class_test_scope() -> None:
    """没有真实阴性样本时不得输出形式上存在但统计上无定义的曲线."""

    with pytest.raises(ValueError, match="同时包含阳性与阴性"):
        build_detection_score_tables(
            (_record("p0", "positive_source", 0.8),),
            _protocol(),
        )


@pytest.mark.quick
def test_complete_sweep_does_not_rescan_observations_per_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """完整阈值 sweep 应按分组线性累计, 不得对每个阈值重新扫描全部记录."""

    records = tuple(
        _record(
            f"p{index}",
            "positive_source" if index % 2 == 0 else "clean_negative",
            1.0 - index / 200.0,
        )
        for index in range(200)
    )
    original_confusion = curve_module._confusion
    call_count = 0

    def counted_confusion(*args: object, **kwargs: object) -> dict[str, int | float]:
        """统计 operating point 混淆矩阵调用次数."""

        nonlocal call_count
        call_count += 1
        return original_confusion(*args, **kwargs)

    monkeypatch.setattr(curve_module, "_confusion", counted_confusion)
    tables = build_detection_score_tables(records, _protocol())
    overall_rows = [
        row
        for row in tables["roc_curve_points"]
        if row["sample_scope"] == "test_overall"
    ]

    assert len(overall_rows) == 202
    assert call_count == 2
