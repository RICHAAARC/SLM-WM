"""对论文表图实际数据执行结构,数值与跨摘要一致性验证."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import fields
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from experiments.ablations.necessity_statistics import (
    ABLATION_NECESSITY_ALPHA,
    ABLATION_NECESSITY_ANALYSIS_SCHEMA,
    ABLATION_NECESSITY_BOOTSTRAP_BIT_GENERATOR,
    ABLATION_NECESSITY_BOOTSTRAP_QUANTILE_METHOD,
    ABLATION_NECESSITY_EFFECT_DIRECTION,
    ABLATION_NECESSITY_FIELDNAMES,
    ABLATION_NECESSITY_MINIMUM_EFFECT_SIZE,
    ABLATION_NECESSITY_PAIRED_SSIM_NONINFERIORITY_MARGIN,
    ABLATION_NECESSITY_P_VALUE_METHOD,
    ABLATION_NECESSITY_PRIMARY_METRIC,
    canonicalize_ablation_necessity_rows,
)
from experiments.ablations.runtime_rerun import FORMAL_RUNTIME_RERUN_ABLATION_IDS
from experiments.artifacts.detection_score_curves import (
    CURVE_POINT_FIELDNAMES,
    SCORE_DISTRIBUTION_FIELDNAMES,
    build_detection_score_tables,
)
from experiments.artifacts.image_only_detection_metrics import (
    build_image_only_test_metric_rows,
)
from experiments.protocol.attacks import (
    attack_config_digest,
    default_attack_configs,
    resolve_formal_attack_config,
)
from experiments.protocol.dataset_quality import (
    FORMAL_DATASET_QUALITY_METRIC_NAMES,
)
from main.core.digest import build_stable_digest
from experiments.runners.image_only_dataset_runtime import (
    FrozenEvidenceProtocol,
    validate_detection_attention_alignment_gate,
)
from paper_experiments.analysis.formal_record_statistics import (
    validate_frozen_evidence_protocol_record,
)


FROZEN_PROTOCOL_FIELDS = {
    field.name for field in fields(FrozenEvidenceProtocol)
}

TEST_METRIC_FIELDS = {
    "attack_family",
    "attack_name",
    "resource_profile",
    "sample_role",
    "record_count",
    "positive_count",
    "positive_rate",
    "content_score_mean",
    "source_to_evaluated_ssim_mean",
    "source_to_evaluated_psnr_mean",
    "positive_rate_upper_95",
    "target_fpr",
    "fixed_fpr_upper_bound_ready",
    "metric_status",
    "supports_paper_claim",
}

ATTACK_METRIC_FIELDS = {
    "attack_id",
    "attack_family",
    "attack_name",
    "resource_profile",
    "attack_config_digest",
    "metric_status",
    "attack_record_count",
    "supported_record_count",
    "unsupported_record_count",
    "positive_count",
    "negative_count",
    "true_positive_rate",
    "false_positive_rate",
    "clean_false_positive_rate",
    "attacked_false_positive_rate",
    "false_positive_rate_upper_95",
    "target_fpr",
    "fixed_fpr_upper_bound_ready",
    "quality_score_mean",
    "quality_ssim_mean",
    "quality_psnr_mean",
    "attacked_positive_source_to_attacked_ssim_mean",
    "score_retention_mean",
    "lf_score_retention_mean",
    "tail_score_retention_mean",
    "geometry_reliable_rate",
    "rescue_rate",
    "supports_paper_claim",
}

BASELINE_COMPARISON_FIELDS = {
    "method_id",
    "method_role",
    "comparison_scope",
    "common_prompt_protocol_ready",
    "common_attack_protocol_ready",
    "common_threshold_protocol_ready",
    "metric_status",
    "true_positive_rate",
    "false_positive_rate",
    "clean_false_positive_rate",
    "attacked_false_positive_rate",
    "quality_score_mean",
    "supports_paper_claim",
}

ABLATION_METRIC_FIELDS = {
    "ablation_id",
    "test_prompt_count",
    "clean_false_positive_rate",
    "wrong_key_false_positive_rate",
    "clean_true_positive_rate",
    "attacked_true_positive_rate",
    "attacked_false_positive_rate",
    "positive_content_score_mean",
    "paired_ssim_mean",
    "frozen_threshold_digest",
    "metric_status",
}

ABLATION_DELTA_FIELDS = {
    "ablation_id",
    "clean_true_positive_rate_delta",
    "attacked_true_positive_rate_delta",
    "paired_ssim_delta",
    "metric_status",
}

ABLATION_NECESSITY_FIELDS = set(ABLATION_NECESSITY_FIELDNAMES)

DATASET_QUALITY_FIELDS = {
    "quality_metric_name",
    "quality_metric_value",
    "metric_status",
    "paper_metric_name",
    "feature_backend",
    "source_image_count",
    "comparison_image_count",
    "sample_pair_count",
    "supports_paper_claim",
}


def _as_bool(value: Any) -> bool:
    """解析 CSV 与 JSON 中允许出现的布尔表示."""

    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "pass", "ready"}


def _finite(value: Any) -> float:
    """解析一个必须有限的统计数值."""

    resolved = float(value)
    if not math.isfinite(resolved):
        raise ValueError("数值不是有限值")
    return resolved


def _rate(value: Any) -> float:
    """解析一个位于闭区间 [0, 1] 的比率."""

    resolved = _finite(value)
    if not 0.0 <= resolved <= 1.0:
        raise ValueError("比率超出 [0, 1]")
    return resolved


def _count(value: Any, *, positive: bool = False) -> int:
    """解析非负或正整数计数."""

    resolved = int(value)
    if str(resolved) != str(value).strip() and float(value) != resolved:
        raise ValueError("计数不是整数")
    if resolved < (1 if positive else 0):
        raise ValueError("计数低于允许下界")
    return resolved


def _same_float(left: Any, right: Any) -> bool:
    """以冻结协议精度比较两个有限浮点数."""

    try:
        return math.isclose(_finite(left), _finite(right), rel_tol=0.0, abs_tol=1e-12)
    except (TypeError, ValueError, OverflowError):
        return False


def _read_csv_exact(path: Path, expected_fields: Iterable[str]) -> list[dict[str, str]]:
    """读取非空 CSV 并要求列集合与受治理 schema 完全一致."""

    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError("文件缺失或为空")
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        actual_fields = tuple(reader.fieldnames or ())
        rows = list(reader)
    expected = tuple(expected_fields)
    if set(actual_fields) != set(expected) or len(actual_fields) != len(expected):
        raise ValueError(
            f"CSV 列集合不一致: actual={','.join(actual_fields)};expected={','.join(expected)}"
        )
    if not rows:
        raise ValueError("CSV 不得只有表头")
    if any(None in row for row in rows):
        raise ValueError("CSV 数据行包含表头之外的额外字段")
    return rows


def _read_curve_csv(path: Path) -> list[dict[str, str]]:
    """读取曲线表并额外固定列顺序."""

    rows = _read_csv_exact(path, CURVE_POINT_FIELDNAMES)
    with path.open(encoding="utf-8-sig", newline="") as stream:
        fieldnames = tuple(csv.DictReader(stream).fieldnames or ())
    if fieldnames != CURVE_POINT_FIELDNAMES:
        raise ValueError("曲线表列顺序不符合受治理 schema")
    return rows


def _read_jsonl_records(path: Path) -> list[dict[str, Any]]:
    """读取非空 JSONL 原始记录, 并拒绝非对象行."""

    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError("文件缺失或为空")
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"JSONL 第{line_number}行不是对象")
            records.append(payload)
    if not records:
        raise ValueError("JSONL 不得为空")
    return records


def _validate_raw_detection_records(
    path: Path,
    context: Mapping[str, Any],
) -> dict[str, Any]:
    """从原始记录重建 test operating point 与三张连续分数表.

    这一实现属于通用证据审计写法: 审计器读取记录级事实来源, 再调用与正式
    runtime 相同的构建函数重算派生产物, 避免只对派生 CSV 做自洽性检查.
    """

    frozen_protocol = context.get("frozen_protocol")
    if not isinstance(frozen_protocol, Mapping):
        raise ValueError("冻结协议无效, 无法重建连续检测表")
    records = _read_jsonl_records(path)
    for record in records:
        validate_detection_attention_alignment_gate(record)
    rebuilt_tables = {
        **build_detection_score_tables(records, frozen_protocol),
        "test_detection_metrics": build_image_only_test_metric_rows(
            records,
            _finite(context["target_fpr"]),
        ),
    }
    return {
        "row_count": len(records),
        "rebuilt_tables": rebuilt_tables,
    }


def _csv_cell_text(value: Any) -> str:
    """复现 ``csv.DictWriter`` 对单元格的稳定文本转换."""

    return "" if value is None else str(value)


def _validate_rebuilt_csv_exact(
    path: Path,
    rebuilt_rows: Iterable[Mapping[str, Any]],
    fieldnames: tuple[str, ...],
) -> None:
    """逐行逐字段比较重建结果与正式 CSV, 不允许重排或数值改写."""

    actual_rows = _read_csv_exact(path, fieldnames)
    with path.open(encoding="utf-8-sig", newline="") as stream:
        actual_fieldnames = tuple(csv.DictReader(stream).fieldnames or ())
    if actual_fieldnames != fieldnames:
        raise ValueError("重建比较的 CSV 列顺序不符合受治理 schema")
    expected_rows = [
        {field_name: _csv_cell_text(row[field_name]) for field_name in fieldnames}
        for row in rebuilt_rows
    ]
    if len(actual_rows) != len(expected_rows):
        raise ValueError(
            f"原始检测记录重建行数不一致: actual={len(actual_rows)};"
            f"expected={len(expected_rows)}"
        )
    for row_index, (actual_row, expected_row) in enumerate(
        zip(actual_rows, expected_rows, strict=True)
    ):
        for field_name in fieldnames:
            if actual_row[field_name] != expected_row[field_name]:
                raise ValueError(
                    "原始检测记录重建值不一致: "
                    f"row={row_index};field={field_name};"
                    f"actual={actual_row[field_name]};expected={expected_row[field_name]}"
                )


def _validate_frozen_protocol(path: Path, context: Mapping[str, Any]) -> dict[str, Any]:
    """验证完整冻结协议的字段与数值边界."""

    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError("文件缺失或为空")
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict) or set(payload) != FROZEN_PROTOCOL_FIELDS:
        raise ValueError("冻结协议字段集合不一致")
    validate_frozen_evidence_protocol_record(
        payload,
        expected_target_fpr=_finite(context["target_fpr"]),
    )
    if not _same_float(payload["target_fpr"], context["target_fpr"]):
        raise ValueError("冻结协议 target_fpr 与运行摘要不一致")
    if str(payload["threshold_digest"]) != str(context["threshold_digest"]):
        raise ValueError("冻结协议摘要与运行摘要不一致")
    return {"row_count": 1, "protocol": payload}


def _validate_test_metrics(path: Path, context: Mapping[str, Any]) -> dict[str, Any]:
    """验证 test split operating point 聚合表."""

    rows = _read_csv_exact(path, TEST_METRIC_FIELDS)
    roles = {row["sample_role"] for row in rows}
    if not {"positive_source", "clean_negative", "wrong_key_negative"}.issubset(roles):
        raise ValueError("test metrics 缺少正式正负 sample role")
    for row in rows:
        record_count = _count(row["record_count"], positive=True)
        positive_count = _count(row["positive_count"])
        if positive_count > record_count:
            raise ValueError("test metrics 阳性计数超过记录数")
        if not math.isclose(
            _rate(row["positive_rate"]),
            positive_count / record_count,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("test metrics 阳性率与计数不一致")
        _finite(row["content_score_mean"])
        _rate(row["positive_rate_upper_95"])
        if not _same_float(row["target_fpr"], context["target_fpr"]):
            raise ValueError("test metrics target_fpr 不一致")
        if row["metric_status"] != "measured_image_only_detection":
            raise ValueError("test metrics 不是实测仅图像检测状态")
    return {"row_count": len(rows)}


def _validate_distribution(path: Path, context: Mapping[str, Any]) -> dict[str, Any]:
    """验证记录级连续分数分布表及冻结 operating point."""

    rows = _read_csv_exact(path, SCORE_DISTRIBUTION_FIELDNAMES)
    with path.open(encoding="utf-8-sig", newline="") as stream:
        if tuple(csv.DictReader(stream).fieldnames or ()) != SCORE_DISTRIBUTION_FIELDNAMES:
            raise ValueError("分数分布表列顺序不符合受治理 schema")
    overall_labels = {
        int(row["binary_label"])
        for row in rows
        if row["sample_scope"] == "test_overall"
    }
    if overall_labels != {0, 1}:
        raise ValueError("分数分布总体 scope 缺少正类或负类")
    for row in rows:
        label = int(row["binary_label"])
        if label not in {0, 1}:
            raise ValueError("binary_label 仅允许0或1")
        _finite(row["score"])
        _finite(row["raw_content_score"])
        if row["aligned_content_score"]:
            _finite(row["aligned_content_score"])
        if not _same_float(row["threshold"], context["content_threshold"]):
            raise ValueError("分数分布 operating point 阈值与冻结协议不一致")
        _validate_confusion_row(row)
        if row["threshold_digest"] != context["threshold_digest"]:
            raise ValueError("分数分布阈值摘要不一致")
        if row["metric_status"] != "measured_continuous_image_only_detection_score":
            raise ValueError("分数分布不是连续实测分数")
    return {"row_count": len(rows)}


def _validate_confusion_row(row: Mapping[str, Any]) -> None:
    """验证一行混淆矩阵,分母与三个比率完全一致."""

    tp = _count(row["tp"])
    fp = _count(row["fp"])
    tn = _count(row["tn"])
    fn = _count(row["fn"])
    sample_count = _count(row["sample_count"], positive=True)
    positive_count = _count(row["positive_count"], positive=True)
    negative_count = _count(row["negative_count"], positive=True)
    if tp + fn != positive_count or fp + tn != negative_count:
        raise ValueError("混淆矩阵与正负样本计数不一致")
    if positive_count + negative_count != sample_count:
        raise ValueError("正负样本计数与 sample_count 不一致")
    expected_rates = (
        ("tpr", tp / positive_count),
        ("fpr", fp / negative_count),
        ("fnr", fn / positive_count),
    )
    for field_name, expected in expected_rates:
        if not math.isclose(_rate(row[field_name]), expected, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"{field_name} 与混淆矩阵不一致")


def _curve_threshold(value: str) -> float:
    """解析允许在完整阈值 sweep 两端出现的无穷阈值."""

    resolved = float(value)
    if math.isnan(resolved):
        raise ValueError("threshold 不得为 NaN")
    return resolved


def _validate_curve(path: Path, context: Mapping[str, Any]) -> dict[str, Any]:
    """验证真实 sweep 的端点,唯一阈值,混淆矩阵与单调性."""

    rows = _read_curve_csv(path)
    grouped: dict[tuple[str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                row["sample_scope"],
                row["attack_family"],
                row["attack_name"],
                row["resource_profile"],
            )
        ].append(row)
    if ("test_overall", "all", "all", "all") not in grouped:
        raise ValueError("曲线缺少 test overall scope")
    for group in grouped.values():
        ordered = sorted(group, key=lambda row: int(row["curve_point_index"]))
        if [int(row["curve_point_index"]) for row in ordered] != list(range(len(ordered))):
            raise ValueError("curve_point_index 不连续")
        thresholds = [_curve_threshold(row["threshold"]) for row in ordered]
        if not (math.isinf(thresholds[0]) and thresholds[0] > 0):
            raise ValueError("曲线缺少正无穷端点")
        if not (math.isinf(thresholds[-1]) and thresholds[-1] < 0):
            raise ValueError("曲线缺少负无穷端点")
        if ordered[0]["threshold_kind"] != "positive_infinity_endpoint":
            raise ValueError("首个阈值类型错误")
        if ordered[-1]["threshold_kind"] != "negative_infinity_endpoint":
            raise ValueError("末个阈值类型错误")
        middle = thresholds[1:-1]
        if not middle or any(not math.isfinite(value) for value in middle):
            raise ValueError("曲线中间阈值必须是非空有限观测分数集合")
        if middle != sorted(set(middle), reverse=True):
            raise ValueError("观测阈值必须严格降序且不重复")
        for row in ordered:
            _validate_confusion_row(row)
            if row["threshold_digest"] != context["threshold_digest"]:
                raise ValueError("曲线阈值摘要不一致")
            if row["metric_status"] != "measured_complete_threshold_sweep":
                raise ValueError("曲线点不是完整实测 threshold sweep")
        if (int(ordered[0]["tp"]), int(ordered[0]["fp"])) != (0, 0):
            raise ValueError("正无穷端点必须预测全部阴性")
        if int(ordered[-1]["tp"]) != int(ordered[-1]["positive_count"]):
            raise ValueError("负无穷端点必须覆盖全部阳性")
        if int(ordered[-1]["fp"]) != int(ordered[-1]["negative_count"]):
            raise ValueError("负无穷端点必须覆盖全部阴性")
        tpr_values = [_rate(row["tpr"]) for row in ordered]
        fpr_values = [_rate(row["fpr"]) for row in ordered]
        fnr_values = [_rate(row["fnr"]) for row in ordered]
        if tpr_values != sorted(tpr_values) or fpr_values != sorted(fpr_values):
            raise ValueError("ROC 坐标不随阈值降低单调增加")
        if fnr_values != sorted(fnr_values, reverse=True):
            raise ValueError("DET 漏检率不随阈值降低单调下降")
    return {"row_count": len(rows), "rows": rows}


def _validate_attack_metrics(path: Path, context: Mapping[str, Any]) -> dict[str, Any]:
    """验证攻击级实测 TPR/FPR 与质量指标表."""

    rows = _read_csv_exact(path, ATTACK_METRIC_FIELDS)
    expected_attack_ids = {
        config.attack_id
        for config in default_attack_configs()
        if config.enabled
        and config.resource_profile in {"full_main", "full_extra"}
    }
    observed_attack_ids: set[str] = set()
    for row in rows:
        try:
            config = resolve_formal_attack_config(
                attack_family=row["attack_family"],
                attack_name=row["attack_name"],
                resource_profile=row["resource_profile"],
            )
        except ValueError as exc:
            raise ValueError("攻击指标包含未登记正式攻击") from exc
        if (
            row["attack_id"] != config.attack_id
            or row["attack_config_digest"] != attack_config_digest(config)
            or row["attack_id"] in observed_attack_ids
        ):
            raise ValueError("攻击指标身份或配置摘要不一致")
        observed_attack_ids.add(row["attack_id"])
        if row["metric_status"] != "measured_real_attacked_image_image_only_detection":
            raise ValueError("攻击指标不是来自真实 attacked image 的仅图像检测")
        positive_count = _count(row["positive_count"], positive=True)
        negative_count = _count(row["negative_count"], positive=True)
        if _count(row["attack_record_count"], positive=True) != positive_count + negative_count:
            raise ValueError("攻击记录数与正负样本数不一致")
        for field_name in (
            "true_positive_rate",
            "false_positive_rate",
            "clean_false_positive_rate",
            "attacked_false_positive_rate",
            "false_positive_rate_upper_95",
            "quality_score_mean",
            "quality_ssim_mean",
            "attacked_positive_source_to_attacked_ssim_mean",
            "geometry_reliable_rate",
            "rescue_rate",
        ):
            _rate(row[field_name])
        for field_name in (
            "quality_psnr_mean",
            "score_retention_mean",
            "lf_score_retention_mean",
            "tail_score_retention_mean",
        ):
            _finite(row[field_name])
        if not _same_float(row["target_fpr"], context["target_fpr"]):
            raise ValueError("攻击指标 target_fpr 不一致")
    if observed_attack_ids != expected_attack_ids:
        raise ValueError("攻击指标未精确覆盖全部正式攻击")
    return {"row_count": len(rows)}


def _validate_baseline_comparison(path: Path, context: Mapping[str, Any]) -> dict[str, Any]:
    """验证主方法与四个主表 baseline 的共同协议实测行."""

    rows = _read_csv_exact(path, BASELINE_COMPARISON_FIELDS)
    if len({row["method_id"] for row in rows}) != len(rows):
        raise ValueError("baseline comparison method_id 重复")
    proposed_rows = [row for row in rows if row["method_id"] == "slm_wm_current"]
    primary_rows = [row for row in rows if row["method_role"] == "external_baseline_primary"]
    if len(proposed_rows) != 1 or len(primary_rows) != 4:
        raise ValueError("baseline comparison 必须包含主方法与四个主表 baseline")
    for row in (*proposed_rows, *primary_rows):
        if row["metric_status"] == "unsupported":
            raise ValueError("主表方法不得为 unsupported")
        if not all(
            _as_bool(row[field_name])
            for field_name in (
                "common_prompt_protocol_ready",
                "common_attack_protocol_ready",
                "common_threshold_protocol_ready",
            )
        ):
            raise ValueError("主表方法未通过共同协议")
        for field_name in (
            "true_positive_rate",
            "false_positive_rate",
            "clean_false_positive_rate",
            "attacked_false_positive_rate",
            "quality_score_mean",
        ):
            _rate(row[field_name])
    if context["baseline_claim_ready"] and not all(
        _as_bool(row["supports_paper_claim"]) for row in (*proposed_rows, *primary_rows)
    ):
        raise ValueError("baseline ready 摘要与主表 claim 标记不一致")
    return {"row_count": len(rows)}


def _validate_ablation_metrics(path: Path, context: Mapping[str, Any]) -> dict[str, Any]:
    """验证唯一正式重运行消融集合的实测指标表."""

    rows = _read_csv_exact(path, ABLATION_METRIC_FIELDS)
    if tuple(row["ablation_id"] for row in rows) != FORMAL_RUNTIME_RERUN_ABLATION_IDS:
        raise ValueError("消融指标标识或顺序不符合唯一正式规范")
    for row in rows:
        _count(row["test_prompt_count"], positive=True)
        for field_name in (
            "clean_false_positive_rate",
            "wrong_key_false_positive_rate",
            "clean_true_positive_rate",
            "attacked_true_positive_rate",
            "attacked_false_positive_rate",
            "paired_ssim_mean",
        ):
            _rate(row[field_name])
        _finite(row["positive_content_score_mean"])
        if not row["frozen_threshold_digest"]:
            raise ValueError("消融冻结阈值摘要不得为空")
        if row["metric_status"] != "measured_full_runtime_rerun":
            raise ValueError("消融指标不是完整 runtime rerun 实测结果")
    return {"row_count": len(rows)}


def _validate_ablation_delta(path: Path, context: Mapping[str, Any]) -> dict[str, Any]:
    """验证全部非完整方法变体相对完整方法的真实消融差值."""

    rows = _read_csv_exact(path, ABLATION_DELTA_FIELDS)
    expected_ids = tuple(
        ablation_id
        for ablation_id in FORMAL_RUNTIME_RERUN_ABLATION_IDS
        if ablation_id != "complete_method"
    )
    if tuple(row["ablation_id"] for row in rows) != expected_ids:
        raise ValueError("消融 delta 标识或顺序不符合唯一正式规范")
    for row in rows:
        for field_name in (
            "clean_true_positive_rate_delta",
            "attacked_true_positive_rate_delta",
            "paired_ssim_delta",
        ):
            value = _finite(row[field_name])
            if not -1.0 <= value <= 1.0:
                raise ValueError("消融 delta 超出可解释范围")
        if row["metric_status"] != "measured_full_runtime_rerun":
            raise ValueError("消融 delta 不是完整 runtime rerun 实测结果")
    return {"row_count": len(rows)}


def _validate_ablation_necessity_statistics(
    path: Path,
    context: Mapping[str, Any],
) -> dict[str, Any]:
    """验证逐 Prompt 配对机制必要性统计及其受治理摘要绑定。"""

    rows = _read_csv_exact(path, ABLATION_NECESSITY_FIELDS)
    expected_ids = tuple(
        ablation_id
        for ablation_id in FORMAL_RUNTIME_RERUN_ABLATION_IDS
        if ablation_id != "complete_method"
    )
    if tuple(row["ablation_id"] for row in rows) != expected_ids:
        raise ValueError("机制必要性统计未精确覆盖全部正式变体")

    paired_prompt_counts: set[int] = set()
    for row in rows:
        if row["primary_metric_name"] != ABLATION_NECESSITY_PRIMARY_METRIC:
            raise ValueError("机制必要性统计主指标不符合预注册协议")
        if row["effect_direction"] != ABLATION_NECESSITY_EFFECT_DIRECTION:
            raise ValueError("机制必要性统计效应方向不符合预注册协议")
        if row["bootstrap_analysis_schema"] != ABLATION_NECESSITY_ANALYSIS_SCHEMA:
            raise ValueError("机制必要性统计 schema 不符合正式协议")
        if row["bootstrap_bit_generator"] != ABLATION_NECESSITY_BOOTSTRAP_BIT_GENERATOR:
            raise ValueError("机制必要性 bootstrap 随机生成器不符合协议")
        if row["bootstrap_quantile_method"] != ABLATION_NECESSITY_BOOTSTRAP_QUANTILE_METHOD:
            raise ValueError("机制必要性 bootstrap 分位数方法不符合协议")
        if row["paired_p_value_method"] != ABLATION_NECESSITY_P_VALUE_METHOD:
            raise ValueError("机制必要性配对检验方法不符合协议")

        paired_prompt_counts.add(_count(row["paired_prompt_count"], positive=True))
        _count(row["bootstrap_resample_count"], positive=True)
        mean_effect = _finite(row["mean_paired_effect"])
        ci_low = _finite(row["mean_paired_effect_ci_low"])
        ci_high = _finite(row["mean_paired_effect_ci_high"])
        minimum_effect = _finite(row["minimum_effect_size"])
        confidence_level = _finite(row["confidence_level"])
        significance_alpha = _finite(row["significance_alpha"])
        p_value = _finite(row["one_sided_paired_p_value"])
        adjusted_p_value = _finite(row["holm_adjusted_p_value"])
        clean_effect = _finite(row["clean_true_positive_mean_paired_effect"])
        clean_ci_low = _finite(
            row["clean_true_positive_mean_paired_effect_ci_low"]
        )
        clean_ci_high = _finite(
            row["clean_true_positive_mean_paired_effect_ci_high"]
        )
        ssim_effect = _finite(row["paired_ssim_mean_paired_effect"])
        ssim_ci_low = _finite(
            row["paired_ssim_mean_paired_effect_ci_low"]
        )
        ssim_ci_high = _finite(
            row["paired_ssim_mean_paired_effect_ci_high"]
        )
        ssim_margin = _finite(row["paired_ssim_noninferiority_margin"])
        if not -1.0 <= ci_low <= mean_effect <= ci_high <= 1.0:
            raise ValueError("机制必要性效应或置信区间超出有效范围")
        if abs(minimum_effect - ABLATION_NECESSITY_MINIMUM_EFFECT_SIZE) > 1e-12:
            raise ValueError("机制必要性最小效应阈值与预注册协议不一致")
        if not 0.0 < confidence_level < 1.0:
            raise ValueError("机制必要性置信水平无效")
        if abs(significance_alpha - ABLATION_NECESSITY_ALPHA) > 1e-12:
            raise ValueError("机制必要性显著性水平与预注册协议不一致")
        if not 0.0 <= p_value <= adjusted_p_value <= 1.0:
            raise ValueError("机制必要性 p 值或 Holm 校正结果无效")
        if not -1.0 <= clean_ci_low <= clean_effect <= clean_ci_high <= 1.0:
            raise ValueError("clean TPR 配对诊断或置信区间无效")
        if not -1.0 <= ssim_ci_low <= ssim_effect <= ssim_ci_high <= 1.0:
            raise ValueError("paired SSIM 配对诊断或置信区间无效")
        if abs(
            ssim_margin
            - ABLATION_NECESSITY_PAIRED_SSIM_NONINFERIORITY_MARGIN
        ) > 1e-12:
            raise ValueError("paired SSIM 非劣界与预注册协议不一致")
        if _as_bool(row["paired_ssim_noninferiority_ready"]) != (
            ssim_ci_low >= -ssim_margin
        ):
            raise ValueError("paired SSIM 非劣判定与置信区间不一致")

        expected_flags = {
            "effect_direction_ready": mean_effect > 0.0,
            "minimum_effect_ready": mean_effect >= minimum_effect,
            "confidence_interval_ready": ci_low > minimum_effect,
            "adjusted_significance_ready": adjusted_p_value < significance_alpha,
        }
        if any(_as_bool(row[name]) != ready for name, ready in expected_flags.items()):
            raise ValueError("机制必要性统计判定标记与数值不一致")
        supported = all(expected_flags.values()) and _as_bool(
            row["paired_ssim_noninferiority_ready"]
        )
        expected_decision = (
            "measured_supported" if supported else "measured_not_supported"
        )
        if (
            _as_bool(row["necessity_component_supported"]) != supported
            or _as_bool(row["supports_paper_claim"]) != supported
            or row["necessity_component_decision"] != expected_decision
        ):
            raise ValueError(
                "机制必要性主张结论未同时满足统计条件与 paired SSIM 质量非劣门禁"
            )
        for digest_field in (
            "bootstrap_seed_digest_random",
            "paired_prompt_id_digest",
            "input_record_digest",
        ):
            digest = str(row[digest_field])
            if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
                raise ValueError("机制必要性统计摘要字段不是 SHA-256")

    summary = context["ablation_component_summary"]
    supported_ids = [
        row["ablation_id"] for row in rows if _as_bool(row["necessity_component_supported"])
    ]
    not_supported_ids = [
        row["ablation_id"] for row in rows if not _as_bool(row["necessity_component_supported"])
    ]
    if not all(
        (
            _as_bool(summary.get("ablation_necessity_statistics_ready")),
            summary.get("necessity_statistic_row_count") == len(expected_ids),
            len(paired_prompt_counts) == 1,
            summary.get("paired_prompt_count") == next(
                iter(paired_prompt_counts),
                -1,
            ),
            summary.get("expected_paired_prompt_count")
            == summary.get("paired_prompt_count"),
            summary.get("necessity_statistic_rows_digest")
            == build_stable_digest(canonicalize_ablation_necessity_rows(rows)),
            summary.get("necessity_component_supported_ablation_ids") == supported_ids,
            summary.get("necessity_component_not_supported_ablation_ids") == not_supported_ids,
            _as_bool(summary.get("all_mechanism_necessity_components_supported"))
            == (not not_supported_ids),
        )
    ):
        raise ValueError("机制必要性统计与受治理摘要不一致")
    return {"row_count": len(rows), "rows": rows}


def _validate_dataset_quality(path: Path, context: Mapping[str, Any]) -> dict[str, Any]:
    """验证 FID 与 KID mean/std 三行正式 Inception 质量证据."""

    rows = _read_csv_exact(path, DATASET_QUALITY_FIELDS)
    if [row["quality_metric_name"] for row in rows] != list(
        FORMAL_DATASET_QUALITY_METRIC_NAMES
    ):
        raise ValueError("数据集质量表必须恰好按 FID, KID mean, KID std 三行排列")
    counts = set()
    for row in rows:
        if row["metric_status"] != "measured":
            raise ValueError("FID/KID 必须是 measured 状态")
        if row["paper_metric_name"] != row["quality_metric_name"]:
            raise ValueError("FID/KID 论文指标名称与机器指标名称不一致")
        value = _finite(row["quality_metric_value"])
        if row["quality_metric_name"] in {"fid", "kid_std"} and value < 0.0:
            raise ValueError("FID 与 KID 子集标准差不得为负数")
        if not row["feature_backend"]:
            raise ValueError("FID/KID feature_backend 不得为空")
        row_counts = tuple(
            _count(row[field_name], positive=True)
            for field_name in (
                "source_image_count",
                "comparison_image_count",
                "sample_pair_count",
            )
        )
        if len(set(row_counts)) != 1:
            raise ValueError("FID/KID 图像数量与配对数不一致")
        counts.add(row_counts)
    if len(counts) != 1:
        raise ValueError("FID 与 KID mean/std 使用的样本集合不一致")
    return {"row_count": len(rows)}


def _relative_or_absolute(path: Path, root_path: Path) -> str:
    """把输入路径尽量记录为仓库相对路径."""

    try:
        return path.resolve().relative_to(root_path.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _sha256(path: Path) -> str:
    """计算实际读取文件的字节级 SHA-256."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_check(
    check_id: str,
    path: Path,
    validator: Callable[[Path, Mapping[str, Any]], dict[str, Any]],
    context: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """执行单项数据验证并把异常收敛为 fail-closed 检查记录."""

    try:
        detail = validator(path, context)
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        return (
            {
                "check_id": check_id,
                "data_ready": False,
                "row_count": 0,
                "issues": [str(exc)],
            },
            None,
        )
    return (
        {
            "check_id": check_id,
            "data_ready": True,
            "row_count": int(detail.get("row_count", 0)),
            "issues": [],
        },
        detail,
    )


def validate_paper_artifact_source_data(
    *,
    root_path: Path,
    source_paths: Mapping[str, Path],
    threshold_report: Mapping[str, Any],
    attack_manifest: Mapping[str, Any],
    baseline_runtime_report: Mapping[str, Any],
    dataset_quality_summary: Mapping[str, Any],
    ablation_component_summary: Mapping[str, Any],
) -> dict[str, Any]:
    """读取并验证论文表图所依赖的12类实际数据文件.

    返回值保留每个阻断原因和全部已存在输入文件的字节摘要.文件缺失时不抛出
    到 writer 外层, 而是让对应 table/figure readiness 明确进入 blocked.
    """

    target_fpr = threshold_report.get("target_fpr")
    threshold_digest = str(threshold_report.get("frozen_threshold_digest", ""))
    context: dict[str, Any] = {
        "target_fpr": target_fpr,
        "threshold_digest": threshold_digest,
        "baseline_claim_ready": _as_bool(
            baseline_runtime_report.get("comparison_table_supports_paper_claim", False)
        ),
        "ablation_component_summary": dict(ablation_component_summary),
    }
    checks: dict[str, dict[str, Any]] = {}
    details: dict[str, dict[str, Any] | None] = {}
    validators = {
        "frozen_evidence_protocol_ready": _validate_frozen_protocol,
        "raw_image_only_detection_records_ready": _validate_raw_detection_records,
        "test_detection_metrics_ready": _validate_test_metrics,
        "score_distribution_table_ready": _validate_distribution,
        "roc_curve_points_ready": _validate_curve,
        "det_curve_points_ready": _validate_curve,
        "attack_family_metrics_ready": _validate_attack_metrics,
        "baseline_comparison_table_ready": _validate_baseline_comparison,
        "mechanism_ablation_metrics_ready": _validate_ablation_metrics,
        "mechanism_pairwise_delta_ready": _validate_ablation_delta,
        "mechanism_necessity_statistics_ready": (
            _validate_ablation_necessity_statistics
        ),
        "dataset_quality_metrics_ready": _validate_dataset_quality,
    }

    protocol_path = source_paths["frozen_evidence_protocol_ready"]
    protocol_check, protocol_detail = _run_check(
        "frozen_evidence_protocol_ready",
        protocol_path,
        _validate_frozen_protocol,
        context,
    )
    checks["frozen_evidence_protocol_ready"] = protocol_check
    details["frozen_evidence_protocol_ready"] = protocol_detail
    if protocol_detail is not None:
        frozen_protocol = protocol_detail["protocol"]
        context["content_threshold"] = frozen_protocol["content_threshold"]
        context["frozen_protocol"] = frozen_protocol
    else:
        context["content_threshold"] = math.nan
        context["frozen_protocol"] = None

    for check_id, validator in validators.items():
        if check_id == "frozen_evidence_protocol_ready":
            continue
        check, detail = _run_check(
            check_id,
            source_paths[check_id],
            validator,
            context,
        )
        checks[check_id] = check
        details[check_id] = detail

    raw_detail = details.get("raw_image_only_detection_records_ready")
    rebuilt_table_specs = (
        (
            "test_detection_metrics_ready",
            "test_detection_metrics",
            tuple(sorted(TEST_METRIC_FIELDS)),
        ),
        (
            "score_distribution_table_ready",
            "score_distribution_table",
            SCORE_DISTRIBUTION_FIELDNAMES,
        ),
        ("roc_curve_points_ready", "roc_curve_points", CURVE_POINT_FIELDNAMES),
        ("det_curve_points_ready", "det_curve_points", CURVE_POINT_FIELDNAMES),
    )
    for check_id, table_name, fieldnames in rebuilt_table_specs:
        if raw_detail is None:
            checks[check_id] = {
                **checks[check_id],
                "data_ready": False,
                "issues": [
                    *checks[check_id]["issues"],
                    "原始仅图像检测记录无效, 无法执行精确重建比较",
                ],
            }
            continue
        try:
            _validate_rebuilt_csv_exact(
                source_paths[check_id],
                raw_detail["rebuilt_tables"][table_name],
                fieldnames,
            )
        except (OSError, ValueError, TypeError, KeyError) as exc:
            checks[check_id] = {
                **checks[check_id],
                "data_ready": False,
                "issues": [*checks[check_id]["issues"], str(exc)],
            }

    roc_detail = details.get("roc_curve_points_ready")
    det_detail = details.get("det_curve_points_ready")
    if roc_detail is not None and det_detail is not None and roc_detail["rows"] != det_detail["rows"]:
        checks["det_curve_points_ready"] = {
            **checks["det_curve_points_ready"],
            "data_ready": False,
            "issues": [
                *checks["det_curve_points_ready"]["issues"],
                "ROC 与 DET 没有共享同一完整 threshold sweep",
            ],
        }

    implications = (
        (
            _as_bool(threshold_report.get("full_method_component_ready", False)),
            (
                "frozen_evidence_protocol_ready",
                "raw_image_only_detection_records_ready",
                "test_detection_metrics_ready",
                "score_distribution_table_ready",
                "roc_curve_points_ready",
                "det_curve_points_ready",
            ),
            "主方法 ready 但连续检测表图数据不完整",
        ),
        (
            _as_bool(attack_manifest.get("attack_metrics_ready", False)),
            ("attack_family_metrics_ready",),
            "攻击 ready 但攻击指标表无效",
        ),
        (
            _as_bool(baseline_runtime_report.get("comparison_table_supports_paper_claim", False)),
            ("baseline_comparison_table_ready",),
            "baseline ready 但共同协议对比表无效",
        ),
        (
            _as_bool(ablation_component_summary.get("ablation_component_ready", False)),
            (
                "mechanism_ablation_metrics_ready",
                "mechanism_pairwise_delta_ready",
                "mechanism_necessity_statistics_ready",
            ),
            "消融 ready 但实测指标, delta 或必要性统计表无效",
        ),
        (
            _as_bool(dataset_quality_summary.get("formal_fid_kid_component_ready", False)),
            ("dataset_quality_metrics_ready",),
            "数据集质量 ready 但 FID/KID 表无效",
        ),
    )
    consistency_issues = [
        issue
        for ready_flag, required_checks, issue in implications
        if ready_flag and not all(checks[check_id]["data_ready"] for check_id in required_checks)
    ]
    checks["ready_flag_consistency_ready"] = {
        "check_id": "ready_flag_consistency_ready",
        "data_ready": not consistency_issues,
        "row_count": 1,
        "issues": consistency_issues,
    }

    normalized_source_paths = {
        check_id: _relative_or_absolute(path, root_path)
        for check_id, path in source_paths.items()
    }
    evidence_source_file_sha256 = {
        normalized_source_paths[check_id]: _sha256(path)
        for check_id, path in source_paths.items()
        if path.is_file()
    }
    raw_detection_source_path = normalized_source_paths[
        "raw_image_only_detection_records_ready"
    ]
    blocked_ids = [
        check_id
        for check_id, check in checks.items()
        if not check["data_ready"]
    ]
    return {
        **{check_id: bool(check["data_ready"]) for check_id, check in checks.items()},
        "artifact_data_validation_ready": not blocked_ids,
        "artifact_data_check_count": len(checks),
        "blocked_artifact_data_count": len(blocked_ids),
        "blocked_artifact_data_ids": blocked_ids,
        "source_paths": normalized_source_paths,
        "evidence_source_file_sha256": evidence_source_file_sha256,
        "raw_image_only_detection_records_sha256": evidence_source_file_sha256.get(
            raw_detection_source_path,
            "",
        ),
        "checks": checks,
        "supports_paper_claim": False,
    }
