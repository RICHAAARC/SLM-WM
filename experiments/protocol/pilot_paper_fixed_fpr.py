"""构建论文运行级 fixed-FPR 共同协议。

该模块描述 probe_paper、pilot_paper 与 full_paper 共用的正式验证协议,
包括受治理输入、阈值边界、攻击矩阵、baseline 导入模板和声明边界。
它不执行 GPU 推理。三个论文运行层级仅通过 Prompt 规模和统计强度区分。
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

from experiments.protocol.attacks import (
    AttackConfig,
    attack_config_digest,
    resolve_formal_attack_config,
)
from experiments.protocol.formal_evidence import contains_nonformal_marker
from experiments.protocol.formal_randomization import (
    formal_randomization_protocol_record,
    resolve_formal_randomization_repeat,
)
from experiments.protocol.paper_run_config import (
    FULL_PAPER_RUN_NAME,
    PaperRunConfig,
    PROBE_PAPER_RUN_NAME,
    RUN_DEFAULTS,
    RUN_EXPECTED_PROMPT_COUNTS,
    build_paper_run_config,
)
from experiments.protocol.prompts import PromptProtocolRecord
from experiments.protocol.splits import apply_split_assignments, build_group_split_counts
from main.core.digest import build_stable_digest

PILOT_PAPER_PROMPT_SET = "pilot_paper"
PILOT_PAPER_PROMPT_FILE = "configs/paper_main_pilot_paper_prompts.txt"
PILOT_PAPER_PROMPT_PROTOCOL_NAME = "paper_main_pilot_paper_prompt_protocol"
PILOT_PAPER_RESULT_PROTOCOL_NAME = "pilot_paper_fixed_fpr_common_protocol"
PILOT_PAPER_RESULT_SCOPE = "pilot_paper_common_protocol"
PILOT_PAPER_CLAIM_BOUNDARY = "pilot_claim"
FULL_PAPER_CLAIM_BOUNDARY = "full_paper_claim_requires_full_paper_sample_scale"
PROBE_PAPER_WORKFLOW_BOUNDARY = (
    "probe_paper_uses_same_formal_protocol_with_smaller_sample_only"
)
PAPER_RUN_CLAIM_SCOPES = {
    PROBE_PAPER_RUN_NAME: "probe_claim",
    PILOT_PAPER_PROMPT_SET: "pilot_claim",
    FULL_PAPER_RUN_NAME: "full_claim",
}
PROBE_PAPER_FIXED_FPR = float(RUN_DEFAULTS[PROBE_PAPER_RUN_NAME]["target_fpr"])
PILOT_PAPER_FIXED_FPR = float(RUN_DEFAULTS[PILOT_PAPER_PROMPT_SET]["target_fpr"])
FULL_PAPER_FIXED_FPR = float(RUN_DEFAULTS[FULL_PAPER_RUN_NAME]["target_fpr"])
PAPER_RUN_FIXED_FPR = {
    PROBE_PAPER_RUN_NAME: PROBE_PAPER_FIXED_FPR,
    PILOT_PAPER_PROMPT_SET: PILOT_PAPER_FIXED_FPR,
    FULL_PAPER_RUN_NAME: FULL_PAPER_FIXED_FPR,
}
PILOT_PAPER_CONFIDENCE_INTERVAL_METHOD = "bounded_hoeffding"
PILOT_PAPER_CONFIDENCE_LEVEL = 0.95
PILOT_PAPER_PAIRED_BOOTSTRAP_RESAMPLE_COUNT = 100_000
PILOT_PAPER_PAIRED_BOOTSTRAP_ANALYSIS_SCHEMA = "paired_prompt_cluster_bootstrap_v1"
PILOT_PAPER_PAIRED_BOOTSTRAP_BIT_GENERATOR = "PCG64"
PILOT_PAPER_PAIRED_BOOTSTRAP_QUANTILE_METHOD = "linear"
PILOT_PAPER_PAIRED_CLAIM_P_VALUE_METHOD = "bounded_hoeffding_prompt_cluster_mean"
PILOT_PAPER_PAIRED_SHARP_NULL_DIAGNOSTIC_METHOD = (
    "exact_prompt_cluster_sign_flip_dp"
)
PILOT_PAPER_MINIMUM_CLEAN_NEGATIVE_COUNT = 340
PILOT_PAPER_METHOD_IDS = ("slm_wm_current", "tree_ring", "gaussian_shading", "shallow_diffuse", "t2smark")
PILOT_PAPER_PRIMARY_METHOD_ID = "slm_wm_current"
PILOT_PAPER_PRIMARY_BASELINE_IDS = ("tree_ring", "gaussian_shading", "shallow_diffuse", "t2smark")
PILOT_PAPER_ATTACK_RESOURCE_PROFILES = ("full_main", "full_extra")

PILOT_PAPER_REQUIRED_METRIC_FIELDS = (
    "positive_count",
    "negative_count",
    "attacked_negative_count",
    "attack_record_count",
    "supported_record_count",
    "true_positive_rate",
    "true_positive_rate_ci_low",
    "true_positive_rate_ci_high",
    "false_positive_rate",
    "false_positive_rate_ci_low",
    "false_positive_rate_ci_high",
    "clean_false_positive_rate",
    "clean_false_positive_rate_ci_low",
    "clean_false_positive_rate_ci_high",
    "attacked_false_positive_rate",
    "attacked_false_positive_rate_ci_low",
    "attacked_false_positive_rate_ci_high",
    "quality_score_mean",
    "quality_score_ci_low",
    "quality_score_ci_high",
    "score_retention_mean",
    "score_retention_ci_low",
    "score_retention_ci_high",
)
PILOT_PAPER_REQUIRED_SOURCE_FIELDS = (
    "attack_id",
    "attack_config_digest",
    "result_protocol_name",
    "result_scope",
    "result_claim_scope",
    "method_id",
    "prompt_protocol_name",
    "prompt_split_digest",
    "attack_matrix_digest",
    "fixed_fpr_protocol_digest",
    "method_threshold_digest",
    "baseline_result_source",
    "baseline_result_source_digest",
    "evidence_paths",
)
PILOT_PAPER_RATE_FIELDS = (
    "true_positive_rate",
    "false_positive_rate",
    "clean_false_positive_rate",
    "attacked_false_positive_rate",
)
PILOT_PAPER_METRIC_BOUNDS = {
    **{field_name: (0.0, 1.0) for field_name in PILOT_PAPER_RATE_FIELDS},
    "quality_score_mean": (-1.0, 1.0),
    "score_retention_mean": (0.0, 1.0),
}
PILOT_PAPER_CI_FIELD_GROUPS = (
    ("true_positive_rate_ci_low", "true_positive_rate", "true_positive_rate_ci_high"),
    ("false_positive_rate_ci_low", "false_positive_rate", "false_positive_rate_ci_high"),
    ("clean_false_positive_rate_ci_low", "clean_false_positive_rate", "clean_false_positive_rate_ci_high"),
    ("attacked_false_positive_rate_ci_low", "attacked_false_positive_rate", "attacked_false_positive_rate_ci_high"),
    ("quality_score_ci_low", "quality_score_mean", "quality_score_ci_high"),
    (
        "score_retention_ci_low",
        "score_retention_mean",
        "score_retention_ci_high",
    ),
)
PILOT_PAPER_CI_COUNT_FIELDS = {
    "true_positive_rate": "positive_count",
    "false_positive_rate": "negative_count",
    "clean_false_positive_rate": "negative_count",
    "attacked_false_positive_rate": "attacked_negative_count",
    "quality_score_mean": "positive_count",
    "score_retention_mean": "positive_count",
}


def clamp_unit_interval(value: float) -> float:
    """把有限指标值裁剪到 [0, 1] 区间."""

    resolved = float(value)
    if not math.isfinite(resolved):
        raise ValueError("指标值必须是有限数值")
    return min(1.0, max(0.0, resolved))


def bounded_metric_value(
    value: float,
    *,
    lower_bound: float,
    upper_bound: float,
) -> float:
    """读取显式有界的有限指标值, 不对越界观测执行静默裁剪."""

    if any(isinstance(item, bool) for item in (value, lower_bound, upper_bound)):
        raise ValueError("指标边界和值不得使用布尔值")
    resolved_value = float(value)
    resolved_lower = float(lower_bound)
    resolved_upper = float(upper_bound)
    if not all(
        math.isfinite(item)
        for item in (resolved_value, resolved_lower, resolved_upper)
    ):
        raise ValueError("指标边界和值必须是有限数值")
    if resolved_lower >= resolved_upper:
        raise ValueError("指标下界必须小于上界")
    if not resolved_lower <= resolved_value <= resolved_upper:
        raise ValueError("指标值超出显式有界范围")
    return resolved_value


def bounded_hoeffding_confidence_interval(
    value: float,
    count: int,
    confidence_level: float,
    *,
    lower_bound: float = 0.0,
    upper_bound: float = 1.0,
) -> tuple[float, float]:
    """对任意显式有界独立观测均值构造分布无关 Hoeffding 区间."""

    resolved_value = bounded_metric_value(
        value,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
    )
    resolved_lower = float(lower_bound)
    resolved_upper = float(upper_bound)
    sample_count = int(count)
    if sample_count <= 0:
        raise ValueError("置信区间样本数必须为正整数")
    alpha = 1.0 - float(confidence_level)
    if not 0.0 < alpha < 1.0:
        raise ValueError("confidence_level 必须位于 (0, 1)")
    range_width = resolved_upper - resolved_lower
    margin = range_width * math.sqrt(
        math.log(2.0 / alpha) / (2.0 * sample_count)
    )
    return (
        max(resolved_lower, resolved_value - margin),
        min(resolved_upper, resolved_value + margin),
    )


def canonical_pilot_paper_result_records(
    rows: Iterable[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    """把论文结果记录规范化为与输入文件行序无关的稳定顺序.

    这一实现属于通用证据治理写法.排序键先使用论文模板身份, 再使用整行
    稳定摘要消除同键记录的输入顺序影响; 是否存在重复模板键仍由正式 schema
    validator 独立判定, 因而摘要计算不会掩盖重复记录.
    """

    materialized = [dict(row) for row in rows]
    return tuple(
        sorted(
            materialized,
            key=lambda row: (
                str(row.get("method_id", "")),
                str(row.get("resource_profile", "")),
                str(row.get("attack_family", "")),
                str(row.get("attack_name", "")),
                build_stable_digest(row),
            ),
        )
    )


def build_pilot_paper_result_record_set_digest(
    rows: Iterable[Mapping[str, Any]],
) -> str:
    """计算正式结果记录稳定有序集合的 SHA-256 摘要."""

    return build_stable_digest(list(canonical_pilot_paper_result_records(rows)))


def build_pilot_paper_result_records_manifest_config(
    *,
    result_records: Iterable[Mapping[str, Any]],
    method_threshold_digest_map: Mapping[str, str],
    randomization_aggregate_digest: str,
    common_code_version: str,
    validation_report: Mapping[str, Any],
    template_coverage_rows: Iterable[Mapping[str, Any]],
    summary: Mapping[str, Any],
    require_existing_evidence: bool,
) -> dict[str, Any]:
    """构造正式 result records manifest 的唯一配置摘要载荷.

    该函数属于通用证据治理写法.生产器与最终门禁共同调用该函数,
    使 manifest 的 config digest 能由落盘 records、报告和覆盖表精确重建.
    """

    result_record_set_digest = build_pilot_paper_result_record_set_digest(
        result_records
    )
    return {
        "result_record_digest": result_record_set_digest,
        "result_record_set_digest": result_record_set_digest,
        "method_threshold_digest_map": dict(
            sorted(
                (str(method_id), str(digest))
                for method_id, digest in method_threshold_digest_map.items()
            )
        ),
        "randomization_aggregate_digest": str(randomization_aggregate_digest),
        "common_code_version": str(common_code_version),
        "validation_report_digest": build_stable_digest(dict(validation_report)),
        "template_coverage_digest": build_stable_digest(
            [dict(row) for row in template_coverage_rows]
        ),
        "summary_digest": build_stable_digest(dict(summary)),
        "require_existing_evidence": bool(require_existing_evidence),
        "image_only_runtime_used": True,
    }


def prompt_protocol_name_for_run(run_name: str) -> str:
    """根据论文运行层级生成 prompt 协议名称。"""

    return f"paper_main_{run_name}_prompt_protocol"


def result_protocol_name_for_run(run_name: str) -> str:
    """根据论文运行层级生成 fixed-FPR 共同协议名称。"""

    return f"{run_name}_fixed_fpr_common_protocol"


def result_scope_for_run(run_name: str) -> str:
    """根据论文运行层级生成结果范围名称。"""

    return f"{run_name}_common_protocol"


def result_claim_scope_for_run(run_name: str) -> str:
    """根据论文运行层级生成论文主张边界名称。"""

    return PAPER_RUN_CLAIM_SCOPES[run_name]


def _default_paper_run_value(field_name: str) -> Any:
    """从统一论文运行配置读取 dataclass 默认值。"""

    return getattr(build_paper_run_config("."), field_name)


@dataclass(frozen=True)
class PilotPaperFixedFprConfig:
    """集中描述论文运行级 fixed-FPR 共同协议配置。

    该对象属于通用工程写法: 把 prompt set、固定 FPR、置信区间方法和
    论文声明边界集中在 dataclass 构造层, 业务函数只消费已经归一化的配置。
    """

    paper_run_name: str = field(default_factory=lambda: _default_paper_run_value("run_name"))
    prompt_set: str = field(default_factory=lambda: _default_paper_run_value("prompt_set"))
    prompt_file: str = field(default_factory=lambda: _default_paper_run_value("prompt_file"))
    prompt_protocol_name: str = field(
        default_factory=lambda: prompt_protocol_name_for_run(_default_paper_run_value("run_name"))
    )
    result_protocol_name: str = field(
        default_factory=lambda: result_protocol_name_for_run(_default_paper_run_value("run_name"))
    )
    result_scope: str = field(default_factory=lambda: result_scope_for_run(_default_paper_run_value("run_name")))
    result_claim_scope: str = field(default_factory=lambda: result_claim_scope_for_run(_default_paper_run_value("run_name")))
    target_fpr: float = field(default_factory=lambda: float(_default_paper_run_value("target_fpr")))
    confidence_interval_method: str = PILOT_PAPER_CONFIDENCE_INTERVAL_METHOD
    confidence_level: float = PILOT_PAPER_CONFIDENCE_LEVEL
    minimum_clean_negative_count: int = field(
        default_factory=lambda: int(_default_paper_run_value("minimum_clean_negative_count"))
    )
    randomization_repeat_id: str = field(
        default_factory=lambda: str(
            _default_paper_run_value("randomization_repeat_id")
        )
    )
    generation_seed_index: int = field(
        default_factory=lambda: int(
            _default_paper_run_value("generation_seed_index")
        )
    )
    generation_seed_offset: int = field(
        default_factory=lambda: int(
            _default_paper_run_value("generation_seed_offset")
        )
    )
    watermark_key_index: int = field(
        default_factory=lambda: int(
            _default_paper_run_value("watermark_key_index")
        )
    )
    formal_randomization_protocol_digest: str = field(
        default_factory=lambda: str(
            _default_paper_run_value(
                "formal_randomization_protocol_digest"
            )
        )
    )
    attack_resource_profiles: tuple[str, ...] = PILOT_PAPER_ATTACK_RESOURCE_PROFILES

    def __post_init__(self) -> None:
        """集中校验不可恢复的协议边界。"""

        if self.prompt_set not in RUN_DEFAULTS:
            raise ValueError(f"未知论文运行 prompt set: {self.prompt_set}")
        if self.paper_run_name not in RUN_DEFAULTS:
            raise ValueError(f"未知论文运行层级: {self.paper_run_name}")
        if not self.prompt_file:
            raise ValueError("prompt_file 不得为空")
        if not 0.0 < self.target_fpr < 1.0:
            raise ValueError("target_fpr 必须位于 (0, 1)")
        if self.confidence_interval_method != PILOT_PAPER_CONFIDENCE_INTERVAL_METHOD:
            raise ValueError("confidence_interval_method 必须为 bounded_hoeffding")
        if not 0.0 < self.confidence_level < 1.0:
            raise ValueError("confidence_level 必须位于 (0, 1)")
        if self.minimum_clean_negative_count <= 0:
            raise ValueError("minimum_clean_negative_count 必须为正整数")
        if not self.randomization_repeat_id:
            raise ValueError("fixed-FPR 配置必须显式携带活动 repeat ID")
        repeat = resolve_formal_randomization_repeat(
            self.randomization_repeat_id
        )
        if (
            self.generation_seed_index != repeat.generation_seed_index
            or self.generation_seed_offset != repeat.generation_seed_offset
            or self.watermark_key_index != repeat.watermark_key_index
            or self.formal_randomization_protocol_digest
            != formal_randomization_protocol_record()[
                "formal_randomization_protocol_digest"
            ]
        ):
            raise ValueError("fixed-FPR 配置的活动 repeat 身份不匹配")

    def to_dict(self) -> dict[str, Any]:
        """转为 JSON 兼容字典。"""

        payload = asdict(self)
        payload["attack_resource_profiles"] = list(self.attack_resource_profiles)
        return payload


def build_paper_fixed_fpr_config_from_paper_run(
    paper_run: PaperRunConfig,
) -> PilotPaperFixedFprConfig:
    """复用已验证论文运行配置构造 fixed-FPR 共同协议配置."""

    return PilotPaperFixedFprConfig(
        paper_run_name=paper_run.run_name,
        prompt_set=paper_run.prompt_set,
        prompt_file=paper_run.prompt_file,
        prompt_protocol_name=prompt_protocol_name_for_run(paper_run.run_name),
        result_protocol_name=result_protocol_name_for_run(paper_run.run_name),
        result_scope=result_scope_for_run(paper_run.run_name),
        result_claim_scope=result_claim_scope_for_run(paper_run.run_name),
        target_fpr=paper_run.target_fpr,
        minimum_clean_negative_count=paper_run.minimum_clean_negative_count,
        randomization_repeat_id=paper_run.randomization_repeat_id,
        generation_seed_index=paper_run.generation_seed_index,
        generation_seed_offset=paper_run.generation_seed_offset,
        watermark_key_index=paper_run.watermark_key_index,
        formal_randomization_protocol_digest=(
            paper_run.formal_randomization_protocol_digest
        ),
    )


def build_paper_fixed_fpr_config(root: str | Path = ".") -> PilotPaperFixedFprConfig:
    """按统一论文运行配置构造 fixed-FPR 共同协议配置。"""

    return build_paper_fixed_fpr_config_from_paper_run(
        build_paper_run_config(root)
    )


@dataclass(frozen=True)
class PilotPaperImportIssue:
    """记录 pilot_paper 结果导入 schema 校验中的单个问题。"""

    row_index: int
    method_id: str
    field_name: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        """转为 JSON 兼容字典。"""

        return asdict(self)


def _str_field(row: Mapping[str, Any], field_name: str) -> str:
    """读取字符串字段, 缺失时返回空字符串。"""

    return str(row.get(field_name, "") or "")


def _bool_field(row: Mapping[str, Any], field_name: str) -> bool:
    """读取布尔字段, 兼容 JSON 与 CSV 文本表示。"""

    value = row.get(field_name)
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _int_field(row: Mapping[str, Any], field_name: str) -> int:
    """读取非负计数字段。"""

    return int(float(row.get(field_name, 0) or 0))


def _float_field(row: Mapping[str, Any], field_name: str) -> float:
    """读取有限浮点字段。"""

    value = float(row.get(field_name, 0.0) or 0.0)
    if value != value or value in {float("inf"), float("-inf")}:
        raise ValueError(f"{field_name} 必须是有限数值")
    return value


def _list_field(row: Mapping[str, Any], field_name: str) -> tuple[str, ...]:
    """读取列表字段, 兼容列表、元组和分号分隔文本。"""

    value = row.get(field_name, ())
    if isinstance(value, list | tuple):
        return tuple(str(item) for item in value if str(item).strip())
    if isinstance(value, str):
        return tuple(part.strip() for part in value.split(";") if part.strip())
    return ()


def _issue(row_index: int, row: Mapping[str, Any], field_name: str, reason: str) -> PilotPaperImportIssue:
    """构造统一 schema issue。"""

    return PilotPaperImportIssue(
        row_index=row_index,
        method_id=_str_field(row, "method_id"),
        field_name=field_name,
        reason=reason,
    )


def _relative_or_absolute(path: Path, root_path: Path) -> str:
    """优先记录相对仓库根目录的路径。"""

    try:
        return path.resolve().relative_to(root_path.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _stable_records_for_prompt_split(records: Iterable[PromptProtocolRecord]) -> list[dict[str, Any]]:
    """提取生成 prompt split 摘要所需的稳定字段。"""

    return [
        {
            "prompt_id": record.prompt_id,
            "prompt_digest": record.prompt_digest,
            "prompt_set": record.prompt_set,
            "risk_profile": record.risk_profile,
            "split": record.split,
        }
        for record in sorted(records, key=lambda item: item.prompt_id)
    ]


def build_pilot_paper_prompt_split_summary(
    prompt_records: Iterable[PromptProtocolRecord],
    config: PilotPaperFixedFprConfig | None = None,
) -> dict[str, Any]:
    """构建 pilot_paper prompt split 摘要。

    此处设计的主要考虑在于: pilot_paper 共同协议必须冻结同一个 prompt split
    digest, 后续 proposed method 与全部 baseline 只能引用该 digest, 不能各自
    重新划分样本。
    """

    resolved_config = config or PilotPaperFixedFprConfig()
    pilot_paper_records = tuple(record for record in prompt_records if record.prompt_set == resolved_config.prompt_set)
    assigned_records = apply_split_assignments(pilot_paper_records)
    split_counts = dict(sorted(Counter(record.split for record in assigned_records).items()))
    risk_profile_counts = dict(sorted(Counter(record.risk_profile for record in assigned_records).items()))
    split_record_payload = _stable_records_for_prompt_split(assigned_records)
    calibration_ids = {record.prompt_id for record in assigned_records if record.split == "calibration"}
    test_ids = {record.prompt_id for record in assigned_records if record.split == "test"}
    calibration_clean_negative_count = len(calibration_ids)
    test_clean_negative_count = len(test_ids)
    expected_prompt_count = RUN_EXPECTED_PROMPT_COUNTS[resolved_config.paper_run_name]
    expected_split_counts = build_group_split_counts(expected_prompt_count)
    allowed_false_positive_count = math.floor(resolved_config.target_fpr * calibration_clean_negative_count)
    prompt_split_digest = build_stable_digest(split_record_payload)
    calibration_prompt_id_digest = build_stable_digest(sorted(calibration_ids))
    test_prompt_id_digest = build_stable_digest(sorted(test_ids))
    minimum_ready = (
        len(assigned_records) == expected_prompt_count
        and calibration_clean_negative_count == expected_split_counts["calibration"]
        and test_clean_negative_count == expected_split_counts["test"]
        and resolved_config.minimum_clean_negative_count == expected_split_counts["test"]
    )
    return {
        "prompt_set": resolved_config.prompt_set,
        "prompt_file": resolved_config.prompt_file,
        "prompt_protocol_name": resolved_config.prompt_protocol_name,
        "pilot_paper_prompt_count": len(assigned_records),
        "expected_prompt_count": expected_prompt_count,
        "split_counts": split_counts,
        "risk_profile_counts": risk_profile_counts,
        "calibration_test_disjoint": calibration_ids.isdisjoint(test_ids),
        "calibration_clean_negative_count": calibration_clean_negative_count,
        "test_clean_negative_count": test_clean_negative_count,
        "pilot_paper_negative_count_minimum_required": resolved_config.minimum_clean_negative_count,
        "pilot_paper_negative_count_ready": minimum_ready,
        "target_fpr": resolved_config.target_fpr,
        "allowed_false_positive_count": allowed_false_positive_count,
        "prompt_split_digest": prompt_split_digest,
        "calibration_prompt_id_digest": calibration_prompt_id_digest,
        "test_prompt_id_digest": test_prompt_id_digest,
        "prompt_split_ready": bool(assigned_records) and calibration_ids.isdisjoint(test_ids) and minimum_ready,
        "supports_paper_claim": False,
    }


def build_pilot_paper_attack_matrix_rows(
    attack_configs: Iterable[AttackConfig],
    config: PilotPaperFixedFprConfig | None = None,
) -> tuple[dict[str, Any], ...]:
    """构建 pilot_paper 共同协议使用的同一攻击矩阵行。

    这一实现属于项目特定写法: 它复用已有攻击配置, 但用 pilot_paper 协议 digest
    约束所有方法共享同一批攻击定义。
    """

    resolved_config = config or PilotPaperFixedFprConfig()
    rows: list[dict[str, Any]] = []
    for attack_config in attack_configs:
        if not attack_config.enabled or attack_config.resource_profile not in resolved_config.attack_resource_profiles:
            continue
        config_digest = attack_config_digest(attack_config)
        row = {
            "attack_id": attack_config.attack_id,
            "attack_family": attack_config.attack_family,
            "attack_name": attack_config.attack_name,
            "attack_strength": attack_config.attack_strength,
            "resource_profile": attack_config.resource_profile,
            "requires_gpu": attack_config.requires_gpu,
            "attack_parameters": dict(attack_config.attack_parameters),
            "attack_config_digest": config_digest,
            "result_scope": resolved_config.result_scope,
            "supports_paper_claim": False,
        }
        rows.append(row)
    return tuple(sorted(rows, key=lambda item: (item["resource_profile"], item["attack_family"], item["attack_name"], item["attack_id"])))


def build_fixed_fpr_protocol_digest(config: PilotPaperFixedFprConfig | None = None) -> str:
    """生成 pilot_paper fixed-FPR 协议摘要。"""

    resolved_config = config or PilotPaperFixedFprConfig()
    payload = {
        "result_protocol_name": resolved_config.result_protocol_name,
        "result_scope": resolved_config.result_scope,
        "target_fpr": resolved_config.target_fpr,
        "calibration_split": "calibration",
        "test_split": "test",
        "calibration_role": "clean_negative",
        "confidence_interval_method": resolved_config.confidence_interval_method,
        "confidence_level": resolved_config.confidence_level,
        "minimum_clean_negative_count": resolved_config.minimum_clean_negative_count,
        "result_claim_scope": resolved_config.result_claim_scope,
    }
    return build_stable_digest(payload)


def build_pilot_paper_result_import_schema(
    *,
    prompt_split_digest: str,
    attack_matrix_digest: str,
    fixed_fpr_protocol_digest: str,
    config: PilotPaperFixedFprConfig | None = None,
) -> dict[str, Any]:
    """构建 pilot_paper 结果受治理导入 schema 描述。"""

    resolved_config = config or PilotPaperFixedFprConfig()
    return {
        "result_protocol_name": resolved_config.result_protocol_name,
        "result_scope": resolved_config.result_scope,
        "result_claim_scope": resolved_config.result_claim_scope,
        "prompt_set": resolved_config.prompt_set,
        "prompt_protocol_name": resolved_config.prompt_protocol_name,
        "prompt_split_digest": prompt_split_digest,
        "attack_matrix_digest": attack_matrix_digest,
        "fixed_fpr_protocol_digest": fixed_fpr_protocol_digest,
        "target_fpr": resolved_config.target_fpr,
        "confidence_interval_method": resolved_config.confidence_interval_method,
        "confidence_level": resolved_config.confidence_level,
        "minimum_result_positive_count": resolved_config.minimum_clean_negative_count,
        "minimum_result_negative_count": resolved_config.minimum_clean_negative_count,
        "minimum_result_attacked_negative_count": resolved_config.minimum_clean_negative_count,
        "method_ids": list(PILOT_PAPER_METHOD_IDS),
        "primary_baseline_ids": list(PILOT_PAPER_PRIMARY_BASELINE_IDS),
        "required_metric_fields": list(PILOT_PAPER_REQUIRED_METRIC_FIELDS),
        "required_source_fields": list(PILOT_PAPER_REQUIRED_SOURCE_FIELDS),
        "required_rate_fields": list(PILOT_PAPER_RATE_FIELDS),
        "metric_bounds": {
            field_name: [lower_bound, upper_bound]
            for field_name, (lower_bound, upper_bound) in (
                PILOT_PAPER_METRIC_BOUNDS.items()
            )
        },
        "ci_field_groups": [list(group) for group in PILOT_PAPER_CI_FIELD_GROUPS],
        "ci_count_fields": dict(PILOT_PAPER_CI_COUNT_FIELDS),
        "supports_paper_claim": True,
        "paper_run_allows_paper_claim": True,
        "paper_run_claim_type": resolved_config.result_claim_scope,
        "strict_formal_evidence_required": True,
        "nonformal_evidence_rejection_policy": "reject_nonformal_records",
        "probe_paper_workflow_boundary": PROBE_PAPER_WORKFLOW_BOUNDARY,
        "paper_claim_scale": resolved_config.prompt_set,
        "full_paper_claim_boundary": FULL_PAPER_CLAIM_BOUNDARY,
    }


def build_pilot_paper_method_registry_rows(
    *,
    prompt_split_digest: str,
    attack_matrix_digest: str,
    fixed_fpr_protocol_digest: str,
    config: PilotPaperFixedFprConfig | None = None,
) -> tuple[dict[str, Any], ...]:
    """构建参与 pilot_paper 共同协议的方法登记表。"""

    resolved_config = config or PilotPaperFixedFprConfig()
    display_names = {
        "slm_wm_current": "SLM-WM",
        "tree_ring": "Tree-Ring",
        "gaussian_shading": "Gaussian Shading",
        "shallow_diffuse": "Shallow Diffuse",
        "t2smark": "T2SMark",
    }
    rows: list[dict[str, Any]] = []
    for method_id in PILOT_PAPER_METHOD_IDS:
        role = "proposed_method" if method_id == "slm_wm_current" else "primary_baseline"
        rows.append(
            {
                "method_id": method_id,
                "method_name": display_names[method_id],
                "method_role": role,
                "prompt_set": resolved_config.prompt_set,
                "prompt_file": resolved_config.prompt_file,
                "prompt_protocol_name": resolved_config.prompt_protocol_name,
                "prompt_split_digest": prompt_split_digest,
                "attack_matrix_digest": attack_matrix_digest,
                "fixed_fpr_protocol_digest": fixed_fpr_protocol_digest,
                "target_fpr": resolved_config.target_fpr,
                "confidence_interval_method": resolved_config.confidence_interval_method,
                "confidence_level": resolved_config.confidence_level,
                "result_protocol_name": resolved_config.result_protocol_name,
                "result_scope": resolved_config.result_scope,
                "result_claim_scope": resolved_config.result_claim_scope,
                "governed_import_required": True,
                "supports_paper_claim": True,
                "paper_claim_scale": resolved_config.prompt_set,
            }
        )
    return tuple(rows)


def build_pilot_paper_result_import_template_rows(
    method_rows: Iterable[Mapping[str, Any]],
    attack_rows: Iterable[Mapping[str, Any]],
    config: PilotPaperFixedFprConfig | None = None,
) -> tuple[dict[str, Any], ...]:
    """构建 method × attack 的 pilot_paper 结果导入模板。

    在其他项目中可复用的部分是模板生成方式: 它不依赖具体 baseline 代码,
    只要求每个方法在同一 prompt split、同一 attack matrix 和同一 fixed-FPR
    协议下提交同构结果记录。
    """

    resolved_config = config or PilotPaperFixedFprConfig()
    rows: list[dict[str, Any]] = []
    for method_row in method_rows:
        for attack_row in attack_rows:
            payload = {
                "method_id": _str_field(method_row, "method_id"),
                "method_role": _str_field(method_row, "method_role"),
                "attack_id": _str_field(attack_row, "attack_id"),
                "attack_config_digest": _str_field(
                    attack_row,
                    "attack_config_digest",
                ),
                "attack_family": _str_field(attack_row, "attack_family"),
                "attack_name": _str_field(attack_row, "attack_name"),
                "resource_profile": _str_field(attack_row, "resource_profile"),
                "requires_gpu": bool(attack_row.get("requires_gpu")),
                "target_fpr": resolved_config.target_fpr,
                "result_protocol_name": resolved_config.result_protocol_name,
                "result_scope": resolved_config.result_scope,
                "result_claim_scope": resolved_config.result_claim_scope,
                "prompt_protocol_name": resolved_config.prompt_protocol_name,
                "prompt_split_digest": _str_field(method_row, "prompt_split_digest"),
                "attack_matrix_digest": _str_field(method_row, "attack_matrix_digest"),
                "fixed_fpr_protocol_digest": _str_field(method_row, "fixed_fpr_protocol_digest"),
                "confidence_interval_method": resolved_config.confidence_interval_method,
                "confidence_level": resolved_config.confidence_level,
                "required_metric_fields": list(PILOT_PAPER_REQUIRED_METRIC_FIELDS),
                "required_source_fields": list(PILOT_PAPER_REQUIRED_SOURCE_FIELDS),
                "required_result_record_path": (
                    "outputs/pilot_paper_fixed_fpr_results/"
                    f"{resolved_config.paper_run_name}/pilot_paper_result_records.jsonl"
                ),
                "supports_paper_claim": False,
                "paper_claim_scale": resolved_config.prompt_set,
            }
            digest = build_stable_digest(payload)
            payload["pilot_paper_result_template_id"] = f"pilot_paper_result_template_{digest[:16]}"
            payload["pilot_paper_result_template_digest"] = digest
            rows.append(payload)
    return tuple(rows)


def _validate_required_fields(row: Mapping[str, Any], row_index: int, schema: Mapping[str, Any]) -> list[PilotPaperImportIssue]:
    """校验 schema 要求的字段是否存在。"""

    issues: list[PilotPaperImportIssue] = []
    for field_name in tuple(schema["required_metric_fields"]) + tuple(schema["required_source_fields"]):
        value = row.get(field_name)
        missing = (
            field_name not in row
            or value is None
            or (isinstance(value, str) and not value.strip())
            or (isinstance(value, list | tuple) and not value)
        )
        if missing:
            issues.append(_issue(row_index, row, field_name, "required_field_missing"))
    return issues


def _validate_counts_and_rates(row: Mapping[str, Any], row_index: int, schema: Mapping[str, Any]) -> list[PilotPaperImportIssue]:
    """校验计数、率值和置信区间边界。"""

    issues: list[PilotPaperImportIssue] = []
    minimum_count_fields = {
        "positive_count": int(schema.get("minimum_result_positive_count", 1)),
        "negative_count": int(schema.get("minimum_result_negative_count", 1)),
        "attacked_negative_count": int(schema.get("minimum_result_attacked_negative_count", 1)),
    }
    for field_name, minimum_count in minimum_count_fields.items():
        if _int_field(row, field_name) < minimum_count:
            issues.append(_issue(row_index, row, field_name, "pilot_paper_minimum_sample_count_required"))
    positive_count = _int_field(row, "positive_count")
    attacked_negative_count = _int_field(row, "attacked_negative_count")
    supported_record_count = _int_field(row, "supported_record_count")
    attack_record_count = _int_field(row, "attack_record_count")
    if supported_record_count <= 0:
        issues.append(_issue(row_index, row, "supported_record_count", "positive_count_required"))
    if supported_record_count != positive_count:
        issues.append(
            _issue(
                row_index,
                row,
                "supported_record_count",
                "supported_record_count_must_equal_attacked_positive_count",
            )
        )
    if attack_record_count != positive_count + attacked_negative_count:
        issues.append(
            _issue(
                row_index,
                row,
                "attack_record_count",
                "attack_record_count_must_equal_attacked_sample_count",
            )
        )
    metric_bounds = schema["metric_bounds"]
    if not isinstance(metric_bounds, Mapping):
        raise ValueError("metric_bounds 必须是字段到有界区间的映射")
    for field_name, bounds in metric_bounds.items():
        if not isinstance(bounds, list | tuple) or len(bounds) != 2:
            raise ValueError(f"{field_name} 的 metric_bounds 必须包含上下界")
        lower_bound, upper_bound = (float(bounds[0]), float(bounds[1]))
        value = _float_field(row, field_name)
        if not lower_bound <= value <= upper_bound:
            issues.append(
                _issue(
                    row_index,
                    row,
                    field_name,
                    "metric_value_must_be_in_declared_bounds",
                )
            )
    for low_name, value_name, high_name in schema["ci_field_groups"]:
        low = _float_field(row, low_name)
        value = _float_field(row, value_name)
        high = _float_field(row, high_name)
        bounds = metric_bounds[value_name]
        lower_bound, upper_bound = (float(bounds[0]), float(bounds[1]))
        if not (lower_bound <= low <= value <= high <= upper_bound):
            issues.append(_issue(row_index, row, value_name, "confidence_interval_must_cover_metric"))
            continue
        sample_count = _int_field(
            row,
            schema["ci_count_fields"][value_name],
        )
        expected_low, expected_high = bounded_hoeffding_confidence_interval(
            value,
            sample_count,
            float(schema["confidence_level"]),
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        )
        if not (
            math.isclose(low, expected_low, rel_tol=0.0, abs_tol=1e-12)
            and math.isclose(high, expected_high, rel_tol=0.0, abs_tol=1e-12)
        ):
            issues.append(
                _issue(
                    row_index,
                    row,
                    value_name,
                    "confidence_interval_must_match_bounded_hoeffding",
                )
            )
    return issues


def _validate_protocol_fields(row: Mapping[str, Any], row_index: int, schema: Mapping[str, Any]) -> list[PilotPaperImportIssue]:
    """校验 prompt、attack、fixed-FPR 和声明边界是否与 schema 一致。"""

    issues: list[PilotPaperImportIssue] = []
    equality_fields = (
        "result_protocol_name",
        "result_scope",
        "result_claim_scope",
        "prompt_protocol_name",
        "prompt_split_digest",
        "attack_matrix_digest",
        "fixed_fpr_protocol_digest",
    )
    for field_name in equality_fields:
        if _str_field(row, field_name) != str(schema[field_name]):
            issues.append(_issue(row_index, row, field_name, "protocol_value_mismatch"))
    if _str_field(row, "method_id") not in set(schema["method_ids"]):
        issues.append(_issue(row_index, row, "method_id", "pilot_paper_method_id_required"))
    try:
        attack_config = resolve_formal_attack_config(
            attack_family=_str_field(row, "attack_family"),
            attack_name=_str_field(row, "attack_name"),
            resource_profile=_str_field(row, "resource_profile"),
        )
    except ValueError:
        issues.append(
            _issue(
                row_index,
                row,
                "attack_id",
                "registered_formal_attack_identity_required",
            )
        )
    else:
        expected_attack_fields = {
            "attack_id": attack_config.attack_id,
            "attack_config_digest": attack_config_digest(attack_config),
        }
        for field_name, expected_value in expected_attack_fields.items():
            if _str_field(row, field_name) != expected_value:
                issues.append(
                    _issue(
                        row_index,
                        row,
                        field_name,
                        "formal_attack_identity_must_match_attack_config",
                    )
                )
    if not math.isclose(_float_field(row, "target_fpr"), float(schema["target_fpr"]), rel_tol=0.0, abs_tol=1e-12):
        issues.append(_issue(row_index, row, "target_fpr", "target_fpr_mismatch"))
    if _str_field(row, "confidence_interval_method") != str(schema["confidence_interval_method"]):
        issues.append(_issue(row_index, row, "confidence_interval_method", "confidence_interval_method_mismatch"))
    if not math.isclose(
        _float_field(row, "confidence_level"),
        float(schema["confidence_level"]),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        issues.append(_issue(row_index, row, "confidence_level", "confidence_level_mismatch"))
    paper_claim_scale = _str_field(row, "paper_claim_scale")
    expected_paper_claim_scale = str(schema.get("paper_claim_scale", PILOT_PAPER_PROMPT_SET))
    if paper_claim_scale and paper_claim_scale != expected_paper_claim_scale:
        reason = (
            "pilot_paper_claim_scale_required"
            if expected_paper_claim_scale == PILOT_PAPER_PROMPT_SET
            else "paper_claim_scale_mismatch"
        )
        issues.append(_issue(row_index, row, "paper_claim_scale", reason))
    if bool(schema.get("strict_formal_evidence_required", True)):
        if not _bool_field(row, "supports_paper_claim"):
            issues.append(_issue(row_index, row, "supports_paper_claim", "strict_formal_claim_record_required"))
        if not _bool_field(row, "strict_formal_result_ready"):
            issues.append(_issue(row_index, row, "strict_formal_result_ready", "strict_formal_result_required"))
        inspected_values = (
            row.get("metric_status", ""),
            row.get("result_source_kind", ""),
            row.get("baseline_result_source", ""),
            row.get("evidence_paths", ()),
        )
        if contains_nonformal_marker(inspected_values):
            issues.append(_issue(row_index, row, "metric_status", "nonformal_result_marker_rejected"))
    return issues


def _validate_evidence_paths(
    row: Mapping[str, Any],
    row_index: int,
    evidence_root: Path,
    require_existing_evidence: bool,
) -> list[PilotPaperImportIssue]:
    """校验证据路径字段是否非空, 并在需要时检查文件存在。"""

    evidence_paths = _list_field(row, "evidence_paths")
    if not evidence_paths:
        return [_issue(row_index, row, "evidence_paths", "evidence_paths_required")]
    if not require_existing_evidence:
        return []
    issues: list[PilotPaperImportIssue] = []
    for evidence_path in evidence_paths:
        candidate = Path(evidence_path)
        resolved_path = candidate if candidate.is_absolute() else evidence_root / candidate
        if not resolved_path.is_file():
            issues.append(_issue(row_index, row, "evidence_paths", "evidence_path_missing"))
    return issues


def validate_pilot_paper_result_import_rows(
    rows: Iterable[Mapping[str, Any]],
    schema: Mapping[str, Any],
    *,
    evidence_root: str | Path = ".",
    require_existing_evidence: bool = False,
) -> dict[str, Any]:
    """校验 pilot_paper 共同协议结果导入记录。

    该函数属于 schema validator 层, 负责收敛重复字段校验。下游对比表只应消费
    accepted_records, 从而避免把小样本或未对齐协议的结果误当作正式论文结论。
    """

    evidence_root_path = Path(evidence_root).resolve()
    materialized_rows = [dict(row) for row in rows]
    accepted: list[dict[str, Any]] = []
    issues: list[PilotPaperImportIssue] = []
    seen_template_keys: set[tuple[str, str, str, str]] = set()
    for row_index, row in enumerate(materialized_rows):
        row_issues: list[PilotPaperImportIssue] = []
        template_key = (
            _str_field(row, "method_id"),
            _str_field(row, "attack_family"),
            _str_field(row, "attack_name"),
            _str_field(row, "resource_profile"),
        )
        if template_key in seen_template_keys:
            row_issues.append(_issue(row_index, row, "method_id", "duplicate_result_template_key"))
        else:
            seen_template_keys.add(template_key)
        row_issues.extend(_validate_required_fields(row, row_index, schema))
        if not row_issues:
            row_issues.extend(_validate_counts_and_rates(row, row_index, schema))
            row_issues.extend(_validate_protocol_fields(row, row_index, schema))
            row_issues.extend(_validate_evidence_paths(row, row_index, evidence_root_path, require_existing_evidence))
        if row_issues:
            issues.extend(row_issues)
        else:
            accepted.append(row)
    claim_records_ready = bool(accepted) and not issues and all(
        _bool_field(row, "supports_paper_claim") for row in accepted
    )
    return {
        "protocol_name": schema["result_protocol_name"],
        "result_scope": schema["result_scope"],
        "target_fpr": schema["target_fpr"],
        "input_record_count": len(materialized_rows),
        "accepted_pilot_paper_import_count": len(accepted),
        "rejected_pilot_paper_import_count": len(materialized_rows) - len(accepted),
        "pilot_paper_import_issue_count": len(issues),
        "pilot_paper_result_import_ready": bool(materialized_rows) and not issues,
        "accepted_records": accepted,
        "issues": [issue.to_dict() for issue in issues],
        "accepted_pilot_paper_claim_record_count": sum(1 for row in accepted if _bool_field(row, "supports_paper_claim")),
        "pilot_paper_claim_record_ready": claim_records_ready,
        "supports_paper_claim": claim_records_ready,
    }


def _record_key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    """提取共同协议模板键, 用于比较同一攻击设置下的方法结果。"""

    return (str(row.get("attack_family", "")), str(row.get("attack_name", "")), str(row.get("resource_profile", "")))


def _mean_rate(rows: tuple[Mapping[str, Any], ...], field_name: str) -> float:
    """计算结果记录中的平均 rate, 空集合返回 0。"""

    return sum(float(row.get(field_name, 0.0)) for row in rows) / len(rows) if rows else 0.0


def build_superiority_gate_summary(
    accepted_records: Iterable[Mapping[str, Any]],
    config: PilotPaperFixedFprConfig,
) -> dict[str, Any]:
    """根据受治理结果记录判断是否支持 SLM-WM 优势性主张。

    该函数属于项目特定的论文声明门禁: common protocol 覆盖完整只说明证据可导入,
    不能自动推出方法优越。这里显式要求 SLM-WM 与主表 baseline 在相同攻击模板
    上比较, 且 SLM-WM 平均 TPR 高于最佳 baseline, 同时平均 FPR 不超过 fixed-FPR
    目标, 才允许支持 superiority claim。
    """

    materialized = tuple(row for row in accepted_records if _bool_field(row, "supports_paper_claim"))
    slm_rows = tuple(row for row in materialized if str(row.get("method_id", "")) == PILOT_PAPER_PRIMARY_METHOD_ID)
    baseline_rows_by_method = {
        method_id: tuple(row for row in materialized if str(row.get("method_id", "")) == method_id)
        for method_id in PILOT_PAPER_PRIMARY_BASELINE_IDS
    }
    slm_keys = {_record_key(row) for row in slm_rows}
    method_template_keys_unique = len(slm_rows) == len(slm_keys) and all(
        len(rows) == len({_record_key(row) for row in rows}) for rows in baseline_rows_by_method.values()
    )
    missing_baseline_methods = tuple(method_id for method_id, rows in baseline_rows_by_method.items() if not rows)
    template_aligned = (
        bool(slm_rows)
        and method_template_keys_unique
        and all({_record_key(row) for row in rows} == slm_keys for rows in baseline_rows_by_method.values())
    )
    slm_mean_tpr = _mean_rate(slm_rows, "true_positive_rate")
    slm_mean_fpr = _mean_rate(slm_rows, "false_positive_rate")
    baseline_mean_tprs = {
        method_id: _mean_rate(rows, "true_positive_rate") for method_id, rows in baseline_rows_by_method.items() if rows
    }
    best_baseline_method_id = max(baseline_mean_tprs, key=baseline_mean_tprs.get) if baseline_mean_tprs else ""
    best_baseline_mean_tpr = baseline_mean_tprs.get(best_baseline_method_id, 0.0)
    fixed_fpr_ready = bool(slm_rows) and slm_mean_fpr <= config.target_fpr
    superiority_ready = (
        bool(materialized)
        and bool(slm_rows)
        and not missing_baseline_methods
        and template_aligned
        and fixed_fpr_ready
        and slm_mean_tpr > best_baseline_mean_tpr
    )
    if superiority_ready:
        reason = "slm_wm_tpr_exceeds_best_baseline_within_fixed_fpr"
    elif not slm_rows:
        reason = "slm_wm_records_missing"
    elif missing_baseline_methods:
        reason = "baseline_records_missing"
    elif not method_template_keys_unique:
        reason = "duplicate_method_attack_template_records"
    elif not template_aligned:
        reason = "method_attack_templates_not_aligned"
    elif not fixed_fpr_ready:
        reason = "slm_wm_fpr_exceeds_fixed_fpr_target"
    else:
        reason = "slm_wm_tpr_not_above_best_baseline"
    return {
        "superiority_gate_ready": superiority_ready,
        "superiority_gate_reason": reason,
        "slm_wm_record_count": len(slm_rows),
        "slm_wm_mean_true_positive_rate": slm_mean_tpr,
        "slm_wm_mean_false_positive_rate": slm_mean_fpr,
        "slm_wm_fixed_fpr_boundary_ready": fixed_fpr_ready,
        "best_baseline_method_id": best_baseline_method_id,
        "best_baseline_mean_true_positive_rate": best_baseline_mean_tpr,
        "missing_baseline_methods": list(missing_baseline_methods),
        "method_attack_template_keys_unique": method_template_keys_unique,
        "method_attack_templates_aligned": template_aligned,
    }


def build_pilot_paper_common_protocol_summary(
    *,
    prompt_summary: Mapping[str, Any],
    attack_rows: Iterable[Mapping[str, Any]],
    method_rows: Iterable[Mapping[str, Any]],
    template_rows: Iterable[Mapping[str, Any]],
    import_validation_report: Mapping[str, Any],
    paired_superiority_summary: Mapping[str, Any],
    config: PilotPaperFixedFprConfig | None = None,
) -> dict[str, Any]:
    """汇总 pilot_paper fixed-FPR 共同协议的运行前治理状态。"""

    resolved_config = config or PilotPaperFixedFprConfig()
    materialized_attack_rows = tuple(attack_rows)
    materialized_method_rows = tuple(method_rows)
    materialized_template_rows = tuple(template_rows)
    method_ids = {str(row["method_id"]) for row in materialized_method_rows}
    accepted_records = tuple(dict(row) for row in import_validation_report.get("accepted_records", ()))
    accepted_claim_record_count = sum(1 for row in accepted_records if _bool_field(row, "supports_paper_claim"))
    import_ready = bool(import_validation_report.get("pilot_paper_result_import_ready", False))
    expected_template_keys = {
        (
            str(row.get("method_id", "")),
            str(row.get("attack_family", "")),
            str(row.get("attack_name", "")),
            str(row.get("resource_profile", "")),
        )
        for row in materialized_template_rows
    }
    accepted_template_key_rows = [
        (
            str(row.get("method_id", "")),
            str(row.get("attack_family", "")),
            str(row.get("attack_name", "")),
            str(row.get("resource_profile", "")),
        )
        for row in accepted_records
    ]
    accepted_template_keys = set(accepted_template_key_rows)
    missing_template_count = len(expected_template_keys - accepted_template_keys)
    unexpected_template_count = len(accepted_template_keys - expected_template_keys)
    duplicate_template_count = len(accepted_template_key_rows) - len(accepted_template_keys)
    template_import_coverage_ready = (
        bool(expected_template_keys)
        and accepted_template_keys == expected_template_keys
        and duplicate_template_count == 0
    )
    claim_coverage_ready = (
        template_import_coverage_ready
        and accepted_claim_record_count == len(accepted_records)
    )
    superiority_gate = build_superiority_gate_summary(accepted_records, resolved_config)
    expected_target_fpr = PAPER_RUN_FIXED_FPR.get(resolved_config.paper_run_name, PILOT_PAPER_FIXED_FPR)
    template_registry_unique = len(expected_template_keys) == len(materialized_template_rows)
    ready = (
        bool(prompt_summary.get("prompt_split_ready"))
        and bool(materialized_attack_rows)
        and method_ids == set(PILOT_PAPER_METHOD_IDS)
        and len(materialized_template_rows) == len(materialized_attack_rows) * len(materialized_method_rows)
        and template_registry_unique
        and math.isclose(float(resolved_config.target_fpr), expected_target_fpr, rel_tol=0.0, abs_tol=1e-12)
    )
    paper_run_allows_claim = True
    paper_run_workflow_validation_ready = ready and import_ready and template_import_coverage_ready
    expected_paired_baseline_ids = set(PILOT_PAPER_PRIMARY_BASELINE_IDS)
    paired_baseline_ids = {
        str(value) for value in paired_superiority_summary.get("primary_baseline_ids", ())
    }
    paired_ready_ids = {
        str(value)
        for value in paired_superiority_summary.get("paired_superiority_ready_ids", ())
    }
    quality_matched_ready_ids = {
        str(value)
        for value in paired_superiority_summary.get(
            "quality_matched_ready_ids",
            (),
        )
    }
    paired_attack_counts = {
        int(value) for value in paired_superiority_summary.get("paired_attack_counts", ())
    }
    paired_prompt_counts = {
        int(value) for value in paired_superiority_summary.get("paired_prompt_counts", ())
    }
    paired_observation_sha256_map = paired_superiority_summary.get(
        "method_observation_source_sha256_map",
        {},
    )
    paired_digest_fields = (
        "paired_outcome_set_digest",
        "paired_superiority_rows_digest",
        "paired_superiority_protocol_digest",
        "quality_matching_protocol_digest",
        "quality_matched_rows_digest",
        "paired_test_prompt_id_digest",
        "paired_attack_registry_digest",
        "threshold_audit_rows_digest",
    )
    paired_superiority_ready = (
        paired_superiority_summary.get("paper_claim_scale")
        == resolved_config.paper_run_name
        and math.isclose(
            float(paired_superiority_summary.get("target_fpr", float("nan"))),
            resolved_config.target_fpr,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        and paired_superiority_summary.get("paired_superiority_exact_set_ready") is True
        and paired_superiority_summary.get("paired_superiority_scale_ready") is True
        and paired_superiority_summary.get("overall_paired_superiority_ready") is True
        and paired_superiority_summary.get(
            "overall_quality_matched_superiority_ready"
        )
        is True
        and paired_superiority_summary.get("quality_matched_exact_set_ready") is True
        and paired_superiority_summary.get("quality_matching_uses_detection_labels")
        is False
        and paired_superiority_summary.get("supports_paper_claim") is True
        and paired_baseline_ids == expected_paired_baseline_ids
        and paired_ready_ids == expected_paired_baseline_ids
        and quality_matched_ready_ids == expected_paired_baseline_ids
        and int(paired_superiority_summary.get("paired_superiority_row_count", 0))
        == len(expected_paired_baseline_ids)
        and int(paired_superiority_summary.get("quality_matched_row_count", 0))
        == len(expected_paired_baseline_ids)
        and int(paired_superiority_summary.get("expected_test_count", 0))
        == resolved_config.minimum_clean_negative_count
        and int(paired_superiority_summary.get("paired_test_prompt_count", 0))
        == resolved_config.minimum_clean_negative_count
        and str(paired_superiority_summary.get("paired_test_prompt_id_digest", ""))
        == str(prompt_summary.get("test_prompt_id_digest", ""))
        and int(paired_superiority_summary.get("expected_attack_count", 0))
        == len(materialized_attack_rows)
        and paired_prompt_counts == {resolved_config.minimum_clean_negative_count}
        and paired_attack_counts == {len(materialized_attack_rows)}
        and int(paired_superiority_summary.get("bootstrap_resample_count", 0))
        == PILOT_PAPER_PAIRED_BOOTSTRAP_RESAMPLE_COUNT
        and math.isclose(
            float(paired_superiority_summary.get("confidence_level", math.nan)),
            PILOT_PAPER_CONFIDENCE_LEVEL,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        and paired_superiority_summary.get("bootstrap_analysis_schema")
        == PILOT_PAPER_PAIRED_BOOTSTRAP_ANALYSIS_SCHEMA
        and paired_superiority_summary.get("bootstrap_bit_generator")
        == PILOT_PAPER_PAIRED_BOOTSTRAP_BIT_GENERATOR
        and paired_superiority_summary.get("bootstrap_quantile_method")
        == PILOT_PAPER_PAIRED_BOOTSTRAP_QUANTILE_METHOD
        and paired_superiority_summary.get("claim_p_value_method")
        == PILOT_PAPER_PAIRED_CLAIM_P_VALUE_METHOD
        and paired_superiority_summary.get("sharp_null_diagnostic_method")
        == PILOT_PAPER_PAIRED_SHARP_NULL_DIAGNOSTIC_METHOD
        and isinstance(paired_observation_sha256_map, Mapping)
        and set(paired_observation_sha256_map)
        == {"slm_wm", *PILOT_PAPER_PRIMARY_BASELINE_IDS}
        and all(
            len(str(value)) == 64
            for value in paired_observation_sha256_map.values()
        )
        and all(
            len(str(paired_superiority_summary.get(field_name, ""))) == 64
            for field_name in paired_digest_fields
        )
    )
    effect_direction_ready = superiority_gate["superiority_gate_ready"]
    effectiveness_ready = effect_direction_ready and paired_superiority_ready
    paper_run_claim_ready = (
        ready
        and paper_run_allows_claim
        and import_ready
        and claim_coverage_ready
        and effectiveness_ready
    )
    probe_paper_claim_ready = paper_run_claim_ready and resolved_config.prompt_set == PROBE_PAPER_RUN_NAME
    probe_paper_workflow_validation_ready = (
        resolved_config.prompt_set == PROBE_PAPER_RUN_NAME and paper_run_workflow_validation_ready
    )
    pilot_paper_claim_ready = paper_run_claim_ready and resolved_config.prompt_set == PILOT_PAPER_PROMPT_SET
    full_paper_claim_ready = paper_run_claim_ready and resolved_config.prompt_set == FULL_PAPER_RUN_NAME
    return {
        "construction_unit_name": "pilot_paper_fixed_fpr_common_protocol",
        "result_protocol_name": resolved_config.result_protocol_name,
        "result_scope": resolved_config.result_scope,
        "result_claim_scope": resolved_config.result_claim_scope,
        "paper_claim_scale": resolved_config.prompt_set,
        "paper_run_claim_type": resolved_config.result_claim_scope,
        "full_paper_claim_boundary": FULL_PAPER_CLAIM_BOUNDARY,
        "probe_paper_workflow_boundary": PROBE_PAPER_WORKFLOW_BOUNDARY,
        "paper_run_allows_paper_claim": paper_run_allows_claim,
        "strict_formal_evidence_required": True,
        "pilot_paper_common_protocol_ready": ready,
        "paper_run_workflow_validation_ready": paper_run_workflow_validation_ready,
        "probe_paper_workflow_validation_ready": probe_paper_workflow_validation_ready,
        "pilot_paper_prompt_count": prompt_summary.get("pilot_paper_prompt_count", 0),
        "paper_prompt_count": prompt_summary.get("pilot_paper_prompt_count", 0),
        "pilot_paper_prompt_split_ready": prompt_summary.get("prompt_split_ready", False),
        "paper_prompt_split_ready": prompt_summary.get("prompt_split_ready", False),
        "calibration_prompt_id_digest": prompt_summary.get(
            "calibration_prompt_id_digest",
            "",
        ),
        "test_prompt_id_digest": prompt_summary.get("test_prompt_id_digest", ""),
        "pilot_paper_target_fpr": resolved_config.target_fpr,
        "paper_target_fpr": resolved_config.target_fpr,
        "expected_target_fpr": expected_target_fpr,
        "pilot_paper_negative_count_minimum_required": resolved_config.minimum_clean_negative_count,
        "minimum_result_positive_count": resolved_config.minimum_clean_negative_count,
        "minimum_result_negative_count": resolved_config.minimum_clean_negative_count,
        "minimum_result_attacked_negative_count": resolved_config.minimum_clean_negative_count,
        "pilot_paper_attack_count": len(materialized_attack_rows),
        "pilot_paper_method_count": len(materialized_method_rows),
        "pilot_paper_import_template_count": len(materialized_template_rows),
        "pilot_paper_result_import_ready": import_ready,
        "accepted_pilot_paper_import_count": int(import_validation_report.get("accepted_pilot_paper_import_count", 0)),
        "accepted_pilot_paper_claim_record_count": accepted_claim_record_count,
        "pilot_paper_claim_record_ready": claim_coverage_ready,
        "paper_run_result_import_coverage_ready": template_import_coverage_ready,
        "paper_run_result_missing_template_count": missing_template_count,
        "paper_run_result_unexpected_template_count": unexpected_template_count,
        "paper_run_result_duplicate_template_count": duplicate_template_count,
        "paper_run_template_registry_unique": template_registry_unique,
        "pilot_paper_evidence_coverage_ready": claim_coverage_ready,
        "point_estimate_effect_direction_ready": effect_direction_ready,
        "pilot_paper_effectiveness_gate_ready": effectiveness_ready,
        "pilot_paper_effectiveness_gate_reason": (
            "paired_superiority_with_fixed_fpr_ready"
            if effectiveness_ready
            else (
                superiority_gate["superiority_gate_reason"]
                if not effect_direction_ready
                else "paired_superiority_not_ready"
            )
        ),
        "paired_superiority_ready": paired_superiority_ready,
        "paired_superiority_exact_set_ready": paired_superiority_summary.get(
            "paired_superiority_exact_set_ready", False
        ),
        "overall_paired_superiority_ready": paired_superiority_summary.get(
            "overall_paired_superiority_ready", False
        ),
        "overall_quality_matched_superiority_ready": (
            paired_superiority_summary.get(
                "overall_quality_matched_superiority_ready",
                False,
            )
        ),
        "quality_matched_exact_set_ready": paired_superiority_summary.get(
            "quality_matched_exact_set_ready",
            False,
        ),
        "quality_matching_uses_detection_labels": paired_superiority_summary.get(
            "quality_matching_uses_detection_labels",
            True,
        ),
        "quality_matching_protocol_schema": paired_superiority_summary.get(
            "quality_matching_protocol_schema",
            "",
        ),
        "quality_matching_protocol_digest": paired_superiority_summary.get(
            "quality_matching_protocol_digest",
            "",
        ),
        "quality_metric_name": paired_superiority_summary.get(
            "quality_metric_name",
            "",
        ),
        "quality_match_caliper": paired_superiority_summary.get(
            "quality_match_caliper",
            0.0,
        ),
        "minimum_matched_prompt_fraction": paired_superiority_summary.get(
            "minimum_matched_prompt_fraction",
            0.0,
        ),
        "quality_matched_rows_digest": paired_superiority_summary.get(
            "quality_matched_rows_digest",
            "",
        ),
        "paired_superiority_protocol_digest": paired_superiority_summary.get(
            "paired_superiority_protocol_digest", ""
        ),
        "paired_superiority_rows_digest": paired_superiority_summary.get(
            "paired_superiority_rows_digest", ""
        ),
        "paired_outcome_set_digest": paired_superiority_summary.get(
            "paired_outcome_set_digest", ""
        ),
        "paired_test_prompt_count": paired_superiority_summary.get(
            "paired_test_prompt_count", 0
        ),
        "paired_test_prompt_id_digest": paired_superiority_summary.get(
            "paired_test_prompt_id_digest", ""
        ),
        "paired_attack_registry_digest": paired_superiority_summary.get(
            "paired_attack_registry_digest", ""
        ),
        "method_observation_source_sha256_map": paired_superiority_summary.get(
            "method_observation_source_sha256_map", {}
        ),
        "threshold_audit_rows_digest": paired_superiority_summary.get(
            "threshold_audit_rows_digest", ""
        ),
        "claim_p_value_method": paired_superiority_summary.get(
            "claim_p_value_method", ""
        ),
        "sharp_null_diagnostic_method": paired_superiority_summary.get(
            "sharp_null_diagnostic_method", ""
        ),
        "bootstrap_analysis_schema": paired_superiority_summary.get(
            "bootstrap_analysis_schema", ""
        ),
        "bootstrap_bit_generator": paired_superiority_summary.get(
            "bootstrap_bit_generator", ""
        ),
        "bootstrap_quantile_method": paired_superiority_summary.get(
            "bootstrap_quantile_method", ""
        ),
        "bootstrap_resample_count": paired_superiority_summary.get(
            "bootstrap_resample_count", 0
        ),
        "slm_wm_mean_true_positive_rate": superiority_gate["slm_wm_mean_true_positive_rate"],
        "slm_wm_mean_false_positive_rate": superiority_gate["slm_wm_mean_false_positive_rate"],
        "best_baseline_mean_true_positive_rate": superiority_gate["best_baseline_mean_true_positive_rate"],
        "best_baseline_method_id": superiority_gate["best_baseline_method_id"],
        "slm_wm_fixed_fpr_boundary_ready": superiority_gate["slm_wm_fixed_fpr_boundary_ready"],
        "pilot_paper_claim_ready": pilot_paper_claim_ready,
        "confidence_interval_method": resolved_config.confidence_interval_method,
        "confidence_level": resolved_config.confidence_level,
        "pilot_paper_supports_superiority_claim": pilot_paper_claim_ready,
        "paper_run_claim_ready": paper_run_claim_ready,
        "paper_run_supports_superiority_claim": paper_run_claim_ready,
        "paper_claim_ready": paper_run_claim_ready,
        "probe_paper_claim_ready": probe_paper_claim_ready,
        "probe_claim_ready": probe_paper_claim_ready,
        "full_paper_claim_ready": full_paper_claim_ready,
        "pilot_claim_ready": pilot_paper_claim_ready,
        "full_claim_ready": full_paper_claim_ready,
        "supports_paper_claim": paper_run_claim_ready,
    }


def build_attack_matrix_digest(attack_rows: Iterable[Mapping[str, Any]]) -> str:
    """生成 pilot_paper 攻击矩阵摘要。"""

    stable_rows = sorted((dict(row) for row in attack_rows), key=lambda item: str(item["attack_id"]))
    return build_stable_digest(stable_rows)


def build_pilot_paper_manifest_config(
    *,
    prompt_summary: Mapping[str, Any],
    attack_rows: Iterable[Mapping[str, Any]],
    method_rows: Iterable[Mapping[str, Any]],
    template_rows: Iterable[Mapping[str, Any]],
    schema: Mapping[str, Any],
    validation_report: Mapping[str, Any],
    summary: Mapping[str, Any],
    config: PilotPaperFixedFprConfig,
) -> dict[str, Any]:
    """构建 manifest 使用的稳定配置摘要输入。"""

    return {
        "config": config.to_dict(),
        "prompt_summary_digest": build_stable_digest(dict(prompt_summary)),
        "attack_matrix_digest": build_stable_digest(tuple(dict(row) for row in attack_rows)),
        "method_registry_digest": build_stable_digest(tuple(dict(row) for row in method_rows)),
        "template_digest": build_stable_digest(tuple(dict(row) for row in template_rows)),
        "schema_digest": build_stable_digest(dict(schema)),
        "validation_report_digest": build_stable_digest(dict(validation_report)),
        "summary_digest": build_stable_digest(dict(summary)),
    }
