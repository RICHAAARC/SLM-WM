"""构建 pilot_paper 级 fixed-FPR 共同协议。

该模块只描述 pilot_paper 共同验证协议的受治理输入、阈值边界、攻击矩阵、
baseline 导入模板和声明边界。它不执行 GPU 推理。pilot_paper 与 full_paper 共用方法协议,
二者仅通过 prompt 规模、样本量和运行资源规模区分。
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

from experiments.protocol.attacks import AttackConfig, attack_config_digest
from experiments.protocol.paper_run_config import (
    FULL_PAPER_RUN_NAME,
    RUN_DEFAULTS,
    build_paper_run_config,
)
from experiments.protocol.prompts import PromptProtocolRecord
from experiments.protocol.splits import apply_split_assignments
from main.core.digest import build_stable_digest

PILOT_PAPER_PROMPT_SET = "pilot_paper"
PILOT_PAPER_PROMPT_FILE = "configs/paper_main_pilot_paper_prompts.txt"
PILOT_PAPER_PROMPT_PROTOCOL_NAME = "paper_main_pilot_paper_prompt_protocol"
PILOT_PAPER_RESULT_PROTOCOL_NAME = "pilot_paper_fixed_fpr_common_protocol"
PILOT_PAPER_RESULT_SCOPE = "pilot_paper_common_protocol"
PILOT_PAPER_CLAIM_BOUNDARY = "pilot_paper_paper_claim"
FULL_PAPER_CLAIM_BOUNDARY = "full_paper_claim_requires_full_paper_sample_scale"
PILOT_PAPER_FIXED_FPR = 0.01
FULL_PAPER_FIXED_FPR = 0.001
PAPER_RUN_FIXED_FPR = {
    PILOT_PAPER_PROMPT_SET: PILOT_PAPER_FIXED_FPR,
    FULL_PAPER_RUN_NAME: FULL_PAPER_FIXED_FPR,
}
PILOT_PAPER_BOOTSTRAP_ITERATION_COUNT = 1000
PILOT_PAPER_CONFIDENCE_LEVEL = 0.95
PILOT_PAPER_MINIMUM_CLEAN_NEGATIVE_COUNT = 100
PILOT_PAPER_METHOD_IDS = ("slm_wm_current", "tree_ring", "gaussian_shading", "shallow_diffuse", "t2smark")
PILOT_PAPER_PRIMARY_METHOD_ID = "slm_wm_current"
PILOT_PAPER_PRIMARY_BASELINE_IDS = ("tree_ring", "gaussian_shading", "shallow_diffuse", "t2smark")
PILOT_PAPER_ATTACK_RESOURCE_PROFILES = ("full_main", "full_extra")

PILOT_PAPER_REQUIRED_METRIC_FIELDS = (
    "positive_count",
    "negative_count",
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
    "result_protocol_name",
    "result_scope",
    "result_claim_scope",
    "method_id",
    "prompt_protocol_name",
    "prompt_split_digest",
    "attack_matrix_digest",
    "fixed_fpr_protocol_digest",
    "baseline_result_source",
    "baseline_result_source_digest",
    "evidence_paths",
)
PILOT_PAPER_RATE_FIELDS = (
    "true_positive_rate",
    "false_positive_rate",
    "clean_false_positive_rate",
    "attacked_false_positive_rate",
    "quality_score_mean",
    "score_retention_mean",
)
PILOT_PAPER_CI_FIELD_GROUPS = (
    ("true_positive_rate_ci_low", "true_positive_rate", "true_positive_rate_ci_high"),
    ("false_positive_rate_ci_low", "false_positive_rate", "false_positive_rate_ci_high"),
    ("clean_false_positive_rate_ci_low", "clean_false_positive_rate", "clean_false_positive_rate_ci_high"),
    ("attacked_false_positive_rate_ci_low", "attacked_false_positive_rate", "attacked_false_positive_rate_ci_high"),
    ("quality_score_ci_low", "quality_score_mean", "quality_score_ci_high"),
    ("score_retention_ci_low", "score_retention_mean", "score_retention_ci_high"),
)


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

    return f"{run_name}_paper_claim"


def _default_paper_run_value(field_name: str) -> Any:
    """从统一论文运行配置读取 dataclass 默认值。"""

    return getattr(build_paper_run_config("."), field_name)


@dataclass(frozen=True)
class PilotPaperFixedFprConfig:
    """集中描述论文运行级 fixed-FPR 共同协议配置。

    该对象属于通用工程写法: 把 prompt set、固定 FPR、bootstrap 次数和
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
    bootstrap_iteration_count: int = PILOT_PAPER_BOOTSTRAP_ITERATION_COUNT
    confidence_level: float = PILOT_PAPER_CONFIDENCE_LEVEL
    minimum_clean_negative_count: int = field(
        default_factory=lambda: int(_default_paper_run_value("minimum_clean_negative_count"))
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
        if self.bootstrap_iteration_count <= 0:
            raise ValueError("bootstrap_iteration_count 必须为正整数")
        if not 0.0 < self.confidence_level < 1.0:
            raise ValueError("confidence_level 必须位于 (0, 1)")
        if self.minimum_clean_negative_count <= 0:
            raise ValueError("minimum_clean_negative_count 必须为正整数")

    def to_dict(self) -> dict[str, Any]:
        """转为 JSON 兼容字典。"""

        payload = asdict(self)
        payload["attack_resource_profiles"] = list(self.attack_resource_profiles)
        return payload


def build_paper_fixed_fpr_config(root: str | Path = ".") -> PilotPaperFixedFprConfig:
    """按统一论文运行配置构造 fixed-FPR 共同协议配置。"""

    paper_run = build_paper_run_config(root)
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
    allowed_false_positive_count = math.floor(resolved_config.target_fpr * calibration_clean_negative_count)
    prompt_split_digest = build_stable_digest(split_record_payload)
    minimum_ready = (
        calibration_clean_negative_count >= resolved_config.minimum_clean_negative_count
        and test_clean_negative_count >= resolved_config.minimum_clean_negative_count
    )
    return {
        "prompt_set": resolved_config.prompt_set,
        "prompt_file": resolved_config.prompt_file,
        "prompt_protocol_name": resolved_config.prompt_protocol_name,
        "pilot_paper_prompt_count": len(assigned_records),
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
        "bootstrap_iteration_count": resolved_config.bootstrap_iteration_count,
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
        "bootstrap_iteration_count": resolved_config.bootstrap_iteration_count,
        "confidence_level": resolved_config.confidence_level,
        "minimum_result_positive_count": resolved_config.minimum_clean_negative_count,
        "minimum_result_negative_count": resolved_config.minimum_clean_negative_count,
        "method_ids": list(PILOT_PAPER_METHOD_IDS),
        "primary_baseline_ids": list(PILOT_PAPER_PRIMARY_BASELINE_IDS),
        "required_metric_fields": list(PILOT_PAPER_REQUIRED_METRIC_FIELDS),
        "required_source_fields": list(PILOT_PAPER_REQUIRED_SOURCE_FIELDS),
        "required_rate_fields": list(PILOT_PAPER_RATE_FIELDS),
        "ci_field_groups": [list(group) for group in PILOT_PAPER_CI_FIELD_GROUPS],
        "supports_paper_claim": True,
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
                "bootstrap_iteration_count": resolved_config.bootstrap_iteration_count,
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
                "bootstrap_iteration_count": resolved_config.bootstrap_iteration_count,
                "confidence_level": resolved_config.confidence_level,
                "required_metric_fields": list(PILOT_PAPER_REQUIRED_METRIC_FIELDS),
                "required_source_fields": list(PILOT_PAPER_REQUIRED_SOURCE_FIELDS),
                "required_result_record_path": "outputs/pilot_paper_fixed_fpr_results/pilot_paper_result_records.jsonl",
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
    }
    for field_name, minimum_count in minimum_count_fields.items():
        if _int_field(row, field_name) < minimum_count:
            issues.append(_issue(row_index, row, field_name, "pilot_paper_minimum_sample_count_required"))
    if _int_field(row, "supported_record_count") <= 0:
        issues.append(_issue(row_index, row, "supported_record_count", "positive_count_required"))
    if _int_field(row, "attack_record_count") < _int_field(row, "supported_record_count"):
        issues.append(_issue(row_index, row, "attack_record_count", "attack_record_count_must_cover_supported_count"))
    for field_name in schema["required_rate_fields"]:
        value = _float_field(row, field_name)
        if not 0.0 <= value <= 1.0:
            issues.append(_issue(row_index, row, field_name, "metric_rate_must_be_in_unit_interval"))
    for low_name, value_name, high_name in schema["ci_field_groups"]:
        low = _float_field(row, low_name)
        value = _float_field(row, value_name)
        high = _float_field(row, high_name)
        if not (0.0 <= low <= value <= high <= 1.0):
            issues.append(_issue(row_index, row, value_name, "confidence_interval_must_cover_metric"))
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
    if not math.isclose(_float_field(row, "target_fpr"), float(schema["target_fpr"]), rel_tol=0.0, abs_tol=1e-12):
        issues.append(_issue(row_index, row, "target_fpr", "target_fpr_mismatch"))
    if _int_field(row, "bootstrap_iteration_count") < int(schema["bootstrap_iteration_count"]):
        issues.append(_issue(row_index, row, "bootstrap_iteration_count", "bootstrap_iteration_count_too_small"))
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
    for row_index, row in enumerate(materialized_rows):
        row_issues: list[PilotPaperImportIssue] = []
        row_issues.extend(_validate_required_fields(row, row_index, schema))
        if not row_issues:
            row_issues.extend(_validate_counts_and_rates(row, row_index, schema))
            row_issues.extend(_validate_protocol_fields(row, row_index, schema))
            row_issues.extend(_validate_evidence_paths(row, row_index, evidence_root_path, require_existing_evidence))
        if row_issues:
            issues.extend(row_issues)
        else:
            accepted.append(row)
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
        "pilot_paper_claim_record_ready": bool(accepted) and all(_bool_field(row, "supports_paper_claim") for row in accepted),
        "supports_paper_claim": bool(accepted) and all(_bool_field(row, "supports_paper_claim") for row in accepted),
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
    missing_baseline_methods = tuple(method_id for method_id, rows in baseline_rows_by_method.items() if not rows)
    template_aligned = bool(slm_rows) and all({_record_key(row) for row in rows} == slm_keys for rows in baseline_rows_by_method.values())
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
        "method_attack_templates_aligned": template_aligned,
    }


def build_pilot_paper_common_protocol_summary(
    *,
    prompt_summary: Mapping[str, Any],
    attack_rows: Iterable[Mapping[str, Any]],
    method_rows: Iterable[Mapping[str, Any]],
    template_rows: Iterable[Mapping[str, Any]],
    import_validation_report: Mapping[str, Any],
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
    claim_coverage_ready = (
        bool(materialized_template_rows)
        and len(accepted_records) >= len(materialized_template_rows)
        and accepted_claim_record_count == len(accepted_records)
    )
    superiority_gate = build_superiority_gate_summary(accepted_records, resolved_config)
    expected_target_fpr = PAPER_RUN_FIXED_FPR.get(resolved_config.paper_run_name, PILOT_PAPER_FIXED_FPR)
    ready = (
        bool(prompt_summary.get("prompt_split_ready"))
        and bool(materialized_attack_rows)
        and method_ids == set(PILOT_PAPER_METHOD_IDS)
        and len(materialized_template_rows) == len(materialized_attack_rows) * len(materialized_method_rows)
        and math.isclose(float(resolved_config.target_fpr), expected_target_fpr, rel_tol=0.0, abs_tol=1e-12)
    )
    paper_run_claim_ready = (
        ready
        and bool(import_validation_report.get("pilot_paper_result_import_ready", False))
        and claim_coverage_ready
        and superiority_gate["superiority_gate_ready"]
    )
    pilot_paper_claim_ready = paper_run_claim_ready and resolved_config.prompt_set == PILOT_PAPER_PROMPT_SET
    full_paper_claim_ready = paper_run_claim_ready and resolved_config.prompt_set == FULL_PAPER_RUN_NAME
    return {
        "construction_unit_name": "pilot_paper_fixed_fpr_common_protocol",
        "result_protocol_name": resolved_config.result_protocol_name,
        "result_scope": resolved_config.result_scope,
        "result_claim_scope": resolved_config.result_claim_scope,
        "paper_claim_scale": resolved_config.prompt_set,
        "full_paper_claim_boundary": FULL_PAPER_CLAIM_BOUNDARY,
        "pilot_paper_common_protocol_ready": ready,
        "pilot_paper_prompt_count": prompt_summary.get("pilot_paper_prompt_count", 0),
        "paper_prompt_count": prompt_summary.get("pilot_paper_prompt_count", 0),
        "pilot_paper_prompt_split_ready": prompt_summary.get("prompt_split_ready", False),
        "paper_prompt_split_ready": prompt_summary.get("prompt_split_ready", False),
        "pilot_paper_target_fpr": resolved_config.target_fpr,
        "paper_target_fpr": resolved_config.target_fpr,
        "expected_target_fpr": expected_target_fpr,
        "pilot_paper_negative_count_minimum_required": resolved_config.minimum_clean_negative_count,
        "minimum_result_positive_count": resolved_config.minimum_clean_negative_count,
        "minimum_result_negative_count": resolved_config.minimum_clean_negative_count,
        "pilot_paper_attack_count": len(materialized_attack_rows),
        "pilot_paper_method_count": len(materialized_method_rows),
        "pilot_paper_import_template_count": len(materialized_template_rows),
        "pilot_paper_result_import_ready": bool(import_validation_report.get("pilot_paper_result_import_ready", False)),
        "accepted_pilot_paper_import_count": int(import_validation_report.get("accepted_pilot_paper_import_count", 0)),
        "accepted_pilot_paper_claim_record_count": accepted_claim_record_count,
        "pilot_paper_claim_record_ready": claim_coverage_ready,
        "pilot_paper_evidence_coverage_ready": claim_coverage_ready,
        "pilot_paper_effectiveness_gate_ready": superiority_gate["superiority_gate_ready"],
        "pilot_paper_effectiveness_gate_reason": superiority_gate["superiority_gate_reason"],
        "slm_wm_mean_true_positive_rate": superiority_gate["slm_wm_mean_true_positive_rate"],
        "slm_wm_mean_false_positive_rate": superiority_gate["slm_wm_mean_false_positive_rate"],
        "best_baseline_mean_true_positive_rate": superiority_gate["best_baseline_mean_true_positive_rate"],
        "best_baseline_method_id": superiority_gate["best_baseline_method_id"],
        "slm_wm_fixed_fpr_boundary_ready": superiority_gate["slm_wm_fixed_fpr_boundary_ready"],
        "pilot_paper_claim_ready": pilot_paper_claim_ready,
        "bootstrap_iteration_count": resolved_config.bootstrap_iteration_count,
        "confidence_level": resolved_config.confidence_level,
        "pilot_paper_supports_superiority_claim": pilot_paper_claim_ready,
        "paper_run_claim_ready": paper_run_claim_ready,
        "paper_run_supports_superiority_claim": paper_run_claim_ready,
        "paper_claim_ready": paper_run_claim_ready,
        "full_paper_claim_ready": full_paper_claim_ready,
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
