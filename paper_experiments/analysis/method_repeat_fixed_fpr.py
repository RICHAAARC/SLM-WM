"""逐方法、逐随机化重复重算论文 fixed-FPR 阈值.

该模块只实现纯分析逻辑, 不读取文件、不写出产物. 调用方必须先从精确9重复
聚合包中提取 raw observation、阈值声明及其嵌套成员字节摘要. 这里随后强制
执行 `9个重复与5个方法` 的笛卡尔积校验, 并在每个 method-repeat 内
独立使用 calibration clean negatives 冻结阈值. 任何跨 repeat 合并 calibration
后只计算5个阈值的输入都无法通过精确键集合检查.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import PurePosixPath
import re
from typing import Any, Iterable, Mapping

from experiments.protocol.fixed_fpr_observation_audit import (
    FORMAL_THRESHOLD_SOURCE,
    audit_fixed_fpr_observation_threshold,
)
from experiments.protocol.formal_randomization import (
    formal_randomization_protocol_record,
    formal_randomization_repeat_ids,
    formal_randomization_repeats,
    resolve_formal_randomization_repeat,
)
from experiments.protocol.paper_run_config import (
    RUN_EXPECTED_PROMPT_COUNTS,
    normalize_paper_run_name,
    validate_frozen_paper_run_target_fpr,
)
from experiments.protocol.prompts import build_prompt_id, normalize_prompt_text
from experiments.protocol.splits import build_group_split_counts
from experiments.runners.image_only_dataset_runtime import (
    FrozenEvidenceProtocol,
    calibrate_complete_evidence_protocol,
)
from main.core.digest import build_stable_digest
from paper_experiments.analysis.fixed_fpr_threshold_audit import (
    FIXED_FPR_THRESHOLD_METHOD_IDS,
    MAIN_THRESHOLD_SOURCE,
    audit_baseline_fixed_fpr,
    audit_main_method_fixed_fpr,
)


METHOD_REPEAT_THRESHOLD_SCHEMA = "method_repeat_fixed_fpr_threshold_record"
METHOD_REPEAT_FAIRNESS_SCHEMA = "method_repeat_randomization_fairness_record"
METHOD_REPEAT_THRESHOLD_REPORT_SCHEMA = "method_repeat_fixed_fpr_threshold_report"
METHOD_REPEAT_THRESHOLD_CALCULATION_UNIT = "method_repeat"
METHOD_REPEAT_THRESHOLD_COUNT = len(formal_randomization_repeat_ids()) * len(
    FIXED_FPR_THRESHOLD_METHOD_IDS
)
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
FORMAL_BASE_LATENT_GENERATION_PROTOCOL = (
    "device_independent_sha256_box_muller_cpu_dtype_cast_then_device_transfer_v1"
)
FORMAL_BASE_LATENT_DTYPE = "torch.float16"
FORMAL_BASE_LATENT_SHAPE = (1, 16, 64, 64)

METHOD_LEAF_PACKAGE_FAMILY = {
    "slm_wm": "image_only_dataset_runtime",
    "tree_ring": "method_faithful_tree_ring",
    "gaussian_shading": "method_faithful_gaussian_shading",
    "shallow_diffuse": "method_faithful_shallow_diffuse",
    "t2smark": "official_reference_t2smark",
}

_PROMPT_IDENTITY_FIELDS = (
    "randomization_repeat_id",
    "generation_seed_index",
    "generation_seed_offset",
    "generation_seed_random",
    "watermark_key_index",
    "watermark_key_seed_random",
    "watermark_key_material_digest_random",
    "formal_randomization_protocol_digest",
    "formal_randomization_identity_digest_random",
    "base_latent_content_digest_random",
    "base_latent_identity_digest_random",
)
_PROMPT_IDENTITY_INTEGER_FIELDS = (
    "generation_seed_index",
    "generation_seed_offset",
    "generation_seed_random",
    "watermark_key_index",
    "watermark_key_seed_random",
)
_PROMPT_IDENTITY_SHA256_FIELDS = (
    "watermark_key_material_digest_random",
    "formal_randomization_protocol_digest",
    "formal_randomization_identity_digest_random",
    "base_latent_content_digest_random",
    "base_latent_identity_digest_random",
)
_MAIN_BASE_LATENT_PROTOCOL_FIELDS = (
    "base_latent_generation_protocol",
    "base_latent_keyed_prg_version",
    "base_latent_dtype",
    "base_latent_shape",
)


class MethodRepeatFixedFprError(ValueError):
    """表示逐方法逐重复 fixed-FPR 输入没有满足论文协议."""


def _require_sha256(value: str, field_name: str) -> None:
    """集中校验嵌套包和成员的字节级 SHA-256."""

    if SHA256_PATTERN.fullmatch(value) is None:
        raise MethodRepeatFixedFprError(f"{field_name} 必须是小写 SHA-256")


def _expected_repeat_component_member(randomization_repeat_id: str) -> str:
    """返回 aggregate 内单重复组件的规范成员路径."""

    return f"repeat_components/{randomization_repeat_id}.zip"


def _expected_leaf_package_member(
    randomization_repeat_id: str,
    package_family: str,
) -> str:
    """返回 repeat component 内 leaf ZIP 的规范成员路径."""

    return (
        f"randomization_repeat_evidence/{randomization_repeat_id}/"
        f"leaf_packages/{package_family}.zip"
    )


def _expected_observation_member(paper_run_name: str, method_id: str) -> str:
    """返回各主表方法 raw observation 的规范 leaf 成员路径."""

    if method_id == "slm_wm":
        return (
            f"outputs/image_only_dataset_runtime/{paper_run_name}/"
            "image_only_detection_records.jsonl"
        )
    if method_id == "t2smark":
        return (
            f"outputs/t2smark_formal_reproduction/{paper_run_name}/"
            "t2smark_adapter/baseline_observations.json"
        )
    return (
        f"outputs/external_baseline_method_faithful/{paper_run_name}/"
        f"split_observations/{method_id}_baseline_observations.json"
    )


def _expected_threshold_declaration_member(
    paper_run_name: str,
    method_id: str,
) -> str:
    """返回用于核对 producer 阈值声明的规范 leaf 成员路径."""

    if method_id == "slm_wm":
        return (
            f"outputs/image_only_dataset_runtime/{paper_run_name}/"
            "frozen_evidence_protocol.json"
        )
    if method_id == "t2smark":
        return (
            f"outputs/t2smark_formal_reproduction/{paper_run_name}/"
            "t2smark_formal_import_candidate_records.jsonl"
        )
    return (
        f"outputs/external_baseline_method_faithful/{paper_run_name}/"
        f"split_observations/{method_id}_baseline_transfer_manifest.json"
    )


@dataclass(frozen=True)
class MethodRepeatObservationSource:
    """保存一个 method-repeat 的 raw observation 与嵌套包字节绑定.

    该对象不接受解压目录作为来源身份.repeat component、leaf package、
    observation 成员和阈值声明成员都必须同时给出规范归档路径及字节摘要.
    ``declared_threshold_protocol`` 是调用方从 producer declaration 规范化后的
    映射.四个 baseline 统一使用 ``calibrated_detection_threshold`` 与
    ``threshold_digest``;T2SMark 不得把 candidate record 的字段误读成
    adapter manifest 顶层 ``threshold``.
    """

    paper_run_name: str
    method_id: str
    randomization_repeat_id: str
    generation_model_id: str
    generation_model_revision: str
    randomization_aggregate_package_sha256: str
    randomization_aggregate_digest: str
    common_code_version: str
    repeat_component_archive_member: str
    randomization_repeat_component_sha256: str
    randomization_repeat_evidence_manifest_digest: str
    component_content_digest: str
    leaf_package_family: str
    leaf_package_archive_member: str
    leaf_package_sha256: str
    observation_archive_member: str
    observation_source_sha256: str
    threshold_declaration_archive_member: str
    threshold_declaration_source_sha256: str
    declared_threshold_protocol: Mapping[str, Any]
    observation_rows: tuple[Mapping[str, Any], ...]

    def __post_init__(self) -> None:
        """集中校验不可由统计函数修复的来源身份边界."""

        run_name = normalize_paper_run_name(self.paper_run_name)
        if self.method_id not in FIXED_FPR_THRESHOLD_METHOD_IDS:
            raise MethodRepeatFixedFprError("method_id 不属于五个主表方法")
        resolve_formal_randomization_repeat(self.randomization_repeat_id)
        if not self.generation_model_id or not self.generation_model_revision:
            raise MethodRepeatFixedFprError("方法来源必须绑定完整生成模型身份")
        expected_family = METHOD_LEAF_PACKAGE_FAMILY[self.method_id]
        expected_paths = {
            "repeat_component_archive_member": _expected_repeat_component_member(
                self.randomization_repeat_id
            ),
            "leaf_package_archive_member": _expected_leaf_package_member(
                self.randomization_repeat_id,
                expected_family,
            ),
            "observation_archive_member": _expected_observation_member(
                run_name,
                self.method_id,
            ),
            "threshold_declaration_archive_member": (
                _expected_threshold_declaration_member(
                    run_name,
                    self.method_id,
                )
            ),
        }
        if self.leaf_package_family != expected_family:
            raise MethodRepeatFixedFprError("method 与 leaf package family 不匹配")
        for field_name, expected_value in expected_paths.items():
            actual_value = str(getattr(self, field_name))
            if PurePosixPath(actual_value).as_posix() != expected_value:
                raise MethodRepeatFixedFprError(
                    f"{field_name} 未匹配规范嵌套成员路径"
                )
        for field_name in (
            "randomization_aggregate_package_sha256",
            "randomization_aggregate_digest",
            "randomization_repeat_component_sha256",
            "randomization_repeat_evidence_manifest_digest",
            "component_content_digest",
            "leaf_package_sha256",
            "observation_source_sha256",
            "threshold_declaration_source_sha256",
        ):
            _require_sha256(str(getattr(self, field_name)), field_name)
        if re.fullmatch(r"[0-9a-f]{40}", self.common_code_version) is None:
            raise MethodRepeatFixedFprError(
                "common_code_version 必须是40位小写 Git commit"
            )
        if not isinstance(self.declared_threshold_protocol, Mapping):
            raise MethodRepeatFixedFprError("阈值声明必须是 mapping")
        rows = tuple(dict(row) for row in self.observation_rows)
        if not rows:
            raise MethodRepeatFixedFprError("raw observation 不得为空")
        object.__setattr__(self, "paper_run_name", run_name)
        object.__setattr__(
            self,
            "declared_threshold_protocol",
            dict(self.declared_threshold_protocol),
        )
        object.__setattr__(self, "observation_rows", rows)


def build_prompt_split_contract(
    prompt_rows: Iterable[Mapping[str, Any]],
    *,
    paper_run_name: str,
) -> dict[str, Any]:
    """构造论文 Prompt exact-set 与 split 的规范纯数据契约."""

    run_name = normalize_paper_run_name(paper_run_name)
    rows = tuple(dict(row) for row in prompt_rows)
    expected_count = int(RUN_EXPECTED_PROMPT_COUNTS[run_name])
    if len(rows) != expected_count:
        raise MethodRepeatFixedFprError("Prompt 数量未匹配论文运行层级")
    normalized: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        prompt_id = str(row.get("prompt_id", "")).strip()
        prompt_digest = str(row.get("prompt_digest", ""))
        prompt_text = str(row.get("prompt_text", ""))
        split = str(row.get("split", ""))
        prompt_index = row.get("prompt_index")
        normalized_prompt_text = normalize_prompt_text(prompt_text)
        if (
            not prompt_id
            or not normalized_prompt_text
            or prompt_text != normalized_prompt_text
            or type(prompt_index) is not int
            or split not in {"dev", "calibration", "test"}
        ):
            raise MethodRepeatFixedFprError(
                f"Prompt 契约字段无效: row={row_index}"
            )
        _require_sha256(prompt_digest, "prompt_digest")
        if prompt_digest != build_stable_digest(
            {"prompt_text": normalized_prompt_text}
        ):
            raise MethodRepeatFixedFprError(
                f"Prompt digest 无法由文本重建: row={row_index}"
            )
        if prompt_id != build_prompt_id(
            run_name,
            prompt_index,
            normalized_prompt_text,
        ):
            raise MethodRepeatFixedFprError(
                f"Prompt ID 无法由索引和文本重建: row={row_index}"
            )
        normalized.append(
            {
                "prompt_id": prompt_id,
                "prompt_index": prompt_index,
                "prompt_text": normalized_prompt_text,
                "split": split,
                "prompt_digest": prompt_digest,
            }
        )
    ordered = tuple(sorted(normalized, key=lambda row: int(row["prompt_index"])))
    if [int(row["prompt_index"]) for row in ordered] != list(
        range(expected_count)
    ):
        raise MethodRepeatFixedFprError("Prompt index 必须连续且从0开始")
    prompt_ids = tuple(str(row["prompt_id"]) for row in ordered)
    if len(set(prompt_ids)) != expected_count:
        raise MethodRepeatFixedFprError("Prompt ID 必须唯一")
    split_counts = {
        split: sum(row["split"] == split for row in ordered)
        for split in ("dev", "calibration", "test")
    }
    if split_counts != build_group_split_counts(expected_count):
        raise MethodRepeatFixedFprError("Prompt split 数量未匹配冻结比例")
    payload = {
        "paper_claim_scale": run_name,
        "prompt_count": expected_count,
        "split_counts": split_counts,
        "prompt_rows": [dict(row) for row in ordered],
    }
    return {
        **payload,
        "prompt_protocol_digest": build_stable_digest(payload),
    }


def _clean_unattacked_negative(row: Mapping[str, Any]) -> bool:
    """判断一行是否为阈值校准允许消费的未攻击 clean negative."""

    return bool(
        row.get("sample_role") == "clean_negative"
        and not str(row.get("attack_id", "")).strip()
        and str(row.get("attack_family", "")).strip() in {"", "clean"}
        and str(row.get("attack_name", "")).strip()
        in {"", "none", "clean", "clean_none"}
    )


def _validate_source_rows(
    source: MethodRepeatObservationSource,
    *,
    prompt_contract: Mapping[str, Any],
    expected_model_id: str,
    expected_model_revision: str,
    expected_base_seed: int,
) -> tuple[
    tuple[dict[str, Any], ...],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
]:
    """校验一个来源的 Prompt exact-set、模型和完整随机身份."""

    if (
        source.generation_model_id != expected_model_id
        or source.generation_model_revision != expected_model_revision
    ):
        raise MethodRepeatFixedFprError("方法来源模型身份未匹配共同论文模型")
    repeat = resolve_formal_randomization_repeat(source.randomization_repeat_id)
    randomization_protocol = formal_randomization_protocol_record()
    protocol_digest = str(
        randomization_protocol["formal_randomization_protocol_digest"]
    )
    base_latent_prg_version = str(
        randomization_protocol["base_latent_keyed_prg_version"]
    )
    prompt_rows = tuple(prompt_contract["prompt_rows"])
    prompt_by_id = {str(row["prompt_id"]): dict(row) for row in prompt_rows}
    identities: dict[str, dict[str, Any]] = {}
    main_base_latent_protocols: dict[str, dict[str, Any]] = {}
    clean_rows_by_prompt: dict[str, list[dict[str, Any]]] = {
        prompt_id: [] for prompt_id in prompt_by_id
    }
    normalized_rows = tuple(source.observation_rows)
    for row_index, row in enumerate(normalized_rows):
        prompt_id = str(row.get("prompt_id", ""))
        prompt = prompt_by_id.get(prompt_id)
        if prompt is None or str(row.get("split", "")) != prompt["split"]:
            raise MethodRepeatFixedFprError(
                f"observation Prompt 或 split 未匹配 exact-set: row={row_index}"
            )
        if "prompt_text" in row and row.get("prompt_text") is not None:
            if str(row["prompt_text"]) != str(prompt["prompt_text"]):
                raise MethodRepeatFixedFprError(
                    "observation Prompt 文本与规范契约不一致"
                )
        if source.method_id != "slm_wm" and str(
            row.get("baseline_id", "")
        ) != source.method_id:
            raise MethodRepeatFixedFprError("baseline observation 方法身份不一致")
        for field_name in ("generation_model_id", "generation_model_revision"):
            if field_name in row and str(row[field_name]).strip():
                expected_value = (
                    expected_model_id
                    if field_name == "generation_model_id"
                    else expected_model_revision
                )
                if str(row[field_name]) != expected_value:
                    raise MethodRepeatFixedFprError("observation 模型身份不一致")
        identity = {field_name: row.get(field_name) for field_name in _PROMPT_IDENTITY_FIELDS}
        if any(
            type(identity[field_name]) is not int
            for field_name in _PROMPT_IDENTITY_INTEGER_FIELDS
        ):
            raise MethodRepeatFixedFprError("随机身份整数类型无效")
        expected_generation_seed = (
            expected_base_seed
            + int(prompt["prompt_index"])
            + repeat.generation_seed_offset
        )
        if (
            int(identity["generation_seed_random"]) < 0
            or identity["generation_seed_random"] != expected_generation_seed
        ):
            raise MethodRepeatFixedFprError(
                "generation seed 未匹配冻结 base seed 公式"
            )
        for field_name in _PROMPT_IDENTITY_SHA256_FIELDS:
            _require_sha256(str(identity[field_name]), field_name)
        expected_repeat_values = {
            "randomization_repeat_id": repeat.randomization_repeat_id,
            "generation_seed_index": repeat.generation_seed_index,
            "generation_seed_offset": repeat.generation_seed_offset,
            "watermark_key_index": repeat.watermark_key_index,
            "formal_randomization_protocol_digest": protocol_digest,
        }
        if any(
            identity[field_name] != expected_value
            for field_name, expected_value in expected_repeat_values.items()
        ):
            raise MethodRepeatFixedFprError("observation 随机化 repeat 身份不一致")
        digest_payload = {
            field_name: identity[field_name]
            for field_name in (
                "randomization_repeat_id",
                "generation_seed_index",
                "generation_seed_offset",
                "watermark_key_index",
                "generation_seed_random",
                "watermark_key_seed_random",
                "formal_randomization_protocol_digest",
                "watermark_key_material_digest_random",
            )
        }
        if build_stable_digest(digest_payload) != identity[
            "formal_randomization_identity_digest_random"
        ]:
            raise MethodRepeatFixedFprError("formal randomization identity 摘要不可重建")
        if source.method_id == "slm_wm":
            raw_base_protocol = {
                field_name: row.get(field_name)
                for field_name in _MAIN_BASE_LATENT_PROTOCOL_FIELDS
            }
            raw_shape = raw_base_protocol["base_latent_shape"]
            if (
                raw_base_protocol["base_latent_generation_protocol"]
                != FORMAL_BASE_LATENT_GENERATION_PROTOCOL
                or raw_base_protocol["base_latent_keyed_prg_version"]
                != base_latent_prg_version
                or raw_base_protocol["base_latent_dtype"]
                != FORMAL_BASE_LATENT_DTYPE
                or not isinstance(raw_shape, (tuple, list))
                or any(type(value) is not int or value <= 0 for value in raw_shape)
                or tuple(raw_shape) != FORMAL_BASE_LATENT_SHAPE
            ):
                raise MethodRepeatFixedFprError(
                    "主方法 base latent 协议、PRG、dtype 或 shape 无效"
                )
            normalized_base_protocol = {
                **raw_base_protocol,
                "base_latent_shape": list(raw_shape),
            }
            base_identity_payload = {
                "generation_seed_random": identity["generation_seed_random"],
                **normalized_base_protocol,
                "formal_randomization_protocol_digest": identity[
                    "formal_randomization_protocol_digest"
                ],
                "base_latent_content_digest_random": identity[
                    "base_latent_content_digest_random"
                ],
            }
            if build_stable_digest(base_identity_payload) != identity[
                "base_latent_identity_digest_random"
            ]:
                raise MethodRepeatFixedFprError(
                    "主方法 base latent identity 摘要无法由精确 payload 重建"
                )
            previous_base_protocol = main_base_latent_protocols.setdefault(
                prompt_id,
                normalized_base_protocol,
            )
            if previous_base_protocol != normalized_base_protocol:
                raise MethodRepeatFixedFprError(
                    "同一主方法 Prompt 的 base latent 协议发生漂移"
                )
        previous = identities.setdefault(prompt_id, identity)
        if previous != identity:
            raise MethodRepeatFixedFprError("同一方法内同 Prompt 随机身份发生漂移")
        if _clean_unattacked_negative(row):
            clean_rows_by_prompt[prompt_id].append(row)
    if any(len(rows) != 1 for rows in clean_rows_by_prompt.values()):
        raise MethodRepeatFixedFprError(
            "每个 Prompt 必须精确提供一条未攻击 clean negative"
        )
    expected_calibration_ids = {
        str(row["prompt_id"])
        for row in prompt_rows
        if row["split"] == "calibration"
    }
    actual_calibration_ids = {
        prompt_id
        for prompt_id, rows in clean_rows_by_prompt.items()
        if rows[0]["split"] == "calibration"
    }
    if actual_calibration_ids != expected_calibration_ids:
        raise MethodRepeatFixedFprError(
            "calibration clean negative 未精确覆盖冻结 Prompt 集合"
        )
    clean_rows = tuple(
        clean_rows_by_prompt[str(row["prompt_id"])][0]
        for row in prompt_rows
    )
    if source.method_id == "slm_wm" and set(
        main_base_latent_protocols
    ) != set(prompt_by_id):
        raise MethodRepeatFixedFprError(
            "主方法 base latent 协议未覆盖完整 Prompt exact-set"
        )
    return clean_rows, identities, main_base_latent_protocols


def _validate_cross_repeat_randomization(
    identity_by_source: Mapping[tuple[str, str], Mapping[str, Mapping[str, Any]]],
    main_base_latent_protocol_by_repeat: Mapping[
        str,
        Mapping[str, Mapping[str, Any]],
    ],
    prompt_contract: Mapping[str, Any],
    *,
    expected_base_seed: int,
) -> tuple[dict[str, Any], ...]:
    """核验五方法公平身份以及3种 seed 与3种 key 的真实交叉."""

    prompt_rows = tuple(prompt_contract["prompt_rows"])
    method_ids = tuple(FIXED_FPR_THRESHOLD_METHOD_IDS)
    fairness_records: list[dict[str, Any]] = []
    for repeat in formal_randomization_repeats():
        for prompt in prompt_rows:
            prompt_id = str(prompt["prompt_id"])
            identities = tuple(
                dict(identity_by_source[(repeat.randomization_repeat_id, method_id)][prompt_id])
                for method_id in method_ids
            )
            if any(identity != identities[0] for identity in identities[1:]):
                raise MethodRepeatFixedFprError(
                    "同一 repeat/Prompt 的五方法 seed、key 或 base latent 不一致"
                )
            payload = {
                "record_schema": METHOD_REPEAT_FAIRNESS_SCHEMA,
                "randomization_repeat_id": repeat.randomization_repeat_id,
                "prompt_id": prompt_id,
                "prompt_index": int(prompt["prompt_index"]),
                "split": str(prompt["split"]),
                "expected_base_seed": expected_base_seed,
                **identities[0],
                **main_base_latent_protocol_by_repeat[
                    repeat.randomization_repeat_id
                ][prompt_id],
                "method_ids": list(method_ids),
                "fairness_identity_match_ready": True,
                "supports_paper_claim": False,
            }
            payload["fairness_record_digest"] = build_stable_digest(payload)
            fairness_records.append(payload)
    by_repeat_prompt = {
        (str(row["randomization_repeat_id"]), str(row["prompt_id"])): row
        for row in fairness_records
    }
    for prompt in prompt_rows:
        prompt_id = str(prompt["prompt_id"])
        prompt_records = tuple(
            by_repeat_prompt[(repeat.randomization_repeat_id, prompt_id)]
            for repeat in formal_randomization_repeats()
        )
        generation_bases = {
            int(row["generation_seed_random"])
            - int(row["generation_seed_offset"])
            for row in prompt_records
        }
        expected_generation_base = expected_base_seed + int(
            prompt["prompt_index"]
        )
        if generation_bases != {expected_generation_base}:
            raise MethodRepeatFixedFprError(
                "跨 repeat Prompt 没有共享冻结 base seed 公式"
            )
        for seed_index in range(3):
            rows = tuple(
                row
                for row in prompt_records
                if int(row["generation_seed_index"]) == seed_index
            )
            if len(rows) != 3 or len(
                {
                    (
                        row["generation_seed_random"],
                        row["base_latent_content_digest_random"],
                        row["base_latent_identity_digest_random"],
                    )
                    for row in rows
                }
            ) != 1:
                raise MethodRepeatFixedFprError(
                    "同一 generation seed 跨 key 的 base latent 不一致"
                )
        for key_index in range(3):
            rows = tuple(
                row
                for row in prompt_records
                if int(row["watermark_key_index"]) == key_index
            )
            if len(rows) != 3 or len(
                {
                    (
                        row["watermark_key_seed_random"],
                        row["watermark_key_material_digest_random"],
                    )
                    for row in rows
                }
            ) != 1:
                raise MethodRepeatFixedFprError(
                    "同一 watermark key 跨 generation seed 的身份不一致"
                )
        if len(
            {
                row["base_latent_content_digest_random"]
                for row in prompt_records
            }
        ) != 3:
            raise MethodRepeatFixedFprError("三个 generation seed 未产生三个 base latent 身份")
        if len(
            {row["watermark_key_seed_random"] for row in prompt_records}
        ) != 3:
            raise MethodRepeatFixedFprError("三个 watermark key 身份没有真实区分")
    return tuple(fairness_records)


def _same_value(left: Any, right: Any) -> bool:
    """比较阈值声明字段, 浮点使用严格绝对容差."""

    if isinstance(left, int | float) and isinstance(right, int | float):
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-12)
    return left == right


def _recompute_threshold_record(
    source: MethodRepeatObservationSource,
    *,
    prompt_contract: Mapping[str, Any],
    fairness_identity_digest: str,
    target_fpr: float,
    main_rescue_margin_low: float,
) -> dict[str, Any]:
    """在一个 method-repeat 内独立重算并核对 producer 阈值声明."""

    rows = tuple(source.observation_rows)
    prompt_rows = tuple(prompt_contract["prompt_rows"])
    calibration_ids = {
        str(row["prompt_id"])
        for row in prompt_rows
        if row["split"] == "calibration"
    }
    test_ids = {
        str(row["prompt_id"])
        for row in prompt_rows
        if row["split"] == "test"
    }
    calibration_rows = tuple(
        row
        for row in rows
        if _clean_unattacked_negative(row)
        and str(row["prompt_id"]) in calibration_ids
    )
    expected_calibration_count = len(calibration_ids)
    expected_test_count = len(test_ids)
    declaration = dict(source.declared_threshold_protocol)
    if source.method_id == "slm_wm":
        recomputed = calibrate_complete_evidence_protocol(
            calibration_rows,
            target_fpr,
            main_rescue_margin_low,
        )
        recomputed_protocol = recomputed.to_dict()
        expected_declaration_fields = {
            field.name
            for field in FrozenEvidenceProtocol.__dataclass_fields__.values()
        }
        if set(declaration) != expected_declaration_fields:
            raise MethodRepeatFixedFprError("主方法阈值声明字段集合不完整")
        if any(
            not _same_value(declaration[field_name], expected_value)
            for field_name, expected_value in recomputed_protocol.items()
        ):
            raise MethodRepeatFixedFprError(
                "主方法阈值声明与 raw observation 重算不一致"
            )
        audit = audit_main_method_fixed_fpr(
            rows,
            declaration,
            observation_source_sha256=source.observation_source_sha256,
            target_fpr=target_fpr,
            expected_calibration_negative_count=expected_calibration_count,
            expected_test_negative_count=expected_test_count,
        )
        if (
            not recomputed.geometry_protocol_calibration_ready
            or recomputed.geometry_calibration_negative_count
            != expected_calibration_count
            or recomputed.registration_calibration_negative_count
            != expected_calibration_count
            or recomputed.sync_calibration_negative_count
            != expected_calibration_count
        ):
            raise MethodRepeatFixedFprError("主方法完整几何 rescue 校准字段不完整")
        false_positive_count = recomputed.calibration_false_positive_count
        threshold_protocol = recomputed_protocol
    else:
        primitive = audit_fixed_fpr_observation_threshold(
            rows,
            target_fpr=target_fpr,
            expected_calibration_negative_count=expected_calibration_count,
        )
        try:
            declared_threshold = float(
                declaration["calibrated_detection_threshold"]
            )
            declared_digest = str(declaration["threshold_digest"])
        except (KeyError, TypeError, ValueError) as exc:
            raise MethodRepeatFixedFprError("baseline 阈值声明缺少数值或摘要") from exc
        audit = audit_baseline_fixed_fpr(
            source.method_id,
            rows,
            observation_source_sha256=source.observation_source_sha256,
            target_fpr=target_fpr,
            expected_calibration_negative_count=expected_calibration_count,
            expected_test_negative_count=expected_test_count,
            declared_threshold=declared_threshold,
            declared_threshold_digest=declared_digest,
        )
        false_positive_count = primitive.calibration_false_positive_count
        threshold_protocol = {
            "threshold": primitive.frozen_threshold,
            "threshold_source": FORMAL_THRESHOLD_SOURCE,
            "threshold_digest": primitive.threshold_digest,
            "target_fpr": target_fpr,
            "calibration_negative_count": primitive.calibration_negative_count,
            "calibration_false_positive_count": false_positive_count,
            "decision_scope": "score_greater_than_or_equal_to_threshold",
        }
    if audit["fixed_fpr_threshold_ready"] is not True:
        raise MethodRepeatFixedFprError("method-repeat fixed-FPR 重算或逐条判定未通过")
    allowed_false_positive_count = max(
        0,
        math.floor(target_fpr * (expected_calibration_count + 1)) - 1,
    )
    repeat = resolve_formal_randomization_repeat(source.randomization_repeat_id)
    payload = {
        "record_schema": METHOD_REPEAT_THRESHOLD_SCHEMA,
        "paper_claim_scale": source.paper_run_name,
        "threshold_calculation_unit": METHOD_REPEAT_THRESHOLD_CALCULATION_UNIT,
        "method_id": source.method_id,
        **repeat.to_dict(),
        "formal_randomization_protocol_digest": formal_randomization_protocol_record()[
            "formal_randomization_protocol_digest"
        ],
        "generation_model_id": source.generation_model_id,
        "generation_model_revision": source.generation_model_revision,
        "target_fpr": target_fpr,
        "threshold_source": (
            MAIN_THRESHOLD_SOURCE
            if source.method_id == "slm_wm"
            else FORMAL_THRESHOLD_SOURCE
        ),
        "calibration_clean_negative_count": expected_calibration_count,
        "test_clean_negative_count": expected_test_count,
        "allowed_calibration_false_positive_count": allowed_false_positive_count,
        "calibration_false_positive_count": false_positive_count,
        "calibration_false_positive_rate": (
            false_positive_count / expected_calibration_count
        ),
        "calibrated_detection_threshold": audit[
            "calibrated_detection_threshold"
        ],
        "threshold_protocol": threshold_protocol,
        "threshold_digest": audit["threshold_digest"],
        "prompt_protocol_digest": prompt_contract["prompt_protocol_digest"],
        "calibration_prompt_ids_digest": build_stable_digest(
            sorted(calibration_ids)
        ),
        "test_prompt_ids_digest": build_stable_digest(sorted(test_ids)),
        "fairness_identity_digest": fairness_identity_digest,
        "randomization_aggregate_package_sha256": (
            source.randomization_aggregate_package_sha256
        ),
        "randomization_aggregate_digest": source.randomization_aggregate_digest,
        "common_code_version": source.common_code_version,
        "repeat_component_archive_member": (
            source.repeat_component_archive_member
        ),
        "randomization_repeat_component_sha256": (
            source.randomization_repeat_component_sha256
        ),
        "randomization_repeat_evidence_manifest_digest": (
            source.randomization_repeat_evidence_manifest_digest
        ),
        "component_content_digest": source.component_content_digest,
        "leaf_package_family": source.leaf_package_family,
        "leaf_package_archive_member": source.leaf_package_archive_member,
        "leaf_package_sha256": source.leaf_package_sha256,
        "observation_archive_member": source.observation_archive_member,
        "observation_source_sha256": source.observation_source_sha256,
        "observation_rows_digest": build_stable_digest(rows),
        "threshold_declaration_archive_member": (
            source.threshold_declaration_archive_member
        ),
        "threshold_declaration_source_sha256": (
            source.threshold_declaration_source_sha256
        ),
        "threshold_declaration_protocol_digest": build_stable_digest(
            declaration
        ),
        "fixed_fpr_threshold_ready": True,
        "supports_paper_claim": False,
    }
    payload["method_repeat_threshold_record_digest"] = build_stable_digest(
        payload
    )
    return payload


def recompute_exact_method_repeat_fixed_fpr(
    sources: Iterable[MethodRepeatObservationSource],
    *,
    prompt_rows: Iterable[Mapping[str, Any]],
    paper_run_name: str,
    target_fpr: float,
    expected_model_id: str,
    expected_model_revision: str,
    expected_base_seed: int,
    main_rescue_margin_low: float,
) -> dict[str, Any]:
    """重算精确45个 method-repeat 阈值并返回可持久化纯数据.

    阈值计算单位固定为 ``method_repeat``. 函数先完成45个来源的 exact-set
    校验, 再校验五方法公平身份, 最后逐键重算. 它从不先合并9个 repeat 的
    calibration 分数, 也不会修改调用方提供的 observation 行.
    """

    run_name = normalize_paper_run_name(paper_run_name)
    resolved_target_fpr = validate_frozen_paper_run_target_fpr(
        run_name,
        target_fpr,
    )
    if not expected_model_id or not expected_model_revision:
        raise MethodRepeatFixedFprError("共同生成模型身份不得为空")
    if type(expected_base_seed) is not int or expected_base_seed < 0:
        raise MethodRepeatFixedFprError(
            "expected_base_seed 必须是非负整数"
        )
    if not math.isfinite(main_rescue_margin_low) or main_rescue_margin_low >= 0.0:
        raise MethodRepeatFixedFprError("main_rescue_margin_low 必须是负有限数")
    prompt_contract = build_prompt_split_contract(
        prompt_rows,
        paper_run_name=run_name,
    )
    materialized = tuple(sources)
    expected_keys = tuple(
        (repeat_id, method_id)
        for repeat_id in formal_randomization_repeat_ids()
        for method_id in FIXED_FPR_THRESHOLD_METHOD_IDS
    )
    source_by_key: dict[tuple[str, str], MethodRepeatObservationSource] = {}
    identity_by_source: dict[
        tuple[str, str], dict[str, dict[str, Any]]
    ] = {}
    main_base_latent_protocol_by_repeat: dict[
        str,
        dict[str, dict[str, Any]],
    ] = {}
    for source in materialized:
        if source.paper_run_name != run_name:
            raise MethodRepeatFixedFprError("来源论文运行层级不一致")
        key = (source.randomization_repeat_id, source.method_id)
        if key in source_by_key:
            raise MethodRepeatFixedFprError("method-repeat 来源键重复")
        _, identities, main_base_protocols = _validate_source_rows(
            source,
            prompt_contract=prompt_contract,
            expected_model_id=expected_model_id,
            expected_model_revision=expected_model_revision,
            expected_base_seed=expected_base_seed,
        )
        source_by_key[key] = source
        identity_by_source[key] = identities
        if source.method_id == "slm_wm":
            main_base_latent_protocol_by_repeat[
                source.randomization_repeat_id
            ] = main_base_protocols
    if (
        tuple(key for key in expected_keys if key in source_by_key)
        != expected_keys
        or len(source_by_key) != METHOD_REPEAT_THRESHOLD_COUNT
    ):
        raise MethodRepeatFixedFprError(
            "阈值来源必须精确覆盖9个重复与5个方法的笛卡尔积"
        )
    aggregate_identities = {
        (
            source.randomization_aggregate_package_sha256,
            source.randomization_aggregate_digest,
            source.common_code_version,
        )
        for source in source_by_key.values()
    }
    if len(aggregate_identities) != 1:
        raise MethodRepeatFixedFprError("45个来源没有绑定同一个精确聚合包")
    for repeat_id in formal_randomization_repeat_ids():
        component_identities = {
            (
                source.randomization_repeat_component_sha256,
                source.randomization_repeat_evidence_manifest_digest,
                source.component_content_digest,
            )
            for (source_repeat_id, _), source in source_by_key.items()
            if source_repeat_id == repeat_id
        }
        if len(component_identities) != 1:
            raise MethodRepeatFixedFprError(
                "同一重复的五方法没有绑定同一个 repeat component"
            )
    fairness_records = _validate_cross_repeat_randomization(
        identity_by_source,
        main_base_latent_protocol_by_repeat,
        prompt_contract,
        expected_base_seed=expected_base_seed,
    )
    fairness_digest_by_repeat = {
        repeat_id: build_stable_digest(
            [
                row
                for row in fairness_records
                if row["randomization_repeat_id"] == repeat_id
            ]
        )
        for repeat_id in formal_randomization_repeat_ids()
    }
    threshold_records = tuple(
        _recompute_threshold_record(
            source_by_key[(repeat_id, method_id)],
            prompt_contract=prompt_contract,
            fairness_identity_digest=fairness_digest_by_repeat[repeat_id],
            target_fpr=resolved_target_fpr,
            main_rescue_margin_low=main_rescue_margin_low,
        )
        for repeat_id, method_id in expected_keys
    )
    repeat_threshold_counts = {
        repeat_id: sum(
            row["randomization_repeat_id"] == repeat_id
            for row in threshold_records
        )
        for repeat_id in formal_randomization_repeat_ids()
    }
    ready = (
        len(threshold_records) == METHOD_REPEAT_THRESHOLD_COUNT
        and all(
            count == len(FIXED_FPR_THRESHOLD_METHOD_IDS)
            for count in repeat_threshold_counts.values()
        )
        and all(row["fixed_fpr_threshold_ready"] is True for row in threshold_records)
    )
    (
        aggregate_package_sha256,
        aggregate_digest,
        common_code_version,
    ) = next(iter(aggregate_identities))
    report_payload = {
        "report_schema": METHOD_REPEAT_THRESHOLD_REPORT_SCHEMA,
        "paper_claim_scale": run_name,
        "target_fpr": resolved_target_fpr,
        "threshold_calculation_unit": METHOD_REPEAT_THRESHOLD_CALCULATION_UNIT,
        "expected_randomization_repeat_ids": list(
            formal_randomization_repeat_ids()
        ),
        "expected_method_ids": list(FIXED_FPR_THRESHOLD_METHOD_IDS),
        "expected_threshold_record_count": METHOD_REPEAT_THRESHOLD_COUNT,
        "threshold_record_count": len(threshold_records),
        "repeat_threshold_counts": repeat_threshold_counts,
        "prompt_count": prompt_contract["prompt_count"],
        "split_counts": prompt_contract["split_counts"],
        "prompt_protocol_digest": prompt_contract["prompt_protocol_digest"],
        "generation_model_id": expected_model_id,
        "generation_model_revision": expected_model_revision,
        "expected_base_seed": expected_base_seed,
        "randomization_aggregate_package_sha256": aggregate_package_sha256,
        "randomization_aggregate_digest": aggregate_digest,
        "common_code_version": common_code_version,
        "repeat_fairness_identity_digest_map": fairness_digest_by_repeat,
        "fairness_records_digest": build_stable_digest(fairness_records),
        "threshold_records_digest": build_stable_digest(threshold_records),
        "method_repeat_fixed_fpr_recomputation_ready": ready,
        "supports_paper_claim": False,
    }
    report_payload["method_repeat_fixed_fpr_report_digest"] = (
        build_stable_digest(report_payload)
    )
    if not ready:
        raise MethodRepeatFixedFprError("逐方法逐重复阈值重算报告未通过")
    return {
        "threshold_records": threshold_records,
        "fairness_records": fairness_records,
        "report": report_payload,
    }


__all__ = [
    "FORMAL_BASE_LATENT_DTYPE",
    "FORMAL_BASE_LATENT_GENERATION_PROTOCOL",
    "FORMAL_BASE_LATENT_SHAPE",
    "METHOD_LEAF_PACKAGE_FAMILY",
    "METHOD_REPEAT_FAIRNESS_SCHEMA",
    "METHOD_REPEAT_THRESHOLD_CALCULATION_UNIT",
    "METHOD_REPEAT_THRESHOLD_COUNT",
    "METHOD_REPEAT_THRESHOLD_REPORT_SCHEMA",
    "METHOD_REPEAT_THRESHOLD_SCHEMA",
    "MethodRepeatFixedFprError",
    "MethodRepeatObservationSource",
    "build_prompt_split_contract",
    "recompute_exact_method_repeat_fixed_fpr",
]
