"""冻结四图配对质量 estimand 并构造不可混淆的原始图像身份记录."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from experiments.protocol.attacks import attack_config_digest, default_attack_configs
from experiments.protocol.independent_semantic_quality import (
    load_independent_semantic_quality_evaluator,
)
from experiments.runtime.scientific_unit_provenance import (
    validate_scientific_unit_provenance,
)
from main.core.digest import build_stable_digest


DEFAULT_ATTACK_CONDITIONED_QUALITY_ESTIMAND_PATH = (
    Path(__file__).resolve().parents[2]
    / "configs"
    / "attack_conditioned_quality_estimand.json"
)
ATTACK_CONDITIONED_QUALITY_RECORD_SCHEMA = (
    "attack_conditioned_four_image_quality_record_v1"
)
ATTACK_CONDITIONED_IMAGE_PAIR_ROLE = "matched_attack_clean_to_watermarked"
FOUR_IMAGE_ROLES = (
    "clean",
    "watermarked",
    "attacked_clean",
    "attacked_watermarked",
)


class AttackConditionedQualityError(ValueError):
    """表示四图配对身份,攻击随机性或 estimand 配置不完整."""


@dataclass(frozen=True)
class AttackConditionedQualityImageRecord:
    """保存同一 Prompt,repeat 和攻击下的四张真实图像身份."""

    attack_quality_record_id: str
    attack_quality_record_digest: str
    record_schema: str
    quality_estimand_id: str
    quality_estimand_protocol_digest: str
    run_id: str
    prompt_id: str
    split: str
    randomization_repeat_id: str
    sample_role: str
    attack_id: str
    attack_name: str
    attack_family: str
    attack_config_digest: str
    attack_seed_random: int
    formal_attack_seed_protocol_digest: str
    attack_parameters: Mapping[str, Any]
    source_image_role: str
    comparison_image_role: str
    image_pair_role: str
    clean_image: Mapping[str, Any]
    watermarked_image: Mapping[str, Any]
    attacked_clean_image: Mapping[str, Any]
    attacked_watermarked_image: Mapping[str, Any]
    generation_scientific_unit_provenance: Mapping[str, Any]
    code_version: str
    scientific_dependency_profile_id: str
    scientific_dependency_profile_digest: str
    scientific_complete_hash_lock_digest: str
    supports_paper_claim: bool

    def to_dict(self) -> dict[str, Any]:
        """转换为可直接写入 JSONL 的普通字典."""

        return {
            field_name: getattr(self, field_name)
            for field_name in self.__dataclass_fields__
        }


def _is_sha256(value: Any) -> bool:
    """判断值是否为规范小写 SHA-256."""

    text = str(value)
    return len(text) == 64 and all(
        character in "0123456789abcdef" for character in text
    )


def load_attack_conditioned_quality_estimand(
    path: str | Path = DEFAULT_ATTACK_CONDITIONED_QUALITY_ESTIMAND_PATH,
) -> dict[str, Any]:
    """读取并验证四图配对,CLIP 和 KID 的冻结科学含义."""

    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AttackConditionedQualityError("四图质量 estimand 配置无法读取") from exc
    base_pairing = payload.get("base_pairing")
    attack_pairing = payload.get("attack_pairing")
    clip_similarity = payload.get("clip_image_similarity")
    independent_similarity = payload.get("independent_semantic_similarity")
    distribution = payload.get("distributional_quality")
    if not all(
        isinstance(value, dict)
        for value in (
            base_pairing,
            attack_pairing,
            clip_similarity,
            independent_similarity,
            distribution,
        )
    ):
        raise AttackConditionedQualityError("四图质量 estimand 组件必须是对象")
    if (
        payload.get("protocol_schema")
        != "attack_conditioned_quality_estimand_v1"
        or tuple(payload.get("four_image_roles", ())) != FOUR_IMAGE_ROLES
        or base_pairing.get("pair_role") != "clean_to_watermarked"
        or attack_pairing.get("pair_role")
        != ATTACK_CONDITIONED_IMAGE_PAIR_ROLE
        or attack_pairing.get("source_image_role") != "attacked_clean"
        or attack_pairing.get("comparison_image_role")
        != "attacked_watermarked"
        or attack_pairing.get("attack_configuration_rule")
        != "identical_registered_attack_config_digest"
        or attack_pairing.get("attack_randomness_rule")
        != "identical_formal_attack_seed_random"
        or clip_similarity.get("model_id")
        != "openai/clip-vit-base-patch32"
        or clip_similarity.get("model_revision")
        != "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"
        or clip_similarity.get("vision_output")
        != "CLIPVisionModelWithProjection.image_embeds"
        or clip_similarity.get("evidence_role")
        != "mechanism_consistency_diagnostic"
        or independent_similarity.get("evidence_role")
        != "independent_semantic_preservation_primary"
        or independent_similarity.get("evaluator_config_path")
        != "configs/independent_semantic_quality_evaluator.json"
        or distribution.get("primary_estimand")
        != "prompt_conditional_kid_mean"
        or distribution.get("primary_sampling_unit") != "prompt"
    ):
        raise AttackConditionedQualityError("四图质量 estimand 的冻结含义发生漂移")
    resolved = dict(payload)
    independent_evaluator = load_independent_semantic_quality_evaluator()
    resolved["independent_semantic_quality_evaluator"] = independent_evaluator
    resolved["quality_estimand_protocol_digest"] = build_stable_digest(resolved)
    resolved["quality_estimand_id"] = (
        "attack_conditioned_quality_estimand_"
        f"{resolved['quality_estimand_protocol_digest'][:16]}"
    )
    return resolved


def _image_identity(record: Mapping[str, Any], prefix: str) -> dict[str, Any]:
    """从检测记录提取路径,文件摘要,像素摘要和分辨率身份."""

    image_path = str(record.get(f"{prefix}_image_path", ""))
    image_digest = str(record.get(f"{prefix}_image_digest", ""))
    pixel_digest = str(
        record.get(f"{prefix}_image_rgb_uint8_content_sha256", "")
    )
    width = record.get(f"{prefix}_image_width")
    height = record.get(f"{prefix}_image_height")
    if (
        not image_path
        or not _is_sha256(image_digest)
        or not _is_sha256(pixel_digest)
        or type(width) is not int
        or type(height) is not int
        or width <= 0
        or height <= 0
    ):
        raise AttackConditionedQualityError("检测记录缺少完整图像文件和像素身份")
    return {
        "image_path": image_path,
        "image_sha256": image_digest,
        "image_rgb_uint8_content_sha256": pixel_digest,
        "image_width": width,
        "image_height": height,
    }


def _registered_attack_identities() -> dict[str, str]:
    """返回启用攻击的精确配置摘要, 防止运行记录替换攻击参数."""

    return {
        config.attack_id: attack_config_digest(config)
        for config in default_attack_configs()
        if config.enabled
    }


def build_attack_conditioned_quality_image_records(
    detection_records: Iterable[Mapping[str, Any]],
    runtime_results: Iterable[Mapping[str, Any]],
    *,
    expected_randomization_repeat_id: str,
    expected_prompt_ids: Iterable[str],
    protocol: Mapping[str, Any] | None = None,
) -> tuple[AttackConditionedQualityImageRecord, ...]:
    """把两类攻击检测记录联接为逐 Prompt,逐攻击四图真实记录."""

    estimand = dict(protocol or load_attack_conditioned_quality_estimand())
    repeat_id = str(expected_randomization_repeat_id).strip()
    prompt_ids = tuple(str(prompt_id).strip() for prompt_id in expected_prompt_ids)
    if (
        not repeat_id
        or not prompt_ids
        or len(set(prompt_ids)) != len(prompt_ids)
        or any(not prompt_id for prompt_id in prompt_ids)
    ):
        raise AttackConditionedQualityError("四图质量预期 repeat 或 Prompt 集合无效")
    registered_attacks = _registered_attack_identities()
    expected_keys = {
        (prompt_id, attack_id)
        for prompt_id in prompt_ids
        for attack_id in registered_attacks
    }
    runtime_by_run_id: dict[str, Mapping[str, Any]] = {}
    for result in runtime_results:
        run_id = str(result.get("run_id", ""))
        if not run_id or run_id in runtime_by_run_id:
            raise AttackConditionedQualityError("运行结果 run_id 缺失或重复")
        runtime_by_run_id[run_id] = result

    grouped: dict[tuple[str, str], dict[str, Mapping[str, Any]]] = {}
    for record in detection_records:
        attack_id = str(record.get("attack_id", ""))
        if not attack_id:
            continue
        prompt_id = str(record.get("prompt_id", ""))
        key = (prompt_id, attack_id)
        if key not in expected_keys:
            continue
        role = str(record.get("sample_role", ""))
        if role not in {"clean_negative", "positive_source"}:
            continue
        role_records = grouped.setdefault(key, {})
        if role in role_records:
            raise AttackConditionedQualityError("同一四图配对的攻击样本角色重复")
        role_records[role] = record
    if set(grouped) != expected_keys or any(
        set(role_records) != {"clean_negative", "positive_source"}
        for role_records in grouped.values()
    ):
        raise AttackConditionedQualityError("四图质量记录未覆盖 Prompt,攻击和两个样本角色")

    records: list[AttackConditionedQualityImageRecord] = []
    for prompt_id in prompt_ids:
        for attack_id in sorted(registered_attacks):
            negative = grouped[(prompt_id, attack_id)]["clean_negative"]
            positive = grouped[(prompt_id, attack_id)]["positive_source"]
            run_id = str(negative.get("run_id", ""))
            result = runtime_by_run_id.get(run_id)
            provenance = (
                result.get("metadata", {}).get("scientific_unit_provenance")
                if isinstance(result, Mapping)
                and isinstance(result.get("metadata"), Mapping)
                else None
            )
            if not isinstance(provenance, Mapping):
                raise AttackConditionedQualityError("四图记录缺少生成科学运行来源")
            validated_provenance = validate_scientific_unit_provenance(provenance)
            scientific_environment = dict(
                validated_provenance["scientific_execution_environment"]
            )
            dependency_id = str(
                scientific_environment.get("dependency_profile_id", "")
            )
            dependency_digest = str(
                scientific_environment.get("dependency_profile_digest", "")
            )
            lock_digest = str(
                scientific_environment.get("complete_hash_lock_digest", "")
            )
            code_version = str(
                scientific_environment.get("formal_execution_commit", "")
            )
            if not all(
                (
                    str(positive.get("run_id", "")) == run_id,
                    negative.get("split") == positive.get("split") == "test",
                    negative.get("randomization_repeat_id")
                    == positive.get("randomization_repeat_id")
                    == repeat_id,
                    negative.get("attack_config_digest")
                    == positive.get("attack_config_digest")
                    == registered_attacks[attack_id],
                    negative.get("attack_seed_random")
                    == positive.get("attack_seed_random"),
                    negative.get("formal_attack_seed_protocol_digest")
                    == positive.get("formal_attack_seed_protocol_digest"),
                    negative.get("attack_parameters")
                    == positive.get("attack_parameters"),
                    dependency_id == "sd35_method_runtime_gpu",
                    _is_sha256(dependency_digest),
                    _is_sha256(lock_digest),
                    len(code_version) == 40,
                )
            ):
                raise AttackConditionedQualityError("四图配对的攻击,repeat 或科学环境身份不一致")
            clean_image = _image_identity(negative, "source")
            watermarked_image = _image_identity(positive, "source")
            attacked_clean_image = _image_identity(negative, "evaluated")
            attacked_watermarked_image = _image_identity(positive, "evaluated")
            if (
                clean_image["image_width"] != watermarked_image["image_width"]
                or clean_image["image_height"] != watermarked_image["image_height"]
                or attacked_clean_image["image_width"]
                != attacked_watermarked_image["image_width"]
                or attacked_clean_image["image_height"]
                != attacked_watermarked_image["image_height"]
            ):
                raise AttackConditionedQualityError("四图配对分辨率不一致")
            core = {
                "record_schema": ATTACK_CONDITIONED_QUALITY_RECORD_SCHEMA,
                "quality_estimand_id": estimand["quality_estimand_id"],
                "quality_estimand_protocol_digest": estimand[
                    "quality_estimand_protocol_digest"
                ],
                "run_id": run_id,
                "prompt_id": prompt_id,
                "split": "test",
                "randomization_repeat_id": repeat_id,
                "sample_role": "four_image_matched_attack_pair",
                "attack_id": attack_id,
                "attack_name": str(negative.get("attack_name", "")),
                "attack_family": str(negative.get("attack_family", "")),
                "attack_config_digest": registered_attacks[attack_id],
                "attack_seed_random": int(negative["attack_seed_random"]),
                "formal_attack_seed_protocol_digest": str(
                    negative.get("formal_attack_seed_protocol_digest", "")
                ),
                "attack_parameters": dict(negative.get("attack_parameters", {})),
                "source_image_role": "attacked_clean",
                "comparison_image_role": "attacked_watermarked",
                "image_pair_role": ATTACK_CONDITIONED_IMAGE_PAIR_ROLE,
                "clean_image": clean_image,
                "watermarked_image": watermarked_image,
                "attacked_clean_image": attacked_clean_image,
                "attacked_watermarked_image": attacked_watermarked_image,
                "generation_scientific_unit_provenance": dict(
                    validated_provenance
                ),
                "code_version": code_version,
                "scientific_dependency_profile_id": dependency_id,
                "scientific_dependency_profile_digest": dependency_digest,
                "scientific_complete_hash_lock_digest": lock_digest,
                "supports_paper_claim": False,
            }
            record_digest = build_stable_digest(core)
            records.append(
                AttackConditionedQualityImageRecord(
                    attack_quality_record_id=(
                        f"attack_quality_record_{record_digest[:16]}"
                    ),
                    attack_quality_record_digest=record_digest,
                    **core,
                )
            )
    return tuple(records)


def attack_quality_dataset_image_records(
    records: Iterable[Mapping[str, Any] | AttackConditionedQualityImageRecord],
) -> tuple[dict[str, Any], ...]:
    """把四图记录映射为通用 Inception/CLIP source-comparison 图像对."""

    result: list[dict[str, Any]] = []
    for raw_record in records:
        record = (
            raw_record.to_dict()
            if isinstance(raw_record, AttackConditionedQualityImageRecord)
            else dict(raw_record)
        )
        source = dict(record.get("attacked_clean_image", {}))
        comparison = dict(record.get("attacked_watermarked_image", {}))
        record_digest = str(record.get("attack_quality_record_digest", ""))
        if (
            record.get("record_schema") != ATTACK_CONDITIONED_QUALITY_RECORD_SCHEMA
            or not _is_sha256(record_digest)
            or record.get("attack_quality_record_id")
            != f"attack_quality_record_{record_digest[:16]}"
            or not _is_sha256(source.get("image_sha256"))
            or not _is_sha256(comparison.get("image_sha256"))
        ):
            raise AttackConditionedQualityError("四图记录不能映射为正式特征图像对")
        dataset_core = {
            "run_id": str(record["run_id"]),
            "prompt_id": str(record["prompt_id"]),
            "attack_name": str(record["attack_id"]),
            "image_pair_index": len(result),
            "image_pair_role": ATTACK_CONDITIONED_IMAGE_PAIR_ROLE,
            "source_image_path": str(source["image_path"]),
            "source_image_digest": str(source["image_sha256"]),
            "comparison_image_path": str(comparison["image_path"]),
            "comparison_image_digest": str(comparison["image_sha256"]),
            "feature_backend": "inception_feature_backend",
            "supports_paper_claim": False,
        }
        dataset_digest = build_stable_digest(dataset_core)
        result.append(
            {
                "dataset_quality_record_id": (
                    f"dataset_quality_record_{dataset_digest[:16]}"
                ),
                "dataset_quality_record_digest": dataset_digest,
                "attack_quality_record_id": record["attack_quality_record_id"],
                "attack_quality_record_digest": record_digest,
                "randomization_repeat_id": record["randomization_repeat_id"],
                "attack_id": record["attack_id"],
                "attack_config_digest": record["attack_config_digest"],
                "attack_seed_random": record["attack_seed_random"],
                **dataset_core,
            }
        )
    return tuple(result)


__all__ = [
    "ATTACK_CONDITIONED_IMAGE_PAIR_ROLE",
    "ATTACK_CONDITIONED_QUALITY_RECORD_SCHEMA",
    "AttackConditionedQualityError",
    "AttackConditionedQualityImageRecord",
    "attack_quality_dataset_image_records",
    "build_attack_conditioned_quality_image_records",
    "load_attack_conditioned_quality_estimand",
]
