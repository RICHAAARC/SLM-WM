"""从精确随机化聚合来源重算逐方法、逐重复 fixed-FPR 阈值.

该模块把聚合包的临时记录读取层与纯统计层连接起来. 公共入口只能接收
``RandomizationAggregateProvenance``，不会接受仓库根目录、解压目录、Prompt
文件或成员路径覆盖. Prompt exact-set 只能从主方法 leaf 内嵌的来源注册表、
冻结选择清单和 Prompt 文件重建, 9个受验证 ``runtime_results`` 必须逐条匹配;
五方法 observation 和阈值声明只能由同一临时工作区提供.

该层不写 ``outputs/``，也不把45个独立阈值提升为论文结论. 返回值保持
``supports_paper_claim=false``，供后续攻击矩阵和跨重复统计继续消费.
"""

from __future__ import annotations

from dataclasses import fields
import hashlib
import math
from typing import Any, Mapping

from experiments.protocol.fixed_fpr_observation_audit import (
    FORMAL_THRESHOLD_SOURCE,
)
from experiments.protocol.formal_randomization import (
    build_canonical_sd35_base_latent,
    formal_generation_seed,
    formal_randomization_protocol_record,
    formal_randomization_repeat_ids,
    formal_watermark_key_plan_record,
    formal_watermark_key_material_from_seed,
    resolve_formal_randomization_repeat,
)
from experiments.protocol.method_runtime_config import (
    FORMAL_METHOD_PACKAGE_ROOT,
    load_formal_method_runtime_config,
)
from experiments.protocol.paper_run_config import (
    RUN_EXPECTED_PROMPT_COUNTS,
    normalize_paper_run_name,
    validate_frozen_paper_run_target_fpr,
)
from experiments.protocol.prompts import (
    build_prompt_records,
    normalize_prompt_text,
)
from experiments.protocol.splits import apply_split_assignments
from experiments.runners.semantic_watermark_runtime import (
    FORMAL_RUNTIME_RESCUE_MARGIN_LOW,
    SemanticWatermarkRuntimeConfig,
    validate_semantic_watermark_runtime_result_provenance,
)
from main.core.digest import build_stable_digest
from paper_experiments.analysis.fixed_fpr_threshold_audit import (
    FIXED_FPR_THRESHOLD_METHOD_IDS,
)
from paper_experiments.analysis.method_repeat_fixed_fpr import (
    METHOD_LEAF_PACKAGE_FAMILY,
    MethodRepeatObservationSource,
    recompute_exact_method_repeat_fixed_fpr,
)
from paper_experiments.baselines.formal_import import (
    build_fixed_fpr_operating_point,
)
from paper_experiments.runners.randomization_aggregate_provenance import (
    RandomizationAggregateProvenance,
)
from paper_experiments.runners.randomization_aggregate_record_workspace import (
    RandomizationAggregateRecordSource,
    RandomizationAggregateRecordWorkspace,
    open_randomization_aggregate_record_workspace,
)
from paper_experiments.runners.randomization_prompt_source_contract import (
    rebuild_randomization_prompt_source_contract,
)


_METHOD_SOURCE_ROLES = {
    "slm_wm": (
        "semantic_watermark_detection_observation",
        "semantic_watermark_frozen_evidence_protocol",
    ),
    "tree_ring": (
        "tree_ring_baseline_observation",
        "tree_ring_baseline_transfer_manifest",
    ),
    "gaussian_shading": (
        "gaussian_shading_baseline_observation",
        "gaussian_shading_baseline_transfer_manifest",
    ),
    "shallow_diffuse": (
        "shallow_diffuse_baseline_observation",
        "shallow_diffuse_baseline_transfer_manifest",
    ),
    "t2smark": (
        "t2smark_baseline_observation",
        "t2smark_formal_import_candidate_record",
    ),
}
_SOURCE_LINEAGE_FIELDS = (
    "randomization_scope",
    "randomization_repeat_id",
    "package_family",
    "leaf_package_sha256",
    "randomization_repeat_component_sha256",
    "randomization_repeat_evidence_manifest_digest",
    "component_content_digest",
    "randomization_aggregate_package_sha256",
    "common_code_version",
    "randomization_aggregate_digest",
)
_SHA256_SOURCE_FIELDS = (
    "record_sha256",
    "leaf_package_sha256",
    "randomization_repeat_component_sha256",
    "randomization_repeat_evidence_manifest_digest",
    "component_content_digest",
    "randomization_aggregate_package_sha256",
    "randomization_aggregate_digest",
)
_FROZEN_METHOD_CONFIG = load_formal_method_runtime_config(
    FORMAL_METHOD_PACKAGE_ROOT
)
_FROZEN_BASE_GENERATION_SEED = _FROZEN_METHOD_CONFIG.seed
_RUNTIME_RANDOMIZATION_IDENTITY_FIELDS = (
    "randomization_repeat_id",
    "generation_seed_index",
    "generation_seed_offset",
    "watermark_key_index",
    "generation_seed_random",
    "watermark_key_seed_random",
    "formal_randomization_protocol_digest",
    "watermark_key_material_digest_random",
    "formal_randomization_identity_digest_random",
)


class RandomizationMethodRepeatThresholdError(ValueError):
    """表示聚合来源无法形成精确45个独立 fixed-FPR 阈值."""


def _plain_protocol_value(value: Any) -> Any:
    """把 tuple 和 mapping 递归转换为可比较的 JSON 语义值."""

    if isinstance(value, Mapping):
        return {
            str(key): _plain_protocol_value(item)
            for key, item in value.items()
        }
    if isinstance(value, (tuple, list)):
        return [_plain_protocol_value(item) for item in value]
    return value


def _validate_formal_runtime_method_config(
    config: Mapping[str, Any],
    *,
    paper_run_name: str,
) -> None:
    """把持久化 runtime payload 重新锁定到完整正式方法与检测配置."""

    expected_payload_fields = {
        field.name for field in fields(SemanticWatermarkRuntimeConfig)
    }
    expected_payload_fields.remove("key_material")
    expected_payload_fields.update(
        {
            "key_material_digest_random",
            "method_definition",
            "method_definition_digest",
        }
    )
    if set(config) != expected_payload_fields:
        raise RandomizationMethodRepeatThresholdError(
            "runtime 配置字段集合未匹配当前持久化 schema"
        )
    expected_method_settings = _FROZEN_METHOD_CONFIG.paper_method_settings()
    drifted_method_fields = tuple(
        field_name
        for field_name, expected_value in expected_method_settings.items()
        if field_name not in config
        or _plain_protocol_value(config[field_name])
        != _plain_protocol_value(expected_value)
    )
    if drifted_method_fields:
        raise RandomizationMethodRepeatThresholdError(
            "runtime 未完整继承冻结方法配置: "
            + ",".join(drifted_method_fields)
        )
    split = str(config.get("split", ""))
    attacks_enabled = split == "test"
    exact_values = {
        "model_family": _FROZEN_METHOD_CONFIG.model_family,
        "model_id": _FROZEN_METHOD_CONFIG.model_id,
        "model_revision": _FROZEN_METHOD_CONFIG.model_revision,
        "vision_model_id": _FROZEN_METHOD_CONFIG.vision_model_id,
        "vision_model_revision": _FROZEN_METHOD_CONFIG.vision_model_revision,
        "negative_prompt": _FROZEN_METHOD_CONFIG.negative_prompt,
        "width": _FROZEN_METHOD_CONFIG.width,
        "height": _FROZEN_METHOD_CONFIG.height,
        "device_name": "cuda",
        "torch_dtype": _FROZEN_METHOD_CONFIG.latent_torch_dtype,
        "injection_step_indices": list(
            _FROZEN_METHOD_CONFIG.injection_step_indices
        ),
        "candidate_count": _FROZEN_METHOD_CONFIG.jacobian_candidate_count,
        "null_rank": _FROZEN_METHOD_CONFIG.null_space_rank,
        "branch_risk_mode": "branch_specific",
        "standard_attack_profiles": (
            ["full_main"] if attacks_enabled else []
        ),
        "diffusion_attacks_enabled": (
            _FROZEN_METHOD_CONFIG.diffusion_attacks_enabled
            if attacks_enabled
            else False
        ),
        "content_threshold": 0.0,
        "geometry_score_threshold": 0.0,
        "registration_confidence_threshold": 0.0,
        "attention_sync_score_threshold": 0.0,
        "rescue_margin_low": FORMAL_RUNTIME_RESCUE_MARGIN_LOW,
        "output_dir": (
            f"outputs/image_only_dataset_runtime/{paper_run_name}/runs"
        ),
    }
    drifted_exact_fields = tuple(
        field_name
        for field_name, expected_value in exact_values.items()
        if field_name not in config
        or _plain_protocol_value(config[field_name])
        != _plain_protocol_value(expected_value)
    )
    required_true_fields = (
        "semantic_routing_enabled",
        "null_space_enabled",
        "lf_enabled",
        "tail_robust_enabled",
        "tail_truncation_enabled",
        "attention_geometry_enabled",
        "image_alignment_enabled",
    )
    disabled_formal_fields = tuple(
        field_name
        for field_name in required_true_fields
        if config.get(field_name) is not True
    )
    if drifted_exact_fields or disabled_formal_fields:
        raise RandomizationMethodRepeatThresholdError(
            "runtime 完整方法或检测配置发生漂移: "
            + ",".join((*drifted_exact_fields, *disabled_formal_fields))
        )


def _is_sha256(value: Any) -> bool:
    """判断值是否为规范的小写 SHA-256."""

    text = str(value)
    return len(text) == 64 and all(
        character in "0123456789abcdef" for character in text
    )


def _require_source_digest_fields(
    source: RandomizationAggregateRecordSource,
) -> None:
    """集中校验工作区描述符的包、组件与成员摘要."""

    if not isinstance(source, RandomizationAggregateRecordSource):
        raise TypeError("记录来源必须由聚合记录工作区提供")
    invalid_fields = tuple(
        field_name
        for field_name in _SHA256_SOURCE_FIELDS
        if not _is_sha256(getattr(source, field_name))
    )
    if invalid_fields:
        raise RandomizationMethodRepeatThresholdError(
            "聚合记录来源缺少规范摘要: " + ",".join(invalid_fields)
        )


def _require_source_aggregate_identity(
    source: RandomizationAggregateRecordSource,
    provenance: RandomizationAggregateProvenance,
) -> None:
    """核对记录描述符与公共入口聚合对象的共同身份."""

    _require_source_digest_fields(source)
    if not all(
        (
            source.randomization_scope == "active_repeat_component",
            source.randomization_aggregate_package_sha256
            == provenance.package_sha256,
            source.randomization_aggregate_digest
            == provenance.randomization_aggregate_digest,
            source.common_code_version == provenance.common_code_version,
        )
    ):
        raise RandomizationMethodRepeatThresholdError(
            "记录来源没有绑定公共入口的精确聚合身份"
        )
    repeat_records = {
        str(record.get("randomization_repeat_id", "")): record
        for record in provenance.randomization_repeat_components
    }
    repeat_record = repeat_records.get(source.randomization_repeat_id)
    if repeat_record is None or not all(
        (
            source.randomization_repeat_component_sha256
            == repeat_record.get("package_sha256"),
            source.randomization_repeat_evidence_manifest_digest
            == repeat_record.get(
                "randomization_repeat_evidence_manifest_digest"
            ),
            source.component_content_digest
            == repeat_record.get("component_content_digest"),
        )
    ):
        raise RandomizationMethodRepeatThresholdError(
            "记录来源的重复组件摘要与聚合清单不一致"
        )


def _runtime_prompt_rows(
    workspace: RandomizationAggregateRecordWorkspace,
    provenance: RandomizationAggregateProvenance,
    *,
    paper_run_name: str,
    expected_prompt_rows: tuple[Mapping[str, Any], ...],
) -> tuple[
    tuple[dict[str, Any], ...],
    str,
    str,
    float,
    tuple[dict[str, Any], ...],
    dict[str, dict[str, dict[str, Any]]],
]:
    """从9个 runtime 成员重建 Prompt 与受治理随机身份锚点."""

    expected_count = int(RUN_EXPECTED_PROMPT_COUNTS[paper_run_name])
    if len(expected_prompt_rows) != expected_count:
        raise RandomizationMethodRepeatThresholdError(
            "内嵌来源 Prompt 契约未匹配论文运行数量"
        )
    frozen_prompt_rows = tuple(dict(row) for row in expected_prompt_rows)
    protocol_digest = str(
        formal_randomization_protocol_record()[
            "formal_randomization_protocol_digest"
        ]
    )
    canonical_prompt_rows: tuple[dict[str, Any], ...] | None = None
    canonical_prompt_text_bytes: tuple[bytes, ...] | None = None
    model_identities: set[tuple[str, str]] = set()
    rescue_margins: set[float] = set()
    repeat_ids = formal_randomization_repeat_ids()
    registered_runtime_sources = {
        source.randomization_repeat_id: source
        for source in workspace.prompt_runtime_sources
    }
    if (
        tuple(registered_runtime_sources) != repeat_ids
        or len(registered_runtime_sources) != len(repeat_ids)
    ):
        raise RandomizationMethodRepeatThresholdError(
            "runtime Prompt 来源必须精确覆盖9个随机化重复"
        )
    runtime_source_records: list[dict[str, Any]] = []
    runtime_randomization_identity_map: dict[
        str,
        dict[str, dict[str, Any]],
    ] = {}

    for repeat_id in repeat_ids:
        source = workspace.find_source(
            randomization_repeat_id=repeat_id,
            package_family="image_only_dataset_runtime",
            record_role="semantic_watermark_runtime_record",
        )
        if source != registered_runtime_sources[repeat_id]:
            raise RandomizationMethodRepeatThresholdError(
                "runtime 成员摘要与工作区登记映射发生漂移"
            )
        _require_source_aggregate_identity(source, provenance)
        expected_member = (
            f"outputs/image_only_dataset_runtime/{paper_run_name}/"
            "runtime_results.jsonl"
        )
        if source.record_member != expected_member:
            raise RandomizationMethodRepeatThresholdError(
                "runtime Prompt 来源成员路径不规范"
            )
        runtime_rows = tuple(workspace.iter_records(source))
        if len(runtime_rows) != expected_count:
            raise RandomizationMethodRepeatThresholdError(
                "runtime Prompt 来源没有覆盖论文运行完整数量"
            )
        repeat = resolve_formal_randomization_repeat(repeat_id)
        configs: list[Mapping[str, Any]] = []
        prompt_texts: list[str] = []
        for runtime_row in runtime_rows:
            try:
                validate_semantic_watermark_runtime_result_provenance(
                    runtime_row
                )
            except (TypeError, ValueError) as exc:
                raise RandomizationMethodRepeatThresholdError(
                    "runtime raw record 的 run_id、配置或科学来源不可重建"
                ) from exc
            metadata = runtime_row.get("metadata")
            if not isinstance(metadata, Mapping):
                raise RandomizationMethodRepeatThresholdError(
                    "runtime 记录缺少科学单元 metadata"
                )
            config = metadata.get("scientific_unit_config")
            if not isinstance(config, Mapping):
                raise RandomizationMethodRepeatThresholdError(
                    "runtime 记录缺少科学单元配置"
                )
            _validate_formal_runtime_method_config(
                config,
                paper_run_name=paper_run_name,
            )
            prompt_text = config.get("prompt")
            if (
                not isinstance(prompt_text, str)
                or prompt_text != normalize_prompt_text(prompt_text)
            ):
                raise RandomizationMethodRepeatThresholdError(
                    "runtime Prompt 文本字段无效"
                )
            if (
                config.get("randomization_repeat_id") != repeat_id
                or config.get("generation_seed_index")
                != repeat.generation_seed_index
                or config.get("generation_seed_offset")
                != repeat.generation_seed_offset
                or config.get("watermark_key_index")
                != repeat.watermark_key_index
                or config.get("formal_randomization_protocol_digest")
                != protocol_digest
                or type(config.get("seed")) is not int
                or type(config.get("watermark_key_seed_random")) is not int
            ):
                raise RandomizationMethodRepeatThresholdError(
                    "runtime Prompt 或随机化身份无法逐字段重建"
                )
            key_seed = int(config["watermark_key_seed_random"])
            try:
                rebuilt_key_material = (
                    formal_watermark_key_material_from_seed(
                        key_seed,
                        repeat,
                    )
                )
            except (TypeError, ValueError) as exc:
                raise RandomizationMethodRepeatThresholdError(
                    "runtime watermark key seed 不属于正式取值域"
                ) from exc
            expected_key_material_digest = build_stable_digest(
                {"key_material": rebuilt_key_material}
            )
            if (
                not _is_sha256(config.get("key_material_digest_random"))
                or config.get("key_material_digest_random")
                != expected_key_material_digest
            ):
                raise RandomizationMethodRepeatThresholdError(
                    "runtime key material 摘要无法由正式 key seed 重建"
                )
            model_id = str(config.get("model_id", ""))
            model_revision = str(config.get("model_revision", ""))
            rescue_margin = config.get("rescue_margin_low")
            if (
                not model_id
                or not model_revision
                or isinstance(rescue_margin, bool)
                or not isinstance(rescue_margin, int | float)
                or not math.isfinite(float(rescue_margin))
                or float(rescue_margin) >= 0.0
            ):
                raise RandomizationMethodRepeatThresholdError(
                    "runtime 记录缺少共同模型或 rescue 配置"
                )
            model_identities.add((model_id, model_revision))
            rescue_margins.add(float(rescue_margin))
            configs.append(config)
            prompt_texts.append(prompt_text)

        rebuilt_records = apply_split_assignments(
            build_prompt_records(paper_run_name, tuple(prompt_texts))
        )
        prompt_rows: list[dict[str, Any]] = []
        prompt_text_bytes: list[bytes] = []
        repeat_randomization_identities: dict[str, dict[str, Any]] = {}
        for config, rebuilt in zip(configs, rebuilt_records, strict=True):
            expected_generation_seed = formal_generation_seed(
                _FROZEN_BASE_GENERATION_SEED,
                rebuilt.prompt_index,
                repeat,
            )
            if (
                config.get("prompt_id") != rebuilt.prompt_id
                or config.get("prompt") != rebuilt.prompt_text
                or config.get("split") != rebuilt.split
                or config.get("seed") != expected_generation_seed
            ):
                raise RandomizationMethodRepeatThresholdError(
                    "runtime Prompt、规范 split 或冻结生成 seed 无法独立重建"
                )
            randomization_identity = {
                "randomization_repeat_id": repeat_id,
                "generation_seed_index": repeat.generation_seed_index,
                "generation_seed_offset": repeat.generation_seed_offset,
                "watermark_key_index": repeat.watermark_key_index,
                "generation_seed_random": expected_generation_seed,
                "watermark_key_seed_random": config[
                    "watermark_key_seed_random"
                ],
                "formal_randomization_protocol_digest": protocol_digest,
                "watermark_key_material_digest_random": config[
                    "key_material_digest_random"
                ],
            }
            randomization_identity[
                "formal_randomization_identity_digest_random"
            ] = build_stable_digest(randomization_identity)
            repeat_randomization_identities[rebuilt.prompt_id] = (
                randomization_identity
            )
            prompt_rows.append(
                {
                    "prompt_id": rebuilt.prompt_id,
                    "prompt_index": rebuilt.prompt_index,
                    "split": rebuilt.split,
                    "prompt_digest": rebuilt.prompt_digest,
                    "prompt_text": rebuilt.prompt_text,
                    "semantic_tags": list(rebuilt.semantic_tags),
                    "risk_profile": rebuilt.risk_profile,
                }
            )
            prompt_text_bytes.append(rebuilt.prompt_text.encode("utf-8"))
        current_prompt_rows = tuple(prompt_rows)
        current_prompt_text_bytes = tuple(prompt_text_bytes)
        if current_prompt_rows != frozen_prompt_rows:
            raise RandomizationMethodRepeatThresholdError(
                "runtime Prompt 未逐条匹配内嵌受治理来源字节"
            )
        if canonical_prompt_rows is None:
            canonical_prompt_rows = current_prompt_rows
            canonical_prompt_text_bytes = current_prompt_text_bytes
        elif (
            current_prompt_rows != canonical_prompt_rows
            or current_prompt_text_bytes != canonical_prompt_text_bytes
        ):
            raise RandomizationMethodRepeatThresholdError(
                "9个 runtime 成员的 Prompt 字节或字段发生漂移"
            )
        repeat_key_identities = {
            (
                identity["watermark_key_seed_random"],
                identity["watermark_key_material_digest_random"],
            )
            for identity in repeat_randomization_identities.values()
        }
        if len(repeat_key_identities) != 1:
            raise RandomizationMethodRepeatThresholdError(
                "同一 runtime 重复内的 watermark key 身份发生漂移"
            )
        key_seed, key_material_digest = next(iter(repeat_key_identities))
        runtime_source_records.append(
            {
                "randomization_repeat_id": repeat_id,
                "record_member": source.record_member,
                "record_sha256": source.record_sha256,
                "leaf_package_sha256": source.leaf_package_sha256,
                "randomization_repeat_component_sha256": (
                    source.randomization_repeat_component_sha256
                ),
                "randomization_repeat_evidence_manifest_digest": (
                    source.randomization_repeat_evidence_manifest_digest
                ),
                "component_content_digest": source.component_content_digest,
                "watermark_key_seed_random": key_seed,
                "watermark_key_material_digest_random": (
                    key_material_digest
                ),
                "runtime_randomization_identity_digest_random": (
                    build_stable_digest(repeat_randomization_identities)
                ),
            }
        )
        runtime_randomization_identity_map[repeat_id] = (
            repeat_randomization_identities
        )

    if (
        canonical_prompt_rows is None
        or len(model_identities) != 1
        or len(rescue_margins) != 1
    ):
        raise RandomizationMethodRepeatThresholdError(
            "9个 runtime 成员没有冻结共同模型与 rescue 配置"
        )
    key_identities_by_index: dict[int, set[tuple[int, str]]] = {
        key_index: set() for key_index in range(3)
    }
    for repeat_id, prompt_identity_map in (
        runtime_randomization_identity_map.items()
    ):
        repeat = resolve_formal_randomization_repeat(repeat_id)
        key_identities_by_index[repeat.watermark_key_index].update(
            (
                int(identity["watermark_key_seed_random"]),
                str(identity["watermark_key_material_digest_random"]),
            )
            for identity in prompt_identity_map.values()
        )
    if any(
        len(identities) != 1
        for identities in key_identities_by_index.values()
    ):
        raise RandomizationMethodRepeatThresholdError(
            "同一 watermark key 跨 generation seed 的 runtime 身份不一致"
        )
    resolved_key_identities = tuple(
        next(iter(key_identities_by_index[key_index]))
        for key_index in range(3)
    )
    key_plan = formal_watermark_key_plan_record()
    expected_key_identities = tuple(
        (
            int(record["watermark_key_seed_random"]),
            str(record["watermark_key_material_digest_random"]),
        )
        for record in key_plan["watermark_key_records"]
    )
    if (
        len({identity[0] for identity in resolved_key_identities}) != 3
        or len({identity[1] for identity in resolved_key_identities}) != 3
        or resolved_key_identities != expected_key_identities
    ):
        raise RandomizationMethodRepeatThresholdError(
            "三个 runtime watermark key 身份未匹配预注册 key plan"
        )
    model_id, model_revision = next(iter(model_identities))
    if (
        model_id != _FROZEN_METHOD_CONFIG.model_id
        or model_revision != _FROZEN_METHOD_CONFIG.model_revision
    ):
        raise RandomizationMethodRepeatThresholdError(
            "runtime 共同模型未匹配冻结方法配置的 ID/revision"
        )
    return (
        frozen_prompt_rows,
        model_id,
        model_revision,
        next(iter(rescue_margins)),
        tuple(runtime_source_records),
        runtime_randomization_identity_map,
    )


def _require_method_source_pair(
    observation_source: RandomizationAggregateRecordSource,
    declaration_source: RandomizationAggregateRecordSource,
    provenance: RandomizationAggregateProvenance,
) -> None:
    """核对 observation 与声明来自同一活动 leaf 和成员摘要链."""

    _require_source_aggregate_identity(observation_source, provenance)
    _require_source_aggregate_identity(declaration_source, provenance)
    if any(
        getattr(observation_source, field_name)
        != getattr(declaration_source, field_name)
        for field_name in _SOURCE_LINEAGE_FIELDS
    ):
        raise RandomizationMethodRepeatThresholdError(
            "observation 与阈值声明没有绑定同一 leaf 来源"
        )
    if observation_source.record_member == declaration_source.record_member:
        raise RandomizationMethodRepeatThresholdError(
            "observation 与阈值声明不得复用同一成员"
        )


def _normalize_transfer_threshold(
    declaration: Mapping[str, Any],
    *,
    observation_source_sha256: str,
    method_id: str,
    paper_run_name: str,
    target_fpr: float,
    expected_model_id: str,
    expected_model_revision: str,
) -> dict[str, Any]:
    """把三种 method-faithful transfer 阈值规范化为统一字段."""

    generation = declaration.get("generation_protocol")
    threshold = declaration.get("threshold")
    threshold_digest = str(declaration.get("threshold_digest", ""))
    if (
        not isinstance(generation, Mapping)
        or declaration.get("baseline_id") != method_id
        or declaration.get("paper_run_name") != paper_run_name
        or declaration.get("transfer_ready") is not True
        or declaration.get("model_id") != expected_model_id
        or declaration.get("model_revision") != expected_model_revision
        or declaration.get("baseline_observations_sha256")
        != observation_source_sha256
        or generation.get("model_id") != expected_model_id
        or generation.get("model_revision") != expected_model_revision
        or isinstance(threshold, bool)
        or not isinstance(threshold, int | float)
        or not math.isfinite(float(threshold))
        or not _is_sha256(threshold_digest)
    ):
        raise RandomizationMethodRepeatThresholdError(
            f"{method_id} transfer 阈值或共同模型身份无效"
        )
    try:
        declared_target_fpr = float(declaration.get("target_fpr"))
    except (TypeError, ValueError) as exc:
        raise RandomizationMethodRepeatThresholdError(
            f"{method_id} transfer 缺少 target FPR"
        ) from exc
    if not math.isclose(
        declared_target_fpr,
        target_fpr,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RandomizationMethodRepeatThresholdError(
            f"{method_id} transfer target FPR 漂移"
        )
    return {
        "calibrated_detection_threshold": float(threshold),
        "threshold_digest": threshold_digest,
    }


def _normalize_t2smark_threshold(
    candidate_rows: tuple[Mapping[str, Any], ...],
    *,
    target_fpr: float,
    observation_rows: tuple[Mapping[str, Any], ...],
    observation_member: str,
) -> dict[str, Any]:
    """从全部 T2SMark candidate 记录提取唯一校准阈值身份."""

    expected_operating_point = build_fixed_fpr_operating_point(target_fpr)
    expected_observation_digest = build_stable_digest(
        [dict(row) for row in observation_rows]
    )
    threshold_identities: set[tuple[float, str]] = set()
    for candidate in candidate_rows:
        threshold = candidate.get("calibrated_detection_threshold")
        threshold_digest = str(candidate.get("threshold_digest", ""))
        try:
            candidate_target_fpr = float(candidate.get("target_fpr"))
        except (TypeError, ValueError) as exc:
            raise RandomizationMethodRepeatThresholdError(
                "T2SMark candidate 缺少显式 target FPR"
            ) from exc
        if (
            candidate.get("baseline_id") != "t2smark"
            or candidate.get("comparable_operating_point")
            != expected_operating_point
            or candidate.get("evaluation_split") != "test"
            or candidate.get("threshold_source") != FORMAL_THRESHOLD_SOURCE
            or candidate.get("fixed_fpr_baseline_calibration_ready")
            is not True
            or candidate.get("fixed_fpr_observation_evidence_path")
            != observation_member
            or candidate.get("fixed_fpr_observation_evidence_digest")
            != expected_observation_digest
            or not math.isclose(
                candidate_target_fpr,
                target_fpr,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            or isinstance(threshold, bool)
            or not isinstance(threshold, int | float)
            or not math.isfinite(float(threshold))
            or not _is_sha256(threshold_digest)
        ):
            raise RandomizationMethodRepeatThresholdError(
                "T2SMark candidate 阈值协议无效"
            )
        threshold_identities.add((float(threshold), threshold_digest))
    if len(threshold_identities) != 1:
        raise RandomizationMethodRepeatThresholdError(
            "T2SMark candidate 出现多阈值或阈值摘要漂移"
        )
    threshold, threshold_digest = next(iter(threshold_identities))
    return {
        "calibrated_detection_threshold": threshold,
        "threshold_digest": threshold_digest,
    }


def _require_observation_model_identity(
    rows: tuple[Mapping[str, Any], ...],
    *,
    method_id: str,
    expected_model_id: str,
    expected_model_revision: str,
) -> None:
    """强制四个 baseline observation 明示共同生成模型身份."""

    if method_id == "slm_wm":
        return
    if any(
        row.get("generation_model_id") != expected_model_id
        or row.get("generation_model_revision") != expected_model_revision
        for row in rows
    ):
        raise RandomizationMethodRepeatThresholdError(
            f"{method_id} observation 未绑定共同模型 ID/revision"
        )


def _clean_unattacked_negative(row: Mapping[str, Any]) -> bool:
    """选择唯一允许参与 latent 身份复验的 clean negative."""

    return bool(
        row.get("sample_role") == "clean_negative"
        and not str(row.get("attack_id", "")).strip()
        and str(row.get("attack_family", "")).strip() in {"", "clean"}
        and str(row.get("attack_name", "")).strip()
        in {"", "none", "clean", "clean_none"}
    )


def _resolve_observation_torch_dtype(dtype_text: str) -> Any:
    """把持久化 dtype 限制到冻结方法配置声明的 Torch 精度."""

    import torch

    expected_text = f"torch.{_FROZEN_METHOD_CONFIG.latent_torch_dtype}"
    supported = {
        "torch.float16": torch.float16,
        "torch.bfloat16": torch.bfloat16,
        "torch.float32": torch.float32,
    }
    if dtype_text != expected_text or dtype_text not in supported:
        raise RandomizationMethodRepeatThresholdError(
            "主方法 observation 的 base latent dtype 未匹配冻结配置"
        )
    return supported[dtype_text]


def _verify_canonical_base_latent_identities(
    sources: tuple[MethodRepeatObservationSource, ...],
    *,
    runtime_randomization_identity_map: Mapping[
        str,
        Mapping[str, Mapping[str, Any]],
    ],
    expected_model_id: str,
    expected_model_revision: str,
) -> tuple[dict[str, Any], ...]:
    """从规范高斯字节流重建主方法 latent，并约束全部 baseline."""

    source_by_key = {
        (source.randomization_repeat_id, source.method_id): source
        for source in sources
    }
    canonical_cache: dict[
        tuple[int, str, str, str, tuple[int, ...]],
        dict[str, Any],
    ] = {}
    verified_main_by_prompt: dict[tuple[str, str], dict[str, Any]] = {}
    canonical_records: list[dict[str, Any]] = []
    for repeat_id in formal_randomization_repeat_ids():
        expected_identities = runtime_randomization_identity_map[repeat_id]
        main_source = source_by_key[(repeat_id, "slm_wm")]
        clean_rows = tuple(
            row
            for row in main_source.observation_rows
            if _clean_unattacked_negative(row)
        )
        clean_by_prompt: dict[str, list[Mapping[str, Any]]] = {
            prompt_id: [] for prompt_id in expected_identities
        }
        for row in clean_rows:
            prompt_id = str(row.get("prompt_id", ""))
            if prompt_id not in clean_by_prompt:
                raise RandomizationMethodRepeatThresholdError(
                    "主方法 clean negative 混入 runtime exact-set 外 Prompt"
                )
            clean_by_prompt[prompt_id].append(row)
        if any(len(rows) != 1 for rows in clean_by_prompt.values()):
            raise RandomizationMethodRepeatThresholdError(
                "主方法 clean negative 未精确覆盖 runtime Prompt exact-set"
            )
        for prompt_id, expected_identity in expected_identities.items():
            expected_seed = int(
                expected_identity["generation_seed_random"]
            )
            row = clean_by_prompt[prompt_id][0]
            raw_shape = row.get("base_latent_shape")
            dtype_text = str(row.get("base_latent_dtype", ""))
            if (
                not isinstance(raw_shape, list | tuple)
                or not raw_shape
                or any(type(value) is not int or value <= 0 for value in raw_shape)
                or row.get("generation_seed_random") != expected_seed
            ):
                raise RandomizationMethodRepeatThresholdError(
                    "主方法 clean negative 的 latent shape 或生成 seed 无效"
                )
            shape = tuple(int(value) for value in raw_shape)
            cache_key = (
                expected_seed,
                expected_model_id,
                expected_model_revision,
                dtype_text,
                shape,
            )
            identity = canonical_cache.get(cache_key)
            if identity is None:
                _latent, rebuilt_identity = build_canonical_sd35_base_latent(
                    shape=shape,
                    generation_seed_random=expected_seed,
                    model_id=expected_model_id,
                    model_revision=expected_model_revision,
                    device="cpu",
                    dtype=_resolve_observation_torch_dtype(dtype_text),
                )
                identity = dict(rebuilt_identity)
                canonical_cache[cache_key] = identity
                canonical_records.append(
                    {
                        "prompt_id": prompt_id,
                        "generation_seed_index": (
                            resolve_formal_randomization_repeat(
                                repeat_id
                            ).generation_seed_index
                        ),
                        "generation_seed_random": expected_seed,
                        "generation_model_id": expected_model_id,
                        "generation_model_revision": expected_model_revision,
                        **identity,
                    }
                )
            if any(
                row.get(field_name) != expected_value
                for field_name, expected_value in identity.items()
            ):
                raise RandomizationMethodRepeatThresholdError(
                    "主方法 base latent 身份无法从规范高斯字节流重建"
                )
            verified_main_by_prompt[(repeat_id, prompt_id)] = identity

    expected_identity_count = 3 * len(
        next(iter(runtime_randomization_identity_map.values()))
    )
    if len(canonical_records) != expected_identity_count:
        raise RandomizationMethodRepeatThresholdError(
            "规范 base latent 缓存未形成3种 seed 与 Prompt 的精确乘积"
        )
    for repeat_id in formal_randomization_repeat_ids():
        expected_prompt_ids = set(
            runtime_randomization_identity_map[repeat_id]
        )
        for method_id in FIXED_FPR_THRESHOLD_METHOD_IDS[1:]:
            source = source_by_key[(repeat_id, method_id)]
            clean_rows = tuple(
                row
                for row in source.observation_rows
                if _clean_unattacked_negative(row)
            )
            clean_by_prompt = {
                str(row.get("prompt_id", "")): row for row in clean_rows
            }
            if (
                set(clean_by_prompt) != expected_prompt_ids
                or len(clean_by_prompt) != len(clean_rows)
            ):
                raise RandomizationMethodRepeatThresholdError(
                    f"{method_id} clean negative 未精确覆盖 runtime Prompt"
                )
            for prompt_id, row in clean_by_prompt.items():
                main_identity = verified_main_by_prompt[(repeat_id, prompt_id)]
                if any(
                    row.get(field_name) != main_identity[field_name]
                    for field_name in (
                        "base_latent_content_digest_random",
                        "base_latent_identity_digest_random",
                    )
                ):
                    raise RandomizationMethodRepeatThresholdError(
                        f"{method_id} base latent 未匹配已重建主方法身份"
                    )
    return tuple(canonical_records)


def _method_repeat_sources(
    workspace: RandomizationAggregateRecordWorkspace,
    provenance: RandomizationAggregateProvenance,
    *,
    paper_run_name: str,
    target_fpr: float,
    expected_model_id: str,
    expected_model_revision: str,
    runtime_randomization_identity_map: Mapping[
        str,
        Mapping[str, Mapping[str, Any]],
    ],
) -> tuple[MethodRepeatObservationSource, ...]:
    """从工作区构造精确45个不可外部覆写的统计来源."""

    expected_source_count = len(formal_randomization_repeat_ids()) * len(
        FIXED_FPR_THRESHOLD_METHOD_IDS
    )
    if (
        len(workspace.observation_sources) != expected_source_count
        or len(workspace.threshold_binding_sources) != expected_source_count
    ):
        raise RandomizationMethodRepeatThresholdError(
            "聚合工作区未提供精确45个 observation 与45个阈值声明"
        )
    sources: list[MethodRepeatObservationSource] = []
    for repeat_id in formal_randomization_repeat_ids():
        for method_id in FIXED_FPR_THRESHOLD_METHOD_IDS:
            package_family = METHOD_LEAF_PACKAGE_FAMILY[method_id]
            observation_role, declaration_role = _METHOD_SOURCE_ROLES[
                method_id
            ]
            observation_source = workspace.find_source(
                randomization_repeat_id=repeat_id,
                package_family=package_family,
                record_role=observation_role,
            )
            declaration_source = workspace.find_source(
                randomization_repeat_id=repeat_id,
                package_family=package_family,
                record_role=declaration_role,
            )
            _require_method_source_pair(
                observation_source,
                declaration_source,
                provenance,
            )
            observation_rows = tuple(
                workspace.iter_records(observation_source)
            )
            if method_id == "slm_wm":
                expected_identities = (
                    runtime_randomization_identity_map.get(repeat_id)
                )
                observed_prompt_ids = {
                    str(row.get("prompt_id", ""))
                    for row in observation_rows
                }
                if (
                    not isinstance(expected_identities, Mapping)
                    or observed_prompt_ids != set(expected_identities)
                    or any(
                        str(row.get("prompt_id", ""))
                        not in expected_identities
                        or any(
                            row.get(field_name)
                            != expected_identities[
                                str(row.get("prompt_id", ""))
                            ][field_name]
                            for field_name in (
                                _RUNTIME_RANDOMIZATION_IDENTITY_FIELDS
                            )
                        )
                        for row in observation_rows
                    )
                ):
                    raise RandomizationMethodRepeatThresholdError(
                        "主方法 observation 随机身份未逐 Prompt 对齐 runtime"
                    )
            _require_observation_model_identity(
                observation_rows,
                method_id=method_id,
                expected_model_id=expected_model_id,
                expected_model_revision=expected_model_revision,
            )
            if method_id == "slm_wm":
                declared_protocol = dict(
                    workspace.read_object(declaration_source)
                )
            elif method_id == "t2smark":
                declared_protocol = _normalize_t2smark_threshold(
                    tuple(workspace.iter_records(declaration_source)),
                    target_fpr=target_fpr,
                    observation_rows=observation_rows,
                    observation_member=observation_source.record_member,
                )
            else:
                declared_protocol = _normalize_transfer_threshold(
                    workspace.read_object(declaration_source),
                    observation_source_sha256=(
                        observation_source.record_sha256
                    ),
                    method_id=method_id,
                    paper_run_name=paper_run_name,
                    target_fpr=target_fpr,
                    expected_model_id=expected_model_id,
                    expected_model_revision=expected_model_revision,
                )
            sources.append(
                MethodRepeatObservationSource(
                    paper_run_name=paper_run_name,
                    method_id=method_id,
                    randomization_repeat_id=repeat_id,
                    generation_model_id=expected_model_id,
                    generation_model_revision=expected_model_revision,
                    randomization_aggregate_package_sha256=(
                        observation_source.randomization_aggregate_package_sha256
                    ),
                    randomization_aggregate_digest=(
                        observation_source.randomization_aggregate_digest
                    ),
                    common_code_version=observation_source.common_code_version,
                    repeat_component_archive_member=(
                        f"repeat_components/{repeat_id}.zip"
                    ),
                    randomization_repeat_component_sha256=(
                        observation_source.randomization_repeat_component_sha256
                    ),
                    randomization_repeat_evidence_manifest_digest=(
                        observation_source.randomization_repeat_evidence_manifest_digest
                    ),
                    component_content_digest=(
                        observation_source.component_content_digest
                    ),
                    leaf_package_family=package_family,
                    leaf_package_archive_member=(
                        f"randomization_repeat_evidence/{repeat_id}/"
                        f"leaf_packages/{package_family}.zip"
                    ),
                    leaf_package_sha256=(
                        observation_source.leaf_package_sha256
                    ),
                    observation_archive_member=(
                        observation_source.record_member
                    ),
                    observation_source_sha256=(
                        observation_source.record_sha256
                    ),
                    threshold_declaration_archive_member=(
                        declaration_source.record_member
                    ),
                    threshold_declaration_source_sha256=(
                        declaration_source.record_sha256
                    ),
                    declared_threshold_protocol=declared_protocol,
                    observation_rows=observation_rows,
                )
            )
    return tuple(sources)


def recompute_randomization_method_repeat_fixed_fpr(
    source: RandomizationAggregateProvenance,
) -> dict[str, Any]:
    """仅从精确聚合 provenance 重算45个独立 fixed-FPR 阈值."""

    if not isinstance(source, RandomizationAggregateProvenance):
        raise TypeError("阈值桥接入口只接受 RandomizationAggregateProvenance")
    try:
        paper_run_name = normalize_paper_run_name(
            str(source.payload["paper_run_name"])
        )
        target_fpr = validate_frozen_paper_run_target_fpr(
            paper_run_name,
            float(source.payload["target_fpr"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise RandomizationMethodRepeatThresholdError(
            "聚合 provenance 缺少冻结论文运行身份"
        ) from exc
    with open_randomization_aggregate_record_workspace(source) as workspace:
        prompt_source_contract = (
            rebuild_randomization_prompt_source_contract(
                workspace,
                source,
                paper_run_name=paper_run_name,
            )
        )
        prompt_source_rows = prompt_source_contract.get("prompt_rows")
        prompt_source_report = prompt_source_contract.get("report")
        if (
            not isinstance(prompt_source_rows, tuple)
            or not isinstance(prompt_source_report, Mapping)
            or prompt_source_report.get("prompt_source_contract_ready")
            is not True
            or prompt_source_report.get("supports_paper_claim") is not False
        ):
            raise RandomizationMethodRepeatThresholdError(
                "aggregate 内嵌 Prompt 来源契约未通过"
            )
        (
            prompt_rows,
            expected_model_id,
            expected_model_revision,
            main_rescue_margin_low,
            runtime_source_records,
            runtime_randomization_identity_map,
        ) = _runtime_prompt_rows(
            workspace,
            source,
            paper_run_name=paper_run_name,
            expected_prompt_rows=prompt_source_rows,
        )
        method_sources = _method_repeat_sources(
            workspace,
            source,
            paper_run_name=paper_run_name,
            target_fpr=target_fpr,
            expected_model_id=expected_model_id,
            expected_model_revision=expected_model_revision,
            runtime_randomization_identity_map=(
                runtime_randomization_identity_map
            ),
        )
        canonical_base_latent_records = (
            _verify_canonical_base_latent_identities(
                method_sources,
                runtime_randomization_identity_map=(
                    runtime_randomization_identity_map
                ),
                expected_model_id=expected_model_id,
                expected_model_revision=expected_model_revision,
            )
        )
        result = recompute_exact_method_repeat_fixed_fpr(
            method_sources,
            prompt_rows=prompt_rows,
            paper_run_name=paper_run_name,
            target_fpr=target_fpr,
            expected_model_id=expected_model_id,
            expected_model_revision=expected_model_revision,
            main_rescue_margin_low=main_rescue_margin_low,
            expected_base_seed=_FROZEN_BASE_GENERATION_SEED,
        )
    threshold_report = result.get("report")
    if not isinstance(threshold_report, Mapping):
        raise RandomizationMethodRepeatThresholdError(
            "逐方法逐重复重算没有返回摘要锁定的报告"
        )
    threshold_report_digest = str(
        threshold_report.get("method_repeat_fixed_fpr_report_digest", "")
    )
    if not _is_sha256(threshold_report_digest):
        raise RandomizationMethodRepeatThresholdError(
            "逐方法逐重复报告缺少规范摘要"
        )
    if threshold_report.get(
        "method_repeat_fixed_fpr_recomputation_ready"
    ) is not True:
        raise RandomizationMethodRepeatThresholdError(
            "逐方法逐重复纯重算尚未通过"
        )
    runtime_source_record_map = {
        str(record["randomization_repeat_id"]): dict(record)
        for record in runtime_source_records
    }
    if tuple(runtime_source_record_map) != formal_randomization_repeat_ids():
        raise RandomizationMethodRepeatThresholdError(
            "runtime 来源摘要映射未精确覆盖9个重复"
        )
    reconstruction_report = {
        "report_schema": (
            "randomization_method_repeat_threshold_reconstruction_report"
        ),
        "paper_claim_scale": paper_run_name,
        "target_fpr": target_fpr,
        "randomization_aggregate_package_sha256": source.package_sha256,
        "randomization_aggregate_digest": (
            source.randomization_aggregate_digest
        ),
        "common_code_version": source.common_code_version,
        "runtime_source_record_map": runtime_source_record_map,
        "runtime_source_records_digest": build_stable_digest(
            runtime_source_records
        ),
        "runtime_randomization_identity_count": sum(
            len(identity_map)
            for identity_map in runtime_randomization_identity_map.values()
        ),
        "runtime_randomization_identity_digest_random": (
            build_stable_digest(runtime_randomization_identity_map)
        ),
        "formal_watermark_key_plan_digest": (
            formal_watermark_key_plan_record()[
                "formal_watermark_key_plan_digest"
            ]
        ),
        "prompt_source_contract_digest": prompt_source_report.get(
            "prompt_source_contract_digest"
        ),
        "prompt_source_records_digest": prompt_source_report.get(
            "prompt_source_records_digest"
        ),
        "prompt_source_record_map": prompt_source_report.get(
            "prompt_source_record_map"
        ),
        "prompt_file_sha256": prompt_source_report.get(
            "prompt_file_sha256"
        ),
        "prompt_source_registry_digest": prompt_source_report.get(
            "prompt_source_registry_digest"
        ),
        "selection_manifest_sha256": prompt_source_report.get(
            "selection_manifest_sha256"
        ),
        "selection_manifest_digest": prompt_source_report.get(
            "selection_manifest_digest"
        ),
        "packaged_prompt_source_audit_digest": prompt_source_report.get(
            "packaged_prompt_source_audit_digest"
        ),
        "prompt_protocol_digest": threshold_report.get(
            "prompt_protocol_digest"
        ),
        "method_repeat_fixed_fpr_report_digest": threshold_report_digest,
        "threshold_records_digest": threshold_report.get(
            "threshold_records_digest"
        ),
        "canonical_base_latent_identity_count": len(
            canonical_base_latent_records
        ),
        "canonical_base_latent_identity_digest": build_stable_digest(
            canonical_base_latent_records
        ),
        "exact_runtime_source_count": len(runtime_source_records),
        "exact_method_repeat_fixed_fpr_ready": True,
        "supports_paper_claim": False,
    }
    reconstruction_report["reconstruction_report_digest"] = (
        build_stable_digest(reconstruction_report)
    )
    return {
        **result,
        "reconstruction_report": reconstruction_report,
    }


__all__ = [
    "RandomizationMethodRepeatThresholdError",
    "recompute_randomization_method_repeat_fixed_fpr",
]
