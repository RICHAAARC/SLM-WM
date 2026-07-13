"""验证精确聚合来源到45个 fixed-FPR 阈值的受限桥接."""

from __future__ import annotations

import copy
from dataclasses import replace
import hashlib
import inspect
from pathlib import Path
from typing import Any, Mapping

import pytest

from experiments.protocol.fixed_fpr_observation_audit import (
    FORMAL_THRESHOLD_SOURCE,
    audit_fixed_fpr_observation_threshold,
    conformal_threshold_from_clean_negative_scores,
)
from experiments.protocol.formal_randomization import (
    formal_generation_seed,
    formal_randomization_protocol_record,
    formal_randomization_repeat_ids,
    formal_randomization_repeats,
    formal_watermark_key_material_from_seed,
    formal_watermark_key_plan_record,
    resolve_formal_randomization_repeat,
)
from experiments.protocol.prompts import build_prompt_records
from experiments.protocol.prompt_sources import (
    PROMPT_CONFIG_NAMES,
    PROMPT_SELECTION_MANIFEST_PATH,
    PROMPT_SOURCE_REGISTRY_PATH,
)
from experiments.protocol.splits import apply_split_assignments
from experiments.runners.semantic_watermark_runtime import (
    SemanticWatermarkRuntimeConfig,
    semantic_watermark_runtime_config_payload,
)
from experiments.runners.image_only_dataset_runtime import (
    apply_frozen_evidence_protocol,
    calibrate_complete_evidence_protocol,
)
from main.core.digest import build_stable_digest
from paper_experiments.analysis.fixed_fpr_threshold_audit import (
    FIXED_FPR_THRESHOLD_METHOD_IDS,
)
from paper_experiments.analysis.method_repeat_fixed_fpr import (
    METHOD_LEAF_PACKAGE_FAMILY,
    recompute_exact_method_repeat_fixed_fpr,
)
from paper_experiments.runners.randomization_aggregate_provenance import (
    RandomizationAggregateProvenance,
)
from paper_experiments.runners.randomization_aggregate_record_workspace import (
    RECORD_FORMAT_JSONL,
    RECORD_FORMAT_JSON_ARRAY,
    RECORD_FORMAT_JSON_OBJECT,
    RECORD_FORMAT_RAW_BYTES,
    RandomizationAggregateRecordSource,
)
from paper_experiments.runners import (
    randomization_method_repeat_thresholds as bridge_module,
)
from paper_experiments.runners.randomization_method_repeat_thresholds import (
    RandomizationMethodRepeatThresholdError,
    recompute_randomization_method_repeat_fixed_fpr,
)


pytestmark = pytest.mark.quick

PAPER_RUN_NAME = "probe_paper"
TARGET_FPR = 0.1
MODEL_ID = "stabilityai/stable-diffusion-3.5-medium"
MODEL_REVISION = "b940f670f0eda2d07fbb75229e779da1ad11eb80"
CODE_VERSION = "a" * 40
BASE_SEED = 1703
RESCUE_MARGIN_LOW = -0.05
AGGREGATE_PACKAGE_SHA256 = build_stable_digest({"aggregate_package": 1})
AGGREGATE_DIGEST = build_stable_digest({"aggregate": 1})


def _watermark_key_seed_random(repeat: Any) -> int:
    """返回 quick 工作区中按 key index 冻结的公开 key seed."""

    record = formal_watermark_key_plan_record()["watermark_key_records"][
        int(repeat.watermark_key_index)
    ]
    return int(record["watermark_key_seed_random"])


def _watermark_key_material_digest_random(
    repeat: Any,
    *,
    key_seed: int | None = None,
) -> str:
    """按正式 key material 结构构造不含原始 secret 的摘要."""

    resolved_seed = (
        _watermark_key_seed_random(repeat)
        if key_seed is None
        else key_seed
    )
    return build_stable_digest(
        {
            "key_material": formal_watermark_key_material_from_seed(
                resolved_seed,
                repeat,
            )
        }
    )


def _formal_randomization_identity(
    repeat: Any,
    generation_seed_random: int,
    *,
    key_seed: int | None = None,
) -> dict[str, Any]:
    """构造 observation 与 runtime 必须共享的完整随机身份."""

    resolved_key_seed = (
        _watermark_key_seed_random(repeat)
        if key_seed is None
        else key_seed
    )
    identity = {
        "randomization_repeat_id": repeat.randomization_repeat_id,
        "generation_seed_index": repeat.generation_seed_index,
        "generation_seed_offset": repeat.generation_seed_offset,
        "watermark_key_index": repeat.watermark_key_index,
        "generation_seed_random": generation_seed_random,
        "watermark_key_seed_random": resolved_key_seed,
        "formal_randomization_protocol_digest": (
            formal_randomization_protocol_record()[
                "formal_randomization_protocol_digest"
            ]
        ),
        "watermark_key_material_digest_random": (
            _watermark_key_material_digest_random(
                repeat,
                key_seed=resolved_key_seed,
            )
        ),
    }
    identity["formal_randomization_identity_digest_random"] = (
        build_stable_digest(identity)
    )
    return identity


def _base_latent_identity(generation_seed_random: int) -> dict[str, Any]:
    """构造由 quick 测试生成器逐字段返回的 latent 身份."""

    identity = {
        "generation_seed_random": generation_seed_random,
        "base_latent_generation_protocol": (
            "device_independent_sha256_normal_icdf_table20_"
            "cpu_dtype_cast_then_device_transfer_v2"
        ),
        "base_latent_keyed_prg_version": (
            "sha256_counter_normal_icdf_table20_float32_v2"
        ),
        "base_latent_keyed_prg_protocol_digest": (
            formal_randomization_protocol_record()[
                "base_latent_keyed_prg_protocol_digest"
            ]
        ),
        "formal_randomization_protocol_digest": (
            formal_randomization_protocol_record()[
                "formal_randomization_protocol_digest"
            ]
        ),
        "base_latent_dtype": "torch.float16",
        "base_latent_shape": [1, 16, 64, 64],
        "base_latent_content_digest_random": build_stable_digest(
            {"canonical_latent_seed": generation_seed_random}
        ),
    }
    identity["base_latent_identity_digest_random"] = build_stable_digest(
        identity
    )
    return identity


def _member_paths(method_id: str) -> tuple[str, str]:
    """返回一个方法的 observation 与声明规范成员."""

    if method_id == "slm_wm":
        prefix = f"outputs/image_only_dataset_runtime/{PAPER_RUN_NAME}/"
        return (
            prefix + "image_only_detection_records.jsonl",
            prefix + "frozen_evidence_protocol.json",
        )
    if method_id == "t2smark":
        prefix = f"outputs/t2smark_formal_reproduction/{PAPER_RUN_NAME}/"
        return (
            prefix + "t2smark_adapter/baseline_observations.json",
            prefix + "t2smark_formal_import_candidate_records.jsonl",
        )
    prefix = (
        f"outputs/external_baseline_method_faithful/{PAPER_RUN_NAME}/"
        "split_observations/"
    )
    return (
        prefix + f"{method_id}_baseline_observations.json",
        prefix + f"{method_id}_baseline_transfer_manifest.json",
    )


def _method_roles(method_id: str) -> tuple[str, str]:
    """返回桥接登记的两个方法来源角色."""

    return bridge_module._METHOD_SOURCE_ROLES[method_id]


class _MemoryAggregateWorkspace:
    """在内存中模拟已经独立验证的工作区读取接口."""

    def __init__(self, provenance: RandomizationAggregateProvenance) -> None:
        self.provenance = provenance
        self._sources: dict[
            tuple[str, str, str], RandomizationAggregateRecordSource
        ] = {}
        self._rows: dict[
            tuple[str, str, str], tuple[Mapping[str, Any], ...]
        ] = {}
        self._objects: dict[tuple[str, str, str], Mapping[str, Any]] = {}
        self._bytes: dict[tuple[str, str, str], bytes] = {}
        self._runtime_property_sources: list[
            RandomizationAggregateRecordSource
        ] = []
        self._prompt_source_property_sources: list[
            RandomizationAggregateRecordSource
        ] = []
        self._build_sources()

    def __enter__(self) -> "_MemoryAggregateWorkspace":
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        return None

    @property
    def prompt_runtime_sources(
        self,
    ) -> tuple[RandomizationAggregateRecordSource, ...]:
        return tuple(self._runtime_property_sources)

    @property
    def prompt_source_sources(
        self,
    ) -> tuple[RandomizationAggregateRecordSource, ...]:
        return tuple(self._prompt_source_property_sources)

    @property
    def observation_sources(
        self,
    ) -> tuple[RandomizationAggregateRecordSource, ...]:
        return tuple(
            source
            for source in self._sources.values()
            if source.record_role.endswith("observation")
        )

    @property
    def threshold_binding_sources(
        self,
    ) -> tuple[RandomizationAggregateRecordSource, ...]:
        declaration_roles = {
            roles[1] for roles in bridge_module._METHOD_SOURCE_ROLES.values()
        }
        return tuple(
            source
            for source in self._sources.values()
            if source.record_role in declaration_roles
        )

    def find_source(
        self,
        *,
        randomization_repeat_id: str,
        package_family: str,
        record_role: str,
    ) -> RandomizationAggregateRecordSource:
        return self._sources[
            (randomization_repeat_id, package_family, record_role)
        ]

    def iter_records(
        self,
        source: RandomizationAggregateRecordSource,
    ) -> Any:
        key = (
            source.randomization_repeat_id,
            source.package_family,
            source.record_role,
        )
        return iter(self._rows[key])

    def read_object(
        self,
        source: RandomizationAggregateRecordSource,
    ) -> Mapping[str, Any]:
        key = (
            source.randomization_repeat_id,
            source.package_family,
            source.record_role,
        )
        return self._objects[key]

    def read_bytes(
        self,
        source: RandomizationAggregateRecordSource,
    ) -> bytes:
        key = (
            source.randomization_repeat_id,
            source.package_family,
            source.record_role,
        )
        return self._bytes[key]

    def _source(
        self,
        *,
        repeat_id: str,
        package_family: str,
        record_role: str,
        record_member: str,
        record_format: str,
        record_sha256: str | None = None,
    ) -> RandomizationAggregateRecordSource:
        repeat_record = next(
            record
            for record in self.provenance.randomization_repeat_components
            if record["randomization_repeat_id"] == repeat_id
        )
        return RandomizationAggregateRecordSource(
            randomization_scope="active_repeat_component",
            randomization_repeat_id=repeat_id,
            package_family=package_family,
            record_group="test_source",
            record_role=record_role,
            record_format=record_format,
            record_member=record_member,
            record_sha256=(
                record_sha256
                or build_stable_digest(
                    {
                        "repeat_id": repeat_id,
                        "package_family": package_family,
                        "record_role": record_role,
                    }
                )
            ),
            leaf_package_sha256=build_stable_digest(
                {"repeat_id": repeat_id, "package_family": package_family}
            ),
            randomization_repeat_component_sha256=str(
                repeat_record["package_sha256"]
            ),
            randomization_repeat_evidence_manifest_digest=str(
                repeat_record[
                    "randomization_repeat_evidence_manifest_digest"
                ]
            ),
            component_content_digest=str(
                repeat_record["component_content_digest"]
            ),
            randomization_aggregate_package_sha256=AGGREGATE_PACKAGE_SHA256,
            common_code_version=CODE_VERSION,
            randomization_aggregate_digest=AGGREGATE_DIGEST,
        )

    def _build_sources(self) -> None:
        """构造9个 runtime 来源与45对方法来源."""

        repository_root = Path(__file__).resolve().parents[2]
        prompt_texts = tuple(
            (
                repository_root
                / "configs"
                / PROMPT_CONFIG_NAMES[PAPER_RUN_NAME]
            )
            .read_text(encoding="utf-8")
            .splitlines()
        )
        prompt_records = apply_split_assignments(
            build_prompt_records(PAPER_RUN_NAME, prompt_texts)
        )
        protocol_digest = formal_randomization_protocol_record()[
            "formal_randomization_protocol_digest"
        ]
        for repeat in formal_randomization_repeats():
            repeat_id = repeat.randomization_repeat_id
            runtime_role = "semantic_watermark_runtime_record"
            runtime_source = self._source(
                repeat_id=repeat_id,
                package_family="image_only_dataset_runtime",
                record_role=runtime_role,
                record_member=(
                    f"outputs/image_only_dataset_runtime/{PAPER_RUN_NAME}/"
                    "runtime_results.jsonl"
                ),
                record_format=RECORD_FORMAT_JSONL,
            )
            runtime_key = (
                repeat_id,
                "image_only_dataset_runtime",
                runtime_role,
            )
            self._sources[runtime_key] = runtime_source
            self._runtime_property_sources.append(runtime_source)
            runtime_rows = []
            for prompt in prompt_records:
                key_seed = _watermark_key_seed_random(repeat)
                runtime_config = SemanticWatermarkRuntimeConfig(
                    prompt_id=prompt.prompt_id,
                    prompt=prompt.prompt_text,
                    split=prompt.split,
                    key_material=formal_watermark_key_material_from_seed(
                        key_seed,
                        repeat,
                    ),
                    seed=formal_generation_seed(
                        BASE_SEED,
                        prompt.prompt_index,
                        repeat,
                    ),
                    randomization_repeat_id=repeat_id,
                    generation_seed_index=repeat.generation_seed_index,
                    generation_seed_offset=repeat.generation_seed_offset,
                    watermark_key_index=repeat.watermark_key_index,
                    watermark_key_seed_random=key_seed,
                    formal_randomization_protocol_digest=protocol_digest,
                    standard_attack_profiles=(
                        ("full_main",) if prompt.split == "test" else ()
                    ),
                    diffusion_attacks_enabled=(prompt.split == "test"),
                    output_dir=(
                        f"outputs/image_only_dataset_runtime/"
                        f"{PAPER_RUN_NAME}/runs"
                    ),
                )
                config = semantic_watermark_runtime_config_payload(
                    runtime_config
                )
                runtime_rows.append(
                    {
                        "run_id": "runtime_"
                        + build_stable_digest(config)[:16],
                        "metadata": {"scientific_unit_config": config},
                    }
                )
            self._rows[runtime_key] = tuple(runtime_rows)

            prompt_source_payloads = {
                "governed_prompt_file_bytes": (
                    f"outputs/image_only_dataset_runtime/{PAPER_RUN_NAME}/"
                    "prompt_source_snapshot/"
                    f"{PROMPT_CONFIG_NAMES[PAPER_RUN_NAME]}",
                    (
                        repository_root
                        / "configs"
                        / PROMPT_CONFIG_NAMES[PAPER_RUN_NAME]
                    ).read_bytes(),
                ),
                "governed_prompt_selection_manifest_bytes": (
                    f"outputs/image_only_dataset_runtime/{PAPER_RUN_NAME}/"
                    "prompt_source_snapshot/prompt_selection_manifest.jsonl",
                    (
                        repository_root / PROMPT_SELECTION_MANIFEST_PATH
                    ).read_bytes(),
                ),
                "governed_prompt_source_registry_bytes": (
                    f"outputs/image_only_dataset_runtime/{PAPER_RUN_NAME}/"
                    "prompt_source_snapshot/prompt_source_registry.json",
                    (
                        repository_root / PROMPT_SOURCE_REGISTRY_PATH
                    ).read_bytes(),
                ),
            }
            for role, (member_name, payload) in (
                prompt_source_payloads.items()
            ):
                prompt_source = self._source(
                    repeat_id=repeat_id,
                    package_family="image_only_dataset_runtime",
                    record_role=role,
                    record_member=member_name,
                    record_format=RECORD_FORMAT_RAW_BYTES,
                    record_sha256=hashlib.sha256(payload).hexdigest(),
                )
                prompt_source_key = (
                    repeat_id,
                    "image_only_dataset_runtime",
                    role,
                )
                self._sources[prompt_source_key] = prompt_source
                self._bytes[prompt_source_key] = payload
                self._prompt_source_property_sources.append(prompt_source)

            for method_id in FIXED_FPR_THRESHOLD_METHOD_IDS:
                package_family = METHOD_LEAF_PACKAGE_FAMILY[method_id]
                observation_role, declaration_role = _method_roles(method_id)
                observation_member, declaration_member = _member_paths(
                    method_id
                )
                observation_source = self._source(
                    repeat_id=repeat_id,
                    package_family=package_family,
                    record_role=observation_role,
                    record_member=observation_member,
                    record_format=RECORD_FORMAT_JSON_ARRAY,
                )
                declaration_source = self._source(
                    repeat_id=repeat_id,
                    package_family=package_family,
                    record_role=declaration_role,
                    record_member=declaration_member,
                    record_format=(
                        RECORD_FORMAT_JSONL
                        if method_id == "t2smark"
                        else RECORD_FORMAT_JSON_OBJECT
                    ),
                )
                observation_key = (
                    repeat_id,
                    package_family,
                    observation_role,
                )
                declaration_key = (
                    repeat_id,
                    package_family,
                    declaration_role,
                )
                self._sources[observation_key] = observation_source
                self._sources[declaration_key] = declaration_source
                observations = []
                method_index = FIXED_FPR_THRESHOLD_METHOD_IDS.index(
                    method_id
                )
                score_offset = (
                    repeat.generation_seed_index * 0.1
                    + method_index * 0.01
                )
                for prompt in prompt_records:
                    generation_seed_random = formal_generation_seed(
                        BASE_SEED,
                        prompt.prompt_index,
                        repeat,
                    )
                    observation = {
                        "prompt_id": prompt.prompt_id,
                        "split": prompt.split,
                        "sample_role": "clean_negative",
                        "attack_id": "",
                        "attack_family": (
                            "" if method_id == "slm_wm" else "clean"
                        ),
                        "attack_name": (
                            "" if method_id == "slm_wm" else "clean_none"
                        ),
                        **_formal_randomization_identity(
                            repeat,
                            generation_seed_random,
                        ),
                        **_base_latent_identity(generation_seed_random),
                    }
                    if method_id != "slm_wm":
                        observation.update(
                            {
                                "baseline_id": method_id,
                                "prompt_text": prompt.prompt_text,
                                "generation_model_id": MODEL_ID,
                                "generation_model_revision": MODEL_REVISION,
                                "score": (
                                    score_offset
                                    + prompt.prompt_index / 1000.0
                                ),
                            }
                        )
                    else:
                        observation.update(
                            {
                                "content_score": (
                                    score_offset
                                    + prompt.prompt_index / 1000.0
                                ),
                                "aligned_content_score": (
                                    score_offset
                                    + prompt.prompt_index / 1000.0
                                ),
                                "attention_geometry_score": 0.0,
                                "registration_confidence": 0.0,
                                "attention_sync_score": 0.0,
                                "alignment": {
                                    "registration_geometry_reliable": False,
                                },
                            }
                        )
                    observations.append(observation)
                calibration_rows = tuple(
                    row
                    for row in observations
                    if row["split"] == "calibration"
                )
                if method_id == "slm_wm":
                    protocol = calibrate_complete_evidence_protocol(
                        calibration_rows,
                        TARGET_FPR,
                        RESCUE_MARGIN_LOW,
                    )
                    observations = list(
                        apply_frozen_evidence_protocol(
                            observations,
                            protocol,
                        )
                    )
                    self._objects[declaration_key] = protocol.to_dict()
                    self._rows[observation_key] = tuple(observations)
                    continue

                threshold = conformal_threshold_from_clean_negative_scores(
                    (float(row["score"]) for row in calibration_rows),
                    TARGET_FPR,
                )
                observations = [
                    {
                        **row,
                        "threshold": threshold,
                        "threshold_source": FORMAL_THRESHOLD_SOURCE,
                        "detection_decision": (
                            float(row["score"]) >= threshold
                        ),
                    }
                    for row in observations
                ]
                self._rows[observation_key] = tuple(observations)
                threshold_audit = audit_fixed_fpr_observation_threshold(
                    observations,
                    target_fpr=TARGET_FPR,
                    expected_calibration_negative_count=33,
                )
                threshold_digest = threshold_audit.threshold_digest
                if method_id == "t2smark":
                    observation_digest = build_stable_digest(observations)
                    candidate = {
                        "baseline_id": "t2smark",
                        "comparable_operating_point": "fixed_fpr_0.1",
                        "target_fpr": TARGET_FPR,
                        "evaluation_split": "test",
                        "threshold_source": FORMAL_THRESHOLD_SOURCE,
                        "fixed_fpr_baseline_calibration_ready": True,
                        "calibrated_detection_threshold": threshold,
                        "threshold_digest": threshold_digest,
                        "fixed_fpr_observation_evidence_path": (
                            observation_member
                        ),
                        "fixed_fpr_observation_evidence_digest": (
                            observation_digest
                        ),
                    }
                    self._rows[declaration_key] = (
                        candidate,
                        {**candidate, "attack_id": "jpeg"},
                    )
                else:
                    self._objects[declaration_key] = {
                        "baseline_id": method_id,
                        "paper_run_name": PAPER_RUN_NAME,
                        "transfer_ready": True,
                        "model_id": MODEL_ID,
                        "model_revision": MODEL_REVISION,
                        "generation_protocol": {
                            "model_id": MODEL_ID,
                            "model_revision": MODEL_REVISION,
                        },
                        "target_fpr": TARGET_FPR,
                        "threshold": threshold,
                        "threshold_digest": threshold_digest,
                        "baseline_observations_sha256": (
                            observation_source.record_sha256
                        ),
                    }

    def replace_runtime_prompt_text(
        self,
        repeat_id: str,
        prompt_index: int,
        prompt_text: str,
    ) -> None:
        """保持单个重复内部自洽，同时制造跨重复 Prompt 漂移."""

        key = (
            repeat_id,
            "image_only_dataset_runtime",
            "semantic_watermark_runtime_record",
        )
        rows = [dict(row) for row in self._rows[key]]
        texts = [
            str(row["metadata"]["scientific_unit_config"]["prompt"])
            for row in rows
        ]
        texts[prompt_index] = prompt_text
        rebuilt = apply_split_assignments(
            build_prompt_records(PAPER_RUN_NAME, tuple(texts))
        )
        for row, prompt in zip(rows, rebuilt, strict=True):
            metadata = dict(row["metadata"])
            config = dict(metadata["scientific_unit_config"])
            config.update(
                {
                    "prompt_id": prompt.prompt_id,
                    "prompt": prompt.prompt_text,
                    "split": prompt.split,
                    "standard_attack_profiles": (
                        ["full_main"] if prompt.split == "test" else []
                    ),
                    "diffusion_attacks_enabled": (
                        prompt.split == "test"
                    ),
                }
            )
            metadata["scientific_unit_config"] = config
            row["metadata"] = metadata
            row["run_id"] = "runtime_" + build_stable_digest(config)[:16]
        self._rows[key] = tuple(rows)


def _replace_runtime_key_identity(
    workspace: _MemoryAggregateWorkspace,
    repeat_id: str,
    forged_key_seed: int,
) -> None:
    """保持 runtime 配置内部自洽地替换一个重复的 key 身份."""

    repeat = resolve_formal_randomization_repeat(repeat_id)
    key = (
        repeat_id,
        "image_only_dataset_runtime",
        "semantic_watermark_runtime_record",
    )
    rows = [dict(row) for row in workspace._rows[key]]
    forged_material_digest = _watermark_key_material_digest_random(
        repeat,
        key_seed=forged_key_seed,
    )
    for row in rows:
        metadata = dict(row["metadata"])
        config = dict(metadata["scientific_unit_config"])
        config["watermark_key_seed_random"] = forged_key_seed
        config["key_material_digest_random"] = forged_material_digest
        metadata["scientific_unit_config"] = config
        row["metadata"] = metadata
        row["run_id"] = "runtime_" + build_stable_digest(config)[:16]
    workspace._rows[key] = tuple(rows)


def _replace_all_runtime_config_field(
    workspace: _MemoryAggregateWorkspace,
    field_name: str,
    value: Any,
) -> None:
    """在9个 runtime 中同步替换字段并重算测试 run id."""

    for repeat_id in formal_randomization_repeat_ids():
        key = (
            repeat_id,
            "image_only_dataset_runtime",
            "semantic_watermark_runtime_record",
        )
        rows = [dict(row) for row in workspace._rows[key]]
        for row in rows:
            metadata = dict(row["metadata"])
            config = dict(metadata["scientific_unit_config"])
            config[field_name] = value
            metadata["scientific_unit_config"] = config
            row["metadata"] = metadata
            row["run_id"] = "runtime_" + build_stable_digest(config)[:16]
        workspace._rows[key] = tuple(rows)


def _replace_observation_key_identity(
    workspace: _MemoryAggregateWorkspace,
    repeat_id: str,
    forged_key_seed: int,
) -> None:
    """让五方法 observation 共享同一伪造 key, 并重算其身份摘要."""

    repeat = resolve_formal_randomization_repeat(repeat_id)
    for method_id in FIXED_FPR_THRESHOLD_METHOD_IDS:
        observation_role, _declaration_role = _method_roles(method_id)
        key = (
            repeat_id,
            METHOD_LEAF_PACKAGE_FAMILY[method_id],
            observation_role,
        )
        rows = [dict(row) for row in workspace._rows[key]]
        for row in rows:
            row.update(
                _formal_randomization_identity(
                    repeat,
                    int(row["generation_seed_random"]),
                    key_seed=forged_key_seed,
                )
            )
        workspace._rows[key] = tuple(rows)


def _provenance(tmp_path: Path) -> RandomizationAggregateProvenance:
    """构造桥接入口需要的不可变精确聚合身份."""

    repeat_components = tuple(
        {
            "randomization_repeat_id": repeat_id,
            "package_sha256": build_stable_digest(
                {"repeat_component": repeat_id}
            ),
            "randomization_repeat_evidence_manifest_digest": (
                build_stable_digest({"repeat_manifest": repeat_id})
            ),
            "component_content_digest": build_stable_digest(
                {"component_content": repeat_id}
            ),
        }
        for repeat_id in formal_randomization_repeat_ids()
    )
    return RandomizationAggregateProvenance(
        package_path=(tmp_path / "aggregate.zip").resolve(),
        package_sha256=AGGREGATE_PACKAGE_SHA256,
        payload_path="aggregate/payload.json",
        payload_sha256=build_stable_digest({"payload": 1}),
        manifest_path="aggregate/manifest.json",
        manifest_sha256=build_stable_digest({"manifest": 1}),
        payload={
            "paper_run_name": PAPER_RUN_NAME,
            "target_fpr": TARGET_FPR,
        },
        manifest={"artifact_id": "aggregate"},
        randomization_repeat_components=repeat_components,
        invariant_packages=(),
        common_code_version=CODE_VERSION,
        randomization_aggregate_digest=AGGREGATE_DIGEST,
    )


def _install_memory_boundaries(
    monkeypatch: pytest.MonkeyPatch,
    workspace: _MemoryAggregateWorkspace,
    captured: dict[str, Any],
) -> None:
    """替换已单测的两侧边界，只验证桥接数据流."""

    monkeypatch.setattr(
        bridge_module,
        "open_randomization_aggregate_record_workspace",
        lambda source: workspace,
    )

    def validate_runtime(row: Mapping[str, Any]) -> dict[str, Any]:
        config = row["metadata"]["scientific_unit_config"]
        expected_run_id = "runtime_" + build_stable_digest(config)[:16]
        if row.get("run_id") != expected_run_id:
            raise ValueError("run_id 与配置摘要不一致")
        return {"runtime_record_ready": True}

    monkeypatch.setattr(
        bridge_module,
        "validate_semantic_watermark_runtime_result_provenance",
        validate_runtime,
    )

    def rebuild_base_latent(
        *,
        shape,
        generation_seed_random,
        model_id,
        model_revision,
        device,
        dtype,
    ):
        assert tuple(shape) == (1, 16, 64, 64)
        assert model_id == MODEL_ID
        assert model_revision == MODEL_REVISION
        assert device == "cpu"
        assert str(dtype) == "torch.float16"
        return None, _base_latent_identity(generation_seed_random)

    monkeypatch.setattr(
        bridge_module,
        "build_canonical_sd35_base_latent",
        rebuild_base_latent,
    )

    def recompute(sources, **kwargs):
        materialized = tuple(sources)
        captured["sources"] = materialized
        captured["kwargs"] = kwargs
        report_payload = {
            "prompt_protocol_digest": build_stable_digest(
                kwargs["prompt_rows"]
            ),
            "threshold_records_digest": build_stable_digest(
                [source.method_id for source in materialized]
            ),
            "method_repeat_fixed_fpr_recomputation_ready": True,
            "supports_paper_claim": False,
        }
        report_payload["method_repeat_fixed_fpr_report_digest"] = (
            build_stable_digest(report_payload)
        )
        return {
            "threshold_records": tuple(
                {
                    "method_id": source.method_id,
                    "randomization_repeat_id": source.randomization_repeat_id,
                    "supports_paper_claim": False,
                }
                for source in materialized
            ),
            "fairness_records": (),
            "report": report_payload,
        }

    monkeypatch.setattr(
        bridge_module,
        "recompute_exact_method_repeat_fixed_fpr",
        recompute,
    )


@pytest.fixture(scope="module")
def bridge_template(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[RandomizationAggregateProvenance, _MemoryAggregateWorkspace]:
    """只构造一次完整内存来源模板, 避免 quick 测试重复读取大清单."""

    provenance = _provenance(tmp_path_factory.mktemp("threshold_bridge"))
    return provenance, _MemoryAggregateWorkspace(provenance)


@pytest.fixture
def bridge_context(
    bridge_template: tuple[
        RandomizationAggregateProvenance,
        _MemoryAggregateWorkspace,
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[
    RandomizationAggregateProvenance,
    _MemoryAggregateWorkspace,
    dict[str, Any],
]:
    provenance, workspace_template = bridge_template
    workspace = copy.deepcopy(workspace_template)
    captured: dict[str, Any] = {}
    _install_memory_boundaries(monkeypatch, workspace, captured)
    return provenance, workspace, captured


def test_bridge_builds_exact_45_sources_and_digest_locked_reconstruction(
    bridge_context,
) -> None:
    provenance, _workspace, captured = bridge_context

    result = recompute_randomization_method_repeat_fixed_fpr(provenance)

    assert len(captured["sources"]) == 45
    assert len(result["threshold_records"]) == 45
    assert captured["kwargs"]["expected_model_id"] == MODEL_ID
    assert captured["kwargs"]["expected_model_revision"] == MODEL_REVISION
    assert captured["kwargs"]["main_rescue_margin_low"] == RESCUE_MARGIN_LOW
    assert captured["kwargs"]["expected_base_seed"] == BASE_SEED
    transfer_source = next(
        source
        for source in captured["sources"]
        if source.method_id == "tree_ring"
    )
    assert set(transfer_source.declared_threshold_protocol) == {
        "calibrated_detection_threshold",
        "threshold_digest",
    }
    t2smark_source = next(
        source
        for source in captured["sources"]
        if source.method_id == "t2smark"
    )
    assert set(t2smark_source.declared_threshold_protocol) == {
        "calibrated_detection_threshold",
        "threshold_digest",
    }
    reconstruction = result["reconstruction_report"]
    assert tuple(reconstruction["runtime_source_record_map"]) == (
        formal_randomization_repeat_ids()
    )
    assert reconstruction["exact_runtime_source_count"] == 9
    assert reconstruction["runtime_randomization_identity_count"] == 630
    assert len(
        reconstruction["runtime_randomization_identity_digest_random"]
    ) == 64
    assert reconstruction["formal_watermark_key_plan_digest"] == (
        formal_watermark_key_plan_record()[
            "formal_watermark_key_plan_digest"
        ]
    )
    for repeat_id, runtime_source in reconstruction[
        "runtime_source_record_map"
    ].items():
        repeat = resolve_formal_randomization_repeat(repeat_id)
        assert runtime_source["watermark_key_seed_random"] == (
            _watermark_key_seed_random(repeat)
        )
        assert runtime_source[
            "watermark_key_material_digest_random"
        ] == _watermark_key_material_digest_random(repeat)
        assert len(
            runtime_source[
                "runtime_randomization_identity_digest_random"
            ]
        ) == 64
    assert reconstruction["canonical_base_latent_identity_count"] == 210
    assert tuple(reconstruction["prompt_source_record_map"]) == (
        formal_randomization_repeat_ids()
    )
    assert len(reconstruction["prompt_source_contract_digest"]) == 64
    assert len(reconstruction["prompt_source_records_digest"]) == 64
    assert len(reconstruction["prompt_file_sha256"]) == 64
    assert len(reconstruction["prompt_source_registry_digest"]) == 64
    assert len(reconstruction["selection_manifest_sha256"]) == 64
    assert len(reconstruction["selection_manifest_digest"]) == 64
    assert reconstruction["exact_method_repeat_fixed_fpr_ready"] is True
    assert reconstruction["supports_paper_claim"] is False
    digest = reconstruction["reconstruction_report_digest"]
    assert digest == build_stable_digest(
        {
            key: value
            for key, value in reconstruction.items()
            if key != "reconstruction_report_digest"
        }
    )


def test_bridge_composes_with_real_exact_threshold_recomputation(
    bridge_context,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证桥接产生的完整 schema 可直接进入真实纯阈值算子."""

    provenance, _workspace, _captured = bridge_context
    monkeypatch.setattr(
        bridge_module,
        "recompute_exact_method_repeat_fixed_fpr",
        recompute_exact_method_repeat_fixed_fpr,
    )

    result = recompute_randomization_method_repeat_fixed_fpr(provenance)

    assert len(result["threshold_records"]) == 45
    assert result["report"][
        "method_repeat_fixed_fpr_recomputation_ready"
    ] is True
    assert result["reconstruction_report"][
        "exact_method_repeat_fixed_fpr_ready"
    ] is True
    assert result["reconstruction_report"]["supports_paper_claim"] is False


def test_bridge_rejects_cross_repeat_prompt_byte_and_field_drift(
    bridge_context,
) -> None:
    provenance, workspace, _captured = bridge_context
    workspace.replace_runtime_prompt_text(
        formal_randomization_repeat_ids()[1],
        10,
        "A governed city scene with a distinct byte sequence",
    )

    with pytest.raises(
        RandomizationMethodRepeatThresholdError,
        match="受治理来源字节",
    ):
        recompute_randomization_method_repeat_fixed_fpr(provenance)


def test_bridge_rejects_nine_runtime_records_with_consistent_prompt_drift(
    bridge_context,
) -> None:
    """9份 runtime 自洽替换同一文本仍必须服从包内 Prompt 来源."""

    provenance, workspace, _captured = bridge_context
    for repeat_id in formal_randomization_repeat_ids():
        workspace.replace_runtime_prompt_text(
            repeat_id,
            0,
            "A completely unregistered adversarial prompt",
        )

    with pytest.raises(
        RandomizationMethodRepeatThresholdError,
        match="受治理来源字节",
    ):
        recompute_randomization_method_repeat_fixed_fpr(provenance)


def test_bridge_rejects_nine_runtime_records_with_method_setting_drift(
    bridge_context,
) -> None:
    """9份 runtime 同步漂移非模型方法参数也不得通过完整配置锁."""

    provenance, workspace, _captured = bridge_context
    _replace_all_runtime_config_field(workspace, "tail_fraction", 0.125)

    with pytest.raises(
        RandomizationMethodRepeatThresholdError,
        match="冻结方法配置",
    ):
        recompute_randomization_method_repeat_fixed_fpr(provenance)


@pytest.mark.parametrize(
    ("field_name", "drift_value"),
    (
        ("injection_step_indices", [1, 2]),
        ("candidate_count", 999),
        ("null_rank", 1),
    ),
)
def test_bridge_rejects_nine_runtime_records_with_alias_drift(
    bridge_context,
    field_name: str,
    drift_value: Any,
) -> None:
    """运行时实际消费的三个科学别名必须匹配唯一正式配置."""

    provenance, workspace, _captured = bridge_context
    _replace_all_runtime_config_field(
        workspace,
        field_name,
        drift_value,
    )

    with pytest.raises(
        RandomizationMethodRepeatThresholdError,
        match="完整方法或检测配置",
    ):
        recompute_randomization_method_repeat_fixed_fpr(provenance)


def test_bridge_rejects_nine_runtime_records_with_rescue_margin_drift(
    bridge_context,
) -> None:
    """rescue margin 必须等于冻结正式协议值, 不能只要求9份相同."""

    provenance, workspace, _captured = bridge_context
    _replace_all_runtime_config_field(workspace, "rescue_margin_low", -0.1)

    with pytest.raises(
        RandomizationMethodRepeatThresholdError,
        match="完整方法或检测配置",
    ):
        recompute_randomization_method_repeat_fixed_fpr(provenance)


def test_bridge_rejects_nine_runtime_records_with_extra_proxy_field(
    bridge_context,
) -> None:
    """9份 runtime 同步加入未登记机制字段也不得通过精确 schema."""

    provenance, workspace, _captured = bridge_context
    _replace_all_runtime_config_field(workspace, "proxy_mode", True)

    with pytest.raises(
        RandomizationMethodRepeatThresholdError,
        match="字段集合",
    ):
        recompute_randomization_method_repeat_fixed_fpr(provenance)


def test_bridge_rejects_same_count_calibration_test_split_swap(
    bridge_context,
) -> None:
    provenance, workspace, _captured = bridge_context
    repeat_id = formal_randomization_repeat_ids()[0]
    key = (
        repeat_id,
        "image_only_dataset_runtime",
        "semantic_watermark_runtime_record",
    )
    rows = [dict(row) for row in workspace._rows[key]]
    calibration_index = next(
        index
        for index, row in enumerate(rows)
        if row["metadata"]["scientific_unit_config"]["split"]
        == "calibration"
    )
    test_index = next(
        index
        for index, row in enumerate(rows)
        if row["metadata"]["scientific_unit_config"]["split"] == "test"
    )
    for index, replacement_split in (
        (calibration_index, "test"),
        (test_index, "calibration"),
    ):
        metadata = dict(rows[index]["metadata"])
        config = dict(metadata["scientific_unit_config"])
        config["split"] = replacement_split
        config["standard_attack_profiles"] = (
            ["full_main"] if replacement_split == "test" else []
        )
        config["diffusion_attacks_enabled"] = (
            replacement_split == "test"
        )
        metadata["scientific_unit_config"] = config
        rows[index]["metadata"] = metadata
        rows[index]["run_id"] = (
            "runtime_" + build_stable_digest(config)[:16]
        )
    workspace._rows[key] = tuple(rows)

    with pytest.raises(
        RandomizationMethodRepeatThresholdError,
        match="规范 split",
    ):
        recompute_randomization_method_repeat_fixed_fpr(provenance)


def test_bridge_rejects_runtime_generation_seed_formula_drift(
    bridge_context,
) -> None:
    provenance, workspace, _captured = bridge_context
    repeat_id = formal_randomization_repeat_ids()[0]
    key = (
        repeat_id,
        "image_only_dataset_runtime",
        "semantic_watermark_runtime_record",
    )
    rows = [dict(row) for row in workspace._rows[key]]
    metadata = dict(rows[0]["metadata"])
    config = dict(metadata["scientific_unit_config"])
    config["seed"] = int(config["seed"]) + 1
    metadata["scientific_unit_config"] = config
    rows[0]["metadata"] = metadata
    rows[0]["run_id"] = "runtime_" + build_stable_digest(config)[:16]
    workspace._rows[key] = tuple(rows)

    with pytest.raises(
        RandomizationMethodRepeatThresholdError,
        match="冻结生成 seed",
    ):
        recompute_randomization_method_repeat_fixed_fpr(provenance)


def test_bridge_rejects_runtime_key_material_not_derived_from_key_seed(
    bridge_context,
) -> None:
    """runtime 即使重算 run id, 也不能提交任意 key material 摘要."""

    provenance, workspace, _captured = bridge_context
    repeat_id = formal_randomization_repeat_ids()[0]
    key = (
        repeat_id,
        "image_only_dataset_runtime",
        "semantic_watermark_runtime_record",
    )
    rows = [dict(row) for row in workspace._rows[key]]
    metadata = dict(rows[0]["metadata"])
    config = dict(metadata["scientific_unit_config"])
    config["key_material_digest_random"] = "f" * 64
    metadata["scientific_unit_config"] = config
    rows[0]["metadata"] = metadata
    rows[0]["run_id"] = "runtime_" + build_stable_digest(config)[:16]
    workspace._rows[key] = tuple(rows)

    with pytest.raises(
        RandomizationMethodRepeatThresholdError,
        match="key material 摘要",
    ):
        recompute_randomization_method_repeat_fixed_fpr(provenance)


def test_bridge_rejects_consistent_single_repeat_runtime_key_forgery(
    bridge_context,
) -> None:
    """一个重复内 runtime 与五方法同步伪造 key 仍不能破坏3x3交叉."""

    provenance, workspace, _captured = bridge_context
    repeat_id = formal_randomization_repeat_ids()[0]
    forged_key_seed = 7_700_001
    _replace_runtime_key_identity(
        workspace,
        repeat_id,
        forged_key_seed,
    )
    _replace_observation_key_identity(
        workspace,
        repeat_id,
        forged_key_seed,
    )

    with pytest.raises(
        RandomizationMethodRepeatThresholdError,
        match="跨 generation seed",
    ):
        recompute_randomization_method_repeat_fixed_fpr(provenance)


def test_bridge_rejects_all_45_sources_with_consistently_forged_key(
    bridge_context,
) -> None:
    """45组 observation 全部自洽伪造时仍必须服从 runtime 身份锚点."""

    provenance, workspace, _captured = bridge_context
    for repeat in formal_randomization_repeats():
        _replace_observation_key_identity(
            workspace,
            repeat.randomization_repeat_id,
            8_800_000 + repeat.watermark_key_index,
        )

    with pytest.raises(
        RandomizationMethodRepeatThresholdError,
        match="随机身份未逐 Prompt 对齐 runtime",
    ):
        recompute_randomization_method_repeat_fixed_fpr(provenance)


def test_bridge_rejects_all_runtime_and_observations_with_alternate_key_plan(
    bridge_context,
) -> None:
    """9个 runtime 与45组 observation 同步换 key 仍须匹配预注册计划."""

    provenance, workspace, _captured = bridge_context
    for repeat in formal_randomization_repeats():
        forged_key_seed = 7_700_000 + repeat.watermark_key_index
        _replace_runtime_key_identity(
            workspace,
            repeat.randomization_repeat_id,
            forged_key_seed,
        )
        _replace_observation_key_identity(
            workspace,
            repeat.randomization_repeat_id,
            forged_key_seed,
        )

    with pytest.raises(
        RandomizationMethodRepeatThresholdError,
        match="预注册 key plan",
    ):
        recompute_randomization_method_repeat_fixed_fpr(provenance)


@pytest.mark.parametrize("drift_field", ("threshold", "threshold_digest"))
def test_bridge_rejects_t2smark_multiple_threshold_identity(
    bridge_context,
    drift_field: str,
) -> None:
    provenance, workspace, _captured = bridge_context
    repeat_id = formal_randomization_repeat_ids()[0]
    role = "t2smark_formal_import_candidate_record"
    key = (repeat_id, "official_reference_t2smark", role)
    rows = [dict(row) for row in workspace._rows[key]]
    if drift_field == "threshold":
        rows[1]["calibrated_detection_threshold"] = 0.75
    else:
        rows[1]["threshold_digest"] = "f" * 64
    workspace._rows[key] = tuple(rows)

    with pytest.raises(
        RandomizationMethodRepeatThresholdError,
        match="多阈值或阈值摘要漂移",
    ):
        recompute_randomization_method_repeat_fixed_fpr(provenance)


def test_bridge_rejects_t2smark_observation_evidence_digest_drift(
    bridge_context,
) -> None:
    provenance, workspace, _captured = bridge_context
    repeat_id = formal_randomization_repeat_ids()[0]
    key = (
        repeat_id,
        "official_reference_t2smark",
        "t2smark_formal_import_candidate_record",
    )
    rows = [dict(row) for row in workspace._rows[key]]
    rows[0]["fixed_fpr_observation_evidence_digest"] = "f" * 64
    workspace._rows[key] = tuple(rows)

    with pytest.raises(
        RandomizationMethodRepeatThresholdError,
        match="candidate 阈值协议无效",
    ):
        recompute_randomization_method_repeat_fixed_fpr(provenance)


def test_bridge_rejects_method_faithful_observation_sha_drift(
    bridge_context,
) -> None:
    provenance, workspace, _captured = bridge_context
    repeat_id = formal_randomization_repeat_ids()[0]
    key = (
        repeat_id,
        "method_faithful_tree_ring",
        "tree_ring_baseline_transfer_manifest",
    )
    declaration = dict(workspace._objects[key])
    declaration["baseline_observations_sha256"] = "f" * 64
    workspace._objects[key] = declaration

    with pytest.raises(
        RandomizationMethodRepeatThresholdError,
        match="tree_ring transfer",
    ):
        recompute_randomization_method_repeat_fixed_fpr(provenance)


def test_bridge_rejects_forged_main_canonical_latent_digest(
    bridge_context,
) -> None:
    provenance, workspace, _captured = bridge_context
    repeat_id = formal_randomization_repeat_ids()[0]
    key = (
        repeat_id,
        "image_only_dataset_runtime",
        "semantic_watermark_detection_observation",
    )
    rows = [dict(row) for row in workspace._rows[key]]
    rows[0]["base_latent_content_digest_random"] = "f" * 64
    workspace._rows[key] = tuple(rows)

    with pytest.raises(
        RandomizationMethodRepeatThresholdError,
        match="规范高斯字节流",
    ):
        recompute_randomization_method_repeat_fixed_fpr(provenance)


def test_bridge_rejects_baseline_latent_digest_drift_from_verified_main(
    bridge_context,
) -> None:
    provenance, workspace, _captured = bridge_context
    repeat_id = formal_randomization_repeat_ids()[0]
    key = (
        repeat_id,
        "method_faithful_gaussian_shading",
        "gaussian_shading_baseline_observation",
    )
    rows = [dict(row) for row in workspace._rows[key]]
    rows[0]["base_latent_identity_digest_random"] = "f" * 64
    workspace._rows[key] = tuple(rows)

    with pytest.raises(
        RandomizationMethodRepeatThresholdError,
        match="已重建主方法身份",
    ):
        recompute_randomization_method_repeat_fixed_fpr(provenance)


def test_bridge_rejects_runtime_member_summary_drift(
    bridge_context,
) -> None:
    provenance, workspace, _captured = bridge_context
    repeat_id = formal_randomization_repeat_ids()[0]
    key = (
        repeat_id,
        "image_only_dataset_runtime",
        "semantic_watermark_runtime_record",
    )
    workspace._sources[key] = replace(
        workspace._sources[key],
        record_sha256="f" * 64,
    )

    with pytest.raises(
        RandomizationMethodRepeatThresholdError,
        match="成员摘要.*漂移",
    ):
        recompute_randomization_method_repeat_fixed_fpr(provenance)


def test_bridge_rejects_incomplete_runtime_summary_map(
    bridge_context,
) -> None:
    provenance, workspace, _captured = bridge_context
    workspace._runtime_property_sources.pop()

    with pytest.raises(
        RandomizationMethodRepeatThresholdError,
        match="精确覆盖9个",
    ):
        recompute_randomization_method_repeat_fixed_fpr(provenance)


def test_bridge_rejects_tampered_runtime_run_id(
    bridge_context,
) -> None:
    provenance, workspace, _captured = bridge_context
    repeat_id = formal_randomization_repeat_ids()[0]
    key = (
        repeat_id,
        "image_only_dataset_runtime",
        "semantic_watermark_runtime_record",
    )
    rows = [dict(row) for row in workspace._rows[key]]
    rows[0]["run_id"] = "tampered_runtime"
    workspace._rows[key] = tuple(rows)

    with pytest.raises(
        RandomizationMethodRepeatThresholdError,
        match="run_id、配置或科学来源",
    ):
        recompute_randomization_method_repeat_fixed_fpr(provenance)


def test_bridge_public_entry_has_no_workspace_path_override(
    bridge_context,
    tmp_path: Path,
) -> None:
    provenance, _workspace, _captured = bridge_context
    assert tuple(
        inspect.signature(
            recompute_randomization_method_repeat_fixed_fpr
        ).parameters
    ) == ("source",)

    with pytest.raises(TypeError):
        recompute_randomization_method_repeat_fixed_fpr(
            provenance,
            workspace_path=tmp_path,
        )
