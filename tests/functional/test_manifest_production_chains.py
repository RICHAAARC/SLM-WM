"""验证生产 writer、manifest 配置正文和正式打包器使用同一契约。"""

from __future__ import annotations

from dataclasses import dataclass
import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from experiments.ablations import runtime_rerun
from experiments.artifacts import dataset_level_quality_outputs
from experiments.artifacts.paired_quality_outputs import (
    FORMAL_CLIP_FEATURE_BACKEND,
    FORMAL_CLIP_FEATURE_DIMENSION,
    PAIRED_QUALITY_METRIC_RECORD_SCHEMA,
)
from experiments.artifacts.artifact_manifest import build_artifact_manifest
from experiments.artifacts.manifest_schema import manifest_config_digest_ready
from experiments.protocol.attacks import (
    attack_config_digest,
    default_attack_configs,
    formal_attack_seed_protocol_record,
    formal_attack_seed_random,
)
from experiments.protocol.dataset_quality import (
    FORMAL_DATASET_QUALITY_METRIC_NAMES,
    build_dataset_quality_image_records,
)
from experiments.protocol.attack_conditioned_quality import (
    ATTACK_CONDITIONED_IMAGE_PAIR_ROLE,
    ATTACK_CONDITIONED_QUALITY_RECORD_SCHEMA,
    attack_quality_dataset_image_records,
    load_attack_conditioned_quality_estimand,
)
from experiments.protocol.detection_key_identity import (
    REGISTERED_WRONG_KEY_ROLE,
    REGISTERED_WATERMARK_KEY_ROLE,
)
from experiments.protocol.independent_semantic_quality import (
    INDEPENDENT_SEMANTIC_FEATURE_BACKEND,
    INDEPENDENT_SEMANTIC_FEATURE_DIMENSION,
    load_independent_semantic_quality_evaluator,
)
from experiments.protocol.prompts import build_prompt_records, read_prompt_file
from experiments.protocol.formal_randomization import (
    formal_generation_seed,
    formal_randomization_sample_reference,
    formal_randomization_protocol_record,
    formal_watermark_key_material_from_seed,
    formal_watermark_key_plan_record,
    resolve_formal_randomization_repeat,
    validate_formal_prompt_randomization_identity,
)
from experiments.protocol.splits import apply_split_assignments
from experiments.protocol.content_routing_reference_quantile import (
    ContentRoutingReferenceScalars,
)
from experiments.runners.semantic_watermark_runtime import (
    SemanticWatermarkRuntimeConfig,
    build_semantic_watermark_run_id,
    semantic_watermark_runtime_config_payload,
)
from experiments.runtime import repository_environment
from main.core.digest import build_stable_digest
from paper_experiments.analysis.result_closure_gate import (
    _ablation_raw_record_rebuild_ready,
)
from tests.helpers.formal_execution_lock import build_test_formal_execution_lock
from tests.helpers.formal_detection_record import bind_formal_detection_record
from tests.helpers.scientific_execution_binding import (
    write_test_scientific_execution_binding,
)
from tests.helpers.scientific_unit_provenance import (
    build_test_scientific_unit_provenance,
)


PAPER_RUN_NAME = "probe_paper"
TARGET_FPR = 0.1
FORMAL_EXECUTION_LOCK = build_test_formal_execution_lock()


@dataclass(frozen=True)
class _FakeRuntimeResult:
    """保存语义运行结果正文和对应检测记录路径。"""

    payload: dict[str, Any]
    detection_record_path: str

    def to_dict(self) -> dict[str, Any]:
        """返回生产 provenance validator 可复算的结果正文。"""

        return dict(self.payload)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """写出测试链路使用的稳定 JSONL。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _minimal_paper_run(prompt_file: str, prompt_count: int) -> SimpleNamespace:
    """构造只包含生产 writer 与打包器实际读取字段的论文运行配置。"""

    split_records = apply_split_assignments(
        build_prompt_records(
            PAPER_RUN_NAME,
            tuple(f"测试 Prompt {index}" for index in range(prompt_count)),
        )
    )
    test_count = sum(record.split == "test" for record in split_records)
    repeat = resolve_formal_randomization_repeat("seed_00_key_00")
    return SimpleNamespace(
        run_name=PAPER_RUN_NAME,
        target_fpr=TARGET_FPR,
        prompt_set=PAPER_RUN_NAME,
        prompt_file=prompt_file,
        prompt_count=prompt_count,
        minimum_clean_negative_count=test_count,
        dataset_level_quality_minimum_count=prompt_count,
        randomization_repeat_id=repeat.randomization_repeat_id,
        generation_seed_index=repeat.generation_seed_index,
        generation_seed_offset=repeat.generation_seed_offset,
        watermark_key_index=repeat.watermark_key_index,
        formal_randomization_protocol_digest=(
            formal_randomization_protocol_record()[
                "formal_randomization_protocol_digest"
            ]
        ),
    )


def _runtime_result_payload(
    config: SemanticWatermarkRuntimeConfig,
    *,
    prompt_index: int,
) -> dict[str, Any]:
    """构造配置、run id 和科学来源完全一致的最小运行结果。"""

    unit_config = semantic_watermark_runtime_config_payload(config)
    run_id = build_semantic_watermark_run_id(config)
    randomization_identity = validate_formal_prompt_randomization_identity(
        base_generation_seed_random=SemanticWatermarkRuntimeConfig().seed,
        prompt_index=prompt_index,
        randomization_repeat_id=config.randomization_repeat_id,
        generation_seed_index=config.generation_seed_index,
        generation_seed_offset=config.generation_seed_offset,
        watermark_key_index=config.watermark_key_index,
        generation_seed_random=config.seed,
        watermark_key_seed_random=config.watermark_key_seed_random,
        key_material=config.key_material,
        formal_randomization_protocol_digest=(
            config.formal_randomization_protocol_digest
        ),
    )
    return {
        "run_id": run_id,
        "run_decision": "pass",
        "metadata": {
            "scientific_unit_config_digest": build_stable_digest(
                unit_config
            ),
            "formal_randomization_reference": (
                formal_randomization_sample_reference(
                    randomization_identity
                )
            ),
            "scientific_unit_provenance": build_test_scientific_unit_provenance(
                run_id,
                build_stable_digest(unit_config),
                seed=config.seed,
                formal_execution_lock=FORMAL_EXECUTION_LOCK,
            ),
            "paired_quality": {"ssim": 0.95},
        },
    }


def _detection_record(
    *,
    prompt_id: str,
    split: str,
    sample_role: str,
    content_score: float,
    attention_geometry_enabled: bool,
    image_alignment_enabled: bool,
    attack: Any | None = None,
    generation_seed_random: int | None = None,
    detector_guided_attack_threshold_digest: str | None = None,
) -> dict[str, Any]:
    """构造能够进入真实 fixed-FPR 校准和攻击覆盖检查的检测原子。"""

    record: dict[str, Any] = {
        "prompt_id": prompt_id,
        "split": split,
        "sample_role": sample_role,
        "detection_key_role": (
            REGISTERED_WRONG_KEY_ROLE
            if sample_role == "wrong_key_negative"
            else REGISTERED_WATERMARK_KEY_ROLE
        ),
        "content_score": content_score,
        "aligned_content_score": (
            content_score if image_alignment_enabled else None
        ),
        "attention_geometry_score": (
            0.0 if image_alignment_enabled else None
        ),
        "raw_attention_geometry_score": (
            0.0 if attention_geometry_enabled else None
        ),
        "registration_confidence": (
            0.0 if image_alignment_enabled else None
        ),
        "attention_sync_score": (
            0.0 if image_alignment_enabled else None
        ),
        "alignment": (
            {"registration_geometry_reliable": False}
            if image_alignment_enabled
            else None
        ),
        "metadata": {
            "attention_geometry_enabled": attention_geometry_enabled,
            "image_alignment_enabled": image_alignment_enabled,
        },
    }
    if attack is not None:
        if generation_seed_random is None:
            raise ValueError("攻击检测记录必须提供实际生成 seed")
        record.update(
            {
                "attack_id": attack.attack_id,
                "attack_family": attack.attack_family,
                "attack_name": attack.attack_name,
                "resource_profile": attack.resource_profile,
                "attack_config_digest": attack_config_digest(attack),
                "attack_parameters": attack.attack_parameters,
                "attack_performed": True,
                "generation_seed_random": generation_seed_random,
                "attack_seed_random": formal_attack_seed_random(
                    generation_seed_random,
                    attack.attack_id,
                ),
                "formal_attack_seed_protocol_digest": (
                    formal_attack_seed_protocol_record()[
                        "formal_attack_seed_protocol_digest"
                    ]
                ),
            }
        )
        if attack.attack_name == "adversarial_removal_attack":
            record["detector_guided_attack_threshold_digest"] = (
                detector_guided_attack_threshold_digest
            )
    return bind_formal_detection_record(record)


@pytest.mark.quick
def test_artifact_manifest_persists_one_immutable_config_snapshot() -> None:
    """manifest 正文与摘要必须来自同一配置快照。"""

    source_config = {"nested": {"value": 3}}
    manifest = build_artifact_manifest(
        artifact_id="manifest_contract",
        artifact_type="local_manifest",
        input_paths=(),
        output_paths=("outputs/manifest.json",),
        config=source_config,
        code_version="a" * 40,
        rebuild_command="python -m tests.manifest_contract",
    ).to_dict()
    source_config["nested"]["value"] = 9

    assert manifest["config"] == {"nested": {"value": 3}}
    assert manifest["config_digest"] == build_stable_digest(manifest["config"])
    assert manifest_config_digest_ready(manifest) is True
    manifest["config"]["nested"]["value"] = 4
    assert manifest_config_digest_ready(manifest) is False


@pytest.mark.quick
def test_runtime_rerun_writer_package_and_closure_share_manifest_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """真实消融 writer 的 manifest 应直接通过 package 与 raw closure。"""

    prompt_relative = "configs/manifest_chain_prompts.txt"
    prompt_path = tmp_path / prompt_relative
    prompt_path.parent.mkdir(parents=True)
    prompt_path.write_text(
        "".join(f"测试 Prompt {index}\n" for index in range(6)),
        encoding="utf-8",
    )
    paper_run = _minimal_paper_run(prompt_relative, 6)
    prompt_records = apply_split_assignments(
        build_prompt_records(
            paper_run.prompt_set,
            read_prompt_file(prompt_path),
        )
    )
    repeat = resolve_formal_randomization_repeat(
        paper_run.randomization_repeat_id
    )
    key_record = next(
        record
        for record in formal_watermark_key_plan_record()[
            "watermark_key_records"
        ]
        if record["watermark_key_index"] == repeat.watermark_key_index
    )
    key_seed = int(key_record["watermark_key_seed_random"])
    base_generation_seed = SemanticWatermarkRuntimeConfig().seed
    base_configs = tuple(
        SemanticWatermarkRuntimeConfig(
            prompt=record.prompt_text,
            prompt_id=record.prompt_id,
            split=record.split,
            seed=formal_generation_seed(
                base_generation_seed,
                record.prompt_index,
                repeat,
            ),
            key_material=formal_watermark_key_material_from_seed(
                key_seed,
                repeat,
            ),
            randomization_repeat_id=repeat.randomization_repeat_id,
            generation_seed_index=repeat.generation_seed_index,
            generation_seed_offset=repeat.generation_seed_offset,
            watermark_key_index=repeat.watermark_key_index,
            watermark_key_seed_random=key_seed,
            formal_randomization_protocol_digest=(
                paper_run.formal_randomization_protocol_digest
            ),
        )
        for record in prompt_records
    )
    prompt_index_by_id = {
        record.prompt_id: record.prompt_index
        for record in prompt_records
    }

    monkeypatch.setattr(
        repository_environment,
        "require_published_formal_execution_lock",
        lambda _root: dict(FORMAL_EXECUTION_LOCK),
    )
    monkeypatch.setattr(runtime_rerun, "build_paper_run_config", lambda _root: paper_run)
    monkeypatch.setattr(runtime_rerun, "restore_role_checkpoints", lambda **_kwargs: None)
    monkeypatch.setattr(runtime_rerun, "clear_progress_checkpoints", lambda **_kwargs: None)

    def load_fake_result(
        config: SemanticWatermarkRuntimeConfig,
        *,
        root: Path,
    ) -> _FakeRuntimeResult:
        """按真实配置写出该完成单元的检测原子并返回结果。"""

        ablation_id = Path(config.output_dir).name
        positive_score = 1.0 if ablation_id == "complete_method" else 0.0
        rows = [
            _detection_record(
                prompt_id=config.prompt_id,
                split=config.split,
                sample_role="clean_negative",
                content_score=0.0,
                attention_geometry_enabled=config.attention_geometry_enabled,
                image_alignment_enabled=config.image_alignment_enabled,
            ),
            _detection_record(
                prompt_id=config.prompt_id,
                split=config.split,
                sample_role="positive_source",
                content_score=positive_score,
                attention_geometry_enabled=config.attention_geometry_enabled,
                image_alignment_enabled=config.image_alignment_enabled,
            ),
            _detection_record(
                prompt_id=config.prompt_id,
                split=config.split,
                sample_role="wrong_key_negative",
                content_score=0.0,
                attention_geometry_enabled=config.attention_geometry_enabled,
                image_alignment_enabled=config.image_alignment_enabled,
            ),
        ]
        if config.split == "test":
            rows.extend(
                    _detection_record(
                        prompt_id=config.prompt_id,
                        split=config.split,
                        sample_role=sample_role,
                    content_score=(
                        positive_score if sample_role == "positive_source" else 0.0
                    ),
                    attention_geometry_enabled=(
                        config.attention_geometry_enabled
                    ),
                    image_alignment_enabled=config.image_alignment_enabled,
                    attack=attack,
                    generation_seed_random=config.seed,
                    detector_guided_attack_threshold_digest=(
                        None
                        if config.detector_guided_attack_threshold_protocol
                        is None
                        else str(
                            config.detector_guided_attack_threshold_protocol[
                                "threshold_digest"
                            ]
                        )
                    ),
                )
                for attack in default_attack_configs()
                if attack.enabled
                and attack.resource_profile in {"full_main", "full_extra"}
                for sample_role in ("clean_negative", "positive_source")
            )
        payload = _runtime_result_payload(
            config,
            prompt_index=prompt_index_by_id[config.prompt_id],
        )
        relative_path = (
            Path("outputs/manifest_chain_runtime")
            / payload["run_id"]
            / "detection_records.jsonl"
        )
        _write_jsonl(root / relative_path, rows)
        return _FakeRuntimeResult(
            payload=payload,
            detection_record_path=relative_path.as_posix(),
        )

    monkeypatch.setattr(
        runtime_rerun,
        "load_completed_semantic_watermark_runtime_result",
        load_fake_result,
    )
    summary = runtime_rerun.run_runtime_rerun_ablations(
        base_configs,
        target_fpr=TARGET_FPR,
        paper_run_name=PAPER_RUN_NAME,
        root=tmp_path,
        content_routing_references=ContentRoutingReferenceScalars(
            reference_gradient=1.0,
            reference_response=0.5,
            reference_sensitivity=0.25,
        ),
    )
    output_dir = (
        tmp_path / "outputs/formal_mechanism_ablation" / PAPER_RUN_NAME
    )
    manifest = json.loads(
        (output_dir / "manifest.local.json").read_text(encoding="utf-8")
    )
    runtime_records = tuple(
        json.loads(line)
        for line in (output_dir / "runtime_rerun_records.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    )
    with (output_dir / "mechanism_necessity_statistics.csv").open(
        encoding="utf-8",
        newline="",
    ) as stream:
        necessity_rows = tuple(dict(row) for row in csv.DictReader(stream))
    necessity_summary = json.loads(
        (output_dir / "mechanism_necessity_summary.json").read_text(
            encoding="utf-8"
        )
    )
    closure_bundle = SimpleNamespace(
        ablation_summary=summary,
        ablation_manifest=manifest,
        ablation_runtime_records=runtime_records,
        ablation_necessity_rows=necessity_rows,
        ablation_necessity_summary=necessity_summary,
        expected_prompt_count=6,
        expected_test_count=paper_run.minimum_clean_negative_count,
        expected_calibration_prompt_id_digest=build_stable_digest(
            sorted(
                record.prompt_id
                for record in prompt_records
                if record.split == "calibration"
            )
        ),
        expected_test_prompt_id_digest=build_stable_digest(
            sorted(
                record.prompt_id
                for record in prompt_records
                if record.split == "test"
            )
        ),
    )

    assert manifest["config_digest"] == build_stable_digest(manifest["config"])
    assert manifest["config"]["record_digest"] == build_stable_digest(
        runtime_records
    )
    assert _ablation_raw_record_rebuild_ready(closure_bundle) is True

    write_test_scientific_execution_binding(
        repository_root=tmp_path,
        artifact_dir=output_dir,
        artifact_role="runtime_rerun_ablation",
        paper_run_name=PAPER_RUN_NAME,
        profile_id="sd35_method_runtime_gpu",
        summary_file_name="ablation_component_summary.json",
        manifest_file_name="manifest.local.json",
        formal_execution_lock=FORMAL_EXECUTION_LOCK,
        execution_route="semantic_watermark_ablation_session",
    )
    archive_path = runtime_rerun.package_runtime_rerun_ablations(
        PAPER_RUN_NAME,
        root=tmp_path,
    )
    assert archive_path.is_file()


@pytest.mark.quick
def test_dataset_quality_writer_and_package_share_manifest_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """真实质量 writer 的完整 config 应由正式打包器直接复验。"""

    prompt_relative = "configs/manifest_chain_prompts.txt"
    prompt_path = tmp_path / prompt_relative
    prompt_path.parent.mkdir(parents=True)
    prompt_path.write_text(
        "测试 Prompt 0\n测试 Prompt 1\n测试 Prompt 2\n",
        encoding="utf-8",
    )
    paper_run = _minimal_paper_run(prompt_relative, 3)
    prompt_ids = dataset_level_quality_outputs.canonical_prompt_ids_for_paper_run(
        root_path=tmp_path,
        prompt_set=paper_run.prompt_set,
        prompt_file=paper_run.prompt_file,
    )
    registry_path = (
        tmp_path
        / "outputs/image_only_dataset_runtime"
        / PAPER_RUN_NAME
        / "watermark_quality_image_registry.jsonl"
    )
    registry_rows: list[dict[str, Any]] = []
    for index, prompt_id in enumerate(prompt_ids):
        source_path = tmp_path / f"outputs/manifest_chain_images/source_{index}.bin"
        attacked_path = tmp_path / f"outputs/manifest_chain_images/attacked_{index}.bin"
        source_path.parent.mkdir(parents=True, exist_ok=True)
        source_path.write_bytes(f"source-{index}".encode("utf-8"))
        attacked_path.write_bytes(f"attacked-{index}".encode("utf-8"))
        registry_rows.append(
            {
                "run_id": f"manifest_quality_run_{index}",
                "prompt_id": prompt_id,
                "attack_name": "watermark_embedding",
                "image_pair_role": "clean_to_watermarked",
                "source_image_path": source_path.relative_to(tmp_path).as_posix(),
                "source_image_digest": hashlib.sha256(
                    source_path.read_bytes()
                ).hexdigest(),
                "attacked_image_path": attacked_path.relative_to(tmp_path).as_posix(),
                "attacked_image_digest": hashlib.sha256(
                    attacked_path.read_bytes()
                ).hexdigest(),
                "supports_paper_claim": False,
            }
        )
    _write_jsonl(registry_path, registry_rows)

    quality_records = build_dataset_quality_image_records(registry_rows, tmp_path)
    item_identity: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    for record_index, record in enumerate(quality_records):
        for role, image_path, image_digest, value in (
            (
                "source",
                record.source_image_path,
                record.source_image_digest,
                float(record_index),
            ),
            (
                "comparison",
                record.comparison_image_path,
                record.comparison_image_digest,
                float(record_index) + 0.25,
            ),
        ):
            identity = {
                "dataset_quality_record_id": record.dataset_quality_record_id,
                "dataset_quality_image_role": role,
                "image_path": image_path,
                "image_digest": image_digest,
            }
            item_identity.append(identity)
            feature_rows.append(
                {
                    **identity,
                    "feature_backend": (
                        dataset_level_quality_outputs.FORMAL_FEATURE_BACKEND
                    ),
                    "feature_extractor_id": (
                        dataset_level_quality_outputs.FORMAL_FEATURE_EXTRACTOR_ID
                    ),
                    "feature_dimension": 2048,
                    "feature_vector": [value] * 2048,
                    "supports_paper_claim": False,
                }
            )
    batch_identity_digest = build_stable_digest(
        [
            (
                identity["dataset_quality_record_id"],
                identity["dataset_quality_image_role"],
            )
            for identity in item_identity
        ]
    )
    batch_provenance = build_test_scientific_unit_provenance(
        f"feature_batch_{batch_identity_digest[:16]}",
        dataset_level_quality_outputs._inception_batch_config_digest(
            item_identity
        ),
        formal_execution_lock=FORMAL_EXECUTION_LOCK,
    )
    for row in feature_rows:
        row["scientific_unit_provenance"] = batch_provenance
    feature_source_path = (
        tmp_path / "outputs/manifest_chain_features/formal_features.jsonl"
    )
    _write_jsonl(feature_source_path, feature_rows)

    estimand = load_attack_conditioned_quality_estimand()
    split_records = apply_split_assignments(
        build_prompt_records(PAPER_RUN_NAME, read_prompt_file(prompt_path))
    )
    test_prompt_ids = tuple(
        record.prompt_id for record in split_records if record.split == "test"
    )
    attack_quality_rows: list[dict[str, Any]] = []
    for prompt_id in test_prompt_ids:
        for attack_index, attack in enumerate(
            config for config in default_attack_configs() if config.enabled
        ):
            identity = {
                "prompt_id": prompt_id,
                "attack_id": attack.attack_id,
            }

            def image_identity(role: str) -> dict[str, Any]:
                """构造 manifest 链测试使用的确定性图像身份."""

                return {
                    "image_path": (
                        "outputs/manifest_chain_attack_images/"
                        f"{prompt_id}_{attack.attack_id}_{role}.bin"
                    ),
                    "image_sha256": build_stable_digest(
                        {**identity, "image_role": role}
                    ),
                    "image_rgb_uint8_content_sha256": build_stable_digest(
                        {**identity, "pixel_role": role}
                    ),
                    "image_width": 512,
                    "image_height": 512,
                }

            attack_core = {
                "record_schema": ATTACK_CONDITIONED_QUALITY_RECORD_SCHEMA,
                "quality_estimand_id": estimand["quality_estimand_id"],
                "quality_estimand_protocol_digest": estimand[
                    "quality_estimand_protocol_digest"
                ],
                "run_id": f"manifest_attack_{prompt_id}",
                "prompt_id": prompt_id,
                "split": "test",
                "randomization_repeat_id": paper_run.randomization_repeat_id,
                "sample_role": "four_image_matched_attack_pair",
                "attack_id": attack.attack_id,
                "attack_name": attack.attack_name,
                "attack_family": attack.attack_family,
                "attack_config_digest": attack_config_digest(attack),
                "attack_seed_random": 2000 + attack_index,
                "formal_attack_seed_protocol_digest": (
                    formal_attack_seed_protocol_record()[
                        "formal_attack_seed_protocol_digest"
                    ]
                ),
                "attack_parameters": dict(attack.attack_parameters),
                "source_image_role": "attacked_clean",
                "comparison_image_role": "attacked_watermarked",
                "image_pair_role": ATTACK_CONDITIONED_IMAGE_PAIR_ROLE,
                "clean_image": image_identity("clean"),
                "watermarked_image": image_identity("watermarked"),
                "attacked_clean_image": image_identity("attacked_clean"),
                "attacked_watermarked_image": image_identity(
                    "attacked_watermarked"
                ),
                "generation_scientific_unit_provenance": batch_provenance,
                "code_version": FORMAL_EXECUTION_LOCK[
                    "formal_execution_commit"
                ],
                "scientific_dependency_profile_id": "sd35_method_runtime_gpu",
                "scientific_dependency_profile_digest": "1" * 64,
                "scientific_complete_hash_lock_digest": "3" * 64,
                "supports_paper_claim": False,
            }
            attack_digest = build_stable_digest(attack_core)
            attack_quality_rows.append(
                {
                    "attack_quality_record_id": (
                        f"attack_quality_record_{attack_digest[:16]}"
                    ),
                    "attack_quality_record_digest": attack_digest,
                    **attack_core,
                }
            )
    attack_registry_path = (
        tmp_path
        / "outputs/image_only_dataset_runtime"
        / PAPER_RUN_NAME
        / "attack_conditioned_quality_image_records.jsonl"
    )
    _write_jsonl(attack_registry_path, attack_quality_rows)
    attack_pair_rows = list(
        attack_quality_dataset_image_records(attack_quality_rows)
    )
    attack_item_identity: list[dict[str, Any]] = []
    attack_feature_rows: list[dict[str, Any]] = []
    for pair_index, pair in enumerate(attack_pair_rows):
        for role, path_field, digest_field in (
            ("source", "source_image_path", "source_image_digest"),
            (
                "comparison",
                "comparison_image_path",
                "comparison_image_digest",
            ),
        ):
            item = {
                "dataset_quality_record_id": pair[
                    "dataset_quality_record_id"
                ],
                "dataset_quality_image_role": role,
                "image_path": pair[path_field],
                "image_digest": pair[digest_field],
            }
            attack_item_identity.append(item)
            attack_feature_rows.append(
                {
                    **item,
                    "feature_backend": (
                        dataset_level_quality_outputs.FORMAL_FEATURE_BACKEND
                    ),
                    "feature_extractor_id": (
                        dataset_level_quality_outputs.FORMAL_FEATURE_EXTRACTOR_ID
                    ),
                    "feature_dimension": 2048,
                    "feature_vector": [float(pair_index % 5)] * 2048,
                    "supports_paper_claim": False,
                }
            )
    attack_batch_identity_digest = build_stable_digest(
        [
            (
                item["dataset_quality_record_id"],
                item["dataset_quality_image_role"],
            )
            for item in attack_item_identity
        ]
    )
    attack_batch_provenance = build_test_scientific_unit_provenance(
        f"feature_batch_{attack_batch_identity_digest[:16]}",
        dataset_level_quality_outputs._inception_batch_config_digest(
            attack_item_identity
        ),
        formal_execution_lock=FORMAL_EXECUTION_LOCK,
    )
    for row in attack_feature_rows:
        row["scientific_unit_provenance"] = attack_batch_provenance
    output_dir = tmp_path / "outputs/dataset_level_quality" / PAPER_RUN_NAME
    attack_output_dir = output_dir / "attack_conditioned_quality"
    _write_jsonl(
        attack_output_dir
        / "attack_conditioned_quality_inception_feature_records.jsonl",
        attack_feature_rows,
    )
    clip_vector = [1.0] + [0.0] * (FORMAL_CLIP_FEATURE_DIMENSION - 1)
    clip_rows: list[dict[str, Any]] = []
    for pair in (*quality_records, *attack_pair_rows):
        for role, path_field, digest_field in (
            ("source", "source_image_path", "source_image_digest"),
            (
                "comparison",
                "comparison_image_path",
                "comparison_image_digest",
            ),
        ):
            clip_rows.append(
                {
                    "dataset_quality_record_id": pair.dataset_quality_record_id
                    if not isinstance(pair, dict)
                    else pair["dataset_quality_record_id"],
                    "dataset_quality_image_role": role,
                    "feature_backend": FORMAL_CLIP_FEATURE_BACKEND,
                    "feature_extractor_id": "formal_test_clip_extractor",
                    "feature_dimension": FORMAL_CLIP_FEATURE_DIMENSION,
                    "image_path": getattr(pair, path_field, None)
                    if not isinstance(pair, dict)
                    else pair[path_field],
                    "image_digest": getattr(pair, digest_field, None)
                    if not isinstance(pair, dict)
                    else pair[digest_field],
                    "feature_vector": clip_vector,
                    "quality_estimand_protocol_digest": estimand[
                        "quality_estimand_protocol_digest"
                    ],
                    "scientific_unit_provenance": batch_provenance,
                    "supports_paper_claim": False,
                }
            )
    _write_jsonl(
        attack_output_dir / "paired_quality_clip_feature_records.jsonl",
        clip_rows,
    )
    independent_protocol = load_independent_semantic_quality_evaluator()
    independent_vector = [1.0] + [0.0] * (
        INDEPENDENT_SEMANTIC_FEATURE_DIMENSION - 1
    )
    independent_provenance = build_test_scientific_unit_provenance(
        "independent_semantic_manifest_fixture",
        build_stable_digest({"independent_semantic_manifest_fixture": 1}),
        formal_execution_lock=FORMAL_EXECUTION_LOCK,
        dependency_profile_digest=independent_protocol[
            "dependency_profile_digest"
        ],
        complete_hash_lock_digest=independent_protocol[
            "complete_hash_lock_digest"
        ],
    )
    independent_rows = [
        {
            "dataset_quality_record_id": row["dataset_quality_record_id"],
            "dataset_quality_image_role": row[
                "dataset_quality_image_role"
            ],
            "feature_backend": INDEPENDENT_SEMANTIC_FEATURE_BACKEND,
            "feature_extractor_id": (
                f"{independent_protocol['model_contract']['model_id']}@"
                f"{independent_protocol['model_contract']['model_revision']}"
            ),
            "feature_dimension": INDEPENDENT_SEMANTIC_FEATURE_DIMENSION,
            "feature_layer": "last_hidden_state_cls_token",
            "feature_normalization": "l2",
            "image_path": row["image_path"],
            "image_digest": row["image_digest"],
            "feature_vector": independent_vector,
            "feature_vector_digest": build_stable_digest(independent_vector),
            "independent_semantic_quality_protocol_digest": (
                independent_protocol[
                    "independent_semantic_quality_protocol_digest"
                ]
            ),
            "scientific_unit_provenance": independent_provenance,
            "supports_paper_claim": False,
        }
        for row in clip_rows
    ]
    _write_jsonl(
        attack_output_dir
        / "paired_quality_independent_semantic_feature_records.jsonl",
        independent_rows,
    )

    monkeypatch.setattr(
        repository_environment,
        "require_published_formal_execution_lock",
        lambda _root: dict(FORMAL_EXECUTION_LOCK),
    )
    monkeypatch.setattr(
        dataset_level_quality_outputs,
        "build_paper_run_config",
        lambda _root: paper_run,
    )
    monkeypatch.setitem(
        dataset_level_quality_outputs.RUN_EXPECTED_PROMPT_COUNTS,
        PAPER_RUN_NAME,
        3,
    )
    monkeypatch.setattr(
        dataset_level_quality_outputs,
        "restore_role_checkpoints",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        dataset_level_quality_outputs,
        "require_dependency_profile_ready",
        lambda *_args, **_kwargs: SimpleNamespace(
            profile_digest="1" * 64,
            complete_hash_lock_digest="3" * 64,
        ),
    )

    def measured_metric_rows(
        records: Any,
        _root_path: Path,
        **_kwargs: Any,
    ) -> list[dict[str, Any]]:
        """避免在 manifest 集成测试中执行 2048 维重型数值分解。"""

        count = len(tuple(records))
        return [
            {
                "quality_metric_name": name,
                "quality_metric_value": value,
                "metric_status": "measured",
                "paper_metric_name": name,
                "feature_backend": (
                    dataset_level_quality_outputs.FORMAL_FEATURE_BACKEND
                ),
                "source_image_count": count,
                "comparison_image_count": count,
                "sample_pair_count": count,
                "supports_paper_claim": False,
            }
            for name, value in zip(
                FORMAL_DATASET_QUALITY_METRIC_NAMES,
                (1.0, 0.01, 0.0),
                strict=True,
            )
        ]

    monkeypatch.setattr(
        dataset_level_quality_outputs,
        "build_dataset_quality_metric_rows",
        measured_metric_rows,
    )

    def paired_metric_records(
        base_records: Any,
        attack_records: Any,
        _clip_rows: Any,
        _independent_semantic_rows: Any,
        *,
        randomization_repeat_id: str,
        **_kwargs: Any,
    ) -> tuple[dict[str, Any], ...]:
        """构造由生产 validator 复验的轻量配对质量记录."""

        rows: list[dict[str, Any]] = []
        for scope, records_for_scope in (
            ("base", tuple(base_records)),
            ("registered_attack", tuple(attack_records)),
        ):
            for record in records_for_scope:
                attack_id = (
                    "none"
                    if scope == "base"
                    else str(getattr(record, "attack_id"))
                )
                core = {
                    "record_schema": PAIRED_QUALITY_METRIC_RECORD_SCHEMA,
                    "dataset_quality_record_id": getattr(
                        record,
                        "dataset_quality_record_id",
                    ),
                    "dataset_quality_record_digest": getattr(
                        record,
                        "dataset_quality_record_digest",
                    ),
                    "attack_quality_record_id": getattr(
                        record,
                        "attack_quality_record_id",
                        "",
                    ),
                    "randomization_repeat_id": randomization_repeat_id,
                    "prompt_id": getattr(record, "prompt_id"),
                    "estimand_scope": scope,
                    "sample_role": (
                        "base_quality_pair"
                        if scope == "base"
                        else "matched_attack_quality_pair"
                    ),
                    "attack_id": attack_id,
                    "attack_config_digest": getattr(
                        record,
                        "attack_config_digest",
                        "",
                    ),
                    "attack_seed_random": getattr(
                        record,
                        "attack_seed_random",
                        None,
                    ),
                    "image_pair_role": getattr(
                        record,
                        "image_pair_role",
                    ),
                    "source_image_digest": getattr(
                        record,
                        "source_image_digest",
                    ),
                    "comparison_image_digest": getattr(
                        record,
                        "comparison_image_digest",
                    ),
                    "paired_ssim": 1.0,
                    "clip_cosine": 1.0,
                    "clip_evidence_role": (
                        "mechanism_consistency_diagnostic"
                    ),
                    "clip_source_feature_digest": build_stable_digest(
                        clip_vector
                    ),
                    "clip_comparison_feature_digest": (
                        build_stable_digest(clip_vector)
                    ),
                    "independent_semantic_cosine": 1.0,
                    "independent_semantic_evidence_role": (
                        "independent_semantic_preservation_primary"
                    ),
                    "independent_semantic_source_feature_digest": (
                        build_stable_digest(independent_vector)
                    ),
                    "independent_semantic_comparison_feature_digest": (
                        build_stable_digest(independent_vector)
                    ),
                    "independent_semantic_quality_protocol_digest": (
                        independent_protocol[
                            "independent_semantic_quality_protocol_digest"
                        ]
                    ),
                    "quality_estimand_protocol_digest": estimand[
                        "quality_estimand_protocol_digest"
                    ],
                    "supports_paper_claim": False,
                }
                digest = build_stable_digest(core)
                rows.append(
                    {
                        "paired_quality_metric_record_id": (
                            f"paired_quality_metric_{digest[:16]}"
                        ),
                        "paired_quality_metric_record_digest": digest,
                        **core,
                    }
                )
        return tuple(rows)

    monkeypatch.setattr(
        dataset_level_quality_outputs,
        "build_paired_quality_metric_records",
        paired_metric_records,
    )
    manifest = dataset_level_quality_outputs.write_dataset_level_quality_outputs(
        paper_run_name=PAPER_RUN_NAME,
        target_fpr=TARGET_FPR,
        root=tmp_path,
        quality_image_registry_path=registry_path,
        attack_quality_registry_path=attack_registry_path,
        formal_feature_records_path=feature_source_path,
        formal_min_sample_count=3,
    )

    assert manifest["config_digest"] == build_stable_digest(manifest["config"])
    assert manifest["config"]["formal_feature_records_sha256"] == (
        dataset_level_quality_outputs.path_digest(
            output_dir / "dataset_quality_formal_feature_records.jsonl"
        )
    )

    write_test_scientific_execution_binding(
        repository_root=tmp_path,
        artifact_dir=output_dir,
        artifact_role="dataset_level_quality",
        paper_run_name=PAPER_RUN_NAME,
        profile_id="sd35_method_runtime_gpu",
        summary_file_name="dataset_quality_summary.json",
        manifest_file_name="manifest.local.json",
        formal_execution_lock=FORMAL_EXECUTION_LOCK,
        execution_route="semantic_watermark_session",
    )
    archive_path = (
        dataset_level_quality_outputs.package_dataset_level_quality_outputs(
            PAPER_RUN_NAME,
            root=tmp_path,
        )
    )
    assert archive_path.is_file()
