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
from experiments.artifacts.artifact_manifest import build_artifact_manifest
from experiments.artifacts.manifest_schema import manifest_config_digest_ready
from experiments.protocol.attacks import attack_config_digest, default_attack_configs
from experiments.protocol.dataset_quality import (
    FORMAL_DATASET_QUALITY_METRIC_NAMES,
    build_dataset_quality_image_records,
)
from experiments.protocol.prompts import build_prompt_records, read_prompt_file
from experiments.protocol.formal_randomization import (
    formal_randomization_protocol_record,
    resolve_formal_randomization_repeat,
)
from experiments.protocol.splits import apply_split_assignments
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
from tests.helpers.scientific_execution_binding import (
    write_test_scientific_execution_binding,
)
from tests.helpers.scientific_unit_provenance import (
    build_test_scientific_unit_provenance,
)


PAPER_RUN_NAME = "probe_paper"
TARGET_FPR = 0.1
FORMAL_EXECUTION_LOCK = build_test_formal_execution_lock()
ATTENTION_ALIGNMENT_GATE = {
    "attention_anchor_count": 12,
    "attention_residual_threshold": 0.20,
    "attention_minimum_inlier_ratio": 0.50,
}


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
) -> dict[str, Any]:
    """构造配置、run id 和科学来源完全一致的最小运行结果。"""

    unit_config = semantic_watermark_runtime_config_payload(config)
    run_id = build_semantic_watermark_run_id(config)
    return {
        "run_id": run_id,
        "run_decision": "pass",
        "metadata": {
            "scientific_unit_config": unit_config,
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
    sample_role: str,
    content_score: float,
    attack: Any | None = None,
) -> dict[str, Any]:
    """构造能够进入真实 fixed-FPR 校准和攻击覆盖检查的检测原子。"""

    record: dict[str, Any] = {
        "sample_role": sample_role,
        "content_score": content_score,
        "aligned_content_score": None,
        "attention_geometry_score": 0.0,
        "registration_confidence": 0.0,
        "attention_sync_score": 0.0,
        "metadata": {
            "attention_alignment_gate": dict(
                ATTENTION_ALIGNMENT_GATE
            ),
            **ATTENTION_ALIGNMENT_GATE,
        },
        "alignment": {
            "registration_geometry_reliable": False,
            "metadata": {
                "attention_alignment_gate": dict(
                    ATTENTION_ALIGNMENT_GATE
                ),
            },
            **ATTENTION_ALIGNMENT_GATE,
        },
    }
    if attack is not None:
        record.update(
            {
                "attack_id": attack.attack_id,
                "attack_family": attack.attack_family,
                "attack_name": attack.attack_name,
                "resource_profile": attack.resource_profile,
                "attack_config_digest": attack_config_digest(attack),
                "attack_parameters": attack.attack_parameters,
                "attack_performed": True,
            }
        )
    return record


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
        "测试 Prompt 0\n测试 Prompt 1\n测试 Prompt 2\n",
        encoding="utf-8",
    )
    paper_run = _minimal_paper_run(prompt_relative, 3)
    prompt_records = apply_split_assignments(
        build_prompt_records(
            paper_run.prompt_set,
            read_prompt_file(prompt_path),
        )
    )
    base_configs = tuple(
        SemanticWatermarkRuntimeConfig(
            prompt=record.prompt_text,
            prompt_id=record.prompt_id,
            split=record.split,
            seed=100 + record.prompt_index,
        )
        for record in prompt_records
    )

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
            _detection_record(sample_role="clean_negative", content_score=0.0),
            _detection_record(
                sample_role="positive_source",
                content_score=positive_score,
            ),
            _detection_record(sample_role="wrong_key_negative", content_score=0.0),
        ]
        if config.split == "test":
            rows.extend(
                _detection_record(
                    sample_role=sample_role,
                    content_score=(
                        positive_score if sample_role == "positive_source" else 0.0
                    ),
                    attack=attack,
                )
                for attack in default_attack_configs()
                if attack.enabled
                and attack.resource_profile in {"full_main", "full_extra"}
                for sample_role in ("clean_negative", "positive_source")
            )
        payload = _runtime_result_payload(config)
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
        expected_prompt_count=3,
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
    manifest = dataset_level_quality_outputs.write_dataset_level_quality_outputs(
        paper_run_name=PAPER_RUN_NAME,
        target_fpr=TARGET_FPR,
        root=tmp_path,
        quality_image_registry_path=registry_path,
        formal_feature_records_path=feature_source_path,
        formal_min_sample_count=3,
    )
    output_dir = tmp_path / "outputs/dataset_level_quality" / PAPER_RUN_NAME

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
