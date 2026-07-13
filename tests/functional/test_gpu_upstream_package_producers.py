"""验证主方法 GPU 上游包与 CPU 闭合选择契约完全一致。"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import pytest

from experiments.runtime import repository_environment
from experiments.runtime.package_input_manifest import (
    validate_exact_package_archive,
)
from experiments.ablations.runtime_rerun import (
    FORMAL_RUNTIME_RERUN_ABLATION_IDS,
    FORMAL_RUNTIME_RERUN_ABLATION_SPEC_DIGEST,
    default_runtime_rerun_ablation_specs,
    package_runtime_rerun_ablations,
)
from experiments.ablations.necessity_statistics import (
    ABLATION_NECESSITY_FIELDNAMES,
    build_ablation_necessity_statistics,
)
from experiments.artifacts.dataset_level_quality_outputs import (
    FORMAL_FEATURE_BACKEND,
    FORMAL_FEATURE_EXTRACTOR_ID,
    _inception_batch_config_digest,
    canonical_prompt_ids_for_paper_run,
    package_dataset_level_quality_outputs,
    path_digest,
)
from experiments.protocol.paper_run_config import build_paper_run_config
from experiments.protocol.dataset_quality import (
    FORMAL_DATASET_QUALITY_METRIC_NAMES,
    formal_dataset_quality_metric_protocol,
)
from experiments.protocol.prompts import (
    PROMPT_FILES,
    build_prompt_records,
    read_prompt_file,
)
from experiments.protocol.splits import (
    apply_split_assignments,
    group_prompt_ids_by_split,
)
from experiments.runtime.scientific_unit_provenance import (
    aggregate_scientific_unit_provenance,
)
from main.core.digest import build_stable_digest
from experiments.runners.image_only_dataset_runtime import (
    _formal_attention_alignment_gate_record_ready,
    _write_prompt_source_snapshot,
    package_image_only_dataset_runtime,
)
from experiments.runners.semantic_watermark_runtime import (
    SemanticWatermarkRuntimeConfig,
    build_semantic_watermark_run_id,
    semantic_watermark_runtime_config_payload,
)
from paper_experiments.runners.closure_package_selection import (
    CLOSURE_PACKAGE_FAMILY_SPECS,
    ClosurePackageSelectionError,
    inspect_closure_package,
)
from tests.helpers.formal_execution_lock import build_test_formal_execution_lock
from tests.helpers.formal_detection_record import bind_formal_detection_record
from tests.helpers.scientific_execution_binding import (
    write_test_scientific_execution_binding,
)
from tests.helpers.scientific_unit_provenance import (
    build_test_scientific_unit_provenance,
)
from tests.functional.test_scientific_content_binding import (
    _artifact_fixture,
)


PAPER_RUN_NAME = "probe_paper"
TARGET_FPR = 0.1
PROMPT_COUNT = 70
GENERATED_AT = "2026-07-11T00:00:00+00:00"
FORMAL_EXECUTION_LOCK = build_test_formal_execution_lock()
ATTENTION_ALIGNMENT_GATE = {
    "attention_anchor_count": 12,
    "attention_residual_threshold": 0.20,
    "attention_minimum_inlier_ratio": 0.50,
}


@pytest.fixture(autouse=True)
def _publish_formal_execution_lock(monkeypatch: pytest.MonkeyPatch) -> None:
    """把临时输出目录绑定到确定性正式执行锁."""

    monkeypatch.setattr(
        repository_environment,
        "require_published_formal_execution_lock",
        lambda _root: dict(FORMAL_EXECUTION_LOCK),
    )


def _write_json(path: Path, payload: object) -> None:
    """写出稳定 JSON 测试产物。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    """写出保持输入顺序的 JSONL 测试记录."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _canonical_prompt_records(root: Path) -> tuple[object, ...]:
    """读取当前测试论文层级的完整规范 Prompt 与 split."""

    repository_root = Path(__file__).resolve().parents[2]
    for relative_path in (
        Path("configs/prompt_source_registry.json"),
        Path("configs/prompt_selection_manifest.jsonl"),
        PROMPT_FILES[PAPER_RUN_NAME],
    ):
        target_path = root / relative_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        if not target_path.is_file():
            target_path.write_bytes((repository_root / relative_path).read_bytes())
    paper_run = build_paper_run_config(root)
    prompt_path = root / paper_run.prompt_file
    return apply_split_assignments(
        build_prompt_records(
            paper_run.prompt_set,
            read_prompt_file(prompt_path),
        )
    )


def _runtime_result_payload(
    config: SemanticWatermarkRuntimeConfig,
    *,
    seed: int,
) -> dict[str, object]:
    """构造配置、run id 与来源自摘要一致的最小完成结果."""

    unit_config = semantic_watermark_runtime_config_payload(config)
    config_digest = build_stable_digest(unit_config)
    run_id = build_semantic_watermark_run_id(config)
    return {
        "run_id": run_id,
        "run_decision": "pass",
        "metadata": {
            "scientific_unit_config": unit_config,
            "scientific_unit_provenance": (
                build_test_scientific_unit_provenance(
                    run_id,
                    config_digest,
                    seed=seed,
                    formal_execution_lock=FORMAL_EXECUTION_LOCK,
                )
            ),
        },
    }


def _base_runtime_config(
    prompt_record: object,
    paper_run: object,
) -> SemanticWatermarkRuntimeConfig:
    """按正式论文配置构造一个 Prompt 的完整主方法运行身份."""

    return SemanticWatermarkRuntimeConfig(
        prompt=prompt_record.prompt_text,
        prompt_id=prompt_record.prompt_id,
        split=prompt_record.split,
        seed=20260711 + prompt_record.prompt_index,
        inference_steps=paper_run.inference_steps,
        guidance_scale=paper_run.guidance_scale,
        injection_step_indices=paper_run.attention_injection_steps,
        output_dir=(
            f"outputs/image_only_dataset_runtime/{PAPER_RUN_NAME}/runs"
        ),
    )


def _write_required_files(directory: Path, filenames: tuple[str, ...]) -> None:
    """写出不承担字段身份的必要文件。"""

    directory.mkdir(parents=True, exist_ok=True)
    for filename in filenames:
        path = directory / filename
        path.write_text("{}\n", encoding="utf-8")


def _randomization_repeat_identity(paper_run: object) -> dict[str, object]:
    """构造上游包 summary 与 manifest 共用的活动 repeat 身份."""

    return {
        "randomization_repeat_id": paper_run.randomization_repeat_id,
        "generation_seed_index": paper_run.generation_seed_index,
        "generation_seed_offset": paper_run.generation_seed_offset,
        "watermark_key_index": paper_run.watermark_key_index,
        "formal_randomization_protocol_digest": (
            paper_run.formal_randomization_protocol_digest
        ),
    }


def _prepare_image_runtime(
    root: Path,
    *,
    digest_only: bool = False,
    tamper_unit_leaf: bool = False,
    omit_unit_leaves: bool = False,
) -> Path:
    """构造仅图像运行打包门禁所需的最小正式形状。"""

    directory = root / "outputs" / "image_only_dataset_runtime" / PAPER_RUN_NAME
    prompt_records = _canonical_prompt_records(root)
    paper_run = build_paper_run_config(root)
    repeat_identity = _randomization_repeat_identity(paper_run)
    formal_detection_record = bind_formal_detection_record(
        {"content_score": 0.0}
    )
    formal_detector_identity = {
        field_name: formal_detection_record[field_name]
        for field_name in (
            "lf_carrier_protocol_digest",
            "tail_carrier_protocol_digest",
            "lf_weight",
            "tail_robust_weight",
            "tail_fraction",
            "image_only_detector_config_digest",
        )
    }
    runtime_results = []
    scientific_unit_output_paths: list[str] = []
    for prompt_record in prompt_records:
        config = _base_runtime_config(prompt_record, paper_run)
        run_id = build_semantic_watermark_run_id(config)
        _, content_result, unit_manifest, _ = _artifact_fixture(
            root,
            runtime_config=config,
            relative_output_dir=f"{config.output_dir}/{run_id}",
            run_id=run_id,
        )
        result = _runtime_result_payload(
            config,
            seed=20260711 + prompt_record.prompt_index,
        )
        result.update(
            {
                field_name: content_result[field_name]
                for field_name in (
                    "update_record_path",
                    "detection_record_path",
                    "clean_image_path",
                    "watermarked_image_path",
                )
            }
        )
        result["metadata"] = {
            **content_result["metadata"],
            **result["metadata"],
        }
        unit_dir = root / config.output_dir / run_id
        result_path = unit_dir / "runtime_result.json"
        unit_manifest_path = unit_dir / "manifest.local.json"
        result["manifest_path"] = unit_manifest_path.relative_to(
            root
        ).as_posix()
        result_path.write_text(
            json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        unit_manifest["output_paths"].extend(
            (
                result_path.relative_to(root).as_posix(),
                unit_manifest_path.relative_to(root).as_posix(),
            )
        )
        unit_config = semantic_watermark_runtime_config_payload(config)
        unit_manifest["config"] = unit_config
        unit_manifest["config_digest"] = build_stable_digest(unit_config)
        unit_manifest_path.write_text(
            json.dumps(
                unit_manifest,
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        scientific_unit_output_paths.extend(unit_manifest["output_paths"])
        runtime_results.append(result)
    if digest_only:
        for result in runtime_results:
            result["metadata"].pop("scientific_content_binding_schema")
            result["metadata"].pop("scientific_content_binding_record")
    if tamper_unit_leaf:
        update_path = root / str(runtime_results[0]["update_record_path"])
        update_records = [
            json.loads(line)
            for line in update_path.read_text(encoding="utf-8").splitlines()
            if line
        ]
        update_records[0]["semantic_risk_signal_content_sha256"] = "f" * 64
        _write_jsonl(update_path, update_records)
    _write_jsonl(directory / "runtime_results.jsonl", runtime_results)
    _write_required_files(
        directory,
        (
            "image_only_detection_records.jsonl",
            "watermark_quality_image_registry.jsonl",
            "frozen_evidence_protocol.json",
            "test_detection_metrics.csv",
            "score_distribution_table.csv",
            "roc_curve_points.csv",
            "det_curve_points.csv",
        ),
    )
    threshold_digest = "c" * 64
    _write_json(
        directory / "frozen_evidence_protocol.json",
        {
            **ATTENTION_ALIGNMENT_GATE,
            **formal_detector_identity,
            "threshold_digest": threshold_digest,
        },
    )
    prompt_source_snapshot_paths, prompt_source_report = (
        _write_prompt_source_snapshot(
            root_path=root,
            output_dir=directory,
            paper_run_name=PAPER_RUN_NAME,
            prompt_path=root / PROMPT_FILES[PAPER_RUN_NAME],
        )
    )
    provenance_summary = aggregate_scientific_unit_provenance(
        (
            result["metadata"]["scientific_unit_provenance"]
            for result in runtime_results
        ),
        expected_reference_count=PROMPT_COUNT,
    )
    scientific_content_binding_digests = [
        result["metadata"]["scientific_content_binding_digest"]
        for result in runtime_results
    ]
    scientific_content_binding_digest = build_stable_digest(
        {
            "scientific_content_binding_digests": (
                scientific_content_binding_digests
            )
        }
    )
    _write_json(
        directory / "dataset_runtime_summary.json",
        {
            "generated_at": GENERATED_AT,
            "paper_run_name": PAPER_RUN_NAME,
            "target_fpr": TARGET_FPR,
            "randomization_repeat_identity": repeat_identity,
            "protocol_decision": "pass",
            "attention_alignment_gate": dict(
                ATTENTION_ALIGNMENT_GATE
            ),
            **ATTENTION_ALIGNMENT_GATE,
            **formal_detector_identity,
            "frozen_threshold_digest": threshold_digest,
            "full_method_component_ready": True,
            "geometry_protocol_calibration_ready": True,
            "detection_curve_data_ready": True,
            "scientific_content_binding_digests": (
                scientific_content_binding_digests
            ),
            "scientific_content_binding_digest": (
                scientific_content_binding_digest
            ),
            "scientific_content_binding_failure_count": 0,
            "scientific_content_binding_gate_ready": True,
            **provenance_summary,
            "repeat_component_ready": True,
            "randomization_aggregate_ready": False,
            "supports_paper_claim": False,
            "prompt_file_sha256": prompt_source_report[
                "prompt_file_sha256"
            ],
            "prompt_source_registry_digest": prompt_source_report[
                "prompt_source_registry_digest"
            ],
            "selection_manifest_sha256": prompt_source_report[
                "selection_manifest_sha256"
            ],
            "selection_manifest_digest": prompt_source_report[
                "selection_manifest_digest"
            ],
            "packaged_prompt_source_audit_digest": prompt_source_report[
                "packaged_prompt_source_audit_digest"
            ],
            "prompt_source_contract_ready": True,
        },
    )
    method_config = runtime_results[0]["metadata"][
        "scientific_unit_config"
    ]
    manifest_config = {
        "paper_run": {
            "run_name": PAPER_RUN_NAME,
            "target_fpr": TARGET_FPR,
            **repeat_identity,
        },
        "method_config": method_config,
    }
    _write_json(
        directory / "manifest.local.json",
        {
            "artifact_id": f"{PAPER_RUN_NAME}_image_only_dataset_runtime_manifest",
            "artifact_type": "local_manifest",
            "code_version": "b" * 40,
            "formal_execution_run_lock": FORMAL_EXECUTION_LOCK,
            "output_paths": [
                (directory / filename).relative_to(root).as_posix()
                for filename in (
                    "runtime_results.jsonl",
                    "image_only_detection_records.jsonl",
                    "watermark_quality_image_registry.jsonl",
                    "frozen_evidence_protocol.json",
                    "test_detection_metrics.csv",
                    "score_distribution_table.csv",
                    "roc_curve_points.csv",
                    "det_curve_points.csv",
                    "dataset_runtime_summary.json",
                    "manifest.local.json",
                )
            ]
            + [
                path.relative_to(root).as_posix()
                for path in prompt_source_snapshot_paths
            ]
            + ([] if omit_unit_leaves else scientific_unit_output_paths),
            "config": manifest_config,
            "config_digest": build_stable_digest(manifest_config),
            "metadata": {
                "attention_alignment_gate": dict(
                    ATTENTION_ALIGNMENT_GATE
                ),
                **ATTENTION_ALIGNMENT_GATE,
                **formal_detector_identity,
                "geometry_protocol_calibration_ready": True,
                "scientific_content_binding_digest": (
                    scientific_content_binding_digest
                ),
                "scientific_content_binding_failure_count": 0,
                "scientific_content_binding_gate_ready": True,
            },
        },
    )
    write_test_scientific_execution_binding(
        repository_root=root,
        artifact_dir=directory,
        artifact_role="image_only_dataset_runtime",
        paper_run_name=PAPER_RUN_NAME,
        profile_id="sd35_method_runtime_gpu",
        summary_file_name="dataset_runtime_summary.json",
        manifest_file_name="manifest.local.json",
        formal_execution_lock=FORMAL_EXECUTION_LOCK,
        execution_route="semantic_watermark_session",
    )
    return package_image_only_dataset_runtime(PAPER_RUN_NAME, root=root)


def _prepare_ablation(root: Path) -> Path:
    """构造正式重运行消融打包门禁所需的最小正式形状。"""

    directory = root / "outputs" / "formal_mechanism_ablation" / PAPER_RUN_NAME
    prompt_records = _canonical_prompt_records(root)
    paper_run = build_paper_run_config(root)
    repeat_identity = _randomization_repeat_identity(paper_run)
    specs = default_runtime_rerun_ablation_specs()
    ablation_records: list[dict[str, object]] = []
    for spec in specs:
        for prompt_record in prompt_records:
            base_config = _base_runtime_config(prompt_record, paper_run)
            run_config = spec.apply(
                base_config,
                f"outputs/formal_mechanism_ablation/{PAPER_RUN_NAME}",
            )
            ablation_records.append(
                {
                    "prompt_id": prompt_record.prompt_id,
                    "prompt_digest": build_stable_digest(
                        {"prompt_text": prompt_record.prompt_text}
                    ),
                    "split": prompt_record.split,
                    "ablation_id": spec.ablation_id,
                    "runtime_config": spec.to_dict(),
                    "runtime_result": _runtime_result_payload(
                        run_config,
                        seed=run_config.seed,
                    ),
                    "formal_attack_coverage_ready": True,
                    "attacked_positive_rate": (
                        1.0 if spec.ablation_id == "complete_method" else 0.0
                    ),
                    "positive_source_positive": (
                        spec.ablation_id == "complete_method"
                    ),
                    "paired_ssim": 0.95,
                }
            )
    _write_jsonl(
        directory / "runtime_rerun_records.jsonl",
        ablation_records,
    )
    _write_required_files(
        directory,
        (
            "formal_detection_records.jsonl",
            "mechanism_pairwise_delta.csv",
        ),
    )
    frozen_protocol_payload = {
        ablation_id: {} for ablation_id in FORMAL_RUNTIME_RERUN_ABLATION_IDS
    }
    _write_json(
        directory / "per_ablation_frozen_protocols.json",
        frozen_protocol_payload,
    )
    (directory / "mechanism_ablation_metrics.csv").write_text(
        "ablation_id\n"
        + "".join(f"{ablation_id}\n" for ablation_id in FORMAL_RUNTIME_RERUN_ABLATION_IDS),
        encoding="utf-8",
    )
    ablation_contract = {
        "expected_ablation_ids": list(FORMAL_RUNTIME_RERUN_ABLATION_IDS),
        "actual_ablation_ids": list(FORMAL_RUNTIME_RERUN_ABLATION_IDS),
        "ablation_spec_digest": FORMAL_RUNTIME_RERUN_ABLATION_SPEC_DIGEST,
        "ablation_exact_set_ready": True,
    }
    split_prompt_ids = group_prompt_ids_by_split(prompt_records)
    prompt_contract = {
        "prompt_id_digest": build_stable_digest(
            sorted(record.prompt_id for record in prompt_records)
        ),
        "calibration_prompt_id_digest": build_stable_digest(
            sorted(split_prompt_ids["calibration"])
        ),
        "test_prompt_id_digest": build_stable_digest(
            sorted(split_prompt_ids["test"])
        ),
        "prompt_protocol_exact_set_ready": True,
    }
    variant_ids = tuple(
        ablation_id
        for ablation_id in FORMAL_RUNTIME_RERUN_ABLATION_IDS
        if ablation_id != "complete_method"
    )
    necessity_rows, necessity_summary = build_ablation_necessity_statistics(
        ablation_records,
        expected_ablation_ids=variant_ids,
        expected_paired_prompt_count=len(split_prompt_ids["test"]),
        bootstrap_resample_count=1000,
    )
    with (directory / "mechanism_necessity_statistics.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=ABLATION_NECESSITY_FIELDNAMES)
        writer.writeheader()
        writer.writerows(necessity_rows)
    _write_json(
        directory / "mechanism_necessity_summary.json",
        necessity_summary,
    )
    provenance_summary = aggregate_scientific_unit_provenance(
        (
            record["runtime_result"]["metadata"][
                "scientific_unit_provenance"
            ]
            for record in ablation_records
        ),
        expected_reference_count=len(ablation_records),
    )
    atom_identity = {
        "formal_detection_records_sha256": path_digest(
            directory / "formal_detection_records.jsonl"
        ),
        "formal_detection_records_digest": build_stable_digest([{}]),
        "per_ablation_frozen_protocols_sha256": path_digest(
            directory / "per_ablation_frozen_protocols.json"
        ),
        "per_ablation_frozen_protocols_digest": build_stable_digest(
            frozen_protocol_payload
        ),
    }
    expected_attacked_run_count = len(split_prompt_ids["test"]) * len(specs)
    _write_json(
        directory / "ablation_component_summary.json",
        {
            "generated_at": GENERATED_AT,
            "paper_run_name": PAPER_RUN_NAME,
            "target_fpr": TARGET_FPR,
            "randomization_repeat_identity": repeat_identity,
            **ablation_contract,
            **prompt_contract,
            **atom_identity,
            "record_count": len(ablation_records),
            "expected_attack_and_detection_rerun_count": (
                expected_attacked_run_count
            ),
            "attack_and_detection_rerun_count": expected_attacked_run_count,
            "formal_attack_coverage_ready_count": len(ablation_records),
            "formal_attack_coverage_ready": True,
            **provenance_summary,
            **necessity_summary,
            "protocol_decision": "pass",
            "ablation_component_ready": True,
            "repeat_component_ready": True,
            "randomization_aggregate_ready": False,
            "supports_paper_claim": False,
        },
    )
    ablation_manifest_config = {
        "specs": [spec.to_dict() for spec in specs],
        "target_fpr": TARGET_FPR,
        "randomization_repeat_identity": repeat_identity,
        **ablation_contract,
        **prompt_contract,
        **atom_identity,
        "prompt_count": len(prompt_records),
        "split_counts": {
            split: sum(record.split == split for record in prompt_records)
            for split in ("dev", "calibration", "test")
        },
        "record_digest": build_stable_digest(ablation_records),
        "necessity_statistic_rows_digest": necessity_summary[
            "necessity_statistic_rows_digest"
        ],
        "necessity_summary_digest": build_stable_digest(
            necessity_summary
        ),
    }
    _write_json(
        directory / "manifest.local.json",
        {
            "artifact_id": "formal_mechanism_ablation_manifest",
            "artifact_type": "local_manifest",
            "code_version": "b" * 40,
            "formal_execution_run_lock": FORMAL_EXECUTION_LOCK,
            "output_paths": [
                (directory / filename).relative_to(root).as_posix()
                for filename in (
                    "runtime_rerun_records.jsonl",
                    "formal_detection_records.jsonl",
                    "per_ablation_frozen_protocols.json",
                    "mechanism_ablation_metrics.csv",
                    "mechanism_pairwise_delta.csv",
                    "mechanism_necessity_statistics.csv",
                    "mechanism_necessity_summary.json",
                    "ablation_component_summary.json",
                    "manifest.local.json",
                )
            ],
            "config": ablation_manifest_config,
            "config_digest": build_stable_digest(ablation_manifest_config),
            "metadata": {
                **ablation_contract,
                **prompt_contract,
                **atom_identity,
                "ablation_necessity_statistics_ready": True,
                "necessity_statistic_rows_digest": necessity_summary[
                    "necessity_statistic_rows_digest"
                ],
                "necessity_component_supported_ablation_ids": necessity_summary[
                    "necessity_component_supported_ablation_ids"
                ],
                "necessity_component_not_supported_ablation_ids": necessity_summary[
                    "necessity_component_not_supported_ablation_ids"
                ],
                "all_mechanism_necessity_components_supported": necessity_summary[
                    "all_mechanism_necessity_components_supported"
                ],
            },
        },
    )
    write_test_scientific_execution_binding(
        repository_root=root,
        artifact_dir=directory,
        artifact_role="runtime_rerun_ablation",
        paper_run_name=PAPER_RUN_NAME,
        profile_id="sd35_method_runtime_gpu",
        summary_file_name="ablation_component_summary.json",
        manifest_file_name="manifest.local.json",
        formal_execution_lock=FORMAL_EXECUTION_LOCK,
        execution_route="semantic_watermark_ablation_session",
    )
    return package_runtime_rerun_ablations(PAPER_RUN_NAME, root=root)


def _prepare_dataset_quality(root: Path) -> Path:
    """构造正式 FID/KID 打包门禁所需的最小正式形状。"""

    directory = root / "outputs" / "dataset_level_quality" / PAPER_RUN_NAME
    _canonical_prompt_records(root)
    paper_run = build_paper_run_config(root)
    repeat_identity = _randomization_repeat_identity(paper_run)
    canonical_ids = canonical_prompt_ids_for_paper_run(
        root_path=root,
        prompt_set=PAPER_RUN_NAME,
        prompt_file=PROMPT_FILES[PAPER_RUN_NAME],
    )
    quality_records: list[dict[str, object]] = []
    resolution_records: list[dict[str, object]] = []
    feature_rows: list[dict[str, object]] = []
    item_identity: list[dict[str, object]] = []
    for index, prompt_id in enumerate(canonical_ids):
        source_path = f"outputs/fixture_images/quality_pair_{index:05d}_source.png"
        comparison_path = (
            f"outputs/fixture_images/quality_pair_{index:05d}_comparison.png"
        )
        source_bytes = f"source-image-{index}".encode("utf-8")
        comparison_bytes = f"comparison-image-{index}".encode("utf-8")
        source_file = root / source_path
        comparison_file = root / comparison_path
        source_file.parent.mkdir(parents=True, exist_ok=True)
        source_file.write_bytes(source_bytes)
        comparison_file.write_bytes(comparison_bytes)
        source_digest = hashlib.sha256(source_bytes).hexdigest()
        comparison_digest = hashlib.sha256(comparison_bytes).hexdigest()
        record_payload = {
            "run_id": f"quality_runtime_{index:05d}",
            "prompt_id": prompt_id,
            "attack_name": "watermark_embedding",
            "image_pair_index": index,
            "image_pair_role": "clean_to_watermarked",
            "source_image_path": source_path,
            "source_image_digest": source_digest,
            "comparison_image_path": comparison_path,
            "comparison_image_digest": comparison_digest,
            "feature_backend": FORMAL_FEATURE_BACKEND,
            "supports_paper_claim": False,
        }
        record_digest = build_stable_digest(record_payload)
        record_id = f"dataset_quality_record_{record_digest[:16]}"
        quality_records.append(
            {
                "dataset_quality_record_id": record_id,
                "dataset_quality_record_digest": record_digest,
                **record_payload,
            }
        )
        for role, image_path, image_digest, value in (
            ("source", source_path, source_digest, float(index)),
            (
                "comparison",
                comparison_path,
                comparison_digest,
                float(index) + 0.25,
            ),
        ):
            resolution_payload = {
                "requested_image_path": image_path,
                "resolved_image_path": image_path,
                "resolved_from_package_path": "",
                "resolution_status": "resolved_existing_image_file",
                "resolved_image_digest": image_digest,
                "materialized_image_input": False,
                "supports_paper_claim": False,
            }
            resolution_digest = build_stable_digest(resolution_payload)
            resolution_records.append(
                {
                    **resolution_payload,
                    "image_resolution_record_digest": resolution_digest,
                    "image_resolution_record_id": (
                        "dataset_quality_image_resolution_"
                        f"{resolution_digest[:16]}"
                    ),
                }
            )
            item_identity.append(
                {
                    "dataset_quality_record_id": record_id,
                    "dataset_quality_image_role": role,
                    "image_path": image_path,
                    "image_digest": image_digest,
                }
            )
            feature_rows.append(
                {
                    "dataset_quality_record_id": record_id,
                    "dataset_quality_image_role": role,
                    "feature_backend": FORMAL_FEATURE_BACKEND,
                    "feature_extractor_id": FORMAL_FEATURE_EXTRACTOR_ID,
                    "feature_dimension": 2048,
                    "image_path": image_path,
                    "image_digest": image_digest,
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
    scientific_unit_id = f"feature_batch_{batch_identity_digest[:16]}"
    batch_provenance = build_test_scientific_unit_provenance(
        scientific_unit_id,
        _inception_batch_config_digest(item_identity),
        formal_execution_lock=FORMAL_EXECUTION_LOCK,
    )
    for row in feature_rows:
        row["scientific_unit_provenance"] = batch_provenance
    _write_jsonl(
        directory / "dataset_quality_image_records.jsonl",
        quality_records,
    )
    _write_jsonl(
        directory / "dataset_quality_formal_feature_records.jsonl",
        feature_rows,
    )
    _write_jsonl(
        directory / "dataset_quality_image_resolution_records.jsonl",
        resolution_records,
    )
    feature_records_path = directory / "dataset_quality_formal_feature_records.jsonl"
    feature_sha256 = path_digest(feature_records_path)
    prompt_digest = build_stable_digest(sorted(canonical_ids))
    provenance_summary = aggregate_scientific_unit_provenance(
        (row["scientific_unit_provenance"] for row in feature_rows),
        expected_reference_count=PROMPT_COUNT * 2,
    )
    coverage = {
        "canonical_prompt_id_digest": prompt_digest,
        "registry_prompt_id_digest": prompt_digest,
        "prompt_registry_exact_set_ready": True,
        "accepted_feature_pair_count": PROMPT_COUNT,
        "missing_feature_pair_count": 0,
        "feature_issue_count": 0,
        "formal_feature_record_count": PROMPT_COUNT * 2,
        "formal_feature_records_sha256": feature_sha256,
        **provenance_summary,
    }
    resolution_contract = {
        "image_resolution_records_digest": build_stable_digest(
            resolution_records
        ),
        "image_resolution_record_count": len(resolution_records),
        "resolved_image_file_count": len(resolution_records),
        "missing_image_file_count": 0,
        "materialized_image_input_count": 0,
        "image_resolution_identity_ready": True,
    }
    metric_protocol = formal_dataset_quality_metric_protocol()
    _write_json(
        directory / "dataset_quality_formal_feature_import_report.json",
        {
            "paper_run_name": PAPER_RUN_NAME,
            "target_fpr": TARGET_FPR,
            "randomization_repeat_identity": repeat_identity,
            "expected_feature_pair_count": PROMPT_COUNT,
            **coverage,
            **resolution_contract,
        },
    )
    (directory / "dataset_quality_metrics.csv").write_text(
        (
            "quality_metric_name,quality_metric_value,metric_status,"
            "paper_metric_name,source_image_count,comparison_image_count,"
            "sample_pair_count\n"
            + "".join(
                f"{metric_name},{metric_value},measured,{metric_name},"
                f"{PROMPT_COUNT},{PROMPT_COUNT},{PROMPT_COUNT}\n"
                for metric_name, metric_value in zip(
                    FORMAL_DATASET_QUALITY_METRIC_NAMES,
                    (1.0, 0.01, 0.0),
                    strict=True,
                )
            )
        ),
        encoding="utf-8",
    )
    _write_json(
        directory / "dataset_quality_summary.json",
        {
            "generated_at": GENERATED_AT,
            "paper_run_name": PAPER_RUN_NAME,
            "target_fpr": TARGET_FPR,
            "randomization_repeat_identity": repeat_identity,
            "expected_prompt_count": PROMPT_COUNT,
            "registry_prompt_count": PROMPT_COUNT,
            "sample_pair_count": PROMPT_COUNT,
            **coverage,
            **resolution_contract,
            "formal_feature_backend_ready": True,
            "formal_sample_scale_ready": True,
            "canonical_formal_feature_extractor_ready": True,
            "scientific_unit_provenance_identity_ready": True,
            "formal_fid_kid_component_ready": True,
            "repeat_component_ready": True,
            "randomization_aggregate_ready": False,
            "supports_paper_claim": False,
            "kid_effective_subset_size": PROMPT_COUNT,
            "formal_metric_protocol": metric_protocol,
            "formal_metric_protocol_digest": metric_protocol[
                "formal_metric_protocol_digest"
            ],
        },
    )
    _write_json(
        directory / "manifest.local.json",
        {
            "artifact_id": "dataset_level_quality_manifest",
            "artifact_type": "local_manifest",
            "code_version": "b" * 40,
            "formal_execution_run_lock": FORMAL_EXECUTION_LOCK,
            "output_paths": [
                (directory / filename).relative_to(root).as_posix()
                for filename in (
                    "dataset_quality_image_records.jsonl",
                    "dataset_quality_image_resolution_records.jsonl",
                    "dataset_quality_formal_feature_records.jsonl",
                    "dataset_quality_formal_feature_import_report.json",
                    "dataset_quality_metrics.csv",
                    "dataset_quality_summary.json",
                    "manifest.local.json",
                )
            ],
            "metadata": {
                "paper_run_name": PAPER_RUN_NAME,
                "target_fpr": TARGET_FPR,
                **coverage,
                **resolution_contract,
            },
            "config": {
                **coverage,
                **resolution_contract,
                "randomization_repeat_identity": repeat_identity,
            },
            "config_digest": build_stable_digest(
                {
                    **coverage,
                    **resolution_contract,
                    "randomization_repeat_identity": repeat_identity,
                }
            ),
        },
    )
    write_test_scientific_execution_binding(
        repository_root=root,
        artifact_dir=directory,
        artifact_role="dataset_level_quality",
        paper_run_name=PAPER_RUN_NAME,
        profile_id="sd35_method_runtime_gpu",
        summary_file_name="dataset_quality_summary.json",
        manifest_file_name="manifest.local.json",
        formal_execution_lock=FORMAL_EXECUTION_LOCK,
        execution_route="semantic_watermark_session",
    )
    return package_dataset_level_quality_outputs(PAPER_RUN_NAME, root=root)


@pytest.mark.quick
def test_summary_and_manifest_gate_reject_float_anchor_type() -> None:
    """结果摘要和 manifest 的嵌套锚点字段必须保留整数类型."""

    record = {
        "attention_alignment_gate": {
            **ATTENTION_ALIGNMENT_GATE,
            "attention_anchor_count": 12.0,
        },
        **ATTENTION_ALIGNMENT_GATE,
    }

    assert _formal_attention_alignment_gate_record_ready(record) is False


@pytest.mark.quick
def test_primary_gpu_package_producers_pass_strict_closure_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """三个主方法上游包应为 run-scoped 且可直接通过精确闭合选择器。"""

    monkeypatch.setenv("SLM_WM_PAPER_RUN_NAME", PAPER_RUN_NAME)
    monkeypatch.delenv("SLM_WM_PROMPT_SET", raising=False)
    monkeypatch.delenv("SLM_WM_PROMPT_FILE", raising=False)
    sentinel = tmp_path / "outside_family.txt"
    sentinel.write_text("不得归档\n", encoding="utf-8")
    archives = (
        _prepare_image_runtime(tmp_path),
        _prepare_ablation(tmp_path),
        _prepare_dataset_quality(tmp_path),
    )
    for archive_path, spec in zip(archives, CLOSURE_PACKAGE_FAMILY_SPECS[:3]):
        candidate = inspect_closure_package(
            archive_path,
            spec=spec,
            paper_run_name=PAPER_RUN_NAME,
            target_fpr=TARGET_FPR,
            randomization_repeat_id="seed_00_key_00",
        )
        assert candidate.package_family == spec.package_family
        assert candidate.randomization_scope == "active_repeat_component"
        assert candidate.randomization_repeat_id == "seed_00_key_00"
        with ZipFile(archive_path) as archive:
            archive_names = set(archive.namelist())
            assert archive_names
            assert all(
                name.startswith("outputs/") and f"/{PAPER_RUN_NAME}/" in name
                for name in archive_names
            )
            assert sentinel.name not in archive_names
            if spec.package_family == "image_only_dataset_runtime":
                snapshot_prefix = (
                    f"outputs/image_only_dataset_runtime/{PAPER_RUN_NAME}/"
                    "prompt_source_snapshot/"
                )
                assert {
                    snapshot_prefix
                    + f"paper_main_{PAPER_RUN_NAME}_prompts.txt",
                    snapshot_prefix + "prompt_selection_manifest.jsonl",
                    snapshot_prefix + "prompt_source_registry.json",
                }.issubset(archive_names)
            assert spec.package_input_manifest_template is not None
            package_input_member = spec.package_input_manifest_template.format(
                paper_run=PAPER_RUN_NAME,
                baseline=spec.baseline_id or "",
            )
            package_input = json.loads(archive.read(package_input_member))
            assert package_input["schema_version"] == 2
            assert package_input["randomization_repeat_identity"] == (
                _randomization_repeat_identity(build_paper_run_config(tmp_path))
            )
            declared_paths = package_input["entry_paths"]
            assert package_input["entry_count"] == len(declared_paths)
            assert set(declared_paths) == archive_names - {package_input_member}
            assert package_input["entry_sha256"] == {
                member_name: hashlib.sha256(archive.read(member_name)).hexdigest()
                for member_name in declared_paths
            }


@pytest.mark.quick
@pytest.mark.parametrize(
    "fixture_mode",
    ("digest_only", "tampered_leaf", "omitted_leaves"),
)
def test_image_runtime_package_requires_rebuilt_unit_content_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fixture_mode: str,
) -> None:
    """打包前必须逐单元重建科学叶子, 不接受伪摘要或被改写叶子。"""

    monkeypatch.setenv("SLM_WM_PAPER_RUN_NAME", PAPER_RUN_NAME)
    with pytest.raises(
        RuntimeError,
        match="科学内容无法从完成包叶子重建|未完整覆盖",
    ):
        _prepare_image_runtime(
            tmp_path,
            digest_only=fixture_mode == "digest_only",
            tamper_unit_leaf=fixture_mode == "tampered_leaf",
            omit_unit_leaves=fixture_mode == "omitted_leaves",
        )


@pytest.mark.quick
def test_primary_package_ignores_stale_file_and_selector_rejects_undeclared_extra(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """打包器不得吸收遗留文件, 选择器也必须拒绝归档后追加的同前缀成员."""

    monkeypatch.setenv("SLM_WM_PAPER_RUN_NAME", PAPER_RUN_NAME)
    archive_path = _prepare_image_runtime(tmp_path)
    archive_path.unlink()
    output_dir = (
        tmp_path
        / "outputs"
        / "image_only_dataset_runtime"
        / PAPER_RUN_NAME
    )
    stale_path = output_dir / "stale_same_prefix.json"
    stale_path.write_text("{}\n", encoding="utf-8")

    archive_path = package_image_only_dataset_runtime(PAPER_RUN_NAME, root=tmp_path)
    spec = CLOSURE_PACKAGE_FAMILY_SPECS[0]
    with ZipFile(archive_path) as archive:
        assert stale_path.relative_to(tmp_path).as_posix() not in archive.namelist()

    undeclared_member = (
        f"outputs/image_only_dataset_runtime/{PAPER_RUN_NAME}/"
        "undeclared_same_prefix.json"
    )
    with ZipFile(archive_path, "a") as archive:
        archive.writestr(undeclared_member, b"{}\n")
    assert spec.package_input_manifest_template is not None
    package_input_path = tmp_path / spec.package_input_manifest_template.format(
        paper_run=PAPER_RUN_NAME,
        baseline=spec.baseline_id or "",
    )
    with pytest.raises(RuntimeError, match="写后成员集合"):
        validate_exact_package_archive(
            archive_path,
            repository_root=tmp_path,
            package_input_manifest_path=package_input_path,
        )
    with pytest.raises(ClosurePackageSelectionError, match="精确成员集合不一致"):
        inspect_closure_package(
            archive_path,
            spec=spec,
            paper_run_name=PAPER_RUN_NAME,
            target_fpr=TARGET_FPR,
            randomization_repeat_id="seed_00_key_00",
        )


@pytest.mark.quick
def test_primary_gpu_package_producers_reject_non_ready_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """上游 summary 未通过时不得先生成可被误选的 ZIP。"""

    monkeypatch.setenv("SLM_WM_PAPER_RUN_NAME", PAPER_RUN_NAME)
    monkeypatch.delenv("SLM_WM_PROMPT_SET", raising=False)
    monkeypatch.delenv("SLM_WM_PROMPT_FILE", raising=False)
    archive_path = _prepare_image_runtime(tmp_path)
    archive_path.unlink()
    summary_path = (
        tmp_path
        / "outputs"
        / "image_only_dataset_runtime"
        / PAPER_RUN_NAME
        / "dataset_runtime_summary.json"
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["repeat_component_ready"] = False
    _write_json(summary_path, summary)
    with pytest.raises(RuntimeError, match="ready 门禁"):
        package_image_only_dataset_runtime(PAPER_RUN_NAME, root=tmp_path)
    assert not tuple(summary_path.parent.glob("image_only_dataset_runtime_package_*.zip"))


@pytest.mark.quick
def test_ablation_and_quality_packages_reject_inexact_scientific_contracts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """消融集合不精确或质量特征缺配对时不得生成新的正式 ZIP。"""

    monkeypatch.setenv("SLM_WM_PAPER_RUN_NAME", PAPER_RUN_NAME)
    ablation_archive = _prepare_ablation(tmp_path)
    ablation_archive.unlink()
    ablation_summary_path = (
        tmp_path
        / "outputs/formal_mechanism_ablation"
        / PAPER_RUN_NAME
        / "ablation_component_summary.json"
    )
    ablation_summary = json.loads(ablation_summary_path.read_text(encoding="utf-8"))
    ablation_summary["actual_ablation_ids"] = list(FORMAL_RUNTIME_RERUN_ABLATION_IDS[:6])
    _write_json(ablation_summary_path, ablation_summary)
    with pytest.raises(RuntimeError, match="精确15项规范"):
        package_runtime_rerun_ablations(PAPER_RUN_NAME, root=tmp_path)

    quality_archive = _prepare_dataset_quality(tmp_path)
    quality_archive.unlink()
    quality_report_path = (
        tmp_path
        / "outputs/dataset_level_quality"
        / PAPER_RUN_NAME
        / "dataset_quality_formal_feature_import_report.json"
    )
    quality_report = json.loads(quality_report_path.read_text(encoding="utf-8"))
    quality_report["accepted_feature_pair_count"] = PROMPT_COUNT - 1
    quality_report["missing_feature_pair_count"] = 1
    _write_json(quality_report_path, quality_report)
    with pytest.raises(RuntimeError, match="精确 Prompt/特征覆盖"):
        package_dataset_level_quality_outputs(PAPER_RUN_NAME, root=tmp_path)


@pytest.mark.quick
def test_dataset_quality_package_rejects_resigned_feature_schema_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """特征文件和上游摘要同步重签也不得绕过逐行正式 schema 复验."""

    monkeypatch.setenv("SLM_WM_PAPER_RUN_NAME", PAPER_RUN_NAME)
    archive_path = _prepare_dataset_quality(tmp_path)
    archive_path.unlink()
    output_dir = (
        tmp_path / "outputs/dataset_level_quality" / PAPER_RUN_NAME
    )
    feature_path = output_dir / "dataset_quality_formal_feature_records.jsonl"
    feature_rows = [
        json.loads(line)
        for line in feature_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    feature_rows[0]["feature_extractor_id"] = "forged_extractor"
    _write_jsonl(feature_path, feature_rows)
    feature_sha256 = path_digest(feature_path)

    summary_path = output_dir / "dataset_quality_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["formal_feature_records_sha256"] = feature_sha256
    _write_json(summary_path, summary)
    report_path = output_dir / "dataset_quality_formal_feature_import_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["formal_feature_records_sha256"] = feature_sha256
    _write_json(report_path, report)
    manifest_path = output_dir / "manifest.local.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["config"]["formal_feature_records_sha256"] = feature_sha256
    manifest["metadata"]["formal_feature_records_sha256"] = feature_sha256
    manifest["config_digest"] = build_stable_digest(manifest["config"])
    _write_json(manifest_path, manifest)

    with pytest.raises(RuntimeError, match="精确 Prompt/特征覆盖"):
        package_dataset_level_quality_outputs(PAPER_RUN_NAME, root=tmp_path)


@pytest.mark.quick
def test_package_removes_archive_when_final_execution_lock_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """归档写出后的 Git 锁漂移必须删除尚未交付的 ZIP."""

    monkeypatch.setenv("SLM_WM_PAPER_RUN_NAME", PAPER_RUN_NAME)
    archive_path = _prepare_image_runtime(tmp_path)
    archive_path.unlink()
    changed_lock = build_test_formal_execution_lock("c" * 40)
    lock_records = iter((FORMAL_EXECUTION_LOCK, changed_lock))
    monkeypatch.setattr(
        repository_environment,
        "require_published_formal_execution_lock",
        lambda _root: dict(next(lock_records)),
    )

    with pytest.raises(repository_environment.FormalExecutionLockError):
        package_image_only_dataset_runtime(PAPER_RUN_NAME, root=tmp_path)

    output_dir = tmp_path / "outputs" / "image_only_dataset_runtime" / PAPER_RUN_NAME
    assert not tuple(output_dir.glob("image_only_dataset_runtime_package_*.zip"))
