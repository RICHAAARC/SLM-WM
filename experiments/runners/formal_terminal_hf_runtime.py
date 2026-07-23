"""Run a compact formal terminal-HF integration screen on the frozen prompts."""

from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
from typing import Any, Mapping

from PIL import Image

from experiments.protocol.content_routing_reference_quantile import (
    ContentRoutingReferenceScalars,
)
from experiments.protocol.content_survival_observation import (
    CONTENT_SURVIVAL_CLAIM_BOUNDARY,
    CONTENT_SURVIVAL_PROMPT_IDS,
    build_content_survival_observation_roster,
    load_content_survival_observation_protocol,
)
from experiments.protocol.detection_key_identity import (
    REGISTERED_WRONG_KEY_ROLE,
    resolve_detection_key_material_and_identity,
)
from experiments.runners.semantic_watermark_runtime import (
    FinalImageEvidenceGateFailure,
    SemanticWatermarkRuntimeConfig,
    SemanticWatermarkRuntimeContext,
    _encode_image_latent,
    load_completed_semantic_watermark_runtime_result,
    load_semantic_watermark_runtime_context,
    write_semantic_watermark_runtime_outputs,
)
from experiments.runners.terminal_content_carrier_runtime import (
    _score_key_roster,
)
from experiments.runtime.repository_environment import (
    file_digest,
    validate_formal_execution_lock_record,
)
from main.core.digest import build_stable_digest


FORMAL_TERMINAL_HF_RESULT_NAME = "formal_terminal_hf_result.json"
FORMAL_TERMINAL_HF_MANIFEST_NAME = "cell_manifest.json"


def _stable_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    if path.exists() or partial.exists():
        raise FileExistsError(f"formal terminal HF artifact exists: {path}")
    with partial.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial, path)


def _detection_records(root: Path, result: Any) -> tuple[dict[str, Any], ...]:
    path = (root / result.detection_record_path).resolve()
    records = tuple(
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if [record.get("sample_role") for record in records] != [
        "clean_negative",
        "positive_source",
        "wrong_key_negative",
    ]:
        raise RuntimeError("formal terminal HF detections are incomplete")
    if any(
        record.get("metadata", {}).get("method_role") != "hf_tail_only_content"
        for record in records
    ):
        raise RuntimeError("formal terminal HF detection role drifted")
    return records


def _load_completed_cell(prompt_root: Path) -> dict[str, Any] | None:
    manifest_path = prompt_root / FORMAL_TERMINAL_HF_MANIFEST_NAME
    result_path = prompt_root / FORMAL_TERMINAL_HF_RESULT_NAME
    if not manifest_path.exists() and not result_path.exists():
        return None
    if not manifest_path.is_file() or not result_path.is_file():
        raise RuntimeError("formal terminal HF cell is partial")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_payload = {
        key: value for key, value in manifest.items() if key != "manifest_digest"
    }
    if manifest.get("manifest_digest") != build_stable_digest(manifest_payload):
        raise RuntimeError("formal terminal HF cell manifest drifted")
    if manifest.get("result_sha256") != file_digest(result_path):
        raise RuntimeError("formal terminal HF cell result digest drifted")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result_payload = {
        key: value for key, value in result.items() if key != "result_digest"
    }
    if result.get("result_digest") != build_stable_digest(result_payload):
        raise RuntimeError("formal terminal HF cell result identity drifted")
    formal_runtime_complete = result.get("formal_runtime_complete") is True
    formal_paths = (
        result.get("formal_runtime_result_path"),
        result.get("formal_runtime_manifest_path"),
    )
    if formal_runtime_complete != all(bool(path) for path in formal_paths):
        raise RuntimeError("formal terminal HF completion state drifted")
    if not formal_runtime_complete and result.get("scientific_gate_failure") != (
        "final_image_evidence"
    ):
        raise RuntimeError("formal terminal HF incomplete cell lacks scientific gate")
    return result


def _publish_cell(prompt_root: Path, result: Mapping[str, Any]) -> None:
    result_path = prompt_root / FORMAL_TERMINAL_HF_RESULT_NAME
    manifest_path = prompt_root / FORMAL_TERMINAL_HF_MANIFEST_NAME
    _write_bytes_atomic(result_path, _stable_json_bytes(dict(result)))
    manifest_payload = {
        "result_path": result_path.name,
        "result_sha256": file_digest(result_path),
        "formal_runtime_complete": result["formal_runtime_complete"],
        "scientific_gate_failure": result["scientific_gate_failure"],
        "formal_runtime_result_path": result["formal_runtime_result_path"],
        "formal_runtime_result_sha256": result["formal_runtime_result_sha256"],
        "formal_runtime_manifest_path": result["formal_runtime_manifest_path"],
        "formal_runtime_manifest_sha256": result[
            "formal_runtime_manifest_sha256"
        ],
        **CONTENT_SURVIVAL_CLAIM_BOUNDARY,
    }
    manifest = {
        **manifest_payload,
        "manifest_digest": build_stable_digest(manifest_payload),
    }
    _write_bytes_atomic(manifest_path, _stable_json_bytes(manifest))


def run_formal_terminal_hf_screen(
    prompt_configs: Mapping[str, SemanticWatermarkRuntimeConfig],
    *,
    references: ContentRoutingReferenceScalars,
    verified_formal_execution_lock: Mapping[str, Any],
    verified_execution_environment_identity: Mapping[str, Any],
    repository_root: str | Path,
    output_dir: str | Path,
    runtime_context: SemanticWatermarkRuntimeContext | None = None,
) -> dict[str, Any]:
    """Run the actual formal writer/loader and a 32-wrong-key rank screen."""

    root = Path(repository_root).resolve()
    output_root = (root / Path(output_dir)).resolve()
    output_root.relative_to((root / "outputs").resolve())
    prompt_ids = tuple(prompt_configs)
    if prompt_ids not in {
        CONTENT_SURVIVAL_PROMPT_IDS[:1],
        CONTENT_SURVIVAL_PROMPT_IDS,
    }:
        raise ValueError("formal terminal HF screen requires the frozen prompt prefix")
    configs = tuple(prompt_configs[prompt_id] for prompt_id in prompt_ids)
    validate_formal_execution_lock_record(verified_formal_execution_lock)
    if not verified_execution_environment_identity:
        raise ValueError("formal terminal HF screen requires execution identity")
    protocol = load_content_survival_observation_protocol(root)
    roster = build_content_survival_observation_roster(
        configs[0].key_material,
        protocol=protocol,
    )
    wrong_keys = tuple(roster["wrong_keys"])
    model_identity_digest = build_stable_digest(
        {
            "model_id": configs[0].model_id,
            "model_revision": configs[0].model_revision,
        }
    )
    context = runtime_context or load_semantic_watermark_runtime_context(
        configs[0],
        verified_formal_execution_lock=verified_formal_execution_lock,
        repository_root=root,
    )
    prompt_results: list[dict[str, Any]] = []
    for config in configs:
        prompt_root = output_root / config.prompt_id
        existing = _load_completed_cell(prompt_root)
        if existing is not None:
            prompt_results.append(existing)
            continue
        runtime_output = prompt_root / "runtime"
        runtime_config = replace(
            config,
            output_dir=runtime_output.relative_to(root).as_posix(),
        )
        completed = load_completed_semantic_watermark_runtime_result(
            runtime_config,
            root,
        )
        diagnostic_gate_failure: FinalImageEvidenceGateFailure | None = None
        if completed is None:
            try:
                write_semantic_watermark_runtime_outputs(
                    runtime_config,
                    root,
                    references=references,
                    verified_formal_execution_lock=verified_formal_execution_lock,
                    runtime_context=context,
                )
            except FinalImageEvidenceGateFailure as exc:
                diagnostic_gate_failure = exc
        if diagnostic_gate_failure is None:
            completed = load_completed_semantic_watermark_runtime_result(
                runtime_config,
                root,
            )
            if completed is None:
                raise RuntimeError("formal terminal HF writer output did not reload")
            detections = _detection_records(root, completed)
            image_path = (root / completed.watermarked_image_path).resolve()
            with Image.open(image_path) as source_image:
                image = source_image.convert("RGB").copy()
            runtime_metadata = completed.metadata
            carrier_image_path = (
                root / runtime_metadata["carrier_only_image_path"]
            ).resolve()
            with Image.open(carrier_image_path) as source_image:
                carrier_image = source_image.convert("RGB").copy()
            runtime_run_id = completed.run_id
            runtime_result_path = (
                root
                / runtime_config.output_dir
                / completed.run_id
                / "runtime_result.json"
            ).resolve()
            runtime_manifest_path = Path(root / completed.manifest_path).resolve()
            formal_runtime_result_path = runtime_result_path.relative_to(
                root
            ).as_posix()
            formal_runtime_result_sha256 = file_digest(runtime_result_path)
            formal_runtime_manifest_path = runtime_manifest_path.relative_to(
                root
            ).as_posix()
            formal_runtime_manifest_sha256 = file_digest(runtime_manifest_path)
            scientific_gate_failure = None
            scientific_gate_failure_reasons: list[str] = []
            formal_runtime_complete = True
        else:
            (
                failed_runtime,
                _,
                _,
                failed_detections,
                _,
                image,
                carrier_image,
                _,
            ) = diagnostic_gate_failure.runtime_outputs
            if failed_runtime.run_decision != "fail":
                raise RuntimeError("final-image gate failure reported a passing runtime")
            detections = tuple(failed_detections)
            runtime_metadata = failed_runtime.metadata
            runtime_run_id = failed_runtime.run_id
            formal_runtime_result_path = ""
            formal_runtime_result_sha256 = ""
            formal_runtime_manifest_path = ""
            formal_runtime_manifest_sha256 = ""
            scientific_gate_failure = "final_image_evidence"
            scientific_gate_failure_reasons = list(
                diagnostic_gate_failure.failure_reasons
            )
            formal_runtime_complete = False
        if [record.get("sample_role") for record in detections] != [
            "clean_negative",
            "positive_source",
            "wrong_key_negative",
        ]:
            raise RuntimeError("formal terminal HF detections are incomplete")
        positive = detections[1]
        fixed_wrong = detections[2]
        fixed_wrong_margin = float(positive["content_score"]) - float(
            fixed_wrong["content_score"]
        )
        reencoded = _encode_image_latent(context.pipeline, image)
        multi_key_scores = _score_key_roster(
            reencoded,
            registered_key_material=config.key_material,
            wrong_keys=wrong_keys,
            model_identity_digest=model_identity_digest,
            carrier_mode="hf_only",
        )
        carrier_reencoded = _encode_image_latent(
            context.pipeline,
            carrier_image,
        )
        carrier_multi_key_scores = _score_key_roster(
            carrier_reencoded,
            registered_key_material=config.key_material,
            wrong_keys=wrong_keys,
            model_identity_digest=model_identity_digest,
            carrier_mode="hf_only",
        )
        fixed_wrong_key, fixed_wrong_identity = (
            resolve_detection_key_material_and_identity(
                config.key_material,
                REGISTERED_WRONG_KEY_ROLE,
            )
        )
        carrier_fixed_wrong_scores = _score_key_roster(
            carrier_reencoded,
            registered_key_material=config.key_material,
            wrong_keys=(
                {
                    "wrong_key_index": 0,
                    "wrong_key_material": fixed_wrong_key,
                    "wrong_key_material_digest_random": (
                        fixed_wrong_identity[
                            "detection_key_material_digest_random"
                        ]
                    ),
                },
            ),
            model_identity_digest=model_identity_digest,
            carrier_mode="hf_only",
        )
        carrier_fixed_records = carrier_fixed_wrong_scores[
            "key_score_records"
        ]
        carrier_fixed_wrong_margin = float(
            carrier_fixed_records[0]["blind_content_score"]
        ) - float(carrier_fixed_records[1]["blind_content_score"])
        result_payload = {
            "result_schema": "slm_wm_formal_terminal_hf_screen",
            "schema_version": 1,
            "prompt_id": config.prompt_id,
            "run_id": runtime_run_id,
            "formal_runtime_complete": formal_runtime_complete,
            "scientific_gate_failure": scientific_gate_failure,
            "scientific_gate_failure_reasons": scientific_gate_failure_reasons,
            "formal_runtime": runtime_metadata["method_runtime"],
            "formal_attribution_carrier": runtime_metadata[
                "formal_attribution_carrier"
            ],
            "formal_attribution_strength_multiplier": runtime_metadata[
                "formal_attribution_strength_multiplier"
            ],
            "formal_fixed_wrong_key_margin": fixed_wrong_margin,
            "formal_fixed_wrong_key_pass": fixed_wrong_margin > 0.0,
            "multi_key_scores": multi_key_scores,
            "registered_rank": multi_key_scores["rank_record"]["registered_rank"],
            "registered_rank_one": (
                multi_key_scores["rank_record"]["registered_rank"] == 1
            ),
            "carrier_only_multi_key_scores": carrier_multi_key_scores,
            "carrier_only_registered_rank": carrier_multi_key_scores[
                "rank_record"
            ]["registered_rank"],
            "carrier_only_registered_rank_one": (
                carrier_multi_key_scores["rank_record"]["registered_rank"] == 1
            ),
            "carrier_only_fixed_wrong_key_margin": carrier_fixed_wrong_margin,
            "carrier_only_fixed_wrong_key_pass": (
                carrier_fixed_wrong_margin > 0.0
            ),
            "hf_attribution_views": {
                "full_combined": "multi_key_scores",
                "carrier_only": "carrier_only_multi_key_scores",
                "qk_geometry_gain_source": (
                    "final_image_attention_observability"
                ),
            },
            "paired_quality": runtime_metadata["paired_quality"],
            "final_image_preservation": runtime_metadata[
                "final_image_preservation"
            ],
            "carrier_only_final_image_preservation": runtime_metadata[
                "carrier_only_final_image_preservation"
            ],
            "final_image_attention_observability": runtime_metadata[
                "final_image_attention_observability"
            ],
            "formal_runtime_result_path": formal_runtime_result_path,
            "formal_runtime_result_sha256": formal_runtime_result_sha256,
            "formal_runtime_manifest_path": formal_runtime_manifest_path,
            "formal_runtime_manifest_sha256": formal_runtime_manifest_sha256,
            "verified_execution_environment_identity_digest": build_stable_digest(
                dict(verified_execution_environment_identity)
            ),
            **CONTENT_SURVIVAL_CLAIM_BOUNDARY,
        }
        result = {
            **result_payload,
            "result_digest": build_stable_digest(result_payload),
        }
        _publish_cell(prompt_root, result)
        prompt_results.append(result)
    rank_one_count = sum(
        result.get("registered_rank_one") is True for result in prompt_results
    )
    fixed_wrong_pass_count = sum(
        result.get("formal_fixed_wrong_key_pass") is True
        for result in prompt_results
    )
    positive_max_wrong_margin_count = sum(
        result.get("multi_key_scores", {})
        .get("rank_record", {})
        .get("registered_minus_max_wrong_margin", 0.0)
        > 0.0
        for result in prompt_results
    )
    carrier_rank_one_count = sum(
        result.get("carrier_only_registered_rank_one") is True
        for result in prompt_results
    )
    carrier_fixed_wrong_pass_count = sum(
        result.get("carrier_only_fixed_wrong_key_pass") is True
        for result in prompt_results
    )
    carrier_positive_max_wrong_margin_count = sum(
        result.get("carrier_only_multi_key_scores", {})
        .get("rank_record", {})
        .get("registered_minus_max_wrong_margin", 0.0)
        > 0.0
        for result in prompt_results
    )
    final_image_evidence_pass_count = sum(
        result.get("formal_runtime_complete") is True
        for result in prompt_results
    )
    return {
        "decision": "pass",
        "method_screening_decision": (
            "pass"
            if rank_one_count == len(prompt_ids)
            and fixed_wrong_pass_count == len(prompt_ids)
            and positive_max_wrong_margin_count == len(prompt_ids)
            and carrier_rank_one_count == len(prompt_ids)
            and carrier_fixed_wrong_pass_count == len(prompt_ids)
            and carrier_positive_max_wrong_margin_count == len(prompt_ids)
            else "fail"
        ),
        "prompt_ids": list(prompt_ids),
        "prompt_count": len(prompt_results),
        "complete_cell_count": len(prompt_results),
        "diffusion_chain_count": len(prompt_results) * 7,
        "key_score_count": len(prompt_results) * 66,
        "registered_rank_one_count": rank_one_count,
        "formal_fixed_wrong_key_pass_count": fixed_wrong_pass_count,
        "carrier_only_registered_rank_one_count": carrier_rank_one_count,
        "carrier_only_fixed_wrong_key_pass_count": (
            carrier_fixed_wrong_pass_count
        ),
        "final_image_evidence_pass_count": final_image_evidence_pass_count,
        "scientific_gate_failure_count": (
            len(prompt_results) - final_image_evidence_pass_count
        ),
        "prompt_results": prompt_results,
        **CONTENT_SURVIVAL_CLAIM_BOUNDARY,
    }


__all__ = [
    "FORMAL_TERMINAL_HF_MANIFEST_NAME",
    "FORMAL_TERMINAL_HF_RESULT_NAME",
    "run_formal_terminal_hf_screen",
]
