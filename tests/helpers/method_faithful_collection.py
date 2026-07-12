"""构造 method-faithful exact-set 测试输入。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from experiments.protocol.attacks import attack_config_digest, default_attack_configs
from experiments.protocol.fixed_fpr_observation_audit import (
    audit_fixed_fpr_observation_threshold,
    conformal_threshold_from_clean_negative_scores,
)
from experiments.protocol.paper_run_config import build_paper_run_config
from experiments.protocol.prompts import build_prompt_records, read_prompt_file
from experiments.protocol.splits import apply_split_assignments
from main.core.digest import build_stable_digest
from paper_experiments.baselines.method_faithful_observation_collection import (
    FORMAL_MODEL_ID,
    FORMAL_MODEL_REVISION,
    METHOD_FAITHFUL_BASELINE_IDS,
    MethodFaithfulCollectionProtocol,
    build_method_faithful_collection_protocol,
    canonical_prompt_protocol_digest,
    file_sha256,
    observation_relative_path,
    transfer_manifest_relative_path,
)
from paper_experiments.baselines.method_faithful_numerical_fidelity import (
    METHOD_FAITHFUL_NUMERICAL_FIDELITY_SCHEMA,
)
from tests.helpers.formal_prompt_source import copy_governed_prompt_file


def write_json(path: Path, payload: Any) -> None:
    """写出稳定 JSON 测试输入。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def numerical_fidelity_report(baseline_id: str) -> dict[str, Any]:
    """构造覆盖登记算子集合且可重建门禁的数值忠实度测试记录."""

    operator_ids = {
        "tree_ring": (
            "tree_ring.mask",
            "tree_ring.ring_key",
            "tree_ring.fourier_injection",
            "tree_ring.negative_l1_detection_score",
        ),
        "gaussian_shading": (
            "gaussian_shading.chacha20_ietf_cipher",
            "gaussian_shading.block_voting",
            "gaussian_shading.conditional_gaussian_sign_mapping",
        ),
        "shallow_diffuse": (
            "shallow_diffuse.ring_mask",
            "shallow_diffuse.complex_random_patch",
            "shallow_diffuse.fourier_injection",
            "shallow_diffuse.negative_l1_detection_score",
            "shallow_diffuse.edit_timestep_floor",
        ),
    }[baseline_id]
    reference_mode = {
        "tree_ring": "executed_official_commit_operator_equivalence",
        "gaussian_shading": (
            "official_source_bound_rfc8439_and_operator_equivalence"
        ),
        "shallow_diffuse": "executed_official_commit_operator_equivalence",
    }[baseline_id]
    records = []
    for operator_id in operator_ids:
        comparison = {
            "operator_id": operator_id,
            "reference_origin": "executed_official_commit_operator",
            "comparison_mode": "exact_tensor",
            "reference_dtype": "torch.float32",
            "adapter_dtype": "torch.float32",
            "reference_shape": [1],
            "adapter_shape": [1],
            "element_count": 1,
            "absolute_tolerance": 0.0,
            "max_absolute_error": 0.0,
            "exact_match": True,
            "reference_value_digest": "1" * 64,
            "adapter_value_digest": "1" * 64,
            "numerical_fidelity_ready": True,
        }
        if operator_id == "gaussian_shading.chacha20_ietf_cipher":
            comparison["official_source_cipher_contract_ready"] = True
        if operator_id == "shallow_diffuse.edit_timestep_floor":
            comparison["official_edit_timestep_formula_ast_digest"] = "2" * 64
        comparison["comparison_record_digest"] = build_stable_digest(comparison)
        records.append(comparison)
    payload = {
        "report_schema": METHOD_FAITHFUL_NUMERICAL_FIDELITY_SCHEMA,
        "baseline_id": baseline_id,
        "numerical_fidelity_reference_mode": reference_mode,
        "official_repository_commit": "a" * 40,
        "official_source_read_mode": "immutable_git_commit_blob",
        "official_source_file": f"external_baseline/{baseline_id}/source.py",
        "official_source_blob_sha256": "3" * 64,
        "official_operator_ast_digest": "4" * 64,
        "adapter_file": f"external_baseline/{baseline_id}/adapter.py",
        "adapter_file_sha256": "5" * 64,
        "execution_device": "cpu",
        "torch_version": "test",
        "numpy_version": "test",
        "operator_ids": list(operator_ids),
        "operator_record_count": len(records),
        "operator_records": records,
        "operator_records_digest": build_stable_digest(records),
        "method_faithful_numerical_fidelity_ready": True,
        "supports_paper_claim": False,
    }
    if baseline_id == "shallow_diffuse":
        payload.update(
            {
                "official_entrypoint_blob_sha256": "6" * 64,
                "official_edit_timestep_formula_ast_digest": "2" * 64,
            }
        )
    payload["numerical_fidelity_report_digest"] = build_stable_digest(payload)
    return payload


def prompt_rows(prompt_set: str, splits: Iterable[str]) -> list[dict[str, Any]]:
    """构造带连续索引的规范 Prompt 计划。"""

    rows: list[dict[str, Any]] = []
    for prompt_index, split in enumerate(splits):
        prompt_text = f"formal prompt {prompt_index}"
        rows.append(
            {
                "prompt_id": f"prompt_{prompt_index:04d}",
                "prompt_index": prompt_index,
                "prompt_set": prompt_set,
                "split": split,
                "prompt_text": prompt_text,
                "prompt_digest": f"{prompt_index + 1:064x}",
            }
        )
    return rows


def collection_protocol(
    rows: list[dict[str, Any]],
    *,
    paper_run_name: str = "probe_paper",
    target_fpr: float = 0.1,
) -> MethodFaithfulCollectionProtocol:
    """根据测试 Prompt 计划构造 expected protocol。"""

    return MethodFaithfulCollectionProtocol(
        paper_run_name=paper_run_name,
        prompt_set=paper_run_name,
        prompt_count=len(rows),
        prompt_protocol_digest=canonical_prompt_protocol_digest(rows),
        target_fpr=target_fpr,
    )


def write_current_paper_protocol(
    root_path: Path,
    *,
    paper_run_name: str = "probe_paper",
) -> tuple[list[dict[str, Any]], MethodFaithfulCollectionProtocol]:
    """写出当前论文层级 Prompt 文件并返回其真实规范协议。"""

    copy_governed_prompt_file(root_path, paper_run_name)
    paper_run = build_paper_run_config(root_path)
    records = apply_split_assignments(
        build_prompt_records(paper_run.prompt_set, read_prompt_file(root_path / paper_run.prompt_file))
    )
    rows = [
        {
            "prompt_id": record.prompt_id,
            "prompt_index": record.prompt_index,
            "prompt_set": record.prompt_set,
            "split": record.split,
            "prompt_text": record.prompt_text,
            "prompt_digest": record.prompt_digest,
        }
        for record in records
    ]
    return rows, build_method_faithful_collection_protocol(root_path)


def formal_observation_rows(
    baseline_id: str,
    prompts: list[dict[str, Any]],
    protocol: MethodFaithfulCollectionProtocol,
) -> list[dict[str, Any]]:
    """构造覆盖规范 Prompt、固定 FPR 和完整攻击集合的 observation。"""

    calibration_scores = [
        0.05 + prompt_index * 0.001
        for prompt_index, row in enumerate(prompts)
        if row["split"] == "calibration"
    ]
    threshold = conformal_threshold_from_clean_negative_scores(
        calibration_scores,
        protocol.target_fpr,
    )
    rows: list[dict[str, Any]] = []
    for prompt_index, prompt in enumerate(prompts):
        for sample_role, score in (("clean_negative", 0.05 + prompt_index * 0.001), ("positive_source", 0.95)):
            rows.append(
                {
                    "event_id": f"{baseline_id}_{prompt['prompt_id']}_{sample_role}",
                    "baseline_id": baseline_id,
                    "prompt_id": prompt["prompt_id"],
                    "prompt_text": prompt["prompt_text"],
                    "split": prompt["split"],
                    "attack_family": "clean",
                    "attack_name": "clean_none",
                    "attack_condition": "clean_none",
                    "sample_role": sample_role,
                    "score": score,
                    "threshold": threshold,
                    "score_name": "baseline_detection_score",
                    "higher_is_positive": True,
                    "threshold_source": "calibration_clean_negative_conformal",
                    "detection_decision": score >= threshold,
                    "adapter_boundary": "method_faithful_sd35_adapter_reproduction",
                    "generation_model_id": protocol.model_id,
                    "generation_model_revision": protocol.model_revision,
                    "execution_device": "cuda",
                    "latent_shape": [1, 16, 64, 64],
                    "image_id": f"{baseline_id}_{prompt['prompt_id']}_{sample_role}",
                    "image_path": f"outputs/{baseline_id}/{prompt['prompt_id']}_{sample_role}.png",
                    "image_digest": f"{prompt_index + 1:064x}",
                    "formal_result_claim": False,
                    "supports_paper_claim": False,
                }
            )
        if prompt["split"] != "test":
            continue
        for attack in default_attack_configs():
            if not attack.enabled or attack.resource_profile not in {"full_main", "full_extra"}:
                continue
            for sample_role, score in (("attacked_negative", 0.04), ("attacked_positive", 0.90)):
                rows.append(
                    {
                        "event_id": f"{baseline_id}_{prompt['prompt_id']}_{attack.attack_name}_{sample_role}",
                        "baseline_id": baseline_id,
                        "prompt_id": prompt["prompt_id"],
                        "prompt_text": prompt["prompt_text"],
                        "split": "test",
                        "attack_family": attack.attack_family,
                        "attack_name": attack.attack_name,
                        "attack_condition": attack.attack_name,
                        "attack_id": attack.attack_id,
                        "resource_profile": attack.resource_profile,
                        "attack_config_digest": attack_config_digest(attack),
                        "sample_role": sample_role,
                        "score": score,
                        "threshold": threshold,
                        "score_name": "baseline_detection_score",
                        "higher_is_positive": True,
                        "threshold_source": "calibration_clean_negative_conformal",
                        "detection_decision": score >= threshold,
                        "adapter_boundary": "method_faithful_sd35_adapter_reproduction",
                        "generation_model_id": protocol.model_id,
                        "generation_model_revision": protocol.model_revision,
                        "execution_device": "cuda",
                        "latent_shape": [1, 16, 64, 64],
                        "image_id": f"{baseline_id}_{prompt['prompt_id']}_{attack.attack_name}_{sample_role}",
                        "image_path": (
                            f"outputs/{baseline_id}/{prompt['prompt_id']}_{attack.attack_name}_{sample_role}.png"
                        ),
                        "image_digest": f"{prompt_index + 1:064x}",
                        "formal_result_claim": False,
                        "supports_paper_claim": False,
                    }
                )
    return rows


def write_collection_source(
    collection_root: Path,
    baseline_id: str,
    observations: list[dict[str, Any]],
    prompts: list[dict[str, Any]],
    protocol: MethodFaithfulCollectionProtocol,
    *,
    manifest_overrides: dict[str, Any] | None = None,
) -> tuple[Path, Path]:
    """写出单个 baseline 的 observation、三类绑定 manifest 和 transfer。"""

    observations_path = collection_root / Path(*observation_relative_path(baseline_id).parts)
    transfer_path = collection_root / Path(*transfer_manifest_relative_path(baseline_id).parts)
    command_path = collection_root / "split_observations" / f"{baseline_id}_baseline_command_results.json"
    run_dir = collection_root / "run_records" / baseline_id
    prompt_path = run_dir / f"{baseline_id}_prompt_plan.json"
    adapter_path = run_dir / f"{baseline_id}_adapter_manifest.json"
    execution_path = run_dir / f"{baseline_id}_execution_manifest.json"
    numerical_fidelity_path = run_dir / f"{baseline_id}_numerical_fidelity_report.json"
    write_json(observations_path, observations)
    write_json(
        command_path,
        [
            {
                "baseline_id": baseline_id,
                "return_code": 0,
                "observation_count": len(observations),
                "output_path": observations_path.as_posix(),
            }
        ],
    )
    write_json(prompt_path, prompts)
    generation_protocol = {
        "model_id": FORMAL_MODEL_ID,
        "model_revision": FORMAL_MODEL_REVISION,
        "num_inference_steps": protocol.num_inference_steps,
        "guidance_scale": protocol.guidance_scale,
        "height": 512,
        "width": 512,
    }
    detection_protocol = {
        "input_access_mode": "image_only",
        "num_inversion_steps": protocol.num_inversion_steps,
        "target_fpr": protocol.target_fpr,
    }
    threshold_audit = audit_fixed_fpr_observation_threshold(
        observations,
        target_fpr=protocol.target_fpr,
        expected_calibration_negative_count=sum(row["split"] == "calibration" for row in prompts),
    )
    write_json(
        adapter_path,
        {
            "artifact_name": f"{baseline_id}_method_faithful_sd35_adapter_manifest.json",
            "baseline_id": baseline_id,
            "adapter_status": "method_faithful_sd35_adapter_ready",
            "adapter_boundary": "method_faithful_sd35_adapter_reproduction",
            "model_id": FORMAL_MODEL_ID,
            "model_revision": FORMAL_MODEL_REVISION,
            "observation_count": len(observations),
            "generation_protocol": generation_protocol,
            "detection_protocol": detection_protocol,
        },
    )
    write_json(
        execution_path,
        {
            "artifact_name": "baseline_execution_manifest.json",
            "baseline_ids": [baseline_id],
            "command_count": 1,
            "failed_command_count": 0,
            "observation_count": len(observations),
        },
    )
    fidelity_report = numerical_fidelity_report(baseline_id)
    write_json(numerical_fidelity_path, fidelity_report)
    transfer = {
        "artifact_name": f"{baseline_id}_baseline_transfer_manifest.json",
        "baseline_id": baseline_id,
        "baseline_observations_path": observation_relative_path(baseline_id).as_posix(),
        "baseline_observation_count": len(observations),
        "baseline_observations_sha256": file_sha256(observations_path),
        "baseline_command_results_path": command_path.relative_to(collection_root).as_posix(),
        "baseline_command_results_sha256": file_sha256(command_path),
        "prompt_plan_path": prompt_path.relative_to(collection_root).as_posix(),
        "prompt_plan_sha256": file_sha256(prompt_path),
        "prompt_protocol_digest": canonical_prompt_protocol_digest(prompts),
        "adapter_manifest_path": adapter_path.relative_to(collection_root).as_posix(),
        "adapter_manifest_sha256": file_sha256(adapter_path),
        "execution_manifest_path": execution_path.relative_to(collection_root).as_posix(),
        "execution_manifest_sha256": file_sha256(execution_path),
        "numerical_fidelity_report_path": numerical_fidelity_path.relative_to(
            collection_root
        ).as_posix(),
        "numerical_fidelity_report_sha256": file_sha256(numerical_fidelity_path),
        "numerical_fidelity_report_digest": fidelity_report[
            "numerical_fidelity_report_digest"
        ],
        "numerical_fidelity_reference_mode": fidelity_report[
            "numerical_fidelity_reference_mode"
        ],
        "method_faithful_numerical_fidelity_ready": True,
        "paper_run_name": protocol.paper_run_name,
        "prompt_set": protocol.prompt_set,
        "prompt_count": protocol.prompt_count,
        "model_id": FORMAL_MODEL_ID,
        "model_revision": FORMAL_MODEL_REVISION,
        "target_fpr": protocol.target_fpr,
        "generation_protocol": generation_protocol,
        "detection_protocol": detection_protocol,
        "formal_attack_names": sorted(
            attack.attack_name
            for attack in default_attack_configs()
            if attack.enabled and attack.resource_profile in {"full_main", "full_extra"}
        ),
        "threshold": threshold_audit.frozen_threshold,
        "threshold_digest": threshold_audit.threshold_digest,
        "transfer_ready": True,
    }
    transfer.update(manifest_overrides or {})
    write_json(transfer_path, transfer)
    return observations_path, transfer_path


def write_complete_collection(
    collection_root: Path,
    observations_by_baseline: dict[str, list[dict[str, Any]]],
    prompts: list[dict[str, Any]],
    protocol: MethodFaithfulCollectionProtocol,
) -> None:
    """写出三个 baseline 的完整 exact-set collection。"""

    for baseline_id in METHOD_FAITHFUL_BASELINE_IDS:
        write_collection_source(
            collection_root,
            baseline_id,
            observations_by_baseline[baseline_id],
            prompts,
            protocol,
        )
