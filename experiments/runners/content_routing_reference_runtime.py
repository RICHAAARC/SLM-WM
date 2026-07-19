"""Materialize real content-routing reference observations into candidate bytes."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

from experiments.protocol.content_routing_reference_registry import (
    _strict_json_object,
)
from experiments.protocol.content_routing_reference_registry_payload import (
    assemble_content_routing_reference_registry_payload,
)
from experiments.protocol.formal_randomization import (
    build_canonical_sd35_base_latent,
)
from experiments.protocol.paper_run_config import build_paper_run_config
from experiments.protocol.prompts import (
    PROMPT_FILES,
    build_prompt_records,
    read_prompt_file,
)
from experiments.protocol.splits import apply_split_assignments
from experiments.runners.image_only_dataset_workload import build_method_config
from experiments.runtime import repository_environment
from experiments.runtime.dependency_profiles import require_dependency_profile_ready
from experiments.runtime.diffusion.sd3_pipeline_runtime import load_pipeline
from main.core.digest import build_stable_digest, tensor_content_sha256
from main.methods.content.latent_response import (
    _measure_adjacent_latent_relative_response,
)
from main.methods.content.local_sensitivity import (
    _measure_public_probe_local_sensitivity,
    build_public_probe_identity,
)
from main.methods.content.texture import _measure_gradient_magnitude


__all__ = ["write_content_routing_reference_runtime_outputs"]


_PARTITION_ID = "probe_paper_dev_method_parameter_v1"
_EXPECTED_DEV_INDICES = (30, 31, 32)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_output_root(root: Path, value: str | Path) -> Path:
    requested = Path(value).expanduser()
    resolved = requested.resolve() if requested.is_absolute() else (root / requested).resolve()
    try:
        resolved.relative_to((root / "outputs").resolve())
    except ValueError as exc:
        raise ValueError("reference runtime output must stay under outputs/") from exc
    return resolved


def _method_parameter_prompts(root: Path) -> tuple[Any, ...]:
    paper_run = build_paper_run_config(root)
    if (
        paper_run.run_name != "probe_paper"
        or paper_run.prompt_set != "probe_paper"
        or Path(paper_run.prompt_file) != PROMPT_FILES["probe_paper"]
    ):
        raise RuntimeError("probe_paper prompt owner identity drifted")
    prompt_path = root / paper_run.prompt_file
    records = apply_split_assignments(
        build_prompt_records("probe_paper", read_prompt_file(prompt_path))
    )
    dev = tuple(record for record in records if record.split == "dev")
    if tuple(record.prompt_index for record in dev) != _EXPECTED_DEV_INDICES:
        raise RuntimeError("probe_paper dev partition identity drifted")
    grouped = {name: {record.prompt_id for record in records if record.split == name} for name in ("dev", "calibration", "test")}
    if (
        tuple(len(grouped[name]) for name in ("dev", "calibration", "test"))
        != (3, 33, 34)
        or grouped["dev"] & grouped["calibration"]
        or grouped["dev"] & grouped["test"]
        or grouped["calibration"] & grouped["test"]
    ):
        raise RuntimeError("probe_paper parameter partition is not isolated")
    return dev


def _decode_latent(pipeline: Any, latent: Any) -> Any:
    import torch

    vae_dtype = next(pipeline.vae.parameters()).dtype
    scaled = latent.to(dtype=vae_dtype) / pipeline.vae.config.scaling_factor
    scaled = scaled + pipeline.vae.config.shift_factor
    decoded = pipeline.vae.decode(scaled, return_dict=False)[0]
    image = pipeline.image_processor.postprocess(decoded, output_type="pt")
    if not isinstance(image, torch.Tensor):
        raise RuntimeError("reference VAE decode must return an RGB Tensor")
    return image.to(device=latent.device, dtype=torch.float32)


def _runtime_identity_payload(
    *,
    config: Any,
    pipeline: Any,
    runtime_versions: Mapping[str, Any],
    formal_execution_lock_digest: str,
    dependency_profile_digest: str,
) -> dict[str, Any]:
    scheduler_config = dict(getattr(pipeline.scheduler, "config", {}))
    return {
        "model_id": config.model_id,
        "model_revision": config.model_revision,
        "dependency_profile_digest": dependency_profile_digest,
        "formal_execution_lock_digest": formal_execution_lock_digest,
        "vae_preprocess_identity_digest": build_stable_digest(
            {
                "vae_class": type(pipeline.vae).__qualname__,
                "image_processor_class": type(pipeline.image_processor).__qualname__,
                "scaling_factor": float(pipeline.vae.config.scaling_factor),
                "shift_factor": float(pipeline.vae.config.shift_factor),
                "postprocess_output_type": "pt",
                "sd35_operator_identity": runtime_versions.get("sd35_operator_identity"),
            }
        ),
        "scheduler_identity_digest": build_stable_digest(
            {
                "scheduler_class": type(pipeline.scheduler).__qualname__,
                "scheduler_config": scheduler_config,
                "capture_indices": [9, 10],
            }
        ),
        "content_observation_formula_identity_digest": build_stable_digest(
            {
                "gradient_formula": "rgb_luminance_sobel_gradient_magnitude_v1",
                "latent_response_formula": "adjacent_latent_relative_channel_rms_v1",
                "local_sensitivity_formula": "public_probe_vae_finite_difference_v1",
                "public_probe_identity": build_public_probe_identity(
                    config.model_revision
                ),
            }
        ),
    }


def _generation_identity(
    config: Any,
    prompt_record: Any,
    seed: int,
    canonical_base_latent_identity: Mapping[str, Any],
) -> str:
    return build_stable_digest(
        {
            "generation_identity_role": "real_sd35_reference_member_v1",
            "prompt_id": prompt_record.prompt_id,
            "prompt_text_digest": prompt_record.prompt_digest,
            "generation_seed_random": seed,
            "model_id": config.model_id,
            "model_revision": config.model_revision,
            "negative_prompt": config.negative_prompt,
            "width": config.width,
            "height": config.height,
            "inference_steps": config.inference_steps,
            "guidance_scale": config.guidance_scale,
            "canonical_base_latent_identity": dict(
                canonical_base_latent_identity
            ),
        }
    )


def _observe_member(
    *,
    pipeline: Any,
    config: Any,
    prompt_record: Any,
    seed: int,
) -> tuple[tuple[Any, Any, Any], dict[str, Any]]:
    import torch

    shape = (
        1,
        int(pipeline.transformer.config.in_channels),
        int(config.height) // int(pipeline.vae_scale_factor),
        int(config.width) // int(pipeline.vae_scale_factor),
    )
    latent, base_latent_identity = build_canonical_sd35_base_latent(
        shape=shape,
        generation_seed_random=seed,
        model_id=config.model_id,
        model_revision=config.model_revision,
        device=pipeline._execution_device,
        dtype=pipeline.transformer.dtype,
    )
    z9: Any | None = None
    observed: tuple[Any, Any, Any] | None = None
    decode_count = 0
    probe_decode_count = 0

    def callback(
        pipe: Any,
        step_index: int,
        timestep: Any,
        callback_kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        del timestep
        nonlocal z9, observed, decode_count
        current = callback_kwargs.get("latents")
        if current is None:
            raise RuntimeError("reference callback requires latents")
        if step_index == 9:
            if z9 is not None:
                raise RuntimeError("reference runtime captured index 9 more than once")
            z9 = current.detach().clone()
        elif step_index == 10:
            if z9 is None or observed is not None:
                raise RuntimeError("reference index 10 requires one prior index 9")
            decode_count += 1
            x10 = _decode_latent(pipe, current)

            def decoder(candidate: Any) -> Any:
                nonlocal probe_decode_count
                probe_decode_count += 1
                return _decode_latent(pipe, candidate)

            gradient = _measure_gradient_magnitude(x10.detach().float())
            response = _measure_adjacent_latent_relative_response(
                z9.detach().float(), current.detach().float()
            )
            sensitivity = _measure_public_probe_local_sensitivity(
                current,
                x10,
                decoder,
                build_public_probe_identity(config.model_revision),
            ).local_difference_sensitivity
            observed = tuple(
                value.detach().to(device="cpu", dtype=torch.float32).contiguous()
                for value in (gradient, response, sensitivity)
            )
        return callback_kwargs

    pipeline(
        prompt=prompt_record.prompt_text,
        negative_prompt=config.negative_prompt,
        width=config.width,
        height=config.height,
        num_inference_steps=config.inference_steps,
        guidance_scale=config.guidance_scale,
        latents=latent,
        callback_on_step_end=callback,
        callback_on_step_end_tensor_inputs=["latents"],
        output_type="latent",
    )
    if observed is None or decode_count != 1 or probe_decode_count != 1:
        raise RuntimeError("reference member did not execute one current and one probe decode")
    return observed, base_latent_identity


def _assemble_pass(
    *,
    root: Path,
    config: Any,
    pipeline: Any,
    runtime_versions: Mapping[str, Any],
    prompts: tuple[Any, ...],
    execution_lock_digest: str,
    dependency_profile_digest: str,
) -> tuple[bytes, dict[str, Any]]:
    seeds = [int(config.seed) + int(record.prompt_index) for record in prompts]
    members = [
        _observe_member(
            pipeline=pipeline,
            config=replace(config, prompt=record.prompt_text, seed=seed),
            prompt_record=record,
            seed=seed,
        )
        for record, seed in zip(prompts, seeds, strict=True)
    ]
    triples = [member[0] for member in members]
    generation_digests = [
        _generation_identity(config, record, seed, member[1])
        for record, seed, member in zip(prompts, seeds, members, strict=True)
    ]
    payload = assemble_content_routing_reference_registry_payload(
        method_parameter_partition_id=_PARTITION_ID,
        prompt_projection=[
            {"prompt_id": record.prompt_id, "prompt_text_digest": record.prompt_digest}
            for record in prompts
        ],
        seed_projection_random=seeds,
        generation_input_identity_digests=generation_digests,
        gradient_observations=[triple[0] for triple in triples],
        response_observations=[triple[1] for triple in triples],
        sensitivity_observations=[triple[2] for triple in triples],
        formal_execution_lock_digest=execution_lock_digest,
        dependency_profile_digest=dependency_profile_digest,
        runtime_component_identity_payload=_runtime_identity_payload(
            config=config,
            pipeline=pipeline,
            runtime_versions=runtime_versions,
            formal_execution_lock_digest=execution_lock_digest,
            dependency_profile_digest=dependency_profile_digest,
        ),
    )
    return payload, {
        "generation_input_identity_digests": generation_digests,
        "observation_member_content_sha256": [
            [tensor_content_sha256(value) for value in triple] for triple in triples
        ],
    }


def write_content_routing_reference_runtime_outputs(
    *,
    root: str | Path,
    output_root: str | Path = "outputs/content_routing_reference_runtime",
) -> dict[str, Any]:
    """Run producer plus replay and persist only a qualified candidate under outputs/."""

    root_path = Path(root).resolve()
    resolved_output_root = _resolve_output_root(root_path, output_root)
    if type(os.environ.get("SLM_WM_KEY_MATERIAL")) is not str or not os.environ.get(
        "SLM_WM_KEY_MATERIAL"
    ):
        raise RuntimeError("reference runtime requires explicit SLM_WM_KEY_MATERIAL")
    execution_lock = repository_environment.require_published_formal_execution_lock(
        root_path
    )
    profile = require_dependency_profile_ready("sd35_method_runtime_gpu")
    if not profile.formal_ready:
        raise RuntimeError("reference runtime dependency profile is not ready")
    config = build_method_config(root_path)
    prompts = _method_parameter_prompts(root_path)
    pipeline, runtime_versions = load_pipeline(config)
    first, first_evidence = _assemble_pass(
        root=root_path,
        config=config,
        pipeline=pipeline,
        runtime_versions=runtime_versions,
        prompts=prompts,
        execution_lock_digest=execution_lock["formal_execution_lock_digest"],
        dependency_profile_digest=profile.profile_digest,
    )
    replay, replay_evidence = _assemble_pass(
        root=root_path,
        config=config,
        pipeline=pipeline,
        runtime_versions=runtime_versions,
        prompts=prompts,
        execution_lock_digest=execution_lock["formal_execution_lock_digest"],
        dependency_profile_digest=profile.profile_digest,
    )
    if first != replay or first_evidence != replay_evidence:
        raise RuntimeError("reference producer and independent replay are not identical")
    registry = _strict_json_object(first)
    semantic_digest = registry.get("content_routing_reference_registry_digest")
    if type(semantic_digest) is not str or len(semantic_digest) != 64:
        raise RuntimeError("reference candidate does not contain a semantic digest")
    file_sha256 = hashlib.sha256(first).hexdigest()
    run_dir = resolved_output_root / semantic_digest
    run_dir.mkdir(parents=True, exist_ok=False)
    candidate_path = run_dir / "content_routing_reference_registry.json"
    candidate_path.write_bytes(first)
    qualification = {
        "report_schema": "content_routing_reference_runtime_qualification_v1",
        "schema_version": 1,
        "method_parameter_partition_id": _PARTITION_ID,
        "candidate_path": candidate_path.relative_to(root_path).as_posix(),
        "content_routing_reference_registry_digest": semantic_digest,
        "content_routing_reference_registry_file_sha256": file_sha256,
        "producer_replay_byte_identical": True,
        "producer_evidence": first_evidence,
        "replay_evidence": replay_evidence,
        "qualification_ready": True,
        "supports_paper_claim": False,
    }
    qualification["qualification_digest"] = build_stable_digest(qualification)
    report_path = run_dir / "content_routing_reference_runtime_qualification.json"
    report_path.write_text(
        json.dumps(qualification, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "report_schema": "content_routing_reference_runtime_invocation_v1",
        "schema_version": 1,
        "candidate_path": candidate_path.relative_to(root_path).as_posix(),
        "candidate_file_sha256": _file_sha256(candidate_path),
        "candidate_registry_digest": semantic_digest,
        "qualification_report_path": report_path.relative_to(root_path).as_posix(),
        "qualification_report_sha256": _file_sha256(report_path),
        "qualification_ready": True,
        "supports_paper_claim": False,
    }
