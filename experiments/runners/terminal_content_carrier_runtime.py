"""Run the compact terminal-latent keyed-carrier observation."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
from typing import Any, Mapping

from experiments.protocol.content_routing_reference_quantile import (
    ContentRoutingReferenceScalars,
)
from experiments.protocol.content_survival_observation import (
    CONTENT_SURVIVAL_CARRIER_MODES,
    CONTENT_SURVIVAL_CLAIM_BOUNDARY,
    CONTENT_SURVIVAL_PROMPT_IDS,
    build_content_survival_observation_roster,
    build_registered_rank_record,
    compute_blind_observation_score,
    load_content_survival_observation_protocol,
)
from experiments.protocol.formal_randomization import build_canonical_sd35_base_latent
from experiments.runners.semantic_watermark_runtime import (
    SemanticWatermarkRuntimeConfig,
    SemanticWatermarkRuntimeContext,
    _decode_content_runtime_latent,
    _encode_image_latent,
    _require_full_content_runtime_config,
    load_semantic_watermark_runtime_context,
)
from experiments.runtime.image_metrics import compute_image_quality_metrics
from experiments.runtime.repository_environment import validate_formal_execution_lock_record
from main.core.digest import build_stable_digest, tensor_content_sha256
from main.core.keyed_prg import KEYED_PRG_VERSION
from main.methods.carrier.high_frequency_tail import build_high_frequency_tail_template
from main.methods.carrier.low_frequency import build_low_frequency_template
from main.methods.carrier.terminal_update import (
    TERMINAL_CONTENT_STRENGTH_MULTIPLIERS,
    build_terminal_content_carrier_update,
)
from main.methods.content.local_sensitivity import build_public_probe_identity
from main.methods.content.runtime_adapter import build_content_observation_routing


TERMINAL_CONTENT_ROUTING_MODES = ("semantic_unit_energy", "uniform")
_TERMINAL_CONTENT_VARIANTS_PER_PROMPT = (
    len(TERMINAL_CONTENT_ROUTING_MODES)
    * len(CONTENT_SURVIVAL_CARRIER_MODES)
    * len(TERMINAL_CONTENT_STRENGTH_MULTIPLIERS)
)
TERMINAL_CONTENT_DIFFUSION_CHAIN_COUNT = len(CONTENT_SURVIVAL_PROMPT_IDS)
TERMINAL_CONTENT_VARIANT_COUNT = (
    TERMINAL_CONTENT_DIFFUSION_CHAIN_COUNT
    * _TERMINAL_CONTENT_VARIANTS_PER_PROMPT
)
TERMINAL_CONTENT_KEY_SCORE_COUNT = (
    TERMINAL_CONTENT_VARIANT_COUNT
    * 2
    * 33
)


@dataclass(frozen=True)
class _TerminalReference:
    terminal_latent: Any
    image_reencoded_latent: Any
    image: Any
    routing: Any
    z9_content_sha256: str
    z10_content_sha256: str


def _stable_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    if partial.exists():
        partial.unlink()
    with partial.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _image_png_bytes(image: Any) -> bytes:
    import torch
    from PIL import Image

    pixels = image.detach().float().cpu()
    if pixels.ndim != 4 or tuple(pixels.shape[:2]) != (1, 3):
        raise ValueError("terminal observation image must have [1,3,H,W] shape")
    rgb = (
        pixels[0]
        .clamp(0.0, 1.0)
        .mul(255.0)
        .round()
        .to(dtype=torch.uint8)
        .permute(1, 2, 0)
        .numpy()
    )
    payload = BytesIO()
    Image.fromarray(rgb).save(payload, format="PNG")
    return payload.getvalue()


def _paired_quality(reference_image: Any, candidate_image: Any) -> dict[str, Any]:
    from PIL import Image

    with Image.open(BytesIO(_image_png_bytes(reference_image))) as reference:
        reference.load()
        with Image.open(BytesIO(_image_png_bytes(candidate_image))) as candidate:
            candidate.load()
            return compute_image_quality_metrics(reference, candidate)


def _extract_terminal_latent(output: Any) -> Any:
    import torch

    value = getattr(output, "images", None)
    if isinstance(value, torch.Tensor) and value.ndim == 4:
        return value
    if isinstance(value, (tuple, list)) and len(value) == 1:
        item = value[0]
        if isinstance(item, torch.Tensor) and item.ndim == 4:
            return item
    raise RuntimeError("pipeline did not return one terminal latent")


def _run_terminal_reference(
    config: SemanticWatermarkRuntimeConfig,
    references: ContentRoutingReferenceScalars,
    *,
    context: SemanticWatermarkRuntimeContext,
    base_latent: Any,
) -> _TerminalReference:
    captured_z9: Any | None = None
    captured_z10: Any | None = None
    routing: Any | None = None
    callback_count = 0
    pipeline = context.pipeline
    public_probe_identity = build_public_probe_identity(config.model_revision)

    def callback(
        pipe: Any,
        step_index: int,
        _timestep: Any,
        callback_kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        nonlocal captured_z9, captured_z10, routing, callback_count
        latent = callback_kwargs.get("latents")
        if latent is None:
            raise RuntimeError("terminal observation callback requires latents")
        if step_index == 9:
            captured_z9 = latent.detach().clone()
            return callback_kwargs
        if step_index != 10:
            return callback_kwargs
        if captured_z9 is None or callback_count:
            raise RuntimeError("terminal observation requires one z9 and one z10")
        callback_count += 1
        captured_z10 = latent.detach().clone()
        decoded = _decode_content_runtime_latent(pipe, latent)
        probe_decode_count = 0

        def vae_decoder(candidate: Any) -> Any:
            nonlocal probe_decode_count
            probe_decode_count += 1
            return _decode_content_runtime_latent(pipe, candidate)

        observations = build_content_observation_routing(
            previous_scheduler_latent=captured_z9,
            current_scheduler_latent=latent,
            decoded_current_image=decoded,
            prompt=config.prompt,
            saliency_runtime=context.prompt_saliency_runtime,
            vae_decoder=vae_decoder,
            public_probe_identity=public_probe_identity,
            reference_gradient=references.reference_gradient,
            reference_response=references.reference_response,
            reference_sensitivity=references.reference_sensitivity,
        )
        if probe_decode_count != 1:
            raise RuntimeError("terminal observation public probe decode drifted")
        routing = observations.routing
        return callback_kwargs

    output = pipeline(
        prompt=config.prompt,
        negative_prompt=config.negative_prompt,
        width=config.width,
        height=config.height,
        num_inference_steps=config.inference_steps,
        guidance_scale=config.guidance_scale,
        output_type="latent",
        latents=base_latent.detach().clone(),
        callback_on_step_end=callback,
        callback_on_step_end_tensor_inputs=["latents"],
    )
    if callback_count != 1 or captured_z9 is None or captured_z10 is None or routing is None:
        raise RuntimeError("terminal observation did not capture its routing reference")
    terminal_latent = _extract_terminal_latent(output)
    image = _decode_content_runtime_latent(pipeline, terminal_latent)
    reencoded = _encode_image_latent(pipeline, image)
    return _TerminalReference(
        terminal_latent=terminal_latent,
        image_reencoded_latent=reencoded,
        image=image,
        routing=routing,
        z9_content_sha256=tensor_content_sha256(captured_z9),
        z10_content_sha256=tensor_content_sha256(captured_z10),
    )


def _score_method_role(carrier_mode: str) -> str:
    if carrier_mode == "lf_only":
        return "lf_only_content"
    if carrier_mode == "hf_only":
        return "hf_tail_only_content"
    return "full_dual_chain"


def _score_key_roster(
    latent: Any,
    *,
    registered_key_material: str,
    wrong_keys: tuple[Mapping[str, Any], ...],
    model_identity_digest: str,
    carrier_mode: str,
) -> dict[str, Any]:
    keys = (
        {
            "key_role": "registered",
            "key_index": None,
            "key_material": registered_key_material,
            "key_material_digest_random": build_stable_digest(
                {"key_material": registered_key_material}
            ),
        },
        *(
            {
                "key_role": "wrong",
                "key_index": record["wrong_key_index"],
                "key_material": record["wrong_key_material"],
                "key_material_digest_random": record[
                    "wrong_key_material_digest_random"
                ],
            }
            for record in wrong_keys
        ),
    )
    records = []
    for key in keys:
        score = compute_blind_observation_score(
            latent,
            key_material=key["key_material"],
            model_identity_digest=model_identity_digest,
            prg_version=KEYED_PRG_VERSION,
            method_role=_score_method_role(carrier_mode),
        )
        records.append(
            {
                "key_role": key["key_role"],
                "key_index": key["key_index"],
                "key_material_digest_random": key["key_material_digest_random"],
                "blind_lf_score": score["blind_lf_score"],
                "blind_hf_tail_score": score["blind_hf_tail_score"],
                "blind_content_score": score["blind_content_score"],
                "score_identity_digest": score["score_identity_digest"],
            }
        )
    rank = build_registered_rank_record(
        float(records[0]["blind_content_score"]),
        [float(record["blind_content_score"]) for record in records[1:]],
    )
    return {
        "latent_content_sha256": tensor_content_sha256(latent),
        "key_score_records": records,
        "rank_record": rank,
    }


def _load_prompt_result(prompt_root: Path) -> dict[str, Any] | None:
    manifest_path = prompt_root / "cell_manifest.json"
    result_path = prompt_root / "terminal_content_carrier_result.json"
    if not manifest_path.is_file() and not result_path.exists():
        return None
    if not manifest_path.is_file() or not result_path.is_file():
        raise RuntimeError("terminal prompt result is partial")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    claimed_manifest_digest = manifest.get("manifest_digest")
    manifest_payload = {
        key: value for key, value in manifest.items() if key != "manifest_digest"
    }
    if claimed_manifest_digest != build_stable_digest(manifest_payload):
        raise RuntimeError("terminal prompt manifest identity drifted")
    if manifest.get("result_sha256") != _sha256(result_path):
        raise RuntimeError("terminal prompt result digest drifted")
    image_files = manifest.get("image_files")
    if type(image_files) is not list:
        raise RuntimeError("terminal prompt image manifest is invalid")
    for record in image_files:
        if type(record) is not dict or set(record) != {"path", "sha256"}:
            raise RuntimeError("terminal prompt image record is invalid")
        image_path = prompt_root / str(record["path"])
        if not image_path.is_file() or record["sha256"] != _sha256(image_path):
            raise RuntimeError("terminal prompt image digest drifted")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("result_digest") != build_stable_digest(
        {key: value for key, value in result.items() if key != "result_digest"}
    ):
        raise RuntimeError("terminal prompt result identity drifted")
    return result


def _publish_prompt_result(
    prompt_root: Path,
    result: Mapping[str, Any],
    images: Mapping[str, bytes],
) -> None:
    prompt_root.mkdir(parents=True, exist_ok=True)
    for name, payload in images.items():
        _write_bytes_atomic(prompt_root / name, payload)
    result_path = prompt_root / "terminal_content_carrier_result.json"
    _write_bytes_atomic(result_path, _stable_json_bytes(result))
    manifest = {
        "result_sha256": _sha256(result_path),
        "image_files": [
            {"path": name, "sha256": _sha256(prompt_root / name)}
            for name in sorted(images)
        ],
    }
    manifest["manifest_digest"] = build_stable_digest(manifest)
    _write_bytes_atomic(
        prompt_root / "cell_manifest.json",
        _stable_json_bytes(manifest),
    )


def run_terminal_content_carrier_observation(
    prompt_configs: Mapping[str, SemanticWatermarkRuntimeConfig],
    *,
    references: ContentRoutingReferenceScalars,
    verified_formal_execution_lock: Mapping[str, Any],
    verified_execution_environment_identity: Mapping[str, Any],
    repository_root: str | Path,
    output_dir: str | Path,
    runtime_context: SemanticWatermarkRuntimeContext | None = None,
) -> dict[str, Any]:
    """Run four diffusion chains and test terminal carrier strength offline."""

    root = Path(repository_root).resolve()
    output_root = (root / Path(output_dir)).resolve()
    output_root.relative_to((root / "outputs").resolve())
    if tuple(prompt_configs) != CONTENT_SURVIVAL_PROMPT_IDS:
        raise ValueError("terminal observation requires the fixed four prompts")
    configs = tuple(prompt_configs[prompt_id] for prompt_id in CONTENT_SURVIVAL_PROMPT_IDS)
    for config in configs:
        _require_full_content_runtime_config(config)
    validate_formal_execution_lock_record(verified_formal_execution_lock)
    if not verified_execution_environment_identity:
        raise ValueError("terminal observation requires an execution environment identity")
    protocol = load_content_survival_observation_protocol(root)
    roster = build_content_survival_observation_roster(
        configs[0].key_material,
        protocol=protocol,
    )
    wrong_keys = tuple(roster["wrong_keys"])
    model_identity_digest = build_stable_digest(
        {"model_id": configs[0].model_id, "model_revision": configs[0].model_revision}
    )
    context = runtime_context or load_semantic_watermark_runtime_context(
        configs[0],
        verified_formal_execution_lock=verified_formal_execution_lock,
        repository_root=root,
    )
    prompt_results = []
    for config in configs:
        prompt_root = output_root / config.prompt_id
        existing = _load_prompt_result(prompt_root)
        if existing is not None:
            prompt_results.append(existing)
            continue
        pipeline = context.pipeline
        shape = (
            1,
            int(pipeline.transformer.config.in_channels),
            int(config.height) // int(pipeline.vae_scale_factor),
            int(config.width) // int(pipeline.vae_scale_factor),
        )
        base_latent, base_identity = build_canonical_sd35_base_latent(
            shape=shape,
            generation_seed_random=int(config.seed),
            model_id=config.model_id,
            model_revision=config.model_revision,
            device=pipeline._execution_device,
            dtype=pipeline.transformer.dtype,
        )
        reference = _run_terminal_reference(
            config,
            references,
            context=context,
            base_latent=base_latent,
        )
        clean_image_bytes = _image_png_bytes(reference.image)
        images: dict[str, bytes] = {"clean.png": clean_image_bytes}
        variants = []
        for routing_mode in TERMINAL_CONTENT_ROUTING_MODES:
            for carrier_mode in CONTENT_SURVIVAL_CARRIER_MODES:
                for multiplier in TERMINAL_CONTENT_STRENGTH_MULTIPLIERS:
                    lf_template = build_low_frequency_template(
                        reference.terminal_latent,
                        config.key_material,
                        model_identity_digest,
                        prg_version=KEYED_PRG_VERSION,
                    )
                    hf_template = build_high_frequency_tail_template(
                        reference.terminal_latent,
                        config.key_material,
                        model_identity_digest,
                        prg_version=KEYED_PRG_VERSION,
                    )
                    update = build_terminal_content_carrier_update(
                        reference.terminal_latent,
                        reference.routing,
                        lf_template,
                        hf_template,
                        routing_mode=routing_mode,
                        carrier_mode=carrier_mode,
                        strength_multiplier=float(multiplier),
                    )
                    image = _decode_content_runtime_latent(
                        pipeline,
                        update.written_latent,
                    )
                    reencoded = _encode_image_latent(pipeline, image)
                    variant_id = (
                        f"{routing_mode}__{carrier_mode}__x{int(multiplier)}"
                    )
                    image_name = f"{variant_id}.png"
                    images[image_name] = _image_png_bytes(image)
                    payload = {
                        "variant_id": variant_id,
                        "routing_mode": routing_mode,
                        "carrier_mode": carrier_mode,
                        "strength_multiplier": float(multiplier),
                        "lf_effective_l2": update.lf_effective_l2,
                        "hf_tail_effective_l2": update.hf_tail_effective_l2,
                        "combined_effective_l2": update.combined_effective_l2,
                        "combined_relative_l2": update.combined_relative_l2,
                        "lf_update_content_sha256": tensor_content_sha256(update.lf_update),
                        "hf_tail_update_content_sha256": tensor_content_sha256(
                            update.hf_tail_update
                        ),
                        "terminal_scores": _score_key_roster(
                            update.written_latent,
                            registered_key_material=config.key_material,
                            wrong_keys=wrong_keys,
                            model_identity_digest=model_identity_digest,
                            carrier_mode=carrier_mode,
                        ),
                        "reencoded_scores": _score_key_roster(
                            reencoded,
                            registered_key_material=config.key_material,
                            wrong_keys=wrong_keys,
                            model_identity_digest=model_identity_digest,
                            carrier_mode=carrier_mode,
                        ),
                        "paired_quality": _paired_quality(
                            reference.image,
                            image,
                        ),
                        "image_file": image_name,
                        "image_sha256": hashlib.sha256(images[image_name]).hexdigest(),
                        **CONTENT_SURVIVAL_CLAIM_BOUNDARY,
                    }
                    payload["variant_digest"] = build_stable_digest(payload)
                    variants.append(payload)
        result = {
            "result_schema": "slm_wm_terminal_content_carrier_observation",
            "schema_version": 1,
            "prompt_id": config.prompt_id,
            "prompt_text_digest": build_stable_digest({"prompt": config.prompt}),
            "base_latent_identity_digest_random": base_identity[
                "base_latent_identity_digest_random"
            ],
            "z9_content_sha256": reference.z9_content_sha256,
            "z10_content_sha256": reference.z10_content_sha256,
            "clean_terminal_latent_content_sha256": tensor_content_sha256(
                reference.terminal_latent
            ),
            "clean_reencoded_latent_content_sha256": tensor_content_sha256(
                reference.image_reencoded_latent
            ),
            "routing_identity_digest": reference.routing.routing_identity_digest,
            "variant_count": len(variants),
            "variants": variants,
            **CONTENT_SURVIVAL_CLAIM_BOUNDARY,
        }
        result["result_digest"] = build_stable_digest(result)
        _publish_prompt_result(prompt_root, result, images)
        prompt_results.append(result)

    summary = {
        "summary_schema": "slm_wm_terminal_content_carrier_observation_summary",
        "schema_version": 1,
        "prompt_count": len(prompt_results),
        "diffusion_chain_count": TERMINAL_CONTENT_DIFFUSION_CHAIN_COUNT,
        "variant_count": sum(int(result["variant_count"]) for result in prompt_results),
        "key_score_count": TERMINAL_CONTENT_KEY_SCORE_COUNT,
        "prompt_result_digests": [result["result_digest"] for result in prompt_results],
        "method_change_tested": "terminal_pre_vae_fixed_energy_keyed_carrier",
        **CONTENT_SURVIVAL_CLAIM_BOUNDARY,
    }
    summary["summary_digest"] = build_stable_digest(summary)
    _write_bytes_atomic(output_root / "observation_summary.json", _stable_json_bytes(summary))
    return summary


__all__ = [
    "TERMINAL_CONTENT_DIFFUSION_CHAIN_COUNT",
    "TERMINAL_CONTENT_KEY_SCORE_COUNT",
    "TERMINAL_CONTENT_ROUTING_MODES",
    "TERMINAL_CONTENT_VARIANT_COUNT",
    "run_terminal_content_carrier_observation",
]
