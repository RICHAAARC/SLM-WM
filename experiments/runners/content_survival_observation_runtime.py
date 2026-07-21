"""Run the isolated GPU observation that localizes content-carrier loss."""

from __future__ import annotations

from dataclasses import dataclass, replace
from io import BytesIO
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping

from experiments.protocol.content_routing_reference_quantile import (
    ContentRoutingReferenceScalars,
)
from experiments.protocol.content_survival_direction import (
    load_content_survival_direction_protocol,
    materialize_content_survival_probe,
)
from experiments.protocol.content_survival_observation import (
    CONTENT_SURVIVAL_CARRIER_MODES,
    CONTENT_SURVIVAL_CHAIN_COUNT,
    CONTENT_SURVIVAL_CHAIN_ROLES,
    CONTENT_SURVIVAL_CLAIM_BOUNDARY,
    CONTENT_SURVIVAL_EVALUATION_COUNT,
    CONTENT_SURVIVAL_GEOMETRY_FAILURE_CODE,
    CONTENT_SURVIVAL_M0_DEVIATIONS,
    CONTENT_SURVIVAL_OBSERVATION_ROLES,
    CONTENT_SURVIVAL_PROMPT_IDS,
    CONTENT_SURVIVAL_ROUTING_MODES,
    build_content_survival_cell_identity,
    build_content_survival_causal_diagnostic,
    build_content_survival_observation_roster,
    build_content_survival_observation_run_identity,
    build_content_survival_observation_summary,
    build_content_survival_observation_attempt_summary,
    build_content_survival_cell_failure_record,
    build_content_survival_parent_child_binding_digest,
    build_registered_rank_record,
    compute_blind_observation_score,
    compute_routed_template_oracle_score,
    load_content_survival_observation_protocol,
    publish_content_survival_cell,
    publish_content_survival_cell_failure,
    select_content_survival_observation_sign,
    validate_content_survival_cell_bundle,
    validate_content_survival_cell_failure_record,
    validate_content_survival_observation_record,
)


_GEOMETRY_FAILURE_MESSAGE = "9项几何回溯均未产生actual-dtype严格关系分数改善"
from experiments.protocol.formal_randomization import (
    build_canonical_sd35_base_latent,
)
from experiments.runners.semantic_watermark_runtime import (
    SemanticWatermarkRuntimeConfig,
    SemanticWatermarkRuntimeContext,
    _build_attention_geometry_sync_update_with_evidence,
    _content_runtime_prompt_embeddings,
    _decode_content_runtime_latent,
    _encode_image_latent,
    _require_full_content_runtime_config,
    _transformer_forward_function,
    load_semantic_watermark_runtime_context,
    semantic_watermark_runtime_config_digest,
)
from experiments.runtime.scientific_content_binding import (
    canonical_rgb_uint8_content_record,
)
from experiments.runtime.repository_environment import (
    validate_formal_execution_lock_record,
)
from main.core.digest import build_stable_digest, tensor_content_sha256
from main.core.keyed_prg import KEYED_PRG_VERSION
from main.methods.carrier.content_update import build_content_carrier_update
from main.methods.carrier.high_frequency_tail import (
    build_high_frequency_tail_template,
)
from main.methods.carrier.low_frequency import build_low_frequency_template
from main.methods.content.local_sensitivity import build_public_probe_identity
from main.methods.content.runtime_adapter import build_content_observation_routing
from main.methods.geometry import DifferentiableAttentionRecorder
from main.methods.update_composition import (
    compose_dual_chain_update_once,
    formal_dual_chain_write_budget,
)


@dataclass(frozen=True)
class _ObservedChain:
    role: str
    pre_write_z10: Any
    post_write_z10: Any
    pre_vae_terminal_latent: Any
    image_reencoded_latent: Any
    image: Any
    routing: Any | None
    chain_record: dict[str, Any]
    lf_update: Any | None
    hf_update: Any | None
    replay_sign: int | None


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


def _publish_summary(output_root: Path, summary: Mapping[str, Any]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / "observation_summary.json"
    payload = _stable_json_bytes(summary)
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError("existing observation summary conflicts with this run")
        return
    partial = output_root / "observation_summary.partial.json"
    if partial.exists():
        partial.unlink()
    with partial.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial, path)
    descriptor = os.open(output_root, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _tensor_bytes(value: Any) -> bytes:
    import torch

    payload = BytesIO()
    torch.save(value.detach().cpu(), payload)
    return payload.getvalue()


def _image_png_bytes(image: Any) -> bytes:
    import torch
    from PIL import Image

    if not isinstance(image, torch.Tensor):
        raise TypeError("decoded observation image must be a Tensor")
    pixels = image.detach().float().cpu()
    if pixels.ndim != 4 or pixels.shape[0] != 1 or pixels.shape[1] != 3:
        raise ValueError("decoded observation image must have [1,3,H,W] shape")
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


def _canonical_image_identity(image: Any) -> dict[str, Any]:
    from PIL import Image

    with Image.open(BytesIO(_image_png_bytes(image))) as canonical_image:
        canonical_image.load()
        return canonical_rgb_uint8_content_record(canonical_image)


def _extract_terminal_latent(output: Any) -> Any:
    import torch

    latent = getattr(output, "images", None)
    if isinstance(latent, torch.Tensor) and latent.ndim == 4:
        return latent.detach().clone()
    if (
        isinstance(latent, (list, tuple))
        and len(latent) == 1
        and isinstance(latent[0], torch.Tensor)
    ):
        value = latent[0]
        return value.detach().clone() if value.ndim == 4 else value.unsqueeze(0).detach().clone()
    raise RuntimeError("latent pipeline output did not contain one [1,C,H,W] Tensor")


def _content_method_role(routing_mode: str, carrier_mode: str) -> str:
    if routing_mode == "uniform":
        return "uniform_content_routing"
    return {
        "lf_only": "lf_only_content",
        "hf_only": "hf_tail_only_content",
        "dual": "full_dual_chain",
    }[carrier_mode]


def _score_method_role(routing_mode: str, carrier_mode: str) -> str:
    if carrier_mode == "lf_only":
        return "lf_only_content"
    if carrier_mode == "hf_only":
        return "hf_tail_only_content"
    return "uniform_content_routing" if routing_mode == "uniform" else "full_dual_chain"


def _active_content_updates(content_update: Any, carrier_mode: str) -> tuple[Any, Any]:
    import torch

    lf_update = content_update.lf_update
    hf_update = content_update.hf_tail_update
    if carrier_mode == "lf_only":
        hf_update = torch.zeros_like(hf_update)
    elif carrier_mode == "hf_only":
        lf_update = torch.zeros_like(lf_update)
    return lf_update, hf_update


def _restrict_content_update(content_update: Any, carrier_mode: str, z10: Any) -> Any:
    """Keep a uniform-routing ablation exact without changing its producer."""

    import torch

    lf_update, hf_update = _active_content_updates(content_update, carrier_mode)
    return replace(
        content_update,
        lf_direction=(
            content_update.lf_direction
            if carrier_mode != "hf_only"
            else torch.zeros_like(content_update.lf_direction)
        ),
        hf_tail_direction=(
            content_update.hf_tail_direction
            if carrier_mode != "lf_only"
            else torch.zeros_like(content_update.hf_tail_direction)
        ),
        lf_update=lf_update,
        hf_tail_update=hf_update,
        content_only_latent_float32=z10.detach().float() + lf_update + hf_update,
        method_role=(
            "lf_only_content"
            if carrier_mode == "lf_only"
            else (
                "hf_tail_only_content"
                if carrier_mode == "hf_only"
                else content_update.method_role
            )
        ),
    )


def _composition_method_role(routing_mode: str, carrier_mode: str) -> str:
    if carrier_mode == "lf_only":
        return "lf_only_content"
    if carrier_mode == "hf_only":
        return "hf_tail_only_content"
    return "uniform_content_routing" if routing_mode == "uniform" else "full_dual_chain"


def _run_observed_chain(
    config: SemanticWatermarkRuntimeConfig,
    references: ContentRoutingReferenceScalars,
    *,
    context: SemanticWatermarkRuntimeContext,
    base_latent: Any,
    base_identity: Mapping[str, Any],
    routing_mode: str,
    carrier_mode: str,
    role: str,
    probe_sign: int | None,
    replay_sign: int | None,
    observation_run_identity_digest: str,
    prompt_text_digest: str,
    prompt_config_digest: str,
    key_roster_digest_random: str,
    cell_identity: Mapping[str, Any] | None,
) -> _ObservedChain:
    """Execute one fresh scheduler fork and retain the three latent observations."""

    import torch

    if role not in (*CONTENT_SURVIVAL_CHAIN_ROLES, "clean_reference"):
        raise ValueError("observation chain role is not governed")
    if role != "clean_reference" and cell_identity is None:
        raise ValueError("content observation chain requires its cell identity")
    pipeline = context.pipeline
    prompt_embeds, pooled_prompt_embeds = _content_runtime_prompt_embeddings(
        pipeline,
        config.prompt,
    )
    model_identity_digest = build_stable_digest(
        {"model_id": config.model_id, "model_revision": config.model_revision}
    )
    public_probe_identity = build_public_probe_identity(config.model_revision)
    direction_protocol = load_content_survival_direction_protocol(
        Path(__file__).resolve().parents[2]
    )
    captured_z9: Any | None = None
    pre_write_z10: Any | None = None
    post_write_z10: Any | None = None
    routing: Any | None = None
    lf_update: Any | None = None
    hf_update: Any | None = None
    callback_record: dict[str, Any] = {}
    callback_count = 0

    def callback(
        pipe: Any,
        step_index: int,
        timestep: Any,
        callback_kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        nonlocal captured_z9, pre_write_z10, post_write_z10, routing, lf_update, hf_update
        nonlocal callback_record, callback_count
        latent = callback_kwargs.get("latents")
        if latent is None:
            raise RuntimeError("observation callback requires latents")
        if step_index == 9:
            if captured_z9 is not None:
                raise RuntimeError("observation chain captured z9 twice")
            captured_z9 = latent.detach().clone()
            return callback_kwargs
        if step_index != 10:
            return callback_kwargs
        if captured_z9 is None or callback_count:
            raise RuntimeError("observation chain requires one z9 and one z10")
        callback_count += 1
        z10 = latent
        pre_write_z10 = z10.detach().clone()
        callback_record = {
            "scheduler_step_index": 10,
            "scheduler_step_timestep": float(timestep.detach().float().item()),
            "z9_content_sha256": tensor_content_sha256(captured_z9),
            "z10_before_write_content_sha256": tensor_content_sha256(z10),
        }
        if role == "clean_reference":
            post_write_z10 = z10.detach().clone()
            callback_kwargs["latents"] = post_write_z10
            return callback_kwargs

        decoded_x10 = _decode_content_runtime_latent(pipe, z10)
        decode_count = 0

        def vae_decoder(candidate: Any) -> Any:
            nonlocal decode_count
            decode_count += 1
            if decode_count != 1:
                raise RuntimeError("routing permits exactly one public probe decode")
            return _decode_content_runtime_latent(pipe, candidate)

        observations = build_content_observation_routing(
            previous_scheduler_latent=captured_z9,
            current_scheduler_latent=z10,
            decoded_current_image=decoded_x10,
            prompt=config.prompt,
            saliency_runtime=context.prompt_saliency_runtime,
            vae_decoder=vae_decoder,
            public_probe_identity=public_probe_identity,
            reference_gradient=references.reference_gradient,
            reference_response=references.reference_response,
            reference_sensitivity=references.reference_sensitivity,
        )
        if decode_count != 1:
            raise RuntimeError("routing public probe decode was not consumed")
        routing = observations.routing
        lf_template = build_low_frequency_template(
            z10,
            config.key_material,
            model_identity_digest,
            prg_version=KEYED_PRG_VERSION,
        )
        hf_template = build_high_frequency_tail_template(
            z10,
            config.key_material,
            model_identity_digest,
            prg_version=KEYED_PRG_VERSION,
        )
        content_update = build_content_carrier_update(
            current_scheduler_latent=z10,
            routing=routing,
            lf_template=lf_template,
            hf_tail_template=hf_template,
            method_role=_content_method_role(routing_mode, carrier_mode),
            content_strength_common_multiplier=1.0,
        )
        content_update = _restrict_content_update(content_update, carrier_mode, z10)
        lf_update, hf_update = _active_content_updates(content_update, carrier_mode)
        full_role = role.startswith("full_") or role.startswith("nominal_")
        geometry = None
        if full_role:
            transformer_forward = _transformer_forward_function(
                pipe,
                timestep,
                prompt_embeds,
                pooled_prompt_embeds,
            )
            recorder = DifferentiableAttentionRecorder(
                context.attention_modules,
                max_tokens=config.max_attention_tokens,
            )
            try:
                geometry, _ = _build_attention_geometry_sync_update_with_evidence(
                    current_scheduler_latent=z10,
                    content_update=content_update,
                    transformer_forward=transformer_forward,
                    recorder=recorder,
                    key_material=config.key_material,
                    prg_version=KEYED_PRG_VERSION,
                )
            finally:
                recorder.close()
        geometry_update = (
            geometry.geometry_update
            if geometry is not None
            else torch.zeros_like(z10, dtype=torch.float32)
        )
        raw_direction = lf_update + hf_update + geometry_update
        callback_record.update(
            {
                "routing_identity_digest": routing.routing_identity_digest,
                "registered_direction_content_sha256": tensor_content_sha256(raw_direction),
                "lf_update_content_sha256": tensor_content_sha256(lf_update),
                "hf_update_content_sha256": tensor_content_sha256(hf_update),
                "geometry_update_content_sha256": tensor_content_sha256(geometry_update),
                "attention_geometry_enabled": full_role,
            }
        )
        if probe_sign is not None:
            written, probe_record = materialize_content_survival_probe(
                z10,
                raw_direction,
                sign=probe_sign,
                role=role,
                protocol=direction_protocol,
            )
            post_write_z10 = written.detach().clone()
            callback_kwargs["latents"] = written
            callback_record.update(probe_record)
            return callback_kwargs
        if replay_sign not in (-1, 1):
            raise RuntimeError("nominal observation requires a frozen sign")
        write_result = compose_dual_chain_update_once(
            z10,
            lf_update * float(replay_sign),
            hf_update * float(replay_sign),
            geometry_update * float(replay_sign),
            formal_dual_chain_write_budget(),
            method_role=_composition_method_role(routing_mode, carrier_mode),
        )
        post_write_z10 = write_result.written_latent.detach().clone()
        callback_kwargs["latents"] = write_result.written_latent
        callback_record.update(
            {
                "selected_sign": replay_sign,
                "common_gamma": float(write_result.accepted_common_scale),
                "write_identity_digest": write_result.write_identity_digest,
                "actual_dtype_single_write_digest": (
                    write_result.actual_dtype_write_digest
                ),
            }
        )
        return callback_kwargs

    common_kwargs = {
        "prompt": config.prompt,
        "negative_prompt": config.negative_prompt,
        "width": config.width,
        "height": config.height,
        "num_inference_steps": config.inference_steps,
        "guidance_scale": config.guidance_scale,
        "output_type": "latent",
    }
    output = pipeline(
        latents=base_latent.detach().clone(),
        callback_on_step_end=callback,
        callback_on_step_end_tensor_inputs=["latents"],
        **common_kwargs,
    )
    if callback_count != 1 or pre_write_z10 is None or post_write_z10 is None:
        raise RuntimeError("observation chain did not materialize post-write z10")
    terminal_latent = _extract_terminal_latent(output)
    image = _decode_content_runtime_latent(pipeline, terminal_latent)
    reencoded = _encode_image_latent(pipeline, image)
    scheduler_identity = {
        "scheduler_class": (
            f"{type(pipeline.scheduler).__module__}."
            f"{type(pipeline.scheduler).__qualname__}"
        ),
        "scheduler_config": dict(pipeline.scheduler.config),
    }
    scheduler_digest = build_stable_digest(scheduler_identity)
    image_identity = _canonical_image_identity(image)
    parent_payload = {
        "role": role,
        "routing_mode": routing_mode,
        "carrier_mode": carrier_mode,
        "base_latent_identity": dict(base_identity),
        "base_latent_identity_digest_random": base_identity[
            "base_latent_identity_digest_random"
        ],
        "scheduler_identity": scheduler_identity,
        "scheduler_identity_digest": scheduler_digest,
        "observation_run_identity_digest": observation_run_identity_digest,
        "prompt_text_digest": prompt_text_digest,
        "prompt_config_digest": prompt_config_digest,
        "key_roster_digest_random": key_roster_digest_random,
        "post_write_z10_content_sha256": tensor_content_sha256(post_write_z10),
        "pre_vae_terminal_latent_content_sha256": tensor_content_sha256(
            terminal_latent
        ),
        "image_rgb_uint8_content_sha256": image_identity[
            "image_rgb_uint8_content_sha256"
        ],
        "image_rgb_uint8_content_schema": image_identity[
            "image_rgb_uint8_content_schema"
        ],
        "image_width": image_identity["image_width"],
        "image_height": image_identity["image_height"],
        "image_reencoded_latent_content_sha256": tensor_content_sha256(reencoded),
        "callback_record": callback_record,
        **CONTENT_SURVIVAL_CLAIM_BOUNDARY,
    }
    if role != "clean_reference":
        assert cell_identity is not None
        parent_payload.update(
            {
                "cell_identity_digest": cell_identity["cell_identity_digest"],
                "protocol_semantic_digest": cell_identity[
                    "protocol_semantic_digest"
                ],
                "prompt_roster_semantic_digest": cell_identity[
                    "prompt_roster_semantic_digest"
                ],
                "prompt_roster_artifact_file_sha256": cell_identity[
                    "prompt_roster_artifact_file_sha256"
                ],
            }
        )
    chain_record = {
        **parent_payload,
        "parent_chain_digest": build_stable_digest(parent_payload),
    }
    return _ObservedChain(
        role=role,
        pre_write_z10=pre_write_z10,
        post_write_z10=post_write_z10,
        pre_vae_terminal_latent=terminal_latent,
        image_reencoded_latent=reencoded,
        image=image,
        routing=routing,
        chain_record=chain_record,
        lf_update=lf_update,
        hf_update=hf_update,
        replay_sign=replay_sign if replay_sign is not None else probe_sign,
    )


def _observation_tensors(chain: _ObservedChain) -> dict[str, Any]:
    return {
        "post_write_z10": chain.post_write_z10,
        "pre_vae_terminal_latent": chain.pre_vae_terminal_latent,
        "image_reencoded_latent": chain.image_reencoded_latent,
    }


def _routed_templates_for_key(
    chain: _ObservedChain,
    *,
    key_material: str,
    model_identity_digest: str,
    routing_mode: str,
    carrier_mode: str,
) -> tuple[Any, Any]:
    import torch

    if chain.routing is None:
        raise ValueError("oracle template requires a routed chain")
    z10 = chain.pre_write_z10
    lf_template = build_low_frequency_template(
        z10,
        key_material,
        model_identity_digest,
        prg_version=KEYED_PRG_VERSION,
    )
    hf_template = build_high_frequency_tail_template(
        z10,
        key_material,
        model_identity_digest,
        prg_version=KEYED_PRG_VERSION,
    )
    update = build_content_carrier_update(
        current_scheduler_latent=z10,
        routing=chain.routing,
        lf_template=lf_template,
        hf_tail_template=hf_template,
        method_role=_content_method_role(routing_mode, carrier_mode),
        content_strength_common_multiplier=1.0,
    )
    update = _restrict_content_update(update, carrier_mode, z10)
    lf_update, hf_update = _active_content_updates(update, carrier_mode)
    sign = float(chain.replay_sign or 1)
    return lf_update * sign, hf_update * sign


def _score_chain_observations(
    chain: _ObservedChain,
    *,
    cell_identity_digest: str,
    registered_key_material: str,
    wrong_keys: tuple[Mapping[str, Any], ...],
    model_identity_digest: str,
    routing_mode: str,
    carrier_mode: str,
    registered_blind_cache: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[dict[str, Any], ...]:
    score_role = _score_method_role(routing_mode, carrier_mode)
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
    routed_by_key = {
        record["key_material_digest_random"]: _routed_templates_for_key(
            chain,
            key_material=record["key_material"],
            model_identity_digest=model_identity_digest,
            routing_mode=routing_mode,
            carrier_mode=carrier_mode,
        )
        for record in keys
    }
    observation_records: list[dict[str, Any]] = []
    for observation_role, latent in _observation_tensors(chain).items():
        key_records: list[dict[str, Any]] = []
        for key in keys:
            cache_key = f"{chain.role}:{observation_role}"
            blind = None
            if key["key_role"] == "registered" and registered_blind_cache:
                blind = registered_blind_cache.get(cache_key)
            if blind is None:
                blind = compute_blind_observation_score(
                    latent,
                    key_material=key["key_material"],
                    model_identity_digest=model_identity_digest,
                    prg_version=KEYED_PRG_VERSION,
                    method_role=score_role,
                )
            routed_lf, routed_hf = routed_by_key[key["key_material_digest_random"]]
            oracle = compute_routed_template_oracle_score(
                latent,
                routed_lf,
                routed_hf,
                lf_weight=float(blind["lf_weight"]),
                hf_weight=float(blind["hf_tail_weight"]),
                oracle_identity={
                    "routing_mode": routing_mode,
                    "carrier_mode": carrier_mode,
                    "chain_role": chain.role,
                    "key_material_digest_random": key[
                        "key_material_digest_random"
                    ],
                },
            )
            key_records.append(
                {
                    "key_role": key["key_role"],
                    "key_index": key["key_index"],
                    "key_material_digest_random": key[
                        "key_material_digest_random"
                    ],
                    "blind_full_template": blind,
                    "routed_template_oracle": oracle,
                    "feedback_allowed": False,
                    **CONTENT_SURVIVAL_CLAIM_BOUNDARY,
                }
            )
        registered = key_records[0]
        wrong = key_records[1:]
        rank_records = {
            family: build_registered_rank_record(
                float(registered[family][
                    "blind_content_score" if family == "blind_full_template" else "score"
                ]),
                [
                    float(record[family][
                        "blind_content_score" if family == "blind_full_template" else "score"
                    ])
                    for record in wrong
                ],
            )
            for family in ("blind_full_template", "routed_template_oracle")
        }
        payload = {
            "cell_identity_digest": cell_identity_digest,
            "chain_role": chain.role,
            "observation_role": observation_role,
            "latent_content_sha256": tensor_content_sha256(latent),
            "parent_chain_digest": chain.chain_record["parent_chain_digest"],
            "key_score_records": key_records,
            "rank_records": rank_records,
            "oracle_feedback_allowed": False,
            "wrong_key_feedback_allowed": False,
            **CONTENT_SURVIVAL_CLAIM_BOUNDARY,
        }
        payload["observation_record_digest"] = build_stable_digest(payload)
        observation_records.append(
            validate_content_survival_observation_record(
                payload,
                expected_cell_identity_digest=cell_identity_digest,
                expected_latent_content_sha256=payload[
                    "latent_content_sha256"
                ],
                expected_registered_key_digest_random=key_records[0][
                    "key_material_digest_random"
                ],
                expected_wrong_key_digests_random=[
                    record["key_material_digest_random"]
                    for record in key_records[1:]
                ],
            )
        )
    return tuple(observation_records)


def _chain_leaf_payloads(chain: _ObservedChain) -> dict[str, bytes]:
    prefix = chain.role
    return {
        f"{prefix}_chain_record.json": _stable_json_bytes(chain.chain_record),
        f"{prefix}_post_write_z10.pt": _tensor_bytes(chain.post_write_z10),
        f"{prefix}_pre_vae_terminal_latent.pt": _tensor_bytes(
            chain.pre_vae_terminal_latent
        ),
        f"{prefix}_image_reencoded_latent.pt": _tensor_bytes(
            chain.image_reencoded_latent
        ),
        f"{prefix}_image.png": _image_png_bytes(chain.image),
    }


def _observation_cell_identity(
    config: SemanticWatermarkRuntimeConfig,
    *,
    routing_mode: str,
    carrier_mode: str,
    roster: Mapping[str, Any],
    protocol: Any,
    formal_execution_lock: Mapping[str, Any],
    execution_environment_identity: Mapping[str, Any],
) -> dict[str, Any]:
    return build_content_survival_cell_identity(
        prompt_id=config.prompt_id,
        routing_mode=routing_mode,
        carrier_mode=carrier_mode,
        generation_seed_random=int(config.seed),
        registered_key_material_digest_random=roster[
            "registered_key_material_digest_random"
        ],
        formal_execution_lock=formal_execution_lock,
        execution_environment_identity=execution_environment_identity,
        model_id=config.model_id,
        model_revision=config.model_revision,
        vision_model_id=config.vision_model_id,
        vision_model_revision=config.vision_model_revision,
        semantic_watermark_runtime_config_digest=(
            semantic_watermark_runtime_config_digest(config)
        ),
        prompt_text_digest=build_stable_digest({"prompt_text": config.prompt}),
        key_roster=roster,
        protocol=protocol,
    )


def _run_observation_cell(
    config: SemanticWatermarkRuntimeConfig,
    references: ContentRoutingReferenceScalars,
    *,
    context: SemanticWatermarkRuntimeContext,
    base_latent: Any,
    base_identity: Mapping[str, Any],
    clean_chain: _ObservedChain,
    routing_mode: str,
    carrier_mode: str,
    roster: Mapping[str, Any],
    protocol: Any,
    formal_execution_lock: Mapping[str, Any],
    execution_environment_identity: Mapping[str, Any],
    observation_run_identity: Mapping[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    prompt_text_digest = build_stable_digest({"prompt_text": config.prompt})
    prompt_config_digest = semantic_watermark_runtime_config_digest(config)
    cell_identity = _observation_cell_identity(
        config,
        routing_mode=routing_mode,
        carrier_mode=carrier_mode,
        roster=roster,
        formal_execution_lock=formal_execution_lock,
        execution_environment_identity=execution_environment_identity,
        protocol=protocol,
    )
    cell_dir = output_root / config.prompt_id / routing_mode / carrier_mode
    complete_path = cell_dir / "cell_manifest.json"
    failure_path = cell_dir / "cell_failure.json"
    if complete_path.exists() and failure_path.exists():
        raise ValueError("observation cell has conflicting terminal outcomes")
    if complete_path.exists():
        validate_content_survival_cell_bundle(
            cell_dir,
            expected_cell_identity_digest=cell_identity["cell_identity_digest"],
            complete=True,
        )
        return json.loads((cell_dir / "cell_result.json").read_text(encoding="utf-8"))
    if failure_path.exists():
        return validate_content_survival_cell_failure_record(
            json.loads(failure_path.read_text(encoding="utf-8")),
            expected_cell_identity_digest=cell_identity["cell_identity_digest"],
        )

    chains: list[_ObservedChain] = []
    early_blind: dict[str, Mapping[str, Any]] = {}
    model_identity_digest = build_stable_digest(
        {"model_id": config.model_id, "model_revision": config.model_revision}
    )
    score_role = _score_method_role(routing_mode, carrier_mode)
    active_role = CONTENT_SURVIVAL_CHAIN_ROLES[0]
    try:
        for role, sign in (
            ("full_probe_positive", 1),
            ("full_probe_negative", -1),
            ("carrier_probe_positive", 1),
            ("carrier_probe_negative", -1),
        ):
            active_role = role
            chain = _run_observed_chain(
                config,
                references,
                context=context,
                base_latent=base_latent,
                base_identity=base_identity,
                routing_mode=routing_mode,
                carrier_mode=carrier_mode,
                role=role,
                probe_sign=sign,
                replay_sign=None,
                observation_run_identity_digest=observation_run_identity[
                    "observation_run_identity_digest"
                ],
                prompt_text_digest=prompt_text_digest,
                prompt_config_digest=prompt_config_digest,
                key_roster_digest_random=roster["roster_digest_random"],
                cell_identity=cell_identity,
            )
            chains.append(chain)
            cache_key = f"{role}:image_reencoded_latent"
            early_blind[cache_key] = compute_blind_observation_score(
                chain.image_reencoded_latent,
                key_material=config.key_material,
                model_identity_digest=model_identity_digest,
                prg_version=KEYED_PRG_VERSION,
                method_role=score_role,
            )
        selection = select_content_survival_observation_sign(
            {
                role: float(
                    early_blind[f"{role}:image_reencoded_latent"][
                        "blind_content_score"
                    ]
                )
                for role in CONTENT_SURVIVAL_CHAIN_ROLES[:4]
            }
        )
        for role, sign in (
            ("nominal_before_positive", 1),
            ("nominal_after_selected", int(selection["selected_sign"])),
        ):
            active_role = role
            chains.append(
                _run_observed_chain(
                    config,
                    references,
                    context=context,
                    base_latent=base_latent,
                    base_identity=base_identity,
                    routing_mode=routing_mode,
                    carrier_mode=carrier_mode,
                    role=role,
                    probe_sign=None,
                    replay_sign=sign,
                    observation_run_identity_digest=observation_run_identity[
                        "observation_run_identity_digest"
                    ],
                    prompt_text_digest=prompt_text_digest,
                    prompt_config_digest=prompt_config_digest,
                    key_roster_digest_random=roster["roster_digest_random"],
                    cell_identity=cell_identity,
                )
            )
    except ValueError as exc:
        if str(exc) != _GEOMETRY_FAILURE_MESSAGE:
            raise
        failure = build_content_survival_cell_failure_record(
            cell_identity=cell_identity,
            failed_chain_role=active_role,
            successful_chain_count=len(chains),
        )
        if failure["failure_code"] != CONTENT_SURVIVAL_GEOMETRY_FAILURE_CODE:
            raise RuntimeError("geometry failure classification drifted")
        return publish_content_survival_cell_failure(
            cell_dir,
            failure_record=failure,
        )

    observation_records: list[dict[str, Any]] = []
    wrong_keys = tuple(roster["wrong_keys"])
    for chain in chains:
        observation_records.extend(
            _score_chain_observations(
                chain,
                cell_identity_digest=cell_identity["cell_identity_digest"],
                registered_key_material=config.key_material,
                wrong_keys=wrong_keys,
                model_identity_digest=model_identity_digest,
                routing_mode=routing_mode,
                carrier_mode=carrier_mode,
                registered_blind_cache=early_blind,
            )
        )

    clean_observations: tuple[dict[str, Any], ...] = ()
    if routing_mode == "semantic" and carrier_mode == "dual":
        clean_template_chain = chains[0]
        clean_proxy = _ObservedChain(
            role="clean_reference",
            pre_write_z10=clean_template_chain.pre_write_z10,
            post_write_z10=clean_chain.post_write_z10,
            pre_vae_terminal_latent=clean_chain.pre_vae_terminal_latent,
            image_reencoded_latent=clean_chain.image_reencoded_latent,
            image=clean_chain.image,
            routing=clean_template_chain.routing,
            chain_record=clean_chain.chain_record,
            lf_update=clean_template_chain.lf_update,
            hf_update=clean_template_chain.hf_update,
            replay_sign=1,
        )
        clean_observations = _score_chain_observations(
            clean_proxy,
            cell_identity_digest=cell_identity["cell_identity_digest"],
            registered_key_material=config.key_material,
            wrong_keys=wrong_keys,
            model_identity_digest=model_identity_digest,
            routing_mode=routing_mode,
            carrier_mode=carrier_mode,
        )

    causal_diagnostic = build_content_survival_causal_diagnostic(
        [chain.chain_record for chain in chains]
    )
    parent_child_binding_digest = (
        build_content_survival_parent_child_binding_digest(
            [chain.chain_record for chain in chains],
            observation_records,
            clean_chain_record=(clean_chain.chain_record if clean_observations else None),
            clean_observation_records=clean_observations,
        )
    )
    result_payload = {
        "result_schema": "slm_wm_content_survival_observation_cell_result",
        "cell_identity": cell_identity,
        "roster_digest_random": roster["roster_digest_random"],
        "selection": selection,
        "chain_records": [chain.chain_record for chain in chains],
        "observation_records": observation_records,
        "clean_observation_records": list(clean_observations),
        "parent_child_binding_digest": parent_child_binding_digest,
        **(
            {"clean_chain_record": clean_chain.chain_record}
            if clean_observations
            else {}
        ),
        **causal_diagnostic,
        "known_m0_deviations": list(CONTENT_SURVIVAL_M0_DEVIATIONS),
        "cell_chain_count": len(chains),
        "cell_evaluation_count": len(observation_records) * 33 * 2,
        **CONTENT_SURVIVAL_CLAIM_BOUNDARY,
    }
    result_payload["result_digest"] = build_stable_digest(result_payload)
    leaves = {"cell_result.json": _stable_json_bytes(result_payload)}
    for chain in chains:
        leaves.update(_chain_leaf_payloads(chain))
    if clean_observations:
        leaves.update(_chain_leaf_payloads(clean_chain))
    publish_content_survival_cell(
        cell_dir,
        cell_identity=cell_identity,
        leaf_payloads=leaves,
    )
    return result_payload


def _load_terminal_observation_cells(
    configs: tuple[SemanticWatermarkRuntimeConfig, ...],
    *,
    roster: Mapping[str, Any],
    protocol: Any,
    formal_execution_lock: Mapping[str, Any],
    execution_environment_identity: Mapping[str, Any],
    output_root: Path,
) -> list[dict[str, Any]] | None:
    results: list[dict[str, Any]] = []
    for config in configs:
        for routing_mode in CONTENT_SURVIVAL_ROUTING_MODES:
            for carrier_mode in CONTENT_SURVIVAL_CARRIER_MODES:
                identity = _observation_cell_identity(
                    config,
                    routing_mode=routing_mode,
                    carrier_mode=carrier_mode,
                    roster=roster,
                    protocol=protocol,
                    formal_execution_lock=formal_execution_lock,
                    execution_environment_identity=execution_environment_identity,
                )
                cell_dir = (
                    output_root / config.prompt_id / routing_mode / carrier_mode
                )
                manifest_path = cell_dir / "cell_manifest.json"
                failure_path = cell_dir / "cell_failure.json"
                if manifest_path.is_file() == failure_path.is_file():
                    return None
                if manifest_path.is_file():
                    validate_content_survival_cell_bundle(
                        cell_dir,
                        expected_cell_identity_digest=identity[
                            "cell_identity_digest"
                        ],
                        complete=True,
                    )
                    results.append(
                        json.loads(
                            (cell_dir / "cell_result.json").read_text(
                                encoding="utf-8"
                            )
                        )
                    )
                else:
                    results.append(
                        validate_content_survival_cell_failure_record(
                            json.loads(failure_path.read_text(encoding="utf-8")),
                            expected_cell_identity_digest=identity[
                                "cell_identity_digest"
                            ],
                        )
                    )
    return results


def _finalize_observation_summary(
    output_root: Path,
    *,
    cell_results: list[dict[str, Any]],
    observation_run_identity: Mapping[str, Any],
    protocol: Any,
) -> dict[str, Any]:
    failed_cells = [
        result for result in cell_results if "failure_schema" in result
    ]
    if failed_cells:
        expected_cells = [
            {
                "relative_path": (
                    Path(result["cell_identity"]["prompt_id"])
                    / result["cell_identity"]["routing_mode"]
                    / result["cell_identity"]["carrier_mode"]
                ).as_posix(),
                "cell_identity_digest": result["cell_identity"][
                    "cell_identity_digest"
                ],
            }
            for result in cell_results
        ]
        summary = build_content_survival_observation_attempt_summary(
            output_root,
            expected_cells=expected_cells,
            observation_run_identity=observation_run_identity,
            protocol=protocol,
        )
        _publish_summary(output_root, summary)
        return summary
    chain_count = len(CONTENT_SURVIVAL_PROMPT_IDS) + sum(
        int(result["cell_chain_count"]) for result in cell_results
    )
    evaluation_count = sum(
        int(result["cell_evaluation_count"]) for result in cell_results
    ) + sum(
        len(result["clean_observation_records"]) * 33 * 2
        for result in cell_results
    )
    if chain_count != CONTENT_SURVIVAL_CHAIN_COUNT:
        raise RuntimeError("observation did not execute the fixed 148 chains")
    if evaluation_count != CONTENT_SURVIVAL_EVALUATION_COUNT:
        raise RuntimeError("observation did not materialize all 29,304 evaluations")
    expected_cells = [
        {
            "relative_path": (
                Path(result["cell_identity"]["prompt_id"])
                / result["cell_identity"]["routing_mode"]
                / result["cell_identity"]["carrier_mode"]
            ).as_posix(),
            "cell_identity_digest": result["cell_identity"][
                "cell_identity_digest"
            ],
        }
        for result in cell_results
    ]
    summary = build_content_survival_observation_summary(
        output_root,
        expected_cells=expected_cells,
        observation_run_identity=observation_run_identity,
        protocol=protocol,
    )
    _publish_summary(output_root, summary)
    return summary


def run_content_survival_observation(
    prompt_configs: Mapping[str, SemanticWatermarkRuntimeConfig],
    *,
    references: ContentRoutingReferenceScalars,
    verified_formal_execution_lock: Mapping[str, Any],
    verified_execution_environment_identity: Mapping[str, Any],
    repository_root: str | Path,
    output_dir: str | Path,
    runtime_context: SemanticWatermarkRuntimeContext | None = None,
) -> dict[str, Any]:
    """Run the fixed 148-chain observation without entering formal evidence."""

    root = Path(repository_root).resolve()
    output_root = (root / Path(output_dir)).resolve()
    output_root.relative_to((root / "outputs").resolve())
    if tuple(prompt_configs) != CONTENT_SURVIVAL_PROMPT_IDS:
        raise ValueError("prompt configs must follow the frozen four-prompt roster")
    configs = tuple(prompt_configs[prompt_id] for prompt_id in CONTENT_SURVIVAL_PROMPT_IDS)
    if any(type(config) is not SemanticWatermarkRuntimeConfig for config in configs):
        raise TypeError("prompt configs must be exact SemanticWatermarkRuntimeConfig values")
    for config in configs:
        _require_full_content_runtime_config(config)
        if config.prompt_id not in CONTENT_SURVIVAL_PROMPT_IDS:
            raise ValueError("runtime prompt identity is not frozen")
        if config.injection_step_indices != (10,):
            raise ValueError("observation requires the formal z10 write index")
    registered_keys = {config.key_material for config in configs}
    if len(registered_keys) != 1:
        raise ValueError("all observation prompts must share one registered key")
    verified_lock = validate_formal_execution_lock_record(
        verified_formal_execution_lock
    )
    protocol = load_content_survival_observation_protocol(root)
    roster = build_content_survival_observation_roster(
        configs[0].key_material,
        protocol=protocol,
    )
    observation_run_identity = build_content_survival_observation_run_identity(
        formal_execution_lock=verified_lock,
        execution_environment_identity=verified_execution_environment_identity,
        model_id=configs[0].model_id,
        model_revision=configs[0].model_revision,
        vision_model_id=configs[0].vision_model_id,
        vision_model_revision=configs[0].vision_model_revision,
        key_roster=roster,
        protocol=protocol,
    )
    if any(
        config.model_id != configs[0].model_id
        or config.model_revision != configs[0].model_revision
        or config.vision_model_id != configs[0].vision_model_id
        or config.vision_model_revision != configs[0].vision_model_revision
        for config in configs
    ):
        raise ValueError("all observation prompts must share one model identity")
    completed_results = _load_terminal_observation_cells(
        configs,
        roster=roster,
        protocol=protocol,
        formal_execution_lock=verified_lock,
        execution_environment_identity=verified_execution_environment_identity,
        output_root=output_root,
    )
    if completed_results is not None:
        return _finalize_observation_summary(
            output_root,
            cell_results=completed_results,
            observation_run_identity=observation_run_identity,
            protocol=protocol,
        )
    context = runtime_context or load_semantic_watermark_runtime_context(
        configs[0],
        verified_formal_execution_lock=verified_formal_execution_lock,
        repository_root=root,
    )
    cell_results: list[dict[str, Any]] = []
    for config in configs:
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
        clean_chain = _run_observed_chain(
            config,
            references,
            context=context,
            base_latent=base_latent,
            base_identity=base_identity,
            routing_mode="semantic",
            carrier_mode="dual",
            role="clean_reference",
            probe_sign=None,
            replay_sign=None,
            observation_run_identity_digest=observation_run_identity[
                "observation_run_identity_digest"
            ],
            prompt_text_digest=build_stable_digest(
                {"prompt_text": config.prompt}
            ),
            prompt_config_digest=semantic_watermark_runtime_config_digest(
                config
            ),
            key_roster_digest_random=roster["roster_digest_random"],
            cell_identity=None,
        )
        for routing_mode in CONTENT_SURVIVAL_ROUTING_MODES:
            for carrier_mode in CONTENT_SURVIVAL_CARRIER_MODES:
                cell_results.append(
                    _run_observation_cell(
                        config,
                        references,
                        context=context,
                        base_latent=base_latent,
                        base_identity=base_identity,
                        clean_chain=clean_chain,
                        routing_mode=routing_mode,
                        carrier_mode=carrier_mode,
                        roster=roster,
                        protocol=protocol,
                        formal_execution_lock=verified_lock,
                        execution_environment_identity=(
                            verified_execution_environment_identity
                        ),
                        observation_run_identity=observation_run_identity,
                        output_root=output_root,
                    )
                )
    return _finalize_observation_summary(
        output_root,
        cell_results=cell_results,
        observation_run_identity=observation_run_identity,
        protocol=protocol,
    )


__all__ = ["run_content_survival_observation"]
