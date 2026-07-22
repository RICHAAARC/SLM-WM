"""CPU contracts for terminal-latent keyed carrier repair and scoring."""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from experiments.runners import terminal_content_carrier_runtime as runtime
from experiments.protocol.content_routing_reference_quantile import (
    ContentRoutingReferenceScalars,
)
from experiments.runners.semantic_watermark_runtime import (
    SemanticWatermarkRuntimeConfig,
    SemanticWatermarkRuntimeContext,
)
from main.core.digest import build_stable_digest
from main.core.keyed_prg import KEYED_PRG_VERSION
from main.methods.carrier.blind_content_score import compute_blind_content_score
from main.methods.carrier.high_frequency_tail import build_high_frequency_tail_template
from main.methods.carrier.low_frequency import build_low_frequency_template
from main.methods.carrier.terminal_update import build_terminal_content_carrier_update
from main.methods.content.routing import ContentRoutingResult


pytestmark = pytest.mark.quick
MODEL_IDENTITY_DIGEST = "a" * 64


def _routing() -> ContentRoutingResult:
    lf_mask = torch.linspace(0.2, 1.0, 16 * 16).reshape(1, 1, 16, 16)
    hf_mask = torch.flip(lf_mask, dims=(2, 3))
    return ContentRoutingResult(
        writable_capacity_map=torch.maximum(lf_mask, hf_mask),
        lf_mask=lf_mask,
        hf_tail_mask=hf_mask,
        routing_identity_digest=build_stable_digest({"routing": "synthetic"}),
    )


def _templates(latent: torch.Tensor, key: str):
    return (
        build_low_frequency_template(
            latent,
            key,
            MODEL_IDENTITY_DIGEST,
            prg_version=KEYED_PRG_VERSION,
        ),
        build_high_frequency_tail_template(
            latent,
            key,
            MODEL_IDENTITY_DIGEST,
            prg_version=KEYED_PRG_VERSION,
        ),
    )


@pytest.mark.parametrize(
    ("carrier_mode", "lf_relative", "hf_relative"),
    (("lf_only", 0.01, 0.0), ("hf_only", 0.0, 0.006), ("dual", 0.01, 0.006)),
)
def test_terminal_update_restores_fixed_branch_energy(
    carrier_mode: str,
    lf_relative: float,
    hf_relative: float,
) -> None:
    latent = torch.randn((1, 4, 16, 16), generator=torch.Generator().manual_seed(9))
    lf_template, hf_template = _templates(latent, "registered-key")
    update = build_terminal_content_carrier_update(
        latent,
        _routing(),
        lf_template,
        hf_template,
        routing_mode="semantic_unit_energy",
        carrier_mode=carrier_mode,
        strength_multiplier=4.0,
    )
    latent_l2 = float(torch.linalg.vector_norm(latent.reshape(-1)).item())

    assert math.isclose(update.lf_effective_l2 / latent_l2, lf_relative, abs_tol=2e-7)
    assert math.isclose(update.hf_tail_effective_l2 / latent_l2, hf_relative, abs_tol=2e-7)
    assert torch.linalg.vector_norm(update.lf_direction).item() == pytest.approx(
        0.0 if carrier_mode == "hf_only" else 1.0
    )
    assert torch.linalg.vector_norm(update.hf_tail_direction).item() == pytest.approx(
        0.0 if carrier_mode == "lf_only" else 1.0
    )
    assert update.combined_relative_l2 > 0.0
    assert not torch.equal(update.written_latent, latent)


@pytest.mark.parametrize(
    ("carrier_mode", "zero_branch"),
    (("lf_only", "hf_tail"), ("hf_only", "lf")),
)
def test_terminal_update_accepts_zero_mask_for_inactive_branch(
    carrier_mode: str,
    zero_branch: str,
) -> None:
    latent = torch.randn((1, 4, 16, 16), generator=torch.Generator().manual_seed(19))
    lf_template, hf_template = _templates(latent, "registered-key")
    routing = _routing()
    routing = ContentRoutingResult(
        writable_capacity_map=routing.writable_capacity_map,
        lf_mask=(torch.zeros_like(routing.lf_mask) if zero_branch == "lf" else routing.lf_mask),
        hf_tail_mask=(
            torch.zeros_like(routing.hf_tail_mask)
            if zero_branch == "hf_tail"
            else routing.hf_tail_mask
        ),
        routing_identity_digest=routing.routing_identity_digest,
    )

    update = build_terminal_content_carrier_update(
        latent,
        routing,
        lf_template,
        hf_template,
        routing_mode="semantic_unit_energy",
        carrier_mode=carrier_mode,
        strength_multiplier=4.0,
    )

    assert update.combined_relative_l2 > 0.0


def test_terminal_update_rejects_zero_mask_for_active_dual_branch() -> None:
    latent = torch.randn((1, 4, 16, 16), generator=torch.Generator().manual_seed(29))
    lf_template, hf_template = _templates(latent, "registered-key")
    routing = _routing()
    routing = ContentRoutingResult(
        writable_capacity_map=routing.writable_capacity_map,
        lf_mask=routing.lf_mask,
        hf_tail_mask=torch.zeros_like(routing.hf_tail_mask),
        routing_identity_digest=routing.routing_identity_digest,
    )

    with pytest.raises(ValueError, match="HF-tail terminal 方向"):
        build_terminal_content_carrier_update(
            latent,
            routing,
            lf_template,
            hf_template,
            routing_mode="semantic_unit_energy",
            carrier_mode="dual",
            strength_multiplier=4.0,
        )


def test_terminal_write_makes_registered_key_rank_first_on_synthetic_latent() -> None:
    latent = torch.ones((1, 4, 16, 16), dtype=torch.float32)
    registered_key = "registered-key"
    lf_template, hf_template = _templates(latent, registered_key)
    update = build_terminal_content_carrier_update(
        latent,
        _routing(),
        lf_template,
        hf_template,
        routing_mode="uniform",
        carrier_mode="dual",
        strength_multiplier=4.0,
    )
    keys = (registered_key, *(f"wrong-key-{index}" for index in range(32)))
    scores = []
    for key in keys:
        lf, hf = _templates(latent, key)
        scores.append(
            compute_blind_content_score(
                update.written_latent,
                lf,
                hf,
                "full_dual_chain",
            ).blind_content_score
        )

    assert scores[0] > max(scores[1:])


def test_compact_terminal_workload_has_four_diffusions_and_48_variants() -> None:
    assert runtime.TERMINAL_CONTENT_DIFFUSION_CHAIN_COUNT == 4
    assert runtime.TERMINAL_CONTENT_VARIANT_COUNT == 48
    assert runtime.TERMINAL_CONTENT_KEY_SCORE_COUNT == 3168
    assert runtime.TERMINAL_CONTENT_ROUTING_MODES == (
        "semantic_unit_energy",
        "uniform",
    )


def test_terminal_quality_adapter_uses_canonical_rgb_images() -> None:
    image = torch.linspace(0.0, 1.0, 3 * 16 * 16).reshape(1, 3, 16, 16)
    quality = runtime._paired_quality(image, image.clone())

    assert quality == {
        "psnr": "inf",
        "ssim": pytest.approx(1.0),
        "mse": 0.0,
        "mean_abs_error": 0.0,
    }


def test_terminal_key_roster_scores_registered_plus_32_wrong_keys() -> None:
    latent = torch.ones((1, 4, 16, 16), dtype=torch.float32)
    registered = "registered-key"
    lf_template, hf_template = _templates(latent, registered)
    written = build_terminal_content_carrier_update(
        latent,
        _routing(),
        lf_template,
        hf_template,
        routing_mode="uniform",
        carrier_mode="dual",
        strength_multiplier=4.0,
    ).written_latent
    wrong_keys = tuple(
        {
            "wrong_key_index": index,
            "wrong_key_material": f"wrong-key-{index}",
            "wrong_key_material_digest_random": build_stable_digest(
                {"key_material": f"wrong-key-{index}"}
            ),
        }
        for index in range(32)
    )
    scored = runtime._score_key_roster(
        written,
        registered_key_material=registered,
        wrong_keys=wrong_keys,
        model_identity_digest=MODEL_IDENTITY_DIGEST,
        carrier_mode="dual",
    )

    assert len(scored["key_score_records"]) == 33
    assert scored["key_score_records"][0]["key_role"] == "registered"
    assert scored["rank_record"]["registered_rank"] == 1


def test_compact_runner_executes_four_diffusions_and_resumes_without_gpu_calls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Pipeline:
        _execution_device = torch.device("cpu")
        vae_scale_factor = 64
        transformer = SimpleNamespace(
            config=SimpleNamespace(in_channels=4),
            dtype=torch.float32,
        )

        def __init__(self) -> None:
            self.call_count = 0

        def __call__(self, *, latents, callback_on_step_end, **kwargs):
            assert kwargs["output_type"] == "latent"
            self.call_count += 1
            value = latents.clone()
            for index in range(11):
                value = callback_on_step_end(
                    self,
                    index,
                    torch.tensor(float(index)),
                    {"latents": value + 0.01},
                )["latents"]
            return SimpleNamespace(images=value)

    pipeline = _Pipeline()
    context = SemanticWatermarkRuntimeContext(
        pipeline=pipeline,
        prompt_saliency_runtime=object(),
        attention_modules=(),
        unconditional_prompt=None,
        unconditional_pooled=None,
        runtime_versions={},
    )
    configs = {
        prompt_id: SemanticWatermarkRuntimeConfig(
            prompt=f"Prompt for {prompt_id}",
            prompt_id=prompt_id,
            key_material="registered-key",
        )
        for prompt_id in runtime.CONTENT_SURVIVAL_PROMPT_IDS
    }
    monkeypatch.setattr(runtime, "_require_full_content_runtime_config", lambda *_a: None)
    monkeypatch.setattr(runtime, "validate_formal_execution_lock_record", lambda value: value)
    monkeypatch.setattr(
        runtime,
        "build_canonical_sd35_base_latent",
        lambda **_kwargs: (
            torch.ones((1, 4, 16, 16)),
            {
                "base_latent_identity_digest_random": "b" * 64,
            },
        ),
    )

    def _routing_observation(**kwargs):
        kwargs["vae_decoder"](kwargs["current_scheduler_latent"])
        return SimpleNamespace(routing=_routing())

    monkeypatch.setattr(runtime, "build_content_observation_routing", _routing_observation)
    monkeypatch.setattr(
        runtime,
        "_decode_content_runtime_latent",
        lambda _pipeline, latent: latent[:, :3].sigmoid(),
    )
    monkeypatch.setattr(
        runtime,
        "_encode_image_latent",
        lambda _pipeline, image: torch.cat(
            (image, image.mean(dim=1, keepdim=True)),
            dim=1,
        ),
    )

    def _scores(latent, **_kwargs):
        return {
            "latent_content_sha256": build_stable_digest(
                {"mean": float(latent.float().mean().item())}
            ),
            "key_score_records": [
                {
                    "key_role": "registered" if index == 0 else "wrong",
                    "key_index": None if index == 0 else index - 1,
                    "blind_content_score": 1.0 if index == 0 else 0.0,
                }
                for index in range(33)
            ],
            "rank_record": {"registered_rank": 1},
        }

    monkeypatch.setattr(runtime, "_score_key_roster", _scores)
    output_root = tmp_path / "terminal_observation"
    output_dir = output_root.relative_to(Path(".").resolve())
    references = ContentRoutingReferenceScalars(
        reference_gradient=1.0,
        reference_response=1.0,
        reference_sensitivity=1.0,
    )
    kwargs = {
        "references": references,
        "verified_formal_execution_lock": {"lock": "test"},
        "verified_execution_environment_identity": {"environment": "test"},
        "repository_root": Path(".").resolve(),
        "output_dir": output_dir,
        "runtime_context": context,
    }
    summary = runtime.run_terminal_content_carrier_observation(configs, **kwargs)

    assert summary["diffusion_chain_count"] == 4
    assert summary["variant_count"] == 48
    assert summary["key_score_count"] == 3168
    assert pipeline.call_count == 4
    assert len(list(output_root.rglob("cell_manifest.json"))) == 4

    resumed = runtime.run_terminal_content_carrier_observation(configs, **kwargs)
    assert resumed == summary
    assert pipeline.call_count == 4

    first_prompt = output_root / runtime.CONTENT_SURVIVAL_PROMPT_IDS[0]
    image_path = first_prompt / "uniform__dual__x4.png"
    image_path.write_bytes(image_path.read_bytes() + b"tamper")
    with pytest.raises(RuntimeError, match="image digest"):
        runtime.run_terminal_content_carrier_observation(configs, **kwargs)
