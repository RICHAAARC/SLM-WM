"""覆盖 terminal registered Q/K 候选协议与 CPU 数据流。"""

from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest

import experiments.runners.semantic_watermark_runtime as runtime
from experiments.protocol.content_survival_direction import (
    load_content_survival_direction_protocol,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.quick
def test_terminal_qk_protocol_freezes_registered_only_exact_image_search() -> None:
    protocol = load_content_survival_direction_protocol(REPOSITORY_ROOT)
    payload = protocol.payload["terminal_qk_sync"]

    assert payload == {
        "full_replay_only": True,
        "terminal_hf_update": "unchanged_before_independent_qk_sync",
        "direction_source": (
            "differentiable_vae_roundtrip_public_noise_registered_qk"
        ),
        "direction_mask": "semantic_writable_capacity_map",
        "score_source": "image_reencoded_public_noise_real_qk",
        "public_detection_schedule_index": 7,
        "selection_key_access": "registered_only_wrong_key_forbidden",
        "candidate_scale_fractions": [0.0, 0.0625, 0.125, 0.25, 0.5, 1.0],
        "geometry_actual_dtype_relative_l2_limit": 0.001,
        "combined_actual_dtype_relative_l2_limit": 0.014,
        "zero_baseline_required": True,
        "acceptance_rule": (
            "first_nonzero_candidate_with_strict_zero_baseline_improvement_"
            "and_both_full_vs_carrier_gains_above_runtime_minimum"
        ),
        "failure_policy": (
            "retain_zero_baseline_and_fail_final_image_attention_gate"
        ),
    }


@pytest.mark.quick
def test_terminal_qk_sync_uses_first_exact_candidate_and_never_wrong_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    protocol = load_content_survival_direction_protocol(REPOSITORY_ROOT)
    terminal = torch.ones((1, 2, 2, 2), dtype=torch.float32)
    terminal_hf = SimpleNamespace(written_latent=terminal + 0.01)
    carrier_image = SimpleNamespace(role="carrier", latent=terminal)
    score_keys: list[str] = []
    extracted_images: list[object] = []
    decoded_latents: list[object] = []

    def fake_extractor(*_args: object, **_kwargs: object):
        def extract(image: object):
            extracted_images.append(image)
            return (("layer0", image, (0,)), ("layer1", image, (0,)))

        return extract

    def fake_score(records, key_material, _config, *, carrier_pair_weights=None):
        score_keys.append(key_material)
        image = records[0][1]
        if getattr(image, "role", "") == "carrier":
            value = 0.1
        else:
            delta = float((image - (terminal + 0.01)).abs().mean().item())
            value = 0.10005 + delta * 1.0e3
        weights = carrier_pair_weights or SimpleNamespace(
            pair_weight_identity_digest="a" * 64,
            pair_weight_realization_digest="b" * 64,
        )
        return (
            {
                "blind_attention_score": value,
                "carrier_paired_attention_score": value,
                "blind_pair_weight_identity_digest": "a" * 64,
                "blind_pair_weight_realization_digest": "b" * 64,
                "carrier_pair_weight_identity_digest": "a" * 64,
                "carrier_pair_weight_realization_digest": "b" * 64,
                "attention_relation_component_identity_digest": "c" * 64,
                "attention_relation_keyed_projection_digest": "d" * 64,
                "attention_relation_qk_operator_metadata_digest": "e" * 64,
                "qk_atomic_content_digest": "f" * 64,
            },
            weights,
        )

    monkeypatch.setattr(runtime, "_image_attention_extractor", fake_extractor)
    monkeypatch.setattr(runtime, "_terminal_registered_qk_score", fake_score)
    monkeypatch.setattr(
        runtime,
        "_terminal_registered_qk_direction",
        lambda **_kwargs: (
            torch.ones_like(terminal)
            / torch.linalg.vector_norm(torch.ones_like(terminal).reshape(-1)),
            {
                "terminal_qk_direction_source": (
                    "differentiable_vae_roundtrip_public_noise_registered_qk"
                ),
                "terminal_qk_direction_score_is_acceptance_evidence": False,
                "terminal_qk_direction_registered_only": True,
                "terminal_qk_direction_wrong_key_accessed": False,
                "terminal_qk_direction_content_sha256": "1" * 64,
                "terminal_qk_direction_gradient_content_sha256": "2" * 64,
                "terminal_qk_direction_geometry_capacity_content_sha256": "3" * 64,
                "terminal_qk_direction_public_schedule_index": 7,
                "terminal_qk_direction_public_timestep": 1.0,
                "terminal_qk_direction_record_digest": "4" * 64,
            },
        ),
    )
    monkeypatch.setattr(
        runtime,
        "_decode_content_runtime_latent_image",
        lambda _pipeline, latent: (
            decoded_latents.append(latent.detach().clone())
            or latent.detach().clone()
        ),
    )
    monkeypatch.setattr(
        runtime,
        "canonical_rgb_uint8_content_record",
        lambda image: {
            "image_rgb_uint8_content_sha256": runtime.tensor_content_sha256(image),
            "image_width": 2,
            "image_height": 2,
        },
    )
    config = SimpleNamespace(
        public_detection_schedule_index=7,
        key_material="registered-key",
        minimum_final_image_attention_score_gain=1.0e-4,
    )
    selected, image, record = runtime._build_terminal_registered_qk_sync(
        terminal_latent=terminal,
        terminal_hf_update=terminal_hf,
        terminal_routing=SimpleNamespace(
            writable_capacity_map=torch.ones_like(terminal)
        ),
        carrier_only_image=carrier_image,
        pipeline=object(),
        config=config,
        modules=(),
        public_prompt_embeds=object(),
        public_pooled_prompt_embeds=object(),
        protocol=protocol,
    )

    assert record["terminal_qk_sync_selected_candidate_index"] == 1
    assert record["terminal_qk_sync_ready"] is True
    assert record["terminal_qk_sync_applied"] is True
    assert record["terminal_qk_sync_wrong_key_accessed"] is False
    assert all(key == "registered-key" for key in score_keys)
    candidate_records = record["terminal_qk_sync_candidate_records"]
    assert len(candidate_records) == 6
    assert candidate_records[0]["zero_baseline"] is True
    assert all(
        candidate["geometry_actual_dtype_relative_l2"] <= 0.001
        and candidate["combined_actual_dtype_relative_l2"] <= 0.014
        for candidate in candidate_records
    )
    assert len(decoded_latents) == 6
    assert len(extracted_images) == 7
    assert extracted_images[0] is carrier_image
    assert torch.equal(selected, image)

    failed_config = SimpleNamespace(
        public_detection_schedule_index=7,
        key_material="registered-key",
        minimum_final_image_attention_score_gain=100.0,
    )
    failed_latent, failed_image, failed_record = (
        runtime._build_terminal_registered_qk_sync(
            terminal_latent=terminal,
            terminal_hf_update=terminal_hf,
            terminal_routing=SimpleNamespace(
                writable_capacity_map=torch.ones_like(terminal)
            ),
            carrier_only_image=carrier_image,
            pipeline=object(),
            config=failed_config,
            modules=(),
            public_prompt_embeds=object(),
            public_pooled_prompt_embeds=object(),
            protocol=protocol,
        )
    )
    assert failed_record["terminal_qk_sync_selected_candidate_index"] == 0
    assert failed_record["terminal_qk_sync_ready"] is False
    assert failed_record["terminal_qk_sync_failure_reason"] == (
        "no_exact_image_qk_candidate_passed"
    )
    assert torch.equal(failed_latent, terminal_hf.written_latent)
    assert torch.equal(failed_image, terminal_hf.written_latent)
    assert all(key == "registered-key" for key in score_keys)


@pytest.mark.quick
def test_full_replay_is_only_terminal_qk_sync_callsite() -> None:
    source = inspect.getsource(
        runtime._run_semantic_watermark_runtime_with_content_strength
    )

    assert 'if role == "full_nominal_replay"' in source
    assert source.count("_build_terminal_registered_qk_sync(") == 1
    assert "public_prompt_embeds=context.unconditional_prompt" in source
    assert '"terminal_qk_sync_applied": False' in source
    assert '"method_runtime": "formal_terminal_hf_content_dual_chain"' in source
    assert '"formal_attribution_carrier": "terminal_pre_vae_hf_tail"' in source
