"""覆盖生成内 late-HF 写入与 post-generation Q/K 解耦。"""

from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest

import experiments.runners.semantic_watermark_runtime as runtime
from experiments.protocol.content_survival_direction import (
    CONTENT_SURVIVAL_REPLAY_ROLES,
    load_content_survival_direction_protocol,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.quick
def test_late_hf_protocol_freezes_generation_callback_and_roles() -> None:
    protocol = load_content_survival_direction_protocol(REPOSITORY_ROOT)

    assert protocol.protocol_version == (
        "content_survival_direction_late_hf_qk_generation"
    )
    assert protocol.payload["late_hf_generation"] == {
        "callback_step_index": 18,
        "expected_inference_step_count": 20,
        "remaining_transformer_scheduler_step_count": 1,
        "roles": list(CONTENT_SURVIVAL_REPLAY_ROLES),
        "non_replay_write": "forbidden",
        "routing_mode": "semantic_unit_energy",
        "carrier_mode": "hf_only",
        "strength_multiplier": 8.0,
        "write_count_per_role": 1,
        "post_pipeline_write": "forbidden",
        "wrong_key_access": "forbidden_until_output_frozen",
    }
    assert "terminal_qk_sync" not in protocol.payload
    assert protocol.payload["late_qk_geometry"] == {
        "callback_step_index": 18,
        "roles": ["full_nominal_replay"],
        "carrier_only_write": "forbidden",
        "probe_write": "forbidden",
        "registered_key_only": True,
        "wrong_key_access": "forbidden_until_output_frozen",
        "geometry_actual_dtype_relative_l2_limit": 0.001,
        "combined_actual_dtype_relative_l2_limit": 0.014,
        "maximum_backtracking_index": 8,
        "backtracking_factor": 0.5,
        "direction_objective": "carrier_frozen_registered_suffix_qk_score",
        "direction_gradient_scope": "suffix_and_additive_delta_only",
        "suffix_operator": (
            "remaining_generation_step_decode_vae_reencode_"
            "public_noise_schedule_7_frozen_qk"
        ),
        "carrier_reference": "frozen_carrier_only_final_image_qk",
        "candidate_order": "maximum_geometry_then_fixed_halving",
        "acceptance": (
            "first_budget_ready_blind_and_carrier_paired_gain_"
            "at_least_formal_minimum"
        ),
        "write_count_per_full_replay": 1,
        "post_pipeline_write": "forbidden",
    }


@pytest.mark.quick
def test_late_hf_update_uses_registered_template_and_fixed_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    protocol = load_content_survival_direction_protocol(REPOSITORY_ROOT)
    latent = torch.ones((1, 2, 2, 2), dtype=torch.float32)
    written = latent + 0.01
    calls: dict[str, object] = {}

    monkeypatch.setattr(
        runtime,
        "build_formal_low_frequency_template",
        lambda value, key, model, **kwargs: (
            calls.setdefault("lf", (value, key, model, kwargs)) or object()
        ),
    )
    monkeypatch.setattr(
        runtime,
        "build_high_frequency_tail_template",
        lambda value, key, model, **kwargs: (
            calls.setdefault("hf", (value, key, model, kwargs)) or object()
        ),
    )

    def build_update(
        value: object,
        routing: object,
        lf_template: object,
        hf_template: object,
        **kwargs: object,
    ) -> SimpleNamespace:
        calls["update"] = (
            value,
            routing,
            lf_template,
            hf_template,
            kwargs,
        )
        return SimpleNamespace(
            written_latent=written,
            hf_tail_update=torch.full_like(latent, 0.01),
            routing_mode=kwargs["routing_mode"],
            carrier_mode=kwargs["carrier_mode"],
            strength_multiplier=kwargs["strength_multiplier"],
            hf_tail_effective_l2=0.1,
            combined_effective_l2=0.1,
            combined_relative_l2=0.01,
        )

    monkeypatch.setattr(
        runtime,
        "build_terminal_content_carrier_update",
        build_update,
    )
    selected, record = runtime._build_late_hf_generation_update(
        latent,
        routing="shared-routing",
        key_material="registered-key",
        model_identity_digest="model-digest",
        protocol=protocol,
    )

    assert selected is written
    assert calls["lf"][1:3] == ("registered-key", "model-digest")
    assert calls["hf"][1:3] == ("registered-key", "model-digest")
    assert calls["update"][4] == {
        "routing_mode": "semantic_unit_energy",
        "carrier_mode": "hf_only",
        "strength_multiplier": 8.0,
    }
    assert record["terminal_pre_vae_carrier_applied"] is True
    assert record["terminal_pre_vae_carrier_mode"] == "hf_only"
    assert record["terminal_pre_vae_strength_multiplier"] == 8.0


@pytest.mark.quick
def test_runtime_writes_late_hf_once_without_output_qk_optimizer() -> None:
    source = inspect.getsource(
        runtime._run_semantic_watermark_runtime_with_content_strength
    )

    assert "if step_index == late_hf_callback_step_index:" in source
    assert "if role not in CONTENT_SURVIVAL_REPLAY_ROLES:" in source
    assert source.count("_build_late_hf_generation_update(") == 1
    assert source.count("_build_late_qk_geometry_generation_update(") == 1
    assert source.count("_replay_late_qk_generation_suffix(") == 1
    assert "registered_suffix_objective" in source
    assert "exact_suffix_evaluator" in source
    assert "late_qk_carrier_score" in source
    assert 'if role == "full_nominal_replay":' in source
    assert "if full_role and probe_sign is not None:" in source
    assert "signed_geometry = torch.zeros_like" in source
    assert 'expected_late_hf_write_count = (' in source
    assert '"late_hf_generation_post_pipeline_write_applied": False' in source
    assert '"late_hf_generation_wrong_key_accessed": False' in source
    assert source.count("_build_terminal_registered_qk_sync(") == 0
    assert 'full_payload["terminal_qk_sync_ready"]' not in source
    assert '"final_image_attention_observability"' in source
    assert '"final_image_attention_observability_gate_ready"' in source


@pytest.mark.quick
def test_late_qk_suffix_replay_isolated_scheduler_and_cfg_gradient() -> None:
    torch = pytest.importorskip("torch")

    class Transformer(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.tensor(0.0))
            self.batch_sizes: list[int] = []

        def forward(self, *, hidden_states: object, **_: object) -> object:
            self.batch_sizes.append(int(hidden_states.shape[0]))
            return (hidden_states * 0.1 + self.anchor * 0.0,)

    class Scheduler:
        total_calls = 0

        def __init__(self) -> None:
            self.calls = 0

        def step(
            self,
            model_output: object,
            _: object,
            sample: object,
            *,
            return_dict: bool,
        ) -> object:
            assert return_dict is False
            type(self).total_calls += 1
            self.calls += 1
            return (sample + model_output,)

    transformer = Transformer()
    scheduler = Scheduler()
    pipeline = SimpleNamespace(
        transformer=transformer,
        scheduler=scheduler,
    )
    latent = torch.ones(
        (1, 2, 2, 2),
        dtype=torch.float32,
        requires_grad=True,
    )
    replayed = runtime._replay_late_qk_generation_suffix(
        pipeline,
        latent,
        timestep=torch.tensor(1.0),
        prompt_embeds=torch.ones((1, 1, 1)),
        pooled_prompt_embeds=torch.ones((1, 1)),
        negative_prompt_embeds=torch.zeros((1, 1, 1)),
        negative_pooled_prompt_embeds=torch.zeros((1, 1)),
        guidance_scale=4.5,
    )
    replayed.sum().backward()

    assert transformer.batch_sizes == [2]
    assert scheduler.calls == 0
    assert Scheduler.total_calls == 1
    assert latent.grad is not None
    assert bool((latent.grad != 0).all())


@pytest.mark.quick
def test_formal_full_replay_binds_cfg_negative_embeddings_before_suffix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    config = runtime.SemanticWatermarkRuntimeConfig(
        standard_attack_profiles=(),
        diffusion_attacks_enabled=False,
    )
    protocol = load_content_survival_direction_protocol(REPOSITORY_ROOT)
    prompt_calls: list[str] = []
    positive_prompt = torch.full((1, 1, 1), 1.0)
    positive_pooled = torch.full((1, 1), 2.0)
    negative_prompt = torch.full((1, 1, 1), -1.0)
    negative_pooled = torch.full((1, 1), -2.0)

    class SuffixReached(RuntimeError):
        pass

    class Pipeline:
        _execution_device = torch.device("cpu")
        vae_scale_factor = 1
        transformer = SimpleNamespace(
            config=SimpleNamespace(in_channels=1),
            dtype=torch.float32,
        )
        vae = object()
        image_processor = object()
        scheduler = SimpleNamespace(timesteps=torch.arange(20))

        def __call__(
            self,
            *,
            latents: object,
            callback_on_step_end: object,
            **_: object,
        ) -> object:
            value = latents
            for step_index in (9, 10, 18):
                value = callback_on_step_end(
                    self,
                    step_index,
                    torch.tensor(float(step_index)),
                    {"latents": value},
                )["latents"]
            return SimpleNamespace(images=value)

    def prompt_embeddings(_: object, prompt: str) -> tuple[object, object]:
        prompt_calls.append(prompt)
        if prompt == config.prompt:
            return positive_prompt, positive_pooled
        if prompt == config.negative_prompt:
            return negative_prompt, negative_pooled
        raise AssertionError("unexpected prompt identity")

    latent = torch.ones((1, 1, 2, 2), dtype=torch.float32)
    routing = SimpleNamespace(
        routing_identity_digest="r" * 64,
        writable_capacity_map=torch.ones((1, 1, 2, 2)),
    )

    def content_observation(**kwargs: object) -> object:
        kwargs["vae_decoder"](kwargs["current_scheduler_latent"])
        return SimpleNamespace(routing=routing)

    content_update = SimpleNamespace(
        lf_update=torch.full_like(latent, 0.001),
        hf_tail_update=torch.full_like(latent, 0.001),
        lf_nominal_strength=0.001,
        hf_tail_nominal_strength=0.001,
    )
    geometry = SimpleNamespace(
        geometry_update=torch.full_like(latent, 0.0001),
        geometry_update_digest="g" * 64,
        qk_atomic_records_digest="q" * 64,
    )
    write_result = SimpleNamespace(
        written_latent=latent + 0.002,
        accepted_common_scale=1.0,
        lf_effective_l2=0.001,
        hf_tail_effective_l2=0.001,
        geometry_effective_l2=0.0,
        combined_effective_l2=0.002,
        actual_dtype_write_digest="w" * 64,
        write_identity_digest="i" * 64,
    )
    context = runtime.SemanticWatermarkRuntimeContext(
        pipeline=Pipeline(),
        prompt_saliency_runtime=object(),
        attention_modules=(),
        unconditional_prompt=None,
        unconditional_pooled=None,
        runtime_versions={},
    )

    monkeypatch.setattr(runtime, "_require_full_content_runtime_config", lambda *_: None)
    monkeypatch.setattr(
        runtime,
        "_require_calibration_content_strength_multiplier",
        lambda *_a, **_k: None,
    )
    monkeypatch.setattr(
        runtime,
        "load_content_survival_direction_protocol",
        lambda *_: protocol,
    )
    monkeypatch.setattr(
        runtime,
        "semantic_conditioned_latent_method_definition_digest",
        lambda: "m" * 64,
    )
    monkeypatch.setattr(
        runtime,
        "build_content_survival_runtime_method_identity",
        lambda **_: {"composite_runtime_method_identity_digest": "c" * 64},
    )
    monkeypatch.setattr(
        runtime,
        "build_canonical_sd35_base_latent",
        lambda **_: (
            latent,
            {
                "base_latent_content_digest_random": "b" * 64,
                "base_latent_identity_digest_random": "a" * 64,
            },
        ),
    )
    monkeypatch.setattr(
        runtime,
        "_content_runtime_prompt_embeddings",
        prompt_embeddings,
    )
    monkeypatch.setattr(
        runtime,
        "build_public_probe_identity",
        lambda *_: {"identity": "public"},
    )
    monkeypatch.setattr(
        runtime,
        "_decode_content_runtime_latent",
        lambda *_: torch.ones((1, 3, 2, 2)),
    )
    monkeypatch.setattr(
        runtime,
        "build_content_observation_routing",
        content_observation,
    )
    monkeypatch.setattr(
        runtime,
        "build_formal_low_frequency_template",
        lambda *_a, **_k: object(),
    )
    monkeypatch.setattr(
        runtime,
        "build_high_frequency_tail_template",
        lambda *_a, **_k: object(),
    )
    monkeypatch.setattr(
        runtime,
        "build_content_carrier_update",
        lambda **_: content_update,
    )
    monkeypatch.setattr(
        runtime,
        "_transformer_forward_function",
        lambda *_: lambda value: value,
    )
    monkeypatch.setattr(
        runtime,
        "_build_attention_geometry_sync_update_with_evidence",
        lambda **_: (geometry, {}),
    )
    monkeypatch.setattr(
        runtime,
        "materialize_content_survival_probe",
        lambda value, *_a, **_k: (
            value + 0.001,
            {"probe_sign": _k["sign"]},
        ),
    )
    monkeypatch.setattr(
        runtime,
        "formal_dual_chain_write_budget",
        lambda: SimpleNamespace(combined_relative_l2_limit=0.014),
    )
    monkeypatch.setattr(
        runtime,
        "compose_dual_chain_update_once",
        lambda *_a, **_k: write_result,
    )
    monkeypatch.setattr(
        runtime,
        "_build_late_hf_generation_update",
        lambda value, **_: (
            value + 0.001,
            {
                "terminal_pre_vae_carrier_applied": True,
                "terminal_pre_vae_carrier_mode": "hf_only",
            },
        ),
    )
    monkeypatch.setattr(
        runtime,
        "_extract_terminal_pipeline_latent",
        lambda output: output.images,
    )
    monkeypatch.setattr(
        runtime,
        "_decode_content_runtime_latent_image",
        lambda *_: object(),
    )
    monkeypatch.setattr(
        runtime,
        "canonical_rgb_uint8_content_record",
        lambda *_: {
            "image_rgb_uint8_content_sha256": "f" * 64,
            "image_width": 2,
            "image_height": 2,
        },
    )
    monkeypatch.setattr(
        runtime,
        "_build_image_only_measurement_config",
        lambda *_: object(),
    )
    monkeypatch.setattr(
        runtime,
        "_image_attention_extractor",
        lambda *_a, **_k: lambda _image: ({"q": 1},),
    )
    monkeypatch.setattr(
        runtime,
        "resolve_detection_key_material_and_identity",
        lambda key, _role: (key, {}),
    )
    monkeypatch.setattr(
        runtime,
        "_public_detection_noise_evidence_cursor",
        lambda *_: 0,
    )
    monkeypatch.setattr(
        runtime,
        "_discard_public_detection_noise_evidence_since",
        lambda *_: None,
    )
    monkeypatch.setattr(
        runtime,
        "measure_image_only_watermark",
        lambda **_: SimpleNamespace(
            to_record=lambda: {
                "content_score": 0.1,
                "measurement_digest": "d" * 64,
            }
        ),
    )
    monkeypatch.setattr(
        runtime,
        "select_shared_content_survival_sign",
        lambda *_a, **_k: {"selected_sign": 1},
    )
    monkeypatch.setattr(
        runtime,
        "_terminal_registered_qk_score",
        lambda *_a, **_k: (
            {
                "blind_attention_score": 0.1,
                "carrier_paired_attention_score": 0.1,
                "qk_atomic_content_digest": "q" * 64,
            },
            (1.0,),
        ),
    )

    def reach_suffix(
        _pipeline: object,
        _candidate: object,
        **kwargs: object,
    ) -> object:
        assert torch.equal(kwargs["prompt_embeds"], positive_prompt)
        assert torch.equal(kwargs["pooled_prompt_embeds"], positive_pooled)
        assert torch.equal(kwargs["negative_prompt_embeds"], negative_prompt)
        assert torch.equal(
            kwargs["negative_pooled_prompt_embeds"],
            negative_pooled,
        )
        raise SuffixReached

    monkeypatch.setattr(
        runtime,
        "_replay_late_qk_generation_suffix",
        reach_suffix,
    )

    with pytest.raises(SuffixReached):
        runtime._run_semantic_watermark_runtime_with_content_strength(
            config,
            references=runtime.ContentRoutingReferenceScalars(1.0, 1.0, 1.0),
            verified_formal_execution_lock={},
            repository_root=REPOSITORY_ROOT,
            runtime_context=context,
            content_strength_common_multiplier=1.0,
            calibration_content_strength_sensitivity=False,
        )
    assert prompt_calls == [config.prompt, config.negative_prompt]


@pytest.mark.quick
def test_late_qk_geometry_uses_exact_suffix_gains_and_fixed_budget() -> None:
    torch = pytest.importorskip("torch")
    protocol = load_content_survival_direction_protocol(REPOSITORY_ROOT)
    latent = torch.ones((1, 2, 2, 2), dtype=torch.float32)
    late_hf = (latent + 0.01).detach().requires_grad_(True)
    objective_inputs: list[object] = []
    evaluated_candidates: list[object] = []

    def registered_suffix_objective(candidate: object) -> object:
        objective_inputs.append(candidate)
        return candidate.square().mean()

    def exact_suffix_evaluator(candidate: object) -> dict[str, object]:
        evaluated_candidates.append(candidate)
        return {
            "blind_attention_score": 0.2002,
            "carrier_paired_attention_score": 0.3002,
            "qk_atomic_content_digest": "b" * 64,
            "suffix_terminal_latent_content_sha256": "c" * 64,
            "output_image_rgb_uint8_content_sha256": "d" * 64,
        }

    written, record = runtime._build_late_qk_geometry_generation_update(
        latent,
        late_hf_latent=late_hf,
        geometry_capacity_map=torch.ones((1, 1, 2, 2)),
        registered_suffix_objective=registered_suffix_objective,
        exact_suffix_evaluator=exact_suffix_evaluator,
        carrier_reference_score={
            "blind_attention_score": 0.2,
            "carrier_paired_attention_score": 0.3,
            "qk_atomic_content_digest": "a" * 64,
        },
        minimum_gain=1.0e-4,
        protocol=protocol,
    )

    assert not torch.equal(written, late_hf)
    assert len(objective_inputs) == 1
    assert len(record["late_qk_geometry_candidate_records"]) == 9
    assert len(evaluated_candidates) == sum(
        candidate["candidate_score_evaluated"]
        for candidate in record["late_qk_geometry_candidate_records"]
    )
    assert late_hf.grad is None
    assert record["late_qk_geometry_ready"] is True
    assert record["late_qk_geometry_failure_reason"] == ""
    assert record["late_qk_geometry_applied"] is True
    assert record["late_qk_geometry_wrong_key_accessed"] is False
    assert record["late_qk_geometry_actual_dtype_relative_l2"] <= 0.001
    assert record["late_qk_geometry_combined_actual_dtype_relative_l2"] <= 0.014
    assert record["late_qk_geometry_backtracking_index"] == next(
        candidate["candidate_index"]
        for candidate in record["late_qk_geometry_candidate_records"]
        if candidate["candidate_acceptance_ready"]
    )
    assert record["late_qk_geometry_blind_full_vs_carrier_gain"] == (
        pytest.approx(0.0002)
    )
    assert record[
        "late_qk_geometry_carrier_paired_full_vs_carrier_gain"
    ] == pytest.approx(0.0002)
    assert all(
        candidate["wrong_key_accessed"] is False
        for candidate in record["late_qk_geometry_candidate_records"]
    )


@pytest.mark.quick
@pytest.mark.parametrize(
    ("direction_ready", "expected_reason"),
    [
        (False, "late_qk_geometry_registered_suffix_direction_not_ready"),
        (True, "late_qk_geometry_no_exact_suffix_candidate_passed"),
    ],
)
def test_late_qk_geometry_method_level_no_candidate_retains_hf_baseline(
    direction_ready: bool,
    expected_reason: str,
) -> None:
    torch = pytest.importorskip("torch")
    protocol = load_content_survival_direction_protocol(REPOSITORY_ROOT)
    latent = torch.ones((1, 2, 2, 2), dtype=torch.float32)
    late_hf = latent + 0.01
    evaluated_candidates: list[object] = []

    def registered_suffix_objective(candidate: object) -> object:
        return candidate.sum() * (1.0 if direction_ready else 0.0)

    def exact_suffix_evaluator(candidate: object) -> dict[str, object]:
        evaluated_candidates.append(candidate)
        return {
            "blind_attention_score": 0.20005,
            "carrier_paired_attention_score": 0.30005,
            "qk_atomic_content_digest": "b" * 64,
            "suffix_terminal_latent_content_sha256": "c" * 64,
            "output_image_rgb_uint8_content_sha256": "d" * 64,
        }

    written, record = runtime._build_late_qk_geometry_generation_update(
        latent,
        late_hf_latent=late_hf,
        geometry_capacity_map=torch.ones((1, 1, 2, 2)),
        registered_suffix_objective=registered_suffix_objective,
        exact_suffix_evaluator=exact_suffix_evaluator,
        carrier_reference_score={
            "blind_attention_score": 0.2,
            "carrier_paired_attention_score": 0.3,
            "qk_atomic_content_digest": "a" * 64,
        },
        minimum_gain=1.0e-4,
        protocol=protocol,
    )

    assert torch.equal(written, late_hf)
    assert record["late_qk_geometry_ready"] is False
    assert record["late_qk_geometry_failure_reason"] == expected_reason
    assert record["late_qk_geometry_applied"] is False
    assert record["late_qk_geometry_write_count"] == 0
    assert record["late_qk_geometry_wrong_key_accessed"] is False
    assert record["late_qk_geometry_relation_score_after"] == pytest.approx(
        record["late_qk_geometry_relation_score_before"]
    )
    assert len(evaluated_candidates) == sum(
        candidate["candidate_score_evaluated"]
        for candidate in record["late_qk_geometry_candidate_records"]
    )
    assert len(record["late_qk_geometry_candidate_records"]) == (
        9 if direction_ready else 0
    )
