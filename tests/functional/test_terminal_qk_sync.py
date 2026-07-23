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
        "stable_token_fraction": 0.5,
        "unstable_pair_weight": 0.25,
        "acceptance": "first_strict_registered_qk_improvement",
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
def test_late_qk_geometry_uses_registered_gradient_and_fixed_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    protocol = load_content_survival_direction_protocol(REPOSITORY_ROOT)
    latent = torch.ones((1, 2, 2, 2), dtype=torch.float32)
    late_hf = latent + 0.01
    calls: dict[str, object] = {}

    class Recorder:
        records = (("layer", torch.ones(1), (0,)),)

        def clear(self) -> None:
            calls["clear_count"] = int(calls.get("clear_count", 0)) + 1

    evidence = SimpleNamespace(
        gradient=torch.ones_like(latent),
        score_before=0.2,
        stable_pair_weights=object(),
        qk_atomic_content_digest="a" * 64,
    )

    def gradient(*args: object, **kwargs: object) -> object:
        calls["gradient"] = (args, kwargs)
        return evidence

    def score(*args: object, **kwargs: object) -> object:
        calls["score"] = (args, kwargs)
        return torch.tensor(0.3)

    monkeypatch.setattr(runtime, "compute_attention_geometry_gradient", gradient)
    monkeypatch.setattr(runtime, "attention_geometry_score", score)
    monkeypatch.setattr(
        runtime,
        "build_attention_relation_graph_identity",
        lambda *args, **kwargs: SimpleNamespace(
            qk_atomic_content_digest="b" * 64
        ),
    )
    written, record = runtime._build_late_qk_geometry_generation_update(
        latent,
        late_hf_latent=late_hf,
        geometry_capacity_map=torch.ones((1, 1, 2, 2)),
        transformer_forward=lambda value: value,
        recorder=Recorder(),
        key_material="registered-key",
        component_weights=(0.25, 0.25, 0.25, 0.25),
        protocol=protocol,
    )

    assert not torch.equal(written, late_hf)
    assert calls["gradient"][0][3] == "registered-key"
    assert calls["score"][0][1] == "registered-key"
    assert record["late_qk_geometry_ready"] is True
    assert record["late_qk_geometry_failure_reason"] == ""
    assert record["late_qk_geometry_applied"] is True
    assert record["late_qk_geometry_wrong_key_accessed"] is False
    assert record["late_qk_geometry_actual_dtype_relative_l2"] <= 0.001
    assert record["late_qk_geometry_combined_actual_dtype_relative_l2"] <= 0.014
    assert record["late_qk_geometry_relation_score_after"] > record[
        "late_qk_geometry_relation_score_before"
    ]


@pytest.mark.quick
@pytest.mark.parametrize(
    ("direction_ready", "expected_reason"),
    [
        (False, "late_qk_geometry_registered_direction_not_ready"),
        (True, "late_qk_geometry_no_strict_actual_dtype_improvement"),
    ],
)
def test_late_qk_geometry_method_level_no_candidate_retains_hf_baseline(
    monkeypatch: pytest.MonkeyPatch,
    direction_ready: bool,
    expected_reason: str,
) -> None:
    torch = pytest.importorskip("torch")
    protocol = load_content_survival_direction_protocol(REPOSITORY_ROOT)
    latent = torch.ones((1, 2, 2, 2), dtype=torch.float32)
    late_hf = latent + 0.01

    class Recorder:
        records = (("layer", torch.ones(1), (0,)),)

        def clear(self) -> None:
            return None

    monkeypatch.setattr(
        runtime,
        "compute_attention_geometry_gradient",
        lambda *args, **kwargs: SimpleNamespace(
            gradient=(
                torch.ones_like(latent)
                if direction_ready
                else torch.zeros_like(latent)
            ),
            score_before=0.2,
            stable_pair_weights=object(),
            qk_atomic_content_digest="a" * 64,
        ),
    )
    monkeypatch.setattr(
        runtime,
        "attention_geometry_score",
        lambda *args, **kwargs: torch.tensor(0.1),
    )
    monkeypatch.setattr(
        runtime,
        "build_attention_relation_graph_identity",
        lambda *args, **kwargs: SimpleNamespace(
            qk_atomic_content_digest="b" * 64
        ),
    )

    written, record = runtime._build_late_qk_geometry_generation_update(
        latent,
        late_hf_latent=late_hf,
        geometry_capacity_map=torch.ones((1, 1, 2, 2)),
        transformer_forward=lambda value: value,
        recorder=Recorder(),
        key_material="registered-key",
        component_weights=(0.25, 0.25, 0.25, 0.25),
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
