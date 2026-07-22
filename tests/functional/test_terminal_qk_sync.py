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
        "content_survival_direction_late_hf_generation"
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
    assert 'expected_late_hf_write_count = (' in source
    assert '"late_hf_generation_post_pipeline_write_applied": False' in source
    assert '"late_hf_generation_wrong_key_accessed": False' in source
    assert source.count("_build_terminal_registered_qk_sync(") == 0
    assert 'full_payload["terminal_qk_sync_ready"]' not in source
    assert '"final_image_attention_observability"' in source
    assert '"final_image_attention_observability_gate_ready"' in source
