"""CPU protocol tests for the executable content-runtime GPU smoke path."""

from __future__ import annotations

from dataclasses import replace
import inspect
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys
from typing import Any

import pytest
import torch

from experiments.protocol.content_routing_reference_quantile import (
    ContentRoutingReferenceScalars,
)
from experiments.runners import semantic_watermark_runtime as runtime


pytestmark = pytest.mark.quick


def test_formal_runtime_records_fixed_registry_reference_role() -> None:
    """正式生成链不得把 fixed registry 输入误标成 smoke reference。"""

    source = inspect.getsource(runtime.run_semantic_watermark_runtime)
    assert "fixed_content_routing_reference_registry" in source
    assert "explicit_smoke_only_unqualified" not in source


class _Attention(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.to_q = torch.nn.Linear(1, 1, bias=False)
        self.to_k = torch.nn.Linear(1, 1, bias=False)
        self.heads = 1


class _Pipeline:
    def __init__(self, config: runtime.SemanticWatermarkRuntimeConfig) -> None:
        self._execution_device = torch.device("cpu")
        self.vae = SimpleNamespace(set_attn_processor=lambda _value: None)
        attention = {_name: _Attention() for _name in config.attention_module_names}
        parameter = torch.nn.Parameter(torch.ones(()))
        self.transformer = SimpleNamespace(
            parameters=lambda: iter((parameter,)),
            named_modules=lambda: iter(attention.items()),
        )

    def encode_prompt(self, **_kwargs: Any) -> tuple[torch.Tensor, ...]:
        value = torch.zeros((1, 1, 1))
        return value, value, value, value


def test_unmigrated_production_variants_fail_before_pipeline_or_writer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """新正式链不得静默把消融配置当作full_dual_chain执行。"""

    base = runtime.SemanticWatermarkRuntimeConfig()
    variants = (
        replace(base, semantic_routing_enabled=False),
        replace(base, branch_risk_mode="shared_global"),
        replace(base, lf_enabled=False),
        replace(base, tail_robust_enabled=False),
        replace(
            base,
            attention_geometry_enabled=False,
            image_alignment_enabled=False,
        ),
        replace(base, tail_truncation_enabled=False),
        replace(
            base,
            attention_relation_component_weights=(0.0, 1 / 3, 1 / 3, 1 / 3),
        ),
        replace(
            base,
            risk_parameter_protocol="single_model_internal_sensitivity",
            lf_content_risk_config=replace(
                base.lf_content_risk_config,
                budget_gain=base.lf_content_risk_config.budget_gain + 0.001,
            ),
        ),
    )
    calls = {"pipeline": 0, "writer_core": 0}

    def load_components(*_args: Any, **_kwargs: Any) -> Any:
        calls["pipeline"] += 1
        raise AssertionError("pipeline must not load")

    def run_core(*_args: Any, **_kwargs: Any) -> Any:
        calls["writer_core"] += 1
        raise AssertionError("writer core must not run")

    monkeypatch.setattr(
        runtime,
        "_load_content_runtime_smoke_components",
        load_components,
    )
    monkeypatch.setattr(runtime, "run_semantic_watermark_runtime", run_core)
    references = ContentRoutingReferenceScalars(1.0, 1.0, 1.0)

    for index, config in enumerate(variants):
        with pytest.raises(RuntimeError, match="full_dual_chain"):
            runtime.run_content_runtime_smoke(
                config,
                references,
                verified_formal_execution_lock={"lock": True},
                repository_root=tmp_path,
            )
        output_dir = tmp_path / "outputs" / f"variant-{index}"
        with pytest.raises(RuntimeError, match="full_dual_chain"):
            runtime.write_semantic_watermark_runtime_outputs(
                replace(config, output_dir=output_dir.relative_to(tmp_path).as_posix()),
                root=tmp_path,
                references=references,
                verified_formal_execution_lock={"lock": True},
            )
        assert not output_dir.exists()

    assert calls == {"pipeline": 0, "writer_core": 0}


def test_formal_import_does_not_reach_legacy_jacobian_or_tail() -> None:
    """正式runtime与新geometry公开入口不得加载或暴露旧执行依赖。"""

    script = """
import sys
import experiments.runners.semantic_watermark_runtime as runtime
import main.methods as methods
_ = methods.build_attention_geometry_sync_update
assert 'main.methods.subspace.jacobian_nullspace' not in sys.modules
assert not hasattr(runtime, 'exact_jvp')
assert not hasattr(runtime, 'build_tail_robust_template')
assert not hasattr(runtime, 'optimize_attention_geometry_update')
import main.methods.geometry as geometry
import main.methods.subspace as subspace
assert 'optimize_attention_geometry_update' not in geometry.__all__
assert 'exact_jvp' not in subspace.__all__
assert 'solve_psd_conjugate_gradient' not in subspace.__all__
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_smoke_component_loader_excludes_legacy_716_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The smoke branch must load Prompt saliency once and no old 716 runtime."""

    config = runtime.SemanticWatermarkRuntimeConfig()
    pipeline = _Pipeline(config)
    calls = {
        "prompt": 0,
        "legacy_loader": 0,
        "feature": 0,
        "protocol": 0,
        "diffusion": 0,
    }

    monkeypatch.setattr(
        runtime,
        "load_pipeline",
        lambda _config: (pipeline, {"sd35_operator_identity": {"ready": True}}),
    )

    def prompt_loader(*_args: Any, **kwargs: Any) -> Any:
        calls["prompt"] += 1
        assert kwargs["local_files_only"] is True
        assert kwargs["verified_formal_execution_lock"] == {"lock": "verified"}
        return SimpleNamespace(model_identity_digest="a" * 64)

    monkeypatch.setattr(runtime, "load_prompt_saliency_clip_runtime", prompt_loader)
    assert not hasattr(runtime, "load_clip_vision_model")
    assert not hasattr(runtime, "DifferentiableSemanticFeatureRuntime")
    assert not hasattr(runtime, "semantic_feature_protocol_record")
    assert not hasattr(runtime, "DiffusionAttackRuntime")
    monkeypatch.setattr(runtime, "_unconditional_embeddings", lambda *_args: (1, 2))
    monkeypatch.setitem(
        __import__("sys").modules,
        "diffusers.models.attention_processor",
        SimpleNamespace(AttnProcessor=lambda: object()),
    )

    components = runtime._load_content_runtime_smoke_components(
        config,
        verified_formal_execution_lock={"lock": "verified"},
        repository_root="/repository",
    )

    assert calls == {
        "prompt": 1,
        "legacy_loader": 0,
        "feature": 0,
        "protocol": 0,
        "diffusion": 0,
    }
    assert components.prompt_saliency_runtime.model_identity_digest == "a" * 64
    assert "semantic_feature_operator_contract" not in str(components.runtime_versions)
    assert not hasattr(components, "feature_runtime")
    assert not hasattr(components, "diffusion_attack_runtime")


def test_smoke_runtime_executes_only_index10_and_one_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The executable runner captures z9 and writes one actual-dtype z10 candidate."""

    config = runtime.SemanticWatermarkRuntimeConfig(key_material="key")
    latent = torch.ones((1, 1, 1, 1), dtype=torch.float32)
    order: list[str] = []

    class Pipeline:
        _execution_device = torch.device("cpu")
        transformer = SimpleNamespace(config=SimpleNamespace(in_channels=1), dtype=torch.float32)
        vae_scale_factor = 1

        def __call__(self, **kwargs: Any) -> Any:
            callback = kwargs["callback_on_step_end"]
            state = callback(self, 9, torch.tensor(9.0), {"latents": latent.clone()})
            state = callback(self, 10, torch.tensor(10.0), state)
            order.append("pipeline_return")
            assert torch.equal(state["latents"], latent + 1.0)
            return SimpleNamespace(images=["image"])

    components = runtime._ContentRuntimeSmokeComponents(
        pipeline=Pipeline(),
        prompt_saliency_runtime=SimpleNamespace(model_identity_digest="a" * 64),
        attention_modules=(),
        unconditional_prompt=None,
        unconditional_pooled=None,
        runtime_versions={"sd35_operator_identity": {}},
    )
    monkeypatch.setattr(runtime, "_load_content_runtime_smoke_components", lambda *_a, **_k: components)
    monkeypatch.setattr(
        runtime,
        "build_canonical_sd35_base_latent",
        lambda **_k: (
            latent,
            {
                "base_latent_content_digest_random": "1" * 64,
                "base_latent_identity_digest_random": "b" * 64,
            },
        ),
    )
    monkeypatch.setattr(runtime, "_content_runtime_prompt_embeddings", lambda *_a: (None, None))
    monkeypatch.setattr(runtime, "_decode_content_runtime_latent", lambda *_a: torch.zeros((1, 3, 1, 1)))
    monkeypatch.setattr(runtime, "build_public_probe_identity", lambda _revision: {})

    routing = SimpleNamespace(routing_identity_digest="c" * 64)
    observations = SimpleNamespace(routing=routing)

    def observe(**_kwargs: Any) -> Any:
        order.append("observations")
        _kwargs["vae_decoder"](latent)
        return observations

    monkeypatch.setattr(runtime, "build_content_observation_routing", observe)
    monkeypatch.setattr(runtime, "build_formal_low_frequency_template", lambda *_a, **_k: "lf")
    monkeypatch.setattr(runtime, "build_high_frequency_tail_template", lambda *_a, **_k: "hf")
    content = SimpleNamespace(
        lf_update=torch.ones_like(latent),
        hf_tail_update=torch.ones_like(latent),
        method_role="full_dual_chain",
    )
    monkeypatch.setattr(runtime, "build_content_carrier_update", lambda **_k: order.append("content") or content)
    monkeypatch.setattr(runtime, "_transformer_forward_function", lambda *_a: lambda _x: None)

    class Recorder:
        def __init__(self, *_a: Any, **_k: Any) -> None: pass
        def close(self) -> None: order.append("close")

    monkeypatch.setattr(runtime, "DifferentiableAttentionRecorder", Recorder)
    geometry = SimpleNamespace(
        geometry_update=torch.ones_like(latent),
        geometry_update_digest="d" * 64,
        qk_atomic_records_digest="e" * 64,
    )
    monkeypatch.setattr(runtime, "_build_attention_geometry_sync_update_with_evidence", lambda **_k: (order.append("geometry") or geometry, "evidence"))
    write = SimpleNamespace(
        written_latent=latent + 1.0,
        accepted_common_scale=0.5,
        actual_dtype_write_digest="e" * 64,
        lf_effective_l2=1.0e-3,
        hf_tail_effective_l2=1.0e-3,
        geometry_effective_l2=1.0e-3,
        combined_effective_l2=2.0e-3,
    )
    def compose(*_args: Any, **kwargs: Any) -> Any:
        assert kwargs["method_role"] == "full_dual_chain"
        return order.append("write") or write

    monkeypatch.setattr(runtime, "compose_dual_chain_update_once", compose)
    scores = iter(((0.1, "f" * 64), (0.2, "0" * 64)))
    monkeypatch.setattr(runtime, "_evaluate_post_write_geometry_relation", lambda **_k: next(scores))

    image, diagnostic = runtime.run_content_runtime_smoke(
        config,
        ContentRoutingReferenceScalars(1.0, 1.0, 1.0),
        verified_formal_execution_lock={"lock": True},
        repository_root="/repository",
    )
    assert image == "image"
    assert diagnostic["callback_write_index"] == 10
    assert diagnostic["current_image_decode_count"] == 1
    assert diagnostic["public_probe_additional_decode_count"] == 1
    assert diagnostic["actual_dtype_single_write_count"] == 1
    assert diagnostic["base_latent_content_digest_random"] == "1" * 64
    assert diagnostic["base_latent_identity_digest_random"] == "b" * 64
    assert diagnostic["lf_effective_l2"] > 0.0
    assert diagnostic["hf_tail_effective_l2"] > 0.0
    assert diagnostic["geometry_effective_l2"] > 0.0
    assert diagnostic["combined_effective_l2_ready"] is True
    assert diagnostic["post_write_qk_strict_ready"] is True
    assert order[:4] == ["observations", "content", "geometry", "write"]
    assert order.count("write") == 1


def test_content_runtime_randomization_reference_binds_actual_base_latent() -> None:
    """新正式链必须把实际seed、key与base latent身份交给writer manifest。"""

    config = runtime.SemanticWatermarkRuntimeConfig(
        key_material="formal-key",
        seed=12345,
    )
    diagnostic = {
        "base_latent_content_digest_random": "1" * 64,
        "base_latent_identity_digest_random": "2" * 64,
    }
    actual = runtime._content_runtime_formal_randomization_reference(
        config,
        diagnostic,
    )
    identity = {
        "randomization_repeat_id": config.randomization_repeat_id,
        "generation_seed_index": config.generation_seed_index,
        "generation_seed_offset": config.generation_seed_offset,
        "watermark_key_index": config.watermark_key_index,
        "generation_seed_random": config.seed,
        "watermark_key_seed_random": config.watermark_key_seed_random,
        "formal_randomization_protocol_digest": (
            config.formal_randomization_protocol_digest
        ),
        "watermark_key_material_digest_random": runtime.build_stable_digest(
            {"key_material": config.key_material}
        ),
    }
    identity["formal_randomization_identity_digest_random"] = (
        runtime.build_stable_digest(identity)
    )
    assert actual == {
        **identity,
        **diagnostic,
    }


@pytest.mark.quick
@pytest.mark.parametrize("probe_decode_attempts", (0, 2))
def test_smoke_runtime_rejects_incorrect_public_probe_decode_count(
    monkeypatch: pytest.MonkeyPatch,
    probe_decode_attempts: int,
) -> None:
    """The diagnostic must come from the Q decoder closure's actual calls."""

    config = runtime.SemanticWatermarkRuntimeConfig(key_material="key")
    latent = torch.ones((1, 1, 1, 1), dtype=torch.float32)
    decode_calls = 0
    content_calls = 0

    class Pipeline:
        _execution_device = torch.device("cpu")
        transformer = SimpleNamespace(
            config=SimpleNamespace(in_channels=1),
            dtype=torch.float32,
        )
        vae_scale_factor = 1

        def __call__(self, **kwargs: Any) -> Any:
            callback = kwargs["callback_on_step_end"]
            state = callback(self, 9, torch.tensor(9.0), {"latents": latent})
            callback(self, 10, torch.tensor(10.0), state)
            raise AssertionError("invalid decoder counts must fail before output")

    components = runtime._ContentRuntimeSmokeComponents(
        pipeline=Pipeline(),
        prompt_saliency_runtime=SimpleNamespace(model_identity_digest="a" * 64),
        attention_modules=(),
        unconditional_prompt=None,
        unconditional_pooled=None,
        runtime_versions={"sd35_operator_identity": {}},
    )
    monkeypatch.setattr(
        runtime,
        "_load_content_runtime_smoke_components",
        lambda *_args, **_kwargs: components,
    )
    monkeypatch.setattr(
        runtime,
        "build_canonical_sd35_base_latent",
        lambda **_kwargs: (
            latent,
            {
                "base_latent_content_digest_random": "1" * 64,
                "base_latent_identity_digest_random": "b" * 64,
            },
        ),
    )
    monkeypatch.setattr(
        runtime,
        "_content_runtime_prompt_embeddings",
        lambda *_args: (None, None),
    )

    def decode(*_args: Any) -> torch.Tensor:
        nonlocal decode_calls
        decode_calls += 1
        return torch.zeros((1, 3, 1, 1))

    monkeypatch.setattr(runtime, "_decode_content_runtime_latent", decode)
    monkeypatch.setattr(runtime, "build_public_probe_identity", lambda _revision: {})

    def observe(**kwargs: Any) -> Any:
        for _ in range(probe_decode_attempts):
            kwargs["vae_decoder"](latent)
        return SimpleNamespace(routing=SimpleNamespace(routing_identity_digest="c" * 64))

    monkeypatch.setattr(runtime, "build_content_observation_routing", observe)

    def content_builder(**_kwargs: Any) -> Any:
        nonlocal content_calls
        content_calls += 1
        raise AssertionError("content construction must not follow bad decode evidence")

    monkeypatch.setattr(runtime, "build_formal_low_frequency_template", content_builder)

    with pytest.raises(RuntimeError, match="one additional VAE decode"):
        runtime.run_content_runtime_smoke(
            config,
            ContentRoutingReferenceScalars(1.0, 1.0, 1.0),
            verified_formal_execution_lock={"lock": True},
            repository_root="/repository",
        )
    assert decode_calls == (1 if probe_decode_attempts == 0 else 2)
    assert content_calls == 0
