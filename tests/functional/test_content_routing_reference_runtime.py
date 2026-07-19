"""CPU properties for the real content-routing reference producer boundary."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from experiments.protocol.content_routing_reference_registry_payload import (
    assemble_content_routing_reference_registry_payload,
)
from experiments.runners import content_routing_reference_runtime as runtime
from main.core.digest import build_stable_digest, stable_json_dumps
from scripts import gpu_method_qualification_host_workflow as host


pytestmark = pytest.mark.quick


def _candidate_bytes() -> bytes:
    observations = [torch.tensor([[[[1.0, 2.0]]]], dtype=torch.float32)] * 3
    return assemble_content_routing_reference_registry_payload(
        method_parameter_partition_id="probe_paper_dev_method_parameter_v1",
        prompt_projection=[
            {"prompt_id": "prompt", "prompt_text_digest": "1" * 64}
        ],
        seed_projection_random=[7],
        generation_input_identity_digests=["2" * 64],
        gradient_observations=[observations[0]],
        response_observations=[observations[1]],
        sensitivity_observations=[observations[2]],
        formal_execution_lock_digest="3" * 64,
        dependency_profile_digest="4" * 64,
        runtime_component_identity_payload={
            "model_id": "stabilityai/stable-diffusion-3.5-medium",
            "model_revision": "b940f670f0eda2d07fbb75229e779da1ad11eb80",
            "dependency_profile_digest": "4" * 64,
            "formal_execution_lock_digest": "3" * 64,
            "vae_preprocess_identity_digest": "5" * 64,
            "scheduler_identity_digest": "6" * 64,
            "content_observation_formula_identity_digest": "7" * 64,
        },
    )


def test_registry_import_and_scalar_validation_do_not_require_torch(
    tmp_path: Path,
) -> None:
    candidate_path = tmp_path / "content_routing_reference_registry.json"
    candidate_path.write_bytes(_candidate_bytes())
    program = """
import builtins
from pathlib import Path
import sys

real_import = builtins.__import__

def reject_torch(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "torch" or name.startswith("torch."):
        raise AssertionError("registry validation imported torch")
    return real_import(name, globals, locals, fromlist, level)

builtins.__import__ = reject_torch
from experiments.protocol.content_routing_reference_registry import (
    ContentRoutingReferenceScalars,
    _strict_json_object,
    _validate_content_routing_reference_registry,
)

raw_payload = Path(sys.argv[1]).read_bytes()
registry = _strict_json_object(raw_payload)
scalars = _validate_content_routing_reference_registry(
    registry,
    raw_payload=raw_payload,
    expected_registry_digest=registry[
        "content_routing_reference_registry_digest"
    ],
)
assert type(scalars) is ContentRoutingReferenceScalars
assert scalars.reference_gradient == 2.0
assert scalars.reference_response == 2.0
assert scalars.reference_sensitivity == 2.0
"""
    completed = subprocess.run(
        [sys.executable, "-c", program, str(candidate_path)],
        cwd=Path.cwd(),
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_probe_paper_dev_partition_is_exact_and_isolated() -> None:
    records = runtime._method_parameter_prompts(Path.cwd())
    assert tuple(record.prompt_index for record in records) == (30, 31, 32)
    assert all(record.split == "dev" for record in records)


def test_generation_identity_binds_actual_canonical_base_latent_identity() -> None:
    config = SimpleNamespace(
        model_id="stabilityai/stable-diffusion-3.5-medium",
        model_revision="b940f670f0eda2d07fbb75229e779da1ad11eb80",
        negative_prompt="",
        width=512,
        height=512,
        inference_steps=28,
        guidance_scale=4.5,
    )
    prompt = SimpleNamespace(
        prompt_id="prompt-id",
        prompt_digest="1" * 64,
    )
    first_identity = {
        "base_latent_identity_digest_random": "2" * 64,
        "base_latent_content_digest_random": "3" * 64,
    }
    second_identity = {
        **first_identity,
        "base_latent_content_digest_random": "4" * 64,
    }

    first = runtime._generation_identity(config, prompt, 17, first_identity)
    repeated = runtime._generation_identity(config, prompt, 17, first_identity)
    changed = runtime._generation_identity(config, prompt, 17, second_identity)

    assert first == repeated
    assert first != changed


def test_reference_member_uses_latent_output_and_only_two_explicit_decodes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    latent = torch.ones((1, 2, 4, 6), dtype=torch.float32)
    calls = {"decode": 0, "output_type": None}

    class FakePipeline:
        transformer = SimpleNamespace(
            config=SimpleNamespace(in_channels=2),
            dtype=torch.float32,
        )
        vae_scale_factor = 2
        _execution_device = torch.device("cpu")

        def __call__(self, **kwargs):
            calls["output_type"] = kwargs["output_type"]
            callback = kwargs["callback_on_step_end"]
            callback(self, 9, None, {"latents": latent})
            callback(self, 10, None, {"latents": latent + 1.0})
            return SimpleNamespace(images=latent)

    monkeypatch.setattr(
        runtime,
        "build_canonical_sd35_base_latent",
        lambda **_kwargs: (
            latent,
            {"base_latent_identity_digest_random": "2" * 64},
        ),
    )

    def decode(_pipeline, _latent):
        calls["decode"] += 1
        return torch.ones((1, 3, 8, 12), dtype=torch.float32)

    monkeypatch.setattr(runtime, "_decode_latent", decode)
    monkeypatch.setattr(
        runtime,
        "_measure_gradient_magnitude",
        lambda _image: torch.ones((1, 1, 8, 12), dtype=torch.float32),
    )
    monkeypatch.setattr(
        runtime,
        "_measure_adjacent_latent_relative_response",
        lambda _z9, _z10: torch.ones((1, 1, 4, 6), dtype=torch.float32),
    )

    def sensitivity(z10, x10, decoder, _identity):
        decoder(z10)
        assert x10.shape == (1, 3, 8, 12)
        return SimpleNamespace(
            local_difference_sensitivity=torch.ones(
                (1, 1, 4, 6), dtype=torch.float32
            )
        )

    monkeypatch.setattr(
        runtime,
        "_measure_public_probe_local_sensitivity",
        sensitivity,
    )
    config = SimpleNamespace(
        model_id="stabilityai/stable-diffusion-3.5-medium",
        model_revision="b940f670f0eda2d07fbb75229e779da1ad11eb80",
        height=8,
        width=12,
        negative_prompt="",
        inference_steps=20,
        guidance_scale=4.5,
    )
    prompt = SimpleNamespace(prompt_text="a governed reference prompt")

    observations, _identity = runtime._observe_member(
        pipeline=FakePipeline(),
        config=config,
        prompt_record=prompt,
        seed=17,
    )

    assert calls == {"decode": 2, "output_type": "latent"}
    assert tuple(value.dtype for value in observations) == (
        torch.float32,
        torch.float32,
        torch.float32,
    )


def test_reference_writer_requires_identical_replay_and_never_promotes_fixed_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = _candidate_bytes()
    calls = {"assemble": 0, "pipeline": 0}
    monkeypatch.setenv("SLM_WM_KEY_MATERIAL", "registered-root-key")
    monkeypatch.setattr(
        runtime.repository_environment,
        "require_published_formal_execution_lock",
        lambda _root: {"formal_execution_lock_digest": "3" * 64},
    )
    monkeypatch.setattr(
        runtime,
        "require_dependency_profile_ready",
        lambda _name: SimpleNamespace(formal_ready=True, profile_digest="4" * 64),
    )
    monkeypatch.setattr(runtime, "build_method_config", lambda _root: object())
    monkeypatch.setattr(runtime, "_method_parameter_prompts", lambda _root: (object(),))

    def load(_config):
        calls["pipeline"] += 1
        return object(), {}

    def assemble(**_kwargs):
        calls["assemble"] += 1
        return candidate, {"member_identity": "8" * 64}

    monkeypatch.setattr(runtime, "load_pipeline", load)
    monkeypatch.setattr(runtime, "_assemble_pass", assemble)

    result = runtime.write_content_routing_reference_runtime_outputs(
        root=tmp_path,
    )

    assert calls == {"assemble": 2, "pipeline": 1}
    assert result["qualification_ready"] is True
    assert result["supports_paper_claim"] is False
    candidate_path = tmp_path / result["candidate_path"]
    assert candidate_path.read_bytes() == candidate
    assert not (tmp_path / "configs/content_routing_reference_registry.json").exists()


def test_reference_writer_replay_mismatch_fails_before_candidate_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = _candidate_bytes()
    monkeypatch.setenv("SLM_WM_KEY_MATERIAL", "registered-root-key")
    monkeypatch.setattr(
        runtime.repository_environment,
        "require_published_formal_execution_lock",
        lambda _root: {"formal_execution_lock_digest": "3" * 64},
    )
    monkeypatch.setattr(
        runtime,
        "require_dependency_profile_ready",
        lambda _name: SimpleNamespace(formal_ready=True, profile_digest="4" * 64),
    )
    monkeypatch.setattr(runtime, "build_method_config", lambda _root: object())
    monkeypatch.setattr(runtime, "_method_parameter_prompts", lambda _root: (object(),))
    monkeypatch.setattr(runtime, "load_pipeline", lambda _config: (object(), {}))
    sequence = iter(
        ((candidate, {"identity": "a"}), (candidate, {"identity": "b"}))
    )
    monkeypatch.setattr(runtime, "_assemble_pass", lambda **_kwargs: next(sequence))

    with pytest.raises(RuntimeError, match="replay"):
        runtime.write_content_routing_reference_runtime_outputs(root=tmp_path)

    assert not (tmp_path / "outputs/content_routing_reference_runtime").exists()


def test_reference_host_revalidates_candidate_without_fixed_loader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "repository"
    candidate = _candidate_bytes()
    registry = json.loads(candidate)
    semantic_digest = registry["content_routing_reference_registry_digest"]
    run_dir = root / "outputs/content_routing_reference_runtime" / semantic_digest
    run_dir.mkdir(parents=True)
    candidate_path = run_dir / "content_routing_reference_registry.json"
    candidate_path.write_bytes(candidate)
    report = {
        "report_schema": "content_routing_reference_runtime_qualification_v1",
        "schema_version": 1,
        "qualification_ready": True,
        "producer_replay_byte_identical": True,
        "content_routing_reference_registry_digest": semantic_digest,
        "content_routing_reference_registry_file_sha256": host._file_sha256(
            candidate_path
        ),
        "candidate_path": candidate_path.relative_to(root).as_posix(),
        "supports_paper_claim": False,
    }
    report["qualification_digest"] = build_stable_digest(report)
    report_path = run_dir / "content_routing_reference_runtime_qualification.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    invocation = {
        "report_schema": host.CONTENT_ROUTING_REFERENCE_INVOCATION_SCHEMA,
        "schema_version": 1,
        "candidate_path": candidate_path.relative_to(root).as_posix(),
        "candidate_file_sha256": host._file_sha256(candidate_path),
        "candidate_registry_digest": semantic_digest,
        "qualification_report_path": report_path.relative_to(root).as_posix(),
        "qualification_report_sha256": host._file_sha256(report_path),
        "qualification_ready": True,
        "supports_paper_claim": False,
    }

    def execute(_profile, _argv, *, execution_report_path, repository_root):
        isolated = {
            "formal_execution_commit": "a" * 40,
            "formal_execution_lock_revalidated_after_child": True,
            "profile_id": host.SCIENTIFIC_PROFILE_ID,
            "profile_digest": "1" * 64,
            "direct_requirements_digest": "2" * 64,
            "complete_hash_lock_digest": "3" * 64,
            "dependency_environment_report_path": "outputs/dependency.json",
            "dependency_environment_report_digest": "4" * 64,
            "python_executable_path": "/managed/python",
            "python_executable_sha256": "5" * 64,
            "execution": {"return_code": 0, "stdout": json.dumps(invocation), "stderr": ""},
            "decision": "pass",
            "supports_paper_claim": False,
        }
        path = Path(execution_report_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(isolated), encoding="utf-8")
        return isolated, path

    monkeypatch.setattr(host, "execute_isolated_scientific_command", execute)
    result = host.run_content_routing_reference_host_workflow(
        root=root,
        repository_commit="a" * 40,
        result_path="outputs/host/reference.json",
        output_root="outputs/content_routing_reference_runtime",
    )

    assert result["decision"] == "pass"
    assert result["workflow_summary"][
        "content_routing_reference_registry_digest"
    ] == semantic_digest
    assert not (root / "configs/content_routing_reference_registry.json").exists()


@pytest.mark.parametrize(
    "mutation",
    ("duplicate_key", "schema", "scalar", "report_digest", "return_code"),
)
def test_reference_host_rejects_candidate_and_execution_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    root = tmp_path / "repository"
    candidate = _candidate_bytes()
    registry = json.loads(candidate)
    if mutation in {"schema", "scalar"}:
        registry.pop("content_routing_reference_registry_digest")
        if mutation == "schema":
            registry["registry_schema"] = "wrong_schema"
        else:
            registry["reference_gradient"] = 1.1
        registry["content_routing_reference_registry_digest"] = (
            build_stable_digest(registry)
        )
        candidate = stable_json_dumps(registry).encode("utf-8") + b"\n"
    elif mutation == "duplicate_key":
        candidate = candidate.replace(
            b'{"content_routing_reference_populations"',
            b'{"registry_schema":"duplicate","content_routing_reference_populations"',
            1,
        )
    semantic_digest = json.loads(candidate)[
        "content_routing_reference_registry_digest"
    ]
    run_dir = root / "outputs/content_routing_reference_runtime" / semantic_digest
    run_dir.mkdir(parents=True)
    candidate_path = run_dir / "content_routing_reference_registry.json"
    candidate_path.write_bytes(candidate)
    report = {
        "report_schema": "content_routing_reference_runtime_qualification_v1",
        "schema_version": 1,
        "qualification_ready": True,
        "producer_replay_byte_identical": True,
        "content_routing_reference_registry_digest": semantic_digest,
        "content_routing_reference_registry_file_sha256": host._file_sha256(
            candidate_path
        ),
        "candidate_path": candidate_path.relative_to(root).as_posix(),
        "supports_paper_claim": False,
    }
    report["qualification_digest"] = build_stable_digest(report)
    if mutation == "report_digest":
        report["qualification_digest"] = "0" * 64
    report_path = run_dir / "content_routing_reference_runtime_qualification.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    invocation = {
        "report_schema": host.CONTENT_ROUTING_REFERENCE_INVOCATION_SCHEMA,
        "schema_version": 1,
        "candidate_path": candidate_path.relative_to(root).as_posix(),
        "candidate_file_sha256": host._file_sha256(candidate_path),
        "candidate_registry_digest": semantic_digest,
        "qualification_report_path": report_path.relative_to(root).as_posix(),
        "qualification_report_sha256": host._file_sha256(report_path),
        "qualification_ready": True,
        "supports_paper_claim": False,
    }

    def execute(_profile, _argv, *, execution_report_path, repository_root):
        del repository_root
        passed = mutation != "return_code"
        isolated = {
            "formal_execution_commit": "a" * 40,
            "formal_execution_lock_revalidated_after_child": True,
            "profile_id": host.SCIENTIFIC_PROFILE_ID,
            "execution": {
                "return_code": 0 if passed else 1,
                "stdout": json.dumps(invocation),
                "stderr": "",
            },
            "decision": "pass" if passed else "fail",
            "supports_paper_claim": False,
        }
        path = Path(execution_report_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(isolated), encoding="utf-8")
        return isolated, path

    monkeypatch.setattr(host, "execute_isolated_scientific_command", execute)
    with pytest.raises(ValueError):
        host.run_content_routing_reference_host_workflow(
            root=root,
            repository_commit="a" * 40,
            result_path="outputs/host/reference.json",
            output_root="outputs/content_routing_reference_runtime",
        )

    assert not (root / "configs/content_routing_reference_registry.json").exists()
