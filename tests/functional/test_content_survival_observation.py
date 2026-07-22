"""CPU contracts for the isolated content-survival observation."""

from __future__ import annotations

from io import BytesIO
import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

from PIL import Image
import pytest
import torch

from experiments.protocol import content_survival_observation as protocol
from experiments.protocol.content_routing_reference_quantile import (
    ContentRoutingReferenceScalars,
)
from experiments.runners import content_survival_observation_runtime as runtime
from experiments.runners.semantic_watermark_runtime import (
    SemanticWatermarkRuntimeConfig,
    SemanticWatermarkRuntimeContext,
    semantic_watermark_runtime_config_digest,
)
from experiments.runtime.scientific_content_binding import (
    canonical_rgb_uint8_content_record,
)
from main.core.digest import build_stable_digest, tensor_content_sha256
from main.methods.carrier.content_update import ContentCarrierUpdateResult
from main.methods.geometry import sync_update as geometry_sync_update
from scripts import run_content_survival_observation as observation_cli
from tests.helpers.formal_execution_lock import build_test_formal_execution_lock


pytestmark = pytest.mark.quick


def _loaded_protocol() -> protocol.ContentSurvivalObservationProtocol:
    return protocol.load_content_survival_observation_protocol(Path("."))


@pytest.mark.parametrize(
    ("carrier_mode", "expected_role", "lf_active", "hf_active"),
    (
        ("lf_only", "lf_only_content", True, False),
        ("hf_only", "hf_tail_only_content", False, True),
        ("dual", "uniform_content_routing", True, True),
    ),
)
def test_carrier_ablation_preserves_nominal_strength_for_geometry_sync(
    carrier_mode: str,
    expected_role: str,
    lf_active: bool,
    hf_active: bool,
) -> None:
    """单载体消融只停用更新，不把正式名义强度伪装为零。"""

    z10 = torch.ones((1, 4, 2, 2), dtype=torch.float32)
    lf_update = torch.full_like(z10, 1.0e-4)
    hf_update = torch.full_like(z10, -5.0e-5)
    source = ContentCarrierUpdateResult(
        geometry_capacity_map=torch.full((1, 1, 2, 2), 0.5),
        lf_direction=torch.ones_like(z10),
        hf_tail_direction=torch.ones_like(z10),
        lf_update=lf_update,
        hf_tail_update=hf_update,
        content_only_latent_float32=z10 + lf_update + hf_update,
        latent_l2=float(torch.linalg.vector_norm(z10.reshape(-1)).item()),
        lf_nominal_strength=0.01,
        hf_tail_nominal_strength=0.006,
        method_role="uniform_content_routing",
    )

    restricted = runtime._restrict_content_update(source, carrier_mode, z10)

    assert restricted.method_role == expected_role
    assert restricted.lf_nominal_strength == source.lf_nominal_strength
    assert restricted.hf_tail_nominal_strength == source.hf_tail_nominal_strength
    assert bool(torch.count_nonzero(restricted.lf_update)) is lf_active
    assert bool(torch.count_nonzero(restricted.hf_tail_update)) is hf_active
    geometry_sync_update._validate_content_update_formula(z10, restricted)


def _execution_identity(character: str = "3") -> dict[str, object]:
    return protocol.build_content_survival_execution_environment_identity(
        {
            "orchestrator_profile_digest": character * 64,
            "orchestrator_complete_hash_lock_digest": "4" * 64,
            "orchestrator_inspection_digest": "5" * 64,
            "scientific_profile_id": "sd35_method_runtime_gpu",
            "scientific_profile_digest": "6" * 64,
            "scientific_direct_requirements_digest": "7" * 64,
            "scientific_complete_hash_lock_digest": "8" * 64,
            "scientific_dependency_environment_report_digest": "9" * 64,
            "scientific_python_executable_sha256": "a" * 64,
        }
    )


def _dependency_environment_fixture(
    tmp_path: Path,
    formal_lock: dict[str, object],
) -> tuple[Path, dict[str, object], dict[str, object]]:
    """Build the minimal sanitized shape observed in the formal failed archive."""

    orchestrator = observation_cli.require_dependency_profile_ready(
        observation_cli.WORKFLOW_ORCHESTRATOR_PROFILE_ID,
        Path("configs/dependency_profile_registry.json"),
    )
    expected_environment = {
        "python_implementation": "CPython",
        "python_version": "3.12.13",
        "operating_system": "linux",
        "machine": "x86_64",
        "accelerator_runtime": "cpu",
        "cuda_version": None,
        "torch_version": None,
        "torchvision_version": None,
        "direct_dependencies": {"uv": "0.11.28"},
        "locked_dependencies": {"uv": "0.11.28"},
    }
    observed_environment = {
        "python_implementation": "CPython",
        "python_version": "3.12.13",
        "operating_system": "linux",
        "machine": "x86_64",
        "torch_module_available": None,
        "torch_module_version": None,
        "torch_cuda_version": None,
        "cuda_available": None,
        "direct_dependencies": {"uv": "0.11.28"},
        "locked_dependencies": {"uv": "0.11.28"},
    }
    inspection = {
        "profile_name": observation_cli.WORKFLOW_ORCHESTRATOR_PROFILE_ID,
        "profile_digest": orchestrator.profile_digest,
        "complete_hash_lock_digest": orchestrator.complete_hash_lock_digest,
        "profile_formal_ready": True,
        "expected_environment": expected_environment,
        "observed_environment": observed_environment,
        "environment_match": True,
        "mismatches": [],
        "readiness_blockers": [],
        "decision": "pass",
    }
    inspection["inspection_digest"] = build_stable_digest(inspection)
    scientific_profile = {
        "profile_id": "sd35_method_runtime_gpu",
        "profile_digest": "6" * 64,
        "direct_requirements_digest": "7" * 64,
        "complete_hash_lock_digest": "8" * 64,
        "complete_hash_lock_dependency_count": 70,
    }
    provision_report = {
        "report_schema": "isolated_dependency_python_provision_report",
        "schema_version": 1,
        "operation_kind": "isolated_python_provision",
        "profile_id": scientific_profile["profile_id"],
        "profile_digest": scientific_profile["profile_digest"],
        "formal_execution_lock": formal_lock,
        "formal_execution_commit": formal_lock["formal_execution_commit"],
        "formal_execution_lock_digest": formal_lock[
            "formal_execution_lock_digest"
        ],
        "formal_execution_lock_ready": True,
        "provisioned": True,
        "formal_ready": False,
        "decision": "provisioned",
        "failure_reasons": [],
        "supports_paper_claim": False,
        "orchestrator_profile_digest": orchestrator.profile_digest,
        "orchestrator_complete_hash_lock_digest": (
            orchestrator.complete_hash_lock_digest
        ),
        "orchestrator_inspection": inspection,
    }
    dependency_report = {
        "report_schema": "isolated_dependency_environment_preparation_report",
        "schema_version": 1,
        "operation_kind": "formal_dependency_environment_preparation",
        **scientific_profile,
        "formal_execution_lock": formal_lock,
        "formal_execution_commit": formal_lock["formal_execution_commit"],
        "formal_execution_lock_digest": formal_lock[
            "formal_execution_lock_digest"
        ],
        "formal_execution_lock_ready": True,
        "provision_report": provision_report,
        "provision_report_digest": observation_cli._dependency_report_object_digest(
            provision_report
        ),
    }
    dependency_path = tmp_path / "isolated_dependency_environment_report.json"
    dependency_path.write_text(
        json.dumps(dependency_report, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    runtime_report = {
        "dependency_environment_ready": True,
        "isolated_scientific_context_ready": True,
        "dependency_profile_id": scientific_profile["profile_id"],
        "dependency_profile_digest": scientific_profile["profile_digest"],
        "direct_requirements_digest": scientific_profile[
            "direct_requirements_digest"
        ],
        "complete_hash_lock_digest": scientific_profile[
            "complete_hash_lock_digest"
        ],
        "complete_hash_lock_dependency_count": scientific_profile[
            "complete_hash_lock_dependency_count"
        ],
        "isolated_scientific_context": {
            "dependency_environment_report_path": str(dependency_path),
            "dependency_environment_report_actual_digest": hashlib.sha256(
                dependency_path.read_bytes()
            ).hexdigest(),
            "reported_python_executable_sha256": "a" * 64,
        },
    }
    return dependency_path, dependency_report, runtime_report


def _rewrite_dependency_environment_fixture(
    dependency_path: Path,
    dependency_report: dict[str, object],
    runtime_report: dict[str, object],
) -> None:
    dependency_path.write_text(
        json.dumps(dependency_report, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    context = runtime_report["isolated_scientific_context"]
    assert isinstance(context, dict)
    context["dependency_environment_report_actual_digest"] = hashlib.sha256(
        dependency_path.read_bytes()
    ).hexdigest()


def test_cli_binds_nested_orchestrator_identity_from_formal_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    formal_lock = build_test_formal_execution_lock("b" * 40)
    _, dependency_report, runtime_report = _dependency_environment_fixture(
        tmp_path,
        formal_lock,
    )
    monkeypatch.setattr(
        observation_cli,
        "build_runtime_environment_report",
        lambda *_args, **_kwargs: runtime_report,
    )
    forbidden_calls = {"model": 0, "pipeline": 0}

    def _forbidden_model(*_args, **_kwargs):
        forbidden_calls["model"] += 1
        raise AssertionError("environment identity gate loaded the model")

    def _forbidden_pipeline(*_args, **_kwargs):
        forbidden_calls["pipeline"] += 1
        raise AssertionError("environment identity gate called the pipeline")

    monkeypatch.setattr(observation_cli, "build_method_config", _forbidden_model)
    monkeypatch.setattr(
        observation_cli,
        "run_formal_terminal_hf_screen",
        _forbidden_pipeline,
    )
    identity = observation_cli._execution_environment_identity(
        Path(".").resolve(),
        formal_lock,
    )
    provision = dependency_report["provision_report"]
    assert isinstance(provision, dict)
    inspection = provision["orchestrator_inspection"]
    assert isinstance(inspection, dict)
    assert identity["orchestrator_profile_digest"] == provision[
        "orchestrator_profile_digest"
    ]
    assert identity["orchestrator_complete_hash_lock_digest"] == provision[
        "orchestrator_complete_hash_lock_digest"
    ]
    assert identity["orchestrator_inspection_digest"] == inspection[
        "inspection_digest"
    ]
    assert identity["scientific_dependency_environment_report_digest"] == (
        runtime_report["isolated_scientific_context"]
        ["dependency_environment_report_actual_digest"]
    )
    assert forbidden_calls == {"model": 0, "pipeline": 0}


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("missing_inspection", "orchestrator inspection identity is absent"),
        ("inspection_type", "orchestrator inspection identity is absent"),
        ("inspection_digest", "orchestrator inspection digest drifted"),
        ("formal_commit", "scientific dependency provision identity drifted"),
        ("formal_lock", "scientific dependency provision identity drifted"),
        ("scientific_identity", "scientific dependency report identity drifted"),
    ),
)
def test_cli_rejects_nested_orchestrator_identity_drift_before_model_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    message: str,
) -> None:
    formal_lock = build_test_formal_execution_lock("b" * 40)
    dependency_path, dependency_report, runtime_report = (
        _dependency_environment_fixture(tmp_path, formal_lock)
    )
    provision = dependency_report["provision_report"]
    assert isinstance(provision, dict)
    inspection = provision["orchestrator_inspection"]
    assert isinstance(inspection, dict)
    if mutation == "missing_inspection":
        del provision["orchestrator_inspection"]
    elif mutation == "inspection_type":
        provision["orchestrator_inspection"] = []
    elif mutation == "inspection_digest":
        inspection["inspection_digest"] = "0" * 64
    elif mutation == "formal_commit":
        provision["formal_execution_commit"] = "c" * 40
    elif mutation == "formal_lock":
        provision["formal_execution_lock_digest"] = "0" * 64
    elif mutation == "scientific_identity":
        dependency_report["profile_digest"] = "0" * 64
    else:
        raise AssertionError("unregistered test mutation")
    dependency_report["provision_report_digest"] = (
        observation_cli._dependency_report_object_digest(provision)
    )
    _rewrite_dependency_environment_fixture(
        dependency_path,
        dependency_report,
        runtime_report,
    )
    monkeypatch.setattr(
        observation_cli,
        "build_runtime_environment_report",
        lambda *_args, **_kwargs: runtime_report,
    )
    monkeypatch.setattr(
        observation_cli,
        "build_method_config",
        lambda *_args, **_kwargs: pytest.fail("identity failure loaded the model"),
    )
    monkeypatch.setattr(
        observation_cli,
            "run_formal_terminal_hf_screen",
        lambda *_args, **_kwargs: pytest.fail("identity failure called the pipeline"),
    )
    with pytest.raises(RuntimeError, match=message):
        observation_cli._execution_environment_identity(
            Path(".").resolve(),
            formal_lock,
        )


def test_cli_rejects_dependency_report_file_digest_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    formal_lock = build_test_formal_execution_lock("b" * 40)
    _, _, runtime_report = _dependency_environment_fixture(tmp_path, formal_lock)
    context = runtime_report["isolated_scientific_context"]
    assert isinstance(context, dict)
    context["dependency_environment_report_actual_digest"] = "0" * 64
    monkeypatch.setattr(
        observation_cli,
        "build_runtime_environment_report",
        lambda *_args, **_kwargs: runtime_report,
    )
    with pytest.raises(RuntimeError, match="scientific dependency report digest drifted"):
        observation_cli._execution_environment_identity(
            Path(".").resolve(),
            formal_lock,
        )


def test_cli_prefetches_prompt_saliency_snapshot_for_local_only_loader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hf_home = tmp_path / "huggingface"
    snapshot = hf_home / "hub" / "snapshots" / "clip"
    snapshot.mkdir(parents=True)
    calls: list[dict[str, object]] = []

    def _download(**kwargs: object) -> str:
        calls.append(dict(kwargs))
        return str(snapshot)

    monkeypatch.setenv("HF_HOME", str(hf_home))
    monkeypatch.setenv("HF_TOKEN", "synthetic-hf-token")
    configs = {
        prompt_id: _config(prompt_id)
        for prompt_id in protocol.CONTENT_SURVIVAL_PROMPT_IDS
    }
    resolved = observation_cli._prepare_prompt_saliency_model_cache(
        configs,
        snapshot_downloader=_download,
    )
    assert resolved == snapshot.resolve()
    assert calls == [
        {
            "repo_id": "openai/clip-vit-base-patch32",
            "revision": "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268",
            "token": "synthetic-hf-token",
            "cache_dir": str(hf_home / "hub"),
        }
    ]


def test_cli_prefetches_clip_after_identity_gate_and_before_gpu_runtime() -> None:
    main_source = inspect.getsource(observation_cli.main)
    identity_index = main_source.index("_execution_environment_identity(root, lock)")
    prefetch_index = main_source.index(
        "_prepare_prompt_saliency_model_cache(prompt_configs)"
    )
    runtime_index = main_source.index(
        "summary = run_formal_terminal_hf_screen("
    )
    assert identity_index < prefetch_index < runtime_index


def _config(
    prompt_id: str = protocol.CONTENT_SURVIVAL_PROMPT_IDS[0],
) -> SemanticWatermarkRuntimeConfig:
    return SemanticWatermarkRuntimeConfig(
        prompt="A governed observation prompt",
        prompt_id=prompt_id,
        key_material="registered-key",
    )


def _identity_bundle(
    *,
    routing_mode: str = "semantic",
    carrier_mode: str = "dual",
    config: SemanticWatermarkRuntimeConfig | None = None,
    formal_lock: dict[str, object] | None = None,
    execution_identity: dict[str, object] | None = None,
    loaded_protocol: protocol.ContentSurvivalObservationProtocol | None = None,
) -> tuple[
    SemanticWatermarkRuntimeConfig,
    protocol.ContentSurvivalObservationProtocol,
    dict[str, object],
    dict[str, object],
    dict[str, object],
    dict[str, object],
]:
    resolved_config = config or _config()
    resolved_protocol = loaded_protocol or _loaded_protocol()
    lock = formal_lock or build_test_formal_execution_lock("b" * 40)
    environment = execution_identity or _execution_identity()
    roster = protocol.build_content_survival_observation_roster(
        resolved_config.key_material,
        protocol=resolved_protocol,
    )
    identity = protocol.build_content_survival_cell_identity(
        prompt_id=resolved_config.prompt_id,
        routing_mode=routing_mode,
        carrier_mode=carrier_mode,
        generation_seed_random=int(resolved_config.seed),
        registered_key_material_digest_random=roster[
            "registered_key_material_digest_random"
        ],
        formal_execution_lock=lock,
        execution_environment_identity=environment,
        model_id=resolved_config.model_id,
        model_revision=resolved_config.model_revision,
        vision_model_id=resolved_config.vision_model_id,
        vision_model_revision=resolved_config.vision_model_revision,
        semantic_watermark_runtime_config_digest=(
            semantic_watermark_runtime_config_digest(resolved_config)
        ),
        prompt_text_digest=build_stable_digest(
            {"prompt_text": resolved_config.prompt}
        ),
        key_roster=roster,
        protocol=resolved_protocol,
    )
    run_identity = protocol.build_content_survival_observation_run_identity(
        formal_execution_lock=lock,
        execution_environment_identity=environment,
        model_id=resolved_config.model_id,
        model_revision=resolved_config.model_revision,
        vision_model_id=resolved_config.vision_model_id,
        vision_model_revision=resolved_config.vision_model_revision,
        key_roster=roster,
        protocol=resolved_protocol,
    )
    return (
        resolved_config,
        resolved_protocol,
        lock,
        environment,
        roster,
        identity,
    )


def _tensor_bytes(value: torch.Tensor) -> bytes:
    buffer = BytesIO()
    torch.save(value, buffer)
    return buffer.getvalue()


def _png_bytes(marker: int) -> tuple[bytes, dict[str, object]]:
    pixels = bytes(
        (marker + index * 7) % 256 for index in range(8 * 8 * 3)
    )
    image = Image.frombytes("RGB", (8, 8), pixels)
    identity = canonical_rgb_uint8_content_record(image)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue(), identity


def _blind_record(
    *,
    latent_digest: str,
    key_digest: str,
    registered_score: float,
    model_identity_digest: str,
) -> dict[str, object]:
    lf_score = float(registered_score)
    hf_score = float(registered_score)
    content_score = 0.70 * lf_score + 0.30 * hf_score
    payload = {
        "observed_latent_content_sha256": latent_digest,
        "lf_template_digest": build_stable_digest(
            {"kind": "lf", "key": key_digest}
        ),
        "lf_template_content_sha256": build_stable_digest(
            {"content": "lf", "key": key_digest}
        ),
        "hf_tail_template_digest": build_stable_digest(
            {"kind": "hf", "key": key_digest}
        ),
        "hf_tail_template_content_sha256": build_stable_digest(
            {"content": "hf", "key": key_digest}
        ),
        "model_identity_digest": model_identity_digest,
        "prg_version": "test_keyed_prg",
        "scoring_key_identity_digest": build_stable_digest(
            {"score_key": key_digest}
        ),
        "method_role": "full_dual_chain",
        "lf_weight": 0.70,
        "hf_tail_weight": 0.30,
        "blind_lf_score": lf_score,
        "blind_hf_tail_score": hf_score,
        "blind_content_score": content_score,
    }
    return {
        "key_material_digest_random": key_digest,
        **payload,
        "score_identity_digest": build_stable_digest(payload),
    }


def _oracle_record(
    *,
    latent_digest: str,
    key_digest: str,
    score: float,
    chain_role: str,
) -> dict[str, object]:
    payload = {
        "oracle_identity": {
            "routing_mode": "semantic",
            "carrier_mode": "dual",
            "chain_role": chain_role,
            "key_material_digest_random": key_digest,
        },
        "observed_latent_content_sha256": latent_digest,
        "routed_lf_template_content_sha256": build_stable_digest(
            {"oracle": "lf", "key": key_digest}
        ),
        "routed_hf_template_content_sha256": build_stable_digest(
            {"oracle": "hf", "key": key_digest}
        ),
        "lf_score": float(score),
        "hf_score": float(score),
        "score": 0.70 * float(score) + 0.30 * float(score),
        "lf_weight": 0.70,
        "hf_weight": 0.30,
        "feedback_allowed": False,
        **protocol.CONTENT_SURVIVAL_CLAIM_BOUNDARY,
    }
    return {
        **payload,
        "oracle_score_digest": protocol._domain_digest(
            protocol._ORACLE_DOMAIN,
            payload,
        ),
    }


def _key_records(
    *,
    latent_digest: str,
    identity: dict[str, object],
    registered_score: float,
    chain_role: str,
) -> list[dict[str, object]]:
    digests = [
        identity["registered_key_material_digest_random"],
        *identity["wrong_key_material_digests_random"],
    ]
    records = []
    for index, key_digest in enumerate(digests):
        score = (
            registered_score
            if index == 0
            else registered_score - 0.25 - index / 1000.0
        )
        records.append(
            {
                "key_role": "registered" if index == 0 else "wrong",
                "key_index": None if index == 0 else index - 1,
                "key_material_digest_random": key_digest,
                "blind_full_template": _blind_record(
                    latent_digest=latent_digest,
                    key_digest=key_digest,
                    registered_score=score,
                    model_identity_digest=identity["model_identity_digest"],
                ),
                "routed_template_oracle": _oracle_record(
                    latent_digest=latent_digest,
                    key_digest=key_digest,
                    score=score / 2.0,
                    chain_role=chain_role,
                ),
                "feedback_allowed": False,
                **protocol.CONTENT_SURVIVAL_CLAIM_BOUNDARY,
            }
        )
    return records


def _observation_record(
    *,
    chain_role: str,
    observation_role: str,
    latent_digest: str,
    parent_chain_digest: str,
    identity: dict[str, object],
    registered_score: float,
) -> dict[str, object]:
    keys = _key_records(
        latent_digest=latent_digest,
        identity=identity,
        registered_score=registered_score,
        chain_role=chain_role,
    )
    ranks = {
        family: protocol.build_registered_rank_record(
            float(keys[0][family][field]),
            [float(record[family][field]) for record in keys[1:]],
        )
        for family, field in (
            ("blind_full_template", "blind_content_score"),
            ("routed_template_oracle", "score"),
        )
    }
    payload = {
        "cell_identity_digest": identity["cell_identity_digest"],
        "chain_role": chain_role,
        "observation_role": observation_role,
        "latent_content_sha256": latent_digest,
        "parent_chain_digest": parent_chain_digest,
        "key_score_records": keys,
        "rank_records": ranks,
        "oracle_feedback_allowed": False,
        "wrong_key_feedback_allowed": False,
        **protocol.CONTENT_SURVIVAL_CLAIM_BOUNDARY,
    }
    return {**payload, "observation_record_digest": build_stable_digest(payload)}


def _chain_artifacts(
    role: str,
    *,
    role_index: int,
    identity: dict[str, object],
) -> tuple[dict[str, object], dict[str, bytes], dict[str, torch.Tensor]]:
    tensors = {
        observation_role: (
            torch.arange(64, dtype=torch.float32).reshape(1, 1, 8, 8)
            + role_index * 10
            + observation_index
        )
        for observation_index, observation_role in enumerate(
            protocol.CONTENT_SURVIVAL_OBSERVATION_ROLES
        )
    }
    image_bytes, image_identity = _png_bytes(role_index + 1)
    if role.startswith("full_probe"):
        direction_digest = "1" * 64
    elif role.startswith("carrier_probe"):
        direction_digest = "2" * 64
    else:
        direction_digest = "3" * 64
    base_identity_payload = {
        "generation_seed_random": 17,
        "base_latent_generation_protocol": "test_canonical_base_latent",
        "base_latent_keyed_prg_version": "sha256_counter_box_muller_v1",
        "base_latent_keyed_prg_protocol_digest": "7" * 64,
        "formal_randomization_protocol_digest": "8" * 64,
        "base_latent_dtype": "float32",
        "base_latent_shape": [1, 1, 8, 8],
        "base_latent_content_digest_random": "9" * 64,
    }
    base_identity = {
        **base_identity_payload,
        "base_latent_identity_digest_random": build_stable_digest(
            base_identity_payload
        ),
    }
    scheduler_identity = {
        "scheduler_class": "tests.FakeScheduler",
        "scheduler_config": {
            "algorithm_type": "test_scheduler",
            "num_train_timesteps": 1000,
        },
    }
    callback: dict[str, object] = {
        "scheduler_step_index": 10,
        "scheduler_step_timestep": 0.5,
        "z9_content_sha256": "a" * 64,
        "z10_before_write_content_sha256": "d" * 64,
    }
    if role != "clean_reference":
        callback.update(
            {
                "routing_identity_digest": "4" * 64,
                "registered_direction_content_sha256": direction_digest,
                "lf_update_content_sha256": "5" * 64,
                "hf_update_content_sha256": "6" * 64,
                "geometry_update_content_sha256": "0" * 64,
                "attention_geometry_enabled": (
                    role.startswith("full_probe") or role.startswith("nominal_")
                ),
            }
        )
    if role in {
        "full_probe_positive",
        "full_probe_negative",
        "carrier_probe_positive",
        "carrier_probe_negative",
    }:
        probe_sign = 1 if role.endswith("positive") else -1
        probe_payload = {
            "probe_role": role,
            "probe_sign": probe_sign,
            "direction_norm": "rms_chw",
            "direction_axes": [1, 2, 3],
            "direction_accumulation_dtype": "float64",
            "direction_unit_content_sha256": "e" * 64,
            "probe_target_ratio_float64": 1.0e-3,
            "probe_actual_tensor_dtype": "float32",
            "probe_realized_ratio_float64": 1.0e-3,
            "probe_ratio_absolute_error_float64": 0.0,
            "probe_ratio_tolerance_float64": 1.0e-7,
            "probe_ratio_comparison": "less_than_or_equal",
            "probe_ratio_ready": True,
            "probe_materialized_latent_content_sha256": tensor_content_sha256(
                tensors["post_write_z10"]
            ),
            "supports_paper_claim": False,
        }
        callback.update(
            {
                **probe_payload,
                "probe_record_digest": build_stable_digest(probe_payload),
            }
        )
    elif role.startswith("nominal_"):
        callback.update(
            {
                "selected_sign": 1,
                "common_gamma": 0.5,
                "write_identity_digest": "f" * 64,
                "actual_dtype_single_write_digest": "1" * 64,
            }
        )
    payload = {
        "role": role,
        "routing_mode": "semantic",
        "carrier_mode": "dual",
        "base_latent_identity": base_identity,
        "base_latent_identity_digest_random": base_identity[
            "base_latent_identity_digest_random"
        ],
        "scheduler_identity": scheduler_identity,
        "scheduler_identity_digest": build_stable_digest(scheduler_identity),
        "observation_run_identity_digest": identity[
            "observation_run_identity_digest"
        ],
        "prompt_text_digest": identity["prompt_text_digest"],
        "prompt_config_digest": identity["prompt_config_digest"],
        "key_roster_digest_random": identity["key_roster_digest_random"],
        "post_write_z10_content_sha256": tensor_content_sha256(
            tensors["post_write_z10"]
        ),
        "pre_vae_terminal_latent_content_sha256": tensor_content_sha256(
            tensors["pre_vae_terminal_latent"]
        ),
        "image_reencoded_latent_content_sha256": tensor_content_sha256(
            tensors["image_reencoded_latent"]
        ),
        **image_identity,
        "callback_record": callback,
        **protocol.CONTENT_SURVIVAL_CLAIM_BOUNDARY,
    }
    if role != "clean_reference":
        payload.update(
            {
                "cell_identity_digest": identity["cell_identity_digest"],
                "protocol_semantic_digest": identity[
                    "protocol_semantic_digest"
                ],
                "prompt_roster_semantic_digest": identity[
                    "prompt_roster_semantic_digest"
                ],
                "prompt_roster_artifact_file_sha256": identity[
                    "prompt_roster_artifact_file_sha256"
                ],
            }
        )
    record = {**payload, "parent_chain_digest": build_stable_digest(payload)}
    leaves = {
        f"{role}_chain_record.json": protocol._canonical_json_bytes(record),
    }
    leaves.update(
        {
            f"{role}_{observation_role}.pt": _tensor_bytes(tensor)
            for observation_role, tensor in tensors.items()
        }
    )
    leaves[f"{role}_image.png"] = image_bytes
    return record, leaves, tensors


def _valid_cell_payloads(
    identity: dict[str, object],
    *,
    include_clean: bool = True,
) -> dict[str, bytes]:
    role_scores = {
        "full_probe_positive": 0.8,
        "full_probe_negative": 0.2,
        "carrier_probe_positive": 0.7,
        "carrier_probe_negative": 0.1,
        "nominal_before_positive": 0.6,
        "nominal_after_selected": 0.65,
    }
    leaves: dict[str, bytes] = {}
    chain_records: list[dict[str, object]] = []
    observation_records: list[dict[str, object]] = []
    for role_index, role in enumerate(protocol.CONTENT_SURVIVAL_CHAIN_ROLES):
        chain, chain_leaves, tensors = _chain_artifacts(
            role,
            role_index=role_index,
            identity=identity,
        )
        chain_records.append(chain)
        leaves.update(chain_leaves)
        for observation_role in protocol.CONTENT_SURVIVAL_OBSERVATION_ROLES:
            observation_records.append(
                _observation_record(
                    chain_role=role,
                    observation_role=observation_role,
                    latent_digest=tensor_content_sha256(tensors[observation_role]),
                    parent_chain_digest=chain["parent_chain_digest"],
                    identity=identity,
                    registered_score=role_scores[role],
                )
            )
    clean_records: list[dict[str, object]] = []
    clean_chain: dict[str, object] | None = None
    if include_clean:
        clean_chain, clean_leaves, clean_tensors = _chain_artifacts(
            "clean_reference",
            role_index=6,
            identity=identity,
        )
        leaves.update(clean_leaves)
        for observation_role in protocol.CONTENT_SURVIVAL_OBSERVATION_ROLES:
            clean_records.append(
                _observation_record(
                    chain_role="clean_reference",
                    observation_role=observation_role,
                    latent_digest=tensor_content_sha256(
                        clean_tensors[observation_role]
                    ),
                    parent_chain_digest=clean_chain["parent_chain_digest"],
                    identity=identity,
                    registered_score=0.0,
                )
            )
    selection = protocol.select_content_survival_observation_sign(
        {
            role: 0.70 * role_scores[role] + 0.30 * role_scores[role]
            for role in protocol.CONTENT_SURVIVAL_CHAIN_ROLES[:4]
        }
    )
    nominal_after = next(
        record
        for record in chain_records
        if record["role"] == "nominal_after_selected"
    )
    nominal_after["callback_record"]["selected_sign"] = selection[
        "selected_sign"
    ]
    nominal_after_payload = dict(nominal_after)
    nominal_after_payload.pop("parent_chain_digest")
    nominal_after["parent_chain_digest"] = build_stable_digest(
        nominal_after_payload
    )
    leaves["nominal_after_selected_chain_record.json"] = (
        protocol._canonical_json_bytes(nominal_after)
    )
    for record in observation_records:
        if record["chain_role"] == "nominal_after_selected":
            record["parent_chain_digest"] = nominal_after[
                "parent_chain_digest"
            ]
            record_payload = dict(record)
            record_payload.pop("observation_record_digest")
            record["observation_record_digest"] = build_stable_digest(
                record_payload
            )
    causal = protocol.build_content_survival_causal_diagnostic(chain_records)
    parent_child_binding_digest = (
        protocol.build_content_survival_parent_child_binding_digest(
            chain_records,
            observation_records,
            clean_chain_record=clean_chain,
            clean_observation_records=clean_records,
        )
    )
    result_payload = {
        "result_schema": "slm_wm_content_survival_observation_cell_result",
        "cell_identity": identity,
        "roster_digest_random": identity["key_roster_digest_random"],
        "selection": selection,
        "chain_records": chain_records,
        "observation_records": observation_records,
        "clean_observation_records": clean_records,
        "parent_child_binding_digest": parent_child_binding_digest,
        **({"clean_chain_record": clean_chain} if clean_chain else {}),
        **causal,
        "known_m0_deviations": list(protocol.CONTENT_SURVIVAL_M0_DEVIATIONS),
        "cell_chain_count": 6,
        "cell_evaluation_count": 18 * 33 * 2,
        **protocol.CONTENT_SURVIVAL_CLAIM_BOUNDARY,
    }
    result = {
        **result_payload,
        "result_digest": build_stable_digest(result_payload),
    }
    leaves["cell_result.json"] = (
        json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()
    return leaves


def _publish_valid_cell(
    cell_dir: Path,
    *,
    identity: dict[str, object],
    include_clean: bool = True,
) -> dict[str, object]:
    return protocol.publish_content_survival_cell(
        cell_dir,
        cell_identity=identity,
        leaf_payloads=_valid_cell_payloads(identity, include_clean=include_clean),
    )


def test_protocol_freezes_source_roster_matrix_and_claim_boundary() -> None:
    loaded = _loaded_protocol()
    assert loaded.semantic_digest == loaded.payload["protocol_semantic_digest"]
    assert loaded.payload["source_evidence"] == protocol.CONTENT_SURVIVAL_SOURCE_EVIDENCE
    assert loaded.payload["routing_reference_registry"] == (
        protocol.CONTENT_SURVIVAL_ROUTING_REFERENCE_IDENTITY
    )
    assert [row["prompt_id"] for row in loaded.payload["roster"]["selection_records"]] == list(
        protocol.CONTENT_SURVIVAL_PROMPT_IDS
    )
    assert loaded.payload["roster"]["selection_records"][0][
        "minimum_registered_minus_wrong_margin"
    ] == 0.023721
    assert loaded.payload["roster"]["selection_records"][-1][
        "maximum_registered_minus_wrong_margin"
    ] == -0.0381536
    assert loaded.payload["matrix"]["chain_count"] == 148
    assert loaded.payload["evaluation_protocol"]["evaluation_count"] == 29_304
    assert loaded.payload["claim_boundary"] == protocol.CONTENT_SURVIVAL_CLAIM_BOUNDARY


def test_protocol_roster_and_semantic_digests_fail_closed() -> None:
    loaded = _loaded_protocol()
    payload = json.loads(json.dumps(loaded.payload))
    payload["roster"]["selection_records"][0][
        "minimum_registered_minus_wrong_margin"
    ] = 0.0
    with pytest.raises(ValueError, match="roster drifted"):
        protocol.validate_content_survival_observation_payload(payload)
    payload = json.loads(json.dumps(loaded.payload))
    payload["protocol_semantic_digest"] = "0" * 64
    with pytest.raises(ValueError, match="semantic digest mismatch"):
        protocol.validate_content_survival_observation_payload(payload)


def test_cell_identity_binds_formal_model_config_routing_and_rosters() -> None:
    _, loaded, lock, environment, roster, identity = _identity_bundle()
    assert identity["formal_execution_lock"] == lock
    assert identity["execution_environment_identity"] == environment
    assert identity["actual_formal_execution_commit"] == lock[
        "formal_execution_commit"
    ]
    assert identity["routing_reference_registry_semantic_digest"] == loaded.payload[
        "routing_reference_registry"
    ]["semantic_digest"]
    assert identity["prompt_roster_semantic_digest"] == loaded.payload["roster"][
        "roster_semantic_digest"
    ]
    assert identity["key_roster_digest_random"] == roster["roster_digest_random"]


def test_wrong_key_roster_is_fixed_domain_separated_and_collision_free() -> None:
    loaded = _loaded_protocol()
    roster = protocol.build_content_survival_observation_roster(
        "registered-key", protocol=loaded
    )
    digests = [row["wrong_key_material_digest_random"] for row in roster["wrong_keys"]]
    assert len(digests) == len(set(digests)) == 32
    assert roster["registered_key_material_digest_random"] not in digests
    assert [row["wrong_key_index"] for row in roster["wrong_keys"]] == list(range(32))


def test_sign_selection_and_rank_recomputation_are_fixed() -> None:
    scores = {
        "full_probe_positive": 0.8,
        "full_probe_negative": 0.2,
        "carrier_probe_positive": 0.7,
        "carrier_probe_negative": 0.1,
    }
    selected = protocol.select_content_survival_observation_sign(scores)
    assert selected["selected_sign"] == 1
    assert selected["wrong_key_accessed"] is False
    with pytest.raises(ValueError, match="exactly four"):
        protocol.select_content_survival_observation_sign({**scores, "wrong": 9.0})
    rank = protocol.build_registered_rank_record(16.0, list(map(float, range(32))))
    assert rank["registered_rank"] == 17
    assert rank["registered_empirical_percentile"] == 0.5
    assert rank["wrong_score_population_variance"] == pytest.approx(85.25)


def test_blind_scorer_calls_frozen_functions_and_oracle_is_nonclaim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    class _Template:
        template = torch.arange(4, dtype=torch.float32).reshape(1, 1, 2, 2)
        template_digest = "1" * 64

    class _Score:
        blind_lf_score = 0.1
        blind_hf_tail_score = 0.2
        blind_content_score = 0.13
        lf_weight = 0.7
        hf_tail_weight = 0.3
        score_identity_digest = "2" * 64
        scoring_key_identity_digest = "3" * 64

    monkeypatch.setattr(protocol, "build_low_frequency_template", lambda *_a, **_k: calls.append("lf") or _Template())
    monkeypatch.setattr(protocol, "build_high_frequency_tail_template", lambda *_a, **_k: calls.append("hf") or _Template())
    monkeypatch.setattr(protocol, "compute_blind_content_score", lambda *_a, **_k: calls.append("score") or _Score())
    result = protocol.compute_blind_observation_score(
        torch.arange(4, dtype=torch.float32).reshape(1, 1, 2, 2),
        key_material="registered-key",
        model_identity_digest="4" * 64,
        prg_version="prg",
        method_role="full_dual_chain",
    )
    assert calls == ["lf", "hf", "score"]
    assert result["blind_content_score"] == 0.13
    assert "measure_image_only_watermark" not in inspect.getsource(
        protocol.compute_blind_observation_score
    )


def test_complete_bundle_reloads_tensors_png_scores_selection_and_confounds(
    tmp_path: Path,
) -> None:
    *_, identity = _identity_bundle()
    cell_dir = tmp_path / "outputs" / "observation" / "cell"
    manifest = _publish_valid_cell(cell_dir, identity=identity)
    assert manifest["artifact_complete"] is True
    validated = protocol.validate_content_survival_cell_bundle(
        cell_dir,
        expected_cell_identity_digest=identity["cell_identity_digest"],
        complete=True,
    )
    assert validated == manifest
    result = json.loads((cell_dir / "cell_result.json").read_text())
    assert result["nominal_pair_unique_difference_ready"] is True
    assert result["selector_confounded"] is True
    assert result["causal_conclusion_ready"] is False
    assert "full_carrier_probe_direction_identity_not_shared" in result[
        "confounded_reasons"
    ]


@pytest.mark.parametrize(
    "invalid",
    [
        {"not": "a tensor"},
        torch.ones((1, 1, 2, 2), dtype=torch.int64),
        torch.ones((2, 2), dtype=torch.float32),
        torch.tensor([[[[float("nan")]]]], dtype=torch.float32),
    ],
)
def test_safe_tensor_reload_rejects_object_dtype_shape_and_nonfinite(
    tmp_path: Path,
    invalid: object,
) -> None:
    path = tmp_path / "invalid.pt"
    torch.save(invalid, path)
    with pytest.raises((TypeError, ValueError)):
        protocol._load_tensor_leaf(path)


def _rewrite_coordinated_json_surface(
    cell_dir: Path,
    result: dict[str, object],
) -> None:
    result["parent_child_binding_digest"] = (
        protocol.build_content_survival_parent_child_binding_digest(
            result["chain_records"],
            result["observation_records"],
            clean_chain_record=result.get("clean_chain_record"),
            clean_observation_records=result["clean_observation_records"],
        )
    )
    result_payload = dict(result)
    result_payload.pop("result_digest")
    result["result_digest"] = build_stable_digest(result_payload)
    result_path = cell_dir / "cell_result.json"
    result_bytes = protocol._canonical_json_bytes(result)
    result_path.write_bytes(result_bytes)

    binding_path = cell_dir / "cell_binding.json"
    binding = json.loads(binding_path.read_text())
    binding["parent_child_binding_digest"] = result[
        "parent_child_binding_digest"
    ]
    for leaf in binding["leaves"]:
        payload = (cell_dir / leaf["path"]).read_bytes()
        leaf["size_bytes"] = len(payload)
        leaf["sha256"] = hashlib.sha256(payload).hexdigest()
    binding_payload = dict(binding)
    binding_payload.pop("cell_binding_digest")
    binding["cell_binding_digest"] = protocol._domain_digest(
        protocol._CELL_BINDING_DOMAIN, binding_payload
    )
    binding_bytes = protocol._canonical_json_bytes(binding)
    binding_path.write_bytes(binding_bytes)

    manifest_path = cell_dir / "cell_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["parent_child_binding_digest"] = result[
        "parent_child_binding_digest"
    ]
    for leaf in manifest["leaves"]:
        payload = (cell_dir / leaf["path"]).read_bytes()
        leaf["size_bytes"] = len(payload)
        leaf["sha256"] = hashlib.sha256(payload).hexdigest()
    manifest_path.write_bytes(protocol._canonical_json_bytes(manifest))


def _rewrite_outer_digests_after_forged_record(cell_dir: Path) -> None:
    result = json.loads((cell_dir / "cell_result.json").read_text())
    record = result["observation_records"][0]
    forged = "f" * 64
    record["latent_content_sha256"] = forged
    for key_record in record["key_score_records"]:
        blind = key_record["blind_full_template"]
        blind["observed_latent_content_sha256"] = forged
        blind_payload = {
            key: blind[key]
            for key in (
                "observed_latent_content_sha256",
                "lf_template_digest",
                "lf_template_content_sha256",
                "hf_tail_template_digest",
                "hf_tail_template_content_sha256",
                "model_identity_digest",
                "prg_version",
                "scoring_key_identity_digest",
                "method_role",
                "lf_weight",
                "hf_tail_weight",
                "blind_lf_score",
                "blind_hf_tail_score",
                "blind_content_score",
            )
        }
        blind["score_identity_digest"] = build_stable_digest(blind_payload)
        oracle = key_record["routed_template_oracle"]
        oracle["observed_latent_content_sha256"] = forged
        oracle_payload = dict(oracle)
        oracle_payload.pop("oracle_score_digest")
        oracle["oracle_score_digest"] = protocol._domain_digest(
            protocol._ORACLE_DOMAIN, oracle_payload
        )
    record_payload = dict(record)
    record_payload.pop("observation_record_digest")
    record["observation_record_digest"] = build_stable_digest(record_payload)
    _rewrite_coordinated_json_surface(cell_dir, result)


def test_coordinated_json_rehash_cannot_override_tensor_truth(tmp_path: Path) -> None:
    *_, identity = _identity_bundle()
    cell_dir = tmp_path / "outputs" / "observation" / "cell"
    _publish_valid_cell(cell_dir, identity=identity)
    _rewrite_outer_digests_after_forged_record(cell_dir)
    with pytest.raises(ValueError, match="persisted tensor"):
        protocol.validate_content_survival_cell_bundle(
            cell_dir,
            expected_cell_identity_digest=identity["cell_identity_digest"],
            complete=True,
        )


def test_coordinated_probe_parent_rehash_cannot_drift_base_identity(
    tmp_path: Path,
) -> None:
    *_, identity = _identity_bundle()
    cell_dir = tmp_path / "outputs" / "observation" / "cell"
    _publish_valid_cell(cell_dir, identity=identity)
    immutable_paths = tuple(cell_dir.glob("*.pt")) + tuple(cell_dir.glob("*.png"))
    immutable_before = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in immutable_paths
    }
    role = "full_probe_positive"
    chain_path = cell_dir / f"{role}_chain_record.json"
    chain = json.loads(chain_path.read_text())
    base_identity = dict(chain["base_latent_identity"])
    base_identity["generation_seed_random"] += 1
    base_payload = dict(base_identity)
    base_payload.pop("base_latent_identity_digest_random")
    base_identity["base_latent_identity_digest_random"] = build_stable_digest(
        base_payload
    )
    chain["base_latent_identity"] = base_identity
    chain["base_latent_identity_digest_random"] = base_identity[
        "base_latent_identity_digest_random"
    ]
    chain_payload = dict(chain)
    chain_payload.pop("parent_chain_digest")
    chain["parent_chain_digest"] = build_stable_digest(chain_payload)
    chain_path.write_bytes(protocol._canonical_json_bytes(chain))

    result = json.loads((cell_dir / "cell_result.json").read_text())
    result["chain_records"][0] = chain
    for record in result["observation_records"]:
        if record["chain_role"] == role:
            record["parent_chain_digest"] = chain["parent_chain_digest"]
            record_payload = dict(record)
            record_payload.pop("observation_record_digest")
            record["observation_record_digest"] = build_stable_digest(
                record_payload
            )
    _rewrite_coordinated_json_surface(cell_dir, result)
    assert {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in immutable_paths
    } == immutable_before
    with pytest.raises(ValueError, match="six-chain parent identity"):
        protocol.validate_content_survival_cell_bundle(
            cell_dir,
            expected_cell_identity_digest=identity["cell_identity_digest"],
            complete=True,
        )


def test_coordinated_nominal_sign_and_selection_rehash_is_rejected(
    tmp_path: Path,
) -> None:
    *_, identity = _identity_bundle()
    cell_dir = tmp_path / "outputs" / "observation" / "cell"
    _publish_valid_cell(cell_dir, identity=identity)
    immutable_paths = tuple(cell_dir.glob("*.pt")) + tuple(cell_dir.glob("*.png"))
    immutable_before = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in immutable_paths
    }
    role = "nominal_after_selected"
    chain_path = cell_dir / f"{role}_chain_record.json"
    chain = json.loads(chain_path.read_text())
    chain["callback_record"]["selected_sign"] = -1
    chain_payload = dict(chain)
    chain_payload.pop("parent_chain_digest")
    chain["parent_chain_digest"] = build_stable_digest(chain_payload)
    chain_path.write_bytes(protocol._canonical_json_bytes(chain))

    result = json.loads((cell_dir / "cell_result.json").read_text())
    result["chain_records"][-1] = chain
    selection = dict(result["selection"])
    selection["selected_sign"] = -1
    selection_payload = dict(selection)
    selection_payload.pop("selection_digest")
    selection["selection_digest"] = build_stable_digest(selection_payload)
    result["selection"] = selection
    for record in result["observation_records"]:
        if record["chain_role"] == role:
            record["parent_chain_digest"] = chain["parent_chain_digest"]
            record_payload = dict(record)
            record_payload.pop("observation_record_digest")
            record["observation_record_digest"] = build_stable_digest(
                record_payload
            )
    _rewrite_coordinated_json_surface(cell_dir, result)
    assert {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in immutable_paths
    } == immutable_before
    with pytest.raises(ValueError, match="registered-only four-probe selection"):
        protocol.validate_content_survival_cell_bundle(
            cell_dir,
            expected_cell_identity_digest=identity["cell_identity_digest"],
            complete=True,
        )


def test_partial_cell_redoes_only_the_same_identity(tmp_path: Path) -> None:
    *_, identity = _identity_bundle()
    cell_dir = tmp_path / "outputs" / "observation" / "cell"
    payloads = _valid_cell_payloads(identity)
    _publish_valid_cell(cell_dir, identity=identity)
    (cell_dir / "cell_manifest.json").replace(
        cell_dir / "cell_manifest.partial.json"
    )
    manifest = protocol.publish_content_survival_cell(
        cell_dir,
        cell_identity=identity,
        leaf_payloads=payloads,
    )
    assert manifest["artifact_complete"] is True
    (cell_dir / "cell_manifest.json").replace(
        cell_dir / "cell_manifest.partial.json"
    )
    conflicting = {**identity, "cell_identity_digest": "e" * 64}
    with pytest.raises(ValueError, match="different roster identity"):
        protocol.publish_content_survival_cell(
            cell_dir,
            cell_identity=conflicting,
            leaf_payloads=payloads,
        )


def test_publisher_cannot_write_source_archive_or_non_outputs_path(
    tmp_path: Path,
) -> None:
    *_, identity = _identity_bundle()
    with pytest.raises(ValueError, match="under outputs"):
        protocol.publish_content_survival_cell(
            Path("/tmp/slm_wm_source_archive_read_only/cell"),
            cell_identity=identity,
            leaf_payloads=_valid_cell_payloads(identity),
        )


@pytest.mark.parametrize(
    "drift", ["formal", "environment", "model", "config", "routing", "roster"]
)
def test_resume_rejects_identity_drift(tmp_path: Path, drift: str) -> None:
    config, loaded, lock, environment, roster, identity = _identity_bundle()
    cell_dir = tmp_path / "outputs" / "observation" / "cell"
    _publish_valid_cell(cell_dir, identity=identity)
    changed_config = config
    changed_lock = lock
    changed_environment = environment
    changed_protocol = loaded
    if drift == "formal":
        changed_lock = build_test_formal_execution_lock("c" * 40)
    elif drift == "environment":
        changed_environment = _execution_identity("d")
    elif drift == "model":
        changed_config = SimpleNamespace(**{**config.__dict__, "model_revision": "changed"})
    elif drift == "config":
        changed_config = SimpleNamespace(**{**config.__dict__, "seed": int(config.seed) + 1})
    elif drift == "routing":
        payload = json.loads(json.dumps(loaded.payload))
        payload["routing_reference_registry"]["semantic_digest"] = "d" * 64
        changed_protocol = protocol.ContentSurvivalObservationProtocol(
            payload=payload,
            semantic_digest=loaded.semantic_digest,
            file_sha256=loaded.file_sha256,
            schema_file_sha256=loaded.schema_file_sha256,
            config_path=loaded.config_path,
            schema_path=loaded.schema_path,
        )
    elif drift == "roster":
        roster = protocol.build_content_survival_observation_roster(
            "other-registered-key", protocol=loaded
        )
    changed_identity = protocol.build_content_survival_cell_identity(
        prompt_id=config.prompt_id,
        routing_mode="semantic",
        carrier_mode="dual",
        generation_seed_random=int(changed_config.seed),
        registered_key_material_digest_random=roster[
            "registered_key_material_digest_random"
        ],
        formal_execution_lock=changed_lock,
        execution_environment_identity=changed_environment,
        model_id=changed_config.model_id,
        model_revision=changed_config.model_revision,
        vision_model_id=changed_config.vision_model_id,
        vision_model_revision=changed_config.vision_model_revision,
        semantic_watermark_runtime_config_digest=build_stable_digest(
            {"config": changed_config.seed}
        ) if drift == "config" else semantic_watermark_runtime_config_digest(config),
        prompt_text_digest=build_stable_digest({"prompt_text": config.prompt}),
        key_roster=roster,
        protocol=changed_protocol,
    )
    with pytest.raises(ValueError, match="manifest identity"):
        protocol.validate_content_survival_cell_bundle(
            cell_dir,
            expected_cell_identity_digest=changed_identity["cell_identity_digest"],
            complete=True,
        )


def test_summary_enumerates_only_24_shared_validator_passes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, loaded, lock, environment, roster, identity = _identity_bundle()
    run_identity = protocol.build_content_survival_observation_run_identity(
        formal_execution_lock=lock,
        execution_environment_identity=environment,
        model_id=identity["model_id"],
        model_revision=identity["model_revision"],
        vision_model_id=identity["vision_model_id"],
        vision_model_revision=identity["vision_model_revision"],
        key_roster=roster,
        protocol=loaded,
    )
    output_root = tmp_path / "outputs" / "observation"
    expected_cells = []
    calls: list[str] = []
    for prompt_id in protocol.CONTENT_SURVIVAL_PROMPT_IDS:
        for routing_mode in protocol.CONTENT_SURVIVAL_ROUTING_MODES:
            for carrier_mode in protocol.CONTENT_SURVIVAL_CARRIER_MODES:
                relative = Path(prompt_id) / routing_mode / carrier_mode
                directory = output_root / relative
                directory.mkdir(parents=True)
                cell_digest = build_stable_digest(
                    {
                        "prompt_id": prompt_id,
                        "routing_mode": routing_mode,
                        "carrier_mode": carrier_mode,
                    }
                )
                (directory / "cell_manifest.json").write_text("{}", encoding="utf-8")
                result = {
                    "cell_identity": {
                        "prompt_id": prompt_id,
                        "routing_mode": routing_mode,
                        "carrier_mode": carrier_mode,
                        "observation_run_identity": run_identity,
                    },
                    "result_digest": build_stable_digest({"cell": cell_digest}),
                    "causal_conclusion_ready": False,
                }
                if routing_mode == "semantic" and carrier_mode == "dual":
                    result["clean_chain_record"] = {
                        "parent_chain_digest": build_stable_digest(
                            {"clean": prompt_id}
                        ),
                        "image_rgb_uint8_content_sha256": build_stable_digest(
                            {"image": prompt_id}
                        ),
                        "key_roster_digest_random": roster["roster_digest_random"],
                    }
                (directory / "cell_result.json").write_text(
                    json.dumps(result), encoding="utf-8"
                )
                expected_cells.append(
                    {
                        "relative_path": relative.as_posix(),
                        "cell_identity_digest": cell_digest,
                    }
                )
    monkeypatch.setattr(
        protocol,
        "validate_content_survival_cell_bundle",
        lambda directory, **_kwargs: calls.append(Path(directory).as_posix()) or {},
    )
    summary = protocol.build_content_survival_observation_summary(
        output_root,
        expected_cells=expected_cells,
        observation_run_identity=run_identity,
        protocol=loaded,
    )
    assert len(calls) == summary["complete_cell_count"] == 24
    assert summary["clean_chain_count"] == 4
    assert summary["all_causal_conclusions_blocked"] is True


def test_runner_level_complete_cell_resume_makes_zero_pipeline_calls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, loaded, lock, environment, roster, identity = _identity_bundle()
    output_root = tmp_path / "outputs" / "observation"
    cell_dir = output_root / config.prompt_id / "semantic" / "dual"
    _publish_valid_cell(cell_dir, identity=identity)
    monkeypatch.setattr(
        runtime,
        "_run_observed_chain",
        lambda *_args, **_kwargs: pytest.fail("resume called the pipeline"),
    )
    result = runtime._run_observation_cell(
        config,
        ContentRoutingReferenceScalars(1.0, 1.0, 1.0),
        context=object(),
        base_latent=object(),
        base_identity={},
        clean_chain=object(),
        routing_mode="semantic",
        carrier_mode="dual",
        roster=roster,
        protocol=loaded,
        formal_execution_lock=lock,
        execution_environment_identity=environment,
        observation_run_identity=identity["observation_run_identity"],
        output_root=output_root,
    )
    assert result["cell_chain_count"] == 6
    assert len(result["observation_records"]) == 18
    assert all(len(row["key_score_records"]) == 33 for row in result["observation_records"])


def test_geometry_backtracking_failure_is_terminal_and_resume_uses_zero_pipeline_calls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, loaded, lock, environment, roster, identity = _identity_bundle()
    output_root = tmp_path / "outputs" / "observation"
    calls = 0

    def fail_geometry(*_args: object, **_kwargs: object) -> None:
        nonlocal calls
        calls += 1
        raise ValueError(runtime._GEOMETRY_FAILURE_MESSAGE)

    monkeypatch.setattr(runtime, "_run_observed_chain", fail_geometry)
    failure = runtime._run_observation_cell(
        config,
        ContentRoutingReferenceScalars(1.0, 1.0, 1.0),
        context=object(),
        base_latent=object(),
        base_identity={},
        clean_chain=object(),
        routing_mode="uniform",
        carrier_mode="lf_only",
        roster=roster,
        protocol=loaded,
        formal_execution_lock=lock,
        execution_environment_identity=environment,
        observation_run_identity=identity["observation_run_identity"],
        output_root=output_root,
    )
    assert calls == 1
    assert failure["failure_code"] == (
        protocol.CONTENT_SURVIVAL_GEOMETRY_FAILURE_CODE
    )
    assert failure["failed_chain_role"] == "full_probe_positive"
    assert failure["successful_chain_count"] == 0
    assert failure["attempted_chain_count"] == 1
    assert failure["fallback_chain_materialized"] is False
    assert not list(output_root.rglob("*.pt"))
    assert not list(output_root.rglob("*.png"))

    monkeypatch.setattr(
        runtime,
        "_run_observed_chain",
        lambda *_args, **_kwargs: pytest.fail("failure resume called the pipeline"),
    )
    resumed = runtime._run_observation_cell(
        config,
        ContentRoutingReferenceScalars(1.0, 1.0, 1.0),
        context=object(),
        base_latent=object(),
        base_identity={},
        clean_chain=object(),
        routing_mode="uniform",
        carrier_mode="lf_only",
        roster=roster,
        protocol=loaded,
        formal_execution_lock=lock,
        execution_environment_identity=environment,
        observation_run_identity=identity["observation_run_identity"],
        output_root=output_root,
    )
    assert resumed == failure


def test_attempt_summary_accepts_24_terminal_method_failures_without_fallback(
    tmp_path: Path,
) -> None:
    _, loaded, lock, environment, roster, identity = _identity_bundle()
    run_identity = protocol.build_content_survival_observation_run_identity(
        formal_execution_lock=lock,
        execution_environment_identity=environment,
        model_id=identity["model_id"],
        model_revision=identity["model_revision"],
        vision_model_id=identity["vision_model_id"],
        vision_model_revision=identity["vision_model_revision"],
        key_roster=roster,
        protocol=loaded,
    )
    output_root = tmp_path / "outputs" / "observation"
    expected_cells = []
    for prompt_id in protocol.CONTENT_SURVIVAL_PROMPT_IDS:
        for routing_mode in protocol.CONTENT_SURVIVAL_ROUTING_MODES:
            for carrier_mode in protocol.CONTENT_SURVIVAL_CARRIER_MODES:
                relative = Path(prompt_id) / routing_mode / carrier_mode
                cell_identity = {
                    "prompt_id": prompt_id,
                    "routing_mode": routing_mode,
                    "carrier_mode": carrier_mode,
                    "observation_run_identity": run_identity,
                }
                cell_identity["cell_identity_digest"] = build_stable_digest(
                    cell_identity
                )
                failure = protocol.build_content_survival_cell_failure_record(
                    cell_identity=cell_identity,
                    failed_chain_role="full_probe_positive",
                    successful_chain_count=0,
                )
                protocol.publish_content_survival_cell_failure(
                    output_root / relative,
                    failure_record=failure,
                )
                expected_cells.append(
                    {
                        "relative_path": relative.as_posix(),
                        "cell_identity_digest": cell_identity[
                            "cell_identity_digest"
                        ],
                    }
                )
    summary = protocol.build_content_survival_observation_attempt_summary(
        output_root,
        expected_cells=expected_cells,
        observation_run_identity=run_identity,
        protocol=loaded,
    )
    assert summary["terminal_cell_count"] == 24
    assert summary["complete_cell_count"] == 0
    assert summary["failed_cell_count"] == 24
    assert summary["attempted_chain_count"] == 24
    assert summary["successful_chain_count"] == 0
    assert summary["validated_clean_chain_count"] == 0
    assert summary["evaluation_count"] == 0
    assert summary["matrix_execution_complete"] is True
    assert summary["method_failures_present"] is True
    assert summary["supports_paper_claim"] is False


def test_cell_failure_rejects_coordinated_internal_identity_tamper(
    tmp_path: Path,
) -> None:
    *_, identity = _identity_bundle()
    failure = protocol.build_content_survival_cell_failure_record(
        cell_identity=identity,
        failed_chain_role="full_probe_positive",
        successful_chain_count=0,
    )
    tampered = json.loads(json.dumps(failure))
    tampered["cell_identity"]["prompt_id"] = "prompt_coordinated_tamper"
    payload = {
        key: value for key, value in tampered.items() if key != "failure_digest"
    }
    tampered["failure_digest"] = build_stable_digest(payload)
    with pytest.raises(ValueError, match="cell failure identity"):
        protocol.validate_content_survival_cell_failure_record(
            tampered,
            expected_cell_identity_digest=identity["cell_identity_digest"],
        )
    assert not (tmp_path / "cell_failure.json").exists()


def test_runner_keeps_three_latents_and_fixed_registry_boundary_visible() -> None:
    source = inspect.getsource(runtime)
    cli_source = Path("scripts/run_content_survival_observation.py").read_text()
    assert '"output_type": "latent"' in source
    assert "_decode_content_runtime_latent" in source
    assert "_encode_image_latent" in source
    assert "compute_blind_observation_score" in source
    assert "measure_image_only_watermark" not in source
    assert "SLM_WM_CONTENT_ROUTING_REFERENCE_REGISTRY" not in cli_source
    assert 'protocol.payload["routing_reference_registry"]' in cli_source


def test_clean_chain_binds_postwrite_terminal_image_and_reencode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Pipeline:
        _execution_device = torch.device("cpu")
        vae_scale_factor = 1
        scheduler = SimpleNamespace(config={"name": "fixed"})

        def __call__(self, *, latents: torch.Tensor, callback_on_step_end, **kwargs):
            assert kwargs["output_type"] == "latent"
            value = latents.clone()
            for index in range(11):
                value = value + 1
                value = callback_on_step_end(
                    self, index, torch.tensor(float(index)), {"latents": value}
                )["latents"]
            return SimpleNamespace(images=value + 7)

    context = SemanticWatermarkRuntimeContext(
        pipeline=_Pipeline(),
        prompt_saliency_runtime=object(),
        attention_modules=(),
        unconditional_prompt=None,
        unconditional_pooled=None,
        runtime_versions={},
    )
    monkeypatch.setattr(runtime, "_content_runtime_prompt_embeddings", lambda *_a: (torch.zeros(1), torch.zeros(1)))
    monkeypatch.setattr(runtime, "load_content_survival_direction_protocol", lambda *_a: object())
    monkeypatch.setattr(runtime, "_decode_content_runtime_latent", lambda _p, latent: torch.ones((1, 3, 2, 2)) * latent.mean())
    monkeypatch.setattr(runtime, "_encode_image_latent", lambda _p, image: torch.ones((1, 2, 2, 2)) * (image.mean() + 3))
    config = _config()
    *_, identity = _identity_bundle(config=config)
    chain = runtime._run_observed_chain(
        config,
        ContentRoutingReferenceScalars(1.0, 1.0, 1.0),
        context=context,
        base_latent=torch.zeros((1, 2, 2, 2)),
        base_identity={"base_latent_identity_digest_random": "b" * 64},
        routing_mode="semantic",
        carrier_mode="dual",
        role="clean_reference",
        probe_sign=None,
        replay_sign=None,
        observation_run_identity_digest=identity["observation_run_identity_digest"],
        prompt_text_digest=identity["prompt_text_digest"],
        prompt_config_digest=identity["prompt_config_digest"],
        key_roster_digest_random=identity["key_roster_digest_random"],
        cell_identity=None,
    )
    assert torch.equal(chain.pre_write_z10, chain.post_write_z10)
    assert not torch.equal(chain.post_write_z10, chain.pre_vae_terminal_latent)
    assert chain.chain_record["image_reencoded_latent_content_sha256"] == tensor_content_sha256(chain.image_reencoded_latent)
    assert chain.chain_record["image_width"] == 2
    assert chain.chain_record["image_height"] == 2


def test_gpu_runtime_is_not_selected_by_default_pytest() -> None:
    source = Path("tests/integration/test_content_survival_observation_gpu.py").read_text()
    assert "pytest.mark.integration" in source
    assert "pytest.mark.slow" in source
    assert "SLM_WM_RUN_CONTENT_SURVIVAL_OBSERVATION_GPU" in source
