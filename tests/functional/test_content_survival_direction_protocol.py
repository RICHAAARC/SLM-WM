"""覆盖 content-survival 协议、方向、三图证据与原子绑定。"""

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

import experiments.runners.semantic_watermark_runtime as runtime_module

from experiments.protocol.content_survival_direction import (
    CONTENT_SURVIVAL_CHAIN_ROLES,
    CONTENT_SURVIVAL_PROBE_ROLES,
    build_content_survival_artifact_binding,
    build_content_survival_direction_record,
    build_content_survival_runtime_method_identity,
    build_three_image_content_survival_evidence,
    content_survival_protocol_semantic_digest,
    encode_frozen_clip_global_image_feature,
    load_content_survival_direction_protocol,
    materialize_content_survival_probe,
    select_shared_content_survival_sign,
    validate_content_survival_artifact_bundle,
    validate_content_survival_direction_payload,
)
from main.core.digest import build_stable_digest
from main.methods.method_definition import (
    semantic_conditioned_latent_method_definition_digest,
)
from experiments.runners.semantic_watermark_runtime import (
    SemanticWatermarkRuntimeConfig,
    SemanticWatermarkRuntimeResult,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _direction_record() -> tuple[object, dict[str, object], dict[str, object]]:
    protocol = load_content_survival_direction_protocol(REPOSITORY_ROOT)
    composite = build_content_survival_runtime_method_identity(
        core_method_definition_digest=(
            semantic_conditioned_latent_method_definition_digest()
        ),
        protocol=protocol,
    )
    chain_records: list[dict[str, object]] = []
    for index, role in enumerate(CONTENT_SURVIVAL_CHAIN_ROLES):
        chain_records.append(
            {
                "role": role,
                "chain_index": index,
                "z10_content_sha256": "1" * 64,
                "scheduler_step_timestep": 10.0,
                "attention_geometry_enabled": role.startswith("full_"),
                "supports_paper_claim": False,
            }
        )
    probe_records = []
    scores = (0.8, 0.4, 0.7, 0.3)
    for role, score in zip(CONTENT_SURVIVAL_PROBE_ROLES, scores):
        item = {
            "probe_role": role,
            "probe_sign": 1 if role.endswith("positive") else -1,
            "registered_content_score": score,
            "probe_ratio_ready": True,
            "wrong_key_used_for_selection": False,
            "supports_paper_claim": False,
        }
        item["probe_artifact_digest"] = build_stable_digest(item)
        probe_records.append(item)
    selection = select_shared_content_survival_sign(
        probe_records,
        protocol=protocol,
    )
    base_identity = "2" * 64
    probe_parent = {
        "protocol_identity": protocol.identity_record(),
        "composite_runtime_method_identity": composite,
        "base_latent_identity_digest_random": base_identity,
        "probe_records": probe_records,
        "selection": selection,
    }
    probe_bundle_digest = build_stable_digest(probe_parent)
    for item in chain_records[-2:]:
        item.update(
            {
                "selected_sign": selection["selected_sign"],
                "probe_bundle_digest": probe_bundle_digest,
                "nominal_replay_parent_ready": True,
                "actual_dtype_single_write_count": 1,
            }
        )
    chain_records[-1]["post_write_qk_strict_ready"] = True
    for item in chain_records[-2:]:
        item["nominal_replay_record_digest"] = build_stable_digest(item)
    for item in chain_records:
        item["chain_record_digest"] = build_stable_digest(item)
    feature_record = {
        "feature_source": (
            "frozen_official_clip_get_image_features_and_real_pixels"
        ),
        "clip_model_identity_digest": "3" * 64,
    }
    feature_record["feature_record_digest"] = build_stable_digest(
        feature_record
    )
    record = build_content_survival_direction_record(
        protocol=protocol,
        composite_method_identity=composite,
        base_latent_identity_digest_random=base_identity,
        chain_records=chain_records,
        probe_records=probe_records,
        selection=selection,
        probe_bundle_digest=probe_bundle_digest,
        final_image_feature_record=feature_record,
    )
    return protocol, composite, record


def _copy_protocol_files(root: Path) -> None:
    for relative in (
        Path("configs/content_survival_direction_protocol.json"),
        Path("configs/content_survival_direction_protocol_schema.json"),
    ):
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((REPOSITORY_ROOT / relative).read_bytes())


@pytest.mark.quick
def test_content_survival_config_has_canonical_self_identity() -> None:
    protocol = load_content_survival_direction_protocol(REPOSITORY_ROOT)
    payload = protocol.payload

    assert content_survival_protocol_semantic_digest(payload) == (
        protocol.semantic_digest
    )
    assert protocol.file_sha256 == _sha(
        (REPOSITORY_ROOT / protocol.config_path).read_bytes()
    )
    composite = build_content_survival_runtime_method_identity(
        core_method_definition_digest="a" * 64,
        protocol=protocol,
    )
    assert tuple(composite) == (
        "core_method_definition_digest",
        "survival_protocol_file_sha256",
        "survival_protocol_semantic_digest",
        "protocol_version",
        "survival_protocol_schema_file_sha256",
        "composite_runtime_method_identity_digest",
    )


@pytest.mark.quick
def test_content_survival_config_rejects_unknown_and_digest_drift() -> None:
    protocol = load_content_survival_direction_protocol(REPOSITORY_ROOT)
    unknown = dict(protocol.payload)
    unknown["unknown"] = True
    with pytest.raises(ValueError, match="fields"):
        validate_content_survival_direction_payload(unknown)

    drifted = json.loads(json.dumps(protocol.payload))
    drifted["direction"]["target_ratio"] = 0.002
    with pytest.raises(ValueError, match="constants"):
        validate_content_survival_direction_payload(drifted)


@pytest.mark.quick
@pytest.mark.parametrize(
    ("field_name", "replacement"),
    (
        ("norm", "l2"),
        ("axes", [2, 3]),
        ("numerical_floor", 0.0),
        ("comparison", "less_than"),
    ),
)
def test_content_survival_config_rejects_direction_rule_drift(
    field_name: str,
    replacement: object,
) -> None:
    protocol = load_content_survival_direction_protocol(REPOSITORY_ROOT)
    drifted = json.loads(json.dumps(protocol.payload))
    drifted["direction"][field_name] = replacement
    drifted["survival_protocol_semantic_digest"] = (
        content_survival_protocol_semantic_digest(drifted)
    )

    with pytest.raises(ValueError, match="constants"):
        validate_content_survival_direction_payload(drifted)


@pytest.mark.quick
def test_actual_dtype_probe_uses_fixed_rms_ratio_and_rejects_zero() -> None:
    torch = pytest.importorskip("torch")
    protocol = load_content_survival_direction_protocol(REPOSITORY_ROOT)
    z10 = torch.linspace(0.5, 1.5, 48, dtype=torch.float32).reshape(1, 3, 4, 4)
    direction = torch.linspace(-1.0, 1.0, 48, dtype=torch.float32).reshape(
        1, 3, 4, 4
    )

    candidate, record = materialize_content_survival_probe(
        z10,
        direction,
        sign=1,
        role="full_probe_positive",
        protocol=protocol,
    )

    assert candidate.dtype == z10.dtype
    assert record["probe_actual_tensor_dtype"] == "float32"
    assert record["probe_target_ratio_float64"] == 1.0e-3
    assert record["probe_ratio_ready"] is True
    assert record["probe_ratio_absolute_error_float64"] <= record[
        "probe_ratio_tolerance_float64"
    ]
    with pytest.raises(RuntimeError, match="non-zero"):
        materialize_content_survival_probe(
            z10,
            torch.zeros_like(z10),
            sign=1,
            role="full_probe_positive",
            protocol=protocol,
        )
    with pytest.raises(ValueError, match="not governed"):
        materialize_content_survival_probe(
            z10.to(dtype=torch.float64),
            direction.to(dtype=torch.float64),
            sign=1,
            role="full_probe_positive",
            protocol=protocol,
        )
    nonfinite = direction.clone()
    nonfinite[0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        materialize_content_survival_probe(
            z10,
            nonfinite,
            sign=1,
            role="full_probe_positive",
            protocol=protocol,
        )


@pytest.mark.quick
def test_shared_sign_is_registered_only_and_tie_fails_closed() -> None:
    protocol = load_content_survival_direction_protocol(REPOSITORY_ROOT)
    records = []
    for role, score in zip(CONTENT_SURVIVAL_PROBE_ROLES, (0.9, 0.2, 0.8, 0.1)):
        records.append(
            {
                "probe_role": role,
                "probe_sign": 1 if role.endswith("positive") else -1,
                "registered_content_score": score,
            }
        )
    selection = select_shared_content_survival_sign(records, protocol=protocol)
    assert selection["selected_sign"] == 1
    assert selection["wrong_key_used_for_selection"] is False

    tied = [dict(item, registered_content_score=0.5) for item in records]
    with pytest.raises(RuntimeError, match="conflict or tie"):
        select_shared_content_survival_sign(tied, protocol=protocol)

    conflict = [dict(item) for item in records]
    conflict[2]["registered_content_score"] = 0.1
    conflict[3]["registered_content_score"] = 0.9
    with pytest.raises(RuntimeError, match="conflict or tie"):
        select_shared_content_survival_sign(conflict, protocol=protocol)


@pytest.mark.quick
def test_wrong_key_is_evaluated_only_after_nominal_replay() -> None:
    source = inspect.getsource(
        runtime_module._run_semantic_watermark_runtime_with_content_strength
    )

    assert source.index("select_shared_content_survival_sign") < source.index(
        '"carrier_nominal_replay"'
    )
    assert source.index('"full_nominal_replay"') < source.index(
        '"wrong_key_negative"'
    )
    assert "wrong_key_used_for_selection" in source


@pytest.mark.quick
def test_official_clip_global_path_produces_real_three_image_evidence() -> None:
    torch = pytest.importorskip("torch")

    class Model:
        def get_image_features(self, *, pixel_values: object) -> object:
            values = pixel_values.to(dtype=torch.float32)
            means = values.mean(dim=(2, 3))
            return means.repeat(1, 171)[:, :512]

    runtime = SimpleNamespace(
        prepare_image_pixels=lambda image: image,
        _model=Model(),
        model_identity_digest="4" * 64,
    )
    clean = torch.full((1, 3, 8, 8), 0.5)
    carrier = clean + 1.0e-4
    full = clean + 2.0e-4

    feature = encode_frozen_clip_global_image_feature(runtime, clean)
    final, carrier_record, source = build_three_image_content_survival_evidence(
        runtime=runtime,
        clean_image=clean,
        carrier_only_image=carrier,
        full_image=full,
        minimum_semantic_cosine=0.99,
        maximum_structure_relative_drift=0.01,
        counterfactual_identity_digest="5" * 64,
    )

    assert tuple(feature.shape) == (1, 512)
    assert final["final_image_preservation_gate_ready"] is True
    assert carrier_record[
        "carrier_only_counterfactual_three_way_preservation_gate_ready"
    ] is True
    assert source["feature_source"] == (
        "frozen_official_clip_get_image_features_and_real_pixels"
    )


@pytest.mark.quick
def test_artifact_bundle_validates_complete_set_and_rejects_tamper(
    tmp_path: Path,
) -> None:
    protocol, composite, direction_record = _direction_record()
    output_dir = tmp_path / "outputs" / "run"
    output_dir.mkdir(parents=True)
    leaves = []
    paths = {}
    for role, name in (
        ("clean_nominal_image", "clean.png"),
        ("carrier_nominal_image", "carrier.png"),
        ("full_nominal_image", "full.png"),
        ("full_nominal_update", "full.jsonl"),
        ("carrier_nominal_update", "carrier.jsonl"),
        ("nominal_detection_records", "detections.jsonl"),
        ("content_survival_direction_record", "protocol.json"),
    ):
        path = output_dir / name
        payload = (
            json.dumps(direction_record, sort_keys=True).encode("utf-8")
            if role == "content_survival_direction_record"
            else role.encode("utf-8")
        )
        path.write_bytes(payload)
        relative = path.relative_to(tmp_path).as_posix()
        paths[role] = relative
        leaves.append(
            {
                "role": role,
                "path": relative,
                "size_bytes": len(payload),
                "sha256": _sha(payload),
            }
        )
    binding = build_content_survival_artifact_binding(
        run_id="run",
        composite_method_identity=composite,
        protocol_record=direction_record,
        leaf_records=leaves,
    )
    binding_path = output_dir / "content_survival_artifact_binding.json"
    binding_path.write_text(json.dumps(binding), encoding="utf-8")
    binding_relative = binding_path.relative_to(tmp_path).as_posix()
    result_path = output_dir / "runtime_result.json"
    manifest_path = output_dir / "manifest.local.json"
    metadata = {
        "composite_runtime_method_identity": composite,
        "content_survival_direction_record": direction_record,
        "prompt_saliency_model_identity_digest": "3" * 64,
        "content_survival_direction_record_path": paths[
            "content_survival_direction_record"
        ],
        "content_survival_artifact_binding_path": binding_relative,
        "content_survival_artifact_binding_digest": binding[
            "content_survival_artifact_binding_digest"
        ],
        "content_survival_artifact_binding_file_sha256": _sha(
            binding_path.read_bytes()
        ),
        "carrier_only_image_path": paths["carrier_nominal_image"],
        "carrier_only_update_record_path": paths["carrier_nominal_update"],
        "final_image_preservation": {
            "final_image_preservation_gate_ready": True
        },
        "carrier_only_final_image_preservation": {
            "carrier_only_counterfactual_three_way_preservation_gate_ready": True
        },
        "final_image_attention_observability": {
            "final_image_attention_observability_gate_ready": True
        },
        "old_content_runtime_artifact_compatible": False,
    }
    result = {
        "run_id": "run",
        "run_decision": "pass",
        "clean_image_path": paths["clean_nominal_image"],
        "watermarked_image_path": paths["full_nominal_image"],
        "update_record_path": paths["full_nominal_update"],
        "detection_record_path": paths["nominal_detection_records"],
        "manifest_path": manifest_path.relative_to(tmp_path).as_posix(),
        "update_count": 1,
        "elapsed_seconds": 1.0,
        "metadata": metadata,
    }
    result_path.write_text(json.dumps(result), encoding="utf-8")
    output_paths = [item["path"] for item in leaves]
    output_paths.extend(
        (
            binding_relative,
            result_path.relative_to(tmp_path).as_posix(),
            manifest_path.relative_to(tmp_path).as_posix(),
        )
    )
    manifest_config = {"identity": "test"}
    manifest = {
        "config": manifest_config,
        "config_digest": build_stable_digest(manifest_config),
        "output_paths": output_paths,
        "metadata": {
            "composite_runtime_method_identity": composite,
            "content_survival_artifact_binding_digest": binding[
                "content_survival_artifact_binding_digest"
            ],
            "content_survival_artifact_binding_file_sha256": _sha(
                binding_path.read_bytes()
            ),
            "runtime_result_sha256": _sha(result_path.read_bytes()),
        },
    }

    assert validate_content_survival_artifact_bundle(
        repository_root=tmp_path,
        result_payload=result,
        manifest=manifest,
        expected_protocol=protocol,
        expected_core_method_definition_digest=(
            semantic_conditioned_latent_method_definition_digest()
        ),
    )
    (tmp_path / paths["clean_nominal_image"]).write_bytes(b"tampered")
    assert not validate_content_survival_artifact_bundle(
        repository_root=tmp_path,
        result_payload=result,
        manifest=manifest,
        expected_protocol=protocol,
        expected_core_method_definition_digest=(
            semantic_conditioned_latent_method_definition_digest()
        ),
    )


@pytest.mark.quick
def test_writer_publishes_manifest_last_after_shared_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _copy_protocol_files(tmp_path)
    protocol = load_content_survival_direction_protocol(tmp_path)
    _source_protocol, composite, direction_record = _direction_record()
    config = SemanticWatermarkRuntimeConfig(
        output_dir="outputs/content_survival_writer_test",
        prompt_id="writer_test",
        split="calibration",
        standard_attack_profiles=(),
        diffusion_attacks_enabled=False,
    )
    counterfactual = {
        "carrier_only_counterfactual_identity_digest": "6" * 64,
        "carrier_only_counterfactual_atom_content_digest": "7" * 64,
    }
    source_result = SemanticWatermarkRuntimeResult(
        run_id="content_survival_writer_test",
        run_decision="pass",
        clean_image_path="",
        watermarked_image_path="",
        update_record_path="",
        detection_record_path="",
        manifest_path="",
        update_count=1,
        elapsed_seconds=1.0,
        metadata={
            "composite_runtime_method_identity": composite,
            "content_survival_direction_record": direction_record,
            "prompt_saliency_model_identity_digest": "3" * 64,
            "carrier_only_counterfactual": counterfactual,
            "final_image_preservation": {
                "final_image_preservation_gate_ready": True
            },
            "carrier_only_final_image_preservation": {
                "carrier_only_counterfactual_three_way_preservation_gate_ready": True
            },
            "final_image_attention_observability": {
                "final_image_attention_observability_gate_ready": True
            },
            "formal_randomization_reference": {"identity": "writer-test"},
            "old_content_runtime_artifact_compatible": False,
        },
    )
    images = (
        Image.new("RGB", (8, 8), "white"),
        Image.new("RGB", (8, 8), "gray"),
        Image.new("RGB", (8, 8), "black"),
    )
    detections = tuple(
        {
            "sample_role": role,
            "metadata": {},
        }
        for role in (
            "clean_negative",
            "positive_source",
            "wrong_key_negative",
        )
    )
    monkeypatch.setattr(
        runtime_module,
        "_run_semantic_watermark_runtime_with_content_strength",
        lambda *_args, **_kwargs: (
            source_result,
            ({"role": "full_nominal_replay"},),
            ({"role": "carrier_nominal_replay"},),
            detections,
            images[0],
            images[2],
            images[1],
            {},
        ),
    )
    monkeypatch.setattr(
        runtime_module,
        "_bind_detection_qk_to_pixels",
        lambda record, _image: record,
    )
    events = []
    original_validator = runtime_module.validate_content_survival_artifact_bundle
    original_replace = runtime_module.os.replace

    def observed_validator(**kwargs: object) -> bool:
        events.append("validate")
        return original_validator(**kwargs)

    def observed_replace(source: object, destination: object) -> None:
        events.append(f"rename:{Path(destination).name}")
        original_replace(source, destination)

    monkeypatch.setattr(
        runtime_module,
        "validate_content_survival_artifact_bundle",
        observed_validator,
    )
    monkeypatch.setattr(runtime_module.os, "replace", observed_replace)

    resolved = (
        runtime_module._write_semantic_watermark_runtime_outputs_with_content_strength(
            config,
            root=tmp_path,
            references=SimpleNamespace(),
            verified_formal_execution_lock={},
            content_strength_common_multiplier=1.0,
            calibration_content_strength_sensitivity=False,
        )
    )
    manifest_path = tmp_path / resolved.manifest_path
    result_path = manifest_path.parent / "runtime_result.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    result = json.loads(result_path.read_text(encoding="utf-8"))

    assert manifest_path.is_file()
    assert not manifest_path.with_name(
        f"{manifest_path.name}.partial"
    ).exists()
    assert events[-2:] == ["validate", "rename:manifest.local.json"]
    assert validate_content_survival_artifact_bundle(
        repository_root=tmp_path,
        result_payload=result,
        manifest=manifest,
        expected_protocol=protocol,
        expected_core_method_definition_digest=(
            semantic_conditioned_latent_method_definition_digest()
        ),
    )
