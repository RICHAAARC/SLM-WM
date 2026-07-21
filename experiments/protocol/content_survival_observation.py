"""Govern an isolated, non-claim observation of content-carrier survival."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import fmean
from typing import Any, Mapping, Sequence

from main.core.digest import build_stable_digest, tensor_content_sha256
from main.methods.carrier.blind_content_score import compute_blind_content_score
from main.methods.carrier.high_frequency_tail import (
    build_high_frequency_tail_template,
)
from main.methods.carrier.low_frequency import build_low_frequency_template
from experiments.runtime.scientific_content_binding import (
    canonical_rgb_uint8_content_record,
)


CONTENT_SURVIVAL_OBSERVATION_SCHEMA = "slm_wm_content_survival_observation"
CONTENT_SURVIVAL_OBSERVATION_CONFIG_PATH = Path(
    "configs/content_survival_observation.json"
)
CONTENT_SURVIVAL_OBSERVATION_SCHEMA_PATH = Path(
    "configs/content_survival_observation_schema.json"
)
CONTENT_SURVIVAL_OBSERVATION_ROLES = (
    "post_write_z10",
    "pre_vae_terminal_latent",
    "image_reencoded_latent",
)
CONTENT_SURVIVAL_ROUTING_MODES = ("semantic", "uniform")
CONTENT_SURVIVAL_CARRIER_MODES = ("lf_only", "hf_only", "dual")
CONTENT_SURVIVAL_CHAIN_ROLES = (
    "full_probe_positive",
    "full_probe_negative",
    "carrier_probe_positive",
    "carrier_probe_negative",
    "nominal_before_positive",
    "nominal_after_selected",
)
CONTENT_SURVIVAL_PROMPT_IDS = (
    "prompt_e6db4109e01246bc",
    "prompt_d026a34e14f1806b",
    "prompt_4ba678eb7a3abc94",
    "prompt_184d2cc052881c3e",
)
CONTENT_SURVIVAL_WRONG_KEY_COUNT = 32
CONTENT_SURVIVAL_CELL_COUNT = 24
CONTENT_SURVIVAL_CHAIN_COUNT = 148
CONTENT_SURVIVAL_EVALUATION_COUNT = 29_304
CONTENT_SURVIVAL_CLAIM_BOUNDARY = {
    "diagnostic_only": True,
    "supports_paper_claim": False,
    "candidate_promotion_allowed": False,
    "qualification_evidence": False,
}
CONTENT_SURVIVAL_M0_DEVIATIONS = (
    "full_and_carrier_directions_and_common_gamma_are_not_mechanically_shared",
    "counterfactual_identity_is_manually_constructed",
    "partial_manifest_bytes_are_not_reloaded_before_validation",
    "bfloat16_realized_ratio_tolerance_is_too_wide",
    "scheduler_identity_is_limited_to_z10_and_timestep",
)
_SEMANTIC_DOMAIN = b"slm_wm_content_survival_observation_semantic_v1\0"
_ROSTER_SEMANTIC_DOMAIN = b"slm_wm_content_survival_observation_roster_semantic_v1\0"
_ORACLE_DOMAIN = b"slm_wm_content_survival_routed_oracle_v1\0"
_CELL_BINDING_DOMAIN = b"slm_wm_content_survival_observation_cell_v1\0"
_PARENT_CHILD_BINDING_DOMAIN = (
    b"slm_wm_content_survival_observation_parent_child_v1\0"
)
_EXECUTION_IDENTITY_DOMAIN = b"slm_wm_content_survival_observation_execution_v1\0"

_CONTENT_CHAIN_PARENT_FIELDS = {
    "role",
    "routing_mode",
    "carrier_mode",
    "cell_identity_digest",
    "base_latent_identity",
    "base_latent_identity_digest_random",
    "scheduler_identity",
    "scheduler_identity_digest",
    "observation_run_identity_digest",
    "prompt_text_digest",
    "prompt_config_digest",
    "key_roster_digest_random",
    "protocol_semantic_digest",
    "prompt_roster_semantic_digest",
    "prompt_roster_artifact_file_sha256",
    "post_write_z10_content_sha256",
    "pre_vae_terminal_latent_content_sha256",
    "image_rgb_uint8_content_sha256",
    "image_rgb_uint8_content_schema",
    "image_width",
    "image_height",
    "image_reencoded_latent_content_sha256",
    "callback_record",
    *CONTENT_SURVIVAL_CLAIM_BOUNDARY,
}
_CLEAN_CHAIN_PARENT_FIELDS = _CONTENT_CHAIN_PARENT_FIELDS - {
    "cell_identity_digest",
    "protocol_semantic_digest",
    "prompt_roster_semantic_digest",
    "prompt_roster_artifact_file_sha256",
}
_CALLBACK_COMMON_FIELDS = {
    "scheduler_step_index",
    "scheduler_step_timestep",
    "z9_content_sha256",
    "z10_before_write_content_sha256",
}
_CALLBACK_CONTENT_FIELDS = {
    "routing_identity_digest",
    "registered_direction_content_sha256",
    "lf_update_content_sha256",
    "hf_update_content_sha256",
    "geometry_update_content_sha256",
    "attention_geometry_enabled",
}
_CALLBACK_PROBE_FIELDS = {
    "probe_role",
    "probe_sign",
    "direction_norm",
    "direction_axes",
    "direction_accumulation_dtype",
    "direction_unit_content_sha256",
    "probe_target_ratio_float64",
    "probe_actual_tensor_dtype",
    "probe_realized_ratio_float64",
    "probe_ratio_absolute_error_float64",
    "probe_ratio_tolerance_float64",
    "probe_ratio_comparison",
    "probe_ratio_ready",
    "probe_materialized_latent_content_sha256",
    "supports_paper_claim",
    "probe_record_digest",
}
_CALLBACK_REPLAY_FIELDS = {
    "selected_sign",
    "common_gamma",
    "write_identity_digest",
    "actual_dtype_single_write_digest",
}
_PROBE_SIGN_BY_ROLE = {
    "full_probe_positive": 1,
    "full_probe_negative": -1,
    "carrier_probe_positive": 1,
    "carrier_probe_negative": -1,
}

CONTENT_SURVIVAL_SOURCE_EVIDENCE = {
    "run_id": "content_strength_sensitivity_20260721T012732CST_db324b7",
    "formal_execution_commit": "db324b7c86a1bef305114fe83db44dfed04fd706",
    "archive_sha256_file_sha256": (
        "039f1be0264f1162c8789d43da5af1eb624c8e3eb00c78cc819019910c0de964"
    ),
    "archive_file_manifest_sha256": (
        "0b1f13bebaf5473a54ae19a8e93a3d23be0514fc20339437cd2f39e2a0739312"
    ),
    "run_manifest_sha256": (
        "35cf0042f85b13b9e03f1a98adc9b1d622cb3f9e34dc9505514ce07a09428338"
    ),
}
CONTENT_SURVIVAL_ROUTING_REFERENCE_IDENTITY = {
    "semantic_digest": (
        "58f84f0d9456806091971fcecb3a06cf980d8e64fd626752b494d46d4170bc8b"
    ),
    "file_sha256": (
        "af2c56004667abf96ca6d1b14d5a58fa3d1ac5935e59dbc3c0de3b55138920c6"
    ),
}
CONTENT_SURVIVAL_SELECTION_RECORDS = (
    {
        "prompt_id": "prompt_e6db4109e01246bc",
        "selection_role": "stable_winner",
        "minimum_registered_minus_wrong_margin": 0.023721,
        "maximum_registered_minus_wrong_margin": 0.0239117,
    },
    {
        "prompt_id": "prompt_d026a34e14f1806b",
        "selection_role": "stable_winner",
        "minimum_registered_minus_wrong_margin": 0.0168801,
        "maximum_registered_minus_wrong_margin": 0.0171455,
    },
    {
        "prompt_id": "prompt_4ba678eb7a3abc94",
        "selection_role": "stable_loser",
        "minimum_registered_minus_wrong_margin": -0.0391,
        "maximum_registered_minus_wrong_margin": -0.0389263,
    },
    {
        "prompt_id": "prompt_184d2cc052881c3e",
        "selection_role": "stable_loser",
        "minimum_registered_minus_wrong_margin": -0.0383577,
        "maximum_registered_minus_wrong_margin": -0.0381536,
    },
)


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _domain_digest(domain: bytes, value: Any) -> str:
    digest = hashlib.sha256()
    digest.update(domain)
    digest.update(_canonical_json_bytes(value))
    return digest.hexdigest()


def _required_sha256(value: Any, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def content_survival_observation_semantic_digest(
    payload: Mapping[str, Any],
) -> str:
    resolved = dict(payload)
    resolved.pop("protocol_semantic_digest", None)
    return _domain_digest(_SEMANTIC_DOMAIN, resolved)


def _roster_semantic_payload(roster: Mapping[str, Any]) -> dict[str, Any]:
    resolved = dict(roster)
    resolved.pop("roster_semantic_digest", None)
    resolved.pop("roster_artifact_file_sha256", None)
    return resolved


def content_survival_observation_roster_semantic_digest(
    roster: Mapping[str, Any],
) -> str:
    return _domain_digest(_ROSTER_SEMANTIC_DOMAIN, _roster_semantic_payload(roster))


def content_survival_observation_roster_artifact_bytes(
    roster: Mapping[str, Any],
) -> bytes:
    resolved = dict(roster)
    resolved.pop("roster_artifact_file_sha256", None)
    return _canonical_json_bytes(resolved) + b"\n"


@dataclass(frozen=True)
class ContentSurvivalObservationProtocol:
    payload: dict[str, Any]
    semantic_digest: str
    file_sha256: str
    schema_file_sha256: str
    config_path: str
    schema_path: str

    def identity_record(self) -> dict[str, Any]:
        return {
            "protocol_schema": CONTENT_SURVIVAL_OBSERVATION_SCHEMA,
            "protocol_version": self.payload["protocol_version"],
            "protocol_semantic_digest": self.semantic_digest,
            "protocol_file_sha256": self.file_sha256,
            "protocol_schema_file_sha256": self.schema_file_sha256,
            "protocol_config_path": self.config_path,
            "protocol_schema_path": self.schema_path,
        }


def validate_content_survival_observation_payload(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate every fixed matrix, scoring, publication, and claim boundary."""

    resolved = dict(payload)
    if set(resolved) != {
        "protocol_schema",
        "protocol_version",
        "roster",
        "routing_reference_registry",
        "source_evidence",
        "matrix",
        "evaluation_protocol",
        "selection_protocol",
        "publish_protocol",
        "known_m0_deviations",
        "claim_boundary",
        "protocol_semantic_digest",
    }:
        raise ValueError("content survival observation fields drifted")
    if (
        resolved["protocol_schema"] != CONTENT_SURVIVAL_OBSERVATION_SCHEMA
        or resolved["protocol_version"] != "content_survival_observation_v1"
    ):
        raise ValueError("content survival observation identity drifted")
    roster = resolved["roster"]
    expected_roster = {
        "prompt_ids": list(CONTENT_SURVIVAL_PROMPT_IDS),
        "selection_policy": (
            "two_stable_winners_by_descending_minimum_margin_then_"
            "two_stable_losers_by_ascending_maximum_margin"
        ),
        "selection_records": [dict(record) for record in CONTENT_SURVIVAL_SELECTION_RECORDS],
        "wrong_key_count": CONTENT_SURVIVAL_WRONG_KEY_COUNT,
        "wrong_key_domain": (
            "slm_wm_content_survival_observation_wrong_key_v1"
        ),
        "roster_semantic_digest": roster.get("roster_semantic_digest"),
        "roster_artifact_file_sha256": roster.get("roster_artifact_file_sha256"),
    }
    if roster != expected_roster:
        raise ValueError("content survival observation roster drifted")
    roster_semantic_digest = _required_sha256(
        roster["roster_semantic_digest"], "roster_semantic_digest"
    )
    if roster_semantic_digest != content_survival_observation_roster_semantic_digest(
        roster
    ):
        raise ValueError("content survival roster semantic digest mismatch")
    roster_file_sha256 = _required_sha256(
        roster["roster_artifact_file_sha256"], "roster_artifact_file_sha256"
    )
    if roster_file_sha256 != hashlib.sha256(
        content_survival_observation_roster_artifact_bytes(roster)
    ).hexdigest():
        raise ValueError("content survival roster artifact digest mismatch")
    if resolved["source_evidence"] != CONTENT_SURVIVAL_SOURCE_EVIDENCE:
        raise ValueError("content survival source evidence drifted")
    if (
        resolved["routing_reference_registry"]
        != CONTENT_SURVIVAL_ROUTING_REFERENCE_IDENTITY
    ):
        raise ValueError("content survival routing registry identity drifted")
    matrix = resolved["matrix"]
    if matrix != {
        "routing_modes": list(CONTENT_SURVIVAL_ROUTING_MODES),
        "carrier_modes": list(CONTENT_SURVIVAL_CARRIER_MODES),
        "chain_roles": list(CONTENT_SURVIVAL_CHAIN_ROLES),
        "cell_chain_count": 6,
        "clean_chain_count_per_prompt": 1,
        "cell_count": CONTENT_SURVIVAL_CELL_COUNT,
        "chain_count": CONTENT_SURVIVAL_CHAIN_COUNT,
    }:
        raise ValueError("content survival observation matrix drifted")
    if resolved["evaluation_protocol"] != {
        "observation_roles": list(CONTENT_SURVIVAL_OBSERVATION_ROLES),
        "score_families": [
            "blind_full_template",
            "routed_template_oracle",
        ],
        "evaluation_count": CONTENT_SURVIVAL_EVALUATION_COUNT,
    }:
        raise ValueError("content survival evaluation protocol drifted")
    if resolved["selection_protocol"] != {
        "probe_roles": list(CONTENT_SURVIVAL_CHAIN_ROLES[:4]),
        "candidate_signs": [1, -1],
        "primary_objective": (
            "maximize_min_registered_blind_content_score"
        ),
        "secondary_objective": (
            "maximize_sum_registered_blind_content_score"
        ),
        "tie_policy": "fail_closed",
        "wrong_key_access": "forbidden_until_sign_selected",
    }:
        raise ValueError("content survival selection protocol drifted")
    if resolved["publish_protocol"] != {
        "complete_marker": "cell_manifest.json",
        "partial_marker": "cell_manifest.partial.json",
        "partial_resume_policy": "redo_same_cell_only",
        "publish_order": [
            "immutable_observations",
            "leaf_digests",
            "result",
            "binding",
            "partial_manifest",
            "full_validation",
            "fsync",
            "complete_marker_atomic_rename",
        ],
    }:
        raise ValueError("content survival publish protocol drifted")
    if tuple(resolved["known_m0_deviations"]) != CONTENT_SURVIVAL_M0_DEVIATIONS:
        raise ValueError("known M0 deviations drifted")
    if resolved["claim_boundary"] != CONTENT_SURVIVAL_CLAIM_BOUNDARY:
        raise ValueError("content survival claim boundary drifted")
    claimed = _required_sha256(
        resolved["protocol_semantic_digest"],
        "protocol_semantic_digest",
    )
    if claimed != content_survival_observation_semantic_digest(resolved):
        raise ValueError("content survival semantic digest mismatch")
    return resolved


def load_content_survival_observation_protocol(
    repository_root: str | Path,
) -> ContentSurvivalObservationProtocol:
    root = Path(repository_root).resolve()
    config_path = (root / CONTENT_SURVIVAL_OBSERVATION_CONFIG_PATH).resolve()
    schema_path = (root / CONTENT_SURVIVAL_OBSERVATION_SCHEMA_PATH).resolve()
    config_path.relative_to(root)
    schema_path.relative_to(root)
    config_bytes = config_path.read_bytes()
    schema_bytes = schema_path.read_bytes()
    payload = json.loads(config_bytes.decode("utf-8"))
    schema = json.loads(schema_bytes.decode("utf-8"))
    if type(payload) is not dict or type(schema) is not dict:
        raise TypeError("content survival protocol files must be JSON objects")
    if schema.get("$id") != CONTENT_SURVIVAL_OBSERVATION_SCHEMA:
        raise ValueError("content survival schema identity drifted")
    validated = validate_content_survival_observation_payload(payload)
    return ContentSurvivalObservationProtocol(
        payload=validated,
        semantic_digest=validated["protocol_semantic_digest"],
        file_sha256=hashlib.sha256(config_bytes).hexdigest(),
        schema_file_sha256=hashlib.sha256(schema_bytes).hexdigest(),
        config_path=CONTENT_SURVIVAL_OBSERVATION_CONFIG_PATH.as_posix(),
        schema_path=CONTENT_SURVIVAL_OBSERVATION_SCHEMA_PATH.as_posix(),
    )


def build_content_survival_observation_roster(
    registered_key_material: str,
    *,
    protocol: ContentSurvivalObservationProtocol,
) -> dict[str, Any]:
    """Build the frozen prompt and domain-separated 32-key null roster."""

    if type(registered_key_material) is not str or not registered_key_material:
        raise ValueError("registered_key_material must be non-empty")
    registered_digest = build_stable_digest(
        {"key_material": registered_key_material}
    )
    domain = protocol.payload["roster"]["wrong_key_domain"]
    wrong_keys: list[dict[str, Any]] = []
    for index in range(CONTENT_SURVIVAL_WRONG_KEY_COUNT):
        derivation = build_stable_digest(
            {
                "domain": domain,
                "registered_key_material_digest_random": registered_digest,
                "wrong_key_index": index,
            }
        )
        material = f"slm-wm-observation-wrong-key:{derivation}"
        wrong_keys.append(
            {
                "wrong_key_index": index,
                "wrong_key_material": material,
                "wrong_key_material_digest_random": build_stable_digest(
                    {"key_material": material}
                ),
            }
        )
    digest_payload = {
        "prompt_ids": list(CONTENT_SURVIVAL_PROMPT_IDS),
        "registered_key_material_digest_random": registered_digest,
        "wrong_key_domain": domain,
        "wrong_key_material_digests_random": [
            item["wrong_key_material_digest_random"] for item in wrong_keys
        ],
        "protocol_semantic_digest": protocol.semantic_digest,
        "prompt_roster_semantic_digest": protocol.payload["roster"][
            "roster_semantic_digest"
        ],
        "prompt_roster_artifact_file_sha256": protocol.payload["roster"][
            "roster_artifact_file_sha256"
        ],
    }
    return {
        **digest_payload,
        "wrong_keys": wrong_keys,
        "roster_digest_random": build_stable_digest(digest_payload),
    }


def build_content_survival_cell_identity(
    *,
    prompt_id: str,
    routing_mode: str,
    carrier_mode: str,
    generation_seed_random: int,
    registered_key_material_digest_random: str,
    formal_execution_lock: Mapping[str, Any],
    execution_environment_identity: Mapping[str, Any],
    model_id: str,
    model_revision: str,
    vision_model_id: str,
    vision_model_revision: str,
    semantic_watermark_runtime_config_digest: str,
    prompt_text_digest: str,
    key_roster: Mapping[str, Any],
    protocol: ContentSurvivalObservationProtocol,
) -> dict[str, Any]:
    if prompt_id not in CONTENT_SURVIVAL_PROMPT_IDS:
        raise ValueError("prompt_id is not in the frozen roster")
    if routing_mode not in CONTENT_SURVIVAL_ROUTING_MODES:
        raise ValueError("routing_mode is not governed")
    if carrier_mode not in CONTENT_SURVIVAL_CARRIER_MODES:
        raise ValueError("carrier_mode is not governed")
    if type(generation_seed_random) is not int or generation_seed_random < 0:
        raise ValueError("generation_seed_random must be a nonnegative integer")
    _required_sha256(
        registered_key_material_digest_random,
        "registered_key_material_digest_random",
    )
    lock = dict(formal_execution_lock)
    if set(lock) != {
        "formal_execution_lock_schema",
        "formal_execution_commit",
        "formal_execution_head_detached",
        "formal_execution_worktree_clean",
        "formal_execution_lock_ready",
        "formal_execution_lock_digest",
    }:
        raise ValueError("formal execution lock fields drifted")
    commit = lock["formal_execution_commit"]
    if type(commit) is not str or len(commit) != 40 or any(
        character not in "0123456789abcdef" for character in commit
    ):
        raise ValueError("formal_execution_commit must be a full Git commit")
    if lock["formal_execution_lock_digest"] != build_stable_digest(
        {key: value for key, value in lock.items() if key != "formal_execution_lock_digest"}
    ):
        raise ValueError("formal execution lock digest mismatch")
    if (
        lock["formal_execution_head_detached"] is not True
        or lock["formal_execution_worktree_clean"] is not True
        or lock["formal_execution_lock_ready"] is not True
    ):
        raise ValueError("formal execution lock is not ready")
    environment_identity = validate_content_survival_execution_environment_identity(
        execution_environment_identity
    )
    if not all((model_id, model_revision, vision_model_id, vision_model_revision)):
        raise ValueError("model ids and revisions must be non-empty")
    config_digest = _required_sha256(
        semantic_watermark_runtime_config_digest,
        "semantic_watermark_runtime_config_digest",
    )
    prompt_digest = _required_sha256(prompt_text_digest, "prompt_text_digest")
    key_roster_payload = {
        key: value
        for key, value in key_roster.items()
        if key not in {"wrong_keys", "roster_digest_random"}
    }
    wrong_keys = key_roster.get("wrong_keys")
    if (
        type(wrong_keys) is not list
        or len(wrong_keys) != CONTENT_SURVIVAL_WRONG_KEY_COUNT
        or [record.get("wrong_key_index") for record in wrong_keys]
        != list(range(CONTENT_SURVIVAL_WRONG_KEY_COUNT))
        or len(
            {
                record.get("wrong_key_material_digest_random")
                for record in wrong_keys
            }
        )
        != CONTENT_SURVIVAL_WRONG_KEY_COUNT
        or key_roster.get("roster_digest_random")
        != build_stable_digest(key_roster_payload)
    ):
        raise ValueError("key roster content or digest drifted")
    if key_roster.get("registered_key_material_digest_random") != (
        registered_key_material_digest_random
    ):
        raise ValueError("registered key roster identity mismatch")
    key_roster_digest = _required_sha256(
        key_roster.get("roster_digest_random"), "key_roster_digest_random"
    )
    observation_run_identity = build_content_survival_observation_run_identity(
        formal_execution_lock=lock,
        execution_environment_identity=environment_identity,
        model_id=model_id,
        model_revision=model_revision,
        vision_model_id=vision_model_id,
        vision_model_revision=vision_model_revision,
        key_roster=key_roster,
        protocol=protocol,
    )
    payload = {
        "prompt_id": prompt_id,
        "routing_mode": routing_mode,
        "carrier_mode": carrier_mode,
        "generation_seed_random": generation_seed_random,
        "registered_key_material_digest_random": (
            registered_key_material_digest_random
        ),
        "actual_formal_execution_commit": commit,
        "formal_execution_lock": lock,
        "formal_execution_lock_digest": lock["formal_execution_lock_digest"],
        "execution_environment_identity": environment_identity,
        "execution_environment_identity_digest": environment_identity[
            "execution_environment_identity_digest"
        ],
        "observation_run_identity": observation_run_identity,
        "observation_run_identity_digest": observation_run_identity[
            "observation_run_identity_digest"
        ],
        "model_id": model_id,
        "model_revision": model_revision,
        "model_identity_digest": build_stable_digest(
            {"model_id": model_id, "model_revision": model_revision}
        ),
        "runtime_model_bundle_identity_digest": build_stable_digest(
            {
                "model_id": model_id,
                "model_revision": model_revision,
                "vision_model_id": vision_model_id,
                "vision_model_revision": vision_model_revision,
            }
        ),
        "vision_model_id": vision_model_id,
        "vision_model_revision": vision_model_revision,
        "semantic_watermark_runtime_config_digest": config_digest,
        "prompt_text_digest": prompt_digest,
        "prompt_config_digest": config_digest,
        "routing_reference_registry_semantic_digest": protocol.payload[
            "routing_reference_registry"
        ]["semantic_digest"],
        "routing_reference_registry_file_sha256": protocol.payload[
            "routing_reference_registry"
        ]["file_sha256"],
        "prompt_roster_semantic_digest": protocol.payload["roster"][
            "roster_semantic_digest"
        ],
        "prompt_roster_artifact_file_sha256": protocol.payload["roster"][
            "roster_artifact_file_sha256"
        ],
        "key_roster_digest_random": key_roster_digest,
        "wrong_key_material_digests_random": [
            record["wrong_key_material_digest_random"]
            for record in wrong_keys
        ],
        "chain_roles": list(CONTENT_SURVIVAL_CHAIN_ROLES),
        "protocol_semantic_digest": protocol.semantic_digest,
        **CONTENT_SURVIVAL_CLAIM_BOUNDARY,
    }
    return {
        **payload,
        "cell_identity_digest": build_stable_digest(payload),
    }


def validate_content_survival_execution_environment_identity(
    identity: Mapping[str, Any],
) -> dict[str, Any]:
    resolved = dict(identity)
    digest = resolved.pop("execution_environment_identity_digest", None)
    expected_fields = {
        "orchestrator_profile_digest",
        "orchestrator_complete_hash_lock_digest",
        "orchestrator_inspection_digest",
        "scientific_profile_id",
        "scientific_profile_digest",
        "scientific_direct_requirements_digest",
        "scientific_complete_hash_lock_digest",
        "scientific_dependency_environment_report_digest",
        "scientific_python_executable_sha256",
    }
    if set(resolved) != expected_fields or resolved["scientific_profile_id"] != (
        "sd35_method_runtime_gpu"
    ):
        raise ValueError("execution environment identity fields drifted")
    for field in expected_fields - {"scientific_profile_id"}:
        _required_sha256(resolved[field], field)
    expected_digest = _domain_digest(_EXECUTION_IDENTITY_DOMAIN, resolved)
    if digest != expected_digest:
        raise ValueError("execution environment identity digest mismatch")
    return {**resolved, "execution_environment_identity_digest": expected_digest}


def build_content_survival_execution_environment_identity(
    identity: Mapping[str, Any],
) -> dict[str, Any]:
    payload = dict(identity)
    if "execution_environment_identity_digest" in payload:
        raise ValueError("execution environment input cannot claim its own digest")
    return validate_content_survival_execution_environment_identity(
        {
            **payload,
            "execution_environment_identity_digest": _domain_digest(
                _EXECUTION_IDENTITY_DOMAIN, payload
            ),
        }
    )


def build_content_survival_observation_run_identity(
    *,
    formal_execution_lock: Mapping[str, Any],
    execution_environment_identity: Mapping[str, Any],
    model_id: str,
    model_revision: str,
    vision_model_id: str,
    vision_model_revision: str,
    key_roster: Mapping[str, Any],
    protocol: ContentSurvivalObservationProtocol,
) -> dict[str, Any]:
    lock = dict(formal_execution_lock)
    if (
        set(lock)
        != {
            "formal_execution_lock_schema",
            "formal_execution_commit",
            "formal_execution_head_detached",
            "formal_execution_worktree_clean",
            "formal_execution_lock_ready",
            "formal_execution_lock_digest",
        }
        or lock.get("formal_execution_head_detached") is not True
        or lock.get("formal_execution_worktree_clean") is not True
        or lock.get("formal_execution_lock_ready") is not True
        or lock.get("formal_execution_lock_digest")
        != build_stable_digest(
            {
                key: value
                for key, value in lock.items()
                if key != "formal_execution_lock_digest"
            }
        )
    ):
        raise ValueError("observation run requires a complete validated formal lock")
    environment_identity = validate_content_survival_execution_environment_identity(
        execution_environment_identity
    )
    payload = {
        "formal_execution_lock": lock,
        "formal_execution_lock_digest": lock[
            "formal_execution_lock_digest"
        ],
        "actual_formal_execution_commit": lock[
            "formal_execution_commit"
        ],
        "execution_environment_identity": environment_identity,
        "execution_environment_identity_digest": environment_identity[
            "execution_environment_identity_digest"
        ],
        "model_id": model_id,
        "model_revision": model_revision,
        "model_identity_digest": build_stable_digest(
            {"model_id": model_id, "model_revision": model_revision}
        ),
        "runtime_model_bundle_identity_digest": build_stable_digest(
            {
                "model_id": model_id,
                "model_revision": model_revision,
                "vision_model_id": vision_model_id,
                "vision_model_revision": vision_model_revision,
            }
        ),
        "vision_model_id": vision_model_id,
        "vision_model_revision": vision_model_revision,
        "protocol_identity": protocol.identity_record(),
        "source_evidence": dict(protocol.payload["source_evidence"]),
        "routing_reference_registry": dict(
            protocol.payload["routing_reference_registry"]
        ),
        "prompt_roster_semantic_digest": protocol.payload["roster"][
            "roster_semantic_digest"
        ],
        "prompt_roster_artifact_file_sha256": protocol.payload["roster"][
            "roster_artifact_file_sha256"
        ],
        "key_roster_digest_random": key_roster["roster_digest_random"],
    }
    return {
        **payload,
        "observation_run_identity_digest": build_stable_digest(payload),
    }


def select_content_survival_observation_sign(
    registered_probe_scores: Mapping[str, float],
) -> dict[str, Any]:
    """Select one sign using only registered blind scores from four probes."""

    expected = set(CONTENT_SURVIVAL_CHAIN_ROLES[:4])
    if set(registered_probe_scores) != expected:
        raise ValueError("sign selection requires exactly four registered probes")
    scores = {key: float(value) for key, value in registered_probe_scores.items()}
    if not all(math.isfinite(value) for value in scores.values()):
        raise ValueError("registered probe scores must be finite")
    positive = (
        min(scores["full_probe_positive"], scores["carrier_probe_positive"]),
        scores["full_probe_positive"] + scores["carrier_probe_positive"],
    )
    negative = (
        min(scores["full_probe_negative"], scores["carrier_probe_negative"]),
        scores["full_probe_negative"] + scores["carrier_probe_negative"],
    )
    if positive == negative:
        raise ValueError("registered-only sign objective tied")
    selected_sign = 1 if positive > negative else -1
    payload = {
        "registered_probe_scores": scores,
        "positive_objective": list(positive),
        "negative_objective": list(negative),
        "selected_sign": selected_sign,
        "wrong_key_accessed": False,
        **CONTENT_SURVIVAL_CLAIM_BOUNDARY,
    }
    return {
        **payload,
        "selection_digest": build_stable_digest(payload),
    }


def _normalized_correlation(left: Any, right: Any) -> float:
    import torch

    left_flat = left.detach().float().reshape(-1)
    right_flat = right.detach().float().reshape(-1)
    if left_flat.numel() != right_flat.numel() or left_flat.numel() == 0:
        raise ValueError("oracle correlation tensors must have equal nonzero size")
    left_centered = left_flat - left_flat.mean()
    right_centered = right_flat - right_flat.mean()
    left_norm = torch.linalg.vector_norm(left_centered)
    right_norm = torch.linalg.vector_norm(right_centered)
    if (
        not bool(torch.isfinite(left_norm))
        or not bool(torch.isfinite(right_norm))
        or float(left_norm.item()) == 0.0
        or float(right_norm.item()) == 0.0
    ):
        raise ValueError("oracle correlation requires finite nonzero energy")
    score = float(
        torch.dot(left_centered / left_norm, right_centered / right_norm).item()
    )
    if not math.isfinite(score):
        raise ValueError("oracle correlation must be finite")
    return score


def compute_routed_template_oracle_score(
    observed_latent: Any,
    routed_lf_template: Any,
    routed_hf_template: Any,
    *,
    lf_weight: float,
    hf_weight: float,
    oracle_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Score routed templates without exposing the result to formal decisions."""

    if (
        type(lf_weight) is not float
        or type(hf_weight) is not float
        or lf_weight < 0.0
        or hf_weight < 0.0
        or lf_weight + hf_weight != 1.0
    ):
        raise ValueError("oracle role weights must be frozen and sum to one")
    lf_score = (
        _normalized_correlation(observed_latent, routed_lf_template)
        if lf_weight > 0.0
        else 0.0
    )
    hf_score = (
        _normalized_correlation(observed_latent, routed_hf_template)
        if hf_weight > 0.0
        else 0.0
    )
    score = lf_weight * lf_score + hf_weight * hf_score
    payload = {
        "oracle_identity": dict(oracle_identity),
        "observed_latent_content_sha256": tensor_content_sha256(
            observed_latent
        ),
        "routed_lf_template_content_sha256": tensor_content_sha256(
            routed_lf_template
        ),
        "routed_hf_template_content_sha256": tensor_content_sha256(
            routed_hf_template
        ),
        "lf_score": lf_score,
        "hf_score": hf_score,
        "score": score,
        "lf_weight": lf_weight,
        "hf_weight": hf_weight,
        "feedback_allowed": False,
        **CONTENT_SURVIVAL_CLAIM_BOUNDARY,
    }
    return {
        **payload,
        "oracle_score_digest": _domain_digest(_ORACLE_DOMAIN, payload),
    }


def compute_blind_observation_score(
    observed_latent: Any,
    *,
    key_material: str,
    model_identity_digest: str,
    prg_version: str,
    method_role: str,
) -> dict[str, Any]:
    """Call the frozen latent scorer directly, without image re-encoding."""

    lf_template = build_low_frequency_template(
        observed_latent,
        key_material,
        model_identity_digest,
        prg_version=prg_version,
    )
    hf_template = build_high_frequency_tail_template(
        observed_latent,
        key_material,
        model_identity_digest,
        prg_version=prg_version,
    )
    score = compute_blind_content_score(
        observed_latent,
        lf_template,
        hf_template,
        method_role,
    )
    return {
        "key_material_digest_random": build_stable_digest(
            {"key_material": key_material}
        ),
        "observed_latent_content_sha256": tensor_content_sha256(observed_latent),
        "lf_template_digest": lf_template.template_digest,
        "lf_template_content_sha256": tensor_content_sha256(lf_template.template),
        "hf_tail_template_digest": hf_template.template_digest,
        "hf_tail_template_content_sha256": tensor_content_sha256(
            hf_template.template
        ),
        "model_identity_digest": model_identity_digest,
        "prg_version": prg_version,
        "method_role": method_role,
        "blind_lf_score": score.blind_lf_score,
        "blind_hf_tail_score": score.blind_hf_tail_score,
        "blind_content_score": score.blind_content_score,
        "lf_weight": score.lf_weight,
        "hf_tail_weight": score.hf_tail_weight,
        "score_identity_digest": score.score_identity_digest,
        "scoring_key_identity_digest": score.scoring_key_identity_digest,
    }


def build_registered_rank_record(
    registered_score: float,
    wrong_scores: Sequence[float],
) -> dict[str, Any]:
    """Summarize a fixed 32-key null with conservative rank and percentile."""

    if len(wrong_scores) != CONTENT_SURVIVAL_WRONG_KEY_COUNT:
        raise ValueError("rank record requires exactly 32 wrong-key scores")
    registered = float(registered_score)
    wrong = tuple(float(value) for value in wrong_scores)
    if not math.isfinite(registered) or not all(math.isfinite(value) for value in wrong):
        raise ValueError("rank scores must be finite")
    mean = fmean(wrong)
    variance = fmean((value - mean) ** 2 for value in wrong)
    rank = 1 + sum(value >= registered for value in wrong)
    percentile = sum(value < registered for value in wrong) / len(wrong)
    payload = {
        "registered_score": registered,
        "wrong_score_count": len(wrong),
        "wrong_score_mean": mean,
        "wrong_score_population_variance": variance,
        "registered_rank": rank,
        "registered_empirical_percentile": percentile,
        "registered_minus_max_wrong_margin": registered - max(wrong),
        "registered_minus_mean_wrong_margin": registered - mean,
    }
    return {**payload, "rank_record_digest": build_stable_digest(payload)}


def validate_content_survival_observation_record(
    record: Mapping[str, Any],
    *,
    expected_cell_identity_digest: str,
    expected_latent_content_sha256: str,
    expected_registered_key_digest_random: str,
    expected_wrong_key_digests_random: Sequence[str],
) -> dict[str, Any]:
    resolved = dict(record)
    if resolved.get("cell_identity_digest") != _required_sha256(
        expected_cell_identity_digest,
        "expected_cell_identity_digest",
    ):
        raise ValueError("observation record cell identity mismatch")
    if resolved.get("observation_role") not in CONTENT_SURVIVAL_OBSERVATION_ROLES:
        raise ValueError("observation role is not governed")
    if resolved.get("chain_role") not in (
        *CONTENT_SURVIVAL_CHAIN_ROLES,
        "clean_reference",
    ):
        raise ValueError("chain role is not governed")
    for field, expected in CONTENT_SURVIVAL_CLAIM_BOUNDARY.items():
        if resolved.get(field) is not expected:
            raise ValueError("observation record crossed the claim boundary")
    latent_digest = _required_sha256(
        resolved.get("latent_content_sha256"), "latent_content_sha256"
    )
    if latent_digest != _required_sha256(
        expected_latent_content_sha256, "expected_latent_content_sha256"
    ):
        raise ValueError("observation record does not bind the persisted tensor")
    _required_sha256(resolved.get("parent_chain_digest"), "parent_chain_digest")
    if (
        resolved.get("oracle_feedback_allowed") is not False
        or resolved.get("wrong_key_feedback_allowed") is not False
    ):
        raise ValueError("diagnostic score feedback must remain disabled")
    key_records = resolved.get("key_score_records")
    if type(key_records) is not list or len(key_records) != 33:
        raise ValueError("observation record requires registered plus 32 wrong keys")
    expected_key_digests = [
        _required_sha256(
            expected_registered_key_digest_random,
            "expected_registered_key_digest_random",
        ),
        *[
            _required_sha256(value, "expected_wrong_key_digest_random")
            for value in expected_wrong_key_digests_random
        ],
    ]
    if len(expected_key_digests) != 33 or len(set(expected_key_digests)) != 33:
        raise ValueError("expected key roster must contain 33 unique identities")
    for index, (key_record, expected_digest) in enumerate(
        zip(key_records, expected_key_digests, strict=True)
    ):
        if type(key_record) is not dict:
            raise ValueError("key score record must be an object")
        if set(key_record) != {
            "key_role",
            "key_index",
            "key_material_digest_random",
            "blind_full_template",
            "routed_template_oracle",
            "feedback_allowed",
            *CONTENT_SURVIVAL_CLAIM_BOUNDARY,
        }:
            raise ValueError("key score record fields drifted")
        expected_role = "registered" if index == 0 else "wrong"
        expected_index = None if index == 0 else index - 1
        if (
            key_record.get("key_role") != expected_role
            or key_record.get("key_index") != expected_index
            or key_record.get("key_material_digest_random") != expected_digest
        ):
            raise ValueError("key score roster identity or order drifted")
        if key_record.get("feedback_allowed") is not False or any(
            key_record.get(field) is not expected
            for field, expected in CONTENT_SURVIVAL_CLAIM_BOUNDARY.items()
        ):
            raise ValueError("key score record crossed the diagnostic boundary")
        _validate_blind_score_record(
            key_record.get("blind_full_template"),
            expected_latent_content_sha256=latent_digest,
            expected_key_material_digest_random=expected_digest,
        )
        _validate_oracle_score_record(
            key_record.get("routed_template_oracle"),
            expected_latent_content_sha256=latent_digest,
            expected_key_material_digest_random=expected_digest,
        )
    for family, score_field in (
        ("blind_full_template", "blind_content_score"),
        ("routed_template_oracle", "score"),
    ):
        expected_rank = build_registered_rank_record(
            float(key_records[0][family][score_field]),
            [float(item[family][score_field]) for item in key_records[1:]],
        )
        if resolved.get("rank_records", {}).get(family) != expected_rank:
            raise ValueError("registered rank or null summary drifted")
    return resolved


def _validate_blind_score_record(
    value: Any,
    *,
    expected_latent_content_sha256: str,
    expected_key_material_digest_random: str,
) -> None:
    if type(value) is not dict:
        raise ValueError("blind score record must be an object")
    expected_fields = {
        "key_material_digest_random",
        "observed_latent_content_sha256",
        "lf_template_digest",
        "lf_template_content_sha256",
        "hf_tail_template_digest",
        "hf_tail_template_content_sha256",
        "model_identity_digest",
        "prg_version",
        "method_role",
        "blind_lf_score",
        "blind_hf_tail_score",
        "blind_content_score",
        "lf_weight",
        "hf_tail_weight",
        "score_identity_digest",
        "scoring_key_identity_digest",
    }
    if set(value) != expected_fields:
        raise ValueError("blind score fields drifted")
    if (
        value["observed_latent_content_sha256"]
        != expected_latent_content_sha256
        or value["key_material_digest_random"]
        != expected_key_material_digest_random
    ):
        raise ValueError("blind score latent or key identity mismatch")
    for field in (
        "lf_template_digest",
        "lf_template_content_sha256",
        "hf_tail_template_digest",
        "hf_tail_template_content_sha256",
        "model_identity_digest",
        "score_identity_digest",
        "scoring_key_identity_digest",
    ):
        _required_sha256(value[field], field)
    numbers = {
        field: float(value[field])
        for field in (
            "blind_lf_score",
            "blind_hf_tail_score",
            "blind_content_score",
            "lf_weight",
            "hf_tail_weight",
        )
        if not isinstance(value[field], bool)
    }
    if len(numbers) != 5 or not all(math.isfinite(item) for item in numbers.values()):
        raise ValueError("blind score values must be finite numbers")
    if numbers["lf_weight"] + numbers["hf_tail_weight"] != 1.0:
        raise ValueError("blind score role weights drifted")
    expected_content_score = (
        numbers["lf_weight"] * numbers["blind_lf_score"]
        + numbers["hf_tail_weight"] * numbers["blind_hf_tail_score"]
    )
    if numbers["blind_content_score"] != expected_content_score:
        raise ValueError("blind content score is not recomputable")
    score_identity_payload = {
        "observed_latent_content_sha256": value[
            "observed_latent_content_sha256"
        ],
        "lf_template_digest": value["lf_template_digest"],
        "lf_template_content_sha256": value["lf_template_content_sha256"],
        "hf_tail_template_digest": value["hf_tail_template_digest"],
        "hf_tail_template_content_sha256": value[
            "hf_tail_template_content_sha256"
        ],
        "model_identity_digest": value["model_identity_digest"],
        "prg_version": value["prg_version"],
        "scoring_key_identity_digest": value["scoring_key_identity_digest"],
        "method_role": value["method_role"],
        "lf_weight": numbers["lf_weight"],
        "hf_tail_weight": numbers["hf_tail_weight"],
        "blind_lf_score": numbers["blind_lf_score"],
        "blind_hf_tail_score": numbers["blind_hf_tail_score"],
        "blind_content_score": numbers["blind_content_score"],
    }
    if value["score_identity_digest"] != build_stable_digest(
        score_identity_payload
    ):
        raise ValueError("blind score identity digest mismatch")


def _validate_oracle_score_record(
    value: Any,
    *,
    expected_latent_content_sha256: str,
    expected_key_material_digest_random: str,
) -> None:
    if type(value) is not dict:
        raise ValueError("oracle score record must be an object")
    payload = dict(value)
    digest = payload.pop("oracle_score_digest", None)
    if digest != _domain_digest(_ORACLE_DOMAIN, payload):
        raise ValueError("oracle score digest mismatch")
    if (
        payload.get("observed_latent_content_sha256")
        != expected_latent_content_sha256
        or payload.get("oracle_identity", {}).get(
            "key_material_digest_random"
        )
        != expected_key_material_digest_random
    ):
        raise ValueError("oracle latent or key identity mismatch")
    for field in (
        "observed_latent_content_sha256",
        "routed_lf_template_content_sha256",
        "routed_hf_template_content_sha256",
    ):
        _required_sha256(payload.get(field), field)
    for field, expected in CONTENT_SURVIVAL_CLAIM_BOUNDARY.items():
        if payload.get(field) is not expected:
            raise ValueError("oracle score crossed the claim boundary")
    if payload.get("feedback_allowed") is not False:
        raise ValueError("oracle feedback must remain disabled")
    numeric = [
        payload.get(field)
        for field in ("lf_score", "hf_score", "score", "lf_weight", "hf_weight")
    ]
    if any(
        isinstance(item, bool)
        or not isinstance(item, (int, float))
        or not math.isfinite(float(item))
        for item in numeric
    ):
        raise ValueError("oracle scores must be finite")
    if float(payload["lf_weight"]) + float(payload["hf_weight"]) != 1.0:
        raise ValueError("oracle role weights drifted")
    if float(payload["score"]) != (
        float(payload["lf_weight"]) * float(payload["lf_score"])
        + float(payload["hf_weight"]) * float(payload["hf_score"])
    ):
        raise ValueError("oracle score is not recomputable")


def build_content_survival_causal_diagnostic(
    chain_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    records = [dict(record) for record in chain_records]
    if [record.get("role") for record in records] != list(
        CONTENT_SURVIVAL_CHAIN_ROLES
    ):
        raise ValueError("chain records do not follow the governed six-role order")
    before = records[-2]
    after = records[-1]
    before_callback = before.get("callback_record", {})
    after_callback = after.get("callback_record", {})
    nominal_pair_ready = bool(
        before_callback.get("registered_direction_content_sha256")
        == after_callback.get("registered_direction_content_sha256")
        and before_callback.get("common_gamma")
        == after_callback.get("common_gamma")
        and before.get("base_latent_identity_digest_random")
        == after.get("base_latent_identity_digest_random")
        and before.get("scheduler_identity_digest")
        == after.get("scheduler_identity_digest")
    )
    probe_callbacks = [record.get("callback_record", {}) for record in records[:4]]
    selector_reasons: list[str] = []
    if (
        probe_callbacks[0].get("registered_direction_content_sha256")
        != probe_callbacks[2].get("registered_direction_content_sha256")
        or probe_callbacks[1].get("registered_direction_content_sha256")
        != probe_callbacks[3].get("registered_direction_content_sha256")
    ):
        selector_reasons.append("full_carrier_probe_direction_identity_not_shared")
    if any("common_gamma" not in callback for callback in probe_callbacks):
        selector_reasons.append("full_carrier_probe_common_gamma_not_shared")
    reasons = [*selector_reasons, *CONTENT_SURVIVAL_M0_DEVIATIONS[1:]]
    return {
        "nominal_pair_unique_difference_ready": nominal_pair_ready,
        "selector_confounded": bool(selector_reasons),
        "confounded_reasons": reasons,
        "causal_conclusion_ready": bool(nominal_pair_ready and not reasons),
    }


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_bytes(path: Path, payload: bytes) -> None:
    partial = path.with_name(f"{path.name}.partial")
    if partial.exists():
        partial.unlink()
    with partial.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial, path)


def _load_tensor_leaf(path: Path) -> Any:
    import torch

    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError("persisted observation tensor cannot be safely loaded") from exc
    if type(value) is not torch.Tensor:
        raise TypeError("persisted observation tensor must be an exact Tensor")
    if (
        not value.dtype.is_floating_point
        or str(value.dtype) not in {
            "torch.float16",
            "torch.bfloat16",
            "torch.float32",
            "torch.float64",
        }
        or value.ndim != 4
        or value.shape[0] != 1
        or any(int(size) <= 0 for size in value.shape[1:])
        or not bool(torch.isfinite(value).all())
    ):
        raise ValueError("persisted observation tensor shape, dtype, or values are invalid")
    return value


def _read_png_identity(path: Path) -> dict[str, Any]:
    from PIL import Image

    try:
        with Image.open(path) as image:
            if image.format != "PNG":
                raise ValueError("observation image must be PNG")
            image.verify()
        with Image.open(path) as image:
            image.load()
            return canonical_rgb_uint8_content_record(image)
    except (OSError, ValueError) as exc:
        raise ValueError("persisted observation PNG is invalid") from exc


def _validate_scheduler_identity(record: Mapping[str, Any]) -> dict[str, Any]:
    resolved = dict(record)
    if set(resolved) != {"scheduler_class", "scheduler_config"}:
        raise ValueError("scheduler identity fields drifted")
    if not isinstance(resolved["scheduler_class"], str) or not resolved[
        "scheduler_class"
    ]:
        raise ValueError("scheduler class identity is missing")
    if type(resolved["scheduler_config"]) is not dict:
        raise ValueError("scheduler config identity must be a complete object")
    return resolved


def _validate_chain_callback_record(
    callback_record: Mapping[str, Any],
    *,
    expected_role: str,
) -> dict[str, Any]:
    callback = dict(callback_record)
    if expected_role == "clean_reference":
        expected_fields = _CALLBACK_COMMON_FIELDS
    elif expected_role in _PROBE_SIGN_BY_ROLE:
        expected_fields = (
            _CALLBACK_COMMON_FIELDS
            | _CALLBACK_CONTENT_FIELDS
            | _CALLBACK_PROBE_FIELDS
        )
    elif expected_role in {
        "nominal_before_positive",
        "nominal_after_selected",
    }:
        expected_fields = (
            _CALLBACK_COMMON_FIELDS
            | _CALLBACK_CONTENT_FIELDS
            | _CALLBACK_REPLAY_FIELDS
        )
    else:
        raise ValueError("chain callback role is not governed")
    if set(callback) != expected_fields:
        raise ValueError("chain callback required fields drifted")
    if callback["scheduler_step_index"] != 10 or (
        type(callback["scheduler_step_timestep"]) not in {int, float}
        or not math.isfinite(float(callback["scheduler_step_timestep"]))
    ):
        raise ValueError("chain callback scheduler observation drifted")
    for field in ("z9_content_sha256", "z10_before_write_content_sha256"):
        _required_sha256(callback[field], field)
    if expected_role == "clean_reference":
        return callback
    for field in (
        "routing_identity_digest",
        "registered_direction_content_sha256",
        "lf_update_content_sha256",
        "hf_update_content_sha256",
        "geometry_update_content_sha256",
    ):
        _required_sha256(callback[field], field)
    expected_geometry = expected_role.startswith("full_probe") or expected_role.startswith(
        "nominal_"
    )
    if callback["attention_geometry_enabled"] is not expected_geometry:
        raise ValueError("chain callback attention geometry role drifted")
    if expected_role in _PROBE_SIGN_BY_ROLE:
        if (
            callback["probe_role"] != expected_role
            or callback["probe_sign"] != _PROBE_SIGN_BY_ROLE[expected_role]
            or callback["direction_norm"] != "rms_chw"
            or callback["direction_axes"] != [1, 2, 3]
            or callback["direction_accumulation_dtype"] != "float64"
            or callback["probe_actual_tensor_dtype"]
            not in {"float16", "bfloat16", "float32", "float64"}
            or callback["probe_ratio_comparison"] != "less_than_or_equal"
            or callback["probe_ratio_ready"] is not True
            or callback["supports_paper_claim"] is not False
        ):
            raise ValueError("probe callback role, sign, or protocol fields drifted")
        for field in (
            "direction_unit_content_sha256",
            "probe_materialized_latent_content_sha256",
        ):
            _required_sha256(callback[field], field)
        for field in (
            "probe_target_ratio_float64",
            "probe_realized_ratio_float64",
            "probe_ratio_absolute_error_float64",
            "probe_ratio_tolerance_float64",
        ):
            if type(callback[field]) is not float or not math.isfinite(callback[field]):
                raise ValueError("probe callback numerical fields drifted")
        probe_payload = {
            field: callback[field]
            for field in _CALLBACK_PROBE_FIELDS
            if field != "probe_record_digest"
        }
        if callback["probe_record_digest"] != build_stable_digest(probe_payload):
            raise ValueError("probe callback record digest mismatch")
    else:
        if (
            type(callback["selected_sign"]) is not int
            or callback["selected_sign"] not in {-1, 1}
            or (
                expected_role == "nominal_before_positive"
                and callback["selected_sign"] != 1
            )
            or type(callback["common_gamma"]) is not float
            or not math.isfinite(callback["common_gamma"])
            or callback["common_gamma"] <= 0.0
        ):
            raise ValueError("nominal replay sign or common gamma drifted")
        for field in (
            "write_identity_digest",
            "actual_dtype_single_write_digest",
        ):
            _required_sha256(callback[field], field)
    return callback


def build_content_survival_parent_child_binding_digest(
    chain_records: Sequence[Mapping[str, Any]],
    observation_records: Sequence[Mapping[str, Any]],
    *,
    clean_chain_record: Mapping[str, Any] | None = None,
    clean_observation_records: Sequence[Mapping[str, Any]] = (),
) -> str:
    """Build the persisted chain-to-observation parent relation digest."""

    chains = [dict(record) for record in chain_records]
    observations = [dict(record) for record in observation_records]
    clean_observations = [dict(record) for record in clean_observation_records]
    if [record.get("role") for record in chains] != list(
        CONTENT_SURVIVAL_CHAIN_ROLES
    ):
        raise ValueError("parent binding requires exact ordered chains")
    expected_pairs = [
        (chain_role, observation_role)
        for chain_role in CONTENT_SURVIVAL_CHAIN_ROLES
        for observation_role in CONTENT_SURVIVAL_OBSERVATION_ROLES
    ]
    if [
        (record.get("chain_role"), record.get("observation_role"))
        for record in observations
    ] != expected_pairs:
        raise ValueError("parent binding requires exact ordered observations")
    chain_digests = {
        record["role"]: _required_sha256(
            record.get("parent_chain_digest"), "parent_chain_digest"
        )
        for record in chains
    }
    relation_records = []
    for record in observations:
        role = record["chain_role"]
        if record.get("parent_chain_digest") != chain_digests[role]:
            raise ValueError("observation parent does not match persisted chain")
        relation_records.append(
            {
                "chain_role": role,
                "observation_role": record["observation_role"],
                "parent_chain_digest": chain_digests[role],
                "observation_record_digest": _required_sha256(
                    record.get("observation_record_digest"),
                    "observation_record_digest",
                ),
                "latent_content_sha256": _required_sha256(
                    record.get("latent_content_sha256"),
                    "latent_content_sha256",
                ),
            }
        )
    clean_relation: dict[str, Any] | None = None
    if clean_chain_record is not None:
        clean_chain = dict(clean_chain_record)
        clean_parent = _required_sha256(
            clean_chain.get("parent_chain_digest"), "clean parent_chain_digest"
        )
        if [record.get("observation_role") for record in clean_observations] != list(
            CONTENT_SURVIVAL_OBSERVATION_ROLES
        ):
            raise ValueError("clean parent binding positions drifted")
        clean_relation = {
            "parent_chain_digest": clean_parent,
            "observations": [],
        }
        for record in clean_observations:
            if (
                record.get("chain_role") != "clean_reference"
                or record.get("parent_chain_digest") != clean_parent
            ):
                raise ValueError("clean observation parent drifted")
            clean_relation["observations"].append(
                {
                    "observation_role": record["observation_role"],
                    "parent_chain_digest": clean_parent,
                    "observation_record_digest": _required_sha256(
                        record.get("observation_record_digest"),
                        "clean observation_record_digest",
                    ),
                    "latent_content_sha256": _required_sha256(
                        record.get("latent_content_sha256"),
                        "clean latent_content_sha256",
                    ),
                }
            )
    elif clean_observations:
        raise ValueError("clean observations require their persisted chain")
    payload = {
        "chain_parent_records": [
            {"role": role, "parent_chain_digest": chain_digests[role]}
            for role in CONTENT_SURVIVAL_CHAIN_ROLES
        ],
        "observation_parent_records": relation_records,
        "clean_parent_record": clean_relation,
    }
    return _domain_digest(_PARENT_CHILD_BINDING_DOMAIN, payload)


def _validate_chain_record(
    record: Mapping[str, Any],
    *,
    expected_role: str,
    cell_identity: Mapping[str, Any],
    tensors: Mapping[str, Any],
    image_identity: Mapping[str, Any],
) -> dict[str, Any]:
    resolved = dict(record)
    digest = resolved.pop("parent_chain_digest", None)
    if digest != build_stable_digest(resolved):
        raise ValueError("chain parent digest mismatch")
    expected_fields = (
        _CLEAN_CHAIN_PARENT_FIELDS
        if expected_role == "clean_reference"
        else _CONTENT_CHAIN_PARENT_FIELDS
    )
    if set(resolved) != expected_fields:
        raise ValueError("chain parent required fields drifted")
    if resolved.get("role") != expected_role:
        raise ValueError("chain role or order drifted")
    if expected_role == "clean_reference":
        if (
            resolved.get("routing_mode") != "semantic"
            or resolved.get("carrier_mode") != "dual"
        ):
            raise ValueError("clean chain ablation identity drifted")
    elif (
        resolved.get("routing_mode") != cell_identity["routing_mode"]
        or resolved.get("carrier_mode") != cell_identity["carrier_mode"]
    ):
        raise ValueError("content chain ablation identity drifted")
    if (
        resolved.get("observation_run_identity_digest")
        != cell_identity["observation_run_identity_digest"]
        or resolved.get("prompt_text_digest")
        != cell_identity["prompt_text_digest"]
        or resolved.get("prompt_config_digest")
        != cell_identity["prompt_config_digest"]
        or resolved.get("key_roster_digest_random")
        != cell_identity["key_roster_digest_random"]
    ):
        raise ValueError("chain run, prompt, or roster identity mismatch")
    if expected_role != "clean_reference" and (
        resolved.get("cell_identity_digest")
        != cell_identity["cell_identity_digest"]
        or resolved.get("protocol_semantic_digest")
        != cell_identity["protocol_semantic_digest"]
        or resolved.get("prompt_roster_semantic_digest")
        != cell_identity["prompt_roster_semantic_digest"]
        or resolved.get("prompt_roster_artifact_file_sha256")
        != cell_identity["prompt_roster_artifact_file_sha256"]
    ):
        raise ValueError("chain cell, protocol, or prompt roster identity mismatch")
    base_identity = resolved["base_latent_identity"]
    if (
        type(base_identity) is not dict
        or base_identity.get("base_latent_identity_digest_random")
        != resolved["base_latent_identity_digest_random"]
        or build_stable_digest(
            {
                key: value
                for key, value in base_identity.items()
                if key != "base_latent_identity_digest_random"
            }
        )
        != resolved["base_latent_identity_digest_random"]
    ):
        raise ValueError("chain base latent identity mismatch")
    scheduler_identity = _validate_scheduler_identity(
        resolved["scheduler_identity"]
    )
    if resolved["scheduler_identity_digest"] != build_stable_digest(
        scheduler_identity
    ):
        raise ValueError("chain scheduler identity digest mismatch")
    callback = _validate_chain_callback_record(
        resolved["callback_record"],
        expected_role=expected_role,
    )
    for observation_role, tensor in tensors.items():
        field = {
            "post_write_z10": "post_write_z10_content_sha256",
            "pre_vae_terminal_latent": (
                "pre_vae_terminal_latent_content_sha256"
            ),
            "image_reencoded_latent": (
                "image_reencoded_latent_content_sha256"
            ),
        }[observation_role]
        if resolved.get(field) != tensor_content_sha256(tensor):
            raise ValueError("chain record does not bind persisted tensor content")
    for field in (
        "image_rgb_uint8_content_schema",
        "image_rgb_uint8_content_sha256",
        "image_width",
        "image_height",
    ):
        if resolved.get(field) != image_identity[field]:
            raise ValueError("chain record does not bind persisted PNG content")
    if callback.get("z10_before_write_content_sha256") == resolved.get(
        "post_write_z10_content_sha256"
    ) and expected_role != "clean_reference":
        raise ValueError("content chain must distinguish pre-write and post-write z10")
    for field, expected in CONTENT_SURVIVAL_CLAIM_BOUNDARY.items():
        if resolved.get(field) is not expected:
            raise ValueError("chain record crossed the claim boundary")
    return {**resolved, "parent_chain_digest": digest}


def _validate_result_from_leaves(
    directory: Path,
    result: Mapping[str, Any],
    *,
    expected_cell_identity_digest: str,
) -> dict[str, Any]:
    resolved = dict(result)
    result_digest = resolved.pop("result_digest", None)
    if result_digest != build_stable_digest(resolved):
        raise ValueError("cell result digest mismatch")
    cell_identity = resolved.get("cell_identity")
    if (
        type(cell_identity) is not dict
        or cell_identity.get("cell_identity_digest")
        != expected_cell_identity_digest
        or build_stable_digest(
            {
                key: value
                for key, value in cell_identity.items()
                if key != "cell_identity_digest"
            }
        )
        != expected_cell_identity_digest
    ):
        raise ValueError("cell result identity mismatch")
    if (
        resolved.get("result_schema")
        != "slm_wm_content_survival_observation_cell_result"
        or resolved.get("cell_chain_count") != 6
        or resolved.get("cell_evaluation_count") != 18 * 33 * 2
        or resolved.get("roster_digest_random")
        != cell_identity.get("key_roster_digest_random")
    ):
        raise ValueError("cell result fixed counts or roster identity mismatch")
    chain_records = resolved.get("chain_records")
    if type(chain_records) is not list or [
        record.get("role") for record in chain_records if type(record) is dict
    ] != list(CONTENT_SURVIVAL_CHAIN_ROLES):
        raise ValueError("cell result must contain exact six ordered chains")
    persisted_chains: dict[str, dict[str, Any]] = {}
    tensors_by_role: dict[str, dict[str, Any]] = {}
    for role in CONTENT_SURVIVAL_CHAIN_ROLES:
        tensors = {
            observation_role: _load_tensor_leaf(
                directory / f"{role}_{observation_role}.pt"
            )
            for observation_role in CONTENT_SURVIVAL_OBSERVATION_ROLES
        }
        image_identity = _read_png_identity(directory / f"{role}_image.png")
        chain_record = json.loads(
            (directory / f"{role}_chain_record.json").read_text(encoding="utf-8")
        )
        persisted_chains[role] = _validate_chain_record(
            chain_record,
            expected_role=role,
            cell_identity=cell_identity,
            tensors=tensors,
            image_identity=image_identity,
        )
        tensors_by_role[role] = tensors
    shared_parent_fields = (
        "cell_identity_digest",
        "base_latent_identity",
        "base_latent_identity_digest_random",
        "scheduler_identity",
        "scheduler_identity_digest",
        "observation_run_identity_digest",
        "prompt_text_digest",
        "prompt_config_digest",
        "key_roster_digest_random",
        "protocol_semantic_digest",
        "prompt_roster_semantic_digest",
        "prompt_roster_artifact_file_sha256",
    )
    anchor_chain = persisted_chains[CONTENT_SURVIVAL_CHAIN_ROLES[0]]
    for role in CONTENT_SURVIVAL_CHAIN_ROLES[1:]:
        current = persisted_chains[role]
        if any(current[field] != anchor_chain[field] for field in shared_parent_fields):
            raise ValueError("six-chain parent identity drifted")
        for field in (
            "z9_content_sha256",
            "z10_before_write_content_sha256",
            "routing_identity_digest",
        ):
            if current["callback_record"][field] != anchor_chain[
                "callback_record"
            ][field]:
                raise ValueError("six-chain callback parent identity drifted")
    if chain_records != [persisted_chains[role] for role in CONTENT_SURVIVAL_CHAIN_ROLES]:
        raise ValueError("result chain records differ from persisted chain records")
    observation_records = resolved.get("observation_records")
    expected_pairs = [
        (chain_role, observation_role)
        for chain_role in CONTENT_SURVIVAL_CHAIN_ROLES
        for observation_role in CONTENT_SURVIVAL_OBSERVATION_ROLES
    ]
    if type(observation_records) is not list or [
        (record.get("chain_role"), record.get("observation_role"))
        for record in observation_records
        if type(record) is dict
    ] != expected_pairs:
        raise ValueError("observation records do not cover six roles by three positions")
    for record in observation_records:
        payload = dict(record)
        record_digest = payload.pop("observation_record_digest", None)
        if record_digest != build_stable_digest(payload):
            raise ValueError("observation record digest mismatch")
        chain_role = record["chain_role"]
        observation_role = record["observation_role"]
        if record.get("parent_chain_digest") != persisted_chains[chain_role][
            "parent_chain_digest"
        ]:
            raise ValueError("observation record parent chain mismatch")
        validate_content_survival_observation_record(
            record,
            expected_cell_identity_digest=expected_cell_identity_digest,
            expected_latent_content_sha256=tensor_content_sha256(
                tensors_by_role[chain_role][observation_role]
            ),
            expected_registered_key_digest_random=cell_identity[
                "registered_key_material_digest_random"
            ],
            expected_wrong_key_digests_random=cell_identity[
                "wrong_key_material_digests_random"
            ],
        )
        for key_record in record["key_score_records"]:
            oracle_identity = key_record["routed_template_oracle"][
                "oracle_identity"
            ]
            if (
                key_record["blind_full_template"].get("model_identity_digest")
                != cell_identity["model_identity_digest"]
                or oracle_identity.get("chain_role") != chain_role
                or oracle_identity.get("routing_mode")
                != cell_identity["routing_mode"]
                or oracle_identity.get("carrier_mode")
                != cell_identity["carrier_mode"]
            ):
                raise ValueError("oracle chain or ablation identity drifted")
    registered_probe_scores = {
        role: float(
            next(
                record
                for record in observation_records
                if record["chain_role"] == role
                and record["observation_role"] == "image_reencoded_latent"
            )["key_score_records"][0]["blind_full_template"][
                "blind_content_score"
            ]
        )
        for role in CONTENT_SURVIVAL_CHAIN_ROLES[:4]
    }
    rebuilt_selection = select_content_survival_observation_sign(
        registered_probe_scores
    )
    if resolved.get("selection") != rebuilt_selection:
        raise ValueError("registered-only four-probe selection drifted")
    if persisted_chains["nominal_before_positive"]["callback_record"][
        "selected_sign"
    ] != 1 or persisted_chains["nominal_after_selected"]["callback_record"][
        "selected_sign"
    ] != rebuilt_selection["selected_sign"]:
        raise ValueError("nominal replay sign does not match persisted probe selection")
    diagnostic = build_content_survival_causal_diagnostic(chain_records)
    for field, expected in diagnostic.items():
        if resolved.get(field) != expected:
            raise ValueError("causal diagnostic or M0 confound drifted")
    if resolved.get("known_m0_deviations") != list(CONTENT_SURVIVAL_M0_DEVIATIONS):
        raise ValueError("known M0 deviations drifted")
    expects_clean = bool(
        cell_identity.get("routing_mode") == "semantic"
        and cell_identity.get("carrier_mode") == "dual"
    )
    clean_records = resolved.get("clean_observation_records")
    if type(clean_records) is not list or len(clean_records) != (3 if expects_clean else 0):
        raise ValueError("clean observation sharing contract drifted")
    clean_chain: dict[str, Any] | None = None
    if expects_clean:
        clean_tensors = {
            observation_role: _load_tensor_leaf(
                directory / f"clean_reference_{observation_role}.pt"
            )
            for observation_role in CONTENT_SURVIVAL_OBSERVATION_ROLES
        }
        clean_image_identity = _read_png_identity(
            directory / "clean_reference_image.png"
        )
        clean_chain = _validate_chain_record(
            json.loads(
                (directory / "clean_reference_chain_record.json").read_text(
                    encoding="utf-8"
                )
            ),
            expected_role="clean_reference",
            cell_identity=cell_identity,
            tensors=clean_tensors,
            image_identity=clean_image_identity,
        )
        if resolved.get("clean_chain_record") != clean_chain:
            raise ValueError("clean chain result identity drifted")
        if [record.get("observation_role") for record in clean_records] != list(
            CONTENT_SURVIVAL_OBSERVATION_ROLES
        ):
            raise ValueError("clean observation positions drifted")
        for record in clean_records:
            payload = dict(record)
            digest = payload.pop("observation_record_digest", None)
            if digest != build_stable_digest(payload):
                raise ValueError("clean observation record digest mismatch")
            observation_role = record["observation_role"]
            if record.get("chain_role") != "clean_reference" or record.get(
                "parent_chain_digest"
            ) != clean_chain["parent_chain_digest"]:
                raise ValueError("clean observation parent identity drifted")
            validate_content_survival_observation_record(
                record,
                expected_cell_identity_digest=expected_cell_identity_digest,
                expected_latent_content_sha256=tensor_content_sha256(
                    clean_tensors[observation_role]
                ),
                expected_registered_key_digest_random=cell_identity[
                    "registered_key_material_digest_random"
                ],
                expected_wrong_key_digests_random=cell_identity[
                    "wrong_key_material_digests_random"
                ],
            )
            for key_record in record["key_score_records"]:
                oracle_identity = key_record["routed_template_oracle"][
                    "oracle_identity"
                ]
                if (
                    key_record["blind_full_template"].get(
                        "model_identity_digest"
                    )
                    != cell_identity["model_identity_digest"]
                    or oracle_identity.get("chain_role") != "clean_reference"
                    or oracle_identity.get("routing_mode") != "semantic"
                    or oracle_identity.get("carrier_mode") != "dual"
                ):
                    raise ValueError("clean oracle identity drifted")
    elif "clean_chain_record" in resolved:
        raise ValueError("non-owner cell cannot duplicate the shared clean chain")
    rebuilt_parent_child_digest = build_content_survival_parent_child_binding_digest(
        [persisted_chains[role] for role in CONTENT_SURVIVAL_CHAIN_ROLES],
        observation_records,
        clean_chain_record=clean_chain,
        clean_observation_records=clean_records,
    )
    if resolved.get("parent_child_binding_digest") != rebuilt_parent_child_digest:
        raise ValueError("result parent-child binding drifted")
    for field, expected in CONTENT_SURVIVAL_CLAIM_BOUNDARY.items():
        if resolved.get(field) is not expected:
            raise ValueError("cell result crossed the claim boundary")
    return {**resolved, "result_digest": result_digest}


def validate_content_survival_cell_bundle(
    cell_dir: str | Path,
    *,
    expected_cell_identity_digest: str,
    complete: bool,
) -> dict[str, Any]:
    """Re-read every published byte and reject partial or tampered bundles."""

    directory = Path(cell_dir).resolve()
    manifest_name = "cell_manifest.json" if complete else "cell_manifest.partial.json"
    manifest_path = directory / manifest_name
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("cell_identity_digest") != expected_cell_identity_digest:
        raise ValueError("cell manifest identity mismatch")
    if (
        manifest.get("cell_identity", {}).get("cell_identity_digest")
        != expected_cell_identity_digest
        or build_stable_digest(
            {
                key: value
                for key, value in manifest["cell_identity"].items()
                if key != "cell_identity_digest"
            }
        )
        != expected_cell_identity_digest
    ):
        raise ValueError("cell manifest full identity mismatch")
    if manifest.get("artifact_complete") is not True:
        raise ValueError("cell manifest does not describe a complete artifact set")
    leaves = manifest.get("leaves")
    if type(leaves) is not list or not leaves:
        raise ValueError("cell manifest requires leaves")
    for leaf in leaves:
        relative = Path(str(leaf.get("path", "")))
        path = (directory / relative).resolve()
        path.relative_to(directory)
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = path.read_bytes()
        if (
            len(payload) != leaf.get("size_bytes")
            or hashlib.sha256(payload).hexdigest() != leaf.get("sha256")
        ):
            raise ValueError("cell leaf size or digest mismatch")
    if manifest.get("leaf_count") != len(leaves):
        raise ValueError("cell leaf count mismatch")
    for field, expected in CONTENT_SURVIVAL_CLAIM_BOUNDARY.items():
        if manifest.get(field) is not expected:
            raise ValueError("cell manifest crossed the claim boundary")
    binding_path = directory / "cell_binding.json"
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    binding_digest = binding.pop("cell_binding_digest", None)
    if (
        binding.get("cell_identity") != manifest["cell_identity"]
        or binding.get("leaves") != [
            leaf for leaf in leaves if leaf.get("role") != "cell_binding"
        ]
        or binding_digest != _domain_digest(_CELL_BINDING_DOMAIN, binding)
    ):
        raise ValueError("cell binding identity, leaves, or digest mismatch")
    for field, expected in CONTENT_SURVIVAL_CLAIM_BOUNDARY.items():
        if binding.get(field) is not expected:
            raise ValueError("cell binding crossed the claim boundary")
    result = json.loads((directory / "cell_result.json").read_text(encoding="utf-8"))
    validated_result = _validate_result_from_leaves(
        directory,
        result,
        expected_cell_identity_digest=expected_cell_identity_digest,
    )
    rebuilt_parent_child_digest = validated_result["parent_child_binding_digest"]
    if (
        binding.get("parent_child_binding_digest")
        != rebuilt_parent_child_digest
        or manifest.get("parent_child_binding_digest")
        != rebuilt_parent_child_digest
    ):
        raise ValueError("result, binding, and manifest parent relation drifted")
    expected_names = {str(leaf["path"]) for leaf in leaves}
    expected_names.add(manifest_name)
    actual_names = {path.name for path in directory.iterdir() if path.is_file()}
    if actual_names != expected_names:
        raise ValueError("cell directory contains missing or unlisted files")
    return manifest


def build_content_survival_observation_summary(
    output_root: str | Path,
    *,
    expected_cells: Sequence[Mapping[str, Any]],
    observation_run_identity: Mapping[str, Any],
    protocol: ContentSurvivalObservationProtocol,
) -> dict[str, Any]:
    root = Path(output_root).resolve()
    cells = [dict(cell) for cell in expected_cells]
    if len(cells) != CONTENT_SURVIVAL_CELL_COUNT:
        raise ValueError("summary requires the frozen 24-cell roster")
    expected_paths = [str(cell.get("relative_path")) for cell in cells]
    if len(set(expected_paths)) != CONTENT_SURVIVAL_CELL_COUNT:
        raise ValueError("summary cell paths must be unique")
    actual_paths = {
        path.parent.relative_to(root).as_posix()
        for path in root.rglob("cell_manifest.json")
        if path.is_file()
    }
    if actual_paths != set(expected_paths):
        raise ValueError("summary complete-manifest roster is incomplete or contains drift")
    manifest_records: list[dict[str, Any]] = []
    clean_chain_records: list[dict[str, Any]] = []
    result_digests: list[str] = []
    all_causal_blocked = True
    for cell in cells:
        relative_path = Path(cell["relative_path"])
        directory = (root / relative_path).resolve()
        directory.relative_to(root)
        identity_digest = _required_sha256(
            cell.get("cell_identity_digest"), "summary cell identity digest"
        )
        validate_content_survival_cell_bundle(
            directory,
            expected_cell_identity_digest=identity_digest,
            complete=True,
        )
        manifest_path = directory / "cell_manifest.json"
        result_path = directory / "cell_result.json"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result["cell_identity"]["observation_run_identity"] != dict(
            observation_run_identity
        ):
            raise ValueError("summary cell run identity drifted")
        manifest_records.append(
            {
                "relative_path": relative_path.as_posix(),
                "cell_identity_digest": identity_digest,
                "cell_manifest_file_sha256": hashlib.sha256(
                    manifest_path.read_bytes()
                ).hexdigest(),
                "cell_result_file_sha256": hashlib.sha256(
                    result_path.read_bytes()
                ).hexdigest(),
            }
        )
        result_digests.append(result["result_digest"])
        all_causal_blocked = bool(
            all_causal_blocked and result.get("causal_conclusion_ready") is False
        )
        if result["cell_identity"]["routing_mode"] == "semantic" and result[
            "cell_identity"
        ]["carrier_mode"] == "dual":
            clean_chain = result["clean_chain_record"]
            clean_chain_records.append(
                {
                    "prompt_id": result["cell_identity"]["prompt_id"],
                    "parent_chain_digest": clean_chain["parent_chain_digest"],
                    "image_rgb_uint8_content_sha256": clean_chain[
                        "image_rgb_uint8_content_sha256"
                    ],
                    "key_roster_digest_random": clean_chain[
                        "key_roster_digest_random"
                    ],
                }
            )
    if [record["prompt_id"] for record in clean_chain_records] != list(
        CONTENT_SURVIVAL_PROMPT_IDS
    ) or len({record["parent_chain_digest"] for record in clean_chain_records}) != 4:
        raise ValueError("summary must bind four unique shared clean chains")
    payload = {
        "summary_schema": "slm_wm_content_survival_observation_summary",
        "protocol_identity": protocol.identity_record(),
        "source_evidence": dict(protocol.payload["source_evidence"]),
        "observation_run_identity": dict(observation_run_identity),
        "observation_run_identity_digest": observation_run_identity[
            "observation_run_identity_digest"
        ],
        "prompt_roster_semantic_digest": protocol.payload["roster"][
            "roster_semantic_digest"
        ],
        "prompt_roster_artifact_file_sha256": protocol.payload["roster"][
            "roster_artifact_file_sha256"
        ],
        "validated_cell_manifest_records": manifest_records,
        "validated_cell_manifest_records_digest": build_stable_digest(
            manifest_records
        ),
        "cell_result_digests": result_digests,
        "clean_chain_records": clean_chain_records,
        "cell_count": CONTENT_SURVIVAL_CELL_COUNT,
        "complete_cell_count": CONTENT_SURVIVAL_CELL_COUNT,
        "clean_chain_count": len(clean_chain_records),
        "chain_count": CONTENT_SURVIVAL_CHAIN_COUNT,
        "evaluation_count": CONTENT_SURVIVAL_EVALUATION_COUNT,
        "all_causal_conclusions_blocked": all_causal_blocked,
        "known_m0_deviations": list(CONTENT_SURVIVAL_M0_DEVIATIONS),
        **CONTENT_SURVIVAL_CLAIM_BOUNDARY,
    }
    if payload["all_causal_conclusions_blocked"] is not True:
        raise ValueError("M0 diagnostic summary cannot claim causal readiness")
    return {**payload, "summary_digest": build_stable_digest(payload)}


def publish_content_survival_cell(
    cell_dir: str | Path,
    *,
    cell_identity: Mapping[str, Any],
    leaf_payloads: Mapping[str, bytes],
) -> dict[str, Any]:
    """Publish one complete cell with the manifest rename as the last write."""

    directory = Path(cell_dir).resolve()
    repository_outputs = Path(__file__).resolve().parents[2] / "outputs"
    try:
        directory.relative_to(repository_outputs.resolve())
    except ValueError:
        raise ValueError("persistent observation output must remain under outputs")
    identity_digest = _required_sha256(
        cell_identity.get("cell_identity_digest"),
        "cell_identity_digest",
    )
    complete_path = directory / "cell_manifest.json"
    partial_path = directory / "cell_manifest.partial.json"
    if complete_path.exists():
        return validate_content_survival_cell_bundle(
            directory,
            expected_cell_identity_digest=identity_digest,
            complete=True,
        )
    directory.mkdir(parents=True, exist_ok=True)
    if partial_path.exists():
        partial = json.loads(partial_path.read_text(encoding="utf-8"))
        if partial.get("cell_identity_digest") != identity_digest:
            raise ValueError("partial cell belongs to a different roster identity")
    leaves: list[dict[str, Any]] = []
    for file_name in sorted(leaf_payloads):
        if (
            type(file_name) is not str
            or not file_name
            or "/" in file_name
            or ".." in file_name
            or Path(file_name).suffix not in {".json", ".pt", ".png"}
        ):
            raise ValueError("cell leaf name must be a safe governed file")
        payload = leaf_payloads[file_name]
        if type(payload) is not bytes or not payload:
            raise ValueError("cell leaf payload must be non-empty bytes")
        path = directory / file_name
        _atomic_bytes(path, payload)
        leaves.append(
            {
                "role": Path(file_name).stem,
                "path": path.name,
                "size_bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    result_path = directory / "cell_result.json"
    if not result_path.is_file():
        raise ValueError("cell result leaf is required before binding")
    result_record = json.loads(result_path.read_text(encoding="utf-8"))
    parent_child_binding_digest = _required_sha256(
        result_record.get("parent_child_binding_digest"),
        "parent_child_binding_digest",
    )
    binding_payload = {
        "cell_identity": dict(cell_identity),
        "parent_child_binding_digest": parent_child_binding_digest,
        "leaves": leaves,
        **CONTENT_SURVIVAL_CLAIM_BOUNDARY,
    }
    binding = {
        **binding_payload,
        "cell_binding_digest": _domain_digest(
            _CELL_BINDING_DOMAIN,
            binding_payload,
        ),
    }
    binding_bytes = _canonical_json_bytes(binding)
    _atomic_bytes(directory / "cell_binding.json", binding_bytes)
    leaves.append(
        {
            "role": "cell_binding",
            "path": "cell_binding.json",
            "size_bytes": len(binding_bytes),
            "sha256": hashlib.sha256(binding_bytes).hexdigest(),
        }
    )
    manifest = {
        "manifest_schema": "slm_wm_content_survival_observation_cell",
        "cell_identity_digest": identity_digest,
        "cell_identity": dict(cell_identity),
        "parent_child_binding_digest": parent_child_binding_digest,
        "artifact_complete": True,
        "leaves": leaves,
        "leaf_count": len(leaves),
        **CONTENT_SURVIVAL_CLAIM_BOUNDARY,
    }
    with partial_path.open("wb") as handle:
        handle.write(_canonical_json_bytes(manifest))
        handle.flush()
        os.fsync(handle.fileno())
    validate_content_survival_cell_bundle(
        directory,
        expected_cell_identity_digest=identity_digest,
        complete=False,
    )
    for leaf in leaves:
        _fsync_file(directory / leaf["path"])
    _fsync_file(partial_path)
    _fsync_directory(directory)
    os.replace(partial_path, complete_path)
    _fsync_directory(directory)
    return validate_content_survival_cell_bundle(
        directory,
        expected_cell_identity_digest=identity_digest,
        complete=True,
    )


__all__ = [
    "CONTENT_SURVIVAL_CARRIER_MODES",
    "CONTENT_SURVIVAL_CHAIN_COUNT",
    "CONTENT_SURVIVAL_CHAIN_ROLES",
    "CONTENT_SURVIVAL_CLAIM_BOUNDARY",
    "CONTENT_SURVIVAL_EVALUATION_COUNT",
    "CONTENT_SURVIVAL_M0_DEVIATIONS",
    "CONTENT_SURVIVAL_OBSERVATION_ROLES",
    "CONTENT_SURVIVAL_PROMPT_IDS",
    "CONTENT_SURVIVAL_ROUTING_REFERENCE_IDENTITY",
    "CONTENT_SURVIVAL_ROUTING_MODES",
    "CONTENT_SURVIVAL_SELECTION_RECORDS",
    "CONTENT_SURVIVAL_SOURCE_EVIDENCE",
    "CONTENT_SURVIVAL_WRONG_KEY_COUNT",
    "ContentSurvivalObservationProtocol",
    "build_content_survival_cell_identity",
    "build_content_survival_causal_diagnostic",
    "build_content_survival_execution_environment_identity",
    "build_content_survival_observation_roster",
    "build_content_survival_observation_run_identity",
    "build_content_survival_observation_summary",
    "build_content_survival_parent_child_binding_digest",
    "build_registered_rank_record",
    "compute_blind_observation_score",
    "compute_routed_template_oracle_score",
    "content_survival_observation_semantic_digest",
    "content_survival_observation_roster_semantic_digest",
    "load_content_survival_observation_protocol",
    "publish_content_survival_cell",
    "select_content_survival_observation_sign",
    "validate_content_survival_cell_bundle",
    "validate_content_survival_observation_payload",
    "validate_content_survival_observation_record",
    "validate_content_survival_execution_environment_identity",
]
