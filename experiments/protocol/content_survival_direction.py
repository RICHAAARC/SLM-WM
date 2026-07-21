"""治理内容载体跨扩散存活方向、正式重放与产物闭合。"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from main.core.digest import build_stable_digest, tensor_content_sha256


CONTENT_SURVIVAL_DIRECTION_SCHEMA = "slm_wm_content_survival_direction_protocol"
CONTENT_SURVIVAL_ARTIFACT_BINDING_SCHEMA = (
    "slm_wm_content_survival_artifact_binding"
)
CONTENT_SURVIVAL_DIRECTION_CONFIG_PATH = Path(
    "configs/content_survival_direction_protocol.json"
)
CONTENT_SURVIVAL_DIRECTION_SCHEMA_PATH = Path(
    "configs/content_survival_direction_protocol_schema.json"
)
CONTENT_SURVIVAL_DIRECTION_RECORD_PATH = (
    "content_survival_direction_record.json"
)
CONTENT_SURVIVAL_ARTIFACT_BINDING_PATH = (
    "content_survival_artifact_binding.json"
)
CONTENT_SURVIVAL_CHAIN_ROLES = (
    "clean_reference",
    "full_probe_positive",
    "full_probe_negative",
    "carrier_probe_positive",
    "carrier_probe_negative",
    "carrier_nominal_replay",
    "full_nominal_replay",
)
CONTENT_SURVIVAL_PROBE_ROLES = CONTENT_SURVIVAL_CHAIN_ROLES[1:5]
CONTENT_SURVIVAL_REPLAY_ROLES = CONTENT_SURVIVAL_CHAIN_ROLES[5:]
_SEMANTIC_DIGEST_DOMAIN = b"slm_wm_content_survival_direction_semantic_v1\0"
_COMPOSITE_DIGEST_DOMAIN = b"slm_wm_runtime_method_identity_v1\0"
_BINDING_DIGEST_DOMAIN = b"slm_wm_content_survival_artifact_binding_v1\0"
_SEMANTIC_DIGEST_FIELD = "survival_protocol_semantic_digest"
_COMPOSITE_KEYS = (
    "core_method_definition_digest",
    "survival_protocol_file_sha256",
    "survival_protocol_semantic_digest",
    "protocol_version",
    "survival_protocol_schema_file_sha256",
)
_ALLOWED_DTYPES = ("float16", "bfloat16", "float32")


def _canonical_json_bytes(value: Any) -> bytes:
    """返回协议摘要唯一消费的 UTF-8 canonical JSON。"""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _domain_sha256(domain: bytes, value: Any) -> str:
    digest = hashlib.sha256()
    digest.update(domain)
    digest.update(_canonical_json_bytes(value))
    return digest.hexdigest()


def _required_sha256(value: Any, field_name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field_name} must be a lowercase SHA-256")
    return value


def _semantic_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    resolved = dict(payload)
    resolved.pop(_SEMANTIC_DIGEST_FIELD, None)
    return resolved


def content_survival_protocol_semantic_digest(
    payload: Mapping[str, Any],
) -> str:
    """对删除摘要声明后的协议语义执行域分离 SHA-256。"""

    return _domain_sha256(_SEMANTIC_DIGEST_DOMAIN, _semantic_payload(payload))


def _load_json_object(path: Path, field_name: str) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes()
    try:
        value = json.loads(raw.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{field_name} must be valid UTF-8 JSON") from exc
    if type(value) is not dict:
        raise TypeError(f"{field_name} must be a JSON object")
    return value, raw


def _require_exact_keys(
    value: Mapping[str, Any],
    expected: Sequence[str],
    field_name: str,
) -> None:
    if set(value) != set(expected):
        raise ValueError(f"{field_name} fields do not match the governed schema")


@dataclass(frozen=True)
class ContentSurvivalDirectionProtocol:
    """保存已验证配置字节、语义摘要和文件摘要。"""

    payload: dict[str, Any]
    protocol_version: str
    semantic_digest: str
    file_sha256: str
    schema_file_sha256: str
    config_path: str
    schema_path: str

    @property
    def target_ratio(self) -> float:
        return float(self.payload["direction"]["target_ratio"])

    @property
    def numerical_floor(self) -> float:
        return float(self.payload["direction"]["numerical_floor"])

    def identity_record(self) -> dict[str, Any]:
        return {
            "protocol_schema": CONTENT_SURVIVAL_DIRECTION_SCHEMA,
            "protocol_version": self.protocol_version,
            "survival_protocol_file_sha256": self.file_sha256,
            "survival_protocol_semantic_digest": self.semantic_digest,
            "survival_protocol_schema_file_sha256": (
                self.schema_file_sha256
            ),
            "survival_protocol_config_path": self.config_path,
            "survival_protocol_schema_path": self.schema_path,
        }


def validate_content_survival_direction_payload(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """集中验证七链、方向、选择、重放和 dtype 语义。"""

    resolved = dict(payload)
    _require_exact_keys(
        resolved,
        (
            "protocol_schema",
            "protocol_version",
            "direction",
            "chain_protocol",
            "sign_selection",
            "replay_protocol",
            "failure_protocol",
            _SEMANTIC_DIGEST_FIELD,
        ),
        "protocol",
    )
    if resolved["protocol_schema"] != CONTENT_SURVIVAL_DIRECTION_SCHEMA:
        raise ValueError("content survival protocol schema mismatch")
    if resolved["protocol_version"] != "content_survival_direction_v1":
        raise ValueError("unsupported content survival protocol version")

    direction = resolved["direction"]
    if type(direction) is not dict:
        raise TypeError("direction must be an object")
    _require_exact_keys(
        direction,
        (
            "target_ratio",
            "norm",
            "axes",
            "accumulation_dtype",
            "numerical_floor",
            "realized_ratio_definition",
            "comparison",
            "relative_tolerance_multiplier",
            "allowed_actual_dtypes",
            "materialization_policy",
        ),
        "direction",
    )
    if (
        type(direction["target_ratio"]) is not float
        or direction["target_ratio"] != 1.0e-3
        or direction["norm"] != "rms_chw"
        or direction["axes"] != [1, 2, 3]
        or direction["accumulation_dtype"] != "float64"
        or type(direction["numerical_floor"]) is not float
        or direction["numerical_floor"] != 1.0e-12
        or direction["realized_ratio_definition"]
        != "rms_chw_float64_materialized_delta_over_rms_chw_float64_z10"
        or direction["comparison"] != "less_than_or_equal"
        or type(direction["relative_tolerance_multiplier"]) is not float
        or direction["relative_tolerance_multiplier"] != 8.0
        or direction["materialization_policy"]
        != "single_cast_no_rescale_no_retry_no_selection"
    ):
        raise ValueError("direction protocol constants drifted")
    dtype_records = direction["allowed_actual_dtypes"]
    if type(dtype_records) is not dict or tuple(dtype_records) != _ALLOWED_DTYPES:
        raise ValueError("allowed actual dtype order drifted")
    expected_dtype_records = {
        "float16": {
            "machine_epsilon_source": "ieee_754_binary16",
            "machine_epsilon": 0.0009765625,
            "absolute_tolerance": 0.00025,
        },
        "bfloat16": {
            "machine_epsilon_source": "brain_floating_point_16",
            "machine_epsilon": 0.0078125,
            "absolute_tolerance": 0.001,
        },
        "float32": {
            "machine_epsilon_source": "ieee_754_binary32",
            "machine_epsilon": 1.1920928955078125e-07,
            "absolute_tolerance": 1.0e-7,
        },
    }
    if dtype_records != expected_dtype_records:
        raise ValueError("actual dtype tolerance constants drifted")

    chains = resolved["chain_protocol"]
    if type(chains) is not dict or chains != {
        "roles": list(CONTENT_SURVIVAL_CHAIN_ROLES),
        "chain_count": 7,
        "scheduler_fork": "canonical_base_latent_fresh_scheduler_replay",
        "shared_z10_identity_required": True,
        "registered_key_only_probe": True,
        "probe_supports_paper_claim": False,
    }:
        raise ValueError("seven-chain role or scheduler protocol drifted")

    selection = resolved["sign_selection"]
    if type(selection) is not dict or selection != {
        "candidate_signs": [1, -1],
        "shared_across_full_and_carrier": True,
        "primary_objective": "maximize_min_registered_content_score",
        "secondary_objective": "maximize_sum_registered_content_score",
        "tie_absolute_tolerance": 0.0,
        "tie_policy": "fail_closed",
        "nonfinite_policy": "fail_closed",
        "degenerate_direction_policy": "fail_closed",
        "wrong_key_access": "forbidden_until_nominal_replay_complete",
    }:
        raise ValueError("shared sign selection protocol drifted")

    replay = resolved["replay_protocol"]
    if type(replay) is not dict or replay != {
        "roles": list(CONTENT_SURVIVAL_REPLAY_ROLES),
        "probe_and_nominal_amplitudes_are_distinct": True,
        "shared_canonical_base_latent": True,
        "shared_scheduler_and_z10": True,
        "shared_registered_direction_and_sign": True,
        "only_full_carrier_difference": "attention_geometry_enabled",
        "parent_binding": "probe_bundle_digest",
        "same_role_deterministic_replay_requires_byte_identity": True,
    }:
        raise ValueError("nominal replay protocol drifted")
    failure = resolved["failure_protocol"]
    if type(failure) is not dict or failure != {
        "continue_after_governed_failure": False,
        "zero_direction": "fail_closed",
        "nonfinite": "fail_closed",
        "unsupported_dtype": "fail_closed",
        "ratio_outside_tolerance": "fail_closed",
        "sign_conflict_or_tie": "fail_closed",
        "missing_artifact": "fail_closed",
        "supports_paper_claim": False,
    }:
        raise ValueError("content survival failure protocol drifted")
    claimed = _required_sha256(
        resolved[_SEMANTIC_DIGEST_FIELD],
        _SEMANTIC_DIGEST_FIELD,
    )
    if content_survival_protocol_semantic_digest(resolved) != claimed:
        raise ValueError("content survival semantic digest mismatch")
    return resolved


def load_content_survival_direction_protocol(
    repository_root: str | Path,
) -> ContentSurvivalDirectionProtocol:
    """从当前提交固定路径加载并验证配置与 schema。"""

    root = Path(repository_root).resolve()
    config_path = root / CONTENT_SURVIVAL_DIRECTION_CONFIG_PATH
    schema_path = root / CONTENT_SURVIVAL_DIRECTION_SCHEMA_PATH
    payload, raw = _load_json_object(config_path, "content survival config")
    schema, schema_raw = _load_json_object(
        schema_path,
        "content survival schema",
    )
    if (
        schema.get("$id") != CONTENT_SURVIVAL_DIRECTION_SCHEMA
        or schema.get("type") != "object"
        or schema.get("additionalProperties") is not False
        or schema.get("required") != [
            "protocol_schema",
            "protocol_version",
            "direction",
            "chain_protocol",
            "sign_selection",
            "replay_protocol",
            "failure_protocol",
            _SEMANTIC_DIGEST_FIELD,
        ]
    ):
        raise ValueError("content survival schema identity drifted")
    validated = validate_content_survival_direction_payload(payload)
    return ContentSurvivalDirectionProtocol(
        payload=validated,
        protocol_version=str(validated["protocol_version"]),
        semantic_digest=str(validated[_SEMANTIC_DIGEST_FIELD]),
        file_sha256=hashlib.sha256(raw).hexdigest(),
        schema_file_sha256=hashlib.sha256(schema_raw).hexdigest(),
        config_path=CONTENT_SURVIVAL_DIRECTION_CONFIG_PATH.as_posix(),
        schema_path=CONTENT_SURVIVAL_DIRECTION_SCHEMA_PATH.as_posix(),
    )


def build_content_survival_runtime_method_identity(
    *,
    core_method_definition_digest: str,
    protocol: ContentSurvivalDirectionProtocol,
) -> dict[str, Any]:
    """由唯一 owner 以固定四键构造复合方法身份。"""

    payload = {
        "core_method_definition_digest": _required_sha256(
            core_method_definition_digest,
            "core_method_definition_digest",
        ),
        "survival_protocol_file_sha256": protocol.file_sha256,
        "survival_protocol_semantic_digest": protocol.semantic_digest,
        "protocol_version": protocol.protocol_version,
        "survival_protocol_schema_file_sha256": (
            protocol.schema_file_sha256
        ),
    }
    if tuple(payload) != _COMPOSITE_KEYS:
        raise RuntimeError("composite runtime method identity key order drifted")
    return {
        **payload,
        "composite_runtime_method_identity_digest": _domain_sha256(
            _COMPOSITE_DIGEST_DOMAIN,
            payload,
        ),
    }


def materialize_content_survival_probe(
    z10: Any,
    raw_direction: Any,
    *,
    sign: int,
    role: str,
    protocol: ContentSurvivalDirectionProtocol,
) -> tuple[Any, dict[str, Any]]:
    """以 RMS_CHW 单位方向物化一次 actual-dtype 对称 probe。"""

    import torch

    if role not in CONTENT_SURVIVAL_PROBE_ROLES:
        raise ValueError("probe role is not governed")
    if type(sign) is not int or sign not in (-1, 1):
        raise ValueError("probe sign must be exact -1 or 1")
    if not isinstance(z10, torch.Tensor) or not isinstance(
        raw_direction,
        torch.Tensor,
    ):
        raise TypeError("z10 and raw_direction must be Tensors")
    if z10.shape != raw_direction.shape or z10.ndim != 4 or z10.shape[0] != 1:
        raise ValueError("probe tensors must share [1,C,H,W] shape")
    dtype_name = str(z10.dtype).removeprefix("torch.")
    dtype_record = protocol.payload["direction"]["allowed_actual_dtypes"].get(
        dtype_name
    )
    if type(dtype_record) is not dict:
        raise ValueError("actual dtype is not governed by the protocol")
    direction64 = raw_direction.detach().to(dtype=torch.float64)
    z1064 = z10.detach().to(dtype=torch.float64)
    if not bool(torch.isfinite(direction64).all()) or not bool(
        torch.isfinite(z1064).all()
    ):
        raise ValueError("probe tensors must be finite")
    axes = tuple(protocol.payload["direction"]["axes"])
    direction_rms = torch.sqrt(torch.mean(direction64.square(), dim=axes, keepdim=True))
    z10_rms = torch.sqrt(torch.mean(z1064.square(), dim=axes, keepdim=True))
    floor = protocol.numerical_floor
    if bool((direction_rms <= floor).any()) or bool((z10_rms <= floor).any()):
        raise RuntimeError("probe direction and z10 RMS must be non-zero")
    unit_direction = direction64 / direction_rms
    perturbation64 = (
        float(sign)
        * protocol.target_ratio
        * z10_rms
        * unit_direction
    )
    candidate = (z1064 + perturbation64).to(dtype=z10.dtype)
    realized_delta64 = candidate.detach().to(dtype=torch.float64) - z1064
    realized_rms = torch.sqrt(
        torch.mean(realized_delta64.square(), dim=axes, keepdim=True)
    )
    realized_ratio_tensor = realized_rms / z10_rms
    if not bool(torch.isfinite(realized_ratio_tensor).all()) or bool(
        (realized_ratio_tensor <= floor).any()
    ):
        raise RuntimeError("materialized probe ratio must be positive and finite")
    realized_ratio = float(realized_ratio_tensor.max().item())
    tolerance = float(dtype_record["absolute_tolerance"]) + (
        float(protocol.payload["direction"]["relative_tolerance_multiplier"])
        * float(dtype_record["machine_epsilon"])
        * protocol.target_ratio
    )
    absolute_error = abs(realized_ratio - protocol.target_ratio)
    if not absolute_error <= tolerance:
        raise RuntimeError("actual-dtype probe ratio exceeds governed tolerance")
    record = {
        "probe_role": role,
        "probe_sign": sign,
        "direction_norm": "rms_chw",
        "direction_axes": list(axes),
        "direction_accumulation_dtype": "float64",
        "direction_unit_content_sha256": tensor_content_sha256(
            unit_direction.to(dtype=torch.float32)
        ),
        "probe_target_ratio_float64": protocol.target_ratio,
        "probe_actual_tensor_dtype": dtype_name,
        "probe_realized_ratio_float64": realized_ratio,
        "probe_ratio_absolute_error_float64": absolute_error,
        "probe_ratio_tolerance_float64": tolerance,
        "probe_ratio_comparison": "less_than_or_equal",
        "probe_ratio_ready": True,
        "probe_materialized_latent_content_sha256": tensor_content_sha256(
            candidate.detach().float()
        ),
        "supports_paper_claim": False,
    }
    record["probe_record_digest"] = build_stable_digest(record)
    return candidate, record


def select_shared_content_survival_sign(
    probe_records: Sequence[Mapping[str, Any]],
    *,
    protocol: ContentSurvivalDirectionProtocol,
) -> dict[str, Any]:
    """仅用 registered content score 为 full/carrier 选择共同 sign。"""

    records = [dict(record) for record in probe_records]
    if [record.get("probe_role") for record in records] != list(
        CONTENT_SURVIVAL_PROBE_ROLES
    ):
        raise ValueError("probe records do not follow the governed role order")
    scores: dict[int, list[float]] = {1: [], -1: []}
    for record in records:
        sign = record.get("probe_sign")
        score = record.get("registered_content_score")
        if sign not in scores or isinstance(score, bool) or not isinstance(
            score,
            (int, float),
        ):
            raise ValueError("probe registered score identity is invalid")
        resolved_score = float(score)
        if not math.isfinite(resolved_score):
            raise RuntimeError("probe registered score must be finite")
        scores[int(sign)].append(resolved_score)
    if any(len(values) != 2 for values in scores.values()):
        raise ValueError("each sign must contain full and carrier scores")
    full_preference = scores[1][0] - scores[-1][0]
    carrier_preference = scores[1][1] - scores[-1][1]
    if (
        full_preference == 0.0
        or carrier_preference == 0.0
        or full_preference * carrier_preference < 0.0
    ):
        raise RuntimeError(
            "full and carrier probe sign preferences conflict or tie"
        )
    objectives = {
        sign: (min(values), sum(values)) for sign, values in scores.items()
    }
    positive = objectives[1]
    negative = objectives[-1]
    if positive == negative:
        raise RuntimeError("shared sign objective is tied")
    selected_sign = 1 if positive > negative else -1
    record = {
        "selected_sign": selected_sign,
        "registered_only": True,
        "wrong_key_used_for_selection": False,
        "positive_objective": list(positive),
        "negative_objective": list(negative),
        "primary_objective": "maximize_min_registered_content_score",
        "secondary_objective": "maximize_sum_registered_content_score",
        "tie_absolute_tolerance": 0.0,
        "selection_ready": True,
    }
    record["selection_digest"] = build_stable_digest(record)
    return record


def encode_frozen_clip_global_image_feature(runtime: Any, image: Any) -> Any:
    """通过已锁定 runtime 的官方 ``model.get_image_features`` 路径编码。"""

    import torch

    prepare = getattr(runtime, "prepare_image_pixels", None)
    model = getattr(runtime, "_model", None)
    if not callable(prepare) or not callable(getattr(model, "get_image_features", None)):
        raise TypeError("frozen CLIP runtime lacks the official global image path")
    pixel_values = prepare(image)
    with torch.inference_mode():
        features = model.get_image_features(pixel_values=pixel_values)
    if (
        not isinstance(features, torch.Tensor)
        or tuple(features.shape) != (1, 512)
        or not bool(torch.isfinite(features).all())
    ):
        raise RuntimeError("official CLIP global image feature is invalid")
    features = features.to(dtype=torch.float32)
    norm = torch.linalg.vector_norm(features, dim=-1, keepdim=True)
    if not bool(torch.isfinite(norm).all()) or bool((norm <= 0.0).any()):
        raise RuntimeError("official CLIP global image feature has zero norm")
    return features / norm


def _structure_features(image: Any) -> Any:
    import torch

    if (
        not isinstance(image, torch.Tensor)
        or image.ndim != 4
        or tuple(image.shape[:2]) != (1, 3)
    ):
        raise ValueError("structure feature image must be [1,3,H,W]")
    value = image.detach().to(dtype=torch.float64)
    if not bool(torch.isfinite(value).all()) or bool(
        ((value < 0.0) | (value > 1.0)).any()
    ):
        raise ValueError("structure feature image must be finite in [0,1]")
    means = value.mean(dim=(0, 2, 3))
    standard_deviations = value.std(dim=(0, 2, 3), unbiased=False)
    horizontal = (value[:, :, :, 1:] - value[:, :, :, :-1]).abs().mean(
        dim=(0, 2, 3)
    )
    vertical = (value[:, :, 1:, :] - value[:, :, :-1, :]).abs().mean(
        dim=(0, 2, 3)
    )
    return torch.cat((means, standard_deviations, horizontal, vertical))


def build_three_image_content_survival_evidence(
    *,
    runtime: Any,
    clean_image: Any,
    carrier_only_image: Any,
    full_image: Any,
    minimum_semantic_cosine: float,
    maximum_structure_relative_drift: float,
    counterfactual_identity_digest: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """从三张真实重放图像生成 CLIP 全局与结构保持证据。"""

    import torch

    _required_sha256(
        counterfactual_identity_digest,
        "counterfactual_identity_digest",
    )
    images = (clean_image, carrier_only_image, full_image)
    clip_features = tuple(
        encode_frozen_clip_global_image_feature(runtime, image)
        for image in images
    )
    structure_features = tuple(_structure_features(image) for image in images)

    def pair(left: int, right: int) -> tuple[float, float, bool]:
        cosine = float(
            torch.sum(clip_features[left] * clip_features[right]).item()
        )
        denominator = max(
            float(torch.linalg.vector_norm(structure_features[left]).item()),
            1.0e-12,
        )
        drift = float(
            torch.linalg.vector_norm(
                structure_features[right] - structure_features[left]
            ).item()
            / denominator
        )
        if not math.isfinite(cosine) or not math.isfinite(drift):
            raise RuntimeError("three-image preservation values must be finite")
        return (
            cosine,
            drift,
            cosine >= minimum_semantic_cosine
            and drift <= maximum_structure_relative_drift,
        )

    clean_full = pair(0, 2)
    clean_carrier = pair(0, 1)
    carrier_full = pair(1, 2)
    final_record = {
        "final_image_semantic_cosine_similarity": clean_full[0],
        "final_image_handcrafted_structure_feature_relative_drift": clean_full[1],
        "minimum_semantic_preservation_cosine": minimum_semantic_cosine,
        "maximum_handcrafted_structure_feature_relative_drift": (
            maximum_structure_relative_drift
        ),
        "final_image_preservation_gate_ready": clean_full[2],
        "preservation_validation_scope": (
            "clean_to_full_official_clip_global_and_real_structure_features"
        ),
    }
    carrier_record = {
        "carrier_only_final_image_preservation_applicable": True,
        "carrier_only_final_image_semantic_cosine_similarity": clean_carrier[0],
        "carrier_only_final_image_handcrafted_structure_feature_relative_drift": (
            clean_carrier[1]
        ),
        "carrier_only_final_image_preservation_gate_ready": clean_carrier[2],
        "carrier_only_to_full_final_image_semantic_cosine_similarity": carrier_full[0],
        "carrier_only_to_full_final_image_handcrafted_structure_feature_relative_drift": (
            carrier_full[1]
        ),
        "carrier_only_to_full_final_image_preservation_gate_ready": carrier_full[2],
        "carrier_only_counterfactual_three_way_preservation_gate_ready": (
            clean_full[2] and clean_carrier[2] and carrier_full[2]
        ),
        "minimum_semantic_preservation_cosine": minimum_semantic_cosine,
        "maximum_handcrafted_structure_feature_relative_drift": (
            maximum_structure_relative_drift
        ),
        "carrier_only_counterfactual_identity_digest": (
            counterfactual_identity_digest
        ),
        "carrier_only_final_image_preservation_status": (
            "measured_from_nominal_clean_carrier_and_full_replays"
        ),
        "preservation_validation_scope": (
            "three_pair_official_clip_global_and_real_structure_features"
        ),
    }
    feature_record = {
        "feature_source": "frozen_official_clip_get_image_features_and_real_pixels",
        "clip_model_identity_digest": _required_sha256(
            getattr(runtime, "model_identity_digest", ""),
            "clip_model_identity_digest",
        ),
        "clip_feature_content_sha256": {
            role: tensor_content_sha256(feature)
            for role, feature in zip(
                ("clean", "carrier_only", "full"),
                clip_features,
            )
        },
        "structure_feature_content_sha256": {
            role: tensor_content_sha256(feature.to(dtype=torch.float32))
            for role, feature in zip(
                ("clean", "carrier_only", "full"),
                structure_features,
            )
        },
    }
    feature_record["feature_record_digest"] = build_stable_digest(
        feature_record
    )
    return final_record, carrier_record, feature_record


def build_content_survival_artifact_binding(
    *,
    run_id: str,
    composite_method_identity: Mapping[str, Any],
    protocol_record: Mapping[str, Any],
    leaf_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """仅绑定不可变叶子，避免 result/binding/manifest 摘要循环。"""

    ordered_leaves = [dict(record) for record in leaf_records]
    if not run_id or not ordered_leaves:
        raise ValueError("artifact binding requires run_id and leaves")
    if len({record.get("path") for record in ordered_leaves}) != len(
        ordered_leaves
    ):
        raise ValueError("artifact binding contains duplicate leaf paths")
    payload = {
        "binding_schema": CONTENT_SURVIVAL_ARTIFACT_BINDING_SCHEMA,
        "run_id": run_id,
        "composite_runtime_method_identity": dict(composite_method_identity),
        "content_survival_direction_record": dict(protocol_record),
        "leaf_records": ordered_leaves,
    }
    return {
        **payload,
        "content_survival_artifact_binding_digest": _domain_sha256(
            _BINDING_DIGEST_DOMAIN,
            payload,
        ),
    }


def build_content_survival_direction_record(
    *,
    protocol: ContentSurvivalDirectionProtocol,
    composite_method_identity: Mapping[str, Any],
    base_latent_identity_digest_random: str,
    chain_records: Sequence[Mapping[str, Any]],
    probe_records: Sequence[Mapping[str, Any]],
    selection: Mapping[str, Any],
    probe_bundle_digest: str,
    final_image_feature_record: Mapping[str, Any],
) -> dict[str, Any]:
    """集中构造并自验七链正式方向记录。"""

    record = {
        "protocol_identity": protocol.identity_record(),
        "composite_runtime_method_identity": dict(
            composite_method_identity
        ),
        "chain_roles": list(CONTENT_SURVIVAL_CHAIN_ROLES),
        "chain_count": len(chain_records),
        "chain_records": [dict(item) for item in chain_records],
        "probe_records": [dict(item) for item in probe_records],
        "selection": dict(selection),
        "base_latent_identity_digest_random": _required_sha256(
            base_latent_identity_digest_random,
            "base_latent_identity_digest_random",
        ),
        "probe_bundle_digest": _required_sha256(
            probe_bundle_digest,
            "probe_bundle_digest",
        ),
        "nominal_replay_records": [
            dict(item) for item in chain_records[-2:]
        ],
        "final_image_feature_record": dict(final_image_feature_record),
        "wrong_key_evaluated_after_nominal_replay": True,
        "supports_paper_claim": False,
    }
    record["content_survival_direction_record_digest"] = (
        build_stable_digest(record)
    )
    if not _content_survival_record_ready(
        record,
        expected_protocol=protocol,
        expected_composite=composite_method_identity,
    ):
        raise RuntimeError("content survival direction record is not self-consistent")
    return record


def _content_survival_record_ready(
    record: Mapping[str, Any],
    *,
    expected_protocol: ContentSurvivalDirectionProtocol,
    expected_composite: Mapping[str, Any],
) -> bool:
    """复算七链、probe 选择与 nominal replay 的父子身份。"""

    try:
        payload = dict(record)
        declared_record_digest = _required_sha256(
            payload.pop("content_survival_direction_record_digest"),
            "content_survival_direction_record_digest",
        )
        if build_stable_digest(payload) != declared_record_digest:
            return False
        if (
            payload.get("protocol_identity")
            != expected_protocol.identity_record()
            or payload.get("composite_runtime_method_identity")
            != dict(expected_composite)
            or payload.get("chain_roles")
            != list(CONTENT_SURVIVAL_CHAIN_ROLES)
            or payload.get("chain_count") != 7
            or payload.get("supports_paper_claim") is not False
            or payload.get("wrong_key_evaluated_after_nominal_replay") is not True
        ):
            return False
        chain_records = payload.get("chain_records")
        if not isinstance(chain_records, list) or len(chain_records) != 7:
            return False
        if [item.get("role") for item in chain_records] != list(
            CONTENT_SURVIVAL_CHAIN_ROLES
        ):
            return False
        for index, item in enumerate(chain_records):
            if type(item) is not dict or item.get("chain_index") != index:
                return False
            item_payload = dict(item)
            item_digest = _required_sha256(
                item_payload.pop("chain_record_digest"),
                "chain_record_digest",
            )
            if build_stable_digest(item_payload) != item_digest:
                return False
        if len({item.get("z10_content_sha256") for item in chain_records}) != 1:
            return False
        if len({item.get("scheduler_step_timestep") for item in chain_records}) != 1:
            return False

        probe_records = payload.get("probe_records")
        if not isinstance(probe_records, list) or len(probe_records) != 4:
            return False
        for item in probe_records:
            item_payload = dict(item)
            item_digest = _required_sha256(
                item_payload.pop("probe_artifact_digest"),
                "probe_artifact_digest",
            )
            if build_stable_digest(item_payload) != item_digest:
                return False
            if (
                item.get("supports_paper_claim") is not False
                or item.get("wrong_key_used_for_selection") is not False
                or item.get("probe_ratio_ready") is not True
            ):
                return False
        rebuilt_selection = select_shared_content_survival_sign(
            probe_records,
            protocol=expected_protocol,
        )
        if payload.get("selection") != rebuilt_selection:
            return False
        probe_parent = {
            "protocol_identity": expected_protocol.identity_record(),
            "composite_runtime_method_identity": dict(expected_composite),
            "base_latent_identity_digest_random": payload.get(
                "base_latent_identity_digest_random"
            ),
            "probe_records": probe_records,
            "selection": rebuilt_selection,
        }
        probe_bundle_digest = build_stable_digest(probe_parent)
        if payload.get("probe_bundle_digest") != probe_bundle_digest:
            return False
        replay_records = payload.get("nominal_replay_records")
        if (
            not isinstance(replay_records, list)
            or replay_records != chain_records[-2:]
            or [item.get("role") for item in replay_records]
            != list(CONTENT_SURVIVAL_REPLAY_ROLES)
        ):
            return False
        selected_sign = rebuilt_selection["selected_sign"]
        for item in replay_records:
            if not (
                item.get("selected_sign") == selected_sign
                and item.get("probe_bundle_digest") == probe_bundle_digest
                and item.get("nominal_replay_parent_ready") is True
                and item.get("actual_dtype_single_write_count") == 1
            ):
                return False
            replay_payload = dict(item)
            replay_digest = _required_sha256(
                replay_payload.pop("nominal_replay_record_digest"),
                "nominal_replay_record_digest",
            )
            replay_payload.pop("chain_record_digest")
            if build_stable_digest(replay_payload) != replay_digest:
                return False
        if not (
            replay_records[0].get("attention_geometry_enabled") is False
            and replay_records[1].get("attention_geometry_enabled") is True
            and replay_records[1].get("post_write_qk_strict_ready") is True
        ):
            return False
        feature_record = payload.get("final_image_feature_record")
        if not isinstance(feature_record, Mapping):
            return False
        feature_payload = dict(feature_record)
        feature_digest = _required_sha256(
            feature_payload.pop("feature_record_digest"),
            "feature_record_digest",
        )
        if (
            build_stable_digest(feature_payload) != feature_digest
            or feature_record.get("feature_source")
            != "frozen_official_clip_get_image_features_and_real_pixels"
        ):
            return False
        return True
    except (KeyError, TypeError, ValueError, RuntimeError):
        return False


def validate_content_survival_artifact_bundle(
    *,
    repository_root: str | Path,
    result_payload: Mapping[str, Any],
    manifest: Mapping[str, Any],
    expected_protocol: ContentSurvivalDirectionProtocol,
    expected_core_method_definition_digest: str,
) -> bool:
    """供 writer 预发布与 loader 完成后共同执行的纯只读 validator。"""

    root = Path(repository_root).resolve()
    try:
        result = dict(result_payload)
        metadata = dict(result["metadata"])
        manifest_payload = dict(manifest)
        manifest_metadata = dict(manifest_payload["metadata"])
        composite = build_content_survival_runtime_method_identity(
            core_method_definition_digest=expected_core_method_definition_digest,
            protocol=expected_protocol,
        )
        if metadata.get("composite_runtime_method_identity") != composite:
            return False
        binding_relative = str(
            metadata.get("content_survival_artifact_binding_path", "")
        )
        binding_path = (root / binding_relative).resolve()
        binding_path.relative_to((root / "outputs").resolve())
        binding, _ = _load_json_object(
            binding_path,
            "content survival artifact binding",
        )
        declared_binding_digest = _required_sha256(
            binding.get("content_survival_artifact_binding_digest"),
            "content_survival_artifact_binding_digest",
        )
        digest_payload = dict(binding)
        digest_payload.pop("content_survival_artifact_binding_digest")
        if _domain_sha256(_BINDING_DIGEST_DOMAIN, digest_payload) != declared_binding_digest:
            return False
        if (
            metadata.get("content_survival_artifact_binding_digest")
            != declared_binding_digest
            or manifest_metadata.get("content_survival_artifact_binding_digest")
            != declared_binding_digest
            or binding.get("run_id") != result.get("run_id")
            or binding.get("composite_runtime_method_identity") != composite
        ):
            return False
        protocol_record = binding.get("content_survival_direction_record")
        if not isinstance(protocol_record, Mapping):
            return False
        if not _content_survival_record_ready(
            protocol_record,
            expected_protocol=expected_protocol,
            expected_composite=composite,
        ):
            return False
        if metadata.get("content_survival_direction_record") != protocol_record:
            return False
        feature_record = protocol_record.get("final_image_feature_record")
        if (
            not isinstance(feature_record, Mapping)
            or metadata.get("prompt_saliency_model_identity_digest")
            != feature_record.get("clip_model_identity_digest")
        ):
            return False
        output_paths = [str(path) for path in manifest_payload["output_paths"]]
        if len(output_paths) != len(set(output_paths)):
            return False
        required_result_paths = {
            str(result["clean_image_path"]),
            str(result["watermarked_image_path"]),
            str(result["update_record_path"]),
            str(result["detection_record_path"]),
            str(result["manifest_path"]),
            binding_relative,
            str(metadata.get("content_survival_direction_record_path", "")),
            str(metadata.get("carrier_only_image_path", "")),
            str(metadata.get("carrier_only_update_record_path", "")),
        }
        if "" in required_result_paths or not required_result_paths <= set(output_paths):
            return False
        result_relative = (
            Path(str(result["manifest_path"])).parent / "runtime_result.json"
        ).as_posix()
        result_path = (root / result_relative).resolve()
        result_path.relative_to((root / "outputs").resolve())
        if (
            not result_path.is_file()
            or hashlib.sha256(result_path.read_bytes()).hexdigest()
            != manifest_metadata.get("runtime_result_sha256")
            or hashlib.sha256(binding_path.read_bytes()).hexdigest()
            != manifest_metadata.get(
                "content_survival_artifact_binding_file_sha256"
            )
            or metadata.get("content_survival_artifact_binding_file_sha256")
            != manifest_metadata.get(
                "content_survival_artifact_binding_file_sha256"
            )
        ):
            return False
        if json.loads(result_path.read_text(encoding="utf-8")) != result:
            return False
        protocol_leaf_path = (
            root / str(metadata["content_survival_direction_record_path"])
        ).resolve()
        protocol_leaf_path.relative_to((root / "outputs").resolve())
        if (
            not protocol_leaf_path.is_file()
            or json.loads(protocol_leaf_path.read_text(encoding="utf-8"))
            != protocol_record
        ):
            return False
        leaf_records = binding.get("leaf_records")
        if not isinstance(leaf_records, list) or [
            leaf.get("role") for leaf in leaf_records
        ] != [
            "clean_nominal_image",
            "carrier_nominal_image",
            "full_nominal_image",
            "full_nominal_update",
            "carrier_nominal_update",
            "nominal_detection_records",
            "content_survival_direction_record",
        ]:
            return False
        for leaf in leaf_records:
            if type(leaf) is not dict:
                return False
            relative = str(leaf.get("path", ""))
            path = (root / relative).resolve()
            path.relative_to((root / "outputs").resolve())
            if (
                relative not in output_paths
                or not path.is_file()
                or path.is_symlink()
                or path.stat().st_size != leaf.get("size_bytes")
                or hashlib.sha256(path.read_bytes()).hexdigest()
                != leaf.get("sha256")
            ):
                return False
        if manifest_payload.get("config_digest") != build_stable_digest(
            manifest_payload.get("config")
        ):
            return False
        if manifest_metadata.get("composite_runtime_method_identity") != composite:
            return False
        if result.get("run_decision") != "pass" or result.get("update_count") != 1:
            return False
        if metadata.get("old_content_runtime_artifact_compatible") is not False:
            return False
        final_preservation = metadata.get("final_image_preservation")
        carrier_preservation = metadata.get(
            "carrier_only_final_image_preservation"
        )
        observability = metadata.get("final_image_attention_observability")
        if not (
            isinstance(final_preservation, Mapping)
            and final_preservation.get("final_image_preservation_gate_ready")
            is True
            and isinstance(carrier_preservation, Mapping)
            and carrier_preservation.get(
                "carrier_only_counterfactual_three_way_preservation_gate_ready"
            )
            is True
            and isinstance(observability, Mapping)
            and observability.get(
                "final_image_attention_observability_gate_ready"
            )
            is True
        ):
            return False
        return True
    except (
        KeyError,
        OSError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ):
        return False
